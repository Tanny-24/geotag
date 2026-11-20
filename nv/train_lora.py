"""Step 6 (GPU, ~40-70 min on a 24 GB card): LoRA fine-tuning of Qwen2.5-VL-3B-Instruct (LiveVQA recipe).

    python -m nv.train_lora --out ckpt/lora_mixed                       # main model (all formats)
    python -m nv.train_lora --out ckpt/lora_qa --formats qa              # ablation: QA format only

Vision tower frozen; LoRA (r=64, alpha=128) on every attention/MLP projection of the language model;
lr 1e-4, cosine, 1 epoch; loss only on the answer tokens. bf16, gradient checkpointing.
"""
import argparse
import math
import os
import random
import time
from functools import partial

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from . import config as C
from .common import read_jsonl

PROJ = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def attn_impl():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_image(split, idx, max_side=896):
    img = Image.open(C.tara_image(split, idx)).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return img


def make_batch(proc, batch, max_len):
    tok = proc.tokenizer
    texts, imgs, tlens = [], [], []
    for ex in batch:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ex["prompt"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": ex["target"]}]}]
        texts.append(proc.apply_chat_template(msgs, tokenize=False))
        imgs.append(load_image(ex["split"], ex["idx"]))
        tlens.append(len(tok(ex["target"] + "<|im_end|>\n", add_special_tokens=False).input_ids))
    enc = proc(text=texts, images=imgs, padding=True, return_tensors="pt")
    labels = torch.full_like(enc["input_ids"], -100)
    for i in range(len(batch)):
        L = int(enc["attention_mask"][i].sum())          # right padding
        labels[i, L - tlens[i]:L] = enc["input_ids"][i, L - tlens[i]:L]
    if enc["input_ids"].shape[1] > max_len:
        print(f"warning: sequence length {enc['input_ids'].shape[1]} > {max_len}")
    return {**enc, "labels": labels}     # plain dict: DataLoader pins it without BatchFeature quirks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=None, help="default: $NV_OUT/sft/train.jsonl")
    ap.add_argument("--out", default="ckpt/lora_mixed", help="relative to $NV_OUT")
    ap.add_argument("--formats", default="qa,mcqa,qar")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--dora", action="store_true")
    ap.add_argument("--max_pixels", type=int, default=448 * 448)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--save_every", type=int, default=100, help="optimizer steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="CPU processes that build batches while the GPU trains")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = os.path.join(C.OUT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    data = read_jsonl(args.train or C.p("sft", "train.jsonl"))
    keep = set(args.formats.split(","))
    data = [d for d in data if d["fmt"] in keep]
    if args.limit:
        data = data[:args.limit]
    print(f"{len(data)} training samples, formats={sorted(keep)}")

    proc = AutoProcessor.from_pretrained(C.MLLM, min_pixels=64 * 28 * 28, max_pixels=args.max_pixels)
    proc.tokenizer.padding_side = "right"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        C.MLLM, torch_dtype=torch.bfloat16, attn_implementation=attn_impl()).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    targets = sorted({n for n, m in model.named_modules()
                      if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in PROJ and "visual" not in n})
    print(f"LoRA on {len(targets)} language-model linear layers (vision tower frozen)")
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05,
                                             target_modules=targets, use_dora=args.dora, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(data) / (args.bs * args.accum))
    total = max(1, int(steps_per_epoch * args.epochs))
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total), total)

    # sanity check of the label mask on the first batch
    b = make_batch(proc, data[:1], args.max_len)
    lab = b["labels"][0][b["labels"][0] != -100]
    print("label span decodes to:", repr(proc.tokenizer.decode(lab)[:120]))

    # Image decoding, Qwen patching and tokenising run in worker processes, overlapped with the GPU step;
    # in the main loop they left the GPU idle while each batch was built.
    dl = DataLoader(data, batch_size=args.bs, shuffle=True, generator=torch.Generator().manual_seed(args.seed),
                    collate_fn=partial(make_batch, proc, max_len=args.max_len), num_workers=args.workers,
                    pin_memory=True, persistent_workers=args.workers > 0)

    model.train()
    step, micro, t0 = 0, 0, time.time()
    run_loss, n_run = torch.zeros((), device="cuda"), 0   # summed on device: .item() per micro-step syncs the GPU
    n_micro_total = total * args.accum
    while step < total:
        for batch in dl:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            (loss / args.accum).backward()
            run_loss += loss.detach()
            n_run += 1
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0 or step == 1:
                    el = time.time() - t0
                    eta = el / micro * (n_micro_total - micro)
                    print(f"step {step}/{total}  loss {run_loss.item() / n_run:.4f}  lr {sched.get_last_lr()[0]:.2e}"
                          f"  elapsed {el / 60:.1f}m  eta {eta / 60:.1f}m  mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G",
                          flush=True)
                    run_loss.zero_()
                    n_run = 0
                if step % args.save_every == 0:
                    model.save_pretrained(os.path.join(out_dir, f"step{step}"))
                if step >= total:
                    break
    model.save_pretrained(os.path.join(out_dir, "final"))
    print("saved", os.path.join(out_dir, "final"))


if __name__ == "__main__":
    main()
