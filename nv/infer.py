"""Step 7 (GPU): run Qwen2.5-VL (zero-shot or + LoRA adapter) on the evaluation prompts.

    python -m nv.infer --split test --fmt qa_noev --name zs_noev                     # MLLM default
    python -m nv.infer --split test --fmt qa      --name zs_qa                       # MLLM + retrieved articles
    python -m nv.infer --split test --fmt mcqa    --name lora_mcqa --adapter ckpt/lora_mixed/final

Writes $NV_OUT/preds/{split}/{name}.jsonl with raw output, parsed Location/Date, option letter and the
probability of the first generated option letter (MCQA confidence, used for calibration).
"""
import argparse
import os

import torch
from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList, Qwen2_5_VLForConditionalGeneration

from . import config as C
from .common import parse_answer, read_jsonl, write_jsonl
from .prompts import LETTERS
from .train_lora import attn_impl, load_image


def load_model(adapter=None, max_pixels=448 * 448):
    proc = AutoProcessor.from_pretrained(C.MLLM, min_pixels=64 * 28 * 28, max_pixels=max_pixels)
    proc.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        C.MLLM, torch_dtype=torch.bfloat16, attn_implementation=attn_impl()).cuda().eval()
    if adapter:
        from peft import PeftModel
        path = adapter if os.path.isabs(adapter) or os.path.exists(adapter) else os.path.join(C.OUT, adapter)
        model = PeftModel.from_pretrained(model, path).merge_and_unload().eval()
    return model, proc


class FirstTokenLetterProbs(LogitsProcessor):
    """Records the option-letter probabilities of the first generated token and leaves the scores unchanged.

    It sits after generate's own processors, so it sees the same scores output_scores=True returned; that flag
    kept a (batch x 151k-vocab) fp32 tensor on the GPU for every step (~1.9 GB per batch of 8 at 400 tokens)
    only for step 0 to be read."""

    def __init__(self, letter_ids):
        self.letter_ids, self.probs = letter_ids, None

    def __call__(self, input_ids, scores):
        if self.probs is None:
            self.probs = torch.softmax(scores.float(), -1)[:, self.letter_ids]
        return scores


@torch.no_grad()
def generate(model, proc, prompts, images, max_new_tokens):
    texts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
                                      tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = proc(text=texts, images=images, padding=True, return_tensors="pt").to("cuda")
    first = FirstTokenLetterProbs([proc.tokenizer.convert_tokens_to_ids(l) for l in LETTERS])
    seq = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         logits_processor=LogitsProcessorList([first]))
    outs = proc.batch_decode(seq[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs, first.probs.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--fmt", required=True, choices=["qa_noev", "qa", "mcqa", "qar"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=0, help="default 64, 400 for qar")
    args = ap.parse_args()
    recs = [r for r in read_jsonl(C.p("sft", f"{args.split}_eval.jsonl")) if r["fmt"] == args.fmt]
    if args.limit:
        recs = recs[:args.limit]
    mnt = args.max_new_tokens or (400 if args.fmt == "qar" else 64)
    model, proc = load_model(args.adapter)
    out_path = C.p("preds", args.split, f"{args.name}.jsonl")
    preds = []
    for s in range(0, len(recs), args.bs):
        batch = recs[s:s + args.bs]
        outs, probs = generate(model, proc, [r["prompt"] for r in batch],
                               [load_image(r["split"], r["idx"]) for r in batch], mnt)
        for r, o, pr in zip(batch, outs, probs):
            loc, date, letter = parse_answer(o)
            opts = r.get("options") or []
            n_valid = len(opts) + 1
            conf_dist = pr[:n_valid] / max(float(pr[:n_valid].sum()), 1e-9)
            if args.fmt == "mcqa" and letter and LETTERS.index(letter) < len(opts):
                o_ = opts[LETTERS.index(letter)]
                loc, date = loc or o_["loc"], date or o_["date"]       # fallback if the final lines are missing
            preds.append({"split": r["split"], "idx": r["idx"], "fmt": args.fmt, "raw": o, "pred_loc": loc,
                          "pred_date": date, "letter": letter,
                          "letter_conf": float(conf_dist.max()) if args.fmt == "mcqa" else None})
        print(f"{min(s + args.bs, len(recs))}/{len(recs)}  last: {outs[-1][:90]!r}", flush=True)
        write_jsonl(preds, out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
