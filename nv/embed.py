"""Step 2 (GPU, ~10-15 min): frozen CLIP ViT-L/14 embeddings.

    python -m nv.embed                    # corpus texts + location templates + train/dev/test images

Writes $NV_OUT/emb/
    corpus_text.npy   (N, 768) fp16, L2-normalised, "headline. abstract"  (bi-encoder side)
    corpus_loc.npy    (N, 768) fp16, "An image from {LOCATION}"           (location cross-encoder side)
    img_{split}.npy   (M, 768) fp16 and img_{split}_idx.json (TARA row indices whose image exists)

NewsRECON fine-tunes CLIP on Qwen-generated news captions. We have no captions and no time, so we use
frozen CLIP on headline+abstract (their Table 3: abstract is the best input before fine-tuning).
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from . import config as C
from .common import load_corpus, load_tara, write_json


@torch.no_grad()
def encode_texts(model, proc, texts, bs=512, device="cuda"):
    out = []
    for s in tqdm(range(0, len(texts), bs), desc="text"):
        enc = proc.tokenizer(texts[s:s + bs], padding=True, truncation=True, max_length=77, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            e = model.text_projection(model.text_model(**enc).pooler_output)  # stable across HF versions
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).half().cpu())
    return torch.cat(out).numpy()


class ImgSet(Dataset):
    def __init__(self, paths, proc):
        self.paths, self.proc = paths, proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.proc(images=img, return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def encode_images(model, proc, paths, bs=128, workers=8, device="cuda"):
    dl = DataLoader(ImgSet(paths, proc), batch_size=bs, num_workers=workers)
    out = []
    for px in tqdm(dl, desc="images"):
        with torch.autocast("cuda", dtype=torch.float16):
            e = model.visual_projection(model.vision_model(pixel_values=px.to(device, torch.float16)).pooler_output)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).half().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--skip_corpus", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    model = CLIPModel.from_pretrained(C.CLIP_MODEL, torch_dtype=torch.float16).cuda().eval()
    proc = CLIPProcessor.from_pretrained(C.CLIP_MODEL)

    if not args.skip_corpus:
        corpus = load_corpus()
        texts = [f"{a['head']}. {a['abs']}" for a in corpus]
        np.save(C.p("emb", "corpus_text.npy"), encode_texts(model, proc, texts))
        locs = [f"An image from {a['loc_str']}" for a in corpus]
        np.save(C.p("emb", "corpus_loc.npy"), encode_texts(model, proc, locs))

    for split in args.splits.split(","):
        rows = load_tara(split)
        keep = [r["idx"] for r in rows if os.path.exists(C.tara_image(split, r["idx"]))]
        print(f"{split}: {len(keep)}/{len(rows)} images found")
        if not keep:
            continue
        emb = encode_images(model, proc, [C.tara_image(split, i) for i in keep], workers=args.workers)
        np.save(C.p("emb", f"img_{split}.npy"), emb)
        write_json(keep, C.p("emb", f"img_{split}_idx.json"))


if __name__ == "__main__":
    main()
