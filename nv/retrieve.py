"""Step 3 (GPU, ~1 min): bi-encoder top-K retrieval + recall evaluation.

    python -m nv.retrieve

Writes $NV_OUT/retr/{split}_bi.npz  with  rows (TARA idx), ids (M, K) corpus indices, scores (M, K)
For train images the image's own source article is excluded (it is in the corpus).
Prints/saves R@1/R@5/R@50 of location- and event-relevant articles (NewsRECON Table 2 style).
"""
import argparse
import json

import numpy as np
import torch

from . import config as C
from .common import is_event_relevant, is_loc_relevant, load_corpus, load_tara, read_json, write_json


def recall_at_k(rows_by_idx, corpus, row_ids, ids, ks=(1, 5, 10, 50)):
    res = {}
    for name, fn in [("location", is_loc_relevant), ("event", is_event_relevant)]:
        hits = {k: [] for k in ks}
        for r_idx, cand in zip(row_ids, ids):
            row = rows_by_idx[int(r_idx)]
            if not row.get("atom"):
                continue
            rel = [fn(row, corpus[int(c)]) for c in cand]
            for k in ks:
                hits[k].append(any(rel[:k]))
        res[name] = {f"R@{k}": round(100 * float(np.mean(v)), 2) for k, v in hits.items() if v}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--k", type=int, default=C.TOPK)
    args = ap.parse_args()
    corpus = load_corpus()
    url2i = {}
    for a in corpus:
        url2i.setdefault(a["url"], []).append(a["i"])
    txt = torch.from_numpy(np.load(C.p("emb", "corpus_text.npy"))).cuda()
    report = {}
    for split in args.splits.split(","):
        try:
            img = torch.from_numpy(np.load(C.p("emb", f"img_{split}.npy"))).cuda()
        except FileNotFoundError:
            print("no image embeddings for", split)
            continue
        row_ids = read_json(C.p("emb", f"img_{split}_idx.json"))
        rows = {r["idx"]: r for r in load_tara(split)}
        all_ids, all_sc = [], []
        for s in range(0, len(row_ids), 1024):
            sim = (img[s:s + 1024] @ txt.T).float()
            if split == "train":                      # never retrieve the image's own article
                for j, r_idx in enumerate(row_ids[s:s + 1024]):
                    for ci in url2i.get(rows[r_idx]["url"], []):
                        sim[j, ci] = -1e4
            sc, ids = sim.topk(args.k, dim=1)
            all_ids.append(ids.cpu().numpy())
            all_sc.append(sc.cpu().numpy())
        ids, sc = np.concatenate(all_ids), np.concatenate(all_sc)
        np.savez(C.p("retr", f"{split}_bi.npz"), rows=np.array(row_ids), ids=ids, scores=sc)
        report[split] = recall_at_k(rows, corpus, row_ids, ids)
        print(split, json.dumps(report[split]))
    write_json(report, C.p("results", "retrieval_bi_recall.json"))


if __name__ == "__main__":
    main()
