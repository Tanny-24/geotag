"""Step 4 (GPU or CPU, ~2-5 min): NewsRECON location cross-encoder on frozen CLIP embeddings.

    python -m nv.cross_encoder

Input  : frozen CLIP image embedding  +  frozen CLIP embedding of "An image from {LOCATION}"
Head   : [i, t, i*t, |i-t|] -> MLP -> match logit, BCE loss (NewsRECON: frozen encoders + trainable layer)
Pairs  : per train image, 1 location-relevant + up to 7 non-relevant articles from its bi-encoder top-50
Final  : s_comb(a) = s_bi(a) * sigmoid(s_loc(a))   (NewsRECON multiplies the two scores)

Train images are scored with 2-fold cross-fitting (head trained on the other half), so the MLLM training
data sees evidence as noisy as at test time. Dev is used to pick the epoch; dev/test use a head trained on
all train images.

Writes $NV_OUT/retr/{split}_final.npz (rows, ids re-ranked, scores) and results/retrieval_ce_recall.json
"""
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .common import is_loc_relevant, load_corpus, load_tara, read_json, write_json
from .retrieve import recall_at_k


class LocHead(nn.Module):
    def __init__(self, d=768, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * d, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, i, t):
        return self.net(torch.cat([i, t, i * t, (i - t).abs()], -1)).squeeze(-1)


def load_split(split, corpus):
    bi = np.load(C.p("retr", f"{split}_bi.npz"))
    img = np.load(C.p("emb", f"img_{split}.npy")).astype(np.float32)
    rows = {r["idx"]: r for r in load_tara(split)}
    rel = np.array([[is_loc_relevant(rows[int(r)], corpus[int(c)]) for c in cand]
                    for r, cand in zip(bi["rows"], bi["ids"])], dtype=bool)
    return dict(rows=bi["rows"], ids=bi["ids"], bi=bi["scores"], img=img, rel=rel, meta=rows)


@torch.no_grad()
def score(head, S, loc_emb, dev="cuda"):
    head.eval()
    out = np.zeros_like(S["bi"], dtype=np.float32)
    for s in range(0, len(S["ids"]), 256):
        ids = S["ids"][s:s + 256]
        i = torch.from_numpy(S["img"][s:s + 256]).to(dev)[:, None, :].expand(-1, ids.shape[1], -1)
        t = loc_emb[torch.from_numpy(ids).to(dev)]
        out[s:s + 256] = torch.sigmoid(head(i, t)).cpu().numpy()
    return out


def rerank(S, ce, alpha=1.0):
    comb = S["bi"] * np.power(np.clip(ce, 1e-6, 1.0), alpha)   # alpha=1: NewsRECON product, alpha=0: bi only
    order = np.argsort(-comb, axis=1)
    return np.take_along_axis(S["ids"], order, 1), np.take_along_axis(comb, order, 1)


def train_head(S, mask, loc_emb, dev_S, epochs, dev="cuda", n_neg=7, seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    head = LocHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    items = [j for j in np.where(mask)[0] if S["rel"][j].any()]
    best, best_state = -1, None
    for ep in range(epochs):
        head.train()
        random.shuffle(items)
        tot = 0
        for s in range(0, len(items), 256):
            I, T, Y = [], [], []
            for j in items[s:s + 256]:
                pos = np.where(S["rel"][j])[0]
                neg = np.where(~S["rel"][j])[0]
                pick = [random.choice(pos)] + random.sample(list(neg), min(n_neg, len(neg)))
                for k in pick:
                    I.append(j)
                    T.append(S["ids"][j, k])
                    Y.append(float(S["rel"][j, k]))
            i = torch.from_numpy(S["img"][I]).to(dev)
            t = loc_emb[torch.tensor(T, device=dev)]
            loss = lossf(head(i, t), torch.tensor(Y, device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if dev_S is not None:
            ce = score(head, dev_S, loc_emb, dev)
            order = np.argsort(-(dev_S["bi"] * ce), axis=1)
            rel = np.take_along_axis(dev_S["rel"], order, 1)
            r1 = 100 * rel[:, 0].mean()
            r5 = 100 * rel[:, :5].any(1).mean()
            print(f"  epoch {ep + 1}: loss {tot:.2f}  dev loc R@1 {r1:.2f}  R@5 {r5:.2f}")
            if r1 > best:
                best, best_state = r1, {k: v.clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    corpus = load_corpus()
    loc_emb = torch.from_numpy(np.load(C.p("emb", "corpus_loc.npy")).astype(np.float32)).to(dev)
    tr = load_split("train", corpus)
    dv = load_split("dev", corpus)
    report = {}

    print("full head (for dev/test):")
    full = train_head(tr, np.ones(len(tr["rows"]), bool), loc_emb, dv, args.epochs, dev)
    torch.save(full.state_dict(), C.p("ckpt", "loc_head.pt"))

    # how much weight the cross-encoder gets: chosen on dev (R@1 + R@5, location)
    ce_dv = score(full, dv, loc_emb, dev)
    report["alpha_search_dev"] = {}
    best_alpha, best = 1.0, -1
    for a in (0.0, 0.1, 0.25, 0.5, 1.0):
        order = np.argsort(-(dv["bi"] * np.power(np.clip(ce_dv, 1e-6, 1), a)), axis=1)
        rel = np.take_along_axis(dv["rel"], order, 1)
        r1, r5 = 100 * rel[:, 0].mean(), 100 * rel[:, :5].any(1).mean()
        report["alpha_search_dev"][str(a)] = {"R@1": round(r1, 2), "R@5": round(r5, 2)}
        print(f"alpha {a}: dev loc R@1 {r1:.2f} R@5 {r5:.2f}")
        if r1 + r5 > best:
            best, best_alpha = r1 + r5, a
    report["alpha"] = best_alpha
    print("using alpha =", best_alpha)

    # 2-fold cross-fitting for train
    fold = (tr["rows"] % 2).astype(int)
    ce_tr = np.zeros_like(tr["bi"], dtype=np.float32)
    for f in (0, 1):
        print(f"fold head trained on fold {1 - f}, applied to fold {f}:")
        h = train_head(tr, fold != f, loc_emb, dv, args.epochs, dev, seed=f + 1)
        ce_tr[fold == f] = score(h, {k: (v[fold == f] if isinstance(v, np.ndarray) else v) for k, v in tr.items()},
                                 loc_emb, dev)
    ids, sc = rerank(tr, ce_tr, best_alpha)
    np.savez(C.p("retr", "train_final.npz"), rows=tr["rows"], ids=ids, scores=sc)

    for split in ("dev", "test"):
        try:
            S = dv if split == "dev" else load_split(split, corpus)
        except FileNotFoundError:
            print("skip", split)
            continue
        ids, sc = rerank(S, score(full, S, loc_emb, dev), best_alpha)
        np.savez(C.p("retr", f"{split}_final.npz"), rows=S["rows"], ids=ids, scores=sc)
        report[split] = recall_at_k(S["meta"], corpus, S["rows"], ids)
        print(split, "after rerank", json.dumps(report[split]))
    report["train_crossfitted"] = recall_at_k(tr["meta"], corpus, tr["rows"],
                                              np.load(C.p("retr", "train_final.npz"))["ids"])
    write_json(report, C.p("results", "retrieval_ce_recall.json"))


if __name__ == "__main__":
    main()
