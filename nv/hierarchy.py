"""Step 9 (CPU): trust layer. Answer only as precisely as the evidence allows, with a conformal guarantee.

    python -m nv.hierarchy --name lora_mixed_mcqa                    # calibrate on dev, report on test

An MCQA run gives a probability p_k for every candidate event k (plus the "none" option). Each candidate's
location resolves to the path country > region > city, its date to year > month > day. Starting from the
most likely candidate k*, the confidence of the node at depth l of its path is the probability of every
candidate that shares that node:

    c_l = sum_k p_k * 1[path_k[:l] == path_k*[:l]]          (c_1 >= c_2 >= c_3)

The answer is k*'s node at the finest depth with c_l >= lambda, or no answer when even the country (year) is
below lambda. Along one path, being right at a fine level implies being right at every coarser one, so the
error of the committed answer can only fall as lambda grows. Conformal risk control (Angelopoulos et al.,
ICLR 2024) then picks lambda on the calibration split so that E[error] <= eps on new images, assuming the
calibration and test images are exchangeable.

Baseline "selective": the same calibration on the top candidate's own probability p_k*, without back-off
(answer in full or not at all). "finest": lambda = 0, the usual always-answer setting.
A gold answer coarser than the committed one (gold "Haiti", answer "Port-au-Prince, Haiti") is judged at
the gold's depth, as TARA's EM@1 does. Writes $NV_OUT/results/trust_{name}.json and .md
"""
import argparse

import numpy as np

from . import config as C
from .common import CONTINENTS, load_tara, pad_date, read_jsonl, write_json
from .geo import Geo

LEVELS = {"location": ("country", "region", "city"), "date": ("year", "month", "day")}
LAM = np.r_[np.linspace(0, 1, 1001), 1 + 1e-9]       # candidate thresholds; the last one never answers


def loc_path(geo, s):
    h = [x.lower() for x in geo.lookup(s)[1] if x.lower() not in CONTINENTS]
    return tuple(h[:2]) + (h[-1],) if len(h) > 2 else tuple(h)


def date_path(s):
    p = (pad_date(s) or "").split("-")
    return tuple("-".join(p[:i]) for i in range(1, len(p) + 1)) if p[0].isdigit() else ()


def build(recs, preds, gold, opt_path, gold_path):
    """-> conf (n, 3): c_l along k*'s path (-inf past its depth), top (n, 3): p_k* in the same places,
    ok (n, 3): committing at depth l would be right. Images without a gold path at this level are skipped."""
    conf, top, ok = [], [], []
    for i, rec in recs.items():
        g = gold_path(gold[i])
        if not g or i not in preds:
            continue
        probs = preds[i]["letter_probs"][:len(rec["options"])]
        paths = [opt_path(o) for o in rec["options"]]
        k = int(np.argmax(probs)) if probs else 0
        pk = paths[k] if probs else ()          # no candidates, or k* not in the gazetteer: never answers
        conf.append([sum(p for p, q in zip(probs, paths) if q[:l] == pk[:l]) if l <= len(pk) else -np.inf
                     for l in (1, 2, 3)])
        top.append([probs[k] if l <= len(pk) else -np.inf for l in (1, 2, 3)])
        ok.append([pk[:min(l, len(g))] == g[:min(l, len(g))] for l in (1, 2, 3)])
    return np.array(conf).reshape(-1, 3), np.array(top).reshape(-1, 3), np.array(ok, bool).reshape(-1, 3)


def commit(conf, ok, lam):
    """Committed depth (0 = no answer) and error for every image and threshold: two (n, len(lam)) arrays."""
    d = (conf[:, None, :] >= lam[None, :, None]).sum(-1)
    wrong = ~np.take_along_axis(ok, np.maximum(d - 1, 0), 1) & (d > 0)
    return d, wrong


def crc(conf, ok, eps):
    """Smallest lambda with (n * empirical risk + 1) / (n + 1) <= eps (loss bounded by 1)."""
    n = len(conf)
    feasible = (n * commit(conf, ok, LAM)[1].mean(0) + 1) / (n + 1) <= eps
    return float(LAM[np.argmax(feasible)])


def report(conf, ok, lam):
    d, wrong = commit(conf, ok, np.array([lam]))
    d, wrong = d[:, 0], wrong[:, 0]
    return {"lambda": lam, "error": 100 * float(wrong.mean()), "answered": 100 * float((d > 0).mean()),
            "depth_%": [100 * float((d == l).mean()) for l in (1, 2, 3)], "specificity": 100 * float(d.mean() / 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lora_mixed_mcqa", help="MCQA prediction file in preds/<split>/")
    ap.add_argument("--cal", default="dev")
    ap.add_argument("--test", default="test")
    ap.add_argument("--eps", default="0.05,0.1,0.2", help="target error rates")
    args = ap.parse_args()
    geo = Geo()
    paths = {"location": (lambda o: loc_path(geo, o["loc"]),
                          lambda r: loc_path(geo, r["target_loc"] or r["gold_loc"]) if r["gold_loc"] else ()),
             "date": (lambda o: date_path(o["date"]), lambda r: date_path(r["gold_date"]))}
    data = {}
    for split in (args.cal, args.test):
        recs = {r["idx"]: r for r in read_jsonl(C.p("sft", f"{split}_eval.jsonl")) if r["fmt"] == "mcqa"}
        preds = {p["idx"]: p for p in read_jsonl(C.p("preds", split, f"{args.name}.jsonl"))}
        if any(p.get("letter_probs") is None for p in preds.values()):
            raise SystemExit(f"{args.name} on {split} has no letter_probs: re-run nv.infer (--fmt mcqa)")
        gold = {r["idx"]: r for r in load_tara(split)}
        data[split] = {dim: build(recs, preds, gold, *fn) for dim, fn in paths.items()}

    out = {"name": args.name, "cal": args.cal, "test": args.test, "levels": LEVELS, "dims": {}}
    md = [f"## Trust layer: `{args.name}`, lambda calibrated on {args.cal}, scored on {args.test}", "",
          "| dim | eps | method | lambda | error % | answered % | depth 1/2/3 % | specificity % |",
          "|---|---|---|---|---|---|---|---|"]
    for dim in paths:
        (cc, ct, co), (tc, tt, to) = data[args.cal][dim], data[args.test][dim]
        res = {"n_cal": len(cc), "n_test": len(tc), "finest": report(tc, to, 0.0), "eps": {},
               "curve": {"lambda": LAM[::10].tolist(),
                         **{m: {"error": (100 * w.mean(0)).tolist(), "specificity": (100 * d.mean(0) / 3).tolist()}
                            for m, (d, w) in (("ours", commit(tc, to, LAM[::10])),
                                              ("selective", commit(tt, to, LAM[::10])))}}}
        rows = [("-", "finest", res["finest"])]
        for eps in map(float, args.eps.split(",")):
            res["eps"][eps] = {"ours": report(tc, to, crc(cc, co, eps)), "selective": report(tt, to, crc(ct, co, eps))}
            rows += [(eps, m, r) for m, r in res["eps"][eps].items()]
        out["dims"][dim] = res
        for eps, m, r in rows:
            md.append(f"| {dim} | {eps} | {m} | {r['lambda']:.3f} | {r['error']:.1f} | {r['answered']:.1f} | "
                      + "/".join(f"{x:.0f}" for x in r["depth_%"]) + f" | {r['specificity']:.1f} |")
    md.append(f"\nImages with a gold answer: location {out['dims']['location']['n_cal']} {args.cal} / "
              f"{out['dims']['location']['n_test']} {args.test}, date {out['dims']['date']['n_cal']} / "
              f"{out['dims']['date']['n_test']}. Depth 1/2/3 = {'/'.join(LEVELS['location'])} or "
              f"{'/'.join(LEVELS['date'])}.")
    text = "\n".join(md)
    print(text)
    with open(C.p("results", f"trust_{args.name}.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    write_json(out, C.p("results", f"trust_{args.name}.json"))


if __name__ == "__main__":
    main()
