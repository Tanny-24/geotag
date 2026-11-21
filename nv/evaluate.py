"""Step 8 (CPU): score every prediction file of a split and write a results table.

    python -m nv.evaluate --split test --baseline zs_noev [--geonames_user YOURNAME] [--names a,b,c]

Outputs $NV_OUT/results/{split}_results.md / .json with, per run:
  date : EM@1, E-F1, Delta, GREAT_date      location : EM@1, E-F1, CO-delta, GREAT_loc      GREAT (mean)
plus
  * flip analysis vs --baseline (NewsRECON Sec. 4.8.4 "blind trust"): location/date answers that were
    correct (EM@1) in the baseline and became wrong, and vice versa
  * calibration of the MCQA first-token option confidence: ECE (10 bins) against location EM@1
  * 95% bootstrap interval of GREAT (1,000 image resamples, shared by all runs so differences are paired)
    and its gain over --baseline with interval
  * GREAT on the --subset images build_sft would pick with --n_eval <subset> (seed-1234 shuffle), i.e. the
    teammates' 500-image subset, so the numbers can be set next to results/
All runs are scored on the indices they share, so the table is a fair comparison.
"""
import argparse
import glob
import os
import random

import numpy as np

from . import config as C
from .common import load_tara, read_jsonl, write_json
from .geo import Geo
from .metrics import score_date, score_location

ORDER = ["retrieval_bi", "retrieval_ce", "retrieval_ce_cluster", "zs_noev", "zs_qa", "zs_mcqa", "zs_qar"]


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--names", default="", help="comma list; default = every file in preds/<split>/")
    ap.add_argument("--baseline", default="zs_noev")
    ap.add_argument("--geonames_user", default=None)
    ap.add_argument("--subset", type=int, default=500, help="0 disables the subset column")
    args = ap.parse_args()
    gold = {r["idx"]: r for r in load_tara(args.split)}
    pdir = C.p("preds", args.split, "x")[:-1]
    names = args.names.split(",") if args.names else sorted(
        (os.path.basename(f)[:-6] for f in glob.glob(os.path.join(pdir, "*.jsonl"))),
        key=lambda n: (ORDER.index(n) if n in ORDER else len(ORDER), n))
    runs = {n: {p["idx"]: p for p in read_jsonl(os.path.join(pdir, n + ".jsonl"))} for n in names}
    common = set.intersection(*(set(r) for r in runs.values()))
    print(f"{len(names)} runs, scored on {len(common)} shared images")
    geo = Geo(username=args.geonames_user)
    idx = sorted(common)
    boot = np.random.default_rng(0).integers(0, len(idx), (1000, len(idx)))
    sub = np.zeros(len(idx), bool)
    eval_file = C.p("sft", f"{args.split}_eval.jsonl")
    if args.subset and os.path.exists(eval_file):
        order = sorted({r["idx"] for r in read_jsonl(eval_file)})   # same list and shuffle as build_sft
        random.Random(1234).shuffle(order)
        sub = np.isin(idx, order[:args.subset])

    per = {}
    table = []
    great = {}                                 # run -> (per-image GREAT_loc, GREAT_date), nan where no gold
    for n in names:
        L, D, conf = {}, {}, []
        for i in sorted(common):
            g, p = gold[i], runs[n][i]
            if g["gold_loc"]:
                L[i] = score_location(p.get("pred_loc"), g["gold_loc"], geo)
            if g["gold_date"]:
                D[i] = score_date(p.get("pred_date"), g["gold_date"])
            if p.get("letter_conf") is not None and i in L:
                conf.append((p["letter_conf"], L[i]["em"]))
        per[n] = (L, D)
        m = lambda d, k: 100 * float(np.mean([v[k] for v in d.values()])) if d else float("nan")
        row = {"run": n, "n_loc": len(L), "n_date": len(D),
               "date_EM@1": m(D, "em"), "date_E-F1": m(D, "ef1"), "Delta": m(D, "delta"), "GREAT_date": m(D, "great"),
               "loc_EM@1": m(L, "em"), "loc_E-F1": m(L, "ef1"), "CO-delta": m(L, "codelta"), "GREAT_loc": m(L, "great")}
        row["GREAT"] = (row["GREAT_date"] + row["GREAT_loc"]) / 2
        gl, gd = (np.array([s[i]["great"] if i in s else np.nan for i in idx]) for s in (L, D))
        great[n] = 50 * (np.nanmean(gl[boot], 1) + np.nanmean(gd[boot], 1))     # GREAT of each resample
        row["GREAT_95CI"] = np.percentile(great[n], [2.5, 97.5]).tolist()
        if sub.any():
            row[f"GREAT@{args.subset}"] = 50 * (np.nanmean(gl[sub]) + np.nanmean(gd[sub]))
        if conf:
            row["MCQA_ECE"] = 100 * ece([c for c, _ in conf], [y for _, y in conf])
            row["MCQA_mean_conf"] = 100 * float(np.mean([c for c, _ in conf]))
        table.append(row)
    geo.save()
    if args.baseline in great:
        base = next(r["GREAT"] for r in table if r["run"] == args.baseline)
        for r in table:
            if r["run"] != args.baseline:
                r["GREAT_gain"] = r["GREAT"] - base
                r["GREAT_gain_95CI"] = np.percentile(great[r["run"]] - great[args.baseline], [2.5, 97.5]).tolist()

    flips = {}
    if args.baseline in per:
        bL, bD = per[args.baseline]
        for n in names:
            if n == args.baseline:
                continue
            L, D = per[n]
            f = {}
            for task, b, x in (("loc", bL, L), ("date", bD, D)):
                ks = set(b) & set(x)
                f[f"{task}_correct_to_wrong"] = sum(b[k]["em"] == 1 and x[k]["em"] == 0 for k in ks)
                f[f"{task}_wrong_to_correct"] = sum(b[k]["em"] == 0 and x[k]["em"] == 1 for k in ks)
            flips[n] = f

    cols = ["run", "date_EM@1", "date_E-F1", "Delta", "GREAT_date", "loc_EM@1", "loc_E-F1", "CO-delta", "GREAT_loc",
            "GREAT", "MCQA_ECE"]
    md = [f"## TARA {args.split} — {len(common)} images (location n={table[0]['n_loc']}, date n={table[0]['n_date']})", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in table:
        md.append("| " + " | ".join(r["run"] if c == "run" else ("" if c not in r else f"{r[c]:.1f}") for c in cols) + " |")
    ci = lambda r, k: f"[{r[k][0]:.1f}, {r[k][1]:.1f}]" if k in r else ""
    sk = f"GREAT@{args.subset}"
    md += ["", f"### GREAT with 95% bootstrap interval (1,000 resamples of the {len(idx)} images)"
               + (f"; `{sk}` = the {int(sub.sum())}-image subset of `results/`" if sub.any() else ""), "",
           f"| run | GREAT | 95% CI | gain vs `{args.baseline}` | 95% CI | {sk} |", "|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {r['run']} | {r['GREAT']:.1f} | {ci(r, 'GREAT_95CI')} | "
                  + (f"{r['GREAT_gain']:+.1f}" if "GREAT_gain" in r else "") + f" | {ci(r, 'GREAT_gain_95CI')} | "
                  + (f"{r[sk]:.1f}" if sk in r else "") + " |")
    if flips:
        md += ["", f"### Flips vs `{args.baseline}` (EM@1)", "",
               "| run | loc correct→wrong | loc wrong→correct | date correct→wrong | date wrong→correct |",
               "|---|---|---|---|---|"]
        for n, f in flips.items():
            md.append(f"| {n} | {f['loc_correct_to_wrong']} | {f['loc_wrong_to_correct']} | "
                      f"{f['date_correct_to_wrong']} | {f['date_wrong_to_correct']} |")
    text = "\n".join(md)
    print(text)
    with open(C.p("results", f"{args.split}_results.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    write_json({"table": table, "flips": flips, "n_images": len(common)}, C.p("results", f"{args.split}_results.json"))


if __name__ == "__main__":
    main()
