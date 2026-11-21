"""Collect the numbers the paper tables need into results/metrics_summary.json (kept under 10 KB).

    python scripts/extract_metrics.py                          # committed copies in results/
    python scripts/extract_metrics.py --src runs/results       # straight from a pipeline run
"""
import argparse
import json
import os

KEEP = ["date_EM@1", "GREAT_date", "loc_EM@1", "GREAT_loc", "GREAT", "GREAT_95CI", "GREAT_gain", "GREAT_gain_95CI",
        "GREAT@500", "MCQA_ECE"]
TRUST = ["lambda", "error", "answered", "specificity"]


def r1(v):
    return [round(x, 1) for x in v] if isinstance(v, list) else round(v, 1)


def load(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/metrics_summary.json")
    args = ap.parse_args()
    res = load(os.path.join(args.src, f"{args.split}_results.json"))
    bi = load(os.path.join(args.src, "retrieval_bi_recall.json"))
    ce = load(os.path.join(args.src, "retrieval_ce_recall.json"))
    trust = load(os.path.join(args.src, "trust_lora_mixed_mcqa.json"))
    runs = {r["run"]: {k: r1(r[k]) for k in KEEP if isinstance(r.get(k), (int, float, list)) and r[k] == r[k]}
            for r in res.get("table", [])}                               # r[k] == r[k] drops NaN
    summary = {
        "split": args.split,
        "n_images": res.get("n_images"),
        "runs": runs,
        "flips_vs_baseline": res.get("flips", {}),
        "retrieval_location": {"bi": bi.get(args.split, {}).get("location"),
                               "reranked": ce.get(args.split, {}).get("location"), "alpha": ce.get("alpha")},
        "trust_layer": {dim: {"n_test": d["n_test"], "finest": {k: r1(d["finest"][k]) for k in TRUST},
                              **{eps: {m: {k: r1(v[k]) for k in TRUST} for m, v in by.items()}
                                 for eps, by in d["eps"].items()}}
                        for dim, d in trust.get("dims", {}).items()},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    size = os.path.getsize(args.out)
    print(f"wrote {args.out}: {len(runs)} runs, {size} bytes")
    if size > 10_000:
        print("warning: over 10 KB")


if __name__ == "__main__":
    main()
