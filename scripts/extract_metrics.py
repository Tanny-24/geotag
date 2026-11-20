"""Collect the numbers the paper tables need into results/metrics_summary.json (kept under 10 KB).

    python scripts/extract_metrics.py                          # committed copies in results/
    python scripts/extract_metrics.py --src runs/results       # straight from a pipeline run
"""
import argparse
import json
import os

KEEP = ["date_EM@1", "GREAT_date", "loc_EM@1", "GREAT_loc", "GREAT", "MCQA_ECE"]


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
    runs = {r["run"]: {k: round(r[k], 1) for k in KEEP if isinstance(r.get(k), (int, float)) and r[k] == r[k]}
            for r in res.get("table", [])}                               # r[k] == r[k] drops NaN
    summary = {
        "split": args.split,
        "n_images": res.get("n_images"),
        "runs": runs,
        "flips_vs_baseline": res.get("flips", {}),
        "retrieval_location": {"bi": bi.get(args.split, {}).get("location"),
                               "reranked": ce.get(args.split, {}).get("location"), "alpha": ce.get("alpha")},
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
