"""Pick example photos for the web app from the TARA test set: photos where the trust layer (evidence model) answers
at city level and is right, backs off to region or country and is right while its city would be wrong, or gives
no answer where even its country would be wrong. Within each case the model's most confident photos come first.
Prints each case with gold labels and captions; the last line is a comma list for `demo.py --examples`.

    python -m scripts.pick_examples [--eps 0.1] [--per 5]
"""
import argparse

import numpy as np

from nv import config as C
from nv.build_sft import ranked_lists
from nv.common import load_corpus, load_tara, read_json, read_jsonl
from nv.geo import Geo
from nv.hierarchy import build, evidence_conf, features, loc_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lora_mixed_mcqa")
    ap.add_argument("--eps", default="0.1", help="allowed error, a key of the trust-layer results")
    ap.add_argument("--per", type=int, default=5, help="photos listed per case")
    args = ap.parse_args()

    geo = Geo()
    recs = {r["idx"]: r for r in read_jsonl(C.p("sft", "test_eval.jsonl")) if r["fmt"] == "mcqa"}
    preds = {p["idx"]: p for p in read_jsonl(C.p("preds", "test", f"{args.name}.jsonl"))}
    gold = {r["idx"]: r for r in load_tara("test")}
    gold_path = lambda r: loc_path(geo, r["target_loc"] or r["gold_loc"]) if r["gold_loc"] else ()
    conf, top, ev, ok = build(recs, preds, gold, ranked_lists("test", load_corpus()),
                              lambda o: loc_path(geo, o["loc"]), gold_path, lambda a: loc_path(geo, a["loc_str"]))
    idx = [i for i in recs if gold_path(gold[i]) and i in preds]        # the rows build() keeps, in its order
    city = np.array([len(gold_path(gold[i])) == 3 for i in idx]) & np.isfinite(conf[:, 2])   # gold and k* have a city

    trust = read_json(C.p("results", f"trust_{args.name}.json"))["dims"]["location"]
    s = evidence_conf(np.array(list(trust["weights"]["evidence"].values())), features(conf, top, ev), conf)
    d = (s >= trust["eps"][args.eps]["evidence"]["lambda"]).sum(1)      # committed depth, as in demo.py
    right = np.take_along_axis(ok, np.maximum(d - 1, 0)[:, None], 1)[:, 0] & (d > 0)
    cases = {"city level, right": city & (d == 3) & right,
             "coarser level, right (its city would be wrong)": city & (d > 0) & (d < 3) & right & ~ok[:, 2],
             "no answer (even its country would be wrong)": np.isfinite(conf[:, 0]) & (d == 0) & ~ok[:, 0]}

    picked = []
    for name, m in cases.items():
        rows = np.flatnonzero(m)[np.argsort(-top[m, 0])][:args.per]
        print(f"\n{name}: {m.sum()} of {len(idx)} test photos")
        for j in rows:
            g = gold[idx[j]]
            print(f"  {idx[j]:5d}  {g['gold_loc']} | {g['gold_date']} | model {top[j, 0]:.0%} sure | {g['caption'][:80]}")
        picked += [idx[j] for j in rows[:2]]
    print("\nExamples for demo.py --examples:")
    print(",".join(map(str, picked)))


if __name__ == "__main__":
    main()
