"""Step 5 (CPU, ~1 min): LiveVQA-style data engine on top of NewsRECON-lite retrieval.

    python -m nv.build_sft --n_train 6000 --mix mcqa=0.4,qa=0.3,qar=0.3 --noise 0.3 --n_eval 500

Writes
  $NV_OUT/sft/train.jsonl            training samples {fmt, split, idx, prompt, target}
  $NV_OUT/sft/{dev,test}_eval.jsonl  evaluation prompts for every format on a fixed random subset
  $NV_OUT/preds/{split}/retrieval_*.jsonl   retrieval-only baselines (no MLLM) on the same subset
  $NV_OUT/sft/stats.json

Supervision comes only from weak relevance labels and TARA gold answers, never from another LLM:
  * MCQA options are distinct (location, date) events clustered from the re-ranked list (hard negatives);
    the correct letter is the first event-relevant option, else the first location-relevant one,
    else the final "none" option.
  * QAR reasoning is a template over per-article verdicts (same event / same place, different date /
    different place), followed by a conflict-resolution sentence.
  * With probability --noise the evidence is replaced by non-relevant articles from ranks 10-50, so the
    model sees cases where the right move is to reject the evidence (anti "blind trust").
"""
import argparse
import os
import random
from collections import Counter

import numpy as np

from . import config as C
from .common import (candidate_events, date_gap_days, is_event_relevant, is_loc_relevant, load_corpus,
                     load_tara, write_json, write_jsonl)
from .prompts import LETTERS, answer_lines, build_prompt


def ranked_lists(split, corpus, kind="final"):
    f = C.p("retr", f"{split}_{kind}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return {int(r): [corpus[int(i)] for i in ids] for r, ids in zip(z["rows"], z["ids"])}


def option_correct(row, opt):
    pseudo = {"atoms": opt["art"]["atoms"], "date": opt["date"]}
    return is_event_relevant(row, pseudo), is_loc_relevant(row, pseudo)


def pick_letter(row, options):
    flags = [option_correct(row, o) for o in options]
    for k, (evt, _) in enumerate(flags):
        if evt:
            return LETTERS[k], "event"
    for k, (_, loc) in enumerate(flags):
        if loc:
            return LETTERS[k], "location"
    return LETTERS[len(options)], "none"


def gap_text(g):
    if g is None:
        return "at an unknown distance from the photo's date"
    if g == 0:
        return "on the day of the photo"
    return f"{abs(g)} days {'after' if g > 0 else 'before'} the photo"


def qar_target(row, arts):
    lines, evt_k, loc_k = [], None, None
    for k, a in enumerate(arts, 1):
        tag = f"Article {k} ({a['loc_str']}, {a['date']})"
        if is_event_relevant(row, a):
            lines.append(f"{tag}: same place, published {gap_text(date_gap_days(row, a))}; consistent with the photo, "
                         f"likely the same event.")
            evt_k = evt_k or k
        elif is_loc_relevant(row, a):
            lines.append(f"{tag}: same place, but published {gap_text(date_gap_days(row, a))}; a different event, "
                         f"so I use it for the place only.")
            loc_k = loc_k or k
        else:
            lines.append(f"{tag}: its location does not fit the photo; rejected.")
    if evt_k:
        concl = f"Article {evt_k} reports this event, so I follow it for both place and date."
    elif loc_k:
        concl = f"Article {loc_k} matches the place but not the event; the date is estimated from the image."
    else:
        concl = "None of the articles match; the answer relies on visual cues in the image alone."
    return "\n".join(lines) + f"\nConclusion: {concl}\n" + answer_lines(row["target_loc"], row["gold_date"])


def make_example(row, ranked, fmt, noisy, rng):
    if noisy:
        pool = [a for a in ranked[10:] if not is_loc_relevant(row, a)]
        rng.shuffle(pool)
        arts, cand_src = pool[:C.N_EVIDENCE], pool
    else:
        arts, cand_src = ranked[:C.N_EVIDENCE], ranked[:20]
    ex = {"split": row["split"], "idx": row["idx"], "fmt": fmt, "noisy": noisy}
    if fmt == "qa":
        ex["prompt"] = build_prompt("qa", arts)
        ex["target"] = answer_lines(row["target_loc"], row["gold_date"])
    elif fmt == "mcqa":
        options = candidate_events(cand_src)
        letter, kind = pick_letter(row, options)
        ex["prompt"] = build_prompt("mcqa", arts, options)
        ex["target"] = f"{letter}\n" + answer_lines(row["target_loc"], row["gold_date"])
        ex["letter_kind"] = kind
    elif fmt == "qar":
        ex["prompt"] = build_prompt("qar", arts)
        ex["target"] = qar_target(row, arts)
    ex["top3_loc"] = sum(is_loc_relevant(row, a) for a in arts)
    ex["top3_evt"] = sum(is_event_relevant(row, a) for a in arts)
    return ex


def retrieval_pred(row, ranked, mode):
    if mode == "cluster":
        ev = candidate_events(ranked[:20], n=1)
        if ev:
            return {"pred_loc": ", ".join(ev[0]["art"]["glocs"]), "pred_date": ev[0]["date"]}
    top = ranked[0]
    return {"pred_loc": ", ".join(top["glocs"]), "pred_date": top["date"]}   # NewsRECON get_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--mix", default="mcqa=0.4,qa=0.3,qar=0.3")
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    corpus = load_corpus()
    stats = {}

    # ---------------- train
    ranked = ranked_lists("train", corpus)
    rows = [r for r in load_tara("train") if r["idx"] in ranked and r["target_loc"] and r["gold_date"]]
    rng.shuffle(rows)
    rows = rows[:args.n_train]
    mix = {k: float(v) for k, v in (x.split("=") for x in args.mix.split(","))}
    fmts = list(mix)
    out = []
    for r in rows:
        fmt = rng.choices(fmts, weights=[mix[f] for f in fmts])[0]
        out.append(make_example(r, ranked[r["idx"]], fmt, rng.random() < args.noise, rng))
    write_jsonl(out, C.p("sft", "train.jsonl"))
    stats["train"] = {
        "n": len(out), "formats": Counter(e["fmt"] for e in out), "noisy": sum(e["noisy"] for e in out),
        "mcqa_letter_kind": Counter(e.get("letter_kind") for e in out if e["fmt"] == "mcqa"),
        "top3_precision_loc": round(sum(e["top3_loc"] for e in out) / (3 * len(out)), 3),
        "top3_precision_evt": round(sum(e["top3_evt"] for e in out) / (3 * len(out)), 3),
    }

    # ---------------- eval prompts + retrieval-only baselines
    for split in ("dev", "test"):
        ranked = ranked_lists(split, corpus)
        if ranked is None:
            continue
        bi = ranked_lists(split, corpus, "bi")
        rows = [r for r in load_tara(split) if r["idx"] in ranked and (r["gold_loc"] or r["gold_date"])]
        rows = sorted(rows, key=lambda r: r["idx"])
        random.Random(1234).shuffle(rows)
        rows = sorted(rows[:args.n_eval], key=lambda r: r["idx"])
        recs, p_loc, p_evt = [], 0, 0
        for r in rows:
            arts = ranked[r["idx"]][:C.N_EVIDENCE]
            options = candidate_events(ranked[r["idx"]][:20])
            p_loc += sum(is_loc_relevant(r, a) for a in arts)
            p_evt += sum(is_event_relevant(r, a) for a in arts)
            for fmt in ("qa_noev", "qa", "mcqa", "qar"):
                recs.append({"split": split, "idx": r["idx"], "fmt": fmt, "options": options if fmt == "mcqa" else None,
                             "prompt": build_prompt(fmt, arts, options)})
        for rec in recs:   # options carry whole articles; keep only what inference needs
            if rec["options"]:
                rec["options"] = [{"loc": o["loc"], "date": o["date"]} for o in rec["options"]]
        write_jsonl(recs, C.p("sft", f"{split}_eval.jsonl"))
        for name, src, mode in [("retrieval_bi", bi, "top1"), ("retrieval_ce", ranked, "top1"),
                                ("retrieval_ce_cluster", ranked, "cluster")]:
            preds = [dict(split=split, idx=r["idx"], fmt=name, **retrieval_pred(r, src[r["idx"]], mode)) for r in rows]
            write_jsonl(preds, C.p("preds", split, f"{name}.jsonl"))
        stats[split] = {"n_eval": len(rows), "top3_precision_loc": round(p_loc / (3 * len(rows)), 3),
                        "top3_precision_evt": round(p_evt / (3 * len(rows)), 3)}
    write_json(stats, C.p("sft", "stats.json"))
    import json
    print(json.dumps(stats, indent=2, default=str))
    print("If train top-3 precision is far above dev, raise --noise.")


if __name__ == "__main__":
    main()
