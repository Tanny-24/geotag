"""Step 9 (CPU): trust layer. Answer only as precisely as the evidence allows, with a conformal guarantee.

    python -m nv.hierarchy --name lora_mixed_mcqa                     # dev calibrates, test scores
    python -m nv.hierarchy --name lora_mixed_mcqa --resplits 200      # plus 200 pooled random re-splits

An MCQA run gives a probability p_k for every candidate event k (plus the "none" option). Each candidate's
location resolves to the path country > region > city, its date to year > month > day. Starting from the
most likely candidate k*, the model's confidence in the node at depth l of its path is the probability of
every candidate that shares that node:

    c_l = sum_k p_k * 1[path_k[:l] == path_k*[:l]]          (c_1 >= c_2 >= c_3)

Evidence-aware confidence: the retrieved articles give a second signal that does not come from the model,
the weighted share of the top-20 re-ranked articles whose own location (publication date) agrees with that
node,

    e_l = sum_a w_a * 1[path_a[:l] == path_k*[:l]] / sum_a w_a,    w_a = 1 / rank(a).

A logistic model on [logit c_l, logit e_l, logit p_k*, depth] turns both into P(correct at depth l); a
running minimum over depth keeps every coarser node at least as confident as a finer one.

For any of these scores, the answer is k*'s node at the finest depth whose score is >= lambda, or no answer
when even the country (year) is below lambda. Along one path a right city implies a right region and
country, so the error of the committed answer can only fall as lambda grows, and conformal risk control
(Angelopoulos et al., ICLR 2024) picks lambda so that E[error] <= eps on new, exchangeable images.

Methods: "finest" always answers in full (lambda = 0); "selective" calibrates p_k* and answers in full or
not at all; "backoff" calibrates c_l; "evidence" calibrates the logistic score. Protocol: dev is split in
half (seed 0); one half fits the logistic model, the other sets lambda for every method; test only scores.
--resplits R pools dev and test and repeats fit 25% / calibrate 25% / score 50% R times, which separates
chance from a real dev/test difference when one split lands above eps.
A gold answer coarser than the committed one (gold "Haiti", answer "Port-au-Prince, Haiti") is judged at
the gold's depth, as TARA's EM@1 does. Writes $NV_OUT/results/trust_{name}.json and .md
"""
import argparse

import numpy as np

from . import config as C
from .build_sft import ranked_lists
from .common import CONTINENTS, load_corpus, load_tara, pad_date, read_jsonl, write_json
from .geo import Geo

LEVELS = {"location": ("country", "region", "city"), "date": ("year", "month", "day")}
LAM = np.r_[np.linspace(0, 1, 1001), 1 + 1e-9]       # candidate thresholds; the last one never answers
METHODS = ("selective", "backoff", "evidence")
FEATURES = ["logit c_l", "logit e_l", "logit p_k*", "depth 1", "depth 2", "depth 3"]


def loc_path(geo, s):
    h = [x.lower() for x in geo.lookup(s)[1] if x.lower() not in CONTINENTS]
    return tuple(h[:2]) + (h[-1],) if len(h) > 2 else tuple(h)


def date_path(s):
    p = (pad_date(s) or "").split("-")
    return tuple("-".join(p[:i]) for i in range(1, len(p) + 1)) if p[0].isdigit() else ()


def build(recs, preds, gold, arts, opt_path, gold_path, art_path):
    """-> conf, top, ev, ok, each (n, 3): c_l, p_k* and e_l along k*'s path (-inf past its depth) and whether
    committing at depth l would be right. Images without a gold path at this level are skipped."""
    conf, top, ev, ok = [], [], [], []
    for i, rec in recs.items():
        g = gold_path(gold[i])
        if not g or i not in preds:
            continue
        probs = preds[i]["letter_probs"][:len(rec["options"])]
        paths = [opt_path(o) for o in rec["options"]]
        k = int(np.argmax(probs)) if probs else 0
        pk = paths[k] if probs else ()          # no candidates, or k* not in the gazetteer: never answers
        ap = [art_path(a) for a in arts.get(i, [])[:20]]
        w = 1 / np.arange(1, len(ap) + 1)
        conf.append([sum(p for p, q in zip(probs, paths) if q[:l] == pk[:l]) if l <= len(pk) else -np.inf
                     for l in (1, 2, 3)])
        top.append([probs[k] if l <= len(pk) else -np.inf for l in (1, 2, 3)])
        ev.append([float(w @ np.array([q[:l] == pk[:l] for q in ap], float)) / max(w.sum(), 1e-9)
                   if l <= len(pk) else -np.inf for l in (1, 2, 3)])
        ok.append([pk[:min(l, len(g))] == g[:min(l, len(g))] for l in (1, 2, 3)])
    return [np.array(x, float if j < 3 else bool).reshape(-1, 3) for j, x in enumerate((conf, top, ev, ok))]


def logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def features(conf, top, ev):
    """(n, 3, 6): the FEATURES of every (image, depth) pair."""
    return np.concatenate([np.stack([logit(conf), logit(ev), logit(top)], -1),
                           np.broadcast_to(np.eye(3), (len(conf), 3, 3))], -1)


def fit(X, y, l2=1e-2, iters=30):
    """L2-regularised logistic regression by Newton steps."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        w -= np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w)), X.T @ (p - y) + l2 * w)
    return w


def evidence_conf(w, X, conf):
    c = np.minimum.accumulate(1 / (1 + np.exp(-X @ w)), axis=1)     # a finer node is never more sure
    return np.where(np.isfinite(conf), c, -np.inf)


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


def run_split(D, fit_i, cal_i, test_i, eps_list):
    """Fit on fit_i, calibrate on cal_i, score test_i -> ({eps: {method: report}}, scores, logistic weights)."""
    valid = np.isfinite(D["conf"][fit_i])
    w = fit(D["X"][fit_i][valid], D["ok"][fit_i][valid])
    scores = {"selective": D["top"], "backoff": D["conf"], "evidence": evidence_conf(w, D["X"], D["conf"])}
    res = {e: {m: report(s[test_i], D["ok"][test_i], crc(s[cal_i], D["ok"][cal_i], e)) for m, s in scores.items()}
           for e in eps_list}
    return res, scores, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lora_mixed_mcqa", help="MCQA prediction file in preds/<split>/")
    ap.add_argument("--cal", default="dev")
    ap.add_argument("--test", default="test")
    ap.add_argument("--eps", default="0.05,0.1,0.2", help="target error rates")
    ap.add_argument("--resplits", type=int, default=0, help="pooled random fit/calibrate/score splits")
    args = ap.parse_args()
    eps_list = [float(e) for e in args.eps.split(",")]
    geo, corpus = Geo(), load_corpus()
    paths = {"location": (lambda o: loc_path(geo, o["loc"]),
                          lambda r: loc_path(geo, r["target_loc"] or r["gold_loc"]) if r["gold_loc"] else (),
                          lambda a: loc_path(geo, a["loc_str"])),
             "date": (lambda o: date_path(o["date"]), lambda r: date_path(r["gold_date"]),
                      lambda a: date_path(a["date"]))}
    parts = {}
    for split in (args.cal, args.test):
        recs = {r["idx"]: r for r in read_jsonl(C.p("sft", f"{split}_eval.jsonl")) if r["fmt"] == "mcqa"}
        preds = {p["idx"]: p for p in read_jsonl(C.p("preds", split, f"{args.name}.jsonl"))}
        if any(p.get("letter_probs") is None for p in preds.values()):
            raise SystemExit(f"{args.name} on {split} has no letter_probs: re-run nv.infer (--fmt mcqa)")
        gold = {r["idx"]: r for r in load_tara(split)}
        arts = ranked_lists(split, corpus)
        parts[split] = {dim: build(recs, preds, gold, arts, *fn) for dim, fn in paths.items()}

    out = {"name": args.name, "cal": args.cal, "test": args.test, "levels": LEVELS, "features": FEATURES, "dims": {}}
    md = [f"## Trust layer: `{args.name}`; {args.cal} halves fit the evidence model and set lambda, "
          f"{args.test} scores", "",
          "| dim | eps | method | lambda | error % | answered % | depth 1/2/3 % | specificity % |",
          "|---|---|---|---|---|---|---|---|"]
    rs_md = [f"### {args.resplits} pooled random splits ({args.cal} + {args.test}: fit 25% / calibrate 25% / "
             "score 50%)", "", "| dim | eps | method | error % mean ± sd | runs above eps % | answered % | "
             "specificity % |", "|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(0)
    for dim in paths:
        conf, top, ev, ok = (np.concatenate(x) for x in zip(parts[args.cal][dim], parts[args.test][dim]))
        D = {"conf": conf, "top": top, "ok": ok, "X": features(conf, top, ev)}
        n_cal = len(parts[args.cal][dim][0])
        dev = rng.permutation(n_cal)
        test_i = np.arange(n_cal, len(conf))
        res, scores, w = run_split(D, dev[: n_cal // 2], dev[n_cal // 2:], test_i, eps_list)
        out["dims"][dim] = r = {
            "n_cal": n_cal, "n_test": len(test_i), "finest": report(conf[test_i], ok[test_i], 0.0), "eps": res,
            "weights": dict(zip(FEATURES, w.tolist())),
            "curve": {"lambda": LAM[::10].tolist(),
                      **{m: {"error": (100 * x.mean(0)).tolist(), "specificity": (100 * d.mean(0) / 3).tolist()}
                         for m in METHODS for d, x in [commit(scores[m][test_i], ok[test_i], LAM[::10])]}}}
        md.append(f"| {dim} | - | finest | 0.000 | {r['finest']['error']:.1f} | {r['finest']['answered']:.1f} | "
                  + "/".join(f"{x:.0f}" for x in r["finest"]["depth_%"]) + f" | {r['finest']['specificity']:.1f} |")
        for e in eps_list:
            for m in METHODS:
                x = res[e][m]
                md.append(f"| {dim} | {e} | {m} | {x['lambda']:.3f} | {x['error']:.1f} | {x['answered']:.1f} | "
                          + "/".join(f"{v:.0f}" for v in x["depth_%"]) + f" | {x['specificity']:.1f} |")
        if args.resplits:
            runs = []
            for _ in range(args.resplits):
                p = rng.permutation(len(conf))
                q = len(p) // 4
                runs.append(run_split(D, p[:q], p[q:2 * q], p[2 * q:], eps_list)[0])
            r["resplit"] = {e: {m: {"error_mean": float(np.mean(err := [x[e][m]["error"] for x in runs])),
                                    "error_sd": float(np.std(err)),
                                    "above_eps_%": 100 * float(np.mean(np.array(err) > 100 * e)),
                                    "answered_mean": float(np.mean([x[e][m]["answered"] for x in runs])),
                                    "specificity_mean": float(np.mean([x[e][m]["specificity"] for x in runs]))}
                                for m in METHODS} for e in eps_list}
            for e in eps_list:
                for m in METHODS:
                    x = r["resplit"][e][m]
                    rs_md.append(f"| {dim} | {e} | {m} | {x['error_mean']:.1f} ± {x['error_sd']:.1f} | "
                                 f"{x['above_eps_%']:.0f} | {x['answered_mean']:.1f} | {x['specificity_mean']:.1f} |")
    md.append(f"\nImages with a gold answer: location {out['dims']['location']['n_cal']} {args.cal} / "
              f"{out['dims']['location']['n_test']} {args.test}, date {out['dims']['date']['n_cal']} / "
              f"{out['dims']['date']['n_test']}. Depth 1/2/3 = {'/'.join(LEVELS['location'])} or "
              f"{'/'.join(LEVELS['date'])}.")
    md += [f"Evidence-model weights ({', '.join(FEATURES)}): "
           + "; ".join(f"{dim} " + ", ".join(f"{v:+.2f}" for v in out["dims"][dim]["weights"].values())
                       for dim in paths)]
    if args.resplits:
        md += [""] + rs_md
    text = "\n".join(md)
    print(text)
    with open(C.p("results", f"trust_{args.name}.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    write_json(out, C.p("results", f"trust_{args.name}.json"))


if __name__ == "__main__":
    main()
