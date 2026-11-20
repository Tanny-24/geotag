"""Step 1 (CPU only): build the article corpus and normalised TARA files.

    python -m nv.prepare

Reads  $NV_DATA/processed_articles/nyt_articles_<year>.json   (raw NYT archive, 2010-2021 used)
       $NV_DATA/TARA_dataset/input/{train,gold_dev,gold_test}.jsonl
Writes $NV_PREP/corpus.jsonl, $NV_PREP/tara_{train,dev,test}.jsonl, $NV_PREP/stats.json

Corpus filter (replaces NewsRECON's Qwen2.5 "visualisable event" filter, which we cannot run in time):
  * 2010-2021, document_type == article, News / News Analysis
  * has at least one glocation keyword (NewsRECON also requires this) and a headline/abstract
  * drops desks that rarely describe photographable events (opinion, real estate, styles, podcasts, ...)
  * LEAKAGE: every TARA dev/test/interest source article is removed (TARA images come from these
    articles). TARA train articles stay in, flagged, and are excluded for their own image later.
"""
import argparse
import gc
import json
import os
from collections import Counter

from . import config as C
from .common import (compact_address, format_gloc, gloc_atoms, pad_date, read_jsonl, write_json, write_jsonl)

DROP_DESKS = {
    "realestate", "real estate", "podcasts", "styles", "tstyle", "style", "magazine", "games", "well", "learning",
    "obits", "oped", "editorial", "letters", "parenting", "society", "travel", "dining", "bookreview",
    "summary", "nytnow", "insider", "crosswords", "weddings", "upshot", "opinion", "sundayreview", "video",
}
KEEP_TYPES = {"news", "news analysis"}


def keep(a):
    if a.get("document_type") != "article":
        return False
    if str(a.get("type_of_material", "")).lower() not in KEEP_TYPES:
        return False
    if str(a.get("news_desk") or "").lower() in DROP_DESKS:
        return False
    if not any(k.get("name") == "glocations" for k in a.get("keywords") or []):
        return False
    head = (a.get("headline") or {}).get("main") or ""
    return bool(head.strip() or (a.get("abstract") or "").strip())


def slim(a, tara_split=""):
    glocs = [k["value"] for k in a.get("keywords") or [] if k.get("name") == "glocations" and k.get("value")]
    glocs = list(dict.fromkeys(glocs))
    atoms = sorted(set().union(*[gloc_atoms(g) for g in glocs])) if glocs else []
    return {
        "url": a["web_url"],
        "head": " ".join(((a.get("headline") or {}).get("main") or "").split()),
        "abs": " ".join((a.get("abstract") or "").split()),
        "lead": " ".join((a.get("lead_paragraph") or "").split())[:400],
        "date": a["pub_date"][:10],
        "glocs": glocs,
        "atoms": atoms,
        "loc_str": format_gloc(glocs[0]) if glocs else "",
        "tara": tara_split,
    }


def tara_rows(split, raw):
    rows = []
    for idx, t in enumerate(raw):
        locs = t.get("locations") or []
        if split == "train":
            gold_loc, gold_date = t.get("location"), pad_date(t.get("date"))
            target_loc = compact_address(gold_loc) or (format_gloc(locs[0]) if locs else None)
        else:
            gold_loc = t.get("gold_location")
            gold_loc = None if gold_loc in (None, "None", "") else gold_loc
            gold_date = pad_date(t.get("gold_time"))
            target_loc = compact_address(gold_loc)
        if locs:   # prefer the NYT location keyword that agrees with the annotated gold location
            bases = [str(l).split("(")[0].strip().lower() for l in locs]
            agree = [b for b in bases if gold_loc and b and b in str(gold_loc).lower()]
            atom = (agree or bases)[0]
        elif gold_loc:
            atom = str(gold_loc).split(",")[0].strip().lower()
        else:
            atom = None
        rows.append({
            "split": split, "idx": idx, "url": t["web_url"], "img_pub": t["pub_date"][:10],
            "gold_loc": gold_loc, "gold_date": gold_date, "target_loc": target_loc, "atom": atom,
            "caption": " ".join((t.get("caption") or "").split()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles_dir", default=os.path.join(C.DATA, "processed_articles"))
    args = ap.parse_args()
    tin = os.path.join(C.DATA, "TARA_dataset", "input")
    raw = {s: read_jsonl(os.path.join(tin, f)) for s, f in
           [("train", "train.jsonl"), ("dev", "gold_dev.jsonl"), ("test", "gold_test.jsonl")]}
    interest = os.path.join(tin, "gold_interest.jsonl")
    held_out = {t["web_url"] for s in ("dev", "test") for t in raw[s]}
    if os.path.exists(interest):
        held_out |= {t["web_url"] for t in read_jsonl(interest)}
    train_urls = {t["web_url"] for t in raw["train"]}

    corpus, seen, stats = [], set(), Counter()
    for y in C.YEARS:
        path = os.path.join(args.articles_dir, f"nyt_articles_{y}.json")
        if not os.path.exists(path):
            print("missing", path)
            continue
        with open(path, encoding="utf-8") as f:
            arts = json.load(f)
        for a in arts:
            stats["raw"] += 1
            url = a.get("web_url")
            if not url or url in seen:
                continue
            if url in held_out:
                stats["removed_dev_test_source"] += 1
                continue
            if not keep(a) and url not in train_urls:
                continue
            seen.add(url)
            corpus.append(slim(a, "train" if url in train_urls else ""))
        print(y, len(arts), "->", len(corpus))
        del arts
        gc.collect()

    # TARA train source articles missing from the archive are added from train.jsonl itself
    for t in raw["train"]:
        if t["web_url"] not in seen and t.get("keywords") and t["pub_date"][:4] in C.YEARS:
            seen.add(t["web_url"])
            corpus.append(slim(t, "train"))
            stats["train_articles_added"] += 1
    corpus = [a for a in corpus if a["glocs"]]
    for i, a in enumerate(corpus):
        a["i"] = i
    os.makedirs(C.PREP, exist_ok=True)
    write_jsonl(corpus, os.path.join(C.PREP, "corpus.jsonl"))

    for s in raw:
        rows = tara_rows(s, raw[s])
        write_jsonl(rows, os.path.join(C.PREP, f"tara_{s}.jsonl"))
        stats[f"tara_{s}"] = len(rows)
        stats[f"tara_{s}_with_target_loc"] = sum(r["target_loc"] is not None for r in rows)
        stats[f"tara_{s}_with_date"] = sum(r["gold_date"] is not None for r in rows)
    stats["corpus"] = len(corpus)
    stats["corpus_tara_train_articles"] = sum(a["tara"] == "train" for a in corpus)
    write_json(dict(stats), os.path.join(C.PREP, "stats.json"))
    print(json.dumps(dict(stats), indent=2))


if __name__ == "__main__":
    main()
