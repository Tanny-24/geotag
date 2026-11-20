"""Shared helpers: I/O, location/date normalisation, relevance labels, evidence formatting."""
import json
import os
import re
from datetime import date, datetime

from . import config as C

# ----------------------------------------------------------------------------- I/O

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_corpus():
    return read_jsonl(os.path.join(C.PREP, "corpus.jsonl"))


def load_tara(split):
    return read_jsonl(os.path.join(C.PREP, f"tara_{split}.jsonl"))

# ----------------------------------------------------------------------------- locations

CONTINENTS = {"africa", "asia", "europe", "north america", "south america", "oceania", "australia", "antarctica"}

# NYT glocation qualifiers -> readable names, e.g. "Brooklyn (NYC)" -> "Brooklyn, New York City"
NYT_ABBR = {
    "nyc": "New York City", "nys": "New York State", "ny": "New York", "nj": "New Jersey", "conn": "Connecticut",
    "calif": "California", "mass": "Massachusetts", "fla": "Florida", "tex": "Texas", "ill": "Illinois",
    "pa": "Pennsylvania", "wash": "Washington", "va": "Virginia", "md": "Maryland", "ga": "Georgia",
    "mich": "Michigan", "la": "Louisiana", "ariz": "Arizona", "colo": "Colorado", "minn": "Minnesota",
    "mo": "Missouri", "nc": "North Carolina", "sc": "South Carolina", "tenn": "Tennessee", "ky": "Kentucky",
    "ala": "Alabama", "miss": "Mississippi", "ore": "Oregon", "wis": "Wisconsin", "ind": "Indiana",
    "kan": "Kansas", "neb": "Nebraska", "nev": "Nevada", "okla": "Oklahoma", "ark": "Arkansas",
    "nm": "New Mexico", "nh": "New Hampshire", "vt": "Vermont", "me": "Maine", "ri": "Rhode Island",
    "del": "Delaware", "wva": "West Virginia", "mont": "Montana", "wyo": "Wyoming", "sd": "South Dakota",
    "nd": "North Dakota", "dc": "Washington, DC", "li": "Long Island", "wis.": "Wisconsin",
}


def _title(s):
    s = s.strip()
    return s.title() if s.isupper() and len(s) > 3 else s


def format_gloc(g):
    """'Dubai (United Arab Emirates)' -> 'Dubai, United Arab Emirates'; 'Brooklyn (NYC)' -> 'Brooklyn, New York City'."""
    g = " ".join(str(g).split())
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", g)
    if not m:
        return _title(g)
    base, qual = m.group(1), m.group(2)
    qual = ", ".join(NYT_ABBR.get(q.strip().lower().rstrip("."), q.strip()) for q in qual.split(","))
    return f"{_title(base)}, {qual}"


def gloc_atoms(g):
    """NewsRECON relevance atoms of one glocation: 'Karachi (Pakistan)' -> {'karachi', 'pakistan'}."""
    return {v.strip().lower().replace(")", "") for v in str(g).split("(") if v.strip()}


def compact_address(addr):
    """Nominatim-style TARA address -> 'place, region, country' (drops postcodes, continent, non-ASCII parts)."""
    if not addr or addr == "None":
        return None
    parts = [x.strip() for x in str(addr).split(",") if x.strip()]
    parts = [x for x in parts if x.lower() not in CONTINENTS and not re.search(r"\d", x) and x.isascii()]
    if not parts:
        return None
    keep = [parts[0]] + parts[-2:] if len(parts) > 2 else parts
    out = []
    for x in keep:
        if not out or out[-1].lower() != x.lower():
            if x.lower() not in [o.lower() for o in out]:
                out.append(x)
    return ", ".join(out)

# ----------------------------------------------------------------------------- dates

def pad_date(d):
    """'2010-5-3' -> '2010-05-03'; '1960' stays; decades ('1960s') stay; anything else -> None."""
    if d is None:
        return None
    s = str(d).strip().split("T")[0]
    if re.fullmatch(r"\d{4}s", s):
        return s
    m = re.fullmatch(r"(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", s)
    if not m:
        return None
    y, mo, da = m.groups()
    out = y
    if mo:
        out += f"-{int(mo):02d}"
        if da:
            out += f"-{int(da):02d}"
    return out


def to_date(s):
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def parse_pred_date(text):
    """Free-form model date -> 'YYYY-MM-DD' / 'YYYY-MM' / 'YYYY' (or None)."""
    if not text:
        return None
    t = text.strip().strip(".")
    d = pad_date(t)
    if d:
        return d
    m = re.search(r"\d{4}-\d{1,2}(-\d{1,2})?", t)
    if m:
        return pad_date(m.group(0))
    try:
        from dateutil import parser as dp
        has_day = bool(re.search(r"\b\d{1,2}(st|nd|rd|th)?\b", re.sub(r"\d{4}", "", t)))
        dt = dp.parse(t, default=datetime(1900, 1, 1))
        if dt.year == 1900:
            return None
        return dt.strftime("%Y-%m-%d") if has_day else dt.strftime("%Y-%m")
    except Exception:
        m = re.search(r"(19|20)\d{2}", t)
        return m.group(0) if m else None

# ----------------------------------------------------------------------------- relevance (weak labels, NewsRECON rules)

def is_loc_relevant(row, art):
    """An article is location-relevant if its glocation atoms contain the image's gold location atom."""
    a = row.get("atom")
    return bool(a) and a in set(art["atoms"])


def date_gap_days(row, art):
    """Days between the article and the image's event date (None if not computable)."""
    gd = row.get("gold_date") or ""
    if len(gd) == 10:
        ref = to_date(gd)
    elif gd and row.get("img_pub", "").startswith(gd):     # coarse gold date: use image pub date (NewsRECON rule)
        ref = to_date(row["img_pub"])
    else:
        ref = None
    ad = to_date(art["date"])
    if ref is None or ad is None:
        return None
    return (ad - ref).days


def is_event_relevant(row, art):
    if not is_loc_relevant(row, art):
        return False
    g = date_gap_days(row, art)
    return g is not None and abs(g) <= C.N_WINDOW

# ----------------------------------------------------------------------------- evidence text

def evidence_block(arts, max_abs=260):
    lines = []
    for k, a in enumerate(arts, 1):
        ab = a["abs"] or a.get("lead", "")
        if len(ab) > max_abs:
            ab = ab[:max_abs].rsplit(" ", 1)[0] + "..."
        lines.append(f"Article {k} | published {a['date']} | location tags: {a['loc_str']}\n"
                     f"Headline: {a['head']}\nSummary: {ab}")
    return "\n\n".join(lines)


def candidate_events(arts, n=C.N_OPTIONS):
    """Distinct (location, date) candidate events from a ranked article list.

    Articles are grouped by location string; within a location the date is the median of the largest
    +/-N_WINDOW day cluster (a light version of NewsRECON's article clustering)."""
    groups = {}
    for a in arts:
        groups.setdefault(a["loc_str"], []).append(a)
    events = []
    for loc, group in groups.items():          # dict keeps rank order of first appearance
        dates = sorted(d for d in (to_date(a["date"]) for a in group) if d)
        if not dates:
            continue
        best = max(([d2 for d2 in dates if abs((d2 - d).days) <= C.N_WINDOW] for d in dates), key=len)
        mid = best[len(best) // 2]
        events.append({"loc": loc, "date": mid.isoformat(), "support": len(best), "art": group[0]})
        if len(events) == n:
            break
    return events


def parse_answer(text):
    """Extract 'Location:' and 'Date:' lines (and a leading option letter) from a model output."""
    loc = date_ = letter = None
    m = re.search(r"Location\s*:\s*(.+)", text or "", flags=re.I)
    if m:
        loc = m.group(1).strip().strip(".")
    m = re.search(r"Date\s*:\s*(.+)", text or "", flags=re.I)
    if m:
        date_ = parse_pred_date(m.group(1))
    m = re.match(r"\s*\(?([A-E])\)?(\s|$|[.:)])", text or "")
    if m:
        letter = m.group(1)
    return loc, date_, letter
