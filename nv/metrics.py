"""Metric functions copied from NewsRECON's src/evaluation_metrics.py (Tonglet et al., 2026, Apache-2.0),
so scores follow the paper's definitions: EM@1 / Example-F1 / Delta / CO-delta / GREAT.
Only change: geocoding goes through nv.geo.Geo (cache -> offline gazetteer -> optional API)."""
import re
import string

from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzutc
from math import asin, cos, radians, sin, sqrt


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(radians, (a[0], a[1], b[0], b[1]))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * asin(sqrt(h))


def normalize_date_str(s):
    s = str(s).strip()
    return s.split("T", 1)[0] if "T" in s else s


def convert_to_date(s):
    try:
        return dateparser.parse(s) if s else None
    except Exception:
        return None


def date_distance_years(dt1, dt2):
    if dt1 is None or dt2 is None:
        return float("inf")
    dt1 = dt1.replace(tzinfo=tzutc()) if dt1.tzinfo is None else dt1
    dt2 = dt2.replace(tzinfo=tzutc()) if dt2.tzinfo is None else dt2
    d = relativedelta(dt1, dt2)
    return abs(d.years + d.months / 12 + d.days / 365.25)


def norm_loc_str(s):
    return (s or "").lower().translate(str.maketrans("", "", string.punctuation)).strip()


def em_location(pred, gt):
    parts = [norm_loc_str(p) for p in str(gt).split(",") if p.strip()]
    return int(bool(pred) and parts[0] in norm_loc_str(pred))


def em_time(pred, gt):
    if not pred:
        return 0
    c, g = normalize_date_str(pred), str(gt).split("-")
    if len(g) == 1:
        if "s" in g[0].lower():
            dec = int(g[0][:4])
            return int(c.split("-")[0] in [str(y) for y in range(dec, dec + 10)])
        return int(g[0] in c)
    if len(g) == 2:
        return int("-".join(g) in c)
    return int("-".join(g) == c)


def time_hierarchy(s):
    s = normalize_date_str(s)
    if s.endswith("s") and s[:-1].isdigit():
        dec = int(s[:-1])
        return [f"{dec}s", f"{(dec // 100) + 1}th century"]
    parts, out = s.split("-"), []
    try:
        if len(parts) == 3:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            out += [f"{y}-{m:02d}-{d:02d}", f"{y}-{m:02d}", f"{y}"]
        elif len(parts) == 2:
            y, m = int(parts[0]), int(parts[1])
            out += [f"{y}-{m:02d}", f"{y}"]
        else:
            y = int(parts[0])
            out += [f"{y}"]
        out += [f"{(y // 10) * 10}s", f"{((y - 1) // 100) + 1}th century"]
    except Exception:
        out = [s]
    return out


def f1_sets(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def cumulative_paths(nodes):
    rev = []
    for x in reversed(nodes or []):
        if not rev or rev[-1].lower() != (x or "").lower():
            rev.append(x)
    return {", ".join(rev[i:]).lower() for i in range(len(rev))}


_TOL = dict(decade=50, year=5, month=6, day=15)
_W = dict(century=1.0, decade=1.0, year=1.25, month=1.5, day=1.5)


def _time_units(s):
    s = normalize_date_str(str(s))
    m = re.match(r"^(\d{4})s$", s)
    if m:
        y = int(m.group(1))
        return dict(century=((y - 1) // 100) + 1, decade=(y // 10) * 10, year=None, month=None, day=None)
    p = s.split("-")
    try:
        y = int(p[0])
        return dict(century=((y - 1) // 100) + 1, decade=(y // 10) * 10, year=y,
                    month=int(p[1]) if len(p) >= 2 and p[1].isdigit() else None,
                    day=int(p[2]) if len(p) >= 3 and p[2].isdigit() else None)
    except Exception:
        return dict(century=None, decade=None, year=None, month=None, day=None)


def _unit(gu, pu, unit):
    if gu is None or pu is None:
        return 0.0
    if unit == "century":
        return 1.0 if gu == pu else 0.0
    return max(0.0, 1.0 - abs(gu - pu) / float(_TOL[unit]))


def great_date(gt, pred):
    g, p = _time_units(gt), _time_units(pred)
    return sum(_W[u] * _unit(g[u], p[u], u) for u in _W) / sum(_W.values())


def score_location(pred, gt, geo):
    """-> dict(em, great, codelta, ef1) for one sample (NewsRECON evaluate(): EM hit gives 1 everywhere)."""
    if not pred:
        return dict(em=0, great=0.0, codelta=0.0, ef1=0.0)
    if em_location(pred, gt):
        return dict(em=1, great=1.0, codelta=1.0, ef1=1.0)
    gc, gh = geo.lookup(gt)
    pc, ph = geo.lookup(pred)
    if gc and pc:
        d = haversine_km(gc, pc)
        great, co = max(0.0, 1 - d / 1000.0), 1 / (1 + d)
    else:
        great = co = 0.0
    gs, ps = cumulative_paths(gh), cumulative_paths(ph)
    ef1 = 1.0 if not gs and not ps else (0.0 if not gs or not ps else 2 * len(gs & ps) / (len(gs) + len(ps)))
    return dict(em=0, great=great, codelta=co, ef1=ef1)


def score_date(pred, gt):
    if not pred:
        return dict(em=0, great=0.0, delta=0.0, ef1=0.0)
    if em_time(pred, gt):
        return dict(em=1, great=1.0, delta=1.0, ef1=1.0)
    yd = date_distance_years(convert_to_date(normalize_date_str(pred)), convert_to_date(normalize_date_str(gt)))
    return dict(em=0, great=great_date(gt, pred), delta=(1 / (1 + yd)) if yd != float("inf") else 0.0,
                ef1=f1_sets(set(time_hierarchy(gt)), set(time_hierarchy(pred))))
