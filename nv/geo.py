"""Location -> coordinates / hierarchy, used by GREAT_loc, CO-delta and Example-F1.

Order of lookup (same key as NewsRECON: first comma-separated part, lower-cased):
  1. NewsRECON's own geonames cache (src/geonames_results.json, copied to $NV_OUT/geonames/)
  2. offline GeoNames gazetteer (cities15000 + admin1 + countries), trying each comma part in turn
  3. online GeoNames API, only if a username is given (rate-limited: ~1000 req/hour on a free account)

Download the offline files once (≈ 3 MB):
  mkdir -p $NV_OUT/geonames && cd $NV_OUT/geonames
  wget https://download.geonames.org/export/dump/cities15000.zip && unzip cities15000.zip
  wget https://download.geonames.org/export/dump/countryInfo.txt
  wget https://download.geonames.org/export/dump/admin1CodesASCII.txt
"""
import json
import os
import time
from collections import defaultdict

import requests

from . import config as C

CONT = {"AF": "Africa", "AS": "Asia", "EU": "Europe", "NA": "North America", "SA": "South America",
        "OC": "Oceania", "AN": "Antarctica"}
ALIASES = {"usa": "united states", "us": "united states", "u.s.": "united states", "america": "united states",
           "united states of america": "united states", "uk": "united kingdom", "britain": "united kingdom",
           "great britain": "united kingdom", "england": "united kingdom", "scotland": "united kingdom",
           "wales": "united kingdom", "northern ireland": "united kingdom", "nyc": "new york city",
           "new york city": "new york city", "gaza": "palestinian territory", "gaza strip": "palestinian territory",
           "west bank": "palestinian territory", "palestine": "palestinian territory",
           "ivory coast": "ivory coast", "burma": "myanmar", "czech republic": "czechia",
           "washington, dc": "washington", "washington dc": "washington", "d.c.": "washington",
           "uae": "united arab emirates", "south korea": "south korea", "north korea": "north korea",
           "congo": "democratic republic of the congo", "the netherlands": "netherlands", "holland": "netherlands"}


class Geo:
    def __init__(self, username=None, dump_dir=None, cache_path=None):
        self.username = username
        self.dump_dir = dump_dir or os.path.join(C.OUT, "geonames")
        self.cache_path = cache_path or os.path.join(self.dump_dir, "geonames_results.json")
        self.cache = {}
        if os.path.exists(self.cache_path):
            for e in json.load(open(self.cache_path, encoding="utf-8")):
                if isinstance(e, dict) and e.get("query"):
                    self.cache[e["query"].lower()] = e
        self.names = defaultdict(list)       # name -> [(pop, (lat, lon), hierarchy)]
        self._load_offline()
        self.memo = {}

    def _load_offline(self):
        cities = os.path.join(self.dump_dir, "cities15000.txt")
        if not os.path.exists(cities):
            print("[geo] offline gazetteer not found in", self.dump_dir, "- only cache/API will be used")
            return
        countries, admin1 = {}, {}
        for line in open(os.path.join(self.dump_dir, "countryInfo.txt"), encoding="utf-8"):
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            countries[f[0]] = (f[4], CONT.get(f[8], ""))
        for line in open(os.path.join(self.dump_dir, "admin1CodesASCII.txt"), encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            admin1[f[0]] = f[1]
        agg = defaultdict(lambda: [0.0, 0.0, 0.0])      # weighted centroid of countries / admin1 regions
        for line in open(cities, encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            lat, lon, cc, a1, pop = float(f[4]), float(f[5]), f[8], f[10], max(int(f[14] or 0), 1)
            cname, cont = countries.get(cc, ("", ""))
            a1name = admin1.get(f"{cc}.{a1}", "")
            hier = [h for h in (cont, cname, a1name, f[1]) if h]
            for n in {f[1].lower(), f[2].lower()} | {x.lower() for x in f[3].split(",") if x.isascii() and len(x) > 3}:
                self.names[n].append((pop, (lat, lon), hier))
            for key in (("C", cc), ("A", f"{cc}.{a1}")):
                w = agg[key]
                w[0] += pop * lat
                w[1] += pop * lon
                w[2] += pop
        for (kind, code), (slat, slon, sp) in agg.items():
            if kind == "C" and code in countries:
                cname, cont = countries[code]
                self.names[cname.lower()].append((sp * 10, (slat / sp, slon / sp), [cont, cname]))
            elif kind == "A" and code in admin1:
                cc = code.split(".")[0]
                cname, cont = countries.get(cc, ("", ""))
                self.names[admin1[code].lower()].append((sp, (slat / sp, slon / sp), [cont, cname, admin1[code]]))

    def _offline(self, query):
        parts = [p.strip().lower() for p in str(query).split(",") if p.strip()]
        hints = set(ALIASES.get(p, p) for p in parts[1:])
        for p in parts:
            p = ALIASES.get(p, p)
            cands = self.names.get(p)
            if not cands:
                continue
            hinted = [c for c in cands if hints & {h.lower() for h in c[2]}]
            pop, coords, hier = max(hinted or cands, key=lambda c: c[0])
            return coords, hier
        return None, []

    def _online(self, norm):
        if not self.username:
            return None
        try:
            r = requests.get("http://api.geonames.org/searchJSON",
                             params={"q": norm, "maxRows": 1, "username": self.username}, timeout=30).json()
            d = (r.get("geonames") or [None])[0]
            if not d:
                return None
            h = requests.get("http://api.geonames.org/hierarchyJSON",
                             params={"geonameId": d["geonameId"], "username": self.username}, timeout=30).json()
            hier = [x["name"] for x in h.get("geonames", []) if x["name"].lower() != "earth"]
            e = {"query": norm, "coordinates": [float(d["lat"]), float(d["lng"])], "hierarchy": hier}
            self.cache[norm] = e
            time.sleep(1.0)
            return e
        except Exception as ex:
            print("[geo] API error", ex)
            return None

    def lookup(self, query):
        """-> ((lat, lon) or None, hierarchy list root->leaf)"""
        if not query:
            return None, []
        if query in self.memo:
            return self.memo[query]
        norm = str(query).split(",")[0].strip().lower()
        e = self.cache.get(norm)
        if e and e.get("coordinates"):
            h = e.get("hierarchy") or []
            h = [x.strip() for x in h.split(";" if ";" in h else ",")] if isinstance(h, str) else list(h)
            res = (tuple(e["coordinates"]), [x for x in h if x and x.lower() != "earth"])
        else:
            res = self._offline(query)
            if res[0] is None:
                e = self._online(norm)
                if e:
                    res = (tuple(e["coordinates"]), e["hierarchy"])
        self.memo[query] = res
        return res

    def save(self):
        """Persist new API lookups. Never fatal (e.g. read-only cache file)."""
        for path in (self.cache_path, os.path.join(C.OUT, "results", "geonames_cache_new.json")):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(list(self.cache.values()), f, indent=1, ensure_ascii=False)
                return
            except OSError as e:
                print("[geo] could not write cache:", e)
