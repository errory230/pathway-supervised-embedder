"""
  <out>/pathways_test.json : [{"pathway_id", "pathway_name", <level keys>..., "references": [{pmid, title, abstract}]}]
  <out>/pairs_test.jsonl   : {"left", "right", "label", "type"} with type pos / hard / negative (standard) / soft
  <out>/meta.json          : how the set was built (date, queries, counts)
"""
import json
import time
import random
import itertools
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class NCBI:

    def __init__(self, email, api_key=None, tool="kegg-context-benchmark"):
        self.base = {"email": email, "tool": tool}
        if api_key:
            self.base["api_key"] = api_key
        self.delay = 0.11 if api_key else 0.35
        self._last = 0.0

    def _request(self, endpoint, params, post=False, retries=4):
        params = {**self.base, **params}
        url = f"{EUTILS}/{endpoint}"
        data = urllib.parse.urlencode(params).encode()
        for attempt in range(retries):
            wait = self.delay - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                req = urllib.request.Request(url, data=data) if post else urllib.request.Request(f"{url}?{data.decode()}")
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.read()
            except Exception as e:  
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def esearch(self, term, retmax=200, sort="relevance"):
        raw = self._request("esearch.fcgi", {"db": "pubmed", "term": term, "retmax": retmax,
                                             "retmode": "json", "sort": sort})
        res = json.loads(raw)["esearchresult"]
        return res.get("idlist", []), int(res.get("count", 0))

    def fetch_abstracts(self, pmids, batch=200):
        out = {}
        pmids = list(dict.fromkeys(str(p) for p in pmids))
        for i in range(0, len(pmids), batch):
            raw = self._request("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids[i:i + batch]),
                                                "retmode": "xml"}, post=True)
            out.update(parse_pubmed_xml(raw))
        return out


def parse_pubmed_xml(raw):
    out = {}
    root = ET.fromstring(raw)
    for art in root.iter("PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID")
        title = "".join(art.find(".//ArticleTitle").itertext()).strip() if art.find(".//ArticleTitle") is not None else ""
        parts = []
        for ab in art.findall(".//Abstract/AbstractText"):
            text = "".join(ab.itertext()).strip()
            if not text:
                continue
            label = ab.get("Label")
            parts.append(f"{label}: {text}" if label else text)
        if pmid and parts:
            out[pmid] = {"title": title, "abstract": " ".join(parts)}
    return out


def kegg_pmids(kegg_dir):
    found = set()
    for f in Path(kegg_dir).glob("pathways_*.json"):
        for e in json.load(open(f, encoding="utf-8")):
            found.update(str(r["pmid"]) for r in e["references"])
    return found


def make_pairs(entries, level_keys, seed=42, max_pos_per_group=60, neg_ratio=3):
    rng = random.Random(seed)
    labels = defaultdict(lambda: defaultdict(set))
    for e in entries:
        for r in e["references"]:
            for k in level_keys:
                labels[r["pmid"]][k].add(e[k])
    papers = sorted(labels)
    finest, depth = level_keys[-1], len(level_keys)

    def relation(a, b):
        if labels[a][finest] & labels[b][finest]:
            return "pos"
        shared = [i for i, k in enumerate(level_keys[:-1]) if labels[a][k] & labels[b][k]]
        d = max(shared) if shared else -1
        return "hard" if d == depth - 2 else ("negative" if d == depth - 3 else "soft")

    pairs, seen = [], set()
    by_group = defaultdict(list)
    for p in papers:
        for g in labels[p][finest]:
            by_group[g].append(p)
    for g, members in sorted(by_group.items()):
        combos = list(itertools.combinations(sorted(members), 2))
        rng.shuffle(combos)
        for a, b in combos[:max_pos_per_group]:
            key = frozenset((a, b))
            if key not in seen:
                seen.add(key)
                pairs.append({"left": a, "right": b, "label": 1, "type": "pos"})
    n_pos = len(pairs)
    per_type = n_pos * neg_ratio // 3
    counts = {"hard": 0, "negative": 0, "soft": 0}
    attempts = 0
    while min(counts.values()) < per_type and attempts < 200 * n_pos:
        attempts += 1
        a, b = rng.sample(papers, 2)
        key = frozenset((a, b))
        if key in seen:
            continue
        t = relation(a, b)
        if t == "pos" or counts[t] >= per_type:
            continue
        seen.add(key)
        counts[t] += 1
        pairs.append({"left": a, "right": b, "label": 0, "type": t})
    rng.shuffle(pairs)
    return pairs, {"pos": n_pos, **counts}


def save_set(out_dir, entries, level_keys, meta, seed=42):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(entries, open(out / "pathways_test.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pairs, counts = make_pairs(entries, level_keys, seed=seed)
    with open(out / "pairs_test.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p) + "\n" for p in pairs)
    n_refs = sum(len(e["references"]) for e in entries)
    n_papers = len({r["pmid"] for e in entries for r in e["references"]})
    meta = {**meta, "level_keys": level_keys, "groups": len(entries), "references": n_refs,
            "papers": n_papers, "pairs": counts}
    for k in level_keys[:-1]:
        meta[f"n_{k}"] = len({e[k] for e in entries})
    json.dump(meta, open(out / "meta.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("terms", "excluded")}, indent=2, ensure_ascii=False))
