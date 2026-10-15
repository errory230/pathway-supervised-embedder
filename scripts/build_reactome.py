"""Build the Reactome external validation set (human pathways)

Example (Colab):
  python build_reactome.py --email you@lab.org --kegg-dir /content/data/splits --out-dir $BASE/external/reactome
"""
import json
import time
import random
import argparse
import urllib.error
import urllib.request
from datetime import date
from collections import defaultdict
from external_utils import NCBI, kegg_pmids, save_set

CS = "https://reactome.org/ContentService"
USER_AGENT = "Mozilla/5.0 (compatible; kegg-context-benchmark/1.0)"
PATHWAY_TYPES = {"TopLevelPathway", "Pathway", "CellLineagePath"}


def get_json(url, data=None, retries=4, delay=0.1):
    for attempt in range(retries):
        time.sleep(delay)
        try:
            headers = {"Accept": "application/json, text/plain;q=0.9, */*;q=0.8", "User-Agent": USER_AGENT}
            if data is not None:
                headers["Content-Type"] = "text/plain"
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
            try:
                return json.loads(raw)
            except ValueError:
                return raw.decode("utf-8", "replace").strip()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 403 and attempt == retries - 1:
                raise SystemExit(f"Reactome refused the request (HTTP 403) for {url}.\n"
                                 "Run the connection check cell in the notebook and send its output.")
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def descendants(node):
    out = []
    for c in node.get("children", []) or []:
        out.append(c["stId"])
        out.extend(descendants(c))
    return out


def pmids_of(obj):
    found, unresolved = set(), False
    for ref in obj.get("literatureReference", []) or []:
        if isinstance(ref, dict):
            if ref.get("pubMedIdentifier"):
                found.add(str(ref["pubMedIdentifier"]))
        else:
            unresolved = True       
    return found, unresolved


def fetch_event_refs(st_ids, batch=20):
    refs, fallback = {}, []
    st_ids = list(dict.fromkeys(st_ids))
    for i in range(0, len(st_ids), batch):
        chunk = st_ids[i:i + batch]
        res = get_json(f"{CS}/data/query/ids", data=",".join(chunk).encode()) or []
        got = set()
        for obj in res:
            sid = obj.get("stId")
            if not sid:
                continue
            got.add(sid)
            found, unresolved = pmids_of(obj)
            refs[sid] = found
            if unresolved:
                fallback.append(sid)
        fallback.extend(s for s in chunk if s not in got)
        if (i // batch) % 50 == 0:
            print(f"  events {i + len(chunk)}/{len(st_ids)}")
    for sid in dict.fromkeys(fallback):
        obj = get_json(f"{CS}/data/query/enhanced/{sid}")
        refs[sid] = pmids_of(obj)[0] if obj else set()
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="contact email required by NCBI")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--kegg-dir", required=True, help="KEGG splits folder")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--species", default="9606")
    ap.add_argument("--exclude-top", nargs="*", default=["Disease"])
    ap.add_argument("--min-refs", type=int, default=5)
    ap.add_argument("--max-refs", type=int, default=30)
    ap.add_argument("--max-units", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    exclude = kegg_pmids(args.kegg_dir)
    global USER_AGENT
    USER_AGENT = f"Mozilla/5.0 (compatible; kegg-context-benchmark/1.0; mailto:{args.email})"

    print("Checking the Reactome ContentService ...")
    print(f"  Reactome release: {get_json(f'{CS}/data/database/version')}")
    print("Downloading the Reactome event hierarchy ...")
    tops = get_json(f"{CS}/data/eventsHierarchy/{args.species}")
    version = get_json(f"{CS}/data/database/version")

    units, seen = [], set()
    for top_node in tops:
        if top_node.get("name") in args.exclude_top:
            continue
        for child in top_node.get("children", []) or []:
            if child.get("type") not in PATHWAY_TYPES:
                continue
            for unit in child.get("children", []) or []:
                if unit.get("type") not in PATHWAY_TYPES or unit["stId"] in seen:
                    continue
                seen.add(unit["stId"])
                units.append({"stId": unit["stId"], "name": unit["name"], "category": top_node["name"],
                              "subcategory": child["name"], "events": [unit["stId"]] + descendants(unit)})
    print(f"{len(units)} candidate units under {len({u['category'] for u in units})} top-level pathways")

    print("Collecting literature references ...")
    event_refs = fetch_event_refs([e for u in units for e in u["events"]])
    for u in units:
        u["pmids"] = sorted(set().union(*(event_refs.get(e, set()) for e in u["events"])) - exclude)
    eligible = [u for u in units if len(u["pmids"]) >= args.min_refs]
    print(f"{len(eligible)} units with >= {args.min_refs} references outside KEGG")

    by_top = defaultdict(list)
    for u in eligible:
        by_top[u["category"]].append(u)
    for lst in by_top.values():
        rng.shuffle(lst)
    order = []
    while any(by_top.values()):
        for top_name in sorted(by_top):
            if by_top[top_name]:
                order.append(by_top[top_name].pop())

    ncbi = NCBI(args.email, args.api_key)
    entries = []
    for u in order:
        if len(entries) >= args.max_units:
            break
        pool = u["pmids"][:]
        rng.shuffle(pool)
        docs = ncbi.fetch_abstracts(pool[: args.max_refs * 2])
        keep = [p for p in pool if p in docs][: args.max_refs]
        if len(keep) < args.min_refs:
            continue
        entries.append({"pathway_id": u["stId"], "pathway_name": u["name"], "category": u["category"],
                        "subcategory": u["subcategory"], "references": [{"pmid": p, **docs[p]} for p in keep]})
        print(f"{u['stId']:16s} {u['category'][:28]:28s} {u['name'][:40]:40s} {len(keep)} papers")

    save_set(args.out_dir, entries, ["category", "subcategory", "pathway_id"],
             {"source": "Reactome ContentService + PubMed", "reactome_version": version,
              "built": date.today().isoformat(), "species": args.species, "excluded_top": args.exclude_top,
              "min_refs": args.min_refs, "max_refs": args.max_refs},
             seed=args.seed)


if __name__ == "__main__":
    main()
