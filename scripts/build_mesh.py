"""Build the MeSH external validation set

Example (Colab):
  python build_mesh.py --email you@lab.org --kegg-dir /content/data/splits --out-dir $BASE/external/mesh
"""
import argparse
import random
from datetime import date
from external_utils import NCBI, kegg_pmids, save_set

TREE = {
    "Diseases": {
        "Eye Diseases": {
            "Eye Infections": ["Eye Infections, Bacterial", "Endophthalmitis", "Corneal Ulcer"],
            "Conjunctival Diseases": ["Pemphigoid, Benign Mucous Membrane", "Conjunctivitis", "Conjunctival Neoplasms"],
        },
        "Nervous System Diseases": {
            "Trauma, Nervous System": ["Peripheral Nerve Injuries", "Craniocerebral Trauma", "Cerebrovascular Trauma"],
            "Autonomic Nervous System Diseases": ["Complex Regional Pain Syndromes", "Autonomic Dysreflexia", "Adie Syndrome"],
        },
    },
    "Chemicals and Drugs": {
        "Organic Chemicals": {
            "Aldehydes": ["Benzaldehydes", "Acrolein", "Acetaldehyde"],
            "Alcohols": ["Chlorohydrins", "Benzyl Alcohols", "Amino Alcohols"],
        },
        "Inorganic Chemicals": {
            "Aluminum Compounds": ["Aluminum Hydroxide", "Aluminum Chloride", "Alum Compounds"],
            "Alkalies": ["Lewis Bases", "Hydroxides", "Carbonates"],
        },
    },
}
LEVEL_KEYS = ["level1", "level2", "level3", "level4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="contact email")
    ap.add_argument("--api-key", default=None, help="NCBI API key (faster)")
    ap.add_argument("--kegg-dir", required=True, help="KEGG splits folder")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-term", type=int, default=50, help="articles kept per Level-4 term")
    ap.add_argument("--search-max", type=int, default=300, help="PubMed hits retrieved per term before filtering")
    ap.add_argument("--min-year", type=int, default=None, help="optional publication year lower bound")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ncbi = NCBI(args.email, args.api_key)
    exclude = kegg_pmids(args.kegg_dir)

    leaves = [(l1, l2, l3, l4) for l1, d2 in TREE.items() for l2, d3 in d2.items() for l3, l4s in d3.items() for l4 in l4s]
    hits, queries = {}, {}
    for l1, l2, l3, l4 in leaves:
        q = f'"{l4}"[Majr:NoExp] AND hasabstract[text] AND english[lang]'
        if args.min_year:
            q += f" AND {args.min_year}:3000[dp]"
        ids, total = ncbi.esearch(q, retmax=args.search_max)
        hits[l4], queries[l4] = ids, {"query": q, "total_hits": total, "retrieved": len(ids)}
        print(f"{l4:40s} {total:6d} hits, retrieved {len(ids)}")

    owner = {}
    for l4, ids in hits.items():
        for p in ids:
            owner.setdefault(p, set()).add(l4)
    multi = {p for p, s in owner.items() if len(s) > 1}
    candidates = {l4: [p for p in ids if p not in multi and p not in exclude] for l4, ids in hits.items()}

    # Keep up to --per-term articles that really have an abstract
    entries = []
    for l1, l2, l3, l4 in leaves:
        pool = candidates[l4][:]
        rng.shuffle(pool)
        docs = ncbi.fetch_abstracts(pool[: args.per_term * 2])
        keep = [p for p in pool if p in docs][: args.per_term]
        queries[l4]["kept"] = len(keep)
        entries.append({"pathway_id": l4, "pathway_name": l4,
                        "level1": l1, "level2": l2, "level3": l3, "level4": l4,
                        "category": l1, "subcategory": l2,
                        "references": [{"pmid": p, **docs[p]} for p in keep]})
        print(f"{l4:40s} kept {len(keep)}")

    save_set(args.out_dir, entries, LEVEL_KEYS,
             {"source": "MeSH via PubMed E-utilities", "built": date.today().isoformat(),
              "terms": queries, "removed_multi_term": len(multi),
              "removed_in_kegg": sum(1 for ids in hits.values() for p in ids if p in exclude)},
             seed=args.seed)


if __name__ == "__main__":
    main()
