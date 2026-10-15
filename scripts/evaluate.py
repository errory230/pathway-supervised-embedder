"""
Example:
  python evaluate.py --data-dir /content/data/splits --emb-dir $BASE/embeddings --out-dir $BASE/eval
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_samples

LEVELS = ("category", "subcategory", "pathway")
LEVEL_KEYS = {"category": "category", "subcategory": "subcategory", "pathway": "pathway_id"}
TYPE_NAMES = {"hard": "hard", "negative": "standard", "soft": "soft"}
REFERENCE = "ours"


def load_labels(data_dir, split):
    with open(Path(data_dir) / f"pathways_{split}.json", encoding='utf-8') as f:
        pathways = json.load(f)
    labels = defaultdict(lambda: {lv: set() for lv in LEVELS})
    for e in pathways:
        for r in e['references']:
            for lv in LEVELS:
                labels[r['pmid']][lv].add(e[LEVEL_KEYS[lv]])
    return labels


def load_pairs(data_dir, split):
    if not (Path(data_dir) / f"pairs_{split}.jsonl").exists():
        return []
    with open(Path(data_dir) / f"pairs_{split}.jsonl", encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def normalize(E):
    return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)


def retrieval_scores(S, rel, k=10):
    n = S.shape[0]
    ap, ndcg = np.full(n, np.nan), np.full(n, np.nan)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        r = rel[i, mask]
        if not r.any():
            continue
        s = S[i, mask]
        ap[i] = average_precision_score(r, s)
        top = np.argsort(-s)[:k]
        dcg = (r[top] * discounts[:len(top)]).sum()
        idcg = discounts[:min(k, r.sum())].sum()
        ndcg[i] = dcg / idcg
    return ap, ndcg


def silhouette_scores(E, label_sets):
    n = len(label_sets)
    out = np.full(n, np.nan)
    keep = [i for i in range(n) if len(label_sets[i]) == 1]
    lab = np.array([next(iter(label_sets[i])) for i in keep])
    counts = pd.Series(lab).value_counts()
    ok = np.array([counts[l] > 1 for l in lab])
    keep, lab = np.array(keep)[ok], lab[ok]
    if len(set(lab)) > 1:
        out[keep] = silhouette_samples(E[keep], lab, metric="cosine")
    return out


def evaluate_split(E, pmids, labels, pairs):
    En = normalize(E)
    S = En @ En.T
    np.fill_diagonal(S, -np.inf)
    per_paper, summary = {}, {}
    for lv in LEVELS:
        sets = [labels[p][lv] for p in pmids]
        # relevance = share at least one label at this level
        all_labels = sorted({l for s in sets for l in s})
        idx = {l: j for j, l in enumerate(all_labels)}
        M = np.zeros((len(pmids), len(all_labels)), bool)
        for i, s in enumerate(sets):
            for l in s:
                M[i, idx[l]] = True
        rel = (M.astype(np.int32) @ M.T.astype(np.int32)) > 0
        ap, ndcg = retrieval_scores(S, rel)
        sil = silhouette_scores(E, sets)
        per_paper[f"{lv}_map"], per_paper[f"{lv}_ndcg10"], per_paper[f"{lv}_silhouette"] = ap, ndcg, sil
    for k, v in per_paper.items():
        summary[k] = float(np.nanmean(v))

    if not pairs:
        return summary, per_paper
    pos = {p: i for i, p in enumerate(pmids)}
    sims = np.array([float(En[pos[p['left']]] @ En[pos[p['right']]]) for p in pairs])
    y = np.array([p['label'] for p in pairs])
    types = np.array([p['type'] for p in pairs])
    summary["pair_auroc"] = roc_auc_score(y, sims)
    summary["pair_auprc"] = average_precision_score(y, sims)
    for t, name in TYPE_NAMES.items():
        m = (types == "pos") | (types == t)
        if m.any() and (types == t).any():
            summary[f"pair_auroc_pos_vs_{name}"] = roc_auc_score(y[m], sims[m])
    return summary, per_paper


def cluster_bootstrap_ci(values, clusters, n_boot=1000, seed=42):
    ok = ~np.isnan(values)
    v, c = values[ok], clusters[ok]
    uniq, inv = np.unique(c, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    ns = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n_boot)
    means = (draws @ sums) / (draws @ ns)
    return np.percentile(means, [2.5, 97.5])


def holm(pvals):
    names = [k for k, v in pvals.items() if not np.isnan(v)]
    order = sorted(names, key=lambda k: pvals[k])
    adj, running = {}, 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvals[k]))
        adj[k] = running
    return {k: adj.get(k, np.nan) for k in pvals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--select-split", default="val", help="split used to choose cls vs mean pooling")
    ap.add_argument("--report-split", default="test")
    ap.add_argument("--select-metric", default=None, help="default: <finest level>_map")
    ap.add_argument("--level-keys", default="category,subcategory,pathway=pathway_id",
                    help="hierarchy levels from broad to fine, as name or name=json_key")
    ap.add_argument("--pooling-from", default=None,
                    help="pooling_choice.json from a KEGG run; reuses its cls/mean/sep choice per model")
    ap.add_argument("--reference", default=REFERENCE,
                    help="model the others are tested against (e.g. kegg_bge); default 'ours'")
    ap.add_argument("--center", action="store_true",
                    help="subtract each model's mean embedding (computed on the split) before scoring, for every model")
    ap.add_argument("--pair-preds", nargs="*", default=[],
                    help="name=path to a *_predictions.jsonl from train.py (e.g. ours_cross=.../test_predictions.jsonl); "
                         "added to the pair table so the cross-encoder can be compared with embedding similarity")
    args = ap.parse_args()
    global LEVELS, LEVEL_KEYS
    LEVEL_KEYS = dict(item.split("=", 1) if "=" in item else (item, item) for item in args.level_keys.split(","))
    LEVELS = tuple(LEVEL_KEYS)
    finest = LEVELS[-1]
    args.select_metric = args.select_metric or f"{finest}_map"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results, per_paper_all, pmids_by_split = {}, {}, {}
    for split in dict.fromkeys([args.select_split, args.report_split]):
        d = Path(args.emb_dir) / split
        if not d.exists():
            continue
        pmids = json.load(open(d / "pmids.json"))
        labels, pairs = load_labels(args.data_dir, split), load_pairs(args.data_dir, split)
        rows = {}
        for f in sorted(d.glob("*.npy")):
            E = np.load(f)
            if args.center:
                E = E - E.mean(axis=0, keepdims=True)
            assert len(E) == len(pmids), f"{f} has {len(E)} rows, expected {len(pmids)}"
            summary, per_paper = evaluate_split(E, pmids, labels, pairs)
            rows[f.stem] = summary
            per_paper_all[(split, f.stem)] = per_paper
            print(f"[{split}] {f.stem:28s} {finest} mAP {summary[f'{finest}_map']:.4f}  "
                  f"{finest} silhouette {summary[f'{finest}_silhouette']:.4f}  "
                  f"pair AUROC {summary.get('pair_auroc', float('nan')):.4f}")
        results[split] = pd.DataFrame(rows).T.sort_index()
        results[split].to_csv(out / f"all_variants_{split}.csv", float_format="%.4f")
        pmids_by_split[split] = (pmids, labels)

    choice = {}
    if args.pooling_from:
        given = json.load(open(args.pooling_from))
        available = set(results[args.report_split].index)
        for variant in sorted(available):
            model = variant.split("__")[0]
            if model in choice:
                continue
            if given.get(model) in available:
                choice[model] = given[model]
            else:   # model not in the KEGG run: prefer cls, then mean, sep, or the single file
                for v in (f"{model}__cls", f"{model}__mean", f"{model}__sep", model):
                    if v in available:
                        choice[model] = v
                        break
                print(f"  {model}: not in {args.pooling_from}, using {choice[model]}")
    else:
        sel = results.get(args.select_split, results[args.report_split])
        for variant in sel.index:
            model = variant.split("__")[0]
            best = choice.get(model)
            if best is None or sel.loc[variant, args.select_metric] > sel.loc[best, args.select_metric]:
                choice[model] = variant
    json.dump(choice, open(out / "pooling_choice.json", "w"), indent=2)
    print("Chosen variants: %s" % choice)

    split = args.report_split
    pmids, labels = pmids_by_split[split]
    clusters = np.array([sorted(labels[p][finest])[0] for p in pmids])
    metrics = [f"{lv}_{m}" for m in ("map", "ndcg10", "silhouette") for lv in LEVELS]
    REF = args.reference
    ref = choice.get(REF)
    deltas = {m: {} for m in metrics}
    table, pvals = {}, {m: {} for m in metrics}
    for model, variant in choice.items():
        pp = per_paper_all[(split, variant)]
        row = {}
        for m in metrics:
            lo, hi = cluster_bootstrap_ci(pp[m], clusters)
            row[m] = f"{np.nanmean(pp[m]):.4f} [{lo:.3f}, {hi:.3f}]"
            if ref and model != REF:
                a, b = per_paper_all[(split, ref)][m], pp[m]
                ok = ~np.isnan(a) & ~np.isnan(b)
                diff = np.where(ok, a - b, np.nan)
                lo, hi = cluster_bootstrap_ci(diff, clusters)
                deltas[m][model] = (float(np.nanmean(diff)), lo, hi)
                pvals[m][model] = wilcoxon(a[ok], b[ok]).pvalue if ok.sum() > 0 and np.any(a[ok] != b[ok]) else np.nan
        for m in [c for c in results[split].columns if c.startswith("pair_")]:
            row[m] = f"{results[split].loc[variant, m]:.4f}"
        row["variant"] = variant
        table[model] = row
    for m in metrics:
        for model, p in holm(pvals[m]).items():
            table[model][f"{m}_p_holm"] = f"{p:.2g}" if not np.isnan(p) else ""
    main_table = pd.DataFrame(table).T
    main_table.to_csv(out / f"main_table_{split}.csv")

    def md(cols, title):
        lines = [f"### {title}", "", "| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for model in main_table.index:
            cells = []
            for c in cols:
                v = main_table.loc[model, c]
                p = main_table.loc[model, f"{c}_p_holm"] if f"{c}_p_holm" in main_table.columns else ""
                mark = "*" if isinstance(p, str) and p and float(p) < 0.05 else ""
                cells.append(f"{v}{mark}")
            lines.append(f"| {model} ({main_table.loc[model, 'variant'].split('__')[-1]}) | " + " | ".join(cells) + " |")
        return "\n".join(lines)
    report = [
        ("Embeddings mean-centered. " if args.center else "") + f"Split: {split}. Mean [95% cluster-bootstrap CI]; * = differs from '{REF}' (Wilcoxon, Holm p<0.05).",
        "", md([f"{lv}_map" for lv in LEVELS], "Retrieval mAP"),
        "", md([f"{lv}_ndcg10" for lv in LEVELS], "Retrieval nDCG@10"),
        "", md([f"{lv}_silhouette" for lv in LEVELS], "Silhouette"),
    ]
    pair_cols = [c for c in results[split].columns if c.startswith("pair_")]
    if pair_cols:
        report += ["", "### Pair similarity (zero-shot)", "",
                   results[split].loc[list(choice.values()), pair_cols]
                   .rename(index={v: k for k, v in choice.items()}).to_markdown(floatfmt=".4f")]
    if ref:
        key = [f"{lv}_{m}" for m in ("map", "silhouette") for lv in LEVELS]
        lines = ["", f"### Paired difference: {REF} minus each model",
                 f"Mean difference over papers [95% cluster-bootstrap CI over {finest} labels]; a CI excluding 0 is the "
                 "conservative evidence of a difference. Wilcoxon p (Holm) treats papers as independent and is optimistic.", "",
                 "| Model | " + " | ".join(key) + " |", "|---" * (len(key) + 1) + "|"]
        for model in main_table.index:
            if model == REF:
                continue
            cells = []
            for m in key:
                d, lo, hi = deltas[m][model]
                flag = "**" if lo > 0 or hi < 0 else ""
                cells.append(f"{flag}{d:+.4f} [{lo:+.3f}, {hi:+.3f}]{flag} (p={main_table.loc[model, f'{m}_p_holm']})")
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        report += lines
    if args.pair_preds:
        extra = {}
        for item in args.pair_preds:
            name, path = item.split("=", 1)
            recs = [json.loads(l) for l in open(path, encoding='utf-8')]
            y = np.array([r['label'] for r in recs]); pr = np.array([r['prob'] for r in recs])
            ty = np.array([r['type'] for r in recs])
            row = {"pair_auroc": roc_auc_score(y, pr), "pair_auprc": average_precision_score(y, pr)}
            for t, tname in TYPE_NAMES.items():
                m = (ty == "pos") | (ty == t)
                row[f"pair_auroc_pos_vs_{tname}"] = roc_auc_score(y[m], pr[m])
            extra[name] = row
        report += ["", "### Pair classification by trained cross-encoders (same pairs)", "",
                   pd.DataFrame(extra).T.to_markdown(floatfmt=".4f")]
    (out / f"report_{split}.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

    rows = []
    for (s, variant), pp in per_paper_all.items():
        df = pd.DataFrame(pp)
        df.insert(0, "pmid", pmids_by_split[s][0])
        df.insert(0, "variant", variant)
        df.insert(0, "split", s)
        rows.append(df)
    pd.concat(rows).to_csv(out / "per_paper_scores.csv.gz", index=False, float_format="%.5f")


if __name__ == "__main__":
    main()
