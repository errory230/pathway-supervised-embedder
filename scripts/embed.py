"""
Example:
  python embed.py --data-dir /content/data/splits --out-dir $BASE/embeddings \
      --ours $BASE/result_crossencoder/run-XXXX/best \
      --models longformer biobert scibert pubmedbert specter medcpt bge tfidf random
  python embed.py ... --models specter2          # needs `pip install adapters` (run last; see notebook)
  python embed.py ... --custom kegg_bge=$BASE/result_bienc/run-XXXX/best   # any fine-tuned checkpoint
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm


MODELS = {
    "longformer": {"hf": "allenai/longformer-base-4096", "max_len": 4096},
    "biobert":    {"hf": "dmis-lab/biobert-base-cased-v1.2", "max_len": 512},
    "scibert":    {"hf": "allenai/scibert_scivocab_uncased", "max_len": 512},
    "pubmedbert": {"hf": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "max_len": 512},
    "specter":    {"hf": "allenai/specter", "max_len": 512, "only": "cls", "native": "title_sep_abstract"},
    "specter2":   {"hf": "allenai/specter2_base", "adapter": "allenai/specter2", "max_len": 512, "only": "cls",
                   "native": "title_sep_abstract"},
    "medcpt":     {"hf": "ncbi/MedCPT-Article-Encoder", "max_len": 512, "only": "cls", "native": "title_abstract_pair"},
    "bge":        {"hf": "BAAI/bge-base-en-v1.5", "max_len": 512, "only": "cls"},
}
NON_NEURAL = ("tfidf", "random")


def load_papers(data_dir, split):
    with open(Path(data_dir) / f"pathways_{split}.json", encoding='utf-8') as f:
        pathways = json.load(f)
    papers = {}
    for entry in pathways:
        for ref in entry['references']:
            papers[ref['pmid']] = {'title': ref.get('title', ''), 'abstract': ref['abstract']}
    pmids = sorted(papers)
    return pmids, [papers[p] for p in pmids]


def make_inputs(tokenizer, batch, fmt, max_len):
    if fmt == "title_sep_abstract":      # SPECTER / SPECTER2 recommended format
        texts = [p['title'] + tokenizer.sep_token + p['abstract'] for p in batch]
        return tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    if fmt == "title_abstract_pair":     # MedCPT recommended format
        return tokenizer([p['title'] for p in batch], [p['abstract'] for p in batch], padding=True,
                         truncation='only_second', max_length=max_len, return_tensors='pt')
    return tokenizer([p['abstract'] for p in batch], padding=True, truncation=True, max_length=max_len,
                     return_tensors='pt')


def load_model(spec):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(spec.get("tokenizer", spec["hf"]))
    if spec.get("adapter"):
        from adapters import AutoAdapterModel   # pip install adapters
        model = AutoAdapterModel.from_pretrained(spec["hf"])
        model.load_adapter(spec["adapter"], source="hf", load_as="proximity", set_active=True)
    else:
        model = AutoModel.from_pretrained(spec["hf"])
    return tokenizer, model


@torch.no_grad()
def embed_transformer(spec, papers, fmt, batch_size, device, bf16):
    tokenizer, model = load_model(spec)
    model.to(device).eval()
    is_longformer = getattr(model.config, "model_type", "") == "longformer"
    max_len = min(spec["max_len"], tokenizer.model_max_length)
    cls_out, mean_out, sep_out = [], [], []
    for i in tqdm(range(0, len(papers), batch_size), desc=spec["hf"], mininterval=10):
        inputs = make_inputs(tokenizer, papers[i:i + batch_size], fmt, max_len)
        inputs = {k: v.to(device) for k, v in inputs.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        if is_longformer:
            gam = torch.zeros_like(inputs["input_ids"])
            gam[:, 0] = 1                       # global attention on <s>, as in training
            inputs["global_attention_mask"] = gam
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            out = model(**inputs)
        hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]).float()
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        cls_out.append(hidden[:, 0].cpu().numpy())
        mean_out.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
        last = inputs["attention_mask"].sum(1) - 1          # final separator token (</s> or [SEP])
        sep_out.append(hidden[torch.arange(hidden.size(0), device=hidden.device), last].cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"cls": np.concatenate(cls_out), "mean": np.concatenate(mean_out), "sep": np.concatenate(sep_out)}


def embed_tfidf(papers, fit_papers):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words='english')
    X_fit = vec.fit_transform([p['abstract'] for p in fit_papers])
    svd = TruncatedSVD(n_components=min(768, X_fit.shape[1] - 1), random_state=42).fit(X_fit)
    return svd.transform(vec.transform([p['abstract'] for p in papers]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--train-dir", default=None,
                    help="folder with pathways_train.json used to fit TF-IDF (default: --data-dir); "
                         "for external sets pass the KEGG splits folder")
    ap.add_argument("--models", nargs="*", default=[], help=f"any of: {', '.join(list(MODELS) + list(NON_NEURAL))}")
    ap.add_argument("--ours", default=None, help="our fine-tuned checkpoint folder (saved as 'ours')")
    ap.add_argument("--custom", nargs="*", default=[], help="extra checkpoints as name=path")
    ap.add_argument("--input", default="abstract", choices=["abstract", "native"],
                    help="abstract: abstract only for every model (default, same as our model); "
                         "native: each baseline's recommended format (title + abstract)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = args.bf16 and device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    print(f"Device: {device}, bf16: {bf16}")

    jobs = []
    if args.ours:
        jobs.append(("ours", {"hf": args.ours, "max_len": 4096}))
    for item in args.custom:
        name, path = item.split("=", 1)
        jobs.append((name, {"hf": path, "max_len": 4096}))
    for name in args.models:
        if name not in MODELS and name not in NON_NEURAL:
            raise SystemExit(f"Unknown model '{name}'")
        jobs.append((name, MODELS.get(name)))

    _, train_papers = load_papers(args.train_dir or args.data_dir, "train")
    for split in args.splits:
        pmids, papers = load_papers(args.data_dir, split)
        split_dir = Path(args.out_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with open(split_dir / "pmids.json", "w") as f:
            json.dump(pmids, f)
        print(f"{split}: {len(pmids)} papers")

        for name, spec in jobs:
            suffix = "_native" if args.input == "native" and spec and spec.get("native") else ""
            tag = name + suffix
            done = sorted(split_dir.glob(f"{tag}__*.npy")) + sorted(split_dir.glob(f"{tag}.npy"))
            if done and not args.overwrite:
                print(f"  {tag}: already done, skipping ({len(done)} file(s))")
                continue
            if name == "random":
                embs = {None: np.random.default_rng(42).standard_normal((len(papers), 768))}
            elif name == "tfidf":
                embs = {None: embed_tfidf(papers, train_papers)}
            else:
                fmt = spec.get("native", "abstract") if suffix else "abstract"
                embs = embed_transformer(spec, papers, fmt, args.batch_size, device, bf16)
                if spec.get("only"):
                    embs = {spec["only"]: embs[spec["only"]]}
            for pooling, E in embs.items():
                fname = f"{tag}.npy" if pooling is None else f"{tag}__{pooling}.npy"
                np.save(split_dir / fname, E.astype(np.float32))
                print(f"  saved {split}/{fname} {E.shape}")


if __name__ == "__main__":
    main()
