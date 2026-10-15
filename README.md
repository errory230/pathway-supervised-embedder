# PathwaySupervisedEmbedder

**Pathway-supervised document embeddings for biomedical literature (KEGG-BGE)**

Code for the paper *Pathway-supervised document embeddings for biological context* (under review).

Biomedical papers that study the same biological process often use different words, and papers that share words may study different processes. We use the literature references of KEGG pathway maps as supervision: papers cited by the same pathway are treated as sharing a biological context. This repository contains

- a **benchmark** that scores document embeddings on unseen KEGG pathways at three levels (category, subcategory, pathway) by retrieval, clustering and pair discrimination;
- **KEGG-BGE**, [BGE-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) fine-tuned with a pathway-supervised contrastive loss;
- the **baselines** (TF-IDF, BioBERT, SciBERT, PubMedBERT, SPECTER, SPECTER2, MedCPT, BGE, Longformer) and a Longformer **cross-encoder** trained on the same pairs;
- **external validation** sets built from Reactome and MeSH.

## Using the model

The fine-tuned model is on the Hugging Face Hub: [`HF_ID/kegg-bge-base-en-v1.5`](https://huggingface.co/HF_ID/kegg-bge-base-en-v1.5).

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("HF_ID/kegg-bge-base-en-v1.5")
emb = model.encode(["<abstract 1>", "<abstract 2>"])   # L2-normalized, 768-d
print(emb @ emb.T)                                        # cosine similarity
```

Give the abstract only, without a query instruction (as in training). Seeds 1 and 2 are available with `revision="seed1"` / `revision="seed2"`.

## Repository layout

```
scripts/
  train_biencoder.py      KEGG-BGE: contrastive fine-tuning (Section 2.4)
  train_crossencoder.py   Longformer cross-encoder pair classifier (Section 2.5)
  embed.py                embed papers with any model, cache one .npy per model and pooling
  evaluate.py             retrieval / clustering / pair metrics, cluster-bootstrap CIs, paired tests
  build_mesh.py           MeSH external validation set (Section 2.6)
  build_reactome.py       Reactome external validation set (Section 2.6)
  external_utils.py       shared helpers (NCBI E-utilities client, pair construction)
notebooks/                Google Colab notebooks that reproduce the paper, in order
  1_crossencoder.ipynb            cross-encoder (Section 3.6)
  2_baselines.ipynb               off-the-shelf models (Table 1)
  3_biencoder.ipynb               KEGG-BGE: learning rate, seeds, centering, ablation (Tables 1-2, S2, S3, S5)
  4_external_validation.ipynb     Reactome and MeSH (Table S6)
  5_figures.ipynb                 Figures 1-3
  6_release_huggingface.ipynb     convert and upload the model to the Hugging Face Hub
```

## Installation

```bash
git clone https://github.com/GITHUB_ID/pathway-supervised-embedder.git
cd pathway-supervised-embedder
pip install -r requirements.txt
pip install adapters          # only for SPECTER2; install last, it may change the transformers version
```

Training was run on a single A100 GPU (bf16) in Google Colab. Each bi-encoder run takes about 20-40 minutes; evaluation and figures run on CPU.

## Data format

The benchmark files are not distributed in this repository (see the Data availability statement of the paper). All scripts expect one folder per data set with the following files:

```
data/splits/
  pathways_{train,val,test}.json
  pairs_{train,val,test}.jsonl
```

`pathways_<split>.json` is a list of pathways with their papers:

```json
[{"pathway_id": "map00010", "pathway_name": "Glycolysis / Gluconeogenesis",
  "category": "Metabolism", "subcategory": "Carbohydrate metabolism",
  "references": [{"pmid": "12345678", "title": "...", "abstract": "..."}]}]
```

`pairs_<split>.jsonl` has one labelled pair per line; `type` is `pos`, `hard` (same subcategory), `negative` (standard: same category) or `soft` (different category):

```json
{"left": "12345678", "right": "23456789", "label": 0, "type": "hard"}
```

Pathways are split 70/15/15 at the pathway level, and no paper occurs in more than one split. The external sets written by `build_mesh.py` and `build_reactome.py` use the same format (test split only; MeSH uses the keys `level1`-`level4`).

## Reproducing the paper

The notebooks use a Google Drive folder `BASE` (default `MyDrive/kegg-bge`) with the benchmark in `BASE/data/splits/`; results are written next to it. Run them in order. The same steps from the command line:

```bash
D=data/splits; O=results

# KEGG-BGE (learning rate 5e-6 chosen on validation; seeds 42, 1, 2)
python scripts/train_biencoder.py --data-dir $D --out-dir $O/bienc --bf16 --lr 5e-6 --eval-every 50 --epochs 1 --seed 42

# Cross-encoder
python scripts/train_crossencoder.py --data-dir $D --out-dir $O/crossencoder --epochs 5 --bf16

# Embeddings for all models (abstract only, then each model's recommended title + abstract input)
python scripts/embed.py --data-dir $D --out-dir $O/embeddings --bf16 \
    --custom kegg_bge_s42=$O/bienc/run-XXXX/best --ours $O/crossencoder/run-XXXX/best \
    --models longformer biobert scibert pubmedbert specter medcpt bge tfidf random
python scripts/embed.py --data-dir $D --out-dir $O/embeddings --bf16 --input native --models specter medcpt

# Evaluation: pooling chosen on validation, reported on test, paired differences against KEGG-BGE
python scripts/evaluate.py --data-dir $D --emb-dir $O/embeddings --out-dir $O/eval --reference kegg_bge_s42 \
    --pair-preds cross_encoder=$O/crossencoder/run-XXXX/test_predictions.jsonl

# External validation (no training)
python scripts/build_mesh.py --email you@lab.org --kegg-dir $D --out-dir $O/external/mesh
python scripts/embed.py --data-dir $O/external/mesh --train-dir $D --splits test --out-dir $O/emb_mesh --bf16 \
    --custom kegg_bge_s42=$O/bienc/run-XXXX/best --models bge tfidf pubmedbert
python scripts/evaluate.py --data-dir $O/external/mesh --emb-dir $O/emb_mesh --out-dir $O/eval_mesh \
    --level-keys level1,level2,level3,level4 --pooling-from $O/eval/pooling_choice.json --reference kegg_bge_s42
```

`evaluate.py` writes `report_test.md` (tables with 95% cluster-bootstrap CIs), `all_variants_test.csv`, `pooling_choice.json` and `per_paper_scores.csv.gz` (used by the figure notebook). Run any script with `-h` for all options.

### Main settings

| | |
|---|---|
| Base model | BAAI/bge-base-en-v1.5, [CLS] pooling, L2-normalized, max 512 tokens |
| Loss | InfoNCE, temperature 0.05, in-batch negatives with same-pathway papers masked, plus one hierarchy-graded explicit negative per anchor |
| Optimization | AdamW, learning rate 5e-6, batch 32 anchors, linear warmup 10%, validation every 50 steps, best checkpoint by validation pathway mAP |
| Evaluation | cosine similarity; 95% CIs from 1,000 cluster-bootstrap resamples over pathways |

## External data sources

- KEGG: Kanehisa Laboratories; KEGG data are subject to the [KEGG license](https://www.kegg.jp/kegg/legal.html).
- Reactome: [ContentService](https://reactome.org/ContentService/), CC BY 4.0.
- PubMed / MeSH: [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/). Please give your own e-mail address (`--email`) and, optionally, an API key.

## Citation

The citation will be added after publication.

## License

Code: MIT (see `LICENSE`). The fine-tuned model is released under the MIT license of its base model.
