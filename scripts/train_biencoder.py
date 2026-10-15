"""
Example:
  python train_biencoder.py --data-dir /content/data/splits --out-dir $BASE/result_bienc --bf16
  python train_biencoder.py ... --neg-types none          # ablation: in-batch negatives only
  python train_biencoder.py ... --neg-types hard          # ablation: hard negatives only
  python train_biencoder.py ... --model allenai/specter2_base
"""
import os
import json
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import average_precision_score
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--pooling", default="cls", choices=["cls", "mean"], help="cls for BGE / SPECTER2, mean for BERT-type")
    ap.add_argument("--neg-types", default="hard,negative,soft",
                    help="comma-separated pair types used as explicit negatives ('negative' = standard), or 'none'")
    ap.add_argument("--no-mask", action="store_true",
                    help="ablation: do not mask same-pathway papers from the in-batch negatives")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32, help="anchors per batch (each brings a positive and a negative)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=200, help="validate every N steps (and at the end of each epoch)")
    ap.add_argument("--patience", type=int, default=4, help="stop after N validations without improvement (0 = off)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="use N positive pairs (quick test)")
    return vars(ap.parse_args())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(data_dir, split):
    with open(Path(data_dir) / f"pathways_{split}.json", encoding='utf-8') as f:
        pathways = json.load(f)
    abstracts, pw = {}, defaultdict(set)
    for e in pathways:
        for r in e['references']:
            abstracts[r['pmid']] = r['abstract']
            pw[r['pmid']].add(e['pathway_id'])
    with open(Path(data_dir) / f"pairs_{split}.jsonl", encoding='utf-8') as f:
        pairs = [json.loads(line) for line in f]
    return abstracts, pw, pairs


class TripletSampler:

    def __init__(self, pairs, neg_types, seed, limit=None):
        self.pos = []
        self.neg = defaultdict(list)
        for p in pairs:
            if p['label'] == 1:
                self.pos += [(p['left'], p['right']), (p['right'], p['left'])]   
            elif p['type'] in neg_types:
                self.neg[p['left']].append(p['right'])
                self.neg[p['right']].append(p['left'])
        if limit:
            self.pos = random.Random(seed).sample(self.pos, min(limit, len(self.pos)))
        self.use_neg = bool(neg_types)
        self.rng = random.Random(seed)

    def epoch(self):
        order = self.pos[:]
        self.rng.shuffle(order)
        out = []
        for a, p in order:
            cands = self.neg.get(a)
            n = self.rng.choice(cands) if (self.use_neg and cands) else None
            out.append((a, p, n))
        return out


def encode(model, tokenizer, texts, max_length, pooling, device):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
    hidden = model(**batch).last_hidden_state
    if pooling == "cls":
        emb = hidden[:, 0]
    else:
        mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        emb = (hidden * mask).sum(1) / mask.sum(1)
    return F.normalize(emb.float(), dim=-1)


def contrastive_loss(anchors, docs, doc_ids, triplets, pw, temperature, mask_pathway=True):
    logits = anchors @ docs.T / temperature                       
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for i, (a, p, _) in enumerate(triplets):
        for j, d in enumerate(doc_ids):
            if j != i and (d == a or (mask_pathway and pw[d] & pw[a])):
                mask[i, j] = True                                
    logits = logits.masked_fill(mask, float('-inf'))
    target = torch.arange(len(triplets), device=logits.device)   
    return F.cross_entropy(logits, target)


@torch.no_grad()
def validate(model, tokenizer, abstracts, pw, max_length, pooling, device, bf16, batch_size=64):
    model.eval()
    pmids = sorted(abstracts)
    embs = []
    for i in range(0, len(pmids), batch_size):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            embs.append(encode(model, tokenizer, [abstracts[p] for p in pmids[i:i + batch_size]],
                               max_length, pooling, device).cpu())
    E = torch.cat(embs).numpy()
    S = E @ E.T
    aps = []
    for i, p in enumerate(pmids):
        rel = np.array([bool(pw[q] & pw[p]) for q in pmids])
        keep = np.arange(len(pmids)) != i
        if rel[keep].any():
            aps.append(average_precision_score(rel[keep], S[i, keep]))
    model.train()
    return float(np.mean(aps))


def main():
    cfg = parse_args()
    set_seed(cfg["seed"])
    neg_types = [] if cfg["neg_types"] == "none" else cfg["neg_types"].split(",")
    save_dir = Path(cfg["out_dir"]) / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(save_dir / "config.json", "w"), indent=2)
    print(f"Results: {save_dir}")

    train_abs, train_pw, train_pairs = load_split(cfg["data_dir"], "train")
    val_abs, val_pw, _ = load_split(cfg["data_dir"], "val")
    sampler = TripletSampler(train_pairs, neg_types, cfg["seed"], cfg["limit"])
    print(f"train anchors per epoch: {len(sampler.pos)}, explicit negatives: {neg_types or 'none (in-batch only)'}, "
          f"pathway-aware masking: {'off' if cfg['no_mask'] else 'on'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = cfg["bf16"] and device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModel.from_pretrained(cfg["model"]).to(device)
    max_length = min(cfg["max_length"], tokenizer.model_max_length)
    print(f"Device: {device}, bf16: {bf16}, model: {cfg['model']}, pooling: {cfg['pooling']}")

    steps_per_epoch = math.ceil(len(sampler.pos) / cfg["batch_size"])
    total = steps_per_epoch * cfg["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg["warmup"] * total), total)

    log = open(save_dir / "log.jsonl", "a")
    best, bad, step = -1.0, 0, 0
    score0 = validate(model, tokenizer, val_abs, val_pw, max_length, cfg["pooling"], device, bf16)
    print(f"step 0 (before training): val pathway mAP {score0:.4f}")
    log.write(json.dumps({"step": 0, "val_pathway_map": score0}) + "\n")

    def checkpoint(score):
        nonlocal best, bad
        log.write(json.dumps({"step": step, "epoch": epoch, "val_pathway_map": score}) + "\n"); log.flush()
        if score > best:
            best, bad = score, 0
            model.save_pretrained(save_dir / "best")
            tokenizer.save_pretrained(save_dir / "best")
            json.dump({"step": step, "epoch": epoch, "val_pathway_map": score, "val_pathway_map_step0": score0},
                      open(save_dir / "best_info.json", "w"), indent=2)
            print(f"  step {step}: val pathway mAP {score:.4f}  -> new best")
        else:
            bad += 1
            print(f"  step {step}: val pathway mAP {score:.4f}  (best {best:.4f})")
        return cfg["patience"] and bad >= cfg["patience"]

    model.train()
    stop = False
    for epoch in range(1, cfg["epochs"] + 1):
        triplets_all = sampler.epoch()
        bar = tqdm(range(0, len(triplets_all), cfg["batch_size"]), desc=f"epoch {epoch}", mininterval=10)
        for i in bar:
            trip = triplets_all[i:i + cfg["batch_size"]]
            negs = [n for _, _, n in trip if n is not None]
            doc_ids = [p for _, p, _ in trip] + negs
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                a_emb = encode(model, tokenizer, [train_abs[a] for a, _, _ in trip], max_length, cfg["pooling"], device)
                d_emb = encode(model, tokenizer, [train_abs[d] for d in doc_ids], max_length, cfg["pooling"], device)
            loss = contrastive_loss(a_emb, d_emb, doc_ids, trip, train_pw, cfg["temperature"], not cfg["no_mask"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            bar.set_postfix({"loss": f"{loss.item():.4f}"})
            if step % 20 == 0:
                log.write(json.dumps({"step": step, "loss": loss.item()}) + "\n")
            if cfg["eval_every"] and step % cfg["eval_every"] == 0:
                if checkpoint(validate(model, tokenizer, val_abs, val_pw, max_length, cfg["pooling"], device, bf16)):
                    stop = True
                    break
        if stop:
            print(f"Early stopping at step {step}")
            break
        if not cfg["eval_every"] or step % cfg["eval_every"] != 0:
            if checkpoint(validate(model, tokenizer, val_abs, val_pw, max_length, cfg["pooling"], device, bf16)):
                print(f"Early stopping at step {step}")
                break

    print(f"Best val pathway mAP {best:.4f} (before training {score0:.4f}); saved to {save_dir / 'best'}")


if __name__ == "__main__":
    main()
