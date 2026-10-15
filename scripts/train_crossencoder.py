"""
Example:
  python train_crossencoder.py --data-dir data/splits --out-dir results/crossencoder --epochs 5 --bf16
  python train_crossencoder.py --resume results/crossencoder/run-YYYYMMDD-HHMMSS --data-dir data/splits
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, average_precision_score
import json
import argparse
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import random
import numpy as np


SEED = 42
BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "data_dir": str(BASE_DIR / "rebuild" / "data" / "splits"),
    "out_dir": str(BASE_DIR / "result"),
    "learning_rate": 1e-5,
    "batch_size": 4,
    "num_epochs": 5,
    "patience": 2,             
    "select_metric": "auroc",  # val metric used to choose the best epoch: auroc | auprc | f1 | loss
    "model": "allenai/longformer-base-4096",
    "attention_window": 512,
    "max_length": 4096,
    "limit": None,
    "seed": SEED,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=CONFIG["data_dir"], help="folder with pairs_<split>.jsonl and pathways_<split>.json")
    ap.add_argument("--out-dir", default=CONFIG["out_dir"])
    ap.add_argument("--lr", dest="learning_rate", type=float, default=CONFIG["learning_rate"])
    ap.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    ap.add_argument("--epochs", dest="num_epochs", type=int, default=CONFIG["num_epochs"])
    ap.add_argument("--patience", type=int, default=CONFIG["patience"])
    ap.add_argument("--select-metric", default=CONFIG["select_metric"], choices=["auroc", "auprc", "f1", "loss"])
    ap.add_argument("--model", default=CONFIG["model"])
    ap.add_argument("--attention-window", type=int, default=CONFIG["attention_window"])
    ap.add_argument("--max-length", type=int, default=CONFIG["max_length"])
    ap.add_argument("--limit", type=int, default=CONFIG["limit"], help="sample N pairs per split (for quick tests)")
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast on CUDA")
    ap.add_argument("--save-every-epoch", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50, help="write batch metrics every N steps")
    ap.add_argument("--resume", default=None, help="run directory to resume (uses its config.json)")
    return vars(ap.parse_args())


def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def append_jsonl(record, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + "\n")


def write_jsonl_lines(records, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


class PathwayDataset:
    def __init__(self, data_dir, split, docs, limit=None, seed=SEED):
        with open(Path(data_dir) / f"pairs_{split}.jsonl", encoding='utf-8') as f:
            pairs = [json.loads(line) for line in f]
        if limit and limit < len(pairs):
            pairs = random.Random(seed).sample(pairs, limit)
        self.items = [{
            'left': {'pmid': p['left'], 'abstract': docs[p['left']]['abstract']},
            'right': {'pmid': p['right'], 'abstract': docs[p['right']]['abstract']},
            'compatible': p['label'],
            'type': p['type'],
        } for p in pairs]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def export_to_json(self):
        return [{'left': it['left']['pmid'], 'right': it['right']['pmid'],
                 'compatible': it['compatible'], 'type': it['type']} for it in self.items]


def get_dataset(data_dir, save_dir, limit=None, seed=SEED):
    docs = {}
    for split in ("train", "val", "test"):
        with open(Path(data_dir) / f"pathways_{split}.json", encoding='utf-8') as f:
            pathways = json.load(f)
        for entry in pathways:
            for ref in entry['references']:
                docs[ref['pmid']] = {'abstract': ref['abstract']}
        references = sum(len(entry['references']) for entry in pathways)
        print(f"{split} pathways: {len(pathways)}, {split} references: {references}")

    datasets = {s: PathwayDataset(data_dir, s, docs, limit, seed) for s in ("train", "val", "test")}

    for s, ds in datasets.items():
        save_json(ds.export_to_json(), os.path.join(save_dir, f"{s}_pairs.json"))
        types = defaultdict(int)
        for item in ds:
            types[item['type']] += 1
        compatible_count = sum(1 for item in ds if item['compatible'] == 1)
        print(f"{s}: {len(ds)} pairs, compatible {compatible_count} ({compatible_count / len(ds) * 100:.2f}%), types {dict(types)}")

    return datasets["train"], datasets["val"], datasets["test"]


def collate_fn(batch, tokenizer, max_length, use_global_attention):
    left_texts = [item['left']['abstract'] for item in batch]
    right_texts = [item['right']['abstract'] for item in batch]
    labels = torch.tensor([item['compatible'] for item in batch], dtype=torch.long)

    inputs = tokenizer(left_texts, text_pair=right_texts, padding=True, truncation='longest_first',
                       max_length=max_length, return_tensors='pt')

    if use_global_attention:
        global_attention_mask = torch.zeros_like(inputs['input_ids'])
        global_attention_mask[:, 0] = 1
        inputs['global_attention_mask'] = global_attention_mask

    meta = [{'left': it['left']['pmid'], 'right': it['right']['pmid'], 'type': it['type']} for it in batch]
    return inputs, labels, meta


def compute_metrics(labels, probs, types, avg_loss):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    m = {
        "loss": avg_loss,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision, "recall": recall, "f1": f1,
        "auroc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else float('nan'),
        "auprc": average_precision_score(labels, probs) if len(set(labels)) > 1 else float('nan'),
    }
    types = np.asarray(types)
    for t in sorted(set(types)):
        mask = types == t
        m[f"acc_{t}"] = accuracy_score(labels[mask], preds[mask])
    return m


def run_epoch(model, dataloader, device, log_path, prefix, optimizer=None, bf16=False, log_every=50):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0
    all_labels, all_probs, all_types, records = [], [], [], []

    progress_bar = tqdm(dataloader, desc=prefix, mininterval=10)
    buffer = []
    with torch.set_grad_enabled(training):
        for inputs, labels, meta in progress_bar:
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
                outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1].detach().cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_labels.extend(labels_np)
            all_probs.extend(probs)
            all_types.extend(m['type'] for m in meta)
            if not training:
                records.extend({**m, 'label': int(l), 'prob': float(p)} for m, l, p in zip(meta, labels_np, probs))

            batch_accuracy = accuracy_score(labels_np, (probs >= 0.5).astype(int))
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_accuracy:.4f}"})
            if training:
                buffer.append({f"{prefix}_batch_loss": loss.item(), f"{prefix}_batch_accuracy": batch_accuracy})
                if len(buffer) >= log_every:
                    write_jsonl_lines(buffer, log_path)
                    buffer = []
    if buffer:
        write_jsonl_lines(buffer, log_path)

    metrics = compute_metrics(all_labels, all_probs, all_types, total_loss / len(dataloader))
    return metrics, records


def save_model_checkpoint(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def fmt(m):
    return ", ".join(f"{k}: {v:.4f}" for k, v in m.items())


def main():
    args = parse_args()
    if args["resume"]:
        save_dir = args["resume"]
        with open(os.path.join(save_dir, "config.json"), encoding='utf-8') as f:
            config = json.load(f)
        config["num_epochs"] = args["num_epochs"]  
        print(f"Resuming {save_dir}")
    else:
        config = args
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(config["out_dir"], f"run-{timestamp}")
        save_json(config, os.path.join(save_dir, "config.json"))
    set_seed(config["seed"])
    batch_log = os.path.join(save_dir, "batch_metrics.jsonl")
    epoch_log = os.path.join(save_dir, "epoch_metrics.jsonl")
    print(f"Results: {save_dir}")

    train_data, val_data, test_data = get_dataset(config["data_dir"], save_dir, config["limit"], config["seed"])

    model_name = config["model"]
    model_config = AutoConfig.from_pretrained(model_name, num_labels=2)
    is_longformer = model_config.model_type == "longformer"
    if is_longformer:
        model_config.attention_window = config["attention_window"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    last_dir = os.path.join(save_dir, "last")
    state_path = os.path.join(save_dir, "trainer_state.pt")
    resuming = bool(args["resume"]) and os.path.exists(state_path)
    if resuming:
        model = AutoModelForSequenceClassification.from_pretrained(last_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

    max_length = min(config["max_length"], tokenizer.model_max_length, model_config.max_position_embeddings)
    print(f"Model: {model_name} ({model_config.model_type}), max_length: {max_length}")

    collate = lambda x: collate_fn(x, tokenizer, max_length, is_longformer)
    g = torch.Generator()
    train_dataloader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, generator=g,
                                  collate_fn=collate, num_workers=config["num_workers"])
    val_dataloader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False,
                                collate_fn=collate, num_workers=config["num_workers"])
    test_dataloader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False,
                                 collate_fn=collate, num_workers=config["num_workers"])

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    bf16 = config["bf16"]
    if bf16 and (device.type != "cuda" or torch.cuda.get_device_capability(0)[0] < 8):
        print("bf16 needs an Ampere+ GPU (A100, L4, RTX 30xx+); training in fp32 instead")
        bf16 = False
    model.to(device)

    metric = config["select_metric"]
    sign = -1 if metric == "loss" else 1
    best_score, best_epoch, bad_epochs, start_epoch = -float('inf'), 0, 0, 1
    best_dir = os.path.join(save_dir, "best")
    if resuming:
        state = torch.load(state_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        best_score, best_epoch, bad_epochs = state["best_score"], state["best_epoch"], state["bad_epochs"]
        start_epoch = state["epoch"] + 1
        print(f"Resumed after epoch {state['epoch']} (best epoch so far: {best_epoch})")
    stopped = resuming and config["patience"] and bad_epochs >= config["patience"]

    for epoch in range(start_epoch, config["num_epochs"] + 1):
        if stopped:
            break
        print(f"Epoch {epoch}/{config['num_epochs']}")
        g.manual_seed(config["seed"] + epoch)  
        train_m, _ = run_epoch(model, train_dataloader, device, batch_log, "train", optimizer, bf16, config.get("log_every", 50))
        print(f"Train  {fmt(train_m)}")
        val_m, val_records = run_epoch(model, val_dataloader, device, batch_log, "val", bf16=bf16)
        print(f"Val    {fmt(val_m)}")

        append_jsonl({"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()},
                      **{f"val_{k}": v for k, v in val_m.items()}}, epoch_log)

        if config["save_every_epoch"]:
            save_model_checkpoint(model, tokenizer, os.path.join(save_dir, f"epoch-{epoch}"))

        score = sign * val_m[metric]
        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            save_model_checkpoint(model, tokenizer, best_dir)
            with open(os.path.join(save_dir, "val_predictions.jsonl"), 'w', encoding='utf-8') as f:
                f.writelines(json.dumps(r) + "\n" for r in val_records)
            print(f"  -> new best (val {metric} = {val_m[metric]:.4f}), saved to {best_dir}")
        else:
            bad_epochs += 1
            if config["patience"] and bad_epochs >= config["patience"]:
                print(f"Early stopping: no improvement in val {metric} for {bad_epochs} epochs")
                stopped = True

        save_model_checkpoint(model, tokenizer, last_dir)
        torch.save({"epoch": epoch, "optimizer": optimizer.state_dict(), "best_score": best_score,
                    "best_epoch": best_epoch, "bad_epochs": bad_epochs}, state_path)

    print(f"Loading best checkpoint (epoch {best_epoch}) for test evaluation")
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    test_m, test_records = run_epoch(model, test_dataloader, device, batch_log, "test", bf16=bf16)
    print(f"Test   {fmt(test_m)}")
    save_json({"best_epoch": best_epoch, "select_metric": metric, **{f"test_{k}": v for k, v in test_m.items()}},
              os.path.join(save_dir, "test_metrics.json"))
    with open(os.path.join(save_dir, "test_predictions.jsonl"), 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(r) + "\n" for r in test_records)


if __name__ == "__main__":
    main()
