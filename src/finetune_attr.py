#!/usr/bin/env python3
"""Few-shot attribute extraction fine-tuning on SWDE (Section 4.3 of the paper).

Few-shot setting: 10 pages per website for train, rest split 50/50 dev/test.
Zero-shot setting: 2/5 seed websites for train, 1 for dev, rest for test.
"""

import argparse
import csv
import json
import pickle
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.domlm.modeling_domlm import DOMLMForTokenClassification


# ─── Dataset ─────────────────────────────────────────────────────────────────

class PageDataset(Dataset):
    def __init__(self, pkl_files: List[Path]):
        self._items: List[dict] = []
        for f in pkl_files:
            with open(f, "rb") as fp:
                self._items.extend(pickle.load(fp))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def make_few_shot_splits(
    website_dir: Path, n_train: int = 10, seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    files = sorted(website_dir.glob("*.pkl"))
    rng = random.Random(seed)
    all_files = list(files)
    rng.shuffle(all_files)
    train_files = all_files[:n_train]
    rest = all_files[n_train:]
    mid = max(1, len(rest) // 2)
    return train_files, rest[:mid], rest[mid:]


def make_zero_shot_splits(
    domain_dir: Path, n_seed: int = 2, seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Use n_seed websites for train (10% pages), 1 for dev, rest for test."""
    websites = sorted(d for d in domain_dir.iterdir() if d.is_dir() and not d.name.endswith('.json'))
    rng = random.Random(seed)
    rng.shuffle(websites)
    seed_sites = websites[:n_seed]
    dev_site = websites[n_seed] if len(websites) > n_seed else None
    test_sites = websites[n_seed + 1:]

    train_files = []
    for site in seed_sites:
        files = sorted(site.glob("*.pkl"))
        n = max(1, len(files) // 10)
        train_files.extend(rng.sample(files, n))

    dev_files = sorted(dev_site.glob("*.pkl")) if dev_site else []
    test_files = []
    for site in test_sites:
        test_files.extend(sorted(site.glob("*.pkl")))

    return train_files, dev_files, test_files


# ─── Data collator ───────────────────────────────────────────────────────────

_PAD_VALUES = {
    "input_ids": 1,
    "attention_mask": 0,
    "labels": -100,
    "node_ids": 0,
    "parent_node_ids": 0,
    "sibling_node_ids": 0,
    "depth_ids": 0,
    "tag_ids": 0,
}

@dataclass
class DataCollatorForDOMTokenClassification:
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {}
        for key, pad_val in _PAD_VALUES.items():
            if key not in features[0]:
                continue
            padded = []
            for f in features:
                arr = list(f[key])
                arr += [pad_val] * (max_len - len(arr))
                padded.append(arr)
            batch[key] = torch.tensor(padded, dtype=torch.long)
        return batch


# ─── Evaluation ──────────────────────────────────────────────────────────────

def compute_f1(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Token-level macro F1 over attribute classes (excluding background class 0)."""
    mask = (labels != -100) & (labels != 0)
    pred_flat = preds[mask]
    label_flat = labels[mask]

    if len(label_flat) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = int((pred_flat == label_flat).sum())
    precision = tp / len(pred_flat) if len(pred_flat) > 0 else 0.0
    recall = tp / len(label_flat)
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def evaluate(model: DOMLMForTokenClassification, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(-1)
        all_preds.append(preds.cpu().numpy().flatten())
        all_labels.append(labels.cpu().numpy().flatten())

    return compute_f1(np.concatenate(all_preds), np.concatenate(all_labels))


# ─── Training ────────────────────────────────────────────────────────────────

def train_one(
    train_files: List[Path],
    dev_files: List[Path],
    test_files: List[Path],
    model_path: str,
    num_labels: int,
    args,
) -> Dict:
    model = DOMLMForTokenClassification.from_pretrained(
        model_path, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    model.to(args.device)

    collator = DataCollatorForDOMTokenClassification()
    train_loader = DataLoader(PageDataset(train_files), batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(PageDataset(dev_files), batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(PageDataset(test_files), batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_dev_f1 = -1.0
    best_state: Optional[Dict] = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(args.device)
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss = model(**batch, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        dev_metrics = evaluate(model, dev_loader, args.device)
        print(f"    epoch {epoch+1}/{args.epochs}  loss={total_loss/len(train_loader):.4f}  dev_f1={dev_metrics['f1']:.4f}")

        if dev_metrics["f1"] >= best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, args.device)
    return {**test_metrics, "best_dev_f1": best_dev_f1}


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DOM-LM for attribute extraction")
    parser.add_argument("--data_dir",     type=str, required=True, help="Root of swde_attr_preprocessed/")
    parser.add_argument("--model_path",   type=str, required=True, help="Pre-trained DOM-LM checkpoint")
    parser.add_argument("--output_dir",   type=str, default="results_attr")
    parser.add_argument("--domain",       type=str, default="restaurant",
                        help="One of: auto,book,camera,job,movie,nbaplayer,restaurant,university")
    parser.add_argument("--setting",      type=str, default="few_shot", choices=["few_shot", "zero_shot"])
    parser.add_argument("--n_train",      type=int, default=10,  help="(few-shot) pages per website for train")
    parser.add_argument("--n_seed",       type=int, default=2,   help="(zero-shot) seed websites for train")
    parser.add_argument("--epochs",       type=int, default=5)
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--device",       type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir) / args.domain
    output_dir = Path(args.output_dir) / args.domain
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "label2id.json") as f:
        label2id = json.load(f)
    num_labels = len(label2id)
    print(f"Domain: {args.domain}  |  Labels ({num_labels}): {label2id}")

    website_results: List[Dict] = []

    if args.setting == "few_shot":
        for website_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
            train_f, dev_f, test_f = make_few_shot_splits(website_dir, args.n_train, args.seed)
            if not train_f or not test_f:
                continue
            print(f"\n  [{website_dir.name}] train={len(train_f)} dev={len(dev_f)} test={len(test_f)} pages")
            result = train_one(train_f, dev_f, test_f, args.model_path, num_labels, args)
            result["website"] = website_dir.name
            website_results.append(result)
            print(f"  → test_f1={result['f1']:.4f}")

    else:  # zero_shot
        train_f, dev_f, test_f = make_zero_shot_splits(data_dir, args.n_seed, args.seed)
        print(f"\n  zero-shot: train={len(train_f)} dev={len(dev_f)} test={len(test_f)} pages")
        result = train_one(train_f, dev_f, test_f, args.model_path, num_labels, args)
        result["website"] = "zero_shot"
        website_results.append(result)
        print(f"  → test_f1={result['f1']:.4f}")

    if website_results:
        avg_f1 = float(np.mean([r["f1"] for r in website_results]))
        print(f"\n=== {args.domain} ({args.setting})  avg_F1 = {avg_f1:.4f} ===")
        csv_path = output_dir / f"results_{args.setting}.csv"
        with open(csv_path, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=website_results[0].keys())
            writer.writeheader()
            writer.writerows(website_results)
        with open(output_dir / f"avg_{args.setting}.json", "w") as fp:
            json.dump({"domain": args.domain, "setting": args.setting, "avg_f1": avg_f1,
                       "num_websites": len(website_results)}, fp, indent=2)
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
