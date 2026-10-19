"""Evaluate a trained SignTransformer: overall + per-class accuracy, confusion matrix.

Usage:
    python src/evaluate.py [--data_dir data/raw_landmarks] [--ckpt models/sign_transformer.pt]

Uses the same stratified split (seed/val_frac) as train_transformer.py, so the
reported numbers are on the held-out validation clips.
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from dataset import LandmarkClips, collate, discover
from model import SignTransformer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["config"]
    model = SignTransformer(num_classes=cfg["num_classes"], d_model=cfg["d_model"],
                            num_layers=cfg["num_layers"], dropout=cfg.get("dropout", 0.1),
                            max_len=cfg["max_len"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck["label_to_idx"], cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "data", "raw_landmarks"))
    ap.add_argument("--ckpt", default=os.path.join(REPO_ROOT, "models", "sign_transformer.pt"))
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model, label_to_idx, cfg = load_model(args.ckpt)
    samples, _ = discover(args.data_dir)
    labels = [label_to_idx[l] for _, l in samples]
    _, va = train_test_split(samples, test_size=args.val_frac, random_state=args.seed,
                             stratify=labels)
    loader = DataLoader(LandmarkClips(va, label_to_idx, cfg["max_len"]), batch_size=16,
                        collate_fn=collate)
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, pad, y in loader:
            y_pred += model(x, pad).argmax(1).tolist()
            y_true += y.tolist()

    names = [n for n, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]
    cm = confusion_matrix(y_true, y_pred, labels=range(len(names)))
    print(f"Validation clips: {len(y_true)}   accuracy: {np.mean(np.array(y_true) == np.array(y_pred)):.3f}\n")
    print("Per-class accuracy:")
    for i, n in enumerate(names):
        print(f"  {n:<14} {cm[i, i]}/{cm[i].sum()}  ({cm[i, i] / max(cm[i].sum(), 1):.2f})")
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(" " * 15 + " ".join(f"{n[:6]:>6}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:<14} " + " ".join(f"{v:>6}" for v in cm[i]))


if __name__ == "__main__":
    main()
