"""Train the SignTransformer on landmark-sequence clips.

Usage:
    python src/train_transformer.py --data_dir data/raw_landmarks --epochs 40

Saves the best-val checkpoint + label map to models/sign_transformer.pt.
This is the real model; train_baseline.py remains only as a disposable
average-pool + MLP sanity check.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import LandmarkClips, collate, discover
from model import SignTransformer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, pad_mask, y in loader:
        x, pad_mask, y = x.to(device), pad_mask.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x, pad_mask)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "data", "raw_landmarks"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "models", "sign_transformer.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples, label_to_idx = discover(args.data_dir)
    if not samples:
        raise SystemExit(f"No .npy clips found under {args.data_dir}. Collect or generate data first.")
    labels = [label_to_idx[l] for _, l in samples]
    n_classes = len(label_to_idx)
    print(f"{len(samples)} clips, {n_classes} classes: {list(label_to_idx)}")

    # Stratify only if every class has >=2 clips (needed for a stratified split).
    counts = np.bincount(labels, minlength=n_classes)
    stratify = labels if counts.min() >= 2 else None
    tr, va = train_test_split(samples, test_size=args.val_frac, random_state=args.seed,
                              stratify=stratify)

    tr_ds = LandmarkClips(tr, label_to_idx, args.max_len)
    va_ds = LandmarkClips(va, label_to_idx, args.max_len)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = SignTransformer(num_classes=n_classes, d_model=args.d_model,
                            num_layers=args.num_layers, max_len=args.max_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, tr_ld, device, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, va_ld, device, criterion)
        sched.step()
        print(f"epoch {epoch:3d}  train_loss {tr_loss:.3f} acc {tr_acc:.3f}  |  "
              f"val_loss {va_loss:.3f} acc {va_acc:.3f}")
        if va_acc >= best_acc:
            best_acc = va_acc
            torch.save({"state_dict": model.state_dict(),
                        "label_to_idx": label_to_idx,
                        "config": {"d_model": args.d_model, "num_layers": args.num_layers,
                                   "max_len": args.max_len, "num_classes": n_classes}},
                       args.out)
    print(f"\nBest val accuracy: {best_acc:.3f}  ->  saved {args.out}")


if __name__ == "__main__":
    main()
