"""Fine-tune the pretrained INCLUDE Transformer on clips recorded with record_signs.py.

Adapts the model to YOUR webcam, distance, lighting and signing style while keeping what it
learned from INCLUDE. Takes about a minute on CPU.

Usage:
    python src/finetune_webcam.py [--epochs 30]
    python src/sign2speech.py --ckpt models/pretrained/finetuned_webcam.pth --vocab <recorded words>
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

from include_features import to_include_features, resample_fps, REF_CENTER
from include_model import load_pretrained, pad_features

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_clips(root):
    """[(label, (T,134) features at 30 fps)] from record_signs.py output."""
    data = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*.npz"))):
        z = np.load(path)
        feats = to_include_features(z["seq"], aspect=float(z["aspect"]))
        data.append((os.path.basename(os.path.dirname(path)), resample_fps(feats, float(z["fps"]), 30.0)))
    return data


def include_clips(labels):
    """INCLUDE test-split clips of these words -> (replay half, held-out half), as 30 fps features.

    Uses the landmark cache written by eval_include.py; extracts (and caches) any missing clip.
    """
    from eval_include import label_of
    from video_to_landmarks import _find_videos, clip_to_vector_sequence_tracked
    root = os.path.join(REPO_ROOT, "data", "include_test")
    cache = os.path.join(REPO_ROOT, "data", "lm_video03")
    os.makedirs(cache, exist_ok=True)
    per = {}
    for v in _find_videos(root, recursive=True):
        lb = label_of(v)
        if lb not in labels:
            continue
        cpath = os.path.join(cache, os.path.relpath(v, root).replace(os.sep, "_") + ".npy")
        if os.path.exists(cpath):
            seq = np.load(cpath)
        else:
            seq = clip_to_vector_sequence_tracked(v)
            np.save(cpath, seq)
        per.setdefault(lb, []).append((lb, to_include_features(seq)))
    replay, held = [], []
    for items in per.values():
        replay += items[0::2]
        held += items[1::2]
    return replay, held


def augment(f, rng):
    """Light augmentation in INCLUDE pixel space; never-detected (zero) values stay zero."""
    nz = f != 0
    pts = f.reshape(len(f), -1, 2)
    pts = (pts - REF_CENTER) * rng.uniform(0.95, 1.05) + REF_CENTER + rng.normal(0, 8, size=(1, 1, 2))
    f = np.where(nz, pts.reshape(len(f), -1) + rng.normal(0, 1.5, size=f.shape), 0).astype(np.float32)
    new_t = max(8, int(len(f) * rng.uniform(0.85, 1.15)))        # time stretch
    idx = np.linspace(0, len(f) - 1, new_t)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(f) - 1)
    w = (idx - lo)[:, None]
    return (f[lo] * (1 - w) + f[hi] * w).astype(np.float32)


def accuracy(model, items, ids):
    """Top-1 accuracy restricted to the recorded words (ids: label -> class index)."""
    model.eval()
    names, cols = list(ids), list(ids.values())
    ok = 0
    with torch.no_grad():
        for label, f in items:
            logits = model(torch.from_numpy(pad_features(f))[None])[0]
            ok += names[int(logits[cols].argmax())] == label
    return ok / max(len(items), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "webcam_clips"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "models", "pretrained", "finetuned_webcam.pth"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--replay_weight", type=int, default=2, help="repeat INCLUDE replay clips this many times")
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    clips = load_clips(args.data)
    if not clips:
        raise SystemExit(f"No clips under {args.data}. Run record_signs.py first.")
    model, idx_to_label = load_pretrained()
    l2i = {lb: i for i, lb in idx_to_label.items()}
    labels = sorted({lb for lb, _ in clips})
    unknown = [lb for lb in labels if lb not in l2i]
    if unknown:
        raise SystemExit(f"Not INCLUDE words: {unknown}")
    ids = {lb: l2i[lb] for lb in labels}

    train, val = [], []
    for lb in labels:
        items = [c for c in clips if c[0] == lb]
        rng.shuffle(items)
        k = int(round(len(items) * args.val_frac)) if len(items) >= 4 else 0
        val += items[:k]
        train += items[k:]
    print(f"{len(clips)} clips, {len(labels)} signs: {labels}  (train {len(train)}, val {len(val)})")

    # Replay: mix in INCLUDE clips of the same words so the model learns YOUR signing without
    # forgetting the real signs (without this it collapsed towards one word). Half are trained
    # on, the other half are kept aside to check the model still recognises other signers.
    replay, inc_val = include_clips(labels)
    train += replay * args.replay_weight
    print(f"INCLUDE replay: {len(replay)} clips trained on (x{args.replay_weight}), {len(inc_val)} held out")
    if val:
        print(f"BEFORE: your held-out clips {accuracy(model, val, ids):.3f}"
              + (f" | INCLUDE held-out {accuracy(model, inc_val, ids):.3f}" if inc_val else ""))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    for ep in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        total = 0.0
        for s in range(0, len(order), 8):
            batch = [train[j] for j in order[s:s + 8]]
            x = torch.from_numpy(np.stack([pad_features(augment(f, rng)) for _, f in batch]))
            y = torch.tensor([ids[lb] for lb, _ in batch])
            loss = crit(model(x), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(batch)
        if ep % 5 == 0 or ep == args.epochs:
            msg = f"epoch {ep:3d}  loss {total / len(train):.3f}"
            if val:
                msg += f"   your held-out {accuracy(model, val, ids):.3f}"
            if inc_val:
                msg += f"   INCLUDE held-out {accuracy(model, inc_val, ids):.3f}"
            print(msg)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(), "labels": labels}, args.out)
    print("saved", args.out)
    print("run:  python src/sign2speech.py --ckpt", args.out, " (listens for all", len(labels), "recorded words)")


if __name__ == "__main__":
    main()
