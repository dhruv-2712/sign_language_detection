"""End-to-end smoke test for the landmark -> Transformer training pipeline.

Generates class-separable SYNTHETIC landmark clips (no webcam, no mediapipe,
no real signs), then trains SignTransformer for a few epochs and asserts it
learns clearly above chance. This proves the model + dataset + training code
are wired correctly before any real data exists.

Run directly:   python tests/test_pipeline.py
Or with pytest:  pytest tests/test_pipeline.py
"""
import os
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from dataset import LandmarkClips, collate, discover, FEATURE_DIM  # noqa: E402
from model import SignTransformer  # noqa: E402


def make_synthetic(root, n_classes=4, clips_per_class=12, rng=None):
    """Write learnable fake clips: each class gets a distinct constant offset
    plus noise, with a random length in [20, 60] frames."""
    rng = rng or np.random.default_rng(0)
    for c in range(n_classes):
        cdir = os.path.join(root, f"class{c}")
        os.makedirs(cdir, exist_ok=True)
        signature = rng.normal(0, 1, size=FEATURE_DIM) * (c + 1)
        for k in range(clips_per_class):
            t = int(rng.integers(20, 60))
            seq = signature[None, :] + rng.normal(0, 0.3, size=(t, FEATURE_DIM))
            np.save(os.path.join(cdir, f"{k}.npy"), seq.astype(np.float32))


def test_pipeline_learns_above_chance():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        n_classes = 4
        make_synthetic(tmp, n_classes=n_classes, clips_per_class=12)
        samples, label_to_idx = discover(tmp)
        assert len(samples) == n_classes * 12
        assert len(label_to_idx) == n_classes

        # simple split: last 3 of each class -> val
        by_class = {}
        for s in samples:
            by_class.setdefault(s[1], []).append(s)
        tr, va = [], []
        for lbl, items in by_class.items():
            va.extend(items[:3])
            tr.extend(items[3:])

        tr_ld = DataLoader(LandmarkClips(tr, label_to_idx), batch_size=8,
                           shuffle=True, collate_fn=collate)
        va_ld = DataLoader(LandmarkClips(va, label_to_idx), batch_size=8,
                           shuffle=False, collate_fn=collate)

        model = SignTransformer(num_classes=n_classes, d_model=64, num_layers=2, max_len=160)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()

        for _ in range(25):
            model.train()
            for x, mask, y in tr_ld:
                opt.zero_grad()
                crit(model(x, mask), y).backward()
                opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, mask, y in va_ld:
                correct += (model(x, mask).argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"synthetic val accuracy: {acc:.3f} (chance = {1/n_classes:.3f})")
        assert acc >= 0.75, f"pipeline failed to learn separable classes (acc={acc:.3f})"


if __name__ == "__main__":
    test_pipeline_learns_above_chance()
    print("OK: pipeline smoke test passed.")
