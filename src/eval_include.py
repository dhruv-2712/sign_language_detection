"""Evaluate the pretrained INCLUDE Transformer on held-out INCLUDE *test* videos,
using OUR MediaPipe Tasks landmark extractor + the 225->134 feature bridge.

This measures the whole pipeline front half (video -> landmarks -> pretrained model),
on videos the pretrained model never trained on.

Usage:
    python src/fetch_include_test.py                 # once: download the test clips
    python src/eval_include.py [--root data/include_test]
"""
import argparse
import collections
import os

import numpy as np

from include_features import to_include_features
from include_model import load_pretrained, predict
from video_to_landmarks import clip_to_vector_sequence_tracked, _find_videos

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def label_of(path):
    cls_dir = os.path.basename(os.path.dirname(path))
    return "".join(c for c in cls_dir if c.isalpha()).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO_ROOT, "data", "include_test"))
    ap.add_argument("--cache", default=os.path.join(REPO_ROOT, "data", "include_test_landmarks"))
    ap.add_argument("--hand_conf", type=float, default=0.3)
    ap.add_argument("--order", default="raised_first", help="two-hand slot order: track|detected|raised_first|wrist")
    args = ap.parse_args()

    model, idx_to_label = load_pretrained()
    vids = _find_videos(args.root, recursive=True)
    if not vids:
        raise SystemExit(f"No videos under {args.root}; run fetch_include_test.py first.")
    os.makedirs(args.cache, exist_ok=True)

    top1 = top3 = 0
    per = collections.defaultdict(lambda: [0, 0])
    wrong = []
    for n, v in enumerate(vids, 1):
        key = os.path.relpath(v, args.root).replace(os.sep, "_") + ".npy"
        cpath = os.path.join(args.cache, key)
        if os.path.exists(cpath):
            seq = np.load(cpath)
        else:
            seq = clip_to_vector_sequence_tracked(v, args.hand_conf)
            np.save(cpath, seq)
        truth = label_of(v)
        preds = [idx_to_label[i] for i, _ in predict(model, to_include_features(seq, args.order), k=3)]
        top1 += preds[0] == truth
        top3 += truth in preds
        per[truth][1] += 1
        per[truth][0] += preds[0] == truth
        if preds[0] != truth:
            wrong.append((truth, preds))
        print(f"[{n}/{len(vids)}] {truth:<14} -> {preds}")
    N = len(vids)
    print(f"\nVideos: {N}   top-1: {top1 / N:.3f}   top-3: {top3 / N:.3f}")
    for k, (c, t) in sorted(per.items()):
        print(f"  {k:<16} {c}/{t}")


if __name__ == "__main__":
    main()
