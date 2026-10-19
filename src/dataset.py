"""Dataset + batching for landmark-sequence clips.

Reads clips saved as data/raw_landmarks/<label>/*.npy, each of shape
(num_frames, 225). Sequences are variable length, so collate() pads to the
longest clip in the batch and returns a padding mask the Transformer consumes.

Deliberately depends only on numpy + torch (NOT mediapipe), so training runs on
any machine, including ones without a webcam or the .task model files.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess import normalize, augment

FEATURE_DIM = 225


def discover(data_dir):
    """Scan data_dir for <label>/*.npy clips.

    Returns (samples, label_to_idx) where samples is a list of (path, label).
    """
    samples, labels = [], set()
    for label in sorted(os.listdir(data_dir)):
        ldir = os.path.join(data_dir, label)
        if not os.path.isdir(ldir):
            continue
        for f in sorted(os.listdir(ldir)):
            if f.endswith(".npy"):
                samples.append((os.path.join(ldir, f), label))
                labels.add(label)
    label_to_idx = {l: i for i, l in enumerate(sorted(labels))}
    return samples, label_to_idx


class LandmarkClips(Dataset):
    def __init__(self, samples, label_to_idx, max_len=160, train=False):
        self.samples = samples
        self.label_to_idx = label_to_idx
        self.max_len = max_len
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        seq = np.load(path).astype(np.float32)               # (T, 225)
        if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM:
            raise ValueError(f"{path}: expected (T,{FEATURE_DIM}), got {seq.shape}")
        seq = normalize(seq)
        if self.train:
            seq = augment(seq)
        if len(seq) > self.max_len:                          # uniform temporal subsample
            idx = np.linspace(0, len(seq) - 1, self.max_len).astype(int)
            seq = seq[idx]
        if len(seq) == 0:
            seq = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        return torch.from_numpy(seq), self.label_to_idx[label]


def collate(batch):
    """Pad a list of (seq, label) to a dense batch + padding mask."""
    seqs, labels = zip(*batch)
    lengths = [s.size(0) for s in seqs]
    t_max = max(lengths)
    b, d = len(seqs), seqs[0].size(1)
    x = torch.zeros(b, t_max, d)
    pad_mask = torch.ones(b, t_max, dtype=torch.bool)        # True = padding
    for i, s in enumerate(seqs):
        x[i, : lengths[i]] = s
        pad_mask[i, : lengths[i]] = False
    return x, pad_mask, torch.tensor(labels, dtype=torch.long)
