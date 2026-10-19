"""Offline tests for normalisation, augmentation and sign segmentation."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from preprocess import normalize, augment  # noqa: E402
from segmenter import SignSegmenter  # noqa: E402


def _frame(hand=True, offset=0.0, scale=1.0):
    v = np.zeros(225, dtype=np.float32)
    pts = v.reshape(75, 3)
    pts[42 + 11] = np.array([0.4, 0.5, 0.0]) * scale + offset   # left shoulder
    pts[42 + 12] = np.array([0.6, 0.5, 0.0]) * scale + offset   # right shoulder
    if hand:
        pts[0] = np.array([0.5, 0.3, 0.1]) * scale + offset
    return v


def test_normalize_is_shift_and_scale_invariant():
    a = normalize(_frame(offset=0.0, scale=1.0)[None])
    b = normalize(_frame(offset=0.2, scale=1.7)[None])
    assert np.allclose(a, b, atol=1e-4)


def test_normalize_keeps_missing_landmarks_zero():
    out = normalize(_frame(hand=False)[None]).reshape(75, 3)
    assert not out[0].any() and not out[21].any()


def test_augment_changes_values_keeps_shape_dim():
    seq = normalize(np.stack([_frame()] * 30))
    out = augment(seq, np.random.default_rng(0))
    assert out.ndim == 2 and out.shape[1] == 225 and not np.allclose(out[:len(seq)], seq[:len(out)])


def test_segmenter_emits_one_clip_per_sign():
    seg = SignSegmenter(start_frames=3, end_gap=5, min_len=8)
    clips = []
    stream = [_frame(hand=False)] * 5 + [_frame()] * 20 + [_frame(hand=False)] * 8 \
        + [_frame()] * 15 + [_frame(hand=False)] * 8
    for f in stream:
        c = seg.push(f)
        if c is not None:
            clips.append(c)
    assert [len(c) for c in clips] == [20, 15]


def test_segmenter_ignores_short_blips():
    seg = SignSegmenter(start_frames=3, end_gap=5, min_len=8)
    stream = [_frame()] * 2 + [_frame(hand=False)] * 10
    assert all(seg.push(f) is None for f in stream)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
