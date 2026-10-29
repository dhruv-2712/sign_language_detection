"""Offline tests for the INCLUDE feature bridge (no mediapipe/torch needed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from include_features import to_include_features, resample_fps  # noqa: E402
from segmenter import hand_present  # noqa: E402


def _seq(T=10, hand_slot=0):
    s = np.zeros((T, 75, 3))
    s[:, 42 + 15] = [0.7, 0.5, 0]      # left wrist
    s[:, 42 + 16] = [0.3, 0.5, 0]      # right wrist
    for j in range(25):
        s[:, 42 + j, :2] += 0.001 * j
    s[:, hand_slot * 21: hand_slot * 21 + 21, :2] = 0.31   # hand near the RIGHT wrist (x=0.3)
    s[:, hand_slot * 21: hand_slot * 21 + 21, 2] = 0.1
    return s.reshape(T, 225)


def test_shape_and_scaling():
    f = to_include_features(_seq())
    assert f.shape == (10, 134)
    assert abs(f[0, 2 * 15] - 0.715 * 1920) < 5       # pose wrist x scaled by 1920
    assert abs(f[0, 2 * 15 + 1] - 0.515 * 1080) < 5   # y scaled by 1080


def test_lone_hand_goes_to_slot_of_nearer_wrist():
    # hand1 slot filled but it sits at the right wrist -> belongs in hand2 (cols 92:134)
    f = to_include_features(_seq(hand_slot=0))
    assert np.all(f[:, 50:92] == 0) and np.all(f[:, 92:] > 0)


def test_missing_hand_stays_zero_and_gaps_interpolate():
    s = _seq(T=6).reshape(6, 75, 3)
    s[2:4, 0:21] = 0                                    # hand lost for two frames
    f = to_include_features(s.reshape(6, 225))
    assert np.all(f[2:4, 92:134] > 0)                   # lone hand sits at right wrist -> slot 2, gap interpolated
    assert np.all(f[:, 50:92] == 0)                     # slot 1 never seen -> zeros


def test_resample_fps_changes_length():
    x = np.arange(60, dtype=float).reshape(30, 2)
    assert len(resample_fps(x, 15, 30)) == 60
    assert len(resample_fps(x, 30, 30)) == 30


def test_hands_at_sides_do_not_count_as_signing():
    v = np.zeros(225)
    v.reshape(75, 3)[0] = [0.5, 0.9, 0]                 # hand low
    v.reshape(75, 3)[42 + 23] = [0.45, 0.6, 0]          # hip y = 0.6
    assert not hand_present(v)
    v.reshape(75, 3)[0] = [0.5, 0.4, 0]                 # raised
    assert hand_present(v)


def test_4_3_frames_are_not_stretched():
    # same physical hand-to-shoulder distances must give the same pixel geometry as 16:9
    f = to_include_features(_seq(), aspect=4 / 3)
    # 4:3 centre (x=0.5) lands at canvas centre; y scale unchanged
    assert abs(f[0, 2 * 15] - ((0.715 - 0.5) * 1080 * 4 / 3 + 960)) < 5
    assert abs(f[0, 2 * 15 + 1] - 0.515 * 1080) < 5
