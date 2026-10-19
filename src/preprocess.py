"""Landmark normalisation + augmentation (numpy only, no mediapipe/torch).

normalize() makes features invariant to where the signer stands / how far they
are from the camera: every landmark is expressed relative to the mid-shoulder
point and scaled by shoulder width. It is applied identically at training time
(dataset.py) and at inference time (live_demo.py).
"""
import numpy as np

FEATURE_DIM = 225
HANDS_END = 126                      # first 126 dims are the two hands
_L_SH, _R_SH = 11, 12                # pose indices of the shoulders


def normalize(seq):
    """(T,225) raw MediaPipe coords -> shoulder-centred, shoulder-width-scaled.

    All-zero landmark triples (missing hand/pose) stay exactly zero. If the
    shoulders were not detected in a frame, that frame is left unnormalised
    rather than guessed.
    """
    seq = np.asarray(seq, dtype=np.float32)
    out = seq.copy()
    pts = out.reshape(len(out), 75, 3)           # 42 hand pts + 33 pose pts
    present = np.any(pts != 0, axis=2)           # (T, 75)
    ls = pts[:, 42 + _L_SH]
    rs = pts[:, 42 + _R_SH]
    ok = present[:, 42 + _L_SH] & present[:, 42 + _R_SH]
    center = (ls + rs) / 2
    width = np.linalg.norm((ls - rs)[:, :2], axis=1)
    ok &= width > 1e-4
    scale = np.where(ok, width, 1.0)[:, None, None]
    center = np.where(ok[:, None], center, 0.0)[:, None, :]
    pts[:] = np.where((present & ok[:, None])[..., None], (pts - center) / scale, pts)
    return out


def augment(seq, rng=None):
    """Random augmentation of a *normalised* (T,225) sequence."""
    rng = rng or np.random.default_rng()
    seq = seq.copy()
    t = len(seq)
    pts = seq.reshape(t, 75, 3)
    present = np.any(pts != 0, axis=2)[..., None]

    # random in-plane rotation + scale + translation
    th = rng.uniform(-0.2, 0.2)
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts[..., :2] = pts[..., :2] @ rot.T
    pts *= rng.uniform(0.9, 1.1)
    pts += np.where(present, rng.normal(0, 0.03, size=(1, 1, 3)), 0)
    # per-landmark jitter
    pts += np.where(present, rng.normal(0, 0.01, size=pts.shape), 0)

    # time stretch (0.8x-1.25x) by linear resampling
    if t > 3:
        new_t = max(4, int(round(t * rng.uniform(0.8, 1.25))))
        src = np.linspace(0, t - 1, new_t)
        lo = np.floor(src).astype(int)
        hi = np.minimum(lo + 1, t - 1)
        w = (src - lo)[:, None]
        seq = (seq[lo] * (1 - w) + seq[hi] * w).astype(np.float32)
        t = new_t

    # drop a few random frames (simulates tracking dropouts)
    if t > 8:
        keep = rng.random(t) > 0.05
        if keep.sum() >= 4:
            seq = seq[keep]
    return seq.astype(np.float32)
