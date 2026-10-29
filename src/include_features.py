"""Bridge: our 225-dim MediaPipe-Tasks landmark vectors -> INCLUDE's 134-dim format.

INCLUDE's pretrained model was trained on keypoints from the (now removed) legacy
`mp.solutions` Hands/Pose, laid out per frame as:
    pose[0:25] (x,y)  |  hand1[21] (x,y)  |  hand2[21] (x,y)      -> 50+42+42 = 134
with x scaled by 1920 and y by 1080, missing landmarks linearly interpolated over
time, and a single visible hand assigned to hand1/hand2 by wrist proximity.
The Tasks-API PoseLandmarker's first 25 landmarks share indices with the legacy
upper-body pose, so the bridge is a re-indexing, not a retrain.
numpy only (no mediapipe/torch) so it is unit-testable.
"""
import numpy as np

FRAME_W, FRAME_H = 1920.0, 1080.0
_LEFT_WRIST, _RIGHT_WRIST = 15, 16


def _interp_nan(a):
    """Linearly interpolate NaN rows over time per column; all-NaN column -> 0."""
    out = a.copy()
    t = np.arange(len(a))
    for c in range(a.shape[1]):
        col = out[:, c]
        good = ~np.isnan(col)
        out[:, c] = np.interp(t, t[good], col[good]) if good.any() else 0.0
    return out


# Typical signer placement in the INCLUDE training videos (1920x1080 px): median shoulder
# midpoint and shoulder width over the test clips. The model uses RAW pixel coordinates, so
# a webcam signer who sits closer / off-centre must be mapped onto this placement.
REF_CENTER = np.array([950.0, 393.0])
REF_SHOULDER_W = 260.0


def to_include_features(seq, two_hand_order="raised_first", aspect=16 / 9, align=True):
    """(T,225) raw landmark sequence -> (T,134) INCLUDE features.

    aspect: width/height of the source frame. The model was trained on 16:9 video, so a 4:3
    webcam frame is mapped to a centred 16:9 canvas with the SAME pixel scale (no stretching).

    align: map the clip's median shoulder midpoint/width onto the INCLUDE reference placement
    (scale + shift, no rotation) so camera distance/position doesn't matter.

    two_hand_order: how to order hand1/hand2 when BOTH hands are visible in a frame
    ("track" = keep each hand in the slot it had in the previous frame, first frame raised hand
    first; "raised_first" = hand with the smaller wrist y first, "detected" = MediaPipe order,
    "wrist" = by proximity to pose wrists)."""
    seq = np.asarray(seq, dtype=np.float64)
    T = len(seq)
    pts = seq.reshape(T, 75, 3)[..., :2]          # x,y only
    present = np.any(seq.reshape(T, 75, 3) != 0, axis=2)

    h1 = np.where(present[:, 0:21, None], pts[:, 0:21], np.nan)
    h2 = np.where(present[:, 21:42, None], pts[:, 21:42], np.nan)
    pose = np.where(present[:, 42:67, None], pts[:, 42:67], np.nan)   # first 25 pose pts

    # Slot assignment: hand1 <-> pose wrist 15, hand2 <-> pose wrist 16, chosen per frame by
    # proximity. (The Tasks API orders detected hands differently from the legacy API the
    # model was trained on, so detection order cannot be trusted.)
    prev1 = prev2 = None
    for t in range(T):
        if not (present[t, 42 + _LEFT_WRIST] and present[t, 42 + _RIGHT_WRIST]):
            continue
        has1, has2 = present[t, 0], present[t, 21]
        if not (has1 or has2):
            continue
        lw, rw = pts[t, 42 + _LEFT_WRIST], pts[t, 42 + _RIGHT_WRIST]
        d = lambda a, w: np.sum((a - w) ** 2)
        if has1 and has2:
            if two_hand_order == "detected":
                swap = False
            elif two_hand_order == "track" and prev1 is not None and prev2 is not None:
                # keep each hand in the slot it occupied last frame (continuity)
                swap = d(pts[t, 0], prev1) + d(pts[t, 21], prev2) > d(pts[t, 0], prev2) + d(pts[t, 21], prev1)
            elif two_hand_order in ("raised_first", "track"):
                swap = pts[t, 0, 1] > pts[t, 21, 1]
            else:
                swap = d(pts[t, 0], lw) + d(pts[t, 21], rw) > d(pts[t, 0], rw) + d(pts[t, 21], lw)
        else:
            hand = pts[t, 0] if has1 else pts[t, 21]
            swap = (d(hand, rw) < d(hand, lw)) == has1
        if swap:
            h1[t], h2[t] = h2[t].copy(), h1[t].copy()
        if not np.isnan(h1[t, 0, 0]):
            prev1 = h1[t, 0]
        if not np.isnan(h2[t, 0, 0]):
            prev2 = h2[t, 0]

    parts = [_interp_nan(p.reshape(T, -1, 2).reshape(T, -1)) for p in (pose, h1, h2)]
    out = np.concatenate(parts, axis=1)
    out = out.reshape(T, -1, 2)
    if abs(aspect - FRAME_W / FRAME_H) < 0.02:
        out = out * np.array([FRAME_W, FRAME_H])
    else:                                    # uniform scale, centred horizontally
        out[..., 0] = (out[..., 0] - 0.5) * FRAME_H * aspect + FRAME_W / 2
        out[..., 1] = out[..., 1] * FRAME_H
    if align:
        sh = out[:, [11, 12]]                                   # pose shoulders, pixel space
        ok = (sh != 0).any(axis=(1, 2))
        if ok.any():
            mid = np.median(sh[ok].mean(axis=1), axis=0)
            width = np.median(np.linalg.norm(sh[ok, 0] - sh[ok, 1], axis=1))
            if width > 20:
                out = (out - mid) * (REF_SHOULDER_W / width) + REF_CENTER
    return out.reshape(T, -1).astype(np.float32)


def resample_fps(seq, src_fps, dst_fps=30.0):
    """Linearly resample a (T,D) sequence recorded at src_fps to dst_fps.

    INCLUDE clips are ~30 fps and the model uses absolute position embeddings, so a
    webcam loop running at a different rate must be time-normalised before inference.
    """
    seq = np.asarray(seq)
    if len(seq) < 2 or src_fps <= 0 or abs(src_fps - dst_fps) < 1.0:
        return seq
    new_t = max(2, int(round(len(seq) * dst_fps / src_fps)))
    src = np.linspace(0, len(seq) - 1, new_t)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, len(seq) - 1)
    w = (src - lo)[:, None]
    return (seq[lo] * (1 - w) + seq[hi] * w).astype(seq.dtype)
