"""Turn a continuous stream of per-frame landmark vectors into isolated sign clips.

A sign is a run of frames in which at least one hand is visible. The clip ends
once the hands have been absent for `end_gap` frames; clips shorter than
`min_len` are discarded as noise. Pure numpy, so it is unit-testable offline.
"""
import numpy as np

HANDS_END = 126


_HIPS = (126 + 23 * 3, 126 + 24 * 3)       # x index of left/right hip in the 225-dim vector
_SH = (126 + 11 * 3, 126 + 12 * 3)         # x index of left/right shoulder
RAISE_BELOW_SHOULDERS = 1.0                # a wrist counts as raised above this many shoulder-widths
                                           # below the shoulder line (about mid-chest)


def hand_present(vec):
    """True if a hand is visible AND raised, i.e. the signer is signing, not resting.

    Preferred test uses the shoulders (works when sitting close, hips out of frame): a wrist
    must be above ~1 shoulder-width below the shoulder line. Falls back to the hip line, and
    finally to "any visible hand" if neither is detected.
    """
    hands = vec[:HANDS_END].reshape(2, 21, 3)
    visible = [h for h in hands if np.any(h != 0)]
    if not visible:
        return False
    lx, ly = vec[_SH[0]], vec[_SH[0] + 1]
    rx, ry = vec[_SH[1]], vec[_SH[1] + 1]
    if (lx or ly) and (rx or ry):
        width = abs(lx - rx)
        if width > 1e-3:
            limit = (ly + ry) / 2 + RAISE_BELOW_SHOULDERS * width
            return any(h[0, 1] < limit for h in visible)        # image y grows downward
    hips_y = [vec[i + 1] for i in _HIPS if vec[i + 1] != 0]
    if not hips_y:
        return True
    return any(h[0, 1] < np.mean(hips_y) for h in visible)


class SignSegmenter:
    """pre_roll/post_roll keep a few frames of context before the first and after the last
    raised-hand frame (the pretrained model saw whole videos, not trimmed hand-up runs)."""

    def __init__(self, start_frames=3, end_gap=10, min_len=8, max_len=150, pre_roll=0, post_roll=0):
        self.start_frames, self.end_gap = start_frames, end_gap
        self.min_len, self.max_len = min_len, max_len
        self.pre_roll, self.post_roll = pre_roll, post_roll
        self._recent = []
        self.reset()

    def reset(self):
        self._buf, self._pending, self._gap, self.active = [], [], 0, False

    def push(self, vec):
        """Feed one frame. Returns a (T,225) clip when a sign just ended, else None."""
        present = hand_present(vec)
        if not self.active:
            self._recent = (self._recent + [vec])[-(self.pre_roll + self.start_frames + 1):]
            self._pending = self._pending + [vec] if present else []
            n = len(self._pending)
            if n >= self.start_frames:
                lead = self._recent[max(len(self._recent) - n - self.pre_roll, 0):len(self._recent) - n]                     if self.pre_roll else []
                self.active, self._buf, self._gap = True, lead + self._pending, 0
                self._pending = []
            return None
        self._buf.append(vec)
        self._gap = 0 if present else self._gap + 1
        if self._gap >= self.end_gap or len(self._buf) >= self.max_len:
            return self._finish()
        return None

    def _finish(self):
        keep = len(self._buf) - max(self._gap - self.post_roll, 0)
        clip = self._buf[:keep]
        self.reset()
        self._recent = []
        return np.stack(clip) if len(clip) >= self.min_len else None
