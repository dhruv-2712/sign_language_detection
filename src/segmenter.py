"""Turn a continuous stream of per-frame landmark vectors into isolated sign clips.

A sign is a run of frames in which at least one hand is visible. The clip ends
once the hands have been absent for `end_gap` frames; clips shorter than
`min_len` are discarded as noise. Pure numpy, so it is unit-testable offline.
"""
import numpy as np

HANDS_END = 126


def hand_present(vec):
    return bool(np.any(vec[:HANDS_END] != 0))


class SignSegmenter:
    def __init__(self, start_frames=3, end_gap=10, min_len=8, max_len=150):
        self.start_frames, self.end_gap = start_frames, end_gap
        self.min_len, self.max_len = min_len, max_len
        self.reset()

    def reset(self):
        self._buf, self._pending, self._gap, self.active = [], [], 0, False

    def push(self, vec):
        """Feed one frame. Returns a (T,225) clip when a sign just ended, else None."""
        present = hand_present(vec)
        if not self.active:
            self._pending = self._pending + [vec] if present else []
            if len(self._pending) >= self.start_frames:
                self.active, self._buf, self._gap = True, self._pending, 0
                self._pending = []
            return None
        self._buf.append(vec)
        self._gap = 0 if present else self._gap + 1
        if self._gap >= self.end_gap or len(self._buf) >= self.max_len:
            return self._finish()
        return None

    def _finish(self):
        clip = self._buf[: len(self._buf) - self._gap] if self._gap else self._buf
        self.reset()
        return np.stack(clip) if len(clip) >= self.min_len else None
