"""Reference ISL videos (INCLUDE test clips) for each word, used by the R key in the demo tools.

Videos come from data/include_test (run fetch_include_test.py once).
"""
import os

import cv2

from video_to_landmarks import _find_videos

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_WINDOW = "reference sign (any key = stop)"


def reference_videos():
    """label -> one INCLUDE reference video."""
    refs = {}
    for v in _find_videos(os.path.join(REPO_ROOT, "data", "include_test"), recursive=True):
        label = "".join(c for c in os.path.basename(os.path.dirname(v)) if c.isalpha()).lower()
        refs.setdefault(label, v)
    return refs


def play_reference(path, title=""):
    """Play a reference clip in its own window (blocking); any key stops it."""
    cap = cv2.VideoCapture(path)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (640, int(640 * f.shape[0] / f.shape[1])))
        if title:
            cv2.putText(f, title, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.imshow(REF_WINDOW, f)
        if cv2.waitKey(33) != -1:
            break
    cap.release()
    cv2.destroyWindow(REF_WINDOW)
