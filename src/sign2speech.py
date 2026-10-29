"""End-to-end pipeline with the PRETRAINED INCLUDE Transformer (263 ISL words):

    camera/video -> MediaPipe hands+pose -> hand-raise segmentation
      -> INCLUDE feature bridge -> pretrained Transformer (top-k words)
      -> LLM via Groq (words + runner-up guesses -> one fluent sentence) -> TTS

Usage:
    python src/sign2speech.py                      # live webcam
    python src/sign2speech.py --video a.MOV b.MOV  # run the same pipeline on video files
Keys (webcam): SPACE = finish sentence (LLM + speech), C = clear, Q = quit.
Without GROQ_API_KEY the recognised words are spoken as-is.
"""
import argparse
import time

import cv2
import numpy as np

from include_features import to_include_features, resample_fps
from include_model import load_pretrained, predict
from segmenter import SignSegmenter

TOPK = 3


class Recognizer:
    """Stream of per-frame 225-dim landmark vectors -> recognised (word, alternatives)."""

    def __init__(self, threshold=0.55, fps=30.0, aspect=16 / 9):
        self.model, self.idx_to_label = load_pretrained()
        self.seg = SignSegmenter(start_frames=3, end_gap=12, min_len=12, max_len=160,
                                pre_roll=8, post_roll=6)
        self.threshold, self.fps, self.aspect = threshold, fps, aspect

    def push(self, vec):
        clip = self.seg.push(vec)
        return None if clip is None else self.classify(clip)

    def flush(self):
        """Close a sign still in progress when the stream ends."""
        if self.seg.active:
            self.seg._gap = self.seg.end_gap
            clip = self.seg._finish()
            return None if clip is None else self.classify(clip)

    def classify(self, clip):
        # Need the signer's shoulders in most frames; otherwise the pose input is garbage.
        sh = clip.reshape(len(clip), 75, 3)[:, [42 + 11, 42 + 12]]
        if np.mean(np.all(np.any(sh != 0, axis=2), axis=1)) < 0.7:
            return None, [("(body not visible)", 0.0)]
        feats = resample_fps(to_include_features(clip, aspect=self.aspect), self.fps, 30.0)
        top = [(self.idx_to_label[i], p) for i, p in predict(self.model, feats, k=TOPK)]
        return (top[0][0], top) if top[0][1] >= self.threshold else (None, top)


def speak_sentence(words, alts):
    from demo import to_sentence, speak
    try:
        sentence = to_sentence(words, alts)
    except Exception as e:
        print(f"LLM unavailable ({type(e).__name__}); using raw words.")
        from demo import pretty
        sentence = " ".join(pretty(w) for w in words)
    print("Sentence:", sentence)
    speak(sentence)
    return sentence


def run_videos(paths, args):
    from landmarks import create_video_landmarkers, extract_video
    rec = Recognizer(args.threshold)
    words, alts = [], []
    for path in paths:
        hlm, plm = create_video_landmarkers()      # fresh tracker per video (timestamps restart)
        cap = cv2.VideoCapture(path)
        rec.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        rec.aspect = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _emit(rec.push(extract_video(frame, int(i * 1000 / rec.fps), hlm, plm)), words, alts)
            i += 1
        _emit(rec.flush(), words, alts)
        cap.release(); hlm.close(); plm.close()
        for _ in range(rec.seg.end_gap):          # pause between videos = gap between signs
            rec.push(np.zeros(225))
    if words:
        speak_sentence(words, alts)


def _emit(result, words, alts):
    if result is None:
        return
    word, top = result
    print("  sign ->", " | ".join(f"{g} {p:.0%}" for g, p in top), "" if word else "(rejected: low confidence)")
    if word:
        words.append(word)
        alts.append(top)


def run_webcam(args):
    from landmarks import extract
    rec = Recognizer(args.threshold)
    words, alts, t, last = [], [], 0, time.time()
    fps_est = 30.0
    cap = cv2.VideoCapture(args.camera)
    status = ""
    warn = ""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        fps_est = 0.9 * fps_est + 0.1 * (1.0 / max(now - last, 1e-3))
        last, rec.fps = now, fps_est
        t += 33
        rec.aspect = frame.shape[1] / frame.shape[0]
        vec = extract(frame, t)
        pts = vec.reshape(75, 3)
        body_ok = bool(pts[42 + 11].any() and pts[42 + 12].any())
        warn = "" if body_ok else "STEP BACK: shoulders not visible"
        res = rec.push(vec)
        if res is not None:
            _emit(res, words, alts)
            status = " | ".join(f"{g} {p:.0%}" for g, p in res[1])
        cv2.putText(frame, ("SIGNING " if rec.seg.active else "idle ") + status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        h, w = frame.shape[:2]
        for x, y, _ in pts[(np.arange(75) < 42) | np.isin(np.arange(75) - 42, [0, 11, 12, 13, 14, 15, 16, 23, 24])]:
            if x or y:
                cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)
        if warn:
            cv2.putText(frame, warn, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, " ".join(words), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("sign2speech", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            words.clear(); alts.clear()
        if key == ord(" ") and words:
            speak_sentence(list(words), list(alts))
            words.clear(); alts.clear()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", nargs="+", help="run on video file(s) instead of the webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.55)
    a = ap.parse_args()
    run_videos(a.video, a) if a.video else run_webcam(a)
