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
import os
import threading
import time

import cv2
import numpy as np

from include_features import to_include_features, resample_fps
from include_model import load_pretrained, probabilities, checkpoint_labels
from segmenter import SignSegmenter

TOPK = 3
# Demo vocabulary: words that are easy to tell apart and have INCLUDE reference videos.
DEMO_VOCAB = ["hello", "howareyou", "thankyou", "goodmorning", "goodafternoon", "goodevening",
              "goodnight", "alright", "pleased", "i", "you", "we", "they", "he", "she", "it",
              "today", "tomorrow", "yesterday"]
SPEAK_WORDS = True       # speak each recognised word as it is accepted
MIN_VOCAB_MASS = 0.3     # prob. the full 263-way model puts on the demo words (non-demo signs ~0.03)


class Recognizer:
    """Stream of per-frame 225-dim landmark vectors -> recognised (word, alternatives)."""

    def __init__(self, threshold=0.6, fps=30.0, aspect=16 / 9, vocab=None, ckpt=None, debug_dir=None):
        self.model, self.idx_to_label = load_pretrained(ckpt) if ckpt else load_pretrained()
        l2i = {l: i for i, l in self.idx_to_label.items()}
        if not vocab and ckpt:                       # fine-tuned model: listen for every word it was trained on
            vocab = checkpoint_labels(ckpt)
        self.vocab = [w for w in (vocab or DEMO_VOCAB) if w in l2i]
        print("Listening for:", ", ".join(self.vocab))
        self.vocab_ids = np.array([l2i[w] for w in self.vocab])
        self.seg = SignSegmenter(start_frames=3, end_gap=12, min_len=12, max_len=160,
                                pre_roll=8, post_roll=6)
        self.threshold, self.fps, self.aspect = threshold, fps, aspect
        self.debug_dir, self.last_info, self._n = debug_dir, {}, 0

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
        word, top, info = self._decide(clip)
        info["frames"] = len(clip)
        self.last_info = info
        if self.debug_dir:                                    # keep every decision for inspection
            os.makedirs(self.debug_dir, exist_ok=True)
            tag = word or "REJECTED"
            np.savez(os.path.join(self.debug_dir, f"{self._n:03d}_{tag}.npz"), seq=clip,
                     fps=self.fps, aspect=self.aspect, **{k: v for k, v in info.items() if k != "why"})
            self._n += 1
        return (word, top)

    def _decide(self, clip):
        if len(clip) >= self.seg.max_len:                       # hands never came down: not one sign
            return None, [("(hands held up too long)", 0.0)], {"why": "too long"}
        # Hands must be visible for most of the sign itself; skip the pre/post-roll context
        # frames (no hands by design), otherwise short signs like "you" fail this check.
        core = clip[self.seg.pre_roll: len(clip) - self.seg.post_roll]
        core = core if len(core) >= 4 else clip
        hands = core[:, :126].reshape(len(core), 2, 21, 3)
        if np.mean(np.any(hands != 0, axis=(1, 2, 3))) < 0.5:
            return None, [("(hands not visible)", 0.0)], {"why": "few hands"}
        sh = clip.reshape(len(clip), 75, 3)[:, [42 + 11, 42 + 12]]
        if np.mean(np.all(np.any(sh != 0, axis=2), axis=1)) < 0.7:
            return None, [("(body not visible)", 0.0)], {"why": "no shoulders"}
        feats = resample_fps(to_include_features(clip, aspect=self.aspect), self.fps, 30.0)
        full = probabilities(self.model, feats)
        mass = float(full[self.vocab_ids].sum())
        p = full[self.vocab_ids] / max(mass, 1e-9)               # renormalised over demo words
        order = np.argsort(-p)[:TOPK]
        top = [(self.vocab[i], float(p[i])) for i in order]
        info = {"mass": round(mass, 3), "conf": round(top[0][1], 3), "why": ""}
        if mass < MIN_VOCAB_MASS:                                # not one of the demo signs
            return None, [("(not a demo sign)", mass)], {**info, "why": "low mass"}
        if top[0][1] < self.threshold:
            return None, top, {**info, "why": "low confidence"}
        return top[0][0], top, info


def draw_skeleton(frame, vec):
    """Draw hand skeletons (green/magenta) and upper-body pose (blue) from a 225-dim vector."""
    from landmarks import HAND_CONNECTIONS, POSE_CONNECTIONS, HAND_COLORS
    h, w = frame.shape[:2]
    pts = vec.reshape(75, 3)[:, :2] * [w, h]
    for k in range(2):
        hand = pts[k * 21:(k + 1) * 21]
        if not vec.reshape(75, 3)[k * 21:(k + 1) * 21].any():
            continue
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(hand[a].astype(int)), tuple(hand[b].astype(int)), HAND_COLORS[k], 2)
        for p in hand:
            cv2.circle(frame, tuple(p.astype(int)), 4, HAND_COLORS[k], -1)
    pose, raw = pts[42:], vec.reshape(75, 3)[42:]
    for a, b in POSE_CONNECTIONS:
        if raw[a].any() and raw[b].any():
            cv2.line(frame, tuple(pose[a].astype(int)), tuple(pose[b].astype(int)), (255, 200, 0), 2)
    for i in (0, 11, 12, 13, 14, 15, 16, 23, 24):
        if raw[i].any():
            cv2.circle(frame, tuple(pose[i].astype(int)), 5, (0, 200, 255), -1)


def speak_sentence(words, alts, wait=True):
    """LLM sentence from the recognised words, then speak it. wait=False returns once queued."""
    from demo import to_sentence, speak, wait_speech
    try:
        sentence = to_sentence(words, alts)
    except Exception as e:
        print(f"LLM unavailable ({type(e).__name__}); using raw words.")
        from demo import pretty
        sentence = " ".join(pretty(w) for w in words)
    print("Sentence:", sentence)
    speak(sentence)
    if wait:
        wait_speech()
    return sentence


def run_videos(paths, args):
    from landmarks import create_video_landmarkers, extract_video
    rec = Recognizer(args.threshold, vocab=args.vocab.split(",") if args.vocab else None, ckpt=args.ckpt)
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
            _emit(rec.push(extract_video(frame, int(i * 1000 / rec.fps), hlm, plm)), words, alts, rec)
            i += 1
        _emit(rec.flush(), words, alts, rec)
        cap.release(); hlm.close(); plm.close()
        for _ in range(rec.seg.end_gap):          # pause between videos = gap between signs
            rec.push(np.zeros(225))
    if words:
        speak_sentence(words, alts)
    from demo import wait_speech
    wait_speech()


def _emit(result, words, alts, rec=None):
    if result is None:
        return
    word, top = result
    info = rec.last_info if rec else {}
    detail = f"  [mass {info.get('mass', 0):.2f}, {info.get('frames', 0)} frames]" if info else ""
    print("  sign ->", " | ".join(f"{g} {p:.0%}" for g, p in top), detail, "" if word else "(REJECTED)")
    if word:
        words.append(word)
        alts.append(top)
        if SPEAK_WORDS:
            from demo import speak, pretty
            speak(pretty(word))


def run_webcam(args):
    from landmarks import extract, open_camera
    rec = Recognizer(args.threshold, vocab=args.vocab.split(",") if args.vocab else None, ckpt=args.ckpt,
                     debug_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "debug_clips"))
    words, alts, t, last = [], [], 0, time.time()
    fps_est = 30.0
    cap = open_camera(args.camera)
    status = ""
    warn = ""
    ui = {"sentence": ""}                       # filled by the speaking thread
    from references import play_reference, reference_videos
    refs = reference_videos()
    ref_words = [w for w in rec.vocab if w in refs]
    ref_i = 0
    print("Reference videos (R = next, 1-9 = pick):",
          ", ".join(f"{n + 1} {w}" for n, w in enumerate(ref_words)) or "none - run fetch_include_test.py")
    try:                                        # speech self-test: you should HEAR this now
        from demo import speak_blocking
        speak_blocking("Ready")
        print("Speech test: OK (you should have heard 'Ready')")
    except Exception as e:
        print("!!! SPEECH NOT WORKING:", e)
    idle_since = time.time()

    def finish():
        w, a = list(words), list(alts)
        words.clear(); alts.clear()
        ui["sentence"] = "..."

        def work():
            ui["sentence"] = speak_sentence(w, a, wait=False)
        threading.Thread(target=work, daemon=True).start()
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
        if rec.seg.active:
            idle_since = time.time()
        if res is not None:
            idle_since = time.time()
            _emit(res, words, alts, rec)
            i = rec.last_info
            status = ("ACCEPTED " if res[0] else "rejected ") + " | ".join(f"{g} {p:.0%}" for g, p in res[1]) + f"  (mass {i.get('mass', 0):.2f}, {i.get('frames', 0)}f)"
        cv2.putText(frame, ("SIGNING " if rec.seg.active else "idle ") + status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        draw_skeleton(frame, vec)
        if warn:
            cv2.putText(frame, warn, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, " ".join(words), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if ui["sentence"]:
            cv2.putText(frame, ui["sentence"][:60], (10, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
        cv2.putText(frame, "R/1-9 reference | SPACE speak | C clear | Q quit" + (f" | auto in {max(0, args.auto - (time.time() - idle_since)):.0f}s" if words and args.auto else ""),
                    (10, frame.shape[0] - 15) if not warn else (10, frame.shape[0] - 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("sign2speech", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            words.clear(); alts.clear(); ui["sentence"] = ""
        pick = None
        if key == ord("r") and ref_words:
            pick, ref_i = ref_words[ref_i % len(ref_words)], ref_i + 1
        elif ord("1") <= key <= ord("9") and key - ord("1") < len(ref_words):
            pick = ref_words[key - ord("1")]
        if pick:
            from demo import pretty
            play_reference(refs[pick], pretty(pick))
            rec.seg.reset()                     # don't let the pause leak into a sign
            idle_since = time.time()
        if (key == ord(" ") and words) or (words and args.auto and not rec.seg.active
                                           and time.time() - idle_since > args.auto):
            finish()
            idle_since = time.time()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", nargs="+", help="run on video file(s) instead of the webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--vocab", help="comma-separated words to recognise (default: all words in --ckpt, else the demo vocabulary)")
    ap.add_argument("--ckpt", help="fine-tuned checkpoint from finetune_webcam.py")
    ap.add_argument("--auto", type=float, default=4.0,
                    help="speak the sentence automatically after this many idle seconds (0 = only on SPACE)")
    ap.add_argument("--no-speak-words", action="store_true", help="don't speak each word as it is recognised")
    a = ap.parse_args()
    SPEAK_WORDS = not a.no_speak_words
    run_videos(a.video, a) if a.video else run_webcam(a)
