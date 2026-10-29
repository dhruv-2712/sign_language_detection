"""Real-time pipeline: webcam -> landmarks -> segmenter -> SignTransformer -> LLM -> TTS.

Usage:
    python src/live_demo.py [--ckpt models/sign_transformer.pt] [--threshold 0.6]

Sign naturally in front of the camera; each time you drop your hands a word is
recognised and appended to the sentence. Keys:
    SPACE  finish sentence: the Groq LLM turns the words into a fluent sentence, spoken aloud
    C      clear the current words
    Q      quit
Without GROQ_API_KEY the raw words are spoken instead.
"""
import argparse
import os
import threading

import cv2
import numpy as np
import torch

from evaluate import load_model
from landmarks import extract
from preprocess import normalize
from segmenter import SignSegmenter, hand_present

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def classify(model, idx_to_label, clip, max_len, threshold):
    """Return (word, confidence) or (None, confidence) if below threshold."""
    seq = normalize(clip)
    if len(seq) > max_len:
        seq = seq[np.linspace(0, len(seq) - 1, max_len).astype(int)]
    x = torch.from_numpy(seq).unsqueeze(0)
    pad = torch.zeros(1, len(seq), dtype=torch.bool)
    with torch.no_grad():
        probs = torch.softmax(model(x, pad), dim=1)[0]
    conf, i = probs.max(0)
    return (idx_to_label[int(i)] if conf >= threshold else None), float(conf)


def finish_sentence(words, status):
    """Words -> LLM sentence -> TTS (runs in a thread so the video doesn't freeze)."""
    def work():
        try:
            from demo import to_sentence, speak
            try:
                sentence = to_sentence(words)
            except Exception as e:  # no API key / offline: fall back to raw words
                print(f"LLM unavailable ({type(e).__name__}); speaking raw words.")
                sentence = " ".join(words)
            status["sentence"] = sentence
            print("Sentence:", sentence)
            speak(sentence)
        except Exception as e:
            print("TTS failed:", e)
    threading.Thread(target=work, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(REPO_ROOT, "models", "sign_transformer.pt"))
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    model, label_to_idx, cfg = load_model(args.ckpt)
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    seg = SignSegmenter()
    words, status = [], {"sentence": "", "last": ""}

    cap = cv2.VideoCapture(args.camera)
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vec = extract(frame, t)
        t += 33
        clip = seg.push(vec)
        if clip is not None:
            word, conf = classify(model, idx_to_label, clip, cfg["max_len"], args.threshold)
            status["last"] = f"{word or '?'} ({conf:.2f})"
            if word:
                words.append(word)
            print(f"clip of {len(clip)} frames -> {status['last']}")

        color = (0, 255, 0) if seg.active else (200, 200, 200)
        cv2.putText(frame, f"{'SIGNING' if seg.active else 'idle'}  last: {status['last']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "words: " + " ".join(words), (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, status["sentence"][:70], (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("live_demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            words.clear()
            status["sentence"] = ""
        if key == ord(" ") and words:
            finish_sentence(list(words), status)
            words.clear()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
