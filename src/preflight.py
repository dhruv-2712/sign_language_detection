"""Run this 5 minutes before the demo: checks everything the live demo depends on.

    python src/preflight.py
"""
import glob
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((True, name, detail or ""))
    except Exception as e:  # report every problem, not just the first
        results.append((False, name, f"{type(e).__name__}: {e}"))


def files():
    need = ["models/hand_landmarker.task", "models/pose_landmarker_lite.task",
            "models/pretrained/include_large.pth", "labels/include_labels.json"]
    missing = [n for n in need if not os.path.exists(os.path.join(REPO_ROOT, n))]
    if missing:
        raise FileNotFoundError(f"missing {missing} (run setup_models.py / download_pretrained.py)")


def model():
    from include_model import load_pretrained
    m, labels = load_pretrained()
    return f"{len(labels)} words loaded"


def groq():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set in this terminal (open a NEW terminal after setting it)")
    from demo import to_sentence
    return "sentence: " + to_sentence(["hello", "you", "howareyou"])


def tts():
    import pyttsx3
    e = pyttsx3.init()
    e.say("Sign language demo is ready")
    e.runAndWait()
    return "you should have just heard a voice"


def camera():
    from landmarks import open_camera, extract
    import numpy as np
    cap = open_camera(0)
    ok, frame = cap.read()
    h, w = frame.shape[:2]
    shoulders = 0
    for i in range(30):
        ok, frame = cap.read()
        v = extract(frame, 33 * (i + 1)).reshape(75, 3)
        shoulders += bool(v[42 + 11].any() and v[42 + 12].any())
    cap.release()
    if shoulders < 20:
        raise RuntimeError(f"camera works ({w}x{h}) but your shoulders were seen in only {shoulders}/30 "
                           "frames: step back, face the light, one person in frame")
    return f"{w}x{h}, shoulders seen in {shoulders}/30 frames"


def custom():
    ft = os.path.join(REPO_ROOT, "models", "pretrained", "finetuned_webcam.pth")
    n = len(glob.glob(os.path.join(REPO_ROOT, "data", "webcam_clips", "*", "*.npz")))
    return (f"fine-tuned model present, {n} recorded clips" if os.path.exists(ft)
            else f"NO fine-tuned model yet ({n} recorded clips) - accuracy on your webcam will be lower")


def videos():
    n = len(glob.glob(os.path.join(REPO_ROOT, "data", "include_test", "*", "*", "*")))
    if n == 0:
        raise FileNotFoundError("no backup demo videos (run fetch_include_test.py)")
    return f"{n} backup demo videos"


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for name, fn in [("files", files), ("pretrained model", model), ("Groq LLM", groq), ("speech (TTS)", tts),
                 ("camera + body visible", camera), ("your fine-tuned model", custom), ("backup videos", videos)]:
    check(name, fn)
    ok, n, d = results[-1]
    print(f"[{'OK ' if ok else 'FAIL'}] {n}: {d}", flush=True)

bad = [n for ok, n, _ in results if not ok]
print("\nALL GOOD" if not bad else f"\nFix before the demo: {', '.join(bad)}")
