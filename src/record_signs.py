"""Record YOUR OWN webcam clips of the demo signs, with a reference video for each sign.

The pretrained INCLUDE model was trained on other signers filmed from far away. A few clips
from your own webcam let finetune_webcam.py adapt it to you, which is what makes the live
demo reliable.

Usage:
    python src/record_signs.py [--labels hello,you,thankyou] [--clips 8]
Keys:  SPACE = arm one recording (then sign; it saves when your hands drop)
       R = play the reference ISL video for this sign   N / P = next / previous sign
       U = delete last clip of this sign                Q = quit
Clips go to data/webcam_clips/<label>/<n>.npz.
"""
import argparse
import os
import time

import cv2
import numpy as np

from landmarks import extract, open_camera
from segmenter import SignSegmenter
from references import play_reference, reference_videos
from sign2speech import DEMO_VOCAB, draw_skeleton

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=",".join(DEMO_VOCAB))
    ap.add_argument("--clips", type=int, default=8, help="target clips per sign")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "webcam_clips"))
    args = ap.parse_args()
    labels = args.labels.split(",")
    refs = reference_videos()

    def count(label):
        d = os.path.join(args.out, label)
        return len([f for f in os.listdir(d) if f.endswith(".npz")]) if os.path.isdir(d) else 0

    seg = SignSegmenter(start_frames=3, end_gap=12, min_len=12, max_len=160, pre_roll=8, post_roll=6)
    cap = open_camera(args.camera)
    i, armed, t, last, fps, flash = 0, False, 0, time.time(), 30.0, ""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
        last = now
        t += 33
        vec = extract(frame, t)
        if armed:
            clip = seg.push(vec)
        else:
            seg.reset()
            clip = None
        label = labels[i]
        if clip is not None:
            os.makedirs(os.path.join(args.out, label), exist_ok=True)
            n = count(label)
            np.savez(os.path.join(args.out, label, f"{n}.npz"), seq=clip, fps=fps,
                     aspect=frame.shape[1] / frame.shape[0])
            flash, armed = f"saved {label} #{n + 1} ({len(clip)} frames)", False
        draw_skeleton(frame, vec)
        cv2.putText(frame, f"{label}  {count(label)}/{args.clips}   [{i + 1}/{len(labels)}]", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        state = "SIGN NOW (hands drop = save)" if armed else "SPACE to arm | R = reference"
        cv2.putText(frame, state, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0) if armed else (255, 255, 255), 2)
        cv2.putText(frame, flash, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 200, 255), 2)
        cv2.imshow("record_signs", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord(" "):
            armed, flash = True, ""
        elif k in (ord("n"), ord("p")):
            i, armed = (i + (1 if k == ord("n") else -1)) % len(labels), False
        elif k == ord("r") and label in refs:
            play_reference(refs[label], label)
        elif k == ord("u") and count(label):
            os.remove(os.path.join(args.out, label, f"{count(label) - 1}.npz"))
    cap.release()
    cv2.destroyAllWindows()
    print({lb: count(lb) for lb in labels if count(lb)})


if __name__ == "__main__":
    main()
