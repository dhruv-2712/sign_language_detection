"""Extract 225-dim landmark sequences from video files into data/raw_landmarks/.

Used to turn real ISL video clips (e.g. the AI4Bharat INCLUDE dataset) into the
same .npy format collect_data.py produces from a webcam, so the SAME model and
training code work on both. Videos are processed through our own MediaPipe
Tasks-API extractor (IMAGE mode) -- we do NOT reuse INCLUDE's keypoint format.

Usage:
    python src/video_to_landmarks.py --src <dir-of-videos> --label hello
    python src/video_to_landmarks.py --map "48. Hello=hello,55. Thank you=thankyou" \
        --src <Greetings-dir> --recursive
"""
import argparse
import os

import cv2
import numpy as np

from landmarks import create_image_landmarkers, extract_image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_EXTS = (".mov", ".mp4", ".avi", ".mkv", ".webm")


def clip_to_vector_sequence(path, hlm, plm, stride=1):
    cap = cv2.VideoCapture(path)
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(extract_image(frame, hlm, plm))
        i += 1
    cap.release()
    return np.array(frames, dtype=np.float32) if frames else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory containing videos")
    ap.add_argument("--label", help="single label for all videos under --src")
    ap.add_argument("--map", help='comma list of "<subfolder>=<label>" to process per class')
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "raw_landmarks"))
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max clips per label (0 = all)")
    args = ap.parse_args()

    # Build the list of (video_path, label) to process.
    jobs = []
    if args.map:
        for pair in args.map.split(","):
            sub, label = pair.split("=")
            subdir = os.path.join(args.src, sub.strip())
            vids = _find_videos(subdir, recursive=True)
            if args.limit:
                vids = vids[: args.limit]
            jobs += [(v, label.strip()) for v in vids]
    else:
        if not args.label:
            raise SystemExit("Provide --label (single class) or --map (multi-class).")
        vids = _find_videos(args.src, recursive=args.recursive)
        if args.limit:
            vids = vids[: args.limit]
        jobs += [(v, args.label) for v in vids]

    if not jobs:
        raise SystemExit(f"No videos found under {args.src}")

    hlm, plm = create_image_landmarkers()
    per_label = {}
    for n, (path, label) in enumerate(jobs, 1):
        seq = clip_to_vector_sequence(path, hlm, plm, stride=args.stride)
        if seq is None or len(seq) == 0:
            print(f"[skip] {os.path.basename(path)}: no frames")
            continue
        ldir = os.path.join(args.out, label)
        os.makedirs(ldir, exist_ok=True)
        idx = per_label.get(label, 0)
        per_label[label] = idx + 1
        np.save(os.path.join(ldir, f"{idx}.npy"), seq)
        print(f"[{n}/{len(jobs)}] {label}: {os.path.basename(path)} -> "
              f"{seq.shape}  (nonzero {int(np.count_nonzero(seq.mean(0)))}/225)")
    print("\nSaved clips per label:", per_label)


def _find_videos(root, recursive=False):
    out = []
    if not os.path.isdir(root):
        return out
    if recursive:
        for dp, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(VIDEO_EXTS):
                    out.append(os.path.join(dp, f))
    else:
        for f in os.listdir(root):
            if f.lower().endswith(VIDEO_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


if __name__ == "__main__":
    main()
