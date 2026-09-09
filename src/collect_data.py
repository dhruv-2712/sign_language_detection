"""Record landmark clips from the webcam.

Controls:
  SPACE - toggle recording a clip on/off (saves clip when you stop)
  N     - cycle to the next label
  Q     - quit

Each clip is saved as data/raw_landmarks/<label>/<n>.npy with shape
(num_frames, 225). Target 15-20 clips per label solo; teammates run the
same script on their own machines for signer diversity.
"""
import os
import cv2
import numpy as np
from landmarks import extract

VOCAB = ["hello", "thankyou", "yes", "no", "please", "water", "help", "name"]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data", "raw_landmarks")


def next_index(label_dir):
    """Return the next unused integer filename in a label directory."""
    os.makedirs(label_dir, exist_ok=True)
    used = []
    for f in os.listdir(label_dir):
        if f.endswith(".npy"):
            try:
                used.append(int(os.path.splitext(f)[0]))
            except ValueError:
                pass
    return (max(used) + 1) if used else 0


def count_clips(label):
    d = os.path.join(DATA_DIR, label)
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith(".npy")])


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    label_idx = 0
    recording = False
    buffer = []
    t = 0

    print("SPACE=record toggle  N=next label  Q=quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vec = extract(frame, t)
        t += 33  # ~30 fps monotonic timestamp for VIDEO mode
        if recording:
            buffer.append(vec)

        label = VOCAB[label_idx]
        status = "REC" if recording else "idle"
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(frame, f"[{label}] clips={count_clips(label)}  {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if recording:
            cv2.putText(frame, f"frames={len(buffer)}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("collect_data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            if not recording:
                label_idx = (label_idx + 1) % len(VOCAB)
        elif key == ord(' '):
            if not recording:
                recording = True
                buffer = []
            else:
                recording = False
                if buffer:
                    label_dir = os.path.join(DATA_DIR, label)
                    idx = next_index(label_dir)
                    path = os.path.join(label_dir, f"{idx}.npy")
                    np.save(path, np.array(buffer))
                    print(f"Saved {path}  shape={np.array(buffer).shape}")
                buffer = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
