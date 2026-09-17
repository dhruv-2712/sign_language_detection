"""Shared landmark extraction. Produces a 225-dim feature vector per frame:
2 hands x 21 landmarks x 3 (xyz) = 126, plus pose 33 x 3 = 99.
Missing hand/pose in a frame => zeros (keeps sequence length consistent).
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BaseOptions = mp_python.BaseOptions
Mode = vision.RunningMode

# Resolve model paths relative to the repo root, not the current working dir,
# so the module works no matter where a script is launched from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HAND_MODEL = os.path.join(_REPO_ROOT, "models", "hand_landmarker.task")
_POSE_MODEL = os.path.join(_REPO_ROOT, "models", "pose_landmarker_lite.task")

hand_lm = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_HAND_MODEL),
        running_mode=Mode.VIDEO, num_hands=2))
pose_lm = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_POSE_MODEL),
        running_mode=Mode.VIDEO))


def _vectorize(hands, pose):
    """Assemble a HandLandmarker + PoseLandmarker result pair into a (225,) vector.

    Layout: 2 hands x 21 landmarks x 3 (xyz) = 126, then pose 33 x 3 = 99.
    Missing hand/pose slots stay zero so every frame is the same length.
    """
    hf = np.zeros((2, 21, 3))
    for i, h in enumerate(hands.hand_landmarks[:2]):
        for j, lm in enumerate(h):
            hf[i, j] = [lm.x, lm.y, lm.z]

    pf = np.zeros((33, 3))
    if pose.pose_landmarks:
        for j, lm in enumerate(pose.pose_landmarks[0]):
            pf[j] = [lm.x, lm.y, lm.z]

    return np.concatenate([hf.flatten(), pf.flatten()])


def extract(frame_bgr, timestamp_ms):
    """Return a (225,) float array of landmarks for one frame (live/VIDEO mode).

    timestamp_ms must be monotonically increasing across calls (VIDEO mode).
    Used by the live webcam paths (collect_data.py, demo.py).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hands = hand_lm.detect_for_video(img, timestamp_ms)
    pose = pose_lm.detect_for_video(img, timestamp_ms)
    return _vectorize(hands, pose)


def create_image_landmarkers():
    """Build fresh HandLandmarker + PoseLandmarker in IMAGE mode.

    IMAGE mode has no monotonic-timestamp constraint, so it is the right choice
    for offline batch processing of many separate video files (see
    video_to_landmarks.py). Returns (hand_landmarker, pose_landmarker).
    """
    hlm = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_HAND_MODEL),
            running_mode=Mode.IMAGE, num_hands=2))
    plm = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_POSE_MODEL),
            running_mode=Mode.IMAGE))
    return hlm, plm


def extract_image(frame_bgr, hand_landmarker, pose_landmarker):
    """Return a (225,) vector for one frame using IMAGE-mode landmarkers."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hands = hand_landmarker.detect(img)
    pose = pose_landmarker.detect(img)
    return _vectorize(hands, pose)


if __name__ == "__main__":
    # Verification loop: confirm nonzero values appear when a hand is in frame.
    cap = cv2.VideoCapture(0)
    t = 0
    print("Press Q to quit. Raise a hand into frame and watch for nonzero counts.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vec = extract(frame, t)
        t += 33  # ~30 fps
        nonzero = int(np.count_nonzero(vec))
        cv2.putText(frame, f"dim={vec.shape[0]} nonzero={nonzero}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("landmarks", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
