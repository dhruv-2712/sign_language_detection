# CLAUDE.md — Sign Language Recognition Prototype

Instructions for whichever Claude/agent is working in this repo. Read this whole file before writing code. Work through the phases in order — don't skip to the classifier before the data pipeline is proven.

## Project context

NIE Mysore ISE minor project (BIS586, Sem 5, Batch A10). Team: Dhruvaprasad R, Karan Raj, Chinmaya, Ekanth R. Guide: Suhas Bharadwaj R.

**Title:** Real-Time Continuous Sign Language Recognition with Transformer-Based Landmark Modeling and LLM-Assisted Contextual Translation

**Synopsis pipeline:** webcam → MediaPipe hand+pose landmarks → Transformer classifier (trained by us) → LLM turns recognized signs into a fluent sentence → TTS speaks it. Everything except the classifier is meant to be a pretrained/inference-only component.

**What this repo is right now:** a scaffolding prototype, not the final deliverable. The classifier trained in Phase 4 is a throwaway sanity check to prove the pipeline wiring — it is NOT the Transformer that goes in the final report. Don't over-invest in its accuracy.

## Critical gotchas — read before writing MediaPipe code

- `mediapipe.solutions.holistic` (and `.hands`, `.pose`, `.drawing_utils`) is **removed** in current MediaPipe releases. Do not write code against `mp.solutions.*` — it will fail to import.
- Use the **Tasks API** instead: `mediapipe.tasks.python.vision.HandLandmarker` and `PoseLandmarker`, run separately, combined by us into one feature vector. There is no unified "HolisticLandmarker" task yet — Google has said one is "coming soon" but it isn't shipped.
- Model files (`.task`) must be downloaded once, they're not bundled with pip. URLs are in `setup_models.py` below. Don't hardcode different URLs from memory — use these exact ones.
- Feature vector per frame is 225-dim: 2 hands × 21 landmarks × 3 (xyz) = 126, plus pose 33 × 3 = 99. Missing hand/pose in a frame = zeros, not a skipped frame (keeps sequence length consistent).

## Directory layout

```
sign-project/
  models/                  # .task model files + trained classifiers (gitignored, large)
  data/raw_landmarks/<label>/*.npy   # one .npy per recorded clip, shape (num_frames, 225)
  src/
    setup_models.py
    landmarks.py           # extract() — shared by every other script
    collect_data.py
    train_baseline.py
    demo.py
  README.md
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install mediapipe opencv-python numpy scikit-learn joblib pyttsx3 anthropic
python src/setup_models.py
export ANTHROPIC_API_KEY=...   # needed for demo.py only
```

`setup_models.py`:
```python
import os, urllib.request
MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}
os.makedirs("models", exist_ok=True)
for name, url in MODELS.items():
    path = f"models/{name}"
    if not os.path.exists(path):
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, path)
```

## Build order (do these in sequence, verify each before moving on)

### Phase 1 — `landmarks.py`
Implement `extract(frame_bgr, timestamp_ms) -> np.ndarray` shape `(225,)` using `HandLandmarker` + `PoseLandmarker` in `VIDEO` running mode. **Verify:** run against a live webcam loop and print the vector — confirm nonzero values appear when a hand is in frame before proceeding.

```python
import cv2, mediapipe as mp, numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BaseOptions = mp_python.BaseOptions
Mode = vision.RunningMode

hand_lm = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=Mode.VIDEO, num_hands=2))
pose_lm = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/pose_landmarker_lite.task"),
        running_mode=Mode.VIDEO))

def extract(frame_bgr, timestamp_ms):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hands = hand_lm.detect_for_video(img, timestamp_ms)
    pose = pose_lm.detect_for_video(img, timestamp_ms)

    hf = np.zeros((2, 21, 3))
    for i, h in enumerate(hands.hand_landmarks[:2]):
        for j, lm in enumerate(h):
            hf[i, j] = [lm.x, lm.y, lm.z]

    pf = np.zeros((33, 3))
    if pose.pose_landmarks:
        for j, lm in enumerate(pose.pose_landmarks[0]):
            pf[j] = [lm.x, lm.y, lm.z]

    return np.concatenate([hf.flatten(), pf.flatten()])
```

### Phase 2 — `collect_data.py`
Vocabulary (edit if the team agrees on different words): `hello, thankyou, yes, no, please, water, help, name`. SPACE toggles recording a clip, N cycles label, Q quits. Save each clip as `data/raw_landmarks/<label>/<n>.npy`. Target 15–20 clips per label solo tonight; other three team members run the same script on their own machines later for signer diversity.

### Phase 3 — dataset acquisition (background, don't block on this)
```bash
git clone https://github.com/AI4Bharat/INCLUDE.git
```
Do **not** download the full INCLUDE zip set from https://zenodo.org/records/4010759 — it's ~57 GB total across category shards. Only pull the specific category zips covering the vocabulary above, checking each zip's listed size first. Also worth checking `github.com/AI4Bharat/OpenHands` for a pretrained Indian Sign Language checkpoint — if one covers our vocabulary, fine-tuning it is faster than training the real Transformer from scratch next session.

### Phase 4 — `train_baseline.py` (placeholder classifier — see note above)
Average-pool each clip's landmark sequence over time, train a small `MLPClassifier` on it, save with `joblib`. This exists only to prove data → features → prediction works, not as the real model.

```python
import os, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

DATA_DIR = "data/raw_landmarks"
X, y = [], []
for label in os.listdir(DATA_DIR):
    for f in os.listdir(os.path.join(DATA_DIR, label)):
        seq = np.load(os.path.join(DATA_DIR, label, f))
        X.append(seq.mean(axis=0))
        y.append(label)

X, y = np.array(X), np.array(y)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500).fit(X_tr, y_tr)
print("test acc:", clf.score(X_te, y_te))
joblib.dump(clf, "models/baseline_classifier.joblib")
```

### Phase 5 — `demo.py`
Wire recognized words → Claude (fluent sentence) → `pyttsx3` (offline TTS, no API key needed for this part). Use model `claude-haiku-4-5-20251001` for speed/cost.

```python
import joblib, pyttsx3
from anthropic import Anthropic

clf = joblib.load("models/baseline_classifier.joblib")
client = Anthropic()
engine = pyttsx3.init()

def to_sentence(words):
    prompt = f"Convert these sign-language glosses into one fluent English sentence, reply with only the sentence: {' '.join(words)}"
    msg = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=50,
                                  messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text.strip()

if __name__ == "__main__":
    words = ["hello", "name", "help"]
    sentence = to_sentence(words)
    print(sentence)
    engine.say(sentence); engine.runAndWait()
```

## Do NOT do tonight

- Don't try to download the full INCLUDE dataset (57 GB) — category-specific zips only.
- Don't spend time tuning the Phase 4 classifier's accuracy — it's disposable scaffolding.
- Don't write code against `mp.solutions.*` — see gotchas above.
- Don't commit `models/*.task` or `data/` to git — add both to `.gitignore` (model files are large binaries, data grows fast once the team starts recording).

## Definition of done for tonight

- `landmarks.py` produces a correct 225-dim vector on a live webcam frame.
- At least 8 labels have 10+ recorded clips each in `data/raw_landmarks/`.
- `train_baseline.py` runs and prints a test accuracy (doesn't need to be high).
- `demo.py` runs end-to-end: hardcoded word list → Claude sentence → spoken output.
- INCLUDE repo cloned and at least one relevant category zip downloading/downloaded.
- Everything committed to git with a short README noting what's real vs. placeholder.

