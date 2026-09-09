# Sign Language Recognition — Prototype Scaffolding

NIE Mysore ISE minor project (BIS586, Sem 5, Batch A10).
**Team:** Dhruvaprasad R, Karan Raj, Chinmaya, Ekanth R. **Guide:** Suhas Bharadwaj R.

**Title:** Real-Time Continuous Sign Language Recognition with Transformer-Based
Landmark Modeling and LLM-Assisted Contextual Translation.

## Pipeline

```
webcam -> MediaPipe hand+pose landmarks -> classifier -> LLM sentence -> TTS
```

Everything except the classifier is a pretrained / inference-only component.

## What's real vs. placeholder

| Component | Status |
|-----------|--------|
| `landmarks.py` — MediaPipe Tasks API extraction (225-dim/frame) | **Real**, used everywhere |
| `collect_data.py` — webcam clip recorder | **Real** tooling |
| `train_baseline.py` — average-pool + MLPClassifier | **Placeholder** — disposable sanity check, NOT the final model |
| `demo.py` — glosses -> Claude -> pyttsx3 TTS | **Real** wiring, hardcoded input for now |
| Transformer classifier | **Not built yet** — the real deliverable, next sessions |

The Phase 4 MLP classifier only proves the data → features → prediction path
works. Do not tune its accuracy; it is thrown away.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate            # Windows  (use: source venv/bin/activate  on Linux/macOS)
pip install mediapipe opencv-python numpy scikit-learn joblib pyttsx3 anthropic
python src/setup_models.py       # downloads the .task model files into models/
$env:ANTHROPIC_API_KEY = "..."   # needed for demo.py only
```

## Feature vector (225-dim per frame)

- 2 hands × 21 landmarks × 3 (xyz) = 126
- pose 33 × 3 = 99
- Missing hand/pose in a frame → zeros (keeps sequence length consistent).

## Usage

```powershell
python src/landmarks.py          # webcam verification loop — watch nonzero count rise with a hand in frame
python src/collect_data.py       # SPACE=record toggle  N=next label  Q=quit
python src/train_baseline.py     # trains + prints test accuracy, saves models/baseline_classifier.joblib
python src/demo.py               # glosses -> Claude sentence -> spoken output
```

Vocabulary: `hello, thankyou, yes, no, please, water, help, name`.
Target 15–20 clips per label per signer; each teammate records on their own machine.

## MediaPipe note

`mp.solutions.*` (holistic/hands/pose) is **removed** in current MediaPipe.
This project uses the **Tasks API** (`HandLandmarker` + `PoseLandmarker` in VIDEO
mode), combined by us into one feature vector. There is no unified HolisticLandmarker task yet.

## Datasets

- `INCLUDE/` — AI4Bharat INCLUDE repo (code) is cloned but **gitignored**.
- Full INCLUDE dataset (~57 GB on Zenodo) is **not** downloaded — only category
  zips covering the vocabulary. See `docs`/session notes for which categories.
- Consider AI4Bharat/OpenHands for a pretrained ISL checkpoint to fine-tune.

## Not committed to git

`models/*.task`, `models/*.joblib`, `data/`, `venv/`, `INCLUDE/` — see `.gitignore`.
