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
| `demo.py` — glosses -> Groq LLM -> pyttsx3 TTS | **Real** wiring, hardcoded input for now |
| `model.py` / `train_transformer.py` — Transformer classifier | **Real** — the actual deliverable; trains on landmark sequences |
| `dataset.py` — variable-length clip loader + padding masks | **Real** |
| `video_to_landmarks.py` — turn ISL videos into `.npy` clips | **Real** — feeds INCLUDE data through our extractor |
| `preprocess.py` — shoulder-centred/scaled normalisation + augmentation | **Real** — used in training and live inference |
| `segmenter.py` — splits a continuous stream into per-sign clips (hand presence) | **Real** |
| `evaluate.py` — per-class accuracy + confusion matrix | **Real** |
| `live_demo.py` — webcam -> segment -> Transformer -> Groq LLM -> TTS | **Real** — the end-to-end demo |
| `tests/test_pipeline.py` — synthetic end-to-end smoke test | **Real** — verifies training with no data/webcam |

The Phase 4 MLP classifier only proves the data → features → prediction path
works. Do not tune its accuracy; it is thrown away. The Transformer in
`model.py` is the real model — a self-attention encoder over the frame sequence
with a learnable `[CLS]` token and padding masks.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate            # Windows  (use: source venv/bin/activate  on Linux/macOS)
pip install mediapipe opencv-python numpy scikit-learn joblib pyttsx3 groq
python src/setup_models.py       # downloads the .task model files into models/
$env:GROQ_API_KEY = "..."   # needed for demo.py only
```

## Feature vector (225-dim per frame)

- 2 hands × 21 landmarks × 3 (xyz) = 126
- pose 33 × 3 = 99
- Missing hand/pose in a frame → zeros (keeps sequence length consistent).

## Usage

```powershell
python src/landmarks.py           # webcam verification loop — watch nonzero count rise with a hand in frame
python src/collect_data.py        # SPACE=record toggle  N=next label  Q=quit
python src/train_baseline.py      # disposable MLP sanity check -> models/baseline_classifier.joblib
python src/train_transformer.py --data_dir data/raw_landmarks --epochs 40   # the real model -> models/sign_transformer.pt
python src/evaluate.py            # per-class accuracy + confusion matrix on the val split
python src/live_demo.py           # LIVE: sign at the webcam; SPACE = speak sentence, C = clear, Q = quit
python src/demo.py                # glosses -> LLM sentence -> spoken output
python tests/test_pipeline.py     # synthetic smoke test (no webcam/data needed)
```

Turn real ISL videos into training clips (same `.npy` format as the webcam recorder):

```powershell
python src/video_to_landmarks.py --src <Greetings-dir> --recursive \
    --map "48. Hello=hello,55. Thank you=thankyou,56. Pleased=please" --limit 15 --stride 2
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

## Current results

SignTransformer on 75 INCLUDE ISL clips (5 classes), with normalisation +
augmentation + label smoothing: **93% val accuracy on 15 held-out clips**
(was 60% without; chance 20%). The validation set is tiny and all clips come from
INCLUDE signers, so this is a pilot figure, not a generalisation claim.
Retrain after changing preprocessing: `python src/train_transformer.py --epochs 80`.

## Still needs people/data (cannot be automated)

- Live webcam use has not been exercised on a real camera in this repo yet.
- Vocabulary is the 5 INCLUDE greetings. Add more by recording (`collect_data.py`) or
  converting more INCLUDE categories (`video_to_landmarks.py`), then retrain.
- Evaluate on signers not in the training set before quoting accuracy.

## Pretrained pipeline (submission 2)

The final pipeline uses the **AI4Bharat INCLUDE pretrained Transformer** (263 Indian Sign
Language words, upstream val accuracy 83.5%) instead of our small from-scratch model:

```
camera -> MediaPipe Tasks (hands+pose, VIDEO-mode tracking) -> hand-raise segmentation
  -> 225->134 feature bridge -> pretrained INCLUDE Transformer (top-3 words)
  -> Groq LLM (words + runner-up guesses -> fluent sentence) -> pyttsx3 TTS
```

```powershell
python src/download_pretrained.py      # one-time, ~205 MB checkpoint -> models/pretrained/
python src/sign2speech.py              # LIVE webcam (SPACE = speak sentence, C = clear, Q = quit)
python src/sign2speech.py --video a.MOV b.MOV   # same pipeline on video files
python src/fetch_include_test.py       # download ONLY held-out INCLUDE test videos (HTTP range)
python src/eval_include.py --root data/include_test   # evaluate the whole front half
```

Key engineering findings (all verified against upstream's legacy `mp.solutions` pipeline,
which scores 38/41 = 92.7% on the Greetings test clips):

- **Feature bridge** (`include_features.py`): the model expects 134 values/frame (25 upper-body
  pose + 2 hands, x,y in 1920x1080 pixels). The Tasks-API pose's first 25 landmarks share the
  legacy indices, so it is a re-indexing, not retraining.
- **Tracking matters**: stateless IMAGE-mode detection dropped the signing hand (17/41);
  VIDEO-mode tracking + hand confidence 0.3 + raised-hand-first slot ordering gives 34/41.
- **Segmentation context**: clips get pre/post-roll frames because the model was trained on
  whole videos, not trimmed hand-up runs.
- The pose model (lite/full/heavy) made no difference; `SIGN_POSE_MODEL` selects it.

### Results (held-out INCLUDE *test-split* videos the pretrained model never trained on)

| Set | Videos | Classes | Top-1 | Top-3 |
|-----|:-----:|:------:|:-----:|:-----:|
| Greetings | 41 | 9 | 82.9% | 92.7% |
| Greetings + Pronouns + Days/Time | 91 | 35 | 78.0% | 90.1% |
| Upstream legacy-MediaPipe reference (Greetings) | 41 | 9 | 92.7% | - |

Limits: the LLM step needs `GROQ_API_KEY` (without it the raw words are spoken; the
prompt logic is unit-tested with a fake client, not yet exercised against the live API);
the live webcam loop has not been run on a physical camera in this repo; the vocabulary is
INCLUDE's 263 single words (no grammar, no name/water/help); webcam signers differ from the
INCLUDE recordings, so expect lower accuracy until tested on the team.
