# Fortnightly Progress Report — Weeks 1–2

**Project:** Real-Time Continuous Sign Language Recognition with Transformer-Based
Landmark Modeling and LLM-Assisted Contextual Translation
**Course:** Minor Project (BIS586), Semester 5 · Batch A10 · Dept. of ISE, NIE Mysore
**Team:** Dhruvaprasad R · Karan Raj · Chinmaya · Ekanth R
**Internal Guide:** Suhas Bharadwaj R

---

## 1. Objectives for this fortnight

- Stand up the full feature pipeline (webcam → MediaPipe landmarks → classifier).
- Implement the **real Transformer classifier** — the core research model — replacing
  the disposable baseline.
- Prove the training pipeline works end-to-end and validate it on **real Indian Sign
  Language (ISL) data**.

## 2. Summary of progress

The end-to-end recognition pipeline is **built and working on real ISL data**. We
implemented the Transformer classifier, a full training/evaluation harness, and a
video-to-landmark extractor; verified the training loop with a synthetic smoke test;
and then trained the model on **75 real ISL video clips** drawn from the AI4Bharat
**INCLUDE** dataset. The model reaches **60% validation accuracy on a 5-class ISL task
(chance = 20%)** — a 3× improvement over chance from only 15 clips per class, confirming
the pipeline learns genuine signs.

## 3. Work completed

**A. Environment & feature pipeline**
- Configured the project environment and verified the MediaPipe **Tasks API**
  (`HandLandmarker` + `PoseLandmarker`), OpenCV, NumPy and **PyTorch** all work.
- `landmarks.py` produces the **225-dim per-frame feature vector** (2 hands × 21 × 3 +
  pose 33 × 3). Refactored into a shared vectoriser plus an offline **IMAGE-mode**
  extractor so the same features are produced from a live webcam *and* from video files.

**B. Real Transformer model (the core deliverable)**
- `model.py` — `SignTransformer`: linear projection → learnable `[CLS]` token →
  sinusoidal positional encoding → stacked self-attention encoder layers with padding
  masks → classification head. ~0.56 M parameters.
- `dataset.py` — variable-length clip loader with padding-mask collation (handles clips
  of different lengths without distorting the sequence).
- `train_transformer.py` — training/evaluation loop: stratified train/val split, AdamW +
  cosine LR schedule, gradient clipping, best-checkpoint saving.

**C. Pipeline validation (no data required)**
- `tests/test_pipeline.py` — generates class-separable **synthetic** landmark clips and
  trains the model for a few epochs. Result: **100% validation accuracy (chance 25%)**,
  proving the model + data + training code are correctly wired before any real data.

**D. Real ISL data acquisition & training**
- Downloaded the **INCLUDE “Greetings” category** (1.5 GB, part 1 of 2) from Zenodo —
  avoiding the full ~47 GB dataset.
- Built `video_to_landmarks.py` and processed **75 real ISL videos** (5 greeting signs ×
  15 clips) into landmark sequences — 2,599 frames, avg 34.7 frames/clip — using our own
  Tasks-API extractor (INCLUDE’s own keypoint script relies on the removed `mp.solutions`
  API and could not be used).
- Trained `SignTransformer` on this data → best **validation accuracy 60%** (5 classes,
  chance 20%); checkpoint saved to `models/sign_transformer.pt`.

## 4. Results & evidence

| Experiment | Classes | Data | Val. accuracy | Chance | Note |
|-----------|:------:|------|:------:|:------:|------|
| Pipeline smoke test | 4 | Synthetic clips | **100%** | 25% | Verifies training loop |
| Real ISL pilot | 5 | 75 INCLUDE clips (real video) | **60%** | 20% | Learns real signs; small-data overfitting |

Real ISL classes trained: `hello`, `how are you`, `alright`, `good morning`,
`good afternoon`. Extracted clips average ~215/225 non-zero landmark values per frame
(hands + body detected in almost every frame).

## 5. Deliverables produced

- New source: `model.py`, `dataset.py`, `train_transformer.py`, `video_to_landmarks.py`,
  `tests/test_pipeline.py` (~426 lines); `landmarks.py` refactored; `requirements.txt`
  added.
- Trained model checkpoint: `models/sign_transformer.pt`.
- 75-clip real ISL landmark dataset under `data/raw_landmarks/`.
- Presentation deck for the guide review (`presentation/`).

## 6. Challenges & how they were handled

- **MediaPipe `solutions` API removed** in current releases → migrated everything to the
  Tasks API; wrote our own video extractor instead of INCLUDE’s outdated one.
- **ISL vocabulary scarcity** — only 2 of our 8 target words exist in INCLUDE → used
  INCLUDE’s greeting signs to obtain a *real-data* pilot while our own recordings are
  pending.
- **Small-data overfitting** (train 83% vs. val 60%) → expected at 15 clips/class;
  addressed next fortnight with more data and augmentation.

## 7. Honest status — real vs. pending

- **Real & working:** landmark extraction, the Transformer model, the training/eval
  harness, synthetic validation, and real-ISL training with genuine results.
- **Pending:** team’s own multi-signer webcam recordings; larger vocabulary (INCLUDE part
  2 + more categories); data augmentation/regularisation; continuous multi-sign
  segmentation; live end-to-end demo hooking the trained model to the LLM + TTS stage.
- The 60% figure is a **proof-of-concept pilot**, not a final accuracy claim.

## 8. Plan for the next fortnight

1. Record 15–20 clips/sign per team member for the target vocabulary (signer diversity).
2. Expand the ISL dataset (INCLUDE Greetings part 2 + additional categories).
3. Add landmark augmentation + regularisation to close the train/val gap.
4. Begin continuous (sentence-level) recognition and segmentation.
5. Integrate the trained Transformer with the LLM sentence-refinement and TTS stages for a
   live demo.
