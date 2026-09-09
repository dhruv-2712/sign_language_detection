"""Placeholder classifier (Phase 4) -- disposable scaffolding.

Average-pools each clip's landmark sequence over time, trains a small
MLPClassifier, saves with joblib. This exists ONLY to prove the
data -> features -> prediction path works. It is NOT the Transformer
that goes in the final report; do not tune its accuracy.
"""
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data", "raw_landmarks")
OUT_PATH = os.path.join(_REPO_ROOT, "models", "baseline_classifier.joblib")

X, y = [], []
for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    for f in os.listdir(label_dir):
        if not f.endswith(".npy"):
            continue
        seq = np.load(os.path.join(label_dir, f))
        X.append(seq.mean(axis=0))
        y.append(label)

if not X:
    raise SystemExit(f"No clips found under {DATA_DIR}. Record data first with collect_data.py.")

X, y = np.array(X), np.array(y)
print(f"Loaded {len(X)} clips across {len(set(y))} labels.")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500).fit(X_tr, y_tr)
print("test acc:", clf.score(X_te, y_te))
joblib.dump(clf, OUT_PATH)
print(f"Saved {OUT_PATH}")
