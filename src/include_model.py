"""AI4Bharat INCLUDE pretrained Transformer (263 Indian Sign Language words).

Re-implements the inference-time architecture from github.com/AI4Bharat/INCLUDE
(models/transformer.py, "large" config) so we don't depend on that repo's training
code. Weights: `include_no_cnn_transformer_large.pth` from the INCLUDE release,
downloaded to models/pretrained/include_large.pth (see README).

Input per frame (134 values): upper-body pose 25 x (x,y) + hand1 21 x (x,y) +
hand2 21 x (x,y), in pixels of a 1920x1080 frame. See include_features.py.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import transformers

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO_ROOT, "models", "pretrained", "include_large.pth")
LABELS = os.path.join(REPO_ROOT, "labels", "include_labels.json")
MAX_FRAMES = 169            # the length the checkpoint was evaluated with upstream


class _PositionEmbedding(nn.Module):
    def __init__(self, hidden, max_pos=256, eps=1e-12, p=0.1):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_pos, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)
        self.dropout = nn.Dropout(p)
        self.register_buffer("position_ids", torch.arange(max_pos).expand((1, -1)))

    def forward(self, x):
        x = x + self.position_embeddings(self.position_ids[:, : x.size(1)])
        return self.dropout(self.LayerNorm(x))


class INCLUDETransformer(nn.Module):
    def __init__(self, n_classes=263, input_size=134, hidden=512, heads=8, layers=4):
        super().__init__()
        cfg = transformers.BertConfig(hidden_size=hidden, num_attention_heads=heads,
                                      num_hidden_layers=layers, max_position_embeddings=256,
                                      attn_implementation="eager")
        self.l1 = nn.Linear(input_size, hidden)
        self.embedding = _PositionEmbedding(hidden)
        self.layers = nn.ModuleList([transformers.BertLayer(cfg) for _ in range(layers)])
        self.l2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        """x: (B, T, 134) -> logits (B, 263)."""
        x = self.embedding(self.l1(x))
        for layer in self.layers:
            out = layer(x)
            x = out[0] if isinstance(out, (tuple, list)) else out
        return self.l2(torch.max(x, dim=1).values)


def load_pretrained(ckpt_path=CKPT, device="cpu"):
    """Load weights with torch's restricted unpickler (weights_only=True).

    The upstream checkpoint also stores a numpy float (its best val score), so only
    that one numpy scalar type is allow-listed -- no arbitrary code is unpickled.
    """
    from numpy._core import multiarray
    safe = [(multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
            type(np.dtype("float64"))]
    with torch.serialization.safe_globals(safe):
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = INCLUDETransformer()
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    missing = [k for k in missing if "position_ids" not in k]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    with open(LABELS) as f:
        label_to_idx = json.load(f)
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    return model.to(device).eval(), idx_to_label


def checkpoint_labels(ckpt_path):
    """Words a fine-tuned checkpoint was trained on: stored in it, else the recorded-clip folders."""
    import glob
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(ck, dict) and ck.get("labels"):
            return list(ck["labels"])
    except Exception:
        pass
    root = os.path.join(REPO_ROOT, "data", "webcam_clips")
    return sorted({os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(root, "*", "*.npz"))})


def pad_features(feats):
    t = feats.shape[0]
    if t > MAX_FRAMES:
        feats = feats[np.linspace(0, t - 1, MAX_FRAMES).astype(int)]
    padded = np.zeros((MAX_FRAMES, feats.shape[1]), dtype=np.float32)
    padded[: len(feats)] = feats
    return padded


@torch.no_grad()
def probabilities(model, feats, device="cpu"):
    """feats: (T,134) -> softmax probabilities over all 263 words, as a numpy array."""
    x = torch.from_numpy(pad_features(feats)).unsqueeze(0).to(device)
    return torch.softmax(model(x), dim=-1)[0].cpu().numpy()


@torch.no_grad()
def predict(model, feats, k=3, device="cpu"):
    """feats: (T,134) from include_features.to_include_features -> [(label_idx, prob)] top-k."""
    t = feats.shape[0]
    if t > MAX_FRAMES:
        feats = feats[np.linspace(0, t - 1, MAX_FRAMES).astype(int)]
    padded = np.zeros((MAX_FRAMES, feats.shape[1]), dtype=np.float32)
    padded[: len(feats)] = feats
    logits = model(torch.from_numpy(padded).unsqueeze(0).to(device))
    probs = torch.softmax(logits, dim=-1)[0]
    p, i = probs.topk(k)
    return list(zip(i.tolist(), p.tolist()))
