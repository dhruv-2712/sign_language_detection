"""Transformer classifier for sign-language landmark sequences.

This is the real research model referenced in CLAUDE.md (it replaces the
throwaway average-pool + MLP baseline in train_baseline.py). It consumes a
variable-length sequence of 225-dim landmark vectors (one per frame) and
predicts a single sign-gloss label per clip.

Architecture: linear projection -> prepend a learnable [CLS] token ->
sinusoidal positional encoding -> a stack of TransformerEncoder layers with a
key-padding mask -> classify from the [CLS] representation.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SignTransformer(nn.Module):
    def __init__(self, input_dim=225, num_classes=8, d_model=128, nhead=8,
                 num_layers=4, dim_ff=256, dropout=0.1, max_len=1024):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos = PositionalEncoding(d_model, max_len + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x, pad_mask):
        """x: (B, T, input_dim). pad_mask: (B, T) bool, True where PADDING."""
        b = x.size(0)
        h = self.input_proj(x)
        cls = self.cls_token.expand(b, -1, -1)
        h = torch.cat([cls, h], dim=1)                       # (B, T+1, d_model)
        h = self.pos(h)
        cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=x.device)
        mask = torch.cat([cls_mask, pad_mask], dim=1)        # CLS never masked
        h = self.encoder(h, src_key_padding_mask=mask)
        return self.head(self.norm(h[:, 0]))                 # classify from CLS
