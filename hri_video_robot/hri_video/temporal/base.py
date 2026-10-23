"""Temporal module contract.

Input : tokens (B, T, P, C) from the frame encoder (T frames, P spatial tokens)
Output: embedding (B, D) — one temporal embedding per clip.

All modules are causal by default: with `pool: last` the embedding is the output
at the newest frame and only attends to past frames, so the offline clip result
equals what the live streaming loop produces.
"""
from __future__ import annotations

import torch
from torch import nn


class TemporalModule(nn.Module):
    out_dim: int

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


def transformer_stack(dim: int, depth: int, heads: int, mlp_ratio: float, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio), dropout=dropout,
        activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


def causal_mask(t: int, device) -> torch.Tensor:
    # True = not allowed to attend (future positions)
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)


def pool_time(x: torch.Tensor, pool: str) -> torch.Tensor:
    """x (B, T, D) -> (B, D)."""
    if pool == "last":
        return x[:, -1]
    if pool == "mean":
        return x.mean(1)
    raise ValueError(f"pool must be 'last' or 'mean', got {pool}")


class FeatureNorm(nn.Module):
    """Normalise frame features before projection.

    'ln' (default): per-token LayerNorm.
    'bn': BatchNorm over channels with running dataset statistics (experimental option; removes the
    dataset-mean feature, e.g. a constant background of a fixed camera).
    """

    def __init__(self, dim: int, kind: str = "ln"):
        super().__init__()
        self.kind = kind
        self.norm = nn.BatchNorm1d(dim) if kind == "bn" else nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (..., C)
        if self.kind != "bn":
            return self.norm(x)
        shp = x.shape
        return self.norm(x.reshape(-1, shp[-1])).reshape(shp)
