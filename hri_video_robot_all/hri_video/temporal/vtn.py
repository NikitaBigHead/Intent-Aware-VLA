"""Video Transformer Network (VTN), lightweight variant of Kozlov et al., 2019
("Lightweight Network Architecture for Real-Time Action Recognition", arXiv:1905.08711):
a 2D CNN/ViT embeds every frame, the per-frame embeddings get a learnable
temporal position encoding and go through a small transformer over time.

Here: spatial tokens are mean-pooled per frame -> (B, T, C) -> Linear -> causal
transformer -> embedding of the newest frame.
"""
from __future__ import annotations

import torch
from torch import nn

from ..registry import TEMPORALS
from .base import FeatureNorm, TemporalModule, causal_mask, pool_time, transformer_stack


@TEMPORALS.register("vtn")
class VTN(TemporalModule):
    def __init__(self, in_dim: int, num_frames: int, num_tokens: int | None = None,
                 d_model: int = 256, depth: int = 2, heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.1, causal: bool = True, pool: str = "last", feat_norm: str = "ln"):
        super().__init__()
        self.causal, self.pool = causal, pool
        self.proj = nn.Sequential(FeatureNorm(in_dim, feat_norm), nn.Linear(in_dim, d_model))
        self.pos = nn.Parameter(torch.zeros(1, num_frames, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = transformer_stack(d_model, depth, heads, mlp_ratio, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t, p, c = tokens.shape
        x = self.proj(tokens.mean(2)) + self.pos[:, -t:]   # newest frame always gets the last position
        mask = causal_mask(t, x.device) if self.causal else None
        x = self.blocks(x, mask=mask, is_causal=self.causal)
        return self.norm(pool_time(x, self.pool))
