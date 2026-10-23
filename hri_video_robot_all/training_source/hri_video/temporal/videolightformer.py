"""VideoLightFormer-style temporal module, re-implemented from the paper description
(Koot & Lu, 2021, "VideoLightFormer: Lightweight Action Recognition using
Transformers", arXiv:2107.00451). Not the authors' code and no released weights.

Idea of the paper: keep the backbone small (2D, TSN-like) and keep the full
spatial-temporal feature structure, applying *small* transformers in a
factorised way instead of pooling features away before the temporal model.

Here:
    tokens (B, T, P, C) -> Linear to small d_model
      -> spatial transformer inside every frame          (B*T, P, d)
      -> causal temporal transformer per spatial location (B*P, T, d)
      -> newest time step, mean over locations -> (B, d)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import TEMPORALS
from .base import FeatureNorm, TemporalModule, causal_mask, pool_time, transformer_stack


@TEMPORALS.register("videolightformer")
class VideoLightFormer(TemporalModule):
    def __init__(self, in_dim: int, num_frames: int, num_tokens: int,
                 d_model: int = 128, spatial_depth: int = 1, temporal_depth: int = 2, heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1, causal: bool = True, pool: str = "last",
                 spatial_grid: int | None = None, feat_norm: str = "ln"):
        super().__init__()
        self.causal, self.pool = causal, pool
        # optional spatial down-sampling of the token grid (cost ~ T * P)
        src = int(round(num_tokens ** 0.5))
        self.src_grid = src if src * src == num_tokens else None
        self.spatial_grid = spatial_grid if (spatial_grid and self.src_grid and spatial_grid < src) else None
        p = self.spatial_grid ** 2 if self.spatial_grid else num_tokens

        self.proj = nn.Sequential(FeatureNorm(in_dim, feat_norm), nn.Linear(in_dim, d_model))
        self.spatial_pos = nn.Parameter(torch.zeros(1, p, d_model))
        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, d_model))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        self.spatial = transformer_stack(d_model, spatial_depth, heads, mlp_ratio, dropout)
        self.temporal = transformer_stack(d_model, temporal_depth, heads, mlp_ratio, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def _downsample(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.spatial_grid:
            return tokens
        b, t, p, c = tokens.shape
        g = self.src_grid
        x = tokens.reshape(b * t, g, g, c).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, self.spatial_grid)
        return x.flatten(2).transpose(1, 2).reshape(b, t, -1, c)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self._downsample(tokens)
        b, t, p, _ = tokens.shape
        x = self.proj(tokens)                                        # (B, T, P, d)
        d = x.shape[-1]
        x = self.spatial(x.reshape(b * t, p, d) + self.spatial_pos)  # within each frame
        x = x.reshape(b, t, p, d).transpose(1, 2).reshape(b * p, t, d)
        x = x + self.temporal_pos[:, -t:]
        mask = causal_mask(t, x.device) if self.causal else None
        x = self.temporal(x, mask=mask, is_causal=self.causal)       # over time, per location
        x = pool_time(x, self.pool).reshape(b, p, d).mean(1)
        return self.norm(x)
