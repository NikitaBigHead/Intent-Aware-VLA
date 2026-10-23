"""Frame encoder contract.

Input : frames  (N, 3, H, W), float in [0, 1]   (N = batch * time, or batch * views)
Output: tokens  (N, P, C)                        (P spatial tokens, C channels)

Pooled per-frame features are tokens.mean(1). Temporal modules that need the
spatial structure (VideoLightFormer) use the tokens; the rest pool them.
Attributes every encoder must expose: out_dim (C), num_tokens (P).
"""
from __future__ import annotations

import torch
from torch import nn


class FrameEncoder(nn.Module):
    out_dim: int
    num_tokens: int
    frozen: bool = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError
