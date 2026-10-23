"""Frame encoders backed by timm: TinyViT, MobileViT, EfficientFormer (+ any timm model).

Weights are downloaded by timm from the Hugging Face hub the first time
`pretrained: true` is used (needs internet on the training machine).
"""
from __future__ import annotations

import math

import timm
import torch
from timm.data import resolve_data_config

from ..registry import ENCODERS
from .base import FrameEncoder


def _to_tokens(f: torch.Tensor) -> torch.Tensor:
    if f.ndim == 4:            # (N, C, H, W) -> (N, H*W, C)
        return f.flatten(2).transpose(1, 2)
    if f.ndim == 3:            # already (N, P, C)
        return f
    raise ValueError(f"Unexpected feature shape {tuple(f.shape)}")


class TimmFrameEncoder(FrameEncoder):
    default_variant: str | None = None

    def __init__(self, variant: str | None = None, pretrained: bool = True, freeze: bool = True,
                 img_size: int = 224, drop_path_rate: float = 0.0):
        super().__init__()
        variant = variant or self.default_variant
        if variant is None:
            raise ValueError("timm encoder needs `variant`")
        self.variant = variant
        kwargs = dict(pretrained=pretrained, num_classes=0)
        if drop_path_rate:
            kwargs["drop_path_rate"] = drop_path_rate
        self.model = timm.create_model(variant, **kwargs)

        dcfg = resolve_data_config({}, model=self.model)
        self.register_buffer("mean", torch.tensor(dcfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(dcfg["std"]).view(1, 3, 1, 1), persistent=False)

        self.model.eval()
        with torch.no_grad():
            tok = _to_tokens(self.model.forward_features(torch.zeros(1, 3, img_size, img_size)))
        self.num_tokens, self.out_dim = int(tok.shape[1]), int(tok.shape[2])
        g = int(math.isqrt(self.num_tokens))
        self.grid = (g, g) if g * g == self.num_tokens else None

        self.frozen = bool(freeze)
        if self.frozen:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:            # keep BN / dropout in inference mode when frozen
            self.model.eval()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = (frames - self.mean) / self.std
        if self.frozen:
            with torch.no_grad():
                f = self.model.forward_features(x)
        else:
            f = self.model.forward_features(x)
        return _to_tokens(f)


@ENCODERS.register("tinyvit")
class TinyViTEncoder(TimmFrameEncoder):
    default_variant = "tiny_vit_5m_224.dist_in22k_ft_in1k"


@ENCODERS.register("mobilevit")
class MobileViTEncoder(TimmFrameEncoder):
    default_variant = "mobilevit_s.cvnets_in1k"


@ENCODERS.register("efficientformer")
class EfficientFormerEncoder(TimmFrameEncoder):
    # NB: EfficientFormer(v1) has fixed attention biases -> only 224x224 input.
    default_variant = "efficientformer_l1.snap_dist_in1k"


@ENCODERS.register("timm")
class GenericTimmEncoder(TimmFrameEncoder):
    """Any timm backbone: model.encoder.name=timm model.encoder.variant=convnext_atto"""
