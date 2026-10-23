"""Multi-view video model:  frames -> frame encoder -> temporal module -> view fusion -> heads.

    batch["frames"][view]: (B, T, 3, H, W) float in [0, 1]
    batch["view_mask"]   : (B, V) bool, False = this camera is missing for the sample
    output["embedding"]  : (B, E)  single temporal embedding for any classification head
    output["logits"][h]  : (B, n_h)
"""
from __future__ import annotations

import torch
from torch import nn

from . import encoders as _enc  # noqa: F401  (populate registries)
from . import temporal as _tmp  # noqa: F401
from .config import Cfg, resolve_encoder_cfg, resolve_temporal_cfg
from .registry import ENCODERS, FUSIONS, TEMPORALS


# ----------------------------------------------------------------------------- fusion
@FUSIONS.register("mean")
class MeanFusion(nn.Module):
    def __init__(self, in_dim: int, num_views: int, dim: int, **_):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.out_dim = dim

    def forward(self, x, mask):           # x (B, V, D), mask (B, V)
        m = mask.unsqueeze(-1).to(x.dtype)
        return self.proj((x * m).sum(1) / m.sum(1).clamp_min(1.0))


@FUSIONS.register("concat")
class ConcatFusion(nn.Module):
    """Concatenate view embeddings (zeros for missing views) + missing-view flags -> MLP."""

    def __init__(self, in_dim: int, num_views: int, dim: int, dropout: float = 0.1, **_):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(num_views * (in_dim + 1), dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim, dim))
        self.out_dim = dim

    def forward(self, x, mask):
        m = mask.unsqueeze(-1).to(x.dtype)
        return self.mlp(torch.cat([x * m, m], dim=-1).flatten(1))


@FUSIONS.register("attention")
class AttentionFusion(nn.Module):
    """A learned query attends over the available camera embeddings (masked)."""

    def __init__(self, in_dim: int, num_views: int, dim: int, heads: int = 4, dropout: float = 0.1, **_):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.view_emb = nn.Parameter(torch.zeros(1, num_views, dim))
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.view_emb, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, x, mask):
        kv = self.proj(x) + self.view_emb
        none = ~mask.any(1)                          # samples with no camera at all
        safe = mask.clone()
        safe[none, 0] = True                         # avoid all-masked rows (NaN)
        q = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, kv, kv, key_padding_mask=~safe)
        out = self.norm(out[:, 0])
        return out.masked_fill(none.unsqueeze(-1), 0.0)


# ----------------------------------------------------------------------------- model
class VideoModel(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        mc, dc = cfg.model, cfg.data
        self.views = list(dc.views)
        self.num_frames = int(dc.num_frames)

        enc_kw = resolve_encoder_cfg(cfg)
        shared_enc = mc.encoder.get("share_across_views", True)
        if shared_enc:
            enc = ENCODERS.build(mc.encoder.name, **enc_kw)
            self.encoders = nn.ModuleDict({"shared": enc})
        else:
            self.encoders = nn.ModuleDict({v: ENCODERS.build(mc.encoder.name, **enc_kw) for v in self.views})
        ref = next(iter(self.encoders.values()))

        tmp_kw = dict(resolve_temporal_cfg(cfg), in_dim=ref.out_dim, num_frames=self.num_frames,
                      num_tokens=ref.num_tokens)
        if mc.temporal.get("share_across_views", False):
            self.temporals = nn.ModuleDict({"shared": TEMPORALS.build(mc.temporal.name, **tmp_kw)})
        else:
            self.temporals = nn.ModuleDict({v: TEMPORALS.build(mc.temporal.name, **tmp_kw) for v in self.views})
        t_dim = next(iter(self.temporals.values())).out_dim

        fc = dict(mc.fusion)
        fusion_name = fc.pop("name", "attention")
        self.fusion = FUSIONS.build(fusion_name, in_dim=t_dim, num_views=len(self.views), **fc)
        self.embed_dim = self.fusion.out_dim

        self.dropout = nn.Dropout(mc.get("dropout", 0.0))
        self.heads = nn.ModuleDict({k: nn.Linear(self.embed_dim, int(n)) for k, n in mc.heads.items()})

    # -- building blocks (also used by the streaming predictor / evaluator) --
    def encoder_for(self, view):
        return self.encoders["shared"] if "shared" in self.encoders else self.encoders[view]

    def temporal_for(self, view):
        return self.temporals["shared"] if "shared" in self.temporals else self.temporals[view]

    def encode_frames(self, view: str, frames: torch.Tensor) -> torch.Tensor:
        """frames (B, T, 3, H, W) -> tokens (B, T, P, C)"""
        b, t = frames.shape[:2]
        tok = self.encoder_for(view)(frames.flatten(0, 1))
        return tok.reshape(b, t, *tok.shape[1:])

    def head_from_tokens(self, tokens: dict, view_mask: torch.Tensor) -> dict:
        """tokens[view] (B, T, P, C) -> embedding + logits"""
        embs = torch.stack([self.temporal_for(v)(tokens[v]) for v in self.views], dim=1)  # (B, V, D)
        z = self.fusion(embs, view_mask)
        zd = self.dropout(z)
        return {"embedding": z, "logits": {k: h(zd) for k, h in self.heads.items()}}

    def forward(self, batch: dict) -> dict:
        frames = batch["frames"]
        tokens = {v: self.encode_frames(v, frames[v]) for v in self.views}
        mask = batch.get("view_mask")
        if mask is None:
            b = next(iter(frames.values())).shape[0]
            mask = torch.ones(b, len(self.views), dtype=torch.bool, device=tokens[self.views[0]].device)
        return self.head_from_tokens(tokens, mask)


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def param_report(model: VideoModel) -> dict:
    return {
        "params_encoder_M": count_params(model.encoders) / 1e6,
        "params_temporal_M": count_params(model.temporals) / 1e6,
        "params_fusion_heads_M": (count_params(model.fusion) + count_params(model.heads)) / 1e6,
        "params_total_M": count_params(model) / 1e6,
        "params_trainable_M": count_params(model, trainable_only=True) / 1e6,
    }
