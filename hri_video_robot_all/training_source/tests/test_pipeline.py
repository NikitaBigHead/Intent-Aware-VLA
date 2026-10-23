"""Checks: every combo builds; streaming (live) == offline clip result; missing cameras are safe.

    python -m pytest -q tests      (or: python tests/test_pipeline.py)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hri_video.config import load_config  # noqa: E402
from hri_video.data import clip_indices  # noqa: E402
from hri_video.model import VideoModel  # noqa: E402
from hri_video.registry import ENCODERS, TEMPORALS  # noqa: E402
from hri_video.streaming import StreamingPredictor  # noqa: E402

CFG = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def make(enc, tmp, T=4, stride=2):
    cfg = load_config(CFG, [f"model.encoder.name={enc}", f"model.temporal.name={tmp}", "model.encoder.pretrained=false",
                            f"data.num_frames={T}", f"data.frame_stride={stride}", "eval.consecutive=1"])
    torch.manual_seed(0)
    return cfg, VideoModel(cfg).eval()


def test_registries():
    assert {"tinyvit", "mobilevit", "efficientformer"} <= set(ENCODERS.keys())
    assert {"vtn", "videolightformer"} <= set(TEMPORALS.keys())


@torch.no_grad()
def test_streaming_matches_offline():
    dev = torch.device("cpu")
    for enc, tmp in itertools.product(["tinyvit", "mobilevit", "efficientformer"], ["vtn", "videolightformer"]):
        cfg, model = make(enc, tmp)
        L, S = 9, int(cfg.data.img_size)
        rng = np.random.default_rng(0)
        frames = {v: rng.integers(0, 255, (L, S, S, 3), dtype=np.uint8) for v in model.views}
        # offline: encode every frame once, causal windows with left replicate padding
        tok = {v: model.encoder_for(v)(torch.from_numpy(frames[v]).permute(0, 3, 1, 2).float() / 255) for v in model.views}
        mask = torch.ones(1, len(model.views), dtype=torch.bool)
        off = []
        for e in range(L):
            idx = clip_indices(e, model.num_frames, int(cfg.data.frame_stride))
            o = model.head_from_tokens({v: tok[v][idx][None] for v in model.views}, mask)
            off.append(o["logits"]["action"].softmax(-1)[0].numpy())
        # live
        sp = StreamingPredictor(model, cfg, dev, amp=False)
        live = [sp.step({v: frames[v][e] for v in model.views}, e / cfg.data.fps)["probs"] for e in range(L)]
        err = np.abs(np.stack(off) - np.stack(live)).max()
        assert err < 1e-4, (enc, tmp, err)
        print(f"ok {enc:15s} {tmp:17s} max|offline-live|={err:.2e}")


@torch.no_grad()
def test_missing_view_is_finite():
    cfg, model = make("tinyvit", "vtn")
    b = {"frames": {v: torch.rand(3, model.num_frames, 3, 224, 224) for v in model.views},
         "view_mask": torch.tensor([[True, False], [False, True], [False, False]])}
    o = model(b)
    assert torch.isfinite(o["embedding"]).all()


if __name__ == "__main__":
    test_registries(); test_missing_view_is_finite(); test_streaming_matches_offline(); print("all ok")
