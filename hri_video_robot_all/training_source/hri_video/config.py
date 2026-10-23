"""YAML config with dotted CLI overrides:  --set model.encoder.name=mobilevit train.lr=1e-4"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


class Cfg(dict):
    """dict with attribute access (cfg.model.encoder.name)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    @staticmethod
    def wrap(obj):
        if isinstance(obj, dict) and not isinstance(obj, Cfg):
            return Cfg({k: Cfg.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Cfg.wrap(v) for v in obj]
        return obj

    def to_dict(self):
        def un(o):
            if isinstance(o, dict):
                return {k: un(v) for k, v in o.items()}
            if isinstance(o, list):
                return [un(v) for v in o]
            return o
        return un(self)


def _deep_update(base: dict, upd: dict):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_config(path: str | Path, overrides: list[str] | None = None) -> Cfg:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    # optional inheritance: `_base_: base.yaml`
    base = cfg.pop("_base_", None)
    if base:
        cfg = _deep_update(load_config(path.parent / base).to_dict(), cfg)
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        if not _:
            raise ValueError(f"Override must look like a.b.c=value, got '{ov}'")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return Cfg.wrap(cfg)


def save_config(cfg: Cfg, path: str | Path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)


def resolve_encoder_cfg(cfg: Cfg) -> dict:
    """Merge model.encoder common keys with the preset of the selected encoder."""
    enc = cfg.model.encoder
    params = dict(enc.get("presets", {}).get(enc.name, {}) or {})
    for k in ("pretrained", "freeze", "variant"):
        if enc.get(k) is not None:
            params[k] = enc[k]
    params["img_size"] = cfg.data.img_size
    return params


def resolve_temporal_cfg(cfg: Cfg) -> dict:
    tmp = cfg.model.temporal
    params = dict(tmp.get("presets", {}).get(tmp.name, {}) or {})
    params.update(tmp.get("params", {}) or {})  # explicit params win over preset
    return params
