"""Tiny registries so new encoders / temporal modules plug in with one decorator.

    @ENCODERS.register("my_encoder")
    class MyEncoder(FrameEncoder): ...

and then `model.encoder.name: my_encoder` in the config.
"""
from __future__ import annotations


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, key: str):
        def deco(cls):
            if key in self._items:
                raise KeyError(f"{self.kind} '{key}' is already registered")
            self._items[key] = cls
            return cls
        return deco

    def get(self, key: str):
        if key not in self._items:
            raise KeyError(f"Unknown {self.kind} '{key}'. Available: {self.keys()}")
        return self._items[key]

    def build(self, key: str, **kwargs):
        return self.get(key)(**kwargs)

    def keys(self):
        return sorted(self._items)


ENCODERS = Registry("frame encoder")
TEMPORALS = Registry("temporal module")
FUSIONS = Registry("view fusion")
