"""Continuous, class-specific intent decisions for the RGB video model."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


class IntentDecisionFilter:
    """Turn per-step argmax probabilities into the shared intent protocol.

    A class is accepted only after its own confidence threshold and number of
    consecutive valid decisions have both been satisfied. Until then the
    output is ``unknown``; this lets the VLA unknown-grace preserve the current
    prompt while a possible transition is being confirmed.
    """

    def __init__(
        self,
        classes: Sequence[str],
        base_threshold: float,
        base_stable_frames: int,
        class_thresholds: Mapping[str, float] | None = None,
        class_stable_frames: Mapping[str, int] | None = None,
    ):
        self.classes = list(classes)
        if len(self.classes) != len(set(self.classes)) or not self.classes:
            raise ValueError("classes must be non-empty and unique")

        self.class_thresholds = {label: float(base_threshold) for label in self.classes}
        self.class_stable_frames = {label: int(base_stable_frames) for label in self.classes}
        self._apply_overrides(self.class_thresholds, class_thresholds, "confidence threshold")
        self._apply_overrides(self.class_stable_frames, class_stable_frames, "stable frames")

        if any(not 0.0 <= value <= 1.0 for value in self.class_thresholds.values()):
            raise ValueError("Class confidence thresholds must be in [0, 1]")
        if any(value < 1 for value in self.class_stable_frames.values()):
            raise ValueError("Class stable frames must be positive integers")
        self.reset()

    def _apply_overrides(self, target, overrides, description):
        if not overrides:
            return
        unknown = set(overrides) - set(self.classes)
        if unknown:
            raise ValueError(f"Unknown class {description}s: {sorted(unknown)}")
        target.update(overrides)

    def reset(self) -> None:
        self._candidate: str | None = None
        self._streak = 0

    def apply(self, probabilities: Sequence[float]) -> dict:
        probs = np.asarray(probabilities, dtype=float)
        if (
            probs.shape != (len(self.classes),)
            or not np.isfinite(probs).all()
            or not np.isclose(probs.sum(), 1.0, atol=1e-5)
        ):
            raise ValueError("Invalid action probabilities")

        index = int(probs.argmax())
        raw_label = self.classes[index]
        confidence = float(probs[index])
        threshold = float(self.class_thresholds[raw_label])

        if confidence < threshold:
            self.reset()
            label, label_index, reason = "unknown", -1, "low_action_confidence"
        else:
            if raw_label == self._candidate:
                self._streak += 1
            else:
                self._candidate, self._streak = raw_label, 1
            if self._streak < int(self.class_stable_frames[raw_label]):
                label, label_index, reason = "unknown", -1, "unstable_prediction"
            else:
                label, label_index, reason = raw_label, index, ""

        return {
            "label": label,
            "label_index": label_index,
            "confidence": confidence,
            "probabilities": dict(zip(self.classes, map(float, probs))),
            "reason": reason,
            "raw_label": raw_label,
            "class_threshold": threshold,
            "stable_count": int(self._streak),
            "stable_required": int(self.class_stable_frames[raw_label]),
        }
