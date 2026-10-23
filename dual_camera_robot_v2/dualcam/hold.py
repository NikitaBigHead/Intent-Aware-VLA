"""Optional, explicit temporal holds applied after the model decision rules."""

from __future__ import annotations

from typing import Any, Mapping


class HugUnknownHold:
    """Keep an accepted hug across a short run of ``unknown`` decisions.

    The timer starts at the last real ``hug`` decision and is deliberately not
    extended by synthetic held decisions. Any other known class cancels it.
    """

    def __init__(self, enabled: bool = False, duration_s: float = 3.0):
        if duration_s <= 0:
            raise ValueError("hug unknown hold duration must be positive")
        self.enabled = bool(enabled)
        self.duration_s = float(duration_s)
        self._last_hug_t: float | None = None

    def reset(self) -> None:
        """Forget a previous hug after a discontinuity in the input stream."""
        self._last_hug_t = None

    def apply(self, decision: Mapping[str, Any], timestamp: float) -> dict[str, Any]:
        result = dict(decision)
        if not self.enabled:
            return result

        timestamp = float(timestamp)
        label = str(result["label"])
        if label == "hug":
            self._last_hug_t = timestamp
            return result

        if label != "unknown":
            self._last_hug_t = None
            return result

        if self._last_hug_t is None:
            return result

        age = timestamp - self._last_hug_t
        if age < 0 or age > self.duration_s:
            self._last_hug_t = None
            return result

        raw_reason = str(result.get("reason", ""))
        result.update(
            label="hug",
            label_index=0,
            raw_label="unknown",
            raw_reason=raw_reason,
            held_from_unknown=True,
            hug_hold_age_s=float(age),
            hug_hold_remaining_s=float(max(0.0, self.duration_s - age)),
            reason=f"held_hug_after_unknown:{raw_reason or 'unknown'}",
        )
        return result
