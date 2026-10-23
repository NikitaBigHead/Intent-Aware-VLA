"""Versioned ZMQ protocol shared with the dual-camera intent predictors."""

from __future__ import annotations

import time
from typing import Any, Mapping


SCHEMA_VERSION = 1


def build_intent_event(
    decision: Mapping[str, Any],
    *,
    seq: int,
    t_person: float,
    t_hand: float,
    skew_ms: float,
    produced_time: float | None = None,
) -> dict[str, Any]:
    label = str(decision["label"])
    event = {
        "type": "human_intent",
        "version": SCHEMA_VERSION,
        "seq": int(seq),
        "capture_time": float(t_person),
        "hand_capture_time": float(t_hand),
        "produced_time": float(time.time() if produced_time is None else produced_time),
        "label": label,
        "accepted": label != "unknown",
        "confidence": float(decision.get("confidence", 0.0)),
        "probabilities": {
            str(name): float(value)
            for name, value in decision.get("probabilities", {}).items()
        },
        "quality": {
            str(name): (bool(value) if isinstance(value, bool) else float(value))
            for name, value in decision.get("quality", {}).items()
        },
        "reason": str(decision.get("reason", "")),
        "skew_ms": float(skew_ms),
        "hand_attention_last": float(decision.get("hand_attention_last", 0.0)),
    }
    for key in (
        "raw_label",
        "raw_reason",
        "held_from_unknown",
        "hug_hold_age_s",
        "hug_hold_remaining_s",
        "class_threshold",
        "stable_count",
        "stable_required",
        "model_label",
    ):
        if key in decision:
            event[key] = decision[key]
    return event


class IntentPublisher:
    """Best-effort latest-intent publisher that never blocks inference."""

    def __init__(self, endpoint: str):
        import zmq

        self._zmq = zmq
        self.endpoint = endpoint
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.bind(endpoint)
        self._seq = 0

    def publish(self, decision, *, t_person: float, t_hand: float, skew_ms: float):
        event = build_intent_event(
            decision,
            seq=self._seq,
            t_person=t_person,
            t_hand=t_hand,
            skew_ms=skew_ms,
        )
        self._seq += 1
        try:
            self._socket.send_json(event, flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            pass
        return event

    def close(self) -> None:
        self._socket.close(0)
        self._context.term()
