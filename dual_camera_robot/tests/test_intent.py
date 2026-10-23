import json
import unittest

from dualcam.intent import SCHEMA_VERSION, build_intent_event


class IntentProtocolTest(unittest.TestCase):
    def test_build_intent_event_is_json_safe_and_marks_known_label_accepted(self):
        event = build_intent_event(
            {
                "label": "handshake",
                "confidence": 0.75,
                "reason": "",
                "probabilities": {"handshake": 0.75, "hug": 0.25},
                "quality": {"union_observed": 0.8, "person_detected_last": True},
                "hand_attention_last": 0.4,
            },
            seq=3,
            t_person=10.0,
            t_hand=10.01,
            skew_ms=10.0,
            produced_time=11.0,
        )

        self.assertEqual(event["version"], SCHEMA_VERSION)
        self.assertEqual(event["seq"], 3)
        self.assertEqual(event["label"], "handshake")
        self.assertIs(event["accepted"], True)
        self.assertIs(event["quality"]["person_detected_last"], True)
        json.dumps(event)

    def test_unknown_is_not_accepted(self):
        event = build_intent_event(
            {"label": "unknown", "confidence": 0.2, "reason": "low_confidence"},
            seq=0,
            t_person=1.0,
            t_hand=1.0,
            skew_ms=0.0,
        )
        self.assertIs(event["accepted"], False)

    def test_hug_hold_metadata_is_forwarded(self):
        event = build_intent_event(
            {
                "label": "hug",
                "confidence": 0.8,
                "reason": "held_hug_after_unknown:low_confidence",
                "raw_label": "unknown",
                "raw_reason": "low_confidence",
                "held_from_unknown": True,
                "hug_hold_age_s": 0.4,
                "hug_hold_remaining_s": 2.6,
            },
            seq=1,
            t_person=1.0,
            t_hand=1.0,
            skew_ms=0.0,
        )
        self.assertIs(event["accepted"], True)
        self.assertEqual(event["raw_label"], "unknown")
        self.assertIs(event["held_from_unknown"], True)
        self.assertEqual(event["hug_hold_remaining_s"], 2.6)
        json.dumps(event)


if __name__ == "__main__":
    unittest.main()
