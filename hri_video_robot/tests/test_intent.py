import json
import unittest

from hri_video.intent import SCHEMA_VERSION, build_intent_event


class IntentProtocolTest(unittest.TestCase):
    def test_event_matches_shared_vla_schema(self):
        event = build_intent_event(
            {
                "label": "fist_bump",
                "confidence": 0.8,
                "probabilities": {"fist_bump": 0.8, "no_interaction": 0.2},
                "quality": {"head_present": True, "ego_present": True},
                "reason": "",
                "raw_label": "fist_bump",
                "stable_count": 2,
                "stable_required": 2,
                "class_threshold": 0.6,
            },
            seq=4,
            t_person=10.0,
            t_hand=10.01,
            skew_ms=10.0,
            produced_time=11.0,
        )
        self.assertEqual(event["version"], SCHEMA_VERSION)
        self.assertEqual(event["label"], "fist_bump")
        self.assertIs(event["accepted"], True)
        self.assertEqual(event["stable_required"], 2)
        json.dumps(event)

    def test_unknown_is_rejected(self):
        event = build_intent_event(
            {"label": "unknown", "confidence": 0.4, "reason": "unstable_prediction"},
            seq=0,
            t_person=1.0,
            t_hand=1.0,
            skew_ms=0.0,
        )
        self.assertIs(event["accepted"], False)


if __name__ == "__main__":
    unittest.main()
