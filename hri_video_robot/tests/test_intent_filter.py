import unittest

from hri_video.intent_filter import IntentDecisionFilter


CLASSES = ["no_interaction", "hug", "handshake", "fist_bump"]
STABLE = {"no_interaction": 5, "hug": 2, "handshake": 3, "fist_bump": 2}


class IntentDecisionFilterTest(unittest.TestCase):
    def make_filter(self):
        return IntentDecisionFilter(CLASSES, 0.25, 2, class_stable_frames=STABLE)

    def test_no_interaction_requires_five_consecutive_decisions(self):
        f = self.make_filter()
        outputs = [f.apply([0.7, 0.1, 0.1, 0.1]) for _ in range(5)]
        self.assertEqual([out["label"] for out in outputs[:4]], ["unknown"] * 4)
        self.assertEqual(outputs[-1]["label"], "no_interaction")

    def test_handshake_requires_three_consecutive_decisions(self):
        f = self.make_filter()
        outputs = [f.apply([0.1, 0.1, 0.7, 0.1]) for _ in range(3)]
        self.assertEqual([out["label"] for out in outputs], ["unknown", "unknown", "handshake"])

    def test_class_change_resets_streak(self):
        f = self.make_filter()
        f.apply([0.1, 0.1, 0.7, 0.1])
        f.apply([0.1, 0.1, 0.1, 0.7])
        out = f.apply([0.1, 0.1, 0.7, 0.1])
        self.assertEqual(out["label"], "unknown")
        self.assertEqual(out["stable_count"], 1)

    def test_class_specific_confidence_rejects_only_target_class(self):
        f = IntentDecisionFilter(
            CLASSES,
            0.25,
            1,
            class_thresholds={"handshake": 0.75},
        )
        rejected = f.apply([0.1, 0.1, 0.7, 0.1])
        accepted = f.apply([0.1, 0.7, 0.1, 0.1])
        self.assertEqual(rejected["reason"], "low_action_confidence")
        self.assertEqual(accepted["label"], "hug")


if __name__ == "__main__":
    unittest.main()
