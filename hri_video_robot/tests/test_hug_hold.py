import unittest

from hri_video.hold import HugUnknownHold


def decision(label, reason=""):
    return {"label": label, "label_index": -1, "reason": reason}


class HugUnknownHoldTest(unittest.TestCase):
    def test_hug_index_matches_video_model_class_order(self):
        hold = HugUnknownHold(enabled=True, duration_s=2.0, hug_label_index=1)
        hold.apply(decision("hug"), 1.0)
        held = hold.apply(decision("unknown", "unstable_prediction"), 1.5)
        self.assertEqual(held["label"], "hug")
        self.assertEqual(held["label_index"], 1)
        self.assertTrue(held["held_from_unknown"])

    def test_hold_expires_and_known_class_cancels_it(self):
        hold = HugUnknownHold(enabled=True, duration_s=1.0, hug_label_index=1)
        hold.apply(decision("hug"), 1.0)
        self.assertEqual(hold.apply(decision("unknown"), 2.1)["label"], "unknown")
        hold.apply(decision("hug"), 3.0)
        hold.apply(decision("fist_bump"), 3.1)
        self.assertEqual(hold.apply(decision("unknown"), 3.2)["label"], "unknown")


if __name__ == "__main__":
    unittest.main()
