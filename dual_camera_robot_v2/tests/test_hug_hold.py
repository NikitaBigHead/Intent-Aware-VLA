import unittest

from dualcam.hold import HugUnknownHold


def decision(label, reason=""):
    return {"label": label, "label_index": -1 if label == "unknown" else 0, "reason": reason}


class HugUnknownHoldTest(unittest.TestCase):
    def test_disabled_does_not_change_unknown(self):
        hold = HugUnknownHold(enabled=False, duration_s=3.0)
        hold.apply(decision("hug"), 1.0)
        self.assertEqual(hold.apply(decision("unknown"), 1.2)["label"], "unknown")

    def test_holds_unknown_from_last_real_hug_for_bounded_time(self):
        hold = HugUnknownHold(enabled=True, duration_s=2.0)
        hold.apply(decision("hug"), 10.0)

        held = hold.apply(decision("unknown", "low_pose_quality"), 11.0)
        self.assertEqual(held["label"], "hug")
        self.assertEqual(held["raw_label"], "unknown")
        self.assertTrue(held["held_from_unknown"])
        self.assertAlmostEqual(held["hug_hold_remaining_s"], 1.0)

        # A held decision does not slide the timer indefinitely.
        expired = hold.apply(decision("unknown"), 12.1)
        self.assertEqual(expired["label"], "unknown")

    def test_known_non_hug_class_cancels_hold(self):
        hold = HugUnknownHold(enabled=True, duration_s=3.0)
        hold.apply(decision("hug"), 1.0)
        hold.apply(decision("handshake"), 1.2)
        self.assertEqual(hold.apply(decision("unknown"), 1.3)["label"], "unknown")

    def test_reset_cancels_hold(self):
        hold = HugUnknownHold(enabled=True, duration_s=3.0)
        hold.apply(decision("hug"), 1.0)
        hold.reset()
        self.assertEqual(hold.apply(decision("unknown"), 1.2)["label"], "unknown")


if __name__ == "__main__":
    unittest.main()
