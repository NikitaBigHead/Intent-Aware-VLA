import unittest

import numpy as np

from scripts.motion_prediction.maskaware.decision import DEFAULT, decide


CLASSES = ["hug", "handshake", "fist_bump", "no_interaction"]


def run_decision(probabilities, threshold):
    probabilities = np.asarray(probabilities, dtype=float)
    quality = np.ones((len(probabilities), 8), dtype=float)
    episodes = np.zeros(len(probabilities), dtype=int)
    times = np.arange(len(probabilities), dtype=float) * 0.2
    return decide(probabilities, quality, episodes, times, threshold, CLASSES, DEFAULT)


class ClassThresholdTest(unittest.TestCase):
    def test_class_threshold_can_reject_handshake_only(self):
        probabilities = [[0.1, 0.6, 0.1, 0.2]] * 2
        thresholds = {label: 0.25 for label in CLASSES}
        thresholds["handshake"] = 0.65

        labels, reasons, _ = run_decision(probabilities, thresholds)

        self.assertEqual(labels[-1], -1)
        self.assertEqual(reasons[-1], "low_action_confidence")

    def test_other_class_keeps_its_own_threshold(self):
        probabilities = [[0.6, 0.1, 0.1, 0.2]] * 2
        thresholds = {label: 0.25 for label in CLASSES}
        thresholds["handshake"] = 0.9

        labels, reasons, _ = run_decision(probabilities, thresholds)

        self.assertEqual(labels[-1], 0)
        self.assertEqual(reasons[-1], "")

    def test_scalar_threshold_remains_supported(self):
        probabilities = [[0.1, 0.6, 0.1, 0.2]] * 2
        labels, _, _ = run_decision(probabilities, 0.25)
        self.assertEqual(labels[-1], 1)


if __name__ == "__main__":
    unittest.main()
