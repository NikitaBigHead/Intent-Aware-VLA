import unittest

import numpy as np

from scripts.motion_prediction.maskaware.decision import DEFAULT, decide


CLASSES = ["hug", "handshake", "fist_bump", "no_interaction"]


def run_decision(probabilities, stable_frames):
    probabilities = np.asarray(probabilities, dtype=float)
    quality = np.ones((len(probabilities), 8), dtype=float)
    episodes = np.zeros(len(probabilities), dtype=int)
    times = np.arange(len(probabilities), dtype=float) * 0.2
    config = {**DEFAULT, "stable_frames": stable_frames}
    return decide(probabilities, quality, episodes, times, 0.25, CLASSES, config)


class ClassStabilityTest(unittest.TestCase):
    def test_no_interaction_requires_its_own_streak(self):
        probabilities = [[0.1, 0.1, 0.1, 0.7]] * 5
        stable = {"hug": 2, "handshake": 3, "fist_bump": 2, "no_interaction": 5}

        labels, reasons, _ = run_decision(probabilities, stable)

        self.assertEqual(labels.tolist(), [-1, -1, -1, -1, 3])
        self.assertEqual(reasons[:4].tolist(), ["unstable_prediction"] * 4)
        self.assertEqual(reasons[-1], "")

    def test_handshake_requires_three_decisions(self):
        probabilities = [[0.1, 0.7, 0.1, 0.1]] * 3
        stable = {"hug": 2, "handshake": 3, "fist_bump": 2, "no_interaction": 5}

        labels, _, _ = run_decision(probabilities, stable)

        self.assertEqual(labels.tolist(), [-1, -1, 1])

    def test_scalar_stability_remains_supported(self):
        probabilities = [[0.7, 0.1, 0.1, 0.1]] * 2
        labels, _, _ = run_decision(probabilities, 2)
        self.assertEqual(labels.tolist(), [-1, 0])


if __name__ == "__main__":
    unittest.main()
