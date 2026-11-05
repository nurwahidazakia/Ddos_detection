import unittest

import numpy as np

from src.train_ddos_candidate import _evaluate


class TrainDdosCandidateMetricsTests(unittest.TestCase):
    def test_reports_false_positive_rate_and_wilson_interval(self):
        labels = np.array([-1, -1, 1, 1], dtype=np.int8)
        scores = np.array([0.9, 0.1, 0.8, 0.2])

        result = _evaluate(labels, scores, threshold=0.5)

        self.assertEqual(result["normal_false_positive_count"], 1)
        self.assertEqual(result["normal_sample_count"], 2)
        self.assertEqual(result["false_positive_rate_normal"], 0.5)
        lower, upper = result["false_positive_rate_normal_wilson_95_ci"]
        self.assertLessEqual(lower, 0.5)
        self.assertGreaterEqual(upper, 0.5)


if __name__ == "__main__":
    unittest.main()
