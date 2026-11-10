import unittest
from metrics_utils import binary_metrics

class MetricsTests(unittest.TestCase):
    def test_known_confusion(self):
        m = binary_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(m['accuracy'], .75)
        self.assertAlmostEqual(m['precision'], 2/3)
        self.assertEqual(m['recall'], 1)
        self.assertAlmostEqual(m['f1'], .8)
    def test_no_positive_predictions(self):
        self.assertEqual(binary_metrics([0, 1], [0, 0])['f1'], 0)
    def test_invalid_inputs(self):
        for y, p in [([], []), ([0], [0, 1]), ([2], [0])]:
            with self.assertRaises(ValueError):
                binary_metrics(y, p)

if __name__ == '__main__':
    unittest.main()
