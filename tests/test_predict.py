"""Single-image inference returns valid probabilities and a mask at the input's original size."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import build_model
from predict import predict


class PredictTests(unittest.TestCase):
    def test_output_shapes(self):
        model = build_model(pretrained=False).eval()
        image = np.random.default_rng(0).integers(0, 256, (300, 200), dtype=np.uint8)
        probs, mask = predict(model, image, 64, "cpu")
        self.assertEqual(probs.shape, (3,))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertEqual(mask.shape, image.shape)
        self.assertTrue(set(np.unique(mask)) <= {0, 1})


if __name__ == "__main__":
    unittest.main()
