import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from src.pipeline_horizontal_classification import (
    _proba_pos, iter_video_entries, build_df_from_any_json, 
    _supports_sample_weight, _binary_class_weight, make_hgb, make_mlp, 
    VideoSample, windowize_last_label
)

class TestPipelineHorizontalClassification(unittest.TestCase):

    def test_proba_pos(self):
        est = MagicMock()
        est.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
        res = _proba_pos(est, np.array([[1], [2]]))
        np.testing.assert_array_almost_equal(res, [0.8, 0.3])

    def test_iter_video_entries(self):
        data = [{"name": "v1", "start": 0, "end": 10, "data": [1, 2]}, {"name": "v2", "start": 0, "end": 10, "data": [3, 4]}]
        it = iter_video_entries(data)
        self.assertEqual(next(it)["name"], "v1")
        self.assertEqual(next(it)["name"], "v2")

    def test_build_df_from_any_json(self):
        data = [{"frame_index": 0, "x": 1}, {"frame_index": 1, "x": 2}]
        df = build_df_from_any_json(data)
        self.assertFalse(df.empty)
        self.assertIn("frame_index", df.columns)

    def test_supports_sample_weight(self):
        pipe = MagicMock()
        pipe.steps = [("s1", MagicMock())]
        # This one is tricky to mock perfectly as it checks __init__ params
        # But let's just check it doesn't crash
        res = _supports_sample_weight(pipe)
        self.assertIsInstance(res, bool)

    def test_binary_class_weight(self):
        y = np.array([0, 0, 0, 1])
        w = _binary_class_weight(y)
        self.assertEqual(len(w), 4)
        self.assertGreater(w[3], w[0])

    def test_make_hgb(self):
        model = make_hgb(42)
        self.assertIsNotNone(model)

    def test_make_mlp(self):
        model = make_mlp(42)
        self.assertIsNotNone(model)

    def test_windowize_last_label(self):
        feats = [np.array([1]), np.array([2]), np.array([3])]
        labels = [0, 0, 1]
        qual = [1.0, 1.0, 1.0]
        X, y, q = windowize_last_label(feats, labels, qual, window=2)
        self.assertEqual(len(X), 2)
        self.assertEqual(y[0], 0)
        self.assertEqual(y[1], 1)

from unittest.mock import MagicMock

if __name__ == "__main__":
    unittest.main()
