import importlib.util
import os
import sys
import unittest

import numpy as np

HAS_PANDAS = importlib.util.find_spec("pandas") is not None
if HAS_PANDAS:
    import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from src.pipeline_horizontal_classification import select_best_pose, _infer_step, window_vector_nan


@unittest.skipUnless(HAS_PANDAS, "pandas not installed")
class TestPipelineHelpers(unittest.TestCase):
    def test_select_best_pose_prefers_highest_score(self):
        low_pose = [{"visibility": 0.1, "presence": 0.1, "x": 0.0, "y": 0.0} for _ in range(33)]
        high_pose = [{"visibility": 0.9, "presence": 0.9, "x": 0.0, "y": 0.0} for _ in range(33)]
        poses = [low_pose, high_pose]
        best = select_best_pose(poses)
        self.assertIs(best, high_pose)

    def test_infer_step_uses_median_difference(self):
        df = pd.DataFrame({"frame_index": [0, 2, 4, 8, 10]})
        self.assertEqual(_infer_step(df), 2)

    def test_window_vector_nan_handles_all_nan_columns(self):
        data = np.array([[1.0, np.nan, 3.0], [3.0, np.nan, 7.0]], dtype=np.float32)
        out = window_vector_nan(data)
        self.assertEqual(out.shape[0], 18)
        self.assertTrue(np.allclose(out[[1, 4, 7, 10, 13, 16]], 0.0))

    def test_window_vector_nan_empty_returns_zero(self):
        out = window_vector_nan(np.array([]))
        self.assertEqual(out.shape, (1,))
        self.assertTrue(np.allclose(out, 0.0))


if __name__ == "__main__":
    unittest.main()
