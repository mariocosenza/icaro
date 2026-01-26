import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from src.pipeline_horizontal_classification import extract_frame_features_fall, extract_frame_features_horizontal
from src.util_landmarks import BodyLandmark


def make_pose(y_nose=0.1, y_shoulders=0.3, y_hips=0.6, y_ankles=1.0):
    pose = []
    for i in range(33):
        pose.append({
            "x": float(i) * 0.01,
            "y": float(i) * 0.02,
            "visibility": 1.0,
            "presence": 1.0,
        })
    pose[BodyLandmark.NOSE] = {"x": 0.05, "y": y_nose, "visibility": 1.0, "presence": 1.0}
    pose[BodyLandmark.LEFT_SHOULDER] = {"x": 0.1, "y": y_shoulders, "visibility": 1.0, "presence": 1.0}
    pose[BodyLandmark.RIGHT_SHOULDER] = {"x": 0.12, "y": y_shoulders, "visibility": 1.0, "presence": 1.0}
    pose[BodyLandmark.LEFT_HIP] = {"x": 0.2, "y": y_hips, "visibility": 1.0, "presence": 1.0}
    pose[BodyLandmark.RIGHT_HIP] = {"x": 0.22, "y": y_hips, "visibility": 1.0, "presence": 1.0}
    pose[BodyLandmark.LEFT_ANKLE] = {"x": 0.3, "y": y_ankles, "visibility": 1.0, "presence": 1.0}
    pose[BodyLandmark.RIGHT_ANKLE] = {"x": 0.32, "y": y_ankles, "visibility": 1.0, "presence": 1.0}
    return pose


class TestFeatureExtractors(unittest.TestCase):
    def test_extract_frame_features_fall_returns_quality_and_features(self):
        pose = make_pose()
        feats, q = extract_frame_features_fall(
            pose,
            BodyLandmark,
            min_vis_point=0.2,
            min_pres_point=0.2,
            min_required_core_points=3,
        )
        self.assertEqual(len(feats), 6)
        self.assertAlmostEqual(q, 1.0)
        self.assertAlmostEqual(feats[0], 0.1)
        self.assertAlmostEqual(feats[1], 0.6)

    def test_extract_frame_features_horizontal_provides_expected_shape(self):
        pose = make_pose()
        feats, q = extract_frame_features_horizontal(
            pose,
            BodyLandmark,
            min_quality=0.1,
            min_good_keypoints=3,
        )
        self.assertEqual(len(feats), 9)
        self.assertAlmostEqual(q, 1.0)
        self.assertGreater(feats[0], 0.0)
        self.assertGreaterEqual(feats[1], 0.0)

    def test_extract_frame_features_horizontal_rejects_low_quality(self):
        pose = make_pose(y_nose=0.1, y_shoulders=0.3, y_hips=0.6, y_ankles=1.0)
        pose[BodyLandmark.LEFT_ANKLE]["visibility"] = 0.1
        pose[BodyLandmark.RIGHT_ANKLE]["visibility"] = 0.1
        result = extract_frame_features_horizontal(
            pose,
            BodyLandmark,
            min_quality=0.9,
            min_good_keypoints=5,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
