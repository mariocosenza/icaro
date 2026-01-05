import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from src.live_fall_detector import LiveManDownDetector, DetectorConfig, MultiPersonDetector
from src.util_landmarks import BodyLandmark


def make_mock_model(prob: float) -> MagicMock:
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - prob, prob]], dtype=np.float32)
    return model


def make_pose(x: float = 0.05, y: float = 0.1):
    pose = []
    for i in range(33):
        pose.append({
            "x": float(i) * 0.01,
            "y": float(i) * 0.02,
            "visibility": 1.0,
            "presence": 1.0,
        })
    pose[BodyLandmark.NOSE] = {"x": x, "y": y, "visibility": 1.0, "presence": 1.0}
    pose[BodyLandmark.LEFT_SHOULDER] = {"x": 0.1, "y": 0.3, "visibility": 1.0, "presence": 1.0}
    pose[BodyLandmark.RIGHT_SHOULDER] = {"x": 0.12, "y": 0.3, "visibility": 1.0, "presence": 1.0}
    pose[BodyLandmark.LEFT_HIP] = {"x": 0.2, "y": 0.6, "visibility": 1.0, "presence": 1.0}
    pose[BodyLandmark.RIGHT_HIP] = {"x": 0.22, "y": 0.6, "visibility": 1.0, "presence": 1.0}
    pose[BodyLandmark.LEFT_ANKLE] = {"x": 0.3, "y": 1.0, "visibility": 1.0, "presence": 1.0}
    pose[BodyLandmark.RIGHT_ANKLE] = {"x": 0.32, "y": 1.0, "visibility": 1.0, "presence": 1.0}
    return pose


def _default_config() -> DetectorConfig:
    return DetectorConfig(
        fall_threshold=0.5,
        horizontal_threshold=0.5,
        consecutive_fall=1,
        consecutive_horizontal=1,
        min_window_quality=0.0,
        horizontal_min_quality=0.0,
        fall_min_vis_point=0.0,
        fall_min_pres_point=0.0,
        fall_min_required_core_points=1,
        horizontal_min_good_keypoints=1,
    )


class TestLiveFallDetector(unittest.TestCase):
    def test_update_returns_events_after_warmup(self):
        detector = LiveManDownDetector(
            BodyLandmark,
            fall_model=make_mock_model(0.9),
            horizontal_model=make_mock_model(0.8),
            window=2,
            config=_default_config(),
        )

        first = detector.update(make_pose())
        self.assertFalse(first.get("ready", False))
        second = detector.update(make_pose())
        self.assertTrue(second.get("ready", False))
        self.assertTrue(second.get("fall_event", False))
        self.assertTrue(second.get("horizontal_event", False))
        self.assertGreaterEqual(second.get("fall_prob", 0.0), 0.5)
        self.assertGreaterEqual(second.get("horizontal_prob", 0.0), 0.5)

    def test_no_pose_returns_reason_without_model_calls(self):
        fall_model = make_mock_model(0.9)
        horiz_model = make_mock_model(0.8)
        detector = LiveManDownDetector(
            BodyLandmark,
            fall_model=fall_model,
            horizontal_model=horiz_model,
            window=2,
            config=_default_config(),
        )

        out = detector.update(None)
        self.assertFalse(out.get("ready", True))
        self.assertEqual(out.get("reason"), "no_pose")
        fall_model.predict_proba.assert_not_called()
        horiz_model.predict_proba.assert_not_called()

    def test_multi_person_tracks_are_created_and_reused(self):
        det1 = MagicMock()
        det1.update.return_value = {"ready": True, "fall_event": False, "fall_prob": 0.1, "horizontal_event": False, "horizontal_prob": None}
        det2 = MagicMock()
        det2.update.return_value = {"ready": True, "fall_event": False, "fall_prob": 0.2, "horizontal_event": False, "horizontal_prob": None}

        factory = MagicMock()
        factory.side_effect = [det1, det2]

        multi = MultiPersonDetector(BodyLandmark, factory, distance_threshold=0.5, max_tracks=4)

        pose_a = make_pose(x=0.1, y=0.1)
        pose_b = make_pose(x=2.0, y=2.0)

        first_outputs = multi.update([pose_a, pose_b])
        self.assertEqual(len(first_outputs), 2)
        track_ids = {entry["track_id"] for entry in first_outputs}
        self.assertEqual(len(track_ids), 2)
        factory.assert_called()

        second_outputs = multi.update([make_pose(x=0.11, y=0.11)])
        self.assertEqual(len(second_outputs), 1)
        self.assertIn(second_outputs[0]["track_id"], track_ids)


if __name__ == "__main__":
    unittest.main()
