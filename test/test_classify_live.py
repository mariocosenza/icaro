import unittest
from unittest.mock import MagicMock, patch
import asyncio
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Mock dependencies
sys.modules['firebase_admin'] = MagicMock()
sys.modules['firebase_admin.credentials'] = MagicMock()
sys.modules['firebase_admin.messaging'] = MagicMock()
sys.modules['joblib'] = MagicMock()

from src.classify_live import (
    _lm_to_dict, _result_to_frame_landmarks, _is_no_movement, 
    _is_abnormal_hr, _can_reenable_detector, classify_live
)
from src.util_landmarks import BodyLandmark

class TestClassifyLive(unittest.TestCase):

    def test_lm_to_dict(self):
        lm = MagicMock()
        lm.x, lm.y, lm.z = 1.0, 2.0, 3.0
        lm.visibility, lm.presence = 0.9, 0.8
        res = _lm_to_dict(lm)
        self.assertEqual(res['x'], 1.0)
        self.assertEqual(res['visibility'], 0.9)

    def test_result_to_frame_landmarks(self):
        result = MagicMock()
        lm = MagicMock()
        lm.x, lm.y, lm.z = 1.0, 2.0, 3.0
        lm.visibility, lm.presence = 0.9, 0.8
        result.pose_landmarks = [[lm] * 33]
        res = _result_to_frame_landmarks(result)
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 33)
        self.assertEqual(res[0][0]['x'], 1.0)

    @patch('src.classify_live.LatestMovement')
    def test_is_no_movement(self, mock_move):
        mock_move.X, mock_move.Y, mock_move.Z = 0.1, 0.1, 0.1
        self.assertTrue(_is_no_movement())
        mock_move.X = 1.0
        self.assertFalse(_is_no_movement())

    @patch('src.classify_live.LatestHeartbeat')
    def test_is_abnormal_hr(self, mock_hr):
        mock_hr.BPM = 100
        self.assertFalse(_is_abnormal_hr())
        mock_hr.BPM = 20
        self.assertTrue(_is_abnormal_hr())
        mock_hr.BPM = 200
        self.assertTrue(_is_abnormal_hr())

    def test_can_reenable_detector(self):
        import src.classify_live as cl
        cl._detector_disabled_until_ms = 4000
        result = MagicMock()
        lm = MagicMock()
        lm.y = 0.5
        # Mocking subscriptable list
        landmarks = [MagicMock()] * 33
        landmarks[BodyLandmark.LEFT_SHOULDER] = lm
        result.pose_landmarks = [landmarks]
        
        with patch('src.classify_live.GroundCoordinates') as mock_ground:
            mock_ground.Y = 0.3
            self.assertTrue(_can_reenable_detector(result, 5000))
            self.assertFalse(_can_reenable_detector(result, 2000))

    @patch('src.classify_live.draw_pose_points')
    @patch('src.classify_live.MultiPersonDetector')
    @patch('src.classify_live._result_to_frame_landmarks')
    @patch('src.classify_live._is_no_movement')
    @patch('src.classify_live._is_abnormal_hr')
    @patch('src.classify_live._schedule')
    def test_classify_live_basic(self, mock_sched, mock_hr, mock_mov, mock_res_to_lm, mock_det_cls, mock_draw):
        # We need to initialize the global detector
        import src.classify_live as cl
        cl._detector = MagicMock()
        cl._detector.update.return_value = [{"ready": True, "fall_event": False, "horizontal_event": False, "track_id": 0}]
        
        mock_res_to_lm.return_value = [[]]
        mock_mov.return_value = False
        mock_hr.return_value = False
        
        result = MagicMock()
        result.pose_landmarks = [True]
        
        classify_live(result, MagicMock(), 1000)
        
        cl._detector.update.assert_called_once()

if __name__ == "__main__":
    unittest.main()
