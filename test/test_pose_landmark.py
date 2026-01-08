import unittest
from unittest.mock import MagicMock, patch
import asyncio
import numpy as np
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Mock firebase and other dependencies that fail on import
sys.modules['firebase_admin'] = MagicMock()
sys.modules['firebase_admin.credentials'] = MagicMock()
sys.modules['firebase_admin.messaging'] = MagicMock()
sys.modules['joblib'] = MagicMock()

from src.pose_landmark import (
    _resolve_width, _resize_frame, _mp_image_from_bgr, 
    _timestamp_live, _timestamp_video, PoseConfig, _make_detector
)

class TestPoseLandmark(unittest.TestCase):

    def test_resolve_width(self):
        self.assertEqual(_resolve_width("low"), 480)
        self.assertEqual(_resolve_width(100), 100)
        self.assertIsNone(_resolve_width(None))

    def test_resize_frame(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        res = _resize_frame(frame, 100)
        self.assertEqual(res.shape[1], 100)
        self.assertEqual(res.shape[0], 50)
        
        res2 = _resize_frame(frame, None)
        self.assertEqual(res2.shape, (100, 200, 3))

    def test_mp_image_from_bgr(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch('cv2.cvtColor') as mock_cvt:
            mock_cvt.return_value = frame
            img = _mp_image_from_bgr(frame)
            self.assertIsNotNone(img)

    def test_timestamp_live(self):
        start = 100.0
        with patch('time.time', return_value=100.5):
            self.assertEqual(_timestamp_live(start), 500)

    def test_timestamp_video(self):
        cap = MagicMock()
        cap.get.return_value = 1000.0
        self.assertEqual(_timestamp_video(cap, 10, 30.0), 1000)
        
        cap.get.return_value = 0.0
        self.assertEqual(_timestamp_video(cap, 3, 30.0), 100)

    @patch('src.pose_landmark.vision.PoseLandmarker')
    def test_make_detector(self, mock_landmarker):
        cfg = PoseConfig()
        from mediapipe.tasks.python.vision import RunningMode
        _make_detector(cfg, RunningMode.VIDEO)
        mock_landmarker.create_from_options.assert_called_once()

if __name__ == "__main__":
    unittest.main()
