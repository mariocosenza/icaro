import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import mediapipe as mp
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from src.calibration import _timestamp_ms, print_result, pose_result_callback, calibrate_ground_for_stream
from src.util_landmarks import BodyLandmark, GroundCoordinates

class TestCalibration(unittest.TestCase):

    def test_timestamp_ms_webcam(self):
        cap = MagicMock()
        res = _timestamp_ms(cap, webcam=True, frame_index=10)
        self.assertEqual(res, int(10 * (1000 / 30)))

    def test_timestamp_ms_video(self):
        cap = MagicMock()
        cap.get.return_value = 500.0
        res = _timestamp_ms(cap, webcam=False, frame_index=10)
        self.assertEqual(res, 500)

    def test_print_result_thumb_up(self):
        import src.calibration as cal
        cal.THUMB_UP = False
        result = MagicMock()
        gesture = MagicMock()
        gesture.category_name = "Thumb_Up"
        result.gestures = [[gesture]]
        
        print_result(result, MagicMock(), 0)
        self.assertTrue(cal.THUMB_UP)

    @patch('src.calibration.draw_pose_points')
    def test_pose_result_callback(self, mock_draw):
        import src.calibration as cal
        cal.CALIBRATED = False
        
        result = MagicMock()
        landmark = MagicMock()
        landmark.x = 1.0
        landmark.y = 2.0
        landmark.z = 3.0
        landmark.presence = 0.9
        landmark.visibility = 0.9
        
        # We need to mock a list that behaves like it has BodyLandmark.LEFT_KNEE and RIGHT_KNEE
        landmarks = [MagicMock()] * 33
        landmarks[BodyLandmark.LEFT_KNEE] = landmark
        landmarks[BodyLandmark.RIGHT_KNEE] = landmark
        
        result.pose_world_landmarks = [landmarks]
        
        pose_result_callback(result, MagicMock(), 0)
        
        self.assertTrue(cal.CALIBRATED)
        self.assertEqual(cal.GROUND['x'], 1.0)
        self.assertEqual(cal.GROUND['y'], 2.0)
        self.assertEqual(cal.GROUND['z'], 3.0)

    @patch('src.calibration.PoseLandmarker')
    @patch('src.calibration.GestureRecognizer')
    @patch('cv2.VideoCapture')
    @patch('cv2.cvtColor')
    @patch('mediapipe.Image')
    @patch('pandas.DataFrame')
    def test_calibrate_ground_for_stream(self, mock_df, mock_mp_img, mock_cvt, mock_cap, mock_gesture, mock_pose):
        import src.calibration as cal
        cal.THUMB_UP = True # skip first loop
        cal.CALIBRATED = False # Allow the loop to run once
        cal.GROUND = {"x": 1, "y": 2, "z": 3}
        
        cap_instance = mock_cap.return_value
        cap_instance.isOpened.side_effect = [True, True, False] # For second loop
        cap_instance.read.return_value = (True, np.zeros((10,10,3)))
        cap_instance.get.return_value = 0.0
        
        # Mock context managers
        mock_gesture.create_from_options.return_value.__enter__.return_value = MagicMock()
        
        mock_pose_instance = MagicMock()
        mock_pose.create_from_options.return_value.__enter__.return_value = mock_pose_instance
        
        # We need to make it calibrated during the loop
        def set_calibrated(*args, **kwargs):
            cal.CALIBRATED = True
        mock_pose_instance.detect_async.side_effect = set_calibrated
        
        calibrate_ground_for_stream("dummy_path", webcam=False)
        
        # In src/calibration.py, GroundCoordinates is imported from util_landmarks.
        # It should have been updated there.
        self.assertEqual(cal.GroundCoordinates.X, cal.GROUND["x"])
        mock_df.assert_called()

if __name__ == "__main__":
    unittest.main()
