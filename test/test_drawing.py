import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

HAS_MEDIAPIPE = importlib.util.find_spec("mediapipe") is not None
HAS_CV2 = importlib.util.find_spec("cv2") is not None

if HAS_MEDIAPIPE:
    import mediapipe as mp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from src.drawing import _fit_to_screen, draw_pose_points, show_single_image, show_video_loop


@unittest.skipUnless(HAS_MEDIAPIPE and HAS_CV2, "mediapipe/cv2 not installed")
class TestDrawing(unittest.TestCase):

    def test_fit_to_screen_no_resize(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = _fit_to_screen(img, max_w=200, max_h=200)
        self.assertEqual(out.shape, (100, 100, 3))

    def test_fit_to_screen_resize_w(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        out = _fit_to_screen(img, max_w=100, max_h=100)
        self.assertEqual(out.shape[1], 100)
        self.assertEqual(out.shape[0], 50)

    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.namedWindow')
    @patch('cv2.getWindowProperty')
    def test_draw_pose_points(self, mock_prop, mock_named, mock_wait, mock_imshow):
        mock_prop.return_value = 1
        mock_wait.return_value = ord('a')

        img_data = np.zeros((100, 100, 3), dtype=np.uint8)
        mp_image = MagicMock(spec=mp.Image)
        mp_image.numpy_view.return_value = img_data

        # Mock detection result with one pose and one landmark
        landmark = MagicMock()
        landmark.x = 0.5
        landmark.y = 0.5
        pose = [landmark]
        detection_result = MagicMock()
        detection_result.pose_landmarks = [pose]

        res = draw_pose_points(mp_image, detection_result)
        self.assertTrue(res)
        mock_imshow.assert_called_once()
        mock_named.assert_called_once()

    @patch('src.drawing.draw_pose_points')
    def test_show_single_image(self, mock_draw):
        mp_image = MagicMock()
        detection_result = MagicMock()
        show_single_image(mp_image, detection_result)
        mock_draw.assert_called_once_with(mp_image, detection_result, window_name="Pose", wait_ms=0)

    @patch('src.drawing.draw_pose_points')
    @patch('cv2.destroyAllWindows')
    def test_show_video_loop(self, mock_destroy, mock_draw):
        mock_draw.side_effect = [True, False]  # Continue then stop
        frames = [MagicMock(), MagicMock()]
        get_result = MagicMock()

        show_video_loop(frames, get_result)

        self.assertEqual(get_result.call_count, 2)
        self.assertEqual(mock_draw.call_count, 2)
        mock_destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
