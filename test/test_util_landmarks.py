import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.util_landmarks import GroundCoordinates, BodyLandmark


class TestUtilLandmarks(unittest.TestCase):
    def test_ground_coordinates_initial_values(self):
        self.assertEqual(GroundCoordinates.X, 0)
        self.assertEqual(GroundCoordinates.Y, 0)
        self.assertEqual(GroundCoordinates.Z, 0)

    def test_body_landmark_enum(self):
        self.assertEqual(BodyLandmark.NOSE, 0)
        self.assertEqual(BodyLandmark.LEFT_SHOULDER, 11)
        self.assertEqual(BodyLandmark.RIGHT_FOOT_INDEX, 32)
        self.assertEqual(len(BodyLandmark), 33)


if __name__ == "__main__":
    unittest.main()
