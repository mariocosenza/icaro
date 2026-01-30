import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies BEFORE importing src
sys.modules['firebase_admin'] = MagicMock()
sys.modules['firebase_admin.credentials'] = MagicMock()
sys.modules['firebase_admin.messaging'] = MagicMock()
sys.modules['joblib'] = MagicMock()
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.tasks'] = MagicMock()
sys.modules['mediapipe.tasks.python'] = MagicMock()
sys.modules['mediapipe.tasks.python.vision'] = MagicMock()

# Setup paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

class TestMonitorFix(unittest.TestCase):
    def setUp(self):
         # Patch internal calls that might execute during import or setup
         with patch('src.mongodb.get_database'), \
              patch('src.pose_landmark.run_pose_async'), \
              patch('src.push_notification.credentials.Certificate'), \
              patch('firebase_admin.initialize_app'):
            
            # Use importlib to reload app if it was already imported (in case of running multiple tests in same process, specific for some runners)
            # But for simple run it's fine.
            from src.app import app
            self.app = app
            from fastapi.testclient import TestClient
            self.client = TestClient(app)

    def test_monitor_negative_values(self):
        # Case causing error: y is negative
        response = self.client.post("/api/v1/monitor/4.587--0.807-8.367")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        self.assertEqual(response.status_code, 200)
        
        # Verify values in LatestMovement
        from src.app import LatestMovement
        self.assertAlmostEqual(LatestMovement.X, 4.587)
        self.assertAlmostEqual(LatestMovement.Y, -0.807)
        self.assertAlmostEqual(LatestMovement.Z, 8.367)

    def test_monitor_positive_values(self):
        # Regression test for standard format
        response = self.client.post("/api/v1/monitor/1.0-2.0-3.0")
        self.assertEqual(response.status_code, 200)
        from src.app import LatestMovement
        self.assertEqual(LatestMovement.X, 1.0)
        self.assertEqual(LatestMovement.Y, 2.0)
        self.assertEqual(LatestMovement.Z, 3.0)

if __name__ == "__main__":
    unittest.main()
