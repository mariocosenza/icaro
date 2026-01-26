import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

HAS_MEDIAPIPE = importlib.util.find_spec("mediapipe") is not None
if not HAS_MEDIAPIPE:
    raise unittest.SkipTest("mediapipe not installed")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Mock external dependencies
sys.modules['firebase_admin'] = MagicMock()
sys.modules['firebase_admin.credentials'] = MagicMock()
sys.modules['firebase_admin.messaging'] = MagicMock()
sys.modules['joblib'] = MagicMock()

# Mock internal src dependencies only during app import
from fastapi.testclient import TestClient

with patch('src.mongodb.get_database'), \
        patch('src.pose_landmark.run_pose_async'), \
        patch('src.push_notification.credentials.Certificate'), \
        patch('firebase_admin.initialize_app'):
    from src.app import app

client = TestClient(app)

mock_mongo = MagicMock()
mock_pose = MagicMock()


async def mock_run_pose_async(*args, **kwargs):
    pass


class TestApp(unittest.TestCase):

    def setUp(self):
        # Patch the functions used in app.py to use our mocks
        self.patch_mongo = patch('src.app.get_all_data_from_mongo_db', mock_mongo.get_all_data_from_mongo_db)
        self.patch_pose = patch('src.app.run_pose_async', mock_run_pose_async)
        self.patch_mongo.start()
        self.patch_pose.start()

    def tearDown(self):
        self.patch_mongo.stop()
        self.patch_pose.stop()

    def test_status(self):
        response = client.get("/api/v1/status")
        self.assertEqual(response.status_code, 200)
        self.assertIn("running", response.json())

    def test_get_alerts(self):
        mock_mongo.get_all_data_from_mongo_db.return_value = {"alerts": []}
        response = client.get("/api/v1/alerts")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"alerts": []})

    def test_post_heartbeat(self):
        response = client.post("/api/v1/measure/80")
        self.assertEqual(response.status_code, 200)
        from src.app import LatestHeartbeat
        self.assertEqual(LatestHeartbeat.BPM, 80)

    def test_post_movement(self):
        response = client.post("/api/v1/monitor/1.0-2.0-3.0")
        self.assertEqual(response.status_code, 200)
        from src.app import LatestMovement
        self.assertEqual(LatestMovement.X, 1.0)

    def test_start_pipeline(self):
        response = client.post("/api/v1/start")
        self.assertEqual(response.status_code, 200)

    def test_stop_pipeline(self):
        response = client.post("/api/v1/stop")
        self.assertEqual(response.status_code, 200)

    def test_set_running_mode_live_stream(self):
        response = client.put("/api/v1/running-mode/live-stream")
        self.assertEqual(response.status_code, 200)

    def test_set_running_mode_video(self):
        response = client.put("/api/v1/running-mode/video")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
