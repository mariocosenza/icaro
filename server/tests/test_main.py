import unittest
import sys
import asyncio
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

sys.modules['gpiozero'] = MagicMock()
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app

class TestServer(unittest.TestCase):
    def setUp(self):
        self.client_ctx = TestClient(app)
        self.client = self.client_ctx.__enter__()

    def tearDown(self):
        try:
           self.client.post("/stop")
        except:
           pass
        self.client_ctx.__exit__(None, None, None)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("buzzer_active", data)

    def test_alert_start_stop(self):
        # Start alert
        response = self.client.post("/alert", json={"message": "Test alert"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")

        # Check health says active
        response = self.client.get("/health")
        self.assertTrue(response.json()["buzzer_active"])

        # Stop alert
        response = self.client.post("/stop")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")

        # Check health says inactive
        response = self.client.get("/health")
        self.assertFalse(response.json()["buzzer_active"])

    def test_alert_ignored_if_active(self):
        self.client.post("/alert", json={})
        response = self.client.post("/alert", json={})
        self.assertEqual(response.json()["status"], "ignored")

    def test_stop_ignored_if_inactive(self):
        # Make sure it's stopped
        self.client.post("/stop")
        
        response = self.client.post("/stop")
        self.assertEqual(response.json()["status"], "ignored")

    @patch("main.buzzer")
    @patch("main.has_gpio", True)
    def test_continuous_buzz_flow(self, mock_buzzer):
        # Start
        self.client.post("/alert", json={})
        # Stop
        self.client.post("/stop")
        pass
