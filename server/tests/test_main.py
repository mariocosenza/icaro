import unittest
import sys
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

sys.modules['gpiozero'] = MagicMock()
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app

class TestServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "online", "gpio_active": False})

    def test_alert_endpoint(self):
        with patch("main.play_buzzer_pattern") as mock_play:
            response = self.client.post("/alert", json={"message": "Test alert", "duration": 2.0})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "success", "message": "Alert triggered", "duration": 2.0})

    @patch("main.buzzer")
    @patch("main.has_gpio", True)
    def test_alert_triggers_buzzer_mock_gpio(self, mock_buzzer):
        from main import play_buzzer_pattern
        import asyncio
        asyncio.run(play_buzzer_pattern(0.1))

    def test_alert_no_gpio(self):
        with patch("main.has_gpio", False):
            response = self.client.post("/alert", json={"duration": 0.1})
            self.assertEqual(response.status_code, 200)
