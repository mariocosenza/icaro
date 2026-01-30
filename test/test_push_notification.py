import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Mocking firebase_admin and messaging before importing src.push_notification
# because it initializes app at module level
mock_firebase_admin = MagicMock()
mock_messaging = MagicMock()
mock_credentials = MagicMock()

sys.modules['firebase_admin'] = mock_firebase_admin
sys.modules['firebase_admin.credentials'] = mock_credentials
sys.modules['firebase_admin.messaging'] = mock_messaging

import src.push_notification as pn


class TestPushNotification(unittest.TestCase):

    @patch('src.push_notification.messaging')
    def test_send_push_notification(self, mock_msg):
        # Mock Notification to return a real-ish object
        def mock_notif(title, body):
            m = MagicMock()
            m.title = title
            m.body = body
            return m

        mock_msg.Notification.side_effect = mock_notif

        # We need to capture the arguments passed to Message
        messages = []

        def capture_message(**kwargs):
            m = MagicMock()
            m._kwargs = kwargs
            messages.append(m)
            return m

        mock_msg.Message.side_effect = capture_message

        asyncio.run(pn.send_push_notification("test title", "test body"))
        self.assertEqual(mock_msg.send.call_count, 2)

        # Check calls to Message
        self.assertEqual(len(messages), 2)
        # First call has data
        self.assertEqual(messages[0]._kwargs['data']['action'], "FALL_DETECTED")
        # Second call has notification
        self.assertEqual(messages[1]._kwargs['notification'].title, "test title")

    @patch('src.push_notification.messaging')
    def test_send_push_notification_heartbeat(self, mock_msg):
        mock_msg.Message.side_effect = lambda **kwargs: MagicMock(**kwargs)
        mock_msg.Notification.side_effect = lambda **kwargs: MagicMock(**kwargs)

        pn.LatestHeartbeat.BPM = 80
        asyncio.run(pn.send_push_notification_heartbeat())
        mock_msg.send.assert_called_once()

        call_args = mock_msg.Message.call_args[1]
        self.assertIn("80", call_args['notification'].body)

    @patch('src.push_notification.messaging')
    def test_send_monitoring_notification(self, mock_msg):
        mock_msg.Message.side_effect = lambda **kwargs: MagicMock(**kwargs)
        mock_msg.Notification.side_effect = lambda **kwargs: MagicMock(**kwargs)

        asyncio.run(pn.send_monitoring_notification())
        mock_msg.send.assert_called_once()

        call_args = mock_msg.Message.call_args[1]
        self.assertEqual(call_args['notification'].title, "Start monitoring")


if __name__ == "__main__":
    unittest.main()
