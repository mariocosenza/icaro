import logging
import firebase_admin
from firebase_admin import credentials, messaging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cred = credentials.Certificate("../data/account/serviceAccountKey.json")
firebase_admin.initialize_app(cred)

class LatestHeartbeat:
    BPM: int = 0
    NOTIED_FALL: bool = False

async def send_push_notification(title, body):
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
         topic="fall",
    )
    messaging.send(messaging.Message(
        data={
            "action": "FALL_DETECTED",
            "urgency": "high"
        },
        topic="fall",
    ))
    response = messaging.send(message)
    logging.info(f"Successfully sent message: {response}")

async def send_push_notification_heartbeat():
    message = messaging.Message(
        notification=messaging.Notification(
            title="Man Fallen is not feeling well",
            body="Man fallen heart rate: " + str(LatestHeartbeat.BPM) + "bpm",
        ),
         topic="fall",
    )
    response = messaging.send(message)
    logging.info(f"Successfully sent message: {response}")