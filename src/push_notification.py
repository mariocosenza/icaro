import logging

import firebase_admin
from firebase_admin import credentials, messaging

import os
import httpx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RASPBERRY_PI_IP = os.getenv("RASPBERRY_PI_IP", "127.0.0.1")

cred = credentials.Certificate("../data/account/serviceAccountKey.json")
firebase_admin.initialize_app(cred)


class LatestHeartbeat:
    BPM: float = 80
    NOTIED_FALL: bool = False


class LatestMovement:
    X: float = 0
    Y: float = 0
    Z: float = 0


async def trigger_buzzer():
    """
    Sends a request to the Raspberry Pi server to trigger the buzzer.
    """
    url = f"http://{RASPBERRY_PI_IP}:8000/alert"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(url, json={"message": "FALL DETECTED", "duration": 5.0})
            logging.info(f"Successfully triggered buzzer at {url}")
    except Exception as e:
        logging.error(f"Failed to trigger buzzer at {url}: {e}")


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

    await trigger_buzzer()


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


async def send_monitoring_notification():
    message = messaging.Message(
        notification=messaging.Notification(
            title="Start monitoring",
            body=""
        ),
        topic="fall",
    )
    response = messaging.send(message)
    logging.info(f"Successfully sent message: {response}")
