import logging

import firebase_admin
from firebase_admin import credentials, messaging

import os
import httpx
import socket

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resolve_hostname_or_default(hostname: str, default_ip: str) -> str:
    try:
        ip = socket.gethostbyname(hostname)
        logging.info(f"Resolved {hostname} to {ip}")
        return ip
    except socket.error:
        logging.warning(f"Could not resolve {hostname}, using fallback: {default_ip}")
        return default_ip

RASPBERRY_PI_IP = resolve_hostname_or_default("raspberrypi6", os.getenv("RASPBERRY_PI_IP", "127.0.0.1"))
logging.info(f"Initial Raspberry Pi IP: {RASPBERRY_PI_IP}")

def set_raspberry_pi_ip(ip: str):
    global RASPBERRY_PI_IP
    RASPBERRY_PI_IP = ip
    logging.info(f"Raspberry Pi IP updated to: {RASPBERRY_PI_IP}")

def get_raspberry_pi_ip() -> str:
    return RASPBERRY_PI_IP

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
