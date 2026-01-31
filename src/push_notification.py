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


RASPBERRY_PI_HOST = resolve_hostname_or_default("raspberrypi6", os.getenv("RASPBERRY_PI_IP", "127.0.0.1"))
logging.info(f"Initial Raspberry Pi Address: {RASPBERRY_PI_HOST}")

def set_raspberry_pi_address(address: str):
    global RASPBERRY_PI_HOST
    RASPBERRY_PI_HOST = address
    logging.info(f"Raspberry Pi Address updated to: {RASPBERRY_PI_HOST}")

def get_raspberry_pi_address() -> str:
    return RASPBERRY_PI_HOST

cred = credentials.Certificate("../data/account/serviceAccountKey.json")
firebase_admin.initialize_app(cred)


class LatestHeartbeat:
    BPM: float = 0
    NOTIFIED_FALL: bool = False


class LatestMovement:
    X: float = 0
    Y: float = 0
    Z: float = 0



async def trigger_buzzer() -> bool:
    """
    Sends a request to the Raspberry Pi server to trigger the buzzer.
    Returns True if successful, False otherwise.
    """
    if RASPBERRY_PI_HOST.startswith("http://") or RASPBERRY_PI_HOST.startswith("https://"):
        # Treat as full base URL
        url = f"{RASPBERRY_PI_HOST}/alert"
    else:
        # Treat as hostname/IP, assume HTTP and port 8080
        url = f"http://{RASPBERRY_PI_HOST}/alert"
        
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(url, json={"message": "FALL DETECTED", "duration": 5.0})
            logging.info(f"Successfully triggered buzzer at {url}")
            return True
    except Exception as e:
        logging.error(f"Failed to trigger buzzer at {url}: {e}")
        return False


async def stop_buzzer():
    """
    Sends a request to the Raspberry Pi server to stop the buzzer.
    """
    if RASPBERRY_PI_HOST.startswith("http://") or RASPBERRY_PI_HOST.startswith("https://"):
        url = f"{RASPBERRY_PI_HOST}/stop"
    else:
        url = f"http://{RASPBERRY_PI_HOST}/stop"

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(url)
            logging.info(f"Successfully stopped buzzer at {url}")
    except Exception as e:
        logging.error(f"Failed to stop buzzer at {url}: {e}")


async def send_push_notification(title, body):
    await trigger_buzzer()


    try:
        # Send data message first
        messaging.send(messaging.Message(
            data={
                "action": "FALL_DETECTED",
                "urgency": "high"
            },
            topic="fall",
        ))
        
        # Send notification message
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            topic="fall",
        )
        response = messaging.send(message)
        logging.info(f"Successfully sent Firebase message: {response}")
        
    except Exception as e:
        logging.error(f"Firebase notification failed: {e}")


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
