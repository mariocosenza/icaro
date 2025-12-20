import logging
import firebase_admin
from firebase_admin import credentials, messaging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cred = credentials.Certificate("../data/account/serviceAccountKey.json")
firebase_admin.initialize_app(cred)

def send_push_notification(title, body):
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
         topic="fall",
    )
    response = messaging.send(message)
    logging.info(f"Successfully sent message: {response}")