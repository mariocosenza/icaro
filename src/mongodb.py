import logging
import os
from datetime import datetime
from typing import Any

try:
    from pymongo import MongoClient

    PYMONGO_AVAILABLE = True
except ModuleNotFoundError:
    MongoClient = None
    PYMONGO_AVAILABLE = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONNECTION_STRING = os.getenv("CONNECTION_STRING")
if not CONNECTION_STRING:
    logging.warning("CONNECTION_STRING not found in environment variables")


def _require_pymongo() -> None:
    if not PYMONGO_AVAILABLE:
        raise RuntimeError("pymongo is required for MongoDB operations.")


def get_database():
    _require_pymongo()
    if not CONNECTION_STRING:
        raise RuntimeError("CONNECTION_STRING not found in environment variables")
    client = MongoClient(CONNECTION_STRING)
    return client['icaro']


async def insert_message_mongo_db(title: str, message: str, alert: bool) -> None:
    dbname = get_database()
    logging.log(level=logging.INFO, msg='Inserting aggregated data into MongoDB...')
    collection = dbname["icaro"]
    collection.insert_one({
        'title': title,
        'message': message,
        'alert': alert,
        'timestamp': datetime.now()
    })


def get_all_data_from_mongo_db() -> dict[str, Any]:
    dbname = get_database()
    collection = dbname["icaro"]

    docs: list[dict[str, Any]] = list(collection.find())

    for d in docs:
        if "_id" in d:
            d["_id"] = str(d["_id"])

    return {"alerts": docs}
