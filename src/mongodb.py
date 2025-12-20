import logging
import os
import sys
from pymongo import MongoClient
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONNECTION_STRING = os.getenv("CONNECTION_STRING")
if not CONNECTION_STRING:
    sys.exit("CONNECTION_STRING not found in environment variables")


def get_database():
    client = MongoClient(CONNECTION_STRING)
    return client['icaro']


async def insert_message_mongo_db(title: str, message: str, alert: bool) -> None:
    dbname = get_database()
    logging.log(level=logging.INFO, msg='Inserting aggregated data into MongoDB...')
    collection = dbname["icaro"]
    collection.insert_one({
        'title': title,
        'message': message,
        'alert': alert
    })


def get_all_data_from_mongo_db() -> list:
    dbname = get_database()
    collection = dbname["icaro"]
    return list(collection.find())