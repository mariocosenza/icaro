import os
from pymongo import MongoClient

try:
    CONNECTION_STRING = os.getenv("CONNECTION_STRING")
except ImportError:
    exit("CONNECTION_STRING not found in environment variables")


def get_database():
    client = MongoClient(CONNECTION_STRING)
    return client['icaro']


def insert_message_mongo_db(title:str, message:str, alert:bool) -> None:
    dbname = get_database()
    print("Inserting aggregated data into MongoDB...")
    collection_name = dbname["icaro"]
    collection_name.insert({
        'title': title,
        'message': message,
        alert: alert
    })

def get_all_data_from_mongo_db() -> list:
    dbname = get_database()
    collection_name = dbname["icaro"]
    return list(collection_name.find())

