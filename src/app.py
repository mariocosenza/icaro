from fastapi import FastAPI

from src.mongodb import get_all_data_from_mongo_db

app = FastAPI()
@app.get("/api/v1/alerts")
def get_alerts():
    return get_all_data_from_mongo_db()