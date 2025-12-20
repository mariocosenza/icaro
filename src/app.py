from fastapi import FastAPI
from mediapipe.tasks.python import vision

from mongodb import get_all_data_from_mongo_db
from pose_landmark import main

app = FastAPI()
main(vision.RunningMode.LIVE_STREAM)
@app.get("/api/v1/alerts")
def get_alerts():
    return get_all_data_from_mongo_db()
