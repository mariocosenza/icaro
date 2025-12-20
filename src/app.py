from fastapi import FastAPI, logger
from mediapipe.tasks.python import vision

from mongodb import get_all_data_from_mongo_db
from pose_landmark import main
from push_notification import LatestHeartbeat

app = FastAPI()
main(vision.RunningMode.LIVE_STREAM, quality="high")
@app.get("/api/v1/alerts")
def get_alerts():
    return get_all_data_from_mongo_db()

@app.post("/api/v1/measure/{heartbeat}")
def post_heartbeat(heartbeat: int):
    LatestHeartbeat.BPM = heartbeat
    logger.logger.log(level=logger.logger.INFO, msg=f"Heartbeat: {heartbeat}")
