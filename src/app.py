from fastapi import FastAPI, logger
from mediapipe.tasks.python import vision
from starlette.middleware.cors import CORSMiddleware

from mongodb import get_all_data_from_mongo_db
from pose_landmark import main
from push_notification import LatestHeartbeat
from src.push_notification import LatestMovement

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|192\.168\.1\.(?:[0-9]{1,3}))(?::\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/api/v1/alerts")
def get_alerts():
    return get_all_data_from_mongo_db()

@app.post("/api/v1/measure/{heartbeat}")
def post_heartbeat(heartbeat: int):
    LatestHeartbeat.BPM = heartbeat
    logger.logger.log(level=logger.logger.INFO, msg=f"Heartbeat: {heartbeat}")

@app.post("/api/v1/monitor/{x}-{y}-{z}")
def post_heartbeat(x: float, y: float, z: float):
    LatestMovement.X = x
    LatestMovement.Y = y
    LatestMovement.Z = z
    logger.logger.log(level=logger.logger.INFO, msg=f"Movement: {x}-{y}-{z}")

main(path='', running_mode=vision.RunningMode.LIVE_STREAM, quality="high")
if __name__ == "__main__":
    main(path='', running_mode=vision.RunningMode.LIVE_STREAM, quality="high")