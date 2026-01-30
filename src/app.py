import asyncio
import logging
import uuid
import re
from contextlib import asynccontextmanager
from urllib.parse import unquote
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from mediapipe.tasks.python import vision
from starlette.middleware.cors import CORSMiddleware

from classify_live import set_main_loop, reset_detector_state
from mongodb import get_all_data_from_mongo_db
from pose_landmark import run_pose_async
from push_notification import LatestHeartbeat, LatestMovement, set_raspberry_pi_address, get_raspberry_pi_address, stop_buzzer

log = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_VIDEO_PATH = BASE_DIR / "../data/archive/Coffee_room_01/Coffee_room_01/Videos/video (11).avi"

VIDEO_PATH = str(DEFAULT_VIDEO_PATH)
RUNNING_MODE = vision.RunningMode.VIDEO
QUALITY = "high"

_task: Optional[asyncio.Task] = None
_stop_event: Optional[asyncio.Event] = None
_state_lock: asyncio.Lock = asyncio.Lock()


def _is_running() -> bool:
    return _task is not None and not _task.done()


def _on_pipeline_done(t: asyncio.Task) -> None:
    global _task, _stop_event
    try:
        exc = t.exception()
        if exc:
            log.exception("Pose pipeline crashed", exc_info=exc)
        else:
            log.info("Pose pipeline completed.")
    except asyncio.CancelledError:
        log.info("Pose pipeline cancelled.")
    finally:
        _task = None
        _stop_event = None


async def _start_pipeline_if_needed() -> None:
    global _task, _stop_event

    async with _state_lock:
        if _is_running():
            return

        _stop_event = asyncio.Event()
        
        # Reset detector state (tracks, flags) before starting
        reset_detector_state()
        
        _task = asyncio.create_task(
            run_pose_async(
                path=VIDEO_PATH,
                running_mode=RUNNING_MODE,
                quality=QUALITY,
                stop_event=_stop_event,
            )
        )
        _task.add_done_callback(_on_pipeline_done)
        log.info(f"Pose pipeline scheduled. path={VIDEO_PATH} mode={RUNNING_MODE.name}")


async def _stop_pipeline_if_running() -> None:
    global _task, _stop_event

    async with _state_lock:
        if not _is_running():
            _task = None
            _stop_event = None
            return

        assert _stop_event is not None
        _stop_event.set()

        t = _task

    try:
        await t
    except Exception:
        log.exception("Pose pipeline crashed while stopping")
    finally:
        _task = None
        _stop_event = None


async def _restart_pipeline() -> None:
    await _stop_pipeline_if_running()
    await _start_pipeline_if_needed()


@asynccontextmanager
async def lifespan(app: FastAPI):
    set_main_loop(asyncio.get_running_loop())
    yield
    await _stop_pipeline_if_running()


app = FastAPI(lifespan=lifespan)

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
    log.info(f"Heartbeat: {heartbeat}")
    return {"ok": True}


@app.post("/api/v1/monitor/{data:path}")
def post_movement(data: str):
    # Regex to capture three floating point numbers (including negatives) separated by hyphens
    # Expected format: x-y-z where x, y, z can be negative e.g. 4.587--0.807-8.367
    match = re.match(r"^([+-]?\d*\.?\d+)-([+-]?\d*\.?\d+)-([+-]?\d*\.?\d+)$", data)
    if not match:
         raise HTTPException(status_code=422, detail="Invalid format. Expected x-y-z")

    x = float(match.group(1))
    y = float(match.group(2))
    z = float(match.group(3))

    LatestMovement.X = x
    LatestMovement.Y = y
    LatestMovement.Z = z
    log.info(f"Movement: {x}-{y}-{z}")
    return {"ok": True}


@app.get("/api/v1/status")
async def status():
    return {
        "ok": True,
        "running": _is_running(),
        "active_video_path": VIDEO_PATH,
        "running_mode": RUNNING_MODE.name,
    }


@app.post("/api/v1/start")
async def start_pipeline():
    await _start_pipeline_if_needed()
    return {
        "ok": True,
        "started": True,
        "running": _is_running(),
        "active_video_path": VIDEO_PATH,
        "running_mode": RUNNING_MODE.name,
    }


@app.post("/api/v1/stop")
async def stop_pipeline():
    await _stop_pipeline_if_running()
    # Resetting state on stop ensures next start is clean
    reset_detector_state()
    return {"ok": True, "running": _is_running()}


@app.post("/api/v1/stop/buzzer")
async def stop_buzzer_endpoint():
    """
    Stops the buzzer on the Raspberry Pi.
    """
    await stop_buzzer()
    return {"ok": True}


@app.post("/api/v1/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".avi", ".mp4", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    saved_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = UPLOAD_DIR / saved_name

    try:
        with saved_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    global VIDEO_PATH
    VIDEO_PATH = str(saved_path)

    if _is_running():
        await _restart_pipeline()

    return {
        "ok": True,
        "saved_path": str(saved_path),
        "active_video_path": VIDEO_PATH,
        "running_mode": RUNNING_MODE.name,
        "running": _is_running(),
    }


@app.patch("/api/v1/running-mode/live-stream")
async def set_running_mode_live_stream():
    global RUNNING_MODE
    RUNNING_MODE = vision.RunningMode.LIVE_STREAM

    if _is_running():
        await _restart_pipeline()

    return {
        "ok": True,
        "running_mode": RUNNING_MODE.name,
        "active_video_path": VIDEO_PATH,
        "running": _is_running(),
    }


@app.patch("/api/v1/running-mode/video")
async def set_running_mode_video():
    global RUNNING_MODE
    RUNNING_MODE = vision.RunningMode.VIDEO

    if _is_running():
        await _restart_pipeline()

    return {
        "ok": True,
        "running_mode": RUNNING_MODE.name,
        "active_video_path": VIDEO_PATH,
        "running": _is_running(),
    }


@app.patch("/api/v1/quality/{quality_level}")
async def set_quality(quality_level: str):
    global QUALITY
    
    if quality_level not in {"low", "medium", "high"}:
         raise HTTPException(status_code=400, detail="Invalid quality. Options: low, medium, high")

    QUALITY = quality_level
    log.info(f"Quality set to {QUALITY}")

    if _is_running():
        await _restart_pipeline()

    return {
        "ok": True,
        "quality": QUALITY,
        "active_video_path": VIDEO_PATH,
        "running": _is_running(),
    }


@app.patch("/api/v1/raspberry-ip")
async def update_raspberry_pi_address_endpoint(address: str):
    """
    Update the Raspberry Pi Address (IP or Domain).
    Query parameter 'address' is used.
    """
    if not address:
         raise HTTPException(status_code=400, detail="Address is required")
    
    decoded_address = unquote(address)
    set_raspberry_pi_address(decoded_address)
    log.info(f"Raspberry Pi address updated to: {decoded_address} (original: {address})")
    
    return {
        "ok": True,
        "raspberry_pi_address": get_raspberry_pi_address()
    }
