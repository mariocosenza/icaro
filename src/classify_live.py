from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult

    MP_AVAILABLE = True
except ModuleNotFoundError:
    mp = None
    PoseLandmarkerResult = Any
    MP_AVAILABLE = False

from drawing import draw_pose_points
from live_fall_detector import LiveManDownDetector, DetectorConfig, MultiPersonDetector
from mongodb import insert_message_mongo_db
from pipeline_horizontal_classification import load_models
from push_notification import (
    LatestHeartbeat,
    LatestMovement,
    send_monitoring_notification,
    send_push_notification,
    send_push_notification_heartbeat,
)
from util_landmarks import BodyLandmark, GroundCoordinates

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

INHIBIT_MS = 120_000

_detector_enabled = True
_detector_disabled_until_ms = 0

_loop: asyncio.AbstractEventLoop | None = None


def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


def _schedule(coro) -> None:
    if _loop is None:
        return
    _loop.call_soon_threadsafe(asyncio.create_task, coro)


def _lm_to_dict(lm) -> dict:
    return {
        "x": float(lm.x),
        "y": float(lm.y),
        "z": float(getattr(lm, "z", 0.0)),
        "visibility": float(getattr(lm, "visibility", 0.0)),
        "presence": float(getattr(lm, "presence", 0.0)),
    }


def _result_to_frame_landmarks(result: PoseLandmarkerResult):
    if not result or not result.pose_landmarks:
        return None

    poses = []
    for pose in result.pose_landmarks:
        if pose and len(pose) >= 33:
            poses.append([_lm_to_dict(lm) for lm in pose])

    return poses or None


def _now_str() -> str:
    return datetime.now().strftime("%A, %B %d, %Y %H:%M")


def _is_no_movement() -> bool:
    return (
            LatestMovement.X < 0.5 and
            LatestMovement.Y < 0.8 and
            LatestMovement.Z < 0.8
    )


def _is_abnormal_hr() -> bool:
    return LatestHeartbeat.BPM < 30 or LatestHeartbeat.BPM > 180


def _can_reenable_detector(result: PoseLandmarkerResult, timestamp_ms: int) -> bool:
    if timestamp_ms < _detector_disabled_until_ms:
        return False
    if not result.pose_landmarks or not result.pose_landmarks[0]:
        return False
    return result.pose_landmarks[0][BodyLandmark.LEFT_SHOULDER].y > GroundCoordinates.Y


def _notify_mongo(title: str, body: str, alert: bool = True) -> None:
    _schedule(insert_message_mongo_db(title, body, alert=alert))


def _notify_push(title: str, body: str) -> None:
    _schedule(send_push_notification(title, body))


def _notify_monitoring() -> None:
    _schedule(send_monitoring_notification())


def _notify_heartbeat() -> None:
    _schedule(send_push_notification_heartbeat())


def _horizontal_trigger(out: dict) -> bool:
    return bool(out.get("horizontal_event")) and out.get("horizontal_prob", 0.0) > 0.70


def _fall_trigger(out: dict) -> bool:
    return bool(out.get("fall_event")) and out.get("fall_prob", 0.0) > 0.98


def _maybe_notify_heartbeat(no_movement: bool, abnormal_hr: bool) -> None:
    if (not LatestHeartbeat.NOTIED_FALL) and abnormal_hr and no_movement:
        LatestHeartbeat.NOTIED_FALL = True
        _notify_heartbeat()


def _handle_disabled_state(
        result: PoseLandmarkerResult,
        frame_landmarks,
        timestamp_ms: int,
        no_movement: bool,
        abnormal_hr: bool,
) -> bool:
    global _detector_enabled, _detector_disabled_until_ms

    _maybe_notify_heartbeat(no_movement, abnormal_hr)

    if _can_reenable_detector(result, timestamp_ms):
        _detector_enabled = True
        LatestHeartbeat.NOTIED_FALL = False
        return False

    _detector.update(frame_landmarks)
    return True


def _handle_horizontal_event(out: dict, timestamp_ms: int, no_movement: bool, abnormal_hr: bool) -> bool:
    if not _horizontal_trigger(out):
        return False

    if not LatestHeartbeat.NOTIED_FALL:
        _notify_monitoring()

    if abnormal_hr and no_movement:
        _notify_push("Man Down Detected", "A man is down please check your app!")
        _notify_mongo(
            "Man Down Detected",
            f"Man down detected at timestamp: {_now_str()}ms",
            alert=True,
        )
        log.info(
            f"[{timestamp_ms}ms] MAN DOWN (HORIZONTAL) prob={out.get('horizontal_prob', 0.0):.3f} "
            f"hits={out.get('horizontal_hits', 0)}"
        )
        return True

    return False


def _handle_fall_event(out: dict, timestamp_ms: int, abnormal_hr: bool) -> bool:
    if not _fall_trigger(out):
        return False

    log.info(
        f"[{timestamp_ms}ms] FALL prob={out.get('fall_prob', 0.0):.3f} hits={out.get('fall_hits', 0)}"
    )
    _notify_push("Man Fall Detected", "A fall was detected please check your app!")
    _notify_mongo(
        "Fall Detected",
        f"Fall detected at timestamp: {_now_str()}ms",
        alert=True,
    )

    if abnormal_hr:
        LatestHeartbeat.NOTIED_FALL = True
        _notify_heartbeat()

    return True


def classify_live(result: PoseLandmarkerResult, image: mp.Image, timestamp_ms: int) -> None:
    global _detector_enabled, _detector_disabled_until_ms, _detector

    draw_pose_points(image, result, wait_ms=1)

    if _detector is None:
        return

    frame_landmarks = _result_to_frame_landmarks(result)
    no_movement = _is_no_movement()
    abnormal_hr = _is_abnormal_hr()

    if not _detector_enabled and _handle_disabled_state(result, frame_landmarks, timestamp_ms, no_movement,
                                                        abnormal_hr):
        return

    outputs = _detector.update(frame_landmarks)
    inhibited = False

    for entry in outputs:
        out = entry.get("output", {})
        if not out.get("ready", False):
            continue

        inhibited = _handle_horizontal_event(out, timestamp_ms, no_movement, abnormal_hr) or inhibited
        inhibited = _handle_fall_event(out, timestamp_ms, abnormal_hr) or inhibited

    if inhibited:
        _detector_enabled = False
        _detector_disabled_until_ms = int(timestamp_ms + INHIBIT_MS)
        log.info(f"[{timestamp_ms}ms] DETECTOR INHIBITED until {_detector_disabled_until_ms}ms")


bundle = load_models("../data/icaro_models.joblib")


def _make_detector_instance() -> LiveManDownDetector:
    return LiveManDownDetector(
        BodyLandmark,
        fall_model=bundle["fall_model"],
        horizontal_model=bundle["horizontal_model"],
        window=bundle["cfg"]["window"],
        config=DetectorConfig(
            fall_min_vis_point=bundle["cfg"]["fall_min_vis_point"],
            fall_min_pres_point=bundle["cfg"]["fall_min_pres_point"],
            fall_min_required_core_points=bundle["cfg"]["fall_min_required_core_points"],
            horizontal_min_quality=bundle["cfg"]["horizontal_min_quality"],
            horizontal_min_good_keypoints=bundle["cfg"]["horizontal_min_good_keypoints"],
            fall_threshold=0.70,
            horizontal_threshold=0.60,
            consecutive_fall=5,
            consecutive_horizontal=3,
            reset_on_invalid=False,
            min_window_quality="high",
        ),
    )


_detector = MultiPersonDetector(
    BodyLandmark,
    _make_detector_instance,
    distance_threshold=0.12,
    max_tracks=8,
)
