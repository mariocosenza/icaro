import mediapipe as mp
from src.drawing import draw_pose_points
from src.mongodb import insert_message_mongo_db
from src.push_notification import send_push_notification
from src.util_landmarks import GroundCoordinates
from util_landmarks import BodyLandmark
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from src.live_fall_detector import LiveManDownDetector
from src.pipeline_horizontal_classification import load_models

DETECTOR_ENABLED = True


INHIBIT_MS = 60_000
DETECTOR_DISABLED_UNTIL_MS = 0

def _lm_to_dict(lm) -> dict:
    return {
        "x": float(lm.x),
        "y": float(lm.y),
        "z": float(getattr(lm, "z", 0.0)),
        "visibility": float(getattr(lm, "visibility", 0.0)),
        "presence": float(getattr(lm, "presence", 0.0)),
    }

def _result_to_frame_landmarks(result: PoseLandmarkerResult):
    if result is None or not result.pose_landmarks:
        return None
    poses = []
    for pose in result.pose_landmarks:
        if pose is None or len(pose) < 33:
            continue
        poses.append([_lm_to_dict(lm) for lm in pose])
    return poses if poses else None

def classify_live(result: PoseLandmarkerResult, image: mp.Image, timestamp_ms: int) -> None:
    draw_pose_points(image, result, wait_ms=1)

    global DETECTOR_ENABLED, DETECTOR_DISABLED_UNTIL_MS, _detector
    if _detector is None:
        return

    if not DETECTOR_ENABLED:
        if timestamp_ms >= DETECTOR_DISABLED_UNTIL_MS and result.pose_landmarks[0][BodyLandmark.LEFT_SHOULDER].y > GroundCoordinates.Y:
            DETECTOR_ENABLED = True
        else:
            frame_landmarks = _result_to_frame_landmarks(result)
            _detector.update(frame_landmarks)
            return

    frame_landmarks = _result_to_frame_landmarks(result)
    out = _detector.update(frame_landmarks)

    if not out.get("ready", False):
        return

    if out["fall_event"]:
        print(f"[{timestamp_ms}ms] FALL prob={out['fall_prob']:.3f} hits={out['fall_hits']}")
        if out['fall_prob'] > 0.92:
            send_push_notification('Man Fall Detected', 'A fall was detected please check your app!')
            insert_message_mongo_db("Fall Detected", "Fall detected at timestamp: " + str(timestamp_ms) + "ms", alert=True)

            DETECTOR_ENABLED = False
            DETECTOR_DISABLED_UNTIL_MS = timestamp_ms + INHIBIT_MS
            print(f"[{timestamp_ms}ms] DETECTOR INHIBITED until {DETECTOR_DISABLED_UNTIL_MS}ms")

        if out["horizontal_event"]:
            if out['fall_prob'] > 0.80:
                send_push_notification('Man Down Detected', 'A man is down please check your app!')
                insert_message_mongo_db("Man Down Detected", "Man down detected at timestamp: " + str(timestamp_ms) + "ms", alert=True)
                print(f"[{timestamp_ms}ms] MAN DOWN (HORIZONTAL) prob={out['horizontal_prob']:.3f} hits={out['horizontal_hits']}")

bundle = load_models("../data/icaro_models.joblib")

_detector = LiveManDownDetector(
        BodyLandmark,
        fall_model=bundle["fall_model"],
        horizontal_model=bundle["horizontal_model"],
        window=bundle["cfg"]["window"],
        fall_min_vis_point=bundle["cfg"]["fall_min_vis_point"],
        fall_min_pres_point=bundle["cfg"]["fall_min_pres_point"],
        fall_min_required_core_points=bundle["cfg"]["fall_min_required_core_points"],
        horizontal_min_quality=bundle["cfg"]["horizontal_min_quality"],
        horizontal_min_good_keypoints=bundle["cfg"]["horizontal_min_good_keypoints"],
        fall_threshold=0.60,
        horizontal_threshold=0.60,
        consecutive_fall=3,
        consecutive_horizontal=3,
        reset_on_invalid=False,
)

