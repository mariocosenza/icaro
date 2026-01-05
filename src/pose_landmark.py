import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from calibration import calibrate_ground_for_stream
from classify_live import classify_live
from drawing import show_single_image
from util_landmarks import GroundCoordinates

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH = "../data/pose_landmarker_heavy.task"
CALIBRATION_JSON = "../data/calibration_result.json"

WidthSpec = Union[int, str]


@dataclass(frozen=True)
class PoseConfig:
    model_path: str = MODEL_PATH
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.6
    min_pose_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    output_segmentation_masks: bool = False


def _resolve_width(width: WidthSpec) -> Optional[int]:
    if width is None:
        return None
    if isinstance(width, int):
        return width
    width_map = {"low": 480, "medium": 640, "high": 1280}
    return width_map.get(width.lower(), 640)


def _open_capture(path: str, webcam: bool) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0 if webcam else path)
    if not cap.isOpened():
        raise IOError(f"Couldn't open {'webcam' if webcam else 'video file'}: {path}")
    return cap


def _get_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        return 30.0
    return float(fps)


def _resize_frame(frame, target_width: Optional[int]):
    if target_width is None:
        return frame
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    new_h = int(h * (target_width / w))
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)


def _mp_image_from_bgr(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


def _timestamp_live(start_time: float) -> int:
    return int((time.time() - start_time) * 1000)


def _timestamp_video(cap: cv2.VideoCapture, frame_index: int, fps: float) -> int:
    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    if pos_msec and pos_msec > 0:
        return int(pos_msec)
    return int(1000.0 * frame_index / fps)


async def _run_stream(
    cap: cv2.VideoCapture,
    stop_event: asyncio.Event,
    target_width: Optional[int],
    frame_stride: int,
    process_frame,
) -> None:
    frame_index = 0
    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_stride > 1 and (frame_index % frame_stride) != 0:
            frame_index += 1
            await asyncio.sleep(0)
            continue

        resized = _resize_frame(frame, target_width)
        mp_image = _mp_image_from_bgr(resized)

        process_frame(mp_image, frame_index)

        frame_index += 1
        await asyncio.sleep(0)


def _ensure_ground_calibrated() -> None:
    if GroundCoordinates.X != 0 or GroundCoordinates.Y != 0 or GroundCoordinates.Z != 0:
        return

    try:
        frame = pd.read_json(CALIBRATION_JSON)
        GroundCoordinates.X = frame["x"].values[0]
        GroundCoordinates.Y = frame["y"].values[0]
        GroundCoordinates.Z = frame["z"].values[0]
        log.info(
            f"Loaded ground coordinates X {GroundCoordinates.X} "
            f"Y {GroundCoordinates.Y} Z {GroundCoordinates.Z}"
        )
    except FileNotFoundError:
        calibrate_ground_for_stream("", webcam=True)


def _make_detector(cfg: PoseConfig, running_mode: vision.RunningMode):
    base_options = python.BaseOptions(model_asset_path=cfg.model_path)

    common = {
        "base_options": base_options,
        "running_mode": running_mode,
        "output_segmentation_masks": cfg.output_segmentation_masks,
        "num_poses": cfg.num_poses,
        "min_pose_detection_confidence": cfg.min_pose_detection_confidence,
        "min_pose_presence_confidence": cfg.min_pose_presence_confidence,
        "min_tracking_confidence": cfg.min_tracking_confidence,
    }

    if running_mode == vision.RunningMode.LIVE_STREAM:
        return vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(**common, result_callback=classify_live)
        )

    return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(**common))


def pose_video_dataset(path: str, resize_width: WidthSpec = "medium", cfg: PoseConfig = PoseConfig()) -> pd.DataFrame:
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=cfg.model_path),
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=cfg.output_segmentation_masks,
        num_poses=cfg.num_poses,
        min_pose_detection_confidence=cfg.min_pose_detection_confidence,
        min_pose_presence_confidence=cfg.min_pose_presence_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )

    target_width = _resolve_width(resize_width)
    cap = _open_capture(path, webcam=False)
    fps = _get_fps(cap)

    frame_index = 0
    rows = []

    with vision.PoseLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame = _resize_frame(frame, target_width)
            mp_image = _mp_image_from_bgr(frame)

            timestamp_ms = int(1000.0 * frame_index / fps)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            rows.append({
                "frame_index": frame_index,
                "timestamp_ms": timestamp_ms,
                "num_poses": len(result.pose_landmarks) if result.pose_landmarks else 0,
                "pose_landmarks": result.pose_landmarks,
                "pose_world_landmarks": result.pose_world_landmarks,
            })

            frame_index += 1

    cap.release()
    return pd.DataFrame(rows)


def pose_point(path: str, running_mode: vision.RunningMode, webcam: bool, quality: WidthSpec = "medium", cfg: PoseConfig = PoseConfig()):
    detector = _make_detector(cfg, running_mode)

    if running_mode == vision.RunningMode.IMAGE:
        image = mp.Image.create_from_file(path)
        result = detector.detect(image)
        show_single_image(image, result)
        detector.close()
        return None

    cap = _open_capture(path, webcam=webcam)
    fps = _get_fps(cap)
    target_width = _resolve_width(quality)
    frame_index = 0
    t0 = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame = _resize_frame(frame, target_width)
            mp_image = _mp_image_from_bgr(frame)

            if running_mode == vision.RunningMode.LIVE_STREAM:
                timestamp_ms = int((time.time() - t0) * 1000)
                detector.detect_async(mp_image, timestamp_ms)
            else:
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_ms = int(pos_msec) if pos_msec and pos_msec > 0 else int(1000.0 * frame_index / fps)

                result = detector.detect_for_video(mp_image, timestamp_ms)
                classify_live(result, mp_image, timestamp_ms)

            frame_index += 1

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def main(running_mode: vision.RunningMode, path: str, quality: WidthSpec = "medium"):
    if running_mode == vision.RunningMode.LIVE_STREAM:
        _ensure_ground_calibrated()
        pose_point(path, vision.RunningMode.LIVE_STREAM, webcam=True, quality=quality)
    else:
        pose_point(path, vision.RunningMode.VIDEO, webcam=False, quality=quality)


async def run_pose_async(
    path: str,
    running_mode: vision.RunningMode,
    quality: WidthSpec = "medium",
    stop_event: Optional[asyncio.Event] = None,
    webcam: Optional[bool] = None,
    frame_stride: int = 1,
    cfg: PoseConfig = PoseConfig(),
):
    if stop_event is None:
        stop_event = asyncio.Event()

    if webcam is None:
        webcam = (running_mode == vision.RunningMode.LIVE_STREAM)

    live_mode = running_mode == vision.RunningMode.LIVE_STREAM
    if live_mode:
        _ensure_ground_calibrated()

    detector = _make_detector(cfg, running_mode)
    cap = _open_capture(path, webcam=bool(webcam))
    fps = _get_fps(cap)
    target_width = _resolve_width(quality)

    start_time = time.time()

    def _process_frame(mp_image: mp.Image, idx: int) -> None:
        if live_mode:
            detector.detect_async(mp_image, _timestamp_live(start_time))
            return
        ts = _timestamp_video(cap, idx, fps)
        result = detector.detect_for_video(mp_image, ts)
        classify_live(result, mp_image, ts)

    try:
        await _run_stream(cap, stop_event, target_width, frame_stride, _process_frame)

    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            detector.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass

        log.info("Pose pipeline completed; resources released.")


if __name__ == "__main__":
    main(vision.RunningMode.VIDEO, path="../data/images/video (10).avi")