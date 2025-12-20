import logging
import time
import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from drawing import show_single_image
from calibration import calibrate_ground_for_stream
from classify_live import classify_live
from util_landmarks import GroundCoordinates

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _open_capture(path: str, webcam: bool) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0 if webcam else path)
    if not cap.isOpened():
        raise IOError(f"Couldn't open {'webcam' if webcam else 'video file'}: {path}")
    return cap


def pose_video_dataset(path: str, resize_width="medium") -> pd.DataFrame:
    base_options = python.BaseOptions(model_asset_path="../data/pose_landmarker_heavy.task")

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    width_map = {
        "low": 480,
        "medium": 640,
        "high": 1280
    }

    if isinstance(resize_width, str):
        target_width = width_map.get(resize_width.lower(), 640)
    else:
        target_width = resize_width

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Couldn't open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    frame_index = 0
    rows = []

    with vision.PoseLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if target_width is not None and frame.shape[1] > target_width:
                h, w = frame.shape[:2]
                new_h = int(h * (target_width / w))
                frame = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

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


def pose_stream(
    path: str,
    detector,
    running_mode: vision.RunningMode,
    webcam: bool = False,
    visualize: bool = False,
    frame_stride: int = 1,  # LATEST CHANGE: do not skip frames to reduce jerk
    resize_width: str = "medium"
) -> pd.DataFrame:
    cap = cv2.VideoCapture(0 if webcam else path)
    if not cap.isOpened():
        raise IOError(f"Couldn't open {'webcam' if webcam else 'video file'}: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    frame_index = 0
    t0 = time.time()
    rows = []

    width_map = {
        "low": 480,
        "medium": 640,
        "high": 1280
    }
    resize_width = width_map.get(resize_width.lower(), resize_width)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_stride > 1 and (frame_index % frame_stride) != 0:
            frame_index += 1
            continue

        if resize_width is not None and frame.shape[1] > resize_width:
            h, w = frame.shape[:2]
            new_h = int(h * (resize_width / w))
            frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if running_mode == vision.RunningMode.LIVE_STREAM:
            timestamp_ms = int((time.time() - t0) * 1000)
            detector.detect_async(mp_image, timestamp_ms)
        else:
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec is not None and pos_msec > 0:
                timestamp_ms = int(pos_msec)
            else:
                timestamp_ms = int(1000.0 * frame_index / fps)

            result = detector.detect_for_video(mp_image, timestamp_ms)
            classify_live(result, mp_image, timestamp_ms)

            if visualize:
                from drawing import draw_pose_points
                draw_pose_points(mp_image, result, wait_ms=1)

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


def pose_point(path: str, running_mode: vision.RunningMode, webcam: bool = True, quality="medium"):
    base_options = python.BaseOptions(model_asset_path="../data/pose_landmarker_heavy.task")

    common = dict(
        base_options=base_options,
        running_mode=running_mode,
        output_segmentation_masks=False,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    if running_mode == vision.RunningMode.LIVE_STREAM:
        options = vision.PoseLandmarkerOptions(
            **common,
            result_callback=classify_live,
        )
    else:
        options = vision.PoseLandmarkerOptions(**common)

    detector = vision.PoseLandmarker.create_from_options(options)

    if running_mode == vision.RunningMode.IMAGE:
        image = mp.Image.create_from_file(path)
        result = detector.detect(image)
        show_single_image(image, result)
        return None

    return pose_stream(path, detector, running_mode=running_mode, webcam=webcam, resize_width=quality)


def main(running_mode: vision.RunningMode, path: str = "", quality="medium"):
    if running_mode == vision.RunningMode.LIVE_STREAM:
        if GroundCoordinates.X == 0 and GroundCoordinates.Y == 0 and GroundCoordinates.Z == 0:
            try:
                frame = pd.read_json("../data/calibration_result.json")
                logging.info('Loaded calibration frame')
                GroundCoordinates.X = frame["x"].values[0]
                GroundCoordinates.Y = frame["y"].values[0]
                GroundCoordinates.Z = frame["z"].values[0]
                logging.info(f'Loaded ground coordinates X {GroundCoordinates.X} Y {GroundCoordinates.Y} Z {GroundCoordinates.Z}')
            except FileNotFoundError:
                calibrate_ground_for_stream("", webcam=True)
        pose_point("", vision.RunningMode.LIVE_STREAM, webcam=True, quality=quality)
    else:
        pose_point(path, vision.RunningMode.VIDEO, webcam=False)


if __name__ == "__main__":
    # DEMO
    main(vision.RunningMode.VIDEO, path="../data/images/video (10).avi")
