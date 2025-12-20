import logging
import os

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import pandas as pd

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerOptions, GestureRecognizer
from mediapipe.tasks.python.vision.gesture_recognizer_result import GestureRecognizerResult
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions, PoseLandmarker, PoseLandmarkerResult

from drawing import draw_pose_points
from util_landmarks import GroundCoordinates, BodyLandmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

THUMB_UP = False
CALIBRATED = False
GROUND = {"x": 0, "y": 0, "z": 0}


def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global THUMB_UP
    if result.gestures and result.gestures[0] and result.gestures[0][0].category_name == "Thumb_Up":
        logging.info("Thumb_Up")
        THUMB_UP = True


def pose_result_callback(result: PoseLandmarkerResult, image: mp.Image, timestamp_ms: int) -> None:
    global GROUND, CALIBRATED

    draw_pose_points(image, result, wait_ms=1)

    if result.pose_world_landmarks:
        left_knee = result.pose_world_landmarks[0][BodyLandmark.LEFT_KNEE]
        right_knee = result.pose_world_landmarks[0][BodyLandmark.RIGHT_KNEE]

        if (
            left_knee.presence > 0.75 and left_knee.visibility > 0.70 and
            right_knee.presence > 0.75 and right_knee.visibility > 0.70
        ):
            GROUND = {
                "x": (left_knee.x + right_knee.x) / 2,
                "y": (left_knee.y + right_knee.y) / 2,
                "z": (left_knee.z + right_knee.z) / 2,
            }
            CALIBRATED = True


def _timestamp_ms(cap: cv2.VideoCapture, webcam: bool, frame_index: int) -> int:
    if not webcam:
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if ms and ms > 0:
            return int(ms)
    return int(frame_index * (1000 / 30))


def calibrate_ground_for_stream(path: str, webcam: bool = True):
    global THUMB_UP, CALIBRATED

    logging.info("Starting calibration...")

    options_pose = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="../data/pose_landmarker_heavy.task"),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=pose_result_callback,
        output_segmentation_masks=False,
    )

    options_gesture = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path="../data/gesture_recognizer.task"),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=print_result,
    )

    with GestureRecognizer.create_from_options(options=options_gesture) as recognizer:
        cap = cv2.VideoCapture(0 if webcam else path)
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            ts = _timestamp_ms(cap, webcam, frame_index)
            recognizer.recognize_async(mp_image, ts)

            frame_index += 1
            if THUMB_UP:
                logging.info("Starting knee detection...")
                break

        cap.release()

    with PoseLandmarker.create_from_options(options=options_pose) as landmark:
        logging.warning("DO NOT MOVE")
        cap = cv2.VideoCapture(0 if webcam else path)
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            ts = _timestamp_ms(cap, webcam, frame_index)
            landmark.detect_async(mp_image, ts)

            frame_index += 1
            if CALIBRATED:
                logging.info("Calibration complete.")
                GroundCoordinates.X = GROUND["x"]
                GroundCoordinates.Y = GROUND["y"]
                GroundCoordinates.Z = GROUND["z"]
                pd.DataFrame([GROUND]).to_json("../data/calibration_result.json")
                break

        cap.release()