import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerOptions, GestureRecognizer
from mediapipe.tasks.python.vision.gesture_recognizer_result import GestureRecognizerResult
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions, PoseLandmarker, PoseLandmarkerResult

from drawing import draw_pose_points
from util_landmarks import GroundCoordinates, BodyLandmark

THUMB_UP = False
CALIBRATED = False
GROUND = {
    'x': 0,
    'y': 0,
    'z': 0
}

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global THUMB_UP
    if result.gestures and result.gestures[0] and result.gestures[0][0].category_name == 'Thumb_Up':
        print('Thumb_Up')
        THUMB_UP = True

def pose_result_callback(
    result: PoseLandmarkerResult,
    image: mp.Image,
    timestamp_ms: int,
) -> None:
     draw_pose_points(image, result, wait_ms=1)
     global GROUND
     global CALIBRATED
     if result.pose_world_landmarks:
        left_knee = result.pose_world_landmarks[0][BodyLandmark.LEFT_KNEE]
        right_knee = result.pose_world_landmarks[0][BodyLandmark.RIGHT_KNEE]
        if left_knee.presence > 0.65 and left_knee.visibility > 0.85 and right_knee.presence > 0.85 and right_knee.visibility > 0.66:
            GROUND = {
                'x': (left_knee.x + right_knee.x) /2,
                'y': (left_knee.y + right_knee.y) /2,
                'z': (left_knee.z + right_knee.z) /2,
            }
            CALIBRATED = True



def calibrate_ground_for_stream(path:str, webcam=True):
    print('Starting calibration...')
    base_options = BaseOptions(model_asset_path='./data/pose_landmarker_heavy.task')

    options_pose = PoseLandmarkerOptions(
        base_options=base_options, running_mode=RunningMode.LIVE_STREAM,
        result_callback=pose_result_callback,
        output_segmentation_masks=True)

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path='./data/gesture_recognizer.task'),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=print_result)

    with GestureRecognizer.create_from_options(options=options) as recognizer:
        if not webcam:
            cap = cv2.VideoCapture(path)
        else:
            cap = cv2.VideoCapture(0)

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            recognizer.recognize_async(mp_image, frame_index)
            frame_index += 1
            if THUMB_UP:
                print('Starting knee detection...')
                break
    cap.release()

    with PoseLandmarker.create_from_options(options=options_pose) as landmark:
        print('DO NOT MOVE')
        if not webcam:
            cap = cv2.VideoCapture(path)
        else:
            cap = cv2.VideoCapture(0)
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmark.detect_async(mp_image, frame_index)
            frame_index += 1
            if CALIBRATED:
                print('Calibration complete.')
                GroundCoordinates.X = GROUND['x']
                GroundCoordinates.Y = GROUND['y']
                GroundCoordinates.Z = GROUND['z']
                data_frame = pd.DataFrame([GROUND])
                data_frame.to_json('./data/calibration_result.json')
                break


calibrate_ground_for_stream('')
