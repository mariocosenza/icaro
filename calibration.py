import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerOptions, GestureRecognizer
from mediapipe.tasks.python.vision.gesture_recognizer_result import GestureRecognizerResult
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions, PoseLandmarker, PoseLandmarkerResult

from util_landmarks import GroundCoordinates, BodyLandmark

THUMB_UP = True
GROUND = {
    'x': 0,
    'y': 0,
    'z': 0
}

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print(f'gesture recognition result: {result.gestures}')
    global THUMB_UP
    if not THUMB_UP:
        THUMB_UP = True

def pose_result_callback(
    result: PoseLandmarkerResult,
    image: mp.Image,
    timestamp_ms: int,
) -> None:
    GroundCoordinates.X = result[0][BodyLandmark.LEFT_FOOT_INDEX][0]
    GroundCoordinates.Y = result[0][BodyLandmark.LEFT_FOOT_INDEX][1]
    GroundCoordinates.Z = result[0][BodyLandmark.LEFT_FOOT_INDEX][2]




def calibrate_ground_for_stream(path:str, webcam=True):
    print('Starting calibration...')
    base_options = BaseOptions(model_asset_path='./data/pose_landmarker_heavy.task')
    options_pose = PoseLandmarkerOptions(
        base_options=base_options, running_mode=RunningMode.LIVE_STREAM,
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

        while cap.isOpened():
            ret, frame = cap.read()
            mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            recognizer.recognize_async(mp_image, frame)
            if THUMB_UP:
                with PoseLandmarker.create_from_options(options=options_pose) as landmark:
                    landmark.recognize_async(mp_image, frame)
        cap.release()


    print('Thumbs UP!')
