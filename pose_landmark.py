import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerResult
from drawing import draw_pose_points, show_single_image


def pose_result_callback(
    result: PoseLandmarkerResult,
    image: mp.Image,
    timestamp_ms: int,
) -> None:
    draw_pose_points(image, result, wait_ms=0)


def _pose_stream(path: str, detector: PoseLandmarker, webcam=True, live = True):
    if not webcam:
        video_capture = cv2.VideoCapture(path)
    else:
        video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise IOError("Couldn't open webcam or video")

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(1000 * frame_index / fps)
        if live:
            detector.detect_async(mp_image, frame_timestamp_ms)
        else:
            detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
            draw_pose_points(mp_image, detection_result, wait_ms=1)
        frame_index += 1

    video_capture.release()

def pose_point(path:str, running_mode: python.vision.RunningMode, webcam=True):
    base_options = python.BaseOptions(model_asset_path='./data/pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=running_mode, result_callback=pose_result_callback,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    if not  running_mode == python.vision.RunningMode.IMAGE:
        _pose_stream(path, detector, webcam=webcam, live=running_mode == python.vision.RunningMode.LIVE_STREAM)
    else:
        detector = vision.PoseLandmarker.create_from_options(options)
        image = mp.Image.create_from_file(path)
        result = detector.detect(image)
        show_single_image(image, result)




if __name__ == '__main__':
    pose_point('./data/images/video (10).avi', python.vision.RunningMode.LIVE_STREAM)