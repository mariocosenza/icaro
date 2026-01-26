from __future__ import annotations

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ModuleNotFoundError:
    cv2 = None
    CV2_AVAILABLE = False

try:
    import mediapipe as mp

    MP_AVAILABLE = True
except ModuleNotFoundError:
    mp = None
    MP_AVAILABLE = False

# MediaPipe Pose connections as index pairs (no mp.solutions needed)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (17, 19), (18, 20),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
    (27, 31), (28, 32),
]


def _fit_to_screen(bgr: np.ndarray, max_w: int = 1280, max_h: int = 720) -> np.ndarray:
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is required for drawing.")
    h, w = bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)  # never upscale
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return bgr


def draw_pose_points(
        mp_image: mp.Image,
        detection_result,
        window_name: str = "Pose",
        max_w: int = 1280,
        max_h: int = 720,
        wait_ms: int = 1,  # 0 = block (single image), 1/10/33 = video/live
        close_key: str = "q",  # press to close window
) -> bool:
    """
    Draw pose + show it in a resizable window.
    Works for: single image, video loop, live stream callback.

    Returns:
        True  -> continue
        False -> user requested close (pressed close_key or window closed)
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is required for drawing.")
    if not MP_AVAILABLE:
        raise RuntimeError("MediaPipe is required for drawing.")
    # mp.Image -> numpy RGB (HxWx3)
    rgb = mp_image.numpy_view()
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).astype(np.uint8)

    annotated = rgb.copy()
    h, w = annotated.shape[:2]

    for pose in getattr(detection_result, "pose_landmarks", []) or []:
        pts = []
        for lm in pose:
            x, y = int(lm.x * w), int(lm.y * h)
            pts.append((x, y))
            cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)

        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(annotated, pts[a], pts[b], (0, 255, 0), 2)

    # Display via OpenCV (BGR) scaled to fit screen
    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    bgr = _fit_to_screen(bgr, max_w=max_w, max_h=max_h)

    # Create window once (cheap to call repeatedly; OpenCV reuses it)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, bgr)

    # If user closes the window manually
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        return False

    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord(close_key):
        cv2.destroyWindow(window_name)
        return False

    return True


# --- Convenience wrappers ---

def show_single_image(mp_image: mp.Image, detection_result, window_name: str = "Pose") -> None:
    draw_pose_points(mp_image, detection_result, window_name=window_name, wait_ms=0)


def show_video_loop(
        frames_iter,
        get_result_fn,
        window_name: str = "Pose",
        max_w: int = 1280,
        max_h: int = 720,
        wait_ms: int = 1,
) -> None:
    """
    Generic loop for video/webcam:
      - frames_iter yields mp.Image (or anything you convert to mp.Image)
      - get_result_fn(mp_image) returns detection_result
    """
    for mp_image in frames_iter:
        result = get_result_fn(mp_image)
        if not draw_pose_points(mp_image, result, window_name, max_w, max_h, wait_ms=wait_ms):
            break
    cv2.destroyAllWindows()
