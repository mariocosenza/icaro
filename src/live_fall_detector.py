from collections import deque
from typing import Union, Dict, Optional, Any

import numpy as np
from sklearn.pipeline import Pipeline

from src.pipeline_horizontal_classification import _proba_pos, window_vector, select_best_pose, Pose33, PoseCell, \
    extract_frame_features


class LiveManDownDetector:
    def __init__(
        self,
        BodyLandmark,
        *,
        fall_model: Pipeline,
        horizontal_model: Pipeline,
        window: int,
        min_quality: float = 0.20,
        min_good_keypoints: int = 4,
        fall_threshold: float = 0.60,
        horizontal_threshold: float = 0.60,
        consecutive_fall: int = 3,
        consecutive_horizontal: int = 3,
        reset_on_invalid: bool = False,
    ):
        self.BodyLandmark = BodyLandmark
        self.fall_model = fall_model
        self.horizontal_model = horizontal_model
        self.window = int(window)
        self.min_quality = float(min_quality)
        self.min_good_keypoints = int(min_good_keypoints)
        self.fall_threshold = float(fall_threshold)
        self.horizontal_threshold = float(horizontal_threshold)
        self.consecutive_fall = int(consecutive_fall)
        self.consecutive_horizontal = int(consecutive_horizontal)
        self.reset_on_invalid = bool(reset_on_invalid)

        self._feat_buf = deque(maxlen=self.window)
        self._qual_buf = deque(maxlen=self.window)
        self._fall_hits = 0
        self._horiz_hits = 0

    def reset(self):
        self._feat_buf.clear()
        self._qual_buf.clear()
        self._fall_hits = 0
        self._horiz_hits = 0

    def _ingest_pose(self, pose33: Pose33) -> bool:
        out = extract_frame_features(
            pose33,
            self.BodyLandmark,
            min_quality=self.min_quality,
            min_good_keypoints=self.min_good_keypoints,
        )
        if out is None:
            if self.reset_on_invalid:
                self.reset()
            return False
        feat, qual = out
        self._feat_buf.append(feat)
        self._qual_buf.append(qual)
        return True

    def update(
        self,
        frame_landmarks: Union[Pose33, PoseCell],
    ) -> Dict[str, Any]:
        pose33: Optional[Pose33] = None

        if frame_landmarks is None:
            if self.reset_on_invalid:
                self.reset()
            return {"ready": False, "reason": "no_pose"}

        if isinstance(frame_landmarks, list) and len(frame_landmarks) > 0:
            if isinstance(frame_landmarks[0], dict):
                pose33 = frame_landmarks  # already best pose33
            else:
                pose33 = select_best_pose(frame_landmarks)  # list of poses
        else:
            if self.reset_on_invalid:
                self.reset()
            return {"ready": False, "reason": "invalid_input"}

        if pose33 is None:
            if self.reset_on_invalid:
                self.reset()
            return {"ready": False, "reason": "no_pose"}

        ok = self._ingest_pose(pose33)
        if not ok:
            return {"ready": False, "reason": "low_quality"}

        if len(self._feat_buf) < self.window:
            return {"ready": False, "reason": "warming_up", "buffer": len(self._feat_buf)}

        W = np.stack(list(self._feat_buf), axis=0)
        x = window_vector(W).reshape(1, -1)

        fall_p = _proba_pos(self.fall_model, x)
        fall_pred = int(fall_p >= self.fall_threshold)
        self._fall_hits = (self._fall_hits + 1) if fall_pred == 1 else 0
        fall_event = self._fall_hits >= self.consecutive_fall

        horiz_p = None
        horiz_event = False
        if fall_event:
            horiz_p = _proba_pos(self.horizontal_model, x)
            horiz_pred = int(horiz_p >= self.horizontal_threshold)
            self._horiz_hits = (self._horiz_hits + 1) if horiz_pred == 1 else 0
            horiz_event = self._horiz_hits >= self.consecutive_horizontal
        else:
            self._horiz_hits = 0

        return {
            "ready": True,
            "fall_prob": float(fall_p),
            "fall_event": bool(fall_event),
            "fall_hits": int(self._fall_hits),
            "horizontal_prob": None if horiz_p is None else float(horiz_p),
            "horizontal_event": bool(horiz_event),
            "horizontal_hits": int(self._horiz_hits),
        }
