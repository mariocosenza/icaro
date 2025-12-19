from collections import deque
from typing import Union, Dict, Optional, Any, Tuple

import numpy as np
from sklearn.pipeline import Pipeline

from src.pipeline_horizontal_classification import (
    Pose33, extract_frame_features_fall, extract_frame_features_horizontal,
    PoseCell, select_best_pose, window_vector_nan, _proba_pos
)

class LiveManDownDetector:
    def __init__(
        self,
        BodyLandmark,
        *,
        fall_model: Pipeline,
        horizontal_model: Pipeline,
        window: int,


        fall_min_vis_point: float = 0.65,
        fall_min_pres_point: float = 0.65,
        fall_min_required_core_points: int = 3,

        horizontal_min_quality: float = 0.65,
        horizontal_min_good_keypoints: int = 4,

        min_window_quality: float = 0.65,

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

        self.fall_min_vis_point = float(fall_min_vis_point)
        self.fall_min_pres_point = float(fall_min_pres_point)
        self.fall_min_required_core_points = int(fall_min_required_core_points)

        self.horizontal_min_quality = float(horizontal_min_quality)
        self.horizontal_min_good_keypoints = int(horizontal_min_good_keypoints)

        self.min_window_quality = float(min_window_quality)

        self.fall_threshold = float(fall_threshold)
        self.horizontal_threshold = float(horizontal_threshold)
        self.consecutive_fall = int(consecutive_fall)
        self.consecutive_horizontal = int(consecutive_horizontal)
        self.reset_on_invalid = bool(reset_on_invalid)

        self._fall_feat_buf = deque(maxlen=self.window)
        self._fall_q_buf = deque(maxlen=self.window)

        self._h_feat_buf = deque(maxlen=self.window)
        self._h_q_buf = deque(maxlen=self.window)

        self._fall_hits = 0
        self._horiz_hits = 0

    def reset(self):
        self._fall_feat_buf.clear()
        self._fall_q_buf.clear()
        self._h_feat_buf.clear()
        self._h_q_buf.clear()
        self._fall_hits = 0
        self._horiz_hits = 0

    def _ingest(self, pose33: Pose33) -> Tuple[bool, bool]:
        fall_ok = False
        out_f = extract_frame_features_fall(
            pose33,
            self.BodyLandmark,
            min_vis_point=self.fall_min_vis_point,
            min_pres_point=self.fall_min_pres_point,
            min_required_core_points=self.fall_min_required_core_points,
        )
        if out_f is not None:
            f_feat, f_q = out_f
            self._fall_feat_buf.append(f_feat)
            self._fall_q_buf.append(f_q)
            fall_ok = True

        h_ok = False
        out_h = extract_frame_features_horizontal(
            pose33,
            self.BodyLandmark,
            min_quality=self.horizontal_min_quality,
            min_good_keypoints=self.horizontal_min_good_keypoints,
        )
        if out_h is not None:
            h_feat, h_q = out_h
            self._h_feat_buf.append(h_feat)
            self._h_q_buf.append(h_q)
            h_ok = True

        return fall_ok, h_ok

    def _window_quality(self, q_buf: deque) -> float:
        if len(q_buf) == 0:
            return 0.0
        return float(np.nanmean(list(q_buf)))

    def update(self, frame_landmarks: Union[Pose33, PoseCell]) -> Dict[str, Any]:
        pose33: Optional[Pose33] = None

        if frame_landmarks is None:
            if self.reset_on_invalid:
                self.reset()
            return {"ready": False, "reason": "no_pose"}

        if isinstance(frame_landmarks, list) and len(frame_landmarks) > 0:
            if isinstance(frame_landmarks[0], dict):
                pose33 = frame_landmarks
            else:
                pose33 = select_best_pose(frame_landmarks)
        else:
            if self.reset_on_invalid:
                self.reset()
            return {"ready": False, "reason": "invalid_input"}

        if pose33 is None:
            if self.reset_on_invalid:
                self.reset()
            return {"ready": False, "reason": "no_pose"}

        fall_ok, _ = self._ingest(pose33)
        if not fall_ok:
            return {"ready": False, "reason": "low_quality_fall_points"}

        if len(self._fall_feat_buf) < self.window:
            return {"ready": False, "reason": "warming_up", "buffer": len(self._fall_feat_buf)}

        # --- NEW: gate ALL predictions if fall window quality < 0.65 ---
        fall_q = self._window_quality(self._fall_q_buf)
        if fall_q < self.min_window_quality:
            self._fall_hits = 0
            self._horiz_hits = 0
            return {
                "ready": False,
                "reason": "quality_gate_fall",
                "fall_quality": fall_q,
                "min_required": self.min_window_quality,
            }

        Wf = np.stack(list(self._fall_feat_buf), axis=0)
        xf = window_vector_nan(Wf).reshape(1, -1)

        fall_p = _proba_pos(self.fall_model, xf)
        fall_pred = int(fall_p >= self.fall_threshold)
        self._fall_hits = (self._fall_hits + 1) if fall_pred == 1 else 0
        fall_event = self._fall_hits >= self.consecutive_fall

        horiz_p = None
        horiz_event = False

        if fall_event:
            if len(self._h_feat_buf) >= self.window:
                # --- NEW: also gate horizontal prediction on its window quality ---
                horiz_q = self._window_quality(self._h_q_buf)
                if horiz_q >= self.min_window_quality:
                    Wh = np.stack(list(self._h_feat_buf), axis=0)
                    xh = window_vector_nan(Wh).reshape(1, -1)
                    horiz_p = _proba_pos(self.horizontal_model, xh)
                    horiz_pred = int(horiz_p >= self.horizontal_threshold)
                    self._horiz_hits = (self._horiz_hits + 1) if horiz_pred == 1 else 0
                    horiz_event = self._horiz_hits >= self.consecutive_horizontal
                else:
                    self._horiz_hits = 0
            else:
                self._horiz_hits = 0
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
            "fall_quality": fall_q,
        }
