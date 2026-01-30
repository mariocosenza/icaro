import math
from collections import deque
from dataclasses import dataclass
from typing import Union, Dict, Optional, Any, Tuple, TYPE_CHECKING
from uuid import uuid4

import numpy as np

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
else:
    Pipeline = Any

from pipeline_horizontal_classification import (
    Pose33, extract_frame_features_fall, extract_frame_features_horizontal,
    PoseCell, select_best_pose, window_vector_nan, _proba_pos,
)


@dataclass(frozen=True)
class DetectorConfig:
    fall_min_vis_point: float = 0.80
    fall_min_pres_point: float = 0.80
    fall_min_required_core_points: int = 5
    horizontal_min_quality: float = 0.65
    horizontal_min_good_keypoints: int = 4
    min_window_quality: Union[float, str] = "high"
    fall_threshold: float = 0.80
    horizontal_threshold: float = 0.70
    consecutive_fall: int = 5
    consecutive_horizontal: int = 4
    post_fall_duration: int = 60
    reset_on_invalid: bool = False
    horizontal_always_active: bool = True


class LiveManDownDetector:
    def __init__(
            self,
            body_landmark_cls,
            *,
            fall_model: Pipeline,
            horizontal_model: Pipeline,
            window: int,
            config: Optional[DetectorConfig] = None,
    ):
        cfg = config or DetectorConfig()

        self.body_landmark_cls = body_landmark_cls
        self.fall_model = fall_model
        self.horizontal_model = horizontal_model
        self.window = int(window)

        self.fall_min_vis_point = float(cfg.fall_min_vis_point)
        self.fall_min_pres_point = float(cfg.fall_min_pres_point)
        self.fall_min_required_core_points = int(cfg.fall_min_required_core_points)

        self.horizontal_min_quality = float(cfg.horizontal_min_quality)
        self.horizontal_min_good_keypoints = int(cfg.horizontal_min_good_keypoints)

        quality_levels = {"low": 0.60, "medium": 0.80, "high": 0.95}
        if isinstance(cfg.min_window_quality, str):
            self.min_window_quality = quality_levels.get(cfg.min_window_quality.lower(), 0.80)
        else:
            self.min_window_quality = float(cfg.min_window_quality)

        self.fall_threshold = float(cfg.fall_threshold)
        self.horizontal_threshold = float(cfg.horizontal_threshold)
        self.consecutive_fall = int(cfg.consecutive_fall)
        self.consecutive_horizontal = int(cfg.consecutive_horizontal)

        self.post_fall_duration = int(cfg.post_fall_duration)
        self._post_fall_timer = 0

        self.reset_on_invalid = bool(cfg.reset_on_invalid)
        self.horizontal_always_active = bool(cfg.horizontal_always_active)

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
        self._post_fall_timer = 0

    def _reset_hits(self) -> None:
        self._fall_hits = 0
        self._horiz_hits = 0

    def _prepare_pose(self, frame_landmarks: Union[Pose33, PoseCell]) -> Tuple[Optional[Pose33], Dict[str, Any]]:
        if frame_landmarks is None:
            if self.reset_on_invalid:
                self.reset()
            return None, {"ready": False, "reason": "no_pose"}

        pose33: Optional[Pose33] = None
        if isinstance(frame_landmarks, list) and len(frame_landmarks) > 0:
            if isinstance(frame_landmarks[0], dict):
                pose33 = frame_landmarks
            else:
                pose33 = select_best_pose(frame_landmarks)
        else:
            if self.reset_on_invalid:
                self.reset()
            return None, {"ready": False, "reason": "invalid_input"}

        if pose33 is None:
            if self.reset_on_invalid:
                self.reset()
            return None, {"ready": False, "reason": "no_pose"}

        return pose33, {}

    def _ingest_fall(self, pose33: Pose33) -> bool:
        out_f = extract_frame_features_fall(
            pose33,
            self.body_landmark_cls,
            min_vis_point=self.fall_min_vis_point,
            min_pres_point=self.fall_min_pres_point,
            min_required_core_points=self.fall_min_required_core_points,
        )
        if out_f is None:
            return False
        f_feat, f_q = out_f
        self._fall_feat_buf.append(f_feat)
        self._fall_q_buf.append(f_q)
        return True

    def _ingest_horizontal(self, pose33: Pose33) -> bool:
        out_h = extract_frame_features_horizontal(
            pose33,
            self.body_landmark_cls,
            min_quality=self.horizontal_min_quality,
            min_good_keypoints=self.horizontal_min_good_keypoints,
        )
        if out_h is None:
            return False
        h_feat, h_q = out_h
        self._h_feat_buf.append(h_feat)
        self._h_q_buf.append(h_q)
        return True

    def _ingest(self, pose33: Pose33) -> bool:
        fall_ok = self._ingest_fall(pose33)
        self._ingest_horizontal(pose33)
        return fall_ok

    def _window_quality(self, q_buf: deque) -> float:
        if len(q_buf) == 0:
            return 0.0
        return float(np.nanmean(list(q_buf)))

    def _evaluate_fall(self) -> Tuple[bool, float]:
        fall_window = np.stack(list(self._fall_feat_buf), axis=0)
        xf = window_vector_nan(fall_window).reshape(1, -1)
        fall_prob = float(np.asarray(_proba_pos(self.fall_model, xf)).ravel()[0])

        if fall_prob >= self.fall_threshold:
            self._fall_hits += 1
        else:
            self._fall_hits = 0

        return self._fall_hits >= self.consecutive_fall, fall_prob

    def _evaluate_horizontal(self) -> Tuple[bool, Optional[float]]:
        if self._post_fall_timer <= 0 and not self.horizontal_always_active:
            self._horiz_hits = 0
            return False, None

        self._post_fall_timer -= 1

        if len(self._h_feat_buf) < self.window:
            self._horiz_hits = 0
            return False, None

        horiz_quality = self._window_quality(self._h_q_buf)
        if horiz_quality < self.min_window_quality:
            self._horiz_hits = 0
            return False, None

        horiz_window = np.stack(list(self._h_feat_buf), axis=0)
        xh = window_vector_nan(horiz_window).reshape(1, -1)
        horiz_prob = float(np.asarray(_proba_pos(self.horizontal_model, xh)).ravel()[0])

        if horiz_prob >= self.horizontal_threshold:
            self._horiz_hits += 1
        else:
            self._horiz_hits = 0

        return self._horiz_hits >= self.consecutive_horizontal, horiz_prob

    def update(self, frame_landmarks: Union[Pose33, PoseCell]) -> Dict[str, Any]:
        pose33, failure = self._prepare_pose(frame_landmarks)
        if failure:
            return failure

        fall_ok = self._ingest(pose33)
        if not fall_ok and self._post_fall_timer == 0:
            return {"ready": False, "reason": "low_quality_fall_points"}

        if len(self._fall_feat_buf) < self.window:
            return {"ready": False, "reason": "warming_up", "buffer": len(self._fall_feat_buf)}

        fall_quality = self._window_quality(self._fall_q_buf)
        if fall_quality < self.min_window_quality and self._post_fall_timer == 0:
            self._reset_hits()
            return {
                "ready": False,
                "reason": "quality_gate_fall",
                "fall_quality": fall_quality,
                "min_required": self.min_window_quality,
            }

        fall_event, fall_prob = self._evaluate_fall()
        if fall_event:
            self._post_fall_timer = self.post_fall_duration

        horiz_event, horiz_prob = self._evaluate_horizontal()

        return {
            "ready": True,
            "fall_prob": float(fall_prob),
            "fall_event": bool(fall_event),
            "fall_hits": int(self._fall_hits),
            "horizontal_prob": None if horiz_prob is None else float(horiz_prob),
            "horizontal_event": bool(horiz_event),
            "horizontal_hits": int(self._horiz_hits),
            "fall_quality": fall_quality,
            "timer": int(self._post_fall_timer)
        }


@dataclass
class PersonTrack:
    track_id: str
    detector: LiveManDownDetector
    center: Tuple[float, float]
    last_seen_frame: int


class MultiPersonDetector:
    def __init__(
            self,
            body_landmark_cls,
            create_detector,
            *,
            distance_threshold: float = 0.12,
            max_tracks: int = 8,
            max_missed: int = 30,
            center_min_vis: float = 0.30,
            center_min_pres: float = 0.30,
            center_smoothing: float = 0.60,
    ) -> None:
        self.body_landmark_cls = body_landmark_cls
        self.create_detector = create_detector
        self.distance_threshold = float(distance_threshold)
        self.max_tracks = int(max_tracks)
        self.max_missed = int(max_missed)
        self.center_min_vis = float(center_min_vis)
        self.center_min_pres = float(center_min_pres)
        self.center_smoothing = float(center_smoothing)
        self.tracks: Dict[str, PersonTrack] = {}
        self._frame_idx = 0

    def _pose_center(self, pose33: Pose33) -> Tuple[float, float]:
        idx = lambda e: int(e.value) if hasattr(e, "value") else int(e)

        def get_xy(i: int, fallback: Tuple[float, float]) -> Tuple[float, float]:
            try:
                lm = pose33[i]
                return float(lm.get("x", fallback[0])), float(lm.get("y", fallback[1]))
            except Exception:
                return fallback

        def get_vis_pres(i: int) -> Tuple[float, float]:
            try:
                lm = pose33[i]
                return float(lm.get("visibility", 0.0)), float(lm.get("presence", 0.0))
            except Exception:
                return 0.0, 0.0

        center_points = []
        for key in (
                self.body_landmark_cls.NOSE,
                self.body_landmark_cls.LEFT_SHOULDER,
                self.body_landmark_cls.RIGHT_SHOULDER,
                self.body_landmark_cls.LEFT_HIP,
                self.body_landmark_cls.RIGHT_HIP,
        ):
            i = idx(key)
            vis, pres = get_vis_pres(i)
            if vis >= self.center_min_vis and pres >= self.center_min_pres:
                center_points.append(get_xy(i, (0.0, 0.0)))

        if center_points:
            xs = [pt[0] for pt in center_points]
            ys = [pt[1] for pt in center_points]
            return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

        nose = get_xy(idx(self.body_landmark_cls.NOSE), (0.0, 0.0))
        hips = get_xy(idx(self.body_landmark_cls.LEFT_HIP), nose)
        return nose if nose != (0.0, 0.0) else hips

    def _nearest_track(self, center: Tuple[float, float]) -> Optional[str]:
        best_id = None
        best_dist = None
        for tid, track in self.tracks.items():
            dx = center[0] - track.center[0]
            dy = center[1] - track.center[1]
            dist = math.hypot(dx, dy)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_id = tid
        if best_dist is not None and best_dist <= self.distance_threshold:
            return best_id
        return None

    def _ensure_capacity(self) -> None:
        if len(self.tracks) < self.max_tracks:
            return
        oldest_id = min(self.tracks, key=lambda tid: self.tracks[tid].last_seen_frame)
        self.tracks.pop(oldest_id, None)

    def _prune_stale(self) -> None:
        if not self.tracks:
            return
        cutoff = self._frame_idx - self.max_missed
        stale = [tid for tid, track in self.tracks.items() if track.last_seen_frame < cutoff]
        for tid in stale:
            self.tracks.pop(tid, None)

    def update(self, poses: Optional[list[Pose33]]) -> list[Dict[str, Any]]:
        self._frame_idx += 1
        self._prune_stale()

        if not poses:
            return []

        outputs: list[Dict[str, Any]] = []
        centers = [self._pose_center(pose33) for pose33 in poses]

        assignments: Dict[int, str] = {}
        used_tracks: set[str] = set()

        if self.tracks:
            pairs = []
            for pose_idx, center in enumerate(centers):
                for tid, track in self.tracks.items():
                    dx = center[0] - track.center[0]
                    dy = center[1] - track.center[1]
                    dist = math.hypot(dx, dy)
                    if dist <= self.distance_threshold:
                        pairs.append((dist, pose_idx, tid))

            pairs.sort(key=lambda item: item[0])
            for _, pose_idx, tid in pairs:
                if pose_idx in assignments or tid in used_tracks:
                    continue
                assignments[pose_idx] = tid
                used_tracks.add(tid)

        for pose_idx, pose33 in enumerate(poses):
            center = centers[pose_idx]
            track_id = assignments.get(pose_idx)

            if track_id is None:
                self._ensure_capacity()
                track_id = uuid4().hex
                self.tracks[track_id] = PersonTrack(
                    track_id=track_id,
                    detector=self.create_detector(),
                    center=center,
                    last_seen_frame=self._frame_idx,
                )
            else:
                track = self.tracks[track_id]
                alpha = self.center_smoothing
                if 0.0 < alpha < 1.0:
                    center = (
                        track.center[0] * (1.0 - alpha) + center[0] * alpha,
                        track.center[1] * (1.0 - alpha) + center[1] * alpha,
                    )
                track.center = center
                track.last_seen_frame = self._frame_idx

            track = self.tracks[track_id]
            out = track.detector.update(pose33)

            outputs.append({
                "track_id": track_id,
                "output": out,
            })

        return outputs
