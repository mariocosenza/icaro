import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from collections import deque
import inspect

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score


NORMAL = 0
FALL = 1

NOT_HORIZONTAL = 0
HORIZONTAL = 1


@dataclass
class VideoSample:
    name: str
    start: int
    end: int
    df: pd.DataFrame


def iter_video_entries(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        if {"name", "start", "end", "data"}.issubset(obj.keys()):
            yield obj
        for v in obj.values():
            yield from iter_video_entries(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_video_entries(v)


def build_df_from_any_json(data: Any) -> pd.DataFrame:
    if data is None:
        raise ValueError("data is None")

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, dict):
        if "frame_index" in data:
            df = pd.DataFrame(data)
        elif "data" in data and isinstance(data["data"], (dict, list)):
            return build_df_from_any_json(data["data"])
        elif "columns" in data and "data" in data and isinstance(data["data"], list):
            df = pd.DataFrame(data["data"], columns=data["columns"])
        else:
            first = next(iter(data.values()), None)
            if isinstance(first, dict) and ("frame_index" in first or "timestamp_ms" in first):
                df = pd.DataFrame.from_dict(data, orient="index")
            else:
                df = pd.DataFrame(data)
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported data type for dataframe: {type(data)}")

    if "frame_index" not in df.columns:
        try:
            idx_as_int = pd.to_numeric(df.index, errors="raise")
            df = df.reset_index().rename(columns={"index": "frame_index"})
            df["frame_index"] = idx_as_int.astype(int)
        except Exception:
            pass

    if "frame_index" not in df.columns:
        raise ValueError(f"Missing frame_index in data. Columns={list(df.columns)}")

    df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["frame_index"]).copy()
    df["frame_index"] = df["frame_index"].astype(int)

    if "timestamp_ms" in df.columns:
        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["timestamp_ms"]).copy()
        df["timestamp_ms"] = df["timestamp_ms"].astype(int)

    return df.sort_values("frame_index").reset_index(drop=True)


def load_dataset_from_json_obj(dataset_obj: Any) -> List[VideoSample]:
    videos: List[VideoSample] = []
    for entry in iter_video_entries(dataset_obj):
        if entry.get("data") is None or entry.get("start") is None or entry.get("end") is None:
            continue
        try:
            df = build_df_from_any_json(entry["data"])
        except Exception:
            continue
        videos.append(VideoSample(name=entry["name"], start=int(entry["start"]), end=int(entry["end"]), df=df))
    if not videos:
        raise ValueError("No usable videos found in dataset JSON.")
    return videos


def _infer_step(df: pd.DataFrame) -> int:
    fi = np.asarray(df["frame_index"].values, dtype=int)
    if len(fi) < 3:
        return 2
    d = np.diff(fi)
    d = d[d > 0]
    return int(np.median(d)) if len(d) else 2


Pose33 = List[Dict[str, float]]
PoseCell = Union[None, List[Pose33]]


def select_best_pose(poses: Any) -> Optional[Pose33]:
    if poses is None or poses == [] or not isinstance(poses, list):
        return None
    candidates = []
    for p in poses:
        if not isinstance(p, list) or len(p) < 33:
            continue
        vis = np.array([float(lm.get("visibility", 0.0)) for lm in p], dtype=np.float32)
        pres = np.array([float(lm.get("presence", 0.0)) for lm in p], dtype=np.float32)
        score = float(np.mean(vis * pres))
        candidates.append((score, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v) + 1e-6)


def _supports_sample_weight(pipeline: Pipeline) -> bool:
    clf = pipeline.named_steps.get("clf", None)
    if clf is None:
        return False
    try:
        return "sample_weight" in inspect.signature(clf.fit).parameters
    except Exception:
        return False


def _binary_class_weight(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    w = (len(y) / (2.0 * counts)).astype(np.float32)
    return w[y]


def _downsample_majority(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    g: np.ndarray,
    *,
    majority_label: int,
    max_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_maj = np.where(y == majority_label)[0]
    idx_min = np.where(y != majority_label)[0]
    if len(idx_min) == 0:
        return X, y, w, g
    n_keep = int(min(len(idx_maj), max_ratio * len(idx_min)))
    keep_maj = rng.choice(idx_maj, size=n_keep, replace=False) if n_keep > 0 else np.array([], dtype=int)
    keep = np.concatenate([idx_min, keep_maj])
    rng.shuffle(keep)
    return X[keep], y[keep], w[keep], g[keep]


def _pick_split_binary(
    X, y, groups, test_size, random_state,
    tries=140, min_pos_test=30, min_pos_train=80
):
    rng = np.random.default_rng(random_state)
    best = None
    best_score = -1
    for _ in range(tries):
        rs = int(rng.integers(0, 1_000_000))
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        tr, te = next(splitter.split(X, y, groups=groups))
        pos_te = int(np.sum(y[te] == 1))
        pos_tr = int(np.sum(y[tr] == 1))
        if pos_te >= min_pos_test and pos_tr >= min_pos_train:
            if pos_te > best_score:
                best = (tr, te)
                best_score = pos_te
    if best is None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        best = next(splitter.split(X, y, groups=groups))
    return best


def window_vector_nan(feat_window: np.ndarray) -> np.ndarray:
    feat_window = np.asarray(feat_window, dtype=np.float32)

    if feat_window.ndim != 2 or feat_window.size == 0:
        return np.zeros((1,), dtype=np.float32)

    col_all_nan = np.all(np.isnan(feat_window), axis=0)
    safe = feat_window.copy()

    if np.any(col_all_nan):
        safe[:, col_all_nan] = 0.0

    mean = np.nanmean(safe, axis=0)
    std = np.nanstd(safe, axis=0)

    mn = np.nanmin(safe, axis=0)
    mx = np.nanmax(safe, axis=0)

    delta = safe[-1] - safe[0]

    out = np.concatenate([mean, std, mn, mx, delta], axis=0).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out



def windowize_last_label(
    feats: List[np.ndarray],
    labels: List[int],
    qual: List[float],
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(feats) < window:
        return np.zeros((0, 1), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32)

    arr = np.stack(feats, axis=0)
    cls = np.asarray(labels, dtype=np.int32)
    q = np.asarray(qual, dtype=np.float32)

    X, y, w = [], [], []
    for i in range(window - 1, len(arr)):
        w_feats = arr[i - window + 1: i + 1]
        label = int(cls[i])
        wq = float(np.nanmean(q[i - window + 1: i + 1]))
        X.append(window_vector_nan(w_feats))
        y.append(label)
        w.append(float(np.clip(wq, 0.05, 1.0)))

    return np.asarray(X, np.float32), np.asarray(y, np.int32), np.asarray(w, np.float32)


def make_hgb(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(
                max_depth=3,
                learning_rate=0.05,
                max_iter=400,
                random_state=random_state,
            )),
        ]
    )


def make_mlp(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                batch_size=128,
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=10,
                validation_fraction=0.1,
                random_state=random_state,
            )),
        ]
    )


def extract_frame_features_horizontal(
    pose33: Pose33,
    BodyLandmark,
    *,
    min_quality: float = 0.20,
    min_good_keypoints: int = 4,
) -> Optional[Tuple[np.ndarray, float]]:
    idx = lambda e: int(e.value) if hasattr(e, "value") else int(e)

    NOSE = idx(BodyLandmark.NOSE)
    L_SH = idx(BodyLandmark.LEFT_SHOULDER)
    R_SH = idx(BodyLandmark.RIGHT_SHOULDER)
    L_HIP = idx(BodyLandmark.LEFT_HIP)
    R_HIP = idx(BodyLandmark.RIGHT_HIP)
    L_ANK = idx(BodyLandmark.LEFT_ANKLE)
    R_ANK = idx(BodyLandmark.RIGHT_ANKLE)

    key_idx = [NOSE, L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK]
    vis = np.array([float(pose33[i].get("visibility", 0.0)) for i in key_idx], dtype=np.float32)
    pres = np.array([float(pose33[i].get("presence", 0.0)) for i in key_idx], dtype=np.float32)
    q = float(np.mean(vis * pres))
    good = int(np.sum((vis >= 0.5) & (pres >= 0.5)))
    if q < min_quality or good < min_good_keypoints:
        return None

    def xyz(i: int):
        lm = pose33[i]
        return float(lm["x"]), float(lm["y"]), float(lm.get("z", 0.0))

    x_ls, y_ls, z_ls = xyz(L_SH)
    x_rs, y_rs, z_rs = xyz(R_SH)
    x_lh, y_lh, z_lh = xyz(L_HIP)
    x_rh, y_rh, z_rh = xyz(R_HIP)
    x_la, y_la, z_la = xyz(L_ANK)
    x_ra, y_ra, z_ra = xyz(R_ANK)
    x_n, y_n, z_n = xyz(NOSE)

    sh_mid = np.array([(x_ls + x_rs) / 2, (y_ls + y_rs) / 2, (z_ls + z_rs) / 2], dtype=np.float32)
    hip_mid = np.array([(x_lh + x_rh) / 2, (y_lh + y_rh) / 2, (z_lh + z_rh) / 2], dtype=np.float32)
    ank_mid = np.array([(x_la + x_ra) / 2, (y_la + y_ra) / 2, (z_la + z_ra) / 2], dtype=np.float32)

    torso = sh_mid - hip_mid
    body_h = _norm(sh_mid - ank_mid)

    sh_w = _norm(np.array([x_ls - x_rs, y_ls - y_rs, z_ls - z_rs], dtype=np.float32))
    hip_w = _norm(np.array([x_lh - x_rh, y_lh - y_rh, z_lh - z_rh], dtype=np.float32))

    vert = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    c = float(np.clip(np.dot(torso, vert) / (_norm(torso) * _norm(vert)), -1.0, 1.0))
    torso_tilt = float(np.arccos(c))

    xs = [float(lm["x"]) for lm in pose33]
    ys = [float(lm["y"]) for lm in pose33]
    bbox_w = (max(xs) - min(xs))
    bbox_h = (max(ys) - min(ys))
    bbox_ar = float(bbox_w / (bbox_h + 1e-6))

    feats = np.array(
        [
            torso_tilt,
            sh_w / (body_h + 1e-6),
            hip_w / (body_h + 1e-6),
            body_h,
            float(sh_mid[1]),
            float(hip_mid[1]),
            float(ank_mid[1]),
            float(sh_mid[0]),
            float(hip_mid[0]),
            float(ank_mid[0]),
            bbox_ar,
            float(y_n),
        ],
        dtype=np.float32,
    )
    return feats, float(np.clip(q, 0.0, 1.0))


def extract_frame_features_fall(
    pose33: Pose33,
    BodyLandmark,
    *,
    min_vis_point: float = 0.55,
    min_pres_point: float = 0.55,
    min_required_core_points: int = 3,
) -> Optional[Tuple[np.ndarray, float]]:
    idx = lambda e: int(e.value) if hasattr(e, "value") else int(e)

    NOSE = idx(BodyLandmark.NOSE)
    L_SH = idx(BodyLandmark.LEFT_SHOULDER)
    R_SH = idx(BodyLandmark.RIGHT_SHOULDER)
    L_HIP = idx(BodyLandmark.LEFT_HIP)
    R_HIP = idx(BodyLandmark.RIGHT_HIP)
    L_ANK = idx(BodyLandmark.LEFT_ANKLE)
    R_ANK = idx(BodyLandmark.RIGHT_ANKLE)

    core = [L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK]

    vis = np.array([float(pose33[i].get("visibility", 0.0)) for i in range(min(33, len(pose33)))], dtype=np.float32)
    pres = np.array([float(pose33[i].get("presence", 0.0)) for i in range(min(33, len(pose33)))], dtype=np.float32)
    ok = (vis >= min_vis_point) & (pres >= min_pres_point)

    core_ok = int(np.sum(ok[core]))
    if core_ok < min_required_core_points:
        return None

    q = float(np.mean((vis[core] * pres[core])[ok[core]])) if np.any(ok[core]) else 0.0
    q = float(np.clip(q, 0.0, 1.0))

    xs = np.array([float(pose33[i].get("x", np.nan)) for i in range(min(33, len(pose33)))], dtype=np.float32)
    ys = np.array([float(pose33[i].get("y", np.nan)) for i in range(min(33, len(pose33)))], dtype=np.float32)
    zs = np.array([float(pose33[i].get("z", 0.0)) for i in range(min(33, len(pose33)))], dtype=np.float32)

    xs[~ok] = np.nan
    ys[~ok] = np.nan
    zs[~ok] = np.nan

    def p(i: int) -> np.ndarray:
        return np.array([xs[i], ys[i], zs[i]], dtype=np.float32)

    def mid(a: int, b: int) -> np.ndarray:
        pa, pb = p(a), p(b)
        if np.any(np.isnan(pa)) and np.any(np.isnan(pb)):
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        if np.any(np.isnan(pa)):
            return pb
        if np.any(np.isnan(pb)):
            return pa
        return (pa + pb) / 2.0

    sh_mid = mid(L_SH, R_SH)
    hip_mid = mid(L_HIP, R_HIP)
    ank_mid = mid(L_ANK, R_ANK)

    torso = sh_mid - hip_mid
    torso_norm = np.linalg.norm(torso) if not np.any(np.isnan(torso)) else np.nan

    sh_w = np.linalg.norm(p(L_SH) - p(R_SH)) if ok[L_SH] and ok[R_SH] else np.nan
    hip_w = np.linalg.norm(p(L_HIP) - p(R_HIP)) if ok[L_HIP] and ok[R_HIP] else np.nan
    body_h = np.linalg.norm(sh_mid - ank_mid) if not (np.any(np.isnan(sh_mid)) or np.any(np.isnan(ank_mid))) else np.nan

    vert = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    if np.isnan(torso_norm) or torso_norm < 1e-6:
        torso_tilt = np.nan
    else:
        c = float(np.clip(np.dot(torso, vert) / (float(torso_norm) * _norm(vert)), -1.0, 1.0))
        torso_tilt = float(np.arccos(c))

    if np.any(~np.isnan(xs)) and np.any(~np.isnan(ys)):
        bx = xs[~np.isnan(xs)]
        by = ys[~np.isnan(ys)]
        if len(bx) >= 2 and len(by) >= 2:
            bbox_w = float(np.max(bx) - np.min(bx))
            bbox_h = float(np.max(by) - np.min(by))
            bbox_ar = float(bbox_w / (bbox_h + 1e-6))
        else:
            bbox_ar = np.nan
    else:
        bbox_ar = np.nan

    y_n = float(ys[NOSE]) if ok[NOSE] and not np.isnan(ys[NOSE]) else np.nan

    feats = np.array(
        [
            torso_tilt,
            (sh_w / (body_h + 1e-6)) if not (np.isnan(sh_w) or np.isnan(body_h)) else np.nan,
            (hip_w / (body_h + 1e-6)) if not (np.isnan(hip_w) or np.isnan(body_h)) else np.nan,
            body_h,
            float(sh_mid[1]) if not np.isnan(sh_mid[1]) else np.nan,
            float(hip_mid[1]) if not np.isnan(hip_mid[1]) else np.nan,
            float(ank_mid[1]) if not np.isnan(ank_mid[1]) else np.nan,
            float(sh_mid[0]) if not np.isnan(sh_mid[0]) else np.nan,
            float(hip_mid[0]) if not np.isnan(hip_mid[0]) else np.nan,
            float(ank_mid[0]) if not np.isnan(ank_mid[0]) else np.nan,
            bbox_ar,
            y_n,
        ],
        dtype=np.float32,
    )

    return feats, q


def build_xy_binary_fall(
    videos: List[VideoSample],
    BodyLandmark,
    *,
    pose_col: str,
    window: int = 7,
    margin: int = 6,
    min_vis_point: float = 0.55,
    min_pres_point: float = 0.55,
    min_required_core_points: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_all, y_all, g_all, w_all = [], [], [], []

    for vid, v in enumerate(videos):
        df = v.df
        if pose_col not in df.columns:
            continue

        step = _infer_step(df)
        feats_seg: List[np.ndarray] = []
        cls_seg: List[int] = []
        q_seg: List[float] = []
        last_fi: Optional[int] = None

        def flush():
            nonlocal feats_seg, cls_seg, q_seg
            if len(feats_seg) >= window:
                Xw, yw, ww = windowize_last_label(feats_seg, cls_seg, q_seg, window=window)
                if len(Xw) > 0:
                    X_all.append(Xw)
                    y_all.append(yw)
                    w_all.append(ww)
                    g_all.append(np.full((len(yw),), vid, dtype=np.int32))
            feats_seg, cls_seg, q_seg = [], [], []

        for _, row in df.iterrows():
            fi = int(row["frame_index"])
            poses = row[pose_col]
            best = select_best_pose(poses)

            if best is None:
                if last_fi is not None and fi - last_fi > step:
                    flush()
                    last_fi = None
                continue

            if ((v.start - margin) <= fi < v.start) or (v.end < fi <= (v.end + margin)):
                if last_fi is not None and fi - last_fi > step:
                    flush()
                    last_fi = None
                continue

            if last_fi is not None and fi - last_fi > (2 * step):
                flush()

            out = extract_frame_features_fall(
                best,
                BodyLandmark,
                min_vis_point=min_vis_point,
                min_pres_point=min_pres_point,
                min_required_core_points=min_required_core_points,
            )
            if out is None:
                if last_fi is not None and fi - last_fi > step:
                    flush()
                    last_fi = None
                continue

            feat, qual = out
            label = FALL if (v.start <= fi <= v.end) else NORMAL

            feats_seg.append(feat)
            cls_seg.append(label)
            q_seg.append(qual)
            last_fi = fi

        flush()

    if not X_all:
        raise ValueError("No samples for FALL model. Relax per-point thresholds or window/margin.")

    X = np.vstack(X_all).astype(np.float32)
    y = np.concatenate(y_all).astype(np.int32)
    g = np.concatenate(g_all).astype(np.int32)
    w = np.concatenate(w_all).astype(np.float32)
    return X, y, g, w


def build_xy_binary_horizontal_from_final_falled(
    videos: List[VideoSample],
    BodyLandmark,
    *,
    pose_col: str,
    window: int = 7,
    falled_tail_frames: int = 6,
    min_quality: float = 0.20,
    min_good_keypoints: int = 4,
    within_fall_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_all, y_all, g_all, w_all = [], [], [], []

    for vid, v in enumerate(videos):
        df = v.df
        if pose_col not in df.columns:
            continue

        step = _infer_step(df)
        tail = max(1, int(falled_tail_frames))
        horiz_start = v.end - (tail - 1) * step

        feats_seg: List[np.ndarray] = []
        cls_seg: List[int] = []
        q_seg: List[float] = []
        last_fi: Optional[int] = None

        def flush():
            nonlocal feats_seg, cls_seg, q_seg
            if len(feats_seg) >= window:
                Xw, yw, ww = windowize_last_label(feats_seg, cls_seg, q_seg, window=window)
                if len(Xw) > 0:
                    X_all.append(Xw)
                    y_all.append(yw)
                    w_all.append(ww)
                    g_all.append(np.full((len(yw),), vid, dtype=np.int32))
            feats_seg, cls_seg, q_seg = [], [], []

        for _, row in df.iterrows():
            fi = int(row["frame_index"])
            if within_fall_only and (fi < v.start or fi > v.end):
                continue

            poses = row[pose_col]
            best = select_best_pose(poses)
            if best is None:
                if last_fi is not None and fi - last_fi > step:
                    flush()
                    last_fi = None
                continue

            if last_fi is not None and fi - last_fi > (2 * step):
                flush()

            out = extract_frame_features_horizontal(
                best,
                BodyLandmark,
                min_quality=min_quality,
                min_good_keypoints=min_good_keypoints,
            )
            if out is None:
                if last_fi is not None and fi - last_fi > step:
                    flush()
                    last_fi = None
                continue

            feat, qual = out
            label = HORIZONTAL if (v.start <= fi <= v.end and fi >= horiz_start) else NOT_HORIZONTAL

            feats_seg.append(feat)
            cls_seg.append(label)
            q_seg.append(qual)
            last_fi = fi

        flush()

    if not X_all:
        raise ValueError("No samples for HORIZONTAL model. Lower thresholds/window or increase tail frames.")

    X = np.vstack(X_all).astype(np.float32)
    y = np.concatenate(y_all).astype(np.int32)
    g = np.concatenate(g_all).astype(np.int32)
    w = np.concatenate(w_all).astype(np.float32)
    return X, y, g, w


def _fit_search(
    base: Pipeline,
    param_dist: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray],
    scoring: str,
    n_iter: int,
    random_state: int,
) -> Pipeline:
    cv = GroupKFold(n_splits=5)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    if sample_weight is not None and _supports_sample_weight(base):
        search.fit(X_train, y_train, clf__sample_weight=sample_weight, groups=groups_train)
    else:
        search.fit(X_train, y_train, groups=groups_train)
    return search.best_estimator_


def train_binary_model_random_search_best_of_two(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    quality_w: np.ndarray,
    *,
    random_state: int,
    test_size: float = 0.2,
    max_majority_ratio: float = 2.5,
    n_iter_search: int = 30,
    scoring: str = "f1",
) -> Pipeline:
    tr, te = _pick_split_binary(X, y, groups, test_size=test_size, random_state=random_state)

    X_train, y_train, g_train, q_train = X[tr], y[tr], groups[tr], quality_w[tr]
    X_test, y_test = X[te], y[te]

    maj_label = 0 if np.sum(y_train == 0) >= np.sum(y_train == 1) else 1
    X_train, y_train, q_train, g_train = _downsample_majority(
        X_train, y_train, q_train, g_train,
        majority_label=maj_label,
        max_ratio=max_majority_ratio,
        seed=random_state,
    )

    cw = _binary_class_weight(y_train).astype(np.float32)
    sw = cw * np.clip(q_train, 0.05, 1.0)

    hgb = make_hgb(random_state=random_state)
    mlp = make_mlp(random_state=random_state)

    hgb_params = {
        "clf__max_depth": [2, 3, 4, 5],
        "clf__learning_rate": [0.02, 0.05, 0.08, 0.12],
        "clf__max_iter": [200, 300, 450, 650],
        "clf__min_samples_leaf": [10, 20, 40, 80],
        "clf__l2_regularization": [0.0, 0.1, 0.5, 1.0],
    }

    mlp_params = {
        "clf__hidden_layer_sizes": [(64,), (128,), (256,), (128, 64), (256, 128), (128, 128)],
        "clf__activation": ["relu", "tanh"],
        "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "clf__learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3],
        "clf__batch_size": [64, 128, 256],
        "clf__max_iter": [300, 500, 700],
    }

    print("Train support:", np.bincount(y_train, minlength=2), "Test support:", np.bincount(y_test, minlength=2))

    hgb_best = _fit_search(
        hgb, hgb_params,
        X_train, y_train, g_train,
        sample_weight=sw,
        scoring=scoring,
        n_iter=n_iter_search,
        random_state=random_state,
    )
    pred_hgb = hgb_best.predict(X_test)
    f1_hgb = float(f1_score(y_test, pred_hgb, pos_label=1))

    print("\n=== HGB ===")
    print("F1(pos):", f"{f1_hgb:.4f}")
    print("Confusion:\n", confusion_matrix(y_test, pred_hgb))
    print(classification_report(y_test, pred_hgb, digits=4, target_names=["no_fall", "fall"]))

    mlp_best = _fit_search(
        mlp, mlp_params,
        X_train, y_train, g_train,
        sample_weight=None,
        scoring=scoring,
        n_iter=max(10, n_iter_search // 2),
        random_state=random_state + 999,
    )
    pred_mlp = mlp_best.predict(X_test)
    f1_mlp = float(f1_score(y_test, pred_mlp, pos_label=1))

    print("\n=== MLP ===")
    print("F1(pos):", f"{f1_mlp:.4f}")
    print("Confusion:\n", confusion_matrix(y_test, pred_mlp))
    print(classification_report(y_test, pred_mlp, digits=4, target_names=["no_fall", "fall"]))

    if f1_mlp > f1_hgb:
        print("\nChosen: MLP")
        return mlp_best

    print("\nChosen: HGB")
    return hgb_best


def train_and_save_models(
    dataset_obj: Any,
    BodyLandmark,
    *,
    save_path: str = "./data/icaro_models.joblib",
    use_world: bool = False,
    window: int = 7,
    margin: int = 6,
    falled_tail_frames: int = 6,
    horizontal_min_quality: float = 0.20,
    horizontal_min_good_keypoints: int = 4,
    fall_min_vis_point: float = 0.55,
    fall_min_pres_point: float = 0.55,
    fall_min_required_core_points: int = 3,
    within_fall_only_for_horizontal: bool = True,
    random_state: int = 42,
    n_iter_search: int = 30,
) -> Dict[str, Any]:
    videos = load_dataset_from_json_obj(dataset_obj)
    pose_col = "pose_world_landmarks" if use_world else "pose_landmarks"

    Xf, yf, gf, wf = build_xy_binary_fall(
        videos,
        BodyLandmark,
        pose_col=pose_col,
        window=window,
        margin=margin,
        min_vis_point=fall_min_vis_point,
        min_pres_point=fall_min_pres_point,
        min_required_core_points=fall_min_required_core_points,
    )
    print("FALL support:", np.bincount(yf, minlength=2))
    fall_model = train_binary_model_random_search_best_of_two(
        Xf, yf, gf, wf,
        random_state=random_state,
        n_iter_search=n_iter_search,
        scoring="f1",
    )

    Xh, yh, gh, wh = build_xy_binary_horizontal_from_final_falled(
        videos,
        BodyLandmark,
        pose_col=pose_col,
        window=window,
        falled_tail_frames=falled_tail_frames,
        min_quality=horizontal_min_quality,
        min_good_keypoints=horizontal_min_good_keypoints,
        within_fall_only=within_fall_only_for_horizontal,
    )
    print("HORIZONTAL support:", np.bincount(yh, minlength=2))
    horizontal_model = train_binary_model_random_search_best_of_two(
        Xh, yh, gh, wh,
        random_state=random_state + 1,
        n_iter_search=n_iter_search,
        scoring="f1",
    )

    bundle = {
        "fall_model": fall_model,
        "horizontal_model": horizontal_model,
        "cfg": {
            "use_world": use_world,
            "pose_col": pose_col,
            "window": window,
            "margin": margin,
            "falled_tail_frames": falled_tail_frames,
            "horizontal_min_quality": horizontal_min_quality,
            "horizontal_min_good_keypoints": horizontal_min_good_keypoints,
            "fall_min_vis_point": fall_min_vis_point,
            "fall_min_pres_point": fall_min_pres_point,
            "fall_min_required_core_points": fall_min_required_core_points,
            "within_fall_only_for_horizontal": within_fall_only_for_horizontal,
        },
    }
    joblib.dump(bundle, save_path)
    return bundle


def load_models(path: str = "./data/icaro_models.joblib") -> Dict[str, Any]:
    return joblib.load(path)


def _proba_pos(model: Pipeline, X: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p is not None and p.shape[1] >= 2:
            return float(p[0, 1])
    if hasattr(model, "decision_function"):
        s = float(model.decision_function(X)[0])
        return float(1.0 / (1.0 + np.exp(-s)))
    pred = int(model.predict(X)[0])
    return float(pred)





if __name__ == "__main__":
    dataset_obj = json.loads(open("../data/archive.json", "r", encoding="utf-8").read())
    from util_landmarks import BodyLandmark

    bundle = train_and_save_models(
        dataset_obj,
        BodyLandmark,
        save_path="../data/icaro_models.joblib",
        use_world=False,
        window=7,
        margin=6,
        falled_tail_frames=6,
        horizontal_min_quality=0.20,
        horizontal_min_good_keypoints=4,
        fall_min_vis_point=0.55,
        fall_min_pres_point=0.55,
        fall_min_required_core_points=3,
        within_fall_only_for_horizontal=True,
        random_state=42,
        n_iter_search=30,
    )

