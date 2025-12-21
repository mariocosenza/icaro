import logging
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import inspect
# Make sure you have util_landmarks.py in the same folder
from util_landmarks import BodyLandmark
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NORMAL = 0
FALL = 1

NOT_HORIZONTAL = 0
HORIZONTAL = 1

# --- Type Definitions & Helpers ---

# Restore PoseCell for backward compatibility with your other scripts
PoseCell = List[Dict[str, float]]
Pose33 = PoseCell


def _proba_pos(est: Any, X: np.ndarray) -> np.ndarray:
    """Helper to get positive class probabilities safely."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    if hasattr(est, "decision_function"):
        return est.decision_function(X)
    return est.predict(X)


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

    if safe.shape[0] > 1:
        diffs = np.diff(safe, axis=0)
        diffs_filled = np.nan_to_num(diffs, nan=0.0)
        avg_speed = np.mean(np.abs(diffs_filled), axis=0)
    else:
        avg_speed = np.zeros_like(mean)

    out = np.concatenate([mean, std, mn, mx, delta, avg_speed], axis=0).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out


def windowize_last_label(
        feats: List[np.ndarray],
        labels: List[int],
        qual: List[float],
        window: int,
        min_window_quality: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(feats) < window:
        return np.zeros((0, 1), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32)

    arr = np.stack(feats, axis=0)
    cls = np.asarray(labels, dtype=np.int32)
    q = np.asarray(qual, dtype=np.float32)

    X, y, w = [], [], []
    for i in range(window - 1, len(arr)):
        wq = float(np.nanmean(q[i - window + 1: i + 1]))

        if wq < min_window_quality:
            continue

        w_feats = arr[i - window + 1: i + 1]
        label = int(cls[i])

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
                max_depth=4,
                learning_rate=0.08,
                max_iter=500,
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

    # Load ALL 33 keypoints so indices match
    vis = np.array([float(pose33[i].get("visibility", 0.0)) for i in range(33)], dtype=np.float32)
    pres = np.array([float(pose33[i].get("presence", 0.0)) for i in range(33)], dtype=np.float32)

    # Calculate quality only on important keys
    key_idx = [NOSE, L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK]
    q = float(np.mean(vis[key_idx] * pres[key_idx]))

    if q < min_quality:
        return None

    xs = np.array([float(lm["x"]) for lm in pose33])
    ys = np.array([float(lm["y"]) for lm in pose33])

    def get_p(i):
        return np.array([xs[i], ys[i]], dtype=np.float32)

    nose_p = get_p(NOSE)
    sh_mid = (get_p(L_SH) + get_p(R_SH)) / 2.0
    hip_mid = (get_p(L_HIP) + get_p(R_HIP)) / 2.0

    # Safely access ankles using indices
    if vis[L_ANK] > 0.4 and vis[R_ANK] > 0.4:
        foot_mid = (get_p(L_ANK) + get_p(R_ANK)) / 2.0
    else:
        # Fallback to hips if feet are cut off
        foot_mid = hip_mid + (hip_mid - sh_mid)

    body_y_span = abs(foot_mid[1] - nose_p[1])
    body_x_span = abs(foot_mid[0] - nose_p[0]) + abs(sh_mid[0] - hip_mid[0])

    verticality_ratio = body_y_span / (body_x_span + 1e-6)

    nose_to_hip = hip_mid[1] - nose_p[1]
    hip_to_foot = foot_mid[1] - hip_mid[1]

    dx = hip_mid[0] - sh_mid[0]
    dy = hip_mid[1] - sh_mid[1]
    angle_2d = float(np.arctan2(abs(dx), abs(dy)))

    bbox_h = (np.max(ys) - np.min(ys))
    bbox_w = (np.max(xs) - np.min(xs))
    bbox_ar = bbox_w / (bbox_h + 1e-6)

    feats = np.array(
        [
            float(verticality_ratio),
            float(angle_2d),
            float(nose_to_hip),
            float(hip_to_foot),
            float(bbox_ar),
            float(body_y_span),
            float(nose_p[1]),
            float(sh_mid[1]),
            float(q)
        ],
        dtype=np.float32,
    )

    return feats, float(np.clip(q, 0.0, 1.0))


def extract_frame_features_fall(
        pose33: Pose33,
        BodyLandmark,
        *,
        min_vis_point: float = 0.40,
        min_pres_point: float = 0.40,
        min_required_core_points: int = 3,
) -> Optional[Tuple[np.ndarray, float]]:
    idx = lambda e: int(e.value) if hasattr(e, "value") else int(e)

    L_SH = idx(BodyLandmark.LEFT_SHOULDER)
    R_SH = idx(BodyLandmark.RIGHT_SHOULDER)
    NOSE = idx(BodyLandmark.NOSE)
    L_HIP = idx(BodyLandmark.LEFT_HIP)
    R_HIP = idx(BodyLandmark.RIGHT_HIP)

    core = [L_SH, R_SH, NOSE, L_HIP, R_HIP]

    vis = np.array([float(pose33[i].get("visibility", 0.0)) for i in range(33)])
    ok = vis >= min_vis_point

    if np.sum(ok[core]) < min_required_core_points:
        return None

    q = float(np.mean(vis[core]))

    ys = np.array([float(lm["y"]) for lm in pose33])
    xs = np.array([float(lm["x"]) for lm in pose33])

    def get_y(indices):
        valid = [i for i in indices if ok[i]]
        if not valid: return np.nan
        return float(np.mean(ys[valid]))

    def get_x(indices):
        valid = [i for i in indices if ok[i]]
        if not valid: return np.nan
        return float(np.mean(xs[valid]))

    y_shoulders = get_y([L_SH, R_SH])
    y_hips = get_y([L_HIP, R_HIP])
    y_nose = get_y([NOSE])

    if np.isnan(y_hips): y_hips = y_shoulders + 0.1
    if np.isnan(y_shoulders): y_shoulders = y_hips - 0.1
    if np.isnan(y_nose): y_nose = y_shoulders - 0.1

    torso_len = abs(y_hips - y_shoulders)

    valid_ys = ys[ok]
    valid_xs = xs[ok]
    if len(valid_ys) > 2:
        h = np.max(valid_ys) - np.min(valid_ys)
        w = np.max(valid_xs) - np.min(valid_xs)
        ar = w / (h + 1e-6)
    else:
        h, w, ar = 0.0, 0.0, 0.0

    feats = np.array(
        [
            float(y_nose),
            float(y_hips),
            float(torso_len),
            float(ar),
            float(h),
            float(y_nose - y_hips)
        ],
        dtype=np.float32
    )

    return feats, float(np.clip(q, 0.0, 1.0))


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
        min_window_quality: float = 0.0,
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
                Xw, yw, ww = windowize_last_label(
                    feats_seg, cls_seg, q_seg,
                    window=window,
                    min_window_quality=min_window_quality
                )
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

            if (v.start - margin) <= fi < v.start:
                if last_fi is not None and fi - last_fi > step:
                    flush()
                    last_fi = None
                continue

            fall_end_strict = v.end - 2
            is_falling = (v.start <= fi <= fall_end_strict)
            label = FALL if is_falling else NORMAL

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

            feats_seg.append(feat)
            cls_seg.append(label)
            q_seg.append(qual)
            last_fi = fi

        flush()

    if not X_all:
        raise ValueError("No samples for FALL model. Relax per-point thresholds, quality gate, or window/margin.")

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
        min_window_quality: float = 0.0,
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
                Xw, yw, ww = windowize_last_label(
                    feats_seg, cls_seg, q_seg,
                    window=window,
                    min_window_quality=min_window_quality
                )
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
        raise ValueError("No samples for HORIZONTAL model. Lower thresholds/window/quality or increase tail frames.")

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

    if len(X_train) == 0:
        raise ValueError("Training set is empty. Reduce window size, relax quality thresholds, or add more data.")

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
        "clf__batch_size": ["auto", 32],
        "clf__max_iter": [300, 500, 700],
    }

    logging.info(f"Train support: {np.bincount(y_train, minlength=2)} Test support: {np.bincount(y_test, minlength=2)}")

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

    logging.info("=== HGB ===")
    logging.info(f"F1(pos): {f1_hgb:.4f}")
    logging.info(f"Confusion:\n{confusion_matrix(y_test, pred_hgb)}")
    logging.info(classification_report(y_test, pred_hgb, digits=4, target_names=["no_fall", "fall"]))

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

    logging.info("=== MLP ===")
    logging.info(f"F1(pos): {f1_mlp:.4f}")
    logging.info(f"Confusion:\n{confusion_matrix(y_test, pred_mlp)}")
    logging.info(classification_report(y_test, pred_mlp, digits=4, target_names=["no_fall", "fall"]))

    if f1_mlp > f1_hgb:
        logging.info("Chosen: MLP")
        return mlp_best

    logging.info("Chosen: HGB")
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
        min_window_quality: Union[float, str] = "medium",
        random_state: int = 42,
        n_iter_search: int = 30,
) -> Dict[str, Any]:
    videos = load_dataset_from_json_obj(dataset_obj)
    pose_col = "pose_world_landmarks" if use_world else "pose_landmarks"

    quality_levels = {"low": 0.30, "medium": 0.60, "high": 0.90}
    if isinstance(min_window_quality, str):
        resolved_window_quality = quality_levels.get(min_window_quality.lower(), 0.60)
    else:
        resolved_window_quality = float(min_window_quality)

    logging.info(f"Using min_window_quality: {resolved_window_quality} (Input: {min_window_quality})")

    Xf, yf, gf, wf = build_xy_binary_fall(
        videos,
        BodyLandmark,
        pose_col=pose_col,
        window=window,
        margin=margin,
        min_vis_point=fall_min_vis_point,
        min_pres_point=fall_min_pres_point,
        min_required_core_points=fall_min_required_core_points,
        min_window_quality=resolved_window_quality,
    )
    logging.info(f"FALL support: {np.bincount(yf, minlength=2)}")
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
        min_window_quality=resolved_window_quality,
    )
    logging.info(f"HORIZONTAL support: {np.bincount(yh, minlength=2)}")
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
            "min_window_quality": resolved_window_quality,
        },
    }
    joblib.dump(bundle, save_path)
    return bundle


def load_models(path: str = "./data/icaro_models.joblib") -> Dict[str, Any]:
    return joblib.load(path)


if __name__ == "__main__":
    dataset_obj = json.loads(open("../data/archive.json", "r", encoding="utf-8").read())

    bundle = train_and_save_models(
        dataset_obj,
        BodyLandmark,
        save_path="../data/icaro_models.joblib",
        use_world=False,
        window=9,
        margin=6,
        falled_tail_frames=10,
        horizontal_min_quality=0.50,
        horizontal_min_good_keypoints=4,
        fall_min_vis_point=0.60,
        fall_min_pres_point=0.60,
        fall_min_required_core_points=5,
        within_fall_only_for_horizontal=True,
        random_state=42,
        n_iter_search=40,
        min_window_quality="high",
    )