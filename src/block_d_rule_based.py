"""
block_d_rule_based.py

Block D - Day 13: Rule-Based anomaly baseline on AI City Track 4.
Reusable functions for both the Day 13 notebook and pipeline_controller.run_block_d().

Pipeline:
    event_loader -> video_resolver -> trajectory_loader ->
    [F11 fragment_filter -> F22 interpolation -> F16 warmup_strip] ->
    f2_extractor -> scaler (no fit) -> rule_based_scorer -> evaluator

Student : MANJOO Ameera Najla | M01014463
Module  : CST3990 Undergraduate Individual Project - Block D
"""

from __future__ import annotations

import copy
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_JSON = REPO_ROOT / "data" / "ai_city" / "events.json"
TRAJ_BASE = REPO_ROOT / "logs" / "block_b" / "trajectories"
SCALER_PKL = REPO_ROOT / "models" / "minmax_scaler.pkl"
BLOCK_D_LOG_DIR = REPO_ROOT / "logs" / "block_d"
SYNC_CHECK_DIR = BLOCK_D_LOG_DIR / "sync_checks"
RESULTS_CSV = BLOCK_D_LOG_DIR / "block_d_results.csv"
VERIF_LOG_CSV = BLOCK_D_LOG_DIR / "event_slicer_verification.csv"

MIN_TRACK_LENGTH = 15
MAX_INTERP_GAP = 3
CONTEXT_SEC = 5.0
MIN_FRAMES_WARN = 15
Z_THRESH = 2.0
IQR_MULT = 1.5
SPEED_WINDOW_SEC = 0.2
TRACKER = "sort"
F2_IDX = [0, 1]

# ── Day 14 constants ──────────────────────────────────────────────────────────
OCSVM_PKL = REPO_ROOT / "logs" / "block_c" / "ocsvm_trained_best.pkl"
IF_PKL = BLOCK_D_LOG_DIR / "isolation_forest_f2_train_normal.pkl"
IF_TRAIN_META_JSON = BLOCK_D_LOG_DIR / "isolation_forest_training_metadata.json"
METADATA_JSON = REPO_ROOT / "logs" / "metadata.json"
BLOCK_C_LOG_DIR = REPO_ROOT / "logs" / "block_c"
EVENT_PREDS_CSV = BLOCK_D_LOG_DIR / "block_d_event_predictions.csv"
SKIP_REASONS_CSV_PATH = BLOCK_D_LOG_DIR / "block_d_skip_reasons.csv"
BLOCK_D_META_JSON = BLOCK_D_LOG_DIR / "block_d_metadata.json"
F2_COLS = ["vel_px_sec", "vel_px_sec_smooth"]

IF_N_ESTIMATORS = 100
IF_CONTAMINATION = "auto"
IF_RANDOM_STATE = 42


def audit_config(cfg: dict) -> None:
    """
    Assert Block D config is scoped to AI City only.
    """
    dataset_val = cfg.get("dataset", cfg.get("datasets", {}).get("primary", ""))
    assert "ai_city" in str(dataset_val).lower(), (
        f"Config audit FAILED: dataset field is '{dataset_val}'. "
        "Expected 'ai_city' or equivalent nested config."
    )

    cfg_str = json.dumps(cfg).lower()
    for banned in ("tudat", "tu_dat", "tailgating"):
        hits = cfg_str.count(banned)
        assert hits == 0, (
            f"Config audit FAILED: banned substring '{banned}' found {hits} time(s). "
            "Block D scope is AI City Track 4 only."
        )

    print("[Block D] Config audit passed - dataset: AI City, no TU-DAT / tailgating references.")


def load_events(events_path: str | Path = EVENTS_JSON) -> List[dict]:
    """
    Load AI City Track 4 anomaly events from the real frame-based schema.
    """
    events_path = Path(events_path)
    if not events_path.exists():
        raise FileNotFoundError(f"events.json not found at {events_path}")

    with open(events_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("events.json must be a top-level dict.")
    if "events" not in data:
        raise KeyError("events.json missing top-level 'events' key.")

    events = data["events"]
    if not isinstance(events, list) or not events:
        raise ValueError("events.json['events'] must be a non-empty list.")

    required = {
        "event_id",
        "video_id",
        "video_file",
        "original_fps",
        "anomaly_start_frame",
        "anomaly_end_frame",
        "anomaly_class_name",
        "used_in_block_d",
        "sync_ok",
    }

    cleaned = []
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"events[{idx}] is not a dict.")
        missing = sorted(required - set(event.keys()))
        if missing:
            raise KeyError(f"events[{idx}] missing required keys: {missing}")
        if str(event.get("anomaly_class_name", "")).lower() != "anomaly":
            continue
        if not bool(event.get("used_in_block_d", False)):
            continue

        start_frame = int(event["anomaly_start_frame"])
        end_frame = int(event["anomaly_end_frame"])
        if end_frame < start_frame:
            raise ValueError(
                f"Invalid frame range for event_id={event['event_id']}: "
                f"start={start_frame}, end={end_frame}"
            )

        event_copy = dict(event)
        event_copy["video_id"] = str(event["video_id"])
        event_copy["video_file"] = str(event["video_file"])
        event_copy["anomaly_start_frame"] = start_frame
        event_copy["anomaly_end_frame"] = end_frame
        cleaned.append(event_copy)

    if not cleaned:
        raise ValueError(
            "No valid anomaly events found after filtering to "
            "anomaly_class_name='anomaly' and used_in_block_d=True."
        )

    sync_ok_count = sum(1 for e in cleaned if bool(e.get("sync_ok", False)))
    unique_videos = sorted({e["video_id"] for e in cleaned})
    print(
        "[Block D] Events loaded: "
        f"{len(cleaned)} anomaly events across {len(unique_videos)} videos "
        f"({sync_ok_count} sync_ok)."
    )
    return cleaned


def get_actual_fps(video_path: str | Path) -> float:
    """
    Extract FPS from video at runtime. Never returns a hardcoded value.
    """
    video_path = str(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if actual_fps <= 0 or actual_fps > 120:
        raise ValueError(
            f"Suspicious FPS={actual_fps} for {video_path}. "
            "AI City clips must have valid FPS (0 < fps <= 120)."
        )
    return float(actual_fps)


def _candidate_ai_city_roots() -> List[Path]:
    """Return likely AI City dataset roots across local and Colab layouts."""
    return [
        Path("/content/drive/MyDrive/CST3990/datasets/ai_city"),
        Path("/content/drive/MyDrive/CST3990/dataset/ai_city"),
        REPO_ROOT / "data" / "ai_city",
        REPO_ROOT / "dataset" / "ai_city",
    ]


def build_video_index(search_roots: Optional[List[Path]] = None) -> Dict[str, List[Path]]:
    """
    Recursively index AI City .mp4 files by filename.
    """
    roots = search_roots or _candidate_ai_city_roots()
    index: Dict[str, List[Path]] = {}

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.mp4"):
            key = path.name.lower()
            index.setdefault(key, []).append(path)

    total = sum(len(v) for v in index.values())
    print(f"[Block D] Video index built: {total} files across {len(index)} unique names.")
    return index


def resolve_video_path(video_file: str, video_index: Dict[str, List[Path]]) -> Optional[Path]:
    """
    Resolve an event's video_file against the pre-built recursive filename index.
    """
    candidates = video_index.get(str(video_file).lower(), [])
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]


def verify_frame_sync(
    video_path: str | Path,
    annotation_frame_id: int,
    expected_content_note: str,
    out_dir: str | Path = SYNC_CHECK_DIR,
) -> bool:
    """
    Fix F12 - verify that the annotation frame numbering aligns with the decoded video.
    Saves a JPEG of the decoded frame for manual inspection.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(str(video_path)):
        print(f"[Fix F12] SKIP - video not found: {video_path}")
        return False

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, annotation_frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"[Fix F12] WARNING - could not decode frame {annotation_frame_id} from {video_path}")
        return False

    clip_id = Path(video_path).stem
    out_path = out_dir / f"sync_check_{clip_id}_frame{annotation_frame_id}.jpg"
    cv2.imwrite(str(out_path), frame)
    print(f"[Fix F12] Saved {out_path}")
    print(f"[Fix F12] Manually verify it matches: {expected_content_note}")
    return True


def extract_event_window(
    video_path: str | Path,
    anomaly_start_frame: int,
    anomaly_end_frame: int,
    actual_fps: float,
    context_sec: float = CONTEXT_SEC,
) -> List[Tuple[int, np.ndarray]]:
    """
    Extract frames for a frame-based event window with context padding.
    Event bounds are already frame-based and must not be converted from seconds.
    """
    context_frames = int(context_sec * actual_fps)
    jump_to = max(0, int(anomaly_start_frame) - context_frames)
    window_end = int(anomaly_end_frame) + context_frames

    if not os.path.exists(str(video_path)):
        warnings.warn(
            f"[EventSlicer] Video not found: {video_path}. Returning empty frame list.",
            stacklevel=2,
        )
        return []

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, jump_to)

    frames: List[Tuple[int, np.ndarray]] = []
    while True:
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_pos > window_end:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((current_pos - 1, frame))

    cap.release()
    return frames


def find_trajectory_path(video_id: str, tracker: str = TRACKER) -> Optional[Path]:
    """
    Locate the Block B *_final.json trajectory file for a given AI City video_id.
    """
    video_id = str(video_id)
    candidate = TRAJ_BASE / f"{video_id}_{tracker}_final.json"
    if candidate.exists():
        return candidate

    exact_stem = f"{video_id}_{tracker}_final"
    for path in TRAJ_BASE.glob(f"*{tracker}*final.json"):
        if path.stem == exact_stem:
            return path

    for path in TRAJ_BASE.glob(f"*{tracker}*final.json"):
        parts = path.stem.split("_")
        if parts and parts[0] == video_id and tracker in path.stem:
            return path

    for path in TRAJ_BASE.glob(f"*{video_id}*{tracker}*final.json"):
        return path

    return None


def load_trajectory_df(traj_path: Path) -> pd.DataFrame:
    """
    Load a Block B *_final.json trajectory file and flatten it to a DataFrame.
    """
    with open(traj_path, encoding="utf-8") as f:
        data = json.load(f)

    tracks_raw = data.get("tracks", {})
    rows = []
    if isinstance(tracks_raw, dict):
        for tid_str, entries in tracks_raw.items():
            tid = int(tid_str)
            for entry in entries:
                row = copy.deepcopy(entry)
                row["track_id"] = tid
                rows.append(row)
    elif isinstance(tracks_raw, list):
        rows = tracks_raw
    else:
        raise ValueError(f"Unexpected 'tracks' type {type(tracks_raw)} in {traj_path}")

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "frame_idx",
                "track_id",
                "cx",
                "cy",
                "velocity_norm",
                "conf",
                "is_interpolated",
            ]
        )

    if "interpolated" in df.columns and "is_interpolated" not in df.columns:
        df = df.rename(columns={"interpolated": "is_interpolated"})

    for col in ("frame_idx", "track_id", "cx", "cy", "velocity_norm", "conf", "is_interpolated"):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from trajectory {traj_path.name}")

    df["frame_idx"] = df["frame_idx"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    return df


def filter_short_tracks(df: pd.DataFrame, min_length: int = MIN_TRACK_LENGTH) -> pd.DataFrame:
    """
    Fix F11 - fragment filter.
    Remove tracks with fewer than min_length genuine detections.

    Robust to bool / string / numeric encodings in is_interpolated.
    """
    if df.empty:
        return df.copy()

    df = df.copy()

    if "is_interpolated" not in df.columns:
        df["is_interpolated"] = False

    def _to_bool_interp(x):
        s = str(x).strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no", "nan", "none", ""}:
            return False
        return bool(x)

    df["is_interpolated_norm"] = df["is_interpolated"].map(_to_bool_interp)

    genuine = df.loc[~df["is_interpolated_norm"]].copy()
    if genuine.empty:
        logger.warning(
            "[F11] No genuine detections found after normalizing is_interpolated. "
            "Returning empty dataframe."
        )
        return df.iloc[0:0].copy()

    counts = genuine.groupby("track_id")["frame_idx"].count()
    valid = counts[counts >= min_length].index
    filtered = df[df["track_id"].isin(valid)].copy()
    if "is_interpolated_norm" in filtered.columns:
        filtered = filtered.drop(columns=["is_interpolated_norm"])
    n_removed = df["track_id"].nunique() - filtered["track_id"].nunique()
    if n_removed > 0:
        logger.debug("[F11] Removed %d short track(s) (< %d genuine frames).", n_removed, min_length)
    return filtered


def interpolate_gaps(df: pd.DataFrame, max_gap: int = MAX_INTERP_GAP) -> pd.DataFrame:
    """
    Fix F22 - Track interpolation with guards against empty concat cases.
    """
    if df.empty:
        return df.copy()

    result_chunks: List[pd.DataFrame] = []

    for tid, grp in df.groupby("track_id"):
        grp_sorted = grp.sort_values("frame_idx").reset_index(drop=True)
        if grp_sorted.empty:
            continue

        result_chunks.append(grp_sorted)
        track_frames = sorted(set(grp_sorted["frame_idx"].tolist()))

        for i in range(len(track_frames) - 1):
            f_prev = track_frames[i]
            f_next = track_frames[i + 1]
            gap = f_next - f_prev - 1

            if not (0 < gap <= max_gap):
                continue

            prev_row = grp_sorted[grp_sorted["frame_idx"] == f_prev].iloc[0]
            next_row = grp_sorted[grp_sorted["frame_idx"] == f_next].iloc[0]
            interp_rows = []

            for f_interp in range(f_prev + 1, f_next):
                t = (f_interp - f_prev) / (f_next - f_prev)
                interp_rows.append(
                    {
                        "frame_idx": f_interp,
                        "track_id": tid,
                        "cx": float(prev_row["cx"] + t * (next_row["cx"] - prev_row["cx"])),
                        "cy": float(prev_row["cy"] + t * (next_row["cy"] - prev_row["cy"])),
                        "velocity_norm": np.nan,
                        "conf": -1.0,
                        "is_interpolated": True,
                    }
                )

            if interp_rows:
                result_chunks.append(pd.DataFrame(interp_rows))

    if not result_chunks:
        return df.iloc[0:0].copy()

    out = pd.concat(result_chunks, ignore_index=True)
    out = out.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)
    return out


def strip_warmup(df: pd.DataFrame, actual_fps: float) -> pd.DataFrame:
    """
    Fix F16 - Warmup strip using max(10, int(0.4 * actual_fps)).
    """
    if df.empty:
        return df.copy()

    warmup_frames = max(10, int(0.4 * actual_fps))
    min_frame = int(df["frame_idx"].min())
    warmup_boundary = min_frame + warmup_frames
    return df[df["frame_idx"] > warmup_boundary].copy()


def extract_f2_features(
    df: pd.DataFrame,
    actual_fps: float,
    speed_window_sec: float = SPEED_WINDOW_SEC,
) -> pd.DataFrame:
    """
    Extract F2_only features: vel_px_sec and vel_px_sec_smooth.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "frame_idx",
                "track_id",
                "vel_px_sec",
                "vel_px_sec_smooth",
                "is_interpolated",
                "cx",
                "cy",
            ]
        )

    speed_window = max(1, int(speed_window_sec * actual_fps))
    df = df.copy()
    df["vel_px_sec"] = df["velocity_norm"].fillna(0.0) * actual_fps

    smooth_chunks = []
    for _, grp in df.groupby("track_id"):
        grp_sorted = grp.sort_values("frame_idx").copy()
        grp_sorted["vel_px_sec_smooth"] = (
            grp_sorted["vel_px_sec"]
            .where(grp_sorted["is_interpolated"] != True)  # noqa: E712
            .rolling(window=speed_window, min_periods=1)
            .mean()
            .fillna(grp_sorted["vel_px_sec"])
        )
        smooth_chunks.append(grp_sorted)

    if not smooth_chunks:
        return pd.DataFrame(
            columns=[
                "frame_idx",
                "track_id",
                "vel_px_sec",
                "vel_px_sec_smooth",
                "is_interpolated",
                "cx",
                "cy",
            ]
        )

    out = pd.concat(smooth_chunks, ignore_index=True)
    out = out.sort_values(["frame_idx", "track_id"]).reset_index(drop=True)
    return out[
        ["frame_idx", "track_id", "vel_px_sec", "vel_px_sec_smooth", "is_interpolated", "cx", "cy"]
    ]


def aggregate_f2_per_frame(df_f2: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-track F2 features to per-frame means."""
    if df_f2.empty:
        return pd.DataFrame(columns=["frame_idx", "vel_px_sec", "vel_px_sec_smooth"])

    agg = (
        df_f2.groupby("frame_idx")[["vel_px_sec", "vel_px_sec_smooth"]]
        .mean()
        .reset_index()
    )
    return agg.sort_values("frame_idx").reset_index(drop=True)


def load_scaler(scaler_path: str | Path = SCALER_PKL):
    """
    Load the Block C fitted MinMaxScaler.
    Must be called once outside all loops - never fit in Block D.
    """
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"MinMaxScaler not found at {scaler_path}")
    scaler = joblib.load(scaler_path)
    print(f"[Block D] Scaler loaded: {scaler_path.name}, fitted on {scaler.n_features_in_} features.")
    return scaler


def scale_f2_and_log(X_raw: np.ndarray, clip_id: str, scaler) -> np.ndarray:
    """
    Scale F2_only features using the Block C MinMaxScaler and log domain shift.
    """
    f2_min = scaler.data_min_[F2_IDX]
    f2_range = scaler.data_range_[F2_IDX]

    with np.errstate(divide="ignore", invalid="ignore"):
        X_scaled = (X_raw - f2_min) / np.where(f2_range == 0, 1.0, f2_range)

    n_clipped = int(((X_scaled > 1.0) | (X_scaled < 0.0)).sum())
    pct_clipped = n_clipped / X_scaled.size * 100.0 if X_scaled.size else 0.0
    X_scaled = np.clip(X_scaled, 0.0, 1.0)

    flag = " WARNING: HIGH DOMAIN SHIFT" if pct_clipped > 20.0 else ""
    print(
        f"[Fix F18] [{clip_id}] Domain-shift indicator: "
        f"{pct_clipped:.1f}% of F2 values clipped to [0,1]{flag}"
    )
    if pct_clipped > 20.0:
        print(
            f"[Fix F18] WARNING: >20% clipping for {clip_id}. "
            "AI City highway speeds likely exceed UA-DETRAC normal training range. "
            "Scaled rule-based (IQR) results may be biased - interpret with caution."
        )

    return X_scaled


def rule_based_score(
    X_raw: np.ndarray,
    X_scaled: np.ndarray,
    z_thresh: float = Z_THRESH,
    iqr_mult: float = IQR_MULT,
) -> Dict[str, np.ndarray]:
    """
    Apply Z-score and IQR thresholding to produce frame-level anomaly masks.
    """
    results: Dict[str, np.ndarray] = {}

    if X_raw.shape[0] < 2:
        z_mask = np.zeros(len(X_raw), dtype=bool)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = zscore(X_raw, axis=0)
        z_mask = (np.abs(z) > z_thresh).any(axis=1)
    results["zscore"] = z_mask

    if X_scaled.shape[0] < 4:
        iqr_mask = np.zeros(len(X_scaled), dtype=bool)
    else:
        q1 = np.percentile(X_scaled, 25, axis=0)
        q3 = np.percentile(X_scaled, 75, axis=0)
        iqr = q3 - q1
        iqr_mask = (
            (X_scaled < q1 - iqr_mult * iqr) |
            (X_scaled > q3 + iqr_mult * iqr)
        ).any(axis=1)
    results["iqr"] = iqr_mask

    return results


def event_level_prediction(frame_mask: np.ndarray) -> int:
    """Convert a per-frame anomaly mask to an event-level binary prediction."""
    return int(frame_mask.any())


def evaluate_per_method(
    y_true: List[int],
    y_pred: Dict[str, List[int]],
) -> pd.DataFrame:
    """
    Compute Precision, Recall, and F1 for anomaly-only Day 13 methods.
    """
    rows = []
    method_map = {
        "zscore": "Rule-Based (Z-score)",
        "iqr": "Rule-Based (IQR)",
    }

    for method_key, method_label in method_map.items():
        yp = y_pred.get(method_key, [])
        if not y_true or not yp:
            continue

        p = precision_score(y_true, yp, zero_division=0)
        r = recall_score(y_true, yp, zero_division=0)
        f = f1_score(y_true, yp, zero_division=0)

        print(f"  [{method_label}]: P={p:.3f}  R={r:.3f}  F1={f:.3f}")
        rows.append(
            {
                "Method": method_label,
                "Dataset": "AI City Track 4",
                "Event_Type": "anomaly",
                "Precision": round(p, 4),
                "Recall": round(r, 4),
                "F1": round(f, 4),
                "FAR (normal clips)": "TBD",
            }
        )

    return pd.DataFrame(rows)


RESULTS_TABLE_SCHEMA = [
    "Method",
    "Dataset",
    "Event_Type",
    "Precision",
    "Recall",
    "F1",
    "FAR (normal clips)",
]


def build_results_table(rule_based_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Day 13 Block D results table.
    """
    placeholder_rows = []
    for method in ("OC-SVM", "Isolation Forest"):
        placeholder_rows.append(
            {
                "Method": method,
                "Dataset": "AI City Track 4",
                "Event_Type": "anomaly",
                "Precision": "-",
                "Recall": "-",
                "F1": "-",
                "FAR (normal clips)": "TBD",
            }
        )

    frames = [rule_based_df] if not rule_based_df.empty else []
    frames.append(pd.DataFrame(placeholder_rows))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RESULTS_TABLE_SCHEMA)
    combined = combined[RESULTS_TABLE_SCHEMA]

    BLOCK_D_LOG_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RESULTS_CSV, index=False)
    print(f"[Block D] Results table saved: {RESULTS_CSV}")
    return combined


def append_verification_log(
    event_id: str,
    video_id: str,
    video_file: str,
    anomaly_start_frame: int,
    computed_start_frame: int,
    visual_match: str = "UNCHECKED",
    notes: str = "",
    short_window: bool = False,
) -> None:
    """
    Append one row to the event slicer verification CSV.
    """
    BLOCK_D_LOG_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "video_id": video_id,
                "video_file": video_file,
                "anomaly_start_frame": anomaly_start_frame,
                "computed_start_frame": computed_start_frame,
                "visual_match": visual_match,
                "short_window": short_window,
                "notes": notes,
            }
        ]
    )
    write_header = not VERIF_LOG_CSV.exists()
    row.to_csv(VERIF_LOG_CSV, mode="a", header=write_header, index=False)


def run_rule_based_baseline(
    cfg_d: dict,
    cfg_c: dict,
    events_path: str | Path = EVENTS_JSON,
    scaler_path: str | Path = SCALER_PKL,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full Day 13 Block D rule-based pipeline for anomaly-only, frame-based AI City events.
    """
    audit_config(cfg_d)
    anomaly_events = load_events(events_path)
    video_index = build_video_index()
    scaler = load_scaler(scaler_path)

    speed_window_sec = float(cfg_c.get("f2", {}).get("speed_window_sec", SPEED_WINDOW_SEC))
    context_sec = CONTEXT_SEC

    y_true: List[int] = []
    y_pred: Dict[str, List[int]] = {"zscore": [], "iqr": []}

    for event in anomaly_events:
        event_id = str(event["event_id"])
        video_id = str(event["video_id"])
        video_file = str(event["video_file"])
        anomaly_start_frame = int(event["anomaly_start_frame"])
        anomaly_end_frame = int(event["anomaly_end_frame"])

        if verbose:
            print(f"\n[Block D] Processing anomaly event {event_id} (video_id={video_id}, file={video_file})")

        video_path = resolve_video_path(video_file, video_index)
        if video_path is None:
            msg = f"Video not found in indexed AI City roots for file '{video_file}'"
            print(f"  SKIP (video): {msg}")
            append_verification_log(
                event_id,
                video_id,
                video_file,
                anomaly_start_frame,
                anomaly_start_frame,
                notes=msg,
            )
            continue

        try:
            actual_fps = get_actual_fps(video_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP (FPS): {e}")
            append_verification_log(
                event_id,
                video_id,
                video_file,
                anomaly_start_frame,
                anomaly_start_frame,
                notes=str(e),
            )
            continue

        computed_start_frame = anomaly_start_frame
        context_frames = int(context_sec * actual_fps)
        jump_to = max(0, anomaly_start_frame - context_frames)
        window_end = anomaly_end_frame + context_frames

        event_dur_frames = anomaly_end_frame - anomaly_start_frame + 1
        warmup_frames = max(10, int(0.4 * actual_fps))
        short_window = (event_dur_frames - warmup_frames) < MIN_FRAMES_WARN
        if short_window:
            print(
                f"  [WARNING] Short event window: ~{event_dur_frames} frames, "
                f"~{event_dur_frames - warmup_frames} after warmup."
            )

        traj_path = find_trajectory_path(video_id)
        if traj_path is None:
            msg = f"No _{TRACKER}_final.json trajectory found for video_id={video_id}"
            print(f"  SKIP (trajectory): {msg}")
            append_verification_log(
                event_id,
                video_id,
                video_file,
                anomaly_start_frame,
                computed_start_frame,
                short_window=short_window,
                notes=msg,
            )
            continue

        try:
            df_traj = load_trajectory_df(traj_path)
        except Exception as e:
            msg = f"Trajectory load error: {e}"
            print(f"  SKIP (trajectory load): {msg}")
            append_verification_log(
                event_id,
                video_id,
                video_file,
                anomaly_start_frame,
                computed_start_frame,
                short_window=short_window,
                notes=msg,
            )
            continue

        if df_traj.empty:
            msg = "Trajectory file loaded but contains no rows"
            print(f"  SKIP: {msg}")
            append_verification_log(
                event_id,
                video_id,
                video_file,
                anomaly_start_frame,
                computed_start_frame,
                short_window=short_window,
                notes=msg,
            )
            continue

        df_window = df_traj[
            (df_traj["frame_idx"] >= jump_to) & (df_traj["frame_idx"] <= window_end)
        ].copy()
        if df_window.empty:
            msg = "No trajectory rows inside event window"
            print(f"  SKIP: {msg}")
            append_verification_log(
                event_id,
                video_id,
                video_file,
                anomaly_start_frame,
                computed_start_frame,
                short_window=short_window,
                notes=msg,
            )
            continue

        df_window["frame_idx"] = df_window["frame_idx"] - jump_to

        df_window = filter_short_tracks(df_window)
        if df_window.empty:
            msg = "Empty after fragment filtering"
            print(f"  SKIP: {msg}")
            append_verification_log(
                event_id,
                video_id,
                video_file,
                anomaly_start_frame,
                computed_start_frame,
                short_window=short_window,
                notes=msg,
            )
            continue

        df_window = interpolate_gaps(df_window)
        if df_window.empty:
            msg = "Empty after interpolation"
            print(f"  SKIP: {msg}")
            append_verification_log(
                event_id,
                video_id,
                video_file,
                anomaly_start_frame,
                computed_start_frame,
                short_window=short_window,
                notes=msg,
            )
            continue

        df_window = strip_warmup(df_window, actual_fps)
        if df_window.empty:
            msg = "Empty after warmup strip"
            print(f"  SKIP: {msg}")
            append_verification_log(
                event_id,
                video_id,
                video_file,
                anomaly_start_frame,
                computed_start_frame,
                short_window=short_window,
                notes=msg,
            )
            continue

        df_f2 = extract_f2_features(df_window, actual_fps, speed_window_sec=speed_window_sec)
        df_f2_agg = aggregate_f2_per_frame(df_f2)
        if df_f2_agg.empty:
            msg = "Empty F2 matrix after aggregation"
            print(f"  SKIP: {msg}")
            append_verification_log(
                event_id,
                video_id,
                video_file,
                anomaly_start_frame,
                computed_start_frame,
                short_window=short_window,
                notes=msg,
            )
            continue

        x_raw = df_f2_agg[["vel_px_sec", "vel_px_sec_smooth"]].to_numpy(dtype=float)
        if x_raw.size == 0:
            msg = "Empty F2 raw matrix"
            print(f"  SKIP: {msg}")
            append_verification_log(
                event_id,
                video_id,
                video_file,
                anomaly_start_frame,
                computed_start_frame,
                short_window=short_window,
                notes=msg,
            )
            continue

        x_scaled = scale_f2_and_log(x_raw, video_id, scaler)
        masks = rule_based_score(x_raw, x_scaled)

        y_true.append(1)
        for method in ("zscore", "iqr"):
            pred = event_level_prediction(masks[method])
            y_pred[method].append(pred)

        append_verification_log(
            event_id,
            video_id,
            video_file,
            anomaly_start_frame,
            computed_start_frame,
            visual_match="UNCHECKED",
            short_window=short_window,
            notes=f"zscore={int(masks['zscore'].any())} iqr={int(masks['iqr'].any())}",
        )

    print("\n[Block D] === Evaluation Results ===")
    results_df = evaluate_per_method(y_true, y_pred)
    full_table = build_results_table(results_df)
    return full_table


# ============================================================
# Day 14 helpers
# ============================================================

def _load_and_validate_ocsvm(model_path: Path = OCSVM_PKL):
    """
    Load and globally validate the Block C OC-SVM artifact.

    Supports either:
      1) raw sklearn model
      2) packaged dict, e.g. {"model": ..., "feature_columns": [...]}

    Validates the model is compatible with F2_only.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"OC-SVM model not found at {model_path}. "
            "Run Block C Day 12 to produce it before running Block D Day 14."
        )
    artifact = joblib.load(model_path)

    if isinstance(artifact, dict):
        if "feature_columns" not in artifact:
            raise RuntimeError(
                f"OC-SVM artifact at {model_path} is a dict but missing "
                "'feature_columns'. Cannot validate F2_only schema."
            )

        feature_columns = list(artifact["feature_columns"])
        if feature_columns != F2_COLS:
            raise RuntimeError(
                f"OC-SVM schema mismatch: artifact feature_columns={feature_columns}, "
                f"expected F2_only columns={F2_COLS}. Aborting OC-SVM method entirely."
            )

        model = (
            artifact.get("model")
            or artifact.get("estimator")
            or artifact.get("ocsvm_model")
            or artifact.get("best_model")
        )

        if model is None:
            raise RuntimeError(
                f"OC-SVM artifact at {model_path} is a dict but no model object was found. "
                "Expected one of keys: model, estimator, ocsvm_model, best_model."
            )

        print(
            f"[Block D Day 14] OC-SVM loaded: {model_path.name}, "
            f"validated via packaged dict on {len(feature_columns)} F2 features ({feature_columns})."
        )
        return model

    model = artifact
    n_expected = len(F2_COLS)
    if hasattr(model, "n_features_in_"):
        n_model = int(model.n_features_in_)
    elif hasattr(model, "support_vectors_"):
        n_model = int(model.support_vectors_.shape[1])
    else:
        raise RuntimeError(
            f"Cannot determine feature count from raw OC-SVM model at {model_path}. "
            "Cannot validate F2_only schema. Aborting OC-SVM method."
        )

    if n_model != n_expected:
        raise RuntimeError(
            f"OC-SVM schema mismatch: model expects {n_model} features, "
            f"but F2_only has {n_expected} ({F2_COLS}). "
            "Aborting OC-SVM method entirely."
        )

    print(
        f"[Block D Day 14] OC-SVM loaded: {model_path.name}, "
        f"validated on {n_model} F2 features ({F2_COLS})."
    )
    return model


def _train_isolation_forest(verbose: bool = True) -> IsolationForest:
    """
    Train an Isolation Forest on Block C train-split scaled features.

    Data source  : logs/block_c/{seq_id}_features_scaled.csv (train split only)
    Features     : vel_px_sec, vel_px_sec_smooth (F2_only)
    Interp rows  : excluded via is_interpolated column; absence → warning + include all
    Missing CSVs : fail with explicit error — never skip silently

    Saves:
      - logs/block_d/isolation_forest_f2_train_normal.pkl
      - logs/block_d/isolation_forest_training_metadata.json
    """
    BLOCK_D_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Load train sequence IDs
    if not METADATA_JSON.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {METADATA_JSON}. "
            "Cannot determine train split for Isolation Forest training."
        )
    with open(METADATA_JSON, encoding="utf-8") as f:
        meta = json.load(f)

    train_seq_ids: List[str] = meta["split_seqs"]["train"]
    if not train_seq_ids:
        raise ValueError("metadata.json['split_seqs']['train'] is empty.")

    if verbose:
        print(f"[Block D Day 14] IF training: {len(train_seq_ids)} train sequences: {train_seq_ids}")

    # Load and validate each train CSV
    all_chunks: List[pd.DataFrame] = []
    rows_per_file: dict = {}
    total_interp_excluded = 0
    source_files: List[str] = []

    for seq_id in train_seq_ids:
        csv_path = BLOCK_C_LOG_DIR / f"{seq_id}_features_scaled.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expected Block C scaled CSV not found: {csv_path}. "
                "Cannot train Isolation Forest without complete train split. "
                "Do not silently skip — run Block C first."
            )

        df = pd.read_csv(csv_path)

        # Validate F2 columns present
        for col in F2_COLS:
            if col not in df.columns:
                raise KeyError(
                    f"Required F2 column '{col}' missing from {csv_path.name}. "
                    "Block C output schema has changed unexpectedly."
                )

        # Exclude interpolated rows
        if "is_interpolated" in df.columns:
            # Handle both bool and string representations
            interp_mask = df["is_interpolated"].map(
                lambda x: str(x).strip().lower() == "true"
            )
            n_before = len(df)
            df = df[~interp_mask].copy()
            n_excluded = n_before - len(df)
            total_interp_excluded += n_excluded
            if verbose and n_excluded > 0:
                print(f"  [{seq_id}] Excluded {n_excluded} interpolated rows.")
        else:
            logger.warning(
                "[Block D Day 14] IF training: 'is_interpolated' column absent in %s — "
                "including all rows from this file.",
                csv_path.name,
            )

        chunk = df[F2_COLS].dropna()
        n_rows = len(chunk)
        rows_per_file[str(csv_path)] = n_rows
        source_files.append(str(csv_path))
        all_chunks.append(chunk)

    if not all_chunks:
        raise ValueError(
            "No valid training rows found after loading all train-split CSVs. "
            "Cannot fit Isolation Forest."
        )

    X_train = pd.concat(all_chunks, ignore_index=True).to_numpy(dtype=float)
    total_rows = len(X_train)

    if verbose:
        print(
            f"[Block D Day 14] IF training matrix: {total_rows} rows × {X_train.shape[1]} features "
            f"({total_interp_excluded} interpolated rows excluded)."
        )

    # Fit
    if_model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
    )
    if_model.fit(X_train)

    fit_ts = datetime.now(timezone.utc).isoformat()

    # Persist model
    joblib.dump(if_model, IF_PKL)
    print(f"[Block D Day 14] IF model saved: {IF_PKL}")

    # Persist training metadata
    train_meta = {
        "training_source_files": source_files,
        "feature_columns": F2_COLS,
        "total_rows_used": total_rows,
        "rows_per_file": rows_per_file,
        "interpolated_rows_excluded": total_interp_excluded,
        "seed": IF_RANDOM_STATE,
        "n_estimators": IF_N_ESTIMATORS,
        "contamination": IF_CONTAMINATION,
        "fit_timestamp": fit_ts,
    }
    with open(IF_TRAIN_META_JSON, "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2)
    print(f"[Block D Day 14] IF training metadata saved: {IF_TRAIN_META_JSON}")

    return if_model


# ============================================================
# Day 14 public entry point
# ============================================================

def run_block_d_day14(
    cfg_d: dict,
    cfg_c: dict,
    events_path: str | Path = EVENTS_JSON,
    scaler_path: str | Path = SCALER_PKL,
    verbose: bool = True,
) -> None:
    """
    Block D Day 14: Full anomaly-comparison pipeline on AI City Track 4.

    Runs four methods on the same event slices:
      - Rule-Based (Z-score)     [from Day 13, reused]
      - Rule-Based (IQR)         [from Day 13, reused]
      - OC-SVM                   [Block C model, never refit]
      - Isolation Forest         [trained here on Block C train-split normals]

    All loaded events are positive anomaly events (gt_label=1).
    FAR / Precision / F1 are left NaN — positive-only benchmark.

    Outputs (all under logs/block_d/):
      - block_d_results.csv
      - block_d_event_predictions.csv
      - block_d_skip_reasons.csv
      - block_d_metadata.json
      - isolation_forest_f2_train_normal.pkl
      - isolation_forest_training_metadata.json
      - event_slicer_verification.csv  (updated, Day 13-compatible schema)
    """
    BLOCK_D_LOG_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = Path(scaler_path)

    print("[Block D Day 14] === Starting Day 14 pipeline ===")
    audit_config(cfg_d)

    anomaly_events = load_events(events_path)
    n_total = len(anomaly_events)
    video_index = build_video_index()
    scaler = load_scaler(scaler_path)
    speed_window_sec = float(cfg_c.get("f2", {}).get("speed_window_sec", SPEED_WINDOW_SEC))

    # ── OC-SVM: load once and validate globally ───────────────────────────────
    ocsvm_model = None
    ocsvm_aborted = False
    ocsvm_abort_reason = ""
    try:
        ocsvm_model = _load_and_validate_ocsvm(OCSVM_PKL)
    except Exception as exc:
        ocsvm_aborted = True
        ocsvm_abort_reason = str(exc)
        print(f"[Block D Day 14] OC-SVM ABORTED globally: {exc}")

    # ── Isolation Forest: train on Block C train-split normals ────────────────
    if_model = _train_isolation_forest(verbose=verbose)

    # ── Event loop ────────────────────────────────────────────────────────────
    METHOD_KEYS = [
        "Rule-Based (Z-score)",
        "Rule-Based (IQR)",
        "OC-SVM",
        "Isolation Forest",
    ]

    method_stats = {
        m: {"evaluable": 0, "detected": 0, "skipped": 0, "anomaly_ratios": []}
        for m in METHOD_KEYS
    }
    all_event_rows: List[dict] = []
    all_skip_rows: List[dict] = []

    for event in anomaly_events:
        event_id = str(event["event_id"])
        video_id = str(event["video_id"])
        video_file = str(event["video_file"])
        anomaly_start_frame = int(event["anomaly_start_frame"])
        anomaly_end_frame = int(event["anomaly_end_frame"])

        if verbose:
            print(
                f"\n[Block D Day 14] Event {event_id} "
                f"(video_id={video_id}, file={video_file})"
            )

        # ── Shared pipeline (same steps as Day 13) ───────────────────────────
        pipeline_skip_reason: Optional[str] = None
        x_raw: Optional[np.ndarray] = None
        x_scaled: Optional[np.ndarray] = None
        n_valid = 0
        actual_fps = 0.0

        video_path = resolve_video_path(video_file, video_index)
        if video_path is None:
            pipeline_skip_reason = (
                f"Video not found in indexed AI City roots for file '{video_file}'"
            )

        if pipeline_skip_reason is None:
            try:
                actual_fps = get_actual_fps(video_path)
            except (FileNotFoundError, ValueError) as exc:
                pipeline_skip_reason = f"FPS error: {exc}"

        if pipeline_skip_reason is None:
            traj_path = find_trajectory_path(video_id)
            if traj_path is None:
                pipeline_skip_reason = (
                    f"No _{TRACKER}_final.json trajectory found for video_id={video_id}"
                )

        if pipeline_skip_reason is None:
            try:
                df_traj = load_trajectory_df(traj_path)
            except Exception as exc:
                pipeline_skip_reason = f"Trajectory load error: {exc}"

        if pipeline_skip_reason is None:
            if df_traj.empty:
                pipeline_skip_reason = "Trajectory file loaded but contains no rows"

        if pipeline_skip_reason is None:
            context_frames = int(CONTEXT_SEC * actual_fps)
            jump_to = max(0, anomaly_start_frame - context_frames)
            window_end = anomaly_end_frame + context_frames
            df_window = df_traj[
                (df_traj["frame_idx"] >= jump_to) & (df_traj["frame_idx"] <= window_end)
            ].copy()
            if df_window.empty:
                pipeline_skip_reason = "No trajectory rows inside event window"

        if pipeline_skip_reason is None:
            df_window["frame_idx"] = df_window["frame_idx"] - jump_to
            df_window = filter_short_tracks(df_window)
            if df_window.empty:
                pipeline_skip_reason = "Empty after fragment filtering"

        if pipeline_skip_reason is None:
            df_window = interpolate_gaps(df_window)
            if df_window.empty:
                pipeline_skip_reason = "Empty after interpolation"

        if pipeline_skip_reason is None:
            df_window = strip_warmup(df_window, actual_fps)
            if df_window.empty:
                pipeline_skip_reason = "Empty after warmup strip"

        if pipeline_skip_reason is None:
            df_f2 = extract_f2_features(df_window, actual_fps, speed_window_sec=speed_window_sec)
            df_f2_agg = aggregate_f2_per_frame(df_f2)
            if df_f2_agg.empty:
                pipeline_skip_reason = "Empty F2 matrix after aggregation"

        if pipeline_skip_reason is None:
            x_raw = df_f2_agg[["vel_px_sec", "vel_px_sec_smooth"]].to_numpy(dtype=float)
            if x_raw.size == 0:
                pipeline_skip_reason = "Empty F2 raw matrix"

        if pipeline_skip_reason is None:
            x_scaled = scale_f2_and_log(x_raw, video_id, scaler)
            n_valid = int(x_raw.shape[0])

        # ── Pipeline failure: skip all methods for this event ────────────────
        if pipeline_skip_reason is not None:
            if verbose:
                print(f"  SKIP (shared pipeline): {pipeline_skip_reason}")
            for method in METHOD_KEYS:
                method_stats[method]["skipped"] += 1
                all_event_rows.append(
                    {
                        "method": method,
                        "event_id": event_id,
                        "video_id": video_id,
                        "video_file": video_file,
                        "event_pred": "",
                        "gt_label": 1,
                        "total_valid_samples": 0,
                        "anomalous_samples": "",
                        "anomalous_ratio": "",
                        "skipped": True,
                        "skip_reason": pipeline_skip_reason,
                    }
                )
                all_skip_rows.append(
                    {
                        "method": method,
                        "event_id": event_id,
                        "video_id": video_id,
                        "skip_reason": pipeline_skip_reason,
                    }
                )
            append_verification_log(
                event_id, video_id, video_file,
                anomaly_start_frame, anomaly_start_frame,
                notes=pipeline_skip_reason,
            )
            continue

        # ── Compute rule-based masks once (shared across both rule-based methods)
        rb_masks = rule_based_score(x_raw, x_scaled)

        # ── Per-method scoring ───────────────────────────────────────────────
        for method in METHOD_KEYS:
            method_skip_reason = ""
            mask: Optional[np.ndarray] = None

            if method == "Rule-Based (Z-score)":
                mask = rb_masks["zscore"]

            elif method == "Rule-Based (IQR)":
                mask = rb_masks["iqr"]

            elif method == "OC-SVM":
                if ocsvm_aborted:
                    method_skip_reason = (
                        f"OC-SVM method aborted globally: {ocsvm_abort_reason}"
                    )
                elif x_scaled.size == 0:
                    method_skip_reason = "Empty F2 matrix for OC-SVM inference"
                else:
                    try:
                        preds = ocsvm_model.predict(x_scaled)
                        mask = preds == -1
                    except Exception as exc:
                        method_skip_reason = f"OC-SVM predict error: {exc}"

            elif method == "Isolation Forest":
                if x_scaled.size == 0:
                    method_skip_reason = "Empty F2 matrix for IF inference"
                else:
                    try:
                        preds = if_model.predict(x_scaled)
                        mask = preds == -1
                    except Exception as exc:
                        method_skip_reason = f"IF predict error: {exc}"

            # ── Record results ───────────────────────────────────────────────
            if method_skip_reason or mask is None:
                final_reason = method_skip_reason or "mask unavailable"
                method_stats[method]["skipped"] += 1
                all_event_rows.append(
                    {
                        "method": method,
                        "event_id": event_id,
                        "video_id": video_id,
                        "video_file": video_file,
                        "event_pred": "",
                        "gt_label": 1,
                        "total_valid_samples": n_valid,
                        "anomalous_samples": "",
                        "anomalous_ratio": "",
                        "skipped": True,
                        "skip_reason": final_reason,
                    }
                )
                all_skip_rows.append(
                    {
                        "method": method,
                        "event_id": event_id,
                        "video_id": video_id,
                        "skip_reason": final_reason,
                    }
                )
            else:
                anomalous_count = int(mask.sum())
                event_pred = 1 if anomalous_count >= 1 else 0
                ratio = anomalous_count / n_valid if n_valid > 0 else 0.0
                method_stats[method]["evaluable"] += 1
                method_stats[method]["detected"] += event_pred
                method_stats[method]["anomaly_ratios"].append(ratio)
                all_event_rows.append(
                    {
                        "method": method,
                        "event_id": event_id,
                        "video_id": video_id,
                        "video_file": video_file,
                        "event_pred": event_pred,
                        "gt_label": 1,
                        "total_valid_samples": n_valid,
                        "anomalous_samples": anomalous_count,
                        "anomalous_ratio": round(ratio, 4),
                        "skipped": False,
                        "skip_reason": "",
                    }
                )

        # Day 13-compatible verification log entry
        append_verification_log(
            event_id, video_id, video_file,
            anomaly_start_frame, anomaly_start_frame,
            visual_match="UNCHECKED",
            notes=(
                f"zscore={int(rb_masks['zscore'].any())} "
                f"iqr={int(rb_masks['iqr'].any())}"
            ),
        )

    # ── Write artifacts ───────────────────────────────────────────────────────

    # 1. block_d_event_predictions.csv
    if all_event_rows:
        pd.DataFrame(all_event_rows).to_csv(EVENT_PREDS_CSV, index=False)
        print(f"[Block D Day 14] Event predictions saved: {EVENT_PREDS_CSV}")
    else:
        print("[Block D Day 14] WARNING: No event rows to write.")

    # 2. block_d_skip_reasons.csv
    skip_cols = ["method", "event_id", "video_id", "skip_reason"]
    if all_skip_rows:
        skip_df = pd.DataFrame(all_skip_rows, columns=skip_cols)
    else:
        skip_df = pd.DataFrame(columns=skip_cols)
    skip_df.to_csv(SKIP_REASONS_CSV_PATH, index=False)
    print(f"[Block D Day 14] Skip reasons saved: {SKIP_REASONS_CSV_PATH}")

    # 3. block_d_results.csv  (Day 14 schema — overwrites Day 13 output)
    results_rows = []
    for method in METHOD_KEYS:
        stats = method_stats[method]
        evaluable = stats["evaluable"]
        detected = stats["detected"]
        skipped = stats["skipped"]
        ratios = stats["anomaly_ratios"]
        recall = detected / evaluable if evaluable > 0 else 0.0
        mean_ratio = float(np.mean(ratios)) if ratios else 0.0

        results_rows.append(
            {
                "Method": method,
                "Event_Type": "anomaly",
                "Events_Total": n_total,
                "Events_Evaluable": evaluable,
                "Events_Detected": detected,
                "Event_Recall": round(recall, 4),
                "Skipped_Events": skipped,
                "Mean_Anomalous_Ratio": round(mean_ratio, 4),
                "FAR": float("nan"),
                "Precision": float("nan"),
                "F1": float("nan"),
                "Notes": (
                    "positive-only benchmark; "
                    "FAR/Precision/F1 deferred: no normal clips evaluated in Day 14"
                ),
            }
        )

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"[Block D Day 14] Results table saved: {RESULTS_CSV}")

    # 4. block_d_metadata.json
    meta_doc = {
        "scaler_path": str(scaler_path),
        "ocsvm_model_path": str(OCSVM_PKL),
        "isolation_forest_model_path": str(IF_PKL),
        "isolation_forest_training_metadata_path": str(IF_TRAIN_META_JSON),
        "feature_columns": F2_COLS,
        "aggregation_rule": "event_pred = 1 if anomalous_count >= 1 else 0",
        "valid_sample_definition": (
            "survives fragment filter + interpolation gap-fill + warmup strip"
        ),
        "event_schema_assumption": (
            "all loaded events are positive anomaly events (gt_label=1)"
        ),
        "evaluation_scope": (
            "positive-only benchmark — FAR/Precision/F1 deferred"
        ),
        "events_total": n_total,
        "per_method_evaluable": {
            m: method_stats[m]["evaluable"] for m in METHOD_KEYS
        },
        "per_method_skipped": {
            m: method_stats[m]["skipped"] for m in METHOD_KEYS
        },
        "ocsvm_aborted": ocsvm_aborted,
        "ocsvm_abort_reason": ocsvm_abort_reason if ocsvm_aborted else "",
        "blank_metrics_note": {
            "FAR": "deferred: no normal clips evaluated in Day 14",
            "Precision": "not computable: no negative events in Day 14 scope",
            "F1": "not computable: no negative events in Day 14 scope",
            "Specificity": "not computable: no negative events in Day 14 scope",
        },
    }
    with open(BLOCK_D_META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta_doc, f, indent=2)
    print(f"[Block D Day 14] Metadata saved: {BLOCK_D_META_JSON}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[Block D Day 14] === Summary ===")
    display_cols = [
        "Method", "Events_Total", "Events_Evaluable",
        "Events_Detected", "Event_Recall", "Mean_Anomalous_Ratio", "Skipped_Events",
    ]
    print(results_df[display_cols].to_string(index=False))
