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
from scipy.stats import zscore
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
    """
    if df.empty:
        return df.copy()

    genuine = df[df["is_interpolated"] == False]  # noqa: E712
    if genuine.empty:
        return df.iloc[0:0].copy()

    counts = genuine.groupby("track_id")["frame_idx"].count()
    valid = counts[counts >= min_length].index
    filtered = df[df["track_id"].isin(valid)].copy()
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
