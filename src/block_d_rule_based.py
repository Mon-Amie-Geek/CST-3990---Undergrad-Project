"""
block_d_rule_based.py

Block D — Day 13: Rule-Based Anomaly Baseline on AI City Track 4.
All reusable functions used by both the Day 13 notebook and pipeline_controller.run_block_d().

Pipeline:
    event_loader → event_slicer → trajectory_loader →
    [F11 fragment_filter → F22 interpolation → F16 warmup_strip] →
    f2_extractor → scaler (no fit) → rule_based_scorer → evaluator

Fix references embedded in inline comments throughout.

Student : MANJOO Ameera Najla | M01014463
Module  : CST3990 Undergraduate Individual Project — Block D
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

# ── Paths (all relative to repo root) ────────────────────────────────────────
REPO_ROOT         = Path(__file__).resolve().parent.parent
EVENTS_JSON       = REPO_ROOT / "data" / "ai_city" / "events.json"
TRAJ_BASE         = REPO_ROOT / "logs" / "block_b" / "trajectories"
SCALER_PKL        = REPO_ROOT / "models" / "minmax_scaler.pkl"
BLOCK_D_LOG_DIR   = REPO_ROOT / "logs" / "block_d"
SYNC_CHECK_DIR    = BLOCK_D_LOG_DIR / "sync_checks"
RESULTS_CSV       = BLOCK_D_LOG_DIR / "block_d_results.csv"
VERIF_LOG_CSV     = BLOCK_D_LOG_DIR / "event_slicer_verification.csv"

# ── Block C / Block D constants ───────────────────────────────────────────────
MIN_TRACK_LENGTH   = 15       # Fix F11 — fragment filter
MAX_INTERP_GAP     = 3        # Fix F22 — interpolation gap ceiling
CONTEXT_SEC        = 5.0      # context window around event (seconds) — Fix F03
MIN_FRAMES_WARN    = 15       # soft guard: warn if fewer frames after warmup
Z_THRESH           = 2.0      # Z-score anomaly threshold
IQR_MULT           = 1.5      # IQR fence multiplier
SPEED_WINDOW_SEC   = 0.2      # Fix F07 — speed smoothing window in seconds
TRACKER            = "sort"   # Block B frozen selection

# SCALER_FIT_COLUMNS order (must match models/minmax_scaler.pkl fitted column order).
# From src/feature_extractor.py SCALER_FIT_COLUMNS (Day 11).
SCALER_FIT_COLUMNS = [
    "vel_px_sec",
    "vel_px_sec_smooth",
    "vehicle_count",
    "roi_occupancy",
    "inter_vehicle_dist_norm",
    "dwell_time_sec",
    "proximity_count_rolling",
]
# F2_only uses columns 0,1 in the scaler's fitted feature order.
F2_IDX = [0, 1]


# =============================================================================
# SECTION 1 — CONFIG AUDIT
# =============================================================================

def audit_config(cfg: dict) -> None:
    """
    Assert Block D config is scoped to AI City only.
    Checks:
      - dataset field points to ai_city (nested or flat structure handled)
      - Zero occurrences of banned strings: tudat, tu_dat, tailgating
    Raises AssertionError on failure.
    """
    # Support both flat (dataset: ai_city_track4) and nested (datasets.primary: ai_city)
    dataset_val = cfg.get("dataset", cfg.get("datasets", {}).get("primary", ""))
    assert "ai_city" in str(dataset_val).lower(), (
        f"Config audit FAILED: dataset field is '{dataset_val}'. "
        "Expected 'ai_city' or 'ai_city_track4'."
    )

    # Grep raw YAML content for banned strings
    cfg_str = json.dumps(cfg).lower()
    for banned in ("tudat", "tu_dat", "tailgating"):
        hits = cfg_str.count(banned)
        assert hits == 0, (
            f"Config audit FAILED: banned substring '{banned}' found {hits} time(s). "
            "Block D scope is AI City Track 4 only."
        )

    print("[Block D] Config audit passed — dataset: AI City, no TU-DAT / tailgating references.")


# =============================================================================
# SECTION 2 — AI CITY EVENT LOADER
# =============================================================================

def load_events(events_path: str | Path = EVENTS_JSON) -> Tuple[List[dict], List[dict]]:
    """
    Load AI City Track 4 ground-truth events.

    Schema expected per event:
        {clip_id, event_type ('stall'|'crash'), gt_start_sec, gt_end_sec, video_path}

    Returns:
        stall_events (list[dict])
        crash_events (list[dict])
    """
    events_path = Path(events_path)
    if not events_path.exists():
        raise FileNotFoundError(f"events.json not found at {events_path}")

    with open(events_path, encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("events", [])
    if not events:
        raise ValueError("events.json contains no events.")

    stall_events = [e for e in events if e["event_type"] == "stall"]
    crash_events = [e for e in events if e["event_type"] == "crash"]

    assert len(stall_events) >= 1, "events.json must contain at least one stall event."
    assert len(crash_events) >= 1, "events.json must contain at least one crash event."

    print(f"[Block D] Events loaded: {len(stall_events)} stall, {len(crash_events)} crash.")
    return stall_events, crash_events


# =============================================================================
# SECTION 3 — FPS EXTRACTION  (never hardcoded — Fix F03)
# =============================================================================

def get_actual_fps(video_path: str | Path) -> float:
    """
    Extract FPS from video at runtime. Never returns a hardcoded value.
    Raises ValueError if FPS is outside plausible range (0, 120].

    Fix F03: all frame-count parameters derived from actual_fps.
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


# =============================================================================
# SECTION 4 — FRAME SYNC CHECK (Fix F12)
# =============================================================================

def verify_frame_sync(
    video_path: str | Path,
    annotation_frame_id: int,
    expected_content_note: str,
    out_dir: str | Path = SYNC_CHECK_DIR,
) -> bool:
    """
    Fix F12 — verify that the annotation frame numbering aligns with the decoded video.

    Saves a JPEG of the decoded frame so the researcher can manually compare it
    to the annotated event. Determines whether AI City uses 0-based or 1-based indexing.

    Args:
        video_path            : path to the AI City clip
        annotation_frame_id   : the frame index from the annotation (may be 0- or 1-based)
        expected_content_note : human-readable description of what should be visible at this frame
        out_dir               : directory to save sync-check JPEGs

    Returns:
        True if frame was successfully decoded and saved, False otherwise.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(str(video_path)):
        print(f"[Fix F12] SKIP — video not found: {video_path}")
        return False

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, annotation_frame_id)  # Fix F27 direct jump
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"[Fix F12] WARNING — could not decode frame {annotation_frame_id} from {video_path}")
        return False

    clip_id = Path(video_path).stem
    out_path = out_dir / f"sync_check_{clip_id}_frame{annotation_frame_id}.jpg"
    cv2.imwrite(str(out_path), frame)
    print(f"[Fix F12] Saved {out_path}")
    print(f"[Fix F12] Manually verify it matches: {expected_content_note}")
    return True


# =============================================================================
# SECTION 5 — EVENT SLICER (Fix F27)
# =============================================================================

def extract_event_window(
    video_path: str | Path,
    gt_start_sec: float,
    gt_end_sec: float,
    actual_fps: float,
    context_sec: float = CONTEXT_SEC,
) -> List[Tuple[int, np.ndarray]]:
    """
    Extract frames for an event window with context padding.

    Fix F27: Uses CAP_PROP_POS_FRAMES direct jump to avoid sequential decode overhead.
    Fix F03: context_frames computed from actual_fps, never hardcoded.

    Args:
        video_path   : path to AI City clip
        gt_start_sec : ground-truth event start (seconds)
        gt_end_sec   : ground-truth event end (seconds)
        actual_fps   : fps extracted at runtime via get_actual_fps()
        context_sec  : seconds of context before/after event (default 5.0)

    Returns:
        List of (frame_idx, frame_bgr) tuples within the window.
        Empty list if video not found.
    """
    # Fix F03 — all window parameters derived from actual_fps
    context_frames = int(context_sec * actual_fps)
    start_frame    = int(gt_start_sec * actual_fps)
    end_frame      = int(gt_end_sec   * actual_fps)
    jump_to        = max(0, start_frame - context_frames)
    window_end     = end_frame + context_frames

    if not os.path.exists(str(video_path)):
        warnings.warn(
            f"[EventSlicer] Video not found: {video_path}. "
            "Returning empty frame list — run Block B on AI City clips first.",
            stacklevel=2,
        )
        return []

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, jump_to)  # Fix F27 — direct jump

    frames: List[Tuple[int, np.ndarray]] = []
    while True:
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_pos > window_end:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((current_pos - 1, frame))  # cap reads advance by 1, back-correct

    cap.release()
    return frames


# =============================================================================
# SECTION 6 — TRAJECTORY LOADING
# =============================================================================

def find_trajectory_path(clip_id: str, tracker: str = TRACKER) -> Optional[Path]:
    """
    Locate the *_final.json trajectory file for a given clip_id and tracker.

    Looks in logs/block_b/trajectories/ for {clip_id}_{tracker}_final.json.
    Returns None if not found (AI City clips need Block B run first).
    """
    candidate = TRAJ_BASE / f"{clip_id}_{tracker}_final.json"
    if candidate.exists():
        return candidate
    # Fallback: look for any _final.json containing clip_id
    for f in TRAJ_BASE.glob(f"*{clip_id}*_final.json"):
        return f
    return None


def load_trajectory_df(traj_path: Path) -> pd.DataFrame:
    """
    Load a Block B *_final.json trajectory file and flatten to a DataFrame.

    Trajectory format (dict-keyed by track_id string — see trajectory_schema.json):
      { "tracks": { "1": [{frame_idx, bbox, cx, cy, velocity_norm,
                            conf, interpolated, track_length}, ...], ... } }

    Renames 'interpolated' → 'is_interpolated' for Block C schema consistency.
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
        # Fallback for legacy list layout
        rows = tracks_raw
    else:
        raise ValueError(f"Unexpected 'tracks' type {type(tracks_raw)} in {traj_path}")

    df = pd.DataFrame(rows)

    # ── Schema normalisation ─────────────────────────────────────────────────
    # Rename 'interpolated' → 'is_interpolated' (Block C schema alignment)
    if "interpolated" in df.columns and "is_interpolated" not in df.columns:
        df = df.rename(columns={"interpolated": "is_interpolated"})

    # Ensure required columns exist
    for col in ("frame_idx", "track_id", "cx", "cy", "velocity_norm", "conf", "is_interpolated"):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from trajectory {traj_path.name}")

    df["frame_idx"] = df["frame_idx"].astype(int)
    df["track_id"]  = df["track_id"].astype(int)
    return df


# =============================================================================
# SECTION 7 — TRAJECTORY PRE-PROCESSING (Fixes F11, F22, F16)
# =============================================================================

def filter_short_tracks(df: pd.DataFrame, min_length: int = MIN_TRACK_LENGTH) -> pd.DataFrame:
    """
    Fix F11 — Fragment filter.
    Remove tracks with fewer than min_length genuine detections (non-interpolated rows).
    Short tracks are tracker fragments; including them degrades F2 speed estimates.
    """
    genuine = df[df["is_interpolated"] == False]  # noqa: E712
    counts  = genuine.groupby("track_id")["frame_idx"].count()
    valid   = counts[counts >= min_length].index
    filtered = df[df["track_id"].isin(valid)].copy()
    n_removed = df["track_id"].nunique() - filtered["track_id"].nunique()
    if n_removed > 0:
        logger.debug(f"[F11] Removed {n_removed} short track(s) (< {min_length} genuine frames).")
    return filtered


def interpolate_gaps(df: pd.DataFrame, max_gap: int = MAX_INTERP_GAP) -> pd.DataFrame:
    """
    Fix F22 — Track interpolation.
    For each track, if it is absent for <= max_gap consecutive frames:
        linearly interpolate centroid (cx, cy) and mark is_interpolated=True.
    If absent for > max_gap frames: leave as-is (no synthetic bridge — gap too large).

    This mirrors the post_processor.py logic from Block B Day 9.
    """
    all_frames = sorted(df["frame_idx"].unique())
    if not all_frames:
        return df

    frame_min, frame_max = int(min(all_frames)), int(max(all_frames))
    result_rows = []

    for tid, grp in df.groupby("track_id"):
        grp_sorted = grp.sort_values("frame_idx").reset_index(drop=True)
        frames_present = set(grp_sorted["frame_idx"].tolist())

        result_rows.append(grp_sorted)

        # Find gaps in this track's frame sequence
        track_frames = sorted(frames_present)
        for i in range(len(track_frames) - 1):
            f_prev = track_frames[i]
            f_next = track_frames[i + 1]
            gap = f_next - f_prev - 1

            if 0 < gap <= max_gap:
                # Linear interpolation of cx, cy
                prev_row = grp_sorted[grp_sorted["frame_idx"] == f_prev].iloc[0]
                next_row = grp_sorted[grp_sorted["frame_idx"] == f_next].iloc[0]
                for f_interp in range(f_prev + 1, f_next):
                    t = (f_interp - f_prev) / (f_next - f_prev)
                    cx_interp = float(prev_row["cx"] + t * (next_row["cx"] - prev_row["cx"]))
                    cy_interp = float(prev_row["cy"] + t * (next_row["cy"] - prev_row["cy"]))
                    # velocity_norm via finite difference on interpolated position
                    prev_cx = float(prev_row["cx"] + (t - 1.0 / (f_next - f_prev)) * (next_row["cx"] - prev_row["cx"]))
                    vel = float(np.sqrt((cx_interp - prev_cx) ** 2))
                    interp_row = {
                        "frame_idx":      f_interp,
                        "track_id":       tid,
                        "cx":             cx_interp,
                        "cy":             cy_interp,
                        "velocity_norm":  vel,
                        "conf":           -1.0,   # Fix F22 sentinel
                        "is_interpolated": True,
                    }
                    result_rows.append(pd.DataFrame([interp_row]))

    out = pd.concat(result_rows, ignore_index=True)
    out = out.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)
    return out


def strip_warmup(df: pd.DataFrame, actual_fps: float) -> pd.DataFrame:
    """
    Fix F16 — Warmup strip.
    Remove the first max(10, int(0.4 * actual_fps)) frames from the event window.
    Applied per event window (not per full clip) so that the warmup is relative
    to the event window start, not the clip start.

    Fix F03: warmup_frames derived from actual_fps, never hardcoded.
    """
    warmup_frames = max(10, int(0.4 * actual_fps))  # Fix F03
    min_frame = df["frame_idx"].min()
    warmup_boundary = min_frame + warmup_frames      # Fix F16: relative to window start
    return df[df["frame_idx"] > warmup_boundary].copy()


# =============================================================================
# SECTION 8 — F2 FEATURE EXTRACTION (F2_only: vel_px_sec + vel_px_sec_smooth)
# =============================================================================

def extract_f2_features(df: pd.DataFrame, actual_fps: float) -> pd.DataFrame:
    """
    Extract F2_only features: vel_px_sec and vel_px_sec_smooth.
    Replicates Block C extractor logic (src/feature_extractor.py extract_f2_features).

    vel_px_sec        = velocity_norm * actual_fps        (raw pixels/sec)
    vel_px_sec_smooth = rolling mean per track with window = int(0.2 * actual_fps)
                        (Fix F07: window in seconds, converted at runtime)

    Excludes interpolated rows from smoothing input (Fix F13).
    Interpolated rows carry conf=-1.0 sentinel (Fix F22).

    Returns:
        DataFrame with columns: [frame_idx, track_id, vel_px_sec, vel_px_sec_smooth,
                                  is_interpolated, cx, cy]
    """
    # Fix F07 — speed window in frames, derived from actual_fps
    speed_window = max(1, int(SPEED_WINDOW_SEC * actual_fps))

    df = df.copy()
    df["vel_px_sec"] = df["velocity_norm"].fillna(0.0) * actual_fps

    # Rolling mean per track (sort by frame_idx within each track)
    smooth_series = []
    for tid, grp in df.groupby("track_id"):
        grp_sorted = grp.sort_values("frame_idx").copy()
        # Fix F13: exclude interpolated rows from the rolling mean computation
        grp_sorted["vel_px_sec_smooth"] = (
            grp_sorted["vel_px_sec"]
            .where(grp_sorted["is_interpolated"] != True)  # noqa: E712
            .rolling(window=speed_window, min_periods=1)
            .mean()
            .fillna(grp_sorted["vel_px_sec"])
        )
        smooth_series.append(grp_sorted)

    if not smooth_series:
        return pd.DataFrame(columns=["frame_idx", "track_id", "vel_px_sec",
                                      "vel_px_sec_smooth", "is_interpolated", "cx", "cy"])

    out = pd.concat(smooth_series, ignore_index=True)
    out = out.sort_values(["frame_idx", "track_id"]).reset_index(drop=True)
    return out[["frame_idx", "track_id", "vel_px_sec", "vel_px_sec_smooth",
                "is_interpolated", "cx", "cy"]]


def aggregate_f2_per_frame(df_f2: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-track F2 features to per-frame means.
    Rule-based scorer operates on frame-level aggregates.

    Returns:
        DataFrame with columns: [frame_idx, vel_px_sec, vel_px_sec_smooth]
    """
    agg = (
        df_f2.groupby("frame_idx")[["vel_px_sec", "vel_px_sec_smooth"]]
        .mean()
        .reset_index()
    )
    return agg.sort_values("frame_idx").reset_index(drop=True)


# =============================================================================
# SECTION 9 — SCALER & DOMAIN-SHIFT LOGGING (Fixes F13, F18)
# =============================================================================

def load_scaler(scaler_path: str | Path = SCALER_PKL):
    """
    Load the Block C fitted MinMaxScaler.
    Must be called ONCE outside all loops — never call scaler.fit() in Block D.
    """
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"MinMaxScaler not found at {scaler_path}")
    scaler = joblib.load(scaler_path)
    print(f"[Block D] Scaler loaded: {scaler_path.name}, "
          f"fitted on {scaler.n_features_in_} features.")
    return scaler


def scale_f2_and_log(X_raw: np.ndarray, clip_id: str, scaler) -> np.ndarray:
    """
    Scale F2_only features using the Block C MinMaxScaler and log domain shift.

    Fix F13: uses the Block C fitted scaler — scaler.fit() is NEVER called here.
    Fix F18: values outside [0,1] are clipped and the percentage is logged.
             High pct_clipped (>20%) indicates AI City highway speeds exceed the
             UA-DETRAC normal training distribution — a construct validity risk.

    Args:
        X_raw    : shape (N, 2), columns [vel_px_sec, vel_px_sec_smooth]
        clip_id  : used for logging only
        scaler   : fitted sklearn MinMaxScaler (7 features)

    Returns:
        X_scaled : shape (N, 2), values clipped to [0, 1]
    """
    # Extract F2-only scale parameters from the 7-feature scaler
    # SCALER_FIT_COLUMNS[0] = vel_px_sec, [1] = vel_px_sec_smooth
    f2_min   = scaler.data_min_[F2_IDX]      # shape (2,)
    f2_range = scaler.data_range_[F2_IDX]    # shape (2,)

    # Manual MinMax transform for F2 columns only
    with np.errstate(divide="ignore", invalid="ignore"):
        X_scaled = (X_raw - f2_min) / np.where(f2_range == 0, 1.0, f2_range)

    # Fix F18 — clip and log domain shift
    n_clipped  = int(((X_scaled > 1.0) | (X_scaled < 0.0)).sum())
    pct_clipped = n_clipped / X_scaled.size * 100.0
    X_scaled   = np.clip(X_scaled, 0.0, 1.0)

    flag = " ⚠ HIGH DOMAIN SHIFT" if pct_clipped > 20.0 else ""
    print(f"[Fix F18] [{clip_id}] Domain-shift indicator: "
          f"{pct_clipped:.1f}% of F2 values clipped to [0,1]{flag}")
    if pct_clipped > 20.0:
        print(f"[Fix F18] WARNING: >20% clipping for {clip_id}. "
              "AI City highway speeds likely exceed UA-DETRAC normal training range. "
              "Scaled rule-based (IQR) results may be biased — interpret with caution.")

    return X_scaled


# =============================================================================
# SECTION 10 — RULE-BASED ANOMALY SCORER
# =============================================================================

def rule_based_score(
    X_raw: np.ndarray,
    X_scaled: np.ndarray,
    z_thresh: float = Z_THRESH,
    iqr_mult: float = IQR_MULT,
) -> Dict[str, np.ndarray]:
    """
    Apply Z-score and IQR thresholding to produce frame-level anomaly masks.

    Deliberate methodological choice — different feature representations per method:
      - Z-score on X_RAW (unscaled): Z-score measures deviation in standard-deviation
        units and is most interpretable on the original feature scale (px/sec). Applying
        it to MinMax-scaled values compresses variance and distorts the statistic.
      - IQR on X_SCALED: IQR thresholding is scale-invariant by nature and works
        correctly on the [0,1] domain produced by the Block C fitted scaler. Using the
        scaled domain ensures the threshold is consistent with the training distribution.

    Args:
        X_raw    : (N, 2) unscaled [vel_px_sec, vel_px_sec_smooth]
        X_scaled : (N, 2) MinMax-scaled, clipped to [0, 1]
        z_thresh : Z-score threshold (default 2.0 σ)
        iqr_mult : IQR fence multiplier (default 1.5 × IQR)

    Returns:
        dict with keys 'zscore' and 'iqr', each a boolean array of shape (N,).
    """
    results: Dict[str, np.ndarray] = {}

    # ── Z-score on raw features (Fix F18 note: do NOT use X_scaled here) ────
    if X_raw.shape[0] < 2:
        z_mask = np.zeros(len(X_raw), dtype=bool)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = zscore(X_raw, axis=0)
        z_mask = (np.abs(z) > z_thresh).any(axis=1)
    results["zscore"] = z_mask

    # ── IQR on scaled features ────────────────────────────────────────────────
    if X_scaled.shape[0] < 4:
        iqr_mask = np.zeros(len(X_scaled), dtype=bool)
    else:
        Q1  = np.percentile(X_scaled, 25, axis=0)
        Q3  = np.percentile(X_scaled, 75, axis=0)
        IQR = Q3 - Q1
        iqr_mask = (
            (X_scaled < Q1 - iqr_mult * IQR) |
            (X_scaled > Q3 + iqr_mult * IQR)
        ).any(axis=1)
    results["iqr"] = iqr_mask

    return results


def event_level_prediction(frame_mask: np.ndarray) -> int:
    """
    Convert a per-frame anomaly mask to an event-level binary prediction.
    Event is predicted positive if ANY frame in the window is flagged.
    (per_event_reporting=true per config_blockD.yaml)
    """
    return int(frame_mask.any())


# =============================================================================
# SECTION 11 — EVALUATION
# =============================================================================

def evaluate_per_type_per_method(
    y_true: Dict[str, List[int]],
    y_pred: Dict[str, Dict[str, List[int]]],
) -> pd.DataFrame:
    """
    Compute Precision, Recall, F1 for each (event_type, method) combination.

    Args:
        y_true : {'stall': [0/1, ...], 'crash': [0/1, ...]}
        y_pred : {'stall': {'zscore': [...], 'iqr': [...]},
                  'crash': {'zscore': [...], 'iqr': [...]}}

    Returns:
        DataFrame with columns [Method, Dataset, Event_Type, Precision, Recall, F1, FAR]
    """
    rows = []
    for event_type in ("stall", "crash"):
        for method in ("zscore", "iqr"):
            yt = y_true.get(event_type, [])
            yp = y_pred.get(event_type, {}).get(method, [])
            if not yt or not yp:
                continue
            p = precision_score(yt, yp, zero_division=0)
            r = recall_score(yt, yp, zero_division=0)
            f = f1_score(yt, yp, zero_division=0)
            method_label = (
                "Rule-Based (Z-score)" if method == "zscore" else "Rule-Based (IQR)"
            )
            print(f"  [{event_type}] [{method_label}]: P={p:.3f}  R={r:.3f}  F1={f:.3f}")
            rows.append({
                "Method":             method_label,
                "Dataset":            "AI City Track 4",
                "Event_Type":         event_type,
                "Precision":          round(p, 4),
                "Recall":             round(r, 4),
                "F1":                 round(f, 4),
                "FAR (normal clips)": "TBD",   # Day 17
            })
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 12 — RESULTS TABLE BUILDER
# =============================================================================

RESULTS_TABLE_SCHEMA = [
    "Method", "Dataset", "Event_Type", "Precision", "Recall", "F1", "FAR (normal clips)"
]

def build_results_table(rule_based_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append rule-based rows to the full Block D results table.
    OC-SVM and Isolation Forest rows are pre-filled as Day 14 placeholders.

    Saves to logs/block_d/block_d_results.csv.
    """
    placeholder_rows = []
    for method in ("OC-SVM", "Isolation Forest"):
        for event_type in ("stall", "crash"):
            placeholder_rows.append({
                "Method":             method,
                "Dataset":            "AI City Track 4",
                "Event_Type":         event_type,
                "Precision":          "—",
                "Recall":             "—",
                "F1":                 "—",
                "FAR (normal clips)": "TBD",
            })

    combined = pd.concat(
        [rule_based_df, pd.DataFrame(placeholder_rows)],
        ignore_index=True
    )

    BLOCK_D_LOG_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RESULTS_CSV, index=False)
    print(f"[Block D] Results table saved: {RESULTS_CSV}")
    return combined


# =============================================================================
# SECTION 13 — EVENT SLICER VERIFICATION LOG
# =============================================================================

def append_verification_log(
    clip_id: str,
    event_type: str,
    gt_start_sec: float,
    computed_start_frame: int,
    visual_match: str = "UNCHECKED",
    notes: str = "",
    short_window: bool = False,
) -> None:
    """
    Append one row to the event slicer verification CSV (mandatory Day 15 input).

    Schema: clip_id, event_type, gt_start_sec, computed_start_frame,
            visual_match (True/False/UNCHECKED), short_window, notes
    """
    BLOCK_D_LOG_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "clip_id":               clip_id,
        "event_type":            event_type,
        "gt_start_sec":          gt_start_sec,
        "computed_start_frame":  computed_start_frame,
        "visual_match":          visual_match,
        "short_window":          short_window,
        "notes":                 notes,
    }])
    write_header = not VERIF_LOG_CSV.exists()
    row.to_csv(VERIF_LOG_CSV, mode="a", header=write_header, index=False)


# =============================================================================
# SECTION 14 — MAIN PIPELINE (called by pipeline_controller.run_block_d())
# =============================================================================

def run_rule_based_baseline(
    cfg_d: dict,
    cfg_c: dict,
    events_path: str | Path = EVENTS_JSON,
    scaler_path: str | Path = SCALER_PKL,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full Day 13 Block D rule-based pipeline.

    Calls (in order):
        audit_config → load_events → load_scaler →
        [per event: get_actual_fps → extract_event_window →
                    load_trajectory_df → filter_short_tracks →
                    interpolate_gaps → strip_warmup →
                    extract_f2_features → scale_f2_and_log →
                    rule_based_score → event_level_prediction →
                    append_verification_log] →
        evaluate_per_type_per_method → build_results_table

    Returns:
        Full Block D results DataFrame (including Day 14 placeholders).
    """
    # Config audit
    audit_config(cfg_d)

    # Load events
    stall_events, crash_events = load_events(events_path)

    # Load scaler ONCE outside all loops (Fix F13)
    scaler = load_scaler(scaler_path)

    y_true: Dict[str, List[int]] = {"stall": [], "crash": []}
    y_pred: Dict[str, Dict[str, List[int]]] = {
        "stall": {"zscore": [], "iqr": []},
        "crash": {"zscore": [], "iqr": []},
    }

    for event in stall_events + crash_events:
        clip_id    = event["clip_id"]
        evt_type   = event["event_type"]
        gt_start   = float(event["gt_start_sec"])
        gt_end     = float(event["gt_end_sec"])
        video_path = REPO_ROOT / event["video_path"]

        if verbose:
            print(f"\n[Block D] Processing {evt_type.upper()} clip: {clip_id}")

        # FPS extraction — never hardcoded (Fix F03)
        try:
            actual_fps = get_actual_fps(video_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP (FPS): {e}")
            append_verification_log(clip_id, evt_type, gt_start,
                                    int(gt_start * 25), notes=str(e))
            continue

        # Computed start frame for verification log
        computed_start_frame = int(gt_start * actual_fps)
        context_frames = int(CONTEXT_SEC * actual_fps)
        jump_to = max(0, computed_start_frame - context_frames)

        # Short-window soft guard (integration risk #4)
        event_dur_frames = int((gt_end - gt_start) * actual_fps)
        warmup_frames    = max(10, int(0.4 * actual_fps))
        short_window     = (event_dur_frames - warmup_frames) < MIN_FRAMES_WARN
        if short_window:
            print(f"  [WARNING] Short event window: ~{event_dur_frames} frames, "
                  f"~{event_dur_frames - warmup_frames} after warmup. "
                  "Proceeding (soft guard — Fix F00 note on integration risk #4).")

        # Trajectory loading
        traj_path = find_trajectory_path(clip_id)
        if traj_path is None:
            print(f"  SKIP (trajectory): No _sort_final.json for {clip_id}. "
                  "Run Block B on AI City clips first.")
            append_verification_log(clip_id, evt_type, gt_start, computed_start_frame,
                                    notes="No trajectory found", short_window=short_window)
            continue

        try:
            df_traj = load_trajectory_df(traj_path)
        except Exception as e:
            print(f"  SKIP (trajectory load error): {e}")
            continue

        # Subset to event window frames (integration risk #5: re-index relative to jump_to)
        end_frame    = int(gt_end * actual_fps) + context_frames
        df_window    = df_traj[
            (df_traj["frame_idx"] >= jump_to) & (df_traj["frame_idx"] <= end_frame)
        ].copy()
        # Re-index frame_idx relative to jump_to (Fix F00 track-ID boundary alignment)
        df_window["frame_idx_rel"] = df_window["frame_idx"] - jump_to
        df_window["frame_idx"]     = df_window["frame_idx_rel"]
        df_window = df_window.drop(columns=["frame_idx_rel"])

        # Apply Block C pre-processing pipeline (F11 → F22 → F16)
        df_window = filter_short_tracks(df_window)       # Fix F11
        df_window = interpolate_gaps(df_window)           # Fix F22
        df_window = strip_warmup(df_window, actual_fps)  # Fix F16

        if df_window.empty:
            print(f"  SKIP: No tracks remain after pre-processing for {clip_id}.")
            append_verification_log(clip_id, evt_type, gt_start, computed_start_frame,
                                    notes="Empty after pre-processing", short_window=short_window)
            continue

        # F2 feature extraction (no F1, no F3 — F2_only per Block C selection)
        df_f2        = extract_f2_features(df_window, actual_fps)
        df_f2_agg    = aggregate_f2_per_frame(df_f2)

        if df_f2_agg.empty:
            print(f"  SKIP: Empty F2 feature frame for {clip_id}.")
            continue

        X_raw    = df_f2_agg[["vel_px_sec", "vel_px_sec_smooth"]].values  # for Z-score
        X_scaled = scale_f2_and_log(X_raw, clip_id, scaler)               # for IQR (Fix F18)

        # Rule-based scoring
        masks = rule_based_score(X_raw, X_scaled)

        y_true[evt_type].append(1)   # all loaded events are positive
        for method in ("zscore", "iqr"):
            pred = event_level_prediction(masks[method])
            y_pred[evt_type][method].append(pred)

        append_verification_log(
            clip_id, evt_type, gt_start, computed_start_frame,
            visual_match="UNCHECKED", short_window=short_window,
            notes=f"zscore={int(masks['zscore'].any())} iqr={int(masks['iqr'].any())}"
        )

    print("\n[Block D] === Evaluation Results ===")
    results_df   = evaluate_per_type_per_method(y_true, y_pred)
    full_table   = build_results_table(results_df)
    return full_table
