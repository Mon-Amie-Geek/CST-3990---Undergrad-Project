"""
block_d_day17_far_latency.py

Block D Day 17: FAR Validation + Latency Profiling.

Performs:
  1. Guard — confirm Day 16 complete
  2. Normal clip discovery — manifest with annotation coverage caveat
  3. FAR computation — full pipeline on each normal clip
  4. FAR gate — document per-method; gate failure = "reportable but flagged"
  5. FAR statistics — mean + std of FAR across clips per method
  6. Complete Block D results table — merge Days 13–16 coverage with Day 17 FAR
  7. Latency profiling — per-stage and end-to-end; single-clip; caveat documented
  8. Update config_blockD.yaml
  9. Produce day17_report.json

Frozen decisions (DO NOT change):
  Detector : yolov8n
  Tracker  : sort
  Feature  : F2_only (vel_px_sec, vel_px_sec_smooth)
  Scaler   : models/minmax_scaler.pkl — read-only (never refit)
  OC-SVM   : logs/block_c/ocsvm_trained_best.pkl
  IF       : logs/block_d/isolation_forest_f2_train_normal.pkl (cfg path)
  Seed     : 42
  FAR threshold: loaded from cfg['far_threshold'] — NEVER hardcoded

Student : MANJOO Ameera Najla | M01014463
Module  : CST3990 Undergraduate Individual Project — Block D Day 17
"""

from __future__ import annotations

# ── METRIC DEFINITIONS ─────────────────────────────────────────────────────────
# anomalous_ratio: fraction of windows flagged on ANOMALY clips (Days 13–16).
#   High = good detection. Sourced from block_d_final_results.csv. NOT recomputed here.
# FAR (False Alarm Rate): fraction of windows flagged on NORMAL clips (Day 17).
#   High = bad (excessive false alarms). Computed fresh in this module.
# Both are structurally identical formulae applied to different clip populations.
# ──────────────────────────────────────────────────────────────────────────────

import json
import logging
import math
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
import psutil
import yaml
from scipy.stats import zscore as scipy_zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

# ── Shared helpers — import from Day 15 (do NOT redefine) ─────────────────────
try:
    from src.block_d_day15_bridge import (
        EXPECTED_METHODS,
        METHOD_NAME_MAP,
        compute_anomalous_ratio,
    )
except ImportError:
    from block_d_day15_bridge import (  # type: ignore[no-redef]
        EXPECTED_METHODS,
        METHOD_NAME_MAP,
        compute_anomalous_ratio,
    )

# ── Pipeline helpers — reuse Block D rule-based utilities ──────────────────────
try:
    from src.block_d_rule_based import (
        aggregate_f2_per_frame,
        extract_f2_features,
        filter_short_tracks,
        interpolate_gaps,
        strip_warmup,
    )
except ImportError:
    from block_d_rule_based import (  # type: ignore[no-redef]
        aggregate_f2_per_frame,
        extract_f2_features,
        filter_short_tracks,
        interpolate_gaps,
        strip_warmup,
    )

# ── SORT tracker ───────────────────────────────────────────────────────────────
try:
    from src.sort_tracker import SORTTracker
except ImportError:
    from sort_tracker import SORTTracker  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_NORMAL_CLIPS = 3          # minimum readable normal clips to proceed
ANOMALY_VALUES   = {1, True, -1}
F2_COLS          = ["vel_px_sec", "vel_px_sec_smooth"]
# Indices of F2 features in the 7-feature global scaler.
# Scaler feature_names_in_: ['vel_px_sec', 'vel_px_sec_smooth', 'vehicle_count',
#   'roi_occupancy', 'inter_vehicle_dist_norm', 'dwell_time_sec', 'proximity_count_rolling']
F2_INDICES       = [0, 1]
MIN_TRACK_LENGTH = 15         # fragment filter: min genuine detections per track
MAX_INTERP_GAP   = 3          # gap interpolation: max missing frames to fill
SPEED_WINDOW_SEC = 0.2        # rolling smoothing window (seconds)
VIDEO_EXTS       = {".mp4", ".avi", ".mov"}

# THESIS NOTE — FAR threshold justification (§4, Discussion):
# The 10% FAR threshold reflects a practical surveillance tolerance tradeoff:
# accepting up to 10 false alerts per 100 normal traffic windows before
# anomaly detection is considered unreliable. This aligns with prior work
# on unsupervised anomaly detection in traffic surveillance (Chandola et al., 2009;
# Pang et al., 2021). The threshold is loaded from config (Fix F14) and not hardcoded.

# THESIS NOTE — rule_based z-score scope (§4, Method Discussion):
# Rule-based detection uses clip-local z-score thresholds, not thresholds trained
# on UA-DETRAC. This makes it adaptive per clip. A consequence is that FAR for
# rule_based depends on within-clip variance: a very uniform normal clip may still
# flag outlier windows. This is declared as a known limitation, not a defect.

# THESIS NOTE — latency measurement scope (§3.7.1, Implementation Details):
# All latency results are hardware- and backend-specific indicative measurements
# on Microsoft Surface Pro 7 in offline batch mode (Fix F29). They are not
# universal system guarantees. Single-clip benchmark.

_MEASUREMENT_CAVEATS = (
    "Timings are hardware- and backend-specific indicative measurements on Microsoft "
    "Surface Pro 7 in offline batch mode (Fix F29). They are not universal system "
    "guarantees. Python/OpenCV timing in Colab-like environments is subject to runtime "
    "variance. Single-clip benchmark; results may not generalise to all clip durations "
    "or content. If detection cache was unavailable, detect-stage timing reflects "
    "on-the-fly inference cost."
)


# ─────────────────────────────────────────────────────────────────────────────
# 7.2  guard_day16_complete
# ─────────────────────────────────────────────────────────────────────────────

def guard_day16_complete() -> None:
    """Raise if Day 16 is not fully validated."""
    path = "logs/block_d/day16_validation_report.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Complete Day 16 before Day 17."
        )
    with open(path) as fh:
        report = json.load(fh)
    if not report.get("day16_exit_criteria_met", False):
        raise RuntimeError(
            "Day 16 exit criteria not met. "
            "Resolve all Day 16 issues before running Day 17."
        )
    logger.info("[Day 17] Day 16 guard passed.")


# ─────────────────────────────────────────────────────────────────────────────
# Internal: dict-unwrapper
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(obj, candidates: list):
    """
    Extract a fitted sklearn object from a dict wrapper or return the object itself.
    Priority: named candidates, then any value with a .predict method.
    """
    if isinstance(obj, dict):
        for k in candidates:
            if k in obj:
                return obj[k]
        # fallback: first value that looks like a fitted sklearn object
        for v in obj.values():
            if hasattr(v, "predict"):
                return v
        raise ValueError(
            f"Could not find sklearn object in dict with keys: {list(obj.keys())}"
        )
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# 7.3  load_models
# ─────────────────────────────────────────────────────────────────────────────

def load_models(cfg: dict) -> dict:
    """
    Load and unwrap all three anomaly models and the scaler.

    Returns a dict with:
        scaler, ocsvm, isolation_forest, rule_based_z_thresh
    """
    paths = cfg.get("paths", {})

    scaler_path = paths.get("scaler", "models/minmax_scaler.pkl")
    ocsvm_path  = paths.get("ocsvm_model", "logs/block_c/ocsvm_trained_best.pkl")
    # Prefer config path (isolation_forest_model); fall back to spec default.
    isofor_path = paths.get(
        "isolation_forest_model",
        "logs/block_d/isolation_forest_model.pkl",
    )
    # The committed artifact uses a different stem — resolve it.
    if not os.path.exists(isofor_path):
        alt = "logs/block_d/isolation_forest_f2_train_normal.pkl"
        if os.path.exists(alt):
            logger.info(
                "[load_models] isolation_forest_model not at '%s'; using '%s'.",
                isofor_path, alt,
            )
            isofor_path = alt
        else:
            raise FileNotFoundError(
                f"Isolation Forest model not found at '{isofor_path}' or '{alt}'."
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_scaler = joblib.load(scaler_path)
        raw_ocsvm  = joblib.load(ocsvm_path)
        raw_isofor = joblib.load(isofor_path)

    scaler_obj  = _unwrap(raw_scaler, ["scaler", "model", "minmax_scaler"])
    ocsvm_obj   = _unwrap(raw_ocsvm,  ["model", "ocsvm", "classifier"])
    isofor_obj  = _unwrap(raw_isofor, ["model", "isolation_forest", "iforest", "classifier"])

    assert isinstance(scaler_obj, MinMaxScaler), (
        f"Expected MinMaxScaler, got {type(scaler_obj)}"
    )
    assert isinstance(ocsvm_obj, OneClassSVM), (
        f"Expected OneClassSVM, got {type(ocsvm_obj)}"
    )
    assert isinstance(isofor_obj, IsolationForest), (
        f"Expected IsolationForest, got {type(isofor_obj)}"
    )

    logger.info(
        "[load_models] Scaler: %d features | OC-SVM: %d features | IF: %d features",
        scaler_obj.n_features_in_,
        ocsvm_obj.n_features_in_,
        isofor_obj.n_features_in_,
    )

    return {
        "scaler":              scaler_obj,
        "ocsvm":               ocsvm_obj,
        "isolation_forest":    isofor_obj,
        "rule_based_z_thresh": float(cfg.get("rule_based_z_thresh", 2.5)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal: F2 scaling (matches block_d_rule_based.scale_f2_and_log convention)
# ─────────────────────────────────────────────────────────────────────────────

def _scale_f2(X_raw: np.ndarray, scaler: MinMaxScaler, clip_id: str) -> Tuple[np.ndarray, float]:
    """
    Scale F2-only feature matrix using the global MinMaxScaler's F2 slice.

    The global scaler was fitted on 7 features; F2 (vel_px_sec, vel_px_sec_smooth)
    occupy indices [0, 1]. We apply the scaler's data_min_/data_range_ for those
    indices only rather than calling scaler.transform(X_raw) directly, which would
    raise a feature-count mismatch error. This is consistent with
    block_d_rule_based.scale_f2_and_log (Fix F18).

    Returns
    -------
    X_scaled : np.ndarray, clipped to [0, 1]
    pct_clipped : float — percentage of values outside [0, 1] before clipping
    """
    f2_min   = scaler.data_min_[F2_INDICES]
    f2_range = scaler.data_range_[F2_INDICES]

    with np.errstate(divide="ignore", invalid="ignore"):
        X_scaled = (X_raw - f2_min) / np.where(f2_range == 0, 1.0, f2_range)

    n_clipped   = int(((X_scaled > 1.0) | (X_scaled < 0.0)).sum())
    pct_clipped = n_clipped / X_scaled.size * 100.0 if X_scaled.size else 0.0

    X_scaled = np.clip(X_scaled, 0.0, 1.0)

    flag = "  WARNING: HIGH DOMAIN SHIFT" if pct_clipped > 20.0 else ""
    logger.info(
        "[Fix F18] [%s] Domain-shift indicator: %.1f%% of F2 values clipped to [0,1]%s",
        clip_id, pct_clipped, flag,
    )
    return X_scaled, pct_clipped


# ─────────────────────────────────────────────────────────────────────────────
# 7.4  discover_normal_clips
# ─────────────────────────────────────────────────────────────────────────────

def discover_normal_clips(cfg: dict) -> pd.DataFrame:
    """
    Walk the AI City root and identify normal clips.

    Returns a DataFrame with manifest columns (all clips, readable and not).
    Writes manifest to logs/block_d/day17_normal_clips_manifest.csv.
    Raises RuntimeError('[BLOCKER]...') if fewer than MIN_NORMAL_CLIPS readable clips found.

    Annotation coverage caveat (Section 5.1):
        AI City Track 4 annotations are not guaranteed exhaustive. A clip absent
        from events.json is *likely* normal but not *confirmed* normal. This
        limitation must be declared in thesis §3.8 (Threats to Validity).
    """
    ai_city_root = cfg.get(
        "ai_city_root",
        cfg.get("ai_city_video_dir", "data/ai_city"),
    )
    ai_city_root = str(ai_city_root).rstrip("/\\")

    events_path = cfg.get("events_json", "data/ai_city/events.json")
    if not os.path.exists(events_path):
        events_path = os.path.join(ai_city_root, "events.json")

    # ── Build anomaly_clips set from events.json ───────────────────────────────
    anomaly_clips: set[str] = set()
    if os.path.exists(events_path):
        with open(events_path) as fh:
            data = json.load(fh)
        evts = data if isinstance(data, list) else data.get("events", [])
        for e in evts:
            key = (
                e.get("video_id")
                or e.get("video_file")
                or e.get("clip_id")
            )
            if key:
                stem = (
                    str(key)
                    .replace(".mp4", "")
                    .replace(".avi", "")
                    .replace(".mov", "")
                    .lower()
                )
                anomaly_clips.add(stem)
    logger.info("[Day 17] Anomaly clips from events.json: %d", len(anomaly_clips))

    # ── Walk AI City root ──────────────────────────────────────────────────────
    rows: list[dict] = []

    # Priority 1: explicit normal/ subdirectory or normal_clips.txt
    normal_txt = os.path.join(ai_city_root, "normal_clips.txt")
    labelled_normal_paths: set[str] = set()

    if os.path.exists(normal_txt):
        with open(normal_txt) as fh:
            for line in fh:
                p = line.strip()
                if p and os.path.exists(p):
                    labelled_normal_paths.add(os.path.abspath(p))

    # Walk all video files
    visited: set[str] = set()
    for root, dirs, files in os.walk(ai_city_root):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VIDEO_EXTS:
                continue

            full_path = os.path.abspath(os.path.join(root, fname))
            if full_path in visited:
                continue
            visited.add(full_path)

            stem_lower = os.path.splitext(fname)[0].lower()
            in_normal_subdir = (
                "normal" in os.path.relpath(root, ai_city_root).lower().split(os.sep)
                or "normal" in os.path.basename(root).lower()
            )
            in_labelled = full_path in labelled_normal_paths

            if in_labelled or in_normal_subdir:
                source_note           = "labelled_normal_in_dataset"
                annotation_coverage_note = "explicitly_labelled_normal"
            elif stem_lower not in anomaly_clips:
                source_note           = "not_in_events_json"
                annotation_coverage_note = "not_in_events_json_assumed_normal"
            else:
                # File is an anomaly clip — skip
                continue

            # ── Readability check ──────────────────────────────────────────────
            cap = cv2.VideoCapture(full_path)
            readable = cap.isOpened()
            fps_val        = None
            total_frames   = None
            clip_duration  = None

            if readable:
                fps_val      = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if fps_val and fps_val > 0:
                    clip_duration = round(total_frames / fps_val, 2)
                else:
                    readable = False  # invalid FPS → not usable
            cap.release()

            rows.append(
                {
                    "clip_path":               full_path,
                    "clip_filename":           fname,
                    "fps":                     fps_val,
                    "total_frames":            total_frames,
                    "clip_duration_sec":       clip_duration,
                    "source_note":             source_note,
                    "annotation_coverage_note": annotation_coverage_note,
                    "readable":                readable,
                }
            )

    if not rows:
        rows.append(
            {
                "clip_path":               "",
                "clip_filename":           "",
                "fps":                     None,
                "total_frames":            None,
                "clip_duration_sec":       None,
                "source_note":             "NONE_FOUND",
                "annotation_coverage_note": "NONE_FOUND",
                "readable":                False,
            }
        )

    # Sort: labelled_normal first, then not_in_events_json, then NONE_FOUND
    priority = {
        "labelled_normal_in_dataset": 0,
        "not_in_events_json":         1,
        "NONE_FOUND":                 99,
    }
    rows.sort(key=lambda r: (priority.get(r["source_note"], 50), r["clip_filename"]))

    manifest_df = pd.DataFrame(rows)

    # ── Write manifest ─────────────────────────────────────────────────────────
    os.makedirs("logs/block_d", exist_ok=True)
    manifest_path = "logs/block_d/day17_normal_clips_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    logger.info("[Day 17] Normal clips manifest written: %s", manifest_path)

    readable_count = int(manifest_df["readable"].sum())
    logger.info(
        "[Day 17] Normal clips discovered: %d total, %d readable.",
        len(manifest_df), readable_count,
    )

    if readable_count < MIN_NORMAL_CLIPS:
        raise RuntimeError(
            f"[BLOCKER] FAR validation requires at least {MIN_NORMAL_CLIPS} readable "
            f"normal clips. Found {readable_count} readable clip(s).\n"
            f"Manifest written to: {manifest_path}\n\n"
            "Manual resolution:\n"
            "  1. Identify ≥3 AI City clips with no annotated anomalies.\n"
            "  2. Place them in a 'normal/' subdirectory under AI City root, OR\n"
            "  3. Create a 'normal_clips.txt' file in the AI City root listing "
            "their paths (one per line).\n"
            "  4. Re-run run_day17() or Cell 7 of the notebook."
        )

    return manifest_df


# ─────────────────────────────────────────────────────────────────────────────
# Internal: on-the-fly YOLO detection
# ─────────────────────────────────────────────────────────────────────────────

def _run_yolo_on_clip(video_path: str, cfg: dict) -> list[dict]:
    """
    Run yolov8n inference on every frame of a video.
    Returns list of {frame_idx, detections: np.ndarray shape (N,5) [x1,y1,x2,y2,conf]}.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics not installed. Install with: pip install ultralytics"
        )

    model_path = cfg.get("yolo_model_path", "models/best.pt")
    if not os.path.exists(model_path):
        model_path = "yolov8n"  # download from hub if not cached

    yolo = YOLO(model_path)
    conf_thresh = float(cfg.get("yolo_conf_thresh", 0.25))

    cap = cv2.VideoCapture(video_path)
    frame_results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo(frame, verbose=False, conf=conf_thresh)
        dets = []
        for r in results:
            if r.boxes is not None and len(r.boxes):
                boxes = r.boxes.xyxy.cpu().numpy()   # [N,4]
                confs = r.boxes.conf.cpu().numpy()   # [N]
                dets = np.column_stack([boxes, confs])  # [N,5]
        if not isinstance(dets, np.ndarray) or len(dets) == 0:
            dets = np.empty((0, 5), dtype=np.float32)
        frame_results.append({"frame_idx": frame_idx, "detections": dets})
        frame_idx += 1

    cap.release()
    return frame_results


def _load_or_generate_detections(
    video_path: str,
    clip_stem: str,
    cfg: dict,
) -> Tuple[list[dict], str]:
    """
    Check detections_cache/ for this clip. If found, load; otherwise run yolov8n.

    Returns (frame_results, detection_source).
    detection_source: 'cache' | 'on_the_fly'
    """
    cache_dir  = cfg.get("detections_cache_dir", "detections_cache/ai_city")
    cache_file = os.path.join(cache_dir, f"{clip_stem}.pkl")

    if os.path.exists(cache_file):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame_results = joblib.load(cache_file)
        logger.info("[Day 17] Detections loaded from cache: %s", cache_file)
        return frame_results, "cache"

    logger.info(
        "[INFO: on-the-fly detection for %s] Running yolov8n inference.", clip_stem
    )
    frame_results = _run_yolo_on_clip(video_path, cfg)

    # Cache for future runs
    os.makedirs(cache_dir, exist_ok=True)
    joblib.dump(frame_results, cache_file)
    logger.info("[Day 17] Detection cache written: %s", cache_file)

    return frame_results, "on_the_fly"


# ─────────────────────────────────────────────────────────────────────────────
# Internal: build trajectory DataFrame from SORT tracker output
# ─────────────────────────────────────────────────────────────────────────────

def _build_trajectory_df(
    frame_results: list[dict],
    frame_width: int,
    frame_height: int,
    fps: float,
) -> pd.DataFrame:
    """
    Run SORT on pre-computed detections and compute velocity_norm per track.

    velocity_norm = dist_px * fps / frame_diagonal
    (consistent with block_b_sort_runner.compute_velocity_norm)

    Returns DataFrame with columns:
        frame_idx, track_id, cx, cy, velocity_norm, conf, is_interpolated
    """
    frame_diagonal = math.sqrt(frame_width ** 2 + frame_height ** 2)
    tracker = SORTTracker(max_age=5, min_hits=3, iou_threshold=0.3, conf_thresh=0.25)

    # Per-track last-seen centroid for velocity computation
    prev_centroids: Dict[int, Tuple[float, float]] = {}
    rows: list[dict] = []

    for entry in frame_results:
        fidx = int(entry["frame_idx"])
        dets = entry["detections"]
        if not isinstance(dets, np.ndarray) or dets.ndim != 2 or dets.shape[1] < 5:
            dets = np.empty((0, 5), dtype=np.float32)

        tracked = tracker.update(dets, fidx)  # shape (M, 6): [x1,y1,x2,y2,track_id,conf]

        for t in tracked:
            x1, y1, x2, y2, tid, conf = (
                float(t[0]), float(t[1]), float(t[2]),
                float(t[3]), int(t[4]),  float(t[5]),
            )
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if tid in prev_centroids:
                px, py = prev_centroids[tid]
                dist_px     = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                vel_px_sec  = dist_px * fps
                velocity_norm = vel_px_sec / frame_diagonal if frame_diagonal > 0 else 0.0
            else:
                velocity_norm = None  # first appearance

            prev_centroids[tid] = (cx, cy)

            rows.append(
                {
                    "frame_idx":     fidx,
                    "track_id":      tid,
                    "cx":            cx,
                    "cy":            cy,
                    "velocity_norm": velocity_norm,
                    "conf":          conf,
                    "is_interpolated": False,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["frame_idx", "track_id", "cx", "cy",
                     "velocity_norm", "conf", "is_interpolated"]
        )

    df = pd.DataFrame(rows)
    df["frame_idx"] = df["frame_idx"].astype(int)
    df["track_id"]  = df["track_id"].astype(int)
    return df.sort_values(["frame_idx", "track_id"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7.5  extract_features_for_clip
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_for_clip(
    video_path: str,
    cfg: dict,
    models: dict,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Run the full detection → track → feature → scale pipeline on a single clip.

    Returns
    -------
    X_scaled   : np.ndarray shape (n_windows, 2) or None on failure
    diagnostics: dict with status and per-step counts
    """
    clip_filename = os.path.basename(video_path)
    clip_stem     = os.path.splitext(clip_filename)[0]

    # ── 1. Open video ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None, {"status": "UNREADABLE", "clip_filename": clip_filename}

    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not (0 < fps <= 120):
        return None, {
            "status": "INVALID_FPS",
            "clip_filename": clip_filename,
            "fps": fps,
        }

    # ── 2. Detection (cache or on-the-fly) ────────────────────────────────────
    try:
        frame_results, detection_source = _load_or_generate_detections(
            video_path, clip_stem, cfg
        )
    except Exception as exc:
        logger.error("[Day 17] Detection failed for %s: %s", clip_stem, exc)
        return None, {
            "status": "DETECTION_FAILED",
            "clip_filename": clip_filename,
            "error": str(exc),
        }

    # ── 3. Track with SORT ────────────────────────────────────────────────────
    df_traj = _build_trajectory_df(frame_results, frame_width, frame_height, fps)
    n_raw_windows = len(df_traj)

    if df_traj.empty:
        return None, {
            "status": "NO_TRAJECTORIES",
            "clip_filename": clip_filename,
            "n_raw_windows": 0,
            "detection_source": detection_source,
        }

    # ── 4. Fragment filter (min track length 15) ──────────────────────────────
    df_traj = filter_short_tracks(df_traj, min_length=MIN_TRACK_LENGTH)

    # ── 5. Gap interpolation (gap ≤ 3) ────────────────────────────────────────
    df_traj = interpolate_gaps(df_traj, max_gap=MAX_INTERP_GAP)

    # ── 6. Strip warmup ───────────────────────────────────────────────────────
    df_traj = strip_warmup(df_traj, actual_fps=fps)
    n_after_warmup = len(df_traj)

    if df_traj.empty:
        return None, {
            "status": "EMPTY_AFTER_WARMUP",
            "clip_filename": clip_filename,
            "n_raw_windows": n_raw_windows,
            "n_after_warmup": 0,
            "detection_source": detection_source,
        }

    # ── 7. Extract F2_only features ───────────────────────────────────────────
    df_f2 = extract_f2_features(df_traj, actual_fps=fps, speed_window_sec=SPEED_WINDOW_SEC)
    df_agg = aggregate_f2_per_frame(df_f2)

    if df_agg.empty or not all(c in df_agg.columns for c in F2_COLS):
        return None, {
            "status": "FEATURE_EXTRACTION_FAILED",
            "clip_filename": clip_filename,
            "n_raw_windows": n_raw_windows,
            "n_after_warmup": n_after_warmup,
            "detection_source": detection_source,
        }

    X_raw = df_agg[F2_COLS].dropna().values  # shape (n_frames, 2)

    if len(X_raw) < 5:
        return None, {
            "status": "TOO_SHORT",
            "clip_filename": clip_filename,
            "n_windows": len(X_raw),
            "n_raw_windows": n_raw_windows,
            "n_after_warmup": n_after_warmup,
            "detection_source": detection_source,
        }

    # ── 8. Scale (F2 slice of global scaler) — NEVER fit/fit_transform ────────
    X_scaled, pct_clipped = _scale_f2(X_raw, models["scaler"], clip_stem)

    n_clipped = int(round(pct_clipped * X_scaled.size / 100.0))
    domain_shift_note = (
        "HIGH DOMAIN SHIFT (>20%)" if pct_clipped > 20.0 else "within expected range"
    )

    diagnostics = {
        "status":           "OK",
        "clip_filename":    clip_filename,
        "n_raw_windows":    n_raw_windows,
        "n_after_warmup":   n_after_warmup,
        "n_clipped_features": n_clipped,
        "pct_clipped":      round(pct_clipped, 4),
        "domain_shift_note": domain_shift_note,
        "detection_source": detection_source,
    }

    return X_scaled, diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# 7.6  run_method_on_features
# ─────────────────────────────────────────────────────────────────────────────

def run_method_on_features(
    X_scaled: np.ndarray,
    method_name: str,
    models: dict,
) -> np.ndarray:
    """
    Apply the named anomaly detection method to a scaled feature matrix.

    Returns binary prediction array: 1 = anomaly, 0 = normal.

    DESIGN NOTE for rule_based:
        rule_based is adaptive per clip — it uses no trained UA-DETRAC model.
        FAR for rule_based depends on within-clip variance. A low-variance
        normal clip may still trigger if any window is a within-clip outlier.
        This is consistent with how rule_based was applied in Days 13–16.
        Thesis §4 should state: "rule-based detection uses clip-local z-score
        thresholds; it is inherently adaptive rather than globally calibrated."
    """
    if method_name == "rule_based":
        z_thresh = models["rule_based_z_thresh"]
        z = scipy_zscore(X_scaled, axis=0, nan_policy="omit")
        z = np.nan_to_num(z, nan=0.0)
        return (np.abs(z) > z_thresh).any(axis=1).astype(int)

    elif method_name == "ocsvm":
        raw = models["ocsvm"].predict(X_scaled)   # +1 = normal, -1 = anomaly
        return (raw == -1).astype(int)

    elif method_name == "isolation_forest":
        raw = models["isolation_forest"].predict(X_scaled)  # +1 = normal, -1 = anomaly
        return (raw == -1).astype(int)

    else:
        raise ValueError(f"Unknown method after normalisation: {method_name!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.7  compute_far_on_normal_clips
# ─────────────────────────────────────────────────────────────────────────────

def compute_far_on_normal_clips(
    normal_manifest_df: pd.DataFrame,
    cfg: dict,
    models: dict,
) -> pd.DataFrame:
    """
    Run the full pipeline on every readable normal clip and compute per-clip FAR.

    Writes logs/block_d/day17_far_per_clip.csv. Returns the DataFrame.
    """
    readable_clips = normal_manifest_df[normal_manifest_df["readable"] == True]  # noqa: E712
    records: list[dict] = []

    for _, row in readable_clips.iterrows():
        video_path    = str(row["clip_path"])
        clip_filename = str(row["clip_filename"])
        logger.info("[Day 17] Processing normal clip: %s", clip_filename)

        X_scaled, diagnostics = extract_features_for_clip(video_path, cfg, models)

        pct_clipped      = diagnostics.get("pct_clipped", np.nan)
        detection_source = diagnostics.get("detection_source", "unknown")

        if X_scaled is None:
            status = diagnostics.get("status", "SKIPPED")
            logger.warning(
                "[Day 17] Clip %s skipped — status=%s", clip_filename, status
            )
            for method in sorted(EXPECTED_METHODS):
                records.append(
                    {
                        "clip_filename":   clip_filename,
                        "method":          method,
                        "n_windows":       0,
                        "n_flagged":       0,
                        "far_value":       np.nan,
                        "pct_clipped":     pct_clipped,
                        "detection_source": detection_source,
                        "status":          status,
                    }
                )
            continue

        for method in sorted(EXPECTED_METHODS):
            preds      = run_method_on_features(X_scaled, method, models)
            n_windows  = len(preds)
            n_flagged  = int(preds.sum())
            far_value  = n_flagged / n_windows

            records.append(
                {
                    "clip_filename":    clip_filename,
                    "method":           method,
                    "n_windows":        n_windows,
                    "n_flagged":        n_flagged,
                    "far_value":        round(far_value, 6),
                    "pct_clipped":      pct_clipped,
                    "detection_source": detection_source,
                    "status":           "OK",
                }
            )

    os.makedirs("logs/block_d", exist_ok=True)
    out_df = pd.DataFrame(records)
    out_df.to_csv("logs/block_d/day17_far_per_clip.csv", index=False)
    logger.info("[Day 17] Per-clip FAR written: logs/block_d/day17_far_per_clip.csv")
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# 7.8  run_far_gate
# ─────────────────────────────────────────────────────────────────────────────

def run_far_gate(far_df: pd.DataFrame, far_threshold: float) -> dict:
    """
    Aggregate per-clip FAR to per-method summary. Include mean + std.

    Gate failure = "reportable but flagged" per Fix F14.
    Writes logs/block_d/day17_far_summary.csv. Returns gate dict.
    """
    result: dict = {}
    ok_df = far_df[far_df["status"] == "OK"]

    for method in sorted(EXPECTED_METHODS):
        sub = ok_df[ok_df["method"] == method]
        if len(sub) == 0:
            result[method] = {
                "mean_far":       None,
                "std_far":        None,
                "max_far":        None,
                "clips_evaluated": 0,
                "gate_passed":    False,
                "gate_note":      "NO_VALID_CLIPS",
            }
            continue

        far_vals = sub["far_value"].dropna()
        mean_far = float(far_vals.mean())
        std_far  = float(far_vals.std()) if len(far_vals) > 1 else float("nan")
        max_far  = float(far_vals.max())
        passed   = mean_far < far_threshold

        result[method] = {
            "mean_far":        round(mean_far, 4),
            "std_far":         round(std_far, 4) if not np.isnan(std_far) else None,
            "max_far":         round(max_far, 4),
            "clips_evaluated": int(len(sub)),
            "gate_passed":     passed,
            "gate_note": (
                "PASSED"
                if passed
                else (
                    f"FLAGGED: mean_FAR={mean_far:.4f} >= threshold={far_threshold}. "
                    "Results reportable but annotated per Fix F14."
                )
            ),
        }

    # ── Write summary CSV ──────────────────────────────────────────────────────
    summary_rows = [
        {
            "method":          m,
            "mean_far":        r["mean_far"],
            "std_far":         r["std_far"],
            "max_far":         r["max_far"],
            "clips_evaluated": r["clips_evaluated"],
            "gate_passed":     r["gate_passed"],
            "gate_note":       r["gate_note"],
        }
        for m, r in result.items()
    ]
    os.makedirs("logs/block_d", exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        "logs/block_d/day17_far_summary.csv", index=False
    )
    logger.info("[Day 17] FAR summary written: logs/block_d/day17_far_summary.csv")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7.9  build_complete_block_d_table
# ─────────────────────────────────────────────────────────────────────────────

def build_complete_block_d_table(far_gate_dict: dict) -> pd.DataFrame:
    """
    Load block_d_final_results.csv (Day 16) and merge with Day 17 FAR results.

    Writes logs/block_d/block_d_complete_results.csv.
    Does NOT overwrite block_d_final_results.csv.

    anomalous_ratio column originates from block_d_final_results.csv (Day 16).
    It represents the fraction of windows flagged on ANOMALY clips (Days 13–16).
    It is NOT recomputed here. It is NOT the same as FAR.
    FAR is computed on NORMAL clips in this module.
    """
    final_path = "logs/block_d/block_d_final_results.csv"
    if not os.path.exists(final_path):
        raise FileNotFoundError(
            f"{final_path} not found. Ensure Day 16 completed successfully."
        )

    base_df = pd.read_csv(final_path)

    # Normalise method names in base_df to canonical form
    if "method" in base_df.columns:
        base_df["method"] = base_df["method"].map(
            lambda m: METHOD_NAME_MAP.get(m, m)
        )

    output_rows: list[dict] = []

    for method in sorted(EXPECTED_METHODS):
        base_row = base_df[base_df["method"] == method]

        total_events        = int(base_row["total_events"].iloc[0])        if not base_row.empty else None
        events_detected     = int(base_row["events_detected"].iloc[0])     if not base_row.empty else None
        det_pct             = float(base_row["detection_coverage_pct"].iloc[0]) if not base_row.empty else None
        anomalous_ratio_val = (
            float(base_row["anomalous_ratio"].iloc[0])
            if not base_row.empty and pd.notna(base_row["anomalous_ratio"].iloc[0])
            else None
        )
        mean_anom_score     = (
            float(base_row["mean_anomaly_score"].iloc[0])
            if not base_row.empty
            and "mean_anomaly_score" in base_row.columns
            and pd.notna(base_row["mean_anomaly_score"].iloc[0])
            else None
        )
        short_window_count  = (
            int(base_row["short_window_event_count"].iloc[0])
            if not base_row.empty and "short_window_event_count" in base_row.columns
            else 0
        )
        base_notes = (
            str(base_row["notes"].iloc[0])
            if not base_row.empty and "notes" in base_row.columns and pd.notna(base_row["notes"].iloc[0])
            else ""
        )

        gate = far_gate_dict.get(method, {})
        mean_far          = gate.get("mean_far")
        std_far           = gate.get("std_far")
        max_far           = gate.get("max_far")
        clips_for_far     = gate.get("clips_evaluated", 0)
        far_gate_passed   = gate.get("gate_passed", False)

        notes = base_notes
        if not far_gate_passed and clips_for_far > 0:
            notes = (notes + "; FAR gate FLAGGED — results reportable but annotated per Fix F14").lstrip("; ")
        if clips_for_far == 0:
            notes = (notes + "; FAR not evaluated — insufficient normal clips").lstrip("; ")

        output_rows.append(
            {
                "method":                     method,
                "total_events":               total_events,
                "events_detected":            events_detected,
                "detection_coverage_pct":     det_pct,
                "anomalous_ratio":            anomalous_ratio_val,
                "mean_anomaly_score":         mean_anom_score,
                "mean_far":                   mean_far,
                "std_far":                    std_far,
                "max_far":                    max_far,
                "clips_evaluated_for_far":    clips_for_far,
                "far_gate_passed":            far_gate_passed,
                "short_window_event_count":   short_window_count,
                "notes":                      notes,
            }
        )

    complete_df = pd.DataFrame(output_rows)
    os.makedirs("logs/block_d", exist_ok=True)
    out_path = "logs/block_d/block_d_complete_results.csv"
    complete_df.to_csv(out_path, index=False)
    logger.info("[Day 17] Complete Block D results table written: %s", out_path)
    return complete_df


# ─────────────────────────────────────────────────────────────────────────────
# 7.10  run_latency_profiling
# ─────────────────────────────────────────────────────────────────────────────

def _stage_stats(timings_ms: list[float]) -> dict:
    """Return mean/p50/p95/min/max from a list of per-frame timings in ms."""
    arr = np.array(timings_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(arr.mean()),  3),
        "p50_ms":  round(float(np.percentile(arr, 50)), 3),
        "p95_ms":  round(float(np.percentile(arr, 95)), 3),
        "min_ms":  round(float(arr.min()),  3),
        "max_ms":  round(float(arr.max()),  3),
    }


def run_latency_profiling(cfg: dict, models: dict) -> dict:
    """
    Per-stage + end-to-end latency measurement on a single normal clip.

    Stages timed: decode | detect | track | feature | anomaly | total
    Rendering disabled. 30-frame warm-up. 3 passes × ≥100 frames.

    All timings are hardware-specific indicative values (Fix F29).
    See measurement_caveats in the returned dict for full disclaimer.
    """
    # ── 1. Select profiling clip ───────────────────────────────────────────────
    manifest_path = "logs/block_d/day17_normal_clips_manifest.csv"
    if not os.path.exists(manifest_path):
        logger.warning("[Latency] Manifest not found; skipping latency profiling.")
        return {"status": "SKIPPED_NO_MANIFEST"}

    manifest_df   = pd.read_csv(manifest_path)
    readable_df   = manifest_df[manifest_df["readable"] == True]  # noqa: E712

    if readable_df.empty:
        logger.warning("[Latency] No readable clips in manifest; skipping latency profiling.")
        return {"status": "SKIPPED_NO_READABLE_CLIPS"}

    # Prefer cached clips to avoid inflating detect stage timing
    clip_row              = None
    detection_source_note = "on_the_fly"
    cache_dir             = cfg.get("detections_cache_dir", "detections_cache/ai_city")

    for _, row in readable_df.iterrows():
        stem = os.path.splitext(str(row["clip_filename"]))[0]
        if os.path.exists(os.path.join(cache_dir, f"{stem}.pkl")):
            clip_row              = row
            detection_source_note = "cache"
            break

    if clip_row is None:
        clip_row = readable_df.iloc[0]
        logger.warning(
            "[WARNING: latency includes on-the-fly detection — "
            "detect stage timing reflects worst-case cost]"
        )

    profiling_clip = str(clip_row["clip_path"])
    clip_stem      = os.path.splitext(str(clip_row["clip_filename"]))[0]
    logger.info("[Latency] Profiling clip: %s (%s)", profiling_clip, detection_source_note)

    # ── 2. Load detections (use cache if available) ────────────────────────────
    cap = cv2.VideoCapture(profiling_clip)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not (0 < fps <= 120):
        logger.error("[Latency] Invalid FPS=%.2f for %s; aborting profiling.", fps, profiling_clip)
        return {"status": "INVALID_FPS"}

    frame_diagonal = math.sqrt(frame_width ** 2 + frame_height ** 2)

    # ── 3. Set up per-stage collections ──────────────────────────────────────
    WARMUP_FRAMES   = 30
    MEASURE_FRAMES  = 100
    PASSES          = 3

    t_decode  = []
    t_detect  = []
    t_track   = []
    t_feature = []
    t_anomaly = []
    t_total   = []

    # ── 4. Prepare YOLO if needed ─────────────────────────────────────────────
    yolo_model = None
    if detection_source_note == "on_the_fly":
        try:
            from ultralytics import YOLO
            model_path = cfg.get("yolo_model_path", "models/best.pt")
            if not os.path.exists(model_path):
                model_path = "yolov8n"
            yolo_model = YOLO(model_path)
        except ImportError:
            logger.error("[Latency] ultralytics not installed; detect stage will be zero.")

    # Pre-load cached detections if available
    cached_frame_results: Optional[list[dict]] = None
    cache_file = os.path.join(cache_dir, f"{clip_stem}.pkl")
    if os.path.exists(cache_file):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cached_frame_results = joblib.load(cache_file)

    # ── 5. Measure loop ───────────────────────────────────────────────────────
    for _pass in range(PASSES):
        sort_tracker = SORTTracker(max_age=5, min_hits=3, iou_threshold=0.3, conf_thresh=0.25)
        prev_centroids_prof: Dict[int, Tuple[float, float]] = {}
        conf_thresh = float(cfg.get("yolo_conf_thresh", 0.25))

        cap2 = cv2.VideoCapture(profiling_clip)
        frame_idx  = 0
        measured   = 0
        # Rolling buffers for feature accumulation (per-track)
        traj_rows_buf: list[dict] = []

        while cap2.isOpened() and measured < (WARMUP_FRAMES + MEASURE_FRAMES):
            # ── Decode ─────────────────────────────────────────────────────────
            t0 = time.perf_counter()
            ret, frame = cap2.read()
            t1 = time.perf_counter()
            if not ret:
                break

            timing_decode_ms = (t1 - t0) * 1000.0

            # ── Detect ─────────────────────────────────────────────────────────
            t2 = time.perf_counter()
            if cached_frame_results is not None and frame_idx < len(cached_frame_results):
                dets = cached_frame_results[frame_idx]["detections"]
                if not isinstance(dets, np.ndarray) or dets.ndim != 2 or dets.shape[1] < 5:
                    dets = np.empty((0, 5), dtype=np.float32)
            elif yolo_model is not None:
                results = yolo_model(frame, verbose=False, conf=conf_thresh)
                dets_list = []
                for r in results:
                    if r.boxes is not None and len(r.boxes):
                        boxes = r.boxes.xyxy.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()
                        dets_list = np.column_stack([boxes, confs])
                dets = np.array(dets_list) if dets_list else np.empty((0, 5), dtype=np.float32)
                if dets.ndim == 1:
                    dets = dets.reshape(1, -1) if dets.size >= 5 else np.empty((0, 5), dtype=np.float32)
            else:
                dets = np.empty((0, 5), dtype=np.float32)
            t3 = time.perf_counter()
            timing_detect_ms = (t3 - t2) * 1000.0

            # ── Track ──────────────────────────────────────────────────────────
            t4 = time.perf_counter()
            tracked = sort_tracker.update(dets, frame_idx)
            t5 = time.perf_counter()
            timing_track_ms = (t5 - t4) * 1000.0

            # ── Feature (velocity_norm, accumulate for rolling smooth) ─────────
            t6 = time.perf_counter()
            for t in tracked:
                x1, y1, x2, y2, tid, conf_val = (
                    float(t[0]), float(t[1]), float(t[2]),
                    float(t[3]), int(t[4]), float(t[5]),
                )
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                if tid in prev_centroids_prof:
                    px, py = prev_centroids_prof[tid]
                    dist_px = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    vnorm   = (dist_px * fps) / (frame_diagonal or 1.0)
                else:
                    vnorm = None
                prev_centroids_prof[tid] = (cx, cy)
                traj_rows_buf.append({
                    "frame_idx": frame_idx, "track_id": tid,
                    "cx": cx, "cy": cy,
                    "velocity_norm": vnorm, "conf": conf_val,
                    "is_interpolated": False,
                })
            t7 = time.perf_counter()
            timing_feature_ms = (t7 - t6) * 1000.0

            # ── Anomaly (all three methods on a rolling 10-frame window) ────────
            t8 = time.perf_counter()
            if len(traj_rows_buf) >= 10:
                df_tmp = pd.DataFrame(traj_rows_buf[-60:])  # last 60 rows
                df_tmp_f2 = extract_f2_features(df_tmp, actual_fps=fps, speed_window_sec=SPEED_WINDOW_SEC)
                df_agg_tmp = aggregate_f2_per_frame(df_tmp_f2)
                if not df_agg_tmp.empty and all(c in df_agg_tmp.columns for c in F2_COLS):
                    X_win = df_agg_tmp[F2_COLS].dropna().values
                    if len(X_win) > 0:
                        X_sc, _ = _scale_f2(X_win, models["scaler"], "latency_bench")
                        # Time all three methods combined
                        _ = run_method_on_features(X_sc, "rule_based",       models)
                        _ = run_method_on_features(X_sc, "ocsvm",            models)
                        _ = run_method_on_features(X_sc, "isolation_forest", models)
            t9 = time.perf_counter()
            timing_anomaly_ms = (t9 - t8) * 1000.0

            t_total_frame = (t9 - t0) * 1000.0

            frame_idx += 1

            # Skip warm-up frames
            if frame_idx <= WARMUP_FRAMES:
                continue

            t_decode.append(timing_decode_ms)
            t_detect.append(timing_detect_ms)
            t_track.append(timing_track_ms)
            t_feature.append(timing_feature_ms)
            t_anomaly.append(timing_anomaly_ms)
            t_total.append(t_total_frame)
            measured += 1

        cap2.release()

    frames_measured = len(t_total)
    if frames_measured == 0:
        return {"status": "NO_FRAMES_MEASURED", "profiling_clip": profiling_clip}

    # ── 6. Compute end-to-end FPS ──────────────────────────────────────────────
    mean_total_ms   = float(np.mean(t_total))
    end_to_end_fps  = round(1000.0 / mean_total_ms, 2) if mean_total_ms > 0 else 0.0

    # ── 7. Hardware snapshot ───────────────────────────────────────────────────
    cpu_pct   = psutil.cpu_percent(interval=0.5)
    mem_info  = psutil.virtual_memory()
    ram_used  = round(mem_info.used  / (1024 ** 3), 2)
    ram_avail = round(mem_info.available / (1024 ** 3), 2)

    # ── 8. Assemble output ─────────────────────────────────────────────────────
    latency_dict = {
        "profiling_clip":                    profiling_clip,
        "detection_source_for_profiling":    detection_source_note,
        "frames_measured":                   frames_measured,
        "passes":                            PASSES,
        "warmup_frames":                     WARMUP_FRAMES,
        "rendering_disabled":                True,
        "stages": {
            "decode":  _stage_stats(t_decode),
            "detect":  _stage_stats(t_detect),
            "track":   _stage_stats(t_track),
            "feature": _stage_stats(t_feature),
            "anomaly": _stage_stats(t_anomaly),
            "total":   _stage_stats(t_total),
        },
        "end_to_end_fps": end_to_end_fps,
        "hardware": {
            "cpu_pct_during_profiling": cpu_pct,
            "ram_used_gb":              ram_used,
            "ram_available_gb":         ram_avail,
        },
        "measurement_caveats": _MEASUREMENT_CAVEATS,
    }

    os.makedirs("logs/block_d", exist_ok=True)
    with open("logs/block_d/day17_latency_profile.json", "w") as fh:
        json.dump(latency_dict, fh, indent=2)
    logger.info("[Day 17] Latency profile written: logs/block_d/day17_latency_profile.json")
    return latency_dict


# ─────────────────────────────────────────────────────────────────────────────
# 7.11  update_config_blockD
# ─────────────────────────────────────────────────────────────────────────────

def update_config_blockD(cfg_path: str, far_gate_dict: dict) -> None:
    """Update config_blockD.yaml with Day 17 FAR validation results."""
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    all_passed = all(r.get("gate_passed", False) for r in far_gate_dict.values())
    cfg["day17_far_validated"]    = all_passed
    cfg["day17_far_gate_results"] = {
        m: {
            "mean_far":    r.get("mean_far"),
            "std_far":     r.get("std_far"),
            "gate_passed": r.get("gate_passed"),
        }
        for m, r in far_gate_dict.items()
    }
    cfg["day17_timestamp"] = datetime.now(timezone.utc).isoformat()

    if not all_passed:
        cfg["day17_far_validation_note"] = (
            "One or more methods exceeded FAR threshold. "
            "Results are reportable but flagged per Fix F14. See day17_report.json."
        )

    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False)

    logger.info("[Day 17] config_blockD.yaml updated (day17_far_validated=%s).", all_passed)


# ─────────────────────────────────────────────────────────────────────────────
# 7.12  produce_day17_report
# ─────────────────────────────────────────────────────────────────────────────

def produce_day17_report(
    normal_manifest_df: pd.DataFrame,
    far_gate_dict: dict,
    latency_dict: dict,
    complete_results_df: pd.DataFrame,
    far_threshold: float,
) -> dict:
    """
    Assemble and write the Day 17 master report JSON.
    day17_exit_criteria_met is True iff all four completion conditions are satisfied.
    FAR gate outcome does NOT affect day17_exit_criteria_met.
    """
    clips_discovered = int(len(normal_manifest_df))
    clips_readable   = int(normal_manifest_df["readable"].sum()) if "readable" in normal_manifest_df.columns else 0
    clips_skipped    = 0
    clips_evaluated  = 0

    if "status" in complete_results_df.columns:
        pass  # not on complete_results_df; check far_gate_dict
    for gate in far_gate_dict.values():
        clips_evaluated = max(clips_evaluated, gate.get("clips_evaluated", 0))

    methods_passed  = [m for m, g in far_gate_dict.items() if g.get("gate_passed", False)]
    methods_flagged = [m for m, g in far_gate_dict.items() if not g.get("gate_passed", False)]
    all_methods_present = set(far_gate_dict.keys()) == EXPECTED_METHODS

    # Exit criteria evaluation
    crit_clips_evaluated   = clips_evaluated >= MIN_NORMAL_CLIPS
    crit_complete_table    = all_methods_present and len(complete_results_df) == len(EXPECTED_METHODS)
    crit_latency_complete  = (
        isinstance(latency_dict, dict)
        and "stages" in latency_dict
        and "end_to_end_fps" in latency_dict
    )
    crit_config_updated    = os.path.exists("logs/block_d/day17_report.json") or True  # written next

    # Actually check config was updated (day17_far_validated key present)
    cfg_ok = False
    cfg_path_check = "configs/config_blockD.yaml"
    if os.path.exists(cfg_path_check):
        with open(cfg_path_check) as fh:
            _cfg_check = yaml.safe_load(fh)
        cfg_ok = "day17_far_validated" in _cfg_check

    day17_exit_criteria_met = (
        crit_clips_evaluated
        and crit_complete_table
        and crit_latency_complete
        and cfg_ok
    )

    report = {
        "day":               17,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "far_threshold_used": far_threshold,
        "metric_definitions": {
            "anomalous_ratio": (
                "Fraction of windows flagged on ANOMALY clips (Days 13–16). "
                "Not recomputed here."
            ),
            "FAR": (
                "Fraction of windows flagged on NORMAL clips (Day 17). "
                "Different clip population."
            ),
        },
        "normal_clips": {
            "clips_discovered":          clips_discovered,
            "clips_readable":            clips_readable,
            "clips_evaluated":           clips_evaluated,
            "clips_skipped":             max(0, clips_readable - clips_evaluated),
            "annotation_coverage_caveat": (
                "Not_in_events_json clips are assumed normal but not confirmed. "
                "See thesis §3.8."
            ),
            "manifest_written_to": "logs/block_d/day17_normal_clips_manifest.csv",
        },
        "far_gate": {
            m: {
                "mean_far":        g.get("mean_far"),
                "std_far":         g.get("std_far"),
                "max_far":         g.get("max_far"),
                "clips_evaluated": g.get("clips_evaluated"),
                "gate_passed":     g.get("gate_passed"),
                "gate_note":       g.get("gate_note"),
            }
            for m, g in far_gate_dict.items()
        },
        "all_methods_passed":       len(methods_flagged) == 0,
        "rule_based_design_note": (
            "rule_based uses clip-local z-score; FAR is clip-variance-dependent. "
            "See thesis §4."
        ),
        "latency_profiling": {
            "end_to_end_fps":                 latency_dict.get("end_to_end_fps"),
            "total_p50_ms":                   (latency_dict.get("stages", {}).get("total", {}).get("p50_ms")),
            "total_p95_ms":                   (latency_dict.get("stages", {}).get("total", {}).get("p95_ms")),
            "detection_source_for_profiling": latency_dict.get("detection_source_for_profiling"),
            "single_clip_caveat": (
                "Single-clip benchmark; may not generalise. "
                "Hardware-specific. See measurement_caveats."
            ),
            "report_written_to": "logs/block_d/day17_latency_profile.json",
        },
        "complete_results_table": {
            "written_to":                    "logs/block_d/block_d_complete_results.csv",
            "methods_with_far_gate_passed":  sorted(methods_passed),
            "methods_with_far_gate_flagged": sorted(methods_flagged),
        },
        "config_blockD_updated":   cfg_ok,
        "day17_exit_criteria_met": day17_exit_criteria_met,
        "exit_criteria_detail": {
            "clips_evaluated_ge_3":    crit_clips_evaluated,
            "complete_table_written":  crit_complete_table,
            "latency_profile_written": crit_latency_complete,
            "config_blockD_updated":   cfg_ok,
        },
    }

    os.makedirs("logs/block_d", exist_ok=True)
    with open("logs/block_d/day17_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info(
        "[Day 17] Report written: logs/block_d/day17_report.json  "
        "(exit_criteria_met=%s)", day17_exit_criteria_met,
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Internal: blocker partial report
# ─────────────────────────────────────────────────────────────────────────────

def _write_blocker_report(blocker_error: str, far_threshold: float) -> None:
    """
    Write a minimal day17_report.json when the normal-clip BLOCKER fires.
    Records the blocker state for auditability; day17_exit_criteria_met is False.
    """
    os.makedirs("logs/block_d", exist_ok=True)
    report = {
        "day":                      17,
        "timestamp":                datetime.now(timezone.utc).isoformat(),
        "far_threshold_used":       far_threshold,
        "metric_definitions": {
            "anomalous_ratio": (
                "Fraction of windows flagged on ANOMALY clips (Days 13–16). "
                "Not recomputed here."
            ),
            "FAR": (
                "Fraction of windows flagged on NORMAL clips (Day 17). "
                "Different clip population."
            ),
        },
        "status":                   "BLOCKER",
        "blocker_message":          blocker_error,
        "day17_exit_criteria_met":  False,
        "resolution": (
            "Add ≥3 readable normal AI City clips under data/ai_city/normal/ "
            "or list them in data/ai_city/normal_clips.txt and re-run run_day17()."
        ),
    }
    with open("logs/block_d/day17_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    logger.warning(
        "[Day 17] BLOCKER report written to logs/block_d/day17_report.json. "
        "day17_exit_criteria_met=False."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7.13  run_day17 (main entry point)
# ─────────────────────────────────────────────────────────────────────────────

def run_day17(cfg_path: str = "configs/config_blockD.yaml") -> dict:
    """
    Day 17 orchestrator.  Must be run after Day 16 is fully validated.

    Returns the day17 master report dict.
    Raises RuntimeError if the normal-clip BLOCKER fires (after writing a partial
    blocker report so that the run is auditable).
    """
    guard_day16_complete()

    cfg = yaml.safe_load(open(cfg_path))

    far_threshold = cfg.get("far_threshold")
    if far_threshold is None:
        raise KeyError(
            "far_threshold missing from config_blockD.yaml. "
            "Add:  far_threshold: 0.10"
        )

    models = load_models(cfg)

    # discover_normal_clips writes the manifest and raises RuntimeError([BLOCKER])
    # if fewer than MIN_NORMAL_CLIPS readable clips are found.
    # We catch it here to write an auditable partial report before re-raising.
    try:
        normal_manifest_df = discover_normal_clips(cfg)
    except RuntimeError as blocker:
        _write_blocker_report(str(blocker), far_threshold)
        raise  # re-raise so callers know the pipeline did not complete

    far_df          = compute_far_on_normal_clips(normal_manifest_df, cfg, models)
    far_gate_dict   = run_far_gate(far_df, far_threshold)
    complete_df     = build_complete_block_d_table(far_gate_dict)
    latency_dict    = run_latency_profiling(cfg, models)
    update_config_blockD(cfg_path, far_gate_dict)
    return produce_day17_report(
        normal_manifest_df, far_gate_dict, latency_dict, complete_df, far_threshold
    )
