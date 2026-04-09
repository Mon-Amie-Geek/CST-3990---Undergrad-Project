"""
block_d_day15_bridge.py

Block D Day 15: Anomaly Bridge + Scope Clarification.

Performs:
  1. Schema validation of all Block D CSVs
  2. Event coverage audit -cross-validates slicer CSV vs predictions CSV
  3. Frame-index sync sanity check (informative / plausibility only)
  4. TU-DAT audit -active source and config files only; scoped blocker policy
  5. Metadata integrity check -no computed numeric deferred metrics
  6. Scope clarification paragraph (Fix F29) ->docs/scope_clarification.txt
  7. Bridge report ->logs/block_d/day15_bridge_report.json

Discovered schema (DO NOT change):
  event_slicer_verification.csv : event_id, video_id, video_file,
                                  anomaly_start_frame, computed_start_frame,
                                  visual_match, short_window, notes
  block_d_event_predictions.csv : method, event_id, video_id, video_file,
                                  event_pred, gt_label, total_valid_samples,
                                  anomalous_samples, anomalous_ratio,
                                  skipped, skip_reason
  block_d_results.csv           : Method, Event_Type, Events_Total,
                                  Events_Evaluable, Events_Detected,
                                  Event_Recall, Skipped_Events,
                                  Mean_Anomalous_Ratio, FAR, Precision,
                                  F1, Notes

Student : MANJOO Ameera Najla | M01014463
Module  : CST3990 Undergraduate Individual Project -Block D Day 15
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ── Canonical method set ──────────────────────────────────────────────────────
EXPECTED_METHODS = {"rule_based", "ocsvm", "isolation_forest"}

# Method name normalisation map (built from schema discovery).
# Two rule-based variants both map to the single canonical 'rule_based'.
# After deduplication on (clip_key, event_key, method) the first row
# (Rule-Based Z-score) is kept.
METHOD_NAME_MAP: dict[str, str] = {
    "Rule-Based (Z-score)": "rule_based",
    "Rule-Based (IQR)":     "rule_based",
    "OC-SVM":               "ocsvm",
    "Isolation Forest":     "isolation_forest",
}

# Prediction values that mean "anomaly" across all three models
ANOMALY_VALUES = {1, True, -1}  # rule_based / IF use 1; OC-SVM uses -1

# ── TU-DAT audit configuration ────────────────────────────────────────────────
TUDAT_SCAN_EXTENSIONS = [".py", ".yaml", ".yml", ".json"]
TUDAT_BLOCKER_DIRS    = ["src", "configs"]       # hits here = blocker
TUDAT_WARNING_DIRS    = ["logs"]                  # hits here = warning only
TUDAT_EXCLUDE_PATTERNS = [
    "__pycache__", ".git",
    "prompts", "notes", "archive", "old",
    "Day15", "Day16", "day15", "day16",           # this file is auto-excluded
]
# Constructed at runtime to avoid being flagged by the scanner itself.
_T1 = "TU" + "-DAT"
_T2 = "tu" + "dat"
_T3 = "TU" + "DAT"
_T4 = "tu_" + "dat"
TUDAT_TERMS: list[str] = [_T1, _T2, _T3, _T4]

# Deferred metric key fragments -any key whose normalised form CONTAINS one of
# these, AND whose value is a non-None numeric, is a metadata integrity violation.
DEFERRED_METRIC_KEYS = {"far", "false_alarm_rate", "precision", "f1", "f1_score"}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_visual_match(val) -> bool | None:
    """
    Coerce visual_match cell to bool or None per Section 3.5.

    True/False (already bool) ->pass through.
    'True'/'true' ->True; 'False'/'false' ->False.
    Anything else (e.g. 'UNCHECKED', NaN) ->None (UNCHECKED).
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, float):  # NaN comes in as float
        return None
    s = str(val).strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    # 'UNCHECKED' is the well-known project sentinel - treat silently as None.
    # Log a warning only for genuinely unexpected values.
    if s.upper() != "UNCHECKED":
        logger.warning("[visual_match] value '%s' could not be coerced ->UNCHECKED", val)
    return None


def _should_scan(filepath: str, repo_root: str) -> bool:
    rel = os.path.relpath(filepath, repo_root)
    for pat in TUDAT_EXCLUDE_PATTERNS:
        if pat in rel:
            return False
    _, ext = os.path.splitext(filepath)
    return ext in TUDAT_SCAN_EXTENSIONS


def _is_blocker(filepath: str, repo_root: str) -> bool:
    rel = os.path.relpath(filepath, repo_root)
    return any(rel.startswith(d + os.sep) or rel.startswith(d + "/")
               for d in TUDAT_BLOCKER_DIRS)


def _recursive_metadata_check(obj, path: str = "") -> None:
    """
    Recursively walk a dict/list.  Raise AssertionError if any key whose
    normalised form contains a deferred metric term has a non-None numeric value.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            k_norm = k.lower().replace("_", "").replace("-", "")
            for term in DEFERRED_METRIC_KEYS:
                if term in k_norm:
                    if isinstance(v, (int, float)) and v is not None:
                        raise AssertionError(
                            f"METADATA INTEGRITY FAIL at '{path}.{k}' = {v}. "
                            f"This is a computed numeric value for a deferred metric. "
                            f"FAR/Precision/F1 are deferred to Day 17. "
                            f"Replace with null or a string label before proceeding."
                        )
            _recursive_metadata_check(v, path=f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _recursive_metadata_check(item, path=f"{path}[{i}]")


# ─────────────────────────────────────────────────────────────────────────────
# 5.2  load_and_validate_inputs
# ─────────────────────────────────────────────────────────────────────────────

def load_and_validate_inputs(cfg: dict) -> dict:
    """
    Load and validate all four Block D input artifacts.

    Returns a dict containing DataFrames, the metadata dict, and the canonical
    column-name keys discovered during schema inspection.
    """
    paths = cfg.get("paths", {})

    slicer_path  = paths.get("event_slicer_verification",
                             "logs/block_d/event_slicer_verification.csv")
    preds_path   = paths.get("block_d_event_predictions",
                             "logs/block_d/block_d_event_predictions.csv")
    results_path = paths.get("block_d_results",
                             "logs/block_d/block_d_results.csv")
    meta_path    = paths.get("block_d_metadata",
                             "logs/block_d/block_d_metadata.json")

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    for label, p in [("event_slicer_verification", slicer_path),
                     ("block_d_event_predictions",  preds_path),
                     ("block_d_results",            results_path)]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required artifact '{label}' not found at: {p}\n"
                "Run Days 13-14 before Day 15."
            )

    slicer_df  = pd.read_csv(slicer_path)
    preds_df   = pd.read_csv(preds_path)
    results_df = pd.read_csv(results_path)

    with open(meta_path, "r") as fh:
        metadata = json.load(fh)

    # ── Canonical key discovery (Section 3) ───────────────────────────────────
    # Clip key -priority: video_id > video_file > clip_id > first column
    for cand in ("video_id", "video_file", "clip_id"):
        if cand in slicer_df.columns and cand in preds_df.columns:
            clip_key = cand
            break
    else:
        clip_key = slicer_df.columns[0]
        logger.warning("[WARNING: using fallback clip key] '%s'", clip_key)

    # Event key -priority: event_id > any column containing ev_id/anomaly_id/event
    event_key = None
    for cand in ("event_id",):
        if cand in slicer_df.columns and cand in preds_df.columns:
            event_key = cand
            break
    if event_key is None:
        for col in slicer_df.columns:
            if any(t in col.lower() for t in ("ev_id", "anomaly_id", "event")):
                if col in preds_df.columns:
                    event_key = col
                    break
    if event_key is None:
        raise ValueError(
            "Cannot discover event-level key. "
            "Expected 'event_id' or a column containing 'ev_id'/'anomaly_id'/'event'."
        )
    logger.info("[Day 15] clip_key='%s'  event_key='%s'", clip_key, event_key)

    # Method column
    if "method" in preds_df.columns:
        method_col = "method"
    else:
        raise ValueError("Method column 'method' not found in block_d_event_predictions.csv")

    # Prediction column -find a column with values in {0, 1, -1, True, False}
    pred_col = None
    for col in preds_df.columns:
        unique_vals = set(preds_df[col].dropna().unique())
        if unique_vals.issubset({0, 1, -1, True, False}):
            pred_col = col
            break
    if pred_col is None:
        raise ValueError(
            "Cannot discover prediction column. "
            "Expected a column with values in {0, 1, -1, True, False}."
        )

    # ── Method normalisation ──────────────────────────────────────────────────
    preds_df[method_col] = preds_df[method_col].map(
        lambda m: METHOD_NAME_MAP.get(m, m)
    )
    missing_methods = EXPECTED_METHODS - set(preds_df[method_col].unique())
    if missing_methods:
        raise ValueError(
            f"After method normalisation, the following canonical methods are absent "
            f"from block_d_event_predictions.csv: {missing_methods}. "
            f"Check METHOD_NAME_MAP and the actual method column values."
        )

    # ── visual_match coercion ─────────────────────────────────────────────────
    if "visual_match" in slicer_df.columns:
        slicer_df["visual_match"] = slicer_df["visual_match"].apply(_coerce_visual_match)

    # ── Deduplication ─────────────────────────────────────────────────────────
    before = len(preds_df)
    preds_df = preds_df.drop_duplicates(
        subset=[clip_key, event_key, method_col], keep="first"
    ).reset_index(drop=True)
    dup_removed = before - len(preds_df)
    if dup_removed > 0:
        logger.info("[Day 15] Deduplicated %d rows from predictions (kept first per (clip, event, method))",
                    dup_removed)

    # ── Schema validation ─────────────────────────────────────────────────────
    # slicer_df must have: clip_key, event_key, computed_start_frame
    for required in [clip_key, event_key, "computed_start_frame"]:
        if required not in slicer_df.columns:
            raise ValueError(
                f"event_slicer_verification.csv is missing required column: '{required}'"
            )

    # preds_df must have: clip_key, event_key, method_col, pred_col
    for required in [clip_key, event_key, method_col, pred_col]:
        if required not in preds_df.columns:
            raise ValueError(
                f"block_d_event_predictions.csv is missing required column: '{required}'"
            )

    # results_df must have: a method column + at least one numeric result column
    results_method_col = None
    for cand in results_df.columns:
        if cand.lower() == "method":
            results_method_col = cand
            break
    if results_method_col is None:
        raise ValueError(
            "block_d_results.csv is missing a method column "
            "(expected a column whose lowercase form equals 'method')."
        )
    numeric_cols = results_df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        raise ValueError(
            "block_d_results.csv has no numeric result columns."
        )

    return {
        "slicer_df":             slicer_df,
        "predictions_df":        preds_df,
        "results_df":            results_df,
        "metadata":              metadata,
        "clip_key":              clip_key,
        "event_key":             event_key,
        "method_col":            method_col,
        "pred_col":              pred_col,
        "duplicate_rows_removed": dup_removed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.3  run_event_coverage_audit
# ─────────────────────────────────────────────────────────────────────────────

def run_event_coverage_audit(
    slicer_df:      pd.DataFrame,
    predictions_df: pd.DataFrame,
    cfg:            dict,
    clip_key:       str,
    event_key:      str,
    method_col:     str,
    dup_removed:    int,
) -> pd.DataFrame:
    """
    For every (clip, event) pair in slicer_df × EXPECTED_METHODS, check whether
    a row exists in predictions_df.

    Status values:
      OK                   -row present
      MISSING_IN_PREDICTIONS -no matching row
      SYNC_WARNING         -row present but visual_match is genuine False
    """
    has_visual_match = "visual_match" in slicer_df.columns

    # Build lookup index: {(clip_val, event_val, method) ->True}
    pred_index = set(
        zip(predictions_df[clip_key],
            predictions_df[event_key],
            predictions_df[method_col])
    )

    # Build visual_match lookup for SYNC_WARNING: only genuine False triggers it
    sync_warn_index: set[tuple] = set()
    if has_visual_match:
        for _, row in slicer_df.iterrows():
            if row["visual_match"] is False:   # genuine False, not None/UNCHECKED
                sync_warn_index.add((row[clip_key], row[event_key]))

    # Iterate over UNIQUE (clip, event) pairs; each event may appear multiple times
    # in the slicer CSV (one row per tracker/context window) but coverage
    # is checked at the (clip, event, method) level against predictions.
    unique_pairs = (
        slicer_df[[clip_key, event_key]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    rows = []
    for _, srow in unique_pairs.iterrows():
        cv = srow[clip_key]
        ev = srow[event_key]
        for method in sorted(EXPECTED_METHODS):
            key = (cv, ev, method)
            if key in pred_index:
                if has_visual_match and (cv, ev) in sync_warn_index:
                    status = "SYNC_WARNING"
                    notes  = "visual_match=False for this (clip, event)"
                else:
                    status = "OK"
                    notes  = ""
            else:
                status = "MISSING_IN_PREDICTIONS"
                notes  = f"No prediction row found for method='{method}'"

            rows.append({
                clip_key:    cv,
                event_key:   ev,
                "method":    method,
                "coverage_status": status,
                "notes":     notes,
            })

    coverage_df = pd.DataFrame(rows)

    paths = cfg.get("paths", {})
    out_path = paths.get("day15_event_coverage_audit",
                         "logs/block_d/day15_event_coverage_audit.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    coverage_df.to_csv(out_path, index=False)
    logger.info("[Day 15] Event coverage audit written ->%s", out_path)

    ok_n       = (coverage_df["coverage_status"] == "OK").sum()
    missing_n  = (coverage_df["coverage_status"] == "MISSING_IN_PREDICTIONS").sum()
    sync_warn_n = (coverage_df["coverage_status"] == "SYNC_WARNING").sum()

    print(f"[Day 15 Coverage Audit] total={len(coverage_df)}  "
          f"OK={ok_n}  MISSING={missing_n}  SYNC_WARNING={sync_warn_n}  "
          f"dup_rows_removed={dup_removed}")
    if missing_n > 0:
        print(f"[BLOCKER] {missing_n} event-method pair(s) have no prediction row.")

    return coverage_df


# ─────────────────────────────────────────────────────────────────────────────
# 5.4  run_frame_sync_sanity_check
# ─────────────────────────────────────────────────────────────────────────────

def run_frame_sync_sanity_check(
    slicer_df: pd.DataFrame,
    cfg:       dict,
    clip_key:  str,
) -> pd.DataFrame:
    """
    Plausibility-only sanity check using OpenCV CAP_PROP_POS_FRAMES.

    This is NOT a hard frame-truth assertion.  OpenCV's backend-dependent
    position reporting means IMPLAUSIBLE is a WARNING only, never a blocker.
    """
    # Resolve video directory from cfg
    video_dir = (
        cfg.get("ai_city_video_dir")
        or cfg.get("video_dir")
    )
    if not video_dir:
        # No path resolution strategy found
        rows = []
        for clip_val in slicer_df[clip_key].unique():
            rows.append({
                clip_key:       clip_val,
                "target_frame": None,
                "pos_after_read": None,
                "fps":          None,
                "plausible":    None,
                "sync_status":  "VIDEO_PATH_UNRESOLVABLE",
            })
        df = pd.DataFrame(rows)
        _write_sync_csv(df, cfg)
        logger.warning("[Day 15] Frame sync: no video_dir in cfg -all clips marked VIDEO_PATH_UNRESOLVABLE")
        return df

    rows = []
    for clip_val in slicer_df[clip_key].unique():
        clip_rows = slicer_df[slicer_df[clip_key] == clip_val]

        # Resolve video filename -prefer video_file column if available
        if "video_file" in slicer_df.columns:
            video_file = clip_rows.iloc[0]["video_file"]
        else:
            video_file = f"{clip_val}.mp4"

        video_path = os.path.join(str(video_dir), str(video_file))

        if not os.path.exists(video_path):
            rows.append({
                clip_key:       clip_val,
                "target_frame": None,
                "pos_after_read": None,
                "fps":          None,
                "plausible":    None,
                "sync_status":  "VIDEO_NOT_FOUND",
            })
            continue

        # Use the row with the smallest computed_start_frame for this clip
        target_frame = int(clip_rows["computed_start_frame"].min())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, _ = cap.read()
        pos_after_read = cap.get(cv2.CAP_PROP_POS_FRAMES)   # position AFTER read
        cap.release()

        if ret:
            plausible = abs(pos_after_read - (target_frame + 1)) <= 2
        else:
            plausible = False

        sync_status = "PLAUSIBLE" if plausible else "IMPLAUSIBLE"
        if not plausible:
            logger.warning(
                "[Day 15] Frame sync WARNING clip=%s target=%d pos_after=%s ->IMPLAUSIBLE",
                clip_val, target_frame, pos_after_read
            )

        rows.append({
            clip_key:         clip_val,
            "target_frame":   target_frame,
            "pos_after_read": pos_after_read,
            "fps":            fps,
            "plausible":      plausible,
            "sync_status":    sync_status,
        })

    df = pd.DataFrame(rows)
    _write_sync_csv(df, cfg)
    return df


def _write_sync_csv(df: pd.DataFrame, cfg: dict) -> None:
    paths = cfg.get("paths", {})
    out_path = paths.get("day15_frame_sync_sanity_check",
                         "logs/block_d/day15_frame_sync_sanity_check.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("[Day 15] Frame sync sanity check written ->%s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5.5  run_tudat_audit
# ─────────────────────────────────────────────────────────────────────────────

def run_tudat_audit(repo_root: str, cfg: dict) -> pd.DataFrame:
    """
    Scan active source and config files for TU-DAT term occurrences.

    Blocker policy: hits in src/ or configs/ are blockers.
    Warning policy: hits in logs/ are warnings only.
    """
    records: list[dict] = []
    files_scanned = 0
    blocker_hits  = 0
    warning_hits  = 0

    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Prune excluded directories in-place
        dirnames[:] = [
            d for d in dirnames
            if not any(pat in d for pat in TUDAT_EXCLUDE_PATTERNS)
        ]
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if not _should_scan(fpath, repo_root):
                continue
            files_scanned += 1
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        for term in TUDAT_TERMS:
                            if term in line:
                                is_blk = _is_blocker(fpath, repo_root)
                                rel_path = os.path.relpath(fpath, repo_root)
                                records.append({
                                    "file_path":    rel_path,
                                    "line_number":  lineno,
                                    "matched_term": term,
                                    "line_content": line.strip(),
                                    "is_blocker":   is_blk,
                                })
                                if is_blk:
                                    blocker_hits += 1
                                    print(f"[BLOCKER] {rel_path}:{lineno} -'{term}' found")
                                else:
                                    warning_hits += 1
                                    logger.warning(
                                        "[Day 15 TU-DAT] WARNING %s:%d -'%s'",
                                        rel_path, lineno, term
                                    )
                                break  # one hit per line is enough
            except Exception as exc:
                logger.warning("[Day 15 TU-DAT] Could not read %s: %s", fpath, exc)

    audit_df = pd.DataFrame(
        records,
        columns=["file_path", "line_number", "matched_term", "line_content", "is_blocker"]
        if records else None,
    )
    audit_df.attrs = {
        "files_scanned": files_scanned,
        "blocker_hits":  blocker_hits,
        "warning_hits":  warning_hits,
    }

    paths = cfg.get("paths", {})
    out_path = paths.get("day15_tudat_audit",
                         "logs/block_d/day15_tudat_audit.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    audit_df.to_csv(out_path, index=False)

    print(f"[Day 15 TU-DAT Audit] files_scanned={files_scanned}  "
          f"blocker_hits={blocker_hits}  warning_hits={warning_hits}")
    if blocker_hits == 0:
        print("[Day 15 TU-DAT Audit] PASS -no blocker hits.")

    return audit_df


# ─────────────────────────────────────────────────────────────────────────────
# 5.6  run_metadata_integrity_check
# ─────────────────────────────────────────────────────────────────────────────

def run_metadata_integrity_check(metadata: dict) -> bool:
    """
    Walk the metadata dict recursively.  Raise AssertionError if any key
    whose normalised form contains a deferred-metric term holds a non-None
    numeric value.  Returns True if no violation found.
    """
    try:
        _recursive_metadata_check(metadata)
        print("[Day 15 Metadata Integrity] PASS")
        return True
    except AssertionError as exc:
        print(f"[Day 15 Metadata Integrity] FAIL -{exc}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 5.7  compute_anomalous_ratio
# ─────────────────────────────────────────────────────────────────────────────

def compute_anomalous_ratio(
    predictions_df: pd.DataFrame,
    pred_col:       str,
    method_col:     str,
) -> dict[str, float]:
    """
    Returns {method_name: fraction_of_events_flagged} for each canonical method.

    Uses pred_col values; ANOMALY_VALUES = {1, True, -1}.
    """
    result: dict[str, float] = {}
    for method in EXPECTED_METHODS:
        sub = predictions_df[predictions_df[method_col] == method]
        if len(sub) == 0:
            result[method] = float("nan")
        else:
            flagged = sub[pred_col].isin(ANOMALY_VALUES).sum()
            result[method] = round(flagged / len(sub), 4)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5.8  write_scope_clarification
# ─────────────────────────────────────────────────────────────────────────────

_SCOPE_TEXT = (
    "SCOPE CLARIFICATION \u2014 Fix F29\n"
    "CST3990 | MANJOO Ameera Najla | M01014463\n"
    "\n"
    "While the pipeline architecture is explicitly designed for real-time operation \u2014\n"
    "with modular components, lightweight model variants (YOLOv8n, SORT), and\n"
    "streaming-compatible data flow \u2014 local hardware constraints of the Microsoft\n"
    "Surface Pro 7 necessitate asynchronous (offline batch) processing for this\n"
    "validation phase. The real-time claim pertains to the architectural capability\n"
    "of the pipeline design, not to the throughput achieved on the development\n"
    "hardware. Deployment on a dedicated edge device (e.g., NVIDIA Jetson Nano)\n"
    "is identified as a direct extension of this work.\n"
    "\n"
    "Block D evaluation scope note:\n"
    "Days 13\u201316 use AI City Track 4 anomaly events only (positive-only benchmark).\n"
    "All loaded events are confirmed positive anomalies defined by\n"
    "anomaly_start_frame / anomaly_end_frame annotations from AI City Track 4.\n"
    "False Alarm Rate (FAR), Precision, Recall (in the TP/TN/FP/FN sense), and\n"
    "full F1 computation are deferred to Day 17, which introduces normal AI City\n"
    "clips as the negative reference set.\n"
    "\n"
    "Block D Days 13\u201316 report:\n"
    "  \u2014 event-level detection coverage (events_detected / total_events)\n"
    "  \u2014 per-method anomalous_ratio (fraction of trajectory windows flagged)\n"
    "The term \"detection coverage\" is used in preference to \"recall\" because\n"
    "the traditional recall definition requires a confirmed negative class,\n"
    "which is not available until Day 17.\n"
)


def write_scope_clarification(cfg: dict) -> bool:
    """Write the Fix F29 scope clarification text.  Returns True on success."""
    paths = cfg.get("paths", {})
    output_path = paths.get("scope_clarification", "docs/scope_clarification.txt")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(_SCOPE_TEXT)
    print(f"[Day 15] Scope clarification written ->{output_path}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 5.9  produce_bridge_report
# ─────────────────────────────────────────────────────────────────────────────

def produce_bridge_report(
    coverage_df:    pd.DataFrame,
    sync_df:        pd.DataFrame,
    tudat_df:       pd.DataFrame,
    metadata_ok:    bool,
    clip_key:       str,
    event_key:      str,
    scope_written:  bool,
    dup_removed:    int,
    cfg:            dict,
) -> dict:
    """Build and persist the Day 15 bridge report JSON."""

    # Coverage counts
    total_pairs  = len(coverage_df)
    ok_count     = int((coverage_df["coverage_status"] == "OK").sum())
    missing_count = int((coverage_df["coverage_status"] == "MISSING_IN_PREDICTIONS").sum())
    sync_warn_count = int((coverage_df["coverage_status"] == "SYNC_WARNING").sum())

    # Frame sync counts
    clips_checked    = len(sync_df)
    plausible_count  = int((sync_df["sync_status"] == "PLAUSIBLE").sum()) \
                       if "sync_status" in sync_df.columns else 0
    implausible_count = int((sync_df["sync_status"] == "IMPLAUSIBLE").sum()) \
                        if "sync_status" in sync_df.columns else 0
    not_found_count  = int(sync_df["sync_status"].isin(
                           ["VIDEO_NOT_FOUND", "VIDEO_PATH_UNRESOLVABLE"]
                       ).sum()) if "sync_status" in sync_df.columns else clips_checked

    # TU-DAT counts (from attrs, fall back to counting the DataFrame)
    attrs           = tudat_df.attrs
    files_scanned   = attrs.get("files_scanned", 0)
    blocker_hits    = attrs.get("blocker_hits",  int(tudat_df["is_blocker"].sum())
                               if not tudat_df.empty and "is_blocker" in tudat_df.columns else 0)
    warning_hits    = attrs.get("warning_hits",
                               int((~tudat_df["is_blocker"]).sum())
                               if not tudat_df.empty and "is_blocker" in tudat_df.columns else 0)

    coverage_blocker  = missing_count > 0
    tudat_blocker     = blocker_hits > 0
    metadata_blocker  = not metadata_ok

    day15_ok = (
        not coverage_blocker
        and not tudat_blocker
        and metadata_ok
        and scope_written
    )

    report = {
        "day": 15,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "canonical_keys_used": {
            "clip_key":                   clip_key,
            "event_key":                  event_key,
            "method_normalization_applied": True,
        },
        "event_coverage_audit": {
            "total_event_method_pairs_checked": total_pairs,
            "ok_count":                         ok_count,
            "missing_in_predictions_count":     missing_count,
            "sync_warning_count":               sync_warn_count,
            "duplicate_rows_removed":           dup_removed,
            "blocker":                          coverage_blocker,
        },
        "frame_sync_sanity_check": {
            "clips_checked":      clips_checked,
            "plausible_count":    plausible_count,
            "implausible_count":  implausible_count,
            "video_not_found_count": not_found_count,
            "note": "Informative sanity check only -not a hard frame-truth assertion.",
            "blocker": False,
        },
        "tudat_audit": {
            "files_scanned":  files_scanned,
            "blocker_hits":   blocker_hits,
            "warning_hits":   warning_hits,
            "blocker":        tudat_blocker,
        },
        "metadata_integrity": {
            "passed":  metadata_ok,
            "blocker": metadata_blocker,
        },
        "scope_clarification_written": scope_written,
        "day15_exit_criteria_met":     day15_ok,
    }

    paths = cfg.get("paths", {})
    out_path = paths.get("day15_bridge_report",
                         "logs/block_d/day15_bridge_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    status = "PASS" if day15_ok else "FAIL"
    print(f"\n[Day 15 Bridge Report] day15_exit_criteria_met = {day15_ok}  [{status}]")
    print(f"  Report written ->{out_path}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# 5.10  run_day15 -orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_day15(cfg_path: str = "configs/config_blockD.yaml") -> dict:
    """
    Full Day 15 orchestrator.  Calls steps in order:
      5.2 load_and_validate_inputs
      5.3 run_event_coverage_audit
      5.4 run_frame_sync_sanity_check
      5.5 run_tudat_audit
      5.6 run_metadata_integrity_check
      5.7 compute_anomalous_ratio
      5.8 write_scope_clarification
      5.9 produce_bridge_report
    """
    print("\n" + "=" * 70)
    print("  Block D -Day 15: Anomaly Bridge + Scope Clarification")
    print("=" * 70)

    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    repo_root = str(Path(cfg_path).resolve().parent.parent)

    # Step 5.2 -load and validate
    print("\n[Step 1/7] Loading and validating inputs ...")
    artifacts = load_and_validate_inputs(cfg)
    slicer_df    = artifacts["slicer_df"]
    preds_df     = artifacts["predictions_df"]
    metadata     = artifacts["metadata"]
    clip_key     = artifacts["clip_key"]
    event_key    = artifacts["event_key"]
    method_col   = artifacts["method_col"]
    pred_col     = artifacts["pred_col"]
    dup_removed  = artifacts["duplicate_rows_removed"]

    # Step 5.3 -event coverage audit
    print("\n[Step 2/7] Running event coverage audit ...")
    coverage_df = run_event_coverage_audit(
        slicer_df, preds_df, cfg, clip_key, event_key, method_col, dup_removed
    )

    # Step 5.4 -frame sync sanity check
    print("\n[Step 3/7] Running frame-index sync sanity check (plausibility only) ...")
    sync_df = run_frame_sync_sanity_check(slicer_df, cfg, clip_key)

    # Step 5.5 -TU-DAT audit
    print("\n[Step 4/7] Running TU-DAT audit ...")
    tudat_df = run_tudat_audit(repo_root, cfg)

    # Step 5.6 -metadata integrity
    print("\n[Step 5/7] Running metadata integrity check ...")
    metadata_ok = run_metadata_integrity_check(metadata)

    # Step 5.7 -anomalous ratio
    print("\n[Step 6/7] Computing anomalous ratio per method ...")
    anomalous_ratios = compute_anomalous_ratio(preds_df, pred_col, method_col)
    for m, r in anomalous_ratios.items():
        print(f"  {m}: {r}")

    # Step 5.8 -scope clarification
    print("\n[Step 7/7] Writing scope clarification ...")
    scope_written = write_scope_clarification(cfg)

    # Step 5.9 -bridge report
    report = produce_bridge_report(
        coverage_df  = coverage_df,
        sync_df      = sync_df,
        tudat_df     = tudat_df,
        metadata_ok  = metadata_ok,
        clip_key     = clip_key,
        event_key    = event_key,
        scope_written= scope_written,
        dup_removed  = dup_removed,
        cfg          = cfg,
    )

    print("\n" + "=" * 70)
    return report
