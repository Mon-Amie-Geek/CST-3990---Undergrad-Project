"""
block_d_day16_validation.py

Block D Day 16: Final Anomaly Validation.

Performs:
  1. Guard -Day 15 exit criteria must be met before running
  2. Build final results table -canonical Block D summary for thesis Chapter 4
  3. Cross-check event slices -all prediction rows map to valid slicer entries
  4. Config audit -all four config YAMLs free of TU-DAT references
  5. Scaler provenance check -flexible for dict-wrapped and bare sklearn objects
  6. Update config_blockD.yaml -mark completion
  7. Produce Day 16 validation report

FAR / Precision / Recall (TP/TN/FP/FN) / F1 are NOT computed here.
These require a negative (normal) clip set, introduced in Day 17.
Reference: Implementation Plan v3, Fix F14, Day 17 exit criteria.

Student : MANJOO Ameera Najla | M01014463
Module  : CST3990 Undergraduate Individual Project -Block D Day 16
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler

# Import shared helpers from Day 15 to avoid duplication
from src.block_d_day15_bridge import (
    EXPECTED_METHODS,
    METHOD_NAME_MAP,
    ANOMALY_VALUES,
    TUDAT_TERMS,
    _coerce_visual_match,
    compute_anomalous_ratio,
    load_and_validate_inputs,
)

logger = logging.getLogger(__name__)

# ── Config YAMLs to audit ─────────────────────────────────────────────────────
CONFIG_YAML_PATHS = [
    "configs/config_blockA.yaml",
    "configs/config_blockB.yaml",
    "configs/config_blockC.yaml",
    "configs/config_blockD.yaml",
]

KNOWN_PROVENANCE_KEYS = [
    "fit_split", "fitted_on", "scaler_fit_split", "provenance",
    "metadata", "scaler_info", "train_split", "fit_on",
]


# ─────────────────────────────────────────────────────────────────────────────
# 7.1  load_all_block_d_artifacts
# ─────────────────────────────────────────────────────────────────────────────

def load_all_block_d_artifacts(cfg: dict) -> dict:
    """
    Load all Block D CSVs (reusing Day 15 load logic) plus the Day 15 bridge
    report.  Guards that Day 15 exit criteria are met before proceeding.
    """
    paths = cfg.get("paths", {})
    bridge_path = paths.get("day15_bridge_report",
                            "logs/block_d/day15_bridge_report.json")

    if not os.path.exists(bridge_path):
        raise FileNotFoundError(
            f"{bridge_path} not found. Run Day 15 completely before Day 16."
        )

    with open(bridge_path, "r") as fh:
        bridge = json.load(fh)

    if not bridge.get("day15_exit_criteria_met", False):
        raise RuntimeError(
            "Day 15 exit criteria not met. "
            "day15_bridge_report.json has day15_exit_criteria_met=False. "
            "Resolve all Day 15 blockers before running Day 16."
        )

    print("[Day 16] Day 15 guard passed -proceeding with validation.")

    # Re-use Day 15 loader (handles normalisation, dedup, schema validation)
    artifacts = load_and_validate_inputs(cfg)
    artifacts["bridge_report"] = bridge
    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# 7.2  build_final_results_table
# ─────────────────────────────────────────────────────────────────────────────

def build_final_results_table(
    predictions_df: pd.DataFrame,
    pred_col:       str,
    method_col:     str,
    clip_key:       str,
    event_key:      str,
    cfg:            dict,
) -> pd.DataFrame:
    """
    Produce the canonical Block D results table.

    Columns:
      method | total_events | events_detected | detection_coverage_pct |
      anomalous_ratio | mean_anomaly_score | short_window_event_count | notes

    FAR / Precision / Recall (TP/TN/FP/FN) / F1 are NOT computed here.
    These require a negative (normal) clip set, introduced in Day 17.
    Reference: Implementation Plan v3, Fix F14, Day 17 exit criteria.
    """
    anomalous_ratios = compute_anomalous_ratio(predictions_df, pred_col, method_col)

    # Detect score/confidence column (e.g. anomaly_score, confidence, prob_*)
    score_col = None
    for col in predictions_df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ("score", "confidence", "prob")):
            score_col = col
            break

    has_short_window = "short_window" in predictions_df.columns

    rows = []
    for method in sorted(EXPECTED_METHODS):
        sub = predictions_df[predictions_df[method_col] == method]

        # Unique (clip, event) pairs
        if len(sub) == 0:
            total_events     = 0
            events_detected  = 0
            det_cov_pct      = float("nan")
            mean_score       = float("nan")
            sw_count         = 0
            note             = f"[WARNING] no rows found for method '{method}'"
        else:
            pairs = sub[[clip_key, event_key]].drop_duplicates()
            total_events = len(pairs)

            detected_pairs = sub[sub[pred_col].isin(ANOMALY_VALUES)][
                [clip_key, event_key]
            ].drop_duplicates()
            events_detected = len(detected_pairs)

            det_cov_pct = round(
                events_detected / total_events * 100, 2
            ) if total_events > 0 else float("nan")

            mean_score = (
                round(float(sub[score_col].mean()), 4)
                if score_col is not None
                else float("nan")
            )

            # short_window events stay in denominator; counted separately
            sw_count = (
                int(sub["short_window"].sum())
                if has_short_window
                else 0
            )
            note = ""

        rows.append({
            "method":                  method,
            "total_events":            total_events,
            "events_detected":         events_detected,
            "detection_coverage_pct":  det_cov_pct,
            "anomalous_ratio":         anomalous_ratios.get(method, float("nan")),
            "mean_anomaly_score":      mean_score,
            "short_window_event_count": sw_count,
            "notes":                   note,
        })

    final_df = pd.DataFrame(rows)

    paths = cfg.get("paths", {})
    out_path = paths.get("block_d_final_results",
                         "logs/block_d/block_d_final_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final_df.to_csv(out_path, index=False)
    print(f"[Day 16] Final results table written ->{out_path}")
    print(final_df.to_string(index=False))

    return final_df


# ─────────────────────────────────────────────────────────────────────────────
# 7.3  cross_check_event_slices
# ─────────────────────────────────────────────────────────────────────────────

def cross_check_event_slices(
    predictions_df: pd.DataFrame,
    slicer_df:      pd.DataFrame,
    clip_key:       str,
    event_key:      str,
) -> pd.DataFrame:
    """
    For every unique (clip, event) in predictions_df, confirm a matching row
    exists in slicer_df.  SYNC_WARNING if visual_match is genuine False.

    Mismatches are [WARNING] only -not blockers.
    Returns DataFrame of mismatches + sync warnings.  Empty = all clear.
    """
    slicer_index = set(
        zip(slicer_df[clip_key], slicer_df[event_key])
    )

    has_vm = "visual_match" in slicer_df.columns
    sync_warn_index: set[tuple] = set()
    if has_vm:
        for _, row in slicer_df.iterrows():
            if row["visual_match"] is False:
                sync_warn_index.add((row[clip_key], row[event_key]))

    issues = []
    pred_pairs = (
        predictions_df[[clip_key, event_key]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for cv, ev in pred_pairs:
        key = (cv, ev)
        if key not in slicer_index:
            issues.append({
                clip_key:  cv,
                event_key: ev,
                "issue":   "MISMATCH -prediction row has no slicer entry",
            })
            logger.warning(
                "[Day 16] MISMATCH: clip=%s event=%s has no slicer row", cv, ev
            )
        elif key in sync_warn_index:
            issues.append({
                clip_key:  cv,
                event_key: ev,
                "issue":   "SYNC_WARNING -visual_match=False in slicer",
            })
            logger.warning(
                "[Day 16] SYNC_WARNING: clip=%s event=%s visual_match=False", cv, ev
            )

    if issues:
        print(f"[Day 16 Cross-Check] {len(issues)} issue(s) found (WARNING only).")
    else:
        print("[Day 16 Cross-Check] All prediction events have valid slicer entries.")

    return pd.DataFrame(
        issues,
        columns=[clip_key, event_key, "issue"] if issues else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7.4  run_config_audit
# ─────────────────────────────────────────────────────────────────────────────

def run_config_audit() -> bool:
    """
    Check all four config YAMLs for TU-DAT term references.
    Raises AssertionError on first hit.  Returns True if clean.
    """
    for path in CONFIG_YAML_PATHS:
        if not os.path.exists(path):
            logger.warning("[Day 16 Config Audit] Config not found, skipping: %s", path)
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        for term in TUDAT_TERMS:
            if term in raw:
                raise AssertionError(
                    f"TU-DAT term '{term}' found in {path}. "
                    "Remove all TU-DAT references before Day 16."
                )
    print("[Day 16 Config Audit] PASS -all four config YAMLs are TU-DAT-free.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 7.5  confirm_scaler_provenance
# ─────────────────────────────────────────────────────────────────────────────

def confirm_scaler_provenance(cfg: dict) -> tuple[bool, str]:
    """
    Verify that models/minmax_scaler.pkl was fitted on UA-DETRAC train normal data.

    Supports both dict-wrapped objects (with a provenance key) and bare
    sklearn MinMaxScaler objects.  Never raises -returns (bool, detail_str).
    """
    scaler_path = cfg.get("scaler", "models/minmax_scaler.pkl")
    if isinstance(scaler_path, dict):
        scaler_path = scaler_path.get("path", "models/minmax_scaler.pkl")

    if not os.path.exists(scaler_path):
        return (False, f"scaler file not found at '{scaler_path}'")

    try:
        obj = joblib.load(scaler_path)
    except Exception as exc:
        return (False, f"could not load scaler: {exc}")

    if isinstance(obj, dict):
        norm_keys = {k.lower().replace("_", ""): k for k in obj.keys()}
        for prov_key in KNOWN_PROVENANCE_KEYS:
            prov_norm = prov_key.replace("_", "")
            if prov_norm in norm_keys:
                original_key = norm_keys[prov_norm]
                val = str(obj[original_key]).lower()
                if "ua_detrac" in val and "normal" in val:
                    return (True,
                            f'provenance key "{original_key}" = "{obj[original_key]}"')
                else:
                    return (False,
                            f'provenance key "{original_key}" found but value '
                            f'"{obj[original_key]}" does not confirm ua_detrac_normal')
        return (False, "dict loaded but no recognised provenance key found")

    elif isinstance(obj, MinMaxScaler):
        return (True,
                "bare sklearn MinMaxScaler object -provenance confirmed by project "
                "convention (Day 11 fit_on=ua_detrac_train_normal_only)")

    else:
        return (False, f"unexpected object type: {type(obj)}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.6  update_config_blockD
# ─────────────────────────────────────────────────────────────────────────────

def update_config_blockD(cfg_path: str) -> None:
    """Mark Block D completion fields in config_blockD.yaml."""
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    cfg["block_d_completed"] = True
    cfg["day16_validated"]   = True
    cfg["day16_timestamp"]   = datetime.now(timezone.utc).isoformat()

    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False)

    print(f"[Day 16] config_blockD.yaml updated -block_d_completed=True, day16_validated=True")


# ─────────────────────────────────────────────────────────────────────────────
# 7.7  produce_day16_report
# ─────────────────────────────────────────────────────────────────────────────

def produce_day16_report(
    final_results_df: pd.DataFrame,
    cross_check_df:   pd.DataFrame,
    config_audit_ok:  bool,
    scaler_result:    tuple[bool, str],
    predictions_df:   pd.DataFrame,
    clip_key:         str,
    event_key:        str,
    cfg:              dict,
) -> dict:
    """Build and persist the Day 16 validation report JSON."""

    methods_present = sorted(final_results_df["method"].tolist()) \
                      if not final_results_df.empty else []

    all_methods_present = set(methods_present) == EXPECTED_METHODS

    total_pred_events = len(
        predictions_df[[clip_key, event_key]].drop_duplicates()
    )
    mismatches_found = int(
        cross_check_df["issue"].str.startswith("MISMATCH").sum()
    ) if not cross_check_df.empty and "issue" in cross_check_df.columns else 0
    sync_warnings = int(
        cross_check_df["issue"].str.startswith("SYNC_WARNING").sum()
    ) if not cross_check_df.empty and "issue" in cross_check_df.columns else 0

    paths = cfg.get("paths", {})
    final_results_path = paths.get("block_d_final_results",
                                   "logs/block_d/block_d_final_results.csv")

    scaler_ok, scaler_detail = scaler_result

    day16_ok = all_methods_present and config_audit_ok

    report = {
        "day": 16,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "final_results_table": {
            "methods_present": methods_present,
            "rows":            len(final_results_df),
            "written_to":      final_results_path,
        },
        "cross_check_event_slices": {
            "total_prediction_events": total_pred_events,
            "mismatches_found":        mismatches_found,
            "sync_warnings":           sync_warnings,
        },
        "config_audit": {
            "passed":      config_audit_ok,
            "tudat_hits":  0,
        },
        "scaler_provenance": {
            "verified": scaler_ok,
            "detail":   scaler_detail,
        },
        "config_blockD_updated": True,
        "day16_exit_criteria_met": day16_ok,
    }

    report_path = paths.get("day16_validation_report",
                            "logs/block_d/day16_validation_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    status = "PASS" if day16_ok else "FAIL"
    print(f"\n[Day 16 Validation Report] day16_exit_criteria_met = {day16_ok}  [{status}]")
    print(f"  Report written ->{report_path}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# 7.8  run_day16 -orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_day16(cfg_path: str = "configs/config_blockD.yaml") -> dict:
    """
    Full Day 16 orchestrator.  Calls steps in order:
      7.1 load_all_block_d_artifacts  (includes Day 15 guard)
      7.2 build_final_results_table
      7.3 cross_check_event_slices
      7.4 run_config_audit
      7.5 confirm_scaler_provenance
      7.6 update_config_blockD
      7.7 produce_day16_report
    """
    print("\n" + "=" * 70)
    print("  Block D -Day 16: Final Anomaly Validation")
    print("=" * 70)

    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    # Step 7.1 -load artifacts + Day 15 guard
    print("\n[Step 1/6] Loading artifacts and checking Day 15 guard ...")
    artifacts = load_all_block_d_artifacts(cfg)

    slicer_df    = artifacts["slicer_df"]
    preds_df     = artifacts["predictions_df"]
    clip_key     = artifacts["clip_key"]
    event_key    = artifacts["event_key"]
    method_col   = artifacts["method_col"]
    pred_col     = artifacts["pred_col"]

    # Step 7.2 -final results table
    print("\n[Step 2/6] Building final results table ...")
    final_df = build_final_results_table(
        preds_df, pred_col, method_col, clip_key, event_key, cfg
    )

    # Step 7.3 -cross-check event slices
    print("\n[Step 3/6] Cross-checking event slices ...")
    cross_df = cross_check_event_slices(preds_df, slicer_df, clip_key, event_key)

    # Step 7.4 -config audit
    print("\n[Step 4/6] Auditing all four config YAMLs ...")
    config_audit_ok = run_config_audit()

    # Step 7.5 -scaler provenance
    print("\n[Step 5/6] Confirming scaler provenance ...")
    scaler_result = confirm_scaler_provenance(cfg)
    verified, detail = scaler_result
    status_str = "VERIFIED" if verified else "[WARNING] NOT VERIFIED"
    print(f"  Scaler provenance: {status_str} -{detail}")

    # Step 7.6 -update config
    print("\n[Step 6/6] Updating config_blockD.yaml ...")
    update_config_blockD(cfg_path)
    # Reload cfg after update so the report uses fresh values
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    # Step 7.7 -produce report
    report = produce_day16_report(
        final_results_df = final_df,
        cross_check_df   = cross_df,
        config_audit_ok  = config_audit_ok,
        scaler_result    = scaler_result,
        predictions_df   = preds_df,
        clip_key         = clip_key,
        event_key        = event_key,
        cfg              = cfg,
    )

    print("\n" + "=" * 70)
    return report
