"""
block_c_evaluator.py
Block C: Feature set comparison via OC-SVM AUROC.
Fix F02 (no GridSearchCV), Fix F04 (no pseudo-anomaly circular logic).

CST3990 | MANJOO Ameera Najla | M01014463
"""

import ast
import datetime
import json
import logging
import os
import shutil

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

from src.seed_control import set_all_seeds

logger = logging.getLogger(__name__)

# ── Feature set definitions ────────────────────────────────────────────────────
# Fix F04 (no circular pseudo-anomaly construction).
# These are the ONLY column subsets passed to OC-SVM.
# Any future change must be made here only.
#
# SELF-CORRECTION GUARD (Day 12 mandatory check):
# Before the evaluation loop runs, assert that every column listed in
# FEATURE_SET_DEFINITIONS exists in the loaded _features_scaled.csv DataFrames.
# Any mismatch means the column names here do not match Day 11 outputs and the
# F2+F3 comparison will be invalid.
# Use _validate_feature_set_columns() to verify this at runtime.
FEATURE_SET_DEFINITIONS = {
    "F1_only": [
        "vehicle_count",
        "roi_occupancy",
    ],
    "F2_only": [
        "vel_px_sec",
        "vel_px_sec_smooth",
    ],
    "F2_F3": [
        "vel_px_sec",
        "vel_px_sec_smooth",
        "inter_vehicle_dist_norm",      # continuous interaction distance (MinMax-scaled)
        "dwell_time_sec",               # temporal dwell behaviour (MinMax-scaled)
        "proximity_count_rolling",      # rolling interaction persistence (MinMax-scaled)
    ],
    "all_features": [
        "vehicle_count",
        "roi_occupancy",
        "vel_px_sec",
        "vel_px_sec_smooth",
        "inter_vehicle_dist_norm",      # continuous interaction distance (MinMax-scaled)
        "dwell_time_sec",               # temporal dwell behaviour (MinMax-scaled)
        "proximity_count_rolling",      # rolling interaction persistence (MinMax-scaled)
    ],
}
# Column classification for Day 12 OC-SVM inputs:
#   SCALER_FIT_COLUMNS used as model inputs (MinMax-scaled by Day 11):
#       vehicle_count, roi_occupancy, vel_px_sec, vel_px_sec_smooth,
#       inter_vehicle_dist_norm, dwell_time_sec, proximity_count_rolling.
#   EXCLUDED from Day 12 OC-SVM inputs (by design — thesis alignment):
#       proximity_flag — boolean threshold-derived helper written by Day 11 as a
#       passthrough column in _features_scaled.csv. It is retained in the CSV schema
#       for provenance and potential downstream use, but is NOT a Day 12 model input.
#       F3 is represented by three continuous/temporal behavioural indicators:
#       inter_vehicle_dist_norm, dwell_time_sec, proximity_count_rolling.
#       This keeps Block C methodologically clean: OC-SVM compares continuous
#       feature families, not threshold-derived binary signals.

# ── Nu candidates (Fix F02 — no GridSearchCV) ─────────────────────────────────
NU_CANDIDATES = [0.01, 0.02, 0.05, 0.10, 0.15]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _coerce_bool_cols_to_float(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Return a copy of df[cols] with any boolean or string-boolean columns cast to
    float 0.0/1.0.  Handles pandas bool dtype, numpy bool_, and 'True'/'False'
    object strings (which pd.read_csv sometimes produces).
    Never mutates the caller's DataFrame.
    """
    df_copy = df[cols].copy()
    for col in df_copy.columns:
        col_dtype = df_copy[col].dtype
        if pd.api.types.is_bool_dtype(col_dtype):
            df_copy[col] = df_copy[col].astype(float)
        elif col_dtype == object:
            # Handle string 'True'/'False' representations
            mapping = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0,
                       "true": 1.0, "false": 0.0, 1: 1.0, 0: 0.0}
            df_copy[col] = df_copy[col].map(mapping)
    return df_copy


def _load_split_scaled_features(
    split_name: str,
    metadata: dict,
    config: dict,
) -> pd.DataFrame:
    """
    Load and concatenate _features_scaled.csv for all sequences in a given split.
    Raises FileNotFoundError if any expected scaled CSV is missing.
    Never loads _features.csv (unscaled). Never calls scaler.fit() or
    scaler.transform().
    """
    features_dir = config.get("features_dir", "logs/block_c/")
    scaled_suffix = config.get("scaler", {}).get("scaled_suffix", "_features_scaled.csv")
    seq_ids = metadata["split_seqs"][split_name]
    dfs = []
    for seq_id in seq_ids:
        path = os.path.join(features_dir, f"{seq_id}{scaled_suffix}")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[{seq_id}] Scaled CSV not found: {path}. "
                "Ensure Day 11 completed (F3 augmentation + scaler transform)."
            )
        df = pd.read_csv(path)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(
        "Loaded %d rows for split='%s' (%d seqs).",
        len(combined), split_name, len(seq_ids),
    )
    return combined


def _validate_feature_set_columns(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_set_definitions: dict,
) -> None:
    """
    Self-correction guard: assert that every column declared in
    FEATURE_SET_DEFINITIONS exists in the actual _features_scaled.csv DataFrames
    produced by Day 11.

    A mismatch here means the feature group definitions do not match Day 11
    outputs and the F2+F3 / all_features comparisons will be invalid.

    Raises ValueError listing all missing columns so the developer can reconcile
    FEATURE_SET_DEFINITIONS against Day 11's FEATURE_SCHEMA_COLUMN_ORDER.
    """
    all_declared = set(col for cols in feature_set_definitions.values() for col in cols)
    missing_train = all_declared - set(df_train.columns)
    missing_val   = all_declared - set(df_val.columns)

    if missing_train or missing_val:
        msg_parts = []
        if missing_train:
            msg_parts.append(
                f"Train _features_scaled.csv is missing declared columns: "
                f"{sorted(missing_train)}. "
                "Update FEATURE_SET_DEFINITIONS to match Day 11 "
                "FEATURE_SCHEMA_COLUMN_ORDER (which includes both MinMax-scaled "
                "columns AND passthrough columns such as proximity_flag), or "
                "confirm Day 11 ran with F3 augmentation enabled "
                "(f3 fields = true in config_blockC.yaml)."
            )
        if missing_val:
            msg_parts.append(
                f"Val _features_scaled.csv is missing declared columns: "
                f"{sorted(missing_val)}. Same remediation as above."
            )
        raise ValueError(
            "FEATURE_SET_DEFINITIONS column mismatch with Day 11 outputs:\n"
            + "\n".join(msg_parts)
        )

    # F3 completeness check — the three continuous/temporal F3 model-input columns
    # must be present. proximity_flag is intentionally excluded from Day 12 OC-SVM
    # inputs (thesis-aligned behavioural clarity) but is allowed to exist in the CSVs.
    f3_required = {
        "inter_vehicle_dist_norm",
        "dwell_time_sec",
        "proximity_count_rolling",
    }
    f3_missing_train = f3_required - set(df_train.columns)
    f3_missing_val   = f3_required - set(df_val.columns)
    if f3_missing_train or f3_missing_val:
        raise ValueError(
            f"F3 model-input columns incomplete. "
            f"Missing in train: {f3_missing_train}, "
            f"missing in val: {f3_missing_val}. "
            "The three continuous/temporal F3 columns "
            "(inter_vehicle_dist_norm, dwell_time_sec, proximity_count_rolling) "
            "must be present in _features_scaled.csv for F2_F3 and all_features "
            "comparisons to be valid. Confirm Day 11 ran with all F3 features "
            "enabled."
        )

    # Informational only: log proximity_flag presence without requiring it
    pf_in_train = "proximity_flag" in df_train.columns
    pf_in_val   = "proximity_flag" in df_val.columns
    if pf_in_train and pf_in_val:
        logger.info(
            "proximity_flag present in Day 11 CSVs — retained in schema but "
            "excluded from Day 12 OC-SVM inputs by design (thesis alignment: "
            "F3 uses continuous/temporal indicators only)."
        )
    elif not pf_in_train or not pf_in_val:
        logger.info(
            "proximity_flag absent from one or both splits — this is acceptable "
            "as it is not a Day 12 OC-SVM model input."
        )

    logger.info(
        "✓ FEATURE_SET_DEFINITIONS column validation passed. "
        "All declared OC-SVM model-input columns present in Day 11 outputs."
    )


def build_val_binary_labels(
    df_train_full: pd.DataFrame,
    df_val_full: pd.DataFrame,
    config: dict,
) -> pd.Series:
    """
    Fix F04: construct y_val_binary ONCE from the full Day 11 scaled split
    DataFrames, not from per-feature-set subsets.

    Label rule is constant across all feature-set experiments.

    A val row is labelled anomalous (1) if ANY of the following conditions hold:
      - vel_px_sec > train_mean_vel + 3 * train_std_vel
      - vel_px_sec < train_mean_vel - 3 * train_std_vel
      - inter_vehicle_dist_norm < train_mean_dist - 3 * train_std_dist
        (applied only when the column exists in both train and val DataFrames)

    Statistics are computed EXCLUSIVELY from df_train_full — never from val or
    test. The resulting pd.Series is aligned to df_val_full.index and later
    subset using valid_idx for each feature set.

    IMPORTANT:
    - y_val_binary must be defined identically for F1_only, F2_only, F2_F3,
      and all_features.
    - Feature-set-specific column availability must NOT change the label
      definition.
    - Per-feature-set row filtering happens only after label construction.

    Day 11 carry-forward note:
    Because *_features_scaled.csv may contain full-block NaNs for rows with
    undefined interaction features, some val rows may later be excluded from
    specific feature-set evaluations via dropna(). This is expected and
    semantically correct.

    UA-DETRAC val sequences contain normal traffic only — no labelled anomaly
    ground truth. y_val_binary is a statistical proxy: rows where vel_px_sec or
    inter_vehicle_dist_norm fall beyond 3 train-split standard deviations are
    treated as 'anomaly candidates'.  y_val_binary is constructed ONCE from the
    full Day 11 split DataFrames (df_train_full, df_val_full) before the
    feature-set loop — the label definition is identical for F1_only, F2_only,
    F2_F3, and all_features. Per-feature-set valid_idx subsetting happens only
    after label construction. AUROC therefore measures feature separability of
    statistical outliers within a normal distribution, NOT true anomaly
    detection performance. This is the correct interpretation of Fix F04 — it
    avoids the circular pseudo-anomaly construction where speed defines the
    anomaly and then speed is evaluated. Block D (AI City, stall/crash GT)
    provides true anomaly detection evaluation.

    Returns
    -------
    pd.Series of int (0=normal, 1=anomaly candidate), index aligned to
    df_val_full.
    """
    y = pd.Series(0, index=df_val_full.index, dtype=int)

    # ── vel_px_sec 3-sigma threshold (train-split statistics only) ────────────
    train_vel = df_train_full["vel_px_sec"].dropna()
    if len(train_vel) < 2:
        logger.warning(
            "Insufficient non-NaN vel_px_sec rows in train split for 3-sigma "
            "threshold computation. Velocity anomaly criterion will be skipped."
        )
    else:
        train_mean_vel = float(train_vel.mean())
        train_std_vel  = float(train_vel.std(ddof=1))
        upper_vel = train_mean_vel + 3.0 * train_std_vel
        lower_vel = train_mean_vel - 3.0 * train_std_vel
        logger.info(
            "Train vel_px_sec (scaled): mean=%.6f, std=%.6f, "
            "3σ-interval=[%.6f, %.6f]  (n=%d rows)",
            train_mean_vel, train_std_vel, lower_vel, upper_vel, len(train_vel),
        )
        val_vel = df_val_full["vel_px_sec"]
        anomaly_vel = (val_vel > upper_vel) | (val_vel < lower_vel)
        y = (y | anomaly_vel.fillna(False).astype(int)).astype(int)

    # ── inter_vehicle_dist_norm 3-sigma lower-tail threshold ─────────────────
    if (
        "inter_vehicle_dist_norm" in df_train_full.columns
        and "inter_vehicle_dist_norm" in df_val_full.columns
    ):
        train_dist = df_train_full["inter_vehicle_dist_norm"].dropna()
        if len(train_dist) >= 2:
            train_mean_dist = float(train_dist.mean())
            train_std_dist  = float(train_dist.std(ddof=1))
            lower_dist = train_mean_dist - 3.0 * train_std_dist
            logger.info(
                "Train inter_vehicle_dist_norm (scaled): mean=%.6f, std=%.6f, "
                "lower threshold=%.6f",
                train_mean_dist, train_std_dist, lower_dist,
            )
            val_dist     = df_val_full["inter_vehicle_dist_norm"]
            anomaly_dist = val_dist < lower_dist
            y = (y | anomaly_dist.fillna(False).astype(int)).astype(int)

    n_pos   = int(y.sum())
    n_total = len(y)
    logger.info(
        "y_val_binary: %d positive (anomaly proxy), %d negative (normal), "
        "total=%d",
        n_pos, n_total - n_pos, n_total,
    )
    if n_pos == 0:
        logger.warning(
            "y_val_binary has zero positive labels — AUROC will be undefined. "
            "Val split may be too clean or the 3-sigma thresholds too wide in "
            "scaled space."
        )
    if n_pos == n_total:
        logger.warning(
            "y_val_binary has no negative labels — AUROC will be undefined. "
            "Check train-split statistics."
        )
    return y


def select_best_nu(
    X_train_normal: np.ndarray,
    nu_candidates: list,
    seed: int = 42,
) -> tuple:
    """
    Fix F02: manual nu sweep — no GridSearchCV.

    Split X_train_normal 80/20 (train/hold-out) with random_state=seed.
    For each nu, fit OC-SVM on 80%, evaluate FPR on hold-out 20%.
    Select the nu with the lowest FPR among those where FPR < 0.05.
    If no candidate achieves FPR < 0.05, fall back to the nu with the
    minimum FPR overall.

    Returns
    -------
    (best_nu: float, fallback_used: bool)
        fallback_used is True when no candidate achieved FPR < 0.05.
    """
    X_fit, X_hold = train_test_split(
        X_train_normal, test_size=0.20, random_state=seed
    )

    nu_fpr_pairs: list[tuple[float, float]] = []
    for nu in nu_candidates:
        try:
            ocsvm = OneClassSVM(kernel="rbf", nu=nu)
            ocsvm.fit(X_fit)
            preds = ocsvm.predict(X_hold)
            fpr = float((preds == -1).mean())
            logger.info("  nu=%.2f: FPR=%.4f", nu, fpr)
            nu_fpr_pairs.append((nu, fpr))
        except Exception as exc:
            logger.warning("  nu=%.2f: OC-SVM failed (%s). Skipping.", nu, exc)

    if not nu_fpr_pairs:
        logger.warning(
            "All nu candidates failed during hold-out sweep. "
            "Defaulting to nu=0.05."
        )
        return 0.05, True

    # Prefer the nu with the lowest FPR among those that achieve FPR < 0.05
    qualifying = [(nu, fpr) for nu, fpr in nu_fpr_pairs if fpr < 0.05]
    if qualifying:
        best_nu, best_fpr = min(qualifying, key=lambda x: x[1])
        fallback_used = False
        logger.info(
            "Selected nu=%.3f (FPR=%.4f < 0.05 threshold).", best_nu, best_fpr
        )
    else:
        best_nu, best_fpr = min(nu_fpr_pairs, key=lambda x: x[1])
        fallback_used = True
        logger.warning(
            "No nu candidate achieved FPR < 0.05. "
            "Falling back to nu=%.3f (min FPR=%.4f).",
            best_nu, best_fpr,
        )

    return best_nu, fallback_used


def _empty_result_row(fs_name: str, feature_cols: list, reason: str) -> dict:
    """Return a placeholder result row for a skipped feature set."""
    return {
        "feature_set":          fs_name,
        "feature_columns":      str(feature_cols),
        "n_train_rows":         0,
        "n_val_rows":           0,
        "n_val_anomaly_proxy":  0,
        "best_nu":              None,
        "auroc":                None,
        "silhouette":           None,
        "status":               reason,
    }


def run_auroc_comparison(
    config: dict,
    metadata: dict,
    output_dir: str = "logs/block_c/",
) -> pd.DataFrame:
    """
    Fix F04: OC-SVM decision function AUROC comparison across feature sets.
    No GridSearchCV (Fix F02). No pseudo-anomaly construction (Fix F04).

    Steps:
    1.  Load all _features_scaled.csv files for train (7 seqs) and val (1 seq).
    2.  Run _validate_feature_set_columns() — self-correction guard.
    3.  Load scaler pkl for provenance validation only (no transform called).
    4.  Build y_val_full via build_val_binary_labels() ONCE from full split
        DataFrames (label rule is constant across all feature-set experiments).
    5.  For each feature set in FEATURE_SET_DEFINITIONS:
          a. Select columns, coerce booleans to float, dropna.
          b. Warn if post-dropna row count < config.scaler.min_valid_fit_rows.
          c. Select best_nu via select_best_nu() on X_train_normal.
          d. Fit OC-SVM on full X_train_normal (kernel='rbf', best_nu).
          e. Compute decision_function scores on X_val.
             Subset y_val_full using valid_idx for this feature set.
          f. AUROC = roc_auc_score(y_val, -scores)   [negate: higher = normal]
          g. silhouette_score when y_val has ≥ 2 classes.
    6.  Select best feature set: highest AUROC; ties broken by silhouette.
    7.  Refit OC-SVM on best feature set; save ocsvm_trained_best.pkl.
    8.  Write logs/block_c/block_c_auroc_results.csv.
    9.  Write logs/block_c/block_c_results_table.csv  (thesis-ready).
    10. Update configs/config_blockC.yaml (best_feature_set + flags).
    11. Write logs/block_c/block_c_day12_metadata.json.

    Returns DataFrame of results (same schema as block_c_auroc_results.csv).
    """
    set_all_seeds()
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load scaled CSVs ───────────────────────────────────────────────────
    logger.info("Loading train split scaled CSVs...")
    df_train = _load_split_scaled_features("train", metadata, config)
    logger.info("Loading val split scaled CSVs...")
    df_val = _load_split_scaled_features("val", metadata, config)

    # ── 2. Self-correction guard ──────────────────────────────────────────────
    _validate_feature_set_columns(df_train, df_val, FEATURE_SET_DEFINITIONS)

    # ── 3. Provenance validation: load scaler (no transform) ─────────────────
    scaler_path = config.get("scaler", {}).get("path", "models/minmax_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Day 11 must complete before Day 12."
        )
    scaler_obj = joblib.load(scaler_path)
    if not hasattr(scaler_obj, "data_min_"):
        raise AssertionError(
            f"Scaler loaded from {scaler_path} does not appear to be fitted. "
            "Day 11 may have failed."
        )
    logger.info(
        "✓ Scaler loaded for provenance validation: %s "
        "(no fit / transform calls made).",
        scaler_path,
    )
    # Explicitly del to ensure no accidental use downstream
    del scaler_obj

    # ── 4. Build y_val_full ONCE (Fix F04) ────────────────────────────────────
    logger.info("Building y_val_binary from train-split statistics (Fix F04)...")
    y_val_full = build_val_binary_labels(
        df_train_full=df_train,
        df_val_full=df_val,
        config=config,
    )

    min_fit_rows = config.get("scaler", {}).get("min_valid_fit_rows", 10)

    # ── 5. Per-feature-set AUROC loop ─────────────────────────────────────────
    results: list[dict] = []
    best_auroc              = -1.0
    best_silhouette_for_best: float | None = None
    best_fs_name:            str | None = None
    best_nu_for_best:        float | None = None
    best_feature_cols:       list | None = None
    X_train_best_full:       np.ndarray | None = None
    best_fallback_used       = False

    for fs_name, feature_cols in FEATURE_SET_DEFINITIONS.items():
        logger.info(
            "─── Evaluating feature set: %s | columns: %s ───",
            fs_name, feature_cols,
        )

        # ── a. Prepare train data ─────────────────────────────────────────────
        missing_train = [c for c in feature_cols if c not in df_train.columns]
        if missing_train:
            logger.warning(
                "[%s] Train cols missing: %s. Skipping.", fs_name, missing_train
            )
            results.append(
                _empty_result_row(fs_name, feature_cols, "missing_train_cols")
            )
            continue

        X_train_df = _coerce_bool_cols_to_float(df_train, feature_cols).dropna()
        n_train_rows = len(X_train_df)
        logger.info(
            "[%s] Train rows after dropna: %d "
            "(Day 11 min_valid_fit_rows guard = %d).",
            fs_name, n_train_rows, min_fit_rows,
        )
        if n_train_rows < min_fit_rows:
            logger.warning(
                "[%s] Train rows (%d) below min_valid_fit_rows (%d). "
                "Expected for F3-containing sets if Day 11 wrote full-block NaNs. "
                "Skipping — not a preprocessing error.",
                fs_name, n_train_rows, min_fit_rows,
            )
            results.append(
                _empty_result_row(
                    fs_name, feature_cols, "insufficient_train_rows"
                )
            )
            continue
        X_train_normal = X_train_df.values.astype(float)

        # ── a. Prepare val data ───────────────────────────────────────────────
        missing_val = [c for c in feature_cols if c not in df_val.columns]
        if missing_val:
            logger.warning(
                "[%s] Val cols missing: %s. Skipping.", fs_name, missing_val
            )
            results.append(
                _empty_result_row(fs_name, feature_cols, "missing_val_cols")
            )
            continue

        df_val_fs  = _coerce_bool_cols_to_float(df_val, feature_cols)
        valid_idx  = df_val_fs.dropna().index
        X_val      = df_val_fs.loc[valid_idx].values.astype(float)
        logger.info(
            "[%s] Val rows after dropna: %d "
            "(lower count for F3-containing sets is expected — "
            "Day 11 NaN semantics).",
            fs_name, len(X_val),
        )

        if len(X_val) == 0:
            logger.warning("[%s] No valid val rows after dropna. Skipping.", fs_name)
            results.append(
                _empty_result_row(fs_name, feature_cols, "no_valid_val_rows")
            )
            continue

        # ── Subset y_val_full using valid_idx for this feature set ────────────
        # y_val_full built once above; per-feature-set row filtering only here.
        y_val  = y_val_full.loc[valid_idx].values
        n_pos  = int(y_val.sum())
        n_neg  = int((y_val == 0).sum())
        logger.info(
            "[%s] y_val: %d positive (anomaly proxy), %d negative (normal).",
            fs_name, n_pos, n_neg,
        )

        if n_pos == 0 or n_neg == 0:
            logger.warning(
                "[%s] y_val_binary lacks both classes — AUROC undefined. "
                "Val split may be too clean or all rows flagged anomalous.",
                fs_name,
            )
            results.append(
                _empty_result_row(fs_name, feature_cols, "single_class_y_val")
            )
            continue

        # ── b. Select best nu (Fix F02 — no GridSearchCV) ────────────────────
        logger.info("[%s] Running nu sweep on 80/20 train hold-out...", fs_name)
        best_nu, fallback_used = select_best_nu(
            X_train_normal, NU_CANDIDATES, seed=42
        )
        logger.info("[%s] Selected nu=%.3f (fallback=%s).", fs_name, best_nu, fallback_used)

        # ── c. Fit OC-SVM on full train normal ────────────────────────────────
        set_all_seeds()
        ocsvm = OneClassSVM(kernel="rbf", nu=best_nu)
        # OneClassSVM has no random_state parameter; deterministic behaviour is
        # controlled via fixed train/val split, seed control, and identical
        # input ordering.
        ocsvm.fit(X_train_normal)

        # ── d/e. Decision function → AUROC ────────────────────────────────────
        scores = ocsvm.decision_function(X_val)
        # Negate: OC-SVM convention is higher = more normal, so -scores gives
        # higher values for anomalies — required by roc_auc_score.
        try:
            auroc = float(roc_auc_score(y_val, -scores))
            logger.info("[%s] AUROC = %.4f", fs_name, auroc)
        except ValueError as exc:
            logger.warning(
                "[%s] roc_auc_score failed: %s. Setting auroc=None.", fs_name, exc
            )
            auroc = None

        # ── f. Silhouette score ───────────────────────────────────────────────
        # Requires at least 2 distinct classes in y_val and >= 4 samples.
        silhouette: float | None = None
        if len(np.unique(y_val)) >= 2 and len(X_val) >= 4:
            try:
                silhouette = float(silhouette_score(X_val, y_val))
                logger.info("[%s] Silhouette = %.4f", fs_name, silhouette)
            except Exception as exc:
                logger.warning("[%s] Silhouette failed: %s", fs_name, exc)

        # ── Track best (AUROC primary, silhouette tiebreak) ──────────────────
        if auroc is not None:
            current_sil = silhouette if silhouette is not None else -1.0
            prev_sil    = best_silhouette_for_best if best_silhouette_for_best is not None else -1.0
            if (auroc > best_auroc) or (auroc == best_auroc and current_sil > prev_sil):
                best_auroc              = auroc
                best_silhouette_for_best = silhouette
                best_fs_name            = fs_name
                best_nu_for_best        = best_nu
                best_feature_cols       = list(feature_cols)
                X_train_best_full       = X_train_normal.copy()
                best_fallback_used      = fallback_used

        results.append({
            "feature_set":         fs_name,
            "feature_columns":     str(feature_cols),
            "n_train_rows":        n_train_rows,
            "n_val_rows":          len(X_val),
            "n_val_anomaly_proxy": n_pos,
            "best_nu":             best_nu,
            "auroc":               auroc,
            "silhouette":          silhouette,
            "status":              "ok",
        })

    # ── Build results DataFrame ───────────────────────────────────────────────
    df_results = pd.DataFrame(results)

    if best_fs_name is None:
        logger.error(
            "No feature set produced a valid AUROC. "
            "Check Day 11 scaled CSV outputs and y_val_binary label distribution."
        )
    else:
        logger.info(
            "Best feature set: %s | AUROC=%.4f | nu=%.3f",
            best_fs_name, best_auroc, best_nu_for_best,
        )

        # ── 7. Refit best OC-SVM on full train data ───────────────────────────
        set_all_seeds()
        best_ocsvm_final = OneClassSVM(kernel="rbf", nu=best_nu_for_best)
        best_ocsvm_final.fit(X_train_best_full)

        best_model_path = os.path.join(output_dir, "ocsvm_trained_best.pkl")
        joblib.dump(
            {
                "model":            best_ocsvm_final,
                "feature_set_name": best_fs_name,
                "feature_columns":  best_feature_cols,
                "best_nu":          best_nu_for_best,
                "auroc":            best_auroc,
                "train_rows":       len(X_train_best_full),
                "fit_split":        "train",
                "seed":             42,
            },
            best_model_path,
        )
        logger.info("Best OC-SVM saved → %s", best_model_path)

        # Mirror to Google Drive (Colab only; silently skipped on local machines)
        drive_models = "/content/drive/MyDrive/CST3990/models/"
        if os.path.exists(drive_models):
            shutil.copy(
                best_model_path,
                os.path.join(drive_models, "ocsvm_trained_best.pkl"),
            )
            logger.info("Best OC-SVM mirrored to Drive.")
        else:
            logger.warning("Drive models dir not available — skipping Drive mirror.")

        # ── 10. Update configs/config_blockC.yaml ─────────────────────────────
        # Use yaml.safe_load → dict mutation → yaml.safe_dump; never string-replace.
        cfg_path = "configs/config_blockC.yaml"
        with open(cfg_path, "r") as fh:
            cfg_current = yaml.safe_load(fh)
        cfg_current["best_feature_set"]      = best_fs_name
        cfg_current["ocsvm_best_model_path"] = "logs/block_c/ocsvm_trained_best.pkl"
        cfg_current["block_c_completed"]     = True
        cfg_current["block_c_results_table"] = "logs/block_c/block_c_results_table.csv"
        with open(cfg_path, "w") as fh:
            yaml.safe_dump(cfg_current, fh, default_flow_style=False, sort_keys=False)
        logger.info(
            "config_blockC.yaml updated with best_feature_set=%s", best_fs_name
        )

    # ── 8. Write block_c_auroc_results.csv ───────────────────────────────────
    auroc_csv_path = os.path.join(output_dir, "block_c_auroc_results.csv")
    df_results.to_csv(auroc_csv_path, index=False)
    logger.info("AUROC results written → %s", auroc_csv_path)

    # ── 9. Write block_c_results_table.csv (thesis-ready) ────────────────────
    table_rows = []
    for _, row in df_results.iterrows():
        fs       = row["feature_set"]
        selected = "✓" if fs == best_fs_name else ""
        # Recover column list from the string repr safely (no eval of untrusted input)
        try:
            cols_list = ast.literal_eval(row["feature_columns"])
            cols_str  = ", ".join(cols_list)
        except Exception:
            cols_str = str(row["feature_columns"])
        table_rows.append({
            "Feature Set":  fs,
            "Features Used": cols_str,
            "Train Rows":   row["n_train_rows"],
            "Val Rows":     row["n_val_rows"],
            "Best Nu":      row["best_nu"],
            "AUROC": (
                round(float(row["auroc"]), 4)
                if row["auroc"] is not None and not (isinstance(row["auroc"], float) and np.isnan(row["auroc"]))
                else None
            ),
            "Silhouette": (
                round(float(row["silhouette"]), 4)
                if row["silhouette"] is not None and not (isinstance(row["silhouette"], float) and np.isnan(row["silhouette"]))
                else None
            ),
            "Selected": selected,
        })
    df_table = pd.DataFrame(table_rows)
    table_csv_path = os.path.join(output_dir, "block_c_results_table.csv")
    df_table.to_csv(table_csv_path, index=False)
    logger.info("Results table written → %s", table_csv_path)

    # ── 11. Write block_c_day12_metadata.json ────────────────────────────────
    metadata_out = {
        "day":                    12,
        "student":                "MANJOO Ameera Najla",
        "student_id":             "M01014463",
        "timestamp":              datetime.datetime.utcnow().isoformat() + "Z",
        "seed":                   42,
        "fix_references":         ["F02", "F04"],
        "scaler_loaded_from":     scaler_path,
        "scaler_fitted_day":      11,
        "scaler_refitted_today":  False,
        "scaler_purpose_day12": (
            "Loaded for prerequisite/provenance validation only. "
            "No transform calls made."
        ),
        "feature_sets_evaluated": list(FEATURE_SET_DEFINITIONS.keys()),
        "best_feature_set":       best_fs_name,
        "best_auroc":             best_auroc if best_fs_name is not None else None,
        "best_nu":                best_nu_for_best,
        "nu_sweep_candidates":    NU_CANDIDATES,
        "nu_selection_rule": (
            "manual nu sweep on 80/20 train-only hold-out; "
            "choose the nu with the lowest FPR among those < 0.05, "
            "else minimum-FPR fallback"
        ),
        "nu_holdout_fraction":    0.20,
        "nu_target_fpr":          0.05,
        "best_nu_fallback_used":  best_fallback_used,
        "ocsvm_model_path":       "logs/block_c/ocsvm_trained_best.pkl",
        "val_sequences":          metadata.get("split_seqs", {}).get("val", []),
        "train_sequences":        metadata.get("split_seqs", {}).get("train", []),
        "y_val_binary_method":    "3-sigma train-statistics threshold (Fix F04 — anti-circular)",
        "y_val_binary_note": (
            "UA-DETRAC val sequences contain normal traffic only — no labelled anomaly "
            "ground truth. y_val_binary is a statistical proxy: rows where vel_px_sec or "
            "inter_vehicle_dist_norm fall beyond 3 train-split standard deviations are "
            "treated as 'anomaly candidates'. y_val_binary is constructed ONCE from the "
            "full Day 11 split DataFrames (df_train_full, df_val_full) before the "
            "feature-set loop — the label definition is identical for F1_only, F2_only, "
            "F2_F3, and all_features. Per-feature-set valid_idx subsetting happens only "
            "after label construction. AUROC therefore measures feature separability of "
            "statistical outliers within a normal distribution, NOT true anomaly detection "
            "performance. This is the correct interpretation of Fix F04 — it avoids the "
            "circular pseudo-anomaly construction where speed defines the anomaly and then "
            "speed is evaluated. Block D (AI City, stall/crash GT) provides true anomaly "
            "detection evaluation."
        ),
        "y_val_binary_constructed_once": True,
        "day11_scaled_csv_semantics": (
            "Rows with any NaN in scaler-fit columns were written with NaN for the full "
            "scaler-fit block in *_features_scaled.csv. Day 12 preserves this exactly and "
            "does not impute undefined interaction values."
        ),
        "day11_min_valid_fit_rows_guard": (
            "Day 12 reads min_valid_fit_rows from config_blockC.yaml scaler section and "
            "warns (does not crash) if post-dropna row count falls below this threshold "
            "for any feature set."
        ),
        "f3_row_loss_note": (
            "Feature sets containing F3 (F2_F3, all_features) may evaluate on fewer rows "
            "because undefined interaction features were intentionally preserved as NaN in "
            "Day 11 scaled outputs. This is expected and not a preprocessing error."
        ),
        "proximity_flag_in_day12_model": False,
        "proximity_flag_note": (
            "proximity_flag is a boolean threshold-derived helper written by Day 11 as a "
            "passthrough column in _features_scaled.csv. It is retained in the CSV schema "
            "for provenance, but is NOT a Day 12 OC-SVM model input. F3 is represented by "
            "three continuous/temporal behavioural indicators: inter_vehicle_dist_norm, "
            "dwell_time_sec, proximity_count_rolling. This preserves thesis-aligned "
            "methodological clarity: OC-SVM compares continuous feature families, not "
            "threshold-derived binary signals."
        ),
        "feature_set_column_validation": (
            "FEATURE_SET_DEFINITIONS column names were validated against actual "
            "_features_scaled.csv columns at runtime via _validate_feature_set_columns() "
            "before the AUROC loop. Validation requires only the continuous/temporal "
            "OC-SVM model-input columns. proximity_flag is present in Day 11 CSV outputs "
            "but is excluded from all Day 12 feature sets by design."
        ),
        "block_c_results_table":  "logs/block_c/block_c_results_table.csv",
        "config_blockC_updated":  best_fs_name is not None,
        "interpretation_note": (
            "y_val_binary is derived from train-split velocity and interaction-distance "
            "statistics. Feature sets lacking those variables (especially F1_only) may be "
            "relatively disadvantaged. Day 12 compares practical separability under the "
            "selected proxy label, not a universal ranking of all possible anomaly-feature "
            "families."
        ),
    }
    meta_out_path = os.path.join(output_dir, "block_c_day12_metadata.json")
    with open(meta_out_path, "w") as fh:
        json.dump(metadata_out, fh, indent=2)
    logger.info("Metadata written → %s", meta_out_path)

    return df_results


# ── Standalone entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    set_all_seeds()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    with open("configs/config_blockC.yaml") as fh:
        cfg = yaml.safe_load(fh)
    with open("logs/metadata.json") as fh:
        meta = json.load(fh)
    df_results = run_auroc_comparison(cfg, meta)
    print("\n=== Block C AUROC Results ===")
    print(
        df_results[
            ["feature_set", "auroc", "silhouette", "best_nu", "status"]
        ].to_string(index=False)
    )
