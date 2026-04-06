"""
feature_extractor.py
Block C — F1 + F2 + F3 Feature Extraction, Generator Loader, and feature schema enforcement.
Day 10: Generator-based clip loader, F1 (Density/Flow), F2 (Speed/Velocity).
Day 11: F3 (Interaction) added — inter_vehicle_dist_norm, dwell_time_sec,
        proximity_flag, proximity_count_rolling. MinMaxScaler fitting (Fix F13).

Student: MANJOO Ameera Najla | M01014463
Module : CST3990 Undergraduate Individual Project
"""

import copy
import json
import logging
import os

import numpy as np
import pandas as pd
import yaml

from src.seed_control import set_all_seeds

# Seeds must be set at module import — before any numpy/torch operation
set_all_seeds()

logger = logging.getLogger(__name__)

# ── Schema column order — single source of truth ──────────────────────────────
# Any change to this list must bump feature_schema.json schema_version and be
# noted in thesis §3.4.3.
# F3 columns added Day 11 (inter_vehicle_dist_norm, dwell_time_sec,
# proximity_flag, proximity_count_rolling) appear after track_length_real
# and before pipeline metadata columns (split, tracker).
FEATURE_SCHEMA_COLUMN_ORDER = [
    # ── Identifiers ───────────────────────────────────────────────────────────
    "seq_id", "frame_idx", "track_id",
    # ── Spatial ───────────────────────────────────────────────────────────────
    "cx", "cy",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    # ── Detection metadata ────────────────────────────────────────────────────
    "conf", "is_interpolated", "velocity_norm",
    # ── F2: Speed/Velocity ────────────────────────────────────────────────────
    "vel_px_sec", "vel_px_sec_smooth",
    # ── F1: Density/Flow ──────────────────────────────────────────────────────
    "vehicle_count", "roi_occupancy",
    # ── Track metadata ────────────────────────────────────────────────────────
    "track_length_real",
    # ── F3: Interaction ── ADDED DAY 11 ──────────────────────────────────────
    "inter_vehicle_dist_norm", "dwell_time_sec",
    "proximity_flag", "proximity_count_rolling",
    # ── Pipeline metadata ─────────────────────────────────────────────────────
    "split", "tracker",
]

# ── F3 constants ──────────────────────────────────────────────────────────────
FRAME_DIAG = np.sqrt(640.0**2 + 640.0**2)
# Normalisation constant for inter_vehicle_dist_norm (Fix F28).
# Value: 640 * sqrt(2) ≈ 905.097. Computed once at module level — do NOT
# recompute per frame.

# ── Scaler fit columns ────────────────────────────────────────────────────────
SCALER_FIT_COLUMNS = [
    "vel_px_sec", "vel_px_sec_smooth",
    "vehicle_count", "roi_occupancy",
    "inter_vehicle_dist_norm", "dwell_time_sec",
    "proximity_count_rolling",
    # proximity_flag is binary (bool) — excluded from MinMaxScaler fit.
    # Identifiers, bbox, conf, velocity_norm, track_length_real, split,
    # tracker also excluded.
]


# ── Layout detection ──────────────────────────────────────────────────────────

def detect_day7_layout(base_dir: str, tracker: str, test_seq: str = "MVI_20062") -> str:
    """
    Detect which trajectory file layout Day 7 used.

    Variant A: logs/block_b/{tracker}/trajectories_{seq}_final.json
    Variant B: logs/block_b/trajectories/{seq}_{tracker}_final.json

    Returns: "variant_a" or "variant_b"
    Raises : FileNotFoundError if neither layout has a _final.json for test_seq
    """
    candidate_a = os.path.join(base_dir, tracker, f"trajectories_{test_seq}_final.json")
    candidate_b = os.path.join(base_dir, "trajectories", f"{test_seq}_{tracker}_final.json")

    if os.path.exists(candidate_a):
        return "variant_a"
    elif os.path.exists(candidate_b):
        return "variant_b"
    else:
        raise FileNotFoundError(
            f"No _final.json found for tracker='{tracker}', seq='{test_seq}'. "
            f"Checked:\n  {candidate_a}\n  {candidate_b}\n"
            "Ensure Day 9 post-processing is complete before running Block C."
        )


def resolve_trajectory_path(base_dir: str, tracker: str, seq_id: str, layout: str) -> str:
    """Construct _final.json path based on detected layout."""
    if layout == "variant_a":
        return os.path.join(base_dir, tracker, f"trajectories_{seq_id}_final.json")
    elif layout == "variant_b":
        return os.path.join(base_dir, "trajectories", f"{seq_id}_{tracker}_final.json")
    else:
        raise ValueError(f"Unknown layout: {layout}")


# ── Trajectory metadata validation ───────────────────────────────────────────

def validate_trajectory_file_metadata(traj_data: dict, seq_id: str, tracker: str) -> None:
    """
    Validate _final.json top-level metadata fields.
    Raises ValueError if Day 9 post-processing was not applied.

    CRITICAL — field locations:
    All validation fields (fragment_filter_applied, interpolation_applied, reid_model)
    are at the TOP LEVEL of traj_data, NOT inside a nested "header" sub-dict.
    The old validate_trajectory_header(header, ...) pattern was reading an empty
    dict and silently passing every guard. This function corrects that.
    """
    if not traj_data.get("fragment_filter_applied", False):
        raise ValueError(
            f"[{seq_id}][{tracker}] fragment_filter_applied=False in _final.json. "
            "Day 9 post-processing must complete before Block C can run."
        )
    if not traj_data.get("interpolation_applied", False):
        raise ValueError(
            f"[{seq_id}][{tracker}] interpolation_applied=False in _final.json. "
            "Day 9 post-processing must complete before Block C can run."
        )
    # Defensive .get() — Day 7 SORT files may lack this field
    reid = traj_data.get("reid_model", "none")
    logger.info(f"[{seq_id}][{tracker}] Metadata valid. reid_model='{reid}'")


# ── Backward-compatibility alias ──────────────────────────────────────────────
# Keep old name importable in case any Day 9 test references it.
# New code must use validate_trajectory_file_metadata().
def validate_trajectory_header(traj_data: dict, seq_id: str, tracker: str) -> None:
    """Deprecated alias — use validate_trajectory_file_metadata() instead."""
    logger.warning(
        "validate_trajectory_header() is deprecated. "
        "Use validate_trajectory_file_metadata(traj_data, ...) — reads top-level fields."
    )
    validate_trajectory_file_metadata(traj_data, seq_id, tracker)


# ── Track dict flattening ─────────────────────────────────────────────────────

def flatten_tracks_dict(traj_data: dict) -> list:
    """
    Flatten the dict-keyed tracks structure used by Day 7–9 _final.json files.

    CRITICAL — actual _final.json structure:
        {
          "fragment_filter_applied": true,
          "interpolation_applied": true,
          "reid_model": "none",
          ...                           — all metadata at top level (no nested "header" dict)
          "tracks": {
              "1": [ {frame_idx, cx, cy, ...}, ... ],
              "42": [ {frame_idx, cx, cy, ...}, ... ],
              ...
          }
        }

    The "tracks" value is a DICT keyed by track_id string, NOT a flat list.
    pd.DataFrame(traj_data["tracks"]) would be wrong — use this function instead.

    Returns a flat list of entry dicts, each with an injected integer "track_id".
    """
    rows = []
    tracks_raw = traj_data.get("tracks", {})
    if isinstance(tracks_raw, list):
        # Defensive: if some future variant stores tracks as a list already, pass through
        logger.warning(
            "tracks field is a list, not a dict — using as-is. Verify _final.json structure."
        )
        return tracks_raw
    for tid_str, entries in tracks_raw.items():
        tid = int(tid_str)
        for e in entries:
            row = copy.deepcopy(e)
            row["track_id"] = tid
            rows.append(row)
    return rows


# ── Schema column-order enforcement ──────────────────────────────────────────

def _assert_schema_column_order(df: pd.DataFrame, seq_id: str) -> None:
    """
    Raise ValueError if the DataFrame's columns deviate from FEATURE_SCHEMA_COLUMN_ORDER.

    Implementation note:
    - F3 columns are not present in Day 10 CSVs — they are excluded from the check.
    - Only the intersection of (actual columns) and (schema columns) is order-checked.
    - If a column is MISSING — it won't appear in actual_today — the ordered lists will
      differ — ValueError raised immediately, before any CSV is written. This is the
      intended "fail loudly early" behaviour — a missing schema column is caught on the
      first clip, not discovered after all 10 CSVs are already written.
    - If a column is REORDERED — lists differ — ValueError raised. Any merge or concat
      operation that silently changes column order will be caught here.
    - This function is robustness-positive: it adds a safety net with zero false positives
      under correct operation and zero performance overhead (list comparison on ≤19 items).
    """
    actual = list(df.columns)
    # Filter both lists to only the Day 10 schema columns (F3 absent today)
    expected_today = [c for c in FEATURE_SCHEMA_COLUMN_ORDER if c in actual]
    actual_today   = [c for c in actual if c in FEATURE_SCHEMA_COLUMN_ORDER]
    if actual_today != expected_today:
        raise ValueError(
            f"[{seq_id}] Feature DataFrame column order deviates from schema.\n"
            f"  Expected (schema order): {expected_today}\n"
            f"  Got (actual order):      {actual_today}\n"
            "Fix the column reordering step in extract_features() — "
            "ensure `merged[available]` uses FEATURE_SCHEMA_COLUMN_ORDER as the filter."
        )


# ── F1 features — Density/Flow ────────────────────────────────────────────────

def extract_f1_features(trajs_df: pd.DataFrame, fps: float, config: dict) -> pd.DataFrame:
    """
    F1: Density/Flow features — vehicle_count and roi_occupancy.
    Returns DataFrame with columns: frame_idx, vehicle_count, roi_occupancy.
    One row per frame_idx.

    # roi_occupancy: relative vehicle occupancy proxy. Full-frame ROI — no pixel
    # segmentation. Declared in config_blockC.yaml: roi_policy: full_frame.
    # Limitation: includes roadside stationary vehicles (Fix F17).
    """
    # F1.1 — vehicle_count: unique active track_ids per frame
    frame_counts = (
        trajs_df.groupby("frame_idx")["track_id"]
        .nunique()
        .reset_index()
        .rename(columns={"track_id": "vehicle_count"})
    )

    # F1.2 — roi_occupancy: relative occupancy (vehicle_count / max_vehicle_count)
    max_count = frame_counts["vehicle_count"].max()
    if max_count == 0:
        logger.warning(
            "max vehicle_count is 0 — roi_occupancy will be 0. Check trajectory data."
        )
        frame_counts["roi_occupancy"] = 0.0
    else:
        frame_counts["roi_occupancy"] = frame_counts["vehicle_count"] / max_count

    return frame_counts  # columns: frame_idx, vehicle_count, roi_occupancy


# ── F2 features — Speed/Velocity ─────────────────────────────────────────────

def extract_f2_features(trajs_df: pd.DataFrame, fps: float, config: dict) -> pd.DataFrame:
    """
    F2: Speed/Velocity features — vel_px_sec, vel_px_sec_smooth, is_interpolated.
    Returns DataFrame at frame_idx + track_id granularity.

    # speed_window = int(0.2 * fps) — Fix F07: all windows in seconds, converted to
    # frames at runtime. At 25 FPS = 5 frames (~0.2s).
    """
    # speed_window = int(0.2 * fps) — Fix F07
    speed_window = int(0.2 * fps)
    if speed_window < 1:
        speed_window = 1
    logger.info(f"F2: speed_window = {speed_window} frames (0.2s at {fps} FPS)")

    required_cols = ["frame_idx", "track_id", "cx", "cy",
                     "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                     "conf", "velocity_norm"]
    missing = [c for c in required_cols if c not in trajs_df.columns]
    if missing:
        raise ValueError(
            f"extract_f2_features: missing required columns: {missing}. "
            "Check flatten_tracks_dict() and bbox unpacking in clip_feature_generator()."
        )

    df = trajs_df[required_cols].copy()

    # is_interpolated — source from _final.json entry field "interpolated".
    # MUST use .eq(True) not .astype(bool): non-interpolated entries have no
    # "interpolated" key in the JSON so the column contains NaN for those rows.
    # NaN.astype(bool) evaluates to True in pandas — would incorrectly flag
    # every genuine detection as interpolated. .eq(True) treats NaN as False.
    df["is_interpolated"] = (
        trajs_df["interpolated"].eq(True)
        if "interpolated" in trajs_df.columns
        else False
    )

    # Tier 1 debug log — clip-level interpolated frame count
    interp_count = int(df["is_interpolated"].sum())
    logger.debug(f"F2: {interp_count} interpolated frames in this clip.")

    # Verify sentinel: interpolated frames should have conf == -1.0 (Fix F22)
    bad_sentinel = df[(df["is_interpolated"] == True) & (df["conf"] != -1.0)]
    if len(bad_sentinel) > 0:
        logger.warning(
            f"F2: {len(bad_sentinel)} interpolated frames have conf != -1.0. "
            "Expected sentinel conf=-1.0 (Fix F22). Check Day 9 post_processor output."
        )

    # vel_px_sec = velocity_norm * fps (Fix F07)
    df["vel_px_sec"] = df["velocity_norm"] * fps
    # Propagate NaN if velocity_norm was NaN — do NOT fill with 0
    # (zero would contaminate OC-SVM training)
    df.loc[df["velocity_norm"].isna(), "vel_px_sec"] = np.nan

    # Rolling smooth per track — Fix F07: window in seconds → frames
    df = df.sort_values(["track_id", "frame_idx"])
    df["vel_px_sec_smooth"] = (
        df.groupby("track_id")["vel_px_sec"]
          .transform(lambda x: x.rolling(window=speed_window, min_periods=1).mean())
    )

    return df  # columns: frame_idx, track_id, cx, cy, bbox_*, conf, is_interpolated,
               #          velocity_norm, vel_px_sec, vel_px_sec_smooth


# ── Feature orchestrator ──────────────────────────────────────────────────────

def extract_features(
    trajs_df: pd.DataFrame,
    fps: float,
    config: dict,
    seq_id: str,
    split: str,
    tracker: str,
    traj_data_meta: dict,
    track_len_real: pd.DataFrame,
) -> pd.DataFrame:
    """
    Orchestrator: calls F1 + F2 today, stubs F3 (Day 11).
    Returns merged DataFrame matching features.csv schema.

    Parameters
    ----------
    trajs_df       : post-warm-up-strip trajectory DataFrame
    fps            : frames per second (from metadata.json)
    config         : loaded config_blockC.yaml dict
    seq_id         : sequence identifier string
    split          : "train" / "val" / "test"
    tracker        : champion tracker name
    traj_data_meta : full traj_data dict (top-level _final.json).
                     Retained for Day 11 F3 extension without breaking the signature.
                     Day 10 code does not read fields from traj_data_meta directly.
    track_len_real : pre-computed track_length_real from FULL pre-strip DataFrame
                     (computed in clip_feature_generator before warm-up stripping).
                     NEVER pass a post-strip computation here.
    """
    # F1 — per frame
    f1 = extract_f1_features(trajs_df, fps, config)

    # F2 — per frame + track
    f2 = extract_f2_features(trajs_df, fps, config)

    # Merge: F2 is the base (frame_idx + track_id). F1 broadcasts to all track rows per frame.
    merged = f2.merge(f1, on="frame_idx", how="left")

    # Metadata columns
    merged["seq_id"]  = seq_id
    merged["split"]   = split
    merged["tracker"] = tracker

    # track_length_real: from pre-strip computation passed in by generator.
    # NEVER recompute from trajs_df (post-strip) — that would under-count tracks
    # that started before the warm-up boundary.
    merged = merged.merge(track_len_real, on="track_id", how="left")

    # F3 stub — columns are absent from Day 10 CSVs and added by Day 11
    # augment_features_with_f3(). Do NOT add placeholder NaN columns here.
    # dwell_time_sec definition LOCKED Day 10:
    #   cumulative: (frame_idx - track_first_frame_idx_post_warmup) / fps
    #   NOT instantaneous spatial dwell — requires camera calibration outside scope.

    # Enforce column order per features.csv schema (F3 columns excluded today).
    # Only include columns that actually exist (guards against missing bbox columns).
    available = [c for c in FEATURE_SCHEMA_COLUMN_ORDER if c in merged.columns]
    return merged[available]


# ── Generator-based clip loader ───────────────────────────────────────────────

def clip_feature_generator(clip_ids, metadata, config):
    """
    Memory-safe generator. Loads one clip's _final.json at a time.
    Yields seq_id string on successful completion of each clip's CSV write.

    ── Feature Contract Gate ──────────────────────────────────────────────────
    Asserts that config["tracker"] is a real tracker name (not the Day 9
    placeholder) before the loop starts. Stops immediately if not populated.
    """
    set_all_seeds()

    # ── Feature Contract Gate ─────────────────────────────────────────────────
    assert config.get("tracker") not in (None, "", "<SELECTED_TRACKER_FROM_DAY9>"), (
        "config_blockC.yaml 'tracker' field is not populated. "
        "Run populate_config_blockc_tracker() before calling clip_feature_generator(). "
        "Confirm Day 9 champion tracker selection is complete."
    )

    base_dir     = config.get("trajectory_base", "logs/block_b/")
    tracker      = config["tracker"]
    features_dir = config.get("features_dir", "logs/block_c/")
    os.makedirs(features_dir, exist_ok=True)

    # Detect layout once — fails loudly if Day 9 incomplete
    layout = detect_day7_layout(base_dir, tracker)
    logger.info(f"Detected trajectory layout: {layout}")

    # Build split membership map for all clips
    split_map = {}
    for split_name in ["train", "val", "test"]:
        for s in metadata["split_seqs"][split_name]:
            split_map[s] = split_name

    # Post-loop assertion: at least one interpolated frame should exist across
    # all clips (confirms Day 9 interpolation is active).
    total_interpolated_frames_seen = 0

    for seq_id in clip_ids:
        try:
            traj_path = resolve_trajectory_path(base_dir, tracker, seq_id, layout)
            if not os.path.exists(traj_path):
                logger.warning(
                    f"[{seq_id}] _final.json not found at {traj_path}. Skipping. "
                    "Run Block B / Day 9 post-processing on this sequence first."
                )
                continue

            with open(traj_path, "r") as f:
                traj_data = json.load(f)

            # ── Validate top-level metadata fields ────────────────────────────
            # _final.json stores fragment_filter_applied, interpolation_applied,
            # reid_model at the TOP LEVEL — not inside a "header" sub-dict.
            validate_trajectory_file_metadata(traj_data, seq_id, tracker)

            # ── Flatten dict-keyed tracks to a list of row dicts ──────────────
            # CRITICAL: traj_data["tracks"] is a dict keyed by track_id string,
            # NOT a flat list. Use flatten_tracks_dict() — never pd.DataFrame(tracks).
            entries = flatten_tracks_dict(traj_data)
            if not entries:
                logger.warning(f"[{seq_id}] No track entries in _final.json. Skipping.")
                continue

            # ── Defensive bbox unpacking (list vs separate keys) ──────────────
            if "bbox" in entries[0] and isinstance(entries[0]["bbox"], list):
                for entry in entries:
                    x1, y1, x2, y2 = entry.pop("bbox")
                    entry["bbox_x1"], entry["bbox_y1"] = x1, y1
                    entry["bbox_x2"], entry["bbox_y2"] = x2, y2

            full_trajs = pd.DataFrame(entries)

            # ── Compute track_length_real BEFORE warm-up stripping (Fix 6.3) ──
            # track_length_real = genuine detection count (interpolated==False)
            # Must use the FULL pre-strip DataFrame.
            # Use .eq(True) for interpolated flag: entries without the key have
            # NaN in the column; NaN == False would be False, but NaN != False is
            # also False in pandas — safer to explicitly mark genuine frames as
            # those where interpolated is NOT True.
            if "interpolated" in full_trajs.columns:
                genuine_mask = ~full_trajs["interpolated"].eq(True)
            else:
                genuine_mask = pd.Series(True, index=full_trajs.index)

            track_len_real = (
                full_trajs[genuine_mask]
                .groupby("track_id")["frame_idx"]
                .count()
                .rename("track_length_real")
                .reset_index()
            )

            # ── Warm-up stripping (Fix F16) ───────────────────────────────────
            fps = metadata["sequences"][seq_id]["fps"]
            warmup_boundary = max(10, int(0.4 * fps))
            trajs = full_trajs[full_trajs["frame_idx"] > warmup_boundary].copy()
            logger.info(
                f"[{seq_id}] After warmup strip (>{warmup_boundary}): {len(trajs)} entries"
            )
            del full_trajs  # release pre-strip DataFrame immediately

            if trajs.empty:
                logger.warning(
                    f"[{seq_id}] No entries remain after warm-up stripping. Skipping."
                )
                continue

            # Guard: raise if split unknown — must not silently write "unknown" split
            clip_split = split_map.get(seq_id)
            if clip_split is None:
                raise ValueError(
                    f"[{seq_id}] seq_id not found in metadata split_seqs. "
                    "All 10 sequences must have a known split. Check logs/metadata.json."
                )

            # Track interpolated frame count for post-loop assert.
            # Use .eq(True) to avoid NaN being counted as True.
            if "interpolated" in trajs.columns:
                total_interpolated_frames_seen += int(trajs["interpolated"].eq(True).sum())

            # ── Extract features ──────────────────────────────────────────────
            feats = extract_features(
                trajs_df=trajs,
                fps=fps,
                config=config,
                seq_id=seq_id,
                split=clip_split,
                tracker=tracker,
                traj_data_meta=traj_data,
                track_len_real=track_len_real,
            )

            # ── Schema column-order enforcement ───────────────────────────────
            # Raises if DataFrame columns deviate from schema order.
            _assert_schema_column_order(feats, seq_id)

            # ── Write CSV ─────────────────────────────────────────────────────
            out_path = os.path.join(features_dir, f"{seq_id}_features.csv")
            feats.to_csv(out_path, index=False)
            logger.info(
                f"[{seq_id}] Features written → {out_path} "
                f"({len(feats)} rows, {len(feats.columns)} cols)"
            )

            # ── Memory hygiene (explicit del) ─────────────────────────────────
            del trajs, feats

            yield seq_id

        except Exception as e:
            logger.error(
                f"[{seq_id}] Feature extraction failed: {e}", exc_info=True
            )
            # Do NOT re-raise — continue to next clip so one bad clip does not
            # abort the entire run.

    # ── Post-loop interpolated-frame pre-flight assert (Tier 2) ──────────────
    if total_interpolated_frames_seen == 0:
        logger.warning(
            "Pre-flight assertion: ZERO interpolated frames encountered across ALL clips. "
            "This is unexpected — Day 9 interpolation should have produced synthetic frames. "
            "Verify that _final.json files were produced by Day 9 post_processor.py and that "
            "the 'interpolated' field is correctly set in trajectory entries."
        )
    else:
        logger.info(
            f"Pre-flight assert passed: {total_interpolated_frames_seen} interpolated frames "
            "found across all processed clips — confirms Day 9 interpolation is active."
        )


# ── Config population helper ──────────────────────────────────────────────────

def populate_config_blockc_tracker(config: dict) -> dict:
    """
    Read logs/block_b_results.json, find the selected tracker,
    update config dict in-place, and rewrite config_blockC.yaml using yaml.safe_dump().

    CRITICAL — JSON structure note:
    block_b_results.json is a dict with a "trackers" key containing the list of tracker
    rows (not a bare list at top-level). Always iterate results["trackers"].

    Returns the updated config dict.

    ── Feature Contract Gate ─────────────────────────────────────────────────
    If the tracker field in config_blockC.yaml is still the placeholder string
    '<SELECTED_TRACKER_FROM_DAY9>', this function MUST populate it before the
    clip_feature_generator loop starts. If block_b_results.json has no entry
    marked 'selected: true', this function raises immediately — Block C cannot
    proceed without a confirmed champion tracker from Day 9.
    """
    results_path = "logs/block_b_results.json"
    config_path  = "configs/config_blockC.yaml"

    with open(results_path, "r") as f:
        results = json.load(f)

    # CRITICAL: block_b_results.json is {"trackers": [...], ...} — NOT a bare list.
    tracker_rows = results["trackers"]

    selected = None
    for entry in tracker_rows:
        if entry.get("selected", False):
            selected = entry["tracker"]
            break

    if selected is None:
        raise ValueError(
            "No tracker marked 'selected: true' in logs/block_b_results.json[\"trackers\"]. "
            "Day 9 tracker selection must be complete before Day 10."
        )

    if config.get("tracker") == selected:
        logger.info(
            f"config_blockC.yaml tracker already set to '{selected}'. No change needed."
        )
        return config

    config["tracker"] = selected
    logger.info(f"Tracker set to '{selected}' from block_b_results.json.")

    # Write back to YAML using yaml.safe_dump() to produce a canonical, parseable file.
    # Do NOT use string replace — it is fragile if the placeholder appears multiple
    # times or if the YAML has already been partially populated.
    with open(config_path, "r") as f:
        existing = yaml.safe_load(f)
    existing["tracker"] = selected
    with open(config_path, "w") as f:
        yaml.safe_dump(existing, f, default_flow_style=False, sort_keys=False)
    logger.info(f"config_blockC.yaml rewritten via yaml.safe_dump(): tracker = {selected}")

    return config


# ── F3: Interaction features ──────────────────────────────────────────────────

def extract_f3_features(trajs_df: pd.DataFrame, fps: float, config: dict) -> pd.DataFrame:
    """
    F3: Interaction features.
    Returns DataFrame at frame_idx + track_id granularity.

    Columns produced:
        inter_vehicle_dist_norm   (float, NaN when only one track in frame)
        dwell_time_sec            (float, cumulative)
        proximity_flag            (bool)
        proximity_count_rolling   (float, rolling sum)

    Requires trajs_df to contain: frame_idx, track_id, cx, cy
    (post-warm-up-stripped DataFrame — warm-up already removed by
    clip_feature_generator before this function is called).
    """
    set_all_seeds()

    # ── Runtime window computation (Fix F07) ──────────────────────────────────
    prox_window = int(config.get("f3", {}).get("prox_window_sec", 2.0) * fps)
    if prox_window < 1:
        prox_window = 1
    # prox_window = int(2.0 * fps) — all windows in seconds (Fix F07).
    # At 25 FPS = 50 frames (~2.0s).
    prox_threshold = config.get("f3", {}).get("proximity_threshold_dist", 0.15)
    logger.info(f"F3: prox_window={prox_window} frames, prox_threshold={prox_threshold}")

    # Safety: check minimum required columns
    required_cols = {"frame_idx", "track_id", "cx", "cy"}
    missing = required_cols - set(trajs_df.columns)
    if missing:
        raise ValueError(f"extract_f3_features: missing required columns {missing}")

    df = trajs_df[["frame_idx", "track_id", "cx", "cy"]].copy()

    # ── F3.1: inter_vehicle_dist_norm ─────────────────────────────────────────
    # inter_vehicle_dist_norm: image-space proxy only. Normalised by frame
    # diagonal (640*sqrt(2)). Perspective effects mean pixel distance does not
    # linearly represent physical distance. Values are relative behavioural
    # indicators only (Fix F28 — perspective disclaimer).
    # No homographic projection applied — declared scope constraint.
    dist_rows = []
    for frame_idx, group in df.groupby("frame_idx"):
        track_ids = group["track_id"].values
        cxs = group["cx"].values
        cys = group["cy"].values
        n = len(track_ids)
        if n < 2:
            # Single-track frame: NaN — do NOT use 0.0 (would be indistinguishable
            # from two tracks at identical centroids).
            for tid in track_ids:
                dist_rows.append({
                    "frame_idx": frame_idx,
                    "track_id": int(tid),
                    "inter_vehicle_dist_norm": np.nan,
                })
        else:
            # Compute pairwise Euclidean distances; assign each track its minimum.
            for i in range(n):
                min_dist = np.inf
                for j in range(n):
                    if i == j:
                        continue
                    d = np.sqrt((cxs[i] - cxs[j]) ** 2 + (cys[i] - cys[j]) ** 2)
                    if d < min_dist:
                        min_dist = d
                norm_d = np.clip(min_dist / FRAME_DIAG, 0.0, 1.0)
                dist_rows.append({
                    "frame_idx": frame_idx,
                    "track_id": int(track_ids[i]),
                    "inter_vehicle_dist_norm": norm_d,
                })

    dist_df = pd.DataFrame(dist_rows)
    df = df.merge(dist_df, on=["frame_idx", "track_id"], how="left")

    # ── F3.2: dwell_time_sec ──────────────────────────────────────────────────
    # dwell_time_sec: cumulative dwell since first post-warm-up appearance.
    # Formula: (frame_idx - track_first_frame_idx_post_warmup) / fps
    # Definition locked Day 10 (feature_schema.json f3_definitions_locked_day10).
    # NOT instantaneous spatial dwell — camera calibration required for that
    # (Fix F19 deferred).
    first_frames = (
        df.groupby("track_id")["frame_idx"]
          .min()
          .rename("track_first_frame_idx")
          .reset_index()
    )
    df = df.merge(first_frames, on="track_id", how="left")
    df["dwell_time_sec"] = (df["frame_idx"] - df["track_first_frame_idx"]) / fps
    df["dwell_time_sec"] = df["dwell_time_sec"].clip(lower=0.0)  # must never be negative
    df.drop(columns=["track_first_frame_idx"], inplace=True)

    # ── F3.3: proximity_flag ──────────────────────────────────────────────────
    # proximity_flag: True when normalised inter-vehicle distance <
    # proximity_threshold_dist. Threshold read from config_blockC.yaml
    # f3.proximity_threshold_dist (default 0.15). Indicates potentially unsafe
    # following distance — relative indicator, not calibrated metres.
    df["proximity_flag"] = (
        df["inter_vehicle_dist_norm"].notna()
        & (df["inter_vehicle_dist_norm"] < prox_threshold)
    )

    # ── Rolling proximity count (Fix F07 — window in seconds) ─────────────────
    # prox_window = int(2.0 * fps) — all windows in seconds (Fix F07).
    # At 25 FPS = 50 frames (~2.0s).
    df = df.sort_values(["track_id", "frame_idx"])
    df["proximity_count_rolling"] = (
        df.groupby("track_id")["proximity_flag"]
          .transform(
              lambda x: x.astype(float).rolling(window=prox_window, min_periods=1).sum()
          )
    )

    # Return only F3 columns + identifiers for merge
    return df[
        ["frame_idx", "track_id",
         "inter_vehicle_dist_norm", "dwell_time_sec",
         "proximity_flag", "proximity_count_rolling"]
    ]


def augment_features_with_f3(seq_id: str, metadata: dict, config: dict) -> None:
    """
    Load Day 10 features.csv, compute F3 features from _final.json, merge,
    and rewrite the CSV in-place. Idempotent: skips if F3 columns already present.

    ── Strategy: Load-Augment-Rewrite ───────────────────────────────────────
    1. Load logs/block_c/{seq_id}_features.csv  (Day 10 output)
    2. Validate Day 10 columns present
    3. Assert F3 NOT yet present (idempotency guard)
    4. Load _final.json for cx, cy per track per frame (canonical source —
       reloaded from _final.json, not reused from CSV centroids, for auditability)
    5. Compute F3 features via extract_f3_features()
    6. Merge F3 into Day 10 DataFrame on [frame_idx, track_id]
    7. Enforce schema column order
    8. Rewrite logs/block_c/{seq_id}_features.csv (in-place)
    """
    set_all_seeds()

    features_dir = config.get("features_dir", "logs/block_c/")
    csv_path = os.path.join(features_dir, f"{seq_id}_features.csv")

    if not os.path.exists(csv_path):
        logger.warning(f"[{seq_id}] features.csv not found — cannot augment with F3.")
        return

    df_day10 = pd.read_csv(csv_path)

    # ── Day 10 presence guard ─────────────────────────────────────────────────
    day10_required = [
        "vel_px_sec", "vel_px_sec_smooth", "vehicle_count", "roi_occupancy",
        "track_length_real", "split", "tracker",
    ]
    missing_day10 = [c for c in day10_required if c not in df_day10.columns]
    if missing_day10:
        raise ValueError(
            f"[{seq_id}] Day 10 columns missing from features.csv: {missing_day10}. "
            "Run Day 10 generator before augmenting."
        )

    # ── Idempotency guard ─────────────────────────────────────────────────────
    if "inter_vehicle_dist_norm" in df_day10.columns:
        logger.info(
            f"[{seq_id}] F3 already present (inter_vehicle_dist_norm found) — "
            "skipping augmentation."
        )
        return

    # ── Load _final.json for trajectory data ──────────────────────────────────
    tracker = config.get("tracker")
    base_dir = config.get("trajectory_base", "logs/block_b/")
    layout = detect_day7_layout(base_dir, tracker)
    traj_path = resolve_trajectory_path(base_dir, tracker, seq_id, layout)

    if not os.path.exists(traj_path):
        logger.warning(
            f"[{seq_id}] _final.json not found at {traj_path} — cannot compute F3."
        )
        return

    with open(traj_path, "r") as f:
        traj_data = json.load(f)

    # Validate and flatten
    validate_trajectory_file_metadata(traj_data, seq_id, tracker)
    entries = flatten_tracks_dict(traj_data)
    if not entries:
        logger.warning(f"[{seq_id}] No entries in _final.json — skipping F3 augmentation.")
        return

    if "bbox" in entries[0] and isinstance(entries[0]["bbox"], list):
        for entry in entries:
            x1, y1, x2, y2 = entry.pop("bbox")
            entry["bbox_x1"], entry["bbox_y1"] = x1, y1
            entry["bbox_x2"], entry["bbox_y2"] = x2, y2

    full_trajs = pd.DataFrame(entries)

    # ── Apply same warm-up stripping as Day 10 ────────────────────────────────
    fps = metadata["sequences"][seq_id]["fps"]
    warmup_boundary = max(10, int(0.4 * fps))
    trajs = full_trajs[full_trajs["frame_idx"] > warmup_boundary].copy()
    del full_trajs

    if trajs.empty:
        logger.warning(
            f"[{seq_id}] No entries after warm-up stripping — skipping F3 augmentation."
        )
        return

    # Single-track clip warning (F3 values will be mostly NaN/False/0)
    n_unique_tracks = trajs["track_id"].nunique() if "track_id" in trajs.columns else 0
    if n_unique_tracks < 2:
        logger.warning(
            f"[{seq_id}] Only {n_unique_tracks} unique track(s) found post-warmup. "
            "inter_vehicle_dist_norm will be all-NaN. This is valid — "
            "F3 requires at least two simultaneous tracks to be meaningful."
        )

    # ── Compute F3 ────────────────────────────────────────────────────────────
    f3_df = extract_f3_features(trajs, fps, config)
    del trajs

    # ── Merge F3 into Day 10 DataFrame ────────────────────────────────────────
    df_merged = df_day10.merge(f3_df, on=["frame_idx", "track_id"], how="left")

    # ── Log merge stats ───────────────────────────────────────────────────────
    n_nan_dist = int(df_merged["inter_vehicle_dist_norm"].isna().sum())
    n_total = len(df_merged)
    logger.info(
        f"[{seq_id}] F3 merged: {n_total} rows total, "
        f"{n_nan_dist} NaN inter_vehicle_dist_norm "
        f"({100.0 * n_nan_dist / n_total:.1f}% — single-track frames)."
    )

    # ── Enforce schema column order ───────────────────────────────────────────
    available = [c for c in FEATURE_SCHEMA_COLUMN_ORDER if c in df_merged.columns]
    df_final = df_merged[available]
    _assert_schema_column_order(df_final, seq_id)

    # ── Rewrite CSV (in-place) ────────────────────────────────────────────────
    df_final.to_csv(csv_path, index=False)
    logger.info(
        f"[{seq_id}] features.csv augmented with F3 — {csv_path} "
        f"({len(df_final)} rows, {len(df_final.columns)} cols)"
    )


def fit_and_save_scaler(config: dict, metadata: dict):
    """
    Fix F13: fit MinMaxScaler on train-split NORMAL sequences only.
    Never fit on val, test, or AI City data.
    Saves scaler to models/minmax_scaler.pkl and Drive models dir.
    Returns fitted MinMaxScaler.
    """
    from sklearn.preprocessing import MinMaxScaler
    import joblib

    set_all_seeds()

    features_dir = config.get("features_dir", "logs/block_c/")
    scaler_path = config.get("scaler", {}).get("path", "models/minmax_scaler.pkl")
    exclude_interp = config.get("exclude_interpolated_from_scaler_fit", True)

    # Load ALL sequence CSVs and concatenate
    all_dfs = []
    for split_name in ["train", "val", "test"]:
        for seq_id in metadata["split_seqs"][split_name]:
            csv_path = os.path.join(features_dir, f"{seq_id}_features.csv")
            if not os.path.exists(csv_path):
                logger.warning(
                    f"[{seq_id}] features.csv not found — skipping in scaler fit pool."
                )
                continue
            df = pd.read_csv(csv_path)
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError(
            "No feature CSVs found. Run Day 10 generator and Day 11 F3 augmentation first."
        )

    full_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Loaded {len(full_df)} total rows across all sequences.")

    # Fix F13: fit on TRAIN ONLY
    if exclude_interp:
        df_fit = full_df[
            (full_df["split"] == "train") & (full_df["is_interpolated"] == False)
        ]
        logger.info(
            f"Fix F13 + exclude_interpolated: fitting on {len(df_fit)} "
            "train-split non-interpolated rows."
        )
    else:
        df_fit = full_df[full_df["split"] == "train"]
        logger.info(
            f"Fix F13: fitting on {len(df_fit)} train-split rows "
            "(interpolated rows included)."
        )

    if df_fit.empty:
        raise RuntimeError(
            "No train-split rows found for scaler fitting. "
            "Check split labels in features.csv."
        )

    # Validate all scaler columns are present before attempting fit
    fit_cols_present = [c for c in SCALER_FIT_COLUMNS if c in df_fit.columns]
    missing_cols = [c for c in SCALER_FIT_COLUMNS if c not in df_fit.columns]
    if missing_cols:
        raise RuntimeError(
            f"Scaler fit columns not found in features.csv: {missing_cols}. "
            "Ensure F3 augmentation ran before scaler fitting."
        )

    # Drop NaN rows for fit columns — do not impute.
    # NaN inter_vehicle_dist_norm rows come from single-track frames where distance
    # is undefined, not zero. These rows are intentionally excluded from scaler
    # fitting rather than imputed.
    df_fit_clean = df_fit[fit_cols_present].dropna()
    n_dropped = len(df_fit) - len(df_fit_clean)
    if n_dropped > 0:
        pct_dropped = 100.0 * n_dropped / len(df_fit)
        logger.warning(
            f"Scaler fit: dropped {n_dropped} NaN rows ({pct_dropped:.1f}%) "
            "from train-split data."
        )

    # Sparse-traffic audit: ensure enough valid multi-track rows remain
    min_valid_rows = int(config.get("scaler", {}).get("min_valid_fit_rows", 100))
    if len(df_fit_clean) < min_valid_rows:
        raise RuntimeError(
            f"Scaler fit pool too small after NaN exclusion: "
            f"{len(df_fit_clean)} valid rows < min_valid_fit_rows={min_valid_rows}. "
            "Train split may be too sparse for stable F3 scaling; inspect NaN drop "
            "counts and inter_vehicle_dist_norm coverage."
        )
    logger.info(
        f"Scaler fit audit: {len(df_fit_clean)} valid rows remain after NaN exclusion "
        f"out of {len(df_fit)} candidate train rows."
    )

    scaler = MinMaxScaler()
    scaler.fit(df_fit_clean)
    logger.info(
        f"MinMaxScaler fitted on {len(df_fit_clean)} rows, "
        f"{len(fit_cols_present)} feature columns."
    )

    # Log fitted range per column (audit trail for Fix F13 and Fix F18)
    for col, (fmin, fmax) in zip(
        fit_cols_present, zip(scaler.data_min_, scaler.data_max_)
    ):
        logger.info(f"  Scaler fit range [{col}]: min={fmin:.4f}, max={fmax:.4f}")

    # Save scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    import joblib as _joblib
    _joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved → {scaler_path}")

    # Mirror to Drive
    drive_scaler_path = "/content/drive/MyDrive/CST3990/models/minmax_scaler.pkl"
    try:
        import shutil
        os.makedirs(os.path.dirname(drive_scaler_path), exist_ok=True)
        shutil.copy2(scaler_path, drive_scaler_path)
        logger.info(f"Scaler mirrored → {drive_scaler_path}")
    except Exception as e:
        logger.warning(f"Drive mirror failed (non-fatal): {e}")

    return scaler


def apply_scaler_transform(df: pd.DataFrame, scaler, config: dict,
                           label: str = "unknown") -> pd.DataFrame:
    """
    Transform feature columns using a fitted MinMaxScaler.
    Clips output to [0,1] and logs clipping % as a domain-shift indicator (Fix F18).

    Rules:
    - Rows with complete scaler-fit features are scaled normally.
    - Rows with any NaN in scaler-fit features are NOT partially preserved in raw form.
      Instead, all scaler-fit columns for those rows are set to NaN in the scaled CSV.
      This keeps scaled outputs semantically consistent for Day 12+ modelling.
    """
    import numpy as np
    import pandas as pd

    fit_cols = [c for c in SCALER_FIT_COLUMNS if c in df.columns]
    df_out = df.copy()

    # Ensure scaler-fit columns are float-compatible before assignment
    for col in fit_cols:
        df_out[col] = df_out[col].astype(float)

    valid_mask = df_out[fit_cols].notna().all(axis=1)
    invalid_mask = ~valid_mask

    n_total = len(df_out)
    n_valid = int(valid_mask.sum())
    n_invalid = int(invalid_mask.sum())

    if n_valid == 0:
        raise RuntimeError(
            f"[{label}] No valid rows available for scaler transform. "
            "All rows contain NaN in at least one scaler-fit column."
        )

    # For scaled CSV semantics: invalid rows should not keep raw unscaled values
    # in scaler-fit columns. Mark them all NaN.
    df_out.loc[invalid_mask, fit_cols] = np.nan

    # Preserve feature names by passing a DataFrame
    X_valid = df.loc[valid_mask, fit_cols].astype(float)
    X_scaled = scaler.transform(X_valid)

    # Domain-shift logging before clipping
    n_clipped = int(np.sum((X_scaled > 1.0) | (X_scaled < 0.0)))
    pct_clipped = 100.0 * n_clipped / X_scaled.size if X_scaled.size > 0 else 0.0

    if pct_clipped > 0:
        logger.warning(
            f"[{label}] Fix F18 domain-shift indicator: "
            f"{n_clipped} values ({pct_clipped:.2f}%) clipped to [0,1] after MinMaxScaler transform. "
            "High clip % may indicate distribution shift between train (UA-DETRAC) and this split."
        )
    else:
        logger.info(f"[{label}] No out-of-range values after scaler transform. Clip % = 0.00%.")

    X_scaled = np.clip(X_scaled, 0.0, 1.0)

    if not np.isfinite(X_scaled).all():
        raise RuntimeError(
            f"[{label}] Non-finite values produced on rows eligible for scaler transform. "
            "Check zero-variance guard and scaler state."
        )

    # Write scaled values back only to valid rows
    df_out.loc[valid_mask, fit_cols] = X_scaled

    if n_invalid > 0:
        logger.warning(
            f"[{label}] {n_invalid} / {n_total} rows marked NaN in scaled CSV "
            "because at least one scaler-fit column was NaN "
            "(expected for undefined interaction features such as single-track frames)."
        )

    numeric_block = df_out[fit_cols].to_numpy()
    if np.isinf(numeric_block).any():
        raise RuntimeError(f"[{label}] Infinite values detected after scaler transform.")

    logger.info(
        f"[{label}] Scaler transform applied to {n_valid} valid rows; "
        f"{n_invalid} rows written with NaN in scaler-fit columns."
    )

    return df_out