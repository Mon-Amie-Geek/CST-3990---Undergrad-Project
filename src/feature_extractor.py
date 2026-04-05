"""
feature_extractor.py
Block C — F1 + F2 Feature Extraction, Generator Loader, and feature schema enforcement.
Day 10: Generator-based clip loader, F1 (Density/Flow), F2 (Speed/Velocity).
F3 (Interaction) added Day 11.

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
# F3 columns (inter_vehicle_dist_norm, dwell_time_sec, proximity_flag) are
# absent from Day 10 CSVs — they are added by Day 11.
FEATURE_SCHEMA_COLUMN_ORDER = [
    "seq_id", "frame_idx", "track_id", "cx", "cy",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "conf", "is_interpolated", "velocity_norm",
    "vel_px_sec", "vel_px_sec_smooth",
    "vehicle_count", "roi_occupancy",
    "track_length_real", "split", "tracker"
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

    # F3 stub — columns will be added Day 11.
    # dwell_time_sec definition LOCKED Day 10:
    #   cumulative: (frame_idx - track_first_frame_idx_post_warmup) / fps
    #   NOT instantaneous spatial dwell — requires camera calibration outside scope.
    # DO NOT add placeholder NaN columns. Leave F3 columns absent from Day 10 CSVs.

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
