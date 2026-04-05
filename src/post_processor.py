"""
post_processor.py — Block B Day 9 post-processing
CST3990 | MANJOO Ameera Najla | M01014463

Fragment filter (Fix F11) + Track interpolation (Fix F22).
Applied to ALL THREE trackers: SORT, DeepSORT, ByteTrack.

IMPORTANT: Trajectory files use a dict-keyed format:
    trajectory_data["tracks"] = {"<track_id_str>": [entry, entry, ...], ...}
  Entries do NOT contain a "track_id" field — the key IS the track id.
  Day 7 SORT entries also lack "interpolated" and "track_length".
  All of this is handled gracefully below.
"""

import os
import json
import copy
import logging
import numpy as np
from collections import defaultdict

from src.seed_control import set_all_seeds
set_all_seeds()  # MUST be at module top — before any numpy/random operation

# Module-level logger — defined at module level (not inside functions)
# to prevent NameError on logger.warning() / logger.info() calls in any function.
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (read from config at runtime; these are defaults)
# ---------------------------------------------------------------------------
MIN_TRACK_LENGTH = 15          # Fix F11 — fragment filter threshold
MAX_INTERP_GAP   = 3           # Fix F22 — max frames to interpolate across
SPLIT_TRACK_ID_MULTIPLIER = 1000  # new_id = orig_int_id * 1000 + split_index

# Reid model audit trail — examiner requirement.
# Every output header must carry honest metadata matching the actual tracker used.
REID_MODEL_MAP = {
    "sort":      "none",
    "bytetrack": "none",
    "deepsort":  "osnet_x1_0_veri_776",
}


# ---------------------------------------------------------------------------
# Path helpers — variant_a vs variant_b layout
# ---------------------------------------------------------------------------

def get_raw_path(tracker: str, seq: str, layout: str) -> str:
    if layout == "variant_a":
        return f"logs/block_b/{tracker}/trajectories_{seq}.json"
    return f"logs/block_b/trajectories/{seq}_{tracker}.json"


def get_filtered_path(tracker: str, seq: str, layout: str) -> str:
    if layout == "variant_a":
        return f"logs/block_b/{tracker}/trajectories_{seq}_filtered.json"
    return f"logs/block_b/trajectories/{seq}_{tracker}_filtered.json"


def get_final_path(tracker: str, seq: str, layout: str) -> str:
    if layout == "variant_a":
        return f"logs/block_b/{tracker}/trajectories_{seq}_final.json"
    return f"logs/block_b/trajectories/{seq}_{tracker}_final.json"


# ---------------------------------------------------------------------------
# Fragment filter — Fix F11
# ---------------------------------------------------------------------------

def compute_track_lengths(tracks_dict: dict) -> dict:
    """
    Returns {track_id_str: int} count of frames per track.
    Works with the dict-keyed trajectory format:
        tracks_dict = {"11": [entry, entry, ...], "10": [...], ...}
    Used when track_length is absent from entries (Day 7 SORT files).
    """
    return {tid: len(entries) for tid, entries in tracks_dict.items()}


def apply_fragment_filter(trajectory_data: dict, tracker_name: str,
                           min_length: int = MIN_TRACK_LENGTH) -> dict:
    """
    Removes tracks shorter than min_length from trajectory_data.

    # Fragment filter applied BEFORE interpolation — avoids interpolating tracks
    # that would later be discarded, saving compute and preventing ghost
    # interpolation artefacts.

    Works with the dict-keyed format: trajectory_data["tracks"] is a dict
    keyed by track_id string; values are lists of frame entries.

    Does NOT mutate the input dict — always deep-copies before modifying.
    Sets fragment_filter_applied: true in the returned copy.
    Backfills/validates reid_model using REID_MODEL_MAP (examiner audit trail).
    """
    result = copy.deepcopy(trajectory_data)
    tracks_dict = result["tracks"]  # {"tid": [entry, ...], ...}

    original_count = len(tracks_dict)
    valid_tracks = {
        tid: entries
        for tid, entries in tracks_dict.items()
        if len(entries) >= min_length
    }
    filtered_count = len(valid_tracks)
    discarded = original_count - filtered_count

    result["tracks"] = valid_tracks
    result["fragment_filter_applied"] = True
    result["fragment_filter_min_length"] = min_length
    result["tracks_discarded_by_filter"] = discarded
    result["interpolation_applied"] = False  # not yet applied at this stage

    # MANDATORY: set reid_model from REID_MODEL_MAP — honest metadata for examiner
    correct_reid = REID_MODEL_MAP.get(tracker_name, "none")
    existing_reid = result.get("reid_model", None)
    if existing_reid is not None and existing_reid != correct_reid:
        logger.warning(
            f"reid_model mismatch for {tracker_name}: header says '{existing_reid}', "
            f"expected '{correct_reid}'. Correcting to '{correct_reid}'."
        )
    result["reid_model"] = correct_reid

    logger.info(
        f"Fragment filter [{tracker_name}]: {original_count} tracks → "
        f"{filtered_count} kept ({discarded} discarded, min_length={min_length})"
    )
    return result


# ---------------------------------------------------------------------------
# Track interpolation — Fix F22
# ---------------------------------------------------------------------------

def apply_interpolation(trajectory_data: dict,
                         max_gap: int = MAX_INTERP_GAP) -> dict:
    """
    Linear interpolation for gaps ≤ max_gap frames.
    Track split (new dict key) for gaps > max_gap frames.

    ⚠️ HALLUCINATED VELOCITY PROTECTION:
    Gaps > max_gap are SPLIT, NOT interpolated. A gap of 30 frames interpolated
    linearly creates a ghost straight line — constant fake velocity data that
    OC-SVM (Block C) will misinterpret as "perfectly smooth," masking real
    anomalies or generating false low-speed features. The split policy is the
    correct defence. MAX_INTERP_GAP=3 is conservative and must not be raised.

    Input/output format: trajectory_data["tracks"] is a dict keyed by track_id
    string; values are lists of frame entries sorted by frame_idx.

    For split tracks: new key = str(original_int_id * 1000 + split_index).
    If no split occurs, the original key is preserved.

    velocity_norm on interpolated frames:
      Raw frame-to-frame Euclidean distance from previous entry's cx, cy.
      # Interpolation performed in image-space units (640×640 normalised).
      No smoothing applied here — Block C applies its own speed_window_sec=0.2
      smoothing. These are separate pipeline stages; do not conflate them.

    track_length on interpolated entries:
      Stores track_length_real = ORIGINAL real-detection count only.
      Does NOT increment for synthetic frames.
      # track_length = track_length_real (original real-detection count;
      #                does not include interpolated synthetic frames)
      # track_length_total is computable as:
      #   track_length_real + count(interpolated==True) per track
      Block C feature engineering uses track_length_total for span coverage,
      track_length_real for "how long was this vehicle genuinely observed."

    conf on interpolated entries: -1.0 (sentinel — marks synthetic).
      Never use 0.0 which could be confused with a real low-confidence detection.

    Does NOT mutate the input — always deep-copies first.
    """
    result = copy.deepcopy(trajectory_data)
    tracks_dict = result["tracks"]  # {"tid": [entry, ...], ...}

    new_tracks_dict = {}
    total_interpolated = 0

    for tid_str, entries in tracks_dict.items():
        tid_int = int(tid_str)
        entries_sorted = sorted(entries, key=lambda e: e["frame_idx"])
        real_length = len(entries_sorted)  # original detection count

        # Determine if any gap exceeds max_gap (split needed)
        needs_split = any(
            entries_sorted[i]["frame_idx"] - entries_sorted[i - 1]["frame_idx"] - 1 > max_gap
            for i in range(1, real_length)
        )

        if not needs_split:
            # No split — interpolate gaps ≤ max_gap, preserve original key
            original_length = entries_sorted[0].get("track_length", real_length)
            new_entries = [copy.deepcopy(entries_sorted[0])]

            for i in range(1, real_length):
                prev = new_entries[-1]
                curr = entries_sorted[i]
                gap = curr["frame_idx"] - prev["frame_idx"] - 1

                if gap == 0:
                    new_entries.append(copy.deepcopy(curr))

                elif 1 <= gap <= max_gap:
                    for g in range(1, gap + 1):
                        alpha = g / (gap + 1)
                        interp_frame = prev["frame_idx"] + g

                        icx  = prev["cx"] + alpha * (curr["cx"] - prev["cx"])
                        icy  = prev["cy"] + alpha * (curr["cy"] - prev["cy"])
                        ix1  = prev["bbox"][0] + alpha * (curr["bbox"][0] - prev["bbox"][0])
                        iy1  = prev["bbox"][1] + alpha * (curr["bbox"][1] - prev["bbox"][1])
                        ix2  = prev["bbox"][2] + alpha * (curr["bbox"][2] - prev["bbox"][2])
                        iy2  = prev["bbox"][3] + alpha * (curr["bbox"][3] - prev["bbox"][3])

                        prev_cx = new_entries[-1]["cx"]
                        prev_cy = new_entries[-1]["cy"]
                        vel = float(np.sqrt((icx - prev_cx) ** 2 + (icy - prev_cy) ** 2))

                        interp_entry = {
                            "frame_idx":    interp_frame,
                            "bbox":         [ix1, iy1, ix2, iy2],
                            "conf":         -1.0,  # sentinel: synthetic entry, NOT 0.0
                            "cx":           icx,
                            "cy":           icy,
                            "velocity_norm": vel,
                            "interpolated": True,
                            # track_length = track_length_real (original real-detection count;
                            # does not include interpolated synthetic frames).
                            # track_length_total = track_length_real + count(interpolated==True)
                            # computable by Block C feature engineering per track.
                            "track_length": original_length,
                        }
                        new_entries.append(interp_entry)
                        total_interpolated += 1

                    new_entries.append(copy.deepcopy(curr))

                else:
                    # gap > max_gap — should not reach here since needs_split=False
                    # but handle defensively: just append without interpolation
                    logger.warning(
                        f"Unexpected large gap ({gap} frames) in track {tid_str} "
                        f"when needs_split was False. Appending without interpolation."
                    )
                    new_entries.append(copy.deepcopy(curr))

            new_tracks_dict[tid_str] = new_entries

        else:
            # Split required — use multiplier scheme for new keys
            split_idx = 0
            original_length = entries_sorted[0].get("track_length", real_length)
            current_key = str(tid_int * SPLIT_TRACK_ID_MULTIPLIER + split_idx)
            current_segment = [copy.deepcopy(entries_sorted[0])]

            for i in range(1, real_length):
                prev = current_segment[-1]
                curr = entries_sorted[i]
                gap = curr["frame_idx"] - prev["frame_idx"] - 1

                if gap == 0:
                    current_segment.append(copy.deepcopy(curr))

                elif 1 <= gap <= max_gap:
                    for g in range(1, gap + 1):
                        alpha = g / (gap + 1)
                        interp_frame = prev["frame_idx"] + g

                        icx  = prev["cx"] + alpha * (curr["cx"] - prev["cx"])
                        icy  = prev["cy"] + alpha * (curr["cy"] - prev["cy"])
                        ix1  = prev["bbox"][0] + alpha * (curr["bbox"][0] - prev["bbox"][0])
                        iy1  = prev["bbox"][1] + alpha * (curr["bbox"][1] - prev["bbox"][1])
                        ix2  = prev["bbox"][2] + alpha * (curr["bbox"][2] - prev["bbox"][2])
                        iy2  = prev["bbox"][3] + alpha * (curr["bbox"][3] - prev["bbox"][3])

                        prev_cx = current_segment[-1]["cx"]
                        prev_cy = current_segment[-1]["cy"]
                        vel = float(np.sqrt((icx - prev_cx) ** 2 + (icy - prev_cy) ** 2))

                        seg_real_length = original_length
                        interp_entry = {
                            "frame_idx":    interp_frame,
                            "bbox":         [ix1, iy1, ix2, iy2],
                            "conf":         -1.0,  # sentinel: synthetic entry
                            "cx":           icx,
                            "cy":           icy,
                            "velocity_norm": vel,
                            "interpolated": True,
                            # track_length = track_length_real for this sub-track segment
                            "track_length": seg_real_length,
                        }
                        current_segment.append(interp_entry)
                        total_interpolated += 1

                    current_segment.append(copy.deepcopy(curr))

                else:
                    # Gap > max_gap — SPLIT here (hallucinated velocity protection)
                    new_tracks_dict[current_key] = current_segment
                    split_idx += 1
                    current_key = str(tid_int * SPLIT_TRACK_ID_MULTIPLIER + split_idx)
                    original_length = curr.get("track_length", 1)
                    current_segment = [copy.deepcopy(curr)]

            new_tracks_dict[current_key] = current_segment

    result["tracks"] = new_tracks_dict
    result["interpolation_applied"]      = True
    result["total_frames_interpolated"]  = total_interpolated
    result["max_interp_gap"]             = max_gap

    logger.info(
        f"Interpolation complete: {total_interpolated} frames added across "
        f"{len(tracks_dict)} input tracks → {len(new_tracks_dict)} output tracks."
    )
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_post_processing(
    trackers: list,
    test_seqs: list,
    day7_layout: str,
    schema_entry_fields: set,
    min_track_length: int = MIN_TRACK_LENGTH,
    max_interp_gap: int = MAX_INTERP_GAP,
) -> tuple:
    """
    Runs fragment filter then interpolation for all trackers × all sequences.

    Validates schema before processing each file (calls validate_trajectory_schema
    imported from src.error_propagation — lazy import to avoid circular import at
    module level since error_propagation imports get_final_path from this module).

    Writes _filtered.json (filter only) and _final.json (filter + interpolation).
    Returns (fragment_report, interpolation_report) dicts.

    NOTE: Call apply_fragment_filter() and apply_interpolation() directly — they
    are defined in this same module. Do NOT self-import.
    """
    # Lazy import to avoid module-level circular dependency
    from src.error_propagation import validate_trajectory_schema

    assert len(test_seqs) >= 1, "test_seqs must not be empty"

    fragment_report     = {}
    interpolation_report = {}

    for tracker in trackers:
        fragment_report[tracker]      = {}
        interpolation_report[tracker] = {}

        for seq in test_seqs:
            # --- Load raw trajectory ---
            raw_p = get_raw_path(tracker, seq, day7_layout)
            with open(raw_p) as f:
                raw = json.load(f)

            # --- Schema validation (Issue 2 fix) ---
            validate_trajectory_schema(raw, schema_entry_fields, raw_p)

            # --- Step 1: Fragment filter ---
            # Fragment filter applied BEFORE interpolation — avoids interpolating
            # tracks that would later be discarded, saving compute and preventing
            # ghost interpolation artefacts.
            filtered = apply_fragment_filter(raw, tracker_name=tracker,
                                              min_length=min_track_length)

            filt_p = get_filtered_path(tracker, seq, day7_layout)
            os.makedirs(os.path.dirname(filt_p) or ".", exist_ok=True)
            with open(filt_p, "w") as f:
                json.dump(filtered, f, indent=2)

            raw_track_count  = len(raw["tracks"])
            filt_track_count = len(filtered["tracks"])

            fragment_report[tracker][seq] = {
                "tracks_before": raw_track_count,
                "tracks_after":  filt_track_count,
                "discarded":     filtered["tracks_discarded_by_filter"],
            }

            # --- Step 2: Interpolation ---
            # Operates on the filtered output, not on raw.
            # Interpolation performed in image-space units (640×640 normalised).
            final = apply_interpolation(filtered, max_gap=max_interp_gap)

            final_p = get_final_path(tracker, seq, day7_layout)
            os.makedirs(os.path.dirname(final_p) or ".", exist_ok=True)
            with open(final_p, "w") as f:
                json.dump(final, f, indent=2)

            interpolation_report[tracker][seq] = {
                "frames_interpolated": final["total_frames_interpolated"],
            }

            logger.info(
                f"{tracker}/{seq}: {raw_track_count} raw tracks "
                f"→ {filt_track_count} after filter "
                f"({filtered['tracks_discarded_by_filter']} discarded), "
                f"{final['total_frames_interpolated']} frames interpolated. "
                f"_filtered: {filt_p} | _final: {final_p}"
            )

    logger.info(
        "run_post_processing complete. "
        f"Trackers: {trackers}. Sequences: {test_seqs}. Layout: {day7_layout}."
    )
    return fragment_report, interpolation_report
