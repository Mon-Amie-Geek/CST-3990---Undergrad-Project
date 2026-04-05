"""
tracker_selector.py — Block B → Block C bridge
CST3990 | MANJOO Ameera Najla | M01014463

Tracker Selection (Fix F01 resolution).

Reads logs/block_b_results.json, selects the best tracker by:
  Primary criterion:  highest idf1_mean  (identity consistency —
                      directly determines feature quality in Block C)
  Tiebreaker 1:       lowest idsw_total
  Tiebreaker 2:       highest fps_median (prefer faster if quality equal)

Freezes the selection into configs/config_blockC.yaml stub.
Selection is always data-driven — NEVER hardcoded.
"""

import json
import os
import logging

import yaml

from src.seed_control import set_all_seeds
set_all_seeds()  # MUST be at module top

logger = logging.getLogger(__name__)

RESULTS_PATH      = "logs/block_b_results.json"
CONFIG_C_PATH     = "configs/config_blockC.yaml"
SELECTION_REPORT_PATH = "logs/block_b/tracker_selection_report.json"


def select_tracker(
    results_path: str = RESULTS_PATH,
    config_c_path: str = CONFIG_C_PATH,
    report_path: str = SELECTION_REPORT_PATH,
) -> str:
    """
    Selects the best tracker from block_b_results.json.

    Steps:
      1. Rank all trackers by (-idf1_mean, +idsw_total, -fps_median).
      2. Update block_b_results.json: exactly one tracker gets "selected": true.
      3. Write configs/config_blockC.yaml stub.
      4. Write logs/block_b/tracker_selection_report.json.

    Returns the selected tracker name (str).
    """
    with open(results_path) as f:
        results = json.load(f)

    trackers = results["trackers"]

    # Data-driven ranking — Fix F01: never hardcode the winner
    ranked = sorted(
        trackers,
        key=lambda t: (
            -t["idf1_mean"],   # primary: highest IDF1 (identity consistency)
            t["idsw_total"],   # tiebreaker 1: lowest IDSW
            -t["fps_median"],  # tiebreaker 2: highest FPS
        ),
    )
    winner      = ranked[0]
    winner_name = winner["tracker"]

    # Update selected flags — exactly one tracker selected
    for t in trackers:
        t["selected"] = (t["tracker"] == winner_name)

    selection_criterion = (
        "Primary: highest idf1_mean (identity consistency — critical for Block C "
        "feature quality). Tiebreaker 1: lowest idsw_total. "
        "Tiebreaker 2: highest fps_median."
    )
    selection_rationale = (
        f"Selected '{winner_name}': idf1_mean={winner['idf1_mean']:.4f}, "
        f"idsw_total={winner['idsw_total']}, fps_median={winner['fps_median']:.2f}. "
        "Tracker frozen for Blocks C and D."
    )

    results["selection_criterion"] = selection_criterion
    results["selection_rationale"]  = selection_rationale

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Tracker selected: {winner_name}")

    # -----------------------------------------------------------------------
    # Write configs/config_blockC.yaml stub
    # -----------------------------------------------------------------------
    # trajectory_base + trajectory_suffix replace the ambiguous trajectory_source
    # (Issue 4 fix). Block C must detect DAY7_LAYOUT at its own runtime and build
    # full paths as:
    #   variant_a: logs/block_b/{tracker}/trajectories_{seq}_final.json
    #   variant_b: logs/block_b/trajectories/{seq}_{tracker}_final.json
    config_c = {
        "detector":    "yolov8n",
        "tracker":     winner_name,

        # Path components — Block C resolves full paths using DAY7_LAYOUT at runtime
        "trajectory_base":   "logs/block_b/",
        "trajectory_suffix": "_final.json",
        # trajectory_stage is a semantic alias for trajectory_suffix (Improvement 4).
        # Block C code can use either field — both present to avoid suffix-string coupling.
        "trajectory_stage":  "final",

        "warmup_frames":    10,
        "seed":             42,
        "metadata_source":  "logs/metadata.json",
        "split_seqs_key":   "split_seqs",

        # fps_source — never hardcode FPS in Block C; always read from metadata.json
        "fps_source":       "metadata.json",

        # schema_version locks trajectory schema evolution (Minor Gap 3).
        # Increment to v2 if trajectory_schema.json changes before running Block C.
        "schema_version":   "v1",

        # Explicit schema reference for Block C feature validation (micro-improvement 1)
        "feature_schema":            "configs/trajectory_schema.json",
        "feature_trajectory_type":   "_final.json",

        "_note_1": "Full population deferred to Day 10 — Block C feature extraction",
        "_note_2": "Tracker frozen here — do not change after Day 9",
        "_note_3": "Fragment filter (min=15) and interpolation (max_gap=3) applied",
        "_note_4": "Detector frozen: YOLOv8n — source: logs/block_a/block_a_results.csv",
        "_note_5": "fps_source=metadata.json — never hardcode FPS in Block C",
    }

    os.makedirs(os.path.dirname(config_c_path), exist_ok=True)
    with open(config_c_path, "w") as f:
        yaml.dump(config_c, f, default_flow_style=False, sort_keys=False)
    logger.info(f"config_blockC.yaml stub written: tracker={winner_name}")

    # -----------------------------------------------------------------------
    # Write tracker selection report
    # -----------------------------------------------------------------------
    report = {
        "selected_tracker": winner_name,
        "ranked_trackers": [
            {
                "rank":      i + 1,
                "tracker":   t["tracker"],
                "idf1_mean": t["idf1_mean"],
                "idsw_total": t["idsw_total"],
                "fps_median": t["fps_median"],
                "selected":  t["selected"],
            }
            for i, t in enumerate(ranked)
        ],
        "selection_criterion": selection_criterion,
        "rationale": selection_rationale,
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Tracker selection report saved to {report_path}")
    return winner_name
