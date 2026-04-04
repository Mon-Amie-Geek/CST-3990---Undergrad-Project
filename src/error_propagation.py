"""
Error Propagation Analysis — Block B → Block C bridge
CST3990 | MANJOO Ameera Najla | M01014463

Fix F30 supplement: Pearson r(IDSW_per_sequence, speed_std_per_sequence)

STATUS: Scaffold committed Day 8.
EXECUTION: Day 9, after Block C speed features are computed.

This module is called by pipeline_controller.run_block_b_analysis()
once speed_std_per_sequence is available from Block C feature extraction.
"""

import numpy as np
from scipy.stats import pearsonr
import json
import logging

logger = logging.getLogger(__name__)


def compute_error_propagation_correlation(
    idsw_per_sequence: dict,
    speed_std_per_sequence: dict,
    output_path: str = "logs/block_b_error_propagation.json"
) -> dict:
    """
    Compute Pearson r between IDSW count and speed_std across sequences.

    Args:
        idsw_per_sequence: dict mapping seq_id to IDSW count, e.g. {"MVI_20062": 21}
        speed_std_per_sequence: dict mapping seq_id to speed standard deviation
        output_path: where to save the result JSON

    Returns:
        dict with pearson_r, p_value, n_sequences, interpretation, significant_at_0.05

    NOTE: Only call after Block C feature extraction is complete (Day 9+).
          Block C must supply speed_std_per_sequence before this runs.
    """
    common_seqs = sorted(
        set(idsw_per_sequence.keys()) & set(speed_std_per_sequence.keys())
    )

    if len(common_seqs) < 3:
        logger.warning(
            f"Only {len(common_seqs)} common sequences — "
            "Pearson r not reliable with fewer than 3 points."
        )
        return {"error": "insufficient_sequences", "n": len(common_seqs)}

    idsw_vals  = [idsw_per_sequence[s]       for s in common_seqs]
    speed_vals = [speed_std_per_sequence[s]  for s in common_seqs]

    r, p_val = pearsonr(idsw_vals, speed_vals)

    result = {
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p_val), 4),
        "n_sequences": len(common_seqs),
        "sequences_used": common_seqs,
        "interpretation": (
            "Strong positive correlation — ID switches propagate to speed noise"
            if r > 0.5 else
            "Weak or no correlation"
        ),
        "significant_at_0.05": bool(p_val < 0.05)
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        f"Error propagation: r={r:.3f}, p={p_val:.4f}, n={len(common_seqs)}"
    )
    return result


def placeholder_note():
    """
    Confirms the module is importable on Day 8.
    The actual compute_error_propagation_correlation() call is in Day 9
    after Block C speed features are available.
    """
    return "error_propagation.py loaded — execution deferred to Day 9"
