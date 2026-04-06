"""
validate_block_b_finals.py
Lightweight check: confirms that _sort_final.json exists for every split sequence.

Run from the project root:
    python src/validate_block_b_finals.py
Or from a notebook cell:
    from src.validate_block_b_finals import check_sort_finals; check_sort_finals()
"""

import json
import os


def check_sort_finals(
    metadata_path: str = "logs/metadata.json",
    traj_dir: str = "logs/block_b/trajectories",
    tracker: str = "sort",
) -> bool:
    """
    Check that {SEQ}_{tracker}_final.json exists for every split sequence.

    Prints a summary and returns True when all files are present.
    """
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    all_seqs = (
        meta["split_seqs"]["train"]
        + meta["split_seqs"]["val"]
        + meta["split_seqs"]["test"]
    )

    missing = []
    present = []
    for seq in all_seqs:
        path = os.path.join(traj_dir, f"{seq}_{tracker}_final.json")
        if os.path.exists(path):
            present.append(seq)
        else:
            missing.append(seq)

    print(f"\n=== Block B _final.json validation ({tracker}) ===")
    print(f"Checked {len(all_seqs)} sequences from {metadata_path}")
    print(f"  Present  ({len(present)}): {present}")
    if missing:
        print(f"  MISSING  ({len(missing)}): {missing}")
        print("\nResult: FAIL — rerun Day 9 to generate missing trajectories.")
        return False
    else:
        print("\nResult: PASS — all 10 _final.json files present.")
        return True


if __name__ == "__main__":
    check_sort_finals()
