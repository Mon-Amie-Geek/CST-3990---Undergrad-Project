"""
Smoke tests for Day 7 — Block B SORT.

Tests:
  T1 : Empty detection frame
  T2 : One-frame sequence
  T3 : Sequence with missing metadata
  T4 : Sequence with no GT matches
  T5 : Cache base mismatch detection
  T6 : Split manifest integrity
  T7 : Detector selection assertion
  T8 : velocity_norm null on first entry, float on second
"""

import numpy as np
import json, os, yaml, tempfile

from sort_tracker import SORTTracker

# Creats a tracker using parameters from config_BlockB.yaml
def _make_tracker(cfg_b, conf_thresh):
    t = SORTTracker(
        max_age       = cfg_b['sort']['max_age'],
        min_hits      = cfg_b['sort']['min_hits'],
        iou_threshold = cfg_b['sort']['iou_threshold'],
        conf_thresh   = conf_thresh
    )
    # Clears previous tracks - Important for multiple sequences
    t.reset()
    return t


# T1: Empty detection frame
# Checks handler frames with no detections
def test_empty_detection_frame(cfg_b, conf_thresh):
    """SORT must handle frames with zero detections without crashing."""
    tracker = _make_tracker(cfg_b, conf_thresh)
    empty   = np.empty((0, 5))
    result  = tracker.update(empty, frame_idx=0)
    assert result.shape == (0, 6), \
        f"T1 FAIL: expected (0,6) on empty frame, got {result.shape}"
    # Feed a real detection then an empty frame
    det = np.array([[100, 200, 200, 300, 0.9]])
    tracker.update(det, frame_idx=1)
    result2 = tracker.update(empty, frame_idx=2)
    # Track survives (max_age=5), but time_since_update=1 so not emitted
    assert isinstance(result2, np.ndarray), "T1 FAIL: result is not ndarray"
    print("T1 PASS: empty detection frame")


#T2: One-frame sequence
# Checks tracker doesn't crash on extremely short sequence
def test_one_frame_sequence(cfg_b, conf_thresh):
    """SORT must not crash on a single-frame input."""
    tracker = _make_tracker(cfg_b, conf_thresh)
    det     = np.array([[50, 50, 150, 150, 0.85]])
    result  = tracker.update(det, frame_idx=0)
    # min_hits=3 means no confirmed output on frame 1
    assert result.shape[1] == 6 or result.shape == (0, 6), \
        f"T2 FAIL: unexpected shape {result.shape}"
    print("T2 PASS: one-frame sequence")


# T3: Missing metadata
# Checks metadata loader raises error when metadata missing
def test_missing_metadata():
    """load_metadata must raise FileNotFoundError for unknown seq_id."""
    # Import here to avoid circular at module level
    import importlib, sys
    # Temporarily import the runner module's load_metadata
    sys.path.insert(0, 'src')
    from block_b_sort_runner import load_metadata
    try:
        load_metadata('MVI_99999_FAKE')
        assert False, "T3 FAIL: should have raised FileNotFoundError"
    except FileNotFoundError:
        print("T3 PASS: missing metadata raises FileNotFoundError")


# T4: No GT matches 
def test_no_gt_matches():
    """
    motmetrics accumulator must handle frames where pred exists but GT is empty,
    and frames where GT exists but pred is empty.
    """
    import motmetrics as mm
    # Creates evaluation accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Frame with GT but no predictions
    acc.update([1, 2], [], np.full((2, 0), np.nan))
    # Frame with predictions but no GT
    acc.update([], [10, 11], np.full((0, 2), np.nan))
    # Frame with both empty
    acc.update([], [], np.empty((0, 0)))

    mh      = mm.metrics.create()
    # Computes metrics
    summary = mh.compute(acc, metrics=['mota', 'num_switches'], name='test')
    # MOTA may be negative or nan — just must not crash
    assert 'mota' in summary.columns, "T4 FAIL: mota missing from summary"
    print(f"T4 PASS: no GT matches — MOTA={summary['mota'].iloc[0]:.3f} (may be negative/nan)")


# T5: Cache base mismatch detection
def test_cache_base_mismatch(cfg_b):
    """
    Detect if cache first frame_idx contradicts cache_frame_index_base in config.
    """
    from cache_utils import load_cache
    from split_utils import load_split_sequences
    # Loads sequences from fozen manifest
    test_seqs = load_split_sequences(cfg_b['split'], cfg_b['split_manifest'])
    seq_id    = test_seqs[0]

    _, frames = load_cache(seq_id, cfg_b['split'],
                           cfg_b['detector'], cfg_b['detections_source'])
    # Reads first cached frame index
    first_idx = frames[0]['frame_idx']
    # Reads expected base from config
    declared_base = cfg_b['cache_frame_index_base']

    # Detect mismatch
    if first_idx != declared_base:
        raise AssertionError(
            f"T5 FAIL: cache first frame_idx={first_idx} but "
            f"config_blockB says cache_frame_index_base={declared_base}.\n"
            f"Fix config_blockB.yaml before running the full pipeline.")
    print(f"T5 PASS: cache base confirmed — first_idx={first_idx} matches config base={declared_base}")


# T6: Split manifest integrity
def test_split_manifest_integrity(cfg_b):
    """Delegates to split_utils.verify_split_integrity."""
    from split_utils import verify_split_integrity
    # No overlapping sequences, correct counts, valid IDs
    verify_split_integrity(cfg_b['split_manifest'])
    print("T6 PASS: split manifest integrity verified, lambda: test_split_manifest_integrity(cfg_b)")


# T7: Detector selection assertion
# detector in config_BlockB.yaml matches Block A selected detector
def test_detector_selection_assertion(cfg_b):
    """
    Asserts that the detector in config_blockB matches selected==True
    in block_a_results.csv. Stricter than just checking cache directory.
    """
    import pandas as pd
    results_path = 'logs/block_a/block_a_results.csv'
    if not os.path.exists(results_path):
        print("T7 SKIP: block_a_results.csv not found (run Block A first)")
        return
    df_a = pd.read_csv(results_path)
    # Finds selected detector
    selected_rows = df_a[df_a['selected'] == True]
    assert len(selected_rows) == 1, \
        f"T7 FAIL: expected 1 selected detector in block_a_results.csv, got {len(selected_rows)}"
    selected_detector = selected_rows.iloc[0]['detector']
    # Ensures config consistency
    assert cfg_b['detector'] == selected_detector, \
        (f"T7 FAIL: config_blockB detector='{cfg_b['detector']}' does not match "
         f"selected detector='{selected_detector}' from block_a_results.csv.\n"
         f"Update config_blockB.yaml before proceeding.")
    print(f"T7 PASS: detector '{cfg_b['detector']}' matches block_a_results.csv selected==True")


# T8: velocity_norm null on first, float on second 
def test_velocity_norm_contract():
    """
    Verifies velocity_norm is None for first track entry and float for second.
    Tests compute_velocity_norm directly.
    """
    # Imports velocity function
    from block_b_sort_runner import compute_velocity_norm

    # First entry — no previous centroid
    v1 = compute_velocity_norm(None, None, 100.0, 200.0, fps=25.0, frame_diagonal=1200.0)
    assert v1 is None, f"T8 FAIL: expected None for first entry, got {v1}"

    # Second entry — has previous centroid
    v2 = compute_velocity_norm(90.0, 190.0, 100.0, 200.0, fps=25.0, frame_diagonal=1200.0)
    assert isinstance(v2, float), f"T8 FAIL: expected float for second entry, got {type(v2)}"
    assert v2 >= 0, f"T8 FAIL: velocity_norm must be non-negative, got {v2}"

    # Zero movement — velocity should be 0.0
    v3 = compute_velocity_norm(100.0, 200.0, 100.0, 200.0, fps=25.0, frame_diagonal=1200.0)
    assert v3 == 0.0, f"T8 FAIL: zero movement should give 0.0, got {v3}"

    print(f"T8 PASS: velocity_norm contract — null={v1}, float={v2:.6f}, zero={v3}")


# Runner 
def run_all_smoke_tests(cfg_b, conf_thresh):
    print(f"\n{'='*60}")
    print("DAY 7 SMOKE TESTS")
    print(f"{'='*60}")
    passed = 0
    failed = 0
    tests  = [
        ("T1 Empty detection frame",    lambda: test_empty_detection_frame(cfg_b, conf_thresh)),
        ("T2 One-frame sequence",       lambda: test_one_frame_sequence(cfg_b, conf_thresh)),
        ("T3 Missing metadata",         test_missing_metadata),
        ("T4 No GT matches",            test_no_gt_matches),
        ("T5 Cache base mismatch",      lambda: test_cache_base_mismatch(cfg_b)),
        ("T6 Split manifest integrity", lambda: test_split_manifest_integrity(cfg_b)),
        ("T7 Detector selection",       lambda: test_detector_selection_assertion(cfg_b)),
        ("T8 velocity_norm contract",   test_velocity_norm_contract),
    ]
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  {name}: FAIL — {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Smoke tests: {passed} passed, {failed} failed")
    if failed > 0:
        raise RuntimeError(
            f"{failed} smoke test(s) failed. Fix all failures before running "
            f"the full sequence loop.")
    print("All smoke tests passed. Safe to run full pipeline.")
    print(f"{'='*60}\n")