"""
Split utility — Day 7.
Replaces all clips_master.txt usage for sequence loading.
Sequences from frozen split_manifest.json only.

Usage:
    from split_utils import load_split_sequences, verify_split_integrity
    test_seqs = load_split_sequences('test')
"""

import json
import os

DEFAULT_MANIFEST = 'data/ua_detrac/split_manifest.json'


def load_split_sequences(split_name, manifest_path=DEFAULT_MANIFEST):
    """
    Loads sequence list for a given split from the frozen manifest.

    Parameters
    ----------
    split_name    : 'train' | 'val' | 'test'
    manifest_path : path to split_manifest.json

    Returns
    -------
    list of sequence ID strings
    """
    assert split_name in ('train', 'val', 'test'), (
        f"split_name must be train/val/test, got: {split_name}"
    )

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}\n"
            f"Run freeze_split_manifest.py before any Block B/C/D code."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert 'splits' in manifest, "Manifest missing top-level key: 'splits'"
    assert split_name in manifest['splits'], (
        f"Manifest missing split '{split_name}'. "
        f"Available: {list(manifest['splits'].keys())}"
    )

    seqs = manifest['splits'][split_name]
    assert len(seqs) > 0, f"Split '{split_name}' is empty in manifest."
    return seqs


def verify_split_integrity(manifest_path=DEFAULT_MANIFEST):
    """
    Verifies the frozen manifest is internally consistent.

    Checks:
      - File exists
      - Required top-level keys present
      - All three splits present
      - No overlap between splits
      - Total count matches sum of splits
      - Official training set sequences only (MVI_20xxx, MVI_40xxx)
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}\n"
            f"Run freeze_split_manifest.py before any Block B/C/D code."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    required_keys = ['splits', 'total_sequences', 'seed', 'frozen_day']
    for key in required_keys:
        assert key in manifest, f"Manifest missing top-level key: '{key}'"

    for split_name in ['train', 'val', 'test']:
        assert split_name in manifest['splits'], (
            f"Manifest missing split: '{split_name}'"
        )

    train = set(manifest['splits']['train'])
    val   = set(manifest['splits']['val'])
    test  = set(manifest['splits']['test'])

    # No overlap
    assert not (train & val),  f"Overlap between train and val: {train & val}"
    assert not (train & test), f"Overlap between train and test: {train & test}"
    assert not (val & test),   f"Overlap between val and test: {val & test}"

    # Total matches
    total = manifest['total_sequences']
    sum_splits = len(train) + len(val) + len(test)
    assert total == sum_splits, (
        f"Total mismatch: manifest says {total}, splits sum to {sum_splits}"
    )

    # Official training-set range check (Fix F08)
    all_seqs = train | val | test
    non_train = []
    for s in all_seqs:
        try:
            num = int(s.replace('MVI_', '').replace('_', ''))
            if not (20000 <= num < 21000 or 40000 <= num < 41000):
                non_train.append(s)
        except ValueError:
            non_train.append(s)

    if non_train:
        raise AssertionError(
            f"Non-training-set sequences found in manifest: {non_train}\n"
            f"Fix F08: only official UA-DETRAC training sequences allowed."
        )

    print("[split_utils] Manifest integrity: OK")
    print(f"  Train={len(train)} | Val={len(val)} | Test={len(test)} | Total={sum_splits}")
    print(f"  Seed={manifest['seed']} | Frozen Day={manifest['frozen_day']}")
    return True