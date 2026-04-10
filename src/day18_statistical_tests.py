"""
day18_statistical_tests.py

Day 18 — Results Consolidation & Statistical Tests
Student: MANJOO Ameera Najla | M01014463
CST3990 Undergraduate Individual Project

Three statistical tests using actual per-clip FAR data from Day 17:

  Test 1 — Wilcoxon signed-rank:   rule_based vs ocsvm FAR per clip (n=10)
  Test 2 — McNemar (exact):        FAR-gate binary outcomes, rule_based vs IF (n=10)
  Test 3 — Pearson r:              pct_clipped vs far_value across all 30 pairs
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

# ── Repo-root resolution (works when run from any cwd) ────────────────────────
_HERE = Path(__file__).resolve().parent          # src/
_ROOT = _HERE.parent                             # repo root

LOGS_DIR   = _ROOT / "logs"
BLOCK_D_DIR = LOGS_DIR / "block_d"
OUT_DIR    = LOGS_DIR / "block_d"

FAR_THRESHOLD = 0.10   # loaded from config; duplicated here for convenience


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_far_per_clip() -> pd.DataFrame:
    """Load day17_far_per_clip.csv and validate shape."""
    path = BLOCK_D_DIR / "day17_far_per_clip.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run block_d_day17_far_latency.py first (Day 17)."
        )
    df = pd.read_csv(path)
    expected_cols = {"clip_filename", "method", "far_value", "pct_clipped"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"day17_far_per_clip.csv missing columns: {missing}")
    return df


def _pivot_far(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot FAR table so each clip is a row and methods are columns."""
    pivot = df.pivot_table(
        index="clip_filename", columns="method", values="far_value", aggfunc="first"
    )
    pivot.columns.name = None
    pivot = pivot.reset_index()
    return pivot


def _fmt(x: float, decimals: int = 4) -> str:
    return f"{x:.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Wilcoxon signed-rank: rule_based vs ocsvm FAR
# ─────────────────────────────────────────────────────────────────────────────

def run_wilcoxon_far_test(df: pd.DataFrame) -> dict[str, Any]:
    """
    Wilcoxon signed-rank test on per-clip FAR values.

    H₀ : median(FAR_rule_based - FAR_ocsvm) = 0
    H₁ : medians differ  (two-sided)

    Pairs: 10 normal clips, each evaluated by both methods.
    Adapted from original plan (mAP per sequence) because Block A has no
    per-sequence breakdown; FAR-per-clip is the richest paired dataset (n=10).
    """
    pivot = _pivot_far(df)
    required = {"rule_based", "ocsvm"}
    missing = required - set(pivot.columns)
    if missing:
        raise ValueError(f"Pivot missing method columns: {missing}")

    rb   = pivot["rule_based"].values
    ocsvm = pivot["ocsvm"].values
    clips = pivot["clip_filename"].tolist()

    diff = rb - ocsvm

    stat, p = wilcoxon(rb, ocsvm, alternative="two-sided", zero_method="wilcox")

    n = len(rb)
    median_rb   = float(np.median(rb))
    median_ocsvm = float(np.median(ocsvm))

    result = {
        "test":           "Wilcoxon signed-rank",
        "hypothesis":     "H0: median(FAR_rule_based - FAR_ocsvm) = 0",
        "alternative":    "two-sided",
        "n_pairs":        n,
        "clips":          clips,
        "far_rule_based": rb.tolist(),
        "far_ocsvm":      ocsvm.tolist(),
        "differences":    diff.tolist(),
        "median_rb":      median_rb,
        "median_ocsvm":   median_ocsvm,
        "statistic":      float(stat),
        "p_value":        float(p),
        "significant_05": bool(p < 0.05),
        "interpretation": (
            "Significant: rule_based and OC-SVM produce statistically different "
            "FAR distributions on normal clips (p < 0.05)."
            if p < 0.05 else
            "Not significant: no statistical difference in FAR distributions "
            "between rule_based and OC-SVM (p ≥ 0.05)."
        ),
        "note": (
            "Test adapted from original plan (Wilcoxon on per-sequence mAP_yolo vs "
            "mAP_mobilenet) because Block A provides only aggregate mAP, not "
            "per-sequence scores. Per-clip FAR (n=10) is the richest comparable "
            "paired dataset available."
        ),
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — McNemar (exact): FAR-gate pass/fail for rule_based vs IF
# ─────────────────────────────────────────────────────────────────────────────

def run_mcnemar_far_gate_test(
    df: pd.DataFrame,
    threshold: float = FAR_THRESHOLD,
) -> dict[str, Any]:
    """
    McNemar exact test on per-clip FAR-gate binary outcomes.

    Gate: clip fails if FAR >= threshold (0.10).
    Methods compared: rule_based vs isolation_forest (n=10 clips).

    Contingency table:
        rows    = isolation_forest outcome (0=pass, 1=fail)
        columns = rule_based outcome       (0=pass, 1=fail)

    Adapted from original plan (McNemar on event-level predictions) because
    all 10 events are detected by all methods (event_pred=1 always), making
    event-level McNemar degenerate (all discordant pairs = 0).
    """
    pivot = _pivot_far(df)
    required = {"rule_based", "isolation_forest"}
    missing = required - set(pivot.columns)
    if missing:
        raise ValueError(f"Pivot missing method columns: {missing}")

    rb_far = pivot["rule_based"].values
    if_far = pivot["isolation_forest"].values
    clips  = pivot["clip_filename"].tolist()

    # Binary: 1 = gate FAIL (FAR >= threshold)
    rb_fail = (rb_far >= threshold).astype(int)
    if_fail = (if_far >= threshold).astype(int)

    # Contingency table [IF_outcome][RB_outcome]
    # table[i][j]: i=IF label (0=pass,1=fail), j=RB label (0=pass,1=fail)
    table = np.zeros((2, 2), dtype=int)
    for i_f, r_f in zip(if_fail, rb_fail):
        table[i_f, r_f] += 1

    # statsmodels mcnemar expects [[a,b],[c,d]] where a=both match outcome-0
    # exact=True required for small n (n=10)
    result_mc = mcnemar(table, exact=True, correction=False)

    b = int(table[1, 0])   # IF fails, RB passes
    c = int(table[0, 1])   # IF passes, RB fails

    per_clip_detail = [
        {
            "clip": cl,
            "rb_far":    float(rb_far[i]),
            "if_far":    float(if_far[i]),
            "rb_fail":   bool(rb_fail[i]),
            "if_fail":   bool(if_fail[i]),
        }
        for i, cl in enumerate(clips)
    ]

    result = {
        "test":            "McNemar (exact)",
        "hypothesis":      "H0: P(IF-fail, RB-pass) = P(IF-pass, RB-fail)",
        "alternative":     "two-sided",
        "threshold":       threshold,
        "n_clips":         len(clips),
        "contingency_table": {
            "layout":   "rows=IF_outcome, cols=RB_outcome (0=pass, 1=fail)",
            "both_pass_a":   int(table[0, 0]),
            "if_fail_rb_pass_b": b,
            "if_pass_rb_fail_c": c,
            "both_fail_d":   int(table[1, 1]),
        },
        "discordant_b":    b,
        "discordant_c":    c,
        "statistic":       float(result_mc.statistic) if result_mc.statistic is not None else None,
        "p_value":         float(result_mc.pvalue),
        "significant_05":  bool(result_mc.pvalue < 0.05),
        "per_clip_detail": per_clip_detail,
        "interpretation": (
            "Significant: the FAR-gate outcomes of rule_based and Isolation Forest "
            "differ significantly across normal clips (p < 0.05). Isolation Forest "
            "fails the gate on clips where rule_based passes, confirming its high FAR."
            if result_mc.pvalue < 0.05 else
            "Not significant: no statistical difference in FAR-gate outcomes "
            "between rule_based and Isolation Forest (p ≥ 0.05)."
        ),
        "note": (
            "Test adapted from original plan (McNemar on per-event F1_rulebased vs "
            "F1_ocsvm predictions) because all 10 anomaly events are detected by all "
            "methods (event_pred=1 for every event and method), making event-level "
            "McNemar degenerate (zero discordant pairs). Per-clip FAR gate (n=10) "
            "provides meaningful discordance. Exact test used per recommendation for n<25."
        ),
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Pearson r: pct_clipped vs far_value (n=30)
# ─────────────────────────────────────────────────────────────────────────────

def run_pearson_domain_shift_test(df: pd.DataFrame) -> dict[str, Any]:
    """
    Pearson correlation between domain-shift indicator and false alarm rate.

    pct_clipped: fraction of F2 feature values outside the UA-DETRAC [0,1]
                 MinMaxScaler range (domain shift proxy).
    far_value:   false alarm rate for that clip-method pair.

    n=30 (10 clips × 3 methods).

    H₀: ρ(pct_clipped, far_value) = 0
    H₁: ρ ≠ 0  (two-sided)

    Tests whether clips with more out-of-distribution speed features also
    produce more false alarms — a key threat-to-validity hypothesis.
    """
    pc  = df["pct_clipped"].values.astype(float)
    fv  = df["far_value"].values.astype(float)
    n   = len(df)

    r, p = pearsonr(pc, fv)

    result = {
        "test":           "Pearson r",
        "hypothesis":     "H0: rho(pct_clipped, far_value) = 0",
        "alternative":    "two-sided",
        "n_pairs":        n,
        "r":              float(r),
        "p_value":        float(p),
        "r_squared":      float(r**2),
        "significant_05": bool(p < 0.05),
        "interpretation": (
            f"Significant positive correlation (r={r:.4f}, p={p:.4f}): clips with "
            "greater domain shift (more out-of-distribution speed features) produce "
            "significantly higher false alarm rates."
            if (p < 0.05 and r > 0) else
            f"Significant negative correlation (r={r:.4f}, p={p:.4f}): "
            "clips with greater domain shift produce lower false alarm rates (unexpected)."
            if (p < 0.05 and r < 0) else
            f"Not significant (r={r:.4f}, p={p:.4f}): no linear relationship "
            "between domain shift and false alarm rate."
        ),
        "note": (
            "Test adapted from original plan (Pearson on per-sequence IDSW vs speed_std) "
            "because Block B has only 2 test sequences (IDSW data for MVI_20062 and "
            "MVI_20063 only), giving n=2 which is statistically useless. "
            "pct_clipped→far_value (n=30) tests the same domain-shift → error-rate "
            "hypothesis at adequate sample size."
        ),
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Results consolidation helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_complete_results_tables() -> dict[str, Any]:
    """
    Pull best results from all four blocks into a single consolidated dict.
    Reads from log CSVs/JSONs that exist on disk.
    Returns a dict suitable for JSON serialisation.
    """
    tables: dict[str, Any] = {}

    # ── Block A ──────────────────────────────────────────────────────────────
    ba_path = LOGS_DIR / "block_a" / "block_a_results.csv"
    if ba_path.exists():
        ba = pd.read_csv(ba_path)
        tables["block_a"] = {
            "frozen_detector": "yolov8n",
            "selection_criterion": "mAP@0.5 delta > 0.05 threshold",
            "detectors": ba.to_dict(orient="records"),
        }
    else:
        tables["block_a"] = {"error": f"Not found: {ba_path}"}

    # ── Block B ──────────────────────────────────────────────────────────────
    bb_path = LOGS_DIR / "block_b" / "tracker_selection_report.json"
    if bb_path.exists():
        with open(bb_path, encoding="utf-8") as fh:
            bb = json.load(fh)
        tables["block_b"] = {
            "frozen_tracker": bb.get("selected_tracker", "sort"),
            "selection_criterion": bb.get("selection_criterion", ""),
            "trackers": bb.get("ranked_trackers", []),
        }
    else:
        tables["block_b"] = {"error": f"Not found: {bb_path}"}

    # ── Block C ──────────────────────────────────────────────────────────────
    bc_path = LOGS_DIR / "block_c" / "block_c_auroc_results.csv"
    if bc_path.exists():
        bc = pd.read_csv(bc_path)
        best = bc.loc[bc["auroc"].idxmax()]
        tables["block_c"] = {
            "frozen_feature_set": str(best["feature_set"]),
            "selection_criterion": "highest AUROC on OC-SVM validation split",
            "feature_sets": bc[["feature_set", "best_nu", "auroc", "silhouette"]].to_dict(orient="records"),
        }
    else:
        tables["block_c"] = {"error": f"Not found: {bc_path}"}

    # ── Block D ──────────────────────────────────────────────────────────────
    bd_path = BLOCK_D_DIR / "block_d_complete_results.csv"
    if bd_path.exists():
        bd = pd.read_csv(bd_path)
        tables["block_d"] = {
            "selection_criterion": "lowest mean_FAR with gate_passed=True",
            "methods": bd.to_dict(orient="records"),
        }
    else:
        tables["block_d"] = {"error": f"Not found: {bd_path}"}

    # ── Latency ──────────────────────────────────────────────────────────────
    lat_path = BLOCK_D_DIR / "day17_latency_profile.json"
    if lat_path.exists():
        with open(lat_path, encoding="utf-8") as fh:
            lat = json.load(fh)
        tables["latency"] = {
            "end_to_end_fps":   lat.get("end_to_end_fps"),
            "total_mean_ms":    lat["stages"]["total"]["mean_ms"],
            "total_p50_ms":     lat["stages"]["total"]["p50_ms"],
            "total_p95_ms":     lat["stages"]["total"]["p95_ms"],
            "anomaly_mean_ms":  lat["stages"]["anomaly"]["mean_ms"],
            "bottleneck_stage": "anomaly",
            "hardware":         lat.get("hardware", {}),
            "caveats":          lat.get("measurement_caveats", ""),
        }
    else:
        tables["latency"] = {"error": f"Not found: {lat_path}"}

    return tables


def document_failure_cases() -> list[dict[str, Any]]:
    """Return a structured list of documented failure cases for the project."""
    return [
        {
            "id":          "F1",
            "block":       "A",
            "description": "SSD300 mAP@0.5 = 0.316 — far below the YOLOv8n fine-tuned "
                           "result of 0.6674. SSD trained on COCO without domain adaptation.",
            "impact":      "Selection criterion triggered: mAP delta = 0.3514 > 0.05 "
                           "threshold → YOLOv8n frozen.",
            "resolution":  "YOLOv8n selected and frozen for Blocks B–D.",
        },
        {
            "id":          "F2",
            "block":       "B",
            "description": "IDF1 scores are low (SORT: 0.0412) because UA-DETRAC "
                           "sequences have dense occlusion. DeepSORT/ByteTrack show "
                           "near-zero IDF1 in this configuration.",
            "impact":      "SORT selected by default as the only tracker with non-trivial "
                           "IDF1. Low IDF1 implies track fragmentation affects feature quality.",
            "resolution":  "SORT frozen. Feature robustness to fragmentation partially "
                           "mitigated by MIN_TRACK_LENGTH=15 filter in Block C/D.",
        },
        {
            "id":          "F3",
            "block":       "C",
            "description": "F1_only (density/flow features) achieves AUROC = 0.4724, "
                           "worse than chance. Vehicle count and ROI occupancy are poor "
                           "discriminators for the anomaly proxy used.",
            "impact":      "F1_only disqualified. F2_only (speed features) achieves "
                           "AUROC = 0.9935.",
            "resolution":  "F2_only frozen: vel_px_sec, vel_px_sec_smooth.",
        },
        {
            "id":          "F14",
            "block":       "D",
            "description": "Isolation Forest FAR gate FAILED: mean_FAR = 0.3354 ≥ "
                           "threshold = 0.10. 6 of 10 normal clips exceed the FAR threshold.",
            "impact":      "Isolation Forest cannot be recommended for deployment. "
                           "Results are reportable but annotated per Fix F14 policy "
                           "(non-blocking for research reporting).",
            "resolution":  "Results reported with annotation. OC-SVM (mean_FAR=0.0283) "
                           "and rule_based (mean_FAR=0.0350) both pass the gate.",
        },
        {
            "id":          "F29",
            "block":       "D (Latency)",
            "description": "Latency measured in Colab-like offline batch mode, not "
                           "real-time streaming. Timings are hardware/backend-specific "
                           "indicative values only.",
            "impact":      "End-to-end FPS=21.11 may not reflect deployment on "
                           "different hardware or with real-time video streams.",
            "resolution":  "Caveats documented in day17_latency_profile.json and "
                           "reported in results with explicit scope limitation.",
        },
        {
            "id":          "STAT-1",
            "block":       "Day 18 (Statistical Tests)",
            "description": "Wilcoxon test adapted: original plan specified per-sequence "
                           "mAP_yolo vs mAP_mobilenet. Block A provides only 2 aggregate "
                           "detector results, not per-sequence breakdowns.",
            "impact":      "Wilcoxon ran on per-clip FAR rule_based vs ocsvm (n=10) "
                           "instead. Different hypothesis tested.",
            "resolution":  "Adaptation justified: FAR-per-clip is the richest paired "
                           "dataset (n=10) and directly relevant to Block D evaluation.",
        },
        {
            "id":          "STAT-2",
            "block":       "Day 18 (Statistical Tests)",
            "description": "McNemar adapted: original plan specified per-event F1 "
                           "predictions (rule_based vs ocsvm). All 10 anomaly events "
                           "are detected by all methods (event_pred=1 always), giving "
                           "zero discordant pairs — McNemar is degenerate.",
            "impact":      "McNemar ran on per-clip FAR-gate binary outcomes "
                           "(rule_based vs isolation_forest), n=10.",
            "resolution":  "6 discordant pairs (IF fails gate, RB passes) — valid test.",
        },
        {
            "id":          "STAT-3",
            "block":       "Day 18 (Statistical Tests)",
            "description": "Pearson r adapted: original plan specified per-sequence "
                           "IDSW vs speed_std. Block B has IDSW for only 2 test sequences "
                           "(n=2 is statistically useless).",
            "impact":      "Pearson ran on pct_clipped vs far_value across all 30 "
                           "method-clip pairs. Same domain-shift→error hypothesis.",
            "resolution":  "n=30 is adequate for Pearson. Tests threat-to-validity "
                           "directly (domain shift → elevated FAR).",
        },
    ]


def generate_day18_report(
    wilcoxon_result: dict,
    mcnemar_result: dict,
    pearson_result: dict,
    tables: dict,
    failures: list,
) -> dict[str, Any]:
    """Assemble the final Day 18 JSON report."""
    report = {
        "day":     18,
        "title":   "Results Consolidation & Statistical Tests",
        "student": "MANJOO Ameera Najla | M01014463",
        "project": "CST3990 Undergraduate Individual Project",
        "statistical_tests": {
            "test1_wilcoxon":  wilcoxon_result,
            "test2_mcnemar":   mcnemar_result,
            "test3_pearson_r": pearson_result,
        },
        "consolidated_results": tables,
        "failure_cases":        failures,
        "threats_to_validity": [
            {
                "threat":      "Dataset bias",
                "description": "UA-DETRAC is a Chinese motorway dataset; AI City Track 4 "
                               "is a US-style intersection dataset. Domain shift (pct_clipped "
                               "measured per clip) is non-trivial for clips 49, 51, 63.",
                "mitigation":  "Pearson test quantifies FAR inflation from domain shift. "
                               "pct_clipped documented per clip. Results scoped to these datasets.",
            },
            {
                "threat":      "Small sample sizes",
                "description": "n=10 normal clips for Wilcoxon and McNemar. n=30 for Pearson. "
                               "Power may be insufficient to detect small effects.",
                "mitigation":  "Exact McNemar used (appropriate for n<25). Wilcoxon is "
                               "non-parametric (robust). Effect sizes reported alongside p-values.",
            },
            {
                "threat":      "Anomaly ground truth quality",
                "description": "AI City Track 4 event timestamps may have annotation noise. "
                               "10 events only — insufficient for learning-based detection "
                               "statistical analysis.",
                "mitigation":  "Event-level analysis is descriptive only. FAR-based "
                               "statistical tests use the richer normal-clip dataset.",
            },
            {
                "threat":      "Latency measurement conditions",
                "description": "Latency profiled on a single normal clip in offline batch mode "
                               "on a Surface Pro 7. Not representative of edge deployment.",
                "mitigation":  "Fix F29 applied: caveats documented. FPS reported as "
                               "indicative, not a system guarantee.",
            },
            {
                "threat":      "OC-SVM threshold sensitivity",
                "description": "OC-SVM nu=0.01 tuned on UA-DETRAC validation split. "
                               "Performance on AI City clips may differ from reported AUROC.",
                "mitigation":  "FAR gate validates OC-SVM on AI City normal clips. "
                               "OC-SVM passes gate (mean_FAR=0.0283).",
            },
        ],
    }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Write outputs
# ─────────────────────────────────────────────────────────────────────────────

def _write_report_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"[Day 18] Report written → {path}")


def _write_stats_csv(
    w: dict, m: dict, p: dict, path: Path
) -> pd.DataFrame:
    rows = [
        {
            "test":          "Wilcoxon signed-rank",
            "comparison":    "rule_based vs ocsvm FAR per clip",
            "n":             w["n_pairs"],
            "statistic":     w["statistic"],
            "p_value":       w["p_value"],
            "significant_05": w["significant_05"],
            "effect_size":   f"median_rb={w['median_rb']:.4f}, median_ocsvm={w['median_ocsvm']:.4f}",
            "data_source":   "day17_far_per_clip.csv",
        },
        {
            "test":          "McNemar (exact)",
            "comparison":    "rule_based vs IF FAR-gate outcomes",
            "n":             m["n_clips"],
            "statistic":     m["statistic"],
            "p_value":       m["p_value"],
            "significant_05": m["significant_05"],
            "effect_size":   f"b={m['discordant_b']}, c={m['discordant_c']}",
            "data_source":   "day17_far_per_clip.csv",
        },
        {
            "test":          "Pearson r",
            "comparison":    "pct_clipped vs far_value (all 30 pairs)",
            "n":             p["n_pairs"],
            "statistic":     p["r"],
            "p_value":       p["p_value"],
            "significant_05": p["significant_05"],
            "effect_size":   f"r²={p['r_squared']:.4f}",
            "data_source":   "day17_far_per_clip.csv",
        },
    ]
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[Day 18] Stats CSV written → {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Day 18 - Results Consolidation & Statistical Tests")
    print("=" * 70)

    # Load data
    df_far = _load_far_per_clip()
    print(f"\n[OK] Loaded day17_far_per_clip.csv  ({len(df_far)} rows, "
          f"{df_far['clip_filename'].nunique()} clips, "
          f"{df_far['method'].nunique()} methods)")

    # Run tests
    print("\n-- Test 1: Wilcoxon signed-rank ---------------------------------")
    w = run_wilcoxon_far_test(df_far)
    print(f"  n={w['n_pairs']}  stat={w['statistic']:.4f}  p={w['p_value']:.4f}  "
          f"significant={w['significant_05']}")
    print(f"  {w['interpretation']}")

    print("\n-- Test 2: McNemar (exact) --------------------------------------")
    m = run_mcnemar_far_gate_test(df_far)
    tbl = m["contingency_table"]
    print(f"  Table [[{tbl['both_pass_a']},{tbl['if_fail_rb_pass_b']}],"
          f"[{tbl['if_pass_rb_fail_c']},{tbl['both_fail_d']}]]  "
          f"p={m['p_value']:.4f}  significant={m['significant_05']}")
    print(f"  {m['interpretation']}")

    print("\n-- Test 3: Pearson r --------------------------------------------")
    p = run_pearson_domain_shift_test(df_far)
    print(f"  n={p['n_pairs']}  r={p['r']:.4f}  r²={p['r_squared']:.4f}  "
          f"p={p['p_value']:.4f}  significant={p['significant_05']}")
    print(f"  {p['interpretation']}")

    # Consolidate
    print("\n-- Consolidating results tables ---------------------------------")
    tables   = build_complete_results_tables()
    failures = document_failure_cases()

    report = generate_day18_report(w, m, p, tables, failures)

    # Write outputs
    _write_report_json(report, OUT_DIR / "day18_report.json")
    stats_df = _write_stats_csv(
        w, m, p, OUT_DIR / "day18_statistical_results.csv"
    )

    print("\n-- Summary table ------------------------------------------------")
    print(stats_df[["test", "n", "p_value", "significant_05"]].to_string(index=False))

    print("\n[Day 18] DONE ✓")


if __name__ == "__main__":
    main()
