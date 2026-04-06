"""
2_Comparison.py

Comparison page showing experimental results.
Block C section (Day 10+) loads actual feature CSVs from logs/block_c/ when available.
Results for Blocks A, B, D remain placeholder until those days complete.
Student: MANJOO Ameera Najla | M01014463
"""

import os
import glob

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Comparison - CST3990 Traffic Anomaly Detection", layout="wide")

st.title("📊 Results Comparison")

st.info("""
📌 **Placeholder Results Notice**

Results tables below show placeholder data as actual experimental results will be populated 
during Days 5-18 of the research phase. All values are currently illustrative examples 
and will be updated with real experimental data from the pipeline testing phase.
""")

st.markdown("---")

# ============================================================================
# BLOCK A — DETECTOR COMPARISON
# ============================================================================
st.header("Block A — Detector Comparison")

detector_data = {
    "Detector": ["YOLOv8n (fine-tuned)", "MobileNet-SSD (COCO)"],
    "mAP@0.5": [0.742, 0.628],
    "Precision": [0.812, 0.704],
    "Recall": [0.698, 0.623],
    "FPS (CPU)": [12.5, 8.2]
}

df_detectors = pd.DataFrame(detector_data)
st.write("**placeholder — to be updated with experimental results**")
st.dataframe(df_detectors, use_container_width=True)

st.caption("""
- **mAP@0.5:** Mean Average Precision at IoU=0.5 on UA-DETRAC test set
- **FPS:** Frames per second on CPU (Intel Iris Plus, no GPU)
- YOLOv8n is fine-tuned on UA-DETRAC; MobileNet-SSD uses COCO pre-training
""")

st.markdown("---")

# ============================================================================
# BLOCK B — TRACKER COMPARISON
# ============================================================================
st.header("Block B — Tracker Comparison")

tracker_data = {
    "Tracker": ["SORT", "DeepSORT", "ByteTrack"],
    "MOTA": [0.651, 0.712, 0.738],
    "IDF1": [0.543, 0.628, 0.672],
    "ID Switches": [142, 89, 64],
    "Fragmentation": [23, 14, 8],
    "FPS (CPU)": [18.2, 14.5, 12.1]
}

df_trackers = pd.DataFrame(tracker_data)
st.write("**placeholder — to be updated with experimental results**")
st.dataframe(df_trackers, use_container_width=True)

st.caption("""
- **MOTA:** Multi-Object Tracking Accuracy
- **IDF1:** ID F1 Score (identity consistency)
- **ID Switches:** Number of identity switches (lower is better)
- **Fragmentation:** Number of fragmented tracks (lower is better)
- **FPS:** Processing speed on CPU
""")

st.markdown("---")

# ============================================================================
# BLOCK C — FEATURE SET COMPARISON (Day 10: loads actual CSVs when available)
# ============================================================================
st.header("Block C — Feature Set Comparison")

# ── Try to load actual feature CSVs produced by Day 10 ──────────────────────
BLOCK_C_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "logs", "block_c"
)
BLOCK_C_DIR = os.path.normpath(BLOCK_C_DIR)

feature_csvs = sorted(glob.glob(os.path.join(BLOCK_C_DIR, "*_features.csv")))

if feature_csvs:
    st.success(
        f"**Day 10 feature CSVs found** — {len(feature_csvs)} sequence(s) in `logs/block_c/`"
    )

    # Load and concatenate all available CSVs
    dfs = []
    for p in feature_csvs:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            st.warning(f"Could not load {os.path.basename(p)}: {e}")

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)

        # ── Summary stats ────────────────────────────────────────────────────
        st.subheader("Feature Dataset Summary (F1 + F2 — Day 10)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total rows",       f"{len(df_all):,}")
        col2.metric("Sequences",        df_all["seq_id"].nunique() if "seq_id" in df_all else "?")
        col3.metric("Tracks",           df_all["track_id"].nunique() if "track_id" in df_all else "?")
        col4.metric("Interpolated rows",
                    f"{int(df_all['is_interpolated'].sum()):,}" if "is_interpolated" in df_all else "?")

        # Split breakdown
        if "split" in df_all.columns:
            st.write("**Row counts per split:**")
            split_counts = df_all["split"].value_counts().reset_index()
            split_counts.columns = ["split", "rows"]
            st.dataframe(split_counts, use_container_width=True)

        # ── F1 velocity / vehicle count chart ───────────────────────────────
        if "vehicle_count" in df_all.columns and "seq_id" in df_all.columns:
            st.subheader("F1: Vehicle Count Distribution per Sequence")
            vc_stats = (
                df_all.groupby("seq_id")["vehicle_count"]
                .agg(["mean", "max"])
                .reset_index()
                .rename(columns={"mean": "Mean Count", "max": "Max Count"})
            )
            fig_vc = go.Figure()
            fig_vc.add_trace(go.Bar(
                x=vc_stats["seq_id"], y=vc_stats["Mean Count"],
                name="Mean", marker_color="#1f77b4"
            ))
            fig_vc.add_trace(go.Bar(
                x=vc_stats["seq_id"], y=vc_stats["Max Count"],
                name="Max", marker_color="#ff7f0e"
            ))
            fig_vc.update_layout(
                barmode="group", title="Vehicle Count (mean / max) per Sequence",
                xaxis_title="Sequence", yaxis_title="Vehicle Count", height=380
            )
            st.plotly_chart(fig_vc, use_container_width=True)

        # ── F2 speed distribution ────────────────────────────────────────────
        if "vel_px_sec_smooth" in df_all.columns:
            st.subheader("F2: Smoothed Speed Distribution (vel_px_sec_smooth)")
            speed_data = df_all["vel_px_sec_smooth"].dropna()
            fig_sp = go.Figure()
            fig_sp.add_trace(go.Histogram(
                x=speed_data, nbinsx=60,
                marker_color="#2ca02c", opacity=0.75,
                name="vel_px_sec_smooth"
            ))
            fig_sp.update_layout(
                title="Distribution of vel_px_sec_smooth across all sequences",
                xaxis_title="Speed (px/sec, smoothed)",
                yaxis_title="Count",
                height=350
            )
            st.plotly_chart(fig_sp, use_container_width=True)
            st.caption(
                "speed_window = int(0.2 × fps) frames (Fix F07). "
                "At 25 FPS = 5-frame rolling mean. Image-space proxy — not calibrated to km/h."
            )

        # ── Per-sequence feature stats table ────────────────────────────────
        st.subheader("Per-Sequence Feature Statistics")
        stat_cols = [c for c in ["vehicle_count", "roi_occupancy", "vel_px_sec", "vel_px_sec_smooth"]
                     if c in df_all.columns]
        if stat_cols and "seq_id" in df_all.columns:
            stats_df = df_all.groupby("seq_id")[stat_cols].mean().round(4).reset_index()
            st.dataframe(stats_df, use_container_width=True)

    st.markdown("---")

else:
    st.info(
        "**Block C feature CSVs not yet generated locally.** "
        "Run Day 10 notebook in Colab to generate `logs/block_c/*_features.csv`, "
        "then pull to this repo for live display here."
    )


# -- OC-SVM AUROC comparison: loads Day 12 results when available
import json as _json_bc

st.subheader('Feature Set AUROC Comparison - OC-SVM (Block C Day 12)')

_RESULTS_TABLE_CSV = os.path.normpath(
    os.path.join(BLOCK_C_DIR, 'block_c_results_table.csv')
)
_DAY12_META_JSON = os.path.normpath(
    os.path.join(BLOCK_C_DIR, 'block_c_day12_metadata.json')
)

_best_fs_for_summary    = None
_best_auroc_for_summary = None
_block_c_day12_done     = False

if os.path.exists(_RESULTS_TABLE_CSV):
    try:
        df_auroc_table = pd.read_csv(_RESULTS_TABLE_CSV)
        _CHECK = chr(10003)  # checkmark

        _brow = df_auroc_table[
            df_auroc_table['Selected'].astype(str).str.strip() == _CHECK
        ]
        if not _brow.empty:
            _best_fs_for_summary    = str(_brow.iloc[0]['Feature Set'])
            _best_auroc_for_summary = _brow.iloc[0]['AUROC']
            _block_c_day12_done     = True
            st.success(
                '**Block C Day 12 complete.** Best feature set: **'
                + _best_fs_for_summary
                + '** - AUROC = '
                + str(_best_auroc_for_summary)
                + ' | best nu = '
                + str(_brow.iloc[0]['Best Nu'])
            )
        else:
            st.success('**Block C Day 12 results loaded.**')

        st.dataframe(df_auroc_table, use_container_width=True)

        # AUROC bar chart
        _vauc = df_auroc_table[df_auroc_table['AUROC'].notna()].copy()
        if not _vauc.empty:
            _bar_cols = [
                '#2ca02c' if str(r['Selected']).strip() == _CHECK else '#1f77b4'
                for _, r in _vauc.iterrows()
            ]
            _fig_a = go.Figure(go.Bar(
                x=_vauc['Feature Set'],
                y=_vauc['AUROC'].astype(float),
                marker_color=_bar_cols,
                text=['{:.4f}'.format(v) for v in _vauc['AUROC'].astype(float)],
                textposition='outside',
            ))
            _fig_a.update_layout(
                title=(
                    'OC-SVM AUROC by Feature Set - Fix F04 (3-sigma train-stats proxy labels)'
                ),
                xaxis_title='Feature Set',
                yaxis_title='AUROC',
                yaxis=dict(range=[0, 1.1]),
                height=420,
                showlegend=False,
            )
            st.plotly_chart(_fig_a, use_container_width=True)

        # Silhouette bar chart
        _vsil = df_auroc_table[df_auroc_table['Silhouette'].notna()].copy()
        if not _vsil.empty:
            _fig_s = go.Figure(go.Bar(
                x=_vsil['Feature Set'],
                y=_vsil['Silhouette'].astype(float),
                marker_color='#9467bd',
                text=['{:.4f}'.format(v) for v in _vsil['Silhouette'].astype(float)],
                textposition='outside',
            ))
            _fig_s.update_layout(
                title='Silhouette Score by Feature Set (val split, y_val_binary labels)',
                xaxis_title='Feature Set',
                yaxis_title='Silhouette Score',
                height=380,
                showlegend=False,
            )
            st.plotly_chart(_fig_s, use_container_width=True)

        # Metadata expander
        if os.path.exists(_DAY12_META_JSON):
            with st.expander('Day 12 Provenance Metadata', expanded=False):
                try:
                    with open(_DAY12_META_JSON, encoding='utf-8') as _fh:
                        _m12 = _json_bc.load(_fh)
                    st.json({k: v for k, v in _m12.items()
                             if k != 'y_val_binary_note'})
                    st.caption(_m12.get('y_val_binary_note', ''))
                except Exception as _me:
                    st.warning('Could not read metadata: ' + str(_me))

        st.caption(
            'AUROC (Fix F04 - anti-circular): OC-SVM decision-function scores on '
            'UA-DETRAC val split (MVI_20061). y_val_binary uses a 3-sigma threshold '
            'from train-split statistics only - no val data contaminates labels. '
            'Measures feature separability of statistical outliers within a normal '
            'distribution, NOT true anomaly detection. Block D (AI City) has true GT. '
            'Silhouette: cluster separation under proxy labels. '
            'Fix F02: nu swept over {0.01, 0.02, 0.05, 0.10, 0.15} on 80/20 train '
            'hold-out; GridSearchCV not used (OC-SVM has no negative class for CV). '
            'Row counts may differ for F3-containing sets due to Day 11 NaN semantics.'
        )

    except Exception as _err:
        st.warning('Could not load Day 12 AUROC results: ' + str(_err))

else:
    st.info(
        'Block C Day 12 AUROC results not yet generated. '
        'Run PipelineController.run_block_c_day12() or '
        'python -m src.block_c_evaluator from the repo root, then pull '
        'logs/block_c/block_c_results_table.csv to this repo.'
    )

st.caption(
    'AUROC: Area Under ROC Curve - higher = better separability (Day 12). '
    'Silhouette: cluster separation under 3-sigma proxy labels (Day 12). '
    'F3 OC-SVM inputs: inter_vehicle_dist_norm, dwell_time_sec, '
    'proximity_count_rolling (continuous/temporal behavioural indicators). '
    'proximity_flag is in Day 11 CSV schema but excluded from Day 12 model '
    'inputs (thesis alignment). F3 added Day 11; AUROC comparison Day 12.'
)

st.markdown('---')

# ============================================================================
# BLOCK D — ANOMALY METHOD COMPARISON
# ============================================================================
st.header("Block D — Anomaly Detection Method Comparison")

anomaly_data = {
    "Method": ["Rule-Based (Z-score)", "One-Class SVM", "Isolation Forest"],
    "Precision": [0.782, 0.851, 0.823],
    "Recall": [0.698, 0.764, 0.751],
    "F1 Score": [0.738, 0.806, 0.786],
    "False Alarm Rate (%)": [12.3, 8.5, 9.1]
}

df_anomaly = pd.DataFrame(anomaly_data)
st.write("**placeholder — to be updated with experimental results**")
st.dataframe(df_anomaly, use_container_width=True)

st.caption("""
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1 Score:** Harmonic mean of Precision and Recall
- **False Alarm Rate:** Percentage of false positive detections
""")

# Create F1 score comparison chart
st.subheader("F1 Score Comparison Across Anomaly Methods")
fig = go.Figure(data=[
    go.Bar(
        x=df_anomaly["Method"],
        y=df_anomaly["F1 Score"],
        marker=dict(color=["#1f77b4", "#ff7f0e", "#2ca02c"]),
        text=df_anomaly["F1 Score"],
        textposition="outside"
    )
])
fig.update_layout(
    title="Anomaly Detection Method Performance (F1 Score)",
    yaxis_title="F1 Score",
    xaxis_title="Detection Method",
    height=400,
    showlegend=False,
    yaxis=dict(range=[0, 1])
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================================================
# SUMMARY
# ============================================================================
st.header("Key Findings (Placeholder)")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Best Detector", "YOLOv8n", "74.2% mAP@0.5")
    st.metric("Best Tracker", "ByteTrack", "IDF1: 0.672")

with col2:
    st.metric(
        "Best Feature Set",
        _best_fs_for_summary if _block_c_day12_done else "All Combined",
        f"AUROC: {_best_auroc_for_summary}" if _block_c_day12_done else "AUROC: 0.918 (placeholder)",
    )
    st.metric("Best Anomaly Method", "One-Class SVM", "F1: 0.806")

with col3:
    st.metric("Avg Processing FPS", "~14 fps", "CPU only")
    st.metric("Trade-off", "Speed vs Accuracy", "3rd-frame skip used")

st.markdown("---")

st.info("""
**Note on Results Validity**

These placeholder results demonstrate the structure and metrics used for evaluation. 
Actual experimental results from Days 5-18 will replace these values. The system tracks:

- Detection performance on UA-DETRAC test set
- Tracking robustness metrics (MOTA, IDF1)
- Feature discriminative power (AUROC, Silhouette)
- Anomaly detection effectiveness (Precision, Recall, F1)
- Real-time feasibility (FPS on CPU hardware)
""")
