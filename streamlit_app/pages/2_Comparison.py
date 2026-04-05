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

# ── AUROC comparison (placeholder — populated Day 12) ───────────────────────
st.subheader("Feature Set AUROC Comparison (Day 12 — placeholder)")

features_data = {
    "Feature Set": [
        "F1: Density/Flow only",
        "F2: Speed only",
        "F3: Distance only",
        "F1 + F2 (combined)",
        "F1 + F3 (combined)",
        "F2 + F3 (combined)",
        "F1 + F2 + F3 (all)"
    ],
    "AUROC": [0.721, 0.814, 0.768, 0.852, 0.823, 0.891, 0.918],
    "Silhouette Score": [0.342, 0.521, 0.468, 0.612, 0.587, 0.728, 0.782]
}

df_features = pd.DataFrame(features_data)
st.write("**placeholder — AUROC will be populated after Day 12 OC-SVM evaluation**")
st.dataframe(df_features, use_container_width=True)

st.caption("""
- **AUROC:** Area Under Receiver Operating Characteristic Curve (anomaly detection) — Day 12
- **Silhouette Score:** Measure of feature quality (range: -1 to 1, higher is better) — Day 12
- F3 features (inter_vehicle_dist_norm, dwell_time_sec, proximity_flag) added Day 11
""")

st.markdown("---")

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
    st.metric("Best Feature Set", "All Combined", "AUROC: 0.918")
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
