"""
2_Comparison.py

Comparison page showing placeholder experimental results.
Results will be populated with actual data from Days 5-18.
Student: MANJOO Ameera Najla | M01014463
"""

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
# BLOCK C — FEATURE COMPARISON
# ============================================================================
st.header("Block C — Feature Set Comparison")

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
st.write("**placeholder — to be updated with experimental results**")
st.dataframe(df_features, use_container_width=True)

st.caption("""
- **AUROC:** Area Under Receiver Operating Characteristic Curve (anomaly detection)
- **Silhouette Score:** Measure of feature quality (range: -1 to 1, higher is better)
- Combined features generally outperform individual features
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
