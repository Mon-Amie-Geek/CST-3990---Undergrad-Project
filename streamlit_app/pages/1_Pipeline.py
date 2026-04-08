"""
1_Pipeline.py

Main pipeline page for processing traffic videos.
Aligns feature naming with Block C schema v1.1 (F1/F2/F3).
Student: MANJOO Ameera Najla | M01014463
"""

import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import json
import torch
import random
from datetime import datetime

# Import components
from components import Detector, Tracker, FeatureExtractor, AnomalyDetector
from utils.video_utils import get_video_metadata, save_uploaded_file
from utils.drawing_utils import draw_tracks, draw_anomaly_banner

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

st.set_page_config(page_title="Pipeline - CST3990 Traffic Anomaly Detection", layout="wide")

st.title("🎥 Live Detection Pipeline")

# Initialize session state
if "processing" not in st.session_state:
    st.session_state.processing    = False
    st.session_state.results       = None
    st.session_state.frame_data    = []
    st.session_state.anomaly_events = []

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
st.sidebar.header("⚙️ Pipeline Configuration")

# 1. Video Upload
uploaded_file = st.sidebar.file_uploader("Upload traffic video (.mp4)", type=["mp4"])

# 2. Block A: Detector
st.sidebar.write("### Block A — Detector")
st.sidebar.caption("Research frozen: **YOLOv8n** (mAP@0.5 = 0.6674, Day 6)")
detector_choice = st.sidebar.radio(
    "Select detector:",
    ["YOLOv8n (fine-tuned)", "SSD300 (COCO)"],
    key="detector_choice"
)

# 3. Confidence / NMS thresholds
conf_thresh = st.sidebar.slider("Confidence threshold:", 0.10, 0.90, 0.25, 0.05)
nms_thresh  = st.sidebar.slider("NMS threshold:",         0.10, 0.90, 0.45, 0.05)

# 4. Block B: Tracker
st.sidebar.write("### Block B — Tracker")
st.sidebar.caption("Research frozen: **SORT** (IDF1 = 0.0412, Day 9)")
tracker_choice = st.sidebar.radio(
    "Select tracker:",
    ["SORT", "DeepSORT", "ByteTrack"],
    key="tracker_choice"
)
st.sidebar.caption(
    "ℹ️ The app uses the workspace Block B tracker implementations and frozen configs. "
    "DeepSORT uses VeRi-776 ReID; ByteTrack uses two-pass confidence association."
)

# 5. Block C: Features
st.sidebar.write("### Block C — Features")
st.sidebar.caption("Research frozen: **F2_only** (AUROC = 0.9935, Day 12)")
features_selected = st.sidebar.multiselect(
    "Select feature groups:",
    ["F1: Density/Flow", "F2: Speed", "F3: Distance/Proximity"],
    default=["F1: Density/Flow", "F2: Speed", "F3: Distance/Proximity"],
    key="features_selected"
)

# 6. Block D: Anomaly Method
st.sidebar.write("### Block D — Anomaly Detection")
anomaly_choice = st.sidebar.radio(
    "Select anomaly method:",
    ["Rule-Based", "One-Class SVM", "Isolation Forest"],
    key="anomaly_choice"
)
st.sidebar.caption(
    "OC-SVM: nu=0.01 (Block C Day 12 best nu for F2_only). "
    "If the pre-trained Block C model is found, it is loaded directly."
)

st.sidebar.markdown("---")

# 7. Run Button
run_button = st.sidebar.button(
    "▶ Run Pipeline",
    disabled=(uploaded_file is None),
    use_container_width=True,
    type="primary"
)

# 8. Visualization Options
st.sidebar.write("### Visualization Options")
show_boxes = st.sidebar.checkbox("Show bounding boxes", value=True)
show_ids   = st.sidebar.checkbox("Show track IDs",      value=True)
show_speed = st.sidebar.checkbox("Show speed labels",   value=True)

# ============================================================================
# MAIN PANEL — TABS
# ============================================================================
tab1, tab2, tab3 = st.tabs(["Live Detection", "Feature Charts", "Results Summary"])

# ============================================================================
# TAB 1: LIVE DETECTION
# ============================================================================
with tab1:
    if run_button and uploaded_file:
        st.session_state.processing = True

        video_path = save_uploaded_file(uploaded_file)

        try:
            metadata    = get_video_metadata(video_path)
            fps         = metadata["fps"]
            frame_count = metadata["frame_count"]
            width       = metadata["width"]
            height      = metadata["height"]

            st.success(
                f"✅ Loaded: {frame_count} frames @ {fps:.1f} FPS ({width}×{height})"
            )

            # Initialise components
            detector_name = "yolov8n" if "YOLOv8n" in detector_choice else "ssd300"
            detector      = Detector(detector_name, conf_thresh, nms_thresh)

            tracker_name  = tracker_choice.lower()
            tracker       = Tracker(tracker_name)

            feature_extractor = FeatureExtractor(fps, width, height, features_selected)

            anomaly_name_map = {
                "Rule-Based":        "rule_based",
                "One-Class SVM":     "ocsvm",
                "Isolation Forest":  "isolation_forest",
            }
            anomaly_detector = AnomalyDetector(anomaly_name_map[anomaly_choice])

            # ── Show if pre-trained OC-SVM was loaded ───────────────────────
            if anomaly_choice == "One-Class SVM" and anomaly_detector._using_pretrained:
                st.info(
                    "✅ Block C pre-trained OC-SVM loaded (`ocsvm_trained_best.pkl`). "
                    "Using F2_only features (vel_px_sec, vel_px_sec_smooth) with "
                    "fitted MinMaxScaler — Block C research-quality inference."
                )
            elif anomaly_choice == "One-Class SVM":
                st.info(
                    "ℹ️ Block C pre-trained OC-SVM not found. "
                    "Training on warmup data (first 200 frames) with nu=0.01."
                )

            # Process frames
            cap = cv2.VideoCapture(video_path)
            process_every_n = 3

            frame_placeholder  = st.empty()
            stats_placeholder  = st.empty()
            progress_bar       = st.progress(0)

            frame_idx          = 0
            processed_frame_idx = 0
            frame_times         = []
            anomaly_events_local = []
            frame_data_list     = []
            speed_data_by_frame = {}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % process_every_n != 0:
                    continue

                processed_frame_idx += 1
                t_start = time.time()

                # Detect → Track → Extract → Predict
                detections     = detector.detect(frame)
                tracks         = tracker.update(detections, processed_frame_idx, frame=frame)
                features       = feature_extractor.extract(tracks, processed_frame_idx)
                frame_data_list.append(features)
                anomaly_result = anomaly_detector.predict(features)

                # Anomalous track IDs (post-warmup only)
                anomalous_ids = set()
                for track in tracks:
                    if track.get("age", 0) >= feature_extractor.warmup_frames:
                        if anomaly_result["is_anomaly"]:
                            anomalous_ids.add(track["track_id"])

                # Per-track speed for label overlay
                speed_data = {}
                for track in tracks:
                    tid = track.get("track_id")
                    if tid and tid in feature_extractor.track_history:
                        spds = feature_extractor.track_history[tid]["speeds"]
                        if spds:
                            speed_data[tid] = float(np.mean(spds[-3:]))

                speed_data_by_frame[processed_frame_idx] = speed_data

                # Draw and display
                annotated = draw_tracks(
                    frame, tracks, anomalous_ids,
                    show_boxes, show_ids, show_speed, speed_data
                )
                if anomaly_result["is_anomaly"]:
                    feat_label = anomaly_result.get("triggered_feature") or "Unknown"
                    annotated = draw_anomaly_banner(
                        annotated,
                        f"ANOMALY — {feat_label} — Frame {processed_frame_idx}"
                    )
                    anomaly_events_local.append({
                        "Frame":              processed_frame_idx,
                        "Track ID":           ("multiple" if len(anomalous_ids) > 1
                                               else list(anomalous_ids)[0] if anomalous_ids
                                               else "unknown"),
                        "Class":              "Anomalous Behaviour",
                        "Feature Triggered":  feat_label,
                        "Timestamp (s)":      round(processed_frame_idx / fps, 2),
                        "Score":              round(float(anomaly_result.get("score", 0.0)), 4),
                    })
                    st.error(
                        f"⚠ ANOMALY DETECTED — Frame {processed_frame_idx} — {feat_label}"
                    )

                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, use_column_width=True)

                t_end = time.time()
                frame_times.append(t_end - t_start)
                fps_current = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0

                with stats_placeholder.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Frame",         f"{processed_frame_idx} / {frame_count // process_every_n}")
                    c2.metric("FPS",           f"{fps_current:.1f}")
                    c3.metric("Active Tracks", len(tracks))
                    c4.metric("Anomalies",     len(anomaly_events_local))

                progress = processed_frame_idx / max(1, frame_count // process_every_n)
                progress_bar.progress(min(1.0, progress))

            cap.release()

            st.session_state.results = {
                "frame_data":         frame_data_list,
                "anomaly_events":     anomaly_events_local,
                "frame_times":        frame_times,
                "speed_data_by_frame": speed_data_by_frame,
                "metadata":           metadata,
                "config": {
                    "detector":       detector_choice,
                    "conf_thresh":    conf_thresh,
                    "nms_thresh":     nms_thresh,
                    "tracker":        tracker_choice,
                    "features":       features_selected,
                    "anomaly_method": anomaly_choice,
                },
            }
            st.success("✅ Pipeline processing complete!")
            st.session_state.processing = False

        except Exception as e:
            st.error(f"❌ Error processing video: {e}")
            st.session_state.processing = False

    elif not st.session_state.processing:
        st.info(
            "👆 Upload a traffic video (.mp4) and configure the pipeline in the sidebar, "
            "then click **▶ Run Pipeline**."
        )

# ============================================================================
# TAB 2: FEATURE CHARTS
# ============================================================================
with tab2:
    if st.session_state.results:
        import plotly.graph_objects as go

        frame_data = st.session_state.results["frame_data"]
        frames     = list(range(len(frame_data)))

        st.caption(
            "Feature naming aligned with Block C schema v1.1. "
            "Speed = F2 (vel_px_sec), Distance = F3 (inter_vehicle_dist_norm proxy), "
            "Dwell = F3 (dwell_time_sec), Density = F1 (vehicle_count)."
        )

        # ── F2: Speed (vel_px_sec) ───────────────────────────────────────────
        if "F2: Speed" in features_selected:
            speeds = [f.get("mean_speed_px_sec", 0.0) for f in frame_data]

            fig_sp = go.Figure()
            fig_sp.add_trace(go.Scatter(
                x=frames, y=speeds,
                mode="lines", name="mean vel_px_sec",
                line=dict(color="#2ca02c", width=2)
            ))

            if speeds and max(speeds) > 0:
                threshold = np.mean(speeds) + 2.0 * np.std(speeds)
                fig_sp.add_hline(
                    y=threshold, line_dash="dash", line_color="red",
                    annotation_text=f"Z-score threshold (mean + 2σ = {threshold:.1f} px/s)"
                )

            fig_sp.update_layout(
                title="F2 — Speed Proxy Over Time (vel_px_sec, image-space, NOT km/h)",
                xaxis_title="Processed Frame",
                yaxis_title="Mean speed (px/sec)",
                hovermode="x unified",
                height=400,
            )
            st.plotly_chart(fig_sp, use_container_width=True)
            st.caption(
                "speed_window = int(0.2 × fps) frames (Fix F07). "
                "Image-space proxy — NOT calibrated to km/h. "
                "Red dashed line = rule-based anomaly threshold (mean + 2σ from full sequence)."
            )

        # ── F3: Inter-Vehicle Distance (inter_vehicle_dist_norm proxy) ───────
        if "F3: Distance/Proximity" in features_selected:
            distances = [f.get("min_distance_norm", 1.0) for f in frame_data]

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Scatter(
                x=frames, y=distances,
                mode="lines+markers", name="min inter-vehicle dist (normalised)",
                line=dict(color="#9467bd", width=2),
                marker=dict(size=3)
            ))

            fig_dist.update_layout(
                title="F3 — Minimum Inter-Vehicle Distance (inter_vehicle_dist_norm proxy, normalised by frame diagonal)",
                xaxis_title="Processed Frame",
                yaxis_title="Normalised distance (0–1)",
                hovermode="x unified",
                height=400,
            )
            fig_dist.add_annotation(
                text="⚠️ Image-space proxy — NOT physical distance (Fix F28)",
                xref="paper", yref="paper",
                x=0.5, y=-0.18, showarrow=False,
                font=dict(size=11, color="red")
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption(
                "Normalised by frame diagonal (640×√2 ≈ 905 px). "
                "NaN/1.0 when only one vehicle is present. "
                "Image-space only — perspective effects make this a relative behavioural indicator."
            )

            # ── F3: Dwell Time ───────────────────────────────────────────────
            dwell_times = [f.get("mean_dwell_sec", 0.0) for f in frame_data]
            if any(d > 0 for d in dwell_times):
                fig_dwell = go.Figure()
                fig_dwell.add_trace(go.Scatter(
                    x=frames, y=dwell_times,
                    mode="lines", name="mean dwell_time_sec",
                    line=dict(color="#d62728", width=2)
                ))
                fig_dwell.update_layout(
                    title="F3 — Mean Track Dwell Time (dwell_time_sec = time in scene)",
                    xaxis_title="Processed Frame",
                    yaxis_title="Dwell time (seconds)",
                    hovermode="x unified",
                    height=360,
                )
                st.plotly_chart(fig_dwell, use_container_width=True)
                st.caption(
                    "Dwell time = (last_frame − first_frame) / fps for each active track. "
                    "Rising dwell times may indicate stopped or slow-moving vehicles."
                )

        # ── F1: Vehicle Count / ROI Occupancy ────────────────────────────────
        if "F1: Density/Flow" in features_selected:
            # Sample every 10 frames to avoid overloading the bar chart
            sampled_frames  = frames[::10]
            vehicle_counts  = [frame_data[i].get("vehicle_count", 0)
                               for i in range(0, len(frame_data), 10)]
            roi_occupancies = [frame_data[i].get("roi_occupancy", 0.0)
                               for i in range(0, len(frame_data), 10)]

            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Bar(
                x=sampled_frames, y=vehicle_counts,
                name="vehicle_count (F1)",
                marker=dict(color="#ff7f0e"),
                yaxis="y",
            ))
            fig_f1.add_trace(go.Scatter(
                x=sampled_frames, y=roi_occupancies,
                mode="lines", name="roi_occupancy (F1)",
                line=dict(color="#1f77b4", width=2),
                yaxis="y2",
            ))
            fig_f1.update_layout(
                title="F1 — Vehicle Count & ROI Occupancy Over Time (sampled every 10 frames)",
                xaxis_title="Processed Frame",
                yaxis=dict(title="Vehicle Count"),
                yaxis2=dict(title="ROI Occupancy (0–1)", overlaying="y", side="right"),
                hovermode="x unified",
                height=400,
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_f1, use_container_width=True)
            st.caption(
                "roi_occupancy is estimated as (n_tracks × 80×60) / (frame_width × frame_height). "
                "This is a simplified proxy — exact bbox areas are not used in the demo."
            )

    else:
        st.info("👈 Run the pipeline first to generate feature charts.")

# ============================================================================
# TAB 3: RESULTS SUMMARY
# ============================================================================
with tab3:
    if st.session_state.results:
        results       = st.session_state.results
        config        = results["config"]
        metadata      = results["metadata"]
        frame_data    = results["frame_data"]
        anomaly_events = results["anomaly_events"]
        frame_times   = results["frame_times"]

        st.subheader("Pipeline Configuration Used")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Block A — Detector:**")
            st.metric("Selected Detector",    config["detector"])
            st.metric("Confidence Threshold", f"{config['conf_thresh']:.2f}")
            st.metric("NMS Threshold",        f"{config['nms_thresh']:.2f}")

            st.write("**Block B — Tracker:**")
            st.metric("Selected Tracker", config["tracker"])

            st.write("**Block C — Feature Groups:**")
            st.write(", ".join(config["features"]) if config["features"] else "None selected")

            st.write("**Block D — Anomaly Method:**")
            st.metric("Method", config["anomaly_method"])

        with col2:
            st.write("**Performance Metrics:**")
            st.metric("Video FPS",            f"{metadata['fps']:.1f}")
            st.metric("Total Frames (video)", metadata["frame_count"])
            st.metric("Frames Processed",     len(frame_data))
            st.metric("Frame Skip",           "Every 3rd frame (3×)")

            if frame_times:
                st.metric("Processing FPS",   f"{1.0 / np.mean(frame_times):.1f}")
                st.metric("Avg Latency (ms)", f"{np.mean(frame_times) * 1000:.1f}")
                st.metric("p95 Latency (ms)", f"{np.percentile(frame_times, 95) * 1000:.1f}")

            st.metric("Anomaly Events",       len(anomaly_events))

        st.markdown("---")

        # Feature summary statistics
        if frame_data:
            st.subheader("Feature Summary Statistics (this video)")
            df_feat = pd.DataFrame(frame_data)
            _stat_cols = [c for c in [
                "vehicle_count", "roi_occupancy",
                "mean_speed_px_sec", "min_distance_norm", "mean_dwell_sec"
            ] if c in df_feat.columns]
            if _stat_cols:
                _stat_display = df_feat[_stat_cols].describe().round(4)
                # Rename columns to match Block C schema names for clarity
                _rename = {
                    "mean_speed_px_sec": "vel_px_sec (approx)",
                    "min_distance_norm": "inter_vehicle_dist_norm (proxy)",
                    "mean_dwell_sec":    "dwell_time_sec (mean)",
                }
                _stat_display = _stat_display.rename(columns=_rename)
                st.dataframe(_stat_display, use_container_width=True)

        st.markdown("---")
        st.subheader("Anomaly Event Log")

        if anomaly_events:
            df_events = pd.DataFrame(anomaly_events)
            st.dataframe(df_events, use_container_width=True)
        else:
            st.info("No anomalies detected in this video.")

        st.markdown("---")

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            if anomaly_events:
                events_json = json.dumps(anomaly_events, indent=2)
                st.download_button(
                    label="📥 Download events.json",
                    data=events_json,
                    file_name=f"anomaly_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )
        with col2:
            if frame_data:
                df_features = pd.DataFrame(frame_data)
                # Rename to match Block C schema names
                df_features = df_features.rename(columns={
                    "mean_speed_px_sec": "vel_px_sec",
                    "min_distance_norm": "inter_vehicle_dist_norm",
                    "mean_dwell_sec":    "dwell_time_sec",
                })
                st.download_button(
                    label="📥 Download features.csv",
                    data=df_features.to_csv(index=False),
                    file_name=f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

    else:
        st.info("👈 Run the pipeline first to see results summary.")
