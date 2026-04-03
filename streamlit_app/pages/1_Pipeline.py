"""
1_Pipeline.py

Main pipeline page for processing traffic videos.
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
    st.session_state.processing = False
    st.session_state.results = None
    st.session_state.frame_data = []
    st.session_state.anomaly_events = []

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
st.sidebar.header("⚙️ Pipeline Configuration")

# 1. Video Upload
uploaded_file = st.sidebar.file_uploader(
    "Upload traffic video (.mp4)",
    type=["mp4"]
)

# 2. Block A: Detector
st.sidebar.write("### Block A — Detector")
detector_choice = st.sidebar.radio(
    "Select detector:",
    ["YOLOv8n (fine-tuned)", "MobileNet-SSD (COCO)"],
    key="detector_choice"
)

# 3. Confidence Threshold
conf_thresh = st.sidebar.slider(
    "Confidence threshold:",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

# 4. NMS Threshold
nms_thresh = st.sidebar.slider(
    "NMS threshold:",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05
)

# 5. Block B: Tracker
st.sidebar.write("### Block B — Tracker")
tracker_choice = st.sidebar.radio(
    "Select tracker:",
    ["SORT", "DeepSORT", "ByteTrack"],
    key="tracker_choice"
)
st.sidebar.caption(
    "ℹ️ DeepSORT (VeRi-776 ReID) and ByteTrack are fully implemented in the "
    "research pipeline (Days 8–9). The live demo uses a Kalman+IoU approximation "
    "for all three — full results are on the Comparison page."
)

# 6. Block C: Features
st.sidebar.write("### Block C — Features")
features_selected = st.sidebar.multiselect(
    "Select features:",
    ["F1: Density/Flow", "F2: Speed", "F3: Distance"],
    default=["F1: Density/Flow", "F2: Speed", "F3: Distance"],
    key="features_selected"
)

# 7. Block D: Anomaly Method
st.sidebar.write("### Block D — Anomaly Detection")
anomaly_choice = st.sidebar.radio(
    "Select anomaly method:",
    ["Rule-Based", "One-Class SVM", "Isolation Forest"],
    key="anomaly_choice"
)

st.sidebar.markdown("---")

# 8. Run Button
run_button = st.sidebar.button(
    "▶ Run Pipeline",
    disabled=(uploaded_file is None),
    use_container_width=True,
    type="primary"
)

# 9-11. Visualization Options
st.sidebar.write("### Visualization Options")
show_boxes = st.sidebar.checkbox("Show bounding boxes", value=True)
show_ids = st.sidebar.checkbox("Show track IDs", value=True)
show_speed = st.sidebar.checkbox("Show speed labels", value=True)

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
        
        # Save uploaded file
        video_path = save_uploaded_file(uploaded_file)
        
        try:
            # Get video metadata
            metadata = get_video_metadata(video_path)
            fps = metadata["fps"]
            frame_count = metadata["frame_count"]
            width = metadata["width"]
            height = metadata["height"]
            
            st.success(f"✅ Loaded video: {frame_count} frames @ {fps:.1f} FPS ({width}x{height})")
            
            # Initialize components
            detector_name = "yolov8n" if "YOLOv8n" in detector_choice else "mobilenet_ssd"
            detector = Detector(detector_name, conf_thresh, nms_thresh)
            
            tracker_name = tracker_choice.lower()
            tracker = Tracker(tracker_name)
            
            feature_extractor = FeatureExtractor(fps, width, height, features_selected)
            
            anomaly_name_map = {
                "Rule-Based": "rule_based",
                "One-Class SVM": "ocsvm",
                "Isolation Forest": "isolation_forest"
            }
            anomaly_detector = AnomalyDetector(anomaly_name_map[anomaly_choice])
            
            # Process frames
            cap = cv2.VideoCapture(video_path)
            process_every_n = 3
            
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_idx = 0
            processed_frame_idx = 0
            frame_times = []
            anomaly_events_local = []
            frame_data_list = []
            speed_data_by_frame = {}
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_idx += 1
                
                # Process every 3rd frame
                if frame_idx % process_every_n != 0:
                    continue
                
                processed_frame_idx += 1
                frame_start = time.time()
                
                # Detect vehicles
                detections = detector.detect(frame)
                
                # Track vehicles
                tracks = tracker.update(detections, processed_frame_idx)
                
                # Extract features
                features = feature_extractor.extract(tracks, processed_frame_idx)
                frame_data_list.append(features)
                
                # Detect anomalies
                anomaly_result = anomaly_detector.predict(features)
                
                # Collect anomalous track IDs
                anomalous_ids = set()
                for track in tracks:
                    if track.get("age", 0) >= feature_extractor.warmup_frames:
                        if anomaly_result["is_anomaly"]:
                            anomalous_ids.add(track["track_id"])
                
                # Prepare speed data for visualization
                speed_data = {}
                for track in tracks:
                    tid = track.get("track_id")
                    if tid and tid in feature_extractor.track_history:
                        speeds = feature_extractor.track_history[tid]["speeds"]
                        if speeds:
                            speed_data[tid] = np.mean(speeds)
                
                speed_data_by_frame[processed_frame_idx] = speed_data
                
                # Draw annotations
                annotated_frame = draw_tracks(
                    frame, tracks, anomalous_ids,
                    show_boxes, show_ids, show_speed,
                    speed_data
                )
                
                # Draw anomaly banner if triggered
                if anomaly_result["is_anomaly"]:
                    anomalous_track = "(mixed)"
                    annotated_frame = draw_anomaly_banner(
                        annotated_frame,
                        f"⚠ ANOMALY — {anomaly_result['triggered_feature']} — Frame {processed_frame_idx}"
                    )
                    
                    # Log anomaly event
                    anomaly_events_local.append({
                        "Frame": processed_frame_idx,
                        "Track ID": "multiple" if len(anomalous_ids) > 1 else list(anomalous_ids)[0] if anomalous_ids else "unknown",
                        "Class": "Anomalous Behavior",
                        "Feature Triggered": anomaly_result["triggered_feature"] or "Unknown",
                        "Timestamp(s)": round(processed_frame_idx / fps, 2)
                    })
                    
                    st.error(f"⚠ ANOMALY DETECTED — Frame {processed_frame_idx} — {anomaly_result['triggered_feature']}")
                
                # Display frame
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, use_container_width=True)
                
                # Frame time
                frame_end = time.time()
                frame_time = frame_end - frame_start
                frame_times.append(frame_time)
                
                fps_current = 1.0 / frame_time if frame_time > 0 else 0
                
                # Stats
                with stats_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Frame", f"{processed_frame_idx} / {frame_count // process_every_n}")
                    col2.metric("FPS", f"{fps_current:.1f}")
                    col3.metric("Active Tracks", len(tracks))
                    col4.metric("Anomalies", len(anomaly_events_local))
                
                # Progress bar
                progress = processed_frame_idx / (frame_count // process_every_n)
                progress_bar.progress(min(1.0, progress))
            
            cap.release()
            
            # Save results to session state
            st.session_state.results = {
                "frame_data": frame_data_list,
                "anomaly_events": anomaly_events_local,
                "frame_times": frame_times,
                "speed_data_by_frame": speed_data_by_frame,
                "metadata": metadata,
                "config": {
                    "detector": detector_choice,
                    "conf_thresh": conf_thresh,
                    "nms_thresh": nms_thresh,
                    "tracker": tracker_choice,
                    "features": features_selected,
                    "anomaly_method": anomaly_choice
                }
            }
            
            st.success("✅ Pipeline processing complete!")
            st.session_state.processing = False
            
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            st.session_state.processing = False
    
    elif not st.session_state.processing:
        st.info("👆 Upload a video and configure the pipeline in the sidebar, then click 'Run Pipeline'")

# ============================================================================
# TAB 2: FEATURE CHARTS
# ============================================================================
with tab2:
    if st.session_state.results:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        frame_data = st.session_state.results["frame_data"]
        
        if "F2: Speed" in features_selected:
            speeds = [f.get("mean_speed_px_sec", 0) for f in frame_data]
            frames = list(range(len(speeds)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=frames, y=speeds,
                mode="lines",
                name="Mean Speed (px/sec)",
                line=dict(color="blue", width=2)
            ))
            
            # Add anomaly threshold line
            if speeds:
                threshold = np.mean(speeds) + 2 * np.std(speeds)
                fig.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Anomaly Threshold"
                )
            
            fig.update_layout(
                title="Speed Proxy Over Time",
                xaxis_title="Frame",
                yaxis_title="Speed (px/sec)",
                hovermode="x unified",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if "F3: Distance" in features_selected:
            distances = [f.get("min_distance_norm", 1.0) for f in frame_data]
            frames = list(range(len(distances)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=frames, y=distances,
                mode="lines+markers",
                name="Min Inter-Vehicle Distance (normalized)",
                line=dict(color="green", width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title="Inter-Vehicle Distance Over Time",
                xaxis_title="Frame",
                yaxis_title="Distance (normalized, 0-1)",
                hovermode="x unified",
                height=400
            )
            fig.add_annotation(
                text="Image-space proxy only. Not physical distance.",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=10, color="red")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if "F1: Density/Flow" in features_selected:
            vehicle_counts = [f.get("vehicle_count", 0) for f in frame_data[::10]]
            frames = list(range(0, len(frame_data), 10))[:len(vehicle_counts)]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=frames, y=vehicle_counts,
                name="Vehicle Count",
                marker=dict(color="orange")
            ))
            
            fig.update_layout(
                title="Vehicle Count Per Frame (sampled every 10 frames)",
                xaxis_title="Frame",
                yaxis_title="Vehicle Count",
                hovermode="x unified",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Run the pipeline first to generate feature charts")

# ============================================================================
# TAB 3: RESULTS SUMMARY
# ============================================================================
with tab3:
    if st.session_state.results:
        results = st.session_state.results
        config = results["config"]
        metadata = results["metadata"]
        frame_data = results["frame_data"]
        anomaly_events = results["anomaly_events"]
        frame_times = results["frame_times"]
        
        st.subheader("Pipeline Configuration Used")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Block A — Detector:**")
            st.metric("Selected Detector", config["detector"])
            st.metric("Confidence Threshold", f"{config['conf_thresh']:.2f}")
            st.metric("NMS Threshold", f"{config['nms_thresh']:.2f}")
            
            st.write("**Block B — Tracker:**")
            st.metric("Selected Tracker", config["tracker"])
            
            st.write("**Block C — Features:**")
            st.write(", ".join(config["features"]))
        
        with col2:
            st.write("**Performance Metrics:**")
            
            st.metric("Total Frames Processed", len(frame_data))
            
            # Count unique tracks
            unique_tracks = set()
            for frame in frame_data:
                # Simplified: count from anomaly events
                pass
            
            st.metric("Processing FPS", f"{1.0 / np.mean(frame_times):.1f}" if frame_times else "N/A")
            
            st.metric("Anomalies Detected", len(anomaly_events))
            
            if frame_times:
                st.metric("Avg Latency (ms)", f"{np.mean(frame_times) * 1000:.1f}")
                st.metric("p95 Latency (ms)", f"{np.percentile(frame_times, 95) * 1000:.1f}")
            
            st.write("**Block D — Anomaly Method:**")
            st.metric("Method", config["anomaly_method"])
        
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
                    mime="application/json"
                )
        
        with col2:
            if frame_data:
                df_features = pd.DataFrame(frame_data)
                csv_data = df_features.to_csv(index=False)
                st.download_button(
                    label="📥 Download features.csv",
                    data=csv_data,
                    file_name=f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    else:
        st.info("👈 Run the pipeline first to see results summary")
