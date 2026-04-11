"""
1_Pipeline.py — Live video processing pipeline
Student: MANJOO Ameera Najla | M01014463
Project: CST3990 Undergraduate Individual Project
"""

import os
import sys
import json
import random
import time
import warnings
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

warnings.filterwarnings("ignore")

# ── Fix import path so pages/ can find streamlit_app/ siblings ─────────────────
_PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR   = os.path.normpath(os.path.join(_PAGES_DIR, ".."))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from components import Detector, Tracker, FeatureExtractor, AnomalyDetector
from utils.video_utils import get_video_metadata, save_uploaded_file
from utils.drawing_utils import draw_tracks, draw_anomaly_banner

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pipeline — CST3990",
    page_icon="🎥",
    layout="wide",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.2rem; }
    div[data-testid="metric-container"] {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 0.4rem 0.6rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎥 Traffic Analysis Pipeline")
st.caption(
    "Upload a traffic video, configure each pipeline block, then run to see "
    "live detection, feature extraction, and anomaly detection in action."
)

# ── Session state ──────────────────────────────────────────────────────────────
_SS_DEFAULTS = {
    "results":          None,
    "annotated_frames": [],
    "processing":       False,
    "last_file_name":   None,
}
for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _image_fill_kwargs():
    """
    Return Streamlit image sizing kwargs compatible with the installed version.
    Streamlit 1.32 uses `use_column_width`; newer releases may prefer
    `use_container_width`.
    """
    try:
        import inspect

        image_params = inspect.signature(st.image).parameters
        if "use_container_width" in image_params:
            return {"use_container_width": True}
    except Exception:
        pass

    return {"use_column_width": True}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")

    # ── Video upload ───────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📂 Upload traffic video (.mp4)",
        type=["mp4"],
        help="Select an .mp4 video file from your local machine.",
    )

    if uploaded_file is not None:
        if st.session_state.last_file_name != uploaded_file.name:
            st.session_state.last_file_name   = uploaded_file.name
            st.session_state.results          = None
            st.session_state.annotated_frames = []

        # Quick metadata preview (reads header only via temp file)
        try:
            _preview_path = save_uploaded_file(uploaded_file)
            try:
                _meta_preview = get_video_metadata(_preview_path)
            finally:
                if os.path.exists(_preview_path):
                    os.unlink(_preview_path)
            st.success(
                f"✅ **{uploaded_file.name}**  \n"
                f"{_meta_preview['frame_count']} frames · "
                f"{_meta_preview['fps']:.1f} FPS · "
                f"{_meta_preview['duration_sec']:.1f}s  \n"
                f"{_meta_preview['width']}×{_meta_preview['height']} px"
            )
        except Exception:
            st.success(f"✅ Loaded: **{uploaded_file.name}**")

    st.divider()

    # ── Block A ────────────────────────────────────────────────────────────────
    st.markdown("**🔍 Block A — Detector**")
    st.caption("Research frozen: **YOLOv8n** (mAP@0.5 = 0.6674, Day 6)")
    detector_choice = st.radio(
        "Detector:",
        ["YOLOv8n (fine-tuned)", "SSD300 (COCO)"],
        key="detector_choice",
        help="YOLOv8n is the frozen research selection (fine-tuned on UA-DETRAC). "
             "SSD300 uses COCO pre-training only.",
    )
    col_conf, col_nms = st.columns(2)
    with col_conf:
        conf_thresh = st.slider("Confidence", 0.10, 0.90, 0.25, 0.05,
                                help="Minimum detection confidence.")
    with col_nms:
        nms_thresh = st.slider("NMS IoU", 0.10, 0.90, 0.45, 0.05,
                               help="NMS suppression threshold.")

    st.divider()

    # ── Block B ────────────────────────────────────────────────────────────────
    st.markdown("**🔗 Block B — Tracker**")
    st.caption("Research frozen: **SORT** (IDF1 = 0.0412, Day 9)")
    tracker_choice = st.radio(
        "Tracker:",
        ["SORT", "DeepSORT", "ByteTrack"],
        key="tracker_choice",
        help="SORT is the frozen selection. DeepSORT requires VeRi-776 ReID weights.",
    )
    _REPO = os.path.normpath(os.path.join(_APP_DIR, ".."))
    _reid_path = os.path.join(_REPO, "models", "osnet_x1_0_veri_776.pth")
    if tracker_choice == "DeepSORT" and not os.path.exists(_reid_path):
        st.warning(
            "⚠️ DeepSORT requires `models/osnet_x1_0_veri_776.pth` (VeRi-776 weights).  \n"
            "File not found — please use **SORT** (frozen selection) instead."
        )

    st.divider()

    # ── Block C ────────────────────────────────────────────────────────────────
    st.markdown("**📐 Block C — Feature Groups**")
    st.caption("Research frozen: **F2_only** (AUROC = 0.9935, Day 12)")
    features_selected = st.multiselect(
        "Feature groups to extract:",
        ["F1: Density/Flow", "F2: Speed", "F3: Distance/Proximity"],
        default=["F1: Density/Flow", "F2: Speed", "F3: Distance/Proximity"],
        key="features_selected",
        help="F2 (Speed) is the frozen selection. F1 and F3 add density/proximity context.",
    )

    st.divider()

    # ── Block D ────────────────────────────────────────────────────────────────
    st.markdown("**🚨 Block D — Anomaly Detection**")
    anomaly_choice = st.radio(
        "Method:",
        ["Rule-Based", "One-Class SVM", "Isolation Forest"],
        key="anomaly_choice",
        help="OC-SVM is the frozen best (FAR=0.0283). "
             "Isolation Forest failed FAR gate (FAR=0.3354).",
    )
    st.caption(
        "OC-SVM: nu=0.01, F2_only (Block C Day 12). "
        "Pre-trained model loaded from `logs/block_c/ocsvm_trained_best.pkl` if available."
    )

    st.divider()

    # ── Visualisation ──────────────────────────────────────────────────────────
    st.markdown("**🎨 Visualisation**")
    show_boxes = st.checkbox("Bounding boxes", value=True)
    show_ids   = st.checkbox("Track IDs",      value=True)
    show_speed = st.checkbox("Speed labels",   value=True)

    # Frame skip control
    process_every_n = st.select_slider(
        "Process every N-th frame:",
        options=[1, 2, 3, 5, 10],
        value=3,
        help="Higher = faster processing; lower = more accurate. "
             "3 is the research default.",
    )

    st.divider()

    run_button = st.button(
        "▶ Run Pipeline",
        disabled=(uploaded_file is None),
        use_container_width=True,
        type="primary",
    )
    if uploaded_file is None:
        st.caption("⬆️ Upload a video file first.")

    if st.session_state.results is not None:
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state.results          = None
            st.session_state.annotated_frames = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PROCESSING — runs at page level (not inside a tab)
# ══════════════════════════════════════════════════════════════════════════════
_TRACKER_NAME_MAP = {
    "SORT":       "sort",
    "DeepSORT":   "deepsort",
    "ByteTrack":  "bytetrack",
}
_ANOMALY_NAME_MAP = {
    "Rule-Based":       "rule_based",
    "One-Class SVM":    "ocsvm",
    "Isolation Forest": "isolation_forest",
}

if run_button and uploaded_file is not None:
    st.session_state.processing       = True
    st.session_state.results          = None
    st.session_state.annotated_frames = []

    video_path = None

    # Status containers at the top of the main area
    status_box   = st.empty()
    progress_bar = st.progress(0, text="Initialising…")

    # Live display: frame + real-time stats side by side
    live_col_frame, live_col_stats = st.columns([3, 1])
    frame_ph = live_col_frame.empty()
    stats_ph = live_col_stats.empty()

    try:
        # Save upload to temp file
        video_path = save_uploaded_file(uploaded_file)
        metadata   = get_video_metadata(video_path)
        fps         = metadata["fps"]
        frame_count = metadata["frame_count"]
        width       = metadata["width"]
        height      = metadata["height"]

        status_box.info(
            f"⚙️ Processing **{uploaded_file.name}** — "
            f"{frame_count} frames @ {fps:.1f} FPS ({width}×{height})  \n"
            f"Processing every {process_every_n}rd frame — "
            f"estimated {frame_count // process_every_n} frames to process."
        )

        # Build pipeline components
        detector_name = "yolov8n" if "YOLOv8n" in detector_choice else "ssd300"
        detector = Detector(detector_name, conf_thresh, nms_thresh)

        tracker_name = _TRACKER_NAME_MAP.get(tracker_choice, "sort")
        tracker = Tracker(tracker_name)

        feature_extractor = FeatureExtractor(fps, width, height, features_selected)

        anomaly_method   = _ANOMALY_NAME_MAP.get(anomaly_choice, "rule_based")
        anomaly_detector = AnomalyDetector(anomaly_method)

        # OC-SVM pre-trained model status
        if anomaly_choice == "One-Class SVM":
            exact_f2_only = set(features_selected) == {"F2: Speed"} and len(features_selected) == 1
            if anomaly_detector._using_pretrained and exact_f2_only:
                st.success(
                    "✅ Block C pre-trained OC-SVM loaded (`ocsvm_trained_best.pkl`). "
                    "Using F2_only features (vel_px_sec, vel_px_sec_smooth) — "
                    "Block C research-quality inference."
                )
            else:
                st.info(
                    "ℹ️ Live OC-SVM will train on the selected feature groups during "
                    f"warmup (first 200 frames, nu=0.01). Selected groups: {', '.join(features_selected)}."
                )

        # Frame processing loop
        cap = cv2.VideoCapture(video_path)

        frame_idx            = 0
        processed_idx        = 0
        frame_times          = []
        anomaly_events_list  = []
        frame_data_list      = []
        speed_data_by_frame  = {}
        annotated_frames_buf = []   # (frame_idx, rgb_image, is_anomaly)
        total_to_process     = max(1, frame_count // process_every_n)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % process_every_n != 0:
                    continue

                processed_idx += 1
                t0 = time.perf_counter()

                # ── Detect → Track → Extract → Predict ────────────────────────
                detections     = detector.detect(frame)
                tracks         = tracker.update(detections, processed_idx, frame=frame)
                features       = feature_extractor.extract(tracks, processed_idx)
                frame_data_list.append(features)
                anomaly_result = anomaly_detector.predict(features)

                # Identify anomalous track IDs (post-warmup only)
                anomalous_ids = set()
                if anomaly_result["is_anomaly"]:
                    for t in tracks:
                        if t.get("age", 0) >= feature_extractor.warmup_frames:
                            anomalous_ids.add(t["track_id"])

                # Per-track speed for label overlay
                speed_data = {}
                for t in tracks:
                    tid = t.get("track_id")
                    if tid and tid in feature_extractor.track_history:
                        spds = feature_extractor.track_history[tid]["speeds"]
                        if spds:
                            speed_data[tid] = float(np.mean(spds[-3:]))
                speed_data_by_frame[processed_idx] = speed_data

                # ── Draw ───────────────────────────────────────────────────────
                annotated = draw_tracks(
                    frame, tracks, anomalous_ids,
                    show_boxes, show_ids, show_speed, speed_data,
                )
                feat_label = anomaly_result.get("triggered_feature") or "Unknown"
                if anomaly_result["is_anomaly"]:
                    annotated = draw_anomaly_banner(
                        annotated,
                        f"⚠ ANOMALY — {feat_label} — Frame {processed_idx}",
                    )
                    anomaly_events_list.append({
                        "Frame":             processed_idx,
                        "Timestamp (s)":     round(processed_idx / fps, 2),
                        "Track IDs":         (
                            str(list(anomalous_ids)) if anomalous_ids else "unknown"
                        ),
                        "Feature Triggered": feat_label,
                        "Score":             round(float(anomaly_result.get("score", 0.0)), 4),
                        "Method":            anomaly_result.get("method", anomaly_method),
                    })

                # Store key frames for post-processing gallery
                # Keep: every Nth (for overview) + all anomaly frames
                _is_anom = anomaly_result["is_anomaly"]
                _store_interval = max(1, total_to_process // 20)   # ~20 overview frames
                if (processed_idx % _store_interval == 0) or _is_anom:
                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    annotated_frames_buf.append({
                        "frame_idx": processed_idx,
                        "image":     frame_rgb,
                        "anomaly":   _is_anom,
                        "n_tracks":  len(tracks),
                    })

                # ── Live display update ────────────────────────────────────────
                frame_rgb_live = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_ph.image(
                    frame_rgb_live,
                    caption=f"Frame {processed_idx} / {total_to_process}",
                    **_image_fill_kwargs(),
                )

                t1 = time.perf_counter()
                elapsed = t1 - t0
                frame_times.append(elapsed)
                fps_live = 1.0 / elapsed if elapsed > 0 else 0.0

                with stats_ph.container():
                    st.metric("Frame",    f"{processed_idx}/{total_to_process}")
                    st.metric("Live FPS", f"{fps_live:.1f}")
                    st.metric("Tracks",   len(tracks))
                    st.metric("Anomalies", len(anomaly_events_list))
                    if anomaly_events_list:
                        st.error(f"⚠ {len(anomaly_events_list)} event(s)")

                progress_pct = processed_idx / total_to_process
                progress_bar.progress(
                    min(1.0, progress_pct),
                    text=f"Frame {processed_idx} / {total_to_process}",
                )

        finally:
            cap.release()

        progress_bar.progress(1.0, text="Processing complete ✓")
        status_box.success(
            f"✅ Pipeline complete — processed {processed_idx} frames  |  "
            f"{len(anomaly_events_list)} anomaly event(s) detected  |  "
            f"avg {1.0/max(float(np.mean(frame_times)), 1e-9):.1f} FPS"
            if frame_times else
            f"✅ Pipeline complete — processed {processed_idx} frames"
        )

        # Store results in session state
        st.session_state.results = {
            "frame_data":          frame_data_list,
            "anomaly_events":      anomaly_events_list,
            "frame_times":         frame_times,
            "speed_data_by_frame": speed_data_by_frame,
            "metadata":            metadata,
            "config": {
                "detector":       detector_choice,
                "conf_thresh":    conf_thresh,
                "nms_thresh":     nms_thresh,
                "tracker":        tracker_choice,
                "features":       features_selected,
                "anomaly_method": anomaly_choice,
                "process_every":  process_every_n,
            },
        }
        st.session_state.annotated_frames = annotated_frames_buf

    except Exception as exc:
        status_box.error(f"❌ Pipeline error: {exc}")
        st.exception(exc)

    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except Exception:
                pass
        st.session_state.processing = False

# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY SECTION — shown when results are available
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.results is not None:
    results        = st.session_state.results
    frame_data     = results["frame_data"]
    anomaly_events = results["anomaly_events"]
    frame_times    = results["frame_times"]
    metadata       = results["metadata"]
    config         = results["config"]
    fps_video      = metadata["fps"]
    ann_frames     = st.session_state.annotated_frames

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs([
        "🖼️ Detection View",
        "📈 Feature Charts",
        "📋 Results Summary",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: DETECTION VIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        if ann_frames:
            anom_frames   = [f for f in ann_frames if f["anomaly"]]
            normal_frames = [f for f in ann_frames if not f["anomaly"]]

            # ── Anomaly frames first ───────────────────────────────────────────
            if anom_frames:
                st.subheader(f"⚠️ Anomaly Frames ({len(anom_frames)} detected)")
                n_cols = min(3, len(anom_frames))
                cols   = st.columns(n_cols)
                for i, fd in enumerate(anom_frames[:n_cols * 2]):
                    with cols[i % n_cols]:
                        st.image(
                            fd["image"],
                            caption=f"Frame {fd['frame_idx']} — ANOMALY",
                            **_image_fill_kwargs(),
                        )
                st.markdown("---")

            # ── Overview frames ────────────────────────────────────────────────
            st.subheader(f"📸 Frame Overview ({len(normal_frames)} sampled)")
            show_n = min(12, len(normal_frames))
            n_cols = 3
            frame_cols = st.columns(n_cols)
            for i, fd in enumerate(normal_frames[:show_n]):
                with frame_cols[i % n_cols]:
                    st.image(
                        fd["image"],
                        caption=(
                            f"Frame {fd['frame_idx']}  ·  "
                            f"{fd['n_tracks']} track(s)"
                        ),
                        **_image_fill_kwargs(),
                    )
        else:
            st.info(
                "No annotated frames were captured during this run.  \n"
                "This can happen if the video had very few frames. Try re-running."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: FEATURE CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        feat_config = config["features"]
        frames_idx  = list(range(len(frame_data)))

        st.caption(
            "Feature names aligned with Block C schema v1.1. "
            "Speed = F2 (vel_px_sec), Distance = F3 (inter_vehicle_dist_norm proxy), "
            "Dwell = F3 (dwell_time_sec), Density = F1 (vehicle_count)."
        )

        # Anomaly event frame indices for marker overlay
        anom_frame_idxs = [e["Frame"] for e in anomaly_events]

        # ── F2: Speed ──────────────────────────────────────────────────────────
        if "F2: Speed" in feat_config:
            speeds = [float(f.get("mean_speed_px_sec", 0.0)) for f in frame_data]

            fig_sp = go.Figure()
            fig_sp.add_trace(go.Scatter(
                x=frames_idx, y=speeds,
                mode="lines", name="vel_px_sec (mean)",
                line=dict(color="#2ca02c", width=2),
            ))

            if speeds and max(speeds) > 0:
                spd_mean = float(np.mean(speeds))
                spd_std  = float(np.std(speeds))
                thresh   = spd_mean + 2.0 * spd_std
                fig_sp.add_hline(
                    y=thresh, line_dash="dash", line_color="#e53e3e",
                    annotation_text=f"Threshold (mean+2σ = {thresh:.0f} px/s)",
                    annotation_position="top right",
                )

            # Mark anomaly events
            if anom_frame_idxs:
                anom_speeds = [
                    speeds[i] if i < len(speeds) else 0.0
                    for i in anom_frame_idxs
                ]
                fig_sp.add_trace(go.Scatter(
                    x=anom_frame_idxs, y=anom_speeds,
                    mode="markers", name="Anomaly event",
                    marker=dict(color="#e53e3e", size=12, symbol="x",
                                line=dict(width=2, color="#e53e3e")),
                ))

            fig_sp.update_layout(
                title="F2 — Speed Proxy Over Time (vel_px_sec, image-space, NOT km/h)",
                xaxis_title="Processed Frame Index",
                yaxis_title="Mean speed (px/sec)",
                height=400,
                hovermode="x unified",
                plot_bgcolor="#f8fafc",
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_sp, use_container_width=True)
            st.caption(
                "speed_window = int(0.2 × fps) frames. "
                "Image-space proxy — NOT calibrated to km/h. "
                "Red dashed line = rule-based anomaly threshold (mean + 2σ). "
                "✕ markers = frames with anomaly events."
            )

        # ── F3: Inter-Vehicle Distance ─────────────────────────────────────────
        if "F3: Distance/Proximity" in feat_config:
            distances = [float(f.get("min_distance_norm", 1.0)) for f in frame_data]
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Scatter(
                x=frames_idx, y=distances,
                mode="lines", name="inter_vehicle_dist_norm",
                line=dict(color="#9467bd", width=2),
            ))
            fig_dist.update_layout(
                title=(
                    "F3 — Minimum Inter-Vehicle Distance "
                    "(inter_vehicle_dist_norm, normalised by frame diagonal ≈ 905 px)"
                ),
                xaxis_title="Processed Frame Index",
                yaxis_title="Normalised distance (0–1)",
                height=380,
                hovermode="x unified",
                plot_bgcolor="#f8fafc",
            )
            fig_dist.add_annotation(
                text="⚠️ Image-space proxy — NOT physical distance",
                xref="paper", yref="paper",
                x=0.5, y=-0.20, showarrow=False,
                font=dict(size=11, color="#e53e3e"),
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption(
                "Normalised by frame diagonal (640×√2 ≈ 905 px). "
                "Value = 1.0 when fewer than 2 vehicles are tracked. "
                "Perspective effects mean this is a relative behavioural indicator only."
            )

            # ── F3: Dwell Time ─────────────────────────────────────────────────
            dwell = [float(f.get("mean_dwell_sec", 0.0)) for f in frame_data]
            if any(d > 0 for d in dwell):
                fig_dwell = go.Figure(go.Scatter(
                    x=frames_idx, y=dwell,
                    mode="lines", name="dwell_time_sec",
                    line=dict(color="#d62728", width=2),
                ))
                fig_dwell.update_layout(
                    title="F3 — Mean Track Dwell Time (dwell_time_sec = cumulative time in scene)",
                    xaxis_title="Processed Frame Index",
                    yaxis_title="Dwell time (seconds)",
                    height=340,
                    hovermode="x unified",
                    plot_bgcolor="#f8fafc",
                )
                st.plotly_chart(fig_dwell, use_container_width=True)
                st.caption(
                    "dwell_time_sec = (last_frame − first_frame) / fps for each active track. "
                    "Persistently rising values may indicate stationary or slow vehicles."
                )

        # ── F1: Density / ROI Occupancy ────────────────────────────────────────
        if "F1: Density/Flow" in feat_config:
            step = max(1, len(frame_data) // 100)   # ~100 bars max for readability
            sframes = frames_idx[::step]
            vc  = [frame_data[i].get("vehicle_count", 0) for i in range(0, len(frame_data), step)]
            roi = [frame_data[i].get("roi_occupancy", 0.0) for i in range(0, len(frame_data), step)]

            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Bar(
                x=sframes, y=vc,
                name="vehicle_count (F1)",
                marker_color="#ff7f0e",
                yaxis="y",
            ))
            fig_f1.add_trace(go.Scatter(
                x=sframes, y=roi,
                mode="lines", name="roi_occupancy (F1)",
                line=dict(color="#1f77b4", width=2),
                yaxis="y2",
            ))
            fig_f1.update_layout(
                title="F1 — Vehicle Count & ROI Occupancy Over Time",
                xaxis_title="Processed Frame Index",
                yaxis=dict(title="Vehicle Count"),
                yaxis2=dict(title="ROI Occupancy (0–1)", overlaying="y", side="right"),
                height=400,
                hovermode="x unified",
                plot_bgcolor="#f8fafc",
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_f1, use_container_width=True)
            st.caption(
                "roi_occupancy estimated as (n_tracks × 80×60) / (frame_w × frame_h). "
                "Simplified proxy — exact bbox areas not used in streaming demo."
            )

        # ── Anomaly Score Timeline ─────────────────────────────────────────────
        if anomaly_events:
            scores    = [e["Score"] for e in anomaly_events]
            a_times   = [e["Timestamp (s)"] for e in anomaly_events]
            a_labels  = [e.get("Feature Triggered", "") for e in anomaly_events]

            fig_scr = go.Figure(go.Scatter(
                x=a_times, y=scores,
                mode="lines+markers",
                name="Anomaly Score",
                line=dict(color="#e53e3e", width=2),
                marker=dict(size=10, color="#e53e3e"),
                text=a_labels,
                hovertemplate="t=%{x:.2f}s<br>Score=%{y:.4f}<br>%{text}<extra></extra>",
            ))
            fig_scr.update_layout(
                title="Anomaly Score at Detection Events",
                xaxis_title="Video time (s)",
                yaxis_title="Anomaly score (higher = more anomalous)",
                height=320,
                plot_bgcolor="#f8fafc",
            )
            st.plotly_chart(fig_scr, use_container_width=True)
        else:
            st.info("✅ No anomaly events detected — anomaly score timeline not available.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: RESULTS SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Pipeline Configuration Used")
        r_col1, r_col2 = st.columns(2)

        with r_col1:
            st.markdown("**Block A — Detector**")
            st.metric("Selected Detector",    config["detector"])
            st.metric("Confidence Threshold", f"{config['conf_thresh']:.2f}")
            st.metric("NMS Threshold",        f"{config['nms_thresh']:.2f}")
            st.markdown("**Block B — Tracker**")
            st.metric("Selected Tracker", config["tracker"])
            st.markdown("**Block C — Feature Groups**")
            st.write(
                ", ".join(config["features"]) if config["features"] else "*(none selected)*"
            )
            st.markdown("**Block D — Anomaly Method**")
            st.metric("Method", config["anomaly_method"])

        with r_col2:
            st.markdown("**Performance Metrics**")
            st.metric("Video FPS",            f"{metadata['fps']:.1f}")
            st.metric("Total Frames (video)", metadata["frame_count"])
            st.metric("Frames Processed",     len(frame_data))
            st.metric("Frame Skip",           f"Every {config.get('process_every', 3)}rd frame")
            if frame_times:
                proc_fps     = 1.0 / float(np.mean(frame_times))
                avg_lat_ms   = float(np.mean(frame_times)) * 1000
                p95_lat_ms   = float(np.percentile(frame_times, 95)) * 1000
                st.metric("Processing FPS",   f"{proc_fps:.1f}")
                st.metric("Avg Latency",      f"{avg_lat_ms:.1f} ms")
                st.metric("p95 Latency",      f"{p95_lat_ms:.1f} ms")
            st.metric("Anomaly Events",       len(anomaly_events))

        st.markdown("---")

        # ── Feature statistics ─────────────────────────────────────────────────
        if frame_data:
            st.subheader("Feature Summary Statistics")
            df_feat = pd.DataFrame(frame_data)
            stat_cols = [c for c in [
                "vehicle_count", "roi_occupancy",
                "mean_speed_px_sec", "min_distance_norm", "mean_dwell_sec",
            ] if c in df_feat.columns]
            if stat_cols:
                st.dataframe(
                    df_feat[stat_cols]
                    .describe()
                    .round(4)
                    .rename(columns={
                        "mean_speed_px_sec": "vel_px_sec (approx)",
                        "min_distance_norm": "inter_vehicle_dist_norm (proxy)",
                        "mean_dwell_sec":    "dwell_time_sec (mean)",
                    }),
                    use_container_width=True,
                )

        st.markdown("---")
        st.subheader("Anomaly Event Log")

        if anomaly_events:
            df_ev = pd.DataFrame(anomaly_events)
            st.dataframe(df_ev, use_container_width=True)
            st.caption(
                f"Total: **{len(anomaly_events)}** anomaly event(s) detected in this video."
            )
        else:
            st.success("✅ No anomalies detected in this video.")

        st.markdown("---")
        st.subheader("Download Results")

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            if anomaly_events:
                _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📥 Download anomaly_events.json",
                    data=json.dumps(anomaly_events, indent=2),
                    file_name=f"anomaly_events_{_ts}.json",
                    mime="application/json",
                    help="Download the full anomaly event log as JSON.",
                )
        with dl_col2:
            if frame_data:
                _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                df_dl = pd.DataFrame(frame_data).rename(columns={
                    "mean_speed_px_sec": "vel_px_sec",
                    "min_distance_norm": "inter_vehicle_dist_norm",
                    "mean_dwell_sec":    "dwell_time_sec",
                })
                st.download_button(
                    label="📥 Download features.csv",
                    data=df_dl.to_csv(index=False),
                    file_name=f"features_{_ts}.csv",
                    mime="text/csv",
                    help="Download per-frame feature data as CSV (Block C schema v1.1 names).",
                )

# ══════════════════════════════════════════════════════════════════════════════
# IDLE STATE — shown when no results and not processing
# ══════════════════════════════════════════════════════════════════════════════
elif not st.session_state.processing and not run_button:
    st.markdown("---")
    col_info1, col_info2 = st.columns([2, 1])
    with col_info1:
        st.markdown("""
### 👈 Getting Started

1. **Upload** a traffic video (`.mp4`) using the sidebar
2. **Configure** each pipeline block:
   - Block A: choose detector & thresholds
   - Block B: choose tracker
   - Block C: select feature groups
   - Block D: choose anomaly detection method
3. Click **▶ Run Pipeline**
4. View results in the tabs:
   - **Detection View** — annotated frame gallery
   - **Feature Charts** — speed, distance, density over time
   - **Results Summary** — metrics, event log, downloads

> The research-frozen selection is shown as a caption under each option.
> All processing runs on CPU — processing speed depends on video length.
""")
    with col_info2:
        st.markdown("""
### 🔒 Frozen Selections

| Block | Selection |
|---|---|
| A | YOLOv8n |
| B | SORT |
| C | F2_only |
| D | OC-SVM |

> See the **Comparison** page for full evaluation results and the **About** page for methodology details.
""")
