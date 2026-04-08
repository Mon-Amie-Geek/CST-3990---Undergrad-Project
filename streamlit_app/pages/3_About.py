"""
3_About.py

About page showing project details, frozen pipeline selections, and important disclaimers.
Student: MANJOO Ameera Najla | M01014463
"""

import os
import json
import streamlit as st

st.set_page_config(page_title="About - CST3990 Traffic Anomaly Detection", layout="wide")

st.title("📋 About This Project")
st.markdown("---")

# ── Path helpers ──────────────────────────────────────────────────────────────
_PAGES_DIR = os.path.dirname(__file__)
_REPO_ROOT  = os.path.normpath(os.path.join(_PAGES_DIR, "..", ".."))
_LOGS_DIR   = os.path.join(_REPO_ROOT, "logs")

def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

_ba = _load_json(os.path.join(_LOGS_DIR, "block_a_results.json"))
_bb = _load_json(os.path.join(_LOGS_DIR, "block_b_results.json"))

# ============================================================================
# PROJECT INFORMATION
# ============================================================================
st.header("Project Details")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Student Information")
    st.write("**Name:** MANJOO Ameera Najla")
    st.write("**Student ID:** M01014463")
    st.write("**University:** Middlesex University Mauritius")
    st.write("**Supervisor:** Mr Karel Veerabudren")
with col2:
    st.subheader("Course Information")
    st.write("**Course Code:** CST3990")
    st.write("**Course Title:** Undergraduate Individual Project")
    st.write("**Project Type:** Final Year Dissertation")

st.markdown("---")

# ============================================================================
# PROJECT TITLE
# ============================================================================
st.header("Project Title")
st.markdown("""
### A Comparative Study of Computer Vision Pipeline Components
### for Real-Time Traffic Analysis and Anomaly Detection

This project investigates and compares different components of a computer vision pipeline
for analysing traffic video. The system includes configurable options for:

| Block | Component | Candidates | Status |
|---|---|---|---|
| **A** | Detection | YOLOv8n (fine-tuned) vs SSD300 (COCO) | ✅ Complete — Day 6 |
| **B** | Tracking | SORT vs DeepSORT vs ByteTrack | ✅ Complete — Day 9 |
| **C** | Feature Extraction | F1 (Density/Flow), F2 (Speed), F2+F3, All | ✅ Complete — Day 12 |
| **D** | Anomaly Detection | Rule-Based, OC-SVM, Isolation Forest | ⏳ Pending |
""")

st.markdown("---")

# ============================================================================
# FROZEN PIPELINE SELECTIONS
# ============================================================================
st.header("🔒 Frozen Pipeline Selections")
st.markdown("Each block selects one component that is then locked for all downstream blocks.")

sel_col1, sel_col2, sel_col3, sel_col4 = st.columns(4)

with sel_col1:
    st.subheader("Block A")
    if _ba:
        _yolo = _ba["detectors"]["yolov8n"]
        st.success("**Frozen: YOLOv8n**")
        st.write(f"mAP@0.5: **{_yolo['mAP50']:.4f}**")
        st.write(f"mAP@0.5:0.95: {_yolo['mAP50-95']:.4f}")
        st.write(f"Precision: {_yolo['precision']:.4f}")
        st.write(f"Recall: {_yolo['recall']:.4f}")
        st.write(f"FPS p50: {_yolo['fps_p50']} (T4 GPU)")
        st.write(f"Latency p95: {_yolo['latency_p95_ms']} ms")
        st.caption(f"Selection: {_ba['selection_reason']}")
    else:
        st.success("**Frozen: YOLOv8n**")
        st.write("mAP@0.5: 0.6674")
        st.write("Precision: 0.7679 | Recall: 0.7703")

with sel_col2:
    st.subheader("Block B")
    if _bb:
        _sel_trk = next((t for t in _bb["trackers"] if t.get("selected")), None)
        if _sel_trk:
            st.success(f"**Frozen: {_sel_trk['tracker'].upper()}**")
            st.write(f"IDF1: **{_sel_trk['idf1_mean']:.4f}**")
            st.write(f"MOTA: {_sel_trk['mota_mean']:.4f}")
            st.write(f"ID Switches: {_sel_trk['idsw_total']}")
            st.write(f"Fragmentations: {_sel_trk['fragmentations_total']}")
            st.write(f"FPS median: {_sel_trk['fps_median']}")
            st.caption("Selection: Highest IDF1 (identity consistency critical for Block C)")
    else:
        st.success("**Frozen: SORT**")
        st.write("IDF1: 0.0412 | MOTA: −0.7775")
        st.write("IDSW: 31 | FPS: 1129.7")

with sel_col3:
    st.subheader("Block C")
    st.success("**Frozen: F2_only**")
    st.write("Features: `vel_px_sec`, `vel_px_sec_smooth`")
    st.write("AUROC: **0.9935**")
    st.write("Silhouette: 0.9431")
    st.write("Best nu: 0.01")
    st.write("Train rows: 61,221")
    st.write("Val rows: 8,990")
    st.caption(
        "Selection: Highest OC-SVM AUROC on UA-DETRAC val split (MVI_20061). "
        "3-sigma proxy labels from train-split statistics (Fix F04)."
    )

with sel_col4:
    st.subheader("Block D")
    st.warning("**Pending evaluation**")
    st.write("Methods: Rule-Based, OC-SVM, Isolation Forest")
    st.write("Dataset: AI City Track 4")
    st.write("FAR threshold: 0.10")
    st.write("Context: runtime (per-event reporting)")
    st.caption("Config frozen 2026-04-07. Results to be populated after Block D evaluation.")

st.markdown("---")

# ============================================================================
# DATASETS
# ============================================================================
st.header("Datasets")
st.markdown("""
**UA-DETRAC** — Urban-scene Detection, Tracking and Counting dataset
- Used for: YOLOv8n fine-tuning (Block A), tracker evaluation (Block B), feature extraction (Block C)
- 10 sequences used: MVI_20011, MVI_20012, MVI_20032, MVI_20034, MVI_20035, MVI_20051, MVI_20052 (train), MVI_20061 (val), MVI_20062, MVI_20063 (test)
- Frame rate: 25 FPS (standard for all sequences)
- All bboxes stored in 640×640 coordinate space for schema consistency

**AI City Track 4 (2021)** — Anomaly ground truth (Block D only)
- Anomaly types: vehicle stalls, crashes, abandoned vehicles
- Used for: final anomaly detection evaluation against true ground truth
""")

st.markdown("---")

# ============================================================================
# MODEL DETAILS
# ============================================================================
st.header("Model Details")
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.subheader("Primary Model: YOLOv8n (`best.pt`)")
    st.markdown("""
- **Architecture:** YOLOv8 Nano (lightweight, single-class)
- **Training data:** UA-DETRAC train split (official sequences)
- **Train image size:** 960×540 px
- **Inference image size:** 640×640 px
- **Task:** Vehicle detection (single class: vehicle)
- **Performance on UA-DETRAC test (MVI_20062, MVI_20063):**
""")
    if _ba:
        _y = _ba["detectors"]["yolov8n"]
        st.markdown(f"""
  | Metric | Value |
  |---|---|
  | mAP@0.5 | **{_y['mAP50']:.4f}** |
  | mAP@0.5:0.95 | {_y['mAP50-95']:.4f} |
  | Precision | {_y['precision']:.4f} |
  | Recall | {_y['recall']:.4f} |
  | FPS p50 (T4 GPU) | {_y['fps_p50']} |
  | Latency p95 | {_y['latency_p95_ms']} ms |
""")
    else:
        st.markdown("mAP@0.5 = 0.6674 | Precision = 0.7679 | Recall = 0.7703 | FPS p50 = 47.2")

with col_m2:
    st.subheader("Alternative Model: SSD300")
    st.markdown("""
- **Source:** torchvision `SSD300_VGG16_Weights.COCO_V1` (Fix F06)
- **Pre-trained on:** COCO (no UA-DETRAC fine-tuning)
- **Inference size:** 300×300 px
- **Thresholds:** conf = 0.25, nms = 0.45 (same as YOLOv8n — Fix F23)
- **Performance on UA-DETRAC test (MVI_20062, MVI_20063):**
""")
    if _ba:
        _s = _ba["detectors"]["ssd300"]
        st.markdown(f"""
  | Metric | Value |
  |---|---|
  | mAP@0.5 | {_s['mAP50']:.4f} |
  | mAP@0.5:0.95 | {_s['mAP50-95']:.4f} |
  | Precision | {_s['precision']:.4f} |
  | Recall | {_s['recall']:.4f} |
  | FPS p50 (T4 GPU) | {_s['fps_p50']} |
  | Latency p95 | {_s['latency_p95_ms']} ms |
""")
    else:
        st.markdown("mAP@0.5 = 0.316 | Precision = 0.2801 | Recall = 0.5582 | FPS p50 = 25.6")

st.markdown("---")

# ============================================================================
# FEATURE SCHEMA (v1.1)
# ============================================================================
st.header("Feature Schema v1.1")
st.markdown("""
The feature schema has 23 columns across three groups:

| Group | Features | Notes |
|---|---|---|
| **F1 — Density/Flow** | `vehicle_count`, `roi_occupancy` | Frame-level aggregates |
| **F2 — Speed** | `vel_px_sec`, `vel_px_sec_smooth` | Per-track, 5-frame rolling mean (Fix F07) |
| **F3 — Interaction** | `inter_vehicle_dist_norm`, `dwell_time_sec`, `proximity_flag`, `proximity_count_rolling` | Added Day 11; NaN for single-track frames |
| **Metadata** | `seq_id`, `frame_idx`, `track_id`, `cx`, `cy`, `bbox_*`, `conf`, `is_interpolated`, `velocity_norm`, `track_length_real`, `split`, `tracker` | Schema bookkeeping |

**MinMaxScaler** fitted on 7 continuous features (train split, normal-only, 60,428 rows):
`vel_px_sec`, `vel_px_sec_smooth`, `vehicle_count`, `roi_occupancy`,
`inter_vehicle_dist_norm`, `dwell_time_sec`, `proximity_count_rolling`
(Note: `proximity_flag` is boolean — excluded from scaler fit by design)
""")

with st.expander("Scaler Fitted Ranges", expanded=False):
    _scaler_meta_path = os.path.join(_LOGS_DIR, "block_c", "scaler_metadata.json")
    _sm = _load_json(_scaler_meta_path)
    if _sm:
        st.json(_sm)
    else:
        st.info("Scaler metadata not found at `logs/block_c/scaler_metadata.json`")

st.markdown("---")

# ============================================================================
# IMPORTANT DISCLAIMERS
# ============================================================================
st.header("⚠️ Important Disclaimers")

st.warning("""
**Distance Measurement Disclaimer (Fix F28)**

Distance values in this system are **image-space proxies normalised by frame diagonal (640×√2 ≈ 905 px)**.

⚠️ **These values do NOT represent physical distances.**

Perspective effects and camera angle mean that pixel distance does not linearly correspond to
real-world distance. All distance metrics should be interpreted as **relative behavioural
indicators only**. See §3.8.2 of the thesis for full construct validity discussion.
""")

st.info("""
**Real-Time Operation Scope**

This system operates in **offline validation mode** on development hardware.

Real-time operation requires dedicated edge hardware such as NVIDIA Jetson Nano or similar.
The pipeline architecture is designed to support real-time deployment, but:
- Current throughput is limited by CPU constraints
- Every 3rd frame is processed in the live demo for speed
- This is a **research/validation tool**, not a production deployment
""")

st.info("""
**Block C AUROC Construct Validity (Fix F04)**

The AUROC metric used in Block C measures OC-SVM separability of **statistical outliers
within a normal-traffic distribution** (UA-DETRAC validation split). It does **not** measure
true anomaly detection performance against labelled anomaly events. Block D evaluation on
AI City Track 4 provides true precision/recall/F1 against real anomaly ground truth.
""")

st.markdown("---")

# ============================================================================
# TECHNICAL STACK
# ============================================================================
st.header("Technical Stack")
col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    st.subheader("Core Framework")
    st.write("- Python 3.11.9")
    st.write("- Streamlit 1.32.0")
    st.write("- PyTorch 2.3.1 (CPU)")
    st.write("- OpenCV 4.9.0.80")
    st.write("- NumPy 1.26.4 | Pandas 2.2.2")
with col_t2:
    st.subheader("Computer Vision")
    st.write("- YOLOv8 (Ultralytics 8.2.0)")
    st.write("- SSD300")
    st.write("- Kalman Filtering (filterpy 1.4.5)")
    st.write("- SORT / DeepSORT / ByteTrack")
    st.write("- motmetrics 1.4.0")
with col_t3:
    st.subheader("Machine Learning")
    st.write("- scikit-learn 1.5.0")
    st.write("- One-Class SVM (nu=0.01, rbf)")
    st.write("- Isolation Forest (100 trees)")
    st.write("- joblib 1.4.2")
    st.write("- Plotly 5.22.0")

st.markdown("---")

# ============================================================================
# SYSTEM REQUIREMENTS
# ============================================================================
st.header("System Requirements")
st.markdown("""
- **OS:** Windows 11 / Linux / macOS
- **Python:** 3.11+
- **RAM:** Minimum 4 GB (8 GB recommended for full Block C feature CSVs)
- **Storage:** ~2 GB for models and dependencies; ~500 MB for Block C feature CSVs
- **GPU:** Optional (all inference runs on CPU)
- **Camera:** Not required (offline video processing)
""")

st.markdown("---")

# ============================================================================
# FUTURE WORK
# ============================================================================
st.header("Future Work")
st.markdown("""
- **Real-time deployment** on NVIDIA Jetson Nano
- **Multi-camera calibration** for physical distance estimation
- **Perspective-aware** homography-based feature extraction
- **Deep learning-based** anomaly detection (e.g., spatiotemporal autoencoders)
- **Regional analysis** with ROI selection per camera
- **Multi-class detection** (cars, motorcycles, trucks, pedestrians)
""")

st.markdown("---")

st.subheader("Contact")
st.write("**Student:** MANJOO Ameera Najla (M01014463)")
st.write("**Supervisor:** Mr Karel Veerabudren")
st.write("**University:** Middlesex University Mauritius")
