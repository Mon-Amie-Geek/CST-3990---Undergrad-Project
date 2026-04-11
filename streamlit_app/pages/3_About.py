"""
3_About.py — Project details, frozen pipeline selections, and disclaimers.
Student: MANJOO Ameera Najla | M01014463
Project: CST3990 Undergraduate Individual Project
"""

import os
import json
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="About — CST3990",
    page_icon="📋",
    layout="wide",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.2rem; }
    .disclaimer-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #0dcaf0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📋 About This Project")
st.markdown("---")

# ── Path helpers ───────────────────────────────────────────────────────────────
_PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.normpath(os.path.join(_PAGES_DIR, "..", ".."))
_LOGS_DIR   = os.path.join(_REPO_ROOT, "logs")
_BLOCK_D    = os.path.join(_LOGS_DIR, "block_d")
_BLOCK_C    = os.path.join(_LOGS_DIR, "block_c")


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


_ba  = _load_json(os.path.join(_LOGS_DIR, "block_a_results.json"))
_bb  = _load_json(os.path.join(_LOGS_DIR, "block_b_results.json"))
_bd  = _load_csv(os.path.join(_BLOCK_D, "block_d_complete_results.csv"))

# ══════════════════════════════════════════════════════════════════════════════
# PROJECT INFORMATION
# ══════════════════════════════════════════════════════════════════════════════
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
    st.write("**Academic Year:** 2025–2026")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PROJECT TITLE
# ══════════════════════════════════════════════════════════════════════════════
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
| **D** | Anomaly Detection | Rule-Based, OC-SVM, Isolation Forest | ✅ Complete — Day 17 |
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FROZEN PIPELINE SELECTIONS
# ══════════════════════════════════════════════════════════════════════════════
st.header("🔒 Frozen Pipeline Selections")
st.markdown(
    "Each block selects one component that is **locked** for all downstream blocks. "
    "Selections are made based on the primary evaluation criterion for each block."
)

sel_col1, sel_col2, sel_col3, sel_col4 = st.columns(4)

# ── Block A ────────────────────────────────────────────────────────────────────
with sel_col1:
    st.subheader("Block A")
    st.success("**Frozen: YOLOv8n**")
    if _ba:
        _y = _ba["detectors"]["yolov8n"]
        st.write(f"mAP@0.5: **{_y.get('mAP50', 0):.4f}**")
        st.write(f"mAP@0.5:0.95: {_y.get('mAP50-95', 0):.4f}")
        st.write(f"Precision: {_y.get('precision', 0):.4f}")
        st.write(f"Recall: {_y.get('recall', 0):.4f}")
        st.write(f"FPS p50 (T4 GPU): {_y.get('fps_p50', '—')}")
        st.caption(_ba.get("selection_reason", "Selection: mAP delta > 0.05 threshold"))
    else:
        st.write("mAP@0.5: **0.6674**")
        st.write("Precision: 0.7679")
        st.write("Recall: 0.7703")
        st.write("FPS p50 (T4 GPU): 47.2")
        st.caption("Selection: mAP@0.5 delta = 0.3514 > 0.05 threshold (Day 6)")

# ── Block B ────────────────────────────────────────────────────────────────────
with sel_col2:
    st.subheader("Block B")
    st.success("**Frozen: SORT**")
    if _bb:
        _sel_trk = next((t for t in _bb["trackers"] if t.get("selected")), None)
        if _sel_trk:
            st.write(f"IDF1: **{_sel_trk.get('idf1_mean', 0):.4f}**")
            st.write(f"MOTA: {_sel_trk.get('mota_mean', 0):.4f}")
            st.write(f"ID Switches: {_sel_trk.get('idsw_total', '—')}")
            st.write(f"Fragmentations: {_sel_trk.get('fragmentations_total', '—')}")
            st.write(f"FPS median: {_sel_trk.get('fps_median', '—')}")
            st.caption("Selection: Highest IDF1 — identity consistency for Block C")
    else:
        st.write("IDF1: **0.0412** (highest)")
        st.write("MOTA: −0.7775")
        st.write("ID Switches: 31 | FPS: 1129.7")
        st.caption(
            "Selection: Highest IDF1 (Day 9). "
            "Negative MOTA due to dense occlusion in UA-DETRAC."
        )

# ── Block C ────────────────────────────────────────────────────────────────────
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
        "3-sigma proxy labels from train-split statistics (Fix F04). Day 12."
    )

# ── Block D ────────────────────────────────────────────────────────────────────
with sel_col4:
    st.subheader("Block D")
    if _bd is not None and not _bd.empty:
        _ocsvm_row = _bd[_bd["method"] == "ocsvm"]
        _rb_row    = _bd[_bd["method"] == "rule_based"]
        if not _ocsvm_row.empty:
            _far = float(_ocsvm_row.iloc[0]["mean_far"])
            _gate = str(_ocsvm_row.iloc[0]["far_gate_passed"]).strip() == "True"
            if _gate:
                st.success(f"**Best: OC-SVM** ✅")
            else:
                st.warning("**Best: Rule-Based**")
            st.write(f"OC-SVM FAR: **{_far:.4f}** ✅")
            if not _rb_row.empty:
                st.write(f"Rule-Based FAR: {float(_rb_row.iloc[0]['mean_far']):.4f} ✅")
            st.write("Isolation Forest: 0.3354 ❌")
            st.write("FAR gate: < 0.10")
            st.write("All 10 events detected (100%)")
            st.caption(
                "FAR gate threshold = 0.10. OC-SVM & rule_based pass. "
                "Isolation Forest flagged (Fix F14). AI City Track 4. Day 17."
            )
    else:
        st.success("**Best: OC-SVM** ✅")
        st.write("OC-SVM FAR: **0.0283** ✅")
        st.write("Rule-Based FAR: 0.0350 ✅")
        st.write("Isolation Forest: 0.3354 ❌")
        st.write("All 10 events detected (100%)")
        st.caption(
            "FAR gate threshold = 0.10. OC-SVM & rule_based pass. "
            "Isolation Forest flagged (Fix F14). AI City Track 4. Day 17."
        )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════════
st.header("Datasets")
dcol1, dcol2 = st.columns(2)

with dcol1:
    st.subheader("UA-DETRAC")
    st.markdown("""
**Purpose:** Training, tracker evaluation (Blocks A–C)

- **Sequences used:** 10 total (MVI_200xx series)
  - Train: MVI_20011, 20012, 20032, 20034, 20035, 20051, 20052 (7 sequences)
  - Val: MVI_20061 (1 sequence)
  - Test: MVI_20062, MVI_20063 (2 sequences)
- **Frame rate:** 25 FPS (all sequences)
- **Coordinate space:** Normalised to 640×640 for all blocks
- **GT format:** XML annotations (bounding boxes)
- **Normal traffic only** — used for Block C OC-SVM training (61,221 train rows)
""")

with dcol2:
    st.subheader("AI City Track 4 (2021)")
    st.markdown("""
**Purpose:** Anomaly ground truth evaluation (Block D only)

- **Anomaly types:** Vehicle stalls, crashes, abandoned vehicles
- **GT format:** Frame-based annotations (start_frame, end_frame per event)
- **Events used:** 10 annotated anomaly clips (Day 16 validation)
- **Normal clips:** 10 clips for FAR computation (Day 17)
- **Annotation coverage:** See `day15_event_coverage_audit.csv`
  for per-event annotation coverage caveats
- **Resolution note:** Mixed resolutions; frames normalised at inference
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL DETAILS
# ══════════════════════════════════════════════════════════════════════════════
st.header("Model Details")
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.subheader("Primary: YOLOv8n (`best.pt`)")
    st.markdown("""
- **Architecture:** YOLOv8 Nano (Ultralytics 8.2.0), single vehicle class
- **Training data:** UA-DETRAC train split (official sequences)
- **Train image size:** 960×540 px
- **Inference image size:** 640×640 px (input resized by YOLO)
- **Task:** Vehicle detection (single class: `vehicle`)
""")
    if _ba:
        _y = _ba["detectors"]["yolov8n"]
        st.markdown(f"""
  | Metric | Value |
  |---|---|
  | mAP@0.5 | **{_y.get('mAP50', 0):.4f}** |
  | mAP@0.5:0.95 | {_y.get('mAP50-95', 0):.4f} |
  | Precision | {_y.get('precision', 0):.4f} |
  | Recall | {_y.get('recall', 0):.4f} |
  | FPS p50 (T4 GPU) | {_y.get('fps_p50', '—')} |
  | Latency p95 | {_y.get('latency_p95_ms', '—')} ms |
""")
    else:
        st.markdown("""
  | Metric | Value |
  |---|---|
  | mAP@0.5 | **0.6674** |
  | Precision | 0.7679 |
  | Recall | 0.7703 |
  | FPS p50 (T4 GPU) | 47.2 |
  | Latency p95 | 21.2 ms |
""")

with col_m2:
    st.subheader("Alternative: SSD300 (`torchvision`)")
    st.markdown("""
- **Source:** `torchvision.models.detection.ssd300_vgg16`
  with `SSD300_VGG16_Weights.COCO_V1` (Fix F06 — no UA-DETRAC fine-tuning)
- **Pre-trained on:** COCO (80 classes)
- **Vehicle classes used:** {3=car, 4=motorcycle, 6=bus, 8=truck}
- **Inference size:** 300×300 px
- **Thresholds:** conf = 0.25, nms = 0.45 (matched to YOLOv8n — Fix F23)
""")
    if _ba:
        _s = _ba["detectors"]["ssd300"]
        st.markdown(f"""
  | Metric | Value |
  |---|---|
  | mAP@0.5 | {_s.get('mAP50', 0):.4f} |
  | mAP@0.5:0.95 | {_s.get('mAP50-95', 0):.4f} |
  | Precision | {_s.get('precision', 0):.4f} |
  | Recall | {_s.get('recall', 0):.4f} |
  | FPS p50 (T4 GPU) | {_s.get('fps_p50', '—')} |
  | Latency p95 | {_s.get('latency_p95_ms', '—')} ms |
""")
    else:
        st.markdown("""
  | Metric | Value |
  |---|---|
  | mAP@0.5 | 0.3160 |
  | Precision | 0.2801 |
  | Recall | 0.5582 |
  | FPS p50 (T4 GPU) | 25.6 |
  | Latency p95 | 38.7 ms |
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SCHEMA v1.1
# ══════════════════════════════════════════════════════════════════════════════
st.header("Feature Schema v1.1 (Block C)")

st.markdown("""
The feature schema defines **23 columns** across three feature groups plus metadata:

| Group | Features | Notes |
|---|---|---|
| **F1 — Density/Flow** | `vehicle_count`, `roi_occupancy` | Frame-level aggregates (count, occupancy fraction) |
| **F2 — Speed** | `vel_px_sec`, `vel_px_sec_smooth` | Per-track px/sec; 5-frame rolling mean (Fix F07) |
| **F3 — Interaction** | `inter_vehicle_dist_norm`, `dwell_time_sec`, `proximity_flag`, `proximity_count_rolling` | Added Day 11; NaN for single-track frames |
| **Metadata** | `seq_id`, `frame_idx`, `track_id`, `cx`, `cy`, `bbox_x1/y1/x2/y2`, `conf`, `is_interpolated`, `velocity_norm`, `track_length_real`, `split`, `tracker` | Schema bookkeeping |

**MinMaxScaler** fitted on 7 continuous features (train split, normal-only, 60,428 rows):
`vel_px_sec`, `vel_px_sec_smooth`, `vehicle_count`, `roi_occupancy`,
`inter_vehicle_dist_norm`, `dwell_time_sec`, `proximity_count_rolling`

> `proximity_flag` is boolean — excluded from scaler fit by design.
""")

with st.expander("📄 Scaler Fitted Ranges", expanded=False):
    _scaler_meta = _load_json(os.path.join(_BLOCK_C, "scaler_metadata.json"))
    if _scaler_meta:
        st.json(_scaler_meta)
    else:
        st.info("Scaler metadata not available at `logs/block_c/scaler_metadata.json`.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTANT DISCLAIMERS
# ══════════════════════════════════════════════════════════════════════════════
st.header("⚠️ Important Disclaimers")

st.warning("""
**Distance Measurement Disclaimer (Fix F28)**

Distance values in this system are **image-space proxies normalised by frame diagonal (640×√2 ≈ 905 px)**.

⚠️ **These values do NOT represent physical distances.**

Perspective effects and camera angle mean that pixel distance does not linearly correspond to
real-world distance. All distance metrics should be interpreted as **relative behavioural
indicators only**. See §3.8.2 of the thesis for the full construct validity discussion.
""")

st.info("""
**Real-Time Operation Scope (Fix F29)**

This system is validated in **offline batch mode** on development hardware (Surface Pro 7, CPU).

The pipeline architecture is designed to support real-time deployment on dedicated edge hardware
(e.g., NVIDIA Jetson Nano), but:
- Current throughput: ~2–21 FPS depending on stage (detect bottleneck)
- The demo processes every 3rd frame for speed
- This is a **research/validation tool**, not a production deployment
- Real-time operation was not claimed for the demo hardware in the dissertation
""")

st.info("""
**Block C AUROC Construct Validity (Fix F04)**

The AUROC metric used in Block C measures OC-SVM separability of **statistical outliers
within a normal-traffic distribution** (UA-DETRAC validation split, MVI_20061).

It does **not** measure true anomaly detection performance against labelled anomaly events —
that is the role of Block D (AI City Track 4 evaluation).

Proxy labels for the AUROC calculation are derived from 3-sigma thresholds computed from the
**train split only** — no validation data contaminates the labelling process (anti-circular design).
""")

st.warning("""
**Isolation Forest FAR Gate Failure (Fix F14)**

Isolation Forest (100 estimators, contamination='auto') achieved mean FAR = 0.3354 on AI City
normal clips — well above the 0.10 gate threshold. This result is **reportable but annotated**:
it is included in statistical analysis but not used as the frozen Block D selection.

The high FAR is attributed to the cold-start training on warmup data, which may not represent
the full normal-traffic distribution present in AI City clips.
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL STACK
# ══════════════════════════════════════════════════════════════════════════════
st.header("Technical Stack")
col_t1, col_t2, col_t3 = st.columns(3)

with col_t1:
    st.subheader("Core Framework")
    st.write("- Python 3.11.9")
    st.write("- Streamlit 1.32.0")
    st.write("- PyTorch 2.3.1 (CPU)")
    st.write("- OpenCV 4.9.0.80")
    st.write("- NumPy 1.26.4")
    st.write("- Pandas 2.2.2")
    st.write("- Plotly 5.22.0")

with col_t2:
    st.subheader("Computer Vision")
    st.write("- YOLOv8 (Ultralytics 8.2.0)")
    st.write("- SSD300-VGG16 (torchvision)")
    st.write("- Kalman Filtering (filterpy 1.4.5)")
    st.write("- SORT (IoU-only association)")
    st.write("- DeepSORT (VeRi-776 ReID)")
    st.write("- ByteTrack (two-pass confidence)")
    st.write("- motmetrics 1.4.0")

with col_t3:
    st.subheader("Machine Learning")
    st.write("- scikit-learn 1.5.0")
    st.write("- One-Class SVM (nu=0.01, RBF kernel)")
    st.write("- Isolation Forest (100 estimators)")
    st.write("- MinMaxScaler (7 feature columns)")
    st.write("- joblib 1.4.2")
    st.write("- scipy (Wilcoxon, McNemar, Pearson r)")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM REQUIREMENTS
# ══════════════════════════════════════════════════════════════════════════════
st.header("System Requirements")
st.markdown("""
| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Windows 11 / Linux / macOS | Any |
| Python | 3.11+ | 3.11.9 |
| RAM | 4 GB | 8 GB (for Block C feature CSVs) |
| Storage | ~2 GB (models + deps) | ~4 GB (models + all logs) |
| GPU | Not required | CUDA 11.8+ for faster YOLO inference |
| Camera | Not required | Not required (offline video) |

> All pipeline processing (detection, tracking, feature extraction, anomaly detection) runs on CPU by default.
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE ARCHITECTURE NOTES
# ══════════════════════════════════════════════════════════════════════════════
st.header("Pipeline Architecture Notes")

with st.expander("Warmup & Post-Processing (Blocks B, C, D)", expanded=False):
    st.markdown("""
**Warmup Frames:**
- Formula: `max(10, int(0.4 × fps))` — yields 10 frames at 25 FPS
- Warmup frames are stripped from Block C feature extraction output
- Block D anomaly detector suppresses predictions during warmup period (first 200 frames by default)

**Block B Post-Processing (Day 9):**
- Fragment filter: tracks shorter than `min_track_length=15` frames are discarded
- Interpolation: gaps ≤ `max_interp_gap=3` frames are filled with linear centroids
- Warmup stripping: first 20 frames of each sequence removed

**Block C OC-SVM (Day 12):**
- MinMaxScaler fitted on train split (7 continuous columns, normal-only rows)
- OC-SVM: `kernel='rbf'`, `nu=0.01` (best per Fix F02 grid search over {0.01, 0.02, 0.05, 0.10, 0.15})
- F2_only inference uses columns 0 and 1 of the scaled 7-column vector

**Block D Warmup in Live Demo:**
- `warmup_frames=200` for online training (OC-SVM / Isolation Forest)
- Pre-trained OC-SVM model bypasses warmup training if artefacts are found
- Rule-Based uses rolling Z-score from history buffer (deque, maxlen=500)
""")

with st.expander("Coordinate Space & Speed Calculations", expanded=False):
    st.markdown("""
**Coordinate Space:**
- All bboxes normalised to 640×640 reference space (not the original frame resolution)
- Applied by `src/normalisation.py` before Block A caching

**Speed Calculation (Fix F07):**
```
vel_px_sec = Δpixels × fps    (pixels per second, NOT per frame)
vel_px_sec_smooth = rolling_mean(vel_px_sec, window=int(0.2 × fps))
                  = 5 frames at 25 FPS
```

**Inter-Vehicle Distance (Fix F28):**
```
inter_vehicle_dist_norm = min_pairwise_centroid_distance / frame_diagonal
frame_diagonal = √(640² + 640²) ≈ 905.1 px
```
Image-space proxy only — perspective effects apply. NOT physical distance.

**ROI Occupancy:**
```
roi_occupancy = (n_active_tracks × 80 × 60) / (frame_width × frame_height)
```
Simplified proxy using average vehicle bbox area (80×60 px). Exact bbox areas
are available in the full feature CSVs.
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FUTURE WORK
# ══════════════════════════════════════════════════════════════════════════════
st.header("Future Work")
st.markdown("""
- **Real-time deployment** on NVIDIA Jetson Nano or similar edge hardware
- **Multi-camera calibration** using homography for physical distance estimation
- **Perspective-aware feature extraction** (lane-level ROI, calibrated speed)
- **Deep learning anomaly detection** (spatiotemporal autoencoders, Video Transformer)
- **Regional analysis** with configurable ROI polygons per camera
- **Multi-class detection** (cars, motorcycles, trucks, pedestrians, cyclists)
- **Longitudinal tracking** across camera handoff zones
""")

st.markdown("---")

st.subheader("Contact & Attribution")
col_c1, col_c2 = st.columns(2)
with col_c1:
    st.write("**Student:** MANJOO Ameera Najla (M01014463)")
    st.write("**Supervisor:** Mr Karel Veerabudren")
    st.write("**University:** Middlesex University Mauritius")
with col_c2:
    st.write("**Project:** CST3990 Undergraduate Individual Project")
    st.write("**Seed:** 42 (all blocks, all experiments)")
    st.write("**Pipeline frozen:** Day 17 (Block D FAR gate evaluation)")
