"""
app.py

Main entry point for the CST3990 Traffic Anomaly Detection Streamlit application.
Student: MANJOO Ameera Najla | M01014463
Project: CST3990 Undergraduate Individual Project
"""

import os
import json
import warnings
import torch
import numpy as np
import random
import streamlit as st

warnings.filterwarnings("ignore")

# Set seeds for reproducibility (seed=42 across all blocks)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="CST3990 Traffic Anomaly Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Path helpers ──────────────────────────────────────────────────────────────
_APP_DIR  = os.path.dirname(__file__)                             # streamlit_app/
_REPO_ROOT = os.path.normpath(os.path.join(_APP_DIR, ".."))      # repo root
_LOGS_DIR  = os.path.join(_REPO_ROOT, "logs")

def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

_ba = _load_json(os.path.join(_LOGS_DIR, "block_a_results.json"))
_bb = _load_json(os.path.join(_LOGS_DIR, "block_b_results.json"))
_bc_table_path = os.path.join(_LOGS_DIR, "block_c", "block_c_results_table.csv")

# ============================================================================
# HEADER
# ============================================================================
col1, col2, col3 = st.columns([2, 3, 2])
with col1:
    st.markdown("## 🎓 CST3990")
    st.markdown("**Undergraduate Individual Project**")
with col2:
    st.markdown("## 🚗 Traffic Anomaly Detection")
    st.markdown("### Computer Vision Pipeline Comparison")
with col3:
    st.markdown("### 👤 MANJOO Ameera Najla")
    st.markdown("**M01014463**")
    st.markdown("**Middlesex University Mauritius**")

st.markdown("---")

# ============================================================================
# PIPELINE STATUS DASHBOARD
# ============================================================================
st.markdown("### 🔬 Experimental Pipeline — Status & Key Results")

col_a, col_b, col_c, col_d = st.columns(4)

# ── Block A ──────────────────────────────────────────────────────────────────
with col_a:
    if _ba:
        _yolo = _ba["detectors"]["yolov8n"]
        _ssd  = _ba["detectors"]["ssd300"]
        st.success("**Block A — Detection ✅**")
        st.markdown(f"""
| | YOLOv8n | SSD300 |
|--|--|--|
| mAP@0.5 | **{_yolo['mAP50']:.4f}** | {_ssd['mAP50']:.4f} |
| Precision | {_yolo['precision']:.4f} | {_ssd['precision']:.4f} |
| Recall | {_yolo['recall']:.4f} | {_ssd['recall']:.4f} |
| FPS p50 | {_yolo['fps_p50']} | {_ssd['fps_p50']} |
""")
        st.info(f"**Frozen:** `yolov8n` (Day 6)  \nmAP delta = {_yolo['mAP50'] - _ssd['mAP50']:.4f} > 0.05 threshold")
    else:
        st.warning("**Block A — Detection**  \nResults file not found.")

# ── Block B ──────────────────────────────────────────────────────────────────
with col_b:
    if _bb:
        _sel_trk = next((t for t in _bb["trackers"] if t.get("selected")), None)
        st.success("**Block B — Tracking ✅**")
        rows_b = []
        for t in _bb["trackers"]:
            rows_b.append(f"| {'**'+t['tracker'].upper()+'**' if t.get('selected') else t['tracker'].upper()} | {t['idf1_mean']:.4f} | {t['mota_mean']:.4f} | {t['idsw_total']} |")
        st.markdown(
            "| Tracker | IDF1 | MOTA | IDSW |\n|--|--|--|--|\n" + "\n".join(rows_b)
        )
        if _sel_trk:
            st.info(f"**Frozen:** `{_sel_trk['tracker']}` (Day 9)  \nHighest IDF1 = {_sel_trk['idf1_mean']:.4f}")
    else:
        st.warning("**Block B — Tracking**  \nResults file not found.")

# ── Block C ──────────────────────────────────────────────────────────────────
with col_c:
    if os.path.exists(_bc_table_path):
        try:
            import pandas as pd
            _df_c = pd.read_csv(_bc_table_path)
            _CHECK = chr(10003)
            _best_row = _df_c[_df_c["Selected"].astype(str).str.strip() == _CHECK]
            st.success("**Block C — Features ✅**")
            rows_c = []
            for _, row in _df_c.iterrows():
                sel = "**✓**" if str(row["Selected"]).strip() == _CHECK else ""
                rows_c.append(f"| {sel}{row['Feature Set']} | {row['AUROC']:.4f} | {row['Silhouette']:.4f} |")
            st.markdown(
                "| Feature Set | AUROC | Silhouette |\n|--|--|--|\n" + "\n".join(rows_c)
            )
            if not _best_row.empty:
                _bfs = _best_row.iloc[0]
                st.info(f"**Frozen:** `{_bfs['Feature Set']}` (Day 12)  \nAUROC = {_bfs['AUROC']:.4f} | nu = {_bfs['Best Nu']}")
        except Exception:
            st.success("**Block C — Features ✅**")
            st.write("Results loaded. See Comparison page.")
    else:
        st.warning("**Block C — Features**  \nResults CSV not found.")

# ── Block D ──────────────────────────────────────────────────────────────────
with col_d:
    st.warning("**Block D — Anomaly Detection ⏳**")
    st.markdown("""
| Method | Status |
|--|--|
| Rule-Based | Pending |
| OC-SVM | Pending |
| Isolation Forest | Pending |
""")
    st.markdown("""
**Config frozen (2026-04-07):**
- Detector: `yolov8n`
- Tracker: `sort`
- Features: `F2_only`
- Dataset: AI City Track 4
- FAR threshold: 0.10
""")

st.markdown("---")

# ============================================================================
# PROJECT DESCRIPTION
# ============================================================================
st.markdown("### 📋 Project Overview")

left, right = st.columns([3, 2])
with left:
    st.markdown("""
This application implements a configurable computer vision pipeline for **traffic analysis
and anomaly detection**. Four experimental blocks are evaluated independently, each frozen
after its selection phase:

| Block | Component | Candidates | Frozen Selection |
|---|---|---|---|
| **A** | Detection | YOLOv8n (fine-tuned) vs SSD300 (COCO) | `yolov8n` — mAP@0.5 = 0.6674 |
| **B** | Tracking | SORT vs DeepSORT vs ByteTrack | `sort` — IDF1 = 0.0412 |
| **C** | Feature Extraction | F1 (Density/Flow), F2 (Speed), F2+F3, All | `F2_only` — AUROC = 0.9935 |
| **D** | Anomaly Detection | Rule-Based, OC-SVM, Isolation Forest | *Pending evaluation* |

**Datasets:**
- Training / Feature evaluation: **UA-DETRAC** (10 sequences, 25 FPS, MVI_200xx)
- Anomaly ground truth (Block D): **AI City Track 4** (stall/crash events)
""")

with right:
    st.markdown("""
**How to use:**
1. Go to the **Pipeline** page
2. Upload a traffic video (`.mp4`)
3. Configure pipeline components in the sidebar
4. Click **▶ Run Pipeline**
5. View live detection, feature charts, and anomaly events

**Navigation:**
- **Pipeline** — Live video processing
- **Comparison** — Full experimental results
- **About** — Disclaimers & technical details

> All processing runs on **CPU**.
> Every 3rd frame is processed for speed.
""")

st.markdown("---")

# ============================================================================
# FOOTER
# ============================================================================
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.subheader("👤 Student")
    st.write("**Name:** MANJOO Ameera Najla")
    st.write("**ID:** M01014463")
with footer_col2:
    st.subheader("🎓 University")
    st.write("**Institution:** Middlesex University Mauritius")
    st.write("**Supervisor:** Mr Karel Veerabudren")
with footer_col3:
    st.subheader("📚 Navigate")
    st.write("• **Pipeline** — Run analysis on a video")
    st.write("• **Comparison** — View experimental results")
    st.write("• **About** — Project details & disclaimers")

st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #888; font-size: 12px;">
CST3990 Undergraduate Individual Project &nbsp;|&nbsp;
Middlesex University Mauritius &nbsp;|&nbsp;
A Comparative Study of Computer Vision Pipeline Components for Real-Time Traffic Analysis and Anomaly Detection
</p>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL FILE CHECK
# ============================================================================
_model_path = os.path.join(_APP_DIR, "models", "best.pt")
if not os.path.exists(_model_path):
    # Also check repo-root models/
    _model_path_root = os.path.join(_REPO_ROOT, "models", "best.pt")
    if not os.path.exists(_model_path_root):
        st.warning("""
⚠️ **YOLOv8n Model File Not Found**

Place the fine-tuned YOLOv8n weights at `models/best.pt` (repo root) or `streamlit_app/models/best.pt`.
SSD300 (COCO) loads via `torchvision` pretrained weights.
""")
