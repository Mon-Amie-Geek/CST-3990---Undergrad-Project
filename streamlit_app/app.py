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
_APP_DIR   = os.path.dirname(__file__)                          # streamlit_app/
_REPO_ROOT = os.path.normpath(os.path.join(_APP_DIR, ".."))    # repo root
_LOGS_DIR  = os.path.join(_REPO_ROOT, "logs")

def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _load_csv(path):
    try:
        import pandas as pd
        return pd.read_csv(path)
    except Exception:
        return None

# ── Block A: logs/block_a/block_a_results.csv ────────────────────────────────
_ba_df = _load_csv(os.path.join(_LOGS_DIR, "block_a", "block_a_results.csv"))

# ── Block B: logs/block_b/tracker_selection_report.json ──────────────────────
_bb_raw = _load_json(os.path.join(_LOGS_DIR, "block_b", "tracker_selection_report.json"))
# Key is "ranked_trackers" (not "trackers")
_bb = None
if _bb_raw and "ranked_trackers" in _bb_raw:
    _bb = {"trackers": _bb_raw["ranked_trackers"],
           "selected":  _bb_raw.get("selected_tracker", "sort")}

# ── Block C: logs/block_c/block_c_auroc_results.csv ──────────────────────────
_bc_df = _load_csv(os.path.join(_LOGS_DIR, "block_c", "block_c_auroc_results.csv"))

# ── Block D: logs/block_d/block_d_complete_results.csv ───────────────────────
_bd_df = _load_csv(os.path.join(_LOGS_DIR, "block_d", "block_d_complete_results.csv"))

# ── Day 18: logs/block_d/day18_report.json ───────────────────────────────────
_day18 = _load_json(os.path.join(_LOGS_DIR, "block_d", "day18_report.json"))

# ── Latency: logs/block_d/day17_latency_profile.json ─────────────────────────
_lat = _load_json(os.path.join(_LOGS_DIR, "block_d", "day17_latency_profile.json"))

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
    if _ba_df is not None and not _ba_df.empty:
        _yolo = _ba_df[_ba_df["detector"] == "yolov8n"].iloc[0]
        _ssd  = _ba_df[_ba_df["detector"] == "ssd300"].iloc[0]
        st.success("**Block A — Detection ✅**")
        st.markdown(f"""
| | YOLOv8n | SSD300 |
|--|--|--|
| mAP@0.5 | **{_yolo['mAP']:.4f}** | {_ssd['mAP']:.4f} |
| Precision | {_yolo['precision']:.4f} | {_ssd['precision']:.4f} |
| Recall | {_yolo['recall']:.4f} | {_ssd['recall']:.4f} |
| FPS | {_yolo['FPS']} | {_ssd['FPS']} |
""")
        _delta = _yolo['mAP'] - _ssd['mAP']
        st.info(f"**Frozen:** `yolov8n` (Day 6)  \nmAP delta = {_delta:.4f} > 0.05 threshold")
    else:
        st.warning("**Block A — Detection**  \nResults file not found.")

# ── Block B ──────────────────────────────────────────────────────────────────
with col_b:
    if _bb:
        st.success("**Block B — Tracking ✅**")
        rows_b = []
        for t in _bb["trackers"]:
            name = t["tracker"].upper()
            is_sel = t.get("selected", False)
            label = f"**{name}**" if is_sel else name
            idf1  = t.get("idf1_mean", 0)
            idsw  = t.get("idsw_total", "-")
            fps   = t.get("fps_median", "-")
            rows_b.append(f"| {label} | {idf1:.4f} | {idsw} | {fps} |")
        st.markdown(
            "| Tracker | IDF1 | IDSW | FPS |\n|--|--|--|--|\n" + "\n".join(rows_b)
        )
        _sel = _bb["selected"]
        _sel_t = next((t for t in _bb["trackers"] if t.get("selected")), None)
        if _sel_t:
            st.info(f"**Frozen:** `{_sel}` (Day 9)  \nIDF1 = {_sel_t['idf1_mean']:.4f}")
    else:
        st.warning("**Block B — Tracking**  \nResults file not found.")

# ── Block C ──────────────────────────────────────────────────────────────────
with col_c:
    if _bc_df is not None and not _bc_df.empty:
        st.success("**Block C — Features ✅**")
        rows_c = []
        for _, row in _bc_df.iterrows():
            fs   = row["feature_set"]
            auroc = row["auroc"]
            sil   = row["silhouette"]
            nu    = row["best_nu"]
            # Highest AUROC is frozen selection
            is_best = auroc == _bc_df["auroc"].max()
            label = f"**{fs}**" if is_best else fs
            rows_c.append(f"| {label} | {auroc:.4f} | {sil:.4f} | {nu} |")
        st.markdown(
            "| Feature Set | AUROC | Silhouette | nu |\n|--|--|--|--|\n" + "\n".join(rows_c)
        )
        _best_fs = _bc_df.loc[_bc_df["auroc"].idxmax()]
        st.info(
            f"**Frozen:** `{_best_fs['feature_set']}` (Day 12)  \n"
            f"AUROC = {_best_fs['auroc']:.4f} | nu = {_best_fs['best_nu']}"
        )
    else:
        st.warning("**Block C — Features**  \nResults CSV not found.")

# ── Block D ──────────────────────────────────────────────────────────────────
with col_d:
    if _bd_df is not None and not _bd_df.empty:
        st.success("**Block D — Anomaly Detection ✅**")
        rows_d = []
        for _, row in _bd_df.iterrows():
            meth  = row["method"]
            cov   = row.get("detection_coverage_pct", 100)
            mfar  = row["mean_far"]
            gate  = row["far_gate_passed"]
            gate_icon = "✅" if str(gate).strip() == "True" else "❌"
            rows_d.append(f"| {meth} | {cov:.0f}% | {mfar:.4f} | {gate_icon} |")
        st.markdown(
            "| Method | Coverage | mean FAR | Gate |\n|--|--|--|--|\n" + "\n".join(rows_d)
        )
        st.info(
            "**Gate threshold:** FAR < 0.10  \n"
            "`OC-SVM` & `rule_based` pass ✅  \n"
            "`isolation_forest` flagged ❌ (Fix F14, annotated)"
        )
    else:
        st.warning("**Block D — Anomaly Detection**  \nResults file not found.")

st.markdown("---")

# ============================================================================
# DAY 18 — STATISTICAL TESTS RESULTS
# ============================================================================
st.markdown("### 📊 Day 18 — Statistical Tests")

if _day18:
    tests = _day18.get("statistical_tests", {})

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        w = tests.get("test1_wilcoxon", {})
        if w:
            sig = "✅ Significant" if w.get("significant_05") else "◻ Not significant"
            st.markdown(f"**Wilcoxon signed-rank**  \n{sig}")
            st.markdown(f"""
| | |
|--|--|
| Comparison | rule_based vs OC-SVM FAR |
| n pairs | {w.get('n_pairs', '—')} |
| W statistic | {w.get('statistic', 0):.4f} |
| p-value | {w.get('p_value', 0):.4f} |
| Median RB | {w.get('median_rb', 0):.4f} |
| Median OCSVM | {w.get('median_ocsvm', 0):.4f} |
""")

    with col_s2:
        m = tests.get("test2_mcnemar", {})
        if m:
            sig = "✅ Significant" if m.get("significant_05") else "◻ Not significant"
            st.markdown(f"**McNemar (exact)**  \n{sig}")
            ct = m.get("contingency_table", {})
            st.markdown(f"""
| | |
|--|--|
| Comparison | rule_based vs IF gate |
| n clips | {m.get('n_clips', '—')} |
| Threshold | {m.get('threshold', 0.10)} |
| Discordant b | {m.get('discordant_b', '—')} |
| Discordant c | {m.get('discordant_c', '—')} |
| p-value | {m.get('p_value', 0):.4f} |
""")

    with col_s3:
        p = tests.get("test3_pearson_r", {})
        if p:
            sig = "✅ Significant" if p.get("significant_05") else "◻ Not significant"
            st.markdown(f"**Pearson r**  \n{sig}")
            st.markdown(f"""
| | |
|--|--|
| Comparison | pct_clipped vs FAR |
| n pairs | {p.get('n_pairs', '—')} |
| r | {p.get('r', 0):.4f} |
| r² | {p.get('r_squared', 0):.4f} |
| p-value | {p.get('p_value', 0):.4f} |
""")
else:
    st.info(
        "Day 18 statistical results not yet generated.  \n"
        "Run `notebooks/day18_results_consolidation.ipynb` to produce `logs/block_d/day18_report.json`."
    )

st.markdown("---")

# ============================================================================
# LATENCY PROFILE
# ============================================================================
if _lat:
    st.markdown("### ⚡ Latency Profile (Day 17)")
    _stages = _lat.get("stages", {})
    col_l1, col_l2 = st.columns([1, 2])
    with col_l1:
        st.metric("End-to-end FPS", f"{_lat.get('end_to_end_fps', '—')} fps")
        st.metric("Total mean latency", f"{_stages.get('total', {}).get('mean_ms', '—')} ms")
        st.metric("Total p95 latency", f"{_stages.get('total', {}).get('p95_ms', '—')} ms")
        st.metric("Bottleneck stage", "anomaly detection")
    with col_l2:
        stage_data = {
            "Stage": ["decode", "detect", "track", "feature", "anomaly", "total"],
            "mean ms": [
                _stages.get(s, {}).get("mean_ms", 0)
                for s in ["decode", "detect", "track", "feature", "anomaly", "total"]
            ],
            "p50 ms": [
                _stages.get(s, {}).get("p50_ms", 0)
                for s in ["decode", "detect", "track", "feature", "anomaly", "total"]
            ],
            "p95 ms": [
                _stages.get(s, {}).get("p95_ms", 0)
                for s in ["decode", "detect", "track", "feature", "anomaly", "total"]
            ],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(stage_data), use_container_width=True)
    st.caption(
        "⚠️ Timings are indicative offline-batch measurements on a Surface Pro 7 (Fix F29). "
        "Not universal system guarantees."
    )
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
| **D** | Anomaly Detection | Rule-Based, OC-SVM, Isolation Forest | `OC-SVM` — mean FAR = 0.0283 ✅ |

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
    _model_path_root = os.path.join(_REPO_ROOT, "models", "best.pt")
    if not os.path.exists(_model_path_root):
        st.warning("""
⚠️ **YOLOv8n Model File Not Found**

Place the fine-tuned YOLOv8n weights at `models/best.pt` (repo root) or `streamlit_app/models/best.pt`.
SSD300 (COCO) loads via `torchvision` pretrained weights.
""")
