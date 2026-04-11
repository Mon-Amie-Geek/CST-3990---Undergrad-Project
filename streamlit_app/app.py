"""
app.py — Main dashboard for CST3990 Traffic Anomaly Detection
Student: MANJOO Ameera Najla | M01014463
Project: CST3990 Undergraduate Individual Project
"""

import os
import json
import warnings
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CST3990 — Traffic Anomaly Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Page background */
    .main .block-container { padding-top: 1.5rem; }

    /* Hero header */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero-title   { font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .hero-sub     { font-size: 1.05rem; color: #a0aec0; margin: 0.3rem 0 0; }
    .hero-student { font-size: 0.9rem; color: #e2e8f0; text-align: right; margin-top: 0.5rem; }

    /* Block status cards */
    .block-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #48bb78;
    }
    .block-card-pending {
        border-left-color: #ed8936;
    }

    /* Metric row */
    div[data-testid="metric-container"] {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
    }

    /* Tables */
    .stDataFrame { border-radius: 8px; }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #2d3748;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Path helpers ───────────────────────────────────────────────────────────────
_APP_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_APP_DIR, ".."))
_LOGS_DIR  = os.path.join(_REPO_ROOT, "logs")
_BLOCK_D   = os.path.join(_LOGS_DIR, "block_d")
_BLOCK_C   = os.path.join(_LOGS_DIR, "block_c")


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


# ── Load data ──────────────────────────────────────────────────────────────────
_ba_df  = _load_csv(os.path.join(_LOGS_DIR, "block_a", "block_a_results.csv"))
_ba_json = _load_json(os.path.join(_LOGS_DIR, "block_a_results.json"))

_bb_raw = _load_json(os.path.join(_LOGS_DIR, "block_b", "tracker_selection_report.json"))
_bb = None
if _bb_raw and "ranked_trackers" in _bb_raw:
    _bb = {"trackers": _bb_raw["ranked_trackers"],
           "selected":  _bb_raw.get("selected_tracker", "sort")}

_bc_df  = _load_csv(os.path.join(_BLOCK_C, "block_c_auroc_results.csv"))
_bd_df  = _load_csv(os.path.join(_BLOCK_D, "block_d_complete_results.csv"))
_day18  = _load_json(os.path.join(_BLOCK_D, "day18_report.json"))
_lat    = _load_json(os.path.join(_BLOCK_D, "day17_latency_profile.json"))

# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div style="display:flex; justify-content:space-between; align-items:flex-start;">
    <div>
      <div class="hero-title">🚗 Traffic Anomaly Detection</div>
      <div class="hero-sub">A Comparative Study of Computer Vision Pipeline Components</div>
      <div style="margin-top:0.6rem; font-size:0.85rem; color:#90cdf4;">
        CST3990 Undergraduate Individual Project &nbsp;|&nbsp; Middlesex University Mauritius
      </div>
    </div>
    <div style="text-align:right; color:#e2e8f0; font-size:0.9rem; line-height:1.7;">
      <div><strong>👤 MANJOO Ameera Najla</strong></div>
      <div>M01014463</div>
      <div style="color:#90cdf4;">Supervisor: Mr Karel Veerabudren</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Navigation hint ────────────────────────────────────────────────────────────
col_nav1, col_nav2, col_nav3 = st.columns(3)
with col_nav1:
    st.info("🎥 **Pipeline** — Upload a video & run live analysis")
with col_nav2:
    st.info("📊 **Comparison** — Full experimental results & charts")
with col_nav3:
    st.info("📋 **About** — Methodology, disclaimers & tech stack")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STATUS — all 4 blocks
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🔬 Experimental Pipeline — Frozen Selections & Results")

col_a, col_b, col_c, col_d = st.columns(4)

# ── Block A ────────────────────────────────────────────────────────────────────
with col_a:
    _yolo_row = None
    _ssd_row  = None
    if _ba_df is not None and not _ba_df.empty:
        try:
            _yolo_row = _ba_df[_ba_df["detector"] == "yolov8n"].iloc[0]
            _ssd_row  = _ba_df[_ba_df["detector"] == "ssd300"].iloc[0]
        except Exception:
            pass

    if _yolo_row is not None:
        st.success("**Block A — Detection ✅**")
        st.markdown(f"""
| | YOLOv8n | SSD300 |
|--|--|--|
| mAP@0.5 | **{float(_yolo_row['mAP']):.4f}** | {float(_ssd_row['mAP']):.4f} |
| Precision | {float(_yolo_row['precision']):.4f} | {float(_ssd_row['precision']):.4f} |
| Recall | {float(_yolo_row['recall']):.4f} | {float(_ssd_row['recall']):.4f} |
| FPS | {_yolo_row['FPS']} | {_ssd_row['FPS']} |
""")
        _delta = float(_yolo_row['mAP']) - float(_ssd_row['mAP'])
        st.info(f"🔒 **Frozen:** `yolov8n` (Day 6)  \nΔmAP = **{_delta:.4f}** > 0.05 ✓")
    elif _ba_json:
        det = _ba_json.get("detectors", {})
        st.success("**Block A — Detection ✅**")
        y = det.get("yolov8n", {}); s = det.get("ssd300", {})
        st.markdown(f"""
| | YOLOv8n | SSD300 |
|--|--|--|
| mAP@0.5 | **{y.get('mAP50', 0):.4f}** | {s.get('mAP50', 0):.4f} |
| Precision | {y.get('precision', 0):.4f} | {s.get('precision', 0):.4f} |
| Recall | {y.get('recall', 0):.4f} | {s.get('recall', 0):.4f} |
| FPS p50 | {y.get('fps_p50', '—')} | {s.get('fps_p50', '—')} |
""")
        _delta = y.get('mAP50', 0) - s.get('mAP50', 0)
        st.info(f"🔒 **Frozen:** `yolov8n` (Day 6)  \nΔmAP = **{_delta:.4f}** > 0.05 ✓")
    else:
        st.warning("**Block A — Detection**")
        st.markdown("""
| | YOLOv8n | SSD300 |
|--|--|--|
| mAP@0.5 | **0.6674** | 0.3160 |
| Precision | 0.7679 | 0.2801 |
| Recall | 0.7703 | 0.5582 |
| FPS p50 | 47.2 | 25.6 |
""")
        st.info("🔒 **Frozen:** `yolov8n` (Day 6)  \nΔmAP = **0.3514** > 0.05 ✓")

# ── Block B ────────────────────────────────────────────────────────────────────
with col_b:
    if _bb:
        st.success("**Block B — Tracking ✅**")
        rows_b = []
        for t in _bb["trackers"]:
            name    = t["tracker"].upper()
            is_sel  = t.get("selected", False)
            label   = f"**{name}**" if is_sel else name
            idf1    = t.get("idf1_mean", 0)
            idsw    = t.get("idsw_total", "—")
            fps_val = t.get("fps_median", "—")
            rows_b.append(f"| {label} | {idf1:.4f} | {idsw} | {fps_val} |")
        st.markdown("| Tracker | IDF1 | IDSW | FPS |\n|--|--|--|--|\n" + "\n".join(rows_b))
        sel_t = next((t for t in _bb["trackers"] if t.get("selected")), None)
        if sel_t:
            st.info(
                f"🔒 **Frozen:** `{_bb['selected']}` (Day 9)  \n"
                f"IDF1 = **{sel_t['idf1_mean']:.4f}** (highest)"
            )
    else:
        st.warning("**Block B — Tracking**")
        st.markdown("""
| Tracker | IDF1 | IDSW | FPS |
|--|--|--|--|
| **SORT** | **0.0412** | 31 | 1129.7 |
| DeepSORT | 0.0008 | 0 | 1063.7 |
| ByteTrack | 0.0007 | 0 | 1231.4 |
""")
        st.info("🔒 **Frozen:** `sort` (Day 9)  \nIDF1 = **0.0412** (highest)")

# ── Block C ────────────────────────────────────────────────────────────────────
with col_c:
    if _bc_df is not None and not _bc_df.empty:
        st.success("**Block C — Features ✅**")
        rows_c = []
        _max_auroc = _bc_df["auroc"].max()
        for _, row in _bc_df.iterrows():
            fs     = row["feature_set"]
            auroc  = row["auroc"]
            sil    = row.get("silhouette", float("nan"))
            nu     = row.get("best_nu", "—")
            label  = f"**{fs}**" if auroc == _max_auroc else fs
            sil_str = f"{sil:.4f}" if not (isinstance(sil, float) and np.isnan(sil)) else "—"
            rows_c.append(f"| {label} | {auroc:.4f} | {sil_str} | {nu} |")
        st.markdown("| Set | AUROC | Sil. | nu |\n|--|--|--|--|\n" + "\n".join(rows_c))
        _best = _bc_df.loc[_bc_df["auroc"].idxmax()]
        st.info(
            f"🔒 **Frozen:** `{_best['feature_set']}` (Day 12)  \n"
            f"AUROC = **{_best['auroc']:.4f}** | nu = {_best.get('best_nu', 0.01)}"
        )
    else:
        st.warning("**Block C — Features**")
        st.markdown("""
| Set | AUROC | Sil. | nu |
|--|--|--|--|
| **F2_only** | **0.9935** | 0.9431 | 0.01 |
| F1_only | 0.4724 | -0.0022 | 0.05 |
| F2+F3 | 0.6410 | 0.4710 | 0.01 |
| All | 0.7312 | 0.3677 | 0.01 |
""")
        st.info("🔒 **Frozen:** `F2_only` (Day 12)  \nAUROC = **0.9935** | nu = 0.01")

# ── Block D ────────────────────────────────────────────────────────────────────
with col_d:
    if _bd_df is not None and not _bd_df.empty:
        st.success("**Block D — Anomaly Detection ✅**")
        rows_d = []
        for _, row in _bd_df.iterrows():
            meth      = row["method"]
            cov       = float(row.get("detection_coverage_pct", 100))
            mfar      = float(row["mean_far"])
            gate      = str(row["far_gate_passed"]).strip()
            gate_icon = "✅" if gate == "True" else "❌"
            rows_d.append(f"| {meth} | {cov:.0f}% | {mfar:.4f} | {gate_icon} |")
        st.markdown("| Method | Cov. | FAR | Gate |\n|--|--|--|--|\n" + "\n".join(rows_d))
        st.info(
            "🔒 Gate: FAR < 0.10  \n"
            "`OC-SVM` & `rule_based` **✅ PASS**  \n"
            "`isolation_forest` **❌ FAIL**"
        )
    else:
        st.warning("**Block D — Anomaly Detection**")
        st.markdown("""
| Method | Cov. | FAR | Gate |
|--|--|--|--|
| ocsvm | 100% | 0.0283 | ✅ |
| rule_based | 100% | 0.0350 | ✅ |
| isolation_forest | 100% | 0.3354 | ❌ |
""")
        st.info("Gate: FAR < 0.10  \nOC-SVM & Rule-Based pass ✅")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK D — FAR COMPARISON CHART
# ══════════════════════════════════════════════════════════════════════════════
if _bd_df is not None and not _bd_df.empty:
    st.markdown("### 📉 Block D — False Alarm Rate (FAR) Comparison")

    methods   = _bd_df["method"].tolist()
    mean_fars = [float(v) for v in _bd_df["mean_far"]]
    std_fars  = [float(v) for v in _bd_df["std_far"]]
    gates     = [str(v).strip() == "True" for v in _bd_df["far_gate_passed"]]
    colors    = ["#48bb78" if g else "#fc8181" for g in gates]

    fig_far = go.Figure()
    fig_far.add_trace(go.Bar(
        x=methods, y=mean_fars,
        error_y=dict(type="data", array=std_fars, visible=True),
        marker_color=colors,
        text=[f"{v:.4f}" for v in mean_fars],
        textposition="outside",
        name="Mean FAR",
    ))
    fig_far.add_hline(
        y=0.10, line_dash="dash", line_color="#e53e3e",
        annotation_text="FAR gate = 0.10",
        annotation_position="top right",
    )
    fig_far.update_layout(
        title="Block D — Mean False Alarm Rate per Method (AI City normal clips, n=10)",
        xaxis_title="Method",
        yaxis_title="Mean FAR (lower is better)",
        yaxis=dict(range=[0, max(0.5, max(mean_fars) * 1.3)]),
        height=380,
        showlegend=False,
        plot_bgcolor="#f8fafc",
    )
    st.plotly_chart(fig_far, use_container_width=True)
    st.caption(
        "Green bars = FAR gate passed (mean FAR < 0.10). Red bar = gate flagged.  "
        "Error bars = ±1 std dev across 10 normal clips.  "
        "Isolation Forest flagged — results annotated, not excluded."
    )
    st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# DAY 18 — STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📊 Day 18 — Statistical Tests")

if _day18:
    tests = _day18.get("statistical_tests", {})

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        w = tests.get("test1_wilcoxon", {})
        if w:
            sig = "✅ Significant (p < 0.05)" if w.get("significant_05") else "◻ Not significant"
            st.markdown(f"**Wilcoxon Signed-Rank**")
            st.markdown(sig)
            st.markdown(f"""
| | |
|--|--|
| Comparison | rule_based vs OC-SVM FAR |
| n pairs | {w.get('n_pairs', '—')} |
| W statistic | {w.get('statistic', 0):.4f} |
| p-value | **{w.get('p_value', 0):.4f}** |
| Median RB | {w.get('median_rb', 0):.4f} |
| Median OCSVM | {w.get('median_ocsvm', 0):.4f} |
""")

    with col_s2:
        m = tests.get("test2_mcnemar", {})
        if m:
            sig = "✅ Significant (p < 0.05)" if m.get("significant_05") else "◻ Not significant"
            st.markdown(f"**McNemar (exact)**")
            st.markdown(sig)
            st.markdown(f"""
| | |
|--|--|
| Comparison | rule_based vs IF gate |
| n clips | {m.get('n_clips', '—')} |
| Threshold | {m.get('threshold', 0.10)} |
| Discordant b | {m.get('discordant_b', '—')} |
| Discordant c | {m.get('discordant_c', '—')} |
| p-value | **{m.get('p_value', 0):.4f}** |
""")

    with col_s3:
        p = tests.get("test3_pearson_r", {})
        if p:
            sig = "✅ Significant (p < 0.05)" if p.get("significant_05") else "◻ Not significant"
            st.markdown(f"**Pearson r**")
            st.markdown(sig)
            st.markdown(f"""
| | |
|--|--|
| Comparison | pct_clipped vs FAR |
| n pairs | {p.get('n_pairs', '—')} |
| r | **{p.get('r', 0):.4f}** |
| r² | {p.get('r_squared', 0):.4f} |
| p-value | **{p.get('p_value', 0):.4f}** |
""")
else:
    st.info(
        "Day 18 statistical results not generated yet.  \n"
        "Run `src/day18_statistical_tests.py` to produce `logs/block_d/day18_report.json`."
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# LATENCY PROFILE
# ══════════════════════════════════════════════════════════════════════════════
if _lat:
    st.markdown("### ⚡ Latency Profile (Day 17 — offline batch, Surface Pro 7)")

    _stages = _lat.get("stages", {})
    col_l1, col_l2 = st.columns([1, 2])

    with col_l1:
        st.metric("End-to-end FPS", f"{_lat.get('end_to_end_fps', '—')}")
        st.metric("Total mean latency", f"{_stages.get('total', {}).get('mean_ms', '—')} ms")
        st.metric("Total p95 latency",  f"{_stages.get('total', {}).get('p95_ms', '—')} ms")
        bottleneck = _lat.get("bottleneck_stage", "detect")
        st.metric("Bottleneck stage", bottleneck)

    with col_l2:
        stage_names = ["decode", "detect", "track", "feature", "anomaly", "total"]
        stage_data = {
            "Stage":    stage_names,
            "mean ms":  [_stages.get(s, {}).get("mean_ms", 0) for s in stage_names],
            "p50 ms":   [_stages.get(s, {}).get("p50_ms",  0) for s in stage_names],
            "p95 ms":   [_stages.get(s, {}).get("p95_ms",  0) for s in stage_names],
        }
        df_lat = pd.DataFrame(stage_data)
        st.dataframe(df_lat, use_container_width=True, hide_index=True)

        # Latency bar chart (p50 per stage, excluding total)
        fig_lat = go.Figure(go.Bar(
            x=stage_names[:-1],
            y=[_stages.get(s, {}).get("p50_ms", 0) for s in stage_names[:-1]],
            marker_color=["#4299e1", "#f6ad55", "#68d391", "#9f7aea", "#fc8181"],
            text=[f"{_stages.get(s, {}).get('p50_ms', 0):.0f} ms" for s in stage_names[:-1]],
            textposition="outside",
        ))
        fig_lat.update_layout(
            title="Per-stage p50 latency (ms)",
            yaxis_title="Latency (ms)",
            height=280,
            showlegend=False,
            plot_bgcolor="#f8fafc",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    st.caption(
        "⚠️ Timings are indicative offline-batch measurements on a Surface Pro 7 CPU. "
        "Not real-time guarantees — dedicated edge hardware required for real-time deployment."
    )
    st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PROJECT OVERVIEW TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📋 Frozen Pipeline Summary")

left, right = st.columns([3, 2])
with left:
    st.markdown("""
| Block | Component | Candidates | **Frozen Selection** |
|---|---|---|---|
| **A** | Detection | YOLOv8n (fine-tuned) vs SSD300 (COCO) | `yolov8n` — mAP@0.5 = 0.6674 |
| **B** | Tracking | SORT vs DeepSORT vs ByteTrack | `sort` — IDF1 = 0.0412 |
| **C** | Feature Extraction | F1 (Density), F2 (Speed), F2+F3, All | `F2_only` — AUROC = 0.9935 |
| **D** | Anomaly Detection | Rule-Based, OC-SVM, Isolation Forest | `OC-SVM` — FAR = 0.0283 ✅ |

**Datasets:**
- **UA-DETRAC** — Training, tracking & feature evaluation (10 sequences, 25 FPS)
- **AI City Track 4** — Anomaly ground truth (stall/crash events, Block D)
""")

with right:
    st.markdown("""
**How to navigate:**

1. 🎥 **Pipeline** → Upload a `.mp4` video
2. Configure each block in the sidebar
3. Click **▶ Run Pipeline**
4. View live detection, feature charts, results

> All processing runs on **CPU**.
> Every 3rd frame processed (3× speedup).
> OC-SVM loads Block C pre-trained model if available.
""")
    # Model file check
    _model_path = os.path.join(_REPO_ROOT, "models", "best.pt")
    if os.path.exists(_model_path):
        st.success(f"✅ YOLOv8n weights found: `models/best.pt`")
    else:
        st.warning("⚠️ `models/best.pt` not found — YOLOv8n will fail to load.")

st.markdown("---")

# ── Footer ─────────────────────────────────────────────────────────────────────
fc1, fc2, fc3 = st.columns(3)
with fc1:
    st.subheader("👤 Student")
    st.write("**Name:** MANJOO Ameera Najla")
    st.write("**ID:** M01014463")
with fc2:
    st.subheader("🎓 University")
    st.write("**Institution:** Middlesex University Mauritius")
    st.write("**Supervisor:** Mr Karel Veerabudren")
with fc3:
    st.subheader("📚 Quick Links")
    st.write("• **Pipeline** — Run analysis on a video")
    st.write("• **Comparison** — View all experimental results")
    st.write("• **About** — Disclaimers & methodology")

st.markdown("""
<p style="text-align:center; color:#718096; font-size:12px; margin-top:1rem;">
  CST3990 Undergraduate Individual Project &nbsp;|&nbsp; Middlesex University Mauritius &nbsp;|&nbsp;
  A Comparative Study of Computer Vision Pipeline Components for Real-Time Traffic Analysis and Anomaly Detection
</p>
""", unsafe_allow_html=True)
