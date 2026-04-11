"""
2_Comparison.py

Comparison page showing experimental results from all blocks.
All blocks load from actual result files in logs/ when available.
Student: MANJOO Ameera Najla | M01014463
"""

import os
import glob
import json

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Comparison - CST3990 Traffic Anomaly Detection", layout="wide")

st.title("📊 Results Comparison")

# ── Shared path helpers ───────────────────────────────────────────────────────
_PAGES_DIR  = os.path.dirname(__file__)                                   # streamlit_app/pages/
_REPO_ROOT  = os.path.normpath(os.path.join(_PAGES_DIR, "..", ".."))     # repo root
_LOGS_DIR   = os.path.join(_REPO_ROOT, "logs")
_BLOCK_C_DIR = os.path.normpath(os.path.join(_LOGS_DIR, "block_c"))
_CHECK = chr(10003)  # ✓

def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

st.info("""
**Research Progress:** Blocks A (Day 6), B (Days 7-9), C (Days 10-12), and D (Days 15-18) are shown from saved experiment artefacts when the corresponding files exist in `logs/`.
If any section is missing, regenerate that block's outputs and refresh the app.
""")
st.markdown("---")

# =============================================================================
# BLOCK A — DETECTOR COMPARISON (logs/block_a_results.json)
# =============================================================================
st.header("Block A — Detector Comparison")

_ba = _load_json(os.path.join(_LOGS_DIR, "block_a_results.json"))

if _ba:
    _det = _ba["detectors"]
    _sel = _ba.get("selected_detector", "yolov8n")

    # Selection callout
    st.success(
        f"**Block A frozen (Day 6) — Selected: `{_sel.upper()}`** &nbsp;|&nbsp; "
        f"mAP@0.5 = {_det[_sel]['mAP50']:.4f} &nbsp;|&nbsp; "
        f"mAP delta vs SSD300 = {_det['yolov8n']['mAP50'] - _det['ssd300']['mAP50']:.4f} "
        f"(threshold 0.05 — accuracy criterion dominates)"
    )

    # Build display table
    _rows_a = [
        {
            "Detector":           "YOLOv8n (fine-tuned on UA-DETRAC)",
            "mAP@0.5":            _det["yolov8n"]["mAP50"],
            "mAP@0.5:0.95":       _det["yolov8n"]["mAP50-95"],
            "Precision":          _det["yolov8n"]["precision"],
            "Recall":             _det["yolov8n"]["recall"],
            "FPS p50 (T4 GPU)":   _det["yolov8n"]["fps_p50"],
            "Latency p95 (ms)":   _det["yolov8n"]["latency_p95_ms"],
            "Fine-tuned":         "Yes",
            "Selected":           _CHECK,
        },
        {
            "Detector":           "SSD300 (COCO)",
            "mAP@0.5":            _det["ssd300"]["mAP50"],
            "mAP@0.5:0.95":       _det["ssd300"]["mAP50-95"],
            "Precision":          _det["ssd300"]["precision"],
            "Recall":             _det["ssd300"]["recall"],
            "FPS p50 (T4 GPU)":   _det["ssd300"]["fps_p50"],
            "Latency p95 (ms)":   _det["ssd300"]["latency_p95_ms"],
            "Fine-tuned":         "No (COCO only)",
            "Selected":           "",
        },
    ]
    df_a = pd.DataFrame(_rows_a)
    st.dataframe(df_a, use_container_width=True)

    # mAP comparison chart
    _names_a  = ["YOLOv8n (fine-tuned)", "SSD300 (COCO)"]
    _colors_a = ["#2ca02c", "#1f77b4"]
    fig_a = go.Figure()
    fig_a.add_trace(go.Bar(
        name="mAP@0.5",
        x=_names_a,
        y=[_det["yolov8n"]["mAP50"], _det["ssd300"]["mAP50"]],
        marker_color=_colors_a,
        text=[f"{_det['yolov8n']['mAP50']:.4f}", f"{_det['ssd300']['mAP50']:.4f}"],
        textposition="outside",
    ))
    fig_a.add_trace(go.Bar(
        name="mAP@0.5:0.95",
        x=_names_a,
        y=[_det["yolov8n"]["mAP50-95"], _det["ssd300"]["mAP50-95"]],
        marker_color=["#98df8a", "#aec7e8"],
        text=[f"{_det['yolov8n']['mAP50-95']:.4f}", f"{_det['ssd300']['mAP50-95']:.4f}"],
        textposition="outside",
    ))
    fig_a.update_layout(
        barmode="group",
        title="Block A — Detector mAP Comparison (UA-DETRAC test split: MVI_20062, MVI_20063)",
        yaxis=dict(range=[0, 0.85], title="mAP"),
        xaxis_title="Detector",
        height=430,
        legend_title="Metric",
    )
    st.plotly_chart(fig_a, use_container_width=True)

    # Precision / Recall chart
    fig_a_pr = go.Figure()
    fig_a_pr.add_trace(go.Bar(
        name="Precision",
        x=_names_a,
        y=[_det["yolov8n"]["precision"], _det["ssd300"]["precision"]],
        marker_color=_colors_a,
        text=[f"{_det['yolov8n']['precision']:.4f}", f"{_det['ssd300']['precision']:.4f}"],
        textposition="outside",
    ))
    fig_a_pr.add_trace(go.Bar(
        name="Recall",
        x=_names_a,
        y=[_det["yolov8n"]["recall"], _det["ssd300"]["recall"]],
        marker_color=["#98df8a", "#aec7e8"],
        text=[f"{_det['yolov8n']['recall']:.4f}", f"{_det['ssd300']['recall']:.4f}"],
        textposition="outside",
    ))
    fig_a_pr.update_layout(
        barmode="group",
        title="Block A — Precision & Recall Comparison",
        yaxis=dict(range=[0, 1.0], title="Score"),
        xaxis_title="Detector",
        height=380,
        legend_title="Metric",
    )
    st.plotly_chart(fig_a_pr, use_container_width=True)

    st.caption(f"""
    - Evaluated on UA-DETRAC test split (MVI_20062, MVI_20063). IoU threshold = {_ba['evaluation_iou_threshold']}
    - **FPS measured on {_ba['inference_hardware']}.**
    - ⚠️ {_ba['hardware_note']}
    - SSD300 uses COCO pre-training only — no UA-DETRAC fine-tuning
    - Selection rule: primary = mAP@0.5 if delta > 0.05; secondary = FPS p50
    - ⚠️ {_ba['resolution_note']}
    """)

else:
    st.warning(f"Block A results not found at `logs/block_a_results.json`")
    st.info("Run Block A evaluation (Day 6) to generate results.")

st.markdown("---")

# =============================================================================
# BLOCK B — TRACKER COMPARISON (logs/block_b_results.json)
# =============================================================================
st.header("Block B — Tracker Comparison")

_bb = _load_json(os.path.join(_LOGS_DIR, "block_b_results.json"))

if _bb:
    _sel_trk = next((t for t in _bb["trackers"] if t.get("selected")), None)

    # Selection callout
    if _sel_trk:
        st.success(
            f"**Block B frozen (Day 9) — Selected: `{_sel_trk['tracker'].upper()}`** &nbsp;|&nbsp; "
            f"IDF1 = {_sel_trk['idf1_mean']:.4f} &nbsp;|&nbsp; "
            f"MOTA = {_sel_trk['mota_mean']:.4f} &nbsp;|&nbsp; "
            f"IDSW = {_sel_trk['idsw_total']}"
        )

    # Build display table
    _name_map = {
        "sort":      "SORT",
        "deepsort":  f"DeepSORT (osnet_x1_0_veri_776)",
        "bytetrack": "ByteTrack",
    }
    _rows_b = []
    for t in _bb["trackers"]:
        _rows_b.append({
            "Tracker":        _name_map.get(t["tracker"], t["tracker"]),
            "MOTA (mean)":    round(t["mota_mean"], 4),
            "IDF1 (mean)":    round(t["idf1_mean"], 4),
            "ID Switches":    t["idsw_total"],
            "Fragmentations": t["fragmentations_total"],
            "FPS (median)":   t["fps_median"],
            "Selected":       _CHECK if t.get("selected") else "",
        })
    df_b = pd.DataFrame(_rows_b)
    st.dataframe(df_b, use_container_width=True)

    # IDF1 chart (primary selection criterion)
    _trk_names  = [r["Tracker"] for r in _rows_b]
    _trk_colors = ["#2ca02c" if r["Selected"] == _CHECK else "#1f77b4" for r in _rows_b]
    fig_b_idf1 = go.Figure(go.Bar(
        x=_trk_names,
        y=[r["IDF1 (mean)"] for r in _rows_b],
        marker_color=_trk_colors,
        text=[f"{r['IDF1 (mean)']:.4f}" for r in _rows_b],
        textposition="outside",
    ))
    fig_b_idf1.update_layout(
        title="Block B — IDF1 Score (primary selection criterion: identity consistency for Block C)",
        yaxis=dict(
            range=[0, max(r["IDF1 (mean)"] for r in _rows_b) * 1.5 + 0.005],
            title="IDF1"
        ),
        xaxis_title="Tracker",
        height=380,
        showlegend=False,
    )
    st.plotly_chart(fig_b_idf1, use_container_width=True)

    # MOTA chart
    fig_b_mota = go.Figure(go.Bar(
        x=_trk_names,
        y=[r["MOTA (mean)"] for r in _rows_b],
        marker_color=_trk_colors,
        text=[f"{r['MOTA (mean)']:.4f}" for r in _rows_b],
        textposition="outside",
    ))
    fig_b_mota.update_layout(
        title="Block B — MOTA (negative due to heavy occlusion in UA-DETRAC; max_age=5 frames)",
        yaxis_title="MOTA",
        xaxis_title="Tracker",
        height=380,
        showlegend=False,
    )
    st.plotly_chart(fig_b_mota, use_container_width=True)

    # Per-sequence breakdown
    with st.expander("Per-Sequence Breakdown (MVI_20062 / MVI_20063)", expanded=False):
        for t in _bb["trackers"]:
            sel_badge = " ✓ Selected" if t.get("selected") else ""
            st.write(f"**{t['tracker'].upper()}{sel_badge}**")
            _seq_rows = []
            for seq_id, seq_data in t.get("per_seq", {}).items():
                _seq_rows.append({
                    "Sequence":       seq_id,
                    "MOTA":           round(seq_data.get("mota", float("nan")), 4),
                    "IDF1":           round(seq_data.get("idf1", float("nan")), 4),
                    "ID Switches":    seq_data.get("idsw", "—"),
                    "Fragmentations": seq_data.get("fragmentations", "—"),
                    "FPS":            seq_data.get("fps", seq_data.get("fps_median", "—")),
                })
            st.dataframe(pd.DataFrame(_seq_rows), use_container_width=True)

    st.caption(f"""
    - **Selection criterion:** {_bb['selection_criterion']}
    - **Why MOTA is negative:** UA-DETRAC sequences have dense occlusion (>50% overlap periods);
      SORT uses max_age=5 to prevent ID fragmentation. False tracks accumulate, dragging MOTA below zero.
      IDF1 is the more informative metric for identity continuity needed by Block C.
    - All trackers run on YOLOv8n detection cache (Block A frozen output, conf=0.25, nms=0.45)
    - DeepSORT uses VeRi-776 ReID (osnet_x1_0_veri_776); SORT and ByteTrack use IoU-only association
    """)

else:
    st.warning("Block B results not found at `logs/block_b_results.json`")
    st.info("Run Block B evaluation (Days 7–9) to generate results.")

st.markdown("---")

# =============================================================================
# BLOCK C — FEATURE SET COMPARISON (logs/block_c/)
# =============================================================================
st.header("Block C — Feature Set Comparison")

# ── Load actual feature CSVs ─────────────────────────────────────────────────
feature_csvs = sorted(glob.glob(os.path.join(_BLOCK_C_DIR, "*_features.csv")))

if feature_csvs:
    st.success(
        f"**Block C feature CSVs found** — {len(feature_csvs)} sequences in `logs/block_c/`"
        " (schema v1.1: F1 + F2 + F3 features)"
    )

    dfs = []
    for p in feature_csvs:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            st.warning(f"Could not load {os.path.basename(p)}: {e}")

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)

        # Summary metrics
        st.subheader("Feature Dataset Summary (F1 + F2 + F3 — schema v1.1)")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total rows",        f"{len(df_all):,}")
        col2.metric("Sequences",         df_all["seq_id"].nunique() if "seq_id" in df_all else "?")
        col3.metric("Unique tracks",     df_all["track_id"].nunique() if "track_id" in df_all else "?")
        col4.metric("Interpolated rows", f"{int(df_all['is_interpolated'].sum()):,}" if "is_interpolated" in df_all else "?")
        col5.metric("F3 non-NaN rows",
                    f"{int(df_all['inter_vehicle_dist_norm'].notna().sum()):,}"
                    if "inter_vehicle_dist_norm" in df_all else "?")

        # Split breakdown
        if "split" in df_all.columns:
            st.write("**Row counts per split:**")
            split_counts = df_all["split"].value_counts().reset_index()
            split_counts.columns = ["split", "rows"]
            st.dataframe(split_counts, use_container_width=True)

        # F1: vehicle count distribution
        if "vehicle_count" in df_all.columns and "seq_id" in df_all.columns:
            st.subheader("F1: Vehicle Count Distribution per Sequence")
            vc_stats = (
                df_all.groupby("seq_id")["vehicle_count"]
                .agg(["mean", "max"])
                .reset_index()
                .rename(columns={"mean": "Mean Count", "max": "Max Count"})
            )
            fig_vc = go.Figure()
            fig_vc.add_trace(go.Bar(x=vc_stats["seq_id"], y=vc_stats["Mean Count"],
                                     name="Mean", marker_color="#1f77b4"))
            fig_vc.add_trace(go.Bar(x=vc_stats["seq_id"], y=vc_stats["Max Count"],
                                     name="Max",  marker_color="#ff7f0e"))
            fig_vc.update_layout(barmode="group",
                                  title="Vehicle Count (mean / max) per Sequence",
                                  xaxis_title="Sequence", yaxis_title="Vehicle Count", height=380)
            st.plotly_chart(fig_vc, use_container_width=True)

        # F2: speed distribution
        if "vel_px_sec_smooth" in df_all.columns:
            st.subheader("F2: Smoothed Speed Distribution (vel_px_sec_smooth — px/sec)")
            speed_data = df_all["vel_px_sec_smooth"].dropna()
            fig_sp = go.Figure(go.Histogram(
                x=speed_data, nbinsx=60,
                marker_color="#2ca02c", opacity=0.75, name="vel_px_sec_smooth"
            ))
            fig_sp.update_layout(
                title="Distribution of vel_px_sec_smooth across all sequences",
                xaxis_title="Speed (px/sec, 5-frame rolling mean at 25 FPS)",
                yaxis_title="Count", height=350
            )
            st.plotly_chart(fig_sp, use_container_width=True)
            st.caption(
                "speed_window = int(0.2 × fps) = 5 frames at 25 FPS. "
                "Image-space proxy — not calibrated to km/h."
            )

        # F3: inter-vehicle distance distribution
        if "inter_vehicle_dist_norm" in df_all.columns:
            st.subheader("F3: Inter-Vehicle Distance Distribution (inter_vehicle_dist_norm)")
            dist_data = df_all["inter_vehicle_dist_norm"].dropna()
            fig_dist = go.Figure(go.Histogram(
                x=dist_data, nbinsx=60,
                marker_color="#9467bd", opacity=0.75, name="inter_vehicle_dist_norm"
            ))
            fig_dist.update_layout(
                title="Distribution of inter_vehicle_dist_norm across all sequences",
                xaxis_title="Normalised distance (frame-diagonal proxy — NOT physical)",
                yaxis_title="Count", height=350
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption(
                "⚠️ Image-space proxy normalised by frame diagonal (640×√2 ≈ 905 px). "
                "NaN when only one track present in frame (single-track frames excluded from F3 sets). "
                "Perspective effects mean this is NOT physical distance."
            )

        # Per-sequence feature stats
        st.subheader("Per-Sequence Feature Statistics (mean across all rows)")
        stat_cols = [c for c in [
            "vehicle_count", "roi_occupancy",
            "vel_px_sec", "vel_px_sec_smooth",
            "inter_vehicle_dist_norm", "dwell_time_sec", "proximity_count_rolling"
        ] if c in df_all.columns]
        if stat_cols and "seq_id" in df_all.columns:
            stats_df = df_all.groupby("seq_id")[stat_cols].mean().round(4).reset_index()
            st.dataframe(stats_df, use_container_width=True)

else:
    st.info(
        "**Block C feature CSVs not found locally.** "
        "Pull `logs/block_c/*_features.csv` from the research environment."
    )

st.markdown("---")

# ── OC-SVM AUROC comparison (Day 12) ─────────────────────────────────────────
st.subheader("Feature Set AUROC Comparison — OC-SVM (Block C Day 12)")

_RESULTS_TABLE_CSV = os.path.normpath(os.path.join(_BLOCK_C_DIR, "block_c_results_table.csv"))
_DAY12_META_JSON   = os.path.normpath(os.path.join(_BLOCK_C_DIR, "block_c_day12_metadata.json"))

_best_fs_for_summary    = None
_best_auroc_for_summary = None
_block_c_day12_done     = False

if os.path.exists(_RESULTS_TABLE_CSV):
    try:
        df_auroc = pd.read_csv(_RESULTS_TABLE_CSV)

        _brow = df_auroc[df_auroc["Selected"].astype(str).str.strip() == _CHECK]
        if not _brow.empty:
            _best_fs_for_summary    = str(_brow.iloc[0]["Feature Set"])
            _best_auroc_for_summary = float(_brow.iloc[0]["AUROC"])
            _block_c_day12_done     = True
            st.success(
                f"**Block C Day 12 complete — Best feature set: `{_best_fs_for_summary}`** &nbsp;|&nbsp; "
                f"AUROC = {_best_auroc_for_summary:.4f} &nbsp;|&nbsp; "
                f"Silhouette = {float(_brow.iloc[0]['Silhouette']):.4f} &nbsp;|&nbsp; "
                f"best nu = {_brow.iloc[0]['Best Nu']}"
            )
        else:
            st.success("**Block C Day 12 results loaded.**")

        st.dataframe(df_auroc, use_container_width=True)

        # AUROC bar chart
        _vauc = df_auroc[df_auroc["AUROC"].notna()].copy()
        if not _vauc.empty:
            _bar_cols = [
                "#2ca02c" if str(r["Selected"]).strip() == _CHECK else "#1f77b4"
                for _, r in _vauc.iterrows()
            ]
            fig_auroc = go.Figure(go.Bar(
                x=_vauc["Feature Set"],
                y=_vauc["AUROC"].astype(float),
                marker_color=_bar_cols,
                text=[f"{v:.4f}" for v in _vauc["AUROC"].astype(float)],
                textposition="outside",
            ))
            fig_auroc.update_layout(
                title="OC-SVM AUROC by Feature Set — 3-sigma train-stats proxy labels",
                xaxis_title="Feature Set",
                yaxis=dict(range=[0, 1.15], title="AUROC"),
                height=430,
                showlegend=False,
            )
            st.plotly_chart(fig_auroc, use_container_width=True)

        # Silhouette bar chart
        _vsil = df_auroc[df_auroc["Silhouette"].notna()].copy()
        if not _vsil.empty:
            _sil_cols = [
                "#2ca02c" if str(r["Selected"]).strip() == _CHECK else "#9467bd"
                for _, r in _vsil.iterrows()
            ]
            fig_sil = go.Figure(go.Bar(
                x=_vsil["Feature Set"],
                y=_vsil["Silhouette"].astype(float),
                marker_color=_sil_cols,
                text=[f"{v:.4f}" for v in _vsil["Silhouette"].astype(float)],
                textposition="outside",
            ))
            fig_sil.update_layout(
                title="Silhouette Score by Feature Set (val split, 3-sigma proxy labels)",
                xaxis_title="Feature Set",
                yaxis_title="Silhouette Score",
                height=380,
                showlegend=False,
            )
            st.plotly_chart(fig_sil, use_container_width=True)

        # Day 12 metadata expander
        if os.path.exists(_DAY12_META_JSON):
            with st.expander("Day 12 Provenance Metadata", expanded=False):
                try:
                    with open(_DAY12_META_JSON, encoding="utf-8") as fh:
                        m12 = json.load(fh)
                    st.json({k: v for k, v in m12.items() if k != "y_val_binary_note"})
                    st.caption(m12.get("y_val_binary_note", ""))
                except Exception as me:
                    st.warning(f"Could not read metadata: {me}")

        st.caption(
            "AUROC: OC-SVM decision-function scores on "
            "UA-DETRAC val split (MVI_20061). y_val_binary uses a 3-sigma threshold "
            "from train-split statistics only — no val data contaminates labels. "
            "Measures feature separability of statistical outliers within a normal "
            "distribution, NOT true anomaly detection. Block D (AI City) provides true GT. "
            "Silhouette: cluster separation under proxy labels. "
            "nu swept over {0.01, 0.02, 0.05, 0.10, 0.15} on 80/20 train "
            "hold-out. Row counts may differ for F3-containing sets due to NaN semantics "
            "(single-track frames have no inter-vehicle distance)."
        )

    except Exception as err:
        st.warning(f"Could not load Day 12 AUROC results: {err}")
else:
    st.info(
        "Block C Day 12 AUROC results not found. "
        "Run `PipelineController.run_block_c_day12()` and pull "
        "`logs/block_c/block_c_results_table.csv` to this repo."
    )

st.markdown("---")

# =============================================================================
# BLOCK D — ANOMALY DETECTION METHOD COMPARISON (Days 15–17, complete)
# =============================================================================
st.header("Block D — Anomaly Detection Method Comparison")

_BD_DIR     = os.path.join(_LOGS_DIR, "block_d")
_bd_results = os.path.join(_BD_DIR, "block_d_complete_results.csv")
_bd_events  = os.path.join(_BD_DIR, "block_d_event_predictions.csv")
_bd_far_sum = os.path.join(_BD_DIR, "day17_far_summary.csv")
_bd_latency = os.path.join(_BD_DIR, "day17_latency_profile.json")
_bd_day18   = os.path.join(_BD_DIR, "day18_report.json")

if os.path.exists(_bd_results):
    try:
        df_d_main = pd.read_csv(_bd_results)

        st.success(
            "**Block D complete (Days 15–17)** — AI City Track 4 evaluation "
            "| FAR gate threshold = 0.10 | Detector: yolov8n | Tracker: sort | Features: F2_only"
        )

        # ── Event detection coverage ─────────────────────────────────────────
        st.subheader("Event Detection Coverage (AI City Track 4)")
        _col_map = {
            "method":                   "Method",
            "total_events":             "Total Events",
            "events_detected":          "Events Detected",
            "detection_coverage_pct":   "Coverage (%)",
            "mean_far":                 "Mean FAR",
            "std_far":                  "Std FAR",
            "max_far":                  "Max FAR",
            "far_gate_passed":          "FAR Gate (< 0.10)",
            "notes":                    "Notes",
        }
        _disp_cols = [c for c in _col_map if c in df_d_main.columns]
        df_d_disp = df_d_main[_disp_cols].rename(columns=_col_map)
        st.dataframe(df_d_disp, use_container_width=True)

        # ── FAR bar chart ─────────────────────────────────────────────────────
        if "method" in df_d_main.columns and "mean_far" in df_d_main.columns:
            _gate_colors = [
                "#2ca02c" if str(r["far_gate_passed"]).strip() == "True" else "#d62728"
                for _, r in df_d_main.iterrows()
            ]
            fig_d = go.Figure()
            fig_d.add_trace(go.Bar(
                x=df_d_main["method"],
                y=df_d_main["mean_far"].astype(float),
                error_y=dict(type="data", array=df_d_main["std_far"].astype(float)),
                marker_color=_gate_colors,
                text=[f"{v:.4f}" for v in df_d_main["mean_far"].astype(float)],
                textposition="outside",
                name="Mean FAR",
            ))
            fig_d.add_hline(
                y=0.10, line_dash="dash", line_color="crimson",
                annotation_text="FAR gate threshold = 0.10",
                annotation_position="top left",
            )
            fig_d.update_layout(
                title="Block D — Mean False Alarm Rate per Method (10 normal clips, AI City Track 4)",
                xaxis_title="Anomaly Method",
                yaxis=dict(title="Mean FAR", range=[0, min(1.05, df_d_main["max_far"].max() * 1.2)]),
                height=430,
                showlegend=False,
            )
            st.plotly_chart(fig_d, use_container_width=True)
            st.caption(
                "Green bars = FAR gate passed (mean FAR < 0.10). "
                "Red bar = Isolation Forest flagged (mean FAR = 0.3354 ≥ threshold — "
                "results reportable but annotated). "
                "Error bars = ±1 std FAR across 10 normal clips."
            )

        # ── FAR summary per clip ─────────────────────────────────────────────
        if os.path.exists(_bd_far_sum):
            with st.expander("FAR Summary per Method (day17_far_summary.csv)", expanded=False):
                st.dataframe(pd.read_csv(_bd_far_sum), use_container_width=True)

        # ── Latency profile ──────────────────────────────────────────────────
        if os.path.exists(_bd_latency):
            with st.expander("Latency Profile (Day 17 — indicative, offline batch)", expanded=False):
                with open(_bd_latency, encoding="utf-8") as _fh:
                    _lat = json.load(_fh)
                _stages = _lat.get("stages", {})
                col_l1, col_l2 = st.columns([1, 2])
                with col_l1:
                    st.metric("End-to-end FPS", f"{_lat.get('end_to_end_fps', '—')} fps")
                    st.metric("Total mean ms", _stages.get("total", {}).get("mean_ms", "—"))
                    st.metric("Total p95 ms",  _stages.get("total", {}).get("p95_ms",  "—"))
                    st.metric("Bottleneck",     "anomaly stage")
                with col_l2:
                    _lat_rows = [
                        {"Stage": s,
                         "mean ms": _stages[s]["mean_ms"],
                         "p50 ms":  _stages[s]["p50_ms"],
                         "p95 ms":  _stages[s]["p95_ms"]}
                        for s in ["decode", "detect", "track", "feature", "anomaly", "total"]
                        if s in _stages
                    ]
                    st.dataframe(pd.DataFrame(_lat_rows), use_container_width=True)
                st.caption(_lat.get("measurement_caveats", ""))

        # ── Day 18 statistical tests ─────────────────────────────────────────
        if os.path.exists(_bd_day18):
            with st.expander("Day 18 — Statistical Tests", expanded=False):
                with open(_bd_day18, encoding="utf-8") as _fh:
                    _d18 = json.load(_fh)
                _tests = _d18.get("statistical_tests", {})
                _w = _tests.get("test1_wilcoxon",  {})
                _m = _tests.get("test2_mcnemar",   {})
                _p = _tests.get("test3_pearson_r", {})
                _summary_rows = []
                for _t, _label in [(_w, "Wilcoxon"), (_m, "McNemar (exact)"), (_p, "Pearson r")]:
                    if _t:
                        _summary_rows.append({
                            "Test":          _label,
                            "Comparison":    _t.get("hypothesis", ""),
                            "n":             _t.get("n_pairs") or _t.get("n_clips"),
                            "p-value":       round(float(_t.get("p_value", 1.0)), 4),
                            "Significant":   "✓" if _t.get("significant_05") else "✗",
                        })
                if _summary_rows:
                    st.dataframe(pd.DataFrame(_summary_rows), use_container_width=True)

        st.caption(
            "All 10 anomaly events detected by all methods (event_pred=1, coverage=100%). "
            "Key differentiator is FAR (false alarm rate on normal clips). "
            "OC-SVM (mean_FAR=0.0283) and rule_based (mean_FAR=0.035) pass the gate. "
            "Isolation Forest fails (mean_FAR=0.3354) — annotated."
        )

    except Exception as _err:
        st.warning(f"Could not display Block D results: {_err}")
else:
    st.info(
        "Block D results not found at `logs/block_d/block_d_complete_results.csv`. "
        "Run `block_d_day17_far_latency.py` (Day 17) to generate results."
    )

st.markdown("---")

# =============================================================================
# SUMMARY
# =============================================================================
st.header("Key Findings")

col1, col2, col3 = st.columns(3)

with col1:
    if _ba:
        st.metric(
            "Best Detector",
            "YOLOv8n (fine-tuned)",
            f"mAP@0.5 = {_ba['detectors']['yolov8n']['mAP50']:.4f}"
        )
    else:
        st.metric("Best Detector", "YOLOv8n (fine-tuned)", "mAP@0.5 = 0.6674")

    if _bb and _sel_trk:
        st.metric(
            "Best Tracker",
            _sel_trk["tracker"].upper(),
            f"IDF1 = {_sel_trk['idf1_mean']:.4f}"
        )
    else:
        st.metric("Best Tracker", "SORT", "IDF1 = 0.0412")

with col2:
    st.metric(
        "Best Feature Set",
        _best_fs_for_summary if _block_c_day12_done else "F2_only",
        f"AUROC = {_best_auroc_for_summary:.4f}" if _block_c_day12_done else "AUROC = 0.9935",
    )
    st.metric("Best Anomaly Method", "OC-SVM", "mean FAR = 0.0283 ✅ gate passed")

with col3:
    st.metric("Scaler", "MinMaxScaler", "Fitted on UA-DETRAC train (normal only)")
    st.metric("Seed", "42", "All blocks, all splits")

st.info("""
**Key Insight (Block C):** F2_only (speed proxy) achieves AUROC = 0.9935 — far above
F1 (density/flow, AUROC = 0.4724), F2+F3 (AUROC = 0.641), and All Features (AUROC = 0.7312).
Adding F3 interaction features reduces separability on UA-DETRAC normal-traffic validation.

**Key Insight (Block D):** All three methods detect 100% of AI City Track 4 anomaly events.
The discriminator is FAR: OC-SVM (0.0283) and rule_based (0.035) pass the gate;
Isolation Forest (0.3354) fails — annotated. Latency: 21.11 FPS end-to-end,
bottleneck is the anomaly stage (43.48 ms mean). McNemar test (p=0.031) confirms
IF vs rule_based gate-failure difference is statistically significant.
""")
