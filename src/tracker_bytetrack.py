"""
tracker_bytetrack.py
ByteTrack multi-object tracker — Block B
CST3990 | MANJOO Ameera Najla | M01014463

Algorithm: two-pass IoU + Kalman filter association.
No neural network weights — purely detection-score-based recovery.

Fixed constraints (inherited from project-wide rules):
  - Seeds: torch 42, numpy 42, random 42
  - Detections read exclusively from detections_cache/ — detector never re-run
  - bbox coordinates in 640x640 space throughout
  - frame_idx is 0-based (cache_frame_index_base = 0)
  - warmup_boundary frames excluded from evaluation
  - Fragment filter: COMPUTED and LOGGED, NOT enforced (deferred to Day 9)
  - Track interpolation: gap sizes COMPUTED and LOGGED, NOT applied (deferred to Day 9)
  - reid_model: "none" — ByteTrack has no ReID component
  - FPS benchmark: pre-load frames before timing; time tracker update only
"""

import os
import sys
import json
import math
import time
import logging
import random
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import motmetrics as mm

# Seeds — set before any model initialisation
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

try:
    from src.seed_control import set_all_seeds
    set_all_seeds()
except ImportError:
    try:
        from seed_control import set_all_seeds
        set_all_seeds()
    except ImportError:
        pass  # seeds already set above

try:
    from src.cache_utils import load_cache
    from src.gt_loader import load_gt_sequence, get_gt_xml_path
except ImportError:
    from cache_utils import load_cache
    from gt_loader import load_gt_sequence, get_gt_xml_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kalman filter tracker (same state vector as sort_tracker.py for consistency)
# State: [x1, y1, x2, y2, vx, vy, vw, vh]
# ---------------------------------------------------------------------------

class _KalmanTracker:
    """Single object tracked with a Kalman filter."""

    _count = 0  # class-level counter; reset via reset_count() between sequences

    @classmethod
    def reset_count(cls):
        cls._count = 0

    def __init__(self, bbox, conf=1.0):
        _KalmanTracker._count += 1
        self.track_id = _KalmanTracker._count
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.conf = conf
        self.kf = self._build_kf(bbox)

    @staticmethod
    def _build_kf(bbox):
        kf = KalmanFilter(dim_x=8, dim_z=4)
        # Constant-velocity model: pos += vel each step
        kf.F = np.eye(8)
        for i in range(4):
            kf.F[i, i + 4] = 1.0
        kf.H = np.eye(4, 8)
        kf.R[2:, 2:] *= 10.0
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        kf.x[:4] = np.array(bbox, dtype=float).reshape(4, 1)
        return kf

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_bbox()

    def update(self, bbox, conf):
        self.kf.update(np.array(bbox, dtype=float).reshape(4, 1))
        self.time_since_update = 0
        self.hits += 1
        self.conf = conf

    def get_bbox(self):
        return self.kf.x[:4].flatten().tolist()


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def _iou_matrix(bboxes_a, bboxes_b):
    """Compute pairwise IoU. Returns (N, M) matrix."""
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)))

    a = np.array(bboxes_a, dtype=float)
    b = np.array(bboxes_b, dtype=float)

    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter_w  = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h  = np.maximum(0.0, inter_y2 - inter_y1)
    inter    = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union   = area_a[:, None] + area_b[None, :] - inter
    iou     = np.where(union > 0, inter / union, 0.0)
    return iou


def _linear_assign(cost_matrix):
    """Run Hungarian algorithm on a cost matrix. Returns (row_ids, col_ids)."""
    return linear_sum_assignment(cost_matrix)


# ---------------------------------------------------------------------------
# ByteTracker — two-pass association
# ---------------------------------------------------------------------------

class ByteTracker:
    """
    ByteTrack implementation.
    Thresholds match config_blockB.yaml exactly (Fix F19):
      track_high_thresh: 0.50
      track_low_thresh:  0.10
      match_thresh:      0.80
      max_time_lost:     30
    """

    def __init__(self, track_high_thresh=0.50, track_low_thresh=0.10,
                 match_thresh=0.80, max_time_lost=30, min_hits=3):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh  = track_low_thresh
        self.match_thresh      = match_thresh
        self.max_time_lost     = max_time_lost
        self.min_hits          = min_hits
        self.active_tracks     = []  # confirmed + tentative

    def reset(self):
        self.active_tracks = []
        _KalmanTracker.reset_count()

    def update(self, detections, frame_idx):
        """
        One frame update.

        Args:
            detections: list of dicts with keys 'bbox' and 'conf'
            frame_idx:  0-based frame index

        Returns:
            list of (x1, y1, x2, y2, track_id, conf) tuples for active tracks
        """
        # --- Predict all active tracks ---
        predicted_bboxes = []
        for t in self.active_tracks:
            predicted_bboxes.append(t.predict())

        # --- Split detections by confidence ---
        high_dets = [(d['bbox'], d.get('conf', 1.0))
                     for d in detections if d.get('conf', 1.0) >= self.track_high_thresh]
        low_dets  = [(d['bbox'], d.get('conf', 1.0))
                     for d in detections
                     if self.track_low_thresh <= d.get('conf', 1.0) < self.track_high_thresh]

        # ---------------------------------------------------------------
        # PASS 1: match active tracks to high-confidence detections
        # ---------------------------------------------------------------
        matched_track_ids = set()
        matched_det_ids   = set()

        if self.active_tracks and high_dets:
            iou = _iou_matrix(predicted_bboxes,
                              [hd[0] for hd in high_dets])
            cost = 1.0 - iou
            cost[cost > (1.0 - self.match_thresh)] = 1.0  # gate: reject low IoU

            row_ids, col_ids = _linear_assign(cost)
            for r, c in zip(row_ids, col_ids):
                if cost[r, c] < 1.0:
                    self.active_tracks[r].update(high_dets[c][0], high_dets[c][1])
                    matched_track_ids.add(r)
                    matched_det_ids.add(c)

        unmatched_tracks = [i for i in range(len(self.active_tracks))
                            if i not in matched_track_ids]
        unmatched_high   = [i for i in range(len(high_dets))
                            if i not in matched_det_ids]

        # ---------------------------------------------------------------
        # PASS 2: match remaining tracks to low-confidence detections
        # ---------------------------------------------------------------
        if unmatched_tracks and low_dets:
            pred_unmatched = [predicted_bboxes[i] for i in unmatched_tracks]
            iou = _iou_matrix(pred_unmatched, [ld[0] for ld in low_dets])
            cost = 1.0 - iou
            cost[cost > (1.0 - self.match_thresh)] = 1.0

            row_ids, col_ids = _linear_assign(cost)
            for r, c in zip(row_ids, col_ids):
                if cost[r, c] < 1.0:
                    track_i = unmatched_tracks[r]
                    self.active_tracks[track_i].update(
                        low_dets[c][0], low_dets[c][1])
                    matched_track_ids.add(track_i)

        # ---------------------------------------------------------------
        # Initialise new tracks from unmatched high-confidence detections
        # ---------------------------------------------------------------
        for i in unmatched_high:
            self.active_tracks.append(_KalmanTracker(high_dets[i][0], high_dets[i][1]))

        # ---------------------------------------------------------------
        # Remove tracks lost > max_time_lost frames
        # ---------------------------------------------------------------
        self.active_tracks = [
            t for t in self.active_tracks
            if t.time_since_update <= self.max_time_lost
        ]

        # ---------------------------------------------------------------
        # Output confirmed tracks (hits >= min_hits)
        # ---------------------------------------------------------------
        output = []
        for t in self.active_tracks:
            if t.hits >= self.min_hits or t.time_since_update == 0:
                bbox = t.get_bbox()
                output.append((bbox[0], bbox[1], bbox[2], bbox[3],
                                t.track_id, t.conf))
        return output


# ---------------------------------------------------------------------------
# Helpers shared with DeepSORT (velocity, gap analysis)
# ---------------------------------------------------------------------------

def _compute_velocity_norm(prev_cx, prev_cy, curr_cx, curr_cy, fps, frame_diagonal):
    if prev_cx is None or frame_diagonal == 0:
        return None
    dist_px = math.sqrt((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2)
    return (dist_px * fps) / frame_diagonal


def _compute_track_gaps(frame_entries):
    """Compute gap analysis for a single track. Entries must be sorted by frame_idx."""
    sorted_e = sorted(frame_entries, key=lambda e: e['frame_idx'])
    gaps = []
    for i in range(1, len(sorted_e)):
        gap = sorted_e[i]['frame_idx'] - sorted_e[i - 1]['frame_idx'] - 1
        if gap > 0:
            gaps.append({
                "after_frame_idx": sorted_e[i - 1]['frame_idx'],
                "gap_size": gap,
                "interpolatable": gap <= 3
            })
    return {
        "max_gap": max((g["gap_size"] for g in gaps), default=0),
        "gap_count": len(gaps),
        "gaps": gaps
    }


# ---------------------------------------------------------------------------
# motmetrics evaluation (mirrors Day 7 SORT approach exactly)
# ---------------------------------------------------------------------------

def _evaluate(seq_id, trajectories_dict, gt_xml_path,
              warmup_boundary, cache_frame_index_base=0, iou_thresh_eval=0.5):
    """Compute MOTA, IDF1, IDSW, Frag via py-motmetrics."""
    gt_frames = load_gt_sequence(gt_xml_path)
    acc = mm.MOTAccumulator(auto_id=True)

    pred_by_frame = defaultdict(dict)
    for tid, entries in trajectories_dict.items():
        for entry in entries:
            fidx = entry['frame_idx']
            if fidx > warmup_boundary:
                pred_by_frame[fidx][tid] = entry['bbox']

    for gt_frame_num in sorted(gt_frames.keys()):
        fidx = gt_frame_num - 1 if cache_frame_index_base == 0 else gt_frame_num
        if fidx <= warmup_boundary:
            continue
        gt_dict   = gt_frames[gt_frame_num]
        pred_dict = pred_by_frame.get(fidx, {})
        gt_ids    = list(gt_dict.keys())
        pred_ids  = list(pred_dict.keys())
        if gt_ids and pred_ids:
            iou = _iou_matrix(
                [gt_dict[i]   for i in gt_ids],
                [pred_dict[i] for i in pred_ids]
            )
            dist_matrix = np.where(iou >= iou_thresh_eval, 1.0 - iou, np.nan)
        else:
            dist_matrix = np.full((len(gt_ids), len(pred_ids)), np.nan)
        acc.update(gt_ids, pred_ids, dist_matrix)

    mh      = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['mota', 'idf1', 'num_switches', 'num_fragmentations',
                 'num_objects', 'num_predictions', 'precision', 'recall'],
        name=seq_id
    )
    int_keys = {'num_switches', 'num_fragmentations', 'num_objects', 'num_predictions'}
    return {
        k: (int(summary[k].iloc[0]) if k in int_keys else float(summary[k].iloc[0]))
        for k in ['mota', 'idf1', 'num_switches', 'num_fragmentations',
                  'num_objects', 'num_predictions', 'precision', 'recall']
    }


# ---------------------------------------------------------------------------
# Detect Day 7 output layout
# ---------------------------------------------------------------------------

def _detect_day7_layout():
    if os.path.exists("logs/block_b/sort/trajectories_MVI_20062.json"):
        return "variant_a"
    if os.path.exists("logs/block_b/trajectories/MVI_20062_sort.json"):
        return "variant_b"
    raise FileNotFoundError(
        "Cannot detect Day 7 SORT output layout.\n"
        "Expected one of:\n"
        "  logs/block_b/sort/trajectories_MVI_20062.json  (variant A)\n"
        "  logs/block_b/trajectories/MVI_20062_sort.json  (variant B)\n"
        "Run Day 7 notebook first."
    )


# ---------------------------------------------------------------------------
# Single-sequence runner
# ---------------------------------------------------------------------------

def run_bytetrack_on_sequence(seq_id, split, config,
                              gt_root=None,
                              day7_layout=None):
    """
    Run ByteTrack on one sequence and return metrics + trajectory path.

    Args:
        seq_id:       sequence ID, e.g. "MVI_20062"
        split:        "test"
        config:       dict loaded from config_blockB.yaml
        gt_root:      path to UA-DETRAC XML annotations directory.
                      Default: data/ua_detrac/annotations (local).
                      In Colab set to /content/drive/MyDrive/CST3990/datasets/ua_detrac/annotations
        day7_layout:  "variant_a" | "variant_b" | None (auto-detect)
    """
    np.random.seed(42)
    random.seed(42)

    if day7_layout is None:
        day7_layout = _detect_day7_layout()

    if gt_root is None:
        gt_root = "data/ua_detrac/annotations"

    # --- Load config values ---
    cache_dir   = config.get('detections_source', 'detections_cache/')
    bt_cfg      = config.get('bytetrack', {})
    track_high  = bt_cfg.get('track_high_thresh', 0.50)
    track_low   = bt_cfg.get('track_low_thresh',  0.10)
    match_thresh= bt_cfg.get('match_thresh',       0.80)
    max_lost    = bt_cfg.get('max_time_lost',      30)
    min_hits    = config.get('sort', {}).get('min_hits', 3)  # reuse SORT min_hits
    min_track_len = config.get('min_track_length', 15)
    max_interp    = config.get('max_interp_gap',   3)
    iou_eval      = config.get('eval', {}).get('iou_threshold', 0.5)
    cache_base    = config.get('cache_frame_index_base', 0)

    # --- Load detection cache ---
    warmup_boundary, frame_results = load_cache(
        seq_id, split, 'yolov8n', cache_dir)

    # --- Load metadata ---
    with open('logs/metadata.json') as f:
        metadata = json.load(f)
    seq_meta      = metadata['sequences'][seq_id]
    fps           = seq_meta['fps']           # 25.0
    frame_width   = seq_meta['image_width']   # 960
    frame_height  = seq_meta['image_height']  # 540
    frame_diagonal = math.sqrt(frame_width ** 2 + frame_height ** 2)

    # --- Initialise tracker ---
    tracker = ByteTracker(
        track_high_thresh=track_high,
        track_low_thresh=track_low,
        match_thresh=match_thresh,
        max_time_lost=max_lost,
        min_hits=min_hits
    )
    tracker.reset()

    # --- Filter to post-warmup frames ---
    active_frames = [f for f in frame_results if f['frame_idx'] > warmup_boundary]

    # -----------------------------------------------------------------------
    # FPS benchmark — time tracker update only (no image I/O to pre-load here;
    # ByteTrack has no ReID, so there is no image dependency in the update loop).
    # This is consistent with DeepSORT's approach of pre-loading images before
    # timing — both methods isolate tracker logic from I/O.
    # -----------------------------------------------------------------------
    raw_trajectories = defaultdict(list)
    prev_centroids   = {}
    latencies        = []

    for frame_data in active_frames:
        frame_idx  = frame_data['frame_idx']
        detections = frame_data.get('detections', [])

        t0      = time.perf_counter()
        tracked = tracker.update(detections, frame_idx)
        t1      = time.perf_counter()
        latencies.append(t1 - t0)

        for (x1, y1, x2, y2, tid, conf) in tracked:
            tid = int(tid)
            cx  = (x1 + x2) / 2.0
            cy  = (y1 + y2) / 2.0
            prev = prev_centroids.get(tid)
            vel_n = _compute_velocity_norm(
                prev[0] if prev else None,
                prev[1] if prev else None,
                cx, cy, fps, frame_diagonal
            )
            prev_centroids[tid] = (cx, cy)

            raw_trajectories[tid].append({
                'frame_idx'    : int(frame_idx),
                'bbox'         : [float(x1), float(y1), float(x2), float(y2)],
                'cx'           : float(cx),
                'cy'           : float(cy),
                'velocity_norm': float(vel_n) if vel_n is not None else None,
                'conf'         : float(conf),
                'interpolated' : False,
                'track_length' : 0  # placeholder; filled after all frames
            })

    # --- Compute track lengths (Fix F11: compute, do NOT discard) ---
    for tid, entries in raw_trajectories.items():
        tlen = len(entries)
        for e in entries:
            e['track_length'] = tlen

    # --- Compute gap analysis per track (Fix F22: compute, do NOT interpolate) ---
    gap_analysis = {}
    for tid, entries in raw_trajectories.items():
        gap_analysis[str(tid)] = _compute_track_gaps(entries)

    fps_median     = float(1.0 / np.median(latencies)) if latencies else 0.0
    latency_p95_ms = float(np.percentile(latencies, 95) * 1000) if latencies else 0.0

    # --- Determine output paths based on detected Day 7 layout ---
    if day7_layout == "variant_a":
        traj_dir  = "logs/block_b/bytetrack"
        traj_path = os.path.join(traj_dir, f"trajectories_{seq_id}.json")
    else:
        traj_dir  = "logs/block_b/trajectories"
        traj_path = os.path.join(traj_dir, f"{seq_id}_bytetrack.json")
    os.makedirs(traj_dir, exist_ok=True)

    metrics_dir  = "logs/block_b/bytetrack"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"metrics_{seq_id}.json")

    # --- Save trajectory JSON ---
    traj_out = {
        "seq_id"                  : seq_id,
        "detector"                : "yolov8n",
        "tracker"                 : "bytetrack",
        "reid_model"              : "none",
        "cache_frame_index_base"  : int(cache_base),
        "warmup_boundary"         : int(warmup_boundary),
        "track_high_thresh"       : float(track_high),
        "track_low_thresh"        : float(track_low),
        "match_thresh"            : float(match_thresh),
        "max_time_lost"           : int(max_lost),
        "min_track_length"        : int(min_track_len),
        "max_interp_gap"          : int(max_interp),
        "fragment_filter_applied" : False,
        "interpolation_applied"   : False,
        "fps"                     : float(fps),
        "frame_width"             : int(frame_width),
        "frame_height"            : int(frame_height),
        "fps_median_tracker"      : round(fps_median, 2),
        "latency_p95_ms"          : round(latency_p95_ms, 4),
        "gap_analysis"            : gap_analysis,
        "tracks"                  : {str(k): v for k, v in raw_trajectories.items()}
    }
    with open(traj_path, 'w') as f:
        json.dump(traj_out, f, indent=2)
    print(f"  [ByteTrack] Saved trajectory: {traj_path}")

    # --- motmetrics evaluation ---
    gt_xml = get_gt_xml_path(seq_id, gt_root)
    metrics = _evaluate(
        seq_id,
        {str(k): v for k, v in raw_trajectories.items()},
        gt_xml,
        warmup_boundary,
        cache_base,
        iou_eval
    )

    n_tracks = len(raw_trajectories)
    print(f"  [ByteTrack] {seq_id}: tracks={n_tracks}  "
          f"MOTA={metrics['mota']:.3f}  IDF1={metrics['idf1']:.3f}  "
          f"IDSW={metrics['num_switches']}  Frag={metrics['num_fragmentations']}  "
          f"FPS={fps_median:.1f}")

    # --- Save per-sequence metrics JSON ---
    seq_metrics = {
        "seq_id"         : seq_id,
        "tracker"        : "bytetrack",
        "reid_model"     : "none",
        "mota"           : round(metrics['mota'],             4),
        "idf1"           : round(metrics['idf1'],             4),
        "idsw"           : metrics['num_switches'],
        "fragmentations" : metrics['num_fragmentations'],
        "fps_median"     : round(fps_median,     2),
        "latency_p95_ms" : round(latency_p95_ms, 4)
    }
    with open(metrics_path, 'w') as f:
        json.dump(seq_metrics, f, indent=2)

    return seq_metrics


# ---------------------------------------------------------------------------
# Multi-sequence entry point
# ---------------------------------------------------------------------------

def run_bytetrack_on_sequences(seq_ids, split, config,
                               gt_root=None,
                               day7_layout=None):
    """
    Run ByteTrack on multiple sequences and return aggregated results.

    Args:
        seq_ids:      list of sequence IDs
        split:        "test"
        config:       dict from config_blockB.yaml
        gt_root:      path to annotation XMLs
        day7_layout:  "variant_a" | "variant_b" | None (auto-detect)

    Returns:
        {
          "per_seq": {seq_id: {mota, idf1, idsw, fragmentations, fps_median, latency_p95_ms}},
          "aggregate": {mota_mean, idf1_mean, idsw_total, fragmentations_total, fps_median}
        }
    """
    if day7_layout is None:
        day7_layout = _detect_day7_layout()
    print(f"\n[ByteTrack] Day 7 layout: {day7_layout}")

    per_seq = {}
    for seq_id in seq_ids:
        print(f"\n{'='*60}")
        print(f"[ByteTrack] Running on {seq_id}")
        try:
            result = run_bytetrack_on_sequence(
                seq_id, split, config,
                gt_root=gt_root,
                day7_layout=day7_layout
            )
            per_seq[seq_id] = result
        except Exception as e:
            import traceback
            print(f"  [ByteTrack] ERROR on {seq_id}: {e}")
            traceback.print_exc()

    if not per_seq:
        return {"per_seq": {}, "aggregate": {}}

    motas  = [v['mota']           for v in per_seq.values()]
    idf1s  = [v['idf1']           for v in per_seq.values()]
    idsws  = [v['idsw']           for v in per_seq.values()]
    frags  = [v['fragmentations'] for v in per_seq.values()]
    fpss   = [v['fps_median']     for v in per_seq.values()]

    aggregate = {
        "mota_mean"            : round(float(np.mean(motas)), 6),
        "idf1_mean"            : round(float(np.mean(idf1s)), 6),
        "idsw_total"           : int(sum(idsws)),
        "fragmentations_total" : int(sum(frags)),
        "fps_median"           : round(float(np.median(fpss)), 2)
    }

    print(f"\n{'='*60}")
    print("BLOCK B — ByteTrack summary")
    print(f"  MOTA mean : {aggregate['mota_mean']:.3f}")
    print(f"  IDF1 mean : {aggregate['idf1_mean']:.3f}")
    print(f"  IDSW total: {aggregate['idsw_total']}")
    print(f"  Frag total: {aggregate['fragmentations_total']}")
    print(f"  FPS median: {aggregate['fps_median']:.1f}")

    return {"per_seq": per_seq, "aggregate": aggregate}
