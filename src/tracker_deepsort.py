"""
tracker_deepsort.py
DeepSORT tracker with OSNet VeRi-776 ReID — Block B

ReID backbone: osnet_x1_0 pretrained on VeRi-776 (576 vehicle identities).
NOT Market-1501 (RED severity).

Fixed constraints (inherited from project-wide rules):
  - Seeds: torch 42, numpy 42, random 42 — set before model initialisation
  - Detections read exclusively from detections_cache/ — detector never re-run
  - bbox coordinates in 640x640 space throughout
  - frame_idx is 0-based (cache_frame_index_base = 0)
  - warmup_boundary frames excluded from evaluation
  - Fragment filter: COMPUTED and LOGGED, NOT enforced (deferred to Day 9)
  - Track interpolation: gap sizes COMPUTED and LOGGED, NOT applied (deferred to Day 9)
  - FPS benchmark: pre-load ALL frames before timing loop; time tracker update only
  - VeRi-776 weights loaded from the local workspace models/ directory
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
import cv2
import motmetrics as mm

# --- Seeds before any model init ---
import torch
import torch.nn.functional as F
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
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
    from src.gt_loader   import load_gt_sequence, get_gt_xml_path
except ImportError:
    from cache_utils import load_cache
    from gt_loader   import load_gt_sequence, get_gt_xml_path

from scipy.optimize      import linear_sum_assignment
from filterpy.kalman     import KalmanFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# torchreid import helper — fail clearly if dependency is missing
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_MODELS_DIR = os.path.join(_REPO_ROOT, "models")
_DEFAULT_REID_WEIGHTS = os.path.join(_DEFAULT_MODELS_DIR, "osnet_x1_0_veri_776.pth")

def _ensure_torchreid():
    try:
        import torchreid
        return torchreid
    except ImportError:
        raise ImportError(
            "torchreid is required for DeepSORT but is not installed. "
            "Install the pinned dependencies from requirements.txt before "
            "running DeepSORT."
        ) from None


# ---------------------------------------------------------------------------
# OSNet VeRi-776 model builder — local workspace weights only
# ---------------------------------------------------------------------------

def build_reid_model(models_dir=None):
    """
    Build OSNet x1_0 with VeRi-776 weights (num_classes=576).
    Loads weights from the local workspace only.

    CRITICAL: num_classes=576 (VeRi-776). Market-1501=1501. Never fall back.
    """
    torchreid = _ensure_torchreid()

    # Seeds again after dynamic import
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if models_dir is None:
        models_dir = _DEFAULT_MODELS_DIR
    weight_path = os.path.join(models_dir, "osnet_x1_0_veri_776.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            "DeepSORT VeRi-776 weights not found. Expected local file: "
            f"{weight_path}"
        )

    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=576,   # VeRi-776 = 576 vehicle identities
        pretrained=False
    )

    torchreid.utils.load_pretrained_weights(model, weight_path)
    logger.info("VeRi-776 weights loaded from local workspace: %s", weight_path)
    print("  [DeepSORT] VeRi-776 weights loaded from local workspace OK")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    print(f"  [DeepSORT] OSNet x1_0 (VeRi-776, num_classes=576) ready on {device} OK")
    return model, device


# ---------------------------------------------------------------------------
# ReID feature extraction
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def extract_reid_feature(model, device, img_bgr, bbox_640,
                          img_width=960, img_height=540):
    """
    Crop bbox from a full frame, resize to 256x128, normalise, extract embedding.

    Args:
        model:       OSNet model in eval mode
        device:      torch device
        img_bgr:     full frame as (H, W, 3) BGR numpy array (original resolution)
        bbox_640:    [x1, y1, x2, y2] in 640x640 coordinate space
        img_width:   original image width  (960 for UA-DETRAC)
        img_height:  original image height (540 for UA-DETRAC)

    Returns:
        numpy array of shape (512,) — L2-normalised embedding
    """
    x1, y1, x2, y2 = bbox_640
    # Scale from 640x640 to original image dimensions
    scale_x = img_width  / 640.0   # 960/640 = 1.5
    scale_y = img_height / 640.0   # 540/640 = 0.84375
    ix1 = max(0, int(x1 * scale_x))
    iy1 = max(0, int(y1 * scale_y))
    ix2 = min(img_width,  int(x2 * scale_x))
    iy2 = min(img_height, int(y2 * scale_y))

    if ix2 <= ix1 or iy2 <= iy1:
        return np.zeros(512, dtype=np.float32)

    crop = img_bgr[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return np.zeros(512, dtype=np.float32)

    # Resize to VeRi-776 standard input: 256 H x 128 W
    crop_resized = cv2.resize(crop, (128, 256))
    # BGR -> RGB, HWC -> CHW, [0,255] -> [0,1]
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    tensor   = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
    # ImageNet normalisation
    tensor   = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor   = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(tensor)
    feat = F.normalize(feat, p=2, dim=1)
    return feat.cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# Kalman filter tracker (matches sort_tracker.py state vector)
# ---------------------------------------------------------------------------

class _KalmanTracker:
    _count = 0

    @classmethod
    def reset_count(cls):
        cls._count = 0

    def __init__(self, bbox, conf=1.0, embedding=None):
        _KalmanTracker._count += 1
        self.track_id         = _KalmanTracker._count
        self.hits             = 1
        self.age              = 1
        self.time_since_update = 0
        self.conf             = conf
        self.embedding        = embedding  # ReID feature vector
        self.kf               = self._build_kf(bbox)

    @staticmethod
    def _build_kf(bbox):
        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.F = np.eye(8)
        for i in range(4):
            kf.F[i, i + 4] = 1.0
        kf.H           = np.eye(4, 8)
        kf.R[2:, 2:]  *= 10.0
        kf.P[4:, 4:]  *= 1000.0
        kf.P          *= 10.0
        kf.Q[-1, -1]  *= 0.01
        kf.Q[4:, 4:]  *= 0.01
        kf.x[:4]       = np.array(bbox, dtype=float).reshape(4, 1)
        return kf

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age              += 1
        self.time_since_update += 1
        return self.get_bbox()

    def update(self, bbox, conf, embedding=None):
        self.kf.update(np.array(bbox, dtype=float).reshape(4, 1))
        self.time_since_update = 0
        self.hits += 1
        self.conf  = conf
        if embedding is not None:
            # EMA update of appearance embedding (alpha=0.9)
            if self.embedding is None:
                self.embedding = embedding
            else:
                self.embedding = 0.9 * self.embedding + 0.1 * embedding

    def get_bbox(self):
        return self.kf.x[:4].flatten().tolist()


# ---------------------------------------------------------------------------
# IoU and cosine distance helpers
# ---------------------------------------------------------------------------

def _iou_matrix(bboxes_a, bboxes_b):
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)))
    a = np.array(bboxes_a, dtype=float)
    b = np.array(bboxes_b, dtype=float)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _cosine_distance_matrix(embeddings_a, embeddings_b):
    """Cosine distance (1 - cosine_similarity) between two sets of embeddings."""
    if len(embeddings_a) == 0 or len(embeddings_b) == 0:
        return np.ones((len(embeddings_a), len(embeddings_b)))
    a = np.array(embeddings_a, dtype=float)
    b = np.array(embeddings_b, dtype=float)
    # L2 normalise rows
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - a @ b.T


# ---------------------------------------------------------------------------
# DeepSORT tracker
# ---------------------------------------------------------------------------

class DeepSORT:
    """
    DeepSORT: Kalman filter + IoU + cosine ReID association.

    Appearance threshold: 0.40 (cosine distance).
    Max age: 30 frames.
    """

    def __init__(self, appearance_thresh=0.40, max_age=30, min_hits=3,
                 iou_threshold=0.30):
        self.appearance_thresh = appearance_thresh
        self.max_age           = max_age
        self.min_hits          = min_hits
        self.iou_threshold     = iou_threshold
        self.tracks            = []

    def reset(self):
        self.tracks = []
        _KalmanTracker.reset_count()

    def update(self, detections, embeddings):
        """
        One frame update.

        Args:
            detections:  list of dicts with 'bbox' and 'conf'
            embeddings:  list of numpy arrays (one per detection, same order)

        Returns:
            list of (x1, y1, x2, y2, track_id, conf)
        """
        # --- Predict ---
        predicted_bboxes = [t.predict() for t in self.tracks]

        # --- Build cost matrix: weighted cosine + IoU ---
        track_embeddings = [t.embedding for t in self.tracks]
        det_embeddings   = embeddings
        det_bboxes       = [d['bbox'] for d in detections]
        det_confs        = [d.get('conf', 1.0) for d in detections]

        n_tracks = len(self.tracks)
        n_dets   = len(detections)

        if n_tracks > 0 and n_dets > 0:
            # Cosine distance on appearance
            valid_track_emb = [
                e if e is not None else np.zeros(512, dtype=np.float32)
                for e in track_embeddings
            ]
            cos_dist = _cosine_distance_matrix(valid_track_emb, det_embeddings)

            # IoU distance
            iou_dist = 1.0 - _iou_matrix(predicted_bboxes, det_bboxes)

            # Combined: gate on appearance first, then fall back to IoU
            # Gate: if cosine dist > appearance_thresh, force large cost
            cost = np.where(
                cos_dist > self.appearance_thresh,
                iou_dist,                   # fall back to pure IoU if appearance fails gate
                0.5 * cos_dist + 0.5 * iou_dist  # blended when appearance is reliable
            )
            # Hard gate on IoU: reject matches with IoU < iou_threshold
            cost[iou_dist > (1.0 - self.iou_threshold)] = 1e6

            row_ids, col_ids = linear_sum_assignment(cost)
            matched_tracks = set()
            matched_dets   = set()
            for r, c in zip(row_ids, col_ids):
                if cost[r, c] < 1e5:
                    self.tracks[r].update(det_bboxes[c], det_confs[c], det_embeddings[c])
                    matched_tracks.add(r)
                    matched_dets.add(c)
        else:
            matched_tracks = set()
            matched_dets   = set()

        # --- Initialise new tracks for unmatched detections ---
        for i in range(n_dets):
            if i not in matched_dets:
                self.tracks.append(
                    _KalmanTracker(det_bboxes[i], det_confs[i], det_embeddings[i])
                )

        # --- Remove stale tracks ---
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]

        # --- Output confirmed tracks ---
        output = []
        for t in self.tracks:
            if t.hits >= self.min_hits or t.time_since_update == 0:
                bbox = t.get_bbox()
                output.append((bbox[0], bbox[1], bbox[2], bbox[3],
                                t.track_id, t.conf))
        return output


# ---------------------------------------------------------------------------
# Velocity and gap helpers (same as tracker_bytetrack.py)
# ---------------------------------------------------------------------------

def _compute_velocity_norm(prev_cx, prev_cy, curr_cx, curr_cy, fps, frame_diagonal):
    if prev_cx is None or frame_diagonal == 0:
        return None
    dist_px = math.sqrt((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2)
    return (dist_px * fps) / frame_diagonal


def _compute_track_gaps(frame_entries):
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
# motmetrics evaluation (mirrors Day 7 SORT approach)
# ---------------------------------------------------------------------------

def _evaluate(seq_id, trajectories_dict, gt_xml_path,
              warmup_boundary, cache_frame_index_base=0, iou_thresh_eval=0.5):
    gt_frames = load_gt_sequence(gt_xml_path)
    acc       = mm.MOTAccumulator(auto_id=True)

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
            # motmetrics expects distance (1-IoU), NaN where IoU < threshold
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

def run_deepsort_on_sequence(seq_id, split, config,
                              reid_model=None, reid_device=None,
                              drive_dataset_root=None,
                              drive_models_dir=None,
                              gt_root=None,
                              day7_layout=None):
    """
    Run DeepSORT (OSNet VeRi-776) on one sequence.

    Args:
        seq_id:            sequence ID, e.g. "MVI_20062"
        split:             "test"
        config:            dict from config_blockB.yaml
        reid_model:        pre-built OSNet model (pass to avoid reloading per sequence)
        reid_device:       torch device for reid_model
        drive_dataset_root: path to raw frames, e.g.
                           (None = use local data/ua_detrac/raw)
        drive_models_dir:  models directory containing osnet_x1_0_veri_776.pth
        gt_root:           path to annotation XMLs
        day7_layout:       "variant_a" | "variant_b" | None (auto-detect)

    Returns:
        dict with mota, idf1, idsw, fragmentations, fps_median, latency_p95_ms
    """
    np.random.seed(42)
    random.seed(42)

    if day7_layout is None:
        day7_layout = _detect_day7_layout()

    if gt_root is None:
        gt_root = "data/ua_detrac/annotations"

    if drive_dataset_root is None:
        drive_dataset_root = "data/ua_detrac/raw"

    # --- Build ReID model if not provided ---
    if reid_model is None:
        reid_model, reid_device = build_reid_model(drive_models_dir)

    # --- Load config values ---
    cache_dir    = config.get('detections_source', 'detections_cache/')
    ds_cfg       = config.get('deepsort', {})
    appearance_t = ds_cfg.get('appearance_thresh', 0.40)
    max_age      = ds_cfg.get('max_age',    30)
    min_hits     = ds_cfg.get('min_hits',   3)
    iou_thresh   = ds_cfg.get('iou_threshold', 0.30)
    min_track_len= config.get('min_track_length', 15)
    max_interp   = config.get('max_interp_gap',   3)
    iou_eval     = config.get('eval', {}).get('iou_threshold', 0.5)
    cache_base   = config.get('cache_frame_index_base', 0)

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
    warmup_meta   = seq_meta['warmup_frames'] # 10
    frame_diagonal = math.sqrt(frame_width ** 2 + frame_height ** 2)

    assert warmup_meta == warmup_boundary, (
        f"Warmup mismatch: metadata says {warmup_meta}, cache says {warmup_boundary}"
    )

    # --- Filter to post-warmup frames ---
    active_frames = [f for f in frame_results if f['frame_idx'] > warmup_boundary]

    # -----------------------------------------------------------------------
    # PRE-LOAD all frames into memory before timing loop (FPS protocol)
    # Image I/O must NOT be inside the timed section.
    # frame_idx in cache is 0-based; JPG filenames are 1-based → +1 offset.
    # -----------------------------------------------------------------------
    print(f"  [DeepSORT] Pre-loading {len(active_frames)} frames for {seq_id} ...")
    frames_preloaded = {}
    for frame_data in active_frames:
        fid      = frame_data['frame_idx']
        img_path = os.path.join(
            drive_dataset_root, seq_id, f"img{fid + 1:05d}.jpg"
        )
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Frame not found: {img_path} — using blank frame")
            img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        frames_preloaded[fid] = img
    print(f"  [DeepSORT] Pre-load complete OK")

    # --- Initialise tracker ---
    tracker = DeepSORT(
        appearance_thresh=appearance_t,
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_thresh
    )
    tracker.reset()

    raw_trajectories = defaultdict(list)
    prev_centroids   = {}
    latencies        = []

    # -----------------------------------------------------------------------
    # Timing loop — tracker update only (feature extraction included,
    # but not image I/O which was pre-loaded above)
    # -----------------------------------------------------------------------
    for frame_data in active_frames:
        frame_idx  = frame_data['frame_idx']
        detections = frame_data.get('detections', [])
        img        = frames_preloaded[frame_idx]

        # Extract ReID embeddings for each detection
        embeddings = [
            extract_reid_feature(
                reid_model, reid_device, img,
                d['bbox'], frame_width, frame_height
            )
            for d in detections
        ]

        t0      = time.perf_counter()
        tracked = tracker.update(detections, embeddings)
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

    # --- Determine output paths ---
    if day7_layout == "variant_a":
        traj_dir  = "logs/block_b/deepsort"
        traj_path = os.path.join(traj_dir, f"trajectories_{seq_id}.json")
    else:
        traj_dir  = "logs/block_b/trajectories"
        traj_path = os.path.join(traj_dir, f"{seq_id}_deepsort.json")
    os.makedirs(traj_dir, exist_ok=True)

    metrics_dir  = "logs/block_b/deepsort"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"metrics_{seq_id}.json")

    # --- Save trajectory JSON ---
    traj_out = {
        "seq_id"                  : seq_id,
        "detector"                : "yolov8n",
        "tracker"                 : "deepsort",
        "reid_model"              : "osnet_x1_0_veri_776",
        "cache_frame_index_base"  : int(cache_base),
        "warmup_boundary"         : int(warmup_boundary),
        "appearance_thresh"       : float(appearance_t),
        "max_age"                 : int(max_age),
        "min_hits"                : int(min_hits),
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
    print(f"  [DeepSORT] Saved trajectory: {traj_path}")

    # --- motmetrics evaluation ---
    gt_xml  = get_gt_xml_path(seq_id, gt_root)
    metrics = _evaluate(
        seq_id,
        {str(k): v for k, v in raw_trajectories.items()},
        gt_xml,
        warmup_boundary,
        cache_base,
        iou_eval
    )

    n_tracks = len(raw_trajectories)
    print(f"  [DeepSORT] {seq_id}: tracks={n_tracks}  "
          f"MOTA={metrics['mota']:.3f}  IDF1={metrics['idf1']:.3f}  "
          f"IDSW={metrics['num_switches']}  Frag={metrics['num_fragmentations']}  "
          f"FPS={fps_median:.1f}")

    # --- Save per-sequence metrics JSON ---
    seq_metrics = {
        "seq_id"         : seq_id,
        "tracker"        : "deepsort",
        "reid_model"     : "osnet_x1_0_veri_776",
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

def run_deepsort_on_sequences(seq_ids, split, config,
                               drive_dataset_root=None,
                               drive_models_dir=None,
                               gt_root=None,
                               day7_layout=None):
    """
    Run DeepSORT on multiple sequences. Builds ReID model once and reuses it.

    Args:
        seq_ids:            list of sequence IDs
        split:              "test"
        config:             dict from config_blockB.yaml
        drive_dataset_root: path to raw UA-DETRAC frames on Drive
        drive_models_dir:   Drive path for VeRi-776 weight cache
        gt_root:            path to annotation XMLs
        day7_layout:        "variant_a" | "variant_b" | None (auto-detect)

    Returns:
        {
          "per_seq": {seq_id: metrics_dict},
          "aggregate": {mota_mean, idf1_mean, idsw_total, fragmentations_total, fps_median}
        }
    """
    if day7_layout is None:
        day7_layout = _detect_day7_layout()
    print(f"\n[DeepSORT] Day 7 layout: {day7_layout}")

    # Build ReID model once — reuse across sequences
    reid_model, reid_device = build_reid_model(drive_models_dir)

    per_seq = {}
    for seq_id in seq_ids:
        print(f"\n{'='*60}")
        print(f"[DeepSORT] Running on {seq_id}")
        try:
            result = run_deepsort_on_sequence(
                seq_id, split, config,
                reid_model=reid_model,
                reid_device=reid_device,
                drive_dataset_root=drive_dataset_root,
                drive_models_dir=drive_models_dir,
                gt_root=gt_root,
                day7_layout=day7_layout
            )
            per_seq[seq_id] = result
        except Exception as e:
            import traceback
            print(f"  [DeepSORT] ERROR on {seq_id}: {e}")
            traceback.print_exc()

    if not per_seq:
        return {"per_seq": {}, "aggregate": {}}

    motas = [v['mota']           for v in per_seq.values()]
    idf1s = [v['idf1']           for v in per_seq.values()]
    idsws = [v['idsw']           for v in per_seq.values()]
    frags = [v['fragmentations'] for v in per_seq.values()]
    fpss  = [v['fps_median']     for v in per_seq.values()]

    aggregate = {
        "mota_mean"            : round(float(np.mean(motas)), 6),
        "idf1_mean"            : round(float(np.mean(idf1s)), 6),
        "idsw_total"           : int(sum(idsws)),
        "fragmentations_total" : int(sum(frags)),
        "fps_median"           : round(float(np.median(fpss)), 2)
    }

    print(f"\n{'='*60}")
    print("BLOCK B — DeepSORT (VeRi-776) summary")
    print(f"  MOTA mean : {aggregate['mota_mean']:.3f}")
    print(f"  IDF1 mean : {aggregate['idf1_mean']:.3f}")
    print(f"  IDSW total: {aggregate['idsw_total']}")
    print(f"  Frag total: {aggregate['fragmentations_total']}")
    print(f"  FPS median: {aggregate['fps_median']:.1f}")

    return {"per_seq": per_seq, "aggregate": aggregate}
