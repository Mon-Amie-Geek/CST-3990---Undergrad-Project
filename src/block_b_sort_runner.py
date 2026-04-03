"""
Block B — SORT tracker Day 7 runner.

Fixes:
  Proper compute speed : velocity px/sec * fps, normalised by frame_diagonal
  Official split  : sequences from frozen split_manifest.json (split_utils)
  Not to re-run detector  : reads detections_cache/ only
  FNo fragment filtering  : fragment filter NOT applied — raw Day 7, Day 9 post-processes
  Frame-index synchronisation  : 0/1-based offset via cache_frame_index_base in config
  Warmup frames not discarded  : warmup_boundary respected
  Extraction method - Reproducibility   : fps from metadata.json
  Control random seed  : seeds (torch + numpy + random)

Corrections:
  A : max_age=5 via config
  B : conf_thresh from config_blockA.yaml, never guessed
  C : velocity_norm embedded in trajectories
  3 : Extended CSV output contract
  5 : Tracker FPS logged

Schema accuracy:
  conf is NEVER null in Day 7 SORT output (time_since_update < 1 guarantee)
  nms_thresh: stored for provenance only, NOT active (cached dets are post-NMS)
"""

import os, json, yaml, time, subprocess, math
import numpy as np
import motmetrics as mm
from collections import defaultdict
import torch, random
import pandas as pd

from cache_utils  import load_cache
from sort_tracker import SORTTracker
from gt_loader    import load_gt_sequence, get_gt_xml_path
from split_utils  import load_split_sequences, verify_split_integrity


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Gives access to detector, split, sort settings, thresholds, frame index base

with open('configs/config_blockB.yaml') as f:
    cfg_b = yaml.safe_load(f)

with open('configs/config_blockA.yaml') as f:
    cfg_a = yaml.safe_load(f)

# inherit from Block A (source of truth)
CONF_THRESH = cfg_a['conf_thresh']
NMS_THRESH  = cfg_a['nms_thresh']

# ensure Block B config matches Block A provenance
assert cfg_b['conf_thresh'] == cfg_a['conf_thresh'], \
    f"Mismatch: config_blockB conf_thresh={cfg_b['conf_thresh']} vs config_blockA conf_thresh={cfg_a['conf_thresh']}"

assert cfg_b['nms_thresh'] == cfg_a['nms_thresh'], \
    f"Mismatch: config_blockB nms_thresh={cfg_b['nms_thresh']} vs config_blockA nms_thresh={cfg_a['nms_thresh']}"

DETECTOR_NAME    = cfg_b['detector']
CACHE_DIR        = cfg_b['detections_source']
# Access for max_age , min_hits, iou_threshold
SORT_CFG         = cfg_b['sort']
# Min track length for filtering
MIN_TRACK_LEN    = cfg_b['min_track_length']
IOU_THRESH_EVAL  = cfg_b['eval']['iou_threshold']
SPLIT            = cfg_b.get('split', 'test')
# Crucial - GT XML is 1-based and cache confirmed at 0-level
CACHE_FRAME_BASE = cfg_b.get('cache_frame_index_base', 0)

# Contain: results CSV, timing CSV. trajectories jso
LOG_DIR  = 'logs/block_b'
TRAJ_DIR = os.path.join(LOG_DIR, 'trajectories')
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(TRAJ_DIR, exist_ok=True)

# Confirmation of details
print(f"[CONFIG] detector         : {DETECTOR_NAME}  (frozen Day 6)")
print(f"[CONFIG] conf_thresh      : {CONF_THRESH}  (inherited from config_blockA)")
print(f"[CONFIG] nms_thresh       : {NMS_THRESH}  "
      f"(PROVENANCE ONLY — cached detections already post-NMS)")
print(f"[CONFIG] max_age          : {SORT_CFG['max_age']}  (Correction A)")
print(f"[CONFIG] cache_base       : {CACHE_FRAME_BASE}  (Fix F12)")
print(f"[CONFIG] split source     : split_manifest.json  (Fix F08 resolved)")
print(f"[CONFIG] fragment filter  : NOT applied Day 7 — Day 9 post-processes")


# Defines helper function to capture current git commit ID

def get_commit_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'

# Loads metadata: FPS, frame width, frame height
def load_metadata(seq_id):
    # Builds per sequence metadata file path
    path = os.path.join('data', 'ua_detrac', 'metadata', f'{seq_id}.json')
    if not os.path.exists(path):
        # Prevent silent failures in velocity calculations
        raise FileNotFoundError(
            f"metadata.json missing for {seq_id}. "
            f"Run metadata_repair.py before Day 7.")
    with open(path) as f:
        meta = json.load(f)
    for key in ['fps', 'frame_width', 'frame_height']:
        if key not in meta:
            raise KeyError(f"metadata.json for {seq_id} missing key: {key}")
    return meta

# Defines function to compute speed-like feature
# Inputs: previous centroid, current centroid, fps, frame diagonal
# Outputs: Normalised velocity
def compute_velocity_norm(prev_cx, prev_cy, curr_cx, curr_cy,
                           fps, frame_diagonal):
    """
    Note : velocity in px/sec
    Correction C: normalised by frame_diagonal for Block C compatibility.
    Returns None for first track appearance (no previous centroid).
    """
    if prev_cx is None or frame_diagonal == 0:
        return None
    # Computes Euclidean pixel movement between previous & current centroid
    dist_px    = math.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
    # Converts frame per frame into movement per second
    vel_px_sec = dist_px * fps
    # Normalise by frame diagonal
    return vel_px_sec / frame_diagonal

# Converts GT XML frame number into cache index
def gt_frame_to_cache_idx(gt_frame_num, cache_base):
    
    return gt_frame_num - 1 if cache_base == 0 else gt_frame_num


# Core: run SORT on one sequence

def run_sort_on_sequence(seq_id):
    # Loads frozen detections from cache
    # Returns warmup_boundary, frame_results (contains detections frame by frame)
    warmup_boundary, frame_results = load_cache(
        seq_id, SPLIT, DETECTOR_NAME, CACHE_DIR)

    # Loads meta - needed for fps, frame size
    meta           = load_metadata(seq_id)
    fps            = meta['fps']
    # Computes frame diagonal for normalised speed
    frame_diagonal = math.sqrt(meta['frame_width']**2 + meta['frame_height']**2)

    tracker = SORTTracker(
        max_age       = SORT_CFG['max_age'],
        min_hits      = SORT_CFG['min_hits'],
        iou_threshold = SORT_CFG['iou_threshold'],
        conf_thresh   = CONF_THRESH              # From Block A
    )
    # Important when moving between sequences
    tracker.reset()

    # Stores trajectory entries for each track ID
    raw_trajectories = defaultdict(list)
    # Stores previous centroid of each track
    prev_centroids   = {}
    # Stores processing time per frame
    frame_times      = []

    # Loops through all cached frames
    for frame_data in frame_results:
        frame_idx = frame_data['frame_idx']

        # Starts list of detections for current frame
        dets = []
        # Loops through cached detections in frame
        for det in frame_data['detections']:
            # Reads bounding box
            x1, y1, x2, y2 = det['bbox']
            # Builds SORT input row - box coordinates, confidance
            # If confidence missing -> default to 1.0
            dets.append([x1, y1, x2, y2, det.get('conf', 1.0)])
        # Converts detections into numpy array
        dets_arr = (np.array(dets, dtype=float) if dets
                    else np.empty((0, 5)))

        # Starts timing
        t0      = time.perf_counter()
        # Runs tracker for this frame & returns tracked boxes
        tracked = tracker.update(dets_arr, frame_idx)
        # Starts processing time in ms
        frame_times.append((time.perf_counter() - t0) * 1000)

        # Loops through tracks output by SORT
        for row in tracked:
            x1, y1, x2, y2, tid, conf = row
            # Ensures track ID is integer
            tid = int(tid)
            # Computes centroid: used for speed and later features
            cx  = (x1 + x2) / 2
            cy  = (y1 + y2) / 2

            # Gets previous centroid track if available
            prev    = prev_centroids.get(tid)
            # Computes normalised velocity if exists
            vel_n   = compute_velocity_norm(
                prev[0] if prev else None,
                prev[1] if prev else None,
                cx, cy, fps, frame_diagonal)
            # Stores current centroid for next frame
            prev_centroids[tid] = (cx, cy)

            # conf is always float here (never null) — see schema note
            # Adds one trajectory entry for this track
            raw_trajectories[tid].append({
                'frame_idx'     : int(frame_idx),
                # Stores tracked box
                'bbox'          : [float(x1), float(y1), float(x2), float(y2)],
                'cx'            : float(cx),
                'cy'            : float(cy),
                # Stores normalised speed
                'velocity_norm' : float(vel_n) if vel_n is not None else None,
                'conf'          : float(conf)
            })

    mean_ms = float(np.mean(frame_times)) if frame_times else 0.0
    timing  = {
        'total_sec'         : round(sum(frame_times) / 1000, 4),
        'mean_ms_per_frame' : round(mean_ms, 4),
        'tracker_fps'       : round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
        'n_frames_processed': len(frame_times)
    }

    # Converts track IDs to strings for JSON compatibility
    return (
        {str(k): v for k, v in raw_trajectories.items()},
        warmup_boundary,
        {'fps': fps, 'frame_width': meta['frame_width'],
         'frame_height': meta['frame_height'],
         'frame_diagonal': float(frame_diagonal)},
        timing
    )



# Writes trajectory artifact to JSON

def save_trajectories(seq_id, raw_trajectories, warmup_boundary, meta):
    """
    Saves RAW trajectories. No fragment filter applied.
    Schema matches trajectories_schema.json exactly.
    conf is always float (never null) in Day 7 SORT output.
    """
    # Stores sequence, tracker, detector
    out = {
        'seq_id'          : seq_id,
        'tracker'         : 'sort',
        'detector'        : DETECTOR_NAME,
        'warmup_boundary' : int(warmup_boundary),
        'fps'             : meta['fps'],
        'frame_width'     : meta['frame_width'],
        'frame_height'    : meta['frame_height'],
        'frame_diagonal'  : meta['frame_diagonal'],
        # Stores Day 9 filter parameter for reference
        'min_track_length': MIN_TRACK_LEN,
        'tracks'          : raw_trajectories
    }
    path = os.path.join(TRAJ_DIR, f'{seq_id}_sort.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    return path


# motmetrics evaluation - Computes Block B tracking metrics

def evaluate_with_motmetrics(seq_id, raw_trajectories,
                              gt_xml_path, warmup_boundary):
    
    # Loads GT XML into frame-by-frame dictionary
    gt_frames = load_gt_sequence(gt_xml_path)
    # Creates motmetrics accumulator accumulator - 
    acc       = mm.MOTAccumulator(auto_id=True)

    # Organises predictions by frame index
    pred_by_frame = defaultdict(dict)
    # Loops through tracks
    for tid, entries in raw_trajectories.items():
        for entry in entries:
            fidx = entry['frame_idx']
            if fidx > warmup_boundary:
                pred_by_frame[fidx][tid] = entry['bbox']
                
    # Loops through GT frames in order
    for gt_frame_num in sorted(gt_frames.keys()):
        # Converts FT-1 based frame number to cache frame index
        fidx = gt_frame_to_cache_idx(gt_frame_num, CACHE_FRAME_BASE)
        if fidx <= warmup_boundary:
            continue

        # Gets GT objects & predictions for this frame
        gt_dict    = gt_frames[gt_frame_num]
        pred_dict  = pred_by_frame.get(fidx, {})
        # Gets GT & predicted object IDs
        gt_ids     = list(gt_dict.keys())
        pred_ids   = list(pred_dict.keys())

        # Computes IoU cost matrix
        dist_matrix = (
            mm.distances.iou_matrix(
                # Supplies GT boxes & predicted boxes
                [gt_dict[i] for i in gt_ids],
                [pred_dict[i] for i in pred_ids],
                # Uses evaluation threshold
                max_iou=1 - IOU_THRESH_EVAL)
            if gt_ids and pred_ids
            # If one side empty, create nan matrix instead of crashing
            else np.full((len(gt_ids), len(pred_ids)), np.nan)
        )
        # Feeds this frame into motmetrics
        acc.update(gt_ids, pred_ids, dist_matrix)

    # Creates metrics handler
    mh      = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['mota','idf1','num_switches','num_fragmentations',
                 'num_objects','num_predictions','precision','recall'],
        name=seq_id
    )
    return {k: (float(summary[k].iloc[0]) if k not in ('num_switches','num_fragmentations','num_objects','num_predictions')
                else int(summary[k].iloc[0]))
            for k in ['mota','idf1','num_switches','num_fragmentations',
                      'num_objects','num_predictions','precision','recall']}


# Main runner: Runs tracker/ evaluation across all sequences in split

def run_all_sequences(seq_ids, gt_root):
    # Captures git version of code
    commit_hash = get_commit_hash()
    # List to collect 
    all_results = []
    timing_rows = []

    for seq_id in seq_ids:
        # Readable separator in console output
        print(f"\n{'='*60}")
        print(f"[SORT] {seq_id}")
        try:
            # Runs tracking on one sequence
            raw_traj, warmup_b, meta, timing = run_sort_on_sequence(seq_id)
            # Shows how many tracks were produced
            print(f"  Raw tracks  : {len(raw_traj)}")
            print(f"  Tracker FPS : {timing['tracker_fps']:.1f} "
                  f"({timing['mean_ms_per_frame']:.2f} ms/frame)")

            # Solves trajectory artifact
            traj_path = save_trajectories(seq_id, raw_traj, warmup_b, meta)
            print(f"  Saved       : {traj_path}")

            # Resolves GT XML path
            gt_xml  = get_gt_xml_path(seq_id, gt_root)
            # Evaluates tracked results
            metrics = evaluate_with_motmetrics(
                seq_id, raw_traj, gt_xml, warmup_b)

            # Creates one result row for CSV
            
            row = {
                'seq_id'            : seq_id,
                'tracker'           : 'sort',
                'mota'              : metrics['mota'],
                'idf1'              : metrics['idf1'],
                'num_switches'      : metrics['num_switches'],
                'num_fragmentations': metrics['num_fragmentations'],
                'num_objects'       : metrics['num_objects'],
                'num_predictions'   : metrics['num_predictions'],
                'precision'         : metrics['precision'],
                'recall'            : metrics['recall'],
                'n_raw_tracks'      : len(raw_traj),
                'detector_name'     : DETECTOR_NAME,
                'conf_thresh'       : CONF_THRESH,
                'nms_thresh'        : NMS_THRESH,   # provenance only
                'split'             : SPLIT,
                'eval_iou'          : IOU_THRESH_EVAL,
                'sort_max_age'      : SORT_CFG['max_age'],
                'sort_min_hits'     : SORT_CFG['min_hits'],
                'tracker_fps'       : timing['tracker_fps'],
                'mean_ms_per_frame' : timing['mean_ms_per_frame'],
                'fragment_filter'   : 'NOT_APPLIED_DAY7',
                'commit_hash'       : commit_hash
            }
            all_results.append(row)
            timing_rows.append({'seq_id': seq_id, **timing})
            # Prints per-sequence summary
            print(f"  MOTA={metrics['mota']:.3f}  IDF1={metrics['idf1']:.3f}  "
                  f"IDSW={metrics['num_switches']}  Frag={metrics['num_fragmentations']}")
            if metrics['mota'] < 0:
                print(f"  [NOTE] Negative MOTA expected for SORT on dense sequences.")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            
    # If all sequences failed, stop.
    if not all_results:
        print("No results produced.")
        return []
    
    # Saves main results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(LOG_DIR, 'block_b_sort_results.csv'), index=False)
    pd.DataFrame(timing_rows).to_csv(
        os.path.join(LOG_DIR, 'sort_timing.csv'), index=False)
    with open(os.path.join(LOG_DIR, 'idsw_per_sequence.json'), 'w') as f:
        json.dump({r['seq_id']: r['num_switches'] for r in all_results}, f, indent=2)

    # Summary for Chapter 4
    print(f"\n{'='*60}")
    print("BLOCK B — SORT (Day 7, RAW, no fragment filter)")
    print(f"Detector    : {DETECTOR_NAME} | conf_thresh : {CONF_THRESH}")
    print(f"nms_thresh  : {NMS_THRESH} (provenance — NOT active in Day 7)")
    print(f"Split       : {SPLIT} (from split_manifest.json — Fix F08 resolved)")
    print(f"SORT config : max_age={SORT_CFG['max_age']} | "
          f"min_hits={SORT_CFG['min_hits']}")
    print(f"Commit      : {commit_hash}")
    print("-"*60)
    disp = df[['seq_id','mota','idf1','num_switches',
               'num_fragmentations','tracker_fps']].copy()
    disp.columns = ['Sequence','MOTA','IDF1','IDSW','Frag','FPS']
    print(disp.round(3).to_string(index=False))
    print("-"*60)
    print(f"Mean MOTA : {df['mota'].mean():.3f}")
    print(f"Mean IDF1 : {df['idf1'].mean():.3f}")
    print(f"Total IDSW: {df['num_switches'].sum()}")
    print(f"Mean FPS  : {df['tracker_fps'].mean():.1f}")
    print("\nFragment filter (Day 9) | DeepSORT+ByteTrack (Day 8)")
    return all_results


# Entry point 
if __name__ == '__main__':
    GT_ROOT = 'dataset/ua_detrac/annotations'

    verify_split_integrity(cfg_b['split_manifest'])
    seq_ids = load_split_sequences(SPLIT, cfg_b['split_manifest'])
    print(f"\nDay 7 — Block B SORT | Sequences: {len(seq_ids)} ({SPLIT} split)")

    # Smoke tests first
    from smoke_tests_day7 import run_all_smoke_tests
    run_all_smoke_tests(cfg_b, CONF_THRESH)

    run_all_sequences(seq_ids, GT_ROOT)