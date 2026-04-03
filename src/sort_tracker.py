"""
SORT-style tracker — Day 7 implementation.
Adapted from the SORT family of Kalman + IoU trackers for this project’s
[x1, y1, x2, y2] box-state representation.

Fixes applied:
  Fix Random Seed not fully controlled : seeds set (numpy + random only — torch not imported here as
            sort_tracker.py has no torch dependency. Torch seed is set
            in block_b_sort_runner.py which does import torch.)
  Fix Frame-index synchronisation : frame_idx contract is 0-based throughout — no conversion here

Corrections applied:
  Correction A : max_age default=5 (not 1) — prevents ID splitting in
                 UA-DETRAC heavy occlusion sequences
  Correction B : conf_thresh filtering applied before association.
                 Low-confidence detections neither update existing tracks
                 nor spawn new tracks. Value is inherited from
                 config_blockA.yaml via config_blockB.yaml, never hardcoded.

Note on torch import:
  sort_tracker.py does NOT import torch. Seeds for numpy/random are set here.
  Torch seed is set in block_b_sort_runner.py (the entry point that does
  use torch indirectly via torchreid on Day 8). Keeping this file
  torch-free avoids unnecessary dependency weight on CPU-only machines.

Dependencies:
  - filterpy
  - scipy
"""
# For arrays, matrix opperations, IoU computation, filtering &
import numpy as np
import random
from filterpy.kalman import KalmanFilter


# Only numpy and random seeds here; no torch dependency
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Defines a class representing one tracked object
class KalmanBoxTracker:
    """
    Single tracked object using a Kalman filter.

    State vector:
        box position & velocity terms
        [x1, y1, x2, y2, vx, vy, vw, vh]
    """
    # Makes sure new tracker gets a new numeric ID
    count = 0
    
    # Creates a neew tracker for one detection
    # bbox : Initial bounding box, conf: detection confidence
    def __init__(self, bbox, conf=1.0):
        """
        Parameters
        ----------
        bbox : array-like of shape (4,)
            Bounding box [x1, y1, x2, y2]
        conf : float
            Detection confidence stored for downstream trajectory output
        """
        # Creates the Kalman filter - internal state has 8 value, measurement has 4
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition — constant velocity model
        # Next position = current position + velocity
        # Velocity stays the same
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=float)

        # Measurement model — observe box coordinates only
        # Detector gives bounding boxes, not velocities - Filter must infer velocities internally
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=float)

        # Noise / covariance tuning
        self.kf.R[2:, 2:] *= 10.0    # size measurements more uncertain
        self.kf.P[4:, 4:] *= 1000.0  # high velocity uncertainty at init
        # Gives filter toom to adapt to real measurements
        self.kf.P *= 10.0
        # Process noise for motion-related parts
        # Encourages smoother motion predictions
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialises first 4 state vlaues using the detected bounding box
        self.kf.x[:4] = np.array(bbox, dtype=float).reshape((4, 1))
        
        # Counts how many frames since last track was matched to a detection - Used to del stale tracks
        self.time_since_update = 0
        # Allows persistent ID for each tracked object
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        # Total times track has been updates with detection
        self.hits = 0
        # How many consecutibe successful matches the track has had recently
        self.hit_streak = 0
        # How man prediction steps the tracker has existed for
        self.age = 0
        # Stores predicted states
        self.history = []
        # Stores latest confidence - Confidence can be returned with tracked box
        self.conf = conf

    # Called when detection is matched to this existing track
    def update(self, bbox, conf=1.0):
        """Update tracker state with a matched detection."""
        self.time_since_update = 0
        # Clears prediction history after a real observation
        self.history = []
        # Track gets one more successful update
        self.hits += 1
        self.hit_streak += 1
        # Helps determine whether a track is mature enough to trust
        self.conf = conf
        # Correction step : Preicted state is adjusted with new observation
        self.kf.update(np.array(bbox, dtype=float).reshape((4, 1)))

    # called once per frame to advance tracker forward
    def predict(self):
        """Advance state by one frame and return predicted box."""
        # Estimates where object should be before matching with detections
        self.kf.predict()
        self.age += 1

        # Prevents intermittent tracks from being treated as continuously reliable
        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        # Stores & returns predicted box coordinates - Matching needs current predictd boxes
        self.history.append(self.kf.x[:4].flatten())
        return self.history[-1]

    # Returns current estimated box
    def get_state(self):
        """Return current estimated bounding box [x1, y1, x2, y2]."""
        return self.kf.x[:4].flatten()

# Computes IoU between two sets of boxes
def iou_batch(bb_test, bb_gt):
    """
    Vectorized IoU between two sets of bboxes.

    Parameters
    ----------
    bb_test : np.ndarray, shape (N, 4)
    bb_gt   : np.ndarray, shape (M, 4)

    Returns
    -------
    np.ndarray, shape (N, M)
        IoU matrix
    """
    # Reshapes array - Efficient batch IoU computation
    bb_gt = np.expand_dims(bb_gt, 0)     # (1, M, 4)
    bb_test = np.expand_dims(bb_test, 1) # (N, 1, 4)

    # Computes overlap rectangle
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    # Computes intersection area : If boxes do not overlap, width/height are clamped at 0
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)

    # Area of detection boxes
    area_test = (
        (bb_test[..., 2] - bb_test[..., 0]) *
        (bb_test[..., 3] - bb_test[..., 1])
    )
    
    # Area of tracker boxes
    area_gt = (
        (bb_gt[..., 2] - bb_gt[..., 0]) *
        (bb_gt[..., 3] - bb_gt[..., 1])
    )
    
    # Computes IoU = intersection / union
    union = area_test + area_gt - inter
    return inter / np.maximum(union, 1e-6)


# Defines helper function for assignment solving
def linear_assignment(cost_matrix):
    """
    Solve linear assignment using Hungarian matching.

    Parameters
    ----------
    cost_matrix : np.ndarray

    Returns
    -------
    np.ndarray, shape (K, 2)
        Row/column match pairs
    """
    
    from scipy.optimize import linear_sum_assignment
    
    # Finds optimal one-to-one matching between detections and trackers
    r, c = linear_sum_assignment(cost_matrix)
    return np.column_stack((r, c))


# Matches detection boxes to predicted track boxes
# Predicts tracking, compares with detections assign best matches
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    IoU-based assignment using Hungarian matching.

    Parameters
    ----------
    detections : np.ndarray, shape (N, 4)
    trackers   : np.ndarray, shape (M, 4)
    iou_threshold : float

    Returns
    -------
    matches : np.ndarray, shape (K, 2)
    unmatched_dets : np.ndarray
    unmatched_trks : np.ndarray
    """
    # If no active trackers exist
    if len(trackers) == 0:
        # No match, all detections are unmatches, no unmatched trackers
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0,), dtype=int)
        )

    # Builds pairwise IoU matrix
    iou_matrix = iou_batch(detections, trackers)
    
    # Higher IoU => Lower cost = Preferred match
    matched_idx = linear_assignment(1 - iou_matrix)

    # Finds unmatched detection
    unmatched_dets = [
        d for d in range(len(detections))
        if d not in matched_idx[:, 0]
    ]
    
    # Finds unmatched trackers
    unmatched_trks = [
        t for t in range(len(trackers))
        if t not in matched_idx[:, 1]
    ]
    
    
    #  Prevents bas associations that would cause ID switches
    matches = []
    for m in matched_idx:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            
    # Converts match into proper NumPy array
    matches = (
        np.concatenate(matches, axis=0)
        if matches else np.empty((0, 2), dtype=int)
    )

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)

# Manages all object trackers across frames in a sequence
class SORTTracker:
    """
    SORT-style tracker with confidence filtering.

    Correction A:
        max_age=5 default prevents ID splitting on UA-DETRAC heavy
        occlusion sequences.

    Correction B:
        conf_thresh filters detections before association.
        Low-confidence detections neither update existing tracks nor
        spawn new tracks.
    """
    # Initialises tracker settings: Track survives 5 missed frames
    # rack needs 3 successful detections before being considered
    
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3, conf_thresh=0.25):
        # Stores configuration values
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.conf_thresh = conf_thresh
        self.trackers = []
        # Counts frames processed
        self.frame_count = 0
        # Resets ID counter when a new SORTTracker instance is created
        KalmanBoxTracker.count = 0

    # Essential when moving from one video sequence to another
    # Prevents IDs from leaking across sequences
    def reset(self):
        """Reset tracker state between sequences."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    # Processes one frame of detections & return current tracked outputs
    def update(self, dets_with_conf, frame_idx):
        """
        Update tracker with detections for one frame.

        Parameters
        ----------
        dets_with_conf : np.ndarray
            Shape (N, 5) -> [x1, y1, x2, y2, conf]
            or shape (N, 4) -> [x1, y1, x2, y2], conf assumed 1.0
            Empty arrays (0, 5) or (0, 4) are allowed.
        frame_idx : int
            0-based frame index.
            Included for interface consistency with the runner, but not
            directly used inside this tracker implementation.

        Returns
        -------
        np.ndarray, shape (M, 6)
            [x1, y1, x2, y2, track_id, conf]

        Correction B detail:
            Low-confidence detections are dropped before association.
            They do NOT update existing tracks and do NOT create new tracks.
        """
        # Avoids confusion & linter warnings
        _ = frame_idx  # explicit: currently unused, kept for interface consistency
        self.frame_count += 1

        # Correction B: confidence filter before association
        # Prevents weak detectipns from damaging tracking
        has_conf = (dets_with_conf.ndim == 2 and dets_with_conf.shape[1] == 5)
        if has_conf and len(dets_with_conf) > 0:
            mask = dets_with_conf[:, 4] >= self.conf_thresh
            dets_filtered = dets_with_conf[mask]
        else:
            dets_filtered = dets_with_conf

        # Matching operates only on box coordinates
        dets = (
            dets_filtered[:, :4].copy()
            if len(dets_filtered) > 0
            else np.empty((0, 4))
        )
        confs = (
            dets_filtered[:, 4]
            if (has_conf and len(dets_filtered) > 0)
            else np.ones(len(dets))
        )

        # Predict all trackers
        predicted = np.zeros((len(self.trackers), 4))
        to_del = []
        
        # Protects pipeline against numerical failure
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            predicted[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        # Popping from the end avoids index shift issues
        for t in reversed(to_del):
            self.trackers.pop(t)

        predicted = np.ma.compress_rows(np.ma.masked_invalid(predicted))

        # Associate detections to predicted trackers
        matched, unmatched_dets, _ = associate_detections_to_trackers(
            dets, predicted, self.iou_threshold
        )

        # Update matched trackers with detection
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]], confs[m[0]])

        # Spawn new trackers for unmatched detections
        # New objects entering the scene need new IDs
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i], conf=confs[i]))

        # Collect confirmed tracks
        ret = []
        i = len(self.trackers)
        
        # Loops through trackers in reverse & gets current box state
        for trk in reversed(self.trackers):
            state = trk.get_state()
            confirmed = (
                trk.time_since_update < 1 and
                (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)
            )
            
            # Decides whether track should be returned
            # Prevents unstable one-frame noise tracks from being emitted too early
            if confirmed:
                ret.append(
                    np.concatenate((state, [trk.id + 1, trk.conf])).reshape(1, 6)
                )
            # Removes stale tracks, Prevents dead objects from lingering forever
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return np.concatenate(ret, axis=0) if ret else np.empty((0, 6))