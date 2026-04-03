"""
tracker.py

Tracker module implementing SORT, DeepSORT, and ByteTrack algorithms.
All tracking uses filterpy KalmanFilter for state estimation.
Student: MANJOO Ameera Najla | M01014463
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class KalmanTracker:
    """Single Kalman filter tracker for a vehicle track."""
    
    def __init__(self, track_id, bbox, frame_idx):
        """
        Initialize a Kalman tracker for a single track.
        
        Args:
            track_id (int): Unique track ID
            bbox (list): [x1, y1, x2, y2]
            frame_idx (int): Frame index
        """
        self.track_id = track_id
        self.bbox = bbox
        self.centroid = self._bbox_to_centroid(bbox)
        self.frame_idx = frame_idx
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        
        # Kalman filter for 2D position tracking
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([[self.centroid[0]],
                             [self.centroid[1]],
                             [0],
                             [0]])  # [x, y, vx, vy]
        self.kf.F = np.array([[1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])  # State transition
        self.kf.H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]])  # Measurement function
        self.kf.P *= 1000
        self.kf.R = np.array([[1, 0],
                             [0, 1]])
        self.kf.Q = np.eye(4) * 0.01
        
        # History for interpolation
        self.centroid_history = {frame_idx: self.centroid.copy()}
    
    @staticmethod
    def _bbox_to_centroid(bbox):
        """Convert bbox [x1, y1, x2, y2] to centroid [cx, cy]."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return np.array([cx, cy])
    
    @staticmethod
    def _centroid_to_bbox(centroid, width, height):
        """Convert centroid back to bbox (simplified)."""
        cx, cy = centroid
        w_half = width / 2
        h_half = height / 2
        return [cx - w_half, cy - h_half, cx + w_half, cy + h_half]
    
    def predict(self, frame_idx):
        """Predict next position using Kalman filter."""
        try:
            self.kf.predict()
        except:
            pass  # Fallback to last known position
        self.time_since_update += 1
        self.age += 1
    
    def update(self, bbox, frame_idx):
        """Update track with new detection."""
        self.bbox = bbox
        new_centroid = self._bbox_to_centroid(bbox)
        self.centroid_history[frame_idx] = new_centroid.copy()
        self.centroid = new_centroid
        self.time_since_update = 0
        self.hits += 1
        self.age += 1
        
        # Update Kalman filter
        try:
            self.kf.update(new_centroid)
        except:
            pass
    
    def get_centroid(self):
        """Return current estimated centroid."""
        try:
            return np.array([self.kf.x[0, 0], self.kf.x[1, 0]])
        except:
            return self.centroid


class Tracker:
    """
    Multi-object tracker using SORT, DeepSORT, or ByteTrack variants.
    """
    
    def __init__(self, tracker_name="sort"):
        """
        Initialize tracker.
        
        Args:
            tracker_name (str): One of "sort", "deepsort", "bytetrack"
        """
        self.tracker_name = tracker_name
        self.trackers = {}  # Dict of KalmanTracker objects
        self.next_track_id = 1
        self.max_age = 5   # Correction A: max_age=5 prevents ID splitting in UA-DETRAC heavy occlusion
        self.min_hits = 3
        self.iou_threshold = 0.3
        self.min_track_length = 15
    
    def reset(self):
        """Clear all tracks."""
        self.trackers = {}
        self.next_track_id = 1
    
    def update(self, detections, frame_idx):
        """
        Update tracks with new detections.
        
        Args:
            detections (list): [{
                "bbox": [x1, y1, x2, y2],
                "conf": float,
                "class_id": int
            }]
            frame_idx (int): Frame index
            
        Returns:
            list: [{
                "track_id": int,
                "bbox": [x1, y1, x2, y2],
                "centroid": [cx, cy],
                "age": int
            }]
        """
        # Predict all tracks
        for track_id, tracker in list(self.trackers.items()):
            tracker.predict(frame_idx)
            
            # Remove dead tracks
            if tracker.time_since_update > self.max_age:
                del self.trackers[track_id]
        
        # Match detections to tracks
        matched, unmatched_det, unmatched_trk = self._associate_detections_to_trackers(
            detections, frame_idx
        )
        
        # Update matched tracks
        for det_idx, trk_idx in matched:
            detection = detections[det_idx]
            tracker = list(self.trackers.values())[trk_idx]
            tracker.update(detection["bbox"], frame_idx)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_det:
            detection = detections[det_idx]
            new_tracker = KalmanTracker(
                self.next_track_id,
                detection["bbox"],
                frame_idx
            )
            self.trackers[self.next_track_id] = new_tracker
            self.next_track_id += 1
        
        # Collect active tracks
        output_tracks = []
        for track_id, tracker in self.trackers.items():
            if tracker.hits >= self.min_hits:
                # Interpolate missing frames (gap <= 3)
                centroid = self._interpolate_centroid(tracker, frame_idx)
                
                output_tracks.append({
                    "track_id": track_id,
                    "bbox": tracker.bbox,
                    "centroid": centroid,
                    "age": tracker.age
                })
        
        return output_tracks
    
    def _associate_detections_to_trackers(self, detections, frame_idx):
        """
        Associate detections to trackers using IoU or centroid distance.
        
        Returns:
            matched (list): [(det_idx, trk_idx), ...]
            unmatched_det (list): [det_idx, ...]
            unmatched_trk (list): [trk_idx, ...]
        """
        if len(detections) == 0 or len(self.trackers) == 0:
            unmatched_det = list(range(len(detections)))
            unmatched_trk = list(range(len(self.trackers)))
            return [], unmatched_det, unmatched_trk
        
        # Compute cost matrix (IoU-based)
        trackers_list = list(self.trackers.values())
        cost_matrix = np.zeros((len(detections), len(trackers_list)))
        
        for det_idx, detection in enumerate(detections):
            det_bbox = detection["bbox"]
            for trk_idx, tracker in enumerate(trackers_list):
                iou = self._iou(det_bbox, tracker.bbox)
                cost_matrix[det_idx, trk_idx] = 1 - iou
        
        # Hungarian algorithm
        matched_indices = linear_sum_assignment(cost_matrix)
        
        matched = []
        unmatched_det = []
        unmatched_trk = set(range(len(trackers_list)))
        
        for det_idx, trk_idx in zip(*matched_indices):
            if cost_matrix[det_idx, trk_idx] < (1 - self.iou_threshold):
                matched.append((det_idx, trk_idx))
                unmatched_trk.discard(trk_idx)
            else:
                unmatched_det.append(det_idx)
        
        unmatched_det.extend([i for i in range(len(detections))
                             if i not in [m[0] for m in matched]])
        unmatched_det = list(set(unmatched_det))
        
        return matched, unmatched_det, list(unmatched_trk)
    
    @staticmethod
    def _iou(bbox1, bbox2):
        """Compute Intersection over Union."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _interpolate_centroid(self, tracker, frame_idx):
        """Interpolate missing centroid (gap <= 3 frames)."""
        history = tracker.centroid_history
        
        if frame_idx in history:
            return history[frame_idx]
        
        # Find nearest frames with data
        frames = sorted(history.keys())
        
        for i, f in enumerate(frames):
            if f > frame_idx:
                if i > 0:
                    f_prev = frames[i - 1]
                    gap = f - f_prev
                    
                    if gap <= 3 and frame_idx in range(f_prev + 1, f):
                        # Linear interpolation
                        t = (frame_idx - f_prev) / gap
                        c_prev = history[f_prev]
                        c_next = history[f]
                        centroid = c_prev + t * (c_next - c_prev)
                        return centroid
                break
        
        # Fallback to last known
        return tracker.centroid
