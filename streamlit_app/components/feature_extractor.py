"""
feature_extractor.py

Feature extraction module for vehicle density, speed, and interaction metrics.
Student: MANJOO Ameera Najla | M01014463
"""

import numpy as np
from collections import defaultdict


class FeatureExtractor:
    """
    Extract behavioral features from tracked vehicles.
    All temporal windows computed from seconds, never hardcoded frames.
    """
    
    def __init__(self, fps, width, height, selected_features=None):
        """
        Initialize feature extractor.
        
        Args:
            fps (float): Frame rate from video (never hardcoded)
            width (int): Frame width in pixels
            height (int): Frame height in pixels
            selected_features (list): ["F1: Density/Flow", "F2: Speed", "F3: Distance"]
        """
        self.fps = fps
        self.width = width
        self.height = height
        self.selected_features = selected_features or ["F1: Density/Flow", "F2: Speed", "F3: Distance"]
        
        # Compute temporal windows in frames from seconds
        self.speed_window_sec = 0.2
        self.speed_window_frames = int(self.speed_window_sec * self.fps)
        self.speed_window_frames = max(1, self.speed_window_frames)
        
        # Warmup frames
        self.warmup_frames = max(10, int(0.4 * self.fps))
        
        # Track history
        self.track_history = defaultdict(lambda: {
            "centroids": [],
            "frame_indices": [],
            "speeds": [],
            "dwell_time_sec": 0.0,
            "first_frame": None,
            "last_frame": None
        })
        
        # Frame diagonal for normalization
        self.frame_diagonal = np.sqrt(width ** 2 + height ** 2)
    
    def extract(self, tracks, frame_idx):
        """
        Extract features for the current frame.
        
        Args:
            tracks (list): [{
                "track_id": int,
                "centroid": [cx, cy],
                "age": int
            }]
            frame_idx (int): Current frame index
            
        Returns:
            dict: {
                "frame_idx": int,
                "vehicle_count": int,
                "roi_occupancy": float,
                "mean_speed_px_sec": float,
                "min_distance_norm": float,
                "mean_dwell_sec": float
            }
        """
        # Update track history
        for track in tracks:
            track_id = track["track_id"]
            centroid = track["centroid"]
            
            history = self.track_history[track_id]
            history["centroids"].append(centroid)
            history["frame_indices"].append(frame_idx)
            history["first_frame"] = history["first_frame"] or frame_idx
            history["last_frame"] = frame_idx
        
        # Compute features
        vehicle_count = len(tracks)
        roi_occupancy = self._compute_roi_occupancy(tracks)
        mean_speed = self._compute_mean_speed(frame_idx)
        min_distance = self._compute_min_distance(tracks)
        mean_dwell = self._compute_mean_dwell(frame_idx)
        
        return {
            "frame_idx": frame_idx,
            "vehicle_count": vehicle_count,
            "roi_occupancy": roi_occupancy,
            "mean_speed_px_sec": mean_speed,
            "min_distance_norm": min_distance,
            "mean_dwell_sec": mean_dwell
        }
    
    def _compute_roi_occupancy(self, tracks):
        """Compute fraction of frame covered by bounding boxes."""
        if not tracks:
            return 0.0
        
        frame_area = self.width * self.height
        bbox_area = 0
        
        for track in tracks:
            centroid = track["centroid"]
            # Estimate bbox area (simplified: 80x60 pixels average vehicle)
            bbox_area += 80 * 60
        
        occupancy = min(1.0, bbox_area / frame_area)
        return occupancy
    
    def _compute_mean_speed(self, frame_idx):
        """
        Compute mean speed across active tracks in px/sec.
        Fix F07: Always pixels per second, never pixels per frame.
        Bug fix: history["speeds"] now records the individual track's own speed,
        not a running cross-track mean (which was corrupting per-track speed labels).
        """
        speeds = []

        for track_id, history in self.track_history.items():
            centroids = np.array(history["centroids"])
            frames    = np.array(history["frame_indices"])

            if len(centroids) < 2:
                continue

            # Skip tracks that haven't been seen recently
            if frames[-1] < frame_idx - self.speed_window_frames:
                continue

            delta_pixels = np.linalg.norm(centroids[-1] - centroids[-2])
            delta_frames = int(frames[-1] - frames[-2])

            if delta_frames > 0:
                # pixels per second for THIS track
                speed_px_sec = delta_pixels * self.fps
                speeds.append(speed_px_sec)
                # Record per-track speed so the Pipeline page can display individual labels
                history["speeds"].append(speed_px_sec)

        return float(np.mean(speeds)) if speeds else 0.0
    
    def _compute_min_distance(self, tracks):
        """
        Compute minimum inter-vehicle distance, normalized by frame diagonal.
        Fix F28: Always use frame diagonal normalization.
        Disclaimer: Image-space proxy only. Not physical distance.
        """
        if len(tracks) < 2:
            return 1.0
        
        centroids = [track["centroid"] for track in tracks]
        min_dist = float('inf')
        
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.linalg.norm(
                    np.array(centroids[i]) - np.array(centroids[j])
                )
                if dist < min_dist:
                    min_dist = dist
        
        if min_dist == float('inf'):
            return 1.0
        
        # Normalize by frame diagonal
        dist_norm = min_dist / self.frame_diagonal
        return min(1.0, max(0.0, dist_norm))
    
    def _compute_mean_dwell(self, frame_idx):
        """Compute mean dwell time (time in scene) in seconds."""
        dwell_times = []
        
        for track_id, history in self.track_history.items():
            if history["first_frame"] is not None and history["last_frame"] is not None:
                dwell_sec = (history["last_frame"] - history["first_frame"]) / self.fps
                dwell_times.append(dwell_sec)
        
        return np.mean(dwell_times) if dwell_times else 0.0
    
    def reset(self):
        """Reset feature extractor for next video."""
        self.track_history.clear()
