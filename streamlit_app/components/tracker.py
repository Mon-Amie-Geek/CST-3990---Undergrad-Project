"""
tracker.py

Streamlit tracker adapter that reuses the workspace Block B implementations
for SORT, DeepSORT, and ByteTrack.
Student: MANJOO Ameera Najla | M01014463
"""

import os
import sys

import numpy as np
import streamlit as st
import yaml


_COMPONENTS_DIR = os.path.dirname(__file__)
_STREAMLIT_APP_DIR = os.path.normpath(os.path.join(_COMPONENTS_DIR, ".."))
_REPO_ROOT = os.path.normpath(os.path.join(_STREAMLIT_APP_DIR, ".."))

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.sort_tracker import SORTTracker
from src.tracker_bytetrack import ByteTracker
from src.tracker_deepsort import DeepSORT, build_reid_model, extract_reid_feature


def _bbox_to_centroid(bbox):
    """Convert [x1, y1, x2, y2] to [cx, cy]."""
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)


def _bbox_to_640_space(bbox, width, height):
    """Convert original-frame bbox coordinates to the project's 640x640 space."""
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 0.0, 0.0]

    x1, y1, x2, y2 = bbox
    scale_x = 640.0 / float(width)
    scale_y = 640.0 / float(height)
    return [
        float(x1) * scale_x,
        float(y1) * scale_y,
        float(x2) * scale_x,
        float(y2) * scale_y,
    ]


@st.cache_resource
def _load_block_b_config(config_path):
    """Load the frozen Block B tracker config once."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def _load_reid_model(models_dir):
    """Build and cache the VeRi-776 ReID model used by DeepSORT."""
    return build_reid_model(models_dir)


class Tracker:
    """
    Adapter that exposes the Streamlit-friendly track dict format while using
    the real tracker implementations from the workspace.
    """

    def __init__(self, tracker_name="sort"):
        self.tracker_name = tracker_name
        self.config = _load_block_b_config(os.path.join(_REPO_ROOT, "configs", "config_blockB.yaml"))
        self.tracker = None
        self.reid_model = None
        self.reid_device = None

        self._build_tracker()

    def _build_tracker(self):
        """Construct the requested tracker using the frozen workspace config."""
        if self.tracker_name == "sort":
            cfg = self.config["sort"]
            self.tracker = SORTTracker(
                max_age=cfg["max_age"],
                min_hits=cfg["min_hits"],
                iou_threshold=cfg["iou_threshold"],
                conf_thresh=self.config["conf_thresh"],
            )
            return

        if self.tracker_name == "bytetrack":
            cfg = self.config["bytetrack"]
            self.tracker = ByteTracker(
                track_high_thresh=cfg["track_high_thresh"],
                track_low_thresh=cfg["track_low_thresh"],
                match_thresh=cfg["match_thresh"],
                max_time_lost=cfg["max_time_lost"],
                min_hits=self.config["sort"]["min_hits"],
            )
            return

        if self.tracker_name == "deepsort":
            cfg        = self.config["deepsort"]
            models_dir = os.path.join(_REPO_ROOT, "models")
            reid_path  = os.path.join(models_dir, "osnet_x1_0_veri_776.pth")
            if not os.path.exists(reid_path):
                st.error(
                    "**DeepSORT — ReID model not found.**  \n"
                    f"Expected: `models/osnet_x1_0_veri_776.pth`  \n"
                    "Place the VeRi-776 OSNet weights there, or select **SORT** (recommended) "
                    "or **ByteTrack** instead."
                )
                raise FileNotFoundError(
                    f"DeepSORT VeRi-776 ReID weights not found at: {reid_path}"
                )
            try:
                self.reid_model, self.reid_device = _load_reid_model(models_dir)
            except Exception as exc:
                st.error(
                    f"**DeepSORT — failed to load ReID model:** {exc}  \n"
                    "Select **SORT** or **ByteTrack** instead."
                )
                raise
            self.tracker = DeepSORT(
                appearance_thresh=cfg["appearance_thresh"],
                max_age=cfg["max_age"],
                min_hits=cfg["min_hits"],
                iou_threshold=cfg["iou_threshold"],
            )
            return

        raise ValueError(f"Unknown tracker: {self.tracker_name!r}")

    def reset(self):
        """Reset tracker state for the next video."""
        if self.tracker is not None:
            self.tracker.reset()

    def update(self, detections, frame_idx, frame=None):
        """
        Update the selected tracker and return Streamlit-friendly track dicts.

        Args:
            detections: list of detection dicts with bbox/conf keys
            frame_idx: current processed frame index
            frame: full BGR frame, required for DeepSORT ReID extraction
        """
        if self.tracker_name == "sort":
            return self._update_sort(detections, frame_idx)
        if self.tracker_name == "bytetrack":
            return self._update_bytetrack(detections, frame_idx)
        if self.tracker_name == "deepsort":
            return self._update_deepsort(detections, frame_idx, frame)
        return []

    def _update_sort(self, detections, frame_idx):
        det_array = np.array(
            [
                [
                    float(d["bbox"][0]),
                    float(d["bbox"][1]),
                    float(d["bbox"][2]),
                    float(d["bbox"][3]),
                    float(d.get("conf", 1.0)),
                ]
                for d in detections
            ],
            dtype=float,
        ) if detections else np.empty((0, 5), dtype=float)

        tracked = self.tracker.update(det_array, frame_idx)
        age_by_id = {int(t.id + 1): int(t.age) for t in self.tracker.trackers}

        output = []
        for row in tracked:
            bbox = [float(v) for v in row[:4]]
            track_id = int(row[4])
            output.append({
                "track_id": track_id,
                "bbox": bbox,
                "centroid": _bbox_to_centroid(bbox),
                "age": age_by_id.get(track_id, 0),
            })
        return output

    def _update_bytetrack(self, detections, frame_idx):
        tracked = self.tracker.update(detections, frame_idx)
        age_by_id = {int(t.track_id): int(t.age) for t in self.tracker.active_tracks}

        output = []
        for x1, y1, x2, y2, track_id, _conf in tracked:
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            track_id = int(track_id)
            output.append({
                "track_id": track_id,
                "bbox": bbox,
                "centroid": _bbox_to_centroid(bbox),
                "age": age_by_id.get(track_id, 0),
            })
        return output

    def _update_deepsort(self, detections, frame_idx, frame):
        if frame is None:
            raise ValueError("DeepSORT update requires the current frame for ReID extraction.")

        height, width = frame.shape[:2]
        embeddings = []
        for detection in detections:
            bbox_640 = _bbox_to_640_space(detection["bbox"], width, height)
            embeddings.append(
                extract_reid_feature(
                    self.reid_model,
                    self.reid_device,
                    frame,
                    bbox_640,
                    img_width=width,
                    img_height=height,
                )
            )

        tracked = self.tracker.update(detections, embeddings)
        age_by_id = {int(t.track_id): int(t.age) for t in self.tracker.tracks}

        output = []
        for x1, y1, x2, y2, track_id, _conf in tracked:
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            track_id = int(track_id)
            output.append({
                "track_id": track_id,
                "bbox": bbox,
                "centroid": _bbox_to_centroid(bbox),
                "age": age_by_id.get(track_id, 0),
            })
        return output
