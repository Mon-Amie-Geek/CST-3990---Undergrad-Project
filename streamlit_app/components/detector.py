"""
detector.py

Detector module supporting YOLOv8n and SSD300 for vehicle detection.
Student: MANJOO Ameera Najla | M01014463

Fix: @st.cache_resource must be on module-level functions, NOT instance methods.
     Instance methods are not hashable by Streamlit's caching machinery,
     which causes either an UnhashableParamError crash or silent cache-miss
     (model reloads on every frame → pipeline unusably slow).
Fix: model path is resolved to an absolute path so it works regardless of
     which directory 'streamlit run' is invoked from.
"""

import os
import cv2
import numpy as np
import streamlit as st
import torch
from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16
from ultralytics import YOLO

# ── Resolve the YOLOv8n weights path absolutely ───────────────────────────────
# detector.py lives at  streamlit_app/components/detector.py
# best.pt is expected at  <repo_root>/models/best.pt
#                      or  streamlit_app/models/best.pt  (fallback)
_COMP_DIR    = os.path.dirname(os.path.abspath(__file__))       # .../streamlit_app/components/
_APP_DIR     = os.path.normpath(os.path.join(_COMP_DIR, ".."))  # .../streamlit_app/
_REPO_ROOT   = os.path.normpath(os.path.join(_APP_DIR, ".."))   # repo root

def _resolve_yolo_path() -> str:
    """Return the first existing path for best.pt, or the repo-root default."""
    candidates = [
        os.path.join(_REPO_ROOT, "models", "best.pt"),       # repo root (primary)
        os.path.join(_APP_DIR,   "models", "best.pt"),       # streamlit_app/models/
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]   # let YOLO raise a clear FileNotFoundError

_YOLO_PATH = _resolve_yolo_path()


# ── Module-level cached loaders ───────────────────────────────────────────────
# These MUST be module-level functions for st.cache_resource to work.
# st.cache_resource on an instance method receives 'self' as a positional arg
# and tries to hash it — Detector objects are not hashable → crash or no-cache.

@st.cache_resource(show_spinner="Loading YOLOv8n model…")
def _load_yolov8n(model_path: str):
    """Load and cache the fine-tuned YOLOv8n model (CPU)."""
    try:
        model = YOLO(model_path)
        model.to("cpu")
        return model
    except FileNotFoundError:
        st.error(
            f"YOLOv8n weights not found: `{model_path}`\n\n"
            "Place `best.pt` in `models/` at the repository root."
        )
        raise
    except Exception as exc:
        st.error(f"Error loading YOLOv8n: {exc}")
        raise


@st.cache_resource(show_spinner="Loading SSD300 model…")
def _load_ssd300():
    """Load and cache the SSD300-VGG16 model from torchvision (COCO weights, CPU)."""
    try:
        model = ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
        model.to(torch.device("cpu"))
        model.eval()
        return model
    except Exception as exc:
        st.error(f"Error loading SSD300: {exc}")
        raise


# ── Detector class ────────────────────────────────────────────────────────────

class Detector:
    """
    Vehicle detector using YOLOv8n or SSD300.
    All inference runs on CPU with explicit device specification.
    """

    def __init__(self, model_name: str, conf_thresh: float = 0.25, nms_thresh: float = 0.45):
        """
        Args:
            model_name  : "yolov8n" or "ssd300"
            conf_thresh : Confidence threshold (0.0–1.0)
            nms_thresh  : NMS / IOU threshold (0.0–1.0)
        """
        self.model_name  = model_name
        self.conf_thresh = conf_thresh
        self.nms_thresh  = nms_thresh
        self.device      = torch.device("cpu")

        # Use module-level cached functions — safe for st.cache_resource
        if model_name == "yolov8n":
            self.model = _load_yolov8n(_YOLO_PATH)
        elif model_name == "ssd300":
            self.model = _load_ssd300()
        else:
            raise ValueError(f"Unknown detector: {model_name!r}")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a BGR frame.

        Returns:
            list of {"bbox": [x1,y1,x2,y2], "conf": float, "class_id": int}
        """
        if self.model is None:
            return []
        if self.model_name == "yolov8n":
            return self._detect_yolov8(frame)
        if self.model_name == "ssd300":
            return self._detect_ssd300(frame)
        return []

    # ── Private inference helpers ─────────────────────────────────────────────

    def _detect_yolov8(self, frame: np.ndarray) -> list[dict]:
        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.nms_thresh,
            verbose=False,
            device="cpu",
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    "bbox":     [float(x1), float(y1), float(x2), float(y2)],
                    "conf":     float(box.conf[0].item()),
                    "class_id": 0,
                })
        return detections

    def _detect_ssd300(self, frame: np.ndarray) -> list[dict]:
        frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = (
            torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model([frame_tensor])[0]

        vehicle_labels = {3, 4, 6, 8}   # COCO: car, motorcycle, bus, truck
        detections = []
        for box, score, label in zip(
            predictions.get("boxes", []),
            predictions.get("scores", []),
            predictions.get("labels", []),
        ):
            conf     = float(score.item())
            class_id = int(label.item())
            if conf < self.conf_thresh or class_id not in vehicle_labels:
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "bbox":     [float(x1), float(y1), float(x2), float(y2)],
                "conf":     conf,
                "class_id": class_id,
            })
        return detections
