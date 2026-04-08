"""
detector.py

Detector module supporting YOLOv8n and SSD300 for vehicle detection.
Student: MANJOO Ameera Najla | M01014463
"""

import cv2
import numpy as np
import streamlit as st
import torch
from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16
from ultralytics import YOLO


class Detector:
    """
    Vehicle detector using YOLOv8n or SSD300.
    All inference runs on CPU with explicit device specification.
    """

    def __init__(self, model_name, conf_thresh=0.25, nms_thresh=0.45):
        """
        Initialize detector.

        Args:
            model_name (str): One of "yolov8n" or "ssd300"
            conf_thresh (float): Confidence threshold (0.0-1.0)
            nms_thresh (float): NMS threshold (0.0-1.0)
        """
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.model = None
        self.device = torch.device("cpu")

        self._load_model()

    @st.cache_resource
    def _load_model_cached(_self, model_name):
        """Cache model loading to avoid reloading every frame."""
        if model_name == "yolov8n":
            try:
                model = YOLO("models/best.pt")
                model.to("cpu")
                return model
            except FileNotFoundError:
                st.error("Model file not found: models/best.pt")
                st.error("Please place the best.pt file in the models/ directory.")
                raise
            except Exception as e:
                st.error(f"Error loading YOLOv8n model: {str(e)}")
                raise

        if model_name == "ssd300":
            try:
                model = ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
                model.to("cpu")
                model.eval()
                return model
            except Exception as e:
                st.error(f"Error loading SSD300 model: {str(e)}")
                raise

        raise ValueError(f"Unknown model: {model_name}")

    def _load_model(self):
        """Load model with caching."""
        self.model = self._load_model_cached(self.model_name)

    def detect(self, frame):
        """
        Run inference on a frame.

        Args:
            frame (np.ndarray): Input frame (BGR)

        Returns:
            list: [{
                "bbox": [x1, y1, x2, y2],
                "conf": float,
                "class_id": int
            }]
        """
        if self.model is None:
            return []

        if self.model_name == "yolov8n":
            return self._detect_yolov8(frame)
        if self.model_name == "ssd300":
            return self._detect_ssd300(frame)
        return []

    def _detect_yolov8(self, frame):
        """YOLOv8 inference pipeline."""
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
                conf = box.conf[0].item()

                detections.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": float(conf),
                        "class_id": 0,
                    }
                )

        return detections

    def _detect_ssd300(self, frame):
        """torchvision SSD300 inference pipeline."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model([frame_tensor])[0]

        detections = []
        vehicle_labels = {3, 4, 6, 8}

        for box, score, label in zip(
            predictions.get("boxes", []),
            predictions.get("scores", []),
            predictions.get("labels", []),
        ):
            conf = float(score.item())
            class_id = int(label.item())

            if conf < self.conf_thresh or class_id not in vehicle_labels:
                continue

            x1, y1, x2, y2 = box.tolist()
            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": conf,
                    "class_id": class_id,
                }
            )

        return detections
