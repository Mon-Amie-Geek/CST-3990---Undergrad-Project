## Block A — Detector Comparison Results

**Dataset:** UA-DETRAC test split (2 sequences)  
**Hardware:** Google Colab T4 GPU (Tesla T4, 15.6 GB)  
**conf_thresh:** 0.25 | **nms_thresh:** 0.45  

| Detector | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | FPS (p50) | Lat p95 (ms) | Selected |
|----------|---------|--------------|-----------|--------|-----------|--------------|----------|
| YOLOv8n (fine-tuned UA-DETRAC) | 0.6674 | 0.4506 | 0.7679 | 0.7703 | 47.2 | 25.1 | ✓ |
| SSD300 (COCO pretrained) | 0.3160 | 0.2152 | 0.2801 | 0.5582 | 25.6 | 51.2 |  |

**Selected:** `yolov8n` — mAP50 difference 0.3514 > threshold 0.05. YOLOv8n=0.6674, SSD300=0.316. Accuracy criterion dominates.

> **Construct validity note:** YOLOv8n trained at 960×540, evaluated at 640×640. SSD300 natively processes 300×300; bboxes scaled to 640×640 for schema consistency. See §3.8.2.

> **Hardware note:** FPS measured on Colab T4 GPU. Day 17 profiling uses Surface Pro 7 CPU. Values are not directly comparable.