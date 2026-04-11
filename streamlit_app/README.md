# CST3990 Traffic Anomaly Detection UI

**Student:** MANJOO Ameera Najla | **M01014463**  
**University:** Middlesex University Mauritius  
**Supervisor:** Mr Karel Veerabudren

## Overview

This Streamlit application demonstrates the project pipeline across four experimental blocks:

- Block A: Detection with YOLOv8n or SSD300
- Block B: Tracking with SORT, DeepSORT, or ByteTrack
- Block C: Feature extraction for density, speed, and proximity signals
- Block D: Anomaly detection with rule-based logic, One-Class SVM, or Isolation Forest

The app reads completed experiment outputs from `../logs/` and also supports live video processing through the interactive pipeline page.

## Prerequisites

- Python `3.11+`
- Dependencies installed from `streamlit_app/requirements.txt`
- Local runtime artefacts in the repository `models/` directory:
  - `best.pt`
  - `osnet_x1_0_veri_776.pth`
  - `minmax_scaler.pkl`

## Setup

From the repository root:

```powershell
cd streamlit_app
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Pages

### Pipeline

- Upload an `.mp4` traffic video
- Choose the detector, tracker, feature groups, and anomaly method
- Run the pipeline and inspect live annotated frames, charts, and summary tables

### Comparison

- Shows completed experiment outputs for Blocks A, B, C, and D from the `logs/` directory
- Includes detector, tracker, feature-set, and anomaly-method comparisons
- Displays Block D FAR and latency summaries when the corresponding result files exist

### About

- Summarises project scope, datasets, implementation constraints, and research disclaimers

## Notes

- YOLO weights are resolved from the repository `models/` directory first, with `streamlit_app/models/` kept only as a fallback.
- DeepSORT requires the local `models/osnet_x1_0_veri_776.pth` file and does not perform runtime downloads.
- Uploaded videos are stored temporarily under `streamlit_app/.streamlit_uploads/` during local use and are ignored by git.
- Distance-related features are image-space proxies, not calibrated physical measurements.
- The current workspace is intended for offline validation and research demonstrations rather than production deployment.
- `streamlit_app/requirements.txt` pins `opencv-python-headless==4.11.0.86` and `packaging<24` to stay compatible with the Streamlit 1.32 setup used by this project.

## Troubleshooting

### `models/best.pt` not found

- Ensure `models/best.pt` exists at the repository root
- Confirm the file is readable by the current Python environment

### DeepSORT ReID model not found

- Ensure `models/osnet_x1_0_veri_776.pth` exists at the repository root
- Install `torchreid` through the pinned project dependencies before using DeepSORT

### Slow processing

- Increase the "Process every N-th frame" value
- Use shorter videos for demos
- Prefer SORT over DeepSORT when you want the lightest pipeline option
