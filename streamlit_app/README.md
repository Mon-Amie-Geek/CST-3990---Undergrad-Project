# CST3990 — Traffic Anomaly Detection UI

**Student:** MANJOO Ameera Najla | **M01014463**  
**University:** Middlesex University Mauritius  
**Supervisor:** Mr Karel Veerabudren  
**Project Title:** A Comparative Study of Computer Vision Pipeline Components for Real-Time Traffic Analysis and Anomaly Detection

---

## 📋 Overview

This Streamlit application implements a configurable computer vision pipeline for analyzing traffic videos and detecting anomalous vehicle behaviors. The system supports multiple swappable components across four experimental blocks:

### Block A — Detection
- **YOLOv8n** (fine-tuned on UA-DETRAC)
- **MobileNet-SSD** (COCO pre-training via PyTorch Hub)

### Block B — Tracking
- **SORT** (Kalman filter + IoU association)
- **DeepSORT** (Kalman + appearance features)
- **ByteTrack** (High + low confidence recovery)

### Block C — Feature Extraction
- **F1: Density/Flow** — Vehicle count, ROI occupancy
- **F2: Speed Proxy** — Pixel displacement per second (smoothed)
- **F3: Interaction** — Inter-vehicle distance (normalized), dwell time

### Block D — Anomaly Detection
- **Rule-Based** — Z-score / IQR thresholds
- **One-Class SVM** — Novelty detection on feature space
- **Isolation Forest** — Anomaly scoring

---

## 🚀 Setup

### Prerequisites
- **Python:** 3.11+
- **OS:** Windows 11, Linux, or macOS
- **RAM:** 4GB minimum (8GB recommended)
- **Storage:** 2GB for models and dependencies
- **GPU:** Optional (system runs on CPU)

### Installation

1. **Navigate to the application directory:**
   ```powershell
   cd "C:\Users\Amii\OneDrive - Middlesex University\Final Year\CST 3990\CST-3990---Undergrad-Project\streamlit_app"
   ```

2. **(Optional) Create a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Place your model file:**
   ```
   Copy your fine-tuned best.pt to:
   models/best.pt
   ```

5. **Run the application:**
   ```powershell
   streamlit run app.py
   ```

The app will open in your default browser at `http://localhost:8501`

---

## 📖 Usage

### Main Pipeline Tab

1. **Upload Video** — Select a `.mp4` traffic video file
2. **Configure Block A** — Choose detector and thresholds:
   - Confidence threshold (0.10–0.90, default 0.25)
   - NMS threshold (0.10–0.90, default 0.45)
3. **Configure Block B** — Select tracking algorithm (SORT, DeepSORT, ByteTrack)
4. **Configure Block C** — Select features (Density, Speed, Distance)
5. **Configure Block D** — Choose anomaly method (Rule-Based, One-Class SVM, Isolation Forest)
6. **Visualization Options** — Toggle bounding boxes, track IDs, speed labels
7. **Run Pipeline** — Click to process the video

### Results Display

- **Live Detection Tab** — Real-time frame display with:
  - Green bounding boxes for normal vehicles
  - Red bounding boxes for anomalous vehicles
  - Track IDs and speed labels (if enabled)
  - Anomaly alerts with feature trigger information
  - Progress bar and FPS counter

- **Feature Charts Tab** — Plotly interactive charts:
  - Speed over time (with anomaly threshold)
  - Inter-vehicle distance over time
  - Vehicle count per frame

- **Results Summary Tab** — Complete analysis:
  - Configuration recap (metrics)
  - Performance statistics (FPS, latency percentiles)
  - Anomaly event log (DataFrame)
  - Download buttons for JSON/CSV exports

### Comparison Tab

- Shows placeholder experimental results from Days 5-18
- Compares detectors (mAP, Precision, Recall, FPS)
- Compares trackers (MOTA, IDF1, ID Switches)
- Evaluates feature sets (AUROC, Silhouette Score)
- Benchmarks anomaly methods (Precision, Recall, F1, FAR)
- Interactive F1 score comparison chart

### About Tab

- Full project details and student information
- Dataset references (UA-DETRAC, AI City Track 4, TU-DAT)
- **Important disclaimers:**
  - Distance values are image-space proxies only (NOT physical)
  - System operates in offline validation mode on CPU
  - Real-time deployment requires edge hardware (Jetson Nano, etc.)

---

## ⚙️ Technical Details

### Core Dependencies
- `streamlit` — Web UI framework
- `ultralytics` — YOLOv8 implementation
- `torch` — Deep learning (CPU mode)
- `torchvision` — Pre-trained models
- `opencv-python-headless` — Video I/O and drawing
- `numpy`, `pandas` — Data processing
- `scikit-learn` — ML algorithms (SVM, Isolation Forest)
- `filterpy` — Kalman filtering (SORT tracker)
- `plotly` — Interactive charts
- `scipy` — Signal processing

### Key Implementation Notes

1. **CPU-Only Operation**
   - All PyTorch operations explicitly set to CPU
   - No CUDA dependency or assumptions
   - Model inference runs on `device="cpu"`

2. **Frame Skipping for Performance**
   - Processes every 3rd frame (`process_every_n = 3`)
   - Continuous track state maintained across skipped frames
   - Trade-off: Speed vs detection frequency

3. **Dynamic FPS Handling**
   - FPS read from video at runtime, never hardcoded
   - Temporal windows computed in seconds, converted to frames
   - Speed always reported in pixels per second

4. **Normalization**
   - Inter-vehicle distance normalized by frame diagonal
   - Disclaimer: Image-space proxy, not physical distance
   - Used as relative behavioral indicator

5. **Stateless Runs**
   - Each video processed independently
   - Trackers reset at video start
   - Models cached with `@st.cache_resource`

---

## 📊 Output Files

When you complete a pipeline run, you can download:

- **events.json** — Anomaly events with frames, track IDs, features triggered
- **features.csv** — Frame-by-frame feature values (density, speed, distance)

Example CSV columns:
```
frame_idx,vehicle_count,roi_occupancy,mean_speed_px_sec,min_distance_norm,mean_dwell_sec
0,3,0.042,2.145,0.876,0.0
1,3,0.041,2.234,0.821,0.033
2,4,0.051,3.012,0.654,0.067
...
```

---

## ⚠️ Important Disclaimers

### Distance Measurement
**Distance values in this system are image-space proxies normalized by frame diagonal.**

They do **NOT** represent actual physical distances due to:
- Camera perspective and angle
- Varying object depth in the frame
- Lens distortion

Use these values as **relative behavioral indicators only**, not for spatial measurements.

### Real-Time Operation
**This system operates in offline validation mode on development hardware.**

Real-time operation requires:
- NVIDIA Jetson Nano or similar edge hardware
- Dedicated traffic cameras with embedded processors
- Optimized C++/CUDA implementations

Current CPU throughput is suitable for:
- Research and validation
- Post-incident analysis  
- Academic comparison studies

Not suitable for:
- Live traffic monitoring
- Production surveillance
- Real-time intervention systems

---

## 🔬 Experimental Flow

### Days 1-4: Setup & Validation
- ✅ Application framework (this deliverable)
- ✅ Component implementations
- Component validation with sample data

### Days 5-18: Experimentation
- Run full pipeline on UA-DETRAC test set
- Populate Block A detector comparison
- Populate Block B tracker comparison
- Populate Block C feature evaluation
- Populate Block D anomaly method comparison
- Generate performance metrics and charts

### Days 19-22: Results & Documentation
- Finalize all comparison tables
- Generate plots and visualizations
- Write project report
- Document findings and conclusions

---

## 🐛 Troubleshooting

### Model File Error
**Error:** `FileNotFoundError: models/best.pt not found`

**Solution:**
1. Ensure `models/best.pt` exists in the root directory
2. Check file permissions (read access required)
3. Alternatively, use MobileNet-SSD from the sidebar

### Out of Memory
**Error:** `RuntimeError: CUDA out of memory` or similar

**Solution:**
- Ensure system is not running other GPU-heavy applications
- Reduce video resolution if available
- Use every 3rd frame (already default)
- Restart Python kernel

### Video Upload Issues
**Error:** Video not accepted or crashes during loading

**Solution:**
- Ensure video is in MP4 format
- Check video codec compatibility (H.264, H.265)
- Verify video is not corrupted: `ffmpeg -v error -i video.mp4 -f null -`
- Try converting: `ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4`

### Slow Processing
**Issue:** FPS too low, processing takes too long

**Cause:** CPU-bound operation is normal on development hardware

**Options:**
1. The system already uses frame skipping (every 3rd frame)
2. Upload shorter video clips for testing  
3. Reduce confidence threshold (faster but lower precision)
4. Use MobileNet-SSD instead of YOLOv8 (faster but less accurate)

---

## 📞 Support & Contact

**Student:** MANJOO Ameera Najla  
**Student ID:** M01014463  
**University:** Middlesex University Mauritius  
**Supervisor:** Mr Karel Veerabudren

---

## 📝 References

- **YOLO:** Ultralytics YOLOv8 Documentation
- **SORT:** [Bewley et al., 2016](https://arxiv.org/abs/1602.00763)
- **DeepSORT:** [Wojke et al., 2017](https://arxiv.org/abs/1703.07402)
- **ByteTrack:** [Zhang et al., 2022](https://arxiv.org/abs/2110.06864)
- **UA-DETRAC:** [Wen et al., 2015](http://isl.cs.washington.edu/udetrac/)
- **Streamlit:** [streamlit.io](https://streamlit.io/)
- **PyTorch:** [pytorch.org](https://pytorch.org/)

---

**Last Updated:** April 2, 2026
