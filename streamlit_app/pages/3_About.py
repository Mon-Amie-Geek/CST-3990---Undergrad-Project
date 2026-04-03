"""
3_About.py

About page showing project details and important disclaimers.
Student: MANJOO Ameera Najla | M01014463
"""

import streamlit as st

st.set_page_config(page_title="About - CST3990 Traffic Anomaly Detection", layout="wide")

st.title("📋 About This Project")

st.markdown("---")

# Project Information
st.header("Project Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Student Information")
    st.write("**Name:** MANJOO Ameera Najla")
    st.write("**Student ID:** M01014463")
    st.write("**University:** Middlesex University Mauritius")
    st.write("**Supervisor:** Mr Karel Veerabudren")

with col2:
    st.subheader("Course Information")
    st.write("**Course Code:** CST3990")
    st.write("**Course Title:** Undergraduate Individual Project")
    st.write("**Project Type:** Final Year Dissertation")

st.markdown("---")

st.header("Project Title")
st.write("""
### A Comparative Study of Computer Vision Pipeline Components 
### for Real-Time Traffic Analysis and Anomaly Detection

This project investigates and compares different components of a computer vision pipeline 
for analyzing traffic video. The system includes configurable options for:

- **Block A — Detection:** YOLOv8n (fine-tuned on UA-DETRAC) vs MobileNet-SSD (COCO, PyTorch Hub)
- **Block B — Tracking:** SORT vs DeepSORT vs ByteTrack
- **Block C — Features:** Density/Flow, Speed, Inter-vehicle Distance
- **Block D — Anomaly Detection:** Rule-Based, One-Class SVM, Isolation Forest
""")

st.markdown("---")

st.header("Datasets")
st.write("""
The model used in this application was trained on:

- **UA-DETRAC** — Urban-scene Detection, Tracking and Counting dataset
  - Focus: Traffic vehicle detection and tracking
  - Model: YOLOv8n fine-tuned on this dataset (best.pt)

Anomaly ground truth dataset (Block D only):
- **AI City Track 4 (2021)** — Stall and crash event annotations
""")

st.markdown("---")

st.header("Model Details")
st.write("""
### Primary Model: YOLOv8n (best.pt)
- **Architecture:** YOLOv8 Nano (lightweight)
- **Training Data:** UA-DETRAC dataset
- **Task:** Vehicle detection (single class: vehicle)
- **Hardware:** Trained for deployment on CPU devices
- **Performance:** Optimized for real-time inference on embedded systems

### Alternative Model: MobileNet-SSD
- **Source:** PyTorch Hub — `NVIDIA/DeepLearningExamples:torchhub` (`nvidia_ssd`, fp32)
- **Pretrained On:** COCO dataset (no fine-tuning — Fix F06)
- **Thresholds:** conf=0.25, nms=0.45 (same as YOLOv8n — Fix F23)
- **Inference:** CPU-only, evaluated directly on UA-DETRAC test split
""")

st.markdown("---")

st.header("⚠️ Important Disclaimers")

st.warning("""
**Distance Measurement Disclaimer**

Distance values displayed in this system are **image-space proxies normalised by frame diagonal**. 

⚠️ **These values do NOT represent physical distances.**

Perspective effects and camera angle mean that pixel distance does not linearly correspond to 
real-world distance. All distance metrics should be interpreted as **relative behavioural indicators only**.

Use for motion pattern analysis and anomaly detection, not for accurate spatial measurements.
""")

st.info("""
**Real-Time Operation Scope**

This system operates in **offline validation mode** on development hardware.

Real-time operation requires dedicated edge hardware such as:
- NVIDIA Jetson Nano
- Edge TPUs
- Specialized traffic cameras with embedded processors

The pipeline architecture is designed to support real-time deployment. However:
- Current throughput is limited by **CPU constraints** on development machines
- This is a **research/validation tool**, not a production deployment
- Frame processing every 3rd frame is used for speed on CPU
""")

st.markdown("---")

st.header("Technical Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Core Framework")
    st.write("""
    - Python 3.11.9
    - Streamlit
    - PyTorch (CPU)
    - OpenCV
    """)

with col2:
    st.subheader("Computer Vision")
    st.write("""
    - YOLOv8 (Ultralytics)
    - MobileNet-SSD
    - Kalman Filtering
    - SORT Tracker
    """)

with col3:
    st.subheader("Machine Learning")
    st.write("""
    - Scikit-Learn
    - One-Class SVM
    - Isolation Forest
    - Filterpy (Kalman)
    """)

st.markdown("---")

st.header("System Requirements")
st.write("""
- **OS:** Windows 11 / Linux / macOS
- **Python:** 3.11+
- **RAM:** Minimum 4GB (8GB recommended)
- **Storage:** 2GB for models and dependencies
- **GPU:** Optional (system runs on CPU)
- **Camera:** Not required (offline video processing)
""")

st.markdown("---")

st.header("Future Work")
st.write("""
- **Real-time deployment** on NVIDIA Jetson Nano
- **Multi-camera calibration** for physical distance estimation
- **Perspective-aware** feature extraction
- **Deep learning-based** anomaly detection models
- **Regional analysis** with ROI selection
- **Multi-class** detection (cars, motorcycles, trucks, etc.)
""")

st.markdown("---")

st.subheader("For Questions or Support")
st.write("Contact: MANJOO Ameera Najla (M01014463)")
st.write("Supervisor: Mr Karel Veerabudren")
st.write("University: Middlesex University Mauritius")
