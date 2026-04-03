"""
app.py

Main entry point for the CST3990 Traffic Anomaly Detection Streamlit application.
Student: MANJOO Ameera Najla | M01014463
Project: CST3990 Undergraduate Individual Project
"""

import streamlit as st
import torch
import numpy as np
import random

# Set seeds for reproducibility (Critical Rule #2)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="CST3990 Traffic Anomaly Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HEADER
# ============================================================================
col1, col2, col3 = st.columns([2, 3, 2])

with col1:
    st.markdown("## 🎓 CST3990")
    st.markdown("**Undergraduate Individual Project**")

with col2:
    st.markdown("## 🚗 Traffic Anomaly Detection")
    st.markdown("### Computer Vision Pipeline Comparison")

with col3:
    st.markdown("### 👤 MANJOO Ameera Najla")
    st.markdown("**M01014463**")
    st.markdown("**Middlesex University Mauritius**")

st.markdown("---")

# ============================================================================
# PROJECT DESCRIPTION
# ============================================================================
st.markdown("""
### 📋 Project Overview

This application implements a configurable computer vision pipeline for **real-time traffic 
analysis and anomaly detection**. The system allows selection of different components across 
four experimental blocks:

- **Block A — Detection:** YOLOv8n (fine-tuned on UA-DETRAC) vs MobileNet-SSD (COCO)
- **Block B — Tracking:** SORT vs DeepSORT vs ByteTrack  
- **Block C — Feature Extraction:** Density/Flow, Speed Proxy, Inter-Vehicle Distance
- **Block D — Anomaly Detection:** Rule-Based (Z-score), One-Class SVM, Isolation Forest

**How to use:**
1. Go to the **Pipeline** page
2. Upload a traffic video (.mp4)
3. Configure your pipeline components using the sidebar
4. Click **Run Pipeline**
5. View live detection results, feature charts, and anomaly summaries

All processing runs on **CPU** (no GPU required). The system processes every 3rd frame 
for performance on development hardware.
""")

st.markdown("---")

# ============================================================================
# PROJECT INFO FOOTER
# ============================================================================
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.subheader("👤 Student")
    st.write("**Name:** MANJOO Ameera Najla")
    st.write("**ID:** M01014463")

with footer_col2:
    st.subheader("🎓 University")
    st.write("**Institution:** Middlesex University Mauritius")
    st.write("**Supervisor:** Mr Karel Veerabudren")

with footer_col3:
    st.subheader("📚 Resources")
    st.write("🔗 Use the sidebar to navigate:")
    st.write("• Pipeline — Run analysis")
    st.write("• Comparison — View results")
    st.write("• About — Project details")

st.markdown("---")

st.markdown("""
<hr style="border: 1px solid #ccc; margin: 20px 0;">
<p style="text-align: center; color: #666; font-size: 12px;">
CST3990 Undergraduate Individual Project | 
Middlesex University Mauritius | 
A Comparative Study of Computer Vision Pipeline Components for Real-Time Traffic Analysis and Anomaly Detection
</p>
""", unsafe_allow_html=True)

# ============================================================================
# REQUIREMENTS CHECK
# ============================================================================
import warnings
warnings.filterwarnings("ignore")

# Check for model file
import os
if not os.path.exists("models/best.pt"):
    st.warning("""
    ⚠️ **Model File Not Found**
    
    The YOLOv8n model file (`models/best.pt`) is not present.
    
    **To set up:**
    1. Place your fine-tuned YOLOv8n model at: `models/best.pt`
    2. Reload this page
    
    You can still use MobileNet-SSD from the pipeline.
    """)
