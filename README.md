# CST3990 Undergrad Project

Traffic anomaly detection research workspace and Streamlit demonstration app for a comparative study of computer vision pipeline components.

## Quick Start

```bash
# Clone and setup
git clone <https://github.com/Mon-Amie-Geek/CST-3990---Undergrad-Project>
cd CST-3990---Undergrad-Project
python -m venv venv
venv\Scripts\activate  # On Windows; use 'source venv/bin/activate' on macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download large files from Drive (see below), then run:
python src/block_a_detector_comparison.py  # Example script
```

---

## Getting Started

### Prerequisites

- **Python 3.11 or higher**
- **pip** or **conda** package manager
- **Git** (to clone the repository)
- **Google Drive access** (for downloading model weights and datasets)

### Installation Steps

#### Step 1: Clone the Repository

```bash
git clone <your-github-repo-url>
cd CST-3990---Undergrad-Project
```

#### Step 2: Create a Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

#### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### Step 4: Download Required Model Files

Large model files are stored on **Google Drive** (not in the GitHub repo):

📥 **[Download from Google Drive](https://drive.google.com/drive/folders/1zoy-AT3xFCTAdrbL2mPdyDSbtIg1RghF?usp=sharing)**

Extract and place the following files in the `models/` directory:
- `best.pt` — YOLOv8 detector weights
- `osnet_x1_0_veri_776.pth` — DeepSORT ReID model
- `minmax_scaler.pkl` — Feature scaler (if applicable)

```
models/
├── best.pt
├── osnet_x1_0_veri_776.pth
└── minmax_scaler.pkl
```

#### Step 5: Download Datasets (Optional, for Re-running Experiments)

Datasets are available at: **[https://drive.google.com/drive/folders/192YQNLgm3JWsh4PFxFd_a5IIjbir-MWS?usp=sharing]**

Extract to the `data/` directory. Expected structure:
```
data/
├── ua_detrac/
├── ai_city/
└── tu_dat/
```

---

## Running the Project

### Option 1: Research Scripts (Blocks A-D)

Run individual pipeline components:

```bash
# Block A: Object Detection Comparison
python src/block_a_detector_comparison.py

# Block B: Multi-Object Tracking (SORT/DeepSORT)
python src/block_b_sort_runner.py

# Block C: Feature Extraction & Anomaly Detection
python src/block_c_evaluator.py

# Block D: Rule-Based Alert System
python src/block_d_day15_bridge.py
```

See `src/` directory for all available scripts.

### Option 2: Jupyter Notebooks

Development and experimentation notebooks are available in the shared location:

📓 **[Jupyter Notebooks](https://drive.google.com/drive/folders/1l7WDwLUMUEA6f0Q3IxMuELo5h7lIADRL?usp=sharing)**

To run locally:
```bash
pip install jupyter
jupyter notebook
# Open and run notebooks from your browser
```

Notebooks follow a day-by-day progression (day01_setup.ipynb → day18_result_consolidation.ipynb).

### Option 3: Streamlit Web App (Interactive Demo)

Launch the interactive pipeline visualization:

```bash
cd streamlit_app
pip install -r streamlit_app/requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

See [streamlit_app/README.md](streamlit_app/README.md) for detailed usage.

---

## Project Structure

```
CST-3990---Undergrad-Project/
├── src/                          # Research pipeline scripts
│   ├── block_a_*.py             # Block A: Detection comparison
│   ├── block_b_*.py             # Block B: Multi-object tracking
│   ├── block_c_*.py             # Block C: Feature extraction & anomaly detection
│   ├── block_d_*.py             # Block D: Alert system
│   └── ...                      # Utility and orchestration scripts
│
├── streamlit_app/               # Interactive web UI
│   ├── app.py                   # Main app entry point
│   ├── requirements.txt          # Streamlit-specific dependencies
│   └── pages/                   # App pages
│
├── configs/                      # Configuration files (YAML, JSON)
│   ├── config_blockA.yaml       # Block A detector configs
│   ├── config_blockB.yaml       # Block B tracker configs
│   ├── config_blockC.yaml       # Block C feature configs
│   ├── config_blockD.yaml       # Block D rule configs
│   └── *.json                   # Schema definitions
│
├── models/                       # Pre-trained weights (NOT in repo)
│   ├── best.pt                  # YOLOv8 detector
│   ├── osnet_x1_0_veri_776.pth  # DeepSORT ReID backbone
│   └── minmax_scaler.pkl        # Feature scaler
│
├── logs/                         # Experiment results & outputs
│   ├── block_a_results.json     # Block A experiment logs
│   ├── block_b_results.json     # Block B experiment logs
│   └── ...                      # Additional outputs
│
├── detections_cache/            # Cached detector outputs (for reproducibility)
│   ├── train/
│   ├── val/
│   └── test/
│
├── data/                         # Dataset directories (structure only)
│   ├── ua_detrac/               # UA-DETRAC dataset
│   ├── ai_city/                 # AI-City dataset
│   └── tu_dat/                  # TU-DAT dataset
│
├── requirements.txt             # Python dependencies (pinned versions)
├── README.md                    # This file
├── SUBMISSION_NOTES.md          # Submission checklist & bundle guidance
└── thesis_notes.md              # Research notes
```

---

## Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'X'"

**Solution:** Reinstall dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For Streamlit issues specifically:
```bash
cd streamlit_app
pip install -r streamlit_app/requirements.txt
```

---

### ❌ "Model file not found" (best.pt, osnet_x1_0_veri_776.pth)

**Solution:** Download from Google Drive and place in `models/` directory

📥 **[Download Models](https://drive.google.com/drive/folders/1Hq3n9oNp24bT4FXM-5QgmO4Qe4-Wjo3f?usp=sharing)**

Verify files exist:
```bash
ls models/  # macOS/Linux
dir models  # Windows
```

---

### ❌ "CUDA out of memory" or GPU not detected

**Solution:** Code will fall back to CPU (slower but functional)

Optional: Install CUDA 12.1+ and cuDNN for GPU acceleration

---

### ❌ "Streamlit won't start"

**Solution:**
1. Ensure you're in the correct directory: `cd streamlit_app`
2. Reinstall Streamlit dependencies: `pip install -r requirements.txt`
3. Try: `streamlit run app.py --logger.level=debug`

---

## Important Notes

### Large Files & Cloud Storage

- **Model weights** (100+ MB) are stored on Google Drive
- **Raw datasets** are NOT included in this repo for size reasons
- **Cached detections** (in `detections_cache/`) ARE included for reproducibility without re-running expensive detectors
- **Experiment logs** (in `logs/`) are included as reference outputs

### Reproducibility

This project is designed for reproducibility:
- All dependency versions are pinned in `requirements.txt`
- Cached detection outputs allow running downstream blocks without re-training detectors
- Configuration files are versioned alongside code

### Virtual Environment

Always activate the virtual environment before running code:

**Windows:**
```powershell
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

---

## Key Implementation Notes

- **DeepSORT** is configured to use locally installed `torchreid` and the `models/osnet_x1_0_veri_776.pth` weights file
- The workspace runs **without hidden downloads or automatic package installation**
- Model inference supports **GPU (CUDA) and CPU** modes
- See [SUBMISSION_NOTES.md](SUBMISSION_NOTES.md) for final submission details

---

## Support & Documentation

- 📋 **Configuration details:** See `configs/` directory
- 📊 **Results & logs:** See `logs/` directory
- 🔬 **Research progress:** See `thesis_notes.md`
- 📓 **Development process:** See shared Jupyter notebooks link above

Contact : najla.manjoo@gmail.com for queries.
