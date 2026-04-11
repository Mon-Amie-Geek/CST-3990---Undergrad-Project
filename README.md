# CST3990 Undergrad Project

Traffic anomaly detection research workspace and Streamlit demonstration app for a comparative study of computer vision pipeline components.

## Workspace Overview

- `src/`: Research scripts for Blocks A-D, evaluation, ingestion, and pipeline orchestration
- `streamlit_app/`: Streamlit UI for interactive pipeline demos and result visualisation
- `configs/`: YAML and JSON schemas for detectors, trackers, features, and trajectories
- `logs/`: Experimental outputs, comparison tables, and saved model artefacts
- `models/`: Local model weights and scaler artefacts required at runtime
- `data/`: Dataset directory structure placeholders only; raw datasets are not committed
- `detections_cache/`: Cached detector outputs used for reproducible tracker experiments

## Submission Scope

This repository is prepared as a reproducible project submission rather than a code-only skeleton.

- Curated experiment outputs under `logs/` are intentionally included
- Cached detections under `detections_cache/` are intentionally included
- Raw datasets are intentionally excluded
- The `.git/` directory is not part of the submission bundle

## Runtime Expectations

- Python `3.11+`
- Install dependencies from `requirements.txt` for research scripts
- Install dependencies from `streamlit_app/requirements.txt` for the Streamlit UI
- Local model files expected in `models/`:
  - `best.pt`
  - `osnet_x1_0_veri_776.pth`
  - `minmax_scaler.pkl`

## Key Notes

- DeepSORT is configured to use locally installed `torchreid` and the local `models/osnet_x1_0_veri_776.pth` weights file.
- The workspace is designed to run without hidden runtime downloads or automatic package installation.
- Large raw datasets and temporary local uploads are intentionally excluded from version control.
- See [SUBMISSION_NOTES.md](/d:/CST%203990/CST-3990---Undergrad-Project/SUBMISSION_NOTES.md) for the final submission checklist and bundle guidance.

## Streamlit App

From the repository root:

```powershell
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

See [streamlit_app/README.md](/d:/CST%203990/CST-3990---Undergrad-Project/streamlit_app/README.md) for app-specific setup and usage details.
