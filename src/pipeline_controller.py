"""
pipeline_controller.py
Fixed central pipeline orchestrator
All blocks are wired through this controller to keep modules independent.
"""
import yaml
from src.seed_control import set_all_seeds

class PipelineController:
    def __init__(self, config_path: str):
        """
        Initialise the pipeline controller with a YAML config file.

        Args:
            config_path: path to the block-level config YAML file.
        """
        # Seeds must be set before anything else runs
        set_all_seeds()
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        print(f"[PipelineController] Loaded config from: {config_path}")

    def run_block_a(self):
        """Block A - Detector comparison (YOlOv8n vs SSD300)."""
        print("[PipelineController] Block A: Detector comparison - not yet implemented.")
        #Day 5-6 : Call src/detector.py

    def run_block_b(self):
        """ Block B - Tracker comparison (SORT vs DeepSORT vs ByteTrack)."""
        print("[PipelineController] Block B: Tracker comparison - not yet implemented.")
        #Day 7-9: Call src/tracker.py, reads detections_cache/

    def run_block_c(self):
        """Block C - Feature extraction comparison (F1, F2, F3)."""
        print("[PipelineController] Block C: Feature extraction - not yet implemented.")
        #Day 10-12: Call src/feature_extractor.py
    
    def run_block_d(self):
        """Block D - Anomaly detection (Rule-based, OC-SVM, Isolation Forest)"""
        print("[PipelineController] Block D: Anomaly detection - not yet implemented.")
        #Day 13-16: Call src/anomaly_detector.py

if __name__ == "__main__":
    import os
    test_cfg = "configs/config_blockA.yaml"
    if os.path.exists(test_cfg):
        controller = PipelineController(test_cfg)
        print("[PipelineController] Smoke test passed.")
    else:
        print(f"[PipelineController] Config not found at {test_cfg}.")
