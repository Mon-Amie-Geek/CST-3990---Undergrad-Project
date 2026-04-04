"""
pipeline_controller.py
Fixed central pipeline orchestrator
All blocks are wired through this controller to keep modules independent.
"""
import json
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
        """Block B — Multi-Object Tracking (Days 7+8).
        Runs all three trackers: SORT (Day 7), DeepSORT and ByteTrack (Day 8).
        Tracker selection is deferred to Day 9 after error propagation correlation.
        """
        from src.block_b_sort_runner import run_all_sequences as _sort_runner
        from src.tracker_deepsort import run_deepsort_on_sequences
        from src.tracker_bytetrack import run_bytetrack_on_sequences

        cfg = self.cfg
        test_seqs = ["MVI_20062", "MVI_20063"]   # eval split — frozen

        # --- SORT (Day 7) ---
        # block_b_sort_runner uses its own config loading; we pass GT root only
        sort_results = _sort_runner(
            seq_ids=test_seqs,
            gt_root="data/ua_detrac/annotations"
        )

        # --- DeepSORT (Day 8) ---
        deepsort_results = run_deepsort_on_sequences(
            seq_ids=test_seqs,
            split="test",
            config=cfg
        )

        # --- ByteTrack (Day 8) ---
        bytetrack_results = run_bytetrack_on_sequences(
            seq_ids=test_seqs,
            split="test",
            config=cfg
        )

        # Aggregate and save block_b_results.json
        self._save_block_b_results(sort_results, deepsort_results, bytetrack_results)

        # Tracker selection deferred to Day 9 (run_block_b_analysis)
        return {
            "sort": sort_results,
            "deepsort": deepsort_results,
            "bytetrack": bytetrack_results
        }

    def _save_block_b_results(self, sort_results, deepsort_results, bytetrack_results):
        """Persist aggregated Block B results to logs/block_b_results.json."""
        results_path = "logs/block_b_results.json"
        try:
            with open(results_path) as f:
                block_b = json.load(f)
        except FileNotFoundError:
            block_b = {
                "block": "B",
                "evaluation_split": ["MVI_20062", "MVI_20063"],
                "selected_detector": "yolov8n",
                "trackers": []
            }

        # Update or append DeepSORT row
        self._upsert_tracker_row(block_b, {
            "tracker": "deepsort",
            "reid_model": "osnet_x1_0_veri_776",
            **deepsort_results.get("aggregate", {}),
            "per_seq": deepsort_results.get("per_seq", {}),
            "selected": False
        })

        # Update or append ByteTrack row
        self._upsert_tracker_row(block_b, {
            "tracker": "bytetrack",
            "reid_model": "none",
            **bytetrack_results.get("aggregate", {}),
            "per_seq": bytetrack_results.get("per_seq", {}),
            "selected": False
        })

        block_b["note"] = (
            "Tracker selection deferred to Day 9 after error propagation correlation "
            "(Fix F30 supplement)"
        )

        with open(results_path, "w") as f:
            json.dump(block_b, f, indent=2)
        print(f"[PipelineController] Block B results saved to {results_path}")

    @staticmethod
    def _upsert_tracker_row(block_b: dict, new_row: dict):
        """Insert or replace a tracker row in block_b['trackers']."""
        name = new_row["tracker"]
        for i, row in enumerate(block_b["trackers"]):
            if row.get("tracker") == name:
                block_b["trackers"][i] = new_row
                return
        block_b["trackers"].append(new_row)

    def run_block_b_analysis(self):
        """Day 9 — error propagation + tracker selection. Scaffold only today."""
        from src.error_propagation import placeholder_note
        print(placeholder_note())
        # Full execution in Day 9 after Block C speed features exist.
        # Call compute_error_propagation_correlation(idsw_per_seq, speed_std_per_seq)
        # then select the best tracker and freeze in config_blockC.yaml.

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
