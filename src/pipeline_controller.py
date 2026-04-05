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
        """
        Day 9: Post-processing, error propagation, and tracker selection.
        Fully implemented — not a stub.

        Steps:
          1. Detect DAY7_LAYOUT + run pre-flight assertions
          2. Fragment filter + track interpolation (run_post_processing)
          3. Save fragment_filter_report.json and interpolation_report.json
          4. Error propagation correlation (run_error_propagation_analysis)
          5. Tracker selection (select_tracker → config_blockC.yaml)
          6. Finalise block_b_results_table.md
        """
        import os
        from src.post_processor import run_post_processing
        from src.error_propagation import (
            run_error_propagation_analysis,
            preflight_checks,
        )
        from src.tracker_selector import select_tracker

        # Load metadata
        with open("logs/metadata.json") as f:
            metadata = json.load(f)
        test_seqs = metadata["split_seqs"]["test"]   # ["MVI_20062", "MVI_20063"]
        trackers  = ["sort", "deepsort", "bytetrack"]

        # Detect Day 7/8 layout
        if os.path.exists("logs/block_b/sort/trajectories_MVI_20062.json"):
            day7_layout = "variant_a"
        elif os.path.exists("logs/block_b/trajectories/MVI_20062_sort.json"):
            day7_layout = "variant_b"
        else:
            raise FileNotFoundError(
                "Cannot detect Day 7/8 output layout. Run Days 7-8 first."
            )
        print(f"Layout detected: {day7_layout}")

        # Step 1: Pre-flight checks
        print("\n=== Step 1: Pre-flight Checks ===")
        schema_entry_fields = preflight_checks(
            metadata, test_seqs, trackers, day7_layout
        )

        # Step 2: Fragment filter + interpolation
        print("\n=== Step 2: Fragment Filter + Track Interpolation ===")
        fragment_report, interp_report = run_post_processing(
            trackers=trackers,
            test_seqs=test_seqs,
            day7_layout=day7_layout,
            schema_entry_fields=schema_entry_fields,
            min_track_length=15,
            max_interp_gap=3,
        )

        # Step 3: Save reports
        os.makedirs("logs/block_b", exist_ok=True)
        with open("logs/block_b/fragment_filter_report.json", "w") as f:
            json.dump({"min_track_length": 15, **fragment_report}, f, indent=2)
        with open("logs/block_b/interpolation_report.json", "w") as f:
            json.dump({"max_interp_gap": 3, **interp_report}, f, indent=2)
        print("Reports saved.")

        # Step 4: Error propagation correlation
        print("\n=== Step 3: Error Propagation Correlation ===")
        run_error_propagation_analysis(
            trackers=trackers,
            test_seqs=test_seqs,
            day7_layout=day7_layout,
            output_path="logs/block_b_error_propagation.json",
        )

        # Step 5: Tracker selection
        print("\n=== Step 4: Tracker Selection ===")
        selected = select_tracker()
        print(f"SELECTED TRACKER: {selected.upper()}")

        # Step 6: Finalise results table
        print("\n=== Step 5: Finalising Results Table ===")
        self._update_block_b_results_table(
            selected_tracker=selected,
            fragment_report=fragment_report,
            interp_report=interp_report,
            test_seqs=test_seqs,
        )

        print("\n=== Block B Analysis Complete ===")
        print("Canonical Block B outputs (_final.json) ready for Block C.")
        print(f"Tracker frozen: {selected} -> configs/config_blockC.yaml")

    def _update_block_b_results_table(self, selected_tracker, fragment_report,
                                       interp_report, test_seqs):
        """Writes the finalised block_b_results_table.md with two tables."""
        with open("logs/block_b_results.json") as f:
            res = json.load(f)

        tracker_rows = {t["tracker"]: t for t in res["trackers"]}
        trackers = ["sort", "deepsort", "bytetrack"]

        def sum_field(report, field):
            return {
                t: sum(report[t][s][field] for s in test_seqs if s in report.get(t, {}))
                for t in trackers
            }

        tb = sum_field(fragment_report, "tracks_before")
        ta = sum_field(fragment_report, "tracks_after")
        td = sum_field(fragment_report, "discarded")
        fi = sum_field(interp_report,   "frames_interpolated")

        lines = ["## Block B - Tracker Comparison Results\n\n"]
        lines.append(
            "| Tracker | MOTA | IDF1 | IDSW | Frag | FPS (median) | Selected |\n"
        )
        lines.append("|---|---|---|---|---|---|---|\n")
        for t_name in trackers:
            t   = tracker_rows[t_name]
            sel = "YES" if t["selected"] else "-"
            display = "DeepSORT (VeRi-776)" if t_name == "deepsort" else t_name.upper()
            lines.append(
                f"| {display} | {t['mota_mean']:.4f} | {t['idf1_mean']:.4f} | "
                f"{t['idsw_total']} | {t['fragmentations_total']} | "
                f"{t['fps_median']:.2f} | {sel} |\n"
            )
        lines.append(f"\n**Selected tracker: {selected_tracker.upper()}**\n")
        lines.append(f"**Rationale:** {res.get('selection_rationale', '')}\n\n")
        lines.append(
            "> **Statistical note (Pearson r):** Computed on n=2 test sequences -- "
            "r is degenerate (always +/-1.0 for 2 data points). "
            "This is a methodological scaffold; Block C will provide n=10 for a "
            "meaningful result.\n\n"
        )
        lines.append("---\n\n## Block B - Post-Processing Impact\n\n")
        lines.append(
            "| Tracker | Tracks Before | Tracks After | Discarded | Frames Interpolated |\n"
        )
        lines.append("|---|---|---|---|---|\n")
        for t_name in trackers:
            display = "DeepSORT (VeRi-776)" if t_name == "deepsort" else t_name.upper()
            lines.append(
                f"| {display} | {tb.get(t_name,'?')} | {ta.get(t_name,'?')} | "
                f"{td.get(t_name,'?')} | {fi.get(t_name,'?')} |\n"
            )
        lines.append(
            "\n*(Values summed across MVI_20062 and MVI_20063)*\n\n"
            "Fragment filter: `min_track_length=15` (Fix F11).\n"
            "Interpolation: `max_interp_gap=3 frames` (Fix F22). "
            "Gaps >3 frames: track split. No ghost interpolation.\n"
            "Block C reads exclusively from `_final.json` files.\n"
        )

        with open("logs/block_b_results_table.md", "w") as f:
            f.writelines(lines)
        print("block_b_results_table.md written.")

    def run_day9(self):
        """Convenience wrapper for full Day 9 execution."""
        self.run_block_b_analysis()

    def run_block_c(self):
        """Block C: Feature Extraction (Days 10-12).

        Day 10 delivers:
          - Generator-based clip loader (clip_feature_generator)
          - F1 features: vehicle_count, roi_occupancy
          - F2 features: vel_px_sec, vel_px_sec_smooth, is_interpolated
          - features.csv written per-clip to logs/block_c/

        Day 11 will add: scaler fitting, F3 features, AUROC computation.
        """
        from src.feature_extractor import clip_feature_generator, populate_config_blockc_tracker

        with open("configs/config_blockC.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        meta = json.load(open("logs/metadata.json"))

        # Day 10: populate tracker selection if not yet set.
        # populate_config_blockc_tracker() rewrites config_blockC.yaml via yaml.safe_dump()
        # and returns the updated in-memory dict — always use the returned dict.
        cfg = populate_config_blockc_tracker(cfg)

        # —— Feature Contract Gate —————————————————————————————————————————————
        # Hard stop if tracker field is still the Day 9 placeholder.
        # This must be asserted here AND inside clip_feature_generator() for defence-in-depth.
        assert cfg.get("tracker") not in (None, "", "<SELECTED_TRACKER_FROM_DAY9>"), (
            "config_blockC.yaml 'tracker' field is not populated after populate_config_blockc_tracker(). "
            "Verify that logs/block_b_results.json[\"trackers\"] has exactly one entry with 'selected: true'."
        )

        # Run generator over ALL sequences (train + val + test)
        all_seqs = (
            meta["split_seqs"]["train"] +
            meta["split_seqs"]["val"]   +
            meta["split_seqs"]["test"]
        )

        completed = 0
        for completed_id in clip_feature_generator(all_seqs, meta, cfg):
            print(f"[Block C] Features written: {completed_id}")
            completed += 1

        print(f"[Block C] Done. {completed}/{len(all_seqs)} clips processed.")
        # Day 11 will add: scaler fitting, F3 features, AUROC computation

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
