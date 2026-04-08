"""
pipeline_controller.py
Fixed central pipeline orchestrator.
All blocks are wired through this controller to keep modules independent.
"""

import glob
import json
import logging
import os

import yaml

from src.seed_control import set_all_seeds

logger = logging.getLogger(__name__)


class PipelineController:
    def __init__(self, config_path: str):
        """
        Initialise the pipeline controller with a YAML config file.

        Args:
            config_path: path to the block-level config YAML file.
        """
        set_all_seeds()  # seeds must be set before anything else runs

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        print(f"[PipelineController] Loaded config from: {config_path}")

    def run_block_a(self):
        """Block A - Detector comparison (YOLOv8n vs SSD300)."""
        print("[PipelineController] Block A: Detector comparison - not yet implemented.")
        # Day 5-6: Call src/detector.py

    def run_block_b(self):
        """
        Block B — Multi-Object Tracking (Days 7+8).

        Runs all three trackers on ALL split sequences (train + val + test) so
        that Block C / Day 9 post-processing can produce _final.json files for
        all 10 clips.  Evaluation metrics stored in block_b_results.json are
        computed exclusively over the test split to preserve comparability.

        Trackers:
          - SORT      (Day 7)
          - DeepSORT  (Day 8)
          - ByteTrack (Day 8)

        Tracker selection is deferred to Day 9 after error propagation correlation.
        """
        from src.block_b_sort_runner import run_all_sequences as _sort_runner
        from src.tracker_bytetrack import run_bytetrack_on_sequences
        from src.tracker_deepsort import run_deepsort_on_sequences

        cfg = self.cfg
        gt_root = "data/ua_detrac/annotations"

        # --- Load all sequences from metadata (source of truth) ---
        with open("logs/metadata.json", "r") as f:
            metadata = json.load(f)

        train_seqs = metadata["split_seqs"]["train"]
        val_seqs   = metadata["split_seqs"]["val"]
        test_seqs  = metadata["split_seqs"]["test"]
        all_seqs   = train_seqs + val_seqs + test_seqs

        # Map each sequence to its detection-cache split
        seq_split_map = (
            {s: "train" for s in train_seqs}
            | {s: "val"   for s in val_seqs}
            | {s: "test"  for s in test_seqs}
        )

        # --- SORT (Day 7): all sequences; test-only aggregate stored ---
        sort_all = _sort_runner(
            seq_ids=all_seqs,
            gt_root=gt_root,
            seq_split_map=seq_split_map,
        )
        # Filter aggregate to test sequences so stored metrics stay comparable
        test_sort_rows = {
            s: v for s, v in sort_all.get("per_seq", {}).items()
            if s in test_seqs
        }
        if test_sort_rows:
            import numpy as np
            sort_results = {
                "per_seq": test_sort_rows,
                "aggregate": {
                    "mota_mean"           : round(float(
                        sum(v["mota"] for v in test_sort_rows.values()) / len(test_sort_rows)
                    ), 6),
                    "idf1_mean"           : round(float(
                        sum(v["idf1"] for v in test_sort_rows.values()) / len(test_sort_rows)
                    ), 6),
                    "idsw_total"          : sum(v["idsw"] for v in test_sort_rows.values()),
                    "fragmentations_total": sum(v["fragmentations"] for v in test_sort_rows.values()),
                    "fps_median"          : round(float(
                        np.median([v["fps_median"] for v in test_sort_rows.values()])
                    ), 2),
                },
            }
        else:
            sort_results = sort_all  # fallback: use whatever was computed

        # --- DeepSORT (Day 8): test split for metrics; other splits for trajectories ---
        deepsort_results = run_deepsort_on_sequences(
            seq_ids=test_seqs,
            split="test",
            config=cfg,
        )
        for split_name, split_seqs in [("val", val_seqs), ("train", train_seqs)]:
            if split_seqs:
                run_deepsort_on_sequences(
                    seq_ids=split_seqs,
                    split=split_name,
                    config=cfg,
                )

        # --- ByteTrack (Day 8): test split for metrics; other splits for trajectories ---
        bytetrack_results = run_bytetrack_on_sequences(
            seq_ids=test_seqs,
            split="test",
            config=cfg,
        )
        for split_name, split_seqs in [("val", val_seqs), ("train", train_seqs)]:
            if split_seqs:
                run_bytetrack_on_sequences(
                    seq_ids=split_seqs,
                    split=split_name,
                    config=cfg,
                )

        self._save_block_b_results(sort_results, deepsort_results, bytetrack_results)

        return {
            "sort": sort_results,
            "deepsort": deepsort_results,
            "bytetrack": bytetrack_results,
        }

    def _save_block_b_results(self, sort_results, deepsort_results, bytetrack_results):
        """Persist aggregated Block B results to logs/block_b_results.json."""
        results_path = "logs/block_b_results.json"

        # Derive test sequences from metadata (source of truth)
        with open("logs/metadata.json", "r") as f:
            _meta = json.load(f)
        _test_seqs = _meta["split_seqs"]["test"]

        try:
            with open(results_path, "r") as f:
                block_b = json.load(f)
        except FileNotFoundError:
            block_b = {
                "block": "B",
                "evaluation_split": _test_seqs,
                "selected_detector": "yolov8n",
                "trackers": [],
            }

        # Keep/refresh SORT row too if already present or newly produced
        self._upsert_tracker_row(
            block_b,
            {
                "tracker": "sort",
                "reid_model": "none",
                **sort_results.get("aggregate", {}),
                "per_seq": sort_results.get("per_seq", {}),
                "selected": False,
            },
        )

        self._upsert_tracker_row(
            block_b,
            {
                "tracker": "deepsort",
                "reid_model": "osnet_x1_0_veri_776",
                **deepsort_results.get("aggregate", {}),
                "per_seq": deepsort_results.get("per_seq", {}),
                "selected": False,
            },
        )

        self._upsert_tracker_row(
            block_b,
            {
                "tracker": "bytetrack",
                "reid_model": "none",
                **bytetrack_results.get("aggregate", {}),
                "per_seq": bytetrack_results.get("per_seq", {}),
                "selected": False,
            },
        )

        block_b["note"] = (
            "Tracker selection deferred to Day 9 after error propagation "
            "correlation (Fix F30 supplement)."
        )

        os.makedirs("logs", exist_ok=True)
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

    def _detect_day7_layout(self) -> str:
        """Detect which Day 7/8 trajectory layout is present."""
        if os.path.exists("logs/block_b/sort/trajectories_MVI_20062.json"):
            return "variant_a"
        if os.path.exists("logs/block_b/trajectories/MVI_20062_sort.json"):
            return "variant_b"
        raise FileNotFoundError(
            "Cannot detect Day 7/8 output layout. Run Days 7-8 first."
        )

    def _discover_available_raw_sequences(self, trackers):
        """
        Discover sequences that actually have raw trajectory JSONs
        for all required trackers.

        This prevents Day 9 from trying to post-process manifest sequences
        whose raw Block B outputs were never generated.
        """
        seq_sets = []

        for tracker in trackers:
            pattern = f"logs/block_b/trajectories/*_{tracker}.json"
            files = glob.glob(pattern)

            seqs = []
            suffix = f"_{tracker}.json"

            for fp in files:
                name = os.path.basename(fp)

                # raw files only; exclude *_filtered.json and *_final.json
                if not name.endswith(suffix):
                    continue

                seq = name[: -len(suffix)]
                if seq.endswith("_filtered") or seq.endswith("_final"):
                    continue

                seqs.append(seq)

            seq_sets.append(set(seqs))

        if not seq_sets:
            return []

        return sorted(set.intersection(*seq_sets))

    def _generate_missing_raw_trajectories(
        self, missing_seqs, seq_split_map, trackers, day7_layout
    ):
        """
        Generate raw trajectory JSONs for manifest sequences that are absent
        from logs/block_b/trajectories/.

        Called by run_block_b_analysis() as a repair step so that Day 9
        post-processing covers all 10 split sequences, not just the 2 test clips.
        Detection caches must exist for each sequence under detections_cache/.
        GT annotations must be present for metrics; trajectories are written
        before the metrics call, so the JSON is safe even if evaluation fails.
        """
        import yaml

        from src.block_b_sort_runner import run_sort_on_sequence, save_trajectories
        from src.tracker_bytetrack import run_bytetrack_on_sequence
        from src.tracker_deepsort import build_reid_model, run_deepsort_on_sequence

        with open("configs/config_blockB.yaml", "r") as f:
            cfg_b = yaml.safe_load(f)

        gt_root = "data/ua_detrac/annotations"

        print(
            f"\n[Day 9 Repair] Generating raw trajectories for "
            f"{len(missing_seqs)} sequences: {missing_seqs}"
        )

        # Build ReID model once — reused across all DeepSORT sequences
        reid_model, reid_device = None, None
        if "deepsort" in trackers:
            reid_model, reid_device = build_reid_model()

        for seq_id in missing_seqs:
            split = seq_split_map.get(seq_id, "test")
            print(f"\n  [{seq_id}]  split={split}")

            if "sort" in trackers:
                try:
                    raw_traj, warmup_b, meta, _ = run_sort_on_sequence(
                        seq_id, split=split
                    )
                    save_trajectories(seq_id, raw_traj, warmup_b, meta)
                    print(f"    SORT      → saved")
                except Exception as exc:
                    print(f"    SORT      ERROR: {exc}")

            if "deepsort" in trackers:
                try:
                    run_deepsort_on_sequence(
                        seq_id,
                        split,
                        cfg_b,
                        reid_model=reid_model,
                        reid_device=reid_device,
                        gt_root=gt_root,
                        day7_layout=day7_layout,
                    )
                    print(f"    DeepSORT  → saved")
                except Exception as exc:
                    print(f"    DeepSORT  ERROR: {exc}")

            if "bytetrack" in trackers:
                try:
                    run_bytetrack_on_sequence(
                        seq_id,
                        split,
                        cfg_b,
                        gt_root=gt_root,
                        day7_layout=day7_layout,
                    )
                    print(f"    ByteTrack → saved")
                except Exception as exc:
                    print(f"    ByteTrack ERROR: {exc}")

    def _write_block_b_csv_exports(self):
        """
        Optional Day 9 CSV recovery export.
        Writes tracker-level CSV from logs/block_b_results.json.
        """
        results_json = "logs/block_b_results.json"
        results_csv = "logs/block_b_results.csv"

        if not os.path.exists(results_json):
            print("[Day 9] block_b_results.json not found; skipping CSV export.")
            return

        with open(results_json, "r") as f:
            res = json.load(f)

        trackers = res.get("trackers", [])
        if not trackers:
            print("[Day 9] No tracker rows found; skipping CSV export.")
            return

        rows = []
        for t in trackers:
            rows.append(
                {
                    "tracker": t.get("tracker"),
                    "reid_model": t.get("reid_model"),
                    "mota_mean": t.get("mota_mean"),
                    "idf1_mean": t.get("idf1_mean"),
                    "idsw_total": t.get("idsw_total"),
                    "fragmentations_total": t.get("fragmentations_total"),
                    "fps_median": t.get("fps_median"),
                    "selected": t.get("selected"),
                }
            )

        headers = list(rows[0].keys())

        os.makedirs("logs", exist_ok=True)
        with open(results_csv, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for row in rows:
                f.write(",".join("" if row[h] is None else str(row[h]) for h in headers) + "\n")

        print(f"[Day 9] CSV summary written: {results_csv}")

    def run_block_b_analysis(self):
        """
        Day 9: Post-processing, error propagation, and tracker selection.

        Recovery policy for this repo state:
          - Pre-flight + evaluation remain test-split only.
          - Post-processing runs only on sequences that actually have raw
            trajectory JSONs for all trackers.
          - This avoids crashing on manifest sequences whose Day 7/8 outputs
            were never generated in the current repo state.

        Steps:
          1. Detect Day 7/8 layout + run pre-flight assertions
          2. Discover valid Day 9 sequences from available raw tracker outputs
          3. Fragment filter + track interpolation on those valid sequences
          4. Save fragment_filter_report.json and interpolation_report.json
          5. Error propagation correlation on test split only
          6. Tracker selection (select_tracker -> config_blockC.yaml)
          7. Finalise block_b_results_table.md
          8. Export block_b_results.csv
        """
        from src.error_propagation import (
            preflight_checks,
            run_error_propagation_analysis,
        )
        from src.post_processor import run_post_processing
        from src.tracker_selector import select_tracker

        # --- Load metadata ---
        with open("logs/metadata.json", "r") as f:
            metadata = json.load(f)

        train_seqs = metadata["split_seqs"]["train"]
        val_seqs = metadata["split_seqs"]["val"]
        test_seqs = metadata["split_seqs"]["test"]
        manifest_all_seqs = train_seqs + val_seqs + test_seqs

        trackers = ["sort", "deepsort", "bytetrack"]

        # --- Detect layout ---
        day7_layout = self._detect_day7_layout()
        print(f"Layout detected: {day7_layout}")

        # --- Step 1: Pre-flight checks on test split only ---
        print("\n=== Step 1: Pre-flight Checks ===")
        schema_entry_fields = preflight_checks(
            metadata,
            test_seqs,
            trackers,
            day7_layout,
        )

        # --- Step 2: discover which sequences can actually be processed ---
        seqs_available = self._discover_available_raw_sequences(trackers)

        # Build a split map so the repair step uses the correct detection cache
        seq_split_map = (
            {s: "train" for s in train_seqs}
            | {s: "val"   for s in val_seqs}
            | {s: "test"  for s in test_seqs}
        )

        # Repair: generate raw trajectories for manifest sequences that are missing
        missing_seqs = [s for s in manifest_all_seqs if s not in seqs_available]
        if missing_seqs:
            self._generate_missing_raw_trajectories(
                missing_seqs, seq_split_map, trackers, day7_layout
            )
            # Re-discover now that the repair has run
            seqs_available = self._discover_available_raw_sequences(trackers)

        seqs_for_day9 = [s for s in manifest_all_seqs if s in seqs_available]

        print("\nManifest sequences:", manifest_all_seqs)
        print("Available raw-output sequences:", seqs_available)
        print("Using Day 9 sequences:", seqs_for_day9)

        if not seqs_for_day9:
            raise FileNotFoundError(
                "No valid Day 9 sequences found: no overlap between manifest "
                "sequences and available raw trajectory files."
            )

        # --- Step 3: fragment filter + interpolation ---
        print("\n=== Step 2: Fragment Filter + Track Interpolation ===")
        fragment_report, interp_report = run_post_processing(
            trackers=trackers,
            seqs=seqs_for_day9,
            day7_layout=day7_layout,
            schema_entry_fields=schema_entry_fields,
            min_track_length=15,
            max_interp_gap=3,
        )

        # --- Step 4: save reports ---
        os.makedirs("logs/block_b", exist_ok=True)

        with open("logs/block_b/fragment_filter_report.json", "w") as f:
            json.dump(
                {
                    "min_track_length": 15,
                    "processed_sequences": seqs_for_day9,
                    **fragment_report,
                },
                f,
                indent=2,
            )

        with open("logs/block_b/interpolation_report.json", "w") as f:
            json.dump(
                {
                    "max_interp_gap": 3,
                    "processed_sequences": seqs_for_day9,
                    **interp_report,
                },
                f,
                indent=2,
            )

        print("Reports saved.")

        # --- Step 5: error propagation on test split only ---
        print("\n=== Step 3: Error Propagation Correlation (TEST split) ===")
        run_error_propagation_analysis(
            trackers=trackers,
            test_seqs=test_seqs,
            day7_layout=day7_layout,
            output_path="logs/block_b_error_propagation.json",
        )

        # --- Step 6: tracker selection ---
        print("\n=== Step 4: Tracker Selection ===")
        selected = select_tracker()
        print(f"SELECTED TRACKER: {selected.upper()}")

        # --- Step 7: finalise results table ---
        print("\n=== Step 5: Finalising Results Table ===")
        self._update_block_b_results_table(
            selected_tracker=selected,
            fragment_report=fragment_report,
            interp_report=interp_report,
            test_seqs=test_seqs,
        )

        # --- Step 8: CSV recovery export ---
        print("\n=== Step 6: CSV Export ===")
        self._write_block_b_csv_exports()

        print("\n=== Block B Analysis Complete ===")
        print("Canonical Block B outputs (_final.json) ready for Block C.")
        print(f"Tracker frozen: {selected} -> configs/config_blockC.yaml")

    def _update_block_b_results_table(
        self,
        selected_tracker,
        fragment_report,
        interp_report,
        test_seqs,
    ):
        """Write the finalised logs/block_b_results_table.md."""
        with open("logs/block_b_results.json", "r") as f:
            res = json.load(f)

        tracker_rows = {t["tracker"]: t for t in res["trackers"]}
        trackers = ["sort", "deepsort", "bytetrack"]

        def sum_field(report, field):
            out = {}
            for t in trackers:
                tracker_block = report.get(t, {})
                out[t] = sum(
                    tracker_block[s][field]
                    for s in test_seqs
                    if s in tracker_block and field in tracker_block[s]
                )
            return out

        tb = sum_field(fragment_report, "tracks_before")
        ta = sum_field(fragment_report, "tracks_after")
        td = sum_field(fragment_report, "discarded")
        fi = sum_field(interp_report, "frames_interpolated")

        lines = ["## Block B - Tracker Comparison Results\n\n"]
        lines.append("| Tracker | MOTA | IDF1 | IDSW | Frag | FPS (median) | Selected |\n")
        lines.append("|---|---|---|---|---|---|---|\n")

        for t_name in trackers:
            t = tracker_rows[t_name]
            sel = "YES" if t.get("selected") else "-"
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
                f"| {display} | {tb.get(t_name, '?')} | {ta.get(t_name, '?')} | "
                f"{td.get(t_name, '?')} | {fi.get(t_name, '?')} |\n"
            )

        lines.append(
            f"\n*(Values summed across {', '.join(test_seqs)} — test split)*\n\n"
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
        """
        Block C: Feature Extraction (Days 10-12).

        Day 10 delivers:
          - Generator-based clip loader (clip_feature_generator)
          - F1 features: vehicle_count, roi_occupancy
          - F2 features: vel_px_sec, vel_px_sec_smooth, is_interpolated
          - features.csv written per-clip to logs/block_c/
        """
        from src.feature_extractor import (
            clip_feature_generator,
            populate_config_blockc_tracker,
        )

        with open("configs/config_blockC.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        with open("logs/metadata.json", "r") as f:
            meta = json.load(f)

        cfg = populate_config_blockc_tracker(cfg)

        assert cfg.get("tracker") not in (None, "", "<SELECTED_TRACKER_FROM_DAY9>"), (
            "config_blockC.yaml 'tracker' field is not populated after "
            "populate_config_blockc_tracker(). Verify that "
            "logs/block_b_results.json['trackers'] has exactly one entry with "
            "'selected: true'."
        )

        all_seqs = (
            meta["split_seqs"]["train"]
            + meta["split_seqs"]["val"]
            + meta["split_seqs"]["test"]
        )

        completed = 0
        for completed_id in clip_feature_generator(all_seqs, meta, cfg):
            print(f"[Block C] Features written: {completed_id}")
            completed += 1

        print(f"[Block C] Done. {completed}/{len(all_seqs)} clips processed.")

    def run_block_c_day11(self):
        """
        Block C Day 11: F3 feature augmentation + MinMaxScaler fitting +
        scaled CSV writing.

        Prerequisites:
          - Day 10 features.csv files must exist in logs/block_c/ for all 10
            sequences.
          - config_blockC.yaml tracker field must already be populated (Day 10).

        Steps:
          1. F3 augmentation — load-augment-rewrite all 10 *_features.csv files
          2. Fit MinMaxScaler on train-split non-interpolated rows (Fix F13)
          3. Transform all splits and write *_features_scaled.csv (Fix F18)
        """
        from src.feature_extractor import (
            augment_features_with_f3,
            fit_and_save_scaler,
            apply_scaler_transform,
        )
        import pandas as pd

        with open("configs/config_blockC.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        meta = json.load(open("logs/metadata.json"))

        # Assertion: tracker must already be populated from Day 10
        assert cfg.get("tracker") not in (None, "", "<SELECTED_TRACKER_FROM_DAY9>"), (
            "config_blockC.yaml tracker is not populated. "
            "Confirm Day 10 completed successfully."
        )

        all_seqs = (
            meta["split_seqs"]["train"]
            + meta["split_seqs"]["val"]
            + meta["split_seqs"]["test"]
        )
        assert len(all_seqs) == 10, f"Expected 10 sequences, got {len(all_seqs)}"
        assert len(set(all_seqs)) == 10, "Duplicate sequence IDs found across splits!"

        # Step 1: F3 augmentation
        print("[Block C Day 11] Step 1: F3 augmentation for all sequences.")
        for seq_id in all_seqs:
            augment_features_with_f3(seq_id, meta, cfg)
            print(f"  [F3] {seq_id} — done")

        # Step 2: Fit scaler on train split
        print("[Block C Day 11] Step 2: Fitting MinMaxScaler on train split (Fix F13).")
        scaler = fit_and_save_scaler(cfg, meta)

        # Step 3: Transform all splits and save scaled CSVs
        print("[Block C Day 11] Step 3: Applying scaler transform + writing scaled CSVs.")
        features_dir = cfg.get("features_dir", "logs/block_c/")
        scaled_suffix = cfg.get("scaler", {}).get("scaled_suffix", "_features_scaled.csv")

        for split_name in ["train", "val", "test"]:
            for seq_id in meta["split_seqs"][split_name]:
                csv_path = os.path.join(features_dir, f"{seq_id}_features.csv")
                if not os.path.exists(csv_path):
                    logger.warning(
                        f"[{seq_id}] features.csv missing — skipping transform."
                    )
                    continue
                df = pd.read_csv(csv_path)
                df_scaled = apply_scaler_transform(
                    df, scaler, cfg, label=f"{split_name}/{seq_id}"
                )
                out_path = os.path.join(features_dir, f"{seq_id}{scaled_suffix}")
                df_scaled.to_csv(out_path, index=False)
                assert os.path.exists(out_path), (
                    f"[{seq_id}] scaled CSV not written at {out_path}"
                )
                # Verify raw file was NOT overwritten
                assert os.path.exists(csv_path), (
                    f"[{seq_id}] Raw features.csv missing after scaling step!"
                )
                print(f"  [scaled] {seq_id} — {out_path}")

        print(
            f"[Block C Day 11] Complete. "
            f"F3 + scaler fit + scaled CSVs for {len(all_seqs)} sequences."
        )

    def run_block_c_day12(self):
        """
        Block C Day 12: Feature comparison via OC-SVM AUROC (Fix F02, Fix F04).

        Prerequisites:
          - Day 11 *_features_scaled.csv files must exist for all 10 sequences.
          - models/minmax_scaler.pkl must exist (Day 11).
          - config_blockC.yaml f3 fields must be true (Day 11).
          - config_blockC.yaml tracker must be populated (Day 10).

        Outputs:
          - logs/block_c/block_c_auroc_results.csv
          - logs/block_c/block_c_results_table.csv
          - logs/block_c/ocsvm_trained_best.pkl
          - logs/block_c/block_c_day12_metadata.json
          - configs/config_blockC.yaml updated: best_feature_set + block_c_completed
        """
        import json
        import os

        import yaml

        from src.block_c_evaluator import run_auroc_comparison

        cfg_path  = "configs/config_blockC.yaml"
        meta_path = "logs/metadata.json"

        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        with open(meta_path) as fh:
            meta = json.load(fh)

        # ── Prerequisite assertions ───────────────────────────────────────────
        assert cfg.get("tracker") not in (None, "", "<SELECTED_TRACKER_FROM_DAY9>"), (
            "config_blockC.yaml tracker is not populated. "
            "Confirm Day 10 completed successfully."
        )

        f3_cfg = cfg.get("f3", {})
        assert f3_cfg.get("inter_vehicle_dist_norm") is True, (
            "config_blockC.yaml f3.inter_vehicle_dist_norm is not True. "
            "Confirm Day 11 ran successfully."
        )

        scaler_path = cfg.get("scaler", {}).get("path", "models/minmax_scaler.pkl")
        assert os.path.exists(scaler_path), (
            f"Scaler not found at {scaler_path}. "
            "Day 11 must complete before Day 12."
        )

        # ── Verify all 10 scaled CSVs exist ──────────────────────────────────
        features_dir   = cfg.get("features_dir", "logs/block_c/")
        scaled_suffix  = cfg.get("scaler", {}).get("scaled_suffix", "_features_scaled.csv")
        all_seqs = (
            meta["split_seqs"]["train"]
            + meta["split_seqs"]["val"]
            + meta["split_seqs"]["test"]
        )
        assert len(all_seqs) == 10 and len(set(all_seqs)) == 10, (
            f"Expected exactly 10 unique sequences, got {len(all_seqs)} "
            f"({len(set(all_seqs))} unique)."
        )
        missing_scaled = [
            s for s in all_seqs
            if not os.path.exists(os.path.join(features_dir, f"{s}{scaled_suffix}"))
        ]
        if missing_scaled:
            raise FileNotFoundError(
                f"Scaled CSVs missing for: {missing_scaled}. "
                "Run Day 11 notebook first."
            )

        # ── Run evaluation ────────────────────────────────────────────────────
        df_results = run_auroc_comparison(cfg, meta)
        logger.info("Block C Day 12 complete.")
        return df_results

    def run_block_d(self, events_path: str = "data/ai_city/events.json"):
        """
        Block D — Anomaly Detection on AI City Track 4.

        Day 13: Rule-based baseline (Z-score on X_raw; IQR on X_scaled).
                Calls: event_loader → event_slicer → feature_extractor (F2_only)
                       → rule_based_detector → evaluator
                Output: logs/block_d/block_d_results.csv (Day 13 rows filled;
                        OC-SVM and Isolation Forest rows as Day 14 placeholders)

        Day 14 will extend: ocsvm_detector (pre-trained Block C model),
                            isolation_forest_detector (fit on UA-DETRAC normal)

        Fix F25: stub updated on Day 13 — no longer a pass-through.
        """
        # ── Import here to avoid circular dependency at module load time ──────
        from src.block_d_rule_based import (  # noqa: PLC0415
            run_rule_based_baseline,
            audit_config,
        )
        import yaml  # noqa: PLC0415

        # Load Block C config for speed_window_sec and other shared params
        cfg_c_path = "configs/config_blockC.yaml"
        if not os.path.exists(cfg_c_path):
            raise FileNotFoundError(f"Block C config not found: {cfg_c_path}")
        with open(cfg_c_path, encoding="utf-8") as f:
            cfg_c = yaml.safe_load(f)

        print("[PipelineController] Block D: Running Day 13 rule-based baseline.")
        print(f"[PipelineController]   Config : configs/config_blockD.yaml")
        print(f"[PipelineController]   Events : {events_path}")

        results_df = run_rule_based_baseline(
            cfg_d=self.cfg,
            cfg_c=cfg_c,
            events_path=events_path,
            scaler_path=self.cfg.get("scaler", "models/minmax_scaler.pkl"),
            verbose=True,
        )

        print("[PipelineController] Block D Day 13 complete.")
        print(f"[PipelineController]   Results saved to: logs/block_d/block_d_results.csv")
        return results_df


if __name__ == "__main__":
    test_cfg = "configs/config_blockA.yaml"
    if os.path.exists(test_cfg):
        controller = PipelineController(test_cfg)
        print("[PipelineController] Smoke test passed.")
    else:
        print(f"[PipelineController] Config not found at {test_cfg}.")