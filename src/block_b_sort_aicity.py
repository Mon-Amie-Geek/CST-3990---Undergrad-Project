"""
block_b_sort_aicity.py

Minimal AI City SORT trajectory generator for Block D support.

Purpose:
  - Reuse the existing SORT tracker implementation
  - Reuse Day 9 fragment filtering + interpolation
  - Write raw / filtered / final SORT trajectories for AI City videos only
  - Avoid changing the original UA-DETRAC Block B pipeline
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from src.block_d_rule_based import (
    build_video_index,
    get_actual_fps,
    load_events,
    resolve_video_path,
)
from src.post_processor import apply_fragment_filter, apply_interpolation
from src.sort_tracker import SORTTracker
from src.seed_control import set_all_seeds


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = REPO_ROOT / "logs" / "block_b" / "trajectories"
MODEL_PATH = REPO_ROOT / "models" / "best.pt"
FRAME_SIZE = 640
FRAME_DIAGONAL = float(math.sqrt(FRAME_SIZE ** 2 + FRAME_SIZE ** 2))


def _load_configs() -> tuple[dict, dict]:
    """Load frozen detector + tracker config."""
    with open(REPO_ROOT / "configs" / "config_blockA.yaml", encoding="utf-8") as f:
        cfg_a = yaml.safe_load(f)
    with open(REPO_ROOT / "configs" / "config_blockB.yaml", encoding="utf-8") as f:
        cfg_b = yaml.safe_load(f)
    return cfg_a, cfg_b


def _compute_velocity_norm(prev_cx, prev_cy, curr_cx, curr_cy, fps) -> float | None:
    """Match the Block B velocity_norm contract in 640x640 image space."""
    if prev_cx is None:
        return None
    dist_px = math.sqrt((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2)
    vel_px_sec = dist_px * fps
    return vel_px_sec / FRAME_DIAGONAL


def _load_model(model_path: Path = MODEL_PATH) -> YOLO:
    """Load the frozen YOLOv8n detector used by the project."""
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found at {model_path}")
    model = YOLO(str(model_path))
    model.to("cpu")
    return model


def _run_sort_on_video(
    video_path: Path,
    video_id: str,
    model: YOLO,
    conf_thresh: float,
    nms_thresh: float,
    sort_cfg: dict,
) -> dict:
    """
    Run detector + SORT on one AI City clip and return raw trajectory payload.
    """
    fps = get_actual_fps(video_path)
    warmup_boundary = max(10, int(0.4 * fps))

    tracker = SORTTracker(
        max_age=sort_cfg["max_age"],
        min_hits=sort_cfg["min_hits"],
        iou_threshold=sort_cfg["iou_threshold"],
        conf_thresh=conf_thresh,
    )
    tracker.reset()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    raw_trajectories = defaultdict(list)
    prev_centroids = {}
    frame_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        frame_640 = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
        results = model(
            frame_640,
            conf=conf_thresh,
            iou=nms_thresh,
            verbose=False,
            device="cpu",
        )

        dets = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].item())
                dets.append([float(x1), float(y1), float(x2), float(y2), conf])

        dets_arr = np.array(dets, dtype=float) if dets else np.empty((0, 5), dtype=float)
        tracked = tracker.update(dets_arr, frame_idx)

        for row in tracked:
            x1, y1, x2, y2, tid, conf = row
            tid = int(tid)
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

            prev = prev_centroids.get(tid)
            vel_n = _compute_velocity_norm(
                prev[0] if prev else None,
                prev[1] if prev else None,
                cx,
                cy,
                fps,
            )
            prev_centroids[tid] = (cx, cy)

            raw_trajectories[str(tid)].append(
                {
                    "frame_idx": int(frame_idx),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "cx": cx,
                    "cy": cy,
                    "velocity_norm": float(vel_n) if vel_n is not None else None,
                    "conf": float(conf),
                }
            )

    cap.release()

    return {
        "seq_id": str(video_id),
        "tracker": "sort",
        "detector": "yolov8n",
        "reid_model": "none",
        "warmup_boundary": int(warmup_boundary),
        "cache_frame_index_base": 0,
        "fps": float(fps),
        "frame_width": FRAME_SIZE,
        "frame_height": FRAME_SIZE,
        "frame_diagonal": FRAME_DIAGONAL,
        "min_track_length": 15,
        "max_interp_gap": 3,
        "fragment_filter_applied": False,
        "interpolation_applied": False,
        "tracks": dict(raw_trajectories),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def generate_ai_city_sort_trajectories(
    events_path: str | Path = REPO_ROOT / "data" / "ai_city" / "events.json",
    video_ids: List[str] | None = None,
) -> Dict[str, dict]:
    """
    Generate raw / filtered / final SORT trajectories for AI City videos referenced
    by events.json.

    Output files:
      logs/block_b/trajectories/{video_id}_sort.json
      logs/block_b/trajectories/{video_id}_sort_filtered.json
      logs/block_b/trajectories/{video_id}_sort_final.json
    """
    set_all_seeds()
    cfg_a, cfg_b = _load_configs()
    model = _load_model()
    events = load_events(events_path)
    index = build_video_index()

    wanted_ids = {str(e["video_id"]) for e in events}
    if video_ids is not None:
        wanted_ids &= {str(v) for v in video_ids}
    wanted_ids = set(sorted(wanted_ids))

    results = {}
    for video_id in sorted(wanted_ids, key=int):
        event = next(e for e in events if str(e["video_id"]) == video_id)
        video_path = resolve_video_path(event["video_file"], index)
        if video_path is None:
            print(f"[AI City SORT] SKIP video_id={video_id}: video file not found for {event['video_file']}")
            continue

        print(f"[AI City SORT] Processing video_id={video_id} -> {video_path.name}")
        raw = _run_sort_on_video(
            video_path=video_path,
            video_id=video_id,
            model=model,
            conf_thresh=float(cfg_a["conf_thresh"]),
            nms_thresh=float(cfg_a["nms_thresh"]),
            sort_cfg=cfg_b["sort"],
        )

        raw_path = TRAJ_DIR / f"{video_id}_sort.json"
        filtered_path = TRAJ_DIR / f"{video_id}_sort_filtered.json"
        final_path = TRAJ_DIR / f"{video_id}_sort_final.json"

        filtered = apply_fragment_filter(raw, tracker_name="sort", min_length=cfg_b["min_track_length"])
        final = apply_interpolation(filtered, max_gap=cfg_b["max_interp_gap"])

        _write_json(raw_path, raw)
        _write_json(filtered_path, filtered)
        _write_json(final_path, final)

        results[video_id] = {
            "video_file": event["video_file"],
            "raw_path": str(raw_path),
            "filtered_path": str(filtered_path),
            "final_path": str(final_path),
            "n_raw_tracks": len(raw["tracks"]),
            "n_final_tracks": len(final["tracks"]),
        }
        print(
            f"[AI City SORT] Saved {raw_path.name}, {filtered_path.name}, {final_path.name} "
            f"(raw_tracks={len(raw['tracks'])}, final_tracks={len(final['tracks'])})"
        )

    return results


if __name__ == "__main__":
    generate_ai_city_sort_trajectories()
