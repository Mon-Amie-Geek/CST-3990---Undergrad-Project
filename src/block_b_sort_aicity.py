"""
block_b_sort_aicity.py

Minimal AI City SORT trajectory generator for Block D support.

Purpose:
  - Reuse the existing SORT tracker implementation
  - Reuse Day 9 fragment filtering + interpolation
  - Write raw / filtered / final SORT trajectories for AI City videos only
  - Avoid changing the original UA-DETRAC Block B pipeline
  - Use GPU when available
  - Process only anomaly windows (+ context padding) for speed
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
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

# Match Block D context padding
CONTEXT_SEC = 5.0


def _load_configs() -> tuple[dict, dict]:
    """Load frozen detector + tracker config."""
    with open(REPO_ROOT / "configs" / "config_blockA.yaml", encoding="utf-8") as f:
        cfg_a = yaml.safe_load(f)
    with open(REPO_ROOT / "configs" / "config_blockB.yaml", encoding="utf-8") as f:
        cfg_b = yaml.safe_load(f)
    return cfg_a, cfg_b


def _compute_velocity_norm(
    prev_cx: Optional[float],
    prev_cy: Optional[float],
    curr_cx: float,
    curr_cy: float,
    fps: float,
) -> float | None:
    """Match the Block B velocity_norm contract in 640x640 image space."""
    if prev_cx is None or prev_cy is None:
        return None
    dist_px = math.sqrt((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2)
    vel_px_sec = dist_px * fps
    return vel_px_sec / FRAME_DIAGONAL


def _get_device() -> str | int:
    """Use GPU when available, otherwise CPU."""
    return 0 if torch.cuda.is_available() else "cpu"


def _load_model(model_path: Path = MODEL_PATH) -> YOLO:
    """Load the frozen YOLOv8n detector used by the project."""
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found at {model_path}")
    model = YOLO(str(model_path))
    return model


def _run_sort_on_video(
    video_path: Path,
    video_id: str,
    model: YOLO,
    conf_thresh: float,
    nms_thresh: float,
    sort_cfg: dict,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> dict:
    """
    Run detector + SORT on one AI City clip and return raw trajectory payload.

    For speed, this can operate only on [start_frame, end_frame] rather than
    the entire video.
    """
    fps = get_actual_fps(video_path)
    warmup_boundary = max(10, int(0.4 * fps))
    device = _get_device()

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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = max(0, total_frames - 1)

    start_frame = max(0, int(start_frame))
    end_frame = int(end_frame)
    if total_frames > 0:
        end_frame = min(end_frame, total_frames - 1)

    if end_frame < start_frame:
        cap.release()
        raise ValueError(
            f"Invalid frame range for {video_path.name}: "
            f"start_frame={start_frame}, end_frame={end_frame}"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    raw_trajectories = defaultdict(list)
    prev_centroids: Dict[int, tuple[float, float]] = {}
    frame_idx = start_frame - 1
    processed_frames = 0

    print(
        f"[AI City SORT] video_id={video_id} "
        f"frames {start_frame}..{end_frame} "
        f"({end_frame - start_frame + 1} frames) "
        f"device={device}"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx > end_frame:
            break

        frame_640 = cv2.resize(
            frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR
        )

        results = model(
            frame_640,
            conf=conf_thresh,
            iou=nms_thresh,
            verbose=False,
            device=device,
        )

        dets = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
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

        processed_frames += 1
        if processed_frames % 200 == 0:
            print(
                f"[AI City SORT] video_id={video_id} "
                f"processed {processed_frames} frame(s)..."
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
        "min_track_length": int(sort_cfg.get("min_track_length", 15)),
        "max_interp_gap": int(sort_cfg.get("max_interp_gap", 3)),
        "fragment_filter_applied": False,
        "interpolation_applied": False,
        "processed_start_frame": int(start_frame),
        "processed_end_frame": int(end_frame),
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

    results: Dict[str, dict] = {}

    for video_id in sorted(wanted_ids, key=int):
        event = next(e for e in events if str(e["video_id"]) == video_id)
        video_path = resolve_video_path(event["video_file"], index)

        if video_path is None:
            print(
                f"[AI City SORT] SKIP video_id={video_id}: "
                f"video file not found for {event['video_file']}"
            )
            continue

        fps = get_actual_fps(video_path)
        context_frames = int(CONTEXT_SEC * fps)
        start_frame = max(0, int(event["anomaly_start_frame"]) - context_frames)
        end_frame = int(event["anomaly_end_frame"]) + context_frames

        print(
            f"[AI City SORT] Processing video_id={video_id} -> {video_path.name} "
            f"(anomaly {event['anomaly_start_frame']}..{event['anomaly_end_frame']}, "
            f"context={context_frames} frames)"
        )

        raw = _run_sort_on_video(
            video_path=video_path,
            video_id=video_id,
            model=model,
            conf_thresh=float(cfg_a["conf_thresh"]),
            nms_thresh=float(cfg_a["nms_thresh"]),
            sort_cfg={
                **cfg_b["sort"],
                "min_track_length": int(cfg_b["min_track_length"]),
                "max_interp_gap": int(cfg_b["max_interp_gap"]),
            },
            start_frame=start_frame,
            end_frame=end_frame,
        )

        raw_path = TRAJ_DIR / f"{video_id}_sort.json"
        filtered_path = TRAJ_DIR / f"{video_id}_sort_filtered.json"
        final_path = TRAJ_DIR / f"{video_id}_sort_final.json"

        filtered = apply_fragment_filter(
            raw,
            tracker_name="sort",
            min_length=int(cfg_b["min_track_length"]),
        )
        final = apply_interpolation(
            filtered,
            max_gap=int(cfg_b["max_interp_gap"]),
        )

        _write_json(raw_path, raw)
        _write_json(filtered_path, filtered)
        _write_json(final_path, final)

        results[video_id] = {
            "video_file": event["video_file"],
            "raw_path": str(raw_path),
            "filtered_path": str(filtered_path),
            "final_path": str(final_path),
            "processed_start_frame": int(start_frame),
            "processed_end_frame": int(end_frame),
            "n_raw_tracks": len(raw["tracks"]),
            "n_filtered_tracks": len(filtered["tracks"]),
            "n_final_tracks": len(final["tracks"]),
        }

        print(
            f"[AI City SORT] Saved {raw_path.name}, {filtered_path.name}, {final_path.name} "
            f"(raw_tracks={len(raw['tracks'])}, "
            f"filtered_tracks={len(filtered['tracks'])}, "
            f"final_tracks={len(final['tracks'])})"
        )

    return results


if __name__ == "__main__":
    generate_ai_city_sort_trajectories()