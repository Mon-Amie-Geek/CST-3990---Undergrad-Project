"""
drawing_utils.py

Utilities for drawing bounding boxes, track IDs, and anomaly banners on frames.
Student: MANJOO Ameera Najla | M01014463
"""

import cv2
import numpy as np


def draw_tracks(frame, tracks, anomalous_ids,
                show_boxes=True, show_ids=True, show_speed=True,
                speed_data=None):
    """
    Draw bounding boxes and metadata on a frame.

    Args:
        frame (np.ndarray): Input BGR frame
        tracks (list): List of track dicts with 'track_id', 'bbox', 'centroid'
        anomalous_ids (set): Set of track IDs flagged as anomalous
        show_boxes (bool): Whether to draw bounding boxes
        show_ids (bool): Whether to draw track IDs
        show_speed (bool): Whether to draw speed labels
        speed_data (dict): Dict mapping track_id to speed in px/sec

    Returns:
        np.ndarray: Annotated frame (copy)
    """
    frame_copy = frame.copy()

    if not show_boxes or not tracks:
        return frame_copy

    h, w = frame_copy.shape[:2]

    for track in tracks:
        track_id = track.get("track_id")
        bbox     = track.get("bbox")

        if not bbox or len(bbox) < 4:
            continue

        try:
            x1, y1, x2, y2 = (
                int(max(0, bbox[0])),
                int(max(0, bbox[1])),
                int(min(w - 1, bbox[2])),
                int(min(h - 1, bbox[3])),
            )
        except (ValueError, TypeError):
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        # Box colour: red for anomalous, green for normal
        color = (0, 0, 255) if (anomalous_ids and track_id in anomalous_ids) else (0, 200, 0)

        # Bounding box
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

        # Labels — stack above the box; clamp so they stay in frame
        label_y = y1 - 6
        label_lines = []

        if show_ids and track_id is not None:
            label_lines.append(f"ID:{track_id}")
        if show_speed and speed_data and track_id in speed_data:
            label_lines.append(f"{speed_data[track_id]:.0f}px/s")

        for i, text in enumerate(label_lines):
            ty = max(14, label_y - (len(label_lines) - 1 - i) * 16)
            # Small dark background for legibility
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_copy, (x1, ty - th - 2), (x1 + tw + 2, ty + 2),
                          (0, 0, 0), -1)
            cv2.putText(
                frame_copy, text,
                (x1 + 1, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

    return frame_copy


def draw_anomaly_banner(frame, message):
    """
    Draw a semi-transparent red banner with anomaly message at the top of the frame.

    Args:
        frame (np.ndarray): Input BGR frame
        message (str): Message to display (truncated if too long)

    Returns:
        np.ndarray: Frame with banner (copy)
    """
    frame_copy = frame.copy()
    fw = frame.shape[1]

    banner_h = 54
    overlay  = frame_copy.copy()
    cv2.rectangle(overlay, (0, 0), (fw, banner_h), (20, 20, 220), -1)
    cv2.addWeighted(overlay, 0.60, frame_copy, 0.40, 0, frame_copy)

    # Truncate message to fit frame width
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness  = 2
    max_chars  = max(10, fw // 14)
    if len(message) > max_chars:
        message = message[:max_chars - 3] + "..."

    cv2.putText(
        frame_copy, message,
        (12, 36),
        font, font_scale,
        (255, 255, 255), thickness, cv2.LINE_AA,
    )

    return frame_copy
