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
        frame (np.ndarray): Input frame
        tracks (list): List of track dicts with 'track_id', 'bbox', 'centroid'
        anomalous_ids (set): Set of track IDs flagged as anomalous
        show_boxes (bool): Whether to draw bounding boxes
        show_ids (bool): Whether to draw track IDs
        show_speed (bool): Whether to draw speed labels
        speed_data (dict): Dict mapping track_id to speed in px/sec
        
    Returns:
        np.ndarray: Annotated frame
    """
    if not show_boxes:
        return frame
    
    frame_copy = frame.copy()
    
    for track in tracks:
        track_id = track.get("track_id")
        bbox = track.get("bbox")
        
        if not bbox:
            continue
        
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Determine box color
        if track_id in anomalous_ids:
            color = (0, 0, 255)  # Red for anomalous
        else:
            color = (0, 255, 0)  # Green for normal
        
        # Draw bounding box
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID above box
        if show_ids and track_id is not None:
            text_x = x1
            text_y = max(y1 - 10, 20)
            cv2.putText(
                frame_copy,
                f"ID: {track_id}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White
                2
            )
        
        # Draw speed below track ID
        if show_speed and speed_data and track_id in speed_data:
            speed = speed_data[track_id]
            text_x = x1
            text_y = max(y1 - 25, 20) if show_ids else max(y1 - 10, 20)
            cv2.putText(
                frame_copy,
                f"{speed:.1f} px/s",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),  # Yellow
                1
            )
    
    return frame_copy


def draw_anomaly_banner(frame, message):
    """
    Draw a red banner with anomaly message at the top of the frame.
    
    Args:
        frame (np.ndarray): Input frame
        message (str): Message to display
        
    Returns:
        np.ndarray: Frame with banner
    """
    frame_copy = frame.copy()
    height = frame.shape[0]
    
    # Draw semi-transparent red banner at top
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.5, frame_copy, 0.5, 0, frame_copy)
    
    # Draw white text
    cv2.putText(
        frame_copy,
        message,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),  # White
        2
    )
    
    return frame_copy
