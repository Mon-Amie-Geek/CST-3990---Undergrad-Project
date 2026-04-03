"""
video_utils.py

Utilities for video loading, metadata extraction, and file management.
Student: MANJOO Ameera Najla | M01014463
"""

import cv2
import tempfile
import os


def get_video_metadata(video_path):
    """
    Extract metadata from a video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Contains fps, frame_count, width, height, duration_sec
        
    Raises:
        ValueError: If video cannot be opened
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    duration_sec = frame_count / fps if fps > 0 else 0
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec
    }


def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file to a temporary location.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Path to the saved temporary file
    """
    with tempfile.NamedTemporaryFile(
        delete=False, 
        suffix=".mp4"
    ) as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name
