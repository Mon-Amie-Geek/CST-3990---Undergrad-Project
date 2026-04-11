"""
video_utils.py

Utilities for video loading, metadata extraction, and file management.
Student: MANJOO Ameera Najla | M01014463
"""

from pathlib import Path
from uuid import uuid4

import cv2


_UTILS_DIR = Path(__file__).resolve().parent
_APP_DIR = _UTILS_DIR.parent
_UPLOAD_DIR = _APP_DIR / ".streamlit_uploads"


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
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    original_name = getattr(uploaded_file, "name", "uploaded_video.mp4")
    suffix = Path(original_name).suffix or ".mp4"
    safe_stem = Path(original_name).stem or "uploaded_video"
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in safe_stem)
    output_path = _UPLOAD_DIR / f"{safe_stem}_{uuid4().hex}{suffix}"

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    if hasattr(uploaded_file, "getbuffer"):
        file_bytes = uploaded_file.getbuffer()
    else:
        file_bytes = uploaded_file.read()

    with open(output_path, "wb") as output_file:
        output_file.write(file_bytes)

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    return str(output_path)
