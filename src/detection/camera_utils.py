"""
Camera utilities for Smart Recycling Detection System.

This module provides utilities for detecting and managing camera devices.
"""

import cv2
from typing import List

from config.logging_config import get_logger

logger = get_logger("camera")


def detect_available_cameras(max_cameras: int = 10) -> List[int]:
    """
    Detect available camera indices.

    Args:
        max_cameras: Maximum number of camera indices to check

    Returns:
        List of available camera indices
    """
    available_cameras = []
    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        except Exception:
            continue

    logger.info(f"Detected {len(available_cameras)} available cameras: {available_cameras}")
    return available_cameras


def get_camera_info(camera_index: int) -> dict:
    """
    Get information about a specific camera.

    Args:
        camera_index: Camera device index

    Returns:
        Dictionary with camera information
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return {"available": False}

        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        return {
            "available": True,
            "index": camera_index,
            "width": width,
            "height": height,
            "fps": fps,
        }

    except Exception as e:
        logger.error(f"Error getting camera info for index {camera_index}: {e}")
        return {"available": False, "error": str(e)}