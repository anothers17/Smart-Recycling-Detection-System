"""
Video output writer for Smart Recycling Detection System.

This module provides functionality for saving processed video output
to files with proper codec handling and error management.
"""

import cv2
from pathlib import Path
from typing import Optional

from config.logging_config import get_logger

logger = get_logger("output")


class VideoOutputWriter:
    """Handles video output writing with proper codec management."""

    def __init__(self, output_path: str, cap: cv2.VideoCapture):
        """
        Initialize video output writer.

        Args:
            output_path: Path to output video file
            cap: OpenCV VideoCapture object for getting video properties
        """
        self.output_path = Path(output_path)
        self.cap = cap
        self.writer: Optional[cv2.VideoWriter] = None
        self.is_initialized = False

    def initialize(self) -> bool:
        """
        Initialize the video writer.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.cap:
                logger.error("No video capture object provided")
                return False

            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Validate properties
            if width <= 0 or height <= 0:
                logger.error(f"Invalid video dimensions: {width}x{height}")
                return False

            if fps <= 0:
                fps = 30.0  # Default fallback
                logger.warning("Invalid FPS detected, using default 30 FPS")

            # Ensure output directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            self.writer = cv2.VideoWriter(
                str(self.output_path), fourcc, fps, (width, height)
            )

            if self.writer.isOpened():
                self.is_initialized = True
                logger.info(f"Output writer initialized: {self.output_path}")
                return True
            else:
                logger.error("Failed to initialize output writer")
                return False

        except Exception as e:
            logger.error(f"Error initializing output writer: {e}")
            return False

    def write_frame(self, frame) -> bool:
        """
        Write a frame to the output video.

        Args:
            frame: Frame to write

        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized or not self.writer:
            logger.error("Output writer not initialized")
            return False

        try:
            self.writer.write(frame)
            return True
        except Exception as e:
            logger.error(f"Error writing frame: {e}")
            return False

    def close(self):
        """Close the video writer and release resources."""
        try:
            if self.writer:
                self.writer.release()
                self.writer = None
                self.is_initialized = False
                logger.info("Output writer closed")
        except Exception as e:
            logger.error(f"Error closing output writer: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()