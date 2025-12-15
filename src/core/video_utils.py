"""
Video processing utility functions.

This module provides utility functions for creating processors
and batch processing video files.
"""

from pathlib import Path
from typing import Union, Dict, Any

from src.core.video_processor import VideoProcessor, WebcamProcessor, FileProcessor
from src.core.detector import RecyclingDetector
from src.core.model_factory import load_detector
from src.core.counter import RecyclingCounter, create_counter

from config.logging_config import get_logger

logger = get_logger("main")


def create_processor(
    detector: RecyclingDetector,
    counter: RecyclingCounter,
    source: Union[str, int, Path],
) -> VideoProcessor:
    """
    Create appropriate video processor based on source type.

    Args:
        detector: Detection engine
        counter: Counting system
        source: Video source (file path or camera index)

    Returns:
        VideoProcessor instance
    """
    if isinstance(source, int):
        return WebcamProcessor(detector, counter, source)
    else:
        return FileProcessor(detector, counter, source)