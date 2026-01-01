"""
Model factory for creating detector instances.

This module provides factory classes and functions for creating
and managing RecyclingDetector instances with proper error handling.
"""

import time
from pathlib import Path
from typing import Optional, Union
import numpy as np

from config.settings import get_config
from config.logging_config import get_logger
from .detector import RecyclingDetector, DetectionResult

logger = get_logger("detection")


class DetectorFactory:
    """Factory for creating detector instances."""

    @staticmethod
    def create_detector(
        model_path: Union[str, Path], device: Optional[str] = None, max_retries: int = 3
    ) -> RecyclingDetector:
        """
        Create a detector instance with robust error handling.

        Args:
            model_path: Path to model file
            device: Device to use for inference
            max_retries: Maximum loading attempts

        Returns:
            RecyclingDetector instance

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model format is unsupported
            RuntimeError: If model loading fails after all retries
        """
        detector = RecyclingDetector()

        if device:
            detector.set_device(device)

        try:
            success = detector.load_model(model_path, max_retries)
            if not success:
                raise RuntimeError(f"Model loading returned False for {model_path}")
        except (FileNotFoundError, ValueError):
            # Re-raise specific exceptions
            raise
        except Exception as e:
            logger.error(f"Failed to create detector: {e}")
            raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e

        # Warm up the model
        try:
            detector.warm_up()
        except Exception as e:
            logger.warning(f"Model warm-up failed, but proceeding: {e}")

        return detector

    @staticmethod
    def create_from_config(config_path: Optional[str] = None) -> RecyclingDetector:
        """
        Create detector from configuration.

        Args:
            config_path: Path to configuration file

        Returns:
            RecyclingDetector instance
        """
        config = get_config()

        # Find model file
        model_files = list(config.paths.models_dir.glob("*.pt"))

        if not model_files:
            raise FileNotFoundError(
                f"No model files found in {config.paths.models_dir}"
            )

        # Use first model file found (or best.pt if available)
        model_path = None
        for model_file in model_files:
            if model_file.name == "best.pt":
                model_path = model_file
                break

        if model_path is None:
            model_path = model_files[0]

        return DetectorFactory.create_detector(model_path, config.detection.device)


# Utility functions
def load_detector(
    model_path: Union[str, Path], device: Optional[str] = None
) -> RecyclingDetector:
    """
    Convenience function to load a detector.

    Args:
        model_path: Path to model file
        device: Device to use

    Returns:
        Loaded detector instance
    """
    return DetectorFactory.create_detector(model_path, device)


def detect_image(
    image: np.ndarray, model_path: Union[str, Path], confidence_threshold: float = 0.5
) -> DetectionResult:
    """
    Convenience function to perform detection on a single image.

    Args:
        image: Input image
        model_path: Path to model
        confidence_threshold: Confidence threshold

    Returns:
        DetectionResult instance
    """
    with RecyclingDetector() as detector:
        detector.load_model(model_path)
        return detector.detect(image, confidence_threshold=confidence_threshold)


def benchmark_detector(
    detector: RecyclingDetector, num_iterations: int = 10, image_size: tuple = (640, 640)
) -> dict:
    """
    Benchmark detector performance.

    Args:
        detector: Detector instance
        num_iterations: Number of iterations
        image_size: Size of dummy image to use

    Returns:
        Dictionary with benchmark results
    """
    image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
    total_time = 0
    total_detections = 0

    # Warm up
    detector.warm_up(num_iterations=2)

    for _ in range(num_iterations):
        result = detector.detect(image)
        total_time += result.processing_time
        total_detections += len(result.detections)

    avg_processing_time = total_time / num_iterations
    avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0

    return {
        "avg_fps": avg_fps,
        "total_time": total_time,
        "num_iterations": num_iterations,
        "avg_processing_time": avg_processing_time,
        "avg_detections": total_detections / num_iterations,
    }