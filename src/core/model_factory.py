"""
Model factory for creating detector instances.

This module provides factory classes and functions for creating
and managing RecyclingDetector instances with proper error handling.
"""

import time
from pathlib import Path
from typing import Optional, Union

from config.settings import get_config
from config.logging_config import get_logger
from src.core.detector import RecyclingDetector

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