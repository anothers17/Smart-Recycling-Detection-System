"""
Object detection engine for Smart Recycling Detection System.

This module provides a comprehensive detection engine using YOLOv8
with enhanced error handling, performance monitoring, and configuration management.
"""

import time
import torch
import numpy as np
import psutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from ultralytics import YOLO

from config.settings import get_config
from config.logging_config import get_logger, log_performance
from src.core.performance_monitor import ModelPerformanceMonitor

logger = get_logger("detection")


@dataclass
class Detection:
    """Individual detection result."""

    xyxy: List[float]  # Bounding box coordinates [x1, y1, x2, y2]
    confidence: float  # Detection confidence (0-1)
    class_id: int  # Class ID
    class_name: str  # Class name
    center: Tuple[float, float] = field(init=False)  # Center point
    area: float = field(init=False)  # Bounding box area

    def __post_init__(self):
        """Calculate derived properties."""
        x1, y1, x2, y2 = self.xyxy
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.area = (x2 - x1) * (y2 - y1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "xyxy": self.xyxy,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "center": self.center,
            "area": self.area,
        }


@dataclass
class DetectionResult:
    """Complete detection result for an image."""

    detections: List[Detection]
    processing_time: float
    image_shape: Tuple[int, int, int]
    model_input_shape: Tuple[int, int]
    timestamp: float = field(default_factory=time.time)

    def __len__(self) -> int:
        """Number of detections."""
        return len(self.detections)

    def filter_by_class(self, class_names: List[str]) -> "DetectionResult":
        """Filter detections by class names."""
        filtered_detections = [
            det for det in self.detections if det.class_name in class_names
        ]

        return DetectionResult(
            detections=filtered_detections,
            processing_time=self.processing_time,
            image_shape=self.image_shape,
            model_input_shape=self.model_input_shape,
            timestamp=self.timestamp,
        )

    def filter_by_confidence(self, min_confidence: float) -> "DetectionResult":
        """Filter detections by minimum confidence."""
        filtered_detections = [
            det for det in self.detections if det.confidence >= min_confidence
        ]

        return DetectionResult(
            detections=filtered_detections,
            processing_time=self.processing_time,
            image_shape=self.image_shape,
            model_input_shape=self.model_input_shape,
            timestamp=self.timestamp,
        )

    def get_class_counts(self) -> Dict[str, int]:
        """Get count of detections per class."""
        counts = {}
        for detection in self.detections:
            counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "detections": [det.to_dict() for det in self.detections],
            "processing_time": self.processing_time,
            "image_shape": self.image_shape,
            "model_input_shape": self.model_input_shape,
            "timestamp": self.timestamp,
            "num_detections": len(self.detections),
            "class_counts": self.get_class_counts(),
        }


class RecyclingDetector:
    """
    Enhanced YOLOv8-based detection engine for recycling materials.

    Provides comprehensive object detection with performance monitoring,
    error handling, and configurable parameters.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize the detector.

        Args:
            model_path: Path to model file (uses config default if None)
        """
        self.config = get_config()
        self.model = None
        self.model_path = None
        self.device = self.config.detection.device
        self.is_loaded = False

        # Performance monitoring
        self.performance_monitor = ModelPerformanceMonitor()

        # Load model if path provided
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: Union[str, Path], max_retries: int = 3) -> bool:
        """
        Load YOLO model from file with retry mechanism.

        Args:
            model_path: Path to model file
            max_retries: Maximum number of loading attempts

        Returns:
            True if successful, False otherwise
        """
        model_path = Path(model_path)

        # Validate file existence and format
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file does not exist: {model_path}")

        if not model_path.suffix.lower() in [".pt", ".pth"]:
            logger.error(
                f"Unsupported model format: {model_path.suffix}. Supported: .pt, .pth"
            )
            raise ValueError(f"Unsupported model format: {model_path.suffix}")

        logger.info(f"Loading model from: {model_path}")

        last_exception = None

        for attempt in range(max_retries):
            try:
                logger.debug(f"Loading attempt {attempt + 1}/{max_retries}")

                # Load model
                self.model = YOLO(str(model_path))
                self.model_path = model_path

                # Move to specified device with error handling
                if self.device != "auto":
                    try:
                        self.model.to(self.device)
                        logger.debug(f"Model moved to device: {self.device}")
                    except Exception as device_error:
                        logger.warning(
                            f"Failed to move model to device {self.device}: {device_error}"
                        )
                        logger.info("Falling back to CPU")
                        self.device = "cpu"
                        self.model.to(self.device)

                # Validate model
                if not self._validate_model():
                    logger.error("Model validation failed")
                    raise RuntimeError("Model validation failed after loading")

                self.is_loaded = True
                logger.info(f"Model loaded successfully on device: {self.device}")
                return True

            except Exception as e:
                last_exception = e
                logger.warning(f"Model loading attempt {attempt + 1} failed: {e}")

                # Clean up on failure
                if self.model is not None:
                    try:
                        del self.model
                    except Exception:
                        pass
                    self.model = None

                # Wait before retry (exponential backoff)
                if attempt < max_retries - 1:
                    import time

                    wait_time = 2**attempt  # 1, 2, 4 seconds
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

        # All attempts failed
        self.is_loaded = False
        error_msg = (
            f"Failed to load model after {max_retries} attempts: {last_exception}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _validate_model(self) -> bool:
        """
        Validate the loaded model.

        Returns:
            True if model is valid, False otherwise
        """
        try:
            if self.model is None:
                return False

            # Check if model has the expected classes
            if not hasattr(self.model, "names"):
                logger.warning("Model doesn't have class names")
                return True  # Still allow to proceed

            class_names = list(self.model.names.values())
            target_classes = self.config.counting.target_classes

            # Check if target classes are in model
            missing_classes = [cls for cls in target_classes if cls not in class_names]
            if missing_classes:
                logger.warning(f"Target classes not found in model: {missing_classes}")
                logger.info(f"Available classes: {class_names}")

            # Test inference with dummy image
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            result = self.model.predict(dummy_image, verbose=False)

            if result is None:
                logger.error("Model inference test failed")
                return False

            logger.debug("Model validation successful")
            return True

        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return False

    @log_performance
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        target_classes: Optional[List[str]] = None,
    ) -> DetectionResult:
        """
        Perform object detection on image.

        Args:
            image: Input image as numpy array
            confidence_threshold: Override default confidence threshold
            target_classes: List of target class names to detect

        Returns:
            DetectionResult object
        """
        start_time = time.time()

        try:
            if not self.is_loaded:
                logger.error("Model not loaded")
                return DetectionResult(
                    detections=[],
                    processing_time=time.time() - start_time,
                    image_shape=image.shape,
                    model_input_shape=(0, 0),
                )

            # Use config values if not specified
            if confidence_threshold is None:
                confidence_threshold = self.config.detection.confidence_threshold

            if target_classes is None:
                target_classes = self.config.counting.target_classes.copy()

            msg = f"Detecting with confidence: {confidence_threshold}, target classes: {target_classes}"
            logger.debug(msg)

            # Run inference
            results = self.model.predict(
                source=image,
                conf=confidence_threshold,
                device=self.device,
                save=False,
                imgsz=self.config.detection.input_size,
                max_det=self.config.detection.max_detections,
                verbose=False,
            )

            # Process results
            detections = []

            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes

                    # Extract data
                    xyxys = boxes.xyxy.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)

                    # Create detection objects
                    for i in range(len(xyxys)):
                        class_name = result.names[class_ids[i]]

                        # FIXED: Enable target class filtering
                        if target_classes and class_name not in target_classes:
                            logger.debug(
                                f"Filtering out class: {class_name} (not in target classes)"
                            )
                            continue

                        detection = Detection(
                            xyxy=xyxys[i].tolist(),
                            confidence=float(confidences[i]),
                            class_id=int(class_ids[i]),
                            class_name=class_name,
                        )

                        detections.append(detection)
                        msg = f"Added detection: {class_name} with confidence {detection.confidence:.3f}"
                        logger.debug(msg)

            processing_time = time.time() - start_time

            # Update performance monitor
            self.performance_monitor.update(processing_time, len(detections))

            # Log detection summary
            class_counts = {}
            for det in detections:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

            logger.debug(
                f"Detection completed: {len(detections)} objects in {processing_time:.4f}s"
            )
            logger.debug(f"Class distribution: {class_counts}")

            return DetectionResult(
                detections=detections,
                processing_time=processing_time,
                image_shape=image.shape,
                model_input_shape=(
                    self.config.detection.input_size,
                    self.config.detection.input_size,
                ),
            )

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            processing_time = time.time() - start_time

            return DetectionResult(
                detections=[],
                processing_time=processing_time,
                image_shape=image.shape,
                model_input_shape=(0, 0),
            )

    def detect_batch(
        self, images: List[np.ndarray], confidence_threshold: Optional[float] = None
    ) -> List[DetectionResult]:
        """
        Perform detection on batch of images.

        Args:
            images: List of images
            confidence_threshold: Confidence threshold

        Returns:
            List of DetectionResult objects
        """
        results = []

        for i, image in enumerate(images):
            logger.debug(f"Processing batch image {i + 1}/{len(images)}")
            result = self.detect(image, confidence_threshold)
            results.append(result)

        return results

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get current performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        return {
            "average_fps": self.performance_monitor.get_average_fps(),
            "average_processing_time": self.performance_monitor.get_average_processing_time(),
            "average_detections_per_frame": self.performance_monitor.get_detection_rate(),
            "average_memory_usage_mb": self.performance_monitor.get_average_memory_usage(),
            "peak_memory_usage_mb": self.performance_monitor.get_peak_memory_usage(),
            "current_memory_usage_mb": self.performance_monitor.get_current_memory_usage(),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {"status": "not_loaded"}

        info = {
            "status": "loaded",
            "model_path": str(self.model_path),
            "device": self.device,
            "input_size": self.config.detection.input_size,
            "confidence_threshold": self.config.detection.confidence_threshold,
        }

        if hasattr(self.model, "names"):
            info["class_names"] = list(self.model.names.values())
            info["num_classes"] = len(self.model.names)

        return info

    def set_confidence_threshold(self, threshold: float):
        """
        Update confidence threshold.

        Args:
            threshold: New confidence threshold (0-1)
        """
        if 0.0 <= threshold <= 1.0:
            self.config.detection.confidence_threshold = threshold
            logger.info(f"Confidence threshold updated to: {threshold}")
        else:
            logger.warning(f"Invalid confidence threshold: {threshold}")

    def set_device(self, device: str):
        """
        Change inference device.

        Args:
            device: Device name ('cpu', 'cuda', 'mps')
        """
        try:
            if self.model is not None:
                self.model.to(device)

            self.device = device
            self.config.detection.device = device
            logger.info(f"Device changed to: {device}")

        except Exception as e:
            logger.error(f"Failed to change device to {device}: {e}")

    def warm_up(self, num_iterations: int = 3):
        """
        Warm up the model with dummy predictions.

        Args:
            num_iterations: Number of warm-up iterations
        """
        if not self.is_loaded:
            logger.warning("Cannot warm up: model not loaded")
            return

        logger.info(f"Warming up model with {num_iterations} iterations...")

        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        for i in range(num_iterations):
            _ = self.detect(dummy_image)
            logger.debug(f"Warm-up iteration {i+1}/{num_iterations} completed")

        logger.info("Model warm-up completed")

    def reset_performance_monitor(self):
        """Reset performance monitoring statistics."""
        self.performance_monitor = ModelPerformanceMonitor()
        logger.info("Performance monitor reset")

    def export_model(
        self, export_format: str = "onnx", output_path: Optional[str] = None
    ) -> bool:
        """
        Export model to different format.

        Args:
            export_format: Export format ('onnx', 'tensorrt', etc.)
            output_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_loaded:
                logger.error("Cannot export: model not loaded")
                return False

            logger.info(f"Exporting model to {export_format} format...")

            success = self.model.export(
                format=export_format, imgsz=self.config.detection.input_size
            )

            if success:
                logger.info(f"Model exported successfully to {export_format}")
                return True
            else:
                logger.error(f"Model export to {export_format} failed")
                return False

        except Exception as e:
            logger.error(f"Model export error: {e}")
            return False

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.model is not None:
                del self.model
                self.model = None

            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_loaded = False
            logger.info("Detector cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


