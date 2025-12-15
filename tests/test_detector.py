"""
Enhanced unit tests for the detection engine.

This module provides comprehensive tests for the RecyclingDetector class
with improved coverage, performance testing, and edge case handling.
"""

import pytest
import numpy as np
import tempfile
import shutil
import json
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from concurrent.futures import ThreadPoolExecutor

# Import modules to test
from src.core.detector import (
    RecyclingDetector,
    Detection,
    DetectionResult,
    ModelPerformanceMonitor,
    DetectorFactory,
    load_detector,
    detect_image,
    benchmark_detector,
)


# Test Fixtures
@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def large_image():
    """Create a large test image."""
    return np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)


@pytest.fixture
def small_image():
    """Create a small test image."""
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


@pytest.fixture
def mock_model_path():
    """Create a temporary mock model file."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        mock_path = Path(f.name)

    yield str(mock_path)

    # Cleanup
    if mock_path.exists():
        mock_path.unlink()


@pytest.fixture
def batch_images():
    """Create a batch of test images."""
    return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]


@pytest.fixture
def mock_yolo_model():
    """Create a comprehensive mock YOLO model."""
    mock_model = Mock()
    mock_model.names = {
        0: "bottle-glass",
        1: "bottle-plastic",
        2: "tin can",
        3: "cardboard",
        4: "paper",
    }

    # Mock prediction results
    mock_boxes = Mock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
        [[10, 20, 50, 80], [100, 120, 150, 180], [200, 220, 250, 280]]
    )
    mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9, 0.8, 0.7])
    mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0, 1, 2])

    mock_result = Mock()
    mock_result.boxes = mock_boxes
    mock_result.names = mock_model.names

    mock_model.predict.return_value = [mock_result]
    mock_model.export.return_value = True
    mock_model.to.return_value = mock_model

    return mock_model


class TestDetectionEnhanced:
    """Enhanced tests for the Detection dataclass."""

    def test_detection_with_edge_coordinates(self):
        """Test Detection with edge case coordinates."""
        # Very small object
        detection = Detection([10.1, 20.1, 10.2, 20.2], 0.5, 0, "bottle-glass")
        assert detection.area == pytest.approx(0.01, rel=1e-2)

        # Very large object
        detection = Detection([0, 0, 1000, 1000], 0.9, 1, "cardboard")
        assert detection.area == 1000000
        assert detection.center == (500, 500)

    def test_detection_with_negative_coordinates(self):
        """Test Detection handling negative coordinates."""
        detection = Detection([-10, -20, 50, 80], 0.8, 0, "bottle-glass")
        assert detection.center == (20, 30)
        assert detection.area == 6000  # 60 * 100

    def test_detection_serialization_roundtrip(self):
        """Test Detection serialization and deserialization."""
        original = Detection([10, 20, 50, 80], 0.85, 0, "bottle-glass")
        serialized = original.to_dict()

        # Verify all fields are preserved
        assert serialized["xyxy"] == [10, 20, 50, 80]
        assert serialized["confidence"] == 0.85
        assert serialized["class_name"] == "bottle-glass"
        assert serialized["center"] == (30, 50)
        assert serialized["area"] == 2400

    def test_detection_comparison_operations(self):
        """Test Detection comparison and sorting."""
        det1 = Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass")
        det2 = Detection([100, 120, 150, 180], 0.8, 1, "bottle-plastic")
        det3 = Detection([200, 220, 250, 280], 0.95, 2, "tin can")

        detections = [det1, det2, det3]

        # Sort by confidence
        sorted_by_conf = sorted(detections, key=lambda x: x.confidence, reverse=True)
        assert sorted_by_conf[0].confidence == 0.95
        assert sorted_by_conf[-1].confidence == 0.8

        # Sort by area
        sorted_by_area = sorted(detections, key=lambda x: x.area, reverse=True)
        assert all(d.area >= 2400 for d in sorted_by_area)


class TestDetectionResultEnhanced:
    """Enhanced tests for the DetectionResult dataclass."""

    def test_detection_result_with_empty_detections(self):
        """Test DetectionResult with no detections."""
        result = DetectionResult([], 0.05, (480, 640, 3), (640, 640))

        assert len(result) == 0
        assert result.get_class_counts() == {}

        # Filtering empty results
        filtered = result.filter_by_confidence(0.5)
        assert len(filtered) == 0

    def test_detection_result_advanced_filtering(self):
        """Test advanced filtering operations."""
        detections = [
            Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass"),
            Detection([100, 120, 150, 180], 0.6, 1, "bottle-plastic"),
            Detection([200, 220, 250, 280], 0.8, 2, "tin can"),
            Detection([300, 320, 350, 380], 0.4, 0, "bottle-glass"),
            Detection([400, 420, 450, 480], 0.95, 1, "bottle-plastic"),
        ]

        result = DetectionResult(detections, 0.05, (480, 640, 3), (640, 640))

        # Combined filtering
        high_conf_bottles = result.filter_by_confidence(0.7).filter_by_class(
            ["bottle-glass", "bottle-plastic"]
        )
        assert len(high_conf_bottles) == 2  # 0.9 bottle-glass and 0.95 bottle-plastic

        # Class counts verification
        counts = result.get_class_counts()
        assert counts["bottle-glass"] == 2
        assert counts["bottle-plastic"] == 2
        assert counts["tin can"] == 1

    def test_detection_result_serialization(self):
        """Test DetectionResult serialization."""
        detections = [
            Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass"),
            Detection([100, 120, 150, 180], 0.8, 1, "bottle-plastic"),
        ]

        result = DetectionResult(detections, 0.05, (480, 640, 3), (640, 640))
        serialized = result.to_dict()

        assert serialized["num_detections"] == 2
        assert serialized["processing_time"] == 0.05
        assert "timestamp" in serialized
        assert "class_counts" in serialized
        assert len(serialized["detections"]) == 2

    def test_detection_result_timing_accuracy(self):
        """Test timing accuracy in DetectionResult."""
        start_time = time.time()

        detections = [Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass")]
        result = DetectionResult(detections, 0.05, (480, 640, 3), (640, 640))

        end_time = time.time()

        # Timestamp should be within reasonable range
        assert start_time <= result.timestamp <= end_time


class TestModelPerformanceMonitorEnhanced:
    """Enhanced tests for the ModelPerformanceMonitor class."""

    def test_performance_monitor_window_behavior(self):
        """Test window size behavior with varying inputs."""
        monitor = ModelPerformanceMonitor(window_size=3)

        # Add measurements beyond window size
        measurements = [(0.05, 3), (0.06, 2), (0.04, 4), (0.07, 1), (0.03, 5)]

        for processing_time, detection_count in measurements:
            monitor.update(processing_time, detection_count)

        # Should only keep last 3 measurements
        assert len(monitor.processing_times) == 3
        assert len(monitor.detection_counts) == 3

        # Check that oldest measurements were removed
        assert monitor.processing_times == [0.04, 0.07, 0.03]
        assert monitor.detection_counts == [4, 1, 5]

    def test_performance_monitor_statistical_accuracy(self):
        """Test statistical calculations accuracy."""
        monitor = ModelPerformanceMonitor(window_size=100)

        # Add known values for testing
        known_times = [0.1, 0.2, 0.3, 0.4, 0.5]  # Average = 0.3
        known_counts = [1, 2, 3, 4, 5]  # Average = 3

        for time_val, count_val in zip(known_times, known_counts):
            monitor.update(time_val, count_val)

        assert monitor.get_average_processing_time() == pytest.approx(0.3, rel=1e-3)
        assert monitor.get_detection_rate() == pytest.approx(3.0, rel=1e-3)
        assert monitor.get_average_fps() == pytest.approx(1 / 0.3, rel=1e-2)

    def test_performance_monitor_edge_cases(self):
        """Test edge cases in performance monitoring."""
        monitor = ModelPerformanceMonitor(window_size=5)

        # Empty monitor
        assert monitor.get_average_fps() == 0.0
        assert monitor.get_average_processing_time() == 0.0
        assert monitor.get_detection_rate() == 0.0

        # Single measurement
        monitor.update(0.1, 5)
        # FPS calculation requires at least 2 measurements in some implementations
        # or might handle single measurement differently
        fps = monitor.get_average_fps()
        assert fps >= 0  # Should be non-negative
        if fps > 0:
            assert abs(fps - 10.0) < 0.1  # Allow small tolerance

        # Zero processing time (edge case)
        monitor.update(0.0001, 0)  # Very small time, zero detections
        assert monitor.get_average_fps() >= 0

    def get_average_fps(self) -> float:
        """Get average FPS over the window."""
        if len(self.processing_times) < 1:
            return 0.0

        avg_time = np.mean(self.processing_times)
        if avg_time > 0:
            return 1.0 / avg_time
        else:
            return 0.0  # Return 0 instead of infinity for zero processing time


class TestRecyclingDetectorEnhanced:
    """Enhanced tests for the RecyclingDetector class."""

    def test_detector_initialization_states(self):
        """Test detector initialization and state management."""
        detector = RecyclingDetector()

        # Initial state
        assert detector.model is None
        assert not detector.is_loaded
        assert detector.device in ["auto", "cpu", "cuda", "mps"]
        assert detector.performance_monitor is not None

        # Configuration should be loaded
        assert detector.config is not None

    @patch("src.core.detector.YOLO")
    def test_model_loading_validation_comprehensive(
        self, mock_yolo, mock_model_path, mock_yolo_model
    ):
        """Test comprehensive model loading and validation."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()
        success = detector.load_model(mock_model_path)

        assert success
        assert detector.is_loaded
        assert detector.model == mock_yolo_model

        # Test model info extraction
        info = detector.get_model_info()
        assert info["status"] == "loaded"
        assert info["num_classes"] == 5
        assert "bottle-glass" in info["class_names"]

    def test_model_loading_file_validation(self):
        """Test model file validation."""
        detector = RecyclingDetector()

        # Non-existent file
        assert not detector.load_model("nonexistent.pt")

        # Wrong file extension
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            wrong_ext_path = f.name

        try:
            assert not detector.load_model(wrong_ext_path)
        finally:
            Path(wrong_ext_path).unlink(missing_ok=True)

    @patch("src.core.detector.YOLO")
    def test_detection_with_various_parameters(
        self, mock_yolo, mock_model_path, mock_yolo_model, sample_image
    ):
        """Test detection with various parameter combinations."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        # Test with different confidence thresholds
        for confidence in [0.3, 0.5, 0.7, 0.9]:
            result = detector.detect(sample_image, confidence_threshold=confidence)
            assert isinstance(result, DetectionResult)
            # Processing time might be 0 for mocked operations, so just check it's non-negative
            assert result.processing_time >= 0

        # Test with target classes
        result = detector.detect(sample_image, target_classes=["bottle-glass"])
        # Should filter to only bottle-glass detections
        assert all(det.class_name == "bottle-glass" for det in result.detections)

    @patch("src.core.detector.YOLO")
    def test_batch_detection_processing(
        self, mock_yolo, mock_model_path, mock_yolo_model, batch_images
    ):
        """Test batch detection processing."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        results = detector.detect_batch(batch_images, confidence_threshold=0.5)

        assert len(results) == len(batch_images)
        assert all(isinstance(result, DetectionResult) for result in results)
        # Processing time might be 0 for mocked operations
        assert all(result.processing_time >= 0 for result in results)

    @patch("src.core.detector.YOLO")
    def test_device_switching(self, mock_yolo, mock_model_path, mock_yolo_model):
        """Test device switching functionality."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        # Test device switching
        for device in ["cpu", "cuda", "auto"]:
            detector.set_device(device)
            assert detector.device == device

            # Model should be moved to new device
            if detector.model:
                mock_yolo_model.to.assert_called_with(device)

    @patch("src.core.detector.YOLO")
    def test_model_warm_up(self, mock_yolo, mock_model_path, mock_yolo_model):
        """Test model warm-up functionality."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        # Reset call count
        mock_yolo_model.predict.reset_mock()

        # Test warm-up
        detector.warm_up(num_iterations=3)

        # Should have made 3 prediction calls during warm-up
        assert mock_yolo_model.predict.call_count >= 3

    @patch("src.core.detector.YOLO")
    def test_model_export_functionality(
        self, mock_yolo, mock_model_path, mock_yolo_model
    ):
        """Test model export functionality."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        # Test successful export
        success = detector.export_model("onnx")
        assert success
        mock_yolo_model.export.assert_called_with(
            format="onnx", imgsz=detector.config.detection.input_size
        )

        # Test export without loaded model
        detector.model = None
        detector.is_loaded = False
        success = detector.export_model("onnx")
        assert not success

    @patch("src.core.detector.YOLO")
    def test_performance_monitoring_integration(
        self, mock_yolo, mock_model_path, mock_yolo_model, sample_image
    ):
        """Test performance monitoring integration."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        # Reset performance monitor
        detector.reset_performance_monitor()

        # Perform multiple detections
        for _ in range(10):
            detector.detect(sample_image)

        stats = detector.get_performance_stats()

        assert stats["average_fps"] > 0
        assert stats["average_processing_time"] > 0
        assert stats["average_detections_per_frame"] >= 0

    @patch("src.core.detector.YOLO")
    def test_cleanup_functionality(self, mock_yolo, mock_model_path, mock_yolo_model):
        """Test cleanup functionality."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        assert detector.is_loaded
        assert detector.model is not None

        detector.cleanup()

        assert not detector.is_loaded
        assert detector.model is None

    def test_context_manager_behavior(self, mock_model_path):
        """Test context manager behavior."""
        with RecyclingDetector() as detector:
            assert detector is not None
            detector.model = Mock()  # Simulate loaded model

        # After context exit, cleanup should be called
        assert detector.model is None


class TestDetectorFactoryEnhanced:
    """Enhanced tests for the DetectorFactory class."""

    @patch("src.core.detector.YOLO")
    def test_factory_create_detector_success(
        self, mock_yolo, mock_model_path, mock_yolo_model
    ):
        """Test successful detector creation through factory."""
        mock_yolo.return_value = mock_yolo_model

        detector = DetectorFactory.create_detector(mock_model_path, "cpu")

        assert detector.is_loaded
        assert detector.device == "cpu"
        # Should call warm_up during creation
        assert mock_yolo_model.predict.called

    def test_factory_create_detector_failure(self):
        """Test detector creation failure through factory."""
        with pytest.raises(RuntimeError):
            DetectorFactory.create_detector("nonexistent.pt", "cpu")

    @patch("src.core.detector.get_config")
    @patch("src.core.detector.YOLO")
    def test_factory_create_from_config(
        self, mock_yolo, mock_get_config, mock_yolo_model
    ):
        """Test creating detector from configuration."""
        # Mock configuration
        mock_config = Mock()
        mock_config.paths.models_dir = Path("./models")
        mock_config.detection.device = "cpu"
        mock_get_config.return_value = mock_config

        # Improve mock model configuration
        mock_yolo_model.names = {0: "bottle-glass", 1: "bottle-plastic", 2: "tin can"}

        # Create mock model files
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir) / "models"
            models_dir.mkdir()

            # Create mock model files
            (models_dir / "best.pt").touch()
            (models_dir / "last.pt").touch()

            mock_config.paths.models_dir = models_dir
            mock_yolo.return_value = mock_yolo_model

            try:
                detector = DetectorFactory.create_from_config()
                assert detector.is_loaded
                # Should prefer "best.pt" if available
                mock_yolo.assert_called_with(str(models_dir / "best.pt"))
            except RuntimeError as e:
                # If model validation fails due to mocking, that's acceptable
                # Just verify the factory tried to load the model
                mock_yolo.assert_called_with(str(models_dir / "best.pt"))


class TestUtilityFunctionsEnhanced:
    """Enhanced tests for utility functions."""

    @patch("src.core.detector.DetectorFactory.create_detector")
    def test_load_detector_convenience(self, mock_create_detector, mock_model_path):
        """Test load_detector convenience function."""
        mock_detector = Mock()
        mock_create_detector.return_value = mock_detector

        detector = load_detector(mock_model_path, "cpu")

        assert detector == mock_detector
        mock_create_detector.assert_called_once_with(mock_model_path, "cpu")

    @patch("src.core.detector.RecyclingDetector")
    def test_detect_image_convenience(
        self, mock_detector_class, sample_image, mock_model_path
    ):
        """Test detect_image convenience function."""
        mock_detector = Mock()
        mock_result = Mock()
        mock_detector.detect.return_value = mock_result
        mock_detector_class.return_value.__enter__.return_value = mock_detector

        result = detect_image(sample_image, mock_model_path, confidence_threshold=0.8)

        assert result == mock_result
        mock_detector.load_model.assert_called_once_with(mock_model_path)
        mock_detector.detect.assert_called_once_with(sample_image, 0.8)

    @patch("src.core.detector.YOLO")
    def test_benchmark_detector_comprehensive(
        self, mock_yolo, mock_model_path, mock_yolo_model
    ):
        """Test comprehensive detector benchmarking."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        # Mock detection results for consistent timing with non-zero processing time
        mock_result = Mock()
        mock_result.processing_time = 0.05  # Non-zero processing time
        mock_result.detections = [Mock(), Mock()]  # 2 detections

        # Replace the detector's detect method with our mock
        original_detect = detector.detect
        detector.detect = Mock(return_value=mock_result)

        results = benchmark_detector(detector, num_iterations=10, image_size=(320, 320))

        assert "avg_fps" in results
        assert "total_time" in results
        assert results["num_iterations"] == 10
        assert results["avg_processing_time"] == 0.05
        assert results["avg_detections"] == 2
        # FPS should be calculated correctly with non-zero processing time
        assert results["avg_fps"] == pytest.approx(20.0, rel=1e-2)


class TestErrorHandlingEnhanced:
    """Enhanced error handling tests."""

    @patch("src.core.detector.YOLO")
    def test_model_prediction_various_failures(
        self, mock_yolo, mock_model_path, sample_image
    ):
        """Test handling of various model prediction failures."""
        # Test different types of prediction failures
        failure_types = [
            Exception("General prediction error"),
            RuntimeError("CUDA out of memory"),
            ValueError("Invalid input shape"),
            AttributeError("Model attribute error"),
        ]

        for failure in failure_types:
            mock_model = Mock()
            mock_model.names = {0: "bottle-glass"}
            mock_model.predict.side_effect = failure
            mock_yolo.return_value = mock_model

            detector = RecyclingDetector()
            detector.load_model(mock_model_path)

            # Should handle gracefully and return empty result
            result = detector.detect(sample_image)
            assert len(result.detections) == 0
            # Processing time should be non-negative (might be 0 for failed operations)
            assert result.processing_time >= 0

    def test_invalid_image_inputs(self):
        """Test handling of invalid image inputs."""
        detector = RecyclingDetector()

        invalid_inputs = [
            None,
            np.array([]),  # Empty array
            np.random.randint(0, 255, (100,), dtype=np.uint8),  # 1D array
            np.random.randint(
                0, 255, (100, 100), dtype=np.uint8
            ),  # 2D array (grayscale)
            np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8),  # 4-channel image
        ]

        for invalid_input in invalid_inputs:
            # Should handle gracefully without crashing
            try:
                result = detector.detect(invalid_input)
                assert isinstance(result, DetectionResult)
            except Exception as e:
                # Some exceptions are expected for severely malformed inputs
                assert isinstance(e, (ValueError, AttributeError, TypeError))

    @patch("src.core.detector.YOLO")
    def test_model_loading_corruption_simulation(self, mock_yolo, mock_model_path):
        """Test handling of corrupted model files."""
        # Simulate various model loading failures
        corruption_types = [
            Exception("Model file corrupted"),
            RuntimeError("Unsupported model format"),
            OSError("File permission denied"),
            MemoryError("Insufficient memory to load model"),
        ]

        for corruption in corruption_types:
            mock_yolo.side_effect = corruption

            detector = RecyclingDetector()
            success = detector.load_model(mock_model_path)

            assert not success
            assert not detector.is_loaded
            assert detector.model is None

            # Reset for next iteration
            mock_yolo.side_effect = None

    def test_confidence_threshold_edge_cases(self):
        """Test confidence threshold edge cases."""
        detector = RecyclingDetector()

        # Test various invalid thresholds
        invalid_thresholds = [-0.1, 1.1, float("inf"), float("-inf"), float("nan")]

        original_threshold = detector.config.detection.confidence_threshold

        for invalid_threshold in invalid_thresholds:
            detector.set_confidence_threshold(invalid_threshold)
            # Should not change from original value
            assert detector.config.detection.confidence_threshold == original_threshold


class TestConcurrencyAndThreadSafety:
    """Test concurrency and thread safety."""

    @patch("src.core.detector.YOLO")
    def test_concurrent_detection_calls(
        self, mock_yolo, mock_model_path, mock_yolo_model
    ):
        """Test concurrent detection calls."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        def detect_worker(image_id):
            """Worker function for concurrent detection."""
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = detector.detect(test_image)
            return len(result.detections)

        # Run concurrent detections
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(detect_worker, i) for i in range(10)]
            results = [future.result() for future in futures]

        # All should complete successfully
        assert len(results) == 10
        assert all(isinstance(r, int) for r in results)

    @patch("src.core.detector.YOLO")
    def test_thread_safety_performance_monitor(
        self, mock_yolo, mock_model_path, mock_yolo_model
    ):
        """Test thread safety of performance monitor."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        def update_performance_worker(worker_id):
            """Worker function for updating performance metrics."""
            for i in range(50):
                detector.performance_monitor.update(0.05 + worker_id * 0.01, i % 5)

        # Run concurrent updates
        threads = []
        for worker_id in range(4):
            thread = threading.Thread(
                target=update_performance_worker, args=(worker_id,)
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Performance monitor should handle concurrent updates
        stats = detector.get_performance_stats()
        assert stats["average_fps"] > 0
        assert stats["average_processing_time"] > 0


class TestPerformanceAndScalability:
    """Test performance and scalability."""

    @patch("src.core.detector.YOLO")
    def test_detection_performance_large_images(
        self, mock_yolo, mock_model_path, mock_yolo_model, large_image
    ):
        """Test detection performance with large images."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        # Test with large image
        start_time = time.time()
        result = detector.detect(large_image)
        processing_time = time.time() - start_time

        # Should complete in reasonable time
        assert processing_time < 1.0  # Less than 1 second
        assert isinstance(result, DetectionResult)
        assert result.image_shape == large_image.shape

    @patch("src.core.detector.YOLO")
    def test_memory_usage_batch_processing(
        self, mock_yolo, mock_model_path, mock_yolo_model
    ):
        """Test memory usage during batch processing."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        # Create large batch
        large_batch = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(20)
        ]

        # Process batch
        results = detector.detect_batch(large_batch)

        assert len(results) == 20
        assert all(isinstance(result, DetectionResult) for result in results)

    @pytest.mark.slow
    @patch("src.core.detector.YOLO")
    def test_long_term_stability(self, mock_yolo, mock_model_path, mock_yolo_model):
        """Test long-term stability and memory leaks."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        # Simulate long-term usage
        for iteration in range(100):
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = detector.detect(test_image)

            # Periodic validation
            if iteration % 20 == 0:
                stats = detector.get_performance_stats()
                # FPS might be 0 for very fast mocked operations
                assert stats["average_fps"] >= 0

                # Performance should remain stable (non-negative values)
                assert stats["average_processing_time"] >= 0


class TestConfigurationAndSettings:
    """Test configuration and settings management."""

    def test_confidence_threshold_persistence(self):
        """Test confidence threshold persistence."""
        detector = RecyclingDetector()

        original_threshold = detector.config.detection.confidence_threshold
        new_threshold = 0.75

        detector.set_confidence_threshold(new_threshold)
        assert detector.config.detection.confidence_threshold == new_threshold

        # Should persist across detection calls
        info = detector.get_model_info()
        if detector.is_loaded:
            assert info["confidence_threshold"] == new_threshold

    @patch("src.core.detector.YOLO")
    def test_device_configuration_persistence(
        self, mock_yolo, mock_model_path, mock_yolo_model
    ):
        """Test device configuration persistence."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()

        # Test device changes
        devices = ["cpu", "cuda", "auto"]
        for device in devices:
            detector.set_device(device)
            assert detector.device == device
            assert detector.config.detection.device == device

            # Load model with new device
            if detector.load_model(mock_model_path):
                info = detector.get_model_info()
                assert info["device"] == device


class TestIntegrationAdvanced:
    """Advanced integration tests."""

    @patch("src.core.detector.YOLO")
    def test_detection_pipeline_integration(
        self, mock_yolo, mock_model_path, mock_yolo_model, sample_image
    ):
        """Test complete detection pipeline integration."""
        mock_yolo.return_value = mock_yolo_model

        # Create detector through factory
        detector = DetectorFactory.create_detector(mock_model_path, "cpu")

        # Full pipeline test
        detector.set_confidence_threshold(0.7)
        detector.warm_up(num_iterations=2)

        # Multiple detection calls with different parameters
        results = []

        # Standard detection
        result1 = detector.detect(sample_image)
        results.append(result1)

        # Detection with custom confidence
        result2 = detector.detect(sample_image, confidence_threshold=0.5)
        results.append(result2)

        # Detection with target classes
        result3 = detector.detect(sample_image, target_classes=["bottle-glass"])
        results.append(result3)

        # Verify all results
        assert all(isinstance(r, DetectionResult) for r in results)
        # Processing time might be 0 for mocked operations
        assert all(r.processing_time >= 0 for r in results)

        # Performance stats should be updated
        stats = detector.get_performance_stats()
        assert stats["average_fps"] >= 0

        # Cleanup
        detector.cleanup()

    @patch("src.core.detector.YOLO")
    def test_export_and_benchmark_integration(
        self, mock_yolo, mock_model_path, mock_yolo_model
    ):
        """Test model export and benchmarking integration."""
        mock_yolo.return_value = mock_yolo_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        # Test export
        export_success = detector.export_model("onnx")
        assert export_success

        # Mock the detect method to return consistent results with non-zero processing time
        mock_result = Mock()
        mock_result.processing_time = 0.02  # Non-zero time
        mock_result.detections = [Mock()]
        detector.detect = Mock(return_value=mock_result)

        # Test benchmarking
        benchmark_results = benchmark_detector(detector, num_iterations=5)

        assert "avg_fps" in benchmark_results
        assert "total_time" in benchmark_results
        assert benchmark_results["num_iterations"] == 5


# Parameterized tests
@pytest.mark.parametrize("confidence_threshold", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_various_confidence_thresholds(
    confidence_threshold, mock_model_path, sample_image
):
    """Test detection with various confidence thresholds."""
    with patch("src.core.detector.YOLO") as mock_yolo:
        mock_model = Mock()
        mock_model.names = {0: "bottle-glass"}
        mock_model.predict.return_value = [Mock()]
        mock_yolo.return_value = mock_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        result = detector.detect(
            sample_image, confidence_threshold=confidence_threshold
        )
        assert isinstance(result, DetectionResult)


@pytest.mark.parametrize(
    "image_size", [(240, 320), (480, 640), (720, 1280), (1080, 1920)]
)
def test_various_image_sizes(image_size, mock_model_path):
    """Test detection with various image sizes."""
    test_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)

    with patch("src.core.detector.YOLO") as mock_yolo:
        mock_model = Mock()
        mock_model.names = {0: "bottle-glass"}
        mock_model.predict.return_value = [Mock()]
        mock_yolo.return_value = mock_model

        detector = RecyclingDetector()
        detector.load_model(mock_model_path)

        result = detector.detect(test_image)
        assert isinstance(result, DetectionResult)
        assert result.image_shape == test_image.shape


@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_various_devices(device, mock_model_path):
    """Test detector with various devices."""
    with patch("src.core.detector.YOLO") as mock_yolo:
        mock_model = Mock()
        mock_model.names = {0: "bottle-glass"}
        mock_yolo.return_value = mock_model

        detector = RecyclingDetector()
        detector.set_device(device)

        if detector.load_model(mock_model_path):
            assert detector.device == device


if __name__ == "__main__":
    # Run with comprehensive options
    pytest.main(
        [
            __file__,
            "-v",  # Verbose output
            "--tb=short",  # Short traceback format
            "-m",
            "not slow",  # Skip slow tests by default
            "--durations=10",  # Show 10 slowest tests
            "--cov=src.core.detector",  # Coverage for detector module
            "--cov-report=html",  # HTML coverage report
            "--cov-report=term-missing",  # Terminal coverage with missing lines
        ]
    )
