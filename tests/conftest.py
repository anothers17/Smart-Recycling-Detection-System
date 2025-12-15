"""
Comprehensive test configuration and fixtures for Smart Recycling Detection System.

This module provides shared fixtures, configuration, and utilities for all test modules.
"""

import pytest
import numpy as np
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any, Generator
import threading
import psutil
import gc

from src.core.detector import Detection, DetectionResult
from src.core.counter import CountingLine, TrackedObject


# Test Configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line(
        "markers", "memory_intensive: marks tests that use significant memory"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests based on their names/paths
        if "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)

        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)

        if "memory" in item.nodeid.lower():
            item.add_marker(pytest.mark.memory_intensive)

        if any(keyword in item.nodeid.lower() for keyword in ["cuda", "gpu", "device"]):
            item.add_marker(pytest.mark.gpu)


# Session-level fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory(prefix="test_recycling_") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="session")
def mock_config():
    """Create mock configuration for testing."""
    config = Mock()

    # Detection configuration
    config.detection.confidence_threshold = 0.7
    config.detection.device = "cpu"
    config.detection.input_size = 640
    config.detection.max_detections = 300

    # Counting configuration
    config.counting.line_position_x = 300
    config.counting.line_position_y = None
    config.counting.target_classes = ["bottle-glass", "bottle-plastic", "tin can"]
    config.counting.tracking_max_distance = 50.0
    config.counting.reset_tracking_after_frames = 30

    # Paths configuration
    config.paths.models_dir = Path("./models")
    config.paths.output_dir = Path("./output")
    config.paths.logs_dir = Path("./logs")

    return config


# Image fixtures
@pytest.fixture
def sample_image_small():
    """Create small test image (240x320)."""
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_medium():
    """Create medium test image (480x640)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_large():
    """Create large test image (1080x1920)."""
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_grayscale():
    """Create grayscale test image."""
    return np.random.randint(0, 255, (480, 640), dtype=np.uint8)


@pytest.fixture
def image_batch_small():
    """Create batch of small test images."""
    return [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(5)]


@pytest.fixture
def image_batch_large():
    """Create batch of large test images."""
    return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]


# Detection fixtures
@pytest.fixture
def simple_detection():
    """Create a simple detection object."""
    return Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass")


@pytest.fixture
def sample_detections():
    """Create sample detection objects for testing."""
    return [
        Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass"),
        Detection([100, 120, 150, 180], 0.8, 1, "bottle-plastic"),
        Detection([200, 220, 250, 280], 0.7, 2, "tin can"),
        Detection([300, 320, 350, 380], 0.85, 0, "bottle-glass"),
        Detection([400, 420, 450, 480], 0.6, 1, "bottle-plastic"),
    ]


@pytest.fixture
def high_confidence_detections():
    """Create high confidence detections."""
    return [
        Detection([10, 20, 50, 80], 0.95, 0, "bottle-glass"),
        Detection([100, 120, 150, 180], 0.92, 1, "bottle-plastic"),
        Detection([200, 220, 250, 280], 0.88, 2, "tin can"),
    ]


@pytest.fixture
def low_confidence_detections():
    """Create low confidence detections."""
    return [
        Detection([10, 20, 50, 80], 0.3, 0, "bottle-glass"),
        Detection([100, 120, 150, 180], 0.25, 1, "bottle-plastic"),
        Detection([200, 220, 250, 280], 0.4, 2, "tin can"),
    ]


@pytest.fixture
def overlapping_detections():
    """Create overlapping detections for testing."""
    return [
        Detection([100, 100, 200, 200], 0.9, 0, "bottle-glass"),
        Detection([150, 150, 250, 250], 0.8, 0, "bottle-glass"),
        Detection([180, 180, 280, 280], 0.7, 0, "bottle-glass"),
    ]


@pytest.fixture
def mixed_class_detections():
    """Create detections with mixed classes."""
    classes = ["bottle-glass", "bottle-plastic", "tin can", "cardboard", "paper"]
    detections = []

    for i, class_name in enumerate(classes):
        detection = Detection(
            [i * 60, 100, i * 60 + 50, 150], 0.8 - i * 0.05, i, class_name
        )
        detections.append(detection)

    return detections


# DetectionResult fixtures
@pytest.fixture
def sample_detection_result(sample_detections):
    """Create sample DetectionResult."""
    return DetectionResult(
        detections=sample_detections,
        processing_time=0.05,
        image_shape=(480, 640, 3),
        model_input_shape=(640, 640),
    )


@pytest.fixture
def empty_detection_result():
    """Create empty DetectionResult."""
    return DetectionResult(
        detections=[],
        processing_time=0.02,
        image_shape=(480, 640, 3),
        model_input_shape=(640, 640),
    )


@pytest.fixture
def large_detection_result():
    """Create DetectionResult with many detections."""
    detections = []
    classes = ["bottle-glass", "bottle-plastic", "tin can"]

    for i in range(50):
        detection = Detection(
            [i * 12, 100 + (i % 10) * 30, i * 12 + 40, 130 + (i % 10) * 30],
            0.7 + (i % 3) * 0.1,
            i % 3,
            classes[i % 3],
        )
        detections.append(detection)

    return DetectionResult(
        detections=detections,
        processing_time=0.1,
        image_shape=(1920, 1080, 3),
        model_input_shape=(640, 640),
    )


# Counting fixtures
@pytest.fixture
def vertical_counting_line():
    """Create vertical counting line."""
    return CountingLine(x=300, tolerance=10)


@pytest.fixture
def horizontal_counting_line():
    """Create horizontal counting line."""
    return CountingLine(y=200, tolerance=10)


@pytest.fixture
def strict_counting_line():
    """Create counting line with strict tolerance."""
    return CountingLine(x=300, tolerance=2)


@pytest.fixture
def loose_counting_line():
    """Create counting line with loose tolerance."""
    return CountingLine(x=300, tolerance=25)


@pytest.fixture
def tracked_object_simple():
    """Create simple tracked object."""
    obj = TrackedObject("bottle_1", "bottle-glass")
    obj.update_position((100, 150), 0.9)
    return obj


@pytest.fixture
def tracked_object_with_history():
    """Create tracked object with position history."""
    obj = TrackedObject("bottle_2", "bottle-plastic")
    positions = [(90, 145), (100, 150), (110, 155), (120, 160), (130, 165)]
    confidences = [0.8, 0.85, 0.9, 0.87, 0.83]

    for pos, conf in zip(positions, confidences):
        obj.update_position(pos, conf)
        time.sleep(0.001)  # Small delay for timing

    return obj


# Mock fixtures
@pytest.fixture
def mock_yolo_model():
    """Create comprehensive mock YOLO model."""
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


@pytest.fixture
def mock_yolo_model_empty():
    """Create mock YOLO model that returns no detections."""
    mock_model = Mock()
    mock_model.names = {0: "bottle-glass", 1: "bottle-plastic", 2: "tin can"}

    # Mock empty results
    mock_result = Mock()
    mock_result.boxes = None
    mock_result.names = mock_model.names

    mock_model.predict.return_value = [mock_result]
    mock_model.export.return_value = True
    mock_model.to.return_value = mock_model

    return mock_model


@pytest.fixture
def mock_yolo_model_variable():
    """Create mock YOLO model with variable detection results."""
    mock_model = Mock()
    mock_model.names = {0: "bottle-glass", 1: "bottle-plastic", 2: "tin can"}

    # Track call count to vary results
    mock_model.call_count = 0

    def side_effect_predict(*args, **kwargs):
        mock_model.call_count += 1

        # Vary results based on call count
        if mock_model.call_count % 3 == 0:
            # No detections every third call
            mock_result = Mock()
            mock_result.boxes = None
            mock_result.names = mock_model.names
            return [mock_result]

        # Normal detections
        mock_boxes = Mock()
        num_detections = mock_model.call_count % 4 + 1  # 1-4 detections

        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.random.randint(
            0, 400, (num_detections, 4)
        )
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.random.uniform(
            0.5, 0.95, num_detections
        )
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.random.randint(
            0, 3, num_detections
        )

        mock_result = Mock()
        mock_result.boxes = mock_boxes
        mock_result.names = mock_model.names

        return [mock_result]

    mock_model.predict.side_effect = side_effect_predict
    mock_model.export.return_value = True
    mock_model.to.return_value = mock_model

    return mock_model


@pytest.fixture
def mock_model_file(test_data_dir):
    """Create temporary mock model file."""
    model_path = test_data_dir / "test_model.pt"
    model_path.touch()  # Create empty file
    return model_path


@pytest.fixture
def mock_model_files(test_data_dir):
    """Create multiple temporary mock model files."""
    models_dir = test_data_dir / "models"
    models_dir.mkdir()

    model_files = ["best.pt", "last.pt", "epoch_100.pt"]
    created_files = []

    for filename in model_files:
        file_path = models_dir / filename
        file_path.touch()
        created_files.append(file_path)

    return created_files


# Performance monitoring fixtures
@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""

    class MemoryMonitor:
        def __init__(self):
            self.initial_memory = psutil.virtual_memory().used
            self.peak_memory = self.initial_memory

        def update(self):
            current_memory = psutil.virtual_memory().used
            self.peak_memory = max(self.peak_memory, current_memory)

        def get_peak_usage_mb(self):
            return (self.peak_memory - self.initial_memory) / (1024 * 1024)

        def cleanup(self):
            gc.collect()  # Force garbage collection

    monitor = MemoryMonitor()
    yield monitor
    monitor.cleanup()


@pytest.fixture
def performance_timer():
    """Timer for performance measurements."""

    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.durations = []

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            if self.start_time is not None:
                self.end_time = time.perf_counter()
                duration = self.end_time - self.start_time
                self.durations.append(duration)
                return duration
            return 0

        def reset(self):
            self.start_time = None
            self.end_time = None
            self.durations.clear()

        def get_average_duration(self):
            return np.mean(self.durations) if self.durations else 0

        def get_total_duration(self):
            return sum(self.durations)

    return PerformanceTimer()


# Test data generators
@pytest.fixture
def detection_sequence_crossing():
    """Generate detection sequence showing object crossing line."""

    def generate_sequence(line_x=300, num_frames=5):
        sequences = []

        # Object starts left of line and moves right
        start_x = line_x - 100
        step_x = 40

        for frame in range(num_frames):
            x = start_x + frame * step_x
            detection = Detection(
                [x, 100, x + 40, 140],
                0.9 - frame * 0.02,  # Slightly decreasing confidence
                0,
                "bottle-glass",
            )
            sequences.append([detection])

        return sequences

    return generate_sequence


@pytest.fixture
def detection_sequence_multiple_objects():
    """Generate detection sequence with multiple objects."""

    def generate_sequence(num_objects=3, num_frames=5):
        sequences = []
        classes = ["bottle-glass", "bottle-plastic", "tin can"]

        for frame in range(num_frames):
            frame_detections = []

            for obj_id in range(num_objects):
                x = 50 + obj_id * 150 + frame * 20
                y = 100 + obj_id * 80

                detection = Detection(
                    [x, y, x + 40, y + 40],
                    0.8 + obj_id * 0.05,
                    obj_id % len(classes),
                    classes[obj_id % len(classes)],
                )
                frame_detections.append(detection)

            sequences.append(frame_detections)

        return sequences

    return generate_sequence


# Configuration fixtures
@pytest.fixture
def temp_config_file(test_data_dir):
    """Create temporary configuration file."""
    config_data = {
        "detection": {
            "confidence_threshold": 0.7,
            "device": "cpu",
            "input_size": 640,
            "max_detections": 300,
        },
        "counting": {
            "line_position_x": 300,
            "target_classes": ["bottle-glass", "bottle-plastic", "tin can"],
            "tracking_max_distance": 50.0,
        },
        "paths": {"models_dir": "models", "output_dir": "output"},
    }

    config_file = test_data_dir / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    return config_file


# Thread safety fixtures
@pytest.fixture
def thread_barrier():
    """Create thread barrier for synchronization."""
    return threading.Barrier(4)  # For 4 threads


@pytest.fixture
def shared_resource():
    """Create shared resource with lock for thread safety tests."""

    class SharedResource:
        def __init__(self):
            self.value = 0
            self.lock = threading.Lock()
            self.access_count = 0

        def increment(self):
            with self.lock:
                self.value += 1
                self.access_count += 1

        def get_value(self):
            with self.lock:
                return self.value

        def reset(self):
            with self.lock:
                self.value = 0
                self.access_count = 0

    return SharedResource()


# Validation fixtures
@pytest.fixture
def detection_validator():
    """Create detection validator for testing."""

    class DetectionValidator:
        @staticmethod
        def validate_detection(detection):
            """Validate detection object."""
            assert isinstance(detection, Detection)
            assert len(detection.xyxy) == 4
            assert 0 <= detection.confidence <= 1
            assert detection.class_id >= 0
            assert detection.class_name
            assert detection.area >= 0

            # Bounding box should be valid
            x1, y1, x2, y2 = detection.xyxy
            assert x2 > x1, f"Invalid bbox: x2 ({x2}) <= x1 ({x1})"
            assert y2 > y1, f"Invalid bbox: y2 ({y2}) <= y1 ({y1})"

        @staticmethod
        def validate_detection_result(result):
            """Validate detection result."""
            assert isinstance(result, DetectionResult)
            assert isinstance(result.detections, list)
            assert result.processing_time >= 0
            assert len(result.image_shape) == 3
            assert len(result.model_input_shape) == 2
            assert result.timestamp > 0

            for detection in result.detections:
                DetectionValidator.validate_detection(detection)

        @staticmethod
        def validate_counting_line(line):
            """Validate counting line."""
            assert isinstance(line, CountingLine)
            assert (line.x is not None) or (line.y is not None)
            assert line.tolerance > 0

    return DetectionValidator()


# Error simulation fixtures
@pytest.fixture
def error_simulator():
    """Create error simulator for testing error handling."""

    class ErrorSimulator:
        def __init__(self):
            self.error_types = [
                ValueError("Invalid input"),
                RuntimeError("Processing failed"),
                MemoryError("Out of memory"),
                OSError("File not found"),
                AttributeError("Missing attribute"),
                TypeError("Invalid type"),
            ]
            self.current_error_index = 0

        def get_next_error(self):
            """Get next error in sequence."""
            error = self.error_types[self.current_error_index]
            self.current_error_index = (self.current_error_index + 1) % len(
                self.error_types
            )
            return error

        def get_random_error(self):
            """Get random error."""
            import random

            return random.choice(self.error_types)

        def simulate_intermittent_failure(self, success_rate=0.7):
            """Simulate intermittent failures."""
            import random

            if random.random() > success_rate:
                raise self.get_random_error()
            return True

    return ErrorSimulator()


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield

    # Force garbage collection
    gc.collect()

    # Clear any matplotlib figures (if used)
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        pass


@pytest.fixture(scope="session", autouse=True)
def session_cleanup():
    """Cleanup after entire test session."""
    yield

    # Final cleanup
    gc.collect()

    # Print session summary
    print("\n" + "=" * 50)
    print("Test session completed")
    print("=" * 50)


# Helper utilities
class TestUtils:
    """Utility functions for testing."""

    @staticmethod
    def create_realistic_detections(image_shape, num_objects=5, min_confidence=0.5):
        """Create realistic detection objects."""
        height, width = image_shape[:2]
        detections = []
        classes = ["bottle-glass", "bottle-plastic", "tin can", "cardboard", "paper"]

        for i in range(num_objects):
            # Random position ensuring object fits in image
            x1 = np.random.randint(0, width - 100)
            y1 = np.random.randint(0, height - 100)
            x2 = x1 + np.random.randint(40, 100)
            y2 = y1 + np.random.randint(40, 100)

            # Ensure within bounds
            x2 = min(x2, width)
            y2 = min(y2, height)

            confidence = np.random.uniform(min_confidence, 1.0)
            class_id = i % len(classes)
            class_name = classes[class_id]

            detection = Detection([x1, y1, x2, y2], confidence, class_id, class_name)
            detections.append(detection)

        return detections

    @staticmethod
    def create_crossing_trajectory(start_x, end_x, y_center, num_steps=5):
        """Create trajectory that crosses a vertical line."""
        trajectory = []

        for i in range(num_steps):
            progress = i / (num_steps - 1)
            x = start_x + progress * (end_x - start_x)
            y = y_center + np.random.randint(-10, 10)  # Small vertical variation

            trajectory.append((x, y))

        return trajectory

    @staticmethod
    def assert_detection_quality(detections, min_confidence=0.5, max_objects=50):
        """Assert detection quality metrics."""
        assert len(detections) <= max_objects, f"Too many detections: {len(detections)}"

        for detection in detections:
            assert (
                detection.confidence >= min_confidence
            ), f"Low confidence detection: {detection.confidence}"
            assert detection.area > 0, "Zero area detection"

            # Check bounding box validity
            x1, y1, x2, y2 = detection.xyxy
            assert x2 > x1 and y2 > y1, "Invalid bounding box"

    @staticmethod
    def measure_performance(func, *args, **kwargs):
        """Measure function performance."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        duration = end_time - start_time
        return result, duration

    @staticmethod
    def compare_detection_results(result1, result2, tolerance=0.1):
        """Compare two detection results for similarity."""
        if len(result1.detections) != len(result2.detections):
            return False

        for det1, det2 in zip(result1.detections, result2.detections):
            # Compare positions with tolerance
            for coord1, coord2 in zip(det1.xyxy, det2.xyxy):
                if abs(coord1 - coord2) > tolerance:
                    return False

            # Compare confidence
            if abs(det1.confidence - det2.confidence) > tolerance:
                return False

            # Compare class
            if det1.class_name != det2.class_name:
                return False

        return True


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils


# Performance benchmarking fixture
@pytest.fixture
def benchmark_runner():
    """Create benchmark runner for performance tests."""

    class BenchmarkRunner:
        def __init__(self):
            self.results = {}

        def run_benchmark(self, name, func, iterations=100, *args, **kwargs):
            """Run benchmark and store results."""
            durations = []

            # Warm-up
            for _ in range(5):
                func(*args, **kwargs)

            # Actual benchmark
            for _ in range(iterations):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                durations.append(end_time - start_time)

            self.results[name] = {
                "avg_duration": np.mean(durations),
                "std_duration": np.std(durations),
                "min_duration": np.min(durations),
                "max_duration": np.max(durations),
                "total_duration": np.sum(durations),
                "iterations": iterations,
                "fps": iterations / np.sum(durations),
            }

            return self.results[name]

        def compare_benchmarks(self, name1, name2):
            """Compare two benchmark results."""
            if name1 not in self.results or name2 not in self.results:
                return None

            result1 = self.results[name1]
            result2 = self.results[name2]

            return {
                "speedup": result2["avg_duration"] / result1["avg_duration"],
                "fps_improvement": result1["fps"] / result2["fps"],
                "difference_ms": (result1["avg_duration"] - result2["avg_duration"])
                * 1000,
            }

        def get_summary(self):
            """Get summary of all benchmarks."""
            return self.results

    return BenchmarkRunner()


# Data generation fixtures for stress testing
@pytest.fixture
def stress_test_data():
    """Generate data for stress testing."""

    class StressTestData:
        @staticmethod
        def generate_large_detection_batch(batch_size=100, detections_per_frame=20):
            """Generate large batch of detection results."""
            results = []
            classes = [
                "bottle-glass",
                "bottle-plastic",
                "tin can",
                "cardboard",
                "paper",
            ]

            for batch_idx in range(batch_size):
                detections = []

                for det_idx in range(detections_per_frame):
                    x = (det_idx % 10) * 60 + np.random.randint(-10, 10)
                    y = (det_idx // 10) * 60 + 100

                    detection = Detection(
                        [x, y, x + 50, y + 50],
                        np.random.uniform(0.5, 0.95),
                        det_idx % len(classes),
                        classes[det_idx % len(classes)],
                    )
                    detections.append(detection)

                result = DetectionResult(
                    detections=detections,
                    processing_time=np.random.uniform(0.02, 0.08),
                    image_shape=(1080, 1920, 3),
                    model_input_shape=(640, 640),
                )
                results.append(result)

            return results

        @staticmethod
        def generate_memory_intensive_data(num_frames=1000):
            """Generate memory-intensive test data."""
            frames = []

            for frame_idx in range(num_frames):
                # Vary number of detections per frame
                num_detections = (frame_idx % 50) + 1

                detections = TestUtils.create_realistic_detections(
                    (1080, 1920), num_detections, min_confidence=0.3
                )

                frames.append(detections)

            return frames

    return StressTestData()
