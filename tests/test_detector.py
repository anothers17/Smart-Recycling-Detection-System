"""
Unit tests for the detection engine.

This module provides comprehensive tests for the RecyclingDetector class
and related detection functionality.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from src.core.detector import (
    RecyclingDetector, Detection, DetectionResult,
    ModelPerformanceMonitor, DetectorFactory
)


class TestDetection:
    """Test the Detection dataclass."""
    
    def test_detection_creation(self):
        """Test creating a Detection object."""
        detection = Detection(
            xyxy=[10.0, 20.0, 50.0, 80.0],
            confidence=0.85,
            class_id=0,
            class_name="bottle-glass"
        )
        
        assert detection.xyxy == [10.0, 20.0, 50.0, 80.0]
        assert detection.confidence == 0.85
        assert detection.class_id == 0
        assert detection.class_name == "bottle-glass"
        assert detection.center == (30.0, 50.0)  # Calculated center
        assert detection.area == 2400.0  # (50-10) * (80-20)
    
    def test_detection_to_dict(self):
        """Test converting Detection to dictionary."""
        detection = Detection(
            xyxy=[10.0, 20.0, 50.0, 80.0],
            confidence=0.85,
            class_id=0,
            class_name="bottle-glass"
        )
        
        result_dict = detection.to_dict()
        
        assert result_dict['xyxy'] == [10.0, 20.0, 50.0, 80.0]
        assert result_dict['confidence'] == 0.85
        assert result_dict['class_name'] == "bottle-glass"
        assert result_dict['center'] == (30.0, 50.0)
        assert result_dict['area'] == 2400.0


class TestDetectionResult:
    """Test the DetectionResult dataclass."""
    
    def test_detection_result_creation(self):
        """Test creating a DetectionResult object."""
        detections = [
            Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass"),
            Detection([100, 120, 150, 180], 0.8, 1, "bottle-plastic")
        ]
        
        result = DetectionResult(
            detections=detections,
            processing_time=0.05,
            image_shape=(480, 640, 3),
            model_input_shape=(640, 640)
        )
        
        assert len(result) == 2
        assert result.processing_time == 0.05
        assert result.image_shape == (480, 640, 3)
    
    def test_filter_by_class(self):
        """Test filtering detections by class name."""
        detections = [
            Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass"),
            Detection([100, 120, 150, 180], 0.8, 1, "bottle-plastic"),
            Detection([200, 220, 250, 280], 0.7, 0, "bottle-glass")
        ]
        
        result = DetectionResult(detections, 0.05, (480, 640, 3), (640, 640))
        filtered = result.filter_by_class(["bottle-glass"])
        
        assert len(filtered) == 2
        assert all(d.class_name == "bottle-glass" for d in filtered.detections)
    
    def test_filter_by_confidence(self):
        """Test filtering detections by confidence threshold."""
        detections = [
            Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass"),
            Detection([100, 120, 150, 180], 0.6, 1, "bottle-plastic"),
            Detection([200, 220, 250, 280], 0.4, 2, "tin can")
        ]
        
        result = DetectionResult(detections, 0.05, (480, 640, 3), (640, 640))
        filtered = result.filter_by_confidence(0.7)
        
        assert len(filtered) == 1
        assert filtered.detections[0].confidence == 0.9
    
    def test_get_class_counts(self):
        """Test getting class counts."""
        detections = [
            Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass"),
            Detection([100, 120, 150, 180], 0.8, 1, "bottle-plastic"),
            Detection([200, 220, 250, 280], 0.7, 0, "bottle-glass")
        ]
        
        result = DetectionResult(detections, 0.05, (480, 640, 3), (640, 640))
        counts = result.get_class_counts()
        
        assert counts["bottle-glass"] == 2
        assert counts["bottle-plastic"] == 1


class TestModelPerformanceMonitor:
    """Test the ModelPerformanceMonitor class."""
    
    def test_performance_monitor_creation(self):
        """Test creating a performance monitor."""
        monitor = ModelPerformanceMonitor(window_size=50)
        assert monitor.window_size == 50
        assert len(monitor.processing_times) == 0
    
    def test_update_metrics(self):
        """Test updating performance metrics."""
        monitor = ModelPerformanceMonitor(window_size=5)
        
        # Add some measurements
        monitor.update(0.05, 3)
        monitor.update(0.06, 2)
        monitor.update(0.04, 4)
        
        assert len(monitor.processing_times) == 3
        assert len(monitor.detection_counts) == 3
        assert monitor.get_average_processing_time() == pytest.approx(0.05, rel=1e-2)
        assert monitor.get_detection_rate() == pytest.approx(3.0, rel=1e-2)
    
    def test_window_size_limit(self):
        """Test that window size is respected."""
        monitor = ModelPerformanceMonitor(window_size=3)
        
        # Add more measurements than window size
        for i in range(5):
            monitor.update(0.05 + i * 0.01, i + 1)
        
        assert len(monitor.processing_times) == 3
        assert len(monitor.detection_counts) == 3


class TestRecyclingDetector:
    """Test the RecyclingDetector class."""
    
    @pytest.fixture
    def mock_model_path(self):
        """Create a temporary mock model file."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            mock_path = Path(f.name)
        
        yield str(mock_path)
        
        # Cleanup
        if mock_path.exists():
            mock_path.unlink()
    
    def test_detector_creation(self):
        """Test creating a RecyclingDetector."""
        detector = RecyclingDetector()
        
        assert detector.model is None
        assert not detector.is_loaded
        assert detector.config is not None
    
    @patch('src.core.detector.YOLO')
    def test_load_model_success(self, mock_yolo, mock_model_path):
        """Test successful model loading."""
        # Mock YOLO model
        mock_model = Mock()
        mock_model.names = {0: 'bottle-glass', 1: 'bottle-plastic', 2: 'tin can'}
        mock_model.predict.return_value = [Mock()]
        mock_yolo.return_value = mock_model
        
        detector = RecyclingDetector()
        success = detector.load_model(mock_model_path)
        
        assert success
        assert detector.is_loaded
        assert detector.model == mock_model
        mock_yolo.assert_called_once_with(mock_model_path)
    
    def test_load_model_file_not_found(self):
        """Test model loading with non-existent file."""
        detector = RecyclingDetector()
        success = detector.load_model("nonexistent_model.pt")
        
        assert not success
        assert not detector.is_loaded
        assert detector.model is None
    
    @patch('src.core.detector.YOLO')
    def test_detect_without_model(self, mock_yolo):
        """Test detection without loaded model."""
        detector = RecyclingDetector()
        
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(test_image)
        
        assert isinstance(result, DetectionResult)
        assert len(result.detections) == 0
        assert result.processing_time > 0
    
    @patch('src.core.detector.YOLO')
    def test_detect_with_model(self, mock_yolo, mock_model_path):
        """Test detection with loaded model."""
        # Mock YOLO model and results
        mock_boxes = Mock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 20, 50, 80]])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
        
        mock_result = Mock()
        mock_result.boxes = mock_boxes
        mock_result.names = {0: 'bottle-glass'}
        
        mock_model = Mock()
        mock_model.names = {0: 'bottle-glass'}
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        # Test detection
        detector = RecyclingDetector()
        detector.load_model(mock_model_path)
        
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(test_image)
        
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "bottle-glass"
        assert result.detections[0].confidence == 0.9
        assert result.processing_time > 0
    
    def test_set_confidence_threshold(self):
        """Test setting confidence threshold."""
        detector = RecyclingDetector()
        
        # Valid threshold
        detector.set_confidence_threshold(0.8)
        assert detector.config.detection.confidence_threshold == 0.8
        
        # Invalid threshold (should be ignored)
        detector.set_confidence_threshold(1.5)
        assert detector.config.detection.confidence_threshold == 0.8  # Unchanged
    
    def test_get_model_info_no_model(self):
        """Test getting model info when no model is loaded."""
        detector = RecyclingDetector()
        info = detector.get_model_info()
        
        assert info['status'] == 'not_loaded'
    
    @patch('src.core.detector.YOLO')
    def test_get_model_info_with_model(self, mock_yolo, mock_model_path):
        """Test getting model info with loaded model."""
        mock_model = Mock()
        mock_model.names = {0: 'bottle-glass', 1: 'bottle-plastic'}
        mock_yolo.return_value = mock_model
        
        detector = RecyclingDetector()
        detector.load_model(mock_model_path)
        
        info = detector.get_model_info()
        
        assert info['status'] == 'loaded'
        assert info['model_path'] == mock_model_path
        assert info['num_classes'] == 2
        assert 'bottle-glass' in info['class_names']
    
    def test_performance_stats(self):
        """Test getting performance statistics."""
        detector = RecyclingDetector()
        stats = detector.get_performance_stats()
        
        assert 'average_fps' in stats
        assert 'average_processing_time' in stats
        assert 'average_detections_per_frame' in stats
        assert all(isinstance(v, float) for v in stats.values())
    
    def test_cleanup(self):
        """Test detector cleanup."""
        detector = RecyclingDetector()
        detector.model = Mock()  # Mock model
        detector.is_loaded = True
        
        detector.cleanup()
        
        assert detector.model is None
        assert not detector.is_loaded


class TestDetectorFactory:
    """Test the DetectorFactory class."""
    
    @patch('src.core.detector.YOLO')
    def test_create_detector(self, mock_yolo, mock_model_path):
        """Test creating detector through factory."""
        mock_model = Mock()
        mock_model.names = {0: 'bottle-glass'}
        mock_model.predict.return_value = [Mock()]
        mock_yolo.return_value = mock_model
        
        detector = DetectorFactory.create_detector(mock_model_path, "cpu")
        
        assert detector.is_loaded
        assert detector.device == "cpu"
        mock_yolo.assert_called_once_with(mock_model_path)
    
    @patch('src.core.detector.YOLO')
    def test_create_detector_invalid_path(self, mock_yolo):
        """Test creating detector with invalid model path."""
        with pytest.raises(RuntimeError):
            DetectorFactory.create_detector("nonexistent.pt", "cpu")


class TestIntegration:
    """Integration tests for detector components."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @patch('src.core.detector.YOLO')
    def test_full_detection_pipeline(self, mock_yolo, mock_model_path, sample_image):
        """Test complete detection pipeline."""
        # Mock YOLO components
        mock_boxes = Mock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
            [10, 20, 50, 80],
            [100, 120, 150, 180]
        ])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9, 0.8])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0, 1])
        
        mock_result = Mock()
        mock_result.boxes = mock_boxes
        mock_result.names = {0: 'bottle-glass', 1: 'bottle-plastic'}
        
        mock_model = Mock()
        mock_model.names = {0: 'bottle-glass', 1: 'bottle-plastic'}
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        # Test pipeline
        detector = RecyclingDetector()
        detector.load_model(mock_model_path)
        result = detector.detect(sample_image)
        
        # Verify results
        assert len(result.detections) == 2
        assert result.detections[0].class_name == "bottle-glass"
        assert result.detections[1].class_name == "bottle-plastic"
        
        # Test filtering
        glass_only = result.filter_by_class(["bottle-glass"])
        assert len(glass_only) == 1
        
        high_conf = result.filter_by_confidence(0.85)
        assert len(high_conf) == 1
        assert high_conf.detections[0].confidence == 0.9
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring during detection."""
        detector = RecyclingDetector()
        
        # Simulate multiple detection calls
        for i in range(5):
            detector.performance_monitor.update(0.05 + i * 0.01, i + 1)
        
        stats = detector.get_performance_stats()
        
        assert stats['average_fps'] > 0
        assert stats['average_processing_time'] > 0
        assert stats['average_detections_per_frame'] > 0


@pytest.mark.parametrize("confidence_threshold", [0.3, 0.5, 0.7, 0.9])
def test_different_confidence_thresholds(confidence_threshold):
    """Test detector with different confidence thresholds."""
    detector = RecyclingDetector()
    detector.set_confidence_threshold(confidence_threshold)
    
    assert detector.config.detection.confidence_threshold == confidence_threshold


@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_different_devices(device):
    """Test detector with different devices."""
    detector = RecyclingDetector()
    
    # Note: This doesn't actually test CUDA functionality
    # as it requires specific hardware
    detector.set_device(device)
    assert detector.device == device


class TestErrorHandling:
    """Test error handling in detector."""
    
    def test_invalid_model_format(self):
        """Test loading model with invalid format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            invalid_path = Path(f.name)
        
        try:
            detector = RecyclingDetector()
            success = detector.load_model(str(invalid_path))
            
            assert not success
            assert not detector.is_loaded
        finally:
            invalid_path.unlink()
    
    @patch('src.core.detector.YOLO')
    def test_model_prediction_failure(self, mock_yolo, mock_model_path):
        """Test handling of model prediction failures."""
        mock_model = Mock()
        mock_model.names = {0: 'bottle-glass'}
        mock_model.predict.side_effect = Exception("Model prediction failed")
        mock_yolo.return_value = mock_model
        
        detector = RecyclingDetector()
        detector.load_model(mock_model_path)
        
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(test_image)
        
        # Should return empty result on error
        assert len(result.detections) == 0
        assert result.processing_time > 0


if __name__ == '__main__':
    pytest.main([__file__])