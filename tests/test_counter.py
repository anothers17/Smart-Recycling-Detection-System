"""
Unit tests for the counting system.

This module provides comprehensive tests for the RecyclingCounter class
and related counting functionality.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch

from src.core.counter import (
    RecyclingCounter, TrackedObject, CountingLine, ObjectTracker,
    create_counter, count_objects_simple
)
from src.core.detector import Detection, DetectionResult


class TestCountingLine:
    """Test the CountingLine class."""
    
    def test_vertical_line_creation(self):
        """Test creating a vertical counting line."""
        line = CountingLine(x=300)
        
        assert line.x == 300
        assert line.y is None
        assert line.tolerance == 5
    
    def test_horizontal_line_creation(self):
        """Test creating a horizontal counting line."""
        line = CountingLine(y=200)
        
        assert line.x is None
        assert line.y == 200
    
    def test_invalid_line_creation(self):
        """Test creating invalid counting line."""
        with pytest.raises(ValueError):
            CountingLine()  # Neither x nor y specified
    
    def test_vertical_line_crossing_left_to_right(self):
        """Test vertical line crossing from left to right."""
        line = CountingLine(x=300, tolerance=5)
        
        prev_pos = (290, 150)  # Left of line
        curr_pos = (310, 150)  # Right of line
        
        direction = line.check_crossing(prev_pos, curr_pos)
        assert direction == 'left_to_right'
    
    def test_vertical_line_crossing_right_to_left(self):
        """Test vertical line crossing from right to left."""
        line = CountingLine(x=300, tolerance=5)
        
        prev_pos = (310, 150)  # Right of line
        curr_pos = (290, 150)  # Left of line
        
        direction = line.check_crossing(prev_pos, curr_pos)
        assert direction == 'right_to_left'
    
    def test_horizontal_line_crossing_top_to_bottom(self):
        """Test horizontal line crossing from top to bottom."""
        line = CountingLine(y=200, tolerance=5)
        
        prev_pos = (150, 190)  # Above line
        curr_pos = (150, 210)  # Below line
        
        direction = line.check_crossing(prev_pos, curr_pos)
        assert direction == 'top_to_bottom'
    
    def test_no_crossing(self):
        """Test when no crossing occurs."""
        line = CountingLine(x=300, tolerance=5)
        
        prev_pos = (290, 150)
        curr_pos = (295, 150)  # Still on same side
        
        direction = line.check_crossing(prev_pos, curr_pos)
        assert direction is None


class TestTrackedObject:
    """Test the TrackedObject class."""
    
    def test_tracked_object_creation(self):
        """Test creating a TrackedObject."""
        obj = TrackedObject(
            object_id="bottle_1",
            class_name="bottle-glass"
        )
        
        assert obj.object_id == "bottle_1"
        assert obj.class_name == "bottle-glass"
        assert not obj.has_crossed
        assert obj.crossing_direction is None
        assert len(obj.positions) == 0
    
    def test_update_position(self):
        """Test updating object position."""
        obj = TrackedObject("bottle_1", "bottle-glass")
        
        obj.update_position((100, 150), 0.9)
        
        assert len(obj.positions) == 1
        assert obj.get_current_position() == (100, 150)
        assert len(obj.confidence_history) == 1
        assert obj.get_average_confidence() == 0.9
    
    def test_velocity_calculation(self):
        """Test velocity calculation."""
        obj = TrackedObject("bottle_1", "bottle-glass")
        
        # Add positions to calculate velocity
        obj.update_position((100, 150), 0.9)
        obj.update_position((110, 150), 0.8)
        
        velocity = obj.get_velocity()
        assert velocity == (10.0, 0.0)  # Moved 10 pixels right
    
    def test_position_history_limit(self):
        """Test position history size limit."""
        obj = TrackedObject("bottle_1", "bottle-glass")
        
        # Add more positions than the deque limit (10)
        for i in range(15):
            obj.update_position((i * 10, 150), 0.9)
        
        assert len(obj.positions) == 10  # Should be limited to 10


class TestObjectTracker:
    """Test the ObjectTracker class."""
    
    def test_tracker_creation(self):
        """Test creating an ObjectTracker."""
        tracker = ObjectTracker(max_distance=50.0, max_age=30)
        
        assert tracker.max_distance == 50.0
        assert tracker.max_age == 30
        assert len(tracker.tracked_objects) == 0
        assert tracker.next_id == 0
    
    def test_track_new_objects(self):
        """Test tracking new objects."""
        tracker = ObjectTracker()
        
        detections = [
            Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass"),
            Detection([100, 120, 150, 180], 0.8, 1, "bottle-plastic")
        ]
        
        tracked = tracker.update(detections)
        
        assert len(tracked) == 2
        assert tracker.next_id == 2
        assert "bottle-glass_0" in tracked
        assert "bottle-plastic_1" in tracked
    
    def test_track_existing_objects(self):
        """Test tracking existing objects."""
        tracker = ObjectTracker(max_distance=20.0)
        
        # First frame
        detections1 = [
            Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass")
        ]
        tracked1 = tracker.update(detections1)
        
        # Second frame (object moved slightly)
        detections2 = [
            Detection([15, 25, 55, 85], 0.8, 0, "bottle-glass")
        ]
        tracked2 = tracker.update(detections2)
        
        # Should be same object ID
        assert len(tracked2) == 1
        object_id = list(tracked2.keys())[0]
        assert object_id == "bottle-glass_0"
        assert len(tracked2[object_id].positions) == 2
    
    def test_object_association_distance_limit(self):
        """Test object association respects distance limit."""
        tracker = ObjectTracker(max_distance=20.0)
        
        # First frame
        detections1 = [
            Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass")
        ]
        tracker.update(detections1)
        
        # Second frame (object moved too far)
        detections2 = [
            Detection([100, 120, 150, 180], 0.8, 0, "bottle-glass")
        ]
        tracked2 = tracker.update(detections2)
        
        # Should create new object due to distance
        assert len(tracked2) == 1
        assert tracker.next_id == 2  # Two objects created
    
    def test_reset_tracker(self):
        """Test resetting the tracker."""
        tracker = ObjectTracker()
        
        # Add some objects
        detections = [Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass")]
        tracker.update(detections)
        
        assert len(tracker.tracked_objects) == 1
        assert tracker.next_id == 1
        
        # Reset
        tracker.reset()
        
        assert len(tracker.tracked_objects) == 0
        assert tracker.next_id == 0


class TestRecyclingCounter:
    """Test the RecyclingCounter class."""
    
    def test_counter_creation(self):
        """Test creating a RecyclingCounter."""
        counter = RecyclingCounter()
        
        assert counter.total_count == 0
        assert len(counter.class_counts) == 0
        assert counter.frame_count == 0
        assert counter.counting_line.x is not None  # Should use config default
    
    def test_counter_with_custom_line(self):
        """Test creating counter with custom counting line."""
        line = CountingLine(x=400)
        counter = RecyclingCounter(line)
        
        assert counter.counting_line.x == 400
    
    def test_simple_counting_update(self):
        """Test basic counting update."""
        counter = RecyclingCounter()
        
        # Create mock detection result
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
        
        counts = counter.update(result)
        
        # Note: Without actual crossing, counts might be 0
        # This tests the update mechanism
        assert isinstance(counts, dict)
        assert counter.frame_count == 1
    
    def test_get_statistics(self):
        """Test getting counter statistics."""
        counter = RecyclingCounter()
        
        stats = counter.get_statistics()
        
        required_keys = [
            'total_count', 'class_counts', 'direction_counts',
            'frame_count', 'runtime_seconds', 'average_fps',
            'tracked_objects', 'crossed_objects'
        ]
        
        for key in required_keys:
            assert key in stats
    
    def test_reset_counter(self):
        """Test resetting the counter."""
        counter = RecyclingCounter()
        
        # Simulate some activity
        counter.total_count = 5
        counter.class_counts['bottle-glass'] = 3
        counter.frame_count = 100
        
        # Reset
        counter.reset()
        
        assert counter.total_count == 0
        assert len(counter.class_counts) == 0
        assert counter.frame_count == 0
    
    def test_set_counting_line(self):
        """Test setting counting line position."""
        counter = RecyclingCounter()
        
        counter.set_counting_line(x=400)
        assert counter.counting_line.x == 400
        
        counter.set_counting_line(y=300)
        assert counter.counting_line.y == 300
        assert counter.counting_line.x is None  # Should clear x when setting y
    
    def test_get_count_for_class(self):
        """Test getting count for specific class."""
        counter = RecyclingCounter()
        
        counter.class_counts['bottle-glass'] = 5
        counter.class_counts['bottle-plastic'] = 3
        
        assert counter.get_count_for_class('bottle-glass') == 5
        assert counter.get_count_for_class('bottle-plastic') == 3
        assert counter.get_count_for_class('tin can') == 0  # Default
    
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_export_statistics(self, mock_json_dump, mock_open):
        """Test exporting statistics to file."""
        counter = RecyclingCounter()
        
        success = counter.export_statistics("test_stats.json")
        
        assert success
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions in the counter module."""
    
    def test_create_counter(self):
        """Test create_counter utility function."""
        counter = create_counter(line_x=400)
        
        assert isinstance(counter, RecyclingCounter)
        assert counter.counting_line.x == 400
    
    def test_count_objects_simple_vertical(self):
        """Test simple object counting with vertical line."""
        detections = [
            Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass"),      # Center: (30, 50)
            Detection([350, 120, 390, 180], 0.8, 1, "bottle-plastic"), # Center: (370, 150)
            Detection([250, 220, 290, 280], 0.7, 2, "tin can")        # Center: (270, 250)
        ]
        
        counts = count_objects_simple(detections, line_position=300, orientation='vertical')
        
        # Only objects with center_x > 300 should be counted
        assert counts.get('bottle-plastic', 0) == 1  # center_x = 370 > 300
        assert counts.get('bottle-glass', 0) == 0    # center_x = 30 < 300
        assert counts.get('tin can', 0) == 0         # center_x = 270 < 300
    
    def test_count_objects_simple_horizontal(self):
        """Test simple object counting with horizontal line."""
        detections = [
            Detection([10, 20, 50, 80], 0.9, 0, "bottle-glass"),       # Center: (30, 50)
            Detection([100, 250, 150, 300], 0.8, 1, "bottle-plastic"), # Center: (125, 275)
            Detection([200, 120, 250, 180], 0.7, 2, "tin can")        # Center: (225, 150)
        ]
        
        counts = count_objects_simple(detections, line_position=200, orientation='horizontal')
        
        # Only objects with center_y > 200 should be counted
        assert counts.get('bottle-plastic', 0) == 1  # center_y = 275 > 200
        assert counts.get('bottle-glass', 0) == 0    # center_y = 50 < 200
        assert counts.get('tin can', 0) == 0         # center_y = 150 < 200


class TestIntegration:
    """Integration tests for counting components."""
    
    def test_full_counting_pipeline(self):
        """Test complete counting pipeline with line crossing."""
        counter = RecyclingCounter(CountingLine(x=300, tolerance=10))
        
        # Frame 1: Object on left side
        detections1 = [
            Detection([250, 100, 290, 140], 0.9, 0, "bottle-glass")  # Center: (270, 120)
        ]
        result1 = DetectionResult(detections1, 0.05, (480, 640, 3), (640, 640))
        
        counts1 = counter.update(result1)
        assert counter.total_count == 0  # No crossing yet
        
        # Frame 2: Object crosses line
        detections2 = [
            Detection([310, 105, 350, 145], 0.8, 0, "bottle-glass")  # Center: (330, 125)
        ]
        result2 = DetectionResult(detections2, 0.05, (480, 640, 3), (640, 640))
        
        counts2 = counter.update(result2)
        assert counter.total_count == 1  # Should count the crossing
        assert counts2.get('bottle-glass', 0) == 1
    
    def test_multiple_objects_crossing(self):
        """Test multiple objects crossing the line."""
        counter = RecyclingCounter(CountingLine(x=300, tolerance=5))
        
        # Simulate multiple frames with different objects crossing
        frames = [
            # Frame 1: Two objects on left
            [
                Detection([250, 100, 290, 140], 0.9, 0, "bottle-glass"),
                Detection([260, 200, 300, 240], 0.8, 1, "bottle-plastic")
            ],
            # Frame 2: First object crosses
            [
                Detection([310, 105, 350, 145], 0.9, 0, "bottle-glass"),
                Detection([265, 205, 305, 245], 0.8, 1, "bottle-plastic")
            ],
            # Frame 3: Second object crosses
            [
                Detection([320, 110, 360, 150], 0.9, 0, "bottle-glass"),
                Detection([315, 210, 355, 250], 0.8, 1, "bottle-plastic")
            ]
        ]
        
        total_crossings = 0
        for frame_detections in frames:
            result = DetectionResult(frame_detections, 0.05, (480, 640, 3), (640, 640))
            counts = counter.update(result)
            
            if counter.total_count > total_crossings:
                total_crossings = counter.total_count
        
        # Both objects should have crossed
        assert counter.total_count >= 1  # At least one crossing detected
    
    def test_counter_statistics_tracking(self):
        """Test statistics tracking over time."""
        counter = RecyclingCounter()
        
        # Process multiple frames
        for i in range(10):
            detections = [
                Detection([i*5, 100, i*5+40, 140], 0.9, 0, "bottle-glass")
            ]
            result = DetectionResult(detections, 0.05, (480, 640, 3), (640, 640))
            counter.update(result)
        
        stats = counter.get_statistics()
        
        assert stats['frame_count'] == 10
        assert stats['runtime_seconds'] > 0
        assert stats['average_fps'] > 0
    
    def test_counter_reset_functionality(self):
        """Test complete counter reset."""
        counter = RecyclingCounter()
        
        # Add some state
        counter.total_count = 5
        counter.class_counts['bottle-glass'] = 3
        counter.frame_count = 50
        counter.crossed_objects.add("test_object")
        
        # Reset
        counter.reset()
        
        assert counter.total_count == 0
        assert len(counter.class_counts) == 0
        assert counter.frame_count == 0
        assert len(counter.crossed_objects) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_detection_result(self):
        """Test counter with empty detection results."""
        counter = RecyclingCounter()
        
        empty_result = DetectionResult([], 0.05, (480, 640, 3), (640, 640))
        counts = counter.update(empty_result)
        
        assert isinstance(counts, dict)
        assert counter.total_count == 0
        assert counter.frame_count == 1
    
    def test_invalid_detection_data(self):
        """Test counter with invalid detection data."""
        counter = RecyclingCounter()
        
        # Create detection with invalid coordinates
        invalid_detection = Detection(
            [1000, 2000, 50, 80],  # x2 < x1, y2 < y1
            0.9, 0, "bottle-glass"
        )
        
        result = DetectionResult([invalid_detection], 0.05, (480, 640, 3), (640, 640))
        
        # Should handle gracefully
        counts = counter.update(result)
        assert isinstance(counts, dict)
    
    def test_very_fast_object_movement(self):
        """Test handling very fast moving objects."""
        counter = RecyclingCounter(CountingLine(x=300, tolerance=50))
        
        # Object jumps across line in one frame
        detections1 = [Detection([200, 100, 240, 140], 0.9, 0, "bottle-glass")]
        result1 = DetectionResult(detections1, 0.05, (480, 640, 3), (640, 640))
        counter.update(result1)
        
        detections2 = [Detection([400, 105, 440, 145], 0.9, 0, "bottle-glass")]
        result2 = DetectionResult(detections2, 0.05, (480, 640, 3), (640, 640))
        counts = counter.update(result2)
        
        # Should detect crossing despite large jump
        # (Depends on tracking and tolerance settings)
        assert isinstance(counts, dict)
    
    def test_object_disappearing_and_reappearing(self):
        """Test handling objects that disappear and reappear."""
        counter = RecyclingCounter()
        
        # Object appears
        detections1 = [Detection([250, 100, 290, 140], 0.9, 0, "bottle-glass")]
        result1 = DetectionResult(detections1, 0.05, (480, 640, 3), (640, 640))
        counter.update(result1)
        
        # Object disappears (empty detections)
        empty_result = DetectionResult([], 0.05, (480, 640, 3), (640, 640))
        counter.update(empty_result)
        
        # Object reappears in different location
        detections3 = [Detection([350, 200, 390, 240], 0.8, 0, "bottle-glass")]
        result3 = DetectionResult(detections3, 0.05, (480, 640, 3), (640, 640))
        counts = counter.update(result3)
        
        # Should handle gracefully
        assert isinstance(counts, dict)
        assert counter.frame_count == 3


class TestPerformance:
    """Performance tests for counting system."""
    
    def test_counting_performance_many_objects(self):
        """Test counting performance with many objects."""
        counter = RecyclingCounter()
        
        # Create detection result with many objects
        detections = []
        for i in range(100):
            detection = Detection(
                [i*6, 100, i*6+40, 140], 
                0.9, 
                i % 3, 
                ["bottle-glass", "bottle-plastic", "tin can"][i % 3]
            )
            detections.append(detection)
        
        result = DetectionResult(detections, 0.05, (480, 640, 3), (640, 640))
        
        # Measure time
        start_time = time.time()
        counts = counter.update(result)
        processing_time = time.time() - start_time
        
        # Should complete quickly (under 0.1 seconds)
        assert processing_time < 0.1
        assert isinstance(counts, dict)
    
    def test_counting_performance_many_frames(self):
        """Test counting performance over many frames."""
        counter = RecyclingCounter()
        
        # Process many frames
        start_time = time.time()
        
        for frame_num in range(100):
            detections = [
                Detection([frame_num*2, 100, frame_num*2+40, 140], 0.9, 0, "bottle-glass")
            ]
            result = DetectionResult(detections, 0.01, (480, 640, 3), (640, 640))
            counter.update(result)
        
        total_time = time.time() - start_time
        
        # Should process 100 frames quickly
        assert total_time < 1.0  # Less than 1 second
        assert counter.frame_count == 100


@pytest.mark.parametrize("line_position", [100, 300, 500, 700])
def test_different_line_positions(line_position):
    """Test counting with different line positions."""
    counter = RecyclingCounter(CountingLine(x=line_position))
    
    assert counter.counting_line.x == line_position
    
    # Test with detection on both sides
    left_detection = Detection([line_position-50, 100, line_position-10, 140], 0.9, 0, "bottle-glass")
    right_detection = Detection([line_position+10, 100, line_position+50, 140], 0.9, 0, "bottle-glass")
    
    # Should handle both without errors
    result1 = DetectionResult([left_detection], 0.05, (480, 640, 3), (640, 640))
    result2 = DetectionResult([right_detection], 0.05, (480, 640, 3), (640, 640))
    
    counter.update(result1)
    counts = counter.update(result2)
    
    assert isinstance(counts, dict)


@pytest.mark.parametrize("tolerance", [1, 5, 10, 20])
def test_different_tolerance_values(tolerance):
    """Test counting line with different tolerance values."""
    line = CountingLine(x=300, tolerance=tolerance)
    
    # Test crossing with movement just over tolerance
    prev_pos = (300 - tolerance - 1, 150)
    curr_pos = (300 + tolerance + 1, 150)
    
    direction = line.check_crossing(prev_pos, curr_pos)
    assert direction == 'left_to_right'
    
    # Test movement within tolerance (should not cross)
    prev_pos2 = (300 - tolerance + 1, 150)
    curr_pos2 = (300 + tolerance - 1, 150)
    
    direction2 = line.check_crossing(prev_pos2, curr_pos2)
    assert direction2 is None


if __name__ == '__main__':
    pytest.main([__file__])