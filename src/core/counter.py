"""
Object counting logic for Smart Recycling Detection System.

This module provides sophisticated object counting capabilities with
line crossing detection, object tracking, and anti-double-counting measures.
"""

import time
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

from config.settings import get_config
from config.logging_config import get_logger
from src.core.detector import Detection, DetectionResult

logger = get_logger('counter')


@dataclass
class TrackedObject:
    """Represents a tracked object for counting purposes."""
    object_id: str
    class_name: str
    positions: deque = field(default_factory=lambda: deque(maxlen=10))
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    has_crossed: bool = False
    crossing_direction: Optional[str] = None  # 'left_to_right', 'right_to_left', etc.
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=5))
    
    def update_position(self, center: Tuple[float, float], confidence: float):
        """Update object position and confidence."""
        self.positions.append(center)
        self.confidence_history.append(confidence)
        self.last_seen = time.time()
    
    def get_current_position(self) -> Optional[Tuple[float, float]]:
        """Get the most recent position."""
        return self.positions[-1] if self.positions else None
    
    def get_average_confidence(self) -> float:
        """Get average confidence over recent detections."""
        return np.mean(self.confidence_history) if self.confidence_history else 0.0
    
    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """Calculate object velocity based on recent positions."""
        if len(self.positions) < 2:
            return None
        
        recent_pos = self.positions[-1]
        prev_pos = self.positions[-2]
        
        # Simple velocity calculation (position difference)
        vx = recent_pos[0] - prev_pos[0]
        vy = recent_pos[1] - prev_pos[1]
        
        return (vx, vy)


@dataclass
class CountingLine:
    """Represents a counting line for object crossing detection."""
    x: Optional[int] = None              # X coordinate for vertical line
    y: Optional[int] = None              # Y coordinate for horizontal line
    tolerance: int = 5                   # Tolerance for crossing detection
    direction_sensitive: bool = True     # Whether to track crossing direction
    
    def __post_init__(self):
        """Validate counting line configuration."""
        if self.x is None and self.y is None:
            raise ValueError("Either x or y coordinate must be specified")
        
        if self.x is not None and self.y is not None:
            logger.warning("Both x and y specified. Using x (vertical line) only.")
            self.y = None
    
    def check_crossing(self, prev_pos: Tuple[float, float], 
                      curr_pos: Tuple[float, float]) -> Optional[str]:
        """
        Check if object crossed the counting line.
        
        Args:
            prev_pos: Previous position (x, y)
            curr_pos: Current position (x, y)
            
        Returns:
            Crossing direction or None if no crossing detected
        """
        if self.x is not None:
            # Vertical line crossing
            prev_x, curr_x = prev_pos[0], curr_pos[0]
            
            # Check if crossed the line
            if (prev_x < self.x - self.tolerance and curr_x > self.x + self.tolerance):
                return 'left_to_right'
            elif (prev_x > self.x + self.tolerance and curr_x < self.x - self.tolerance):
                return 'right_to_left'
        
        elif self.y is not None:
            # Horizontal line crossing
            prev_y, curr_y = prev_pos[1], curr_pos[1]
            
            # Check if crossed the line
            if (prev_y < self.y - self.tolerance and curr_y > self.y + self.tolerance):
                return 'top_to_bottom'
            elif (prev_y > self.y + self.tolerance and curr_y < self.y - self.tolerance):
                return 'bottom_to_top'
        
        return None


class ObjectTracker:
    """Simple object tracker for counting purposes."""
    
    def __init__(self, max_distance: float = 50.0, max_age: int = 30):
        """
        Initialize tracker.
        
        Args:
            max_distance: Maximum distance for object association
            max_age: Maximum frames to keep unmatched objects
        """
        self.max_distance = max_distance
        self.max_age = max_age
        self.tracked_objects: Dict[str, TrackedObject] = {}
        self.next_id = 0
    
    def update(self, detections: List[Detection]) -> Dict[str, TrackedObject]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of current detections
            
        Returns:
            Dictionary of tracked objects
        """
        # Associate detections with existing tracked objects
        matched_objects = {}
        unmatched_detections = list(detections)
        
        # Simple distance-based matching
        for obj_id, tracked_obj in self.tracked_objects.items():
            if not tracked_obj.positions:
                continue
            
            curr_pos = tracked_obj.get_current_position()
            best_match = None
            min_distance = float('inf')
            
            for detection in unmatched_detections:
                if detection.class_name != tracked_obj.class_name:
                    continue
                
                # Calculate distance
                distance = self._calculate_distance(curr_pos, detection.center)
                
                if distance < min_distance and distance < self.max_distance:
                    min_distance = distance
                    best_match = detection
            
            if best_match:
                # Update existing object
                tracked_obj.update_position(best_match.center, best_match.confidence)
                matched_objects[obj_id] = tracked_obj
                unmatched_detections.remove(best_match)
        
        # Create new objects for unmatched detections
        for detection in unmatched_detections:
            obj_id = f"{detection.class_name}_{self.next_id}"
            self.next_id += 1
            
            new_object = TrackedObject(
                object_id=obj_id,
                class_name=detection.class_name
            )
            new_object.update_position(detection.center, detection.confidence)
            matched_objects[obj_id] = new_object
        
        # Remove old objects that haven't been seen
        current_time = time.time()
        for obj_id, tracked_obj in list(self.tracked_objects.items()):
            if current_time - tracked_obj.last_seen > self.max_age:
                logger.debug(f"Removing old tracked object: {obj_id}")
        
        # Update tracked objects
        self.tracked_objects = matched_objects
        
        return self.tracked_objects
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                          pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_object_count(self) -> int:
        """Get current number of tracked objects."""
        return len(self.tracked_objects)
    
    def reset(self):
        """Reset tracker state."""
        self.tracked_objects.clear()
        self.next_id = 0
        logger.info("Object tracker reset")


class RecyclingCounter:
    """
    Advanced counting system for recycling detection.
    
    Provides accurate object counting with line crossing detection,
    object tracking, and comprehensive statistics.
    """
    
    def __init__(self, counting_line: Optional[CountingLine] = None):
        """
        Initialize counter.
        
        Args:
            counting_line: CountingLine configuration
        """
        self.config = get_config()
        
        # Set up counting line
        if counting_line is None:
            self.counting_line = CountingLine(
                x=self.config.counting.line_position_x,
                y=self.config.counting.line_position_y
            )
        else:
            self.counting_line = counting_line
        
        # Initialize tracking
        self.tracker = ObjectTracker(
            max_distance=self.config.counting.tracking_max_distance,
            max_age=self.config.counting.reset_tracking_after_frames
        )
        
        # Counting statistics
        self.total_count = 0
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.direction_counts: Dict[str, int] = defaultdict(int)
        self.crossed_objects: Set[str] = set()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        logger.info("Recycling counter initialized")
    
    def update(self, detection_result: DetectionResult) -> Dict[str, int]:
        """
        Update counter with new detection results.
        
        Args:
            detection_result: Detection results from current frame
            
        Returns:
            Current count statistics
        """
        self.frame_count += 1
        
        try:
            # Update object tracking
            tracked_objects = self.tracker.update(detection_result.detections)
            
            # Check for line crossings
            new_crossings = self._check_crossings(tracked_objects)
            
            # Update counts
            for obj_id, direction in new_crossings.items():
                tracked_obj = tracked_objects[obj_id]
                
                if obj_id not in self.crossed_objects:
                    self.total_count += 1
                    self.class_counts[tracked_obj.class_name] += 1
                    self.direction_counts[direction] += 1
                    self.crossed_objects.add(obj_id)
                    
                    logger.info(f"Object crossed: {tracked_obj.class_name} ({direction})")
            
            # Log periodic statistics
            if self.frame_count % 100 == 0:
                self._log_statistics()
            
            return dict(self.class_counts)
            
        except Exception as e:
            logger.error(f"Error updating counter: {e}")
            return dict(self.class_counts)
    
    def _check_crossings(self, tracked_objects: Dict[str, TrackedObject]) -> Dict[str, str]:
        """
        Check for line crossings in tracked objects.
        
        Args:
            tracked_objects: Dictionary of tracked objects
            
        Returns:
            Dictionary mapping object IDs to crossing directions
        """
        new_crossings = {}
        
        for obj_id, tracked_obj in tracked_objects.items():
            # Skip if already crossed
            if tracked_obj.has_crossed:
                continue
            
            # Need at least 2 positions to check crossing
            if len(tracked_obj.positions) < 2:
                continue
            
            # Check for crossing
            prev_pos = tracked_obj.positions[-2]
            curr_pos = tracked_obj.positions[-1]
            
            crossing_direction = self.counting_line.check_crossing(prev_pos, curr_pos)
            
            if crossing_direction:
                tracked_obj.has_crossed = True
                tracked_obj.crossing_direction = crossing_direction
                new_crossings[obj_id] = crossing_direction
        
        return new_crossings
    
    def _log_statistics(self):
        """Log current counting statistics."""
        runtime = time.time() - self.start_time
        fps = self.frame_count / runtime if runtime > 0 else 0
        
        logger.info(f"Counting Statistics - Total: {self.total_count}, "
                   f"Frame: {self.frame_count}, FPS: {fps:.2f}")
        logger.info(f"Class counts: {dict(self.class_counts)}")
        logger.info(f"Direction counts: {dict(self.direction_counts)}")
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive counting statistics.
        
        Returns:
            Dictionary with all statistics
        """
        runtime = time.time() - self.start_time
        fps = self.frame_count / runtime if runtime > 0 else 0
        
        return {
            'total_count': self.total_count,
            'class_counts': dict(self.class_counts),
            'direction_counts': dict(self.direction_counts),
            'frame_count': self.frame_count,
            'runtime_seconds': runtime,
            'average_fps': fps,
            'tracked_objects': self.tracker.get_object_count(),
            'crossed_objects': len(self.crossed_objects)
        }
    
    def get_count_for_class(self, class_name: str) -> int:
        """
        Get count for specific class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            Count for the class
        """
        return self.class_counts.get(class_name, 0)
    
    def reset(self):
        """Reset all counting statistics and tracking."""
        self.total_count = 0
        self.class_counts.clear()
        self.direction_counts.clear()
        self.crossed_objects.clear()
        self.frame_count = 0
        self.start_time = time.time()
        
        # Reset tracker
        self.tracker.reset()
        
        logger.info("Counter reset completed")
    
    def set_counting_line(self, x: Optional[int] = None, y: Optional[int] = None):
        """
        Update counting line position.
        
        Args:
            x: X coordinate for vertical line
            y: Y coordinate for horizontal line
        """
        try:
            self.counting_line = CountingLine(x=x, y=y)
            logger.info(f"Counting line updated: x={x}, y={y}")
        except ValueError as e:
            logger.error(f"Invalid counting line configuration: {e}")
    
    def export_statistics(self, filepath: str) -> bool:
        """
        Export statistics to file.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            from pathlib import Path
            
            stats = self.get_statistics()
            stats['export_timestamp'] = time.time()
            stats['counting_line'] = {
                'x': self.counting_line.x,
                'y': self.counting_line.y
            }
            
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            logger.info(f"Statistics exported to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export statistics: {e}")
            return False


# Utility functions
def create_counter(line_x: Optional[int] = None, 
                  line_y: Optional[int] = None) -> RecyclingCounter:
    """
    Create a recycling counter with specified counting line.
    
    Args:
        line_x: X coordinate for vertical counting line
        line_y: Y coordinate for horizontal counting line
        
    Returns:
        RecyclingCounter instance
    """
    counting_line = CountingLine(x=line_x, y=line_y) if (line_x or line_y) else None
    return RecyclingCounter(counting_line)


def count_objects_simple(detections: List[Detection], 
                        line_position: int, 
                        orientation: str = 'vertical') -> Dict[str, int]:
    """
    Simple object counting without tracking (for basic use cases).
    
    Args:
        detections: List of detections
        line_position: Position of counting line
        orientation: 'vertical' or 'horizontal'
        
    Returns:
        Dictionary with class counts
    """
    counts = defaultdict(int)
    
    for detection in detections:
        center_x, center_y = detection.center
        
        if orientation == 'vertical':
            # Count objects that have crossed vertical line
            if center_x > line_position:
                counts[detection.class_name] += 1
        else:
            # Count objects that have crossed horizontal line
            if center_y > line_position:
                counts[detection.class_name] += 1
    
    return dict(counts)