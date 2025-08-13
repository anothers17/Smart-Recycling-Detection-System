"""
Enhanced plotting utilities for Smart Recycling Detection System.

This module provides comprehensive visualization tools for object detection,
including bounding boxes, labels, confidence scores, and counting overlays.
Based on the original plotting.py but with significant enhancements.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass

from config.logging_config import get_logger

logger = get_logger('main')


class Colors:
    """
    Enhanced color palette for object detection visualization.
    
    Based on Ultralytics color palette but with additional functionality
    and better color management.
    """

    def __init__(self):
        """Initialize colors with hex codes from matplotlib tableau colors."""
        hexs = (
            'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', 
            '92CC17', '3DDB86', '1A9334', '00D4BB', '2C99A8', '00C2FF', 
            '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 
            'FF95C8', 'FF37C7'
        )
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        
        # Pose estimation color palette
        self.pose_palette = np.array([
            [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], 
            [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255], 
            [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102], 
            [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51], 
            [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]
        ], dtype=np.uint8)
        
        # Predefined colors for specific classes
        self.class_colors = {
            'bottle-glass': (0, 255, 255),      # Cyan
            'bottle-plastic': (255, 165, 0),    # Orange  
            'tin can': (128, 128, 128),         # Gray
            'can': (128, 128, 128),             # Gray (alternative)
            'bottle': (0, 255, 0),              # Green (generic bottle)
        }

    def __call__(self, i: int, bgr: bool = False) -> Tuple[int, int, int]:
        """
        Get color by index.
        
        Args:
            i: Color index
            bgr: Whether to return BGR format (default RGB)
            
        Returns:
            Color tuple (R,G,B) or (B,G,R)
        """
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    def get_class_color(self, class_name: str, bgr: bool = False) -> Tuple[int, int, int]:
        """
        Get color for specific class name.
        
        Args:
            class_name: Name of the class
            bgr: Whether to return BGR format
            
        Returns:
            Color tuple
        """
        if class_name.lower() in self.class_colors:
            color = self.class_colors[class_name.lower()]
            return (color[2], color[1], color[0]) if bgr else color
        else:
            # Fallback to index-based color
            hash_val = hash(class_name.lower()) % self.n
            return self(hash_val, bgr)

    @staticmethod
    def hex2rgb(h: str) -> Tuple[int, int, int]:
        """Convert hex color codes to RGB values."""
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    def rgb2hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB values to hex color code."""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


@dataclass
class TextProperties:
    """Properties for text rendering."""
    font_scale: float = 0.5
    thickness: int = 1
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    background: bool = True
    background_alpha: float = 0.7


class TextSizeCalculator:
    """Enhanced text size calculator with caching."""
    
    def __init__(self):
        self._cache = {}
    
    def get_text_size(self, text: str, font_scale: float, thickness: int, 
                     font: int = cv2.FONT_HERSHEY_SIMPLEX) -> Tuple[Tuple[int, int], int]:
        """
        Get text size with caching for performance.
        
        Args:
            text: Text to measure
            font_scale: Font scale
            thickness: Text thickness
            font: Font type
            
        Returns:
            ((width, height), baseline)
        """
        cache_key = (text, font_scale, thickness, font)
        
        if cache_key not in self._cache:
            size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
            self._cache[cache_key] = (size, baseline)
        
        return self._cache[cache_key]
    
    def clear_cache(self):
        """Clear the text size cache."""
        self._cache.clear()


class EnhancedAnnotator:
    """
    Enhanced annotator for object detection visualization.
    
    Provides comprehensive annotation capabilities including bounding boxes,
    labels, confidence scores, tracking IDs, and counting information.
    """

    def __init__(self, image: np.ndarray, line_width: Optional[int] = None):
        """
        Initialize annotator.
        
        Args:
            image: Input image
            line_width: Line width for drawings (auto-calculated if None)
        """
        self.image = image.copy()
        self.original_image = image.copy()
        
        # Calculate line width based on image size
        if line_width is None:
            self.lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        else:
            self.lw = line_width
        
        # Initialize components
        self.colors = Colors()
        self.text_calculator = TextSizeCalculator()
        self.text_props = TextProperties()
        
        # Adjust text properties based on image size
        img_area = image.shape[0] * image.shape[1]
        if img_area > 1000000:  # Large image
            self.text_props.font_scale = 0.7
            self.text_props.thickness = 2
        elif img_area < 300000:  # Small image
            self.text_props.font_scale = 0.4
            self.text_props.thickness = 1

    def draw_box(self, xyxy: List[float], color: Tuple[int, int, int], 
                 thickness: Optional[int] = None) -> 'EnhancedAnnotator':
        """
        Draw bounding box.
        
        Args:
            xyxy: Bounding box coordinates [x1, y1, x2, y2]
            color: Box color (BGR)
            thickness: Line thickness
            
        Returns:
            Self for method chaining
        """
        try:
            x1, y1, x2, y2 = map(int, xyxy)
            thickness = thickness or self.lw
            
            cv2.rectangle(self.image, (x1, y1), (x2, y2), color, thickness)
            
        except Exception as e:
            logger.error(f"Error drawing box: {e}")
        
        return self

    def draw_label(self, xyxy: List[float], text: str, color: Tuple[int, int, int],
                   background_color: Optional[Tuple[int, int, int]] = None,
                   position: str = 'top') -> 'EnhancedAnnotator':
        """
        Draw text label with background.
        
        Args:
            xyxy: Bounding box coordinates for positioning
            text: Text to draw
            color: Text color (BGR)
            background_color: Background color (BGR)
            position: Label position ('top', 'bottom', 'center')
            
        Returns:
            Self for method chaining
        """
        try:
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Get text size
            (text_width, text_height), baseline = self.text_calculator.get_text_size(
                text, self.text_props.font_scale, self.text_props.thickness, self.text_props.font
            )
            
            # Calculate label position
            if position == 'top':
                label_y = y1
                text_y = y1 - 5 if y1 - text_height - 10 > 0 else y1 + text_height + 5
                outside = y1 - text_height - 10 > 0
            elif position == 'bottom':
                label_y = y2
                text_y = y2 + text_height + 5
                outside = True
            else:  # center
                label_y = (y1 + y2) // 2
                text_y = label_y + text_height // 2
                outside = True
            
            # Draw background rectangle
            if self.text_props.background:
                if background_color is None:
                    background_color = color
                
                pad = 3
                bg_x1 = x1
                bg_x2 = x1 + text_width + 2 * pad
                
                if outside and position == 'top':
                    bg_y1 = y1 - text_height - 2 * pad
                    bg_y2 = y1
                else:
                    bg_y1 = text_y - text_height - pad
                    bg_y2 = text_y + pad
                
                cv2.rectangle(self.image, (bg_x1, bg_y1), (bg_x2, bg_y2), 
                             background_color, -1)
            
            # Draw text
            cv2.putText(self.image, text, (x1 + 3, text_y - 2), 
                       self.text_props.font, self.text_props.font_scale,
                       (255, 255, 255), self.text_props.thickness, cv2.LINE_AA)
            
        except Exception as e:
            logger.error(f"Error drawing label: {e}")
        
        return self

    def draw_detection(self, xyxy: List[float], class_name: str, 
                      confidence: float, track_id: Optional[int] = None,
                      show_confidence: bool = True) -> 'EnhancedAnnotator':
        """
        Draw complete detection annotation.
        
        Args:
            xyxy: Bounding box coordinates
            class_name: Class name
            confidence: Detection confidence
            track_id: Tracking ID (optional)
            show_confidence: Whether to show confidence score
            
        Returns:
            Self for method chaining
        """
        try:
            # Get color for this class
            color = self.colors.get_class_color(class_name, bgr=True)
            
            # Draw bounding box
            self.draw_box(xyxy, color)
            
            # Prepare label text
            label_parts = [class_name]
            
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            if track_id is not None:
                label_parts.append(f"ID:{track_id}")
            
            label_text = " | ".join(label_parts)
            
            # Draw label
            self.draw_label(xyxy, label_text, color)
            
        except Exception as e:
            logger.error(f"Error drawing detection: {e}")
        
        return self

    def draw_counting_line(self, x: Optional[int] = None, y: Optional[int] = None,
                          color: Tuple[int, int, int] = (255, 0, 0),
                          thickness: Optional[int] = None,
                          label: Optional[str] = None) -> 'EnhancedAnnotator':
        """
        Draw counting line with optional label.
        
        Args:
            x: X coordinate for vertical line
            y: Y coordinate for horizontal line  
            color: Line color (BGR)
            thickness: Line thickness
            label: Optional label for the line
            
        Returns:
            Self for method chaining
        """
        try:
            h, w = self.image.shape[:2]
            thickness = thickness or max(self.lw, 3)
            
            if x is not None:
                # Draw vertical line
                cv2.line(self.image, (x, 0), (x, h), color, thickness)
                
                if label:
                    # Draw label at top of line
                    self.draw_text_at_position(label, (x + 5, 25), color)
            
            if y is not None:
                # Draw horizontal line
                cv2.line(self.image, (0, y), (w, y), color, thickness)
                
                if label:
                    # Draw label at left of line
                    self.draw_text_at_position(label, (5, y - 5), color)
                    
        except Exception as e:
            logger.error(f"Error drawing counting line: {e}")
        
        return self

    def draw_text_at_position(self, text: str, position: Tuple[int, int],
                             color: Tuple[int, int, int],
                             background: bool = True) -> 'EnhancedAnnotator':
        """
        Draw text at specific position.
        
        Args:
            text: Text to draw
            position: (x, y) position
            color: Text color (BGR)
            background: Whether to draw background
            
        Returns:
            Self for method chaining
        """
        try:
            x, y = position
            
            if background:
                # Get text size for background
                (text_width, text_height), baseline = self.text_calculator.get_text_size(
                    text, self.text_props.font_scale, self.text_props.thickness
                )
                
                # Draw background
                pad = 3
                cv2.rectangle(self.image, 
                             (x - pad, y - text_height - pad),
                             (x + text_width + pad, y + pad),
                             (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(self.image, text, (x, y),
                       self.text_props.font, self.text_props.font_scale,
                       color, self.text_props.thickness, cv2.LINE_AA)
            
        except Exception as e:
            logger.error(f"Error drawing text at position: {e}")
        
        return self

    def draw_statistics(self, stats: Dict[str, Any], position: Tuple[int, int] = (10, 30),
                       background: bool = True) -> 'EnhancedAnnotator':
        """
        Draw statistics overlay on image.
        
        Args:
            stats: Dictionary with statistics to display
            position: Starting position for text
            background: Whether to draw background
            
        Returns:
            Self for method chaining
        """
        try:
            x, y = position
            line_height = 25
            
            for i, (key, value) in enumerate(stats.items()):
                text = f"{key}: {value}"
                current_y = y + i * line_height
                
                self.draw_text_at_position(text, (x, current_y), 
                                         (255, 255, 255), background)
                
        except Exception as e:
            logger.error(f"Error drawing statistics: {e}")
        
        return self

    def draw_fps(self, fps: float, position: Tuple[int, int] = (10, 30)) -> 'EnhancedAnnotator':
        """
        Draw FPS counter.
        
        Args:
            fps: FPS value
            position: Position to draw at
            
        Returns:
            Self for method chaining
        """
        fps_text = f"FPS: {fps:.1f}"
        color = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 0, 255)
        self.draw_text_at_position(fps_text, position, color)
        return self

    def draw_confidence_bar(self, xyxy: List[float], confidence: float,
                           bar_width: int = 4) -> 'EnhancedAnnotator':
        """
        Draw confidence bar next to bounding box.
        
        Args:
            xyxy: Bounding box coordinates
            confidence: Confidence value (0-1)
            bar_width: Width of confidence bar
            
        Returns:
            Self for method chaining
        """
        try:
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Calculate bar position (right side of box)
            bar_x = x2 + 5
            bar_height = y2 - y1
            
            # Background bar (gray)
            cv2.rectangle(self.image, (bar_x, y1), (bar_x + bar_width, y2),
                         (128, 128, 128), -1)
            
            # Confidence bar (colored)
            conf_height = int(bar_height * confidence)
            conf_y = y2 - conf_height
            
            # Color based on confidence
            if confidence >= 0.8:
                color = (0, 255, 0)  # Green
            elif confidence >= 0.6:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            cv2.rectangle(self.image, (bar_x, conf_y), (bar_x + bar_width, y2),
                         color, -1)
            
        except Exception as e:
            logger.error(f"Error drawing confidence bar: {e}")
        
        return self

    def draw_center_point(self, xyxy: List[float], color: Tuple[int, int, int],
                         radius: int = 3) -> 'EnhancedAnnotator':
        """
        Draw center point of bounding box.
        
        Args:
            xyxy: Bounding box coordinates
            color: Point color (BGR)
            radius: Point radius
            
        Returns:
            Self for method chaining
        """
        try:
            x1, y1, x2, y2 = map(int, xyxy)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            cv2.circle(self.image, (center_x, center_y), radius, color, -1)
            
        except Exception as e:
            logger.error(f"Error drawing center point: {e}")
        
        return self

    def draw_trajectory(self, points: List[Tuple[int, int]], 
                       color: Tuple[int, int, int] = (255, 255, 0),
                       thickness: int = 2) -> 'EnhancedAnnotator':
        """
        Draw trajectory line from list of points.
        
        Args:
            points: List of (x, y) points
            color: Line color (BGR)
            thickness: Line thickness
            
        Returns:
            Self for method chaining
        """
        try:
            if len(points) < 2:
                return self
            
            for i in range(1, len(points)):
                cv2.line(self.image, points[i-1], points[i], color, thickness)
            
        except Exception as e:
            logger.error(f"Error drawing trajectory: {e}")
        
        return self

    def get_result(self) -> np.ndarray:
        """Get the annotated image."""
        return self.image

    def reset(self) -> 'EnhancedAnnotator':
        """Reset to original image."""
        self.image = self.original_image.copy()
        return self


# Legacy compatibility classes (based on your original code)
class GetTextSizeForLabel:
    """Legacy text size calculator for backward compatibility."""
    
    def __init__(self, label: str, im_shape: Tuple[int, ...], line_width: Optional[int] = None):
        self.label = label
        self.lw = line_width or max(round(sum(im_shape) / 2 * 0.003), 2)
        self.tf = max(self.lw - 1, 1)
        self.sf = self.lw / 3
    
    def getText(self) -> Tuple[Tuple[int, int], float, int]:
        """Get text size information."""
        size, _ = cv2.getTextSize(self.label, 0, fontScale=self.sf, thickness=self.tf)
        return size, self.sf, self.tf


class Annotator:
    """
    Legacy annotator class for backward compatibility with your original code.
    """

    def __init__(self, img: np.ndarray, xyxys: List[List[float]], 
                 classnames: List[str], confidences: List[float], 
                 colors: List[Tuple[int, int, int]], lw: int = 2):
        self.img = img
        self.xyxys = xyxys
        self.classnames = classnames
        self.confidences = confidences
        self.colors = colors
        self.lw = lw

    def drawClass(self) -> np.ndarray:
        """Draw class labels only."""
        for i in range(len(self.classnames)):
            # Draw bounding box
            p1 = (int(self.xyxys[i][0]), int(self.xyxys[i][1]))
            p2 = (int(self.xyxys[i][2]), int(self.xyxys[i][3]))
            cv2.rectangle(self.img, p1, p2, self.colors[i], int(round(self.lw)))
            
            # Draw text
            labelText = f'{self.classnames[i]}'
            (w, h), sf, tf = GetTextSizeForLabel(
                label=labelText, im_shape=self.img.shape, line_width=int(round(self.lw))
            ).getText()
            
            outside = p1[1] - h >= 3
            p2_text = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            
            cv2.rectangle(self.img, p1, p2_text, self.colors[i], -1, cv2.LINE_AA)
            cv2.putText(self.img, labelText, 
                       (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                       0, sf, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)
        
        return self.img

    def drawQuantity(self, classnames_xyxys_count_object: List[Dict[str, Any]]) -> np.ndarray:
        """Draw quantity labels only."""
        for i in range(len(classnames_xyxys_count_object)):
            # Draw bounding box
            p1 = (int(self.xyxys[i][0]), int(self.xyxys[i][1]))
            p2 = (int(self.xyxys[i][2]), int(self.xyxys[i][3]))
            cv2.rectangle(self.img, p1, p2, self.colors[i], int(round(self.lw)))
            
            # Draw text
            labelText = str(classnames_xyxys_count_object[i]['qty'])
            (w, h), sf, tf = GetTextSizeForLabel(
                label=labelText, im_shape=self.img.shape, line_width=int(round(self.lw))
            ).getText()
            
            outside = p1[1] - h >= 3
            p2_text = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            
            cv2.rectangle(self.img, p1, p2_text, self.colors[i], -1, cv2.LINE_AA)
            cv2.putText(self.img, labelText,
                       (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                       0, sf, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)
        
        return self.img
    
    def drawClassAndConfidence(self) -> np.ndarray:
        """Draw class labels with confidence scores."""
        for i in range(len(self.classnames)):
            # Draw bounding box
            p1 = (int(self.xyxys[i][0]), int(self.xyxys[i][1]))
            p2 = (int(self.xyxys[i][2]), int(self.xyxys[i][3]))
            cv2.rectangle(self.img, p1, p2, self.colors[i], int(round(self.lw)))
            
            # Draw text
            labelText = f'{self.classnames[i]}: {int(self.confidences[i] * 100)} %'
            (w, h), sf, tf = GetTextSizeForLabel(
                label=labelText, im_shape=self.img.shape, line_width=int(round(self.lw))
            ).getText()
            
            outside = p1[1] - h >= 3
            p2_text = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            
            cv2.rectangle(self.img, p1, p2_text, self.colors[i], -1, cv2.LINE_AA)
            cv2.putText(self.img, labelText,
                       (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                       0, sf, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)
        
        return self.img


# Create global color instance for backward compatibility
colors = Colors()


# Utility functions
def create_annotator(image: np.ndarray, line_width: Optional[int] = None) -> EnhancedAnnotator:
    """Create an enhanced annotator instance."""
    return EnhancedAnnotator(image, line_width)


def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]], 
                   show_confidence: bool = True, show_tracking: bool = False) -> np.ndarray:
    """
    Draw multiple detections on image.
    
    Args:
        image: Input image
        detections: List of detection dictionaries with keys:
                   - 'xyxy': bounding box coordinates
                   - 'class_name': class name
                   - 'confidence': confidence score
                   - 'track_id': tracking ID (optional)
        show_confidence: Whether to show confidence scores
        show_tracking: Whether to show tracking IDs
        
    Returns:
        Annotated image
    """
    annotator = create_annotator(image)
    
    for detection in detections:
        track_id = detection.get('track_id') if show_tracking else None
        
        annotator.draw_detection(
            detection['xyxy'],
            detection['class_name'],
            detection['confidence'],
            track_id,
            show_confidence
        )
    
    return annotator.get_result()


def draw_counting_statistics(image: np.ndarray, counts: Dict[str, int], 
                           position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """
    Draw counting statistics on image.
    
    Args:
        image: Input image
        counts: Dictionary with class counts
        position: Position to draw statistics
        
    Returns:
        Image with statistics overlay
    """
    annotator = create_annotator(image)
    annotator.draw_statistics(counts, position)
    return annotator.get_result()


def create_detection_summary_image(detections: List[Dict[str, Any]], 
                                 image_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
    """
    Create a summary image showing detection statistics.
    
    Args:
        detections: List of detections
        image_size: Size of output image
        
    Returns:
        Summary image
    """
    # Create blank image
    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    
    # Count detections by class
    class_counts = {}
    total_count = len(detections)
    
    for detection in detections:
        class_name = detection['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Create summary statistics
    stats = {'Total Detections': total_count}
    stats.update(class_counts)
    
    # Draw statistics
    annotator = EnhancedAnnotator(img)
    annotator.draw_statistics(stats, (20, 50))
    
    return annotator.get_result()