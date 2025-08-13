"""
Image processing utilities for Smart Recycling Detection System.

This module provides various image processing functions including
resizing, format conversion, enhancement, and PyQt5 integration.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, List
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import logging

from config.logging_config import get_logger

logger = get_logger('main')


def numpy_to_qimage(np_array: np.ndarray) -> QImage:
    """
    Convert numpy array to QImage for PyQt5 display.
    
    Args:
        np_array: Numpy array representing an image
        
    Returns:
        QImage object
    """
    try:
        if len(np_array.shape) == 3:
            h, w, ch = np_array.shape
            if ch == 3:  # RGB
                bytes_per_line = ch * w
                qt_image = QImage(np_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
            elif ch == 4:  # RGBA
                bytes_per_line = ch * w
                qt_image = QImage(np_array.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
            else:
                raise ValueError(f"Unsupported number of channels: {ch}")
        elif len(np_array.shape) == 2:  # Grayscale
            h, w = np_array.shape
            bytes_per_line = w
            qt_image = QImage(np_array.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            raise ValueError(f"Unsupported array shape: {np_array.shape}")
        
        return qt_image
        
    except Exception as e:
        logger.error(f"Error converting numpy array to QImage: {e}")
        # Return a blank image as fallback
        return QImage(640, 480, QImage.Format_RGB888)


def numpy_to_qpixmap(np_array: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> QPixmap:
    """
    Convert numpy array to QPixmap for PyQt5 display.
    
    Args:
        np_array: Numpy array representing an image
        width: Target width for scaling (optional)
        height: Target height for scaling (optional)
        
    Returns:
        QPixmap object
    """
    try:
        qt_image = numpy_to_qimage(np_array)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale if dimensions provided
        if width is not None or height is not None:
            if width is None:
                width = pixmap.width()
            if height is None:
                height = pixmap.height()
            pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        return pixmap
        
    except Exception as e:
        logger.error(f"Error converting numpy array to QPixmap: {e}")
        # Return a blank pixmap as fallback
        return QPixmap(640, 480)


def resize_image(image: np.ndarray, target_size: Union[int, Tuple[int, int]], maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image as numpy array
        target_size: Target size as (width, height) or single dimension
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    try:
        h, w = image.shape[:2]
        
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        
        target_w, target_h = target_size
        
        if maintain_aspect:
            # Calculate scaling factor
            scale_w = target_w / w
            scale_h = target_h / h
            scale = min(scale_w, scale_h)
            
            # Calculate new dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create padded image if needed
            if new_w != target_w or new_h != target_h:
                # Calculate padding
                pad_w = target_w - new_w
                pad_h = target_h - new_h
                top, bottom = pad_h // 2, pad_h - (pad_h // 2)
                left, right = pad_w // 2, pad_w - (pad_w // 2)
                
                # Add padding
                if len(image.shape) == 3:
                    color = [0, 0, 0]  # Black padding for color images
                else:
                    color = 0  # Black padding for grayscale
                
                resized = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                           cv2.BORDER_CONSTANT, value=color)
        else:
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        return resized
        
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return image


def enhance_image(image: np.ndarray, brightness: float = 0, contrast: float = 1.0, saturation: float = 1.0) -> np.ndarray:
    """
    Enhance image with brightness, contrast, and saturation adjustments.
    
    Args:
        image: Input image
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast multiplier (0.5 to 3.0)
        saturation: Saturation multiplier (0.0 to 2.0)
        
    Returns:
        Enhanced image
    """
    try:
        enhanced = image.copy()
        
        # Brightness adjustment
        if brightness != 0:
            enhanced = cv2.add(enhanced, np.ones(enhanced.shape, dtype=np.uint8) * brightness)
        
        # Contrast adjustment
        if contrast != 1.0:
            enhanced = cv2.multiply(enhanced, np.ones(enhanced.shape) * contrast)
        
        # Saturation adjustment (for color images)
        if saturation != 1.0 and len(enhanced.shape) == 3:
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
        
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return image


def draw_detection_line(image: np.ndarray, x: Optional[int] = None, y: Optional[int] = None, 
                       color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw detection line on image.
    
    Args:
        image: Input image
        x: X coordinate for vertical line
        y: Y coordinate for horizontal line
        color: Line color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with detection line
    """
    try:
        result = image.copy()
        h, w = result.shape[:2]
        
        if x is not None:
            # Draw vertical line
            cv2.line(result, (x, 0), (x, h), color, thickness)
        
        if y is not None:
            # Draw horizontal line
            cv2.line(result, (0, y), (w, y), color, thickness)
        
        return result
        
    except Exception as e:
        logger.error(f"Error drawing detection line: {e}")
        return image


def create_blank_image(width: int, height: int, color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Create a blank image with specified dimensions and color.
    
    Args:
        width: Image width
        height: Image height
        color: Background color (BGR)
        
    Returns:
        Blank image
    """
    try:
        image = np.full((height, width, 3), color, dtype=np.uint8)
        return image
        
    except Exception as e:
        logger.error(f"Error creating blank image: {e}")
        return np.zeros((height, width, 3), dtype=np.uint8)


def add_text_overlay(image: np.ndarray, text: str, position: Tuple[int, int], 
                    font_scale: float = 1.0, color: Tuple[int, int, int] = (255, 255, 255),
                    thickness: int = 2, background: bool = True) -> np.ndarray:
    """
    Add text overlay to image.
    
    Args:
        image: Input image
        text: Text to add
        position: Text position (x, y)
        font_scale: Font scale
        color: Text color (BGR)
        thickness: Text thickness
        background: Whether to add background rectangle
        
    Returns:
        Image with text overlay
    """
    try:
        result = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x, y = position
        
        if background:
            # Draw background rectangle
            cv2.rectangle(result, 
                         (x - 5, y - text_height - 5),
                         (x + text_width + 5, y + baseline + 5),
                         (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(result, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        return result
        
    except Exception as e:
        logger.error(f"Error adding text overlay: {e}")
        return image


def save_image(image: np.ndarray, filepath: Union[str, Path], quality: int = 95) -> bool:
    """
    Save image to file.
    
    Args:
        image: Image to save
        filepath: Output file path
        quality: JPEG quality (0-100)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Set compression parameters
        if filepath.suffix.lower() in ['.jpg', '.jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif filepath.suffix.lower() == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        else:
            params = []
        
        success = cv2.imwrite(str(filepath), image, params)
        
        if success:
            logger.info(f"Image saved to {filepath}")
        else:
            logger.error(f"Failed to save image to {filepath}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return False


def load_image(filepath: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load image from file.
    
    Args:
        filepath: Image file path
        
    Returns:
        Loaded image or None if failed
    """
    try:
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"Image file not found: {filepath}")
            return None
        
        image = cv2.imread(str(filepath))
        
        if image is None:
            logger.error(f"Failed to load image: {filepath}")
            return None
        
        logger.debug(f"Image loaded: {filepath} - Shape: {image.shape}")
        return image
        
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None


def convert_color_space(image: np.ndarray, conversion: str) -> np.ndarray:
    """
    Convert image color space.
    
    Args:
        image: Input image
        conversion: Conversion type ('BGR2RGB', 'RGB2BGR', 'BGR2GRAY', etc.)
        
    Returns:
        Converted image
    """
    try:
        conversion_code = getattr(cv2, f"COLOR_{conversion}", None)
        
        if conversion_code is None:
            logger.error(f"Invalid color conversion: {conversion}")
            return image
        
        converted = cv2.cvtColor(image, conversion_code)
        return converted
        
    except Exception as e:
        logger.error(f"Error converting color space: {e}")
        return image


def calculate_image_statistics(image: np.ndarray) -> dict:
    """
    Calculate basic image statistics.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image statistics
    """
    try:
        stats = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'size_bytes': image.nbytes,
            'min_value': int(np.min(image)),
            'max_value': int(np.max(image)),
            'mean_value': float(np.mean(image)),
            'std_value': float(np.std(image))
        }
        
        if len(image.shape) == 3:
            stats['channels'] = image.shape[2]
            stats['channel_means'] = [float(np.mean(image[:, :, i])) for i in range(image.shape[2])]
        else:
            stats['channels'] = 1
            stats['channel_means'] = [stats['mean_value']]
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating image statistics: {e}")
        return {}


def create_grid_image(images: List[np.ndarray], grid_size: Optional[Tuple[int, int]] = None,
                     padding: int = 10, background_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Create a grid image from multiple images.
    
    Args:
        images: List of images
        grid_size: Grid dimensions (rows, cols). If None, auto-calculate
        padding: Padding between images
        background_color: Background color
        
    Returns:
        Grid image
    """
    try:
        if not images:
            return create_blank_image(640, 480)
        
        # Calculate grid size if not provided
        if grid_size is None:
            num_images = len(images)
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
            grid_size = (rows, cols)
        
        rows, cols = grid_size
        
        # Resize all images to same size
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        
        resized_images = []
        for img in images:
            resized = resize_image(img, (max_w, max_h), maintain_aspect=True)
            resized_images.append(resized)
        
        # Fill remaining slots with blank images
        while len(resized_images) < rows * cols:
            blank = create_blank_image(max_w, max_h, background_color)
            resized_images.append(blank)
        
        # Create grid
        grid_h = rows * max_h + (rows + 1) * padding
        grid_w = cols * max_w + (cols + 1) * padding
        grid = np.full((grid_h, grid_w, 3), background_color, dtype=np.uint8)
        
        for i, img in enumerate(resized_images[:rows * cols]):
            row = i // cols
            col = i % cols
            
            y_start = padding + row * (max_h + padding)
            y_end = y_start + max_h
            x_start = padding + col * (max_w + padding)
            x_end = x_start + max_w
            
            grid[y_start:y_end, x_start:x_end] = img
        
        return grid
        
    except Exception as e:
        logger.error(f"Error creating grid image: {e}")
        return create_blank_image(640, 480)