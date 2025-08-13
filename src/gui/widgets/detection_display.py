"""
Detection display widget for Smart Recycling Detection System.

This module provides a sophisticated video display widget with
real-time detection visualization, statistics overlay, and user controls.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGroupBox, QFileDialog, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QPixmap, QFont, QPalette

from config.settings import get_config
from config.logging_config import get_logger
from src.utils.image_utils import numpy_to_qpixmap, create_blank_image
from src.utils.file_utils import is_video_file, is_model_file

logger = get_logger('gui')


class ScalableLabel(QLabel):
    """Label that maintains aspect ratio when scaling images."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignCenter)
        self._pixmap = None
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def setPixmap(self, pixmap: QPixmap):
        """Set pixmap with aspect ratio preservation."""
        self._pixmap = pixmap
        self._update_pixmap()
    
    def _update_pixmap(self):
        """Update displayed pixmap based on current size."""
        if self._pixmap is None:
            return
        
        # Scale pixmap to fit label while maintaining aspect ratio
        scaled_pixmap = self._pixmap.scaled(
            self.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        super().setPixmap(scaled_pixmap)
    
    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self._update_pixmap()


class FileSelectionWidget(QWidget):
    """Widget for file selection with drag-and-drop support."""
    
    fileSelected = pyqtSignal(str)  # Emitted when file is selected
    
    def __init__(self, file_type: str = "model", parent=None):
        """
        Initialize file selection widget.
        
        Args:
            file_type: Type of file to select ("model", "video")
            parent: Parent widget
        """
        super().__init__(parent)
        self.file_type = file_type
        self.current_file = None
        self.config = get_config()
        
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # File path display
        self.path_display = QLabel("No file selected")
        self.path_display.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 2px solid #E0E0E0;
                border-radius: 6px;
                padding: 8px 12px;
                color: #424242;
            }
        """)
        self.path_display.setMinimumHeight(32)
        
        # Browse button
        self.browse_button = QPushButton(f"Browse {self.file_type.title()}")
        self.browse_button.setMinimumHeight(32)
        self.browse_button.setMaximumWidth(150)
        
        # Apply styling based on file type
        if self.file_type == "model":
            self.browse_button.setProperty("class", "success")
        elif self.file_type == "video":
            self.browse_button.setProperty("class", "warning")
        
        layout.addWidget(self.path_display, 1)
        layout.addWidget(self.browse_button, 0)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
    
    def setup_connections(self):
        """Set up signal connections."""
        self.browse_button.clicked.connect(self.browse_file)
    
    def browse_file(self):
        """Open file browser dialog."""
        try:
            if self.file_type == "model":
                file_filter = "Model files (*.pt *.pth *.onnx);;All files (*)"
                start_dir = str(self.config.paths.models_dir)
            elif self.file_type == "video":
                file_filter = "Video files (*.mp4 *.avi *.mkv *.mov);;All files (*)"
                start_dir = str(self.config.paths.sample_videos_dir)
            else:
                file_filter = "All files (*)"
                start_dir = str(self.config.paths.project_root)
            
            filepath, _ = QFileDialog.getOpenFileName(
                self, 
                f"Select {self.file_type} file",
                start_dir,
                file_filter
            )
            
            if filepath:
                self.set_file(filepath)
                
        except Exception as e:
            logger.error(f"Error browsing for {self.file_type} file: {e}")
    
    def set_file(self, filepath: str):
        """Set the selected file."""
        try:
            file_path = Path(filepath)
            
            # Validate file type
            if self.file_type == "model" and not is_model_file(file_path):
                logger.warning(f"Invalid model file: {filepath}")
                return
            elif self.file_type == "video" and not is_video_file(file_path):
                logger.warning(f"Invalid video file: {filepath}")
                return
            
            self.current_file = filepath
            
            # Update display
            display_text = file_path.name
            if len(display_text) > 50:
                display_text = "..." + display_text[-47:]
            
            self.path_display.setText(display_text)
            self.path_display.setToolTip(filepath)
            
            # Emit signal
            self.fileSelected.emit(filepath)
            
            logger.info(f"{self.file_type.title()} file selected: {filepath}")
            
        except Exception as e:
            logger.error(f"Error setting {self.file_type} file: {e}")
    
    def get_file(self) -> Optional[str]:
        """Get currently selected file."""
        return self.current_file
    
    def clear_file(self):
        """Clear selected file."""
        self.current_file = None
        self.path_display.setText("No file selected")
        self.path_display.setToolTip("")
    
    def dragEnterEvent(self, event):
        """Handle drag enter events."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        """Handle drop events."""
        try:
            urls = event.mimeData().urls()
            if urls:
                filepath = urls[0].toLocalFile()
                self.set_file(filepath)
                event.accept()
        except Exception as e:
            logger.error(f"Error handling drop event: {e}")
            event.ignore()


class DetectionDisplayWidget(QWidget):
    """
    Main detection display widget combining video display with controls.
    
    This widget provides:
    - Scalable video display
    - File selection for models and videos
    - Real-time statistics display
    - Performance metrics
    """
    
    # Signals
    modelSelected = pyqtSignal(str)
    videoSelected = pyqtSignal(str)
    frameDisplayed = pyqtSignal(np.ndarray)
    
    def __init__(self, parent=None):
        """Initialize detection display widget."""
        super().__init__(parent)
        self.config = get_config()
        
        # State
        self.current_frame = None
        self.display_fps = 0.0
        self.last_update_time = 0.0
        
        self.setup_ui()
        self.setup_connections()
        self.setup_placeholder()
        
        logger.info("Detection display widget initialized")
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        
        # File selection section
        file_selection_group = QGroupBox("File Selection")
        file_selection_layout = QVBoxLayout(file_selection_group)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_selector = FileSelectionWidget("model")
        model_layout.addWidget(self.model_selector)
        file_selection_layout.addLayout(model_layout)
        
        # Video selection
        video_layout = QHBoxLayout()
        video_layout.addWidget(QLabel("Video:"))
        self.video_selector = FileSelectionWidget("video")
        video_layout.addWidget(self.video_selector)
        file_selection_layout.addLayout(video_layout)
        
        main_layout.addWidget(file_selection_group)
        
        # Display section
        display_group = QGroupBox("Detection Display")
        display_layout = QVBoxLayout(display_group)
        
        # Video display
        self.video_display = ScalableLabel()
        self.video_display.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 2px solid #E0E0E0;
                border-radius: 8px;
            }
        """)
        self.video_display.setMinimumSize(640, 480)
        
        display_layout.addWidget(self.video_display)
        
        # Display info bar
        info_layout = QHBoxLayout()
        
        self.fps_label = QLabel("FPS: 0.0")
        self.resolution_label = QLabel("Resolution: N/A")
        self.frame_label = QLabel("Frame: 0")
        
        info_layout.addWidget(self.fps_label)
        info_layout.addWidget(self.resolution_label)
        info_layout.addWidget(self.frame_label)
        info_layout.addStretch()
        
        display_layout.addLayout(info_layout)
        main_layout.addWidget(display_group)
        
        # Set proportions
        main_layout.setStretchFactor(file_selection_group, 0)
        main_layout.setStretchFactor(display_group, 1)
    
    def setup_connections(self):
        """Set up signal connections."""
        self.model_selector.fileSelected.connect(self.modelSelected.emit)
        self.video_selector.fileSelected.connect(self.videoSelected.emit)
    
    def setup_placeholder(self):
        """Set up placeholder display."""
        placeholder_image = create_blank_image(640, 480, (64, 64, 64))
        
        # Add placeholder text
        cv2.putText(
            placeholder_image,
            "Smart Recycling Detection",
            (160, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        cv2.putText(
            placeholder_image,
            "Select model and video to start",
            (180, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        
        self.update_display(placeholder_image)
    
    def update_display(self, frame: np.ndarray):
        """
        Update the video display with new frame.
        
        Args:
            frame: New frame to display
        """
        try:
            self.current_frame = frame.copy()
            
            # Convert to QPixmap and display
            pixmap = numpy_to_qpixmap(frame)
            self.video_display.setPixmap(pixmap)
            
            # Update resolution label
            h, w = frame.shape[:2]
            self.resolution_label.setText(f"Resolution: {w}x{h}")
            
            # Emit signal
            self.frameDisplayed.emit(frame)
            
        except Exception as e:
            logger.error(f"Error updating display: {e}")
    
    def update_fps(self, fps: float):
        """Update FPS display."""
        self.display_fps = fps
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
        # Color code FPS label based on performance
        if fps >= 20:
            color = "color: #4CAF50;"  # Green
        elif fps >= 10:
            color = "color: #FF9800;"  # Orange
        else:
            color = "color: #F44336;"  # Red
        
        self.fps_label.setStyleSheet(f"QLabel {{ {color} font-weight: bold; }}")
    
    def update_frame_count(self, frame_count: int):
        """Update frame count display."""
        self.frame_label.setText(f"Frame: {frame_count:,}")
    
    def get_selected_model(self) -> Optional[str]:
        """Get currently selected model file."""
        return self.model_selector.get_file()
    
    def get_selected_video(self) -> Optional[str]:
        """Get currently selected video file."""
        return self.video_selector.get_file()
    
    def set_model_file(self, filepath: str):
        """Set model file programmatically."""
        self.model_selector.set_file(filepath)
    
    def set_video_file(self, filepath: str):
        """Set video file programmatically."""
        self.video_selector.set_file(filepath)
    
    def clear_selections(self):
        """Clear all file selections."""
        self.model_selector.clear_file()
        self.video_selector.clear_file()
        self.setup_placeholder()
    
    def save_current_frame(self) -> bool:
        """Save currently displayed frame."""
        try:
            if self.current_frame is None:
                logger.warning("No frame to save")
                return False
            
            # Open save dialog
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Save Current Frame",
                f"frame_{int(self.last_update_time)}.png",
                "Image files (*.png *.jpg *.jpeg);;All files (*)"
            )
            
            if filepath:
                success = cv2.imwrite(filepath, self.current_frame)
                if success:
                    logger.info(f"Frame saved: {filepath}")
                    return True
                else:
                    logger.error(f"Failed to save frame: {filepath}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            return False


class DetectionInfoPanel(QFrame):
    """Panel displaying detection information and statistics."""
    
    def __init__(self, parent=None):
        """Initialize detection info panel."""
        super().__init__(parent)
        self.setup_ui()
        self.reset_display()
    
    def setup_ui(self):
        """Set up the user interface."""
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Detection Information")
        title.setProperty("class", "subheading")
        layout.addWidget(title)
        
        # Statistics display
        self.stats_layout = QVBoxLayout()
        layout.addLayout(self.stats_layout)
        
        layout.addStretch()
    
    def update_detection_info(self, detections: int, processing_time: float,
                            class_counts: Dict[str, int]):
        """
        Update detection information display.
        
        Args:
            detections: Number of detections
            processing_time: Processing time in seconds
            class_counts: Count per class
        """
        try:
            # Clear previous stats
            self.clear_stats()
            
            # Add new statistics
            self.add_stat("Total Detections", str(detections))
            self.add_stat("Processing Time", f"{processing_time:.3f}s")
            self.add_stat("Speed", f"{1/processing_time:.1f} FPS" if processing_time > 0 else "N/A")
            
            # Add class counts
            if class_counts:
                self.add_separator()
                for class_name, count in class_counts.items():
                    display_name = class_name.replace('-', ' ').title()
                    self.add_stat(display_name, str(count))
            
        except Exception as e:
            logger.error(f"Error updating detection info: {e}")
    
    def add_stat(self, label: str, value: str):
        """Add a statistic row."""
        row_layout = QHBoxLayout()
        
        label_widget = QLabel(f"{label}:")
        label_widget.setAlignment(Qt.AlignLeft)
        
        value_widget = QLabel(value)
        value_widget.setAlignment(Qt.AlignRight)
        value_widget.setStyleSheet("font-weight: bold;")
        
        row_layout.addWidget(label_widget)
        row_layout.addWidget(value_widget)
        
        self.stats_layout.addLayout(row_layout)
    
    def add_separator(self):
        """Add a visual separator."""
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("color: #E0E0E0;")
        self.stats_layout.addWidget(separator)
    
    def clear_stats(self):
        """Clear all statistics."""
        while self.stats_layout.count():
            child = self.stats_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self.clear_layout(child.layout())
    
    def clear_layout(self, layout):
        """Recursively clear a layout."""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self.clear_layout(child.layout())
    
    def reset_display(self):
        """Reset display to default state."""
        self.clear_stats()
        self.add_stat("Status", "Ready")
        self.add_stat("Detections", "0")
        self.add_stat("Processing Time", "0.000s")


class PerformanceMonitorWidget(QFrame):
    """Widget for displaying real-time performance metrics."""
    
    def __init__(self, parent=None):
        """Initialize performance monitor widget."""
        super().__init__(parent)
        self.setup_ui()
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self.update_display)
        self.performance_timer.start(1000)  # Update every second
    
    def setup_ui(self):
        """Set up the user interface."""
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #F8F9FA;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Performance Monitor")
        title.setProperty("class", "subheading")
        layout.addWidget(title)
        
        # Metrics
        self.fps_label = QLabel("FPS: 0.0")
        self.cpu_label = QLabel("CPU: 0%")
        self.memory_label = QLabel("Memory: 0 MB")
        self.gpu_label = QLabel("GPU: N/A")
        
        for label in [self.fps_label, self.cpu_label, self.memory_label, self.gpu_label]:
            label.setFont(QFont("Consolas", 9))
            layout.addWidget(label)
        
        layout.addStretch()
    
    def update_performance(self, metrics: Dict[str, Any]):
        """
        Update performance metrics display.
        
        Args:
            metrics: Dictionary with performance data
        """
        try:
            # Update FPS
            fps = metrics.get('current_fps', 0.0)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            
            # Update processing time
            proc_time = metrics.get('processing_time', 0.0)
            if proc_time > 0:
                theoretical_fps = 1.0 / proc_time
                self.fps_label.setText(f"FPS: {fps:.1f} (Max: {theoretical_fps:.1f})")
            
        except Exception as e:
            logger.error(f"Error updating performance display: {e}")
    
    def update_display(self):
        """Update display with system metrics."""
        try:
            # Get system metrics (simplified)
            import psutil
            
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            memory_mb = memory_info.used / (1024 * 1024)
            
            self.cpu_label.setText(f"CPU: {cpu_percent:.1f}%")
            self.memory_label.setText(f"Memory: {memory_mb:.0f} MB")
            
            # GPU info (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    self.gpu_label.setText(f"GPU: {gpu_memory:.0f} MB")
                else:
                    self.gpu_label.setText("GPU: Not Available")
            except ImportError:
                self.gpu_label.setText("GPU: N/A")
                
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")