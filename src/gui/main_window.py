"""
Main application window for Smart Recycling Detection System.

This module provides the main GUI window, integrating all components
and managing the overall application flow. Enhanced version of the original GUIPro.py.
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QMenuBar, QStatusBar, QAction, QMessageBox,
    QProgressBar, QLabel, QApplication
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QIcon, QKeySequence, QCloseEvent

from config.settings import get_config
from config.logging_config import get_logger
from src.core.detector import RecyclingDetector
from src.core.counter import RecyclingCounter
from src.core.video_processor import VideoProcessor, create_processor
from src.gui.widgets.detection_display import DetectionDisplayWidget
from src.gui.widgets.control_panel import ControlPanelWidget
from src.gui.styles.modern_style import theme_manager, apply_modern_style

logger = get_logger('gui')


class MainWindow(QMainWindow):
    """
    Main application window.
    
    This is the enhanced version of your original GUIPro.py, providing:
    - Modern, responsive interface
    - Real-time detection and counting
    - Professional error handling
    - Performance monitoring
    - Activity logging
    """
    
    def __init__(self):
        """Initialize main window."""
        super().__init__()
        
        self.config = get_config()
        
        # Core components
        self.detector: Optional[RecyclingDetector] = None
        self.counter: Optional[RecyclingCounter] = None
        self.video_processor: Optional[VideoProcessor] = None
        
        # State management
        self.is_processing = False
        self.is_paused = False
        self.current_model_path = None
        self.current_video_path = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        
        # Set up UI
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
        self.setup_connections()
        self.apply_styling()
        
        # Initialize components
        self.initialize_components()
        
        logger.info("Main window initialized")
    
    def setup_ui(self):
        """Set up the main user interface."""
        # Set window properties
        self.setWindowTitle(self.config.ui.window_title)
        self.setMinimumSize(1200, 800)
        self.resize(self.config.ui.window_width, self.config.ui.window_height)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Detection display
        self.detection_display = DetectionDisplayWidget()
        splitter.addWidget(self.detection_display)
        
        # Right panel - Control panel
        self.control_panel = ControlPanelWidget()
        self.control_panel.setMaximumWidth(400)
        self.control_panel.setMinimumWidth(350)
        splitter.addWidget(self.control_panel)
        
        # Set splitter proportions (70% display, 30% controls)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)
    
    def setup_menu_bar(self):
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # Load model action
        load_model_action = QAction('&Load Model...', self)
        load_model_action.setShortcut(QKeySequence.Open)
        load_model_action.triggered.connect(self._load_model_dialog)
        file_menu.addAction(load_model_action)
        
        # Load video action
        load_video_action = QAction('Load &Video...', self)
        load_video_action.setShortcut('Ctrl+V')
        load_video_action.triggered.connect(self._load_video_dialog)
        file_menu.addAction(load_video_action)
        
        file_menu.addSeparator()
        
        # Save settings action
        save_settings_action = QAction('&Save Settings...', self)
        save_settings_action.setShortcut('Ctrl+S')
        save_settings_action.triggered.connect(self._save_settings)
        file_menu.addAction(save_settings_action)
        
        # Load settings action
        load_settings_action = QAction('&Load Settings...', self)
        load_settings_action.triggered.connect(self._load_settings)
        file_menu.addAction(load_settings_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        # Theme submenu
        theme_menu = view_menu.addMenu('&Theme')
        
        for theme_name in theme_manager.get_available_themes():
            theme_action = QAction(theme_name.title(), self)
            theme_action.triggered.connect(lambda checked, t=theme_name: self._change_theme(t))
            theme_menu.addAction(theme_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Set up the status bar."""
        self.status_bar = self.statusBar()
        
        # Status message
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Performance indicator
        self.performance_label = QLabel("FPS: 0.0")
        self.status_bar.addPermanentWidget(self.performance_label)
    
    def setup_connections(self):
        """Set up signal connections between components."""
        # Detection display signals
        self.detection_display.modelSelected.connect(self._on_model_selected)
        self.detection_display.videoSelected.connect(self._on_video_selected)
        
        # Control panel signals
        self.control_panel.startRequested.connect(self._start_processing)
        self.control_panel.stopRequested.connect(self._stop_processing)
        self.control_panel.pauseRequested.connect(self._pause_processing)
        self.control_panel.resumeRequested.connect(self._resume_processing)
        self.control_panel.resetRequested.connect(self._reset_all)
        self.control_panel.settingsChanged.connect(self._on_settings_changed)
    
    def apply_styling(self):
        """Apply modern styling to the application."""
        try:
            app = QApplication.instance()
            if app:
                apply_modern_style(app)
            logger.debug("Modern styling applied")
        except Exception as e:
            logger.error(f"Error applying styling: {e}")
    
    def initialize_components(self):
        """Initialize core components."""
        try:
            # Initialize counter
            self.counter = RecyclingCounter()
            
            # Add initial log entry
            self.control_panel.add_log_entry("Application started", "SUCCESS")
            self.control_panel.add_log_entry("Ready to load model and video", "INFO")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            self._show_error("Initialization Error", str(e))
    
    def _load_model_dialog(self):
        """Open model loading dialog."""
        self.detection_display.model_selector.browse_file()
    
    def _load_video_dialog(self):
        """Open video loading dialog."""
        self.detection_display.video_selector.browse_file()
    
    @pyqtSlot(str)
    def _on_model_selected(self, model_path: str):
        """Handle model selection."""
        try:
            self.current_model_path = model_path
            
            # Load detector
            if self.detector:
                self.detector.cleanup()
            
            self.detector = RecyclingDetector()
            success = self.detector.load_model(model_path)
            
            if success:
                self.control_panel.add_log_entry(f"Model loaded: {Path(model_path).name}", "SUCCESS")
                self.status_label.setText(f"Model loaded: {Path(model_path).name}")
                
                # Get model info
                model_info = self.detector.get_model_info()
                self.control_panel.add_log_entry(
                    f"Classes: {', '.join(model_info.get('class_names', []))}", "INFO"
                )
            else:
                self.control_panel.add_log_entry(f"Failed to load model: {Path(model_path).name}", "ERROR")
                self._show_error("Model Loading Error", f"Failed to load model from {model_path}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.control_panel.add_log_entry(f"Model loading error: {e}", "ERROR")
            self._show_error("Model Loading Error", str(e))
    
    @pyqtSlot(str)
    def _on_video_selected(self, video_path: str):
        """Handle video selection."""
        try:
            self.current_video_path = video_path
            self.control_panel.add_log_entry(f"Video selected: {Path(video_path).name}", "SUCCESS")
            self.status_label.setText(f"Video ready: {Path(video_path).name}")
            
        except Exception as e:
            logger.error(f"Error selecting video: {e}")
            self.control_panel.add_log_entry(f"Video selection error: {e}", "ERROR")
    
    @pyqtSlot()
    def _start_processing(self):
        """Start detection processing."""
        try:
            if not self.detector or not self.detector.is_loaded:
                self._show_error("Detection Error", "Please load a model first")
                return
            
            if not self.current_video_path:
                self._show_error("Detection Error", "Please select a video file first")
                return
            
            # Create video processor
            self.video_processor = create_processor(
                self.detector, self.counter, self.current_video_path
            )
            
            # Connect processor signals
            self.video_processor.frameProcessed.connect(self._on_frame_processed)
            self.video_processor.statisticsUpdated.connect(self._on_statistics_updated)
            self.video_processor.performanceUpdated.connect(self._on_performance_updated)
            self.video_processor.errorOccurred.connect(self._on_processing_error)
            self.video_processor.processingFinished.connect(self._on_processing_finished)
            
            # Start processing
            self.video_processor.start_processing()
            
            # Update UI state
            self.is_processing = True
            self.is_paused = False
            self.frame_count = 0
            self.start_time = time.time()
            
            self.control_panel.set_processing_state(True, False)
            self.status_label.setText("Processing...")
            self.progress_bar.setVisible(True)
            
        except Exception as e:
            logger.error(f"Error starting processing: {e}")
            self.control_panel.add_log_entry(f"Start error: {e}", "ERROR")
            self._show_error("Processing Error", str(e))
    
    @pyqtSlot()
    def _stop_processing(self):
        """Stop detection processing."""
        try:
            if self.video_processor:
                self.video_processor.stop_processing()
            
            self._cleanup_processing()
            
        except Exception as e:
            logger.error(f"Error stopping processing: {e}")
            self.control_panel.add_log_entry(f"Stop error: {e}", "ERROR")
    
    @pyqtSlot()
    def _pause_processing(self):
        """Pause detection processing."""
        try:
            if self.video_processor and self.is_processing:
                self.video_processor.pause_processing()
                self.is_paused = True
                self.control_panel.set_processing_state(True, True)
                self.status_label.setText("Paused")
                
        except Exception as e:
            logger.error(f"Error pausing processing: {e}")
            self.control_panel.add_log_entry(f"Pause error: {e}", "ERROR")
    
    @pyqtSlot()
    def _resume_processing(self):
        """Resume detection processing."""
        try:
            if self.video_processor and self.is_processing and self.is_paused:
                self.video_processor.resume_processing()
                self.is_paused = False
                self.control_panel.set_processing_state(True, False)
                self.status_label.setText("Processing...")
                
        except Exception as e:
            logger.error(f"Error resuming processing: {e}")
            self.control_panel.add_log_entry(f"Resume error: {e}", "ERROR")
    
    @pyqtSlot()
    def _reset_all(self):
        """Reset all components and statistics."""
        try:
            # Stop processing if running
            if self.is_processing:
                self._stop_processing()
            
            # Reset counter
            if self.counter:
                self.counter.reset()
            
            # Reset displays
            self.control_panel.reset_counters()
            self.detection_display.setup_placeholder()
            
            # Reset state
            self.frame_count = 0
            self.start_time = None
            
            # Update UI
            self.status_label.setText("Reset completed")
            self.performance_label.setText("FPS: 0.0")
            
            self.control_panel.add_log_entry("System reset completed", "SUCCESS")
            
            logger.info("System reset completed")
            
        except Exception as e:
            logger.error(f"Error during reset: {e}")
            self.control_panel.add_log_entry(f"Reset error: {e}", "ERROR")
            self._show_error("Reset Error", str(e))
    
    def _cleanup_processing(self):
        """Clean up processing resources."""
        try:
            self.is_processing = False
            self.is_paused = False
            
            # Update UI state
            self.control_panel.set_processing_state(False)
            self.status_label.setText("Stopped")
            self.progress_bar.setVisible(False)
            
            # Clean up video processor
            if self.video_processor:
                self.video_processor = None
            
            logger.info("Processing cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    @pyqtSlot(np.ndarray)
    def _on_frame_processed(self, frame):
        """Handle processed frame from video processor."""
        try:
            self.frame_count += 1
            
            # Update display
            self.detection_display.update_display(frame)
            self.detection_display.update_frame_count(self.frame_count)
            
            # Calculate and display FPS
            if self.start_time:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                self.detection_display.update_fps(fps)
                self.performance_label.setText(f"FPS: {fps:.1f}")
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    @pyqtSlot(dict)
    def _on_statistics_updated(self, stats):
        """Handle updated counting statistics."""
        try:
            self.control_panel.update_counters(stats)
            
            # Log significant count changes
            total = sum(stats.values())
            if total > 0 and total % 10 == 0:  # Log every 10 detections
                self.control_panel.add_log_entry(f"Total count reached: {total}", "INFO")
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    @pyqtSlot(dict)
    def _on_performance_updated(self, metrics):
        """Handle updated performance metrics."""
        try:
            fps = metrics.get('current_fps', 0.0)
            if fps > 0:
                self.performance_label.setText(f"FPS: {fps:.1f}")
            
            # Update progress bar based on video progress (if applicable)
            frame_count = metrics.get('frame_count', 0)
            if frame_count > 0:
                self.progress_bar.setValue(min(frame_count % 100, 99))
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    @pyqtSlot(str)
    def _on_processing_error(self, error_message):
        """Handle processing errors."""
        self.control_panel.add_log_entry(f"Processing error: {error_message}", "ERROR")
        self._show_error("Processing Error", error_message)
    
    @pyqtSlot()
    def _on_processing_finished(self):
        """Handle processing completion."""
        self.control_panel.add_log_entry("Processing completed", "SUCCESS")
        self._cleanup_processing()
    
    @pyqtSlot(dict)
    def _on_settings_changed(self, settings):
        """Handle settings changes from control panel."""
        try:
            # Update detector settings
            if self.detector and 'confidence_threshold' in settings:
                self.detector.set_confidence_threshold(settings['confidence_threshold'])
            
            # Update counter settings
            if self.counter and 'line_position_x' in settings:
                self.counter.set_counting_line(x=settings['line_position_x'])
            
            self.control_panel.add_log_entry("Settings updated", "INFO")
            
        except Exception as e:
            logger.error(f"Error applying settings: {e}")
            self.control_panel.add_log_entry(f"Settings error: {e}", "ERROR")
    
    def _change_theme(self, theme_name: str):
        """Change application theme."""
        try:
            app = QApplication.instance()
            theme_manager.apply_theme(app, theme_name)
            self.control_panel.add_log_entry(f"Theme changed to: {theme_name}", "INFO")
            
        except Exception as e:
            logger.error(f"Error changing theme: {e}")
            self.control_panel.add_log_entry(f"Theme change error: {e}", "ERROR")
    
    def _save_settings(self):
        """Save current settings to file."""
        try:
            from PyQt5.QtWidgets import QFileDialog
            
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Save Settings",
                "settings.json",
                "JSON files (*.json);;All files (*)"
            )
            
            if filepath:
                settings = self.control_panel.get_current_settings()
                
                # Add model and video paths
                settings['model_path'] = self.current_model_path
                settings['video_path'] = self.current_video_path
                
                # Save to file
                import json
                with open(filepath, 'w') as f:
                    json.dump(settings, f, indent=2)
                
                self.control_panel.add_log_entry(f"Settings saved: {filepath}", "SUCCESS")
                
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            self.control_panel.add_log_entry(f"Save settings error: {e}", "ERROR")
    
    def _load_settings(self):
        """Load settings from file."""
        try:
            from PyQt5.QtWidgets import QFileDialog
            
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                "Load Settings",
                "",
                "JSON files (*.json);;All files (*)"
            )
            
            if filepath:
                import json
                with open(filepath, 'r') as f:
                    settings = json.load(f)
                
                # Apply settings
                self.control_panel.apply_settings(settings)
                
                # Load model and video if specified
                if 'model_path' in settings and settings['model_path']:
                    self.detection_display.set_model_file(settings['model_path'])
                
                if 'video_path' in settings and settings['video_path']:
                    self.detection_display.set_video_file(settings['video_path'])
                
                self.control_panel.add_log_entry(f"Settings loaded: {filepath}", "SUCCESS")
                
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            self.control_panel.add_log_entry(f"Load settings error: {e}", "ERROR")
    
    def _show_about(self):
        """Show about dialog."""
        about_text = f"""
        <h2>{self.config.app_name}</h2>
        <p><b>Version:</b> {self.config.app_version}</p>
        <p><b>Author:</b> {self.config.author}</p>
        <p><b>Description:</b> {self.config.description}</p>
        <br>
        <p>This application uses YOLOv8 for real-time detection and counting of recyclable materials.</p>
        <p>Built with PyQt5 and modern software engineering practices.</p>
        """
        
        QMessageBox.about(self, "About", about_text)
    
    def _show_error(self, title: str, message: str):
        """Show error message dialog."""
        QMessageBox.critical(self, title, message)
    
    def _show_warning(self, title: str, message: str):
        """Show warning message dialog."""
        QMessageBox.warning(self, title, message)
    
    def _show_info(self, title: str, message: str):
        """Show information message dialog."""
        QMessageBox.information(self, title, message)
    
    def closeEvent(self, event: QCloseEvent):
        """Handle application close event."""
        try:
            # Stop processing if running
            if self.is_processing:
                self._stop_processing()
                
                # Give time for cleanup
                QApplication.processEvents()
                time.sleep(0.5)
            
            # Cleanup resources
            if self.detector:
                self.detector.cleanup()
            
            # Save window state
            self._save_window_state()
            
            logger.info("Application closing")
            event.accept()
            
        except Exception as e:
            logger.error(f"Error during close: {e}")
            event.accept()  # Close anyway
    
    def _save_window_state(self):
        """Save window state and geometry."""
        try:
            # This could save window geometry, splitter positions, etc.
            # For now, just log
            logger.debug("Window state saved")
            
        except Exception as e:
            logger.error(f"Error saving window state: {e}")
    
    def get_current_statistics(self) -> Dict[str, Any]:
        """Get current application statistics."""
        stats = {
            'frame_count': self.frame_count,
            'is_processing': self.is_processing,
            'is_paused': self.is_paused,
            'model_loaded': self.detector is not None and self.detector.is_loaded,
            'video_selected': self.current_video_path is not None
        }
        
        if self.counter:
            stats.update(self.counter.get_statistics())
        
        if self.detector:
            stats.update(self.detector.get_performance_stats())
        
        return stats