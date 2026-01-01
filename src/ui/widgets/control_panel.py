"""
Control panel widget for Smart Recycling Detection System.

This module provides comprehensive control widgets including
start/stop controls, counting displays, settings, and activity logging.
"""

import time
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QGroupBox,
    QLCDNumber,
    QTextEdit,
    QSlider,
    QSpinBox,
    QCheckBox,
    QTabWidget,
    QScrollArea,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QTextCursor

from config.settings import get_config
from config.logging_config import get_logger

logger = get_logger("gui")


class CountingDisplayWidget(QFrame):
    """Widget for displaying object counting statistics with LCD displays."""

    def __init__(self, parent=None):
        """Initialize counting display widget."""
        super().__init__(parent)
        self.config = get_config()
        self.class_lcds = {}
        self.setup_ui()
        self.reset_counters()
        self.setMinimumWidth(200)

        logger.debug("Counting display widget initialized")

    def setup_ui(self):
        """Set up the user interface."""
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 12px;
            }
        """
        )

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Detection Counters")
        title.setProperty("class", "subheading")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Create LCD displays for each target class
        target_classes = self.config.counting.target_classes

        for class_name in target_classes:
            class_layout = self._create_class_counter(class_name)
            layout.addLayout(class_layout)

        # Total counter
        layout.addWidget(self._create_separator())
        total_layout = self._create_total_counter()
        layout.addLayout(total_layout)

        layout.addStretch()

    def _create_class_counter(self, class_name: str) -> QHBoxLayout:
        """Create counter display for a specific class."""
        layout = QHBoxLayout()

        # Class label
        display_name = class_name.replace("-", " ").title()
        label = QLabel(display_name)
        label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        label.setFixedWidth(200)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # LCD display
        lcd = QLCDNumber(5)
        lcd.setSegmentStyle(QLCDNumber.Flat)
        lcd.setFixedHeight(60)
        lcd.setMinimumWidth(80)

        # Style LCD based on class
        if "glass" in class_name.lower():
            lcd.setProperty("class", "bottle-glass")
            lcd.setStyleSheet("QLCDNumber { color: #2196F3; }")
        elif "plastic" in class_name.lower():
            lcd.setProperty("class", "bottle-plastic")
            lcd.setStyleSheet("QLCDNumber { color: #FF9800; }")
        elif "can" in class_name.lower():
            lcd.setProperty("class", "tin-can")
            lcd.setStyleSheet("QLCDNumber { color: #757575; }")

        # Store reference
        self.class_lcds[class_name] = lcd

        layout.addWidget(label)
        layout.addWidget(lcd)

        return layout

    def _create_total_counter(self) -> QHBoxLayout:
        """Create total counter display."""
        layout = QHBoxLayout()

        # Total label
        label = QLabel("Total Count")
        label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        label.setFixedWidth(200)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # Total LCD
        self.total_lcd = QLCDNumber(5)
        self.total_lcd.setSegmentStyle(QLCDNumber.Flat)
        self.total_lcd.setFixedHeight(60)
        self.total_lcd.setMinimumWidth(80)
        self.total_lcd.setStyleSheet(
            "QLCDNumber { color: #4CAF50; background-color: #E8F5E8; }"
        )

        layout.addWidget(label)
        layout.addWidget(self.total_lcd)

        return layout

    def _create_separator(self) -> QFrame:
        """Create visual separator."""
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("color: #E0E0E0;")
        return separator

    def update_counts(self, class_counts: Dict[str, int]):
        """
        Update all counter displays.

        Args:
            class_counts: Dictionary with count per class
        """
        try:
            total = 0

            # Update individual class counters
            for class_name, lcd in self.class_lcds.items():
                count = class_counts.get(class_name, 0)
                lcd.display(count)
                total += count

            # Update total counter
            self.total_lcd.display(total)

        except Exception as e:
            logger.error(f"Error updating counters: {e}")

    def reset_counters(self):
        """Reset all counters to zero."""
        try:
            for lcd in self.class_lcds.values():
                lcd.display(0)

            self.total_lcd.display(0)

            logger.info("Counters reset")

        except Exception as e:
            logger.error(f"Error resetting counters: {e}")

    def get_current_counts(self) -> Dict[str, int]:
        """Get current counter values."""
        counts = {}

        for class_name, lcd in self.class_lcds.items():
            counts[class_name] = lcd.intValue()

        counts["total"] = self.total_lcd.intValue()

        return counts


class MotorStatusItem(QWidget):
    """Small widget representing a single motor's status."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        # Motor name
        self.label = QLabel(self.name)
        self.label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.label.setFixedWidth(120)

        # Status indicator
        self.status_label = QLabel("CLOSE")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedWidth(80)
        self.status_label.setFixedHeight(24)
        self.status_label.setStyleSheet(
            "background-color: #E0E0E0; color: #424242; border-radius: 4px; font-weight: bold; font-size: 11px;"
        )

        layout.addWidget(self.label)
        layout.addWidget(self.status_label)
        layout.addStretch()

    def set_status(self, is_open: bool):
        """Set motor status and update styling."""
        if is_open:
            self.status_label.setText("OPEN")
            self.status_label.setStyleSheet(
                "background-color: #4CAF50; color: white; border-radius: 4px; font-weight: bold; font-size: 11px;"
            )
        else:
            self.status_label.setText("CLOSE")
            self.status_label.setStyleSheet(
                "background-color: #E0E0E0; color: #424242; border-radius: 4px; font-weight: bold; font-size: 11px;"
            )


class HardwareStatusWidget(QGroupBox):
    """Widget displaying the status of all hardware motors."""

    def __init__(self, parent=None):
        super().__init__("Hardware Status", parent)
        self.motors = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(12, 16, 12, 12)

        # Define motors based on classes
        motor_names = ["Bottle Glass", "Bottle Plastic", "Tin Can"]
        for name in motor_names:
            motor_item = MotorStatusItem(name)
            # Normalize key to lowercase with hyphens
            key = name.lower().replace(" ", "-").replace("_", "-")
            self.motors[key] = motor_item
            layout.addWidget(motor_item)

    def update_status(self, class_name: str, status: str):
        """Update a specific motor's status."""
        # Normalize: lowercase, replace spaces/underscores with hyphens
        key = class_name.lower().replace(" ", "-").replace("_", "-")
        if key in self.motors:
            is_open = status.upper() == "OPEN"
            self.motors[key].set_status(is_open)
            
            # If opening, set a timer to close it after a short delay (simulation)
            if is_open:
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(1500, lambda: self.motors[key].set_status(False))


class ControlButtonsWidget(QWidget):
    """Widget containing main control buttons."""

    # Signals
    startRequested = pyqtSignal()
    stopRequested = pyqtSignal()
    pauseRequested = pyqtSignal()
    resumeRequested = pyqtSignal()
    resetRequested = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize control buttons widget."""
        super().__init__(parent)
        self.is_running = False
        self.is_paused = False
        self.setup_ui()
        self.setup_connections()

        logger.debug("Control buttons widget initialized")

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # First row: Start and Stop
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(20)

        self.start_button = QPushButton("Start")
        self.start_button.setMinimumHeight(50)
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; border-radius: 6px;")
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setMinimumHeight(50)
        self.stop_button.setStyleSheet("background-color: #F44336; color: white; font-weight: bold; border-radius: 6px;")
        self.stop_button.setEnabled(False)

        row1_layout.addWidget(self.start_button)
        row1_layout.addWidget(self.stop_button)

        # Second row: Pause and Reset
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(20)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setMinimumHeight(50)
        self.pause_button.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; border-radius: 6px;")
        self.pause_button.setEnabled(False)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setMinimumHeight(50)
        self.reset_button.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; border-radius: 6px;")

        row2_layout.addWidget(self.pause_button)
        row2_layout.addWidget(self.reset_button)

        layout.addLayout(row1_layout)
        layout.addLayout(row2_layout)

    def setup_connections(self):
        """Set up signal connections."""
        self.start_button.clicked.connect(self._handle_start)
        self.pause_button.clicked.connect(self._handle_pause)
        self.stop_button.clicked.connect(self._handle_stop)
        self.reset_button.clicked.connect(self._handle_reset)

    def _handle_start(self):
        """Handle start button click."""
        if not self.is_running:
            self.startRequested.emit()
        elif self.is_paused:
            self.resumeRequested.emit()

    def _handle_pause(self):
        """Handle pause button click."""
        if self.is_running and not self.is_paused:
            self.pauseRequested.emit()

    def _handle_stop(self):
        """Handle stop button click."""
        if self.is_running:
            self.stopRequested.emit()

    def _handle_reset(self):
        """Handle reset button click."""
        self.resetRequested.emit()

    def set_state(self, running: bool, paused: bool = False):
        """
        Update button states based on application state.

        Args:
            running: Whether detection is running
            paused: Whether detection is paused
        """
        self.is_running = running
        self.is_paused = paused

        if not running:
            # Stopped state
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.reset_button.setEnabled(True)
        elif paused:
            # Paused state
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.reset_button.setEnabled(True)
        else:
            # Running state
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.reset_button.setEnabled(False)

        self._update_button_states()

    def _update_button_states(self):
        """Update button texts based on current state."""
        if self.is_paused:
            self.start_button.setText("Resume")
        else:
            self.start_button.setText("Start")


class SettingsWidget(QFrame):
    """Widget for adjusting detection and counting settings."""

    # Signals
    settingsChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        """Initialize settings widget."""
        super().__init__(parent)
        self.config = get_config()
        self.setup_ui()
        self.setup_connections()
        self.load_current_settings()

        logger.debug("Settings widget initialized")

    def setup_ui(self):
        """Set up the user interface."""
        self.setFrameStyle(QFrame.StyledPanel)

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Settings")
        title.setProperty("class", "subheading")
        layout.addWidget(title)

        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))

        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 95)
        self.confidence_slider.setValue(70)

        self.confidence_label = QLabel("0.70")
        self.confidence_label.setMinimumWidth(40)

        conf_layout.addWidget(self.confidence_slider)
        conf_layout.addWidget(self.confidence_label)
        layout.addLayout(conf_layout)

        # Line position
        line_layout = QHBoxLayout()
        line_layout.addWidget(QLabel("Line Position:"))

        self.line_spinbox = QSpinBox()
        self.line_spinbox.setRange(50, 1000)
        self.line_spinbox.setValue(300)
        self.line_spinbox.setSuffix(" px")

        line_layout.addWidget(self.line_spinbox)
        line_layout.addStretch()
        layout.addLayout(line_layout)

        # Display options
        options_group = QGroupBox("Display Options")
        options_layout = QVBoxLayout(options_group)

        self.show_confidence_cb = QCheckBox("Show Confidence Scores")
        self.show_confidence_cb.setChecked(True)

        self.show_fps_cb = QCheckBox("Show FPS Counter")
        self.show_fps_cb.setChecked(True)

        self.show_line_cb = QCheckBox("Show Counting Line")
        self.show_line_cb.setChecked(True)

        options_layout.addWidget(self.show_confidence_cb)
        options_layout.addWidget(self.show_fps_cb)
        options_layout.addWidget(self.show_line_cb)

        layout.addWidget(options_group)
        layout.addStretch()

    def setup_connections(self):
        """Set up signal connections."""
        self.confidence_slider.valueChanged.connect(self._update_confidence_label)
        self.confidence_slider.valueChanged.connect(self._emit_settings_changed)
        self.line_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.show_confidence_cb.toggled.connect(self._emit_settings_changed)
        self.show_fps_cb.toggled.connect(self._emit_settings_changed)
        self.show_line_cb.toggled.connect(self._emit_settings_changed)

    def load_current_settings(self):
        """Load current settings from config."""
        try:
            # Load confidence threshold
            conf_value = int(self.config.detection.confidence_threshold * 100)
            self.confidence_slider.setValue(conf_value)

            # Load line position
            line_pos = self.config.counting.line_position_x
            if line_pos:
                self.line_spinbox.setValue(line_pos)

            logger.debug("Settings loaded from config")

        except Exception as e:
            logger.error(f"Error loading settings: {e}")

    def _update_confidence_label(self, value: int):
        """Update confidence threshold label."""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")

    def _emit_settings_changed(self):
        """Emit settings changed signal with current values."""
        settings = self.get_current_settings()
        self.settingsChanged.emit(settings)

    def get_current_settings(self) -> Dict[str, Any]:
        """Get current settings values."""
        return {
            "confidence_threshold": self.confidence_slider.value() / 100.0,
            "line_position_x": self.line_spinbox.value(),
            "show_confidence": self.show_confidence_cb.isChecked(),
            "show_fps": self.show_fps_cb.isChecked(),
            "show_counting_line": self.show_line_cb.isChecked(),
        }

    def apply_settings(self, settings: Dict[str, Any]):
        """Apply settings to the widget."""
        try:
            if "confidence_threshold" in settings:
                value = int(settings["confidence_threshold"] * 100)
                self.confidence_slider.setValue(value)

            if "line_position_x" in settings:
                self.line_spinbox.setValue(settings["line_position_x"])

            if "show_confidence" in settings:
                self.show_confidence_cb.setChecked(settings["show_confidence"])

            if "show_fps" in settings:
                self.show_fps_cb.setChecked(settings["show_fps"])

            if "show_counting_line" in settings:
                self.show_line_cb.setChecked(settings["show_counting_line"])

            logger.debug("Settings applied to widget")

        except Exception as e:
            logger.error(f"Error applying settings: {e}")


class ActivityLogWidget(QFrame):
    """Widget for displaying real-time activity log."""

    def __init__(self, parent=None):
        """Initialize activity log widget."""
        super().__init__(parent)
        self.max_lines = 1000
        self.setup_ui()

        # Auto-scroll timer
        self.scroll_timer = QTimer()
        self.scroll_timer.timeout.connect(self._auto_scroll)
        self.scroll_timer.start(100)

        logger.debug("Activity log widget initialized")

    def setup_ui(self):
        """Set up the user interface."""
        self.setFrameStyle(QFrame.StyledPanel)

        layout = QVBoxLayout(self)

        # Header
        header_layout = QHBoxLayout()

        title = QLabel("Activity Log")
        title.setProperty("class", "subheading")

        clear_button = QPushButton("Clear")
        clear_button.setMaximumWidth(80)
        clear_button.clicked.connect(self.clear_log)

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(clear_button)

        layout.addLayout(header_layout)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 9))
        self.log_display.setMaximumHeight(200)
        self.log_display.setStyleSheet(
            """
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #666666;
                border-radius: 4px;
            }
        """
        )

        layout.addWidget(self.log_display)

    def add_log_entry(self, message: str, level: str = "INFO"):
        """
        Add entry to activity log.

        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        try:
            timestamp = time.strftime("%H:%M:%S")

            # Color code by level
            colors = {
                "DEBUG": "#888888",
                "INFO": "#ffffff",
                "WARNING": "#ffaa00",
                "ERROR": "#ff4444",
                "SUCCESS": "#44ff44",
            }

            color = colors.get(level.upper(), "#ffffff")

            # Format log entry
            log_entry = (
                f'<span style="color: #888888;">[{timestamp}]</span> '
                f'<span style="color: {color};">[{level}]</span> '
                f'<span style="color: #ffffff;">{message}</span>'
            )

            # Add to display
            self.log_display.append(log_entry)

            # Limit number of lines
            self._trim_log()

        except Exception as e:
            logger.error(f"Error adding log entry: {e}")

    def _trim_log(self):
        """Trim log to maximum number of lines."""
        try:
            document = self.log_display.document()

            if document.blockCount() > self.max_lines:
                # Remove old lines
                cursor = QTextCursor(document)
                cursor.movePosition(QTextCursor.Start)

                lines_to_remove = document.blockCount() - self.max_lines
                for _ in range(lines_to_remove):
                    cursor.select(QTextCursor.BlockUnderCursor)
                    cursor.movePosition(QTextCursor.NextBlock, QTextCursor.KeepAnchor)
                    cursor.removeSelectedText()

        except Exception as e:
            logger.error(f"Error trimming log: {e}")

    def _auto_scroll(self):
        """Auto-scroll to bottom of log."""
        try:
            scrollbar = self.log_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception:
            pass

    def clear_log(self):
        """Clear the activity log."""
        self.log_display.clear()
        self.add_log_entry("Activity log cleared", "INFO")

    def save_log(self, filepath: str) -> bool:
        """
        Save log to file.

        Args:
            filepath: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.log_display.toPlainText())

            self.add_log_entry(f"Log saved to {filepath}", "SUCCESS")
            return True

        except Exception as e:
            logger.error(f"Error saving log: {e}")
            self.add_log_entry(f"Failed to save log: {e}", "ERROR")
            return False


class ControlPanelWidget(QWidget):
    """
    Main control panel widget combining all control elements.

    This widget serves as the main control interface, providing:
    - Detection counters with LCD displays
    - Control buttons for start/stop/reset
    - Settings adjustment
    - Activity logging
    """

    # Signals
    startRequested = pyqtSignal()
    stopRequested = pyqtSignal()
    pauseRequested = pyqtSignal()
    resumeRequested = pyqtSignal()
    resetRequested = pyqtSignal()
    settingsChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        """Initialize control panel widget."""
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()

        logger.info("Control panel widget initialized")

    def setup_ui(self):
        """Set up the user interface."""
        # Main layout for the widget
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setStyleSheet("background-color: transparent;")

        # Container widget for scroll area
        self.container = QWidget()
        container_layout = QVBoxLayout(self.container)
        container_layout.setSpacing(16)
        container_layout.setContentsMargins(12, 12, 12, 12)

        # Control buttons
        self.control_buttons = ControlButtonsWidget()
        container_layout.addWidget(self.control_buttons)

        # Counting display
        self.counting_display = CountingDisplayWidget()
        container_layout.addWidget(self.counting_display)

        # Hardware status
        self.hardware_status = HardwareStatusWidget()
        container_layout.addWidget(self.hardware_status)

        # Create tabbed interface for settings and log
        self.tab_widget = QTabWidget()

        # Settings tab
        self.settings_widget = SettingsWidget()
        self.tab_widget.addTab(self.settings_widget, "Settings")

        # Activity log tab
        self.activity_log = ActivityLogWidget()
        self.tab_widget.addTab(self.activity_log, "Activity Log")

        container_layout.addWidget(self.tab_widget)

        # Set stretch factors for container
        container_layout.setStretchFactor(self.control_buttons, 0)
        container_layout.setStretchFactor(self.counting_display, 0)
        container_layout.setStretchFactor(self.hardware_status, 0)
        container_layout.setStretchFactor(self.tab_widget, 1)

        # Add container to scroll area
        self.scroll_area.setWidget(self.container)

        # Add scroll area to main layout
        main_layout.addWidget(self.scroll_area)

    def setup_connections(self):
        """Set up signal connections."""
        # Control button signals
        self.control_buttons.startRequested.connect(self.startRequested.emit)
        self.control_buttons.stopRequested.connect(self.stopRequested.emit)
        self.control_buttons.pauseRequested.connect(self.pauseRequested.emit)
        self.control_buttons.resumeRequested.connect(self.resumeRequested.emit)
        self.control_buttons.resetRequested.connect(self.resetRequested.emit)

        # Settings signals
        self.settings_widget.settingsChanged.connect(self.settingsChanged.emit)

    def update_counters(self, class_counts: Dict[str, int]):
        """Update counting displays."""
        self.counting_display.update_counts(class_counts)

    def update_hardware_status(self, class_name: str, action: str, status: str):
        """Update hardware status display and log."""
        # Update visual indicators
        self.hardware_status.update_status(class_name, status)

        # Add to activity log with device-style formatting
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] [Device] Sorting {class_name.upper()}... Status: {status}"
        self.activity_log.add_log_entry(log_msg, "INFO")

    def reset_counters(self):
        """Reset all counters."""
        self.counting_display.reset_counters()

    def set_processing_state(self, running: bool, paused: bool = False):
        """Update control state."""
        self.control_buttons.set_state(running, paused)

        if running and not paused:
            self.add_log_entry("Detection started", "SUCCESS")
        elif paused:
            self.add_log_entry("Detection paused", "WARNING")
        elif not running:
            self.add_log_entry("Detection stopped", "INFO")

    def add_log_entry(self, message: str, level: str = "INFO"):
        """Add entry to activity log."""
        self.activity_log.add_log_entry(message, level)

    def get_current_settings(self) -> Dict[str, Any]:
        """Get current settings."""
        return self.settings_widget.get_current_settings()

    def apply_settings(self, settings: Dict[str, Any]):
        """Apply settings to widgets."""
        self.settings_widget.apply_settings(settings)

    def get_current_counts(self) -> Dict[str, int]:
        """Get current counter values."""
        return self.counting_display.get_current_counts()

    def save_activity_log(self) -> bool:
        """Save activity log to file."""
        try:
            from PyQt5.QtWidgets import QFileDialog

            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Save Activity Log",
                f"activity_log_{int(time.time())}.txt",
                "Text files (*.txt);;All files (*)",
            )

            if filepath:
                return self.activity_log.save_log(filepath)

            return False

        except Exception as e:
            logger.error(f"Error saving activity log: {e}")
            return False
