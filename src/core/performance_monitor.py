"""
Performance monitoring utilities for Smart Recycling Detection System.

This module provides performance tracking and monitoring capabilities
for detection and processing operations.
"""

import time
import numpy as np
import psutil
from typing import Dict


class ModelPerformanceMonitor:
    """Monitor model performance metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.processing_times = []
        self.detection_counts = []
        self.timestamps = []
        self.memory_usages = []  # Memory usage in MB

    def update(self, processing_time: float, detection_count: int):
        """Update performance metrics."""
        current_time = time.time()

        # Get current memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024  # Convert to MB

        self.processing_times.append(processing_time)
        self.detection_counts.append(detection_count)
        self.memory_usages.append(memory_mb)
        self.timestamps.append(current_time)

        # Keep only recent measurements
        if len(self.processing_times) > self.window_size:
            self.processing_times.pop(0)
            self.detection_counts.pop(0)
            self.memory_usages.pop(0)
            self.timestamps.pop(0)

    def get_average_fps(self) -> float:
        """Get average FPS over the window."""
        if len(self.processing_times) < 2:
            return 0.0

        return 1.0 / np.mean(self.processing_times)

    def get_average_processing_time(self) -> float:
        """Get average processing time."""
        if not self.processing_times:
            return 0.0

        return np.mean(self.processing_times)

    def get_detection_rate(self) -> float:
        """Get average detections per frame."""
        if not self.detection_counts:
            return 0.0

        return np.mean(self.detection_counts)

    def get_average_memory_usage(self) -> float:
        """Get average memory usage in MB."""
        if not self.memory_usages:
            return 0.0

        return np.mean(self.memory_usages)

    def get_peak_memory_usage(self) -> float:
        """Get peak memory usage in MB."""
        if not self.memory_usages:
            return 0.0

        return np.max(self.memory_usages)

    def get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not self.memory_usages:
            return 0.0

        return self.memory_usages[-1] if self.memory_usages else 0.0