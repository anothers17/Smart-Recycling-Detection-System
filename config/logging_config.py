"""
Logging configuration for Smart Recycling Detection System.

This module provides comprehensive logging setup with multiple handlers,
formatters, and log levels for different components of the application.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        """Format log record with colors for console output."""
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            record.levelname = colored_levelname
        
        # Format the message
        formatted = super().format(record)
        
        # Reset levelname for other handlers
        record.levelname = levelname
        
        return formatted


class PerformanceFilter(logging.Filter):
    """Filter for performance-related log messages."""
    
    def filter(self, record):
        """Only allow performance-related messages."""
        performance_keywords = ['fps', 'processing_time', 'memory', 'performance', 'speed']
        return any(keyword in record.getMessage().lower() for keyword in performance_keywords)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    enable_performance_logging: bool = True,
    max_log_files: int = 5,
    max_file_size_mb: int = 10
) -> Dict[str, logging.Logger]:
    """
    Set up comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_file_logging: Whether to enable file logging
        enable_console_logging: Whether to enable console logging
        enable_performance_logging: Whether to enable performance logging
        max_log_files: Maximum number of log files to keep
        max_file_size_mb: Maximum size of each log file in MB
        
    Returns:
        Dictionary of configured loggers
    """
    
    # Set up log directory
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    colored_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Dictionary to store all loggers
    loggers = {}
    
    # 1. Main Application Logger
    main_logger = logging.getLogger('recycling_detection')
    main_logger.setLevel(numeric_level)
    
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(colored_formatter)
        main_logger.addHandler(console_handler)
    
    if enable_file_logging:
        # Main application log file with rotation
        main_file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "application.log",
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=max_log_files,
            encoding='utf-8'
        )
        main_file_handler.setLevel(numeric_level)
        main_file_handler.setFormatter(detailed_formatter)
        main_logger.addHandler(main_file_handler)
        
        # Error log file (errors and critical only)
        error_file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "errors.log",
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=max_log_files,
            encoding='utf-8'
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(detailed_formatter)
        main_logger.addHandler(error_file_handler)
    
    loggers['main'] = main_logger
    
    # 2. Detection Logger
    detection_logger = logging.getLogger('recycling_detection.detection')
    detection_logger.setLevel(numeric_level)
    
    if enable_file_logging:
        detection_file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "detection.log",
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=max_log_files,
            encoding='utf-8'
        )
        detection_file_handler.setLevel(numeric_level)
        detection_file_handler.setFormatter(detailed_formatter)
        detection_logger.addHandler(detection_file_handler)
    
    loggers['detection'] = detection_logger
    
    # 3. GUI Logger
    gui_logger = logging.getLogger('recycling_detection.gui')
    gui_logger.setLevel(numeric_level)
    
    if enable_file_logging:
        gui_file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "gui.log",
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=max_log_files,
            encoding='utf-8'
        )
        gui_file_handler.setLevel(numeric_level)
        gui_file_handler.setFormatter(detailed_formatter)
        gui_logger.addHandler(gui_file_handler)
    
    loggers['gui'] = gui_logger
    
    # 4. Performance Logger
    if enable_performance_logging:
        performance_logger = logging.getLogger('recycling_detection.performance')
        performance_logger.setLevel(logging.DEBUG)
        
        if enable_file_logging:
            perf_file_handler = logging.handlers.RotatingFileHandler(
                filename=log_dir / "performance.log",
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=max_log_files,
                encoding='utf-8'
            )
            perf_file_handler.setLevel(logging.DEBUG)
            perf_file_handler.setFormatter(simple_formatter)
            performance_logger.addHandler(perf_file_handler)
        
        # Performance console handler (optional)
        if enable_console_logging and numeric_level <= logging.DEBUG:
            perf_console_handler = logging.StreamHandler(sys.stdout)
            perf_console_handler.setLevel(logging.DEBUG)
            perf_console_handler.setFormatter(simple_formatter)
            perf_console_handler.addFilter(PerformanceFilter())
            performance_logger.addHandler(perf_console_handler)
        
        loggers['performance'] = performance_logger
    
    # 5. Counter Logger
    counter_logger = logging.getLogger('recycling_detection.counter')
    counter_logger.setLevel(numeric_level)
    
    if enable_file_logging:
        counter_file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "counting.log",
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=max_log_files,
            encoding='utf-8'
        )
        counter_file_handler.setLevel(numeric_level)
        counter_file_handler.setFormatter(detailed_formatter)
        counter_logger.addHandler(counter_file_handler)
    
    loggers['counter'] = counter_logger
    
    # Log the initialization
    main_logger.info("="*60)
    main_logger.info("Smart Recycling Detection System - Logging Initialized")
    main_logger.info(f"Log Level: {log_level}")
    main_logger.info(f"Log Directory: {log_dir}")
    main_logger.info(f"File Logging: {enable_file_logging}")
    main_logger.info(f"Console Logging: {enable_console_logging}")
    main_logger.info(f"Performance Logging: {enable_performance_logging}")
    main_logger.info("="*60)
    
    return loggers


def get_logger(name: str = 'main') -> logging.Logger:
    """
    Get a logger by name.
    
    Args:
        name: Logger name ('main', 'detection', 'gui', 'performance', 'counter')
        
    Returns:
        Logger instance
    """
    logger_names = {
        'main': 'recycling_detection',
        'detection': 'recycling_detection.detection',
        'gui': 'recycling_detection.gui',
        'performance': 'recycling_detection.performance',
        'counter': 'recycling_detection.counter'
    }
    
    logger_name = logger_names.get(name, 'recycling_detection')
    logger = logging.getLogger(logger_name)
    
    # If logger has no handlers, set up basic logging
    if not logger.handlers:
        setup_logging()
        logger = logging.getLogger(logger_name)
    
    return logger


class LoggingContext:
    """Context manager for temporarily changing log levels."""
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.original_level = None
    
    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


def log_performance(func):
    """Decorator to log function performance."""
    def wrapper(*args, **kwargs):
        logger = get_logger('performance')
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"{func.__name__} completed in {duration:.4f}s")
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {duration:.4f}s: {e}")
            raise
    
    return wrapper


def log_method_calls(cls):
    """Class decorator to log all method calls."""
    logger = get_logger('main')
    
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith('_'):
            def logged_method(original_method):
                def wrapper(self, *args, **kwargs):
                    logger.debug(f"{cls.__name__}.{original_method.__name__} called")
                    return original_method(self, *args, **kwargs)
                return wrapper
            
            setattr(cls, attr_name, logged_method(attr))
    
    return cls


# Initialize default logging if this module is imported
if __name__ != '__main__':
    # Set up basic logging when module is imported
    try:
        setup_logging()
    except Exception as e:
        # Fallback to basic logging if setup fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        logging.getLogger('recycling_detection').warning(
            f"Failed to set up advanced logging: {e}. Using basic logging."
        )