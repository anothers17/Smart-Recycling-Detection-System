"""
Configuration package for Smart Recycling Detection System.

This package contains all configuration-related modules including:
- Application settings and constants
- Logging configuration
- Environment variable management
- Default values and validation
"""

from .settings import AppConfig, get_config
from .logging_config import setup_logging, get_logger

__all__ = [
    'AppConfig',
    'get_config', 
    'setup_logging',
    'get_logger'
]

__version__ = '1.0.0'