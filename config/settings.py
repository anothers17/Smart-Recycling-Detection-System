"""
Application configuration and settings management.

This module provides centralized configuration management for the Smart Recycling Detection System.
It handles default values, environment variable overrides, and configuration validation.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class DetectionConfig:
    """Configuration for object detection parameters."""
    confidence_threshold: float = 0.7
    iou_threshold: float = 0.45
    max_detections: int = 1000
    input_size: int = 640
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    
    def __post_init__(self):
        """Validate detection configuration."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("IOU threshold must be between 0.0 and 1.0")


@dataclass
class CountingConfig:
    """Configuration for object counting parameters."""
    line_position_x: int = 300
    line_position_y: Optional[int] = None
    tracking_enabled: bool = True
    tracking_max_distance: float = 50.0
    reset_tracking_after_frames: int = 30
    
    # Class names for recyclable materials
    target_classes: List[str] = field(default_factory=lambda: [
        'bottle-glass',
        'bottle-plastic', 
        'tin can'
    ])


@dataclass
class UIConfig:
    """Configuration for user interface."""
    window_title: str = "Smart Recycling Detection System"
    window_width: int = 1240
    window_height: int = 730
    theme: str = "modern"  # "modern", "dark", "light"
    update_interval_ms: int = 30
    display_fps: bool = True
    display_confidence: bool = True
    enable_logging_panel: bool = True


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    fps_limit: int = 30
    buffer_size: int = 1
    skip_frames: int = 0
    output_format: str = "mp4"
    save_detections: bool = False
    detection_output_dir: str = "output/detections"


@dataclass
class PathConfig:
    """Configuration for file paths."""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    models_dir: Path = field(init=False)
    resources_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    sample_videos_dir: Path = field(init=False)
    icons_dir: Path = field(init=False)
    
    def __post_init__(self):
        """Initialize derived paths."""
        self.models_dir = self.project_root / "src" / "resources" / "models"
        self.resources_dir = self.project_root / "src" / "resources"
        self.logs_dir = self.project_root / "logs"
        self.output_dir = self.project_root / "output"
        self.sample_videos_dir = self.resources_dir / "sample_videos"
        self.icons_dir = self.resources_dir / "icons"
        
        # Create directories if they don't exist
        for directory in [self.logs_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """Main application configuration class."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Application metadata
    app_name: str = "Smart Recycling Detection"
    app_version: str = "1.0.0"
    author: str = "Your Name"
    description: str = "AI-powered recycling detection system using YOLOv8"
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._load_from_environment()
        self._validate_config()
    
    def _load_from_environment(self):
        """Load configuration overrides from environment variables."""
        env_mappings = {
            'DETECTION_CONFIDENCE': ('detection', 'confidence_threshold', float),
            'DETECTION_DEVICE': ('detection', 'device', str),
            'UI_THEME': ('ui', 'theme', str),
            'VIDEO_FPS_LIMIT': ('video', 'fps_limit', int),
            'COUNTING_LINE_X': ('counting', 'line_position_x', int),
        }
        
        for env_var, (section, attr, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = type_func(value)
                    setattr(getattr(self, section), attr, converted_value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {env_var}: {value}. Error: {e}")
    
    def _validate_config(self):
        """Validate the complete configuration."""
        # Validate theme
        valid_themes = ["modern", "dark", "light"]
        if self.ui.theme not in valid_themes:
            print(f"Warning: Invalid theme '{self.ui.theme}'. Using 'modern'.")
            self.ui.theme = "modern"
        
        # Validate device setting
        if self.detection.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.detection.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.detection.device = "mps"
            else:
                self.detection.device = "cpu"
    
    def save_to_file(self, filepath: str):
        """Save configuration to a JSON file."""
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def load_from_file(self, filepath: str):
        """Load configuration from a JSON file."""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Update configuration from loaded data
        self._update_from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_name in ['detection', 'counting', 'ui', 'video']:
            section = getattr(self, field_name)
            result[field_name] = {
                k: v for k, v in section.__dict__.items()
                if not k.startswith('_')
            }
        
        # Add paths as strings
        result['paths'] = {
            k: str(v) for k, v in self.paths.__dict__.items()
            if not k.startswith('_')
        }
        
        return result
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section_name, section_data in config_dict.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)


# Global configuration instance
_config_instance: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
        
        # Try to load from config file
        config_file = _config_instance.paths.project_root / "config" / "app_config.json"
        if config_file.exists():
            _config_instance.load_from_file(str(config_file))
    
    return _config_instance


def reload_config() -> AppConfig:
    """Reload the configuration from scratch."""
    global _config_instance
    _config_instance = None
    return get_config()


# Convenience functions for accessing common config values
def get_model_path(model_name: str = "best.pt") -> Path:
    """Get the path to a model file."""
    return get_config().paths.models_dir / model_name


def get_detection_confidence() -> float:
    """Get the detection confidence threshold."""
    return get_config().detection.confidence_threshold


def get_target_classes() -> List[str]:
    """Get the list of target classes for detection."""
    return get_config().counting.target_classes.copy()


def get_device() -> str:
    """Get the device for running inference."""
    return get_config().detection.device