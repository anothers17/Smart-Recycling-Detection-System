"""
Application configuration using Pydantic Settings.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from . import constants

class DetectionConfig(BaseSettings):
    confidence_threshold: float = constants.DEFAULT_CONFIDENCE_THRESHOLD
    iou_threshold: float = constants.DEFAULT_IOU_THRESHOLD
    max_detections: int = constants.DEFAULT_MAX_DETECTIONS
    input_size: int = constants.DEFAULT_INPUT_SIZE
    device: str = constants.DEFAULT_DEVICE

class CountingConfig(BaseSettings):
    line_position_x: int = constants.DEFAULT_LINE_POSITION_X
    line_position_y: Optional[int] = None
    tracking_enabled: bool = True
    tracking_max_distance: float = constants.TRACKING_MAX_DISTANCE
    reset_tracking_after_frames: int = constants.RESET_TRACKING_AFTER_FRAMES
    min_distance: int = constants.MIN_DISTANCE
    target_classes: List[str] = constants.TARGET_CLASSES

class UIConfig(BaseSettings):
    window_title: str = constants.DEFAULT_WINDOW_TITLE
    window_width: int = constants.DEFAULT_WINDOW_WIDTH
    window_height: int = constants.DEFAULT_WINDOW_HEIGHT
    theme: str = constants.DEFAULT_THEME
    update_interval_ms: int = constants.UI_UPDATE_INTERVAL_MS
    display_fps: bool = True
    display_confidence: bool = True
    enable_logging_panel: bool = True

class VideoConfig(BaseSettings):
    fps_limit: int = constants.DEFAULT_FPS_LIMIT
    buffer_size: int = constants.DEFAULT_BUFFER_SIZE
    skip_frames: int = 0
    output_format: str = constants.DEFAULT_OUTPUT_FORMAT
    save_detections: bool = False
    detection_output_dir: str = "output/detections"

class HardwareConfig(BaseSettings):
    port: str = "COM3"
    baud_rate: int = 115200
    timeout: float = 1.0
    auto_connect: bool = True

class PathConfig(BaseSettings):
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    
    @computed_field
    def models_dir(self) -> Path:
        return self.project_root / "src" / "resources" / "models"
    
    @computed_field
    def resources_dir(self) -> Path:
        return self.project_root / "src" / "resources"
    
    @computed_field
    def logs_dir(self) -> Path:
        return self.project_root / "logs"
    
    @computed_field
    def output_dir(self) -> Path:
        return self.project_root / "output"
        
    @computed_field
    def icons_dir(self) -> Path:
        return self.resources_dir / "icons"
        
    @computed_field
    def sample_videos_dir(self) -> Path:
        return self.resources_dir / "sample_videos"

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore"
    )

    # Sub-configurations
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    counting: CountingConfig = Field(default_factory=CountingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    
    # Global settings
    has_hardware: bool = False
    app_name: str = "Smart Recycling Detection"
    app_version: str = "1.0.0"

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure directories exist
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)

# Global instance
_config_instance: Optional[AppConfig] = None

def get_config() -> AppConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance

# Helper functions for backward compatibility
def get_model_path(model_name: str = "best.pt") -> Path:
    return get_config().paths.models_dir / model_name

def get_detection_confidence() -> float:
    return get_config().detection.confidence_threshold

def get_target_classes() -> List[str]:
    return get_config().counting.target_classes

def get_device() -> str:
    return get_config().detection.device
