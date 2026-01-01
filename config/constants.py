"""
Global constants for the Smart Recycling Detection System.
"""

# Object Detection
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_MAX_DETECTIONS = 1000
DEFAULT_INPUT_SIZE = 640
DEFAULT_DEVICE = "auto"

# Object Counting & Classes
TARGET_CLASSES = ["bottle-glass", "bottle-plastic", "tin can"]

# Tracking
TRACKING_MAX_DISTANCE = 80.0
RESET_TRACKING_AFTER_FRAMES = 60
MIN_DISTANCE = 15
DEFAULT_LINE_POSITION_X = 200

# UI
DEFAULT_WINDOW_TITLE = "Smart Recycling Detection System"
DEFAULT_WINDOW_WIDTH = 1240
DEFAULT_WINDOW_HEIGHT = 730
DEFAULT_THEME = "modern"
UI_UPDATE_INTERVAL_MS = 30

# Video
DEFAULT_FPS_LIMIT = 30
DEFAULT_BUFFER_SIZE = 1
DEFAULT_OUTPUT_FORMAT = "mp4"

# Hardware / Servo Configuration
# Map class names to servo angles or IDs
CLASS_TO_SERVO_MAP = {
    "bottle-glass": {"id": 1, "angle_open": 90, "angle_close": 0},
    "bottle-plastic": {"id": 2, "angle_open": 90, "angle_close": 0},
    "tin can": {"id": 3, "angle_open": 90, "angle_close": 0},
}
SERVO_DELAY_SEC = 2.0  # Time to keep the gate open
