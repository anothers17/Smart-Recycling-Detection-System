
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# MOCK EXTERNAL DEPENDENCIES
sys.modules["torch"] = MagicMock()
sys.modules["ultralytics"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["PyQt5"] = MagicMock()
sys.modules["PyQt5.QtWidgets"] = MagicMock()
sys.modules["PyQt5.QtCore"] = MagicMock()
sys.modules["PyQt5.QtGui"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["psutil"] = MagicMock()

# Mock settings to avoid file read errors if config missing
sys.modules["pydantic"] = MagicMock()
sys.modules["pydantic_settings"] = MagicMock()

# Create a more realistic mock for config
mock_config = MagicMock()
mock_config.detection.confidence_threshold = 0.5
mock_config.paths.models_dir = Path("models")
mock_config.has_hardware = False

# Mock get_config
def mock_get_config():
    return mock_config

# We need to inject this mock into config.settings BEFORE importing other modules
# But config.settings might be imported by them.
# So we mock config.settings entirely.
sys.modules["config.settings"] = MagicMock()
sys.modules["config.settings"].get_config = mock_get_config

# Mock logging
sys.modules["config.logging_config"] = MagicMock()
sys.modules["config.logging_config"].get_logger = lambda name: MagicMock()


def test_imports():
    print("Testing imports...")
    try:
        from src.detection.processor import VideoProcessor
        from src.hardware.mock import MockHardware
        # detector imports YOLO, which is mocked.
        from src.detection.detector import RecyclingDetector 
        from src.detection.counter import RecyclingCounter
        print("Imports successful!")
    except ImportError as e:
        print(f"Import failed: {e}")
        # Print traceback to help debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

def test_instantiation():
    print("Testing instantiation...")
    try:
        from src.hardware.factory import get_hardware_interface
        
        # Test hardware factory
        # We need to ensure settings.HAS_HARDWARE is handled.
        # But get_hardware_interface uses get_config().
        
        hw = get_hardware_interface()
        print(f"Hardware initialized: {type(hw).__name__}")
        
    except Exception as e:
        print(f"Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
    test_instantiation()
    print("Smoke test passed!")
