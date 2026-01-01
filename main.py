"""
Smart Recycling Detection System - Main Entry Point.
"""

import sys
import argparse
from pathlib import Path
from PyQt5.QtWidgets import QApplication

from config.settings import get_config
from config.logging_config import setup_logging, get_logger
from src.hardware import get_hardware_interface
from src.detection.model import load_detector
from src.detection.counter import create_counter
from src.detection.processor import create_processor
from src.ui.main_window import MainWindow

logger = get_logger("main")

def main() -> None:
    """
    Main application entry point.
    
    Initializes configuration, logging, hardware, and the GUI application.
    Supports both headless execution (headless mode) and GUI mode.

    Returns:
        None
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Smart Recycling Detection System")
    parser.add_argument("--source", type=str, default="0", help="Video source (camera index or file path)")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to YOLO model")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI (headless mode)")
    args = parser.parse_args()

    # --- DEBUG STARTUP ---
    print("\n[DEBUG] Application is starting...")
    print(f"[DEBUG] Arguments: {args}")

    # Load configuration
    config = get_config()

    # Setup logging
    setup_logging(log_level="INFO", log_dir=config.paths.logs_dir)
    logger.info(f"Starting {config.app_name} v{config.app_version}")
    
    # Initialize Hardware
    hardware = get_hardware_interface()
    
    try:
        # Load Detector
        model_path = config.paths.models_dir / args.model
        if not model_path.exists() and not Path(args.model).exists():
             # Fallback to absolute path or just filename if in current dir
             if (Path.cwd() / args.model).exists():
                 model_path = Path.cwd() / args.model
             else:
                 logger.warning(f"Model file not found at {model_path}. Using default 'yolov8n.pt' for testing if available.")
                 model_path = "yolov8n.pt" # Library will try to download

        logger.info(f"Loading model from: {model_path}")
        detector = load_detector(str(model_path)) # detection.model.load_detector accepts str or Path

        # Initialize Counter
        counter = create_counter()
        
        # Determine Source
        source = args.source
        if source.isdigit():
            source = int(source)
            
        # Create Video Processor
        processor = create_processor(detector, counter, source, hardware)

        if args.no_gui:
            logger.info("Running in headless mode...")
            processor.start_processing()
            # In headless, we need a way to keep main thread alive or join
            # For QThread, we usually use an EventLoop even without GUI
            # But simple join might work if logic allows.
            # Implementation for headless is minimal here.
            processor.wait()
            
        else:
            logger.info("Starting GUI...")
            app = QApplication(sys.argv)
            
            # Apply theme?
            
            window = MainWindow()
            window.show()
            
            sys.exit(app.exec_())

    except Exception as e:
        logger.critical(f"Application crashed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Final cleanup if needed
        pass

if __name__ == "__main__":
    main()
