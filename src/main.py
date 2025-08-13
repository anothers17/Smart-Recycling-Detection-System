"""
Main entry point for Smart Recycling Detection System.

This module provides the application entry point and initialization.
Enhanced version of the original projectdeep.py with professional structure.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, List
from PyQt5.QtWidgets import QApplication, QMessageBox, QSplashScreen
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QIcon

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import application modules
try:
    from config.settings import get_config, reload_config
    from config.logging_config import setup_logging, get_logger
    from src.gui.main_window import MainWindow
    from src.gui.styles.modern_style import apply_modern_style
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


class ApplicationManager:
    """Manages application lifecycle and initialization."""
    
    def __init__(self):
        self.app: Optional[QApplication] = None
        self.main_window: Optional[MainWindow] = None
        self.config = None
        self.logger = None
    
    def initialize(self, args: Optional[List[str]] = None) -> bool:
        """
        Initialize the application.
        
        Args:
            args: Command line arguments
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Set up logging first
            loggers = setup_logging(
                log_level=os.getenv('LOG_LEVEL', 'INFO'),
                enable_console_logging=True,
                enable_file_logging=True,
                enable_performance_logging=True
            )
            
            self.logger = loggers['main']
            self.logger.info("="*60)
            self.logger.info("Smart Recycling Detection System Starting")
            self.logger.info("="*60)
            
            # Load configuration
            self.config = get_config()
            self.logger.info("Configuration loaded successfully")
            
            # Create QApplication
            if args is None:
                args = sys.argv
            
            self.app = QApplication(args)
            self.app.setApplicationName(self.config.app_name)
            self.app.setApplicationVersion(self.config.app_version)
            self.app.setOrganizationName(self.config.author)
            
            # Set application icon
            self._set_application_icon()
            
            # Apply modern styling
            apply_modern_style(self.app)
            
            self.logger.info("QApplication initialized")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Initialization failed: {e}")
            else:
                print(f"Initialization failed: {e}")
            return False
    
    def _set_application_icon(self):
        """Set application icon if available."""
        try:
            icon_path = self.config.paths.icons_dir / "app_icon.png"
            
            if icon_path.exists():
                icon = QIcon(str(icon_path))
                self.app.setWindowIcon(icon)
                self.logger.debug(f"Application icon set: {icon_path}")
            else:
                self.logger.debug("No application icon found")
                
        except Exception as e:
            self.logger.warning(f"Failed to set application icon: {e}")
    
    def show_splash_screen(self) -> Optional[QSplashScreen]:
        """
        Show splash screen during initialization.
        
        Returns:
            QSplashScreen instance or None
        """
        try:
            # Create splash screen
            splash_pixmap = QPixmap(400, 300)
            splash_pixmap.fill(Qt.white)
            
            splash = QSplashScreen(splash_pixmap)
            splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.SplashScreen)
            
            # Show splash
            splash.show()
            splash.showMessage(
                f"Loading {self.config.app_name}...",
                Qt.AlignBottom | Qt.AlignCenter,
                Qt.black
            )
            
            QApplication.processEvents()
            
            self.logger.debug("Splash screen displayed")
            return splash
            
        except Exception as e:
            self.logger.warning(f"Failed to show splash screen: {e}")
            return None
    
    def create_main_window(self) -> bool:
        """
        Create and show main window.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.main_window = MainWindow()
            self.main_window.show()
            
            self.logger.info("Main window created and displayed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create main window: {e}")
            self._show_critical_error("Startup Error", 
                                    f"Failed to create main window: {e}")
            return False
    
    def run(self) -> int:
        """
        Run the application main loop.
        
        Returns:
            Exit code
        """
        try:
            if not self.app:
                self.logger.error("Application not initialized")
                return 1
            
            if not self.main_window:
                self.logger.error("Main window not created")
                return 1
            
            self.logger.info("Starting application main loop")
            
            # Install exception hook
            sys.excepthook = self._exception_hook
            
            # Run application
            exit_code = self.app.exec_()
            
            self.logger.info(f"Application finished with exit code: {exit_code}")
            return exit_code
            
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            return 1
        finally:
            self._cleanup()
    
    def _exception_hook(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        import traceback
        
        # Log the exception
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        if self.logger:
            self.logger.critical(f"Uncaught exception: {error_msg}")
        else:
            print(f"Uncaught exception: {error_msg}")
        
        # Show error dialog
        self._show_critical_error("Critical Error", 
                                f"An unexpected error occurred:\n\n{exc_value}")
    
    def _show_critical_error(self, title: str, message: str):
        """Show critical error dialog."""
        try:
            if self.app:
                QMessageBox.critical(None, title, message)
            else:
                print(f"CRITICAL ERROR - {title}: {message}")
        except Exception:
            print(f"CRITICAL ERROR - {title}: {message}")
    
    def _cleanup(self):
        """Clean up application resources."""
        try:
            if self.main_window:
                # Main window cleanup is handled in its closeEvent
                pass
            
            if self.logger:
                self.logger.info("Application cleanup completed")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during cleanup: {e}")
            else:
                print(f"Cleanup error: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Smart Recycling Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                          # Start with GUI
  python src/main.py --model path/to/model.pt # Start with specific model
  python src/main.py --video path/to/video.mp4 # Start with specific video
  python src/main.py --debug                  # Enable debug logging
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to model file (.pt, .pth)'
    )
    
    parser.add_argument(
        '--video', '-v', 
        type=str,
        help='Path to video file or camera index'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--log-level', '-l',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--no-gui', '-n',
        action='store_true',
        help='Run without GUI (batch processing mode)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser.parse_args()


def check_dependencies() -> bool:
    """
    Check if all required dependencies are available.
    
    Returns:
        True if all dependencies available, False otherwise
    """
    required_modules = [
        'cv2', 'numpy', 'torch', 'ultralytics', 'PyQt5'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("Missing required dependencies:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nPlease install missing dependencies:")
        print("pip install -r requirements.txt")
        return False
    
    return True


def validate_environment() -> bool:
    """
    Validate the runtime environment.
    
    Returns:
        True if environment is valid, False otherwise
    """
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            print(f"Python 3.8+ required, got {sys.version}")
            return False
        
        # Check dependencies
        if not check_dependencies():
            return False
        
        # Check for CUDA availability (optional)
        try:
            import torch
            if torch.cuda.is_available():
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print("CUDA not available, using CPU")
        except ImportError:
            print("PyTorch not available")
        
        return True
        
    except Exception as e:
        print(f"Environment validation failed: {e}")
        return False


def main(args: Optional[List[str]] = None) -> int:
    """
    Main application entry point.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    try:
        # Parse arguments
        parsed_args = parse_arguments()
        
        # Set log level from arguments
        if parsed_args.debug:
            os.environ['LOG_LEVEL'] = 'DEBUG'
        else:
            os.environ['LOG_LEVEL'] = parsed_args.log_level
        
        # Validate environment
        if not validate_environment():
            return 1
        
        # Handle no-GUI mode
        if parsed_args.no_gui:
            return run_batch_mode(parsed_args)
        
        # Initialize application manager
        app_manager = ApplicationManager()
        
        # Initialize application
        if not app_manager.initialize(args):
            return 1
        
        # Show splash screen
        splash = app_manager.show_splash_screen()
        
        try:
            # Create main window
            if not app_manager.create_main_window():
                return 1
            
            # Apply command line arguments
            if parsed_args.model:
                app_manager.main_window.detection_display.set_model_file(parsed_args.model)
            
            if parsed_args.video:
                app_manager.main_window.detection_display.set_video_file(parsed_args.video)
            
            # Close splash screen
            if splash:
                splash.finish(app_manager.main_window)
            
            # Run application
            return app_manager.run()
            
        finally:
            if splash:
                splash.close()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


def run_batch_mode(args: argparse.Namespace) -> int:
    """
    Run in batch mode without GUI.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code
    """
    try:
        from src.core.video_processor import process_video_file
        
        if not args.model:
            print("Error: Model file required for batch mode")
            return 1
        
        if not args.video:
            print("Error: Video file required for batch mode")
            return 1
        
        print(f"Processing video: {args.video}")
        print(f"Using model: {args.model}")
        
        # Process video file
        results = process_video_file(
            video_path=args.video,
            model_path=args.model,
            output_path=args.video.replace('.mp4', '_processed.mp4') if args.video.endswith('.mp4') else None
        )
        
        print("Processing completed:")
        print(f"Status: {results.get('status', 'unknown')}")
        
        if results.get('status') == 'error':
            print(f"Error: {results.get('error', 'unknown error')}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Batch processing error: {e}")
        return 1


def create_desktop_shortcut():
    """Create desktop shortcut for the application."""
    try:
        if sys.platform == "win32":
            # Windows shortcut creation
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            shortcut_path = os.path.join(desktop, "Smart Recycling Detection.lnk")
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f'"{__file__}"'
            shortcut.WorkingDirectory = str(PROJECT_ROOT)
            shortcut.IconLocation = sys.executable
            shortcut.save()
            
            print(f"Desktop shortcut created: {shortcut_path}")
            
        else:
            print("Desktop shortcut creation not supported on this platform")
            
    except ImportError:
        print("Desktop shortcut creation requires: pip install winshell pywin32")
    except Exception as e:
        print(f"Failed to create desktop shortcut: {e}")


if __name__ == '__main__':
    # Handle special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == '--create-shortcut':
            create_desktop_shortcut()
            sys.exit(0)
        elif sys.argv[1] == '--check-env':
            if validate_environment():
                print("Environment validation passed")
                sys.exit(0)
            else:
                print("Environment validation failed")
                sys.exit(1)
    
    # Run main application
    exit_code = main()
    sys.exit(exit_code)