"""
Video processing thread for Smart Recycling Detection System.

This module provides threaded video processing capabilities with
real-time detection, counting, and display updates.
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional, Union, Callable, Dict, Any
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
import queue
import threading

from config.settings import get_config
from config.logging_config import get_logger
from src.core.detector import RecyclingDetector, DetectionResult
from src.core.counter import RecyclingCounter
from src.utils.plotting import EnhancedAnnotator, draw_detections
from src.utils.image_utils import convert_color_space

logger = get_logger('main')


class VideoProcessor(QThread):
    """
    Threaded video processor for real-time detection and counting.
    
    Handles video input from files or webcam, performs detection and counting,
    and emits signals for GUI updates.
    """
    
    # PyQt signals for communication with GUI
    frameProcessed = pyqtSignal(np.ndarray)  # Processed frame with annotations
    statisticsUpdated = pyqtSignal(dict)      # Updated counting statistics
    performanceUpdated = pyqtSignal(dict)     # Performance metrics
    errorOccurred = pyqtSignal(str)           # Error messages
    processingFinished = pyqtSignal()         # Processing completed
    
    def __init__(self, detector: RecyclingDetector, counter: RecyclingCounter):
        """
        Initialize video processor.
        
        Args:
            detector: Detection engine
            counter: Counting system
        """
        super().__init__()
        
        self.config = get_config()
        self.detector = detector
        self.counter = counter
        
        # Video source
        self.video_source: Optional[Union[str, int]] = None
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Processing control
        self.is_running = False
        self.is_paused = False
        self.should_stop = False
        
        # Threading synchronization
        self.mutex = QMutex()
        self.pause_condition = QWaitCondition()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.fps_counter = 0
        self.fps_timer = time.time()
        
        # Processing options
        self.skip_frames = 0
        self.current_frame_skip = 0
        self.save_output = False
        self.output_path = None
        
        logger.info("Video processor initialized")
    
    def set_video_source(self, source: Union[str, int, Path]) -> bool:
        """
        Set video input source.
        
        Args:
            source: Video file path, camera index, or Path object
            
        Returns:
            True if source is valid, False otherwise
        """
        try:
            if isinstance(source, Path):
                source = str(source)
            
            # Validate video file if string path
            if isinstance(source, str) and source.isdigit():
                source = int(source)  # Convert to camera index
            
            if isinstance(source, str):
                if not Path(source).exists():
                    logger.error(f"Video file not found: {source}")
                    return False
            
            self.video_source = source
            logger.info(f"Video source set to: {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting video source: {e}")
            return False
    
    def _initialize_capture(self) -> bool:
        """
        Initialize video capture object.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.video_source is None:
                logger.error("No video source specified")
                return False
            
            # Create capture object
            if isinstance(self.video_source, int):
                # Webcam
                self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
            else:
                # Video file
                self.cap = cv2.VideoCapture(str(self.video_source))
            
            if not self.cap or not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.video_source}")
                return False
            
            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video initialized: {width}x{height} @ {fps:.2f} FPS, "
                       f"Total frames: {total_frames}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing capture: {e}")
            return False
    
    def start_processing(self):
        """Start video processing."""
        if not self.detector.is_loaded:
            self.errorOccurred.emit("Detector not loaded")
            return
        
        if self.video_source is None:
            self.errorOccurred.emit("No video source specified")
            return
        
        self.should_stop = False
        self.is_paused = False
        self.start()
        
        logger.info("Video processing started")
    
    def stop_processing(self):
        """Stop video processing."""
        self.should_stop = True
        self.is_paused = False
        
        # Wake up thread if paused
        self.mutex.lock()
        self.pause_condition.wakeAll()
        self.mutex.unlock()
        
        logger.info("Video processing stop requested")
    
    def pause_processing(self):
        """Pause video processing."""
        self.is_paused = True
        logger.info("Video processing paused")
    
    def resume_processing(self):
        """Resume video processing."""
        self.mutex.lock()
        self.is_paused = False
        self.pause_condition.wakeAll()
        self.mutex.unlock()
        
        logger.info("Video processing resumed")
    
    def set_skip_frames(self, skip_count: int):
        """Set number of frames to skip for faster processing."""
        self.skip_frames = max(0, skip_count)
        logger.info(f"Frame skipping set to: {self.skip_frames}")
    
    def enable_output_saving(self, output_path: str):
        """Enable saving of processed video output."""
        self.save_output = True
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output saving enabled: {output_path}")
    
    def disable_output_saving(self):
        """Disable saving of processed video output."""
        self.save_output = False
        self.output_path = None
        logger.info("Output saving disabled")
    
    def run(self):
        """Main processing loop (runs in separate thread)."""
        try:
            # Initialize video capture
            if not self._initialize_capture():
                self.errorOccurred.emit("Failed to initialize video capture")
                return
            
            # Initialize output writer if needed
            output_writer = None
            if self.save_output and self.output_path:
                output_writer = self._initialize_output_writer()
            
            # Reset counters
            self.frame_count = 0
            self.start_time = time.time()
            self.fps_timer = time.time()
            self.fps_counter = 0
            
            self.is_running = True
            
            logger.info("Starting video processing loop")
            
            # Main processing loop
            while not self.should_stop:
                # Handle pause
                self.mutex.lock()
                if self.is_paused:
                    self.pause_condition.wait(self.mutex)
                self.mutex.unlock()
                
                if self.should_stop:
                    break
                
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.info("End of video or failed to read frame")
                    break
                
                # Skip frames if configured
                if self.current_frame_skip < self.skip_frames:
                    self.current_frame_skip += 1
                    continue
                else:
                    self.current_frame_skip = 0
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Emit processed frame
                self.frameProcessed.emit(processed_frame)
                
                # Save output if enabled
                if output_writer is not None:
                    output_writer.write(processed_frame)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Small delay to prevent overwhelming the GUI
                self.msleep(1)
            
            logger.info("Video processing loop ended")
            
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            self.errorOccurred.emit(str(e))
        
        finally:
            # Cleanup
            self._cleanup(output_writer)
            self.is_running = False
            self.processingFinished.emit()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with detection and counting.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with annotations
        """
        try:
            self.frame_count += 1
            
            # Convert frame to RGB for detection
            rgb_frame = convert_color_space(frame, 'BGR2RGB')
            
            # Perform detection
            detection_result = self.detector.detect(rgb_frame)
            
            # Update counter
            count_stats = self.counter.update(detection_result)
            
            # Create annotated frame
            annotated_frame = self._create_annotated_frame(
                frame, detection_result, count_stats
            )
            
            # Emit statistics update
            self.statisticsUpdated.emit(count_stats)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error processing frame {self.frame_count}: {e}")
            return frame  # Return original frame on error
    
    def _create_annotated_frame(self, frame: np.ndarray, 
                               detection_result: DetectionResult,
                               count_stats: Dict[str, int]) -> np.ndarray:
        """
        Create annotated frame with detections and counting information.
        
        Args:
            frame: Original frame
            detection_result: Detection results
            count_stats: Counting statistics
            
        Returns:
            Annotated frame
        """
        try:
            # Create annotator
            annotator = EnhancedAnnotator(frame)
            
            # Draw detections
            for detection in detection_result.detections:
                annotator.draw_detection(
                    detection.xyxy,
                    detection.class_name,
                    detection.confidence,
                    show_confidence=True
                )
            
            # Draw counting line
            line_x = self.counter.counting_line.x
            line_y = self.counter.counting_line.y
            
            if line_x is not None or line_y is not None:
                annotator.draw_counting_line(
                    x=line_x,
                    y=line_y,
                    label="Counting Line"
                )
            
            # Draw statistics
            stats_text = {
                'Total': self.counter.total_count,
                'Frame': self.frame_count
            }
            stats_text.update(count_stats)
            
            annotator.draw_statistics(stats_text, position=(10, 30))
            
            # Draw FPS
            current_fps = self._calculate_current_fps()
            annotator.draw_fps(current_fps, position=(10, frame.shape[0] - 30))
            
            return annotator.get_result()
            
        except Exception as e:
            logger.error(f"Error creating annotated frame: {e}")
            return frame
    
    def _calculate_current_fps(self) -> float:
        """Calculate current processing FPS."""
        current_time = time.time()
        time_diff = current_time - self.fps_timer
        
        if time_diff >= 1.0:  # Update every second
            current_fps = self.fps_counter / time_diff
            self.fps_counter = 0
            self.fps_timer = current_time
            return current_fps
        else:
            self.fps_counter += 1
            return 0.0  # Return 0 if not ready to calculate
    
    def _update_performance_metrics(self):
        """Update and emit performance metrics."""
        try:
            current_time = time.time()
            
            if self.start_time:
                total_runtime = current_time - self.start_time
                average_fps = self.frame_count / total_runtime if total_runtime > 0 else 0
            else:
                average_fps = 0
            
            # Get detector performance stats
            detector_stats = self.detector.get_performance_stats()
            
            # Get counter statistics
            counter_stats = self.counter.get_statistics()
            
            # Combine all metrics
            performance_metrics = {
                'frame_count': self.frame_count,
                'average_fps': average_fps,
                'current_fps': self._calculate_current_fps(),
                'detector_fps': detector_stats.get('average_fps', 0),
                'processing_time': detector_stats.get('average_processing_time', 0),
                'total_detections': counter_stats.get('total_count', 0),
                'tracked_objects': counter_stats.get('tracked_objects', 0)
            }
            
            # Emit every 10 frames to avoid overwhelming GUI
            if self.frame_count % 10 == 0:
                self.performanceUpdated.emit(performance_metrics)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _initialize_output_writer(self) -> Optional[cv2.VideoWriter]:
        """Initialize video writer for output saving."""
        try:
            if not self.cap:
                return None
            
            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                fps,
                (width, height)
            )
            
            if writer.isOpened():
                logger.info(f"Output writer initialized: {self.output_path}")
                return writer
            else:
                logger.error("Failed to initialize output writer")
                return None
                
        except Exception as e:
            logger.error(f"Error initializing output writer: {e}")
            return None
    
    def _cleanup(self, output_writer: Optional[cv2.VideoWriter] = None):
        """Clean up resources."""
        try:
            # Release video capture
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # Release output writer
            if output_writer:
                output_writer.release()
            
            logger.info("Video processor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_video_info(self) -> Dict[str, Any]:
        """Get information about current video source."""
        if not self.cap:
            return {}
        
        try:
            info = {
                'source': str(self.video_source),
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'current_frame': int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
                'is_camera': isinstance(self.video_source, int)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}


class WebcamProcessor(VideoProcessor):
    """Specialized video processor for webcam input."""
    
    def __init__(self, detector: RecyclingDetector, counter: RecyclingCounter, 
                 camera_index: int = 0):
        """
        Initialize webcam processor.
        
        Args:
            detector: Detection engine
            counter: Counting system
            camera_index: Camera device index
        """
        super().__init__(detector, counter)
        self.set_video_source(camera_index)
        logger.info(f"Webcam processor initialized for camera {camera_index}")


class FileProcessor(VideoProcessor):
    """Specialized video processor for file input."""
    
    def __init__(self, detector: RecyclingDetector, counter: RecyclingCounter, 
                 video_path: Union[str, Path]):
        """
        Initialize file processor.
        
        Args:
            detector: Detection engine
            counter: Counting system
            video_path: Path to video file
        """
        super().__init__(detector, counter)
        self.set_video_source(video_path)
        logger.info(f"File processor initialized for: {video_path}")


# Utility functions
def create_processor(detector: RecyclingDetector, counter: RecyclingCounter,
                    source: Union[str, int, Path]) -> VideoProcessor:
    """
    Create appropriate video processor based on source type.
    
    Args:
        detector: Detection engine
        counter: Counting system
        source: Video source (file path or camera index)
        
    Returns:
        VideoProcessor instance
    """
    if isinstance(source, int):
        return WebcamProcessor(detector, counter, source)
    else:
        return FileProcessor(detector, counter, source)


def process_video_file(video_path: Union[str, Path], 
                      model_path: Union[str, Path],
                      output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Process a video file and return statistics (non-GUI batch processing).
    
    Args:
        video_path: Path to input video
        model_path: Path to model file
        output_path: Optional path for output video
        
    Returns:
        Processing statistics
    """
    from src.core.detector import load_detector
    from src.core.counter import create_counter
    
    try:
        # Load detector and create counter
        detector = load_detector(model_path)
        counter = create_counter()
        
        # Create processor
        processor = FileProcessor(detector, counter, video_path)
        
        if output_path:
            processor.enable_output_saving(str(output_path))
        
        # Process video (this would need modification for non-GUI use)
        # For now, return placeholder statistics
        stats = {
            'status': 'completed',
            'video_path': str(video_path),
            'model_path': str(model_path),
            'output_path': str(output_path) if output_path else None
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error processing video file: {e}")
        return {'status': 'error', 'error': str(e)}