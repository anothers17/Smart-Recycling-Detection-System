"""
Video processing thread for Smart Recycling Detection System.

This module provides threaded video processing capabilities with
real-time detection, counting, and display updates.
"""

import cv2
import time
import numpy as np
import psutil
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

from config.settings import get_config
from config.logging_config import get_logger
from src.core.detector import RecyclingDetector, DetectionResult
from src.core.counter import RecyclingCounter
from src.core.output_writer import VideoOutputWriter
from src.core.camera_utils import detect_available_cameras
from src.utils.plotting import EnhancedAnnotator
from src.utils.image_utils import convert_color_space

logger = get_logger("main")


class VideoProcessor(QThread):
    """
    Threaded video processor for real-time detection and counting.

    Handles video input from files or webcam, performs detection and counting,
    and emits signals for GUI updates.
    """

    # PyQt signals for communication with GUI
    frameProcessed = pyqtSignal(np.ndarray)  # Processed frame with annotations
    statisticsUpdated = pyqtSignal(dict)  # Updated counting statistics
    performanceUpdated = pyqtSignal(dict)  # Performance metrics
    errorOccurred = pyqtSignal(str)  # Error messages
    processingFinished = pyqtSignal()  # Processing completed

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

            # Handle camera format (e.g., "camera:0")
            if isinstance(source, str) and source.startswith("camera:"):
                try:
                    camera_index = int(source.split(":")[1])
                    source = camera_index
                    logger.info(f"Parsed camera index from {source} to {camera_index}")
                except (IndexError, ValueError) as e:
                    logger.error(f"Invalid camera format: {source}")
                    return False

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
        Initialize video capture object with comprehensive fallback options.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.video_source is None:
                logger.error("No video source specified")
                raise ValueError("Video source not specified")

            # Create capture object with fallback backends
            if isinstance(self.video_source, int):
                # Webcam - try multiple backends in order of preference
                backends = [
                    (None, "Default"),
                    (cv2.CAP_DSHOW, "DSHOW"),
                    (cv2.CAP_MSMF, "MSMF"),
                    (cv2.CAP_VFW, "VFW"),
                    (cv2.CAP_ANY, "ANY"),
                ]

                for backend, backend_name in backends:
                    try:
                        if backend is not None:
                            self.cap = cv2.VideoCapture(self.video_source, backend)
                        else:
                            self.cap = cv2.VideoCapture(self.video_source)

                        if self.cap.isOpened():
                            # Test by reading a frame
                            ret, test_frame = self.cap.read()
                            if ret and test_frame is not None:
                                # Reset to beginning for video files
                                if not isinstance(self.video_source, int):
                                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                msg = f"Successfully opened camera {self.video_source} with {backend_name} backend"
                                logger.info(msg)
                                break
                            else:
                                self.cap.release()
                                self.cap = None
                        else:
                            self.cap = None

                    except Exception as backend_error:
                        logger.debug(
                            f"Failed with {backend_name} backend: {backend_error}"
                        )
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                        continue

                if not self.cap or not self.cap.isOpened():
                    available_cameras = self._detect_available_cameras()
                    error_msg = f"Failed to open camera {self.video_source}. "
                    if available_cameras:
                        error_msg += f"Available cameras: {available_cameras}"
                    else:
                        error_msg += "No cameras detected."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

            else:
                # Video file
                video_path = Path(self.video_source)
                if not video_path.exists():
                    raise FileNotFoundError(f"Video file not found: {video_path}")

                if not video_path.suffix.lower() in [
                    ".mp4",
                    ".avi",
                    ".mov",
                    ".mkv",
                    ".wmv",
                    ".flv",
                ]:
                    logger.warning(
                        f"Unsupported video format: {video_path.suffix}. Attempting to open anyway."
                    )

                self.cap = cv2.VideoCapture(str(video_path))

                if not self.cap.isOpened():
                    # Try alternative codecs/backends for files
                    backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
                    for backend in backends:
                        try:
                            self.cap = cv2.VideoCapture(str(video_path), backend)
                            if self.cap.isOpened():
                                logger.info(
                                    f"Successfully opened video file with backend {backend}"
                                )
                                break
                            else:
                                self.cap = None
                        except Exception as e:
                            logger.debug(f"Failed with backend {backend}: {e}")
                            continue

                if not self.cap or not self.cap.isOpened():
                    raise RuntimeError(f"Failed to open video file: {video_path}")

            # Validate capture properties
            try:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Sanity checks
                if width <= 0 or height <= 0:
                    raise ValueError(f"Invalid video dimensions: {width}x{height}")

                if fps <= 0:
                    fps = 30.0  # Default fallback
                    logger.warning("Invalid FPS detected, using default 30 FPS")

                logger.info(
                    f"Video initialized: {width}x{height} @ {fps:.2f} FPS, "
                    f"Total frames: {total_frames}"
                )

            except Exception as prop_error:
                logger.warning(f"Error reading video properties: {prop_error}")
                logger.info("Video capture initialized but properties unavailable")

            return True

        except (FileNotFoundError, ValueError, RuntimeError):
            # Re-raise specific exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing capture: {e}")
            raise RuntimeError(f"Failed to initialize video capture: {e}") from e

    def start_processing(self):
        """Start video processing with enhanced error handling."""
        try:
            # Validate detector
            if not self.detector or not self.detector.is_loaded:
                error_msg = "Detector not loaded or invalid"
                logger.error(error_msg)
                self.errorOccurred.emit(error_msg)
                return

            # Validate video source
            if self.video_source is None:
                error_msg = "No video source specified"
                logger.error(error_msg)
                self.errorOccurred.emit(error_msg)
                return

            # Test video capture initialization
            if not self._initialize_capture():
                error_msg = "Failed to initialize video capture"
                logger.error(error_msg)
                self.errorOccurred.emit(error_msg)
                return

            self.should_stop = False
            self.is_paused = False
            self.start()

            logger.info("Video processing started successfully")

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            error_msg = f"Failed to start processing: {e}"
            logger.error(error_msg)
            self.errorOccurred.emit(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error starting processing: {e}"
            logger.error(error_msg)
            self.errorOccurred.emit(error_msg)

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
                    output_writer.write_frame(processed_frame)

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
            rgb_frame = convert_color_space(frame, "BGR2RGB")

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

    def _create_annotated_frame(
        self,
        frame: np.ndarray,
        detection_result: DetectionResult,
        count_stats: Dict[str, int],
    ) -> np.ndarray:
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
                    show_confidence=True,
                )

            # Draw counting line
            line_x = self.counter.counting_line.x
            line_y = self.counter.counting_line.y

            if line_x is not None or line_y is not None:
                annotator.draw_counting_line(x=line_x, y=line_y, label="Counting Line")

            # Draw statistics
            stats_text = {"Total": self.counter.total_count, "Frame": self.frame_count}
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
                average_fps = (
                    self.frame_count / total_runtime if total_runtime > 0 else 0
                )
            else:
                average_fps = 0

            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / 1024 / 1024  # Convert to MB

            # Get system memory info
            system_memory = psutil.virtual_memory()
            system_memory_percent = system_memory.percent

            # Get detector performance stats
            detector_stats = self.detector.get_performance_stats()

            # Get counter statistics
            counter_stats = self.counter.get_statistics()

            # Combine all metrics
            performance_metrics = {
                "frame_count": self.frame_count,
                "average_fps": average_fps,
                "current_fps": self._calculate_current_fps(),
                "detector_fps": detector_stats.get("average_fps", 0),
                "processing_time": detector_stats.get("average_processing_time", 0),
                "memory_usage_mb": memory_usage_mb,
                "system_memory_percent": system_memory_percent,
                "detector_memory_mb": detector_stats.get("current_memory_usage_mb", 0),
                "total_detections": counter_stats.get("total_count", 0),
                "tracked_objects": counter_stats.get("tracked_objects", 0),
            }

            # Emit every 10 frames to avoid overwhelming GUI
            if self.frame_count % 10 == 0:
                self.performanceUpdated.emit(performance_metrics)

                # Log memory usage in performance logger
                perf_logger = get_logger("performance")
                perf_logger.info(
                    f"Memory usage: {memory_usage_mb:.1f} MB, "
                    f"System memory: {system_memory_percent:.1f}%, "
                    f"Detector memory: {detector_stats.get('current_memory_usage_mb', 0):.1f} MB"
                )

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def _initialize_output_writer(self) -> Optional[VideoOutputWriter]:
        """Initialize video writer for output saving."""
        if not self.cap or not self.output_path:
            return None

        writer = VideoOutputWriter(str(self.output_path), self.cap)
        if writer.initialize():
            return writer
        return None

    def _cleanup(self, output_writer: Optional[VideoOutputWriter] = None):
        """Clean up resources."""
        try:
            # Release video capture
            if self.cap:
                self.cap.release()
                self.cap = None

            # Release output writer
            if output_writer:
                output_writer.close()

            logger.info("Video processor cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _detect_available_cameras(self) -> List[int]:
        """
        Detect available camera indices.

        Returns:
            List of available camera indices
        """
        return detect_available_cameras()

    def get_video_info(self) -> Dict[str, Any]:
        """Get information about current video source."""
        if not self.cap:
            return {}

        try:
            info = {
                "source": str(self.video_source),
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "current_frame": int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
                "is_camera": isinstance(self.video_source, int),
            }

            return info

        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}


class WebcamProcessor(VideoProcessor):
    """Specialized video processor for webcam input."""

    def __init__(
        self,
        detector: RecyclingDetector,
        counter: RecyclingCounter,
        camera_index: int = 0,
    ):
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

    def __init__(
        self,
        detector: RecyclingDetector,
        counter: RecyclingCounter,
        video_path: Union[str, Path],
    ):
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
def create_processor(
    detector: RecyclingDetector,
    counter: RecyclingCounter,
    source: Union[str, int, Path],
) -> VideoProcessor:
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


def process_video_file(
    video_path: Union[str, Path],
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Process a video file and return statistics (non-GUI batch processing).

    Args:
        video_path: Path to input video
        model_path: Path to model file
        output_path: Optional path for output video

    Returns:
        Processing statistics
    """
    from src.core.model_factory import load_detector
    from src.core.counter import create_counter

    try:
        # Load detector and create counter
        detector = load_detector(model_path)
        counter = create_counter()

        # Create processor
        processor = FileProcessor(detector, counter, video_path)

        if output_path:
            processor.enable_output_saving(str(output_path))

        # Initialize video capture
        if not processor._initialize_capture():
            return {"status": "error", "error": "Failed to initialize video capture"}

        # Initialize output writer if needed
        output_writer = None
        if processor.save_output and processor.output_path:
            output_writer = VideoOutputWriter(str(processor.output_path), processor.cap)
            output_writer.initialize()

        # Reset counters and timers
        processor.frame_count = 0
        processor.start_time = time.time()
        processor.fps_timer = time.time()
        processor.fps_counter = 0
        processor.current_frame_skip = 0

        logger.info("Starting batch video processing...")

        # Main processing loop (synchronous)
        while True:
            # Read frame
            ret, frame = processor.cap.read()

            if not ret:
                logger.info("End of video reached")
                break

            # Skip frames if configured
            if processor.current_frame_skip < processor.skip_frames:
                processor.current_frame_skip += 1
                continue
            else:
                processor.current_frame_skip = 0

            # Process frame (without GUI signals)
            processor.frame_count += 1

            # Convert frame to RGB for detection
            rgb_frame = convert_color_space(frame, "BGR2RGB")

            # Perform detection
            detection_result = processor.detector.detect(rgb_frame)

            # Update counter
            count_stats = processor.counter.update(detection_result)

            # Create annotated frame (for output saving)
            if output_writer is not None:
                annotated_frame = processor._create_annotated_frame(
                    frame, detection_result, count_stats
                )
                output_writer.write_frame(annotated_frame)

            # Update FPS counter
            processor.fps_counter += 1

            # Log progress every 100 frames
            if processor.frame_count % 100 == 0:
                current_time = time.time()
                time_diff = current_time - processor.fps_timer
                if time_diff >= 1.0:
                    current_fps = processor.fps_counter / time_diff
                    processor.fps_counter = 0
                    processor.fps_timer = current_time
                    logger.info(
                        f"Processed {processor.frame_count} frames, current FPS: {current_fps:.2f}"
                    )

        # Processing completed
        logger.info(
            f"Batch processing completed: {processor.frame_count} frames processed"
        )

        # Collect final statistics
        final_stats = {
            "status": "completed",
            "video_path": str(video_path),
            "model_path": str(model_path),
            "output_path": str(output_path) if output_path else None,
            "total_frames_processed": processor.frame_count,
            "processing_time_seconds": time.time() - processor.start_time,
        }

        # Add counter statistics
        counter_stats = processor.counter.get_statistics()
        final_stats.update(
            {
                "total_objects_counted": counter_stats.get("total_count", 0),
                "class_counts": counter_stats.get("class_counts", {}),
                "direction_counts": counter_stats.get("direction_counts", {}),
                "tracked_objects": counter_stats.get("tracked_objects", 0),
                "crossed_objects": counter_stats.get("crossed_objects", 0),
                "target_classes": counter_stats.get("target_classes", []),
            }
        )

        # Add detector performance stats
        detector_stats = processor.detector.get_performance_stats()
        final_stats.update(
            {
                "average_detection_fps": detector_stats.get("average_fps", 0),
                "average_processing_time": detector_stats.get(
                    "average_processing_time", 0
                ),
                "average_detections_per_frame": detector_stats.get(
                    "average_detections_per_frame", 0
                ),
                "average_memory_usage_mb": detector_stats.get("average_memory_usage_mb", 0),
                "peak_memory_usage_mb": detector_stats.get("peak_memory_usage_mb", 0),
            }
        )

        # Calculate overall FPS
        if final_stats["processing_time_seconds"] > 0:
            final_stats["overall_fps"] = (
                processor.frame_count / final_stats["processing_time_seconds"]
            )
        else:
            final_stats["overall_fps"] = 0

        # Cleanup
        if output_writer:
            output_writer.close()
        processor._cleanup()

        logger.info(
            f"Batch processing results: {final_stats['total_objects_counted']} objects counted"
        )
        return final_stats

    except Exception as e:
        logger.error(f"Error processing video file: {e}")
        return {"status": "error", "error": str(e)}
