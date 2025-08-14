"""
Enhanced debug script to analyze what's happening during video processing.
This will help us understand why only bottle-plastic is being counted.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_config
from config.logging_config import get_logger
from src.core.detector import RecyclingDetector, DetectorFactory
from src.core.counter import RecyclingCounter, create_counter
import cv2
import numpy as np

# Enable detailed debug logging
logging.getLogger('recycling_detection').setLevel(logging.DEBUG)
logging.getLogger('recycling_detection.counter').setLevel(logging.DEBUG)
logging.getLogger('recycling_detection.detection').setLevel(logging.DEBUG)

logger = get_logger('debug')


def analyze_video_detections(video_path: str, sample_frames: int = 10):
    """
    Analyze detections in a video file to see what classes are being detected.
    
    Args:
        video_path: Path to the video file
        sample_frames: Number of frames to sample for analysis
    """
    try:
        print("🎥 ANALYZING VIDEO DETECTIONS")
        print("=" * 60)
        
        # Load detector
        detector = DetectorFactory.create_from_config()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video file: {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📹 Video info: {total_frames} frames at {fps:.2f} FPS")
        
        # Sample frames evenly throughout the video
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        
        all_detections = {}
        detection_summary = {
            'bottle-glass': 0,
            'bottle-plastic': 0,
            'tin can': 0,
            'other': 0
        }
        
        for i, frame_idx in enumerate(frame_indices):
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            print(f"\n🔍 Analyzing frame {frame_idx} ({i+1}/{sample_frames})")
            
            # Convert to RGB for detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform detection without target class filtering first
            result = detector.detect(rgb_frame, target_classes=None)
            
            if result.detections:
                print(f"   Total detections: {len(result.detections)}")
                
                for detection in result.detections:
                    class_name = detection.class_name
                    confidence = detection.confidence
                    center = detection.center
                    
                    print(f"   - {class_name}: confidence={confidence:.3f}, center=({center[0]:.1f}, {center[1]:.1f})")
                    
                    # Count by class
                    if class_name in detection_summary:
                        detection_summary[class_name] += 1
                    else:
                        detection_summary['other'] += 1
                
                # Now test with target class filtering
                config = get_config()
                filtered_result = detector.detect(rgb_frame, target_classes=config.counting.target_classes)
                
                if len(filtered_result.detections) != len(result.detections):
                    print(f"   🔄 After filtering: {len(filtered_result.detections)} detections")
                    
            else:
                print("   No detections in this frame")
        
        cap.release()
        detector.cleanup()
        
        print("\n📊 DETECTION SUMMARY")
        print("=" * 40)
        total_detections = sum(detection_summary.values())
        
        for class_name, count in detection_summary.items():
            if count > 0:
                percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                print(f"{class_name}: {count} detections ({percentage:.1f}%)")
        
        print(f"\nTotal detections across {sample_frames} sample frames: {total_detections}")
        
        # Analysis
        print("\n🔍 ANALYSIS")
        print("=" * 40)
        
        if detection_summary['bottle-plastic'] > 0 and (detection_summary['bottle-glass'] == 0 and detection_summary['tin can'] == 0):
            print("⚠️  Only bottle-plastic detected in sampled frames")
            print("   This could mean:")
            print("   - Your video only contains bottle-plastic objects")
            print("   - Other objects are present but not being detected by the model")
            print("   - Other objects are too small, blurry, or at difficult angles")
        
        elif detection_summary['bottle-glass'] > 0 or detection_summary['tin can'] > 0:
            print("✅ Multiple target classes detected!")
            print("   The counting issue might be related to:")
            print("   - Object tracking (objects not being tracked consistently)")
            print("   - Counting line position (objects not crossing the line)")
            print("   - Validation criteria (objects failing distance/confidence checks)")
        
        elif total_detections == 0:
            print("❌ No detections found in sampled frames")
            print("   This could indicate:")
            print("   - Model confidence threshold too high")
            print("   - Objects too small or unclear in the video")
            print("   - Model not properly trained for this type of content")
        
    except Exception as e:
        logger.error(f"Error analyzing video: {e}")


def test_counting_line_position():
    """Test if the counting line position makes sense for the video."""
    print("\n📏 COUNTING LINE ANALYSIS")
    print("=" * 60)
    
    config = get_config()
    line_x = config.counting.line_position_x
    line_y = config.counting.line_position_y
    
    print(f"Counting line position: x={line_x}, y={line_y}")
    
    # Load a sample frame to visualize
    video_path = "src/resources/sample_videos"  # Adjust path as needed
    sample_videos = list(Path(video_path).glob("*.mp4")) if Path(video_path).exists() else []
    
    if sample_videos:
        cap = cv2.VideoCapture(str(sample_videos[0]))
        ret, frame = cap.read()
        
        if ret:
            height, width = frame.shape[:2]
            print(f"Video dimensions: {width}x{height}")
            
            if line_x is not None:
                if 0 <= line_x <= width:
                    print(f"✅ Vertical line at x={line_x} is within frame bounds")
                    position_percent = (line_x / width) * 100
                    print(f"   Line is at {position_percent:.1f}% from left edge")
                else:
                    print(f"❌ Vertical line at x={line_x} is outside frame bounds (0-{width})")
            
            if line_y is not None:
                if 0 <= line_y <= height:
                    print(f"✅ Horizontal line at y={line_y} is within frame bounds")
                    position_percent = (line_y / height) * 100
                    print(f"   Line is at {position_percent:.1f}% from top edge")
                else:
                    print(f"❌ Horizontal line at y={line_y} is outside frame bounds (0-{height})")
        
        cap.release()


def main():
    """Run enhanced debug analysis."""
    print("🔍 ENHANCED RECYCLING DETECTION DEBUG")
    print("=" * 60)
    
    # Find video file
    config = get_config()
    sample_videos_dir = config.paths.sample_videos_dir
    
    video_files = []
    if sample_videos_dir.exists():
        video_files = list(sample_videos_dir.glob("*.mp4"))
    
    if not video_files:
        print("❌ No video files found in sample_videos directory")
        print(f"   Expected location: {sample_videos_dir}")
        print("   Please provide a video file path manually:")
        video_path = input("Enter video file path: ").strip()
        if video_path and Path(video_path).exists():
            video_files = [Path(video_path)]
        else:
            print("❌ Invalid video path. Exiting.")
            return
    
    # Analyze the first video file
    video_path = str(video_files[0])
    print(f"📹 Analyzing video: {video_path}")
    
    # Test 1: Analyze detections
    analyze_video_detections(video_path, sample_frames=20)
    
    # Test 2: Check counting line position
    test_counting_line_position()
    
    print("\n🎯 RECOMMENDATIONS BASED ON ANALYSIS:")
    print("=" * 60)
    print("1. If only bottle-plastic is detected:")
    print("   - Check if your video actually contains tin cans and glass bottles")
    print("   - Verify model training data included diverse examples")
    print("   - Consider lowering confidence threshold for testing")
    
    print("\n2. If multiple classes are detected but not counted:")
    print("   - Check counting line position relative to object movement")
    print("   - Verify objects are actually crossing the counting line")
    print("   - Check validation criteria (distance, confidence thresholds)")
    
    print("\n3. Enable real-time debug logging:")
    print("   - Set logging level to DEBUG")
    print("   - Watch for 'CROSSING DETECTED' and 'OBJECT COUNTED' messages")
    print("   - Look for 'Filtered out class' messages")


if __name__ == "__main__":
    main()