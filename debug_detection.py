"""
Debug script to test detection and counting functionality.

This script helps verify that all target classes are being detected and counted correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_config
from config.logging_config import get_logger
from src.core.detector import RecyclingDetector, DetectorFactory
from src.core.counter import RecyclingCounter, create_counter

logger = get_logger('debug')


def test_model_classes():
    """Test what classes the model can detect."""
    try:
        config = get_config()
        detector = DetectorFactory.create_from_config()
        
        model_info = detector.get_model_info()
        
        print("=" * 60)
        print("MODEL INFORMATION")
        print("=" * 60)
        print(f"Model Status: {model_info.get('status')}")
        print(f"Model Path: {model_info.get('model_path')}")
        print(f"Device: {model_info.get('device')}")
        print(f"Number of Classes: {model_info.get('num_classes')}")
        
        if 'class_names' in model_info:
            print("\nAvailable Classes in Model:")
            for i, class_name in enumerate(model_info['class_names']):
                print(f"  {i}: {class_name}")
        
        print("\nTarget Classes from Config:")
        target_classes = config.counting.target_classes
        for class_name in target_classes:
            print(f"  - {class_name}")
        
        # Check for mismatches
        if 'class_names' in model_info:
            model_classes = set(model_info['class_names'])
            target_set = set(target_classes)
            
            missing = target_set - model_classes
            extra = model_classes - target_set
            
            if missing:
                print(f"\n⚠️  WARNING: Target classes not found in model: {missing}")
            
            if extra:
                print(f"\n📝 Note: Model has additional classes: {extra}")
            
            if not missing:
                print(f"\n✅ All target classes are available in the model!")
        
        detector.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"Error testing model classes: {e}")
        return False


def test_detection_filtering():
    """Test detection with target class filtering."""
    try:
        import numpy as np
        
        config = get_config()
        detector = DetectorFactory.create_from_config()
        
        # Create a dummy image for testing
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        print("\n" + "=" * 60)
        print("DETECTION FILTERING TEST")
        print("=" * 60)
        
        # Test with target classes
        target_classes = config.counting.target_classes
        print(f"Testing detection with target classes: {target_classes}")
        
        result = detector.detect(dummy_image, target_classes=target_classes)
        
        print(f"Detections returned: {len(result.detections)}")
        
        if result.detections:
            detected_classes = [det.class_name for det in result.detections]
            print(f"Detected classes: {detected_classes}")
            
            # Check if all detected classes are in target classes
            for class_name in detected_classes:
                if class_name in target_classes:
                    print(f"  ✅ {class_name} - Target class")
                else:
                    print(f"  ❌ {class_name} - NOT in target classes")
        else:
            print("No detections (expected with random image)")
        
        # Test without target class filtering
        print(f"\nTesting detection without filtering...")
        result_all = detector.detect(dummy_image, target_classes=None)
        print(f"Detections without filtering: {len(result_all.detections)}")
        
        detector.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"Error testing detection filtering: {e}")
        return False


def test_counter_configuration():
    """Test counter configuration and target classes."""
    try:
        config = get_config()
        
        print("\n" + "=" * 60)
        print("COUNTER CONFIGURATION TEST")
        print("=" * 60)
        
        # Create counter
        counter = create_counter()
        
        print(f"Counter target classes: {counter.target_classes}")
        print(f"Config target classes: {config.counting.target_classes}")
        
        # Test setting custom target classes
        custom_classes = ['bottle-plastic', 'tin can']
        counter.set_target_classes(custom_classes)
        print(f"After setting custom classes: {counter.target_classes}")
        
        # Get statistics
        stats = counter.get_statistics()
        print(f"\nCounter statistics: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing counter configuration: {e}")
        return False


def debug_class_names():
    """Debug class name matching issues."""
    print("\n" + "=" * 60)
    print("CLASS NAME DEBUGGING")
    print("=" * 60)
    
    # Common class name variations
    common_variations = [
        'bottle-plastic',
        'bottle_plastic', 
        'plastic bottle',
        'plastic_bottle',
        'bottle-glass',
        'bottle_glass',
        'glass bottle',
        'glass_bottle',
        'tin can',
        'tin_can',
        'can',
        'metal can'
    ]
    
    config = get_config()
    target_classes = config.counting.target_classes
    
    print("Target classes from config:")
    for cls in target_classes:
        print(f"  '{cls}' (length: {len(cls)})")
    
    print("\nCommon variations to check:")
    for variation in common_variations:
        if variation in target_classes:
            print(f"  ✅ '{variation}' - Found in targets")
        else:
            print(f"  ❌ '{variation}' - Not in targets")
    
    # Check for whitespace issues
    print("\nChecking for whitespace issues:")
    for cls in target_classes:
        if cls != cls.strip():
            print(f"  ⚠️  '{cls}' has leading/trailing whitespace")
        else:
            print(f"  ✅ '{cls}' - No whitespace issues")


def main():
    """Run all debug tests."""
    print("🔍 RECYCLING DETECTION DEBUG TOOL")
    print("=" * 60)
    
    # Test 1: Model classes
    print("\n1. Testing model classes...")
    test_model_classes()
    
    # Test 2: Detection filtering
    print("\n2. Testing detection filtering...")
    test_detection_filtering()
    
    # Test 3: Counter configuration
    print("\n3. Testing counter configuration...")
    test_counter_configuration()
    
    # Test 4: Class name debugging
    print("\n4. Debugging class names...")
    debug_class_names()
    
    print("\n" + "=" * 60)
    print("🎯 DEBUG COMPLETE")
    print("=" * 60)
    
    print("\n📋 RECOMMENDATIONS:")
    print("1. Replace the detector.py file with the fixed version")
    print("2. Replace the counter.py file with the improved version")
    print("3. Verify that model class names match config target classes exactly")
    print("4. Check for case sensitivity and whitespace in class names")
    print("5. Enable debug logging to see detection filtering in action")


if __name__ == "__main__":
    main()