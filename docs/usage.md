# Usage Guide

This comprehensive guide covers all aspects of using the Smart Recycling Detection System, from basic operation to advanced features.

## 🚀 Getting Started

### First Launch

1. **Start the application**
   ```bash
   python src/main.py
   ```

2. **The main window will open** with two main panels:
   - **Left Panel**: Detection display and file selection
   - **Right Panel**: Control panel with counters, settings, and activity log

### Basic Workflow

1. **Load Model** → **Select Video** → **Start Detection** → **Monitor Results**

## 📁 Loading Files

### Loading a Model File

**Method 1: Browse Button**
1. Click **"Browse Model"** in the file selection area
2. Navigate to your model file (`.pt`, `.pth`, or `.onnx`)
3. Select your trained YOLOv8 model
4. The model path will appear in the display field

**Method 2: Drag and Drop**
1. Drag your model file from file explorer
2. Drop it onto the model selection area
3. The file will be automatically loaded

**Supported Model Formats:**
- `.pt` - PyTorch model (recommended)
- `.pth` - PyTorch model 
- `.onnx` - ONNX format
- `.trt` - TensorRT engine

### Loading a Video File

**Method 1: Browse Button**
1. Click **"Browse Video"** in the file selection area
2. Navigate to your video file
3. Select the video you want to process

**Method 2: Drag and Drop**
1. Drag your video file from file explorer
2. Drop it onto the video selection area

**Method 3: Webcam**
1. Leave video field empty
2. The system will default to webcam (camera index 0)

**Supported Video Formats:**
- `.mp4` - MP4 video (recommended)
- `.avi` - AVI video
- `.mkv` - Matroska video
- `.mov` - QuickTime video
- `.wmv` - Windows Media Video
- `.flv` - Flash Video
- `.webm` - WebM video

## 🎮 Main Controls

### Control Buttons

#### Start Detection
- **Purpose**: Begin real-time detection and counting
- **Requirements**: Model and video must be loaded
- **Keyboard Shortcut**: `Space` (when window is focused)
- **Visual Feedback**: Button becomes disabled, processing starts

#### Pause/Resume
- **Purpose**: Temporarily pause processing
- **Function**: Preserves current state while pausing video processing
- **Resume**: Click the same button to resume

#### Stop
- **Purpose**: Stop processing completely
- **Function**: Stops video processing and resets to ready state
- **Note**: Counting statistics are preserved until reset

#### Reset All
- **Purpose**: Reset all counters and clear display
- **Function**: 
  - Clears all counting statistics
  - Resets LCD displays to zero
  - Clears video display
  - Resets performance metrics
- **Warning**: This action cannot be undone

## 📊 Understanding the Interface

### Detection Display Panel

**Video Display Area**
- Shows real-time video with detection overlays
- Bounding boxes around detected objects
- Class labels and confidence scores
- Counting line visualization
- Performance metrics overlay

**Information Bar**
- **FPS**: Current processing speed
- **Resolution**: Video resolution
- **Frame**: Current frame number

### Control Panel

#### Counting Displays
- **Individual Counters**: LCD displays for each recyclable type
  - 🔵 **Bottle Glass**: Blue LCD display
  - 🟠 **Bottle Plastic**: Orange LCD display  
  - ⚫ **Tin Can**: Gray LCD display
- **Total Counter**: Green LCD showing total count across all types

#### Settings Tab
**Confidence Threshold**
- **Range**: 0.10 to 0.95
- **Default**: 0.70
- **Function**: Minimum confidence for object detection
- **Impact**: Higher values = fewer false positives, might miss some objects

**Line Position**
- **Range**: 50 to 1000 pixels
- **Default**: 300
- **Function**: X-coordinate of the vertical counting line
- **Usage**: Objects crossing this line are counted

**Display Options**
- ☑️ **Show Confidence Scores**: Display confidence percentages
- ☑️ **Show FPS Counter**: Display performance metrics
- ☑️ **Show Counting Line**: Display the red counting line

#### Activity Log Tab
- **Real-time Logging**: Live activity feed with timestamps
- **Color Coding**: 
  - 🟢 Green: Success messages
  - 🟡 Yellow: Warnings
  - 🔴 Red: Errors
  - ⚪ White: Information
- **Auto-scroll**: Automatically scrolls to show latest entries
- **Clear Button**: Clear all log entries
- **Save Log**: Export log to text file

## 🎯 Detection and Counting

### How Detection Works

1. **Frame Capture**: Video frames are captured from source
2. **AI Processing**: YOLOv8 analyzes each frame for recyclable objects
3. **Object Tracking**: Objects are tracked across frames
4. **Line Crossing**: When objects cross the counting line, they're counted
5. **Display Update**: Results are displayed in real-time

### Understanding the Counting Line

**Purpose**: The red vertical line determines when objects are counted

**Behavior**:
- Objects crossing from **left to right** are counted
- Objects crossing from **right to left** are counted
- Objects must completely cross the line (with tolerance)
- Same object won't be counted twice (anti-double-counting)

**Positioning Tips**:
- Place line where objects naturally flow
- Avoid areas with complex object interactions
- Consider camera angle and object movement patterns

### Counting Accuracy

**Factors Affecting Accuracy**:
- **Video Quality**: Higher resolution = better detection
- **Lighting**: Good lighting improves detection
- **Object Size**: Objects should be clearly visible
- **Movement Speed**: Moderate speed works best
- **Camera Angle**: Front-facing view recommended

**Optimization Tips**:
- Use good lighting conditions
- Position camera at appropriate distance
- Ensure objects pass through frame clearly
- Adjust confidence threshold based on your model's performance

## ⚙️ Advanced Features

### Command Line Usage

#### Basic Command Line
```bash
# Start with specific files
python src/main.py --model models/best.pt --video videos/test.mp4

# Enable debug logging
python src/main.py --debug

# Batch processing without GUI
python src/main.py --no-gui --model models/best.pt --video videos/test.mp4
```

#### Environment Variables
```bash
# Set detection confidence
export DETECTION_CONFIDENCE=0.8

# Force CPU usage
export DETECTION_DEVICE=cpu

# Set UI theme
export UI_THEME=dark

# Set log level
export LOG_LEVEL=DEBUG
```

### Configuration Files

#### Creating Custom Configuration
```bash
# Copy default settings
cp config/settings.py config/user_settings.py

# Edit settings
nano config/user_settings.py
```

#### Configuration Options
```python
# Detection settings
detection:
  confidence_threshold: 0.7    # Detection confidence (0.0-1.0)
  input_size: 640             # Model input size (320, 640, 1280)
  device: "auto"              # "auto", "cpu", "cuda", "mps"

# Counting settings
counting:
  line_position_x: 300        # Counting line X position
  tracking_enabled: true      # Enable object tracking
  target_classes:             # Classes to detect and count
    - "bottle-glass"
    - "bottle-plastic" 
    - "tin can"

# UI settings
ui:
  theme: "modern"             # "modern", "dark", "light"
  window_width: 1240          # Window width in pixels
  window_height: 730          # Window height in pixels
```

### Batch Processing

#### Process Multiple Videos
```python
# Python script for batch processing
from src.core.video_processor import process_video_file

videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
model_path = "models/best.pt"

for video in videos:
    print(f"Processing {video}...")
    results = process_video_file(
        video_path=video,
        model_path=model_path,
        output_path=f"processed_{video}"
    )
    print(f"Results: {results}")
```

#### Automated Processing Script
```bash
#!/bin/bash
# Process all videos in a directory

MODEL_PATH="models/best.pt"
INPUT_DIR="input_videos"
OUTPUT_DIR="output_videos"

for video in "$INPUT_DIR"/*.mp4; do
    filename=$(basename "$video" .mp4)
    echo "Processing $filename..."
    
    python src/main.py --no-gui \
        --model "$MODEL_PATH" \
        --video "$video" \
        --output "$OUTPUT_DIR/${filename}_processed.mp4"
done
```

## 📈 Performance Optimization

### Hardware Optimization

#### GPU Acceleration
```bash
# Enable CUDA
export DETECTION_DEVICE=cuda

# Verify GPU usage
nvidia-smi  # Monitor GPU utilization while running
```

#### CPU Optimization
```bash
# Set thread count
export OMP_NUM_THREADS=4

# For Intel CPUs with MKL
export MKL_NUM_THREADS=4
```

### Software Optimization

#### Model Size vs Speed
- **YOLOv8n**: Fastest, lower accuracy
- **YOLOv8s**: Balanced speed and accuracy  
- **YOLOv8m**: Higher accuracy, slower
- **YOLOv8l/x**: Highest accuracy, slowest

#### Input Size Optimization
```python
# Fast processing (lower accuracy)
detection.input_size = 320

# Balanced
detection.input_size = 640  # Default

# High accuracy (slower)
detection.input_size = 1280
```

#### Frame Skipping
```python
# Skip frames for faster processing
video_processor.set_skip_frames(2)  # Process every 3rd frame
```

## 💾 Saving and Exporting

### Save Current Frame
1. Right-click on video display
2. Select "Save Current Frame"
3. Choose location and filename
4. Frame with annotations will be saved

### Export Statistics
1. Go to **File → Export Statistics**
2. Choose output format (JSON, CSV)
3. Statistics include:
   - Total counts per class
   - Processing performance metrics
   - Timing information
   - Configuration used

### Export Activity Log
1. In Activity Log tab, click **"Save Log"**
2. Choose location for log file
3. Complete activity history will be saved

### Export Settings
1. Go to **File → Save Settings**
2. Choose location for settings file
3. Current configuration will be saved as JSON
4. Can be loaded later with **File → Load Settings**

## 🔧 Customization

### Themes

**Change Theme**:
1. Go to **View → Theme**
2. Select from available themes:
   - **Modern**: Clean, professional appearance
   - **Dark**: Dark mode for low-light environments
   - **Light**: High contrast light theme

**Custom Themes**:
- Edit `src/gui/styles/custom_theme.qss`
- Restart application to apply changes

### Counting Line Customization

**Adjust Position**:
- Use slider in Settings tab
- Or set via configuration file
- Or use environment variable: `COUNTING_LINE_X=400`

**Multiple Lines** (Advanced):
- Edit configuration to enable horizontal line
- Set both `line_position_x` and `line_position_y`

### Detection Classes

**Modify Target Classes**:
```python
# In configuration file
counting:
  target_classes:
    - "bottle-glass"
    - "bottle-plastic"
    - "tin can"
    - "paper"          # Add new class
    - "cardboard"      # Add new class
```

**Class-Specific Colors**:
```python
# In src/utils/plotting.py
class_colors = {
    'bottle-glass': (0, 255, 255),    # Cyan
    'bottle-plastic': (255, 165, 0),  # Orange
    'tin can': (128, 128, 128),       # Gray
    'paper': (255, 255, 0),           # Yellow - new
    'cardboard': (165, 42, 42)        # Brown - new
}
```

## 📝 Best Practices

### For Accurate Counting

1. **Camera Positioning**
   - Position camera perpendicular to object flow
   - Ensure good lighting
   - Minimize background clutter
   - Stable mounting (avoid vibrations)

2. **Counting Line Placement**
   - Place where objects naturally cross
   - Avoid areas where objects might reverse direction
   - Consider object size and movement patterns

3. **Model Selection**
   - Use well-trained model specific to your objects
   - Validate model accuracy before deployment
   - Regular retraining with new data

4. **Performance Monitoring**
   - Monitor FPS to ensure real-time performance
   - Check accuracy periodically
   - Review activity logs for issues

### For System Maintenance

1. **Regular Updates**
   - Keep Python and dependencies updated
   - Update model with new training data
   - Monitor for software updates

2. **Log Management**
   - Regularly review activity logs
   - Clean up old log files
   - Monitor disk space usage

3. **Backup**
   - Backup trained models
   - Export and save configurations
   - Document any customizations

## 🎓 Training Your Own Model

### Preparing Training Data

1. **Collect Images**
   - Gather diverse images of recyclable materials
   - Include various lighting conditions
   - Different angles and backgrounds
   - At least 100-500 images per class

2. **Annotation**
   - Use tools like LabelImg or Roboflow
   - Create bounding boxes around objects
   - Use consistent class names
   - Follow YOLO annotation format

3. **Data Organization**
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train/
       ├── val/
       └── test/
   ```

### Training Process

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')

# Train on your dataset
results = model.train(
    data='path/to/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda'  # or 'cpu'
)

# The best model will be saved as 'runs/detect/train/weights/best.pt'
```

### Model Validation

```python
# Validate model performance
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

# Test prediction
results = model.predict('test_image.jpg')
results[0].show()
```

## 🔬 Advanced Configuration

### Performance Tuning

#### For High FPS Requirements
```python
# Reduce input size
detection.input_size = 320

# Skip frames
video.skip_frames = 1  # Process every other frame

# Disable tracking for speed
counting.tracking_enabled = False
```

#### For High Accuracy Requirements
```python
# Increase input size
detection.input_size = 1280

# Lower confidence threshold
detection.confidence_threshold = 0.5

# Enable advanced tracking
counting.tracking_enabled = True
counting.tracking_max_distance = 30.0
```

### Multi-Camera Setup

```python
# Configuration for multiple cameras
cameras = [
    {'id': 0, 'name': 'Camera 1', 'line_x': 300},
    {'id': 1, 'name': 'Camera 2', 'line_x': 400},
]

# Launch multiple instances
for camera in cameras:
    # Each camera would need separate process
    # This is an advanced use case requiring custom implementation
```

### Integration with External Systems

#### Database Integration
```python
# Example: Save results to database
def save_to_database(detection_results):
    import sqlite3
    
    conn = sqlite3.connect('recycling_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO detections (timestamp, class_name, count, confidence)
        VALUES (?, ?, ?, ?)
    ''', (time.time(), class_name, count, confidence))
    
    conn.commit()
    conn.close()
```

#### REST API Integration
```python
# Example: Send results to web service
import requests

def send_to_api(count_data):
    response = requests.post(
        'https://your-api.com/recycling-data',
        json=count_data,
        headers={'Authorization': 'Bearer YOUR_TOKEN'}
    )
    return response.status_code == 200
```

## 📊 Monitoring and Analytics

### Real-time Monitoring

**Performance Metrics**:
- **FPS**: Processing speed (frames per second)
- **Processing Time**: Time per frame in milliseconds
- **Memory Usage**: Current memory consumption
- **GPU Utilization**: GPU memory and compute usage

**Detection Metrics**:
- **Objects per Frame**: Average detections per frame
- **Class Distribution**: Breakdown by object type
- **Confidence Distribution**: Average confidence scores

### Historical Analysis

**Export Data for Analysis**:
```python
# Export statistics
counter.export_statistics('daily_stats.json')

# Load and analyze
import json
import pandas as pd

with open('daily_stats.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['class_counts'].items(), 
                  columns=['Class', 'Count'])
print(df)
```

**Visualization Examples**:
```python
import matplotlib.pyplot as plt

# Plot class distribution
classes = list(data['class_counts'].keys())
counts = list(data['class_counts'].values())

plt.figure(figsize=(10, 6))
plt.bar(classes, counts)
plt.title('Recyclable Material Distribution')
plt.xlabel('Material Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### "No objects detected"
**Possible Causes**:
- Confidence threshold too high
- Model not suitable for your objects
- Poor lighting or image quality

**Solutions**:
- Lower confidence threshold to 0.5 or lower
- Check model training data matches your objects
- Improve lighting conditions
- Verify model file is valid

#### "