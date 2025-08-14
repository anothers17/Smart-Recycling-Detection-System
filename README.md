🌱 Smart Recycling Detection System
Key Features:

- Real-time Object Detection: Detects objects using YOLOv8 for high accuracy.

- Smart Counting System: Tracks objects crossing a line, preventing double counting.

- Multi-class Recognition: Identifies glass bottles, plastic bottles, and tin cans.

- Live Video Processing: Analyzes video or webcam feed in real time.

User Interface:

- Clean Design: Professional, responsive interface built with PyQt5.

- Live Stats: Real-time counting and statistics display.

- Performance Monitoring: Displays FPS, processing time, and memory usage.

- Activity Logging: Keeps a log of activities with timestamps.

- File Upload: Drag and drop support for easy file loading.

- Theme Options: Switch between light, dark, and modern themes.

Advanced Features:

- Object Tracking: Prevents double counting with accurate tracking.

- Custom Detection Settings: Adjust confidence thresholds and counting lines.

- Batch Processing: Process multiple videos using command-line mode.

- Export Data: Save results, logs, and statistics.

- Error Recovery: Handles errors gracefully.

- Cross-platform Support: Works on Windows, macOS, and Linux.

🚀 Quick Start
Prerequisites:

- Python 3.8 or higher

- CUDA-compatible GPU (recommended, but CPU also supported)

- Webcam or video files for testing

Installation:
Clone the repository:

git clone https://github.com/anothers17/smart-recycling-detection.git
cd smart-recycling-detection


Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Add your trained YOLOv8 model:

cp path/to/your/best.pt src/resources/models/


Run the application:

python src/main.py

📖 Usage
GUI Mode (Default):

1.Launch the application:

python src/main.py


2.Load your model:

- Click "Browse Model" to select your .pt file.

- Or, drag and drop the model file into the window.

3.Select video source:

- Click "Browse Video" to choose a video file.

- Or, use the webcam (camera index 0).

4.Start detection:

- Click "Start Detection" to begin real-time analysis.

- Pause, stop, or reset as needed.

5.Adjust settings:

- Use the "Settings" tab to change detection confidence, counting line position, and display options.

Command Line Mode:

To process video without the GUI:

python src/main.py --no-gui --model models/best.pt --video test_video.mp4


Enable debug logging:

python src/main.py --debug


Use specific model and video:

python src/main.py --model path/to/model.pt --video path/to/video.mp4

Batch Processing:

Process videos programmatically:

from src.core.video_processor import process_video_file

results = process_video_file(
    video_path="input_video.mp4",
    model_path="models/best.pt", 
    output_path="output_video.mp4"
)

Configuration
Settings File:

You can configure key settings in the settings.yaml file.

Example:

detection:
  confidence_threshold: 0.7
  input_size: 640
  device: "auto"  # Options: "auto", "cpu", "cuda", "mps"

counting:
  line_position_x: 300
  target_classes: ["bottle-glass", "bottle-plastic", "tin can"]
  tracking_enabled: true

ui:
  theme: "modern"  # Options: "modern", "dark", "light"
  window_width: 1240
  window_height: 730

Environment Variables:

You can override settings using environment variables:

export DETECTION_CONFIDENCE=0.8
export DETECTION_DEVICE=cuda
export UI_THEME=dark

Performance
Benchmarks:

Tested on various hardware:

RTX 3080: YOLOv8n, 45+ FPS, 95.2% accuracy

RTX 3070: YOLOv8s, 35+ FPS, 96.8% accuracy

CPU i7-10700K: YOLOv8n, 12+ FPS, 95.2% accuracy

Optimization Tips:

Use GPU for better performance.

Adjust input size (320, 640, 1280) for a balance between speed and accuracy.

Use YOLOv8n for faster performance, YOLOv8s/m for better accuracy.

🧪 Testing

Run tests using:

python -m pytest tests/


Run tests with coverage:

python -m pytest tests/ --cov=src --cov-report=html

Deployment
Create Executable:

Use PyInstaller to create a standalone executable:

pip install pyinstaller
python scripts/build_executable.sh

Docker Deployment:

Build and run the Docker container:

docker build -t recycling-detection .
docker run -it --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY recycling-detection

🤝 Contributing

Follow the development setup instructions to contribute:

git clone https://github.com/anothers17/smart-recycling-detection.git
cd smart-recycling-detection
pip install -e ".[dev]"
pre-commit install
black src/
flake8 src/
mypy src/

🐛 Troubleshooting
Common Issues:

Model Loading Fails:
Check the model file path and format.

CUDA Out of Memory:
Use CPU or reduce input size.

GUI Not Displaying:
Ensure PyQt5 is installed correctly.

Getting Help:

Issues: GitHub Issues

Discussions: GitHub Discussions

Email: sulhee8@gmail.com

📈 Roadmap

Version 1.1: Web dashboard, REST API, Database integration, Advanced reporting

Version 1.2: Mobile app, Cloud deployment, Multi-camera support, Real-time alerts

Version 2.0: 3D detection, AR visualization, IoT sensor integration

License

This project is licensed under the MIT License. See the LICENSE file for details.

Ultralytics: For the excellent YOLOv8 framework
OpenCV: For computer vision capabilities
PyQt5: For the powerful GUI framework
Python Community: For the amazing ecosystem

📞 Contact
FB: Sulhee Sama-alee

GitHub: @anothers17
Email: sulhee8@gmail.com
LinkedIn: -----------


⭐ If this project helped you, please give it a star on GitHub!
Built with ❤️ for a sustainable future