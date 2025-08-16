# 🌱 Smart Recycling Detection System 

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-windows%20%7C%20macos%20%7C%20linux-lightgrey.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)
![Academic](https://img.shields.io/badge/Academic-Deep%20Learning%20Project-purple.svg)

An intelligent real-time recycling detection system powered by YOLOv8, designed to identify and count recyclable items with high accuracy and smart tracking capabilities.

**Academic Project**: Developed as part of a Deep Learning course to demonstrate practical applications of computer vision and object detection in environmental sustainability.

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Performance](#performance)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contact](#contact)

## Features

### Core Capabilities
- **Real-time Object Detection**: High-accuracy detection using YOLOv8
- **Smart Counting System**: Advanced tracking prevents double counting
- **Multi-class Recognition**: Identifies glass bottles, plastic bottles, and tin cans
- **Live Video Processing**: Real-time analysis of video feeds or webcam input

### User Interface
- **Clean Design**: Professional, responsive interface built with PyQt5
- **Live Statistics**: Real-time counting and performance metrics
- **Performance Monitoring**: FPS, processing time, and memory usage display
- **Activity Logging**: Timestamped activity logs
- **File Upload**: Drag-and-drop support for easy file loading
- **Theme Options**: Light, dark, and modern theme variants

### Advanced Features
- **Object Tracking**: Sophisticated tracking algorithms prevent double counting
- **Custom Settings**: Adjustable confidence thresholds and counting lines
- **Batch Processing**: Command-line mode for processing multiple videos
- **Data Export**: Save results, logs, and statistics
- **Error Recovery**: Graceful error handling and recovery
- **Cross-platform**: Full support for Windows, macOS, and Linux

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.8+ | Required |
| GPU | CUDA-compatible | Recommended for performance |
| Camera | Webcam or video files | For testing |

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/anothers17/smart-recycling-detection.git
   cd smart-recycling-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Linux/macOS
   source venv/bin/activate
   # On Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your trained model**
   ```bash
   cp path/to/your/best.pt src/resources/models/
   ```

5. **Run the application**
   ```bash
   python src/main.py
   ```

## 📖 Usage

### GUI Mode (Default)

1. **Launch application**
   ```bash
   python src/main.py
   ```

2. **Load your model**
   - Click "Browse Model" to select your `.pt` file
   - Or drag and drop the model file into the window

3. **Select video source**
   - Click "Browse Video" to choose a video file
   - Or use webcam (camera index 0)

4. **Start detection**
   - Click "Start Detection" to begin real-time analysis
   - Use pause, stop, or reset controls as needed

5. **Adjust settings**
   - Use the "Settings" tab for detection confidence
   - Configure counting line position and display options

### Command Line Mode

**Process video without GUI:**
```bash
python src/main.py --no-gui --model models/best.pt --video test_video.mp4
```

**Enable debug logging:**
```bash
python src/main.py --debug
```

**Specify custom model and video:**
```bash
python src/main.py --model path/to/model.pt --video path/to/video.mp4
```

### Batch Processing

Process videos programmatically:

```python
from src.core.video_processor import process_video_file

results = process_video_file(
    video_path="input_video.mp4",
    model_path="models/best.pt", 
    output_path="output_video.mp4"
)
```

## Configuration

### Settings File

Configure settings in `settings.yaml`:

```yaml
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
```

### Environment Variables

Override settings using environment variables:

```bash
export DETECTION_CONFIDENCE=0.8
export DETECTION_DEVICE=cuda
export UI_THEME=dark
```

## Performance

### Benchmarks

| Hardware | Model | FPS | Accuracy |
|----------|-------|-----|----------|
| RTX 3080 | YOLOv8n | 45+ | 95.2% |
| RTX 3070 | YOLOv8s | 35+ | 96.8% |
| CPU i7-10700K | YOLOv8n | 12+ | 95.2% |

### Optimization Tips

- **GPU Usage**: Use CUDA-compatible GPU for better performance
- **Input Size**: Adjust input size (320, 640, 1280) to balance speed and accuracy
- **Model Selection**: Use YOLOv8n for speed, YOLOv8s/m for accuracy

## 🧪 Testing

**Run all tests:**
```bash
python -m pytest tests/
```

**Run tests with coverage:**
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Deployment

### Create Executable

Build standalone executable with PyInstaller:

```bash
pip install pyinstaller
bash scripts/build_executable.sh
```

### Docker Deployment

**Build and run container:**
```bash
# Build image
docker build -t recycling-detection .

# Run with GPU support
docker run -it --gpus all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  recycling-detection
```

## 🤝 Contributing

### Development Setup

1. **Clone and setup**
   ```bash
   git clone https://github.com/anothers17/smart-recycling-detection.git
   cd smart-recycling-detection
   pip install -e ".[dev]"
   ```

2. **Setup pre-commit hooks**
   ```bash
   pre-commit install
   ```

3. **Code formatting**
   ```bash
   black src/
   flake8 src/
   mypy src/
   ```

## 🐛 Troubleshooting

### Common Issues

<details>
<summary><strong>Model Loading Fails</strong></summary>

- Check the model file path and format
- Ensure the `.pt` file is a valid YOLOv8 model
- Verify file permissions

</details>

<details>
<summary><strong>CUDA Out of Memory</strong></summary>

- Switch to CPU mode: `--device cpu`
- Reduce input size in settings
- Close other GPU-intensive applications

</details>

<details>
<summary><strong>GUI Not Displaying</strong></summary>

- Ensure PyQt5 is installed correctly: `pip install PyQt5`
- Check display settings on Linux systems
- Try running with `--no-gui` flag first

</details>

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/anothers17/smart-recycling-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/anothers17/smart-recycling-detection/discussions)
- **Email**: sulhee8@gmail.com

## 📈 Roadmap

### Version 1.1
- Web dashboard
- REST API
- Database integration
- Advanced reporting

### Version 1.2
- Mobile app
- Cloud deployment
- Multi-camera support
- Real-time alerts

### Version 2.0
- 3D detection
- AR visualization
- IoT sensor integration

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **[Ultralytics](https://ultralytics.com/)**: For the excellent YOLOv8 framework
- **[OpenCV](https://opencv.org/)**: For computer vision capabilities
- **[PyQt5](https://riverbankcomputing.com/software/pyqt/)**: For the powerful GUI framework
- **Python Community**: For the amazing ecosystem

## 📞 Contact

- **Facebook**: Sulhee Sama-alee
- **GitHub**: [@anothers17](https://github.com/anothers17)
- **Email**: sulhee8@gmail.com
- **LinkedIn**: [Connect with me](https://www.linkedin.com/in/sulhee/)

---

⭐ **If this project helped you, please give it a star on GitHub!**

*Built with ❤️ for a sustainable future*
