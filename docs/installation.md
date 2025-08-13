# Installation Guide

This guide provides detailed installation instructions for the Smart Recycling Detection System across different platforms and deployment scenarios.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 2 GB free space
- **CPU**: Multi-core processor recommended
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)

### Recommended Requirements
- **Python**: 3.10 or higher
- **RAM**: 16 GB or more
- **GPU**: NVIDIA GPU with CUDA support (GTX 1060 or better)
- **Storage**: 4 GB free space (for models and sample data)

### GPU Support (Optional but Recommended)
- **NVIDIA GPU**: GTX 1060 or better
- **CUDA**: Version 11.0 or higher
- **cuDNN**: Compatible version with your CUDA installation
- **VRAM**: 4 GB minimum, 8 GB recommended

## 🔧 Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/smart-recycling-detection.git
cd smart-recycling-detection
```

#### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

#### Step 4: Verify Installation
```bash
# Test basic imports
python -c "import ultralytics, cv2, PyQt5; print('All dependencies installed successfully!')"

# Check environment
python src/main.py --check-env
```

### Method 2: Development Installation

For developers who want to modify the code:

```bash
# Clone repository
git clone https://github.com/yourusername/smart-recycling-detection.git
cd smart-recycling-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Method 3: Docker Installation

For containerized deployment:

```bash
# Build Docker image
docker build -t smart-recycling-detection .

# Run with GUI support (Linux)
docker run -it --rm \
    --gpus all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -v $(pwd)/models:/app/src/resources/models \
    smart-recycling-detection

# Run without GUI (batch processing)
docker run -it --rm \
    --gpus all \
    -v $(pwd)/models:/app/src/resources/models \
    -v $(pwd)/videos:/app/videos \
    smart-recycling-detection \
    python src/main.py --no-gui --model /app/src/resources/models/best.pt --video /app/videos/test.mp4
```

### Method 4: Pre-built Executable

Download pre-built executables from the [Releases](https://github.com/yourusername/smart-recycling-detection/releases) page:

1. Download the appropriate version for your platform
2. Extract the archive
3. Run the executable directly

## 🖥️ Platform-Specific Instructions

### Windows 10/11

#### Prerequisites
```powershell
# Install Python from python.org or Microsoft Store
# Ensure Python is added to PATH during installation

# Verify installation
python --version
pip --version
```

#### CUDA Setup (Optional)
```powershell
# Download and install CUDA Toolkit from NVIDIA
# Download and install cuDNN
# Verify CUDA installation
nvcc --version
```

#### Install Application
```powershell
# Clone repository
git clone https://github.com/yourusername/smart-recycling-detection.git
cd smart-recycling-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python src/main.py
```

### macOS

#### Prerequisites
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Install OpenCV dependencies
brew install opencv
```

#### Install Application
```bash
# Clone repository
git clone https://github.com/yourusername/smart-recycling-detection.git
cd smart-recycling-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python src/main.py
```

### Ubuntu/Debian Linux

#### Prerequisites
```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Install OpenCV dependencies
sudo apt install libopencv-dev python3-opencv

# Install Qt5 dependencies
sudo apt install python3-pyqt5 python3-pyqt5.qtmultimedia

# For GPU support (optional)
# Follow NVIDIA CUDA installation guide for your Ubuntu version
```

#### Install Application
```bash
# Clone repository
git clone https://github.com/yourusername/smart-recycling-detection.git
cd smart-recycling-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python src/main.py
```

### CentOS/RHEL/Fedora

#### Prerequisites
```bash
# Install Python and development tools
sudo dnf install python3 python3-pip python3-devel gcc gcc-c++

# Install OpenCV dependencies
sudo dnf install opencv opencv-devel python3-opencv

# Install Qt5 dependencies
sudo dnf install python3-qt5 python3-qt5-devel
```

## 🚀 Post-Installation Setup

### 1. Add Your Trained Model

```bash
# Copy your trained YOLOv8 model to the models directory
cp path/to/your/best.pt src/resources/models/

# Or create a symbolic link
ln -s /absolute/path/to/your/best.pt src/resources/models/best.pt
```

### 2. Add Sample Videos (Optional)

```bash
# Copy test videos to sample directory
cp path/to/your/videos/*.mp4 src/resources/sample_videos/
```

### 3. Configure Application

```bash
# Copy default configuration
cp config/settings.py config/user_settings.py

# Edit configuration as needed
nano config/user_settings.py
```

### 4. Test Installation

```bash
# Run basic functionality test
python src/main.py --check-env

# Run with sample data
python src/main.py --model src/resources/models/best.pt
```

## 🔧 Troubleshooting

### Common Installation Issues

#### Issue: "No module named 'cv2'"
```bash
# Solution: Install OpenCV
pip install opencv-python

# On Linux, also install system packages:
sudo apt install libopencv-dev python3-opencv
```

#### Issue: "No module named 'PyQt5'"
```bash
# Solution: Install PyQt5
pip install PyQt5

# On Linux:
sudo apt install python3-pyqt5
```

#### Issue: "CUDA out of memory"
```bash
# Solution: Use CPU mode or reduce model size
export DETECTION_DEVICE=cpu
export DETECTION_INPUT_SIZE=320
```

#### Issue: "Permission denied" on Linux/macOS
```bash
# Solution: Make scripts executable
chmod +x scripts/*.sh

# Run installation script
./scripts/install_dependencies.sh
```

#### Issue: Qt platform plugin error
```bash
# Solution: Install Qt platform plugins
# On Ubuntu:
sudo apt install python3-pyqt5.qtx11extras

# Set environment variable:
export QT_QPA_PLATFORM=xcb
```

### Performance Optimization

#### GPU Acceleration
```bash
# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Force GPU usage
export DETECTION_DEVICE=cuda
```

#### Memory Optimization
```bash
# Reduce memory usage
export DETECTION_INPUT_SIZE=320
export VIDEO_BUFFER_SIZE=1
```

#### CPU Optimization
```bash
# Use multiple CPU cores
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## 🔍 Verification

### Quick Test
```bash
# Test basic functionality
python src/main.py --version

# Test import capabilities
python -c "
import sys
sys.path.append('.')
from src.core.detector import RecyclingDetector
from src.core.counter import RecyclingCounter
from src.gui.main_window import MainWindow
print('All components imported successfully!')
"
```

### Comprehensive Test
```bash
# Run test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📦 Creating Standalone Executable

### Build Executable
```bash
# Make build script executable
chmod +x scripts/build_executable.sh

# Build executable
./scripts/build_executable.sh

# The executable will be created in dist/ directory
```

### Distribute Executable
```bash
# The build script creates a packaged version
# Share the generated .zip or .tar.gz file

# Recipients can simply:
# 1. Extract the archive
# 2. Run the executable
# 3. No Python installation required!
```

## 🆘 Getting Help

### Support Channels
- **GitHub Issues**: [Report bugs and issues](https://github.com/yourusername/smart-recycling-detection/issues)
- **Discussions**: [Ask questions and get help](https://github.com/yourusername/smart-recycling-detection/discussions)
- **Email**: your.email@example.com

### Diagnostic Information
When reporting issues, please include:

```bash
# Generate diagnostic information
python src/main.py --check-env > diagnostic_info.txt

# System information
python -c "
import sys, platform, cv2, torch
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'OpenCV: {cv2.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

## 🔄 Updating

### Update from Git
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run any migration scripts
python scripts/migrate_config.py  # If available
```

### Update from PyPI (When Available)
```bash
# Update package
pip install --upgrade smart-recycling-detection

# Or for development version
pip install --upgrade --pre smart-recycling-detection
```

---

**Next Steps**: After installation, see the [Usage Guide](usage.md) for detailed instructions on using the application.