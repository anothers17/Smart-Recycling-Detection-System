# ⚙️ Installation Guide

Welcome to the **Smart Recycling Detection System** installation guide. This document provides step-by-step instructions to get your system up and running, whether you are a developer, a student, or an end-user.

> [!TIP]
> **New to CLI?** Use our **Quick Management Tools**! We've provided scripts to automate almost everything for you.

---

## 📋 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8+ | 3.10+ |
| **RAM** | 4 GB | 16 GB+ |
| **GPU** | N/A | NVIDIA (CUDA Support) |
| **Storage** | 2 GB | 4 GB+ |
| **MCU** | Optional | ESP32 (for Real Hardware) |

---

## � Quick Installation (Recommended)

Choose the method that matches your Operating System for the fastest setup.

### 🪟 Windows (The Easiest Way)
1. **Clone** this repository:
   ```powershell
   git clone https://github.com/anothers17/Smart-Recycling-Detection-System.git
   cd Smart-Recycling-Detection-System
   ```
2. **Run Automation**: Double-click `manage.bat` or run it in your CMD:
   - Select **Option 1 (Setup)**: This will automatically create a virtual environment and install all dependencies.
   - Select **Option 2 (Simulator)**: To start testing the application immediately!

### 🍎/� Linux & macOS
1. **Clone** this repository:
   ```bash
   git clone https://github.com/anothers17/Smart-Recycling-Detection-System.git
   cd Smart-Recycling-Detection-System
   ```
2. **Use Makefile**:
   ```bash
   make setup       # Installs everything
   make simulator   # Runs the app in simulator mode
   ```

---

## 🛠️ Manual Installation (Advanced)

If you prefer to manage your environment manually, follow these steps:

### 1. Environment Setup
```bash
# Create and activate environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Unix/macOS:
source venv/bin/activate
```

### 2. Dependency Installation
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. GPU Acceleration (NVIDIA Users)
If you have an NVIDIA GPU, ensure you have the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed. The system will automatically detect and use your GPU for much faster detection.

---

## 🖥️ Platform-Specific Tips

> [!IMPORTANT]
> **Linux Users**: You may need to install additional system libraries for OpenCV and PyQt5:
> `sudo apt update && sudo apt install -y libgl1-mesa-glx libglib2.0-0`

### Windows 10/11
- Use **PowerShell** or **Command Prompt** as Administrator for the initial setup.
- Ensure "Add Python to PATH" was checked during Python installation.

### macOS (Apple Silicon/Intel)
- If you encounter issues with `cv2`, try: `pip install opencv-python-headless`
- For Apple Silicon (M1/M2), ensure you are using a native ARM64 Python version for best performance.

---

## 🔍 Verification

After installation, verify that everything is working correctly:

```bash
# Verify imports and hardware connection
python src/main.py --check-env
```

If you see **"Environment Verification Successful"**, you are ready to go!

---

## 🆘 Troubleshooting

- **"No module named 'cv2'"**: Run `pip install opencv-python`.
- **"CUDA out of memory"**: Reduce `input_size` in `.env` or use CPU mode (`HAS_HARDWARE=False`).
- **"Qt platform plugin error"**: Ensure all dependencies in `requirements.txt` are installed. On Linux, check for missing `X11` libraries.

---

**Next Steps**: Head over to the [Usage Guide](usage.md) to learn how to load your first model and start detecting! ♻️🚀