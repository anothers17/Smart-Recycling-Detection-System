#!/bin/bash

# Smart Recycling Detection System - Executable Builder
# This script builds standalone executables for different platforms

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="smart-recycling-detection"
VERSION="1.0.0"
MAIN_SCRIPT="src/main.py"
DIST_DIR="dist"
BUILD_DIR="build"

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the project root
check_project_root() {
    if [[ ! -f "src/main.py" ]]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check Python version
    python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_status "Python version: $python_version"
    
    # Check if PyInstaller is installed
    if ! python -c "import PyInstaller" &> /dev/null; then
        print_warning "PyInstaller not found. Installing..."
        pip install pyinstaller
    fi
    
    # Check if required packages are installed
    if ! python -c "import ultralytics, cv2, PyQt5" &> /dev/null; then
        print_warning "Some required packages missing. Installing..."
        pip install -r requirements.txt
    fi
    
    print_success "Dependencies check completed"
}

# Clean previous builds
clean_build() {
    print_status "Cleaning previous builds..."
    
    if [[ -d "$BUILD_DIR" ]]; then
        rm -rf "$BUILD_DIR"
        print_status "Removed build directory"
    fi
    
    if [[ -d "$DIST_DIR" ]]; then
        rm -rf "$DIST_DIR"
        print_status "Removed dist directory"
    fi
    
    # Remove spec files
    if [[ -f "$APP_NAME.spec" ]]; then
        rm "$APP_NAME.spec"
        print_status "Removed previous spec file"
    fi
}

# Create PyInstaller spec file
create_spec_file() {
    print_status "Creating PyInstaller spec file..."
    
    cat > "$APP_NAME.spec" << EOF
# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__name__).parent
sys.path.insert(0, str(project_root))

block_cipher = None

# Data files to include
datas = [
    ('src/resources', 'src/resources'),
    ('config', 'config'),
    ('src/gui/styles/*.qss', 'src/gui/styles'),
]

# Hidden imports
hiddenimports = [
    'ultralytics',
    'cv2',
    'numpy',
    'torch',
    'torchvision',
    'PyQt5.QtCore',
    'PyQt5.QtGui', 
    'PyQt5.QtWidgets',
    'yaml',
    'matplotlib',
    'PIL',
    'scipy',
    'psutil'
]

a = Analysis(
    ['$MAIN_SCRIPT'],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='$APP_NAME',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path if available
)
EOF
    
    print_success "Spec file created: $APP_NAME.spec"
}

# Build executable
build_executable() {
    print_status "Building executable..."
    
    # Build with PyInstaller
    pyinstaller "$APP_NAME.spec" --clean --noconfirm
    
    if [[ $? -eq 0 ]]; then
        print_success "Executable built successfully"
    else
        print_error "Executable build failed"
        exit 1
    fi
}

# Test executable

#windowns
test_executable() {
    print_status "Testing executable..."
    
    # Determine executable name based on platform
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        EXE_NAME="$APP_NAME.exe"
    else
        EXE_NAME="$APP_NAME"
    fi
    
    EXE_PATH="$DIST_DIR/$EXE_NAME"
    
    if [[ -f "$EXE_PATH" ]]; then
        print_success "Executable found: $EXE_PATH"
        
        # Test basic functionality
        print_status "Testing executable startup..."
        
        # Run with version flag (if implemented)
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            start /wait "" "$EXE_PATH" --version &> /dev/null
            if [[ $? -eq 0 ]]; then
                print_success "Executable test passed"
            else
                print_warning "Executable test failed or timed out"
            fi
        else
            if (Start-Process "$EXE_PATH" -ArgumentList '--version' -Wait -PassThru | Out-Null); then
                print_success "Executable test passed"
            else
                print_warning "Executable test failed or timed out"
            fi
        fi
    else
        print_error "Executable not found: $EXE_PATH"
        exit 1
    fi
}
# test_executable() {
#     print_status "Testing executable..."
    
#     # Determine executable name based on platform
#     if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
#         EXE_NAME="$APP_NAME.exe"
#     else
#         EXE_NAME="$APP_NAME"
#     fi
    
#     EXE_PATH="$DIST_DIR/$EXE_NAME"
    
#     if [[ -f "$EXE_PATH" ]]; then
#         print_success "Executable found: $EXE_PATH"
        
#         # Test basic functionality
#         print_status "Testing executable startup..."
        
#         # Run with version flag (if implemented)
#         if timeout 10s "$EXE_PATH" --version &> /dev/null; then
#             print_success "Executable test passed"
#         else
#             print_warning "Executable test failed or timed out"
#         fi
#     else
#         print_error "Executable not found: $EXE_PATH"
#         exit 1
#     fi
# }

# Package executable
package_executable() {
    print_status "Packaging executable..."
    
    # Create package directory
    PACKAGE_DIR="$APP_NAME-$VERSION"
    mkdir -p "$PACKAGE_DIR"
    
    # Copy executable
    cp -r "$DIST_DIR"/* "$PACKAGE_DIR/"
    
    # Copy additional files
    cp README.md "$PACKAGE_DIR/" 2>/dev/null || true
    cp LICENSE "$PACKAGE_DIR/" 2>/dev/null || true
    
    # Create sample config
    mkdir -p "$PACKAGE_DIR/config"
    echo "# Sample configuration file" > "$PACKAGE_DIR/config/sample_config.json"
    
    # Create documentation
    cat > "$PACKAGE_DIR/INSTALL.txt" << EOF
Smart Recycling Detection System v$VERSION
=========================================

Installation Instructions:
1. Extract all files to a directory
2. Place your trained model (.pt file) in the same directory
3. Run the executable: ./$APP_NAME (Linux/Mac) or $APP_NAME.exe (Windows)

Requirements:
- Camera or video files for detection
- Trained YOLOv8 model file

For more information, see README.md or visit:
https://github.com/yourusername/smart-recycling-detection
EOF
    
    # Create archive
    if command -v zip &> /dev/null; then
        zip -r "$PACKAGE_DIR.zip" "$PACKAGE_DIR"
        print_success "Created package: $PACKAGE_DIR.zip"
    elif command -v tar &> /dev/null; then
        tar -czf "$PACKAGE_DIR.tar.gz" "$PACKAGE_DIR"
        print_success "Created package: $PACKAGE_DIR.tar.gz"
    else
        print_warning "No archiving tool found. Package directory created: $PACKAGE_DIR"
    fi
    
    # Cleanup package directory
    rm -rf "$PACKAGE_DIR"
}

# Get build information
get_build_info() {
    print_status "Build Information:"
    echo "  App Name: $APP_NAME"
    echo "  Version: $VERSION"
    echo "  Platform: $OSTYPE"
    echo "  Python: $(python --version)"
    echo "  PyInstaller: $(pyinstaller --version)"
    echo "  Build Date: $(date)"
    echo ""
}

# Main execution
main() {
    echo "======================================"
    echo "Smart Recycling Detection - Build Script"
    echo "======================================"
    echo ""
    
    get_build_info
    
    check_project_root
    check_dependencies
    clean_build
    create_spec_file
    build_executable
    test_executable
    package_executable
    
    print_success "Build process completed successfully!"
    print_status "Executable location: $DIST_DIR/"
    
    if [[ -f "$APP_NAME-$VERSION.zip" ]]; then
        print_status "Package location: $APP_NAME-$VERSION.zip"
    elif [[ -f "$APP_NAME-$VERSION.tar.gz" ]]; then
        print_status "Package location: $APP_NAME-$VERSION.tar.gz"
    fi
    
    echo ""
    echo "======================================"
}

# Handle command line arguments
case "${1:-}" in
    --clean-only)
        check_project_root
        clean_build
        print_success "Clean completed"
        ;;
    --test-only)
        check_project_root
        test_executable
        ;;
    --no-test)
        echo "Building without testing..."
        check_project_root
        check_dependencies
        clean_build
        create_spec_file
        build_executable
        package_executable
        print_success "Build completed (no testing)"
        ;;
    --help|-h)
        echo "Smart Recycling Detection - Build Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --clean-only    Only clean previous builds"
        echo "  --test-only     Only test existing executable"
        echo "  --no-test       Build without testing"
        echo "  --help, -h      Show this help message"
        echo ""
        echo "Default: Full build with testing"
        ;;
    *)
        main
        ;;
esac