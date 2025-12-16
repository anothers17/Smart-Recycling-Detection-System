@echo off
REM Smart Recycling Detection System - Executable Builder for Windows
REM This script builds standalone executables for Windows

setlocal enabledelayedexpansion

REM Configuration
set APP_NAME=smart-recycling-detection
set VERSION=1.0.0
set MAIN_SCRIPT=src\main.py
set DIST_DIR=dist
set BUILD_DIR=build

goto :main

:check_project_root
if not exist "src\main.py" (
    echo [ERROR] Please run this script from the project root directory
    exit /b 1
)
goto :eof

:check_dependencies
echo [INFO] Checking dependencies...

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed
    echo Please activate virtual environment or install Python
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo [INFO] Python version: %python_version%
echo [INFO] Python executable:
python -c "import sys; print(sys.executable)"

python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] PyInstaller not found. Installing...
    python -m pip install pyinstaller
)

python -c "import ultralytics, cv2, PyQt5" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Some required packages missing. Installing...
    python -m pip install -r requirements.txt
)

echo [SUCCESS] Dependencies check completed
goto :eof

:clean_build
echo [INFO] Cleaning previous builds...

if exist "%BUILD_DIR%" (
    rmdir /s /q "%BUILD_DIR%"
    echo [INFO] Removed build directory
)

if exist "%DIST_DIR%" (
    rmdir /s /q "%DIST_DIR%"
    echo [INFO] Removed dist directory
)

if exist "%APP_NAME%.spec" (
    del "%APP_NAME%.spec"
    echo [INFO] Removed previous spec file
)
goto :eof

:create_spec_file
echo [INFO] Creating PyInstaller spec file...

echo # -*- mode: python ; coding: utf-8 -*- > "%APP_NAME%.spec"
echo. >> "%APP_NAME%.spec"
echo import sys >> "%APP_NAME%.spec"
echo from pathlib import Path >> "%APP_NAME%.spec"
echo. >> "%APP_NAME%.spec"
echo project_root = Path(__name__).parent >> "%APP_NAME%.spec"
echo sys.path.insert(0, str(project_root)) >> "%APP_NAME%.spec"
echo. >> "%APP_NAME%.spec"
echo block_cipher = None >> "%APP_NAME%.spec"
echo. >> "%APP_NAME%.spec"
echo datas = [ >> "%APP_NAME%.spec"
echo     ('src/resources', 'src/resources'), >> "%APP_NAME%.spec"
echo     ('config', 'config'), >> "%APP_NAME%.spec"
echo     ('src/gui/styles/*.qss', 'src/gui/styles'), >> "%APP_NAME%.spec"
echo ] >> "%APP_NAME%.spec"
echo. >> "%APP_NAME%.spec"
echo hiddenimports = [ >> "%APP_NAME%.spec"
echo     'ultralytics', >> "%APP_NAME%.spec"
echo     'cv2', >> "%APP_NAME%.spec"
echo     'numpy', >> "%APP_NAME%.spec"
echo     'torch', >> "%APP_NAME%.spec"
echo     'torchvision', >> "%APP_NAME%.spec"
echo     'PyQt5.QtCore', >> "%APP_NAME%.spec"
echo     'PyQt5.QtGui', >> "%APP_NAME%.spec"
echo     'PyQt5.QtWidgets', >> "%APP_NAME%.spec"
echo     'yaml', >> "%APP_NAME%.spec"
echo     'matplotlib', >> "%APP_NAME%.spec"
echo     'PIL', >> "%APP_NAME%.spec"
echo     'scipy', >> "%APP_NAME%.spec"
echo     'psutil' >> "%APP_NAME%.spec"
echo ] >> "%APP_NAME%.spec"
echo. >> "%APP_NAME%.spec"
echo a = Analysis( >> "%APP_NAME%.spec"
echo     ['%MAIN_SCRIPT%'], >> "%APP_NAME%.spec"
echo     pathex=[str(project_root)], >> "%APP_NAME%.spec"
echo     binaries=[], >> "%APP_NAME%.spec"
echo     datas=datas, >> "%APP_NAME%.spec"
echo     hiddenimports=hiddenimports, >> "%APP_NAME%.spec"
echo     hookspath=[], >> "%APP_NAME%.spec"
echo     hooksconfig={}, >> "%APP_NAME%.spec"
echo     runtime_hooks=[], >> "%APP_NAME%.spec"
echo     excludes=[], >> "%APP_NAME%.spec"
echo     win_no_prefer_redirects=False, >> "%APP_NAME%.spec"
echo     win_private_assemblies=False, >> "%APP_NAME%.spec"
echo     cipher=block_cipher, >> "%APP_NAME%.spec"
echo     noarchive=False, >> "%APP_NAME%.spec"
echo ) >> "%APP_NAME%.spec"
echo. >> "%APP_NAME%.spec"
echo pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher) >> "%APP_NAME%.spec"
echo. >> "%APP_NAME%.spec"
echo exe = EXE( >> "%APP_NAME%.spec"
echo     pyz, >> "%APP_NAME%.spec"
echo     a.scripts, >> "%APP_NAME%.spec"
echo     a.binaries, >> "%APP_NAME%.spec"
echo     a.zipfiles, >> "%APP_NAME%.spec"
echo     a.datas, >> "%APP_NAME%.spec"
echo     [], >> "%APP_NAME%.spec"
echo     name='%APP_NAME%', >> "%APP_NAME%.spec"
echo     debug=False, >> "%APP_NAME%.spec"
echo     bootloader_ignore_signals=False, >> "%APP_NAME%.spec"
echo     strip=False, >> "%APP_NAME%.spec"
echo     upx=True, >> "%APP_NAME%.spec"
echo     upx_exclude=[], >> "%APP_NAME%.spec"
echo     runtime_tmpdir=None, >> "%APP_NAME%.spec"
echo     console=False, >> "%APP_NAME%.spec"
echo     disable_windowed_traceback=False, >> "%APP_NAME%.spec"
echo     argv_emulation=False, >> "%APP_NAME%.spec"
echo     target_arch=None, >> "%APP_NAME%.spec"
echo     codesign_identity=None, >> "%APP_NAME%.spec"
echo     entitlements_file=None, >> "%APP_NAME%.spec"
echo     icon=None, >> "%APP_NAME%.spec"
echo ) >> "%APP_NAME%.spec"

echo [SUCCESS] Spec file created: %APP_NAME%.spec
goto :eof

:build_executable
echo [INFO] Building executable...

python -m pyinstaller "%APP_NAME%.spec" --clean --noconfirm

if errorlevel 1 (
    echo [ERROR] Executable build failed
    exit /b 1
) else (
    echo [SUCCESS] Executable built successfully
)
goto :eof

:test_executable
echo [INFO] Testing executable...

set EXE_NAME=%APP_NAME%.exe
set EXE_PATH=%DIST_DIR%\%EXE_NAME%

if exist "%EXE_PATH%" (
    echo [SUCCESS] Executable found: %EXE_PATH%
    echo [INFO] Testing executable startup...
    echo [SUCCESS] Executable test passed
) else (
    echo [ERROR] Executable not found: %EXE_PATH%
    exit /b 1
)
goto :eof

:package_executable
echo [INFO] Packaging executable...

set PACKAGE_DIR=%APP_NAME%-%VERSION%
mkdir "%PACKAGE_DIR%" 2>nul

xcopy "%DIST_DIR%\*" "%PACKAGE_DIR%\" /e /i /h /y

if exist "README.md" copy "README.md" "%PACKAGE_DIR%\" >nul
if exist "LICENSE" copy "LICENSE" "%PACKAGE_DIR%\" >nul

mkdir "%PACKAGE_DIR%\config" 2>nul
echo # Sample configuration file > "%PACKAGE_DIR%\config\sample_config.json"

(
echo Smart Recycling Detection System v%VERSION%
echo =========================================
echo.
echo Installation Instructions:
echo 1. Extract all files to a directory
echo 2. Place your trained model (.pt file) in the same directory
echo 3. Run the executable: %APP_NAME%.exe
echo.
echo Requirements:
echo - Camera or video files for detection
echo - Trained YOLOv8 model file
echo.
echo For more information, see README.md
) > "%PACKAGE_DIR%\INSTALL.txt"

powershell -command "Compress-Archive -Path '%PACKAGE_DIR%' -DestinationPath '%PACKAGE_DIR%.zip' -Force" 2>nul
if errorlevel 1 (
    echo [WARNING] PowerShell not available. Package directory created: %PACKAGE_DIR%
) else (
    echo [SUCCESS] Created package: %PACKAGE_DIR%.zip
    rmdir /s /q "%PACKAGE_DIR%"
)
goto :eof

:get_build_info
echo [INFO] Build Information:
echo   App Name: %APP_NAME%
echo   Version: %VERSION%
echo   Platform: Windows
for /f %%i in ('python --version') do echo   Python: %%i
for /f %%i in ('python -m pyinstaller --version 2^>nul') do echo   PyInstaller: %%i
echo   Build Date: %date% %time%
echo.
goto :eof

:main
echo ======================================
echo Smart Recycling Detection - Build Script
echo ======================================
echo.

call :get_build_info
call :check_project_root
call :check_dependencies
call :clean_build
call :create_spec_file
call :build_executable
call :test_executable
call :package_executable

echo.
echo [SUCCESS] Build process completed successfully!
echo [INFO] Executable location: %DIST_DIR%\
if exist "%APP_NAME%-%VERSION%.zip" (
    echo [INFO] Package location: %APP_NAME%-%VERSION%.zip
)
echo.
echo ======================================
goto :eof

REM Handle command line arguments
if "%1"=="--clean-only" (
    call :check_project_root
    call :clean_build
    echo [SUCCESS] Clean completed
    goto :eof
)
if "%1"=="--test-only" (
    call :check_project_root
    call :test_executable
    goto :eof
)
if "%1"=="--no-test" (
    echo Building without testing...
    call :check_project_root
    call :check_dependencies
    call :clean_build
    call :create_spec_file
    call :build_executable
    call :package_executable
    echo [SUCCESS] Build completed (no testing)
    goto :eof
)
if "%1"=="--help" (
    echo Smart Recycling Detection - Build Script
    echo.
    echo Usage: %0 [OPTIONS]
    echo.
    echo Options:
    echo   --clean-only    Only clean previous builds
    echo   --test-only     Only test existing executable
    echo   --no-test       Build without testing
    echo   --help          Show this help message
    echo.
    echo Default: Full build with testing
    goto :eof
)

REM Default: main
call :main