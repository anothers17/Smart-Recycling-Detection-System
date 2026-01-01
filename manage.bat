@echo off
setlocal

:: 1. Try 'python' command
python --version >nul 2>&1
if %errorlevel% EQU 0 (
    set PYTHON_CMD=python
    goto :FOUND
)

:: 2. Try 'py' command (Python Launcher)
py --version >nul 2>&1
if %errorlevel% EQU 0 (
    set PYTHON_CMD=py
    goto :FOUND
)

:: 3. If neither found
echo ==============================================
echo ERROR: Python not found!
echo ==============================================
echo tested: 'python' and 'py' commands.
echo.
echo Please ensure Python is installed and added to your PATH.
echo You can download it from python.org
echo.
pause
exit /b 1

:FOUND
:: ==============================================
:: Main Menu
:: ==============================================
:MENU
CLS
ECHO ==============================================
ECHO    Smart Recycling Detection - Control
ECHO    (Using: %PYTHON_CMD%)
ECHO ==============================================
ECHO 1. Setup (Install Dependencies)
ECHO 2. Run Simulator Mode (No Hardware)
ECHO 3. Run Real Mode (App Only)
ECHO 4. Start Docker Services
ECHO 5. Stop Docker Services
ECHO 6. Exit
ECHO.
SET /P M="Select Option (1-6): "

IF %M%==1 GOTO SETUP
IF %M%==2 GOTO SIMULATOR
IF %M%==3 GOTO REAL
IF %M%==4 GOTO DOCKER_UP
IF %M%==5 GOTO DOCKER_DOWN
IF %M%==6 GOTO EOF

:SETUP
ECHO Installing dependencies...
%PYTHON_CMD% -m pip install -r requirements.txt
PAUSE
GOTO MENU

:SIMULATOR
ECHO Setting up simulator environment...
if not exist .env copy .env.example .env
ECHO Starting Application (SIMULATOR MODE)...
set HAS_HARDWARE=False
%PYTHON_CMD% main.py
PAUSE
GOTO MENU

:REAL
ECHO Starting Application (REAL MODE)...
set HAS_HARDWARE=True
%PYTHON_CMD% main.py
PAUSE
GOTO MENU

:DOCKER_UP
ECHO Starting Docker services...
docker-compose up -d
ECHO Services started!
PAUSE
GOTO MENU

:DOCKER_DOWN
ECHO Stopping Docker services...
docker-compose down
ECHO Services stopped!
PAUSE
GOTO MENU

:EOF
EXIT
