@echo off
echo 🚀 Starting FocusMind with Real Camera Access
echo ============================================

REM Check if virtual environment exists
if not exist "focusmind_env\" (
    echo ❌ Virtual environment not found!
    echo Please run setup_windows.bat first
    pause
    exit /b 1
)

REM Kill any existing processes
taskkill /f /im python.exe 2>nul
taskkill /f /im node.exe 2>nul

echo 🔧 Starting backend server...
start "FocusMind Backend" cmd /k "focusmind_env\Scripts\activate.bat && python main.py"

REM Wait for backend to start
timeout /t 3 /nobreak >nul

echo 📹 Starting face tracking with camera...
start "Face Tracking" cmd /k "focusmind_env\Scripts\activate.bat && python face_focus_tracker.py --source 0"

echo 🎨 Starting frontend...
start "FocusMind Frontend" cmd /k "cd frontend && npm start"

echo.
echo ✅ FocusMind is starting up!
echo 🌐 Frontend will be available at: http://localhost:3003
echo 📹 Camera tracking is now active!
echo.
echo Press any key to close this window...
pause >nul