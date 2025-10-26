#!/bin/bash

# FocusMind Face Tracking Startup Script
# This script helps you easily start the face tracking system

echo "🎯 FocusMind Face Tracking Startup"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import cv2, mediapipe, numpy, matplotlib, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing required packages. Installing..."
    pip install -r requirements.txt
fi

# Check if camera is available
echo "📹 Checking camera access..."
python -c "import cv2; cap = cv2.VideoCapture(0); print('✅ Camera accessible') if cap.isOpened() else print('❌ Camera not accessible'); cap.release()" 2>/dev/null

# Check if backend is running
echo "🌐 Checking backend connection..."
python -c "import requests; requests.get('http://localhost:8000/', timeout=2); print('✅ Backend is running')" 2>/dev/null || echo "⚠️ Backend not detected. Make sure to start it with: python main.py"

echo ""
echo "🚀 Starting Face Tracking System..."
echo "================================="
echo "📹 This will open a camera window showing face tracking"
echo "🎯 Focus scores will be automatically sent to FocusMind"
echo "🔲 Press ESC in the camera window to stop tracking"
echo "⚠️ Make sure your FocusMind app is open in the browser"
echo ""

# Start the face tracking system
python face_focus_tracker.py --source 0 --backend http://localhost:8000

echo "🛑 Face tracking stopped"