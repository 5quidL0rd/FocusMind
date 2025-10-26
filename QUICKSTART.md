# FocusMind Quick Start Guide

## 🚀 Easy Start (One-Click!)

Just double-click: **`start_focusmind.bat`**

This will automatically:
1. ✅ Clean up any existing processes
2. ✅ Start the backend (FastAPI on port 8000)
3. ✅ Start the frontend (React on port 3001)
4. ✅ Start the face tracker (webcam)
5. ✅ Open your browser to http://localhost:3001

## ⏱️ Quick Test Mode

The Pomodoro timer is currently set to **30 seconds** for quick testing!
The face tracker updates every **5 seconds** to ensure enough data points for the chart.

### What You'll See After a 30-Second Session:

When the timer completes, you'll get:
- 🎤 **Break nudge** with David Goggins motivation audio
- 📊 **Focus Chart** showing:
  - Your focus score over time (line graph)
  - **Average Score** (purple dashed line)
  - **Highest Score** (orange dotted line)  
  - **Lowest Score** (red dotted line)
  - Statistics box with all metrics
  - Session duration and total measurements

## 🎯 Using the App

1. **Start Face Tracking:** The webcam will activate automatically
2. **Watch Your Score:** Real-time attention score updates every second
3. **Start Pomodoro:** Click "▶️ Start" for a 30-second focus session
4. **Get Motivated:** Voice nudges trigger automatically when focus drops
5. **Take Breaks:** At the end, hear break advice and see your performance chart

## ⏹️ Stopping the App

Press any key in the startup window to stop all services cleanly.

## 🔧 Changing Timer Duration

To change back to 25 minutes (production mode):
1. Edit `frontend/src/App.tsx`:
   - Line ~72: Change `useState(30)` to `useState(25 * 60)`
   - Line ~653: Change `setPomodoroTime(30)` to `setPomodoroTime(25 * 60)`
   - Line ~732: Change `pomodoroTime === 30` to `pomodoroTime === 25 * 60`

2. Edit `start_focusmind.bat`:
   - Change `--interval 5` to `--interval 20` (for less frequent updates)

## 📊 Focus Chart Features

The chart automatically includes:
- **Time-series line graph** of your focus throughout the session
- **Average focus line** (purple dashed)
- **Highest score line** (orange dotted)
- **Lowest score line** (red dotted)
- **Statistics box** showing:
  - Average Score
  - Highest Score
  - Lowest Score
  - Total Measurements
- **Session summary** in the title

All of this data is already built into the `FocusScore.py` module!

## 🎮 Controls

- **Pomodoro Timer:**
  - ▶️ Start - Begin focus session
  - ⏸️ Pause - Pause timer
  - 🔄 Reset - Reset to 30 seconds

- **Face Tracker Window:**
  - Press **ESC** to stop tracking
  - Shows live video with face landmarks

## 🔊 Audio Features

- **Voice Nudges:** Automatic when focus drops
- **Break Reminders:** David Goggins motivation after each session
- **Notification Nudges:** Browser notifications (enable permissions)

## ✨ Tips

1. Ensure good lighting for face tracking
2. Look at your screen naturally - the system rewards natural behavior
3. Take the break nudges seriously - they help performance!
4. Check the stats after each session to see your patterns

Enjoy your focus sessions! 💪🔥
