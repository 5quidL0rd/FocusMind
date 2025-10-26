# Note-Taking Grace Period Feature

## Overview
The FocusMind face tracker now includes a **Note-Taking Grace Period** system that allows you to look down to take notes without immediately impacting your focus score.

## How It Works

### Detection
The system automatically detects when you're in a note-taking position by monitoring:
- **Gaze direction**: Looking "Down" with your eyes
- **Head pitch**: Tilting your head down more than 15 degrees
- **Head yaw**: Allowing slight left/right head turns (up to 30 degrees) for natural note-taking positions

### Grace Period
When note-taking is detected:
1. A **5-second grace period** begins automatically
2. During this period, focus score penalties for looking down are greatly reduced or eliminated
3. The system displays the remaining grace period time in the HUD
4. Console messages inform you when the grace period starts and ends

### Visual Indicators
- **HUD Display**: Shows `📝{time}s` next to your focus score during the grace period
- **Console Messages**: 
  - `📝 Note-taking detected - starting 5.0s grace period`
  - `📝 Note-taking grace period: X.Xs remaining`
  - `📝 Note-taking grace period expired - applying focus penalties`
  - `📝 Note-taking ended - grace period reset`

## Supported Note-Taking Positions

✅ **Allowed (triggers grace period)**:
- Looking straight down
- Looking down and slightly left (up to 30°)
- Looking down and slightly right (up to 30°)
- Head tilted down while eyes look forward
- Combination of head tilt and downward gaze

❌ **Not allowed (no grace period)**:
- Looking far left or right without looking down
- Head turned more than 30° to either side
- Looking up
- Looking away from the screen entirely

## Technical Details

### Settings
- **Grace period duration**: 5.0 seconds (configurable)
- **Head pitch threshold**: -15 degrees (looking down)
- **Head yaw tolerance**: ±30 degrees (slight side turns)

### Score Impact
- **During grace period**: Gaze direction "Down" is treated as "Center" (no penalty)
- **During grace period**: Head pitch penalties are greatly reduced
- **After grace period**: Normal focus scoring resumes with standard penalties

### Integration
The grace period system is integrated into:
- `face_focus_tracker.py`: Main detection and timing logic
- `FocusScore.py`: Modified scoring to respect grace period
- Real-time HUD display
- Console logging for debugging

## Usage Tips

1. **Quick notes**: The 5-second grace period is perfect for jotting down quick notes or ideas
2. **Longer writing**: For extended note-taking, expect gradual score reduction after the grace period
3. **Natural positioning**: You can naturally position your notebook slightly to the left or right
4. **Reset mechanism**: Look back up briefly to reset the grace period if needed

## Example Workflow

1. You're focused on your screen (high focus score)
2. You look down to write a note → Grace period starts (5 seconds)
3. You continue writing → Score remains high during grace period
4. After 5 seconds of continuous note-taking → Gradual penalties begin
5. You look back up → Grace period resets, focus score can recover

This system balances maintaining focus awareness while allowing natural note-taking behavior without harsh immediate penalties.