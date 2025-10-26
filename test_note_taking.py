#!/usr/bin/env python3
"""
Test script for note-taking grace period functionality
"""

import time
from face_focus_tracker import FaceFocusTracker

def test_note_taking_detection():
    """Test the note-taking detection logic"""
    tracker = FaceFocusTracker(show_video=False)
    
    print("🧪 Testing note-taking detection logic...")
    
    # Test scenarios
    test_cases = [
        # (gaze_direction, head_pitch, head_yaw, expected_result, description)
        ("Center", 0.0, 0.0, False, "Looking straight ahead"),
        ("Down", 0.0, 0.0, True, "Looking down with gaze"),
        ("Center", -20.0, 0.0, True, "Looking down with head pitch"),
        ("Down", -20.0, 0.0, True, "Looking down with both gaze and head"),
        ("Down", 0.0, 25.0, True, "Looking down and slightly right (note-taking position)"),
        ("Down", 0.0, -25.0, True, "Looking down and slightly left (note-taking position)"),
        ("Down", 0.0, 40.0, False, "Looking down but too far right"),
        ("Left", 0.0, 0.0, False, "Looking left without looking down"),
        ("Right", 0.0, 0.0, False, "Looking right without looking down"),
        ("Center", -10.0, 45.0, False, "Head turned too far right"),
    ]
    
    for gaze_direction, head_pitch, head_yaw, expected, description in test_cases:
        result = tracker.is_note_taking_gesture(gaze_direction, head_pitch, head_yaw)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"{status} {description}: gaze={gaze_direction}, pitch={head_pitch:.1f}°, yaw={head_yaw:.1f}° -> {result}")

def test_grace_period_timing():
    """Test the grace period timing logic"""
    tracker = FaceFocusTracker(show_video=False)
    tracker.note_taking_grace_period = 2.0  # Shorter for testing
    
    print("\n🧪 Testing grace period timing...")
    
    current_time = time.perf_counter()
    
    # Start note-taking
    print("1. Starting note-taking gesture...")
    in_grace = tracker.update_note_taking_grace_period(current_time, "Down", -20.0, 10.0)
    print(f"   Grace period active: {in_grace} (should be True)")
    
    # Check during grace period
    print("2. Checking during grace period (1s later)...")
    current_time += 1.0
    in_grace = tracker.update_note_taking_grace_period(current_time, "Down", -20.0, 10.0)
    print(f"   Grace period active: {in_grace} (should be True)")
    
    # Check at end of grace period
    print("3. Checking at end of grace period (2s later)...")
    current_time += 1.0
    in_grace = tracker.update_note_taking_grace_period(current_time, "Down", -20.0, 10.0)
    print(f"   Grace period active: {in_grace} (should be False)")
    
    # Stop note-taking
    print("4. Stopping note-taking gesture...")
    current_time += 0.1
    in_grace = tracker.update_note_taking_grace_period(current_time, "Center", 0.0, 0.0)
    print(f"   Grace period active: {in_grace} (should be False)")

if __name__ == "__main__":
    print("🎯 Testing FocusMind Note-Taking Grace Period System")
    print("=" * 60)
    
    test_note_taking_detection()
    test_grace_period_timing()
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("\n📝 How to use the note-taking grace period:")
    print("   • Look down (gaze or head) to start the 5-second grace period")
    print("   • You can look slightly left or right (up to 30°) while taking notes")
    print("   • The grace period resets when you stop looking down")
    print("   • After 5 seconds of continuous note-taking, penalties gradually apply")