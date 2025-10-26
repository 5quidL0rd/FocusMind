#!/usr/bin/env python3
"""
Test script for dynamic penalty system
"""

import time
from face_focus_tracker import FaceFocusTracker

def test_dynamic_penalty_system():
    """Test the dynamic penalty escalation system"""
    tracker = FaceFocusTracker(show_video=False)
    
    print("🧪 Testing Dynamic Penalty System")
    print("=" * 60)
    
    # Simulate different score scenarios
    test_scenarios = [
        (85, "Excellent score - no penalties"),
        (75, "Concerning score (70s) - should trigger penalty tracking"),
        (65, "Poor score (60s) - should have higher base penalty"),
        (45, "Critical score (40s) - should have highest penalty"),
    ]
    
    current_time = time.perf_counter()
    
    for score, description in test_scenarios:
        print(f"\n📊 Testing: {description}")
        print(f"   Score: {score}%")
        print(f"   Range: {tracker.get_score_range(score).title()}")
        
        # Test penalty at different time intervals
        for interval in [0, 2, 6, 12, 20, 35]:
            test_time = current_time + interval
            penalty = tracker.update_low_score_penalty_system(score, test_time)
            final_score = score * penalty
            
            if penalty < 1.0:
                print(f"   After {interval}s: {score}% → {final_score:.1f}% (×{penalty:.2f})")
            else:
                print(f"   After {interval}s: {score}% (no penalty)")
        
        # Reset for next test
        tracker.low_score_tracking = {
            'start_time': None,
            'current_range': None,
            'last_range_change': 0.0,
            'penalty_multiplier': 1.0,
            'escalation_intervals': [5.0, 10.0, 15.0, 30.0],
            'last_escalation_message': 0.0
        }

def test_penalty_escalation_messages():
    """Test the escalation warning messages"""
    tracker = FaceFocusTracker(show_video=False)
    
    print("\n\n🚨 Testing Escalation Warning Messages")
    print("=" * 60)
    
    # Simulate staying in concerning range for extended time
    score = 72  # Concerning range
    current_time = time.perf_counter()
    
    print(f"Simulating sustained score of {score}% (concerning range)...")
    
    # Test escalation at different intervals
    for seconds in range(0, 40, 3):
        test_time = current_time + seconds
        penalty = tracker.update_low_score_penalty_system(score, test_time)
        final_score = score * penalty
        
        if seconds % 6 == 0:  # Print every 6 seconds to see progression
            print(f"After {seconds}s: {score}% → {final_score:.1f}% (×{penalty:.2f})")

if __name__ == "__main__":
    print("🎯 Testing FocusMind Dynamic Penalty System")
    print("This system penalizes sustained low focus scores")
    print("\nKey Features:")
    print("• 70-79% (concerning): 5% base penalty + time escalation")
    print("• 60-69% (poor): 10% base penalty + time escalation") 
    print("• 50-59% (critical): 15% base penalty + time escalation")
    print("• <50% (emergency): 20% base penalty + time escalation")
    print("• Time escalation: +5% penalty every 5s, 10s, 15s, 30s")
    print("\n" + "=" * 60)
    
    test_dynamic_penalty_system()
    test_penalty_escalation_messages()
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("\n📈 How the Dynamic Penalty System Works:")
    print("   • Scores 80%+ = No penalties (good/excellent range)")
    print("   • Scores 70-79% = Immediate 5% penalty + escalation after 5s")
    print("   • Lower scores = Higher base penalties + faster escalation")
    print("   • Recovery to 80%+ immediately removes all penalties")
    print("   • Visual warnings at 5s, 10s, 15s, and 30s intervals")