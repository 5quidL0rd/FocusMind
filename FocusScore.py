import math
import json
import subprocess
import base64
import io
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

# Import crucial functions from face_track.py for enhanced metrics
from face_track import (
    compute_average_ear, 
    normalize_ear, 
    update_blink_metrics,
    estimate_head_orientation,
    rotation_matrix_to_euler_angles,
    HEAD_POSE_LANDMARKS,
    MODEL_POINTS
)

# Import eye calibration for personalized scoring
try:
    from eye_calibration import load_eye_calibration, normalize_ear_personalized
    EYE_CALIBRATION_AVAILABLE = True
    # Load calibration once at module level to avoid repeated file reads
    _CACHED_EYE_CALIBRATION = None
    try:
        _CACHED_EYE_CALIBRATION = load_eye_calibration()
    except:
        pass
except ImportError:
    EYE_CALIBRATION_AVAILABLE = False
    _CACHED_EYE_CALIBRATION = None


# -------------------------
# 1) Focus score calculator
# -------------------------
def compute_focus_score(
    face_present: bool,
    eyes_open_ratio: float,
    eyes_closed_duration: float,
    gaze_direction: str,
    gaze_away_ratio: float,
    head_pitch: float,
    head_yaw: float,
    blink_rate: float,
    keys_per_30s: int,
    typing_active: bool,
    focus_trend: float,
    prev_score: float,
    sustained_focus_time: float = 0.0,
    current_time: float = None,
    in_note_taking_grace_period: bool = False
) -> tuple:
    """
    Return a smooth focus score (0-100).
    Face presence is MANDATORY - score goes to 0 when head is off screen.
    """
    # Face-dominant weights - face presence controls everything
    w_face = 1.0       # Face presence is MANDATORY - 100% weight when absent
    w_gaze = 0.40      # Gaze direction - important when face is present  
    w_eyes = 0.30      # Eye behavior - important when face is present
    w_head = 0.20      # Head position - moderate when face is present
    w_blink = 0.10     # Blink rate - minimal when face is present

    # MANDATORY FACE PRESENCE CHECK
    face_score = 1.0 if face_present else 0.0
    
    # If no face detected, return 0 immediately (with smooth transition)
    if not face_present:
        # Smooth transition to zero when face disappears
        alpha = 0.15  # Gradual transition to zero
        focus_score = (1.0 - alpha) * float(prev_score) + alpha * 0.0
        return round(max(0.0, focus_score), 2), sustained_focus_time

    # Eye scoring - More generous when eyes are open normally
    eye_score = float(max(0.0, min(1.0, eyes_open_ratio)))
    
    # Apply a boost for normal eye openness to reward natural looking
    if eye_score >= 0.6:  # If eyes are reasonably open
        eye_score = min(1.0, eye_score * 1.2)  # 20% boost for normal eye openness
    
    if eyes_closed_duration > 3.0:  # Eyes closed for 3+ seconds - STRONG penalty
        penalty_factor = min((eyes_closed_duration - 3.0) / 5.0, 0.7)  # Progressive penalty up to 70%
        eye_score = max(0.2, eye_score - penalty_factor)  # Strong penalty after 3 seconds
    # No penalty for brief closures (blinking, brief rests)

    # Gaze scoring - More forgiving and allows higher scores
    # Apply note-taking grace period protection
    if in_note_taking_grace_period:
        # During grace period, treat "down" gaze as if it were "center"
        # This allows note-taking without immediate penalty
        effective_gaze_direction = "Center" if gaze_direction == "Down" else gaze_direction
        gaze_map = {
            "forward": 1.0,    # Perfect focus
            "Center": 1.0,     # Also perfect (includes protected "down" during grace period)
            "down": 1.0,       # Protected during grace period
            "up": 0.90,        # Slight penalty - less bouncy
            "left": 0.70,      # Moderate penalty - distraction
            "right": 0.70,     # Moderate penalty - distraction
            "Away": 0.50       # Moderate penalty - less harsh for brief glances
        }
        gaze_score = gaze_map.get(effective_gaze_direction, 0.90)
        # No gaze away penalty during grace period for protected directions
        gaze_away_penalty = 0.0 if gaze_direction == "Down" else float(max(0.0, min(0.10, gaze_away_ratio)))
    else:
        # Normal gaze scoring when not in grace period
        gaze_map = {
            "forward": 1.0,    # Perfect focus
            "Center": 1.0,     # Also perfect
            "down": 0.95,      # Minor penalty - allows note-taking
            "up": 0.90,        # Slight penalty - less bouncy
            "left": 0.70,      # Moderate penalty - distraction
            "right": 0.70,     # Moderate penalty - distraction
            "Away": 0.50       # Moderate penalty - less harsh for brief glances
        }
        gaze_score = gaze_map.get(gaze_direction, 0.90)  # Higher default for unknown directions
        # Reduced additional gaze away penalty
        gaze_away_penalty = float(max(0.0, min(0.10, gaze_away_ratio)))  # Reduced to 10% max penalty
    
    gaze_score *= (1.0 - gaze_away_penalty)

    # Head position scoring - More responsive to moderate head movements
    # Apply note-taking grace period protection for head pitch
    head_score = 1.0
    
    # Yaw (left/right) penalty - start penalizing earlier with more noticeable impact
    if abs(head_yaw) > 5:  # Start penalizing after just 5 degrees (was 15)
        yaw_penalty = min((abs(head_yaw) - 5) / 30.0, 0.60)  # Up to 60% penalty, more aggressive curve
        head_score -= yaw_penalty
    
    # Pitch (up/down) penalty - more responsive to moderate movements
    # Apply grace period protection for looking down
    if in_note_taking_grace_period and head_pitch < -3:
        # During grace period, reduce or eliminate head pitch penalty for looking down
        print(f"📝 Head pitch penalty reduced during grace period: {head_pitch:.1f}°")
        # Apply much smaller penalty during grace period
        pitch_penalty = min((abs(head_pitch) - 15) / 50.0, 0.10)  # Very small penalty, starts at 15°
        head_score -= max(0, pitch_penalty)  # Only apply if significant downward look
    elif head_pitch < -3:  # Looking down very slightly (was -5)
        pitch_penalty = min((abs(head_pitch) - 3) / 35.0, 0.40)  # Up to 40% penalty for down movement
        head_score -= pitch_penalty
    elif head_pitch > 8:  # Looking up (was 10)
        pitch_penalty = min((head_pitch - 8) / 30.0, 0.45)  # Up to 45% penalty for looking up
        head_score -= pitch_penalty
    
    head_score = max(0.30, min(1.0, head_score))  # Allow lower scores for poor head position

    # Blink scoring - very forgiving, natural behavior
    optimal_blink_rate = 18.0  # Natural blink rate
    blink_score = 1.0 - min(abs(blink_rate - optimal_blink_rate) / 50.0, 0.10)  # Very forgiving
    blink_score = max(0.90, blink_score)  # High minimum - blinking is natural

    # Typing score - not heavily weighted since we're focusing on visual attention
    if typing_active:
        typing_score = min(keys_per_30s / 20.0, 1.0)  # 20 keys/30s target
    else:
        typing_score = 0.9  # High baseline - not typing doesn't hurt focus

    # Face is present - calculate focus based on other factors
    # Normalize weights for when face is present (face weight is handled separately)
    total_weight = w_gaze + w_eyes + w_head + w_blink
    normalized_weights = {
        'gaze': w_gaze / total_weight,
        'eyes': w_eyes / total_weight, 
        'head': w_head / total_weight,
        'blink': w_blink / total_weight
    }

    weighted_sum = (
        normalized_weights['gaze'] * gaze_score +
        normalized_weights['eyes'] * eye_score +
        normalized_weights['head'] * head_score +
        normalized_weights['blink'] * blink_score
    )  # Face presence is mandatory - only calculate other factors when face is present

    # Add much more aggressive baseline boost for genuine focus
    # When all metrics are good, provide a substantial boost to get into 90-100 range
    baseline_boost = 0.0
    if (gaze_score >= 0.95 and eye_score >= 0.85 and 
        head_score >= 0.80 and blink_score >= 0.85):
        baseline_boost = 0.25  # 25% boost for excellent conditions - much more aggressive
    elif (gaze_score >= 0.90 and eye_score >= 0.75 and 
          head_score >= 0.65 and blink_score >= 0.80):
        baseline_boost = 0.20  # 20% boost for good conditions - much more aggressive
    elif (gaze_score >= 0.80 and eye_score >= 0.60 and 
          head_score >= 0.50 and blink_score >= 0.75):
        baseline_boost = 0.15  # 15% boost for decent conditions - much more aggressive

    # Apply DYNAMIC EMA smoothing based on score direction and magnitude
    raw_score = (weighted_sum + baseline_boost) * 100.0
    
   
    # Dynamic alpha based on score change direction and current score level
    score_change = raw_score - prev_score
    
    # Base alpha values - much more conservative for ultra-smooth updates
    alpha_decline = 0.04    # Very slow decline when performance drops
    alpha_improve = 0.02    # Very slow improvement when performance recovers
    
    # Dynamic adjustments based on score level - much less aggressive
    if raw_score < 70:  # In concerning range (70s)
        alpha_decline = 0.06   # Gentle decline in 70s - smooth but responsive
        alpha_improve = 0.015  # Slow recovery from 70s - earned improvement
        if abs(score_change) > 5:  # Large changes in 70s
            print(f"⚠️ Focus in 70s range - dynamic response: {prev_score:.1f}% → {raw_score:.1f}%")
    elif raw_score < 60:  # In poor range (60s)
        alpha_decline = 0.08   # Moderate decline in 60s - still smooth
        alpha_improve = 0.01   # Very slow recovery from 60s
        if abs(score_change) > 5:
            print(f"🚨 Focus in 60s range - urgent response: {prev_score:.1f}% → {raw_score:.1f}%")
    elif raw_score < 50:  # Critical range
        alpha_decline = 0.10   # Noticeable but smooth decline below 50
        alpha_improve = 0.008  # Extremely slow recovery from critical levels
        if abs(score_change) > 5:
            print(f"🔴 CRITICAL focus level - maximum response: {prev_score:.1f}% → {raw_score:.1f}%")
    
    # Choose alpha based on direction of change
    if raw_score > prev_score:
        alpha = alpha_improve  # Slow improvement
    else:
        alpha = alpha_decline  # Faster decline but controlled
    
    # Additional boost for sustained good performance
    if raw_score >= 85 and prev_score >= 85:
        alpha = 0.05  # Slower response in good range for more stability
    
    focus_score = (1.0 - alpha) * float(prev_score) + alpha * raw_score
    
    # SUSTAINED ATTENTION BOOST SYSTEM
    # Track sustained focus time and provide gradual boosts for consistency
    if current_time is None:
        import time
        current_time = time.perf_counter()
    
    # Update sustained focus tracking
    good_focus_threshold = 80.0  # Score threshold for "good focus"
    new_sustained_time = sustained_focus_time
    
    if focus_score >= good_focus_threshold:
        # Maintaining good focus - increment sustained time
        new_sustained_time = sustained_focus_time + 1.0  # Roughly 1 second increment
    else:
        # Focus dropped - reset sustained time
        new_sustained_time = 0.0
    
    # Apply sustained attention boosts
    sustained_boost = 0.0
    if new_sustained_time >= 10.0:  # After 10 seconds of sustained focus
        # Progressive boost based on sustained time
        # Boost starts small and grows gradually
        boost_duration = min(new_sustained_time - 10.0, 60.0)  # Cap at 60 seconds of boost time
        sustained_boost = min(boost_duration * 0.1, 5.0)  # Max 5 point boost after 50 seconds
        
        # Apply boost gradually to maintain smoothness
        focus_score = min(100.0, focus_score + sustained_boost)
    
    focus_score = max(0.0, min(100.0, focus_score))
    return round(focus_score, 2), new_sustained_time


# -------------------------
# 1.5) Enhanced face tracking integration functions
# -------------------------
def compute_enhanced_face_metrics(landmarks_array, frame_shape, current_time, blink_state=None):
    """
    Compute enhanced face metrics using face_track.py functions with personalized eye calibration.
    
    Args:
        landmarks_array: numpy array of MediaPipe face landmarks (468 points)
        frame_shape: tuple of (height, width) of the frame
        current_time: current timestamp
        blink_state: dict with blink tracking state (optional)
    
    Returns:
        dict with enhanced metrics including accurate blink rate and head pose
    """
    # Initialize blink state if not provided
    if blink_state is None:
        blink_state = {
            'blink_in_progress': False,
            'blink_count': 0,
            'eyes_closed_start_time': None,
            'eyes_closed_duration': 0.0,
            'last_reset_time': current_time
        }
    
    # Compute accurate Eye Aspect Ratio using face_track functions
    ear = compute_average_ear(landmarks_array)
    
    # Use personalized eye normalization if available
    if EYE_CALIBRATION_AVAILABLE and _CACHED_EYE_CALIBRATION:
        eyes_open_ratio = normalize_ear_personalized(ear, _CACHED_EYE_CALIBRATION)
    else:
        eyes_open_ratio = normalize_ear(ear)
    
    # Update blink metrics with proper hysteresis
    (blink_in_progress, blink_count, eyes_closed_start_time, 
     eyes_closed_duration, blink_detected) = update_blink_metrics(
        eyes_open_ratio,
        current_time,
        blink_state['blink_in_progress'],
        blink_state['blink_count'],
        blink_state['eyes_closed_start_time'],
        blink_state['eyes_closed_duration']
    )
    
    # Update blink state
    blink_state.update({
        'blink_in_progress': blink_in_progress,
        'blink_count': blink_count,
        'eyes_closed_start_time': eyes_closed_start_time,
        'eyes_closed_duration': eyes_closed_duration
    })
    
    # Calculate blink rate (blinks per minute)
    time_elapsed = current_time - blink_state['last_reset_time']
    if time_elapsed >= 60.0:  # Reset every minute
        blink_rate = (blink_count / time_elapsed) * 60.0
        blink_state['blink_count'] = 0
        blink_state['last_reset_time'] = current_time
    else:
        # Estimate current rate
        blink_rate = (blink_count / max(time_elapsed, 1.0)) * 60.0
    
    # Compute accurate head pose using face_track functions
    head_orientation = estimate_head_orientation(landmarks_array, frame_shape)
    if head_orientation:
        head_pitch, head_yaw = head_orientation
    else:
        head_pitch, head_yaw = 0.0, 0.0
    
    return {
        'eyes_open_ratio': eyes_open_ratio,
        'eyes_closed_duration': eyes_closed_duration,
        'blink_rate': blink_rate,
        'head_pitch': head_pitch,
        'head_yaw': head_yaw,
        'blink_detected': blink_detected,
        'ear': ear
    }, blink_state


def compute_focus_score_with_landmarks(landmarks_array, frame_shape, gaze_direction, 
                                     gaze_away_ratio, current_time, prev_score, 
                                     blink_state=None, face_present=True, 
                                     in_note_taking_grace_period=False):
    """
    Enhanced focus score computation using actual MediaPipe landmarks.
    Integrates face_track.py functions for accurate blink and head pose detection.
    
    Args:
        landmarks_array: numpy array of MediaPipe face landmarks
        frame_shape: tuple of (height, width)
        gaze_direction: string direction of gaze
        gaze_away_ratio: float ratio of time looking away
        current_time: current timestamp
        prev_score: previous focus score
        blink_state: blink tracking state
        face_present: whether face is detected
    
    Returns:
        tuple of (focus_score, updated_blink_state)
    """
    # Get enhanced metrics using face_track.py functions
    enhanced_metrics, updated_blink_state = compute_enhanced_face_metrics(
        landmarks_array, frame_shape, current_time, blink_state
    )
    
    # Use the enhanced metrics in the main focus score computation
    result = compute_focus_score(
        face_present=face_present,
        eyes_open_ratio=enhanced_metrics['eyes_open_ratio'],
        eyes_closed_duration=enhanced_metrics['eyes_closed_duration'],
        gaze_direction=gaze_direction,
        gaze_away_ratio=gaze_away_ratio,
        head_pitch=enhanced_metrics['head_pitch'],
        head_yaw=enhanced_metrics['head_yaw'],
        blink_rate=enhanced_metrics['blink_rate'],
        keys_per_30s=0,  # Not tracking typing
        typing_active=False,  # Not tracking typing
        focus_trend=0.0,  # Could be enhanced
        prev_score=prev_score,
        sustained_focus_time=0.0,  # Default value for sustained time
        current_time=current_time,
        in_note_taking_grace_period=in_note_taking_grace_period
    )
    
    focus_score, sustained_time = result  # Unpack the tuple
    
    return focus_score, updated_blink_state


# -------------------------
# 2) Chart generator (PNG + base64)
# -------------------------
def generate_focus_chart_base64(focus_data: List[Dict[str, Any]], dpi: int = 120) -> Tuple[str, bytes]:
    """
    - focus_data: list of {'timestamp': 'HH:MM:SS', 'focus_score': float}
    - returns (png_path, base64_bytes)
    The PNG is created in-memory and encoded to base64; png_path is a suggested filename.
    WORKS WITH ANY NUMBER OF DATA POINTS - even just 1!
    """
    if not focus_data:
        raise ValueError("focus_data is empty")

    print(f"🎨 Generating chart with {len(focus_data)} data points...")
    
    df = pd.DataFrame(focus_data)
    # ensure chronological
    df = df.copy()
    
    # Handle timestamp
    try:
        # If timestamp strings, keep as labels; do not convert to datetime to keep tick labels readable
        x = list(df["timestamp"])
    except Exception as e:
        print(f"⚠️ Timestamp issue: {e}, using indices instead")
        x = list(range(len(df)))

    # Handle focus_score
    try:
        y = list(df["focus_score"])
    except Exception as e:
        print(f"❌ ERROR: Could not find 'focus_score' column in data!")
        print(f"Available columns: {df.columns.tolist()}")
        print(f"Sample data: {focus_data[0] if focus_data else 'None'}")
        raise

    # Calculate session statistics
    avg_score = sum(y) / len(y)
    min_score = min(y)
    max_score = max(y)
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    ax.plot(x, y, linewidth=2, color='#2E86AB', label='Focus Score')
    
    # Add horizontal lines for statistics
    ax.axhline(y=avg_score, color='#A23B72', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Average: {avg_score:.1f}')
    ax.axhline(y=max_score, color='#F18F01', linestyle=':', linewidth=1.5, alpha=0.8, label=f'Highest: {max_score:.1f}')
    ax.axhline(y=min_score, color='#C73E1D', linestyle=':', linewidth=1.5, alpha=0.8, label=f'Lowest: {min_score:.1f}')
    
    # Enhanced title with session summary
    session_duration = len(y) / 60 if len(y) > 60 else len(y) / 60  # Approximate duration in minutes
    title = f"Focus Score — Session Summary\nAvg: {avg_score:.1f} | Range: {min_score:.1f} - {max_score:.1f} | Duration: ~{session_duration:.1f}min"
    ax.set_title(title, fontsize=14, pad=20)
    
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Focus Score (0–100)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.4, linestyle="--")
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add text box with detailed statistics
    stats_text = f"📊 Session Statistics:\n"
    stats_text += f"• Average Score: {avg_score:.1f}\n"
    stats_text += f"• Highest Score: {max_score:.1f}\n"
    stats_text += f"• Lowest Score: {min_score:.1f}\n"
    stats_text += f"• Total Measurements: {len(y)}"
    
    # Position text box in the lower left corner
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
            verticalalignment='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    # reduce x-ticks if too many
    if len(x) > 30:
        step = max(1, len(x) // 20)
        for label in ax.xaxis.get_ticklabels()[::step]:
            label.set_rotation(45)
    else:
        for label in ax.xaxis.get_ticklabels():
            label.set_rotation(45)

    fig.tight_layout()

    # in-memory PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()
    b64 = base64.b64encode(png_bytes).decode("ascii")
    # timestamped filename suggestion
    png_path = f"focus_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    return png_path, b64.encode("ascii")


# -------------------------
# 3) Session numeric breakdown / stats
# -------------------------
def generate_session_stats(focus_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Produce numeric breakdown for the ENTIRE session. Returns a dict you can send to GPT.
    Includes time-series list of raw scores and derived metrics.
    """
    if not focus_data:
        return {}

    df = pd.DataFrame(focus_data)
    scores = df["focus_score"].astype(float).values
    times = df["timestamp"].tolist()

    deltas = np.diff(scores, prepend=scores[0])
    ups = deltas[deltas > 0]
    downs = deltas[deltas < 0]

    def count_recoveries(arr, threshold=10.0):
        # number of times a drop of >= threshold is followed later by an increase of >= threshold
        count = 0
        for i in range(len(arr) - 1):
            if arr[i] <= -threshold:
                # search forward for a corresponding recovery
                for j in range(i+1, len(arr)):
                    if arr[j] >= threshold:
                        count += 1
                        break
        return count

    stats = {
        "duration_seconds": len(scores),
        "average_focus": float(np.mean(scores)),
        "median_focus": float(np.median(scores)),
        "std_focus": float(np.std(scores)),
        "min_focus": float(np.min(scores)),
        "min_focus_time": times[int(np.argmin(scores))] if len(times) else None,
        "max_focus": float(np.max(scores)),
        "max_focus_time": times[int(np.argmax(scores))] if len(times) else None,
        "first_score": float(scores[0]),
        "last_score": float(scores[-1]),
        "total_up_changes": int(np.sum(ups > 0)),
        "total_down_changes": int(np.sum(downs < 0)),
        "sum_positive_deltas": float(np.sum(ups)) if ups.size else 0.0,
        "sum_negative_deltas": float(np.sum(downs)) if downs.size else 0.0,
        "largest_single_drop": float(np.min(deltas)),
        "largest_single_gain": float(np.max(deltas)),
        "recovery_moments_est": int(count_recoveries(deltas, threshold=8.0)),
        "focus_time_series": [{"timestamp": t, "score": float(s)} for t, s in zip(times, scores)]
    }
    return stats


# -------------------------
# 4) Trigger nudge (send JSON via stdin)
# -------------------------
def trigger_nudge_with_session(
    nudge_script_path: str,
    session_stats: Dict[str, Any],
    chart_b64_bytes: bytes,
    extra_context: Dict[str, Any] = None,
    timeout: int = 30
) -> str:
    """
    Calls `nudge.py` (or other script) and sends a JSON payload on stdin.
    Payload contains:
      - session_stats: numeric breakdown (see generate_session_stats)
      - chart_base64: base64 string (PNG)
      - extra_context: any additional keys (e.g., user id)
    The target script should read stdin and parse the JSON.
    Returns stdout from the called script (the LLM-generated message).
    """
    payload = {
        "session_stats": session_stats,
        "chart_base64": chart_b64_bytes.decode("ascii")
    }
    if extra_context:
        payload["extra_context"] = extra_context

    payload_str = json.dumps(payload)
    try:
        proc = subprocess.run(
            ["python", nudge_script_path],
            input=payload_str,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            return f"[nudge.py error code {proc.returncode}] stderr: {stderr}\nstdout: {stdout}"
        return stdout or "[nudge.py returned no text]"
    except Exception as e:
        return f"[trigger_nudge exception] {e}"



