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
    prev_score: float
) -> float:
    """
    Return a smooth focus score (0-100).
    Improved version with more forgiving scoring and better recovery.
    """
    # More nuanced and balanced weights
    w_face = 0.25      # Face presence (reduced from 0.35)
    w_gaze = 0.25      # Gaze direction (reduced from 0.30)
    w_eyes = 0.20      # Eye openness 
    w_head = 0.15      # Head position (increased from 0.10)
    w_blink = 0.08     # Blink rate (increased from 0.03)
    w_typing = 0.07    # Typing activity (increased from 0.02)

    # Normalize sub-scores (0..1)
    face_score = 1.0 if face_present else 0.0

    # More nuanced eye scoring
    eye_score = float(max(0.0, min(1.0, eyes_open_ratio)))
    if eyes_closed_duration > 3.0:  # Eyes closed for 3+ seconds
        eye_score = max(0.6, eye_score - 0.2)  # Less harsh penalty, min 0.6 instead of 0.5

    # More nuanced gaze scoring - less volatile
    gaze_map = {"forward": 1.0, "down": 0.9, "up": 0.9, "left": 0.8, "right": 0.8}  # Less harsh differences
    gaze_score = gaze_map.get(gaze_direction, 0.8)  # Higher default
    # Gentle gaze away penalty
    gaze_away_penalty = float(max(0.0, min(0.25, gaze_away_ratio)))  # Max 25% penalty (reduced)
    gaze_score *= (1.0 - gaze_away_penalty)

    # More balanced head movement scoring - less sensitive
    head_score = 1.0 - min(abs(head_yaw) / 120.0, 0.5)  # 120° tolerance, max 50% penalty (reduced)
    # Gentle pitch penalty
    if head_pitch < -45:  # Looking down significantly
        head_score = max(0.6, head_score - 0.15)  # Gentler penalty, min 0.6
    head_score = max(0.6, min(1.0, head_score))  # Higher minimum head score of 0.6

    # Simplified blink scoring - less sensitive
    blink_score = 1.0 - min(abs(blink_rate - 20.0) / 80.0, 0.3)  # Even more forgiving range
    blink_score = max(0.7, blink_score)  # Higher minimum blink score

    # Typing score with moderate baseline
    if typing_active:
        typing_score = min(keys_per_30s / 15.0, 1.0)  # 15 keys/30s target
    else:
        typing_score = 0.6  # Moderate baseline when not typing

    weighted_sum = (
        w_face * face_score +
        w_gaze * gaze_score +
        w_eyes * eye_score +
        w_head * head_score +
        w_blink * blink_score +
        w_typing * typing_score
    )

    # Much slower, more gradual EMA smoothing
    raw_score = weighted_sum * 100.0
    
    # Very conservative adaptive smoothing for slow changes
    if raw_score > prev_score:
        alpha = 0.2  # Very slow improvement (was 0.6)
    else:
        alpha = 0.15  # Very slow decline (was 0.5)
    
    focus_score = (1.0 - alpha) * float(prev_score) + alpha * raw_score
    
    # Reduce bonuses to prevent sudden jumps
    if focus_score > 85 and prev_score > 80:
        focus_score = min(100.0, focus_score + 0.5)  # Tiny sustained bonus (was 1.5)
    
    # Remove the high performance boost entirely to prevent jumps
    # if focus_score > 85:
    #     boost_factor = min(3.0, (focus_score - 85) * 0.3)
    #     focus_score = min(100.0, focus_score + boost_factor)
    #     print(f"🔧 After high performance boost: {focus_score:.1f}")
    
    focus_score = max(0.0, min(100.0, focus_score))
    return round(focus_score, 2)


# -------------------------
# 1.5) Enhanced face tracking integration functions
# -------------------------
def compute_enhanced_face_metrics(landmarks_array, frame_shape, current_time, blink_state=None):
    """
    Compute enhanced face metrics using face_track.py functions.
    
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
                                     blink_state=None, face_present=True):
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
    focus_score = compute_focus_score(
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
        prev_score=prev_score
    )
    
    return focus_score, updated_blink_state


# -------------------------
# 2) Chart generator (PNG + base64)
# -------------------------
def generate_focus_chart_base64(focus_data: List[Dict[str, Any]], dpi: int = 120) -> Tuple[str, bytes]:
    """
    - focus_data: list of {'timestamp': 'HH:MM:SS', 'focus_score': float}
    - returns (png_path, base64_bytes)
    The PNG is created in-memory and encoded to base64; png_path is a suggested filename.
    """
    if not focus_data:
        raise ValueError("focus_data is empty")

    df = pd.DataFrame(focus_data)
    # ensure chronological
    df = df.copy()
    try:
        # If timestamp strings, keep as labels; do not convert to datetime to keep tick labels readable
        x = list(df["timestamp"])
    except Exception:
        x = list(range(len(df)))

    y = list(df["focus_score"])

    fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
    ax.plot(x, y, linewidth=2)
    ax.set_title("Focus Score — Session")
    ax.set_xlabel("Time")
    ax.set_ylabel("Focus Score (0–100)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.4, linestyle="--")
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



