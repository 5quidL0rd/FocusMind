"""
AI-Powered Face Tracking Focus Monitor
Integrates face_track.py with FocusScore.py to automatically monitor user focus
and send real-time focus scores to the FocusMind backend.
"""

import argparse
import math
import time
import requests
import json
import threading
from collections import deque
from typing import Optional, Dict, Any

import cv2
import mediapipe as mp
import numpy as np

# Import existing modules
from ema_smoother import GazeSmoother
from face_tracking_utils import RIGHT_EYE, LEFT_EYE, get_eye_pts, eye_gaze_vector, landmarks_to_np_array
from calibration import calibrate_user, load_calibration
from FocusScore import compute_focus_score, compute_focus_score_with_landmarks

# Import eye calibration for personalized scoring
try:
    from eye_calibration import load_eye_calibration, normalize_ear_personalized
    EYE_CALIBRATION_AVAILABLE = True
except ImportError:
    EYE_CALIBRATION_AVAILABLE = False

# Face tracking constants
HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_EYE_VERTICAL = (159, 145)
RIGHT_EYE_VERTICAL = (386, 374)

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -63.6, -12.5),    # Chin
    (-43.3, 32.7, -26.0),   # Left eye outer corner
    (43.3, 32.7, -26.0),    # Right eye outer corner
    (-28.9, -28.9, -24.1),  # Left mouth corner
    (28.9, -28.9, -24.1),   # Right mouth corner
], dtype=np.float32)

class FaceFocusTracker:
    """
    Real-time face tracking focus monitor that integrates with FocusMind backend
    """
    
    def __init__(self, backend_url="http://localhost:8000", update_interval=2.0, show_video=True, force_calibrate=False):
        self.backend_url = backend_url
        self.update_interval = update_interval  # seconds between focus score updates
        self.show_video = show_video
        self.force_calibrate = force_calibrate
        self.running = False
        
        # Load eye calibration if available
        self.eye_calibration = None
        if EYE_CALIBRATION_AVAILABLE:
            try:
                self.eye_calibration = load_eye_calibration()
                if self.eye_calibration:
                    print("👁️ Personal eye calibration loaded")
                else:
                    print("⚠️ No eye calibration found - run 'python eye_calibration.py' for better accuracy")
            except Exception as e:
                print(f"⚠️ Could not load eye calibration: {e}")
        
        # Focus tracking state
        self.current_focus_score = 100.0
        self.last_update_time = 0
        self.last_quote_threshold = 100  # Track last threshold that triggered a quote
        
        # Sustained attention tracking
        self.sustained_focus_time = 0.0  # Time spent in good focus (80+)
        
        # Video capture and MediaPipe
        self.cap = None
        self.face_mesh = None
        self.smoother = None
        self.cfg = None
        
        # Tracking variables
        self.eyes_closed_start_time = None
        self.eyes_closed_duration = 0.0
        self.blink_in_progress = False
        self.blink_count = 0
        self.session_start_time = 0
        
        # Enhanced blink state for face_track.py integration
        self.blink_state = {
            'blink_in_progress': False,
            'blink_count': 0,
            'eyes_closed_start_time': None,
            'eyes_closed_duration': 0.0,
            'last_reset_time': 0.0
        }
        
        # Rolling buffers for smoothing
        self.eye_horizontal_history = deque(maxlen=5)
        self.eye_vertical_history = deque(maxlen=5)
        self.head_pitch_history = deque(maxlen=5)
        self.head_yaw_history = deque(maxlen=5)
        
        # Face tracking data
        self.expression = None
        self.eyes_open_ratio = 0.0
        self.eye_direction = None
        self.head_direction = None
        self.gaze_label = ""
        
        print("🎯 FaceFocusTracker initialized")
        print(f"📡 Backend URL: {self.backend_url}")
        print(f"⏱️ Update interval: {self.update_interval}s")

    def initialize_camera(self, source=0):
        """Initialize camera and MediaPipe components"""
        try:
            # Open video source
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {source}")
            
            # Initialize MediaPipe Face-Mesh with higher sensitivity
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.7,  # Increased from 0.5 for better detection
                min_tracking_confidence=0.7,   # Increased from 0.5 for better tracking
            )
            
            self.smoother = GazeSmoother(alpha=0.3)
            
            # Try to load existing calibration first, or force calibration if requested
            if self.force_calibrate:
                print("🔧 Starting forced gaze calibration...")
                self.cfg = calibrate_user(
                    self.cap, self.face_mesh,
                    int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    get_eye_pts, eye_gaze_vector, smoothing_alpha=0.2
                )
                print("✅ Calibration completed")
            else:
                print("🔧 Loading gaze calibration...")
                try:
                    self.cfg = load_calibration()
                    print("✅ Loaded existing calibration")
                except (FileNotFoundError, json.JSONDecodeError):
                    print("⚠️ No calibration found, using default values")
                    # Default calibration values that work reasonably well
                    self.cfg = {
                        "left_thresh": -0.15,
                        "right_thresh": 0.15, 
                        "up_thresh": -0.10,
                        "down_thresh": 0.10,
                        "center_x": 0.0,
                        "center_y": 0.0
                    }
                    print("💡 Run with --calibrate flag to perform custom calibration")
            print("✅ Camera and MediaPipe initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize camera: {e}")
            return False

    def infer_expression(self, pts):
        """Simple expression inference from facial landmarks"""
        # Basic expression detection (simplified version)
        # This could be enhanced with more sophisticated analysis
        
        # Check if mouth is open (simple approximation)
        mouth_top = pts[13]    # Upper lip
        mouth_bottom = pts[14] # Lower lip
        mouth_opening = np.linalg.norm(mouth_top - mouth_bottom)
        
        if mouth_opening > 10:  # Threshold for mouth being open
            return "talking"
        
        return "neutral"

    def gaze_vector_to_label(self, vec):
        """Convert smoothed eye gaze vector to directional label with improved sensitivity"""
        if not self.cfg:
            return "Center"
            
        dx, dy = vec
        
        # More sensitive thresholds for better "away" detection
        sensitivity_factor = 0.7  # Make thresholds 30% more sensitive
        
        left_thresh = self.cfg["left_thresh"] * sensitivity_factor
        right_thresh = self.cfg["right_thresh"] * sensitivity_factor
        top_thresh = self.cfg["top_thresh"] * sensitivity_factor  
        down_thresh = self.cfg["down_thresh"] * sensitivity_factor
        
        # Additional extreme gaze detection for "Away"
        extreme_threshold = 0.8  # Very far from center
        
        if abs(dx) > extreme_threshold or abs(dy) > extreme_threshold:
            return "Away"  # Clearly looking away from screen
            
        if dx < left_thresh:
            return "Right"
        if dx > right_thresh:
            return "Left"
        if dy < top_thresh:
            return "Down"
        if dy > down_thresh:
            return "Up"
        
        return "Center"

    def process_frame(self, frame):
        """Process a single video frame and extract focus metrics"""
        current_time = time.perf_counter()
        
        # Convert to RGB for MediaPipe
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        face_present = bool(results.multi_face_landmarks)
        
        if face_present:
            # Use first detected face
            lm = results.multi_face_landmarks[0].landmark
            pts = landmarks_to_np_array(lm, frame.shape)
            
            # ENHANCED FACE PRESENCE VALIDATION
            # Check head pose to ensure person is actually facing camera
            from face_track import estimate_head_orientation
            head_orientation = estimate_head_orientation(pts, (h, w))
            
            if head_orientation:
                head_pitch, head_yaw = head_orientation
                
                # Only flag as "face away" for extreme head positions
                # Allow natural head movement during work
                max_yaw = 100.0        # Very tolerant - only flag extreme turns
                max_pitch_up = 50.0    # Very tolerant - looking way up
                max_pitch_down = 80.0  # Very tolerant - looking way down
                
                if (abs(head_yaw) > max_yaw or 
                    head_pitch > max_pitch_up or 
                    head_pitch < -max_pitch_down):
                    print(f"🔄 Head turned away: yaw={head_yaw:.1f}°, pitch={head_pitch:.1f}°")
                    face_present = False  # Override face detection if head is turned away
        
        if face_present:
            # Continue with normal face processing
            
            # Extract eye data for gaze tracking
            right_pts = get_eye_pts(RIGHT_EYE, lm, w, h)
            left_pts = get_eye_pts(LEFT_EYE, lm, w, h)
            
            vec_r = eye_gaze_vector(right_pts)
            vec_l = eye_gaze_vector(left_pts)
            
            # Average and smooth gaze vectors
            raw_vec = (vec_r + vec_l) / 2.0
            smoothed_vec = self.smoother.update(raw_vec)
            self.gaze_label = self.gaze_vector_to_label(smoothed_vec)
            
            # Use enhanced face_track.py functions for accurate eye tracking
            from face_track import compute_average_ear, normalize_ear, update_blink_metrics
            
            # Get accurate Eye Aspect Ratio with enhanced sensitivity
            ear = compute_average_ear(pts)
            
            # Use personalized eye normalization if available
            if self.eye_calibration:
                self.eyes_open_ratio = normalize_ear_personalized(ear, self.eye_calibration)
            else:
                self.eyes_open_ratio = normalize_ear(ear)
            
            # Eye closure detection - less sensitive to avoid false positives
            eye_closure_threshold = 0.15  # Much less sensitive - only for actual closures
            if self.eyes_open_ratio < eye_closure_threshold:
                if not hasattr(self, '_debug_eye_closure'):
                    print(f"👁️ Eyes partially closed: {self.eyes_open_ratio:.3f}")
                    self._debug_eye_closure = True
            else:
                if hasattr(self, '_debug_eye_closure'):
                    print(f"👁️ Eyes reopened: {self.eyes_open_ratio:.3f}")
                    delattr(self, '_debug_eye_closure')
            
            # Update blink metrics with proper tracking
            (self.blink_state['blink_in_progress'], 
             self.blink_state['blink_count'], 
             self.blink_state['eyes_closed_start_time'], 
             self.blink_state['eyes_closed_duration'], 
             blink_detected) = update_blink_metrics(
                self.eyes_open_ratio,
                current_time,
                self.blink_state['blink_in_progress'],
                self.blink_state['blink_count'],
                self.blink_state['eyes_closed_start_time'],
                self.blink_state['eyes_closed_duration']
            )
            
            # Update session blink count for rate calculation
            if blink_detected:
                self.blink_count += 1
            
            # Use the properly tracked eyes closed duration
            self.eyes_closed_duration = self.blink_state['eyes_closed_duration']
            
            # Detect expression
            self.expression = self.infer_expression(pts)
            
            # Calculate head pose using enhanced functions
            from face_track import estimate_head_orientation
            head_orientation = estimate_head_orientation(pts, (h, w))
            if head_orientation:
                head_pitch, head_yaw = head_orientation
            else:
                head_pitch, head_yaw = 0.0, 0.0
            
            # Store in history for smoothing
            self.head_pitch_history.append(head_pitch)
            self.head_yaw_history.append(head_yaw)
            
            # Determine gaze away ratio
            gaze_away_ratio = 0.0 if self.gaze_label == "Center" else 1.0
            
        else:
            # No face detected
            self.eyes_open_ratio = 0.0
            self.eyes_closed_duration = 0.0
            self.expression = None
            self.gaze_label = "Away"
            gaze_away_ratio = 1.0
            head_pitch = 0.0
            head_yaw = 0.0
        
        # Calculate blink rate using the enhanced tracking
        time_elapsed = current_time - self.blink_state.get('last_reset_time', self.session_start_time)
        if time_elapsed >= 60.0:  # Reset every minute
            blink_rate = (self.blink_state['blink_count'] / time_elapsed) * 60.0
            self.blink_state['blink_count'] = 0
            self.blink_state['last_reset_time'] = current_time
        else:
            # Estimate current rate
            blink_rate = (self.blink_state['blink_count'] / max(time_elapsed, 1.0)) * 60.0
        
        return {
            'face_present': face_present,
            'eyes_open_ratio': self.eyes_open_ratio,
            'eyes_closed_duration': self.eyes_closed_duration,
            'gaze_direction': self.gaze_label,
            'gaze_away_ratio': gaze_away_ratio,
            'head_pitch': head_pitch,
            'head_yaw': head_yaw,
            'blink_rate': blink_rate,
            'expression': self.expression,
            'landmarks_array': pts if face_present else None,
            'frame_shape': (h, w)
        }

    def compute_and_update_focus_score(self, metrics, landmarks_array=None, frame_shape=None):
        """Compute focus score using enhanced FocusScore.py functions and update backend if needed"""
        try:
            current_time = time.perf_counter()
            
            # Use enhanced face tracking if landmarks are available
            if landmarks_array is not None and frame_shape is not None:
                new_focus_score, self.blink_state = compute_focus_score_with_landmarks(
                    landmarks_array=landmarks_array,
                    frame_shape=frame_shape,
                    gaze_direction=metrics['gaze_direction'],
                    gaze_away_ratio=metrics['gaze_away_ratio'],
                    current_time=current_time,
                    prev_score=self.current_focus_score,
                    blink_state=self.blink_state,
                    face_present=metrics['face_present']
                )
            else:
                # Fallback to original method with sustained attention
                result = compute_focus_score(
                    face_present=metrics['face_present'],
                    eyes_open_ratio=metrics['eyes_open_ratio'],
                    eyes_closed_duration=metrics['eyes_closed_duration'],
                    gaze_direction=metrics['gaze_direction'],
                    gaze_away_ratio=metrics['gaze_away_ratio'],
                    head_pitch=metrics['head_pitch'],
                    head_yaw=metrics['head_yaw'],
                    blink_rate=metrics['blink_rate'],
                    keys_per_30s=0,  # Not tracking typing in this implementation
                    typing_active=False,  # Not tracking typing in this implementation
                    focus_trend=0.0,  # Could be enhanced to track trend
                    prev_score=self.current_focus_score,
                    sustained_focus_time=self.sustained_focus_time,
                    current_time=current_time
                )
                new_focus_score, self.sustained_focus_time = result
            
            # Print score changes and sustained attention info
            if abs(new_focus_score - self.current_focus_score) > 2.0:
                boost_info = ""
                if self.sustained_focus_time >= 10.0:
                    boost_info = f" (🔥 {self.sustained_focus_time:.0f}s sustained)"
                print(f"🎯 Focus score: {self.current_focus_score:.1f} → {new_focus_score:.1f}{boost_info}")
            
            # Debug: Print eye tracking details every few seconds
            if abs(new_focus_score - self.current_focus_score) > 2.0:
                print(f"👁️ Eye openness: {metrics['eyes_open_ratio']:.3f}, Eyes closed: {metrics['eyes_closed_duration']:.2f}s, Blinks: {self.blink_state['blink_count']}")
            
            self.current_focus_score = new_focus_score
            
            # Send update to backend
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                self.send_focus_update(new_focus_score)
                self.last_update_time = current_time
                
                # Check if we should trigger a motivational quote
                self.check_quote_thresholds(new_focus_score)
            
        except Exception as e:
            print(f"❌ Error computing focus score: {e}")

    def send_focus_update(self, focus_score):
        """Send focus score update to backend"""
        try:
            response = requests.post(f"{self.backend_url}/update-focus-score", 
                                   json={"focus_score": focus_score}, 
                                   timeout=1.0)
            if response.status_code == 200:
                print(f"📊 Focus score updated: {focus_score:.1f}")
            else:
                print(f"⚠️ Failed to update backend: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"🔌 Backend connection failed: {e}")

    def check_quote_thresholds(self, focus_score):
        """Check if focus score has crossed thresholds and trigger quotes accordingly"""
        thresholds = [80, 60, 50, 40, 20]  # Define your threshold levels
        
        for threshold in thresholds:
            # Check if we've crossed below this threshold since last quote
            if focus_score < threshold and self.last_quote_threshold >= threshold:
                self.trigger_motivational_quote(threshold)
                self.last_quote_threshold = threshold
                break
        
        # Reset threshold tracking if score improves significantly
        if focus_score > self.last_quote_threshold + 10:
            self.last_quote_threshold = 100

    def trigger_motivational_quote(self, threshold):
        """Trigger a motivational quote via the backend"""
        try:
            print(f"🚨 Focus dropped below {threshold}% - triggering motivational quote!")
            response = requests.post(f"{self.backend_url}/trigger-auto-motivation", 
                                   json={"threshold": threshold, "focus_score": self.current_focus_score}, 
                                   timeout=15.0)
            if response.status_code == 200:
                print("💪 Motivational quote triggered successfully")
            else:
                print(f"⚠️ Failed to trigger quote: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"🔌 Failed to trigger motivation: {e}")

    def draw_overlay(self, frame, metrics):
        """Draw focus tracking overlay on video frame"""
        if not self.show_video:
            return
        
        # Draw facial landmarks if face is present
        if metrics['face_present'] and metrics.get('landmarks_array') is not None:
            pts = metrics['landmarks_array']
            
            # Draw key facial landmarks (green dots like in face_track.py)
            # Eye corners and vertical points - Green for eye tracking
            for idx in (159, 145, 386, 374, 33, 133, 362, 263):
                cv2.circle(frame, tuple(pts[idx].astype(int)), 2, (0, 255, 0), -1)
            
            # Mouth corners and center points - Green 
            for idx in (61, 291, 13, 14):
                cv2.circle(frame, tuple(pts[idx].astype(int)), 2, (0, 255, 0), -1)
            
            # Head pose landmarks - Red for head tracking
            for idx in [1, 152, 33, 263, 61, 291]:  # HEAD_POSE_LANDMARKS
                cv2.circle(frame, tuple(pts[idx].astype(int)), 3, (0, 0, 255), -1)
            
            # Face contour points - Blue for face boundary
            face_contour = [10, 151, 9, 8, 168, 6, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
            for idx in face_contour[::3]:  # Every 3rd point to avoid clutter
                cv2.circle(frame, tuple(pts[idx].astype(int)), 1, (255, 0, 0), -1)
            
            # Nose tip and bridge - Yellow for nose tracking
            for idx in [1, 2, 5, 4, 6, 19, 20, 94, 125]:
                cv2.circle(frame, tuple(pts[idx].astype(int)), 2, (0, 255, 255), -1)
            
            # Draw eye regions if available
            try:
                from face_tracking_utils import RIGHT_EYE, LEFT_EYE, get_eye_pts
                h, w = frame.shape[:2]
                
                # Convert landmarks to format expected by get_eye_pts
                lm_list = []
                for point in pts:
                    lm_list.append(type('obj', (object,), {'x': point[0]/w, 'y': point[1]/h})())
                
                # Draw eye points
                right_pts = get_eye_pts(RIGHT_EYE, lm_list, w, h)
                left_pts = get_eye_pts(LEFT_EYE, lm_list, w, h)
                
                for pt in (right_pts["outer"], right_pts["inner"], right_pts["iris"],
                          left_pts["outer"], left_pts["inner"], left_pts["iris"]):
                    cv2.circle(frame, tuple(pt.astype(int)), 2, (255, 255, 0), -1)  # Cyan for eyes
            except Exception as e:
                # Fallback to basic eye landmark drawing if get_eye_pts fails
                pass
            
        # Draw HUD information
        sustained_info = ""
        if self.sustained_focus_time >= 10.0:
            sustained_info = f" 🔥{self.sustained_focus_time:.0f}s"
        
        hud_lines = [
            f"Face: {'Yes' if metrics['face_present'] else 'No'}",
            f"Focus Score: {self.current_focus_score:.1f}%{sustained_info}",
            f"Expression: {metrics['expression'] or '--'}",
            f"Eye openness: {metrics['eyes_open_ratio']:.3f}",
            f"Eyes closed: {metrics['eyes_closed_duration']:.2f}s",
            f"Blink rate: {metrics['blink_rate']:.1f}/min",
            f"Blink count: {self.blink_state['blink_count']}",
            f"Gaze: {metrics['gaze_direction']}",
            f"Head pitch: {metrics['head_pitch']:.1f}°",
            f"Head yaw: {metrics['head_yaw']:.1f}°"
        ]
        
        # Color code based on focus score
        if self.current_focus_score >= 80:
            color = (0, 255, 0)  # Green
        elif self.current_focus_score >= 60:
            color = (0, 255, 255)  # Yellow
        elif self.current_focus_score >= 40:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        y = 30
        for line in hud_lines:
            cv2.putText(frame, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y += 30
        
        # Draw focus score bar
        bar_width = 300
        bar_height = 20
        bar_x = 30
        bar_y = y + 10
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Fill bar based on focus score
        fill_width = int((self.current_focus_score / 100.0) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

    def run(self, source=0):
        """Main tracking loop"""
        print("🚀 Starting FaceFocusTracker...")
        
        if not self.initialize_camera(source):
            return False
        
        self.running = True
        self.session_start_time = time.perf_counter()
        
        # Initialize blink state with current time
        self.blink_state['last_reset_time'] = self.session_start_time
        
        print("📹 Face tracking started - monitoring focus...")
        print("Press ESC to stop tracking")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("📹 Video stream ended")
                    break
                
                # Process frame and get focus metrics
                metrics = self.process_frame(frame)
                
                # Compute and update focus score with enhanced landmarks
                self.compute_and_update_focus_score(
                    metrics, 
                    landmarks_array=metrics.get('landmarks_array'),
                    frame_shape=metrics.get('frame_shape')
                )
                
                # Draw overlay if video display is enabled
                if self.show_video:
                    self.draw_overlay(frame, metrics)
                    cv2.imshow("FocusMind - AI Focus Tracker", frame)
                    
                    # Check for ESC key
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopping face tracking...")
        finally:
            self.stop()
        
        return True

    def stop(self):
        """Stop the tracker and cleanup resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.show_video:
            cv2.destroyAllWindows()
        print("🛑 FaceFocusTracker stopped")

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Face Focus Tracker for FocusMind")
    parser.add_argument("--source", default="0", help="Video source (0 for webcam, or video file path)")
    parser.add_argument("--backend", default="http://localhost:8000", help="FocusMind backend URL")
    parser.add_argument("--interval", type=float, default=20.0, help="Focus score update interval (seconds)")
    parser.add_argument("--no-video", action="store_true", help="Run without video display (headless)")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode without camera (simulates focus tracking)")
    parser.add_argument("--calibrate", action="store_true", help="Force gaze calibration (otherwise uses defaults/saved)")
    parser.add_argument("--calibrate-eyes", action="store_true", help="Calibrate personal eye shape for better accuracy")
    
    args = parser.parse_args()
    
    # If eye calibration requested, run that instead
    if args.calibrate_eyes:
        print("🎯 Starting eye calibration process...")
        if EYE_CALIBRATION_AVAILABLE:
            import subprocess
            import sys
            result = subprocess.run([sys.executable, "eye_calibration.py"])
            if result.returncode == 0:
                print("✅ Eye calibration completed successfully!")
                print("Now run the tracker again to use your personalized settings.")
            else:
                print("❌ Eye calibration failed")
        else:
            print("❌ Eye calibration module not available")
        return
    
    # Convert source to int if it's a digit
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Create and run tracker
    tracker = FaceFocusTracker(
        backend_url=args.backend,
        update_interval=args.interval,
        show_video=not args.no_video,
        force_calibrate=args.calibrate
    )
    
    success = tracker.run(source)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())