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
        
        # Focus score tracking for session statistics
        self.focus_score_history = deque()  # Keep all scores for session average
        
        # Note-taking grace period system
        self.note_taking_grace_period = 5.0  # 5 seconds grace period
        self.note_taking_start_time = None
        self.note_taking_active = False
        self.last_note_taking_check = 0.0
        
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
            print("📸 Initializing camera...")
            
            # Open video source with timeout protection
            self.cap = cv2.VideoCapture(source)
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set reasonable FPS
            
            # Test camera with timeout
            start_time = time.perf_counter()
            max_init_time = 10.0  # 10 second timeout
            
            camera_ready = False
            while time.perf_counter() - start_time < max_init_time:
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        camera_ready = True
                        break
                time.sleep(0.1)
            
            if not camera_ready:
                raise RuntimeError(f"Camera timeout: Cannot initialize video source {source} within {max_init_time}s")
            
            print("✅ Camera initialized successfully")
            
            # Initialize MediaPipe Face-Mesh with optimized settings
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

    def is_note_taking_gesture(self, gaze_direction, head_pitch, head_yaw):
        """
        Detect if the user is in a note-taking position.
        Note-taking is characterized by:
        - Looking down (gaze or head pitch)
        - Optionally looking slightly left or right (for notebook position)
        """
        # Check if looking down via gaze
        looking_down_gaze = gaze_direction == "Down"
        
        # Check if looking down via head pitch (more than 15 degrees down)
        looking_down_head = head_pitch < -15.0
        
        # Check if looking slightly to the side (common when writing in notebooks)
        # Allow up to 30 degrees left or right
        looking_slightly_sideways = abs(head_yaw) <= 30.0
        
        # Note-taking gesture: looking down + not extremely turned away
        is_note_taking = (looking_down_gaze or looking_down_head) and looking_slightly_sideways
        
        return is_note_taking

    def update_note_taking_grace_period(self, current_time, gaze_direction, head_pitch, head_yaw):
        """
        Update the note-taking grace period state.
        Returns True if currently in grace period, False otherwise.
        """
        is_note_taking_now = self.is_note_taking_gesture(gaze_direction, head_pitch, head_yaw)
        
        # If we detect note-taking and haven't started the grace period yet
        if is_note_taking_now and not self.note_taking_active:
            self.note_taking_start_time = current_time
            self.note_taking_active = True
            print(f"📝 Note-taking detected - starting {self.note_taking_grace_period}s grace period")
        
        # If we're not note-taking anymore, reset the grace period
        elif not is_note_taking_now and self.note_taking_active:
            self.note_taking_active = False
            self.note_taking_start_time = None
            print("📝 Note-taking ended - grace period reset")
        
        # Check if we're still within the grace period
        if self.note_taking_active and self.note_taking_start_time is not None:
            grace_time_elapsed = current_time - self.note_taking_start_time
            
            if grace_time_elapsed <= self.note_taking_grace_period:
                # Still in grace period
                remaining_time = self.note_taking_grace_period - grace_time_elapsed
                if current_time - self.last_note_taking_check > 1.0:  # Print update every second
                    print(f"📝 Note-taking grace period: {remaining_time:.1f}s remaining")
                    self.last_note_taking_check = current_time
                return True
            else:
                # Grace period expired, start applying penalties
                if self.note_taking_active:  # Only print once when grace period expires
                    print("📝 Note-taking grace period expired - applying focus penalties")
                    self.note_taking_active = False
        return False

    def process_frame(self, frame):
        """Process a single video frame and extract focus metrics"""
        current_time = time.perf_counter()
        
        # Convert to RGB for MediaPipe with error handling
        try:
            h, w = frame.shape[:2]
            
            # Resize frame if too large to prevent processing overload
            if w > 1280 or h > 720:
                scale = min(1280/w, 720/h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
                h, w = new_h, new_w
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe - this is where freezes often occur
            results = self.face_mesh.process(rgb)
            
        except Exception as e:
            print(f"⚠️ MediaPipe processing error: {e}")
            # Return safe defaults on error
            return {
                'face_present': False,
                'eyes_open_ratio': 0.0,
                'eyes_closed_duration': 0.0,
                'gaze_direction': "Away",
                'gaze_away_ratio': 1.0,
                'head_pitch': 0.0,
                'head_yaw': 0.0,
                'blink_rate': 0.0,
                'expression': None,
                'landmarks_array': None,
                'frame_shape': (h, w) if 'h' in locals() else (480, 640),
                'in_note_taking_grace_period': False,
                'note_taking_active': False
            }
        
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
            
            # Update note-taking grace period
            in_grace_period = self.update_note_taking_grace_period(
                current_time, self.gaze_label, head_pitch, head_yaw
            )
            
        else:
            # No face detected
            self.eyes_open_ratio = 0.0
            self.eyes_closed_duration = 0.0
            self.expression = None
            self.gaze_label = "Away"
            gaze_away_ratio = 1.0
            head_pitch = 0.0
            head_yaw = 0.0
            in_grace_period = False
            
            # Reset note-taking grace period when face is not present
            if self.note_taking_active:
                self.note_taking_active = False
                self.note_taking_start_time = None
        
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
            'frame_shape': (h, w),
            'in_note_taking_grace_period': in_grace_period,
            'note_taking_active': self.note_taking_active
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
                    face_present=metrics['face_present'],
                    in_note_taking_grace_period=metrics.get('in_note_taking_grace_period', False)
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
                    current_time=current_time,
                    in_note_taking_grace_period=metrics.get('in_note_taking_grace_period', False)
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
            
            # Track focus score for session statistics
            self.focus_score_history.append(new_focus_score)
            
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
        
        # Note-taking status info
        note_taking_info = ""
        if metrics.get('in_note_taking_grace_period', False):
            if self.note_taking_start_time:
                remaining_time = self.note_taking_grace_period - (time.perf_counter() - self.note_taking_start_time)
                note_taking_info = f" 📝{remaining_time:.1f}s"
        elif metrics.get('note_taking_active', False):
            note_taking_info = " 📝Active"
        
        hud_lines = [
            f"Face: {'Yes' if metrics['face_present'] else 'No'}",
            f"Focus Score: {self.current_focus_score:.1f}%{sustained_info}{note_taking_info}",
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
            frame_count = 0
            last_fps_time = time.perf_counter()
            fps_counter = 0
            
            while self.running:
                loop_start_time = time.perf_counter()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("📹 Video stream ended")
                    break
                
                # Skip frame processing if it's taking too long (prevent freeze)
                frame_count += 1
                if frame_count % 2 == 0:  # Process every other frame to reduce load
                    # Yield to other threads occasionally
                    if frame_count % 10 == 0:
                        time.sleep(0.001)  # 1ms break every 10 frames
                    continue
                
                try:
                    # Process frame and get focus metrics with timeout protection
                    metrics = self.process_frame(frame)
                    
                    # Compute and update focus score with enhanced landmarks
                    self.compute_and_update_focus_score(
                        metrics, 
                        landmarks_array=metrics.get('landmarks_array'),
                        frame_shape=metrics.get('frame_shape')
                    )
                except Exception as e:
                    print(f"⚠️ Frame processing error (skipping): {e}")
                    continue
                
                # Draw overlay if video display is enabled
                if self.show_video:
                    try:
                        self.draw_overlay(frame, metrics)
                        cv2.imshow("FocusMind - AI Focus Tracker", frame)
                    except Exception as e:
                        print(f"⚠️ Display error: {e}")
                    
                    # Check for ESC key (non-blocking)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                
                # FPS monitoring and performance check
                fps_counter += 1
                current_time = time.perf_counter()
                if current_time - last_fps_time >= 5.0:  # Every 5 seconds
                    fps = fps_counter / (current_time - last_fps_time)
                    if fps < 5:  # If FPS drops below 5, warn user
                        print(f"⚠️ Performance warning: FPS={fps:.1f} (consider closing other apps)")
                    fps_counter = 0
                    last_fps_time = current_time
                
                # Prevent runaway processing - ensure minimum frame time
                loop_duration = time.perf_counter() - loop_start_time
                min_frame_time = 1.0 / 30.0  # Max 30 FPS to prevent overload
                if loop_duration < min_frame_time:
                    time.sleep(min_frame_time - loop_duration)
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopping face tracking...")
            
            # Calculate and display session statistics
            if self.focus_score_history:
                avg_score = sum(self.focus_score_history) / len(self.focus_score_history)
                session_duration = time.perf_counter() - self.session_start_time
                total_measurements = len(self.focus_score_history)
                min_score = min(self.focus_score_history)
                max_score = max(self.focus_score_history)
                
                print(f"📊 Session Summary:")
                print(f"   ⏱️  Duration: {session_duration/60:.1f} minutes")
                print(f"   📈 Average Focus Score: {avg_score:.1f}")
                print(f"   📊 Total Measurements: {total_measurements}")
                print(f"   🔻 Lowest Score: {min_score:.1f}")
                print(f"   🔺 Highest Score: {max_score:.1f}")
            else:
                print("📊 No focus data collected during session")
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            print("🔄 Attempting to recover...")
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