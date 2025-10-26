"""
Eye calibration system to personalize focus tracking for different eye shapes.
"""
import cv2
import numpy as np
import json
import time
from face_track import compute_average_ear, normalize_ear
import mediapipe as mp

def calibrate_eye_baseline(cap, face_mesh, duration=10):
    """
    Calibrate user's baseline eye openness for personalized scoring.
    
    Args:
        cap: cv2.VideoCapture object
        face_mesh: MediaPipe FaceMesh object
        duration: calibration duration in seconds
    
    Returns:
        dict with calibration values
    """
    print("🎯 Eye Calibration Process")
    print("=" * 50)
    print("This will calibrate the system to your natural eye shape.")
    print("Please follow the instructions carefully.\n")
    
    # Step 1: Normal eye state
    print("STEP 1: Look directly at the camera with your eyes in a comfortable, natural state.")
    print("Keep your eyes relaxed and normal - don't force them wide or squint.")
    input("Press ENTER when ready...")
    
    normal_ears = []
    start_time = time.time()
    print(f"Recording normal eye state for {duration} seconds...")
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            
            ear = compute_average_ear(landmarks_array)
            normal_ears.append(ear)
            
            # Visual feedback
            remaining = duration - (time.time() - start_time)
            cv2.putText(frame, f"Normal Eyes: {remaining:.1f}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Eye Calibration", frame)
            cv2.waitKey(1)
    
    # Step 2: Wide eyes state
    print("\nSTEP 2: Open your eyes wider than normal - but don't strain them!")
    print("This represents your 'fully alert' state.")
    input("Press ENTER when ready...")
    
    wide_ears = []
    start_time = time.time()
    print(f"Recording wide eye state for {duration} seconds...")
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            
            ear = compute_average_ear(landmarks_array)
            wide_ears.append(ear)
            
            # Visual feedback
            remaining = duration - (time.time() - start_time)
            cv2.putText(frame, f"Wide Eyes: {remaining:.1f}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Eye Calibration", frame)
            cv2.waitKey(1)
    
    # Step 3: Nearly closed eyes
    print("\nSTEP 3: Squint or nearly close your eyes (but still able to see).")
    print("This represents your 'tired/unfocused' state.")
    input("Press ENTER when ready...")
    
    squint_ears = []
    start_time = time.time()
    print(f"Recording squinted eye state for {duration//2} seconds...")
    
    while time.time() - start_time < duration//2:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            
            ear = compute_average_ear(landmarks_array)
            squint_ears.append(ear)
            
            # Visual feedback
            remaining = (duration//2) - (time.time() - start_time)
            cv2.putText(frame, f"Squinted: {remaining:.1f}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
            cv2.imshow("Eye Calibration", frame)
            cv2.waitKey(1)
    
    cv2.destroyAllWindows()
    
    # Calculate calibration values
    normal_ear = np.mean(normal_ears) if normal_ears else 0.25
    wide_ear = np.mean(wide_ears) if wide_ears else 0.35
    squint_ear = np.mean(squint_ears) if squint_ears else 0.15
    
    # Ensure logical ordering
    if wide_ear <= normal_ear:
        wide_ear = normal_ear * 1.2
    if squint_ear >= normal_ear:
        squint_ear = normal_ear * 0.7
    
    calibration = {
        'normal_ear': float(normal_ear),
        'wide_ear': float(wide_ear),
        'squint_ear': float(squint_ear),
        'ear_range': float(wide_ear - squint_ear),
        'normal_threshold': float(normal_ear * 0.8),  # 80% of normal for "focused"
        'alert_threshold': float(normal_ear * 1.1),   # 110% of normal for "very focused"
        'timestamp': time.time()
    }
    
    print("\n✅ Eye calibration complete!")
    print(f"Normal EAR: {normal_ear:.3f}")
    print(f"Wide EAR: {wide_ear:.3f}")
    print(f"Squinted EAR: {squint_ear:.3f}")
    print(f"Range: {calibration['ear_range']:.3f}")
    
    return calibration

def save_eye_calibration(calibration, filename="eye_calibration.json"):
    """Save eye calibration to file"""
    with open(filename, 'w') as f:
        json.dump(calibration, f, indent=2)
    print(f"💾 Eye calibration saved to {filename}")

def load_eye_calibration(filename="eye_calibration.json"):
    """Load eye calibration from file"""
    try:
        with open(filename, 'r') as f:
            calibration = json.load(f)
        print(f"📂 Eye calibration loaded from {filename}")
        return calibration
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"⚠️ No eye calibration found at {filename}")
        return None

def normalize_ear_personalized(ear, calibration):
    """
    Normalize EAR based on personal calibration - more forgiving version.
    
    Args:
        ear: raw eye aspect ratio
        calibration: personal calibration dict
    
    Returns:
        normalized score 0-1 where 1 = fully alert, 0 = eyes closed
    """
    if not calibration:
        # Fallback to generic normalization
        return normalize_ear(ear)
    
    normal_ear = calibration['normal_ear']
    wide_ear = calibration['wide_ear']
    squint_ear = calibration['squint_ear']
    
    # Much more forgiving mapping - give benefit of the doubt
    # Expand the range below squint for true "eyes closed"
    eyes_closed_threshold = squint_ear * 0.7  # Much lower threshold for "eyes closed"
    
    if ear <= eyes_closed_threshold:
        return 0.0  # Truly closed eyes
    elif ear >= wide_ear:
        return 1.0  # Fully alert
    elif ear >= normal_ear:
        # Above normal - map to 0.85-1.0 range (was 0.8-1.0)
        ratio = (ear - normal_ear) / (wide_ear - normal_ear)
        return 0.85 + (ratio * 0.15)  # 0.85 to 1.0
    else:
        # Below normal but above closed threshold - map to 0.4-0.85 range (much more forgiving)
        ratio = (ear - eyes_closed_threshold) / (normal_ear - eyes_closed_threshold)
        return 0.4 + (ratio * 0.45)  # 0.4 to 0.85 (much higher minimum)

if __name__ == "__main__":
    # Standalone calibration tool
    print("🎯 FocusMind Eye Calibration Tool")
    print("This will customize the system to your eye shape.\n")
    
    # Initialize camera and MediaPipe
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        exit(1)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    
    try:
        calibration = calibrate_eye_baseline(cap, face_mesh)
        save_eye_calibration(calibration)
        print("\n🎉 Calibration complete! Your personalized settings have been saved.")
        print("The focus tracker will now use your custom eye calibration.")
    finally:
        cap.release()
        cv2.destroyAllWindows()