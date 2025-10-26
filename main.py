import os
import subprocess
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel
import random

# Import the focus scoring system
from FocusScore import generate_focus_chart_base64, generate_session_stats

# Import the face tracking system
from face_focus_tracker import FaceFocusTracker

# Global attention score variable (shared state)
attention_score = 100

# Global focus score tracking for the current session
focus_score_history = []

# Global face tracking variables
face_tracker = None
tracking_active = False

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="FocusMind API", description="Motivational Study Coach API")

# Create audio directory if it doesn't exist
audio_dir = Path("audio_files")
audio_dir.mkdir(exist_ok=True)

# Mount static files for audio serving
app.mount("/audio", StaticFiles(directory="audio_files"), name="audio")

# Add CORS middleware to allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3003", "http://localhost:3004", "http://localhost:3005", "http://localhost:3006", "http://localhost:3007", "http://localhost:3008"],  # React dev server on multiple ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=api_key)

class MotivationResponse(BaseModel):
    message: str
    attention_score: int

@app.get("/")
async def root():
    return {"message": "FocusMind API is running"}

@app.get("/motivation", response_model=MotivationResponse)
async def get_motivation(reset: bool = False):
    """Get a motivational quote from David Goggins style coach"""
    global attention_score, focus_score_history
    
    # Reset focus history if requested (page refresh), but preserve attention_score from face tracking
    if reset:
        # Only reset focus history, NOT attention_score (preserve real-time face tracking data)
        focus_score_history = []
        # Keep the current attention_score from face tracking
    
    # Add initial score to history if it's the first entry
    if not focus_score_history:
        current_time = datetime.now().strftime("%H:%M:%S")
        focus_score_history.append({
            "timestamp": current_time,
            "focus_score": attention_score
        })
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a study coach loosely inspired by David Goggins. Give intense, motivational study advice in a strictly PG version of his style. (No swearing). Keep it under 30 words."},
                {"role": "user", "content": "Give me motivation to study hard"}
            ],
            max_tokens=150
        )
        
        message = response.choices[0].message.content
        
        return MotivationResponse(message=message, attention_score=attention_score)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating motivation: {str(e)}")

@app.get("/attention-score")
async def get_attention_score():
    """Get current attention score"""
    global attention_score
    return {"attention_score": attention_score}

@app.post("/decrease-attention")
async def decrease_attention():
    """Decrease attention score by 15"""
    global attention_score, focus_score_history
    attention_score = max(0, attention_score - 15)  # Don't go below 0
    
    # Add to focus score history with timestamp
    current_time = datetime.now().strftime("%H:%M:%S")
    focus_score_history.append({
        "timestamp": current_time,
        "focus_score": attention_score
    })
    
    return {"attention_score": attention_score, "message": f"Attention score decreased to {attention_score}"}

@app.post("/get-focus-chart")
async def get_focus_chart():
    """Generate and return focus chart for the current session"""
    global focus_score_history
    
    try:
        print(f"📊 Chart request received. Focus history length: {len(focus_score_history)}")
        
        if not focus_score_history:
            print("⚠️ No focus data available for chart generation")
            return {
                "success": False,
                "error": "No focus data available for chart generation"
            }
        
        print(f"📈 Generating chart with {len(focus_score_history)} data points")
        
        # Generate chart
        png_path, chart_b64_bytes = generate_focus_chart_base64(focus_score_history)
        
        # Generate session stats
        session_stats = generate_session_stats(focus_score_history)
        
        print(f"✅ Chart generated successfully: {png_path}")
        
        return {
            "success": True,
            "chart_base64": chart_b64_bytes.decode("ascii"),
            "session_stats": session_stats,
            "png_filename": png_path,
            "data_points": len(focus_score_history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating focus chart: {str(e)}")

@app.get("/focus-history-status")
async def get_focus_history_status():
    """Debug endpoint to check focus history status"""
    global focus_score_history
    
    recent_entries = focus_score_history[-5:] if len(focus_score_history) > 5 else focus_score_history
    
    return {
        "total_entries": len(focus_score_history),
        "recent_entries": recent_entries,
        "has_data": len(focus_score_history) > 0
    }

@app.post("/reset-focus-session")
async def reset_focus_session():
    """Reset focus score history for a new session"""
    global focus_score_history
    focus_score_history = []
    return {"success": True, "message": "Focus session reset"}

@app.get("/focus-session-stats")
async def get_focus_session_stats():
    """Get current session statistics"""
    global focus_score_history
    
    if not focus_score_history:
        return {"data_points": 0, "session_active": False}
    
    stats = generate_session_stats(focus_score_history)
    return {
        "data_points": len(focus_score_history),
        "session_active": True,
        "stats": stats
    }

@app.post("/get-voice-nudge")
async def get_voice_nudge():
    """Get a motivational quote with voiceover by running nudge.py script with voice argument"""
    global attention_score
    try:
        # Run nudge.py script with 'voice' argument and current attention score
        result = subprocess.run(
            ["python3", "nudge.py", "voice", str(attention_score)], 
            capture_output=True, 
            text=True, 
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            # Parse JSON output from nudge.py
            output_data = json.loads(result.stdout.strip())
            if output_data.get("success"):
                audio_filename = output_data.get("audio_file")
                audio_url = f"/audio/{audio_filename}" if audio_filename else None
                
                return {
                    "success": True,
                    "message": output_data["message"],
                    "audio_url": audio_url,
                    "audio_file": audio_filename,
                    "source": output_data.get("source", "David Goggins AI"),
                    "nudge_type": "voice",
                    "attention_score": attention_score  # Include current attention score
                }
            else:
                raise HTTPException(status_code=500, detail=f"Voice nudge script error: {output_data.get('error', 'Unknown error')}")
        else:
            raise HTTPException(status_code=500, detail=f"Script execution failed: {result.stderr}")
            
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse script output: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running voice nudge script: {str(e)}")

class VoiceAudioRequest(BaseModel):
    message: str

@app.post("/generate-voice-audio")
async def generate_voice_audio(request: VoiceAudioRequest):
    """Generate voice audio for a specific message without getting a new quote"""
    try:
        # Use nudge.py to generate audio for the provided message
        result = subprocess.run(
            ["python3", "nudge.py", "generate_audio", request.message], 
            capture_output=True, 
            text=True, 
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            # Parse JSON output from nudge.py
            output_data = json.loads(result.stdout.strip())
            if output_data.get("success"):
                audio_filename = output_data.get("audio_file")
                audio_url = f"/audio/{audio_filename}" if audio_filename else None
                
                return {
                    "success": True,
                    "message": request.message,
                    "audio_url": audio_url,
                    "audio_file": audio_filename,
                    "source": "David Goggins AI"
                }
            else:
                raise HTTPException(status_code=500, detail=f"Audio generation error: {output_data.get('error', 'Unknown error')}")
        else:
            raise HTTPException(status_code=500, detail=f"Script execution failed: {result.stderr}")
            
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse script output: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating voice audio: {str(e)}")

@app.post("/get-notification-nudge")
async def get_notification_nudge():
    """Send a system notification by running nudge.py script with notification argument"""
    global attention_score
    try:
        # Run nudge.py script with 'notification' argument and current attention score
        result = subprocess.run(
            ["python3", "nudge.py", "notification", str(attention_score)], 
            capture_output=True, 
            text=True, 
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        if result.returncode == 0:
            # Parse JSON output from nudge.py
            output_data = json.loads(result.stdout.strip())
            if output_data.get("success"):
                return {
                    "success": True,
                    "message": output_data["message"],
                    "source": output_data.get("source", "AI study coach"),
                    "nudge_type": "notification",
                    "platform": output_data.get("platform", "unknown")
                }
            else:
                raise HTTPException(status_code=500, detail=f"Notification nudge script error: {output_data.get('error', 'Unknown error')}")
        else:
            raise HTTPException(status_code=500, detail=f"Script execution failed: {result.stderr}")
            
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse script output: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running notification nudge script: {str(e)}")

@app.post("/get-break-nudge")
async def get_break_nudge():
    """Get a motivational break message for Pomodoro breaks"""
    try:
        # Run nudge.py script with 'break' argument (no attention score needed for breaks)
        result = subprocess.run(
            ["python3", "nudge.py", "break"], 
            capture_output=True, 
            text=True, 
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            # Parse JSON output from nudge.py
            output_data = json.loads(result.stdout.strip())
            if output_data.get("success"):
                audio_filename = output_data.get("audio_file")
                audio_url = f"/audio/{audio_filename}" if audio_filename else None
                
                return {
                    "success": True,
                    "message": output_data["message"],
                    "audio_url": audio_url,
                    "audio_file": audio_filename,
                    "source": output_data.get("source", "David Goggins Break Coach"),
                    "nudge_type": "break"
                }
            else:
                raise HTTPException(status_code=500, detail=f"Break nudge script error: {output_data.get('error', 'Unknown error')}")
        else:
            raise HTTPException(status_code=500, detail=f"Script execution failed: {result.stderr}")
            
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse script output: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running break nudge script: {str(e)}")

# Keep the old endpoint for backward compatibility
@app.post("/get-nudge-quote")
async def get_nudge_quote():
    """Get a motivational quote with voiceover (backward compatibility - calls voice nudge)"""
    return await get_voice_nudge()

# Face Tracking Integration Endpoints
class FocusScoreUpdate(BaseModel):
    focus_score: float

class AutoMotivationTrigger(BaseModel):
    threshold: int
    focus_score: float

@app.post("/update-focus-score")
async def update_focus_score(request: FocusScoreUpdate):
    """Update the attention score from face tracking system"""
    global attention_score, focus_score_history
    
    # Update global attention score
    attention_score = max(0, min(100, request.focus_score))
    
    # Add to focus score history for analytics
    focus_score_history.append({
        "timestamp": datetime.now(),
        "score": attention_score
    })
    
    # Keep only last 1000 entries to prevent memory issues
    if len(focus_score_history) > 1000:
        focus_score_history = focus_score_history[-1000:]
    
    return {
        "success": True,
        "updated_score": attention_score,
        "message": "Focus score updated successfully"
    }

@app.post("/trigger-auto-motivation")
async def trigger_auto_motivation(request: AutoMotivationTrigger):
    """Trigger automatic motivational quote when focus drops below thresholds"""
    global attention_score
    
    try:
        print(f"🚨 Auto-motivation triggered! Focus dropped below {request.threshold}% (current: {request.focus_score:.1f}%)")
        
        # Run nudge.py script with 'voice' argument and current attention score
        result = subprocess.run(
            ["python3", "nudge.py", "voice", str(int(request.focus_score))], 
            capture_output=True, 
            text=True, 
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            # Parse JSON output from nudge.py
            output_data = json.loads(result.stdout.strip())
            if output_data.get("success"):
                audio_filename = output_data.get("audio_file")
                audio_url = f"/audio/{audio_filename}" if audio_filename else None
                
                return {
                    "success": True,
                    "message": output_data["message"],
                    "audio_url": audio_url,
                    "audio_file": audio_filename,
                    "source": output_data.get("source", "David Goggins AI"),
                    "nudge_type": "auto_voice",
                    "threshold": request.threshold,
                    "focus_score": request.focus_score
                }
            else:
                raise HTTPException(status_code=500, detail=f"Auto-motivation script error: {output_data.get('error', 'Unknown error')}")
        else:
            raise HTTPException(status_code=500, detail=f"Script execution failed: {result.stderr}")
            
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse script output: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running auto-motivation script: {str(e)}")

@app.get("/face-tracking-status")
async def get_face_tracking_status():
    """Get the current status of face tracking."""
    global face_tracker, tracking_active, focus_score_history, attention_score
    
    print(f"🔍 DEBUG BACKEND: attention_score = {attention_score}, focus_history_length = {len(focus_score_history)}")
    
    if face_tracker is None:
        print(f"🔍 DEBUG BACKEND: Face tracker is None, returning attention_score: {attention_score}")
        return {
            "active": False,
            "attention_score": attention_score,
            "focus_history_length": len(focus_score_history),
            "last_update": None,
            "tracking_active": False
        }
    
    print(f"🔍 DEBUG BACKEND: Face tracker exists, returning attention_score: {attention_score}")
    return {
        "active": tracking_active,
        "attention_score": attention_score,
        "focus_history_length": len(focus_score_history),
        "last_update": str(focus_score_history[-1]["timestamp"]) if focus_score_history else None,
        "tracking_active": tracking_active
    }

@app.post("/start-face-tracking")
async def start_face_tracking():
    """Start the face tracking system (placeholder for future implementation)"""
    # This could be enhanced to actually start the face tracking process
    # For now, it's a placeholder that the frontend can call
    return {
        "success": True,
        "message": "Face tracking start signal sent. Please run face_focus_tracker.py manually.",
        "command": "python3 face_focus_tracker.py --source 0"
    }

@app.post("/stop-face-tracking")
async def stop_face_tracking():
    """Stop the face tracking system (placeholder for future implementation)"""
    # This could be enhanced to actually stop the face tracking process
    return {
        "success": True,
        "message": "Face tracking stop signal sent."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
