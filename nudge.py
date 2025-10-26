import os
import sys
import json
import uuid
import random
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print(json.dumps({"error": "OPENAI_API_KEY not found in environment variables"}))
    exit(1)

client = OpenAI(api_key=api_key)

# Create audio directory if it doesn't exist
audio_dir = Path("audio_files")
audio_dir.mkdir(exist_ok=True)

def break_nudge():
    """Generate a motivational break message with voiceover using OpenAI for Pomodoro breaks"""
    try:
        system_prompt = """You are David Goggins giving advice for taking a productive break during Pomodoro sessions.

Guidelines:
- Encourage the user to take a REAL break (not phone/social media)
- Suggest physical activities: stretch, walk, drink water, do push-ups, talk to someone
- Keep it energetic and motivating but focused on recovery
- Remind them that breaks make them stronger and more focused
- Keep it under 30 words


Examples: "Break time! Get up, stretch those muscles, hydrate your body. Real recovery, not phone time. Come back stronger!"
"""

        break_prompts = [
            "Time for a break! What should I do to recharge properly?",
            "Pomodoro session complete! How should I use this break to come back stronger?",
            "Session done! What's the best way to recover and reset?",
            "Time to take a break! How do I make it count?"
        ]
        
        user_prompt = random.choice(break_prompts)
        
        # Generate text
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100,
            temperature=0.8  # Add some randomness for variety
        )
        
        message = response.choices[0].message.content
        
        # Generate audio using OpenAI TTS
        audio_filename = f"break_nudge_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = audio_dir / audio_filename
        
        tts_response = client.audio.speech.create(
            model="tts-1",
            voice="onyx",  # Deep, authoritative voice similar to David Goggins
            input=message,
            speed=1.0
        )
        
        # Save audio file
        tts_response.stream_to_file(audio_path)
        
        # Return as JSON for easy parsing
        result = {
            "success": True,
            "message": message,
            "audio_file": audio_filename,
            "audio_path": str(audio_path),
            "source": "David Goggins Break Coach",
            "nudge_type": "break"
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "nudge_type": "break"
        }
        print(json.dumps(error_result))

def voice_nudge(attention_score=100):
    """Generate a motivational quote with voiceover using OpenAI based on attention score"""
    try:
        # Create dynamic system prompt based on attention score
        if attention_score >= 80:
            intensity = "gentle but firm"
            approach = "Give a supportive nudge to maintain focus. Be encouraging but direct."
            max_words = "30 words"
        elif attention_score >= 60:
            intensity = "moderate"
            approach = "Give a more assertive push to regain focus. Be direct and motivating."
            max_words = "40 words"
        elif attention_score >= 40:
            intensity = "strong"
            approach = "Give an intense wake-up call. Be forceful and demanding."
            max_words = "50 words"
        else:  # attention_score < 40
            intensity = "maximum intensity"
            approach = "Give the most intense, no-nonsense reality check. Be brutal and uncompromising."
            max_words = "60 words"
        
        system_prompt = f"""You are an AI study coach loosely based off David Goggins. The user's attention score is {attention_score}/100.
        
Based on this score, use {intensity} motivation. {approach}

CRITICAL: EVERY message MUST start with "Dude, I see you slacking off! " followed by a space, then continue with the rest of the short motivational message.
ALSO: The message should not reference the user's attention score numerical directly. 

Guidelines:
- If score is 80+: Gentle encouragement to stay on track
- If score is 60-79: Firm reminder to refocus  
- If score is 40-59: Strong wake-up call to get serious
- If score is <40: Maximum intensity reality check

Keep it under {max_words}."""

        # Add variety to prompts to ensure different messages each time
        variety_prompts = [
            f"My attention score has dropped to {attention_score}. I need motivation to get back on track with studying.",
            f"I'm losing focus and my score is at {attention_score}. Hit me with some motivation.",
            f"My concentration is slipping to {attention_score}/100. I need a reality check to refocus.",
            f"Attention score: {attention_score}. I need you to get me back in the zone.",
            f"I'm at {attention_score}/100 focus level. Give me the push I need to study harder."
        ]
        
        user_prompt = random.choice(variety_prompts)
        
        # Generate text
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            temperature=0.8  # Add some randomness for variety
        )
        
        message = response.choices[0].message.content
        
        # Generate audio using OpenAI TTS with the actual motivational quote
        audio_filename = f"motivation_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = audio_dir / audio_filename
        
        tts_response = client.audio.speech.create(
            model="tts-1",
            voice="onyx",  # Deep, authoritative voice similar to David Goggins
            input=message,  # Using the actual motivational quote from GPT-4
            speed=1.0
        )
        
        # Save audio file
        tts_response.stream_to_file(audio_path)
        
        # Return as JSON for easy parsing
        result = {
            "success": True,
            "message": message,
            "audio_file": audio_filename,
            "audio_path": str(audio_path),
            "source": "AI study coach with Voice",
            "nudge_type": "voice"
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "nudge_type": "voice"
        }
        print(json.dumps(error_result))

def notification_nudge(attention_score=100):
    """Send a system notification to get user's attention based on attention score"""
    try:
        # Create dynamic system prompt based on attention score
        if attention_score >= 80:
            tone = "gentle reminder"
            max_words = "15 words"
            examples = "'Stay focused!' or 'Keep it up!'"
        elif attention_score >= 60:
            tone = "firm nudge"
            max_words = "18 words"
            examples = "'Focus up! Stay on track!' or 'Get back to work!'"
        elif attention_score >= 40:
            tone = "strong alert"
            max_words = "20 words"
            examples = "'FOCUS UP! Your dreams won't wait!' or 'Stop scrolling! Get back to work!'"
        else:  # attention_score < 40
            tone = "urgent wake-up call"
            max_words = "25 words"
            examples = "'WAKE UP! You're wasting precious time!' or 'STOP! Your future depends on this moment!'"
        
        system_prompt = f"""You are a motivational study coach loosely based off David Goggins. The user's attention score is {attention_score}/100.
        
Generate a short, punchy notification message (max {max_words}) using a {tone} approach.

Examples for this score level: {examples}

Be direct and motivating but adjust intensity based on how low the attention score is."""

        user_prompt = f"My attention score is {attention_score}. I need a notification to refocus."
        
        # Generate a short, punchy notification message
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=50
        )
        
        message = response.choices[0].message.content
        # Clean the message for system notifications (remove quotes that cause issues)
        clean_message = message.replace('"', '').replace("'", "")
        
        # Send system notification based on OS
        if sys.platform == "darwin":  # macOS
            subprocess.run([
                "osascript", "-e", 
                f'display notification "{clean_message}" with title "FocusMind Nudge 💪" sound name "Glass"'
            ])
        elif sys.platform == "linux":  # Linux
            subprocess.run(["notify-send", "FocusMind Nudge 💪", message])
        elif sys.platform == "win32":  # Windows
            # For Windows, we'll use a simple print for now (can be enhanced with plyer library)
            # print(f"NOTIFICATION: {message}")
            pass
        
        # Return as JSON for easy parsing
        result = {
            "success": True,
            "message": message,
            "source": "AI Notification",
            "nudge_type": "notification",
            "platform": sys.platform
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "nudge_type": "notification"
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    # Check command line arguments to determine which nudge to run
    # Usage: python nudge.py [voice|notification|break|generate_audio] [attention_score|message]
    nudge_type = "voice"  # default
    attention_score = 100  # default
    
    if len(sys.argv) > 1:
        nudge_type = sys.argv[1]
    
    if nudge_type == "generate_audio":
        # For generate_audio, the second argument is the message to convert to audio
        if len(sys.argv) > 2:
            message = sys.argv[2]
            try:
                # Generate audio for the provided message
                audio_filename = f"message_{uuid.uuid4()}.mp3"
                audio_path = audio_dir / audio_filename
                
                # Use OpenAI TTS to generate speech
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="onyx",  # David Goggins-like voice
                    input=message,
                    speed=1.0
                )
                
                # Save audio file
                response.stream_to_file(audio_path)
                
                result = {
                    "success": True,
                    "message": message,
                    "audio_file": audio_filename,
                    "source": "David Goggins AI"
                }
                print(json.dumps(result))
            except Exception as e:
                error_result = {
                    "success": False,
                    "error": str(e)
                }
                print(json.dumps(error_result))
        else:
            print(json.dumps({"error": "Message required for generate_audio command"}))
    else:
        # Original logic for other commands
        if len(sys.argv) > 2:
            try:
                attention_score = int(sys.argv[2])
                # Ensure attention score is within valid range
                attention_score = max(0, min(100, attention_score))
            except ValueError:
                print(json.dumps({"error": "Attention score must be a number between 0 and 100"}))
                exit(1)
        
        if nudge_type == "voice":
            voice_nudge(attention_score)
        elif nudge_type == "notification":
            notification_nudge(attention_score)
        elif nudge_type == "break":
            break_nudge()  # Break nudges don't need attention score
        else:
            print(json.dumps({"error": f"Unknown nudge type: {nudge_type}. Use 'voice', 'notification', 'break', or 'generate_audio'"}))
            exit(1)