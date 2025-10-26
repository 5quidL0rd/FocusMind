import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import Dashboard from './components/Dashboard';
import Header from './components/Header';

interface MotivationData {
  message: string;
  attention_score: number;
}

interface AttentionResponse {
  attention_score: number;
  message: string;
}

interface NudgeResponse {
  success: boolean;
  message: string;
  audio_url?: string;
  audio_file?: string;
  source: string;
  nudge_type?: string;
  platform?: string;
  attention_score?: number;  // Add attention score to the response
}

interface FocusChartResponse {
  success: boolean;
  chart_base64?: string;
  session_stats?: any;
  png_filename?: string;
  data_points?: number;
  error?: string;
}

interface FaceTrackingResponse {
  success: boolean;
  message: string;
  command?: string;
}

interface AutoMotivation {
  audio_url: string | null;
  message: string | null;
  timestamp: string | null;
  played: boolean;
}

interface FaceTrackingStatus {
  attention_score: number;
  focus_history_length: number;
  last_update: string | null;
  tracking_active: boolean;
  auto_motivation: AutoMotivation;
}

function App() {
  // Use relative endpoints; CRA dev server proxy will route to backend

  const [motivationData, setMotivationData] = useState<MotivationData | null>(null);
  const [loading, setLoading] = useState(false);
  const [nudgeExecuted, setNudgeExecuted] = useState(false);
  const [notificationSent, setNotificationSent] = useState(false);
  const [notificationPermission, setNotificationPermission] = useState<NotificationPermission>('default');

  // Pomodoro Timer State
  const [pomodoroTime, setPomodoroTime] = useState(30); // 30 seconds for quick testing
  const [pomodoroRunning, setPomodoroRunning] = useState(false);
  const [pomodoroSessions, setPomodoroSessions] = useState(0);
  const [isBreakTime, setIsBreakTime] = useState(false); // Track if we're in break mode
  const sessionIncrementedRef = useRef(false); // Use ref to prevent double increment
  const initialMotivationFetchedRef = useRef(false); // Use ref to prevent multiple initial fetches
  
  // Focus chart state
  const [showFocusChart, setShowFocusChart] = useState(false);
  const [focusChartData, setFocusChartData] = useState<FocusChartResponse | null>(null);
  
  // Face tracking state
  const [faceTrackingActive, setFaceTrackingActive] = useState(false);
  const [lastFaceTrackingUpdate, setLastFaceTrackingUpdate] = useState<string | null>(null);
  
  // Global audio management to prevent multiple audio tracks playing simultaneously
  const [currentAudio, setCurrentAudio] = useState<HTMLAudioElement | null>(null);

  // Helper function to stop any existing audio and play new audio
  const playAudio = async (audioUrl: string, audioType: string = 'voice') => {
    // Stop any currently playing audio
    if (currentAudio) {
      console.log('🔇 Stopping existing audio before playing new audio');
      currentAudio.pause();
      currentAudio.currentTime = 0;
      setCurrentAudio(null);
    }

    console.log(`🎵 Creating new audio element for ${audioType} with URL:`, audioUrl);
    const audio = new Audio(audioUrl);
    audio.volume = 0.8;
    setCurrentAudio(audio);

    // Add event listeners
    audio.onloadstart = () => console.log(`🎵 Loading ${audioType} audio...`);
    audio.oncanplay = () => console.log(`🎵 ${audioType} audio ready to play`);
    audio.onplay = () => console.log(`🎵 ${audioType} audio started playing`);
    audio.onended = () => {
      console.log(`🎵 ${audioType} audio finished playing`);
      setCurrentAudio(null);
      if (audioType === 'break') {
        setIsBreakTime(false);
        console.log('🛌 Exiting break time mode');
      }
      setTimeout(() => setNudgeExecuted(false), 3000);
    };
    audio.onerror = (e) => {
      console.error(`🚫 ${audioType} audio playback error:`, e);
      console.error(`🚫 Failed audio URL:`, audioUrl);
      setCurrentAudio(null);
      if (audioType === 'break') {
        setIsBreakTime(false);
      }
    };

    try {
      console.log(`🎵 Attempting to play ${audioType} audio...`);
      await audio.play();
      console.log(`✅ ${audioType} audio playing successfully!`);
      return true;
    } catch (playError) {
      console.error(`❌ ${audioType} audio play failed:`, playError);
      console.error(`❌ Failed audio URL:`, audioUrl);
      setCurrentAudio(null);
      if (audioType === 'break') {
        setIsBreakTime(false);
      }
      setTimeout(() => setNudgeExecuted(false), 5000);
      return false;
    }
  };

  // Request notification permission on app load
  useEffect(() => {
    if ('Notification' in window) {
      setNotificationPermission(Notification.permission);
      
      if (Notification.permission === 'default') {
        Notification.requestPermission().then((permission) => {
          setNotificationPermission(permission);
          console.log('🔔 Notification permission:', permission);
        });
      }
    } else {
      console.warn('🚫 This browser does not support notifications');
    }
  }, []);

  // Function to show browser notification
  const showBrowserNotification = (title: string, message: string, icon?: string) => {
    console.log('🔔 Attempting to show notification...');
    console.log('📋 Title:', title);
    console.log('📋 Message:', message);
    console.log('🌐 Notification support:', 'Notification' in window);
    console.log('🔐 Permission status:', Notification.permission);
    
    if (!('Notification' in window)) {
      console.error('🚫 Browser does not support notifications');
      alert('Your browser does not support notifications');
      return null;
    }

    if (Notification.permission === 'denied') {
      console.error('🚫 Notifications are blocked. Please enable them in browser settings.');
      alert('Notifications are blocked. Please enable them in your browser settings:\n\n1. Click the lock icon in the address bar\n2. Set Notifications to "Allow"');
      return null;
    }

    if (Notification.permission === 'default') {
      console.log('🔔 Requesting notification permission...');
      Notification.requestPermission().then((permission) => {
        console.log('🔐 Permission result:', permission);
        setNotificationPermission(permission);
        if (permission === 'granted') {
          // Retry showing notification after permission granted
          showBrowserNotification(title, message, icon);
        }
      });
      return null;
    }

    if (Notification.permission === 'granted') {
      console.log('✅ Permission granted, creating notification...');
      try {
        const notification = new Notification(title, {
          body: message,
          icon: icon || '/favicon.ico',
          badge: '/favicon.ico',
          tag: 'focusmind-nudge',
          requireInteraction: false, // Changed to false for better compatibility
          silent: false
        });

        console.log('✅ Notification created successfully!');

        // Auto-close after 8 seconds
        setTimeout(() => {
          notification.close();
          console.log('🔔 Notification auto-closed');
        }, 8000);

        // Handle notification events
        notification.onshow = () => {
          console.log('🔔 Notification is now visible');
        };

        notification.onclick = () => {
          console.log('🔔 Notification clicked');
          window.focus();
          notification.close();
        };

        notification.onerror = (error) => {
          console.error('🚫 Notification error:', error);
        };

        notification.onclose = () => {
          console.log('🔔 Notification closed');
        };

        return notification;
      } catch (error) {
        console.error('🚫 Error creating notification:', error);
        const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
        alert('Error creating notification: ' + errorMessage);
        return null;
      }
    }

    console.warn('🚫 Unknown permission state:', Notification.permission);
    return null;
  };

  const fetchMotivation = async (reset: boolean = false) => {
    setLoading(true);
    try {
      const url = reset 
        ? `/motivation?reset=true`
        : `/motivation`;
      
      console.log(`📡 Fetching motivation from: ${url} (reset: ${reset})`);
      const response = await axios.get<MotivationData>(url);
      setMotivationData(response.data);
      
      if (reset) {
        console.log('🎯 Initial motivation loaded (reset=true):', response.data.message);
        console.log('📊 Initial attention score:', response.data.attention_score);
      } else {
        console.log('📋 Updated motivation loaded:', response.data.message);
      }
    } catch (error) {
      console.error('Error fetching motivation:', error);
    } finally {
      setLoading(false);
    }
  };

  const decreaseAttention = async () => {
    setLoading(true);
    try {
  const response = await axios.post<AttentionResponse>(`/decrease-attention`);
      const newScore = response.data.attention_score;
      
      console.log(`🔢 Attention score changed: ${motivationData?.attention_score} → ${newScore}`);
      
      // Update the attention score in our state
      if (motivationData) {
        setMotivationData({
          ...motivationData,
          attention_score: newScore
        });
        
        // ONLY trigger automatic voice nudge if we're NOT in break time
        if (!isBreakTime) {
          console.log('🚨 Attention score dropped! Automatically getting new motivational voice nudge...');
          // Trigger voice nudge after a short delay to show the score change first
          setTimeout(() => {
            getVoiceNudge();
          }, 500);
        } else {
          console.log('🛌 In break time - skipping automatic voice nudge');
        }
      }
    } catch (error) {
      console.error('Error decreasing attention:', error);
    } finally {
      setLoading(false);
    }
  };

  const getVoiceNudge = async () => {
    setLoading(true);
    setNudgeExecuted(false); // Reset indicator
    
    try {
      console.log('🚀 Calling voice nudge...');
  const response = await axios.post<NudgeResponse>(`/get-voice-nudge`);
      
      if (response.data.success && motivationData) {
        // Set visual indicator that nudge script executed
        setNudgeExecuted(true);
        
        // ONLY update message if we're NOT in break time to prevent flickering
        if (!isBreakTime) {
          // Update both message and attention score from the response
          setMotivationData({
            ...motivationData,
            message: response.data.message,
            attention_score: response.data.attention_score ?? motivationData.attention_score  // Use response score if available, fallback to current
          });
          console.log('💪 Updated message with new David Goggins voice quote!');
        } else {
          console.log('🛌 In break mode - voice nudge triggered but not updating message to prevent flickering');
        }
        
        console.log('🎯 Voice nudge executed successfully!');
        
        // Play audio if available
        if (response.data.audio_url) {
          const audioUrl = `${response.data.audio_url}`;
          console.log('🔊 Playing voiceover:', audioUrl);
          
          const success = await playAudio(audioUrl, 'voice');
          
          if (!success) {
            // Keep indicator visible even if audio fails
            setTimeout(() => setNudgeExecuted(false), 5000);
          }
        } else {
          // No audio, hide indicator after 3 seconds
          setTimeout(() => setNudgeExecuted(false), 3000);
        }
      }
    } catch (error) {
      console.error('Error getting voice nudge:', error);
      setNudgeExecuted(false);
    } finally {
      setLoading(false);
    }
  };

  const readCurrentMessage = async () => {
    if (!motivationData?.message) {
      console.log('❌ No message to read');
      return;
    }

    setLoading(true);
    setNudgeExecuted(false);
    
    try {
      console.log('🎤 Reading current message aloud:', motivationData.message);
      
      // Call backend to generate audio for the current message
      const response = await axios.post<{
        success: boolean;
        message: string;
        audio_url?: string;
        audio_file?: string;
        source: string;
  }>(`/generate-voice-audio`, {
        message: motivationData.message
      });
      
      if (response.data.success && response.data.audio_url) {
        setNudgeExecuted(true);
        
  const audioUrl = `${response.data.audio_url}`;
        console.log('🔊 Playing current message audio:', audioUrl);
        
        const success = await playAudio(audioUrl, 'voice');
        
        if (!success) {
          setTimeout(() => setNudgeExecuted(false), 5000);
        }
      }
    } catch (error) {
      console.error('Error reading current message:', error);
      setNudgeExecuted(false);
    } finally {
      setLoading(false);
    }
  };

  const getNotificationNudge = async () => {
    setLoading(true);
    setNotificationSent(false); // Reset indicator
    
    try {
      console.log('🔔 Generating browser notification nudge...');
      
      // Generate motivational message using our backend
  const response = await axios.post<NudgeResponse>(`/get-notification-nudge`);
      
      if (response.data.success) {
        // Set visual indicator that notification was sent
        setNotificationSent(true);
        
        console.log('📢 Notification nudge generated successfully!');
        console.log('💪 Message:', response.data.message);
        
        // Show browser notification
        const cleanMessage = response.data.message.replace(/"/g, ''); // Remove quotes from message
        const notification = showBrowserNotification(
          '💪 FocusMind Nudge',
          cleanMessage,
          '/favicon.ico'
        );
        
        if (notification) {
          console.log('🔔 Browser notification shown successfully!');
        } else {
          console.warn('⚠️ Could not show browser notification - check permissions');
        }
        
        // Hide indicator after 4 seconds
        setTimeout(() => setNotificationSent(false), 4000);
      }
    } catch (error) {
      console.error('Error generating notification nudge:', error);
      
      // Fallback: Show a generic notification even if API fails
      const fallbackMessage = "Time to refocus! Get back to your studies and crush those goals! 💪";
      const notification = showBrowserNotification(
        '💪 FocusMind Nudge',
        fallbackMessage
      );
      
      if (notification) {
        setNotificationSent(true);
        setTimeout(() => setNotificationSent(false), 4000);
      } else {
        setNotificationSent(false);
      }
    } finally {
      setLoading(false);
    }
  };

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const getBreakNudge = async () => {
    setLoading(true);
    setNudgeExecuted(false); // Reset indicator
    
    try {
      console.log('🍅 Pomodoro break time! Getting break nudge...');
      
      // Set break time mode to prevent other nudges from interfering
      setIsBreakTime(true);
      
  const response = await axios.post<NudgeResponse>(`/get-break-nudge`);
      
      console.log('🔍 Break nudge response:', response.data); // Debug log
      
      if (response.data.success) {
        // Set visual indicator that nudge script executed
        setNudgeExecuted(true);
        
        // ALWAYS update the message with the break nudge (this should show the break message)
        if (motivationData) {
          setMotivationData({
            ...motivationData,
            message: response.data.message
          });
        }
        
        console.log('🛌 Got David Goggins break advice:', response.data.message);
        console.log('🎯 Break nudge executed successfully!');
        
        // FORCE audio playback for break nudges
        if (response.data.audio_url) {
          const audioUrl = `${response.data.audio_url}`;
          console.log('🔊 FORCING break voiceover playback:', audioUrl);
          
          // Stop any existing audio first
          if (currentAudio) {
            currentAudio.pause();
            currentAudio.currentTime = 0;
            setCurrentAudio(null);
          }
          
          // Create and play break audio with higher priority
          const breakAudio = new Audio(audioUrl);
          breakAudio.volume = 0.9; // Slightly higher volume for break nudges
          setCurrentAudio(breakAudio);
          
          breakAudio.onloadstart = () => console.log('🎵 Loading BREAK audio...');
          breakAudio.oncanplay = () => console.log('🎵 BREAK audio ready to play');
          breakAudio.onplay = () => console.log('🎵 ✅ BREAK AUDIO STARTED PLAYING!');
          breakAudio.onended = () => {
            console.log('🎵 ✅ BREAK AUDIO FINISHED PLAYING');
            setCurrentAudio(null);
            setIsBreakTime(false);
            console.log('🛌 Exiting break time mode');
            setTimeout(() => setNudgeExecuted(false), 3000);
            
            // Show focus chart after break audio finishes
            setTimeout(() => {
              getFocusChart();
            }, 1000);
          };
          breakAudio.onerror = (e) => {
            console.error('🚫 BREAK audio playback error:', e);
            console.error('🚫 Failed break audio URL:', audioUrl);
            setCurrentAudio(null);
            setIsBreakTime(false);
          };
          
          try {
            await breakAudio.play();
            console.log('✅ BREAK AUDIO PLAYING SUCCESSFULLY!');
          } catch (playError) {
            console.error('❌ BREAK audio play failed:', playError);
            console.error('❌ Failed break audio URL:', audioUrl);
            setIsBreakTime(false);
            setTimeout(() => setNudgeExecuted(false), 5000);
            
            // Show focus chart even if audio fails
            setTimeout(() => {
              getFocusChart();
            }, 1000);
          }
        } else {
          console.warn('⚠️ No audio URL provided in break nudge response');
          setIsBreakTime(false);
          setTimeout(() => setNudgeExecuted(false), 3000);
          
          // Show focus chart even without audio
          setTimeout(() => {
            getFocusChart();
          }, 1000);
        }
      } else {
        console.error('❌ Break nudge request failed:', response.data);
        setIsBreakTime(false);
      }
    } catch (error) {
      console.error('❌ Error getting break nudge:', error);
      setNudgeExecuted(false);
      setIsBreakTime(false);
    } finally {
      setLoading(false);
    }
  };

  const getFocusChart = async () => {
    try {
      console.log('📊 Fetching focus chart for completed session...');
      
  const response = await axios.post<FocusChartResponse>(`/get-focus-chart`);
      
      if (response.data.success) {
        console.log('📈 Focus chart generated successfully!');
        console.log('📊 Session stats:', response.data.session_stats);
        
        setFocusChartData(response.data);
        setShowFocusChart(true);
      } else {
        console.warn('⚠️ Focus chart generation failed:', response.data.error);
      }
    } catch (error) {
      console.error('❌ Error fetching focus chart:', error);
    }
  };

  const closeFocusChart = () => {
    setShowFocusChart(false);
    setFocusChartData(null);
  };

  // Pomodoro Timer Functions
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const startPomodoro = () => setPomodoroRunning(true);
  const pausePomodoro = () => setPomodoroRunning(false);
  const resetPomodoro = () => {
    setPomodoroRunning(false);
    setPomodoroTime(30); // Reset to 30 seconds for quick testing
    setIsBreakTime(false); // Exit break mode when resetting
    sessionIncrementedRef.current = false; // Reset session increment flag
    // Stop face tracking when resetting Pomodoro
    if (faceTrackingActive) {
      stopFaceTracking();
    }
  };

  // Face Tracking Functions
  const startFaceTracking = async () => {
    try {
      console.log('📹 Starting face tracking...');
  const response = await axios.post<FaceTrackingResponse>(`/start-face-tracking`);
      
      if (response.data.success) {
        setFaceTrackingActive(true);
        console.log('✅ Face tracking started successfully');
        console.log('💡 Command to run manually:', response.data.command);
        
        // Show notification to user about manual command
        alert(`Face tracking enabled! Please run this command in a new terminal:\n\n${response.data.command}`);
      }
    } catch (error) {
      console.error('❌ Error starting face tracking:', error);
      alert('Failed to start face tracking. Make sure the backend is running.');
    }
  };

  const stopFaceTracking = async () => {
    try {
      console.log('📹 Stopping face tracking...');
  const response = await axios.post<FaceTrackingResponse>(`/stop-face-tracking`);
      
      if (response.data.success) {
        setFaceTrackingActive(false);
        console.log('✅ Face tracking stopped successfully');
      }
    } catch (error) {
      console.error('❌ Error stopping face tracking:', error);
    }
  };

  const checkFaceTrackingStatus = async () => {
    console.log('🔄 DEBUG FRONTEND: checkFaceTrackingStatus() called at', new Date().toLocaleTimeString());
    try {
  const response = await axios.get<FaceTrackingStatus>(`/face-tracking-status`);
      console.log('🔍 DEBUG FRONTEND: Received from backend:', response.data);
      console.log('🔍 DEBUG FRONTEND: Current motivationData.attention_score:', motivationData?.attention_score);
      
      setLastFaceTrackingUpdate(response.data.last_update);
      
      // Check for auto-motivation audio that hasn't been played yet
      if (response.data.auto_motivation && !response.data.auto_motivation.played && response.data.auto_motivation.audio_url) {
        console.log('🎤 DEBUG FRONTEND: Auto-motivation audio detected!', response.data.auto_motivation);
        
        // Play the audio
  const audioUrl = `${response.data.auto_motivation.audio_url}`;
        console.log('🔊 Playing auto-motivation audio:', audioUrl);
        
        const success = await playAudio(audioUrl, 'auto-voice');
        
        if (success) {
          // Mark as played on the backend
          await axios.post(`/mark-auto-motivation-played`);
          console.log('✅ Auto-motivation audio played and marked as complete');
          
          // Update the message if available
          if (response.data.auto_motivation.message && motivationData) {
            setMotivationData({
              ...motivationData,
              message: response.data.auto_motivation.message
            });
          }
        }
      }
      
      // Update attention score from face tracking if available
      if (response.data.attention_score !== undefined) {
        console.log('🔍 DEBUG FRONTEND: Updating attention_score from', motivationData?.attention_score, 'to', response.data.attention_score);
        
        if (motivationData) {
          // Update existing motivationData
          setMotivationData({
            ...motivationData,
            attention_score: response.data.attention_score
          });
        } else {
          // Initialize motivationData if it's null
          setMotivationData({
            message: "Welcome to FocusMind! Your attention score is being tracked.",
            attention_score: response.data.attention_score
          });
        }
        console.log('✅ DEBUG FRONTEND: State updated successfully');
      } else {
        console.log('🔍 DEBUG FRONTEND: NOT updating attention_score. attention_score undefined?', response.data.attention_score === undefined);
      }
    } catch (error) {
      console.error('❌ Error checking face tracking status:', error);
    }
  };

  // Pomodoro Timer Effect
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;

    if (pomodoroRunning && pomodoroTime > 0) {
      interval = setInterval(() => {
        setPomodoroTime(prev => {
          if (prev <= 1) {
            setPomodoroRunning(false);
            
            // Only increment session once using ref
            if (!sessionIncrementedRef.current) {
              sessionIncrementedRef.current = true;
              
              setPomodoroSessions(currentSessions => {
                const newSessionCount = currentSessions + 1;
                console.log(`🍅 Pomodoro session #${newSessionCount} complete! Triggering break nudge...`);
                
                // Trigger break nudge when Pomodoro completes
                setTimeout(() => {
                  getBreakNudge();
                }, 100);
                
                return newSessionCount;
              });
            }
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }

    // Reset the increment flag when starting a new timer
    if (pomodoroRunning && pomodoroTime === 30) {
      sessionIncrementedRef.current = false;
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [pomodoroRunning, pomodoroTime]);

  useEffect(() => {
    // Reset attention score to 100 on page load/refresh - but only once
    if (!initialMotivationFetchedRef.current) {
      initialMotivationFetchedRef.current = true;
      console.log('🎯 Initial app load - fetching motivation once');
      fetchMotivation(true);
    }
  }, []);

  // Cleanup function to stop audio when component unmounts
  useEffect(() => {
    return () => {
      if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
      }
    };
  }, [currentAudio]);

  // Face tracking status monitoring - Always poll to detect external face tracking
  useEffect(() => {
    // Check face tracking status every 1 second for more real-time updates
    const statusInterval = setInterval(() => {
      checkFaceTrackingStatus();
    }, 1000);
    
    // Also check immediately on mount
    checkFaceTrackingStatus();
    
    return () => {
      clearInterval(statusInterval);
    };
  }, []);

  // Auto-start face tracking when Pomodoro begins
  useEffect(() => {
    if (pomodoroRunning && !faceTrackingActive) {
      console.log('🍅 Pomodoro started - enabling automatic face tracking');
      startFaceTracking();
    } else if (!pomodoroRunning && faceTrackingActive) {
      console.log('⏸️ Pomodoro stopped - disabling face tracking');
      stopFaceTracking();
    }
  }, [pomodoroRunning]);

  // Pomodoro Timer JSX
  const pomodoroTimerJSX = (
    <div style={{ 
        padding: '2rem', 
        background: 'rgba(13, 27, 42, 0.7)',
        backdropFilter: 'blur(20px) saturate(180%)',
        WebkitBackdropFilter: 'blur(20px) saturate(180%)',
        borderRadius: '20px',
        textAlign: 'center',
        boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(0, 168, 255, 0.2), inset 0 1px 0 rgba(0, 168, 255, 0.1)',
        border: '1px solid rgba(0, 168, 255, 0.2)',
        position: 'relative'
      }}>
        <h2 style={{ 
          margin: '0 0 1rem 0', 
          fontSize: '1.75rem',
          fontWeight: '800',
          color: '#ffffff',
          letterSpacing: '-0.02em',
          textShadow: '0 0 20px rgba(0, 168, 255, 0.5), 0 2px 4px rgba(0, 0, 0, 0.5)'
        }}>
          🍅 Pomodoro Timer
        </h2>
        
        <div style={{ 
          fontSize: '0.875rem', 
          color: '#7dd3fc',
          marginBottom: '0.75rem',
          fontWeight: '500'
        }}>
          Session #{pomodoroSessions + 1} • 
          {pomodoroRunning ? ' ⏰ Running' : ' ⏸️ Paused'}
        </div>
        
        {/* Face Tracking Status */}
        <div style={{
          fontSize: '0.75rem',
          padding: '0.4rem 0.875rem',
          borderRadius: '16px',
          margin: '0 0 0.75rem 0',
          background: faceTrackingActive ? 'rgba(34, 197, 94, 0.1)' : 'rgba(156, 163, 175, 0.1)',
          color: faceTrackingActive ? '#059669' : '#6b7280',
          border: faceTrackingActive ? '1px solid rgba(34, 197, 94, 0.3)' : '1px solid rgba(156, 163, 175, 0.3)'
        }}>
          📹 Face Tracking: {faceTrackingActive ? 'Active' : 'Inactive'}
          {lastFaceTrackingUpdate && faceTrackingActive && (
            <div style={{ fontSize: '0.7rem', marginTop: '0.25rem' }}>
              Last update: {new Date(lastFaceTrackingUpdate).toLocaleTimeString()}
            </div>
          )}
        </div>
        
        <div style={{
          fontSize: '3rem',
          fontWeight: '800',
          color: pomodoroTime <= 60 ? '#ef4444' : '#00d4ff',
          margin: '1.5rem 0',
          fontFamily: '"SF Mono", Monaco, monospace',
          letterSpacing: '0.05em',
          textShadow: pomodoroTime <= 60 
            ? '0 0 30px rgba(239, 68, 68, 0.8), 0 2px 4px rgba(0, 0, 0, 0.5)' 
            : '0 0 30px rgba(0, 212, 255, 0.6), 0 2px 4px rgba(0, 0, 0, 0.5)'
        }}>
          {formatTime(pomodoroTime)}
        </div>

        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          {!pomodoroRunning ? (
            <button 
              onClick={startPomodoro}
              style={{
                padding: '0.875rem 1.75rem',
                background: 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
                color: 'white',
                border: '1px solid rgba(2, 132, 199, 0.5)',
                borderRadius: '12px',
                fontWeight: '600',
                fontSize: '1rem',
                cursor: 'pointer',
                boxShadow: '0 8px 24px rgba(2, 132, 199, 0.4), 0 0 40px rgba(2, 132, 199, 0.2)',
                transition: 'all 0.3s ease'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 12px 32px rgba(2, 132, 199, 0.6), 0 0 60px rgba(2, 132, 199, 0.4)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 8px 24px rgba(2, 132, 199, 0.4), 0 0 40px rgba(2, 132, 199, 0.2)';
              }}
            >
              ▶️ Start
            </button>
          ) : (
            <button 
              onClick={pausePomodoro}
              style={{
                padding: '0.875rem 1.75rem',
                background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
                color: 'white',
                border: '1px solid rgba(245, 158, 11, 0.5)',
                borderRadius: '12px',
                fontWeight: '600',
                fontSize: '1rem',
                cursor: 'pointer',
                boxShadow: '0 8px 24px rgba(245, 158, 11, 0.4), 0 0 40px rgba(245, 158, 11, 0.2)',
                transition: 'all 0.3s ease'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 12px 32px rgba(245, 158, 11, 0.6), 0 0 60px rgba(245, 158, 11, 0.4)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 8px 24px rgba(245, 158, 11, 0.4), 0 0 40px rgba(245, 158, 11, 0.2)';
              }}
            >
              ⏸️ Pause
            </button>
          )}
          
          <button 
            onClick={resetPomodoro}
            style={{
              padding: '0.875rem 1.75rem',
              background: 'linear-gradient(135deg, #475569 0%, #334155 100%)',
              color: 'white',
              border: '1px solid rgba(71, 85, 105, 0.5)',
              borderRadius: '12px',
              fontWeight: '600',
              fontSize: '1rem',
              cursor: 'pointer',
              boxShadow: '0 8px 24px rgba(71, 85, 105, 0.4), 0 0 40px rgba(71, 85, 105, 0.2)',
              transition: 'all 0.3s ease'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 12px 32px rgba(71, 85, 105, 0.6), 0 0 60px rgba(71, 85, 105, 0.4)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 8px 24px rgba(71, 85, 105, 0.4), 0 0 40px rgba(71, 85, 105, 0.2)';
            }}
          >
            🔄 Reset
          </button>
        </div>

        <div style={{ 
          marginTop: '1.5rem', 
          fontSize: '0.8rem', 
          color: '#7dd3fc',
          padding: '1rem 1.25rem',
          background: 'rgba(0, 168, 255, 0.1)',
          borderRadius: '12px',
          border: '1px solid rgba(0, 168, 255, 0.2)',
          textAlign: 'left',
          lineHeight: '1.6',
          boxShadow: 'inset 0 0 20px rgba(0, 168, 255, 0.1)'
        }}>
          <div style={{ fontWeight: '600', marginBottom: '0.5rem', color: '#ffffff' }}>📋 How it works:</div>
          • Click Start for 30-second focus session (testing mode)<br/>
          • Timer turns red in final minute<br/>
          • Auto break nudge when timer reaches 0<br/>
          • David Goggins tells you to stretch & hydrate!<br/>
          • After break → See your focus performance chart!<br/>
          • {isBreakTime ? '🛌 BREAK MODE: No other nudges will interrupt' : '💪 FOCUS MODE: Ready for motivation'}
        </div>
        
        {pomodoroTime === 0 && (
          <div style={{
            marginTop: '1rem',
            padding: '1rem 1.25rem',
            background: 'rgba(16, 185, 129, 0.15)',
            color: '#34d399',
            borderRadius: '12px',
            border: '1px solid rgba(16, 185, 129, 0.4)',
            fontWeight: '600',
            fontSize: '0.95rem',
            boxShadow: '0 8px 24px rgba(16, 185, 129, 0.3), inset 0 0 20px rgba(16, 185, 129, 0.1)',
            animation: 'pulse 2s infinite'
          }}>
            🎉 Session Complete! Time for a break!
          </div>
        )}
      </div>
  );

  return (
    <div className="App">
      <Header 
        message={motivationData?.message || "Welcome to FocusMind! Click 'Get Voice Nudge' to hear David Goggins motivation."} 
        loading={loading}
      />
      <Dashboard 
        attentionScore={motivationData?.attention_score || 0}
        onDecreaseAttention={decreaseAttention}
        onGetVoiceNudge={readCurrentMessage}
        onGetNotificationNudge={getNotificationNudge}
        loading={loading}
        nudgeExecuted={nudgeExecuted}
        notificationSent={notificationSent}
        notificationPermission={notificationPermission}
        onRequestNotificationPermission={() => {
          if ('Notification' in window) {
            Notification.requestPermission().then((permission) => {
              setNotificationPermission(permission);
            });
          }
        }}
        pomodoroTimer={pomodoroTimerJSX}
      />
      
      {/* Focus Chart Modal */}
      {showFocusChart && focusChartData && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            borderRadius: '16px',
            padding: '2rem',
            maxWidth: '90vw',
            maxHeight: '90vh',
            overflow: 'auto',
            position: 'relative'
          }}>
            <button
              onClick={closeFocusChart}
              style={{
                position: 'absolute',
                top: '1rem',
                right: '1rem',
                background: '#ef4444',
                color: 'white',
                border: 'none',
                borderRadius: '50%',
                width: '2rem',
                height: '2rem',
                cursor: 'pointer',
                fontSize: '1.2rem'
              }}
            >
              ×
            </button>
            
            <h2 style={{ textAlign: 'center', marginBottom: '1.5rem', color: '#1f2937' }}>
              📊 Focus Session Analysis
            </h2>
            
            {focusChartData.chart_base64 && (
              <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                <img 
                  src={`data:image/png;base64,${focusChartData.chart_base64}`}
                  alt="Focus Score Chart"
                  style={{ maxWidth: '100%', height: 'auto', borderRadius: '8px' }}
                />
              </div>
            )}
            
            {focusChartData.session_stats && (
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '1rem',
                marginBottom: '1.5rem'
              }}>
                <div style={{ background: '#dbeafe', padding: '1rem', borderRadius: '8px', border: '2px solid #3b82f6' }}>
                  <h4 style={{ margin: '0 0 0.5rem 0', color: '#1e40af', fontSize: '0.9rem' }}>📊 Average Focus</h4>
                  <p style={{ margin: 0, fontSize: '2rem', fontWeight: 'bold', color: '#2563eb' }}>
                    {focusChartData.session_stats.average_focus ? Math.round(focusChartData.session_stats.average_focus) : 0}%
                  </p>
                </div>
                
                <div style={{ background: '#fed7aa', padding: '1rem', borderRadius: '8px', border: '2px solid #f97316' }}>
                  <h4 style={{ margin: '0 0 0.5rem 0', color: '#9a3412', fontSize: '0.9rem' }}>🔥 Highest Score</h4>
                  <p style={{ margin: 0, fontSize: '2rem', fontWeight: 'bold', color: '#ea580c' }}>
                    {focusChartData.session_stats.max_focus ? Math.round(focusChartData.session_stats.max_focus) : 0}%
                  </p>
                </div>
                
                <div style={{ background: '#fecaca', padding: '1rem', borderRadius: '8px', border: '2px solid #dc2626' }}>
                  <h4 style={{ margin: '0 0 0.5rem 0', color: '#991b1b', fontSize: '0.9rem' }}>📉 Lowest Score</h4>
                  <p style={{ margin: 0, fontSize: '2rem', fontWeight: 'bold', color: '#dc2626' }}>
                    {focusChartData.session_stats.min_focus ? Math.round(focusChartData.session_stats.min_focus) : 0}%
                  </p>
                </div>
                
                <div style={{ background: '#f3f4f6', padding: '1rem', borderRadius: '8px' }}>
                  <h4 style={{ margin: '0 0 0.5rem 0', color: '#374151', fontSize: '0.9rem' }}>⏱️ Duration</h4>
                  <p style={{ margin: 0, fontSize: '2rem', fontWeight: 'bold', color: '#6b7280' }}>
                    {focusChartData.session_stats.duration_seconds ? Math.round(focusChartData.session_stats.duration_seconds / 60) : 0} min
                  </p>
                </div>
                
                <div style={{ background: '#f3f4f6', padding: '1rem', borderRadius: '8px' }}>
                  <h4 style={{ margin: '0 0 0.5rem 0', color: '#374151', fontSize: '0.9rem' }}>📈 Data Points</h4>
                  <p style={{ margin: 0, fontSize: '2rem', fontWeight: 'bold', color: '#6b7280' }}>
                    {focusChartData.data_points || 0}
                  </p>
                </div>
              </div>
            )}
            
            <div style={{ textAlign: 'center' }}>
              <button
                onClick={closeFocusChart}
                style={{
                  padding: '0.75rem 2rem',
                  background: 'linear-gradient(135deg, #059669 0%, #047857 100%)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '12px',
                  fontWeight: '600',
                  cursor: 'pointer'
                }}
              >
                Continue Studying 💪
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;