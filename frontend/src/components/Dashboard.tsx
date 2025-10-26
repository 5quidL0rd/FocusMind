import React from 'react';
import './Dashboard.css';
import AttentionScore from './AttentionScore';

interface DashboardProps {
  attentionScore: number;
  onDecreaseAttention: () => void;
  onGetVoiceNudge: () => void;
  onGetNotificationNudge: () => void;
  loading: boolean;
  nudgeExecuted: boolean;
  notificationSent: boolean;
  notificationPermission: NotificationPermission;
  onRequestNotificationPermission: () => void;
  pomodoroTimer: React.ReactNode;
}

const Dashboard: React.FC<DashboardProps> = ({ 
  attentionScore, 
  onDecreaseAttention, 
  onGetVoiceNudge, 
  onGetNotificationNudge, 
  loading, 
  nudgeExecuted, 
  notificationSent,
  notificationPermission,
  onRequestNotificationPermission,
  pomodoroTimer
}) => {
  return (
    <main className="dashboard">
      <div className="dashboard-content">
        <div className="dashboard-grid">
          {/* Left side: Attention Score + Buttons */}
          <div className="attention-section">
            <h2 className="section-title">Attention Score</h2>
            
            {/* Attention Score Circle */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.5rem', paddingLeft: '1rem' }}>
              <AttentionScore score={attentionScore} />
            </div>
            
            {/* Buttons below */}
            <div className="nudge-buttons">
              
              <button 
                className={`voice-nudge-button ${nudgeExecuted ? 'nudge-executed' : ''}`}
                onClick={onGetVoiceNudge}
                disabled={loading}
                title="Get a new voice nudge (happens automatically when attention drops)"
              >
                {loading ? 'Loading...' : 'Get Motivation ⚡'}
              </button>
              
              <button 
                className={`notification-nudge-button ${notificationSent ? 'notification-sent' : ''} ${notificationPermission === 'denied' ? 'permission-denied' : ''}`}
                onClick={notificationPermission === 'granted' ? onGetNotificationNudge : onRequestNotificationPermission}
                disabled={loading}
                title={
                  notificationPermission === 'granted' 
                    ? 'Send browser notification' 
                    : notificationPermission === 'denied'
                    ? 'Notifications blocked - check browser settings'
                    : 'Click to enable notifications'
                }
              >
                {loading ? 'Loading...' : 
                 notificationPermission === 'granted' ? 'Notification Nudge 🔔' :
                 notificationPermission === 'denied' ? 'Notifications Blocked 🚫' :
                 'Enable Notifications 🔔'}
              </button>
            </div>
            
            {nudgeExecuted && (
              <div className="nudge-indicator voice-indicator">
                🎤 Voice Nudge Executed! Audio Playing...
              </div>
            )}
            
            {notificationSent && (
              <div className="nudge-indicator notification-indicator">
                🔔 Browser Notification Sent! Check your notifications.
              </div>
            )}
            
            {/* Debug: Permission status */}
            <div style={{
              marginTop: '1rem',
              padding: '0.5rem',
              background: 'rgba(0, 168, 255, 0.1)',
              border: '1px solid rgba(0, 168, 255, 0.2)',
              borderRadius: '8px',
              fontSize: '0.75rem',
              color: '#7dd3fc'
            }}>
              🔐 Notification Permission: <strong>{notificationPermission}</strong>
              {notificationPermission === 'denied' && (
                <div style={{ marginTop: '0.25rem', color: '#ef4444' }}>
                  ⚠️ To enable: Click the lock icon in address bar → Allow notifications
                </div>
              )}
            </div>
          </div>
          
          {/* Right side: Pomodoro Timer */}
          <div className="pomodoro-section">
            {pomodoroTimer}
          </div>
        </div>
      </div>
    </main>
  );
};

export default Dashboard;
