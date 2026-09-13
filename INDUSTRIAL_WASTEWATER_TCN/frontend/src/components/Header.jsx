import React from 'react';
import { ShieldAlert, CheckCircle, AlertTriangle, Radio, Volume2, VolumeX, Activity } from 'lucide-react';

export default function Header({ status, hsraeResult, isConnected, isMuted, onToggleMute }) {
  const riskLevel = hsraeResult?.overall_risk_level || 'Normal';
  const riskScore = hsraeResult?.risk_score ?? 0;

  const getRiskBadgeClass = () => {
    if (riskLevel === 'Critical') return 'status-pill critical';
    if (riskLevel === 'Warning') return 'status-pill warning';
    return 'status-pill normal';
  };

  return (
    <header className="app-header">
      <div className="header-left">
        <div className="header-titles">
          <h1 className="header-main-title">
            Industrial Wastewater Early Warning &amp; Prediction System
          </h1>
          <p className="header-sub-title">
            Facility: ETP Basin #04 • 24h Predictive Spike Telemetry (Simulation / Historical Replay)
          </p>
        </div>
      </div>

      <div className="header-right">
        {/* Connection Status Pill */}
        <div className="status-pill status-pill-neutral">
          <span
            className={`dot-indicator ${isConnected ? 'dot-online' : 'dot-offline'}`}
          />
          <span className="pill-text">{isConnected ? 'API Live (FastAPI)' : 'API Disconnected'}</span>
        </div>

        {/* Model Architecture Stack */}
        <div className="status-pill status-pill-neutral hide-mobile">
          <Radio size={12} className="model-radio-icon" />
          <span className="pill-text">TCN v2 + Conv-AE + HSRAE</span>
        </div>

        {/* Audio Alert Mute Toggle */}
        <button
          onClick={onToggleMute}
          className={`status-pill status-pill-btn ${isMuted ? 'muted' : 'active'}`}
          title={isMuted ? 'Audio alerts muted (click to unmute)' : 'Audio alerts active (click to mute)'}
        >
          {isMuted ? (
            <VolumeX size={13} style={{ color: '#94a3b8' }} />
          ) : (
            <Volume2 size={13} style={{ color: riskLevel === 'Critical' ? '#dc2626' : '#059669' }} />
          )}
          <span className="pill-text">{isMuted ? 'Muted' : 'Sound On'}</span>
        </button>

        {/* HSRAE Risk Level Badge */}
        <div className={getRiskBadgeClass()} style={{ fontWeight: 700 }}>
          {riskLevel === 'Critical' && <ShieldAlert size={14} className="pulse-dot" />}
          {riskLevel === 'Warning' && <AlertTriangle size={14} />}
          {riskLevel === 'Normal' && <CheckCircle size={14} />}
          <span className="pill-text">HSRAE: {riskLevel.toUpperCase()} ({riskScore}%)</span>
        </div>
      </div>
    </header>
  );
}
