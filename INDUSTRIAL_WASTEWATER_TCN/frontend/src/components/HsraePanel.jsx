import React from 'react';
import { ShieldAlert, AlertTriangle, CheckCircle, Clock, FileText, CheckSquare, Zap } from 'lucide-react';

export default function HsraePanel({ hsraeResult }) {
  if (!hsraeResult) {
    return (
      <div className="panel">
        <div style={{ padding: '30px', textAlign: 'center', color: '#9ca3af' }}>
          Loading HSRAE Risk Intelligence...
        </div>
      </div>
    );
  }

  const {
    overall_risk_level,
    risk_score,
    earliest_spike_hour,
    primary_driver,
    breached_parameters,
    autoencoder_impact,
    protocols,
  } = hsraeResult;

  // Calculate SVG stroke offset for the circular gauge (circumference = 2 * PI * 60 ~= 377)
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(100, Math.max(0, risk_score));
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  const getRiskColor = () => {
    if (overall_risk_level === 'Critical') return '#dc2626';
    if (overall_risk_level === 'Warning') return '#d97706';
    return '#059669';
  };

  return (
    <div className="panel animate-fade">
      <div className="panel-header">
        <div>
          <div className="panel-title">
            <ShieldAlert size={17} />
            <span>Module 5: HSRAE (Hybrid Spike Risk Assessment Engine)</span>
          </div>
          <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '3px' }}>
            Multi-vector synthesis of TCN 24h forecasts, Autoencoder anomaly scores, and regulatory limits
          </div>
        </div>

        <div className={`status-pill ${overall_risk_level.toLowerCase()}`}>
          <span className="dot-indicator" />
          <span>Status: {overall_risk_level.toUpperCase()}</span>
        </div>
      </div>

      <div className="hsrae-grid">
        {/* Left Column: Risk Gauge & Spike Time */}
        <div className="risk-summary-card">
          <span style={{ fontSize: '12px', fontWeight: '600', color: '#4b5563', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Spike Risk Index
          </span>

          {/* SVG Circular Dial */}
          <div className="risk-dial-container">
            <svg width="150" height="150" viewBox="0 0 150 150">
              {/* Background Track */}
              <circle
                cx="75"
                cy="75"
                r={radius}
                fill="none"
                stroke="#e5e7eb"
                strokeWidth="10"
              />
              {/* Progress Stroke */}
              <circle
                cx="75"
                cy="75"
                r={radius}
                fill="none"
                stroke={getRiskColor()}
                strokeWidth="10"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                transform="rotate(-90 75 75)"
                style={{ transition: 'stroke-dashoffset 0.5s ease' }}
              />
            </svg>
            <div style={{ position: 'absolute', textAlign: 'center' }}>
              <div className="risk-score-num">{risk_score}</div>
              <div className="risk-score-label">out of 100</div>
            </div>
          </div>

          {/* Earliest Spike Countdown */}
          <div className="countdown-box">
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Clock size={14} style={{ color: '#4b5563' }} />
              <span className="countdown-label">Time to Spike:</span>
            </div>
            <span
              className="countdown-val"
              style={{
                color: earliest_spike_hour ? (overall_risk_level === 'Critical' ? '#dc2626' : '#d97706') : '#059669',
              }}
            >
              {earliest_spike_hour ? `${earliest_spike_hour} Hours Ahead` : 'No Spike (> 24h)'}
            </span>
          </div>

          {/* Primary Parameter Driver */}
          <div className="countdown-box" style={{ marginTop: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Zap size={14} style={{ color: '#4b5563' }} />
              <span className="countdown-label">Primary Driver:</span>
            </div>
            <span className="countdown-val">{primary_driver}</span>
          </div>

          {/* Autoencoder Contribution */}
          <div className="countdown-box" style={{ marginTop: '8px' }}>
            <span className="countdown-label">Autoencoder Score:</span>
            <span className="countdown-val">
              {autoencoder_impact?.score ?? 0} ({autoencoder_impact?.status ?? 'Normal'})
            </span>
          </div>
        </div>

        {/* Right Column: Breaches & Actionable Protocols */}
        <div>
          {/* Breached Parameters Notice */}
          <div style={{ marginBottom: '18px' }}>
            <div style={{ fontSize: '13px', fontWeight: '600', color: '#111827', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <FileText size={14} />
              <span>Forecasted Parameter Breaches ({breached_parameters?.length || 0})</span>
            </div>

            {breached_parameters && breached_parameters.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {breached_parameters.map((b, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '10px 14px',
                      background: b.level === 'Critical' ? '#fef2f2' : '#fffbeb',
                      border: `1px solid ${b.level === 'Critical' ? '#fecaca' : '#fde68a'}`,
                      borderRadius: '6px',
                      fontSize: '12px',
                    }}
                  >
                    <div>
                      <strong style={{ color: b.level === 'Critical' ? '#dc2626' : '#d97706' }}>
                        {b.param} {b.level.toUpperCase()} BREACH
                      </strong>
                      <span style={{ color: '#4b5563', marginLeft: '8px' }}>
                        Peak expected at {b.peak_value.toFixed(1)}
                      </span>
                    </div>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: '600', color: '#111827' }}>
                      Hour +{b.hour_of_breach}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div
                style={{
                  padding: '12px 14px',
                  background: '#f0fdf4',
                  border: '1px solid #bbf7d0',
                  borderRadius: '6px',
                  fontSize: '12px',
                  color: '#059669',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                }}
              >
                <CheckCircle size={15} />
                <span>All 5 parameters are forecasted to remain within normal regulatory bounds over the next 24 hours.</span>
              </div>
            )}
          </div>

          {/* Plant Operator Action Protocol Checklist */}
          <div>
            <div style={{ fontSize: '13px', fontWeight: '600', color: '#111827', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <CheckSquare size={14} />
              <span>Plant Operator Action Protocol</span>
            </div>

            <div className="protocols-list">
              {protocols && protocols.map((p, idx) => (
                <div key={idx} className="protocol-item">
                  <div className="protocol-icon">
                    <CheckSquare size={14} />
                  </div>
                  <div>{p}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
