import React from 'react';
import { Activity, ShieldAlert, CheckCircle, AlertTriangle, Radio } from 'lucide-react';

export default function Header({ status, hsraeResult, isConnected }) {
  const riskLevel = hsraeResult?.overall_risk_level || 'Normal';
  const riskScore = hsraeResult?.risk_score ?? 0;

  const getRiskBadgeClass = () => {
    if (riskLevel === 'Critical') return 'status-pill critical';
    if (riskLevel === 'Warning') return 'status-pill warning';
    return 'status-pill normal';
  };

  return (
    <header className="top-header">
      <div className="header-inner">
        <div className="brand-section">
          <div className="brand-icon-box">
            <Activity size={18} strokeWidth={2.5} />
          </div>
          <div>
            <h1 className="brand-title">Wastewater AI Telemetry &amp; HSRAE Engine</h1>
            <p className="brand-subtitle">
              Facility: ETP Basin #04 • 24h Early Spike Forecasting &amp; Anomaly Detection
            </p>
          </div>
        </div>

        <div className="header-status-group">
          {/* Connection Status */}
          <div className="status-pill">
            <span
              className={`dot-indicator ${isConnected ? 'normal' : 'critical'}`}
              style={{ color: isConnected ? '#059669' : '#dc2626' }}
            />
            <span>{isConnected ? 'API Live (FastAPI)' : 'API Connecting...'}</span>
          </div>

          {/* Active Model Stack */}
          <div className="status-pill">
            <Radio size={13} style={{ color: '#4b5563' }} />
            <span>TCN v2 + Autoencoder</span>
          </div>

          {/* Global HSRAE Risk Level */}
          <div className={getRiskBadgeClass()} style={{ fontWeight: 600 }}>
            {riskLevel === 'Critical' && <ShieldAlert size={14} />}
            {riskLevel === 'Warning' && <AlertTriangle size={14} />}
            {riskLevel === 'Normal' && <CheckCircle size={14} />}
            <span>HSRAE: {riskLevel.toUpperCase()} ({riskScore}%)</span>
          </div>
        </div>
      </div>
    </header>
  );
}
