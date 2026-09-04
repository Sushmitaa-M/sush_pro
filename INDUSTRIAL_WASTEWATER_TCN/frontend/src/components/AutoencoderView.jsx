import React from 'react';
import { Cpu, AlertCircle, CheckCircle, BarChart2, Layers } from 'lucide-react';

export default function AutoencoderView({ autoencoderData }) {
  if (!autoencoderData) {
    return (
      <div className="panel">
        <div style={{ padding: '30px', textAlign: 'center', color: '#9ca3af' }}>
          Loading Autoencoder Diagnostics...
        </div>
      </div>
    );
  }

  const {
    is_anomaly,
    reconstruction_error,
    anomaly_score,
    threshold,
    param_contributions,
  } = autoencoderData;

  const scorePct = Math.min(100, Math.round(anomaly_score * 100));

  return (
    <div className="panel animate-fade">
      <div className="panel-header">
        <div>
          <div className="panel-title">
            <Cpu size={17} />
            <span>Module 4: Autoencoder Unsupervised Anomaly Detection</span>
          </div>
          <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '3px' }}>
            1D Convolutional Neural Network reconstructing 24h operational states to detect sensor deviations
          </div>
        </div>

        <div className={`status-pill ${is_anomaly ? 'critical' : 'normal'}`}>
          <span className="dot-indicator" />
          <span>{is_anomaly ? 'ANOMALOUS SEQUENCE DETECTED' : 'STANDARD OPERATIONAL PATTERN'}</span>
        </div>
      </div>

      <div className="ae-grid">
        {/* Metric Overview Card */}
        <div style={{ background: '#f9fafb', padding: '20px', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
          <div style={{ fontSize: '13px', fontWeight: '600', color: '#111827', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Layers size={15} />
            <span>Reconstruction Loss &amp; Threshold</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '18px' }}>
            <div style={{ background: '#ffffff', padding: '12px', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
              <div style={{ fontSize: '11px', color: '#6b7280', marginBottom: '4px' }}>Current Sequence MSE</div>
              <div style={{ fontSize: '20px', fontWeight: '700', fontFamily: "'JetBrains Mono', monospace", color: is_anomaly ? '#dc2626' : '#111827' }}>
                {reconstruction_error}
              </div>
            </div>

            <div style={{ background: '#ffffff', padding: '12px', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
              <div style={{ fontSize: '11px', color: '#6b7280', marginBottom: '4px' }}>95th Percentile Threshold</div>
              <div style={{ fontSize: '20px', fontWeight: '700', fontFamily: "'JetBrains Mono', monospace", color: '#4b5563' }}>
                {threshold}
              </div>
            </div>
          </div>

          {/* Anomaly Score Progress Bar */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
              <span style={{ color: '#4b5563', fontWeight: '500' }}>Normalized Anomaly Index:</span>
              <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: '700', color: is_anomaly ? '#dc2626' : '#111827' }}>
                {anomaly_score} ({scorePct}%)
              </span>
            </div>
            <div className="progress-track" style={{ height: '10px' }}>
              <div
                className="progress-fill"
                style={{
                  width: `${scorePct}%`,
                  backgroundColor: is_anomaly ? '#dc2626' : (scorePct > 35 ? '#d97706' : '#111827'),
                }}
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#9ca3af', marginTop: '4px' }}>
              <span>0.0 (Normal baseline)</span>
              <span>0.50 (95th % Threshold)</span>
              <span>1.0 (Severe anomaly)</span>
            </div>
          </div>
        </div>

        {/* Parameter Error Attribution Breakdown */}
        <div style={{ background: '#f9fafb', padding: '20px', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
          <div style={{ fontSize: '13px', fontWeight: '600', color: '#111827', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <BarChart2 size={15} />
            <span>Anomaly Attribution per Parameter (% Error)</span>
          </div>

          <div>
            {param_contributions &&
              Object.entries(param_contributions).map(([param, pct]) => (
                <div key={param} className="bar-row">
                  <div className="bar-label-row">
                    <span style={{ fontWeight: '500', color: '#111827' }}>{param}</span>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#4b5563' }}>
                      {pct}%
                    </span>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill"
                      style={{
                        width: `${pct}%`,
                        backgroundColor: pct > 40 ? '#111827' : '#6b7280',
                      }}
                    />
                  </div>
                </div>
              ))}
          </div>

          <div style={{ marginTop: '16px', fontSize: '11px', color: '#6b7280', lineHeight: '1.4' }}>
            Parameters with higher attribution percentages represent the primary operational deviation triggering the Autoencoder reconstruction loss.
          </div>
        </div>
      </div>
    </div>
  );
}
