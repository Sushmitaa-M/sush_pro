import React, { useState } from 'react';
import { ShieldAlert, AlertTriangle, CheckCircle, Clock, Volume2, VolumeX, AlertOctagon, ChevronDown, ChevronUp, Activity } from 'lucide-react';

export default function AlertBanner({
  hsraeResult,
  autoencoderData,
  isMuted,
  onToggleMute,
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  const riskLevel = hsraeResult?.overall_risk_level || 'Normal';
  const riskScore = hsraeResult?.risk_score ?? 0;
  const primaryDriver = hsraeResult?.primary_driver || 'None';
  const earliestSpikeHour = hsraeResult?.earliest_spike_hour;
  const breachedParams = hsraeResult?.breached_parameters || [];
  const protocols = hsraeResult?.protocols || [];
  const isAnomaly = autoencoderData?.is_anomaly;

  const isNormal = riskLevel === 'Normal' && !isAnomaly;
  const isWarning = riskLevel === 'Warning' || (isAnomaly && riskLevel === 'Normal');
  const isCritical = riskLevel === 'Critical';

  return (
    <div
      className={`alert-banner-container ${isCritical ? 'banner-critical' : isWarning ? 'banner-warning' : 'banner-normal'}`}
      style={{
        borderRadius: '16px',
        padding: '16px 22px',
        marginBottom: '22px',
        border: '1px solid',
        transition: 'all 0.3s ease',
        boxShadow: isCritical
          ? '0 6px 20px rgba(220, 38, 38, 0.16)'
          : isWarning
          ? '0 6px 20px rgba(217, 119, 6, 0.14)'
          : '0 2px 8px rgba(5, 150, 105, 0.06)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
        {/* Left: Icon & Alert Heading */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flex: 1, minWidth: '280px' }}>
          <div
            style={{
              width: '42px',
              height: '42px',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              backgroundColor: isCritical ? '#dc2626' : isWarning ? '#d97706' : '#059669',
              color: '#ffffff',
            }}
          >
            {isCritical && <AlertOctagon size={24} className="pulse-dot" />}
            {isWarning && <AlertTriangle size={24} />}
            {isNormal && <CheckCircle size={24} />}
          </div>

          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              <span
                style={{
                  fontSize: '15px',
                  fontWeight: '700',
                  letterSpacing: '-0.01em',
                  color: isCritical ? '#991b1b' : isWarning ? '#92400e' : '#065f46',
                }}
              >
                {isCritical
                  ? '🚨 CRITICAL WASTEWATER SPIKE PREDICTED'
                  : isWarning
                  ? '⚠️ PREDICTED SPIKE / ELEVATED RISK DETECTED'
                  : '🟢 ALL WASTEWATER TELEMETRY NORMAL'}
              </span>

              <span
                style={{
                  fontSize: '11px',
                  fontWeight: '700',
                  padding: '2px 8px',
                  borderRadius: '4px',
                  textTransform: 'uppercase',
                  backgroundColor: isCritical ? '#fee2e2' : isWarning ? '#fef3c7' : '#d1fae5',
                  color: isCritical ? '#991b1b' : isWarning ? '#92400e' : '#065f46',
                  fontFamily: "'JetBrains Mono', monospace",
                }}
              >
                Risk Index: {riskScore}/100 • {riskLevel.toUpperCase()}
              </span>
            </div>

            <div style={{ fontSize: '13px', color: '#4b5563', marginTop: '3px' }}>
              {isCritical && (
                <span>
                  <strong>Urgent Action Required:</strong> {primaryDriver} spike forecasted within{' '}
                  <strong>{earliestSpikeHour ? `+${earliestSpikeHour} Hours` : 'immediate window'}</strong>.
                </span>
              )}
              {isWarning && (
                <span>
                  <strong>Elevated Trend:</strong> {primaryDriver !== 'None' ? primaryDriver : 'Influent'} approaching regulatory threshold limits.
                </span>
              )}
              {isNormal && (
                <span>
                  Biological and physical effluent parameters operating within standard environmental compliance bands.
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Right: Controls (Mute & Expand Action Protocols) */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* Mute Audio Button */}
          <button
            onClick={onToggleMute}
            title={isMuted ? 'Unmute alert siren' : 'Mute alert siren'}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              fontSize: '12px',
              fontWeight: '600',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              background: isMuted ? '#f3f4f6' : '#ffffff',
              border: '1px solid #d1d5db',
              color: isMuted ? '#6b7280' : '#111827',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            {isMuted ? <VolumeX size={15} /> : <Volume2 size={15} style={{ color: isCritical ? '#dc2626' : undefined }} />}
            <span>{isMuted ? 'Alert Muted' : 'Sound Active'}</span>
          </button>

          {/* Expand Details Button */}
          <button
            onClick={() => setIsExpanded((prev) => !prev)}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              fontSize: '12px',
              fontWeight: '600',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              background: '#ffffff',
              border: '1px solid #d1d5db',
              color: '#111827',
              cursor: 'pointer',
            }}
          >
            <span>{isExpanded ? 'Hide Protocols' : 'View Protocols'}</span>
            {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>
      </div>

      {/* Expanded Protocols & Parameter Details */}
      {isExpanded && (
        <div
          style={{
            marginTop: '14px',
            paddingTop: '14px',
            borderTop: '1px solid rgba(0,0,0,0.08)',
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '14px',
          }}
        >
          {/* Operator Action Checklist */}
          <div style={{ background: '#ffffff', padding: '12px 16px', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
            <div style={{ fontSize: '12px', fontWeight: '700', color: '#111827', marginBottom: '8px', textTransform: 'uppercase' }}>
              Standard Operator Protocols:
            </div>
            <ul style={{ margin: 0, paddingLeft: '18px', fontSize: '12px', color: '#374151', lineHeight: '1.6' }}>
              {protocols.map((p, idx) => (
                <li key={idx}>{p}</li>
              ))}
            </ul>
          </div>

          {/* Breach Breakdown */}
          <div style={{ background: '#ffffff', padding: '12px 16px', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
            <div style={{ fontSize: '12px', fontWeight: '700', color: '#111827', marginBottom: '8px', textTransform: 'uppercase' }}>
              Parameter Risk Breakdown:
            </div>
            <div style={{ fontSize: '12px', color: '#4b5563', lineHeight: '1.6' }}>
              <div><strong>Primary Contributing Driver:</strong> {primaryDriver}</div>
              <div><strong>Earliest Breach Horizon:</strong> {earliestSpikeHour ? `Hour +${earliestSpikeHour}` : 'None predicted'}</div>
              <div><strong>Autoencoder Anomaly Status:</strong> {isAnomaly ? '🚨 Reconstruction Deviation Flagged' : '✓ Standard Pattern'}</div>
              {breachedParams.length > 0 && (
                <div style={{ marginTop: '6px' }}>
                  <strong>Breached Channels: </strong>
                  {breachedParams.map((b, i) => (
                    <span
                      key={i}
                      style={{
                        display: 'inline-block',
                        marginRight: '6px',
                        padding: '1px 6px',
                        borderRadius: '4px',
                        fontSize: '11px',
                        background: b.level === 'Critical' ? '#fee2e2' : '#fef3c7',
                        color: b.level === 'Critical' ? '#dc2626' : '#d97706',
                        fontWeight: '600',
                      }}
                    >
                      {b.param} ({b.peak_value}) - {b.level}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
