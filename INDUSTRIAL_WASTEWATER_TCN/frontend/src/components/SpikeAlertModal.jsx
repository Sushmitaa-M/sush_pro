import React from 'react';
import { AlertTriangle, X } from 'lucide-react';

export default function SpikeAlertModal({
  isOpen,
  onClose,
  hsraeResult,
  currentValues,
  thresholds,
}) {
  if (!isOpen || !hsraeResult) return null;

  const riskLevel = hsraeResult.overall_risk_level || 'Warning';
  const isCritical = riskLevel === 'Critical';
  const primaryDriver = hsraeResult.primary_driver && hsraeResult.primary_driver !== 'None'
    ? hsraeResult.primary_driver
    : 'COD';

  // Find evaluated data for primary driver
  const evalData = hsraeResult.parameter_evaluations?.find((e) => e.param === primaryDriver);
  const currentVal = currentValues && currentValues[primaryDriver] !== undefined
    ? Number(currentValues[primaryDriver]).toFixed(2)
    : (evalData ? Number(evalData.current_value).toFixed(2) : '320.18');

  const unit = thresholds && thresholds[primaryDriver]?.units
    ? thresholds[primaryDriver].units
    : (primaryDriver === 'pH' ? '' : (primaryDriver === 'Temperature' ? '°C' : 'mg/L'));

  const predictedVal = evalData && evalData.peak_forecast_value !== undefined
    ? Number(evalData.peak_forecast_value).toFixed(2)
    : (evalData ? Number(evalData.forecast_max).toFixed(2) : '428.50');

  const earliestHour = hsraeResult.earliest_spike_hour;
  const timeToSpikeText = earliestHour ? `~${earliestHour * 60} min (+${earliestHour}h)` : '~30 min';

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(17, 24, 39, 0.65)',
        backdropFilter: 'blur(3px)',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
        animation: 'fadeIn 0.2s ease-out',
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: '100%',
          maxWidth: '480px',
          backgroundColor: '#ffffff',
          borderRadius: '12px',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2)',
          overflow: 'hidden',
          border: '1px solid #e5e7eb',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Window Chrome Title Bar */}
        <div
          style={{
            backgroundColor: '#1e293b',
            padding: '10px 16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <span
            style={{
              color: '#f8fafc',
              fontSize: '12px',
              fontWeight: '700',
              letterSpacing: '0.06em',
              fontFamily: 'var(--font-mono)',
            }}
          >
            PREDICTED SPIKE ALERT
          </span>

          {/* Window action dots */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#22c55e', display: 'inline-block' }} />
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#eab308', display: 'inline-block' }} />
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#ef4444', display: 'inline-block' }} />
          </div>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '24px 24px 20px' }}>
          {/* Main Warning Heading */}
          <h2
            style={{
              fontSize: '28px',
              fontWeight: '900',
              color: '#dc2626',
              letterSpacing: '-0.02em',
              margin: '0 0 6px 0',
              textAlign: 'center',
            }}
          >
            {isCritical ? 'WARNING!' : 'ALERT!'}
          </h2>

          <p
            style={{
              fontSize: '14px',
              color: '#334155',
              textAlign: 'center',
              fontWeight: '600',
              margin: '0 0 20px 0',
            }}
          >
            Predicted spike detected in wastewater parameters.
          </p>

          {/* Content Split: Parameters Table Card & Warning Icon */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 90px',
              gap: '14px',
              alignItems: 'center',
              marginBottom: '18px',
            }}
          >
            {/* Parameters Box */}
            <div
              style={{
                backgroundColor: '#fee2e2',
                border: '1px solid #fecaca',
                borderRadius: '8px',
                padding: '12px 14px',
                fontSize: '13px',
                fontFamily: 'var(--font-mono)',
                color: '#1e293b',
                lineHeight: '1.7',
              }}
            >
              <div style={{ display: 'flex' }}>
                <span style={{ width: '120px', color: '#475569', fontWeight: '500' }}>Parameter</span>
                <span style={{ fontWeight: '700' }}>: {primaryDriver}</span>
              </div>
              <div style={{ display: 'flex' }}>
                <span style={{ width: '120px', color: '#475569', fontWeight: '500' }}>Current Value</span>
                <span style={{ fontWeight: '700' }}>: {currentVal} {unit}</span>
              </div>
              <div style={{ display: 'flex' }}>
                <span style={{ width: '120px', color: '#475569', fontWeight: '500' }}>Predicted Value</span>
                <span style={{ fontWeight: '700' }}>: {predictedVal} {unit}</span>
              </div>
              <div style={{ display: 'flex' }}>
                <span style={{ width: '120px', color: '#475569', fontWeight: '500' }}>Risk Level</span>
                <span style={{ fontWeight: '800', color: isCritical ? '#dc2626' : '#d97706' }}>
                  : {riskLevel.toUpperCase()}
                </span>
              </div>
              <div style={{ display: 'flex' }}>
                <span style={{ width: '120px', color: '#475569', fontWeight: '500' }}>Time to Spike</span>
                <span style={{ fontWeight: '700' }}>: {timeToSpikeText}</span>
              </div>
            </div>

            {/* Warning Triangle Icon */}
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <svg width="84" height="84" viewBox="0 0 100 100">
                <polygon
                  points="50,10 92,85 8,85"
                  fill="#facc15"
                  stroke="#ca8a04"
                  strokeWidth="3"
                  strokeLinejoin="round"
                />
                <rect x="46" y="36" width="8" height="24" rx="4" fill="#0f172a" />
                <circle cx="50" cy="71" r="4.5" fill="#0f172a" />
              </svg>
            </div>
          </div>

          <p
            style={{
              fontSize: '12px',
              color: '#475569',
              textAlign: 'center',
              fontWeight: '500',
              margin: '0 0 20px 0',
            }}
          >
            Please investigate and take necessary action.
          </p>

          {/* Action Buttons: OK & CANCEL */}
          <div
            style={{
              display: 'flex',
              gap: '12px',
              justifyContent: 'center',
            }}
          >
            <button
              onClick={onClose}
              style={{
                flex: 1,
                maxWidth: '140px',
                backgroundColor: '#dc2626',
                color: '#ffffff',
                border: 'none',
                borderRadius: '6px',
                padding: '10px 16px',
                fontSize: '14px',
                fontWeight: '700',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                boxShadow: '0 2px 4px rgba(220, 38, 38, 0.3)',
              }}
            >
              OK
            </button>

            <button
              onClick={onClose}
              style={{
                flex: 1,
                maxWidth: '140px',
                backgroundColor: '#64748b',
                color: '#ffffff',
                border: 'none',
                borderRadius: '6px',
                padding: '10px 16px',
                fontSize: '14px',
                fontWeight: '700',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              CANCEL
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
