import React from 'react';
import { Zap, RotateCcw, Info, Activity, AlertTriangle } from 'lucide-react';
import MetricCards from './MetricCards';
import ForecastChart from './ForecastChart';
import HsraePanel from './HsraePanel';
import AutoencoderView from './AutoencoderView';

export default function WhatIfStudio({
  isSimulatingSpike,
  isSpikeActive,
  spikeStep,
  totalSpikeSteps = 6,
  elapsedSeconds,
  currentStepLabel,
  onStartSpikeSimulation,
  onResetBaseline,
  predictionData,
  config,
  activeParam,
  onSelectParam,
}) {
  const getStatusBadge = () => {
    if (isSimulatingSpike) {
      return <span style={{ color: '#d97706', fontWeight: '700' }}>● Simulating (Step {spikeStep}/{totalSpikeSteps})</span>;
    }
    if (isSpikeActive) {
      return <span style={{ color: '#dc2626', fontWeight: '700' }}>● Spike Peak Active</span>;
    }
    return <span style={{ color: '#059669', fontWeight: '700' }}>● Ready</span>;
  };

  return (
    <div className="animate-fade">
      {/* Simulation Control Hero Card */}
      <div
        style={{
          background: '#ffffff',
          border: '1px solid var(--border-light)',
          borderRadius: '18px',
          padding: '24px 28px',
          marginBottom: '24px',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        {/* Header with Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
          <span style={{ fontSize: '24px' }}>🧪</span>
          <h2 style={{ fontSize: '20px', fontWeight: '800', color: '#111827', margin: 0 }}>
            What-If Spike Simulation
          </h2>
        </div>

        {/* Info Callout Box (Blue) */}
        <div
          style={{
            background: '#eff6ff',
            border: '1px solid #bfdbfe',
            borderRadius: '8px',
            padding: '14px 18px',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '12px',
            marginBottom: '24px',
          }}
        >
          <Info size={18} style={{ color: '#2563eb', flexShrink: 0, marginTop: '2px' }} />
          <div>
            <div style={{ fontSize: '13px', fontWeight: '600', color: '#1e40af', marginBottom: '2px' }}>
              Click the button below to simulate a gradual increase in wastewater parameters and observe the model's response in real-time.
            </div>
            <div style={{ fontSize: '12px', color: '#3b82f6' }}>
              This will increase the parameter values step by step and run the full prediction pipeline (TCN → Autoencoder → HSRAE).
            </div>
          </div>
        </div>

        {/* Center: Prominent Red Simulate Spike Button */}
        <div style={{ textAlign: 'center', margin: '20px 0 28px' }}>
          {!isSpikeActive ? (
            <button
              onClick={onStartSpikeSimulation}
              disabled={isSimulatingSpike}
              style={{
                background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
                color: '#ffffff',
                border: 'none',
                borderRadius: '10px',
                padding: '16px 36px',
                fontSize: '18px',
                fontWeight: '800',
                letterSpacing: '0.01em',
                cursor: isSimulatingSpike ? 'wait' : 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '10px',
                boxShadow: '0 6px 18px rgba(220, 38, 38, 0.35)',
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={(e) => {
                if (!isSimulatingSpike) e.currentTarget.style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
              }}
            >
              <Zap size={22} fill="#ffffff" />
              <span>{isSimulatingSpike ? `Simulating Step ${spikeStep}/${totalSpikeSteps}...` : '⚡ Simulate Spike'}</span>
            </button>
          ) : (
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap' }}>
              <div
                style={{
                  background: '#fee2e2',
                  border: '1px solid #fecaca',
                  borderRadius: '8px',
                  padding: '10px 18px',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '8px',
                  color: '#dc2626',
                  fontWeight: '700',
                  fontSize: '14px',
                }}
              >
                <AlertTriangle size={18} className="pulse-dot" />
                <span>Spike Injected (Active Peak State)</span>
              </div>

              <button
                onClick={onResetBaseline}
                style={{
                  background: '#ffffff',
                  color: '#111827',
                  border: '1px solid #cbd5e1',
                  borderRadius: '8px',
                  padding: '10px 20px',
                  fontSize: '14px',
                  fontWeight: '700',
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '8px',
                  boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
                  transition: 'all 0.15s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#f8fafc';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = '#ffffff';
                }}
              >
                <RotateCcw size={16} />
                <span>↻ Reset / Return to Normal</span>
              </button>
            </div>
          )}

          <div style={{ fontSize: '12px', color: '#64748b', marginTop: '12px' }}>
            ❖ Values will increase gradually. The dashboard, predictions and risk status will update automatically.
          </div>
        </div>

        {/* Bottom: Simulation Status Grid */}
        <div
          style={{
            borderTop: '1px solid #f1f5f9',
            paddingTop: '16px',
          }}
        >
          <div style={{ fontSize: '13px', fontWeight: '700', color: '#334155', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Activity size={15} />
            <span>Simulation Status</span>
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
              gap: '12px',
              background: '#f8fafc',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              padding: '14px 18px',
              fontSize: '12px',
            }}
          >
            <div>
              <div style={{ color: '#64748b', marginBottom: '3px' }}>Status</div>
              <div>{getStatusBadge()}</div>
            </div>

            <div>
              <div style={{ color: '#64748b', marginBottom: '3px' }}>Elapsed Time</div>
              <div style={{ fontWeight: '700', color: '#1e293b', fontFamily: 'var(--font-mono)' }}>
                {elapsedSeconds > 0 ? `${elapsedSeconds}s` : '-'}
              </div>
            </div>

            <div>
              <div style={{ color: '#64748b', marginBottom: '3px' }}>Current Step</div>
              <div style={{ fontWeight: '700', color: '#1e293b' }}>
                {currentStepLabel || (isSimulatingSpike ? `Step ${spikeStep} of ${totalSpikeSteps}` : '-')}
              </div>
            </div>

            <div>
              <div style={{ color: '#64748b', marginBottom: '3px' }}>Target Condition</div>
              <div style={{ fontWeight: '700', color: '#dc2626' }}>
                High Load Surge Scenario
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Real-Time Telemetry & Forecast Response (Evaluated in real-time by TCN & HSRAE) */}
      <div style={{ marginBottom: '16px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: '700', color: '#1e293b', marginBottom: '12px' }}>
          Real-Time Pipeline Response (TCN v2 Forecasting &amp; HSRAE Risk)
        </h3>
      </div>

      <MetricCards
        currentValues={predictionData?.current_values}
        hsraeEvaluations={predictionData?.hsrae?.parameter_evaluations}
        thresholds={config?.thresholds}
        activeParam={activeParam}
        onSelectParam={onSelectParam}
      />

      <ForecastChart
        historyTimeline={predictionData?.history_timeline}
        forecastTimeline={predictionData?.forecast_timeline}
        thresholds={config?.thresholds}
        activeParam={activeParam}
        onSelectParam={onSelectParam}
      />

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
        <HsraePanel hsraeResult={predictionData?.hsrae} />
        <AutoencoderView autoencoderData={predictionData?.autoencoder} />
      </div>
    </div>
  );
}
