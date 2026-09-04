import React from 'react';
import { Play, Pause, Radio, RefreshCw, Zap, Sliders, FastForward } from 'lucide-react';

export default function LiveStreamBar({
  isStreaming,
  onToggleStream,
  streamSpeed,
  onChangeSpeed,
  applyNoise,
  onToggleNoise,
  currentTimestamp,
  currentIndex,
  totalRows,
  onJumpToIndex,
}) {
  const jumpPresets = [
    { label: 'Normal Baseline', index: 120 },
    { label: 'COD Surge Event', index: 445 },
    { label: 'Acidic pH Excursion', index: 815 },
    { label: 'TDS Salinity Spike', index: 1195 },
    { label: 'Mid-Year Feed (#5000)', index: 5000 },
  ];

  return (
    <div
      style={{
        background: '#ffffff',
        border: '1px solid #e5e7eb',
        borderRadius: '10px',
        padding: '12px 18px',
        marginBottom: '20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '14px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.03)',
      }}
    >
      {/* Left: Stream Toggle & Live Pulse */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap' }}>
        <button
          onClick={onToggleStream}
          style={{
            background: isStreaming ? '#111827' : '#111827',
            color: '#ffffff',
            padding: '8px 16px',
            borderRadius: '6px',
            fontSize: '13px',
            fontWeight: '600',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '7px',
            transition: 'all 0.15s ease',
          }}
        >
          {isStreaming ? (
            <>
              <Pause size={14} />
              <span>Pause Live Feed</span>
            </>
          ) : (
            <>
              <Play size={14} fill="#ffffff" />
              <span>Start Auto Live Stream</span>
            </>
          )}
        </button>

        {/* Live Indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
          <span
            className={isStreaming ? 'pulse-dot' : ''}
            style={{
              width: '9px',
              height: '9px',
              borderRadius: '50%',
              backgroundColor: isStreaming ? '#059669' : '#9ca3af',
              display: 'inline-block',
            }}
          />
          <span style={{ fontWeight: '600', color: isStreaming ? '#059669' : '#6b7280' }}>
            {isStreaming ? 'LIVE INCOMING TELEMETRY' : 'FEED PAUSED'}
          </span>
          {currentTimestamp && (
            <span
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                background: '#f3f4f6',
                padding: '2px 8px',
                borderRadius: '4px',
                color: '#111827',
                fontWeight: '500',
              }}
            >
              {currentTimestamp}
            </span>
          )}
          {currentIndex && totalRows && (
            <span style={{ color: '#9ca3af', fontSize: '11px', fontFamily: "'JetBrains Mono', monospace" }}>
              (Row #{currentIndex} / {totalRows})
            </span>
          )}
        </div>
      </div>

      {/* Right: Controls (Speed, Realistic Noise, Event Jump) */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
        {/* Realistic Jitter Checkbox */}
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '12px',
            color: '#4b5563',
            cursor: 'pointer',
            userSelect: 'none',
          }}
        >
          <input
            type="checkbox"
            checked={applyNoise}
            onChange={(e) => onToggleNoise(e.target.checked)}
            style={{ accentColor: '#111827', cursor: 'pointer' }}
          />
          <span>Realistic Sensor Jitter (±1.5%)</span>
        </label>

        {/* Speed Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#6b7280' }}>
          <FastForward size={14} />
          <span>Speed:</span>
          <select
            value={streamSpeed}
            onChange={(e) => onChangeSpeed(Number(e.target.value))}
            style={{
              padding: '4px 8px',
              borderRadius: '5px',
              border: '1px solid #d1d5db',
              fontSize: '12px',
              background: '#ffffff',
              color: '#111827',
              cursor: 'pointer',
            }}
          >
            <option value={1000}>1 sec / hour (Fast)</option>
            <option value={2000}>2 sec / hour (Normal)</option>
            <option value={4000}>4 sec / hour (Slow)</option>
          </select>
        </div>

        {/* Jump To Event Dropdown */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#6b7280' }}>
          <span>Jump to:</span>
          <select
            onChange={(e) => onJumpToIndex(Number(e.target.value))}
            defaultValue=""
            style={{
              padding: '4px 8px',
              borderRadius: '5px',
              border: '1px solid #d1d5db',
              fontSize: '12px',
              background: '#ffffff',
              color: '#111827',
              cursor: 'pointer',
            }}
          >
            <option value="" disabled>Select event...</option>
            {jumpPresets.map((jp, i) => (
              <option key={i} value={jp.index}>
                {jp.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}
