import React from 'react';
import {
  Play,
  Pause,
  FastForward,
  Volume2,
  VolumeX,
} from 'lucide-react';

export default function OperatorControls({
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
  isMuted,
  onToggleMute,
}) {
  const jumpPresets = [
    { label: 'Normal Baseline (#120)', index: 120 },
    { label: 'COD Surge Event (#445)', index: 445 },
    { label: 'Acidic pH Excursion (#815)', index: 815 },
    { label: 'TDS Salinity Spike (#1195)', index: 1195 },
    { label: 'Mid-Year Feed (#5000)', index: 5000 },
  ];

  return (
    <div className="operator-controls-card animate-fade">
      <div className="controls-row-main">
        {/* Left Section: Live Stream / Replay Telemetry Controls */}
        <div className="controls-group-left">
          <button
            onClick={onToggleStream}
            className={`btn-stream-toggle ${isStreaming ? 'streaming' : 'paused'}`}
            title={isStreaming ? 'Pause historical replay stream' : 'Resume live historical replay'}
          >
            {isStreaming ? (
              <>
                <Pause size={14} />
                <span>Pause Replay</span>
              </>
            ) : (
              <>
                <Play size={14} fill="currentColor" />
                <span>Start Live Replay</span>
              </>
            )}
          </button>

          {/* Replay Telemetry State Badge */}
          <div className="telemetry-badge-container">
            <span className={`telemetry-dot ${isStreaming ? 'active' : 'idle'}`} />
            <div className="telemetry-text-group">
              <span className="telemetry-label">
                {isStreaming ? 'LIVE REPLAY STREAMING' : 'REPLAY PAUSED'}
              </span>
              <span className="telemetry-notice">(Historical Sensor Simulation)</span>
            </div>
          </div>

          {currentTimestamp && (
            <span className="telemetry-timestamp">
              {currentTimestamp}
            </span>
          )}

          {currentIndex && totalRows && (
            <span className="telemetry-row-index">
              Row #{currentIndex} / {totalRows}
            </span>
          )}
        </div>

        {/* Right Section: Sound Mute Toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button
            onClick={onToggleMute}
            className={`btn-mute-toggle ${isMuted ? 'muted' : 'active'}`}
            title={isMuted ? 'Click to enable audio alerts' : 'Click to mute audio alerts'}
          >
            {isMuted ? <VolumeX size={14} /> : <Volume2 size={14} />}
            <span>{isMuted ? 'Alarm Muted' : 'Alarm Sound On'}</span>
          </button>
        </div>
      </div>

      {/* Secondary Controls Bar: Speed, Jitter, Historical Event Jump */}
      <div className="controls-row-secondary">
        <div className="secondary-options">
          {/* Realistic Transducer Jitter */}
          <label className="checkbox-control">
            <input
              type="checkbox"
              checked={applyNoise}
              onChange={(e) => onToggleNoise(e.target.checked)}
            />
            <span>Sensor Jitter (±1.5%)</span>
          </label>

          {/* Speed Selector */}
          <div className="selector-group">
            <FastForward size={13} style={{ color: '#64748b' }} />
            <span>Speed:</span>
            <select
              value={streamSpeed}
              onChange={(e) => onChangeSpeed(Number(e.target.value))}
            >
              <option value={1000}>1s / hr (Fast)</option>
              <option value={2000}>2s / hr (Normal)</option>
              <option value={4000}>4s / hr (Slow)</option>
            </select>
          </div>

          {/* Jump To Historical Event */}
          <div className="selector-group">
            <span>Historical Preset:</span>
            <select
              onChange={(e) => onJumpToIndex(Number(e.target.value))}
              defaultValue=""
            >
              <option value="" disabled>Select event preset...</option>
              {jumpPresets.map((jp, i) => (
                <option key={i} value={jp.index}>
                  {jp.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}
