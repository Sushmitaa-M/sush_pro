import React, { useState } from 'react';
import { Sliders, Play, RotateCcw, AlertTriangle, Sparkles } from 'lucide-react';

export default function Simulator({ samples, activeSampleKey, onSelectSample, onRunSimulation, isSimulating }) {
  const [offsets, setOffsets] = useState({
    pH: 0.0,
    COD: 0.0,
    BOD: 0.0,
    TDS: 0.0,
    Temperature: 0.0,
  });

  const handleSliderChange = (param, val) => {
    setOffsets((prev) => ({ ...prev, [param]: parseFloat(val) }));
  };

  const handleResetSliders = () => {
    setOffsets({
      pH: 0.0,
      COD: 0.0,
      BOD: 0.0,
      TDS: 0.0,
      Temperature: 0.0,
    });
  };

  const handleExecute = () => {
    onRunSimulation(offsets);
  };

  const sliderConfigs = [
    { key: 'pH', label: 'pH Offset', min: -2.5, max: 2.5, step: 0.1, unit: '' },
    { key: 'COD', label: 'COD Spike Inject', min: -50, max: 350, step: 10, unit: 'mg/L' },
    { key: 'BOD', label: 'BOD Load Inject', min: -30, max: 150, step: 5, unit: 'mg/L' },
    { key: 'TDS', label: 'TDS Salinity Inject', min: -200, max: 1200, step: 25, unit: 'mg/L' },
    { key: 'Temperature', label: 'Thermal Offset', min: -10, max: 15, step: 0.5, unit: '°C' },
  ];

  return (
    <div className="panel animate-fade">
      <div className="panel-header">
        <div>
          <div className="panel-title">
            <Sliders size={17} />
            <span>Interactive "What-If" Spike Simulation Studio</span>
          </div>
          <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '3px' }}>
            Load historical scenarios or manually perturb wastewater parameters to evaluate real-time AI model response
          </div>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="secondary-btn" onClick={handleResetSliders} disabled={isSimulating}>
            <RotateCcw size={14} />
            <span>Reset Offsets</span>
          </button>
          <button className="primary-btn" onClick={handleExecute} disabled={isSimulating}>
            <Play size={14} />
            <span>{isSimulating ? 'Evaluating AI...' : 'Run Simulation'}</span>
          </button>
        </div>
      </div>

      {/* Preset Scenario Cards */}
      <div style={{ marginBottom: '20px' }}>
        <div style={{ fontSize: '13px', fontWeight: '600', color: '#111827', marginBottom: '10px' }}>
          Select Pre-Recorded Historical Scenario:
        </div>
        <div className="scenario-btn-grid">
          {samples &&
            Object.entries(samples).map(([key, item]) => {
              const isActive = activeSampleKey === key;
              return (
                <button
                  key={key}
                  className={`scenario-card-btn ${isActive ? 'active' : ''}`}
                  onClick={() => {
                    handleResetSliders();
                    onSelectSample(key);
                  }}
                >
                  <div className="scenario-title">
                    {isActive ? '● ' : ''}
                    {item.name}
                  </div>
                  <div className="scenario-desc">{item.description}</div>
                </button>
              );
            })}
        </div>
      </div>

      {/* Manual Perturbation Sliders */}
      <div>
        <div style={{ fontSize: '13px', fontWeight: '600', color: '#111827', marginBottom: '10px' }}>
          Inject Custom Parameter Offsets into Recent 6 Hours:
        </div>
        <div className="slider-group">
          {sliderConfigs.map(({ key, label, min, max, step, unit }) => {
            const currentVal = offsets[key];
            const isModified = currentVal !== 0;

            return (
              <div key={key} className="slider-item">
                <div className="slider-header">
                  <span style={{ fontWeight: '500', color: '#111827' }}>{label}</span>
                  <span
                    style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontWeight: isModified ? '700' : '400',
                      color: isModified ? '#111827' : '#9ca3af',
                    }}
                  >
                    {currentVal > 0 ? `+${currentVal}` : currentVal} {unit}
                  </span>
                </div>
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={step}
                  value={currentVal}
                  className="slider-input"
                  onChange={(e) => handleSliderChange(key, e.target.value)}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#9ca3af' }}>
                  <span>{min}</span>
                  <span>0</span>
                  <span>+{max}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
