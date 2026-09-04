import React from 'react';
import { ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';

export default function MetricCards({
  currentValues,
  hsraeEvaluations,
  thresholds,
  activeParam = 'COD',
  onSelectParam,
}) {
  const paramList = [
    { key: 'COD', label: 'Chemical Oxygen Demand', unit: 'mg/L' },
    { key: 'pH', label: 'pH Level', unit: '' },
    { key: 'BOD', label: 'Biological Oxygen Demand', unit: 'mg/L' },
    { key: 'TDS', label: 'Total Dissolved Solids', unit: 'mg/L' },
    { key: 'Temperature', label: 'Water Temperature', unit: '°C' },
  ];

  return (
    <div className="metrics-grid">
      {paramList.map(({ key, label, unit }) => {
        const curr = currentValues ? currentValues[key] : '--';
        const evalData = hsraeEvaluations?.find((e) => e.param === key);
        const peak = evalData ? evalData.peak_forecast_value : '--';
        const status = evalData ? evalData.status : 'Normal';
        const th = thresholds ? thresholds[key] : null;
        const isSelected = activeParam === key;

        const getStatusClass = (s) => {
          if (s === 'Critical') return 'status-pill critical';
          if (s === 'Warning') return 'status-pill warning';
          return 'status-pill normal';
        };

        const isIncreasing = peak !== '--' && curr !== '--' && peak > curr;
        const isDecreasing = peak !== '--' && curr !== '--' && peak < curr;

        return (
          <div
            key={key}
            className="metric-card animate-fade"
            onClick={() => onSelectParam && onSelectParam(key)}
            style={{
              cursor: 'pointer',
              borderColor: isSelected ? '#111827' : undefined,
              boxShadow: isSelected ? '0 0 0 1px #111827' : undefined,
              transition: 'all 0.15s ease',
            }}
          >
            <div className="metric-card-top">
              <span className="metric-param-name">{key}</span>
              <span className={getStatusClass(status)} style={{ fontSize: '11px', padding: '3px 8px' }}>
                {status}
              </span>
            </div>

            <div className="metric-value-row">
              <span className="metric-current-val">{curr}</span>
              {unit && <span className="metric-param-unit">{unit}</span>}
              {isIncreasing && (
                <ArrowUpRight
                  size={16}
                  style={{
                    color: status === 'Critical' ? '#dc2626' : (status === 'Warning' ? '#d97706' : '#111827'),
                    marginLeft: 'auto',
                  }}
                />
              )}
              {isDecreasing && (
                <ArrowDownRight size={16} style={{ color: '#4b5563', marginLeft: 'auto' }} />
              )}
              {!isIncreasing && !isDecreasing && (
                <Minus size={16} style={{ color: '#9ca3af', marginLeft: 'auto' }} />
              )}
            </div>

            <div className="metric-meta-grid">
              <div>
                <div className="meta-label">24h Peak</div>
                <div className="meta-value">
                  {peak} {unit}
                </div>
              </div>
              <div>
                <div className="meta-label">Standard Limit</div>
                <div className="meta-value">
                  {th ? (key === 'pH' ? `${th.normal_min}-${th.normal_max}` : `< ${th.normal_max}`) : '--'}
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
