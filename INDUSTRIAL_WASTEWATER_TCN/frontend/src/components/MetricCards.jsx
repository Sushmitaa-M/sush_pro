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

        const isIncreasing = peak !== '--' && curr !== '--' && peak > curr;
        const isDecreasing = peak !== '--' && curr !== '--' && peak < curr;

        // Card style: Active card gets the signature deep forest green treatment inspired by Reference Image 1
        let cardClass = 'metric-card animate-fade';
        if (isSelected) {
          if (status === 'Critical') cardClass += ' card-active-critical';
          else if (status === 'Warning') cardClass += ' card-active-warning';
          else cardClass += ' card-active-forest';
        }

        return (
          <div
            key={key}
            className={cardClass}
            onClick={() => onSelectParam && onSelectParam(key)}
            title={`Click to focus 48h forecast chart on ${key}`}
          >
            <div className="metric-card-top">
              <div className="metric-param-title-group">
                <span className="metric-param-name">{key}</span>
                <span className="metric-param-full-label">{label}</span>
              </div>
              <span className={`status-pill ${status.toLowerCase()}`} style={{ fontSize: '11px', padding: '3px 9px' }}>
                {status}
              </span>
            </div>

            <div className="metric-value-row">
              <span className="metric-current-val">{curr}</span>
              {unit && <span className="metric-param-unit">{unit}</span>}

              {/* Trend Icon in pill circle */}
              <div className="trend-indicator-box">
                {isIncreasing && <ArrowUpRight size={15} />}
                {isDecreasing && <ArrowDownRight size={15} />}
                {!isIncreasing && !isDecreasing && <Minus size={15} />}
              </div>
            </div>

            <div className="metric-meta-grid">
              <div className="meta-col">
                <div className="meta-label">24h Peak</div>
                <div className="meta-value">
                  {peak} {unit}
                </div>
              </div>
              <div className="meta-col">
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
