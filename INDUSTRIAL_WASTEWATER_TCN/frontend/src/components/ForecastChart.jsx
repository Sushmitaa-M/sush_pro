import React, { useState } from 'react';
import { TrendingUp, Radio, AlertTriangle } from 'lucide-react';

export default function ForecastChart({
  historyTimeline,
  forecastTimeline,
  thresholds,
  activeParam = 'COD',
  onSelectParam,
}) {
  const [internalParam, setInternalParam] = useState('COD');
  const [hoveredPoint, setHoveredPoint] = useState(null);

  const currentParam = onSelectParam ? activeParam : internalParam;
  const setParam = onSelectParam || setInternalParam;

  const paramList = [
    { key: 'COD', label: 'COD (mg/L)' },
    { key: 'pH', label: 'pH' },
    { key: 'BOD', label: 'BOD (mg/L)' },
    { key: 'TDS', label: 'TDS (mg/L)' },
    { key: 'Temperature', label: 'Temp (°C)' },
  ];

  const th = thresholds ? thresholds[currentParam] : null;

  // Build combined sequence: past 24 points (-23 to 0) + future 24 points (+1 to +24)
  const pastPoints = (historyTimeline || []).map((pt) => ({
    hour: pt.hour,
    label: pt.label,
    value: pt[`${currentParam}_actual`] ?? 0,
    recon: pt[`${currentParam}_recon`] ?? null,
    isFuture: false,
  }));

  const futurePoints = (forecastTimeline || []).map((pt) => ({
    hour: pt.hour,
    label: pt.label,
    value: pt[`${currentParam}_forecast`] ?? 0,
    isFuture: true,
  }));

  const allPoints = [...pastPoints, ...futurePoints];

  if (allPoints.length === 0) {
    return (
      <div className="panel">
        <div style={{ padding: '40px', textAlign: 'center', color: '#94a3b8' }}>
          Initializing live forecast telemetry...
        </div>
      </div>
    );
  }

  // Calculate SVG ViewBox coordinates
  const width = 960;
  const height = 300;
  const padding = { top: 35, right: 30, bottom: 40, left: 60 };

  const allValues = allPoints.map((p) => p.value);
  let minVal = Math.min(...allValues);
  let maxVal = Math.max(...allValues);

  // Include thresholds in scale bounds
  if (th) {
    if (th.normal_min !== undefined) minVal = Math.min(minVal, th.normal_min * 0.9);
    if (th.normal_max !== undefined) maxVal = Math.max(maxVal, th.normal_max * 1.15);
    if (th.warning_max !== undefined && !isFinite(th.warning_max)) {
      maxVal = Math.max(maxVal, th.normal_max * 1.3);
    } else if (th.warning_max !== undefined) {
      maxVal = Math.max(maxVal, th.warning_max * 1.1);
    }
  }

  const range = maxVal - minVal || 1;
  const yMin = Math.max(0, minVal - range * 0.08);
  const yMax = maxVal + range * 0.08;

  const getX = (idx) => {
    return padding.left + (idx / (allPoints.length - 1)) * (width - padding.left - padding.right);
  };

  const getY = (val) => {
    return padding.top + (1 - (val - yMin) / (yMax - yMin)) * (height - padding.top - padding.bottom);
  };

  // Generate paths
  const pastCoords = pastPoints.map((p, idx) => ({ x: getX(idx), y: getY(p.value) }));
  const futureCoords = futurePoints.map((p, idx) => ({
    x: getX(pastPoints.length + idx),
    y: getY(p.value),
  }));

  const pastPath = pastCoords.length > 0
    ? `M ${pastCoords.map((c) => `${c.x},${c.y}`).join(' L ')}`
    : '';

  const futurePath = futureCoords.length > 0
    ? `M ${pastCoords[pastCoords.length - 1].x},${pastCoords[pastCoords.length - 1].y} L ${futureCoords.map((c) => `${c.x},${c.y}`).join(' L ')}`
    : '';

  const futureAreaPath = futureCoords.length > 0
    ? `M ${pastCoords[pastCoords.length - 1].x},${getY(yMin)} L ${pastCoords[pastCoords.length - 1].x},${pastCoords[pastCoords.length - 1].y} L ${futureCoords.map((c) => `${c.x},${c.y}`).join(' L ')} L ${futureCoords[futureCoords.length - 1].x},${getY(yMin)} Z`
    : '';

  // Now coordinates
  const nowPoint = pastCoords[pastCoords.length - 1];
  const nowX = nowPoint ? nowPoint.x : getX(pastPoints.length - 1);
  const nowY = nowPoint ? nowPoint.y : getY(0);

  // Peak future value
  const peakFutureVal = futurePoints.length > 0 ? Math.max(...futurePoints.map((p) => p.value)) : 0;
  const currentVal = pastPoints.length > 0 ? pastPoints[pastPoints.length - 1].value : 0;

  // Threshold lines
  const warningY = th && th.warning_max !== undefined && isFinite(th.warning_max) ? getY(th.warning_max) : null;
  const normalMaxY = th && th.normal_max !== undefined ? getY(th.normal_max) : null;
  const normalMinY = th && th.normal_min !== undefined && th.normal_min > 0 ? getY(th.normal_min) : null;

  return (
    <div className="panel animate-fade">
      <div className="panel-header">
        <div>
          <div className="panel-title">
            <TrendingUp size={18} className="panel-icon-emerald" />
            <span>Interactive 48-Hour Continuous Telemetry &amp; TCN Forecast</span>
            <span className="live-stream-badge">
              <Radio size={11} className="pulse-dot" /> LIVE TELEMETRY
            </span>
          </div>
          <div style={{ fontSize: '13px', color: '#64748b', marginTop: '4px' }}>
            Current {currentParam}: <strong style={{ color: '#064e3b' }}>{currentVal.toFixed(1)} {th?.units}</strong> • 24h Forecast Peak: <strong style={{ color: '#0f172a' }}>{peakFutureVal.toFixed(1)} {th?.units}</strong>
          </div>
        </div>

        {/* Parameter Switcher Pill Buttons */}
        <div className="button-group">
          {paramList.map(({ key }) => (
            <button
              key={key}
              className={`filter-btn ${currentParam === key ? 'active' : ''}`}
              onClick={() => {
                setParam(key);
                setHoveredPoint(null);
              }}
            >
              {key}
            </button>
          ))}
        </div>
      </div>

      {/* SVG Chart */}
      <div className="chart-container">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="chart-svg"
          onMouseLeave={() => setHoveredPoint(null)}
        >
          <defs>
            <linearGradient id="emeraldForecastGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#059669" stopOpacity="0.18" />
              <stop offset="100%" stopColor="#059669" stopOpacity="0.01" />
            </linearGradient>
          </defs>

          {/* Subtle horizontal gridlines */}
          {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => {
            const val = yMin + pct * (yMax - yMin);
            const y = getY(val);
            return (
              <g key={i}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={width - padding.right}
                  y2={y}
                  stroke="#f1f5f9"
                  strokeWidth="1"
                />
                <text
                  x={padding.left - 10}
                  y={y + 4}
                  fill="#94a3b8"
                  fontSize="10"
                  fontFamily="'JetBrains Mono', monospace"
                  textAnchor="end"
                >
                  {val.toFixed(currentParam === 'pH' ? 1 : 0)}
                </text>
              </g>
            );
          })}

          {/* Threshold Boundary Lines */}
          {warningY && warningY > padding.top && (
            <g>
              <line
                x1={padding.left}
                y1={warningY}
                x2={width - padding.right}
                y2={warningY}
                stroke="#dc2626"
                strokeWidth="1.2"
                strokeDasharray="4 3"
                opacity="0.85"
              />
              <text
                x={width - padding.right - 8}
                y={warningY - 5}
                fill="#dc2626"
                fontSize="9"
                fontFamily="'JetBrains Mono', monospace"
                textAnchor="end"
                fontWeight="700"
              >
                CRITICAL LIMIT ({th.warning_max})
              </text>
            </g>
          )}

          {normalMaxY && normalMaxY > padding.top && (
            <g>
              <line
                x1={padding.left}
                y1={normalMaxY}
                x2={width - padding.right}
                y2={normalMaxY}
                stroke="#d97706"
                strokeWidth="1"
                strokeDasharray="3 3"
                opacity="0.75"
              />
              <text
                x={width - padding.right - 8}
                y={normalMaxY - 5}
                fill="#d97706"
                fontSize="9"
                fontFamily="'JetBrains Mono', monospace"
                textAnchor="end"
                fontWeight="600"
              >
                WARNING LIMIT ({th.normal_max})
              </text>
            </g>
          )}

          {normalMinY && normalMinY < height - padding.bottom && (
            <g>
              <line
                x1={padding.left}
                y1={normalMinY}
                x2={width - padding.right}
                y2={normalMinY}
                stroke="#d97706"
                strokeWidth="1"
                strokeDasharray="3 3"
                opacity="0.75"
              />
              <text
                x={width - padding.right - 8}
                y={normalMinY + 12}
                fill="#d97706"
                fontSize="9"
                fontFamily="'JetBrains Mono', monospace"
                textAnchor="end"
              >
                MIN NORMAL ({th.normal_min})
              </text>
            </g>
          )}

          {/* Soft Emerald Future Forecast Shading */}
          <path d={futureAreaPath} fill="url(#emeraldForecastGrad)" />

          {/* Past History Path (Deep Teal / Forest) */}
          <path
            d={pastPath}
            fill="none"
            stroke="#0d9488"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Future TCN Prediction Path (Deep Forest Emerald with Dashed Continuity) */}
          <path
            d={futurePath}
            fill="none"
            stroke="#064e3b"
            strokeWidth="2.8"
            strokeDasharray="4 2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Vertical Dividing Line: "NOW" (Hour 0) */}
          <line
            x1={nowX}
            y1={padding.top}
            x2={nowX}
            y2={height - padding.bottom}
            stroke="#064e3b"
            strokeWidth="1.5"
            strokeDasharray="3 3"
            opacity="0.7"
          />
          <text
            x={nowX}
            y={padding.top - 10}
            fill="#064e3b"
            fontSize="10"
            fontWeight="800"
            fontFamily="'JetBrains Mono', monospace"
            textAnchor="middle"
          >
            NOW (0h)
          </text>

          {/* Live Sensor Radar Ping at Hour 0 */}
          {nowPoint && (
            <g>
              <circle
                cx={nowX}
                cy={nowY}
                r="10"
                fill="none"
                stroke="#10b981"
                strokeWidth="1.5"
                opacity="0.6"
                className="pulse-dot"
              />
              <circle
                cx={nowX}
                cy={nowY}
                r="4.5"
                fill="#059669"
                stroke="#ffffff"
                strokeWidth="1.5"
              />
            </g>
          )}

          {/* Interactive Data Point Dots */}
          {allPoints.map((pt, idx) => {
            const x = getX(idx);
            const y = getY(pt.value);
            const isHovered = hoveredPoint && hoveredPoint.idx === idx;

            return (
              <circle
                key={idx}
                cx={x}
                cy={y}
                r={isHovered ? 6 : (pt.isFuture ? 3 : 1.8)}
                fill={pt.isFuture ? '#064e3b' : '#0d9488'}
                stroke="#ffffff"
                strokeWidth={isHovered ? 2.5 : 1}
                style={{ cursor: 'pointer', transition: 'r 0.15s ease' }}
                onMouseEnter={() => setHoveredPoint({ ...pt, x, y, idx })}
              />
            );
          })}

          {/* Time Axis Labels */}
          {[-24, -18, -12, -6, 0, 6, 12, 18, 24].map((hr) => {
            const ptIdx = hr <= 0 ? 24 + hr - 1 : 24 + hr - 1;
            const safeIdx = Math.max(0, Math.min(allPoints.length - 1, ptIdx));
            const x = getX(safeIdx);
            return (
              <text
                key={hr}
                x={x}
                y={height - padding.bottom + 18}
                fill="#64748b"
                fontSize="10"
                fontFamily="'JetBrains Mono', monospace"
                textAnchor="middle"
                fontWeight={hr === 0 ? '700' : '500'}
              >
                {hr === 0 ? '0h' : (hr > 0 ? `+${hr}h` : `${hr}h`)}
              </text>
            );
          })}
        </svg>

        {/* Hover Tooltip Overlay */}
        {hoveredPoint && (
          <div
            style={{
              position: 'absolute',
              left: `${(hoveredPoint.x / width) * 100}%`,
              top: `${(hoveredPoint.y / height) * 100}%`,
              transform: 'translate(-50%, -120%)',
              background: '#04362a',
              color: '#ffffff',
              padding: '6px 12px',
              borderRadius: '8px',
              fontSize: '11px',
              fontFamily: "'JetBrains Mono', monospace",
              pointerEvents: 'none',
              boxShadow: '0 8px 16px rgba(4, 54, 42, 0.25)',
              whiteSpace: 'nowrap',
              zIndex: 10,
              border: '1px solid rgba(167, 243, 208, 0.3)',
            }}
          >
            <div style={{ color: '#a7f3d0', fontSize: '10px' }}>
              {hoveredPoint.isFuture ? 'Forecast ' : 'Observed '}
              <strong>{hoveredPoint.label}</strong>
            </div>
            <div style={{ color: '#ffffff', fontSize: '13px', fontWeight: '800' }}>
              {hoveredPoint.value.toFixed(2)} {th?.units}
            </div>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="chart-legend">
        <div className="legend-item">
          <div className="legend-line" style={{ background: '#0d9488', height: '2.5px' }} />
          <span>Past 24 Hours Observed (Telemetry)</span>
        </div>
        <div className="legend-item">
          <div className="legend-line" style={{ background: '#064e3b', height: '3px', borderTop: '2px dashed #064e3b' }} />
          <span>Next 24 Hours TCN Forecast</span>
        </div>
        <div className="legend-item">
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#059669', display: 'inline-block' }} />
          <span>Current Live Reading (0h)</span>
        </div>
        {warningY && (
          <div className="legend-item">
            <div className="legend-dash" style={{ borderColor: '#d97706' }} />
            <span>Warning Limit</span>
          </div>
        )}
        {warningY && (
          <div className="legend-item">
            <div className="legend-dash" style={{ borderColor: '#dc2626' }} />
            <span>Critical Regulatory Threshold</span>
          </div>
        )}
      </div>
    </div>
  );
}
