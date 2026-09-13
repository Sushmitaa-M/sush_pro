import React, { useState } from 'react';
import { Activity, FlaskConical, ShieldAlert, Cpu, Leaf, Radio } from 'lucide-react';

export default function NavigationRail({ activeTab, onSelectTab, isConnected }) {
  const [isHovered, setIsHovered] = useState(false);

  const navItems = [
    { id: 'overview', label: 'Overview & 24h Monitoring', shortLabel: 'Overview', icon: Activity },
    { id: 'whatif', label: 'What-If Spike Simulation', shortLabel: 'What-If Simulation', icon: FlaskConical },
    { id: 'hsrae', label: 'HSRAE Risk Decision Engine', shortLabel: 'HSRAE Risk Engine', icon: ShieldAlert },
    { id: 'autoencoder', label: 'Autoencoder Anomaly Detection', shortLabel: 'Anomaly Detection', icon: Cpu },
  ];

  return (
    <aside
      className={`nav-rail ${isHovered ? 'expanded' : 'collapsed'}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      aria-label="Sidebar Navigation"
    >
      {/* Brand Header: Sleek Environmental Mark (No water drop) */}
      <div className="rail-brand">
        <div className="rail-logo-box">
          <Leaf size={19} strokeWidth={2.2} />
        </div>
        <div className="rail-brand-text">
          <span className="rail-brand-name">EcoTelemetry</span>
          <span className="rail-brand-sub">ETP Basin AI</span>
        </div>
      </div>

      {/* Navigation Items */}
      <nav className="rail-nav">
        {navItems.map(({ id, label, shortLabel, icon: Icon }) => {
          const isActive = activeTab === id;
          return (
            <button
              key={id}
              onClick={() => onSelectTab(id)}
              className={`rail-item ${isActive ? 'active' : ''}`}
              title={label}
            >
              <div className="rail-item-icon">
                <Icon size={18} strokeWidth={isActive ? 2.5 : 2} />
              </div>
              <span className="rail-item-label">{shortLabel}</span>
            </button>
          );
        })}
      </nav>

      {/* Rail Bottom: Live telemetry pulse */}
      <div className="rail-footer">
        <div className="rail-status-dot" style={{ backgroundColor: isConnected ? '#10b981' : '#ef4444' }} />
        <span className="rail-footer-text">
          {isConnected ? 'AI Pipeline Online' : 'Connecting API'}
        </span>
      </div>
    </aside>
  );
}
