import React, { useState, useEffect, useRef } from 'react';
import NavigationRail from './components/NavigationRail';
import Header from './components/Header';
import OperatorControls from './components/OperatorControls';
import AlertBanner from './components/AlertBanner';
import MetricCards from './components/MetricCards';
import ForecastChart from './components/ForecastChart';
import HsraePanel from './components/HsraePanel';
import AutoencoderView from './components/AutoencoderView';
import WhatIfStudio from './components/WhatIfStudio';
import SpikeAlertModal from './components/SpikeAlertModal';
import { soundAlerts } from './utils/audioAlerts';
import './App.css';

// Realistic multi-step industrial organic surge progression
const SPIKE_STEPS = [
  { step: 1, label: 'Step 1 of 6: Early Influent Organic Inflow', offsets: { pH: -0.15, COD: 35, BOD: 20, TDS: 80, Temperature: 0.8 } },
  { step: 2, label: 'Step 2 of 6: Moderate Hydraulic & Organic Buildup', offsets: { pH: -0.30, COD: 85, BOD: 48, TDS: 180, Temperature: 1.6 } },
  { step: 3, label: 'Step 3 of 6: Elevated Load Approaching Limit', offsets: { pH: -0.50, COD: 150, BOD: 85, TDS: 340, Temperature: 2.6 } },
  { step: 4, label: 'Step 4 of 6: Regulatory Warning Threshold Breached', offsets: { pH: -0.75, COD: 220, BOD: 125, TDS: 560, Temperature: 3.8 } },
  { step: 5, label: 'Step 5 of 6: Severe Industrial Effluent Discharge', offsets: { pH: -1.00, COD: 290, BOD: 165, TDS: 800, Temperature: 4.9 } },
  { step: 6, label: 'Step 6 of 6: Critical Organic Spike Peak', offsets: { pH: -1.25, COD: 360, BOD: 210, TDS: 1050, Temperature: 5.8 } },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [isConnected, setIsConnected] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [config, setConfig] = useState(null);
  const [samples, setSamples] = useState(null);

  const [currentSequence, setCurrentSequence] = useState(null);
  const [baselineSequence, setBaselineSequence] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Active parameter selected for chart view
  const [activeParam, setActiveParam] = useState('COD');

  // Audio mute state
  const [isMuted, setIsMuted] = useState(false);

  // Alert Popup Modal State
  const [showSpikeModal, setShowSpikeModal] = useState(false);
  const prevRiskLevelRef = useRef('Normal');

  // Auto Live Stream Replay State (Monitoring only)
  const [isStreaming, setIsStreaming] = useState(true);
  const [streamSpeed, setStreamSpeed] = useState(2000);
  const [applyNoise, setApplyNoise] = useState(true);
  const [streamIndex, setStreamIndex] = useState(120);
  const [currentTimestamp, setCurrentTimestamp] = useState('');
  const [totalRows, setTotalRows] = useState(10000);

  // Spike Simulation State (What-If page)
  const [isSimulatingSpike, setIsSimulatingSpike] = useState(false);
  const [isSpikeActive, setIsSpikeActive] = useState(false);
  const [spikeStep, setSpikeStep] = useState(0);
  const [currentStepLabel, setCurrentStepLabel] = useState('');
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const spikeTimerRef = useRef(null);
  const elapsedTimerRef = useRef(null);

  // Initialize Application & fetch initial state
  useEffect(() => {
    async function initApp() {
      try {
        const [statusRes, configRes, samplesRes] = await Promise.all([
          fetch('/api/status').then((r) => r.json()).catch(() => null),
          fetch('/api/config').then((r) => r.json()).catch(() => null),
          fetch('/api/samples').then((r) => r.json()).catch(() => null),
        ]);

        if (statusRes && statusRes.status === 'ready') {
          setIsConnected(true);
          setApiError(null);
        } else {
          setIsConnected(false);
          setApiError('Prediction service is initializing or unreachable.');
        }

        if (configRes) setConfig(configRes);

        if (samplesRes) {
          setSamples(samplesRes);
          const initialSeq = samplesRes['normal']?.data;
          if (initialSeq) {
            setCurrentSequence(initialSeq);
            setBaselineSequence(initialSeq);
            await executePrediction(initialSeq);
          }
        }
      } catch (err) {
        console.error('Initialization error:', err);
        setApiError('Unable to connect to backend prediction API.');
      } finally {
        setIsLoading(false);
      }
    }

    initApp();
  }, []);

  // Process risk changes for Audio Alarms and Warning Popup
  const handleRiskTransition = (newRiskLevel, data) => {
    const prev = prevRiskLevelRef.current;
    if (newRiskLevel !== prev) {
      prevRiskLevelRef.current = newRiskLevel;

      if (newRiskLevel === 'Warning' || newRiskLevel === 'Critical') {
        soundAlerts.handleRiskLevelChange(newRiskLevel);
        setShowSpikeModal(true);
      } else if (newRiskLevel === 'Normal') {
        soundAlerts.handleRiskLevelChange('Normal');
        setShowSpikeModal(false);
      }
    }
  };

  // Execute real ML prediction pipeline
  async function executePrediction(sequence) {
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setPredictionData(data);
      setIsConnected(true);
      setApiError(null);

      // Evaluate risk transitions
      if (data.hsrae?.overall_risk_level) {
        handleRiskTransition(data.hsrae.overall_risk_level, data);
      }
      return data;
    } catch (err) {
      console.error('Prediction request failed:', err);
      setIsConnected(false);
      setApiError('Prediction service request failed. Check API connection.');
      return null;
    }
  }

  // Auto Live Stream Replay Interval (Main Monitoring Dashboard)
  useEffect(() => {
    if (!isStreaming || isSimulatingSpike || isSpikeActive) return;

    const interval = setInterval(async () => {
      try {
        const noiseParam = applyNoise ? 0.015 : 0.0;
        const res = await fetch(
          `/api/stream/tick?index=${streamIndex}&noise=${noiseParam}&step=1`
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        setPredictionData(data);
        setCurrentTimestamp(data.timestamp);
        setStreamIndex(data.next_index);
        if (data.total_rows) setTotalRows(data.total_rows);
        setIsConnected(true);
        setApiError(null);

        if (data.hsrae?.overall_risk_level) {
          handleRiskTransition(data.hsrae.overall_risk_level, data);
        }
      } catch (err) {
        console.error('Stream tick error:', err);
        setIsConnected(false);
      }
    }, streamSpeed);

    return () => clearInterval(interval);
  }, [isStreaming, isSimulatingSpike, isSpikeActive, streamSpeed, applyNoise, streamIndex]);

  // Toggle Live Replay Stream
  const handleToggleStream = () => {
    soundAlerts.initContext();
    setIsStreaming((prev) => !prev);
  };

  // Jump to specific historical event
  const handleJumpToIndex = async (newIndex) => {
    soundAlerts.initContext();
    setStreamIndex(newIndex);
    setIsSpikeActive(false);
    setIsSimulatingSpike(false);
    if (spikeTimerRef.current) clearTimeout(spikeTimerRef.current);
    if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);

    try {
      const noiseParam = applyNoise ? 0.015 : 0.0;
      const res = await fetch(
        `/api/stream/tick?index=${newIndex}&noise=${noiseParam}&step=1`
      );
      if (res.ok) {
        const data = await res.json();
        setPredictionData(data);
        setCurrentTimestamp(data.timestamp);
        setStreamIndex(data.next_index);
        if (data.hsrae?.overall_risk_level) {
          handleRiskTransition(data.hsrae.overall_risk_level, data);
        }
      }
    } catch (err) {
      console.error('Jump error:', err);
    }
  };

  // Mute / Unmute Audio Alerts
  const handleToggleMute = () => {
    soundAlerts.initContext();
    const newMuted = !isMuted;
    setIsMuted(newMuted);
    soundAlerts.setMuted(newMuted);
  };

  // 🔴 WHAT-IF SPIKE SIMULATION: Gradual multi-step industrial surge
  const handleStartSpikeSimulation = async () => {
    soundAlerts.initContext();

    setIsStreaming(false);
    setIsSimulatingSpike(true);
    setIsSpikeActive(true);
    setElapsedSeconds(0);

    if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
    elapsedTimerRef.current = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);

    const baseSeq = (samples && samples['normal']?.data) || currentSequence || [
      [7.2, 110, 55, 520, 26.5],
    ];
    setBaselineSequence(baseSeq);

    const executeStep = async (stepIndex) => {
      if (stepIndex >= SPIKE_STEPS.length) {
        setIsSimulatingSpike(false);
        return;
      }

      const stepConfig = SPIKE_STEPS[stepIndex];
      setSpikeStep(stepConfig.step);
      setCurrentStepLabel(stepConfig.label);

      const perturbedSeq = baseSeq.map((row, idx) => {
        if (idx >= 14) {
          const rampFactor = (idx - 13) / 10.0;
          const offsets = stepConfig.offsets;
          return [
            parseFloat(Math.max(4.0, Math.min(10.0, row[0] + offsets.pH * rampFactor)).toFixed(2)),
            parseFloat(Math.max(0, row[1] + offsets.COD * rampFactor).toFixed(1)),
            parseFloat(Math.max(0, row[2] + offsets.BOD * rampFactor).toFixed(1)),
            parseFloat(Math.max(0, row[3] + offsets.TDS * rampFactor).toFixed(1)),
            parseFloat(Math.max(10, row[4] + offsets.Temperature * rampFactor).toFixed(1)),
          ];
        }
        return [...row];
      });

      setCurrentSequence(perturbedSeq);
      await executePrediction(perturbedSeq);

      spikeTimerRef.current = setTimeout(() => {
        executeStep(stepIndex + 1);
      }, 1300);
    };

    executeStep(0);
  };

  // Reset to Normal Baseline
  const handleResetBaseline = async () => {
    soundAlerts.initContext();
    if (spikeTimerRef.current) clearTimeout(spikeTimerRef.current);
    if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);

    setIsSimulatingSpike(false);
    setIsSpikeActive(false);
    setSpikeStep(0);
    setCurrentStepLabel('');
    setElapsedSeconds(0);
    setShowSpikeModal(false);

    prevRiskLevelRef.current = 'Normal';
    soundAlerts.handleRiskLevelChange('Normal');

    const normalSeq = (samples && samples['normal']?.data) || baselineSequence;
    if (normalSeq) {
      setCurrentSequence(normalSeq);
      await executePrediction(normalSeq);
    }
  };

  useEffect(() => {
    return () => {
      if (spikeTimerRef.current) clearTimeout(spikeTimerRef.current);
      if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
      soundAlerts.stopAll();
    };
  }, []);

  return (
    <div className="app-layout" onClick={() => soundAlerts.initContext()}>
      {/* Slim Expandable Environmental Navigation Rail (Reference Image 2) */}
      <NavigationRail
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        isConnected={isConnected}
      />

      {/* Main Content Area */}
      <div className="app-main-wrapper">
        {/* Minimal Clean Header */}
        <Header
          status={predictionData?.hsrae?.overall_risk_level}
          hsraeResult={predictionData?.hsrae}
          isConnected={isConnected}
          isMuted={isMuted}
          onToggleMute={handleToggleMute}
        />

        {/* Dynamic Page Views */}
        <main className="main-content">
          {/* Connection Alert if backend is offline */}
          {apiError && (
            <div
              style={{
                padding: '12px 18px',
                backgroundColor: '#fef2f2',
                border: '1px solid #fecaca',
                borderRadius: '12px',
                marginBottom: '18px',
                color: '#991b1b',
                fontSize: '13px',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
              }}
            >
              <span>
                <strong>System Notice:</strong> {apiError} (Ensure FastAPI backend is active on port 8000)
              </span>
            </div>
          )}

          {/* TAB 1: Main Dashboard (MONITORING ONLY) */}
          {activeTab === 'overview' && (
            <div>
              {/* Dynamic Alert Banner (Normal / Warning / Critical) */}
              <AlertBanner
                hsraeResult={predictionData?.hsrae}
                autoencoderData={predictionData?.autoencoder}
                isMuted={isMuted}
                onToggleMute={handleToggleMute}
              />

              {/* Operator Controls / Historical Replay Bar (Monitoring Only) */}
              <OperatorControls
                isStreaming={isStreaming}
                onToggleStream={handleToggleStream}
                streamSpeed={streamSpeed}
                onChangeSpeed={setStreamSpeed}
                applyNoise={applyNoise}
                onToggleNoise={setApplyNoise}
                currentTimestamp={currentTimestamp}
                currentIndex={streamIndex}
                totalRows={totalRows}
                onJumpToIndex={handleJumpToIndex}
                isMuted={isMuted}
                onToggleMute={handleToggleMute}
              />

              {/* 5 Real-Time Wastewater Metric Cards (Reference Image 1 Green Design) */}
              <MetricCards
                currentValues={predictionData?.current_values}
                hsraeEvaluations={predictionData?.hsrae?.parameter_evaluations}
                thresholds={config?.thresholds}
                activeParam={activeParam}
                onSelectParam={setActiveParam}
              />

              {/* 48-Hour Continuous Telemetry & TCN Forecast Chart */}
              <ForecastChart
                historyTimeline={predictionData?.history_timeline}
                forecastTimeline={predictionData?.forecast_timeline}
                thresholds={config?.thresholds}
                activeParam={activeParam}
                onSelectParam={setActiveParam}
              />

              {/* HSRAE Risk Engine & Autoencoder Anomaly Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
                <HsraePanel hsraeResult={predictionData?.hsrae} />
                <AutoencoderView autoencoderData={predictionData?.autoencoder} />
              </div>
            </div>
          )}

          {/* TAB 2: What-If Spike Simulation Page (Dedicated page with red Simulate Spike button) */}
          {activeTab === 'whatif' && (
            <WhatIfStudio
              isSimulatingSpike={isSimulatingSpike}
              isSpikeActive={isSpikeActive}
              spikeStep={spikeStep}
              totalSpikeSteps={SPIKE_STEPS.length}
              elapsedSeconds={elapsedSeconds}
              currentStepLabel={currentStepLabel}
              onStartSpikeSimulation={handleStartSpikeSimulation}
              onResetBaseline={handleResetBaseline}
              predictionData={predictionData}
              config={config}
              activeParam={activeParam}
              onSelectParam={setActiveParam}
            />
          )}

          {/* TAB 3: HSRAE Risk Decision Engine */}
          {activeTab === 'hsrae' && (
            <div>
              <HsraePanel hsraeResult={predictionData?.hsrae} />
              <ForecastChart
                historyTimeline={predictionData?.history_timeline}
                forecastTimeline={predictionData?.forecast_timeline}
                thresholds={config?.thresholds}
                activeParam={activeParam}
                onSelectParam={setActiveParam}
              />
            </div>
          )}

          {/* TAB 4: Autoencoder Anomaly Detection */}
          {activeTab === 'autoencoder' && (
            <div>
              <AutoencoderView autoencoderData={predictionData?.autoencoder} />
              <ForecastChart
                historyTimeline={predictionData?.history_timeline}
                forecastTimeline={predictionData?.forecast_timeline}
                thresholds={config?.thresholds}
                activeParam={activeParam}
                onSelectParam={setActiveParam}
              />
            </div>
          )}
        </main>

        {/* Warning / Critical Alert Popup Modal */}
        <SpikeAlertModal
          isOpen={showSpikeModal}
          onClose={() => setShowSpikeModal(false)}
          hsraeResult={predictionData?.hsrae}
          currentValues={predictionData?.current_values}
          thresholds={config?.thresholds}
        />

        {/* Environmental Industrial Telemetry Footer */}
        <footer className="app-footer">
          <div>
            <strong>Industrial Wastewater Early Spike Prediction System</strong> • Continuous Deep Learning Telemetry
          </div>
          <div style={{ fontFamily: 'var(--font-mono)' }}>
            TCN v2 (Spike-Aware) • Autoencoder (Reconstruction MSE) • HSRAE v2.0
          </div>
        </footer>
      </div>
    </div>
  );
}
