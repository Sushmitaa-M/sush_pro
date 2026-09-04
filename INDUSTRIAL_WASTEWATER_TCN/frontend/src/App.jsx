import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import LiveStreamBar from './components/LiveStreamBar';
import MetricCards from './components/MetricCards';
import ForecastChart from './components/ForecastChart';
import HsraePanel from './components/HsraePanel';
import AutoencoderView from './components/AutoencoderView';
import Simulator from './components/Simulator';
import BenchmarksView from './components/BenchmarksView';
import { Activity, ShieldAlert, Cpu, Sliders, BarChart3 } from 'lucide-react';
import './App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('telemetry');
  const [isConnected, setIsConnected] = useState(false);
  const [config, setConfig] = useState(null);
  const [samples, setSamples] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [activeSampleKey, setActiveSampleKey] = useState('normal');
  const [currentSequence, setCurrentSequence] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSimulating, setIsSimulating] = useState(false);

  const [activeParam, setActiveParam] = useState('COD');

  // Auto Live Stream State (Defaulting to ON for instant live experience)
  const [isStreaming, setIsStreaming] = useState(true);
  const [streamSpeed, setStreamSpeed] = useState(2000);
  const [applyNoise, setApplyNoise] = useState(true);
  const [streamIndex, setStreamIndex] = useState(120);
  const [currentTimestamp, setCurrentTimestamp] = useState('');
  const [totalRows, setTotalRows] = useState(10000);

  // Initialize data on mount
  useEffect(() => {
    async function initApp() {
      try {
        const [statusRes, configRes, samplesRes, metricsRes] = await Promise.all([
          fetch('/api/status').then((r) => r.json()).catch(() => null),
          fetch('/api/config').then((r) => r.json()).catch(() => null),
          fetch('/api/samples').then((r) => r.json()).catch(() => null),
          fetch('/api/metrics').then((r) => r.json()).catch(() => null),
        ]);

        if (statusRes && statusRes.status === 'ready') {
          setIsConnected(true);
        }
        if (configRes) setConfig(configRes);
        if (samplesRes) {
          setSamples(samplesRes);
          const initialKey = 'normal';
          const initialSeq = samplesRes[initialKey]?.data;
          if (initialSeq) {
            setCurrentSequence(initialSeq);
            await fetchPrediction(initialSeq);
          }
        }
        if (metricsRes) setMetrics(metricsRes);
      } catch (err) {
        console.error('Initialization error:', err);
      } finally {
        setIsLoading(false);
      }
    }

    initApp();
  }, []);

  // Fetch prediction from backend for static sequence
  async function fetchPrediction(sequence) {
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
    } catch (err) {
      console.error('Prediction request failed:', err);
    }
  }

  // Auto Live Stream Interval
  useEffect(() => {
    if (!isStreaming) return;

    const interval = setInterval(async () => {
      try {
        const noiseParam = applyNoise ? 0.015 : 0.0;
        const res = await fetch(
          `/api/stream/tick?index=${streamIndex}&noise=${noiseParam}&step=1`
        );
        if (!res.ok) return;
        const data = await res.json();
        setPredictionData(data);
        setCurrentTimestamp(data.timestamp);
        setStreamIndex(data.next_index);
        if (data.total_rows) setTotalRows(data.total_rows);
      } catch (err) {
        console.error('Stream tick error:', err);
      }
    }, streamSpeed);

    return () => clearInterval(interval);
  }, [isStreaming, streamSpeed, applyNoise, streamIndex]);

  const handleToggleStream = () => {
    setIsStreaming((prev) => !prev);
  };

  const handleJumpToIndex = async (newIndex) => {
    setStreamIndex(newIndex);
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
      }
    } catch (err) {
      console.error('Jump error:', err);
    }
  };

  // Switch sample scenario manually
  const handleSelectSample = async (sampleKey) => {
    if (!samples || !samples[sampleKey]) return;
    setIsStreaming(false); // pause live stream when manually selecting static scenario
    setActiveSampleKey(sampleKey);
    const seq = samples[sampleKey].data;
    setCurrentSequence(seq);
    setIsLoading(true);
    await fetchPrediction(seq);
    setIsLoading(false);
  };

  // Run What-If simulation with parameter offsets
  const handleRunSimulation = async (offsets) => {
    if (!currentSequence) return;
    setIsStreaming(false);
    setIsSimulating(true);

    // Apply offsets to the last 8 hours of the 24h window
    const modifiedSeq = currentSequence.map((row, idx) => {
      if (idx >= 16) {
        return [
          parseFloat((row[0] + offsets.pH).toFixed(2)),
          Math.max(0, parseFloat((row[1] + offsets.COD).toFixed(1))),
          Math.max(0, parseFloat((row[2] + offsets.BOD).toFixed(1))),
          Math.max(0, parseFloat((row[3] + offsets.TDS).toFixed(1))),
          parseFloat((row[4] + offsets.Temperature).toFixed(1)),
        ];
      }
      return [...row];
    });

    await fetchPrediction(modifiedSeq);
    setIsSimulating(false);
  };

  // Tab definitions
  const tabs = [
    { id: 'telemetry', label: 'Overview & 24h Forecasting', icon: Activity },
    { id: 'hsrae', label: 'HSRAE Risk Decision Engine', icon: ShieldAlert },
    { id: 'autoencoder', label: 'Autoencoder Anomaly Detection', icon: Cpu },
    { id: 'simulator', label: 'Interactive "What-If" Studio', icon: Sliders },
    { id: 'benchmarks', label: 'Model Benchmarks', icon: BarChart3 },
  ];

  return (
    <div className="app-container">
      {/* Top Header */}
      <Header
        status={predictionData?.hsrae?.overall_risk_level}
        hsraeResult={predictionData?.hsrae}
        isConnected={isConnected}
      />

      {/* Navigation Tabs */}
      <nav className="nav-bar">
        <div className="nav-inner">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`nav-tab ${activeTab === id ? 'active' : ''}`}
              onClick={() => setActiveTab(id)}
            >
              <Icon size={15} />
              <span>{label}</span>
            </button>
          ))}
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="main-content">
        {/* Auto Live Stream Controller Bar */}
        <LiveStreamBar
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
        />

        {/* Metric Cards Banner (Always visible across tabs for continuous plant awareness) */}
        <MetricCards
          currentValues={predictionData?.current_values}
          hsraeEvaluations={predictionData?.hsrae?.parameter_evaluations}
          thresholds={config?.thresholds}
          activeParam={activeParam}
          onSelectParam={setActiveParam}
        />

        {/* Tab 1: Overview & 24h Forecasting */}
        {activeTab === 'telemetry' && (
          <div>
            <ForecastChart
              historyTimeline={predictionData?.history_timeline}
              forecastTimeline={predictionData?.forecast_timeline}
              thresholds={config?.thresholds}
              activeParam={activeParam}
              onSelectParam={setActiveParam}
            />

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
              <HsraePanel hsraeResult={predictionData?.hsrae} />
              <AutoencoderView autoencoderData={predictionData?.autoencoder} />
            </div>
          </div>
        )}

        {/* Tab 2: HSRAE Risk Decision Engine */}
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

        {/* Tab 3: Autoencoder Anomaly Detection */}
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

        {/* Tab 4: Interactive "What-If" Studio */}
        {activeTab === 'simulator' && (
          <div>
            <Simulator
              samples={samples}
              activeSampleKey={activeSampleKey}
              onSelectSample={handleSelectSample}
              onRunSimulation={handleRunSimulation}
              isSimulating={isSimulating}
            />

            <ForecastChart
              historyTimeline={predictionData?.history_timeline}
              forecastTimeline={predictionData?.forecast_timeline}
              thresholds={config?.thresholds}
              activeParam={activeParam}
              onSelectParam={setActiveParam}
            />

            <HsraePanel hsraeResult={predictionData?.hsrae} />
          </div>
        )}

        {/* Tab 5: Model Benchmarks */}
        {activeTab === 'benchmarks' && (
          <BenchmarksView metricsData={metrics} />
        )}
      </main>

      {/* Minimal Footer */}
      <footer className="app-footer">
        <div>
          <strong>Industrial Wastewater Early Spike Prediction System</strong> • Deep Learning &amp; Temporal Convolutional Network
        </div>
        <div style={{ fontFamily: 'var(--font-mono)' }}>
          TCN v2 (Spike-Aware) • Autoencoder (Recon MSE) • HSRAE v2.0
        </div>
      </footer>
    </div>
  );
}
