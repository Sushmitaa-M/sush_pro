/**
 * Web Audio API Industrial Alert Sound Engine
 * Provides synthesized warning chimes and critical alarm sirens without external audio files.
 * Complies with browser autoplay policies via lazy AudioContext initialization.
 */

class SoundAlertManager {
  constructor() {
    this.audioCtx = null;
    this.isMuted = false;
    this.criticalInterval = null;
    this.currentAlertLevel = 'Normal';
  }

  // Ensure AudioContext is initialized on user interaction
  initContext() {
    if (!this.audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        this.audioCtx = new AudioContextClass();
      }
    }
    if (this.audioCtx && this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
  }

  setMuted(muted) {
    this.isMuted = muted;
    if (muted) {
      this.stopAll();
    }
  }

  // Play a single clean warning chime (Dual-frequency high/mid tone)
  playWarningTone() {
    if (this.isMuted) return;
    this.initContext();
    if (!this.audioCtx) return;

    try {
      const now = this.audioCtx.currentTime;

      // Tone 1: 660 Hz
      const osc1 = this.audioCtx.createOscillator();
      const gain1 = this.audioCtx.createGain();
      osc1.type = 'sine';
      osc1.frequency.setValueAtTime(660, now);
      gain1.gain.setValueAtTime(0.2, now);
      gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
      osc1.connect(gain1);
      gain1.connect(this.audioCtx.destination);
      osc1.start(now);
      osc1.stop(now + 0.35);

      // Tone 2: 880 Hz
      const osc2 = this.audioCtx.createOscillator();
      const gain2 = this.audioCtx.createGain();
      osc2.type = 'sine';
      osc2.frequency.setValueAtTime(880, now + 0.15);
      gain2.gain.setValueAtTime(0.001, now);
      gain2.gain.setValueAtTime(0.25, now + 0.15);
      gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.55);
      osc2.connect(gain2);
      gain2.connect(this.audioCtx.destination);
      osc2.start(now + 0.15);
      osc2.stop(now + 0.6);
    } catch (e) {
      console.warn('Warning audio failed:', e);
    }
  }

  // Start continuous critical alert siren pulses (alternating two-tone pulse)
  startCriticalSiren() {
    if (this.isMuted) return;
    this.initContext();
    if (!this.audioCtx) return;

    // Avoid duplicate intervals
    if (this.criticalInterval) return;

    const playPulse = () => {
      if (this.isMuted || !this.audioCtx) return;
      try {
        const now = this.audioCtx.currentTime;
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();

        osc.type = 'sawtooth';
        // Alternating siren frequency sweep from 750Hz to 950Hz
        osc.frequency.setValueAtTime(750, now);
        osc.frequency.linearRampToValueAtTime(950, now + 0.18);
        osc.frequency.linearRampToValueAtTime(700, now + 0.35);

        gain.gain.setValueAtTime(0.18, now);
        gain.gain.linearRampToValueAtTime(0.22, now + 0.18);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.38);

        osc.connect(gain);
        gain.connect(this.audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.4);
      } catch (e) {
        console.warn('Critical siren pulse failed:', e);
      }
    };

    // Play immediate first pulse and loop every 800ms
    playPulse();
    this.criticalInterval = setInterval(playPulse, 800);
  }

  stopCriticalSiren() {
    if (this.criticalInterval) {
      clearInterval(this.criticalInterval);
      this.criticalInterval = null;
    }
  }

  stopAll() {
    this.stopCriticalSiren();
  }

  /**
   * Updates audio state when risk level changes.
   * Only triggers sound on transition to prevent spamming on unchanged state.
   */
  handleRiskLevelChange(newLevel) {
    if (newLevel === this.currentAlertLevel) {
      return; // No state change, do not re-trigger
    }

    this.currentAlertLevel = newLevel;

    if (newLevel === 'Normal') {
      this.stopAll();
    } else if (newLevel === 'Warning') {
      this.stopCriticalSiren();
      this.playWarningTone();
    } else if (newLevel === 'Critical') {
      this.startCriticalSiren();
    }
  }
}

export const soundAlerts = new SoundAlertManager();
export default soundAlerts;
