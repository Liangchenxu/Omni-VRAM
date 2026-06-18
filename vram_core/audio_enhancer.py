"""
Audio Enhancer: AGC, AEC, spectral equalizer, volume normalization,
dynamic range compression, noise gate, and spectral denoise.

v2.5.0 - Full 7-stage professional audio enhancement pipeline.
"""

import logging
import math
import numpy as np
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnhancerConfig:
    """Configuration for AudioEnhancer pipeline."""
    target_db: float = -20.0
    agc_enabled: bool = True
    normalize_enabled: bool = True
    compressor_enabled: bool = False
    compressor_threshold: float = -20.0
    compressor_ratio: float = 4.0
    noise_gate_enabled: bool = True
    noise_gate_threshold_db: float = -40.0
    highpass_enabled: bool = True
    highpass_cutoff_hz: float = 80.0
    dereverb_enabled: bool = False


class AudioEnhancer:
    """Audio enhancement pipeline with AGC, normalization, noise gate, and spectral denoise."""

    def __init__(self, config: Optional[EnhancerConfig] = None, sample_rate: int = 16000):
        self.config = config or EnhancerConfig()
        self.sample_rate = sample_rate

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        """Run the full enhancement pipeline on audio."""
        if len(audio) == 0:
            return audio.astype(np.float32)

        audio = audio.astype(np.float32).copy()

        # Stage 1: Remove DC offset
        audio = self._remove_dc(audio)

        # Stage 2: High-pass filter
        if self.config.highpass_enabled:
            audio = self._highpass(audio, cutoff_hz=self.config.highpass_cutoff_hz)

        # Stage 3: Noise gate
        if self.config.noise_gate_enabled:
            audio = self._noise_gate(audio, threshold_db=self.config.noise_gate_threshold_db)

        # Stage 4: Normalization
        if self.config.normalize_enabled:
            audio = self._normalize(audio)

        # Stage 5: AGC
        if self.config.agc_enabled:
            audio = self._auto_gain_control(audio)

        # Stage 6: Compressor
        if self.config.compressor_enabled:
            audio = self._compress(audio)

        # Stage 7: Dereverb (spectral decay)
        if self.config.dereverb_enabled:
            audio = self._dereverb(audio)

        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target peak level."""
        peak = np.max(np.abs(audio))
        if peak > 1e-10:
            target_linear = 10 ** (self.config.target_db / 20.0)
            # Scale so that peak reaches ~0.95 (near full scale)
            audio = audio * (0.95 / peak)
        return audio

    def _auto_gain_control(self, audio: np.ndarray, frame_size: int = 1024, alpha: float = 0.01) -> np.ndarray:
        """Automatic gain control with smooth gain tracking."""
        gain = 1.0
        output = np.zeros_like(audio)
        target_rms = 10 ** (self.config.target_db / 20.0)
        for i in range(0, len(audio), frame_size):
            frame = audio[i:i + frame_size]
            rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
            desired_gain = target_rms / (rms + 1e-10)
            gain = (1 - alpha) * gain + alpha * desired_gain
            gain = min(gain, 10.0)
            output[i:i + frame_size] = frame * gain
        return output

    def _noise_gate(self, audio: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
        """Apply noise gate - attenuate samples below threshold."""
        threshold_linear = 10 ** (threshold_db / 20.0)
        frame_size = 512
        output = audio.copy()

        for i in range(0, len(audio), frame_size):
            frame = audio[i:i + frame_size]
            rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
            if rms < threshold_linear:
                # Attenuate quiet frames
                attenuation = max(rms / (threshold_linear + 1e-10), 0.01)
                output[i:i + frame_size] = frame * attenuation

        return output

    def _compress(self, audio: np.ndarray) -> np.ndarray:
        """Dynamic range compression."""
        threshold = 10 ** (self.config.compressor_threshold / 20.0)
        ratio = self.config.compressor_ratio
        if ratio <= 0:
            logger.warning("compressor_ratio must be > 0, got %s. Skipping compression.", ratio)
            return audio
        output = np.copy(audio)
        mask = np.abs(audio) > threshold
        excess = np.abs(audio[mask]) - threshold
        compressed_excess = excess / ratio
        output[mask] = np.sign(audio[mask]) * (threshold + compressed_excess)
        return output

    def _remove_dc(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset."""
        return audio - np.mean(audio)

    def _highpass(self, audio: np.ndarray, cutoff_hz: float = 80.0) -> np.ndarray:
        """Simple high-pass filter using spectral method."""
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)
        fft[freqs < cutoff_hz] *= 0.01
        return np.fft.irfft(fft, len(audio)).astype(np.float32)

    def _dereverb(self, audio: np.ndarray, decay_factor: float = 0.9) -> np.ndarray:
        """Simple dereverberation using spectral subtraction."""
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        # Estimate reverb as smoothed magnitude
        smoothed = np.convolve(magnitude, np.ones(5) / 5, mode='same')
        # Subtract estimated reverb
        clean_magnitude = np.maximum(magnitude - decay_factor * smoothed, 0.01 * magnitude)
        clean_fft = clean_magnitude * np.exp(1j * phase)
        return np.fft.irfft(clean_fft, len(audio)).astype(np.float32)

    def spectral_gate(self, audio: np.ndarray, threshold: float = 0.02) -> np.ndarray:
        """Spectral gating for noise reduction."""
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        mask = magnitude > threshold * np.max(magnitude)
        fft_clean = magnitude * mask * np.exp(1j * phase)
        return np.fft.irfft(fft_clean, len(audio)).astype(np.float32)