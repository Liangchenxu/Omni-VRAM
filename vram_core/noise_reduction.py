"""
Noise Reduction Module for Omni-VRAM
=====================================

Spectral subtraction-based noise reduction using only numpy and scipy.
Designed for real-time audio preprocessing before VAD and ASR.

Algorithm:
    1. FFT transform to frequency domain
    2. Estimate noise spectrum from initial silent frames
    3. Spectral subtraction to suppress noise
    4. IFFT to recover time-domain signal

Usage:
    from vram_core.noise_reduction import NoiseReducer

    reducer = NoiseReducer(strength="medium")
    clean_audio = reducer.process(audio_array, sample_rate=16000)
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import stft, istft

logger = logging.getLogger(__name__)


class NoiseStrength(Enum):
    """Noise reduction strength presets."""
    LIGHT = "light"
    MEDIUM = "medium"
    AGGRESSIVE = "aggressive"


# Preset parameters for each strength level
_STRENGTH_PRESETS = {
    NoiseStrength.LIGHT: {
        "alpha": 1.0,           # Over-subtraction factor
        "beta": 0.02,           # Spectral floor (higher = less aggressive)
        "noise_frames": 6,      # Frames for noise estimation
    },
    NoiseStrength.MEDIUM: {
        "alpha": 2.0,
        "beta": 0.01,
        "noise_frames": 8,
    },
    NoiseStrength.AGGRESSIVE: {
        "alpha": 4.0,
        "beta": 0.005,
        "noise_frames": 12,
    },
}


@dataclass
class NoiseReductionResult:
    """Result of noise reduction processing."""
    audio: np.ndarray
    noise_estimate: np.ndarray
    snr_before: float
    snr_after: float
    frames_processed: int


class NoiseReducer:
    """
    Spectral subtraction-based noise reducer.

    Uses Short-Time Fourier Transform (STFT) to analyze audio in the
    frequency domain, estimates noise from initial silent frames, and
    subtracts the estimated noise spectrum from the signal.

    Args:
        strength: Noise reduction strength ("light", "medium", "aggressive").
        alpha: Over-subtraction factor (higher = more aggressive).
               Overrides strength preset if provided.
        beta: Spectral floor factor (higher = less distortion).
              Overrides strength preset if provided.
        noise_frames: Number of initial frames for noise estimation.
                      Overrides strength preset if provided.
        frame_length: STFT frame length in samples (default 512).
        hop_length: STFT hop length in samples (default 256).

    Usage:
        reducer = NoiseReducer(strength="medium")
        clean = reducer.process(noisy_audio, sample_rate=16000)

        # Or with custom parameters
        reducer = NoiseReducer(alpha=2.5, beta=0.01, noise_frames=10)
        clean = reducer.process(noisy_audio, sample_rate=16000)
    """

    def __init__(
        self,
        strength: str = "medium",
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        noise_frames: Optional[int] = None,
        frame_length: int = 512,
        hop_length: int = 256,
    ):
        self.frame_length = frame_length
        self.hop_length = hop_length

        # Load strength preset
        try:
            preset = NoiseStrength(strength)
        except ValueError:
            logger.warning(
                f"Unknown strength '{strength}', falling back to 'medium'"
            )
            preset = NoiseStrength.MEDIUM

        defaults = _STRENGTH_PRESETS[preset]

        # Apply overrides or defaults
        self.alpha = alpha if alpha is not None else defaults["alpha"]
        self.beta = beta if beta is not None else defaults["beta"]
        self.noise_frames = (
            noise_frames if noise_frames is not None
            else defaults["noise_frames"]
        )

        self.strength = preset

        logger.info(
            "NoiseReducer initialized: strength=%s, alpha=%.1f, beta=%.3f, "
            "noise_frames=%d",
            self.strength.value, self.alpha, self.beta, self.noise_frames,
        )

    def estimate_noise_spectrum(
        self,
        magnitude: np.ndarray,
        n_noise_frames: Optional[int] = None,
    ) -> np.ndarray:
        """
        Estimate noise power spectrum from initial frames.

        Uses the mean of the first N frames as the noise estimate,
        assuming the recording starts with a brief silence.

        Args:
            magnitude: STFT magnitude spectrogram (freq_bins, time_frames).
            n_noise_frames: Number of frames to use for estimation.

        Returns:
            Estimated noise magnitude spectrum (freq_bins,).
        """
        n = n_noise_frames or self.noise_frames
        n = min(n, magnitude.shape[1])

        if n == 0:
            return np.zeros(magnitude.shape[0], dtype=np.float32)

        # Mean magnitude of first N frames as noise estimate
        noise_estimate = np.mean(magnitude[:, :n], axis=1)
        return noise_estimate.astype(np.float32)

    def spectral_subtract(
        self,
        magnitude: np.ndarray,
        noise_estimate: np.ndarray,
    ) -> np.ndarray:
        """
        Apply spectral subtraction to magnitude spectrogram.

        Subtracts the estimated noise magnitude from each frame, with
        an over-subtraction factor and a spectral floor to prevent
        musical noise artifacts.

        Formula:
            |S_clean|^2 = |S_noisy|^2 - alpha * |N|^2
            |S_clean| = sqrt(max(|S_clean|^2, beta * |S_noisy|^2))

        Args:
            magnitude: Noisy magnitude spectrogram (freq_bins, time_frames).
            noise_estimate: Estimated noise magnitude spectrum (freq_bins,).

        Returns:
            Clean magnitude spectrogram (freq_bins, time_frames).
        """
        # Squared magnitudes
        mag_sq = magnitude ** 2
        noise_sq = noise_estimate ** 2

        # Broadcast noise estimate across time frames
        noise_sq_expanded = noise_sq[:, np.newaxis]

        # Spectral subtraction with over-subtraction factor
        clean_sq = mag_sq - self.alpha * noise_sq_expanded

        # Apply spectral floor (beta fraction of original)
        spectral_floor = self.beta * mag_sq
        clean_sq = np.maximum(clean_sq, spectral_floor)

        # Ensure non-negative
        clean_sq = np.maximum(clean_sq, 0.0)

        return np.sqrt(clean_sq).astype(np.float32)

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Apply noise reduction to an audio signal.

        Args:
            audio: Input audio signal (float32, mono).
            sample_rate: Sample rate in Hz.

        Returns:
            Noise-reduced audio signal (float32).
        """
        if len(audio) == 0:
            return audio

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Adjust hop/frame length for very short audio
        frame_length = min(self.frame_length, len(audio))
        hop_length = min(self.hop_length, frame_length // 2)

        # STFT
        freqs, times, Zxx = stft(
            audio,
            fs=sample_rate,
            nperseg=frame_length,
            noverlap=frame_length - hop_length,
        )

        # Magnitude and phase
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Estimate noise from initial frames
        noise_estimate = self.estimate_noise_spectrum(magnitude)

        # Spectral subtraction
        clean_magnitude = self.spectral_subtract(magnitude, noise_estimate)

        # Reconstruct complex spectrogram with clean magnitude + original phase
        clean_Zxx = clean_magnitude * np.exp(1j * phase)

        # ISTFT to recover time-domain signal
        _, clean_audio = istft(
            clean_Zxx,
            fs=sample_rate,
            nperseg=frame_length,
            noverlap=frame_length - hop_length,
        )

        # Match output length to input
        clean_audio = clean_audio[:len(audio)]

        # If output is shorter (edge case), pad with zeros
        if len(clean_audio) < len(audio):
            clean_audio = np.pad(
                clean_audio,
                (0, len(audio) - len(clean_audio)),
                mode="constant",
            )

        return clean_audio.astype(np.float32)

    def process_with_stats(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> NoiseReductionResult:
        """
        Apply noise reduction and return detailed statistics.

        Args:
            audio: Input audio signal (float32, mono).
            sample_rate: Sample rate in Hz.

        Returns:
            NoiseReductionResult with cleaned audio and statistics.
        """
        if len(audio) == 0:
            return NoiseReductionResult(
                audio=audio,
                noise_estimate=np.array([]),
                snr_before=0.0,
                snr_after=0.0,
                frames_processed=0,
            )

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        frame_length = min(self.frame_length, len(audio))
        hop_length = min(self.hop_length, frame_length // 2)

        # STFT
        freqs, times, Zxx = stft(
            audio,
            fs=sample_rate,
            nperseg=frame_length,
            noverlap=frame_length - hop_length,
        )

        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        noise_estimate = self.estimate_noise_spectrum(magnitude)
        clean_magnitude = self.spectral_subtract(magnitude, noise_estimate)

        clean_Zxx = clean_magnitude * np.exp(1j * phase)
        _, clean_audio = istft(
            clean_Zxx,
            fs=sample_rate,
            nperseg=frame_length,
            noverlap=frame_length - hop_length,
        )
        clean_audio = clean_audio[:len(audio)]
        if len(clean_audio) < len(audio):
            clean_audio = np.pad(
                clean_audio,
                (0, len(audio) - len(clean_audio)),
                mode="constant",
            )
        clean_audio = clean_audio.astype(np.float32)

        # Compute SNR estimates
        noise_frames = min(self.noise_frames, magnitude.shape[1])
        signal_power = np.mean(magnitude[:, noise_frames:] ** 2)
        noise_power = np.mean(noise_estimate ** 2) + 1e-10

        snr_before = float(10 * np.log10(signal_power / (noise_power + 1e-10)))

        clean_noise = clean_magnitude[:, :noise_frames]
        clean_signal = clean_magnitude[:, noise_frames:]
        clean_noise_power = np.mean(clean_noise ** 2) + 1e-10
        clean_signal_power = np.mean(clean_signal ** 2)
        snr_after = float(
            10 * np.log10(clean_signal_power / (clean_noise_power + 1e-10))
        )

        return NoiseReductionResult(
            audio=clean_audio,
            noise_estimate=noise_estimate,
            snr_before=snr_before,
            snr_after=snr_after,
            frames_processed=magnitude.shape[1],
        )

    @staticmethod
    def create_preset(strength: str = "medium") -> "NoiseReducer":
        """
        Create a NoiseReducer with preset parameters.

        Args:
            strength: One of "light", "medium", "aggressive".

        Returns:
            Configured NoiseReducer instance.
        """
        return NoiseReducer(strength=strength)