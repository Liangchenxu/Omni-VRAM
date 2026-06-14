"""
Emotion Recognition Module for Omni-VRAM
=========================================

Lightweight emotion classification based on audio signal features.
No pretrained models required — uses handcrafted acoustic features:
    - Energy (loudness)
    - Zero-crossing rate (speech rate proxy)
    - Fundamental frequency F0 (pitch)
    - Rhythm/tempo variation

Supported emotions: happy, sad, angry, neutral, surprised

Usage:
    from vram_core.emotion_recognition import EmotionRecognizer

    recognizer = EmotionRecognizer()
    result = recognizer.analyze(audio_array, sample_rate=16000)
    print(result.emotion, result.confidence)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """Extracted audio features for emotion classification."""
    rms_energy: float = 0.0
    zero_crossing_rate: float = 0.0
    mean_f0: float = 0.0
    std_f0: float = 0.0
    energy_variance: float = 0.0
    energy_range: float = 0.0
    speech_rate_proxy: float = 0.0


@dataclass
class EmotionResult:
    """Result of emotion analysis."""
    emotion: str
    confidence: float
    features: AudioFeatures
    all_scores: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"EmotionResult(emotion='{self.emotion}', "
            f"confidence={self.confidence:.3f})"
        )


class EmotionRecognizer:
    """
    Audio feature-based emotion recognizer.

    Extracts acoustic features (energy, ZCR, F0, rhythm) and maps them
    to emotions using a rule-based scoring system. Each emotion has a
    characteristic feature profile:

        - **angry**:  high energy, high ZCR, high F0 variance
        - **happy**:  medium-high energy, high ZCR, high F0
        - **sad**:    low energy, low ZCR, low F0
        - **neutral**: medium energy, medium ZCR, stable F0
        - **surprised**: high energy, sharp onset, high F0

    Args:
        frame_size_ms: Analysis frame size in milliseconds.
        min_f0: Minimum F0 for pitch estimation (Hz).
        max_f0: Maximum F0 for pitch estimation (Hz).

    Usage:
        recognizer = EmotionRecognizer()
        result = recognizer.analyze(audio, sample_rate=16000)
        print(result.emotion, result.confidence, result.features)
    """

    def __init__(
        self,
        frame_size_ms: int = 25,
        min_f0: float = 50.0,
        max_f0: float = 500.0,
    ):
        self.frame_size_ms = frame_size_ms
        self.min_f0 = min_f0
        self.max_f0 = max_f0

    def extract_features(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> AudioFeatures:
        """
        Extract acoustic features from audio signal.

        Args:
            audio: Audio signal (float32, mono).
            sample_rate: Sample rate in Hz.

        Returns:
            AudioFeatures with all computed feature values.
        """
        if len(audio) == 0:
            return AudioFeatures()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        frame_size = int(sample_rate * self.frame_size_ms / 1000)
        n_frames = max(1, len(audio) // frame_size)

        # 1. RMS Energy per frame
        energies = []
        for i in range(n_frames):
            start = i * frame_size
            end = min(start + frame_size, len(audio))
            frame = audio[start:end]
            if len(frame) > 0:
                energies.append(float(np.sqrt(np.mean(frame ** 2))))

        energies = np.array(energies) if energies else np.array([0.0])

        rms_energy = float(np.mean(energies))
        energy_variance = float(np.var(energies))
        energy_range = float(np.max(energies) - np.min(energies)) if len(energies) > 1 else 0.0

        # 2. Zero-Crossing Rate per frame
        zcr_values = []
        for i in range(n_frames):
            start = i * frame_size
            end = min(start + frame_size, len(audio))
            frame = audio[start:end]
            if len(frame) >= 2:
                crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
                zcr_values.append(crossings / (len(frame) - 1))

        zcr = float(np.mean(zcr_values)) if zcr_values else 0.0

        # 3. F0 estimation via autocorrelation
        f0_values = self._estimate_f0_series(audio, sample_rate, frame_size)
        mean_f0 = float(np.mean(f0_values)) if len(f0_values) > 0 else 0.0
        std_f0 = float(np.std(f0_values)) if len(f0_values) > 0 else 0.0

        # 4. Speech rate proxy (energy envelope modulation)
        speech_rate_proxy = self._compute_speech_rate(energies)

        return AudioFeatures(
            rms_energy=rms_energy,
            zero_crossing_rate=zcr,
            mean_f0=mean_f0,
            std_f0=std_f0,
            energy_variance=energy_variance,
            energy_range=energy_range,
            speech_rate_proxy=speech_rate_proxy,
        )

    def _estimate_f0_series(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_size: int,
    ) -> np.ndarray:
        """
        Estimate F0 contour using autocorrelation method.

        Args:
            audio: Audio signal.
            sample_rate: Sample rate.
            frame_size: Frame size in samples.

        Returns:
            Array of F0 estimates per voiced frame (Hz).
        """
        min_lag = int(sample_rate / self.max_f0)
        max_lag = int(sample_rate / self.min_f0)

        f0_values = []
        n_frames = max(1, len(audio) // frame_size)

        for i in range(n_frames):
            start = i * frame_size
            end = min(start + frame_size, len(audio))
            frame = audio[start:end]

            if len(frame) < max_lag + 1:
                continue

            # Autocorrelation via numpy
            frame_centered = frame - np.mean(frame)
            energy = np.sum(frame_centered ** 2)
            if energy < 1e-10:
                continue

            # Compute autocorrelation for the lag range
            autocorr = np.correlate(frame_centered, frame_centered, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]  # Take positive lags only

            if len(autocorr) <= max_lag:
                continue

            # Search for peak in [min_lag, max_lag]
            search_region = autocorr[min_lag:max_lag + 1]
            if len(search_region) == 0:
                continue

            peak_idx = np.argmax(search_region)
            peak_val = search_region[peak_idx] / autocorr[0]  # Normalize

            # Voiced frame threshold
            if peak_val > 0.3:
                lag = peak_idx + min_lag
                if lag > 0:
                    f0 = sample_rate / lag
                    if self.min_f0 <= f0 <= self.max_f0:
                        f0_values.append(f0)

        return np.array(f0_values, dtype=np.float32)

    def _compute_speech_rate(self, energies: np.ndarray) -> float:
        """
        Estimate speech rate proxy from energy envelope modulation.

        Counts the number of energy peaks (syllable-like units) per second.

        Args:
            energies: Per-frame energy values.

        Returns:
            Speech rate proxy (peaks per second).
        """
        if len(energies) < 3:
            return 0.0

        # Smooth energy envelope
        kernel_size = min(3, len(energies))
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(energies, kernel, mode='same')

        # Detect peaks (local maxima above mean)
        threshold = np.mean(smoothed)
        peaks = 0
        for i in range(1, len(smoothed) - 1):
            if (smoothed[i] > smoothed[i - 1] and
                    smoothed[i] > smoothed[i + 1] and
                    smoothed[i] > threshold):
                peaks += 1

        # Estimate time duration from frame count
        duration_frames = len(energies)
        # Assuming 25ms frames: duration in seconds ≈ duration_frames * 0.025
        duration_s = max(duration_frames * (self.frame_size_ms / 1000.0), 0.001)

        return peaks / duration_s

    def classify(self, features: AudioFeatures) -> EmotionResult:
        """
        Classify emotion from extracted features.

        Uses a rule-based scoring system where each emotion has a
        characteristic feature profile. The emotion with the highest
        aggregate score wins.

        Args:
            features: Extracted AudioFeatures.

        Returns:
            EmotionResult with emotion label and confidence.
        """
        scores: Dict[str, float] = {}

        # Normalize features to [0, 1] range for scoring
        e = min(features.rms_energy * 10, 1.0)  # Energy → [0, 1]
        z = min(features.zero_crossing_rate * 5, 1.0)  # ZCR → [0, 1]
        f0_mean = features.mean_f0 / 400.0  # F0 normalized ~[0, 1]
        f0_std = min(features.std_f0 / 100.0, 1.0)  # F0 variability
        e_var = min(features.energy_variance * 200, 1.0)  # Energy variance
        e_range = min(features.energy_range * 10, 1.0)  # Energy range
        rate = min(features.speech_rate_proxy / 10.0, 1.0)  # Speech rate

        # ── Angry ────────────────────────────────────────────────
        # High energy, high ZCR, high F0 variance, high energy variance
        scores["angry"] = (
            0.30 * e +
            0.20 * z +
            0.20 * f0_std +
            0.15 * e_var +
            0.15 * e_range
        )

        # ── Happy ────────────────────────────────────────────────
        # Medium-high energy, higher ZCR, moderate F0, fast speech rate
        scores["happy"] = (
            0.20 * e +
            0.25 * z +
            0.15 * f0_mean +
            0.15 * rate +
            0.10 * e_var +
            0.15 * max(0, 1.0 - f0_std)  # Somewhat stable pitch
        )

        # ── Sad ──────────────────────────────────────────────────
        # Low energy, low ZCR, low F0, slow speech rate, low variance
        scores["sad"] = (
            0.30 * (1.0 - e) +
            0.20 * (1.0 - z) +
            0.15 * (1.0 - f0_mean) +
            0.15 * (1.0 - rate) +
            0.20 * (1.0 - e_var)
        )

        # ── Neutral ──────────────────────────────────────────────
        # Medium energy, medium ZCR, stable F0, moderate variance
        scores["neutral"] = (
            0.25 * (1.0 - abs(e - 0.5) * 2) +
            0.20 * (1.0 - abs(z - 0.4) * 2) +
            0.25 * (1.0 - f0_std) +
            0.15 * (1.0 - e_var) +
            0.15 * (1.0 - abs(rate - 0.4) * 2)
        )

        # ── Surprised ────────────────────────────────────────────
        # High energy, sharp onset, high F0, high energy range
        scores["surprised"] = (
            0.25 * e +
            0.15 * z +
            0.25 * f0_mean +
            0.20 * e_range +
            0.15 * f0_std
        )

        # Normalize scores
        total = sum(scores.values()) + 1e-10
        scores = {k: v / total for k, v in scores.items()}

        # Find winner
        best_emotion = max(scores, key=scores.get)
        confidence = scores[best_emotion]

        return EmotionResult(
            emotion=best_emotion,
            confidence=confidence,
            features=features,
            all_scores=scores,
        )

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> EmotionResult:
        """
        Analyze audio and return emotion classification.

        Args:
            audio: Audio signal (float32, mono).
            sample_rate: Sample rate in Hz.

        Returns:
            EmotionResult with detected emotion and features.
        """
        features = self.extract_features(audio, sample_rate)
        return self.classify(features)