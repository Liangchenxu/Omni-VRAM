"""
Speech Quality Assessment: SNR estimation, spectral clarity, clipping detection,
and overall quality grading.

v2.5.0 - Professional audio quality analysis.
"""

import logging
import numpy as np
from typing import List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Report from speech quality assessment."""
    quality_grade: str = "poor"          # excellent / good / fair / poor
    snr_db: float = 0.0
    spectral_clarity: float = 0.0        # 0.0 - 1.0
    clipping_ratio: float = 0.0          # 0.0 - 1.0
    dynamic_range_db: float = 0.0
    rms_level_db: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class SpeechQualityAssessor:
    """Assess speech audio quality with multiple metrics."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def assess(self, audio: np.ndarray) -> QualityReport:
        """Assess audio quality and return a QualityReport."""
        report = QualityReport()

        # Handle empty audio
        if len(audio) == 0:
            report.quality_grade = "poor"
            report.issues.append("Empty audio signal")
            report.recommendations.append("Provide non-empty audio data")
            return report

        audio = audio.astype(np.float32)

        # 1. SNR estimation
        report.snr_db = self._estimate_snr(audio)

        # 2. Clipping detection
        report.clipping_ratio = self._detect_clipping(audio)

        # 3. Dynamic range
        report.dynamic_range_db = self._compute_dynamic_range(audio)

        # 4. RMS level
        report.rms_level_db = self._compute_rms_level(audio)

        # 5. Spectral clarity
        report.spectral_clarity = self._compute_spectral_clarity(audio)

        # 6. Identify issues
        self._identify_issues(report)

        # 7. Grade overall quality
        report.quality_grade = self._grade_quality(report)

        return report

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio in dB."""
        if len(audio) == 0:
            return -np.inf

        # Estimate signal as the louder parts and noise as the quieter parts
        frame_size = max(256, len(audio) // 100)
        if frame_size == 0:
            return 0.0

        frame_rms = []
        for i in range(0, len(audio) - frame_size + 1, frame_size):
            frame = audio[i:i + frame_size]
            rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
            frame_rms.append(rms)

        if len(frame_rms) < 4:
            # Not enough frames, compute overall
            overall_rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
            peak = np.max(np.abs(audio))
            if peak > 0:
                return 20 * np.log10(overall_rms / (peak * 0.01 + 1e-10))
            return 0.0

        frame_rms = np.array(frame_rms)
        # Signal = loud frames (top 20%)
        threshold = np.percentile(frame_rms, 80)
        signal_frames = frame_rms[frame_rms >= threshold]
        noise_frames = frame_rms[frame_rms < threshold]

        signal_power = np.mean(signal_frames ** 2) if len(signal_frames) > 0 else 1e-10
        noise_power = np.mean(noise_frames ** 2) if len(noise_frames) > 0 else 1e-10

        if noise_power < 1e-10:
            noise_power = 1e-10

        snr = 10 * np.log10(signal_power / noise_power)
        return float(np.clip(snr, -50, 60))

    def _detect_clipping(self, audio: np.ndarray, threshold: float = 0.99) -> float:
        """Detect clipping ratio - fraction of samples near saturation."""
        if len(audio) == 0:
            return 0.0
        clipped = np.sum(np.abs(audio) >= threshold)
        return float(clipped / len(audio))

    def _compute_dynamic_range(self, audio: np.ndarray) -> float:
        """Compute dynamic range in dB."""
        if len(audio) == 0:
            return 0.0
        peak = np.max(np.abs(audio))
        # Compute 5th percentile RMS as "noise floor"
        frame_size = max(256, len(audio) // 50)
        if frame_size == 0:
            return 0.0
        frame_rms = []
        for i in range(0, len(audio) - frame_size + 1, frame_size):
            frame = audio[i:i + frame_size]
            rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
            frame_rms.append(rms)

        if len(frame_rms) == 0:
            return 0.0

        noise_floor = np.percentile(frame_rms, 5)
        if noise_floor < 1e-10:
            noise_floor = 1e-10
        if peak < 1e-10:
            return 0.0
        return float(20 * np.log10(peak / noise_floor))

    def _compute_rms_level(self, audio: np.ndarray) -> float:
        """Compute RMS level in dB."""
        if len(audio) == 0:
            return -np.inf
        rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
        return float(20 * np.log10(rms))

    def _compute_spectral_clarity(self, audio: np.ndarray) -> float:
        """Compute spectral clarity score (0.0 - 1.0)."""
        if len(audio) < 256:
            return 0.0

        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        total_energy = np.sum(magnitude ** 2)
        if total_energy < 1e-10:
            return 0.0

        # Spectral flatness (Wiener entropy) - higher = more tonal = clearer
        log_magnitude = np.log(magnitude + 1e-10)
        spectral_flatness = np.exp(np.mean(log_magnitude)) / (np.mean(magnitude) + 1e-10)

        # Convert to clarity: low flatness = more tonal = higher clarity
        clarity = 1.0 - min(spectral_flatness, 1.0)
        return float(np.clip(clarity, 0.0, 1.0))

    def _identify_issues(self, report: QualityReport) -> None:
        """Identify audio quality issues."""
        if report.snr_db < 5:
            report.issues.append(f"Low SNR: {report.snr_db:.1f} dB")
            report.recommendations.append("Apply noise reduction")

        if report.clipping_ratio > 0.001:
            report.issues.append(f"Clipping detected: {report.clipping_ratio * 100:.1f}% of samples")
            report.recommendations.append("Reduce input gain or apply limiter")

        if report.rms_level_db < -40:
            report.issues.append(f"Very low signal level: {report.rms_level_db:.1f} dB")
            report.recommendations.append("Increase input gain or apply normalization")

        if report.rms_level_db > -5:
            report.issues.append(f"Very high signal level: {report.rms_level_db:.1f} dB")
            report.recommendations.append("Reduce input gain to avoid clipping")

        if report.dynamic_range_db < 3:
            report.issues.append("Very low dynamic range")
            report.recommendations.append("Check for heavy compression or silence")

    def _grade_quality(self, report: QualityReport) -> str:
        """Assign overall quality grade."""
        score = 0.0

        # SNR contribution (0-40 points)
        score += min(max(report.snr_db + 10, 0), 40)

        # Clipping penalty (0-30 points deducted)
        if report.clipping_ratio > 0.5:
            score -= 30
        elif report.clipping_ratio > 0.01:
            score -= 15
        elif report.clipping_ratio > 0.001:
            score -= 5

        # Dynamic range (0-20 points)
        score += min(report.dynamic_range_db, 20)

        # Spectral clarity (0-10 points)
        score += report.spectral_clarity * 10

        # Grade
        if score >= 60:
            return "excellent"
        elif score >= 40:
            return "good"
        elif score >= 20:
            return "fair"
        else:
            return "poor"