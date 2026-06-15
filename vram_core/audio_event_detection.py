"""
Audio Event Detection (AED) Module for vram_core
==================================================

Detects environmental sounds and audio events beyond speech:
glass breaking, alarms, sirens, dog barks, music, applause, etc.

Backends:
1. **YAMNet** (preferred): Google's YAMNet with 521 audio event classes
   - Requires: pip install tensorflow (or tflite-runtime)
   - Based on AudioSet ontology

2. **Energy-based** (fallback): Simple energy-based sound detection
   - No ML dependencies, works with numpy/scipy only
   - Detects: silence, speech-like, loud events

Usage:
    from vram_core.audio_event_detection import AudioEventDetector

    detector = AudioEventDetector()
    events = detector.detect(audio, sample_rate=16000)
    for event in events:
        print(f"[{event.start_time:.1f}s] {event.label} ({event.confidence:.2f})")
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from vram_core.utils import ensure_float32, simple_resample, merge_adjacent_events

logger = logging.getLogger(__name__)


# 鈹€鈹€ Backend Detection 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_YAMNET_AVAILABLE = False
try:
    import tensorflow as tf
    _YAMNET_AVAILABLE = True
    logger.info("TensorFlow detected 锟?YAMNet backend available")
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        _YAMNET_AVAILABLE = True
        logger.info("TFLite Runtime detected 锟?YAMNet backend available")
    except ImportError:
        logger.info("No ML runtime for YAMNet, using energy-based fallback")


@dataclass
class AudioEvent:
    """Detected audio event."""
    label: str
    start_time: float
    end_time: float
    confidence: float
    category: str = "unknown"

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class AEDResult:
    """Audio Event Detection result."""
    events: List[AudioEvent]
    total_events: int
    duration_seconds: float
    backend_used: str


# 鈹€鈹€ Energy-based Fallback 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class EnergyDetector:
    """
    Energy-based audio event detection (fallback).
    
    Uses RMS energy and zero-crossing rate to classify audio segments.
    """

    CATEGORIES = {
        "silence": {"energy_range": (0.0, 0.01), "zcr_range": (0.0, 1.0)},
        "speech": {"energy_range": (0.01, 0.3), "zcr_range": (0.02, 0.3)},
        "music": {"energy_range": (0.05, 0.5), "zcr_range": (0.01, 0.15)},
        "loud_event": {"energy_range": (0.3, 1.0), "zcr_range": (0.0, 1.0)},
    }

    def __init__(self, segment_ms: float = 500.0, energy_threshold: float = 0.01):
        self.segment_ms = segment_ms
        self.energy_threshold = energy_threshold

    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> List[AudioEvent]:
        """Detect events based on energy analysis."""
        if len(audio) == 0:
            return []

        audio = ensure_float32(audio)

        segment_samples = int(sample_rate * self.segment_ms / 1000)
        events = []

        for i in range(0, len(audio), segment_samples):
            seg = audio[i:i + segment_samples]
            if len(seg) < segment_samples // 2:
                continue

            rms = float(np.sqrt(np.mean(seg ** 2)))
            zcr = float(np.mean(np.abs(np.diff(np.sign(seg))))) / 2 if len(seg) > 1 else 0

            start_time = i / sample_rate
            end_time = min((i + segment_samples) / sample_rate, len(audio) / sample_rate)

            if rms < self.energy_threshold:
                continue  # Skip silence

            # Classify based on energy and ZCR
            if rms > 0.3:
                label, category = "loud_event", "loud"
            elif rms > 0.1 and zcr > 0.1:
                label, category = "speech_like", "speech"
            elif rms > 0.05:
                label, category = "ambient_sound", "ambient"
            else:
                label, category = "low_sound", "ambient"

            confidence = min(rms * 2, 1.0)

            events.append(AudioEvent(
                label=label,
                start_time=start_time,
                end_time=end_time,
                confidence=confidence,
                category=category,
            ))

        return self._merge_events(events)

    def _merge_events(self, events: List[AudioEvent]) -> List[AudioEvent]:
        """Merge consecutive events of the same type."""
        return merge_adjacent_events(events)


# 鈹€鈹€ Main Detector 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class AudioEventDetector:
    """
    Multi-backend audio event detection.

    Args:
        backend: Backend to use ("auto", "yamnet", "energy").
        confidence_threshold: Minimum confidence to report event.
        segment_ms: Analysis segment duration in ms.

    Usage:
        detector = AudioEventDetector()
        events = detector.detect(audio, sample_rate=16000)
        for evt in events:
            print(f"{evt.label}: {evt.confidence:.2f}")
    """

    def __init__(
        self,
        backend: str = "auto",
        confidence_threshold: float = 0.3,
        segment_ms: float = 500.0,
    ):
        self.confidence_threshold = confidence_threshold
        self._energy_detector = EnergyDetector(segment_ms=segment_ms)
        self._yamnet_model = None
        self._yamnet_labels = None
        self._active_backend = "energy"

        if backend in ("auto", "yamnet") and _YAMNET_AVAILABLE:
            try:
                self._load_yamnet()
                self._active_backend = "yamnet"
            except Exception as e:
                logger.warning("YAMNet load failed (%s), using energy fallback", e)

    def _load_yamnet(self):
        """Load YAMNet model."""
        try:
            import tensorflow as tf
            self._yamnet_model = tf.saved_model.load("yamnet")
            logger.info("YAMNet model loaded")
        except Exception as e:
            logger.warning("Could not load YAMNet from hub: %s", e)
            raise

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        top_k: int = 5,
    ) -> AEDResult:
        """
        Detect audio events.

        Args:
            audio: Audio signal (float32, mono).
            sample_rate: Sample rate.
            top_k: Number of top predictions per segment (YAMNet only).

        Returns:
            AEDResult with detected events.
        """
        if self._active_backend == "yamnet" and self._yamnet_model:
            events = self._detect_yamnet(audio, sample_rate, top_k)
        else:
            events = self._energy_detector.detect(audio, sample_rate)

        # Filter by confidence
        events = [e for e in events if e.confidence >= self.confidence_threshold]
        duration = len(audio) / sample_rate if len(audio) > 0 else 0.0

        return AEDResult(
            events=events,
            total_events=len(events),
            duration_seconds=duration,
            backend_used=self._active_backend,
        )

    def _detect_yamnet(
        self,
        audio: np.ndarray,
        sample_rate: int,
        top_k: int,
    ) -> List[AudioEvent]:
        """Detect events using YAMNet."""
        try:
            import tensorflow as tf

            # YAMNet expects 16kHz mono float32
            if sample_rate != 16000:
                audio = simple_resample(audio, sample_rate, 16000)

            audio = ensure_float32(audio)

            # Run inference
            waveform = tf.constant(audio, dtype=tf.float32)
            scores, embeddings, spectrogram = self._yamnet_model(waveform)
            scores_np = scores.numpy()

            # Map to events
            events = []
            # Each score frame covers ~0.48s with 0.5s overlap
            frame_duration = 0.48
            for frame_idx in range(scores_np.shape[0]):
                frame_scores = scores_np[frame_idx]
                top_indices = np.argsort(frame_scores)[-top_k:][::-1]

                for class_idx in top_indices:
                    conf = float(frame_scores[class_idx])
                    if conf < self.confidence_threshold:
                        continue

                    label = f"class_{class_idx}"
                    if self._yamnet_labels and class_idx < len(self._yamnet_labels):
                        label = self._yamnet_labels[class_idx]

                    start_time = frame_idx * frame_duration
                    events.append(AudioEvent(
                        label=label,
                        start_time=start_time,
                        end_time=start_time + frame_duration,
                        confidence=conf,
                        category="yamnet",
                    ))

            return events
        except Exception as e:
            logger.error("YAMNet detection failed: %s", e)
            return self._energy_detector.detect(audio, 16000)

    @staticmethod
    def available_backends() -> List[str]:
        """List available AED backends."""
        backends = ["energy"]
        if _YAMNET_AVAILABLE:
            backends.insert(0, "yamnet")
        return backends

    @property
    def backend(self) -> str:
        return self._active_backend