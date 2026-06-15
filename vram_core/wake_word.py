"""
Wake Word / Keyword Detection Module for vram_core
====================================================

Lightweight keyword spotting using energy-based or model-based detection.

Features:
    - Energy-based detection (no dependencies)
    - Custom keyword vocabulary
    - Real-time stream processing
    - Callback on detection

Usage:
    from vram_core.wake_word import WakeWordDetector
    detector = WakeWordDetector(keywords=["hey computer", "wake up"])
    detector.on_detect(lambda kw, conf: print(f"Detected: {kw}"))
    detector.process_chunk(audio_chunk)
"""

import logging
import time
import re
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WakeWordEvent:
    """A wake word detection event."""
    keyword: str = ""
    confidence: float = 0.0
    timestamp: float = 0.0
    audio_start: float = 0.0
    audio_end: float = 0.0


class WakeWordDetector:
    """
    Real-time wake word / keyword detector.

    Supports two modes:
        1. Energy-based: detects sudden energy increases (clap, snap, etc.)
        2. Whisper-based: uses WhisperBridge for actual keyword recognition

    Usage:
        # Simple energy-based wake
        detector = WakeWordDetector(mode="energy", energy_threshold=0.05)

        # Whisper-based keyword detection
        detector = WakeWordDetector(
            mode="whisper",
            keywords=["hey computer", "stop recording"],
            whisper_bridge=bridge,
        )
    """

    def __init__(
        self,
        mode: str = "energy",
        keywords: Optional[List[str]] = None,
        energy_threshold: float = 0.05,
        whisper_bridge=None,
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,
        cooldown: float = 1.0,
        sensitivity: float = 0.8,
    ):
        """
        Initialize wake word detector.

        Args:
            mode: "energy" for energy-based, "whisper" for ASR-based.
            keywords: List of keywords to detect (whisper mode).
            energy_threshold: Energy threshold for energy mode.
            whisper_bridge: WhisperBridge instance for whisper mode.
            sample_rate: Audio sample rate.
            chunk_duration: Duration of each analysis chunk in seconds.
            cooldown: Minimum seconds between detections.
            sensitivity: Detection sensitivity (0.0-1.0).
        """
        self.mode = mode
        self.keywords = [kw.lower() for kw in (keywords or [])]
        self.energy_threshold = energy_threshold
        self.whisper_bridge = whisper_bridge
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.cooldown = cooldown
        self.sensitivity = sensitivity

        self._buffer: deque = deque(maxlen=self.chunk_samples * 2)
        self._last_detect_time = 0.0
        self._callbacks: List[Callable] = []
        self._history: deque = deque(maxlen=100)
        self._total_processed = 0.0

        logger.info(
            f"WakeWordDetector: mode={mode}, keywords={self.keywords}, "
            f"threshold={energy_threshold}, sensitivity={sensitivity}"
        )

    def on_detect(self, callback: Callable[[str, float], None]) -> None:
        """Register a callback for wake word detection."""
        self._callbacks.append(callback)

    def process_chunk(self, audio: np.ndarray) -> Optional[WakeWordEvent]:
        """
        Process an audio chunk and check for wake words.

        Args:
            audio: Float32 mono audio chunk.

        Returns:
            WakeWordEvent if detected, None otherwise.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        self._buffer.extend(audio)
        chunk_duration = len(audio) / self.sample_rate
        self._total_processed += chunk_duration

        # Cooldown check
        now = time.time()
        if now - self._last_detect_time < self.cooldown:
            return None

        if self.mode == "energy":
            return self._detect_energy(audio)
        elif self.mode == "whisper":
            return self._detect_whisper()
        else:
            logger.warning(f"Unknown mode: {self.mode}")
            return None

    def process_stream(
        self,
        audio_stream: np.ndarray,
        chunk_size: int = 4096,
    ) -> List[WakeWordEvent]:
        """
        Process a long audio stream in chunks.

        Args:
            audio_stream: Full audio array.
            chunk_size: Samples per chunk.

        Returns:
            List of detected events.
        """
        events = []
        for i in range(0, len(audio_stream), chunk_size):
            chunk = audio_stream[i:i + chunk_size]
            event = self.process_chunk(chunk)
            if event:
                events.append(event)
        return events

    def _detect_energy(self, audio: np.ndarray) -> Optional[WakeWordEvent]:
        """Energy-based detection."""
        rms = np.sqrt(np.mean(audio ** 2))
        # Adjust threshold by sensitivity
        effective_threshold = self.energy_threshold * (1.1 - self.sensitivity)

        if rms > effective_threshold:
            now = time.time()
            self._last_detect_time = now

            confidence = min(1.0, rms / (effective_threshold * 3))
            event = WakeWordEvent(
                keyword="[energy_spike]",
                confidence=confidence,
                timestamp=now,
                audio_start=self._total_processed - len(audio) / self.sample_rate,
                audio_end=self._total_processed,
            )

            self._history.append(event)
            logger.info(f"Energy wake detected: rms={rms:.4f}, conf={confidence:.3f}")

            for cb in self._callbacks:
                try:
                    cb(event.keyword, event.confidence)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            return event

        return None

    def _detect_whisper(self) -> Optional[WakeWordEvent]:
        """Whisper ASR-based keyword detection."""
        if not self.whisper_bridge or not self.keywords:
            return None

        if len(self._buffer) < self.chunk_samples:
            return None

        audio = np.array(list(self._buffer), dtype=np.float32)[-self.chunk_samples:]

        try:
            result = self.whisper_bridge.transcribe(
                audio, sample_rate=self.sample_rate, beam_size=1,
            )
            text = result.text.lower().strip()

            for keyword in self.keywords:
                if keyword in text:
                    now = time.time()
                    self._last_detect_time = now

                    event = WakeWordEvent(
                        keyword=keyword,
                        confidence=result.confidence,
                        timestamp=now,
                        audio_start=self._total_processed - len(audio) / self.sample_rate,
                        audio_end=self._total_processed,
                    )

                    self._history.append(event)
                    self._buffer.clear()

                    logger.info(f"Keyword detected: '{keyword}' (conf={result.confidence:.3f})")

                    for cb in self._callbacks:
                        try:
                            cb(event.keyword, event.confidence)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

                    return event

        except Exception as e:
            logger.debug(f"Whisper detection error: {e}")

        return None

    def get_history(self) -> List[WakeWordEvent]:
        """Get recent detection history."""
        return list(self._history)

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "mode": self.mode,
            "keywords": self.keywords,
            "total_processed_seconds": round(self._total_processed, 1),
            "total_detections": len(self._history),
            "energy_threshold": self.energy_threshold,
            "sensitivity": self.sensitivity,
            "cooldown": self.cooldown,
        }

    def reset(self) -> None:
        """Reset detector state."""
        self._buffer.clear()
        self._history.clear()
        self._last_detect_time = 0.0
        self._total_processed = 0.0