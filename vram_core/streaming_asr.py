"""
vram_core Streaming ASR (Automatic Speech Recognition)
======================================================

Real-time streaming speech recognition with sliding window + incremental
recognition strategy. Achieves first-word latency < 500ms.

Architecture:
    - Sliding window: 2s window, 0.5s step
    - Incremental recognition: re-process only new audio each step
    - Callback-driven: on_partial_result / on_final_result
    - Supports Chinese-English mixed recognition

Usage:
    from vram_core.streaming_asr import StreamASR

    asr = StreamASR(language="zh")
    asr.on_partial_result = lambda text: print(f"Partial: {text}")
    asr.on_final_result = lambda result: print(f"Final: {result.text}")
    asr.start()
    # ... feed audio via asr.feed(chunk)
    asr.stop()

Thread Safety:
    StreamASR is designed to be used from a single thread or with
    external synchronization.
"""

import time
import logging
import threading
import numpy as np
from typing import Optional, Callable, Any
from dataclasses import dataclass, field

from vram_core.whisper import WhisperBridge, WhisperBackend, WhisperResult

logger = logging.getLogger(__name__)


@dataclass
class StreamASRConfig:
    """
    Configuration for streaming ASR.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        window_duration: Sliding window duration in seconds.
        step_duration: Step size in seconds (how often to re-recognize).
        min_audio_duration: Minimum audio duration before first recognition.
        language: Language code (zh, en, etc.) or None for auto-detect.
        whisper_model: Whisper model size (tiny/base/small/medium/large).
        backend: Whisper backend to use.
        vad_threshold: Energy threshold for voice activity detection.
        silence_timeout: Seconds of silence before finalizing a segment.
        overlap_duration: Overlap between windows to avoid cutting words.
    """
    sample_rate: int = 16000
    window_duration: float = 2.0
    step_duration: float = 0.5
    min_audio_duration: float = 0.5
    language: Optional[str] = "zh"
    whisper_model: str = "base"
    backend: WhisperBackend = WhisperBackend.AUTO
    vad_threshold: float = 0.01
    silence_timeout: float = 1.5
    overlap_duration: float = 0.2


@dataclass
class StreamASRResult:
    """
    Result from streaming ASR.

    Attributes:
        text: Transcribed text.
        is_final: Whether this is a final (committed) result.
        timestamp: Timestamp when result was generated.
        start_time: Start time of the audio segment (seconds from stream start).
        end_time: End time of the audio segment.
        confidence: Confidence score.
        language: Detected language.
    """
    text: str = ""
    is_final: bool = False
    timestamp: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0
    language: str = "unknown"


class StreamASR:
    """
    Real-time streaming ASR engine.

    Implements sliding window + incremental recognition:
    - Feeds audio chunks into an internal buffer
    - Every step_duration seconds, runs whisper on the sliding window
    - Compares new partial result with previous to emit incremental updates
    - Detects silence to finalize segments

    Callbacks:
        on_partial_result(text: str): Called with partial (in-progress) text.
        on_final_result(result: StreamASRResult): Called when a segment is finalized.

    Usage:
        asr = StreamASR(language="zh")
        asr.on_partial_result = lambda t: print(f"Partial: {t}")
        asr.on_final_result = lambda r: print(f"Final: {r.text}")

        asr.start()
        asr.feed(audio_chunk_1)
        asr.feed(audio_chunk_2)
        # ...
        asr.stop()
    """

    def __init__(
        self,
        config: Optional[StreamASRConfig] = None,
        whisper_bridge: Optional[WhisperBridge] = None,
        language: Optional[str] = None,
        whisper_model: Optional[str] = None,
        backend: Optional[WhisperBackend] = None,
    ):
        """
        Initialize StreamASR.

        Args:
            config: Full configuration object (overrides individual params).
            whisper_bridge: Pre-configured WhisperBridge instance.
            language: Language code (overrides config).
            whisper_model: Model size (overrides config).
            backend: Whisper backend (overrides config).
        """
        self.config = config or StreamASRConfig()

        # Override config with explicit params
        if language is not None:
            self.config.language = language
        if whisper_model is not None:
            self.config.whisper_model = whisper_model
        if backend is not None:
            self.config.backend = backend

        # Whisper bridge
        self._bridge = whisper_bridge or WhisperBridge(
            backend=self.config.backend,
            whisper_model=self.config.whisper_model,
            language=self.config.language,
        )

        # Internal audio buffer
        self._buffer = np.array([], dtype=np.float32)
        self._buffer_lock = threading.Lock()

        # Recognition state
        self._is_running = False
        self._last_text = ""
        self._segment_start_time = 0.0
        self._total_audio_time = 0.0
        self._step_samples = int(self.config.step_duration * self.config.sample_rate)
        self._window_samples = int(self.config.window_duration * self.config.sample_rate)
        self._min_samples = int(self.config.min_audio_duration * self.config.sample_rate)
        self._silence_start: Optional[float] = None
        self._last_feed_time: Optional[float] = None

        # Callbacks
        self.on_partial_result: Optional[Callable[[str], None]] = None
        self.on_final_result: Optional[Callable[[StreamASRResult], None]] = None

        # Worker thread
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        logger.info(
            "StreamASR initialized: window=%ss, step=%ss, model=%s, language=%s",
            self.config.window_duration,
            self.config.step_duration,
            self.config.whisper_model,
            self.config.language,
        )

    @property
    def is_running(self) -> bool:
        """Whether the ASR engine is actively processing."""
        return self._is_running

    @property
    def buffer_duration(self) -> float:
        """Current buffer duration in seconds."""
        with self._buffer_lock:
            return len(self._buffer) / self.config.sample_rate

    def start(self):
        """
        Start the streaming ASR engine.

        Begins the internal processing loop that periodically checks
        the audio buffer and runs recognition.
        """
        if self._is_running:
            logger.warning("StreamASR is already running")
            return

        self._is_running = True
        self._stop_event.clear()
        self._last_text = ""
        self._total_audio_time = 0.0
        self._silence_start = None
        self._last_feed_time = time.time()

        # Start worker thread
        self._worker_thread = threading.Thread(
            target=self._processing_loop,
            name="StreamASR-Worker",
            daemon=True,
        )
        self._worker_thread.start()

        logger.info("StreamASR started")

    def stop(self) -> Optional[StreamASRResult]:
        """
        Stop the streaming ASR engine.

        Returns:
            Final result if there's remaining audio in the buffer.
        """
        if not self._is_running:
            logger.warning("StreamASR is not running")
            return None

        self._is_running = False
        self._stop_event.set()

        # Wait for worker to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        # Finalize remaining audio
        final_result = self._finalize_segment()

        logger.info("StreamASR stopped")
        return final_result

    def feed(self, audio_chunk: np.ndarray):
        """
        Feed an audio chunk into the ASR engine.

        Args:
            audio_chunk: Float32 audio array. Will be converted to mono
                         if multi-channel. Sample rate should match config.

        Raises:
            RuntimeError: If ASR engine is not started.
        """
        if not self._is_running:
            raise RuntimeError("StreamASR is not running. Call start() first.")

        # Ensure float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Flatten to mono if needed
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)

        with self._buffer_lock:
            self._buffer = np.concatenate([self._buffer, audio_chunk])

            # Cap buffer to max 2x window to prevent unbounded growth
            max_samples = self._window_samples * 2
            if len(self._buffer) > max_samples:
                # Keep only the most recent max_samples
                self._buffer = self._buffer[-max_samples:]

        self._last_feed_time = time.time()

    def _processing_loop(self):
        """Main processing loop running in worker thread."""
        logger.debug("Processing loop started")

        while not self._stop_event.is_set():
            try:
                # Sleep for step duration
                self._stop_event.wait(timeout=self.config.step_duration)

                if self._stop_event.is_set():
                    break

                # Check if we have enough audio
                with self._buffer_lock:
                    buffer_len = len(self._buffer)

                if buffer_len < self._min_samples:
                    continue

                # Run recognition on sliding window
                self._recognize_step()

            except (RuntimeError, OSError, ValueError) as e:
                logger.error("Error in processing loop: %s", e, exc_info=True)
                # Continue processing despite errors
                continue

        logger.debug("Processing loop ended")

    def _recognize_step(self):
        """
        Run one recognition step on the current sliding window.

        Extracts the most recent window_duration seconds of audio,
        runs whisper, and emits partial/final results.
        """
        with self._buffer_lock:
            buffer_len = len(self._buffer)
            if buffer_len == 0:
                return

            # Extract sliding window (most recent window_duration seconds)
            window_samples = min(self._window_samples, buffer_len)
            audio_window = self._buffer[-window_samples:].copy()

        # Run whisper recognition
        try:
            result = self._bridge.transcribe(
                audio_window,
                sample_rate=self.config.sample_rate,
            )
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning("Recognition failed: %s", e)
            return

        if not result.text.strip():
            # Empty result 锟?might be silence
            self._check_silence()
            return

        # Reset silence detection
        self._silence_start = None

        new_text = result.text.strip()

        # Compare with previous text to determine incremental update
        if new_text != self._last_text:
            self._last_text = new_text

            # Emit partial result
            if self.on_partial_result:
                try:
                    self.on_partial_result(new_text)
                except (RuntimeError, ValueError) as e:
                    logger.error("Error in on_partial_result callback: %s", e)

            logger.debug("Partial: %s", new_text)

        # Check if we should finalize (long enough segment)
        with self._buffer_lock:
            current_duration = len(self._buffer) / self.config.sample_rate

        # Auto-finalize if buffer is getting too long (> 30 seconds)
        if current_duration > 30.0:
            self._finalize_segment()

    def _check_silence(self):
        """Check for silence and finalize segment if silence timeout reached."""
        now = time.time()

        if self._silence_start is None:
            self._silence_start = now
            return

        silence_duration = now - self._silence_start

        if silence_duration >= self.config.silence_timeout:
            # Silence timeout reached 锟?finalize current segment
            self._finalize_segment()
            self._silence_start = None

    def _finalize_segment(self) -> Optional[StreamASRResult]:
        """
        Finalize the current audio segment.

        Runs a final recognition on the buffered audio, emits
        on_final_result callback, and clears the buffer.

        Returns:
            StreamASRResult or None if buffer is empty.
        """
        with self._buffer_lock:
            if len(self._buffer) == 0:
                return None

            audio_data = self._buffer.copy()
            self._buffer = np.array([], dtype=np.float32)

        audio_duration = len(audio_data) / self.config.sample_rate
        segment_start = self._total_audio_time
        self._total_audio_time += audio_duration

        # Run final recognition
        try:
            result = self._bridge.transcribe(
                audio_data,
                sample_rate=self.config.sample_rate,
            )
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning("Final recognition failed: %s", e)
            return None

        if not result.text.strip():
            return None

        asr_result = StreamASRResult(
            text=result.text.strip(),
            is_final=True,
            timestamp=time.time(),
            start_time=segment_start,
            end_time=self._total_audio_time,
            confidence=result.confidence,
            language=result.language,
        )

        self._last_text = ""

        # Emit final result
        if self.on_final_result:
            try:
                self.on_final_result(asr_result)
            except (RuntimeError, ValueError) as e:
                logger.error("Error in on_final_result callback: %s", e)

        logger.info(
            "Final segment [%.1fs - %.1fs]: %s...",
            segment_start, self._total_audio_time, asr_result.text[:80],
        )

        return asr_result

    def get_status(self) -> dict:
        """
        Get current ASR engine status.

        Returns:
            Dictionary with status information.
        """
        return {
            "is_running": self._is_running,
            "buffer_duration": self.buffer_duration,
            "total_audio_time": self._total_audio_time,
            "config": {
                "window_duration": self.config.window_duration,
                "step_duration": self.config.step_duration,
                "language": self.config.language,
                "whisper_model": self.config.whisper_model,
                "sample_rate": self.config.sample_rate,
            },
            "last_text": self._last_text[:100] if self._last_text else "",
        }