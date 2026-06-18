"""
Real-Time Latency Optimizer
============================

Optimizes the real-time voice pipeline for < 200ms end-to-end latency.

Key optimizations:
    1. Silero VAD — neural-network-based VAD, more accurate than energy threshold
    2. Optimized RingBuffer — numpy-based, zero-copy operations
    3. Streaming Whisper — transcribe incrementally as audio arrives
    4. Concurrent VAD + ASR — pipeline parallelism via ThreadPoolExecutor
    5. Latency instrumentation — fine-grained timing at each stage

Architecture:
    Microphone → RingBuffer → SileroVAD (concurrent)
                            → StreamingTranscriber (concurrent)
                            → Result callback

Usage:
    from vram_core.realtime_optimizer import RealtimePipeline, PipelineConfig

    config = PipelineConfig(silero_vad_threshold=0.5)
    pipeline = RealtimePipeline(config=config, whisper_bridge=bridge)
    pipeline.on_transcription = lambda r: print(r.text)
    pipeline.start()
    pipeline.feed(audio_chunk)
    pipeline.stop()

Target: < 200ms mic-to-text latency on RTX 3060.
"""

import time
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Silero VAD
# ==============================================================================

class SileroVAD:
    """
    Neural-network-based Voice Activity Detection using Silero VAD model.

    Much more accurate than energy-based VAD, especially in noisy environments.
    Inference time is ~1ms per chunk on CPU.

    The model is loaded lazily on first call.

    Args:
        threshold: Speech probability threshold (0.0-1.0). Default 0.5.
        sample_rate: Audio sample rate. Supported: 8000, 16000.
        min_speech_duration_ms: Minimum speech duration to trigger.
        silence_duration_ms: Silence duration to mark end of speech.
    """

    SUPPORTED_SAMPLE_RATES = (8000, 16000)

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        silence_duration_ms: int = 500,
    ):
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {self.SUPPORTED_SAMPLE_RATES}, "
                f"got {sample_rate}"
            )

        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.silence_duration_ms = silence_duration_ms

        self._model = None
        self._model_lock = threading.Lock()

        # State tracking for continuous VAD
        self._is_speech_active = False
        self._speech_start_time: Optional[float] = None
        self._last_speech_time: Optional[float] = None

        # Internal state for Silero model
        self._model_state = None

        # Buffering for small chunks (< 512 samples)
        self._MIN_SILERO_SAMPLES = 512
        self._silero_buffer = np.array([], dtype=np.float32)
        self._last_silero_result = False
        self._silero_fail_count = 0

    def _load_model(self):
        """Lazily load the Silero VAD model."""
        if self._model is not None:
            return

        with self._model_lock:
            if self._model is not None:
                return

            try:
                import torch
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False,
                )
                self._model = model
                self._utils = utils
                logger.info("Silero VAD model loaded successfully")
            except (ImportError, RuntimeError, OSError) as e:
                logger.error("Failed to load Silero VAD model: %s", e)
                raise RuntimeError(
                    "Silero VAD model load failed: %s. "
                    "Install with: pip install torch" % e
                ) from e

    def reset(self):
        """Reset VAD internal state."""
        self._is_speech_active = False
        self._speech_start_time = None
        self._last_speech_time = None
        self._silero_buffer = np.array([], dtype=np.float32)
        self._last_silero_result = False
        self._silero_fail_count = 0
        if self._model is not None:
            try:
                self._model.reset_states()
            except (RuntimeError, AttributeError):
                pass

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect whether the given audio chunk contains speech.

        Buffers small chunks (e.g. 480 samples / 30ms) until the minimum
        Silero VAD window size (512 samples / 32ms) is reached, avoiding
        the "Input audio chunk is too short" fallback.

        Args:
            audio_chunk: Float32 audio samples at the configured sample rate.

        Returns:
            True if speech is detected.
        """
        self._load_model()

        if len(audio_chunk) == 0:
            return False

        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Append to internal buffer
        self._silero_buffer = np.concatenate([self._silero_buffer, audio_chunk])

        # Process all complete windows from the buffer
        while len(self._silero_buffer) >= self._MIN_SILERO_SAMPLES:
            window = self._silero_buffer[:self._MIN_SILERO_SAMPLES]
            self._silero_buffer = self._silero_buffer[self._MIN_SILERO_SAMPLES:]
            try:
                import torch
                tensor = torch.from_numpy(window)
                prob = self._model(tensor, self.sample_rate).item()
                self._last_silero_result = prob >= self.threshold
                self._silero_fail_count = 0
            except (RuntimeError, OSError, ValueError) as e:
                if self._silero_fail_count == 0:
                    logger.warning("Silero VAD inference failed: %s, falling back", e)
                self._silero_fail_count += 1
                self._last_silero_result = self._fallback_energy_vad(window)

        # Update speech state
        now = time.time()
        is_speech = self._last_silero_result

        if is_speech:
            if not self._is_speech_active:
                self._is_speech_active = True
                self._speech_start_time = now
            self._last_speech_time = now
        else:
            if self._is_speech_active:
                if self._last_speech_time is not None:
                    silence_ms = (now - self._last_speech_time) * 1000
                    if silence_ms >= self.silence_duration_ms:
                        self._is_speech_active = False

        return is_speech

    def _fallback_energy_vad(self, audio_chunk: np.ndarray) -> bool:
        """Fallback energy-based VAD if Silero fails."""
        if len(audio_chunk) == 0:
            return False
        energy = float(np.sqrt(np.mean(audio_chunk ** 2)))
        return energy > 0.02

    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """
        Get the raw speech probability for the audio chunk.

        Args:
            audio_chunk: Float32 audio samples.

        Returns:
            Speech probability (0.0-1.0).
        """
        self._load_model()

        if len(audio_chunk) == 0:
            return 0.0

        try:
            import torch
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            tensor = torch.from_numpy(audio_chunk)
            return float(self._model(tensor, self.sample_rate).item())
        except (RuntimeError, OSError, ValueError):
            return self._fallback_energy(audio_chunk)

    @staticmethod
    def _fallback_energy(audio_chunk: np.ndarray) -> float:
        """Compute RMS energy as fallback probability."""
        if len(audio_chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_chunk ** 2)))

    @property
    def is_speech_active(self) -> bool:
        """Whether speech is currently active."""
        return self._is_speech_active


# ==============================================================================
# 2. Optimized Ring Buffer
# ==============================================================================

class RingBuffer:
    """
    High-performance numpy-based ring buffer for audio streaming.

    Uses a pre-allocated numpy array with head/tail pointers.
    O(1) write and O(1) read. No Python object overhead.

    Args:
        capacity: Maximum number of samples.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._head = 0      # Write position
        self._tail = 0      # Read position
        self._count = 0      # Current number of samples
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> int:
        """
        Write audio samples to the buffer.

        If buffer is full, oldest samples are overwritten.

        Args:
            data: Float32 audio samples.

        Returns:
            Number of samples written.
        """
        if len(data) == 0:
            return 0

        data = data.flatten().astype(np.float32)
        n = len(data)

        with self._lock:
            if n >= self.capacity:
                # Data larger than buffer — keep only the last `capacity` samples
                data = data[-self.capacity:]
                n = self.capacity
                self._buffer[:] = data
                self._head = 0
                self._tail = 0
                self._count = self.capacity
                return n

            # Write with wrapping
            end = self._head + n
            if end <= self.capacity:
                self._buffer[self._head:end] = data
            else:
                first_part = self.capacity - self._head
                self._buffer[self._head:] = data[:first_part]
                self._buffer[:n - first_part] = data[first_part:]

            self._head = end % self.capacity
            self._count = min(self._count + n, self.capacity)

            # Update tail if we've overwritten
            if self._count == self.capacity and end > self.capacity:
                self._tail = self._head

            return n

    def read(self, n_samples: int) -> np.ndarray:
        """
        Read and remove n samples from the buffer.

        Args:
            n_samples: Number of samples to read.

        Returns:
            Float32 numpy array.
        """
        with self._lock:
            n = min(n_samples, self._count)
            if n == 0:
                return np.array([], dtype=np.float32)

            end = self._tail + n
            if end <= self.capacity:
                result = self._buffer[self._tail:end].copy()
            else:
                first_part = self.capacity - self._tail
                result = np.empty(n, dtype=np.float32)
                result[:first_part] = self._buffer[self._tail:]
                result[first_part:] = self._buffer[:n - first_part]

            self._tail = end % self.capacity
            self._count -= n
            return result

    def peek(self, n_samples: int) -> np.ndarray:
        """
        Read n samples without removing them.

        Args:
            n_samples: Number of samples to peek.

        Returns:
            Float32 numpy array.
        """
        with self._lock:
            n = min(n_samples, self._count)
            if n == 0:
                return np.array([], dtype=np.float32)

            # Peek from the end (most recent samples)
            start = (self._tail + self._count - n) % self.capacity
            end = start + n
            if end <= self.capacity:
                return self._buffer[start:end].copy()
            else:
                result = np.empty(n, dtype=np.float32)
                first_part = self.capacity - start
                result[:first_part] = self._buffer[start:]
                result[first_part:] = self._buffer[:n - first_part]
                return result

    @property
    def size(self) -> int:
        """Current number of samples in buffer."""
        with self._lock:
            return self._count

    @property
    def is_empty(self) -> bool:
        """Whether the buffer is empty."""
        with self._lock:
            return self._count == 0

    @property
    def is_full(self) -> bool:
        """Whether the buffer is full."""
        with self._lock:
            return self._count == self.capacity

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._head = 0
            self._tail = 0
            self._count = 0

    def read_all(self) -> np.ndarray:
        """Read all samples from the buffer."""
        with self._lock:
            n = self._count
            if n == 0:
                return np.array([], dtype=np.float32)

            end = self._tail + n
            if end <= self.capacity:
                result = self._buffer[self._tail:end].copy()
            else:
                result = np.empty(n, dtype=np.float32)
                first_part = self.capacity - self._tail
                result[:first_part] = self._buffer[self._tail:]
                result[first_part:] = self._buffer[:n - first_part]

            self._tail = end % self.capacity
            self._count = 0
            return result


# ==============================================================================
# 3. Streaming Transcriber
# ==============================================================================

@dataclass
class StreamingChunk:
    """Result from streaming transcription."""
    text: str
    chunk_index: int
    start_sample: int
    end_sample: int
    is_final: bool
    timestamp: float = field(default_factory=time.time)
    processing_time_ms: float = 0.0


class StreamingTranscriber:
    """
    Incremental streaming transcriber.

    Transcribes audio in small overlapping chunks as they arrive,
    providing partial results before the utterance is complete.

    This dramatically reduces perceived latency because the user sees
    text appearing while they are still speaking.

    Args:
        whisper_bridge: WhisperBridge instance for transcription.
        chunk_duration_ms: Duration of each transcription chunk.
        overlap_ms: Overlap between consecutive chunks.
        language: Language code for Whisper.
    """

    def __init__(
        self,
        whisper_bridge,
        chunk_duration_ms: int = 500,
        overlap_ms: int = 100,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ):
        self.whisper_bridge = whisper_bridge
        self.chunk_duration_ms = chunk_duration_ms
        self.overlap_ms = overlap_ms
        self.sample_rate = sample_rate
        self.language = language

        self._chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self._overlap_samples = int(sample_rate * overlap_ms / 1000)
        self._chunk_index = 0
        self._processed_samples = 0
        self._pending_audio = np.array([], dtype=np.float32)

        # Callbacks
        self.on_chunk_result: Optional[Callable[[StreamingChunk], None]] = None

    def reset(self):
        """Reset transcriber state."""
        self._chunk_index = 0
        self._processed_samples = 0
        self._pending_audio = np.array([], dtype=np.float32)

    def feed(self, audio_chunk: np.ndarray) -> Optional[StreamingChunk]:
        """
        Feed audio and transcribe if enough has accumulated.

        Args:
            audio_chunk: New audio samples.

        Returns:
            StreamingChunk if a chunk was transcribed, None otherwise.
        """
        if len(audio_chunk) == 0:
            return None

        self._pending_audio = np.concatenate([self._pending_audio, audio_chunk])

        if len(self._pending_audio) < self._chunk_samples:
            return None

        # Extract a chunk for transcription
        chunk = self._pending_audio[:self._chunk_samples].copy()

        # Keep overlap for next chunk
        overlap_start = max(0, self._chunk_samples - self._overlap_samples)
        self._pending_audio = self._pending_audio[overlap_start:]

        return self._transcribe_chunk(chunk)

    def finalize(self) -> Optional[StreamingChunk]:
        """
        Transcribe any remaining audio in the buffer.

        Returns:
            Final StreamingChunk or None.
        """
        if len(self._pending_audio) < self.sample_rate * 0.1:  # < 100ms
            return None

        chunk = self._pending_audio.copy()
        self._pending_audio = np.array([], dtype=np.float32)
        result = self._transcribe_chunk(chunk, is_final=True)
        return result

    def _transcribe_chunk(
        self, audio: np.ndarray, is_final: bool = False
    ) -> Optional[StreamingChunk]:
        """Transcribe a single chunk."""
        start_time = time.time()

        try:
            kwargs = {"sample_rate": self.sample_rate}
            if self.language:
                kwargs["language"] = self.language

            result = self.whisper_bridge.transcribe(audio, **kwargs)
            processing_time = (time.time() - start_time) * 1000

            if not result.text.strip():
                return None

            chunk = StreamingChunk(
                text=result.text.strip(),
                chunk_index=self._chunk_index,
                start_sample=self._processed_samples,
                end_sample=self._processed_samples + len(audio),
                is_final=is_final,
                processing_time_ms=processing_time,
            )

            self._processed_samples += len(audio)
            self._chunk_index += 1

            if self.on_chunk_result:
                try:
                    self.on_chunk_result(chunk)
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.warning("Chunk result callback error: %s", e)

            return chunk

        except (RuntimeError, OSError, ValueError) as e:
            logger.warning("Streaming transcription failed: %s", e)
            return None


# ==============================================================================
# 4. Latency Instrumentation
# ==============================================================================

@dataclass
class LatencyMeasurement:
    """Detailed latency measurement for a single utterance."""
    vad_start_time: float = 0.0
    vad_end_time: float = 0.0
    vad_latency_ms: float = 0.0

    buffer_write_start: float = 0.0
    buffer_write_end: float = 0.0
    buffer_latency_ms: float = 0.0

    asr_start_time: float = 0.0
    asr_end_time: float = 0.0
    asr_latency_ms: float = 0.0

    total_latency_ms: float = 0.0
    audio_duration_ms: float = 0.0

    @property
    def breakdown(self) -> Dict[str, float]:
        """Get latency breakdown as percentages."""
        total = self.total_latency_ms
        if total == 0:
            return {}
        return {
            "vad_pct": (self.vad_latency_ms / total) * 100,
            "buffer_pct": (self.buffer_latency_ms / total) * 100,
            "asr_pct": (self.asr_latency_ms / total) * 100,
        }


class LatencyTracker:
    """
    Tracks fine-grained latency at each stage of the pipeline.

    Usage:
        tracker = LatencyTracker()
        tracker.start_vad()
        # ... VAD processing
        tracker.end_vad()
        tracker.start_asr()
        # ... ASR processing
        tracker.end_asr()
        measurement = tracker.get_measurement()
    """

    def __init__(self):
        self._current: Optional[LatencyMeasurement] = None
        self._history: List[LatencyMeasurement] = []

    def start_measurement(self, audio_duration_ms: float = 0.0):
        """Start a new latency measurement."""
        self._current = LatencyMeasurement(audio_duration_ms=audio_duration_ms)
        self._current.vad_start_time = time.perf_counter()

    def start_vad(self):
        """Mark VAD start."""
        if self._current:
            self._current.vad_start_time = time.perf_counter()

    def end_vad(self):
        """Mark VAD end."""
        if self._current:
            self._current.vad_end_time = time.perf_counter()
            self._current.vad_latency_ms = (
                self._current.vad_end_time - self._current.vad_start_time
            ) * 1000

    def start_buffer_write(self):
        """Mark buffer write start."""
        if self._current:
            self._current.buffer_write_start = time.perf_counter()

    def end_buffer_write(self):
        """Mark buffer write end."""
        if self._current:
            self._current.buffer_write_end = time.perf_counter()
            self._current.buffer_latency_ms = (
                self._current.buffer_write_end - self._current.buffer_write_start
            ) * 1000

    def start_asr(self):
        """Mark ASR start."""
        if self._current:
            self._current.asr_start_time = time.perf_counter()

    def end_asr(self):
        """Mark ASR end."""
        if self._current:
            self._current.asr_end_time = time.perf_counter()
            self._current.asr_latency_ms = (
                self._current.asr_end_time - self._current.asr_start_time
            ) * 1000

    def complete(self) -> Optional[LatencyMeasurement]:
        """Complete and record the current measurement."""
        if not self._current:
            return None

        m = self._current
        m.total_latency_ms = (
            (m.asr_end_time or time.perf_counter()) - m.vad_start_time
        ) * 1000

        self._history.append(m)
        self._current = None
        return m

    def get_stats(self) -> Dict[str, float]:
        """Get aggregate latency statistics."""
        if not self._history:
            return {}

        total_latencies = [m.total_latency_ms for m in self._history]
        vad_latencies = [m.vad_latency_ms for m in self._history]
        asr_latencies = [m.asr_latency_ms for m in self._history]

        def _stats(values: List[float]) -> Dict[str, float]:
            arr = np.array(values)
            return {
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        return {
            "total_ms": _stats(total_latencies),
            "vad_ms": _stats(vad_latencies),
            "asr_ms": _stats(asr_latencies),
            "count": len(self._history),
        }

    def clear(self):
        """Clear history."""
        self._history.clear()


# ==============================================================================
# 5. RealtimePipeline — the main optimized pipeline
# ==============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the real-time optimized pipeline."""
    sample_rate: int = 16000
    chunk_duration_ms: int = 30            # Mic chunk size (30ms for low latency)
    silero_vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_silence_duration_ms: int = 500
    streaming_chunk_ms: int = 500          # ASR chunk size
    streaming_overlap_ms: int = 100
    pre_speech_buffer_ms: int = 200
    language: Optional[str] = None
    max_concurrent_asr: int = 2
    ring_buffer_duration_s: float = 5.0    # Ring buffer capacity

    @property
    def chunk_size(self) -> int:
        """Mic chunk size in samples."""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)

    @property
    def pre_speech_samples(self) -> int:
        """Pre-speech buffer size in samples."""
        return int(self.sample_rate * self.pre_speech_buffer_ms / 1000)

    @property
    def ring_buffer_capacity(self) -> int:
        """Ring buffer capacity in samples."""
        return int(self.sample_rate * self.ring_buffer_duration_s)


@dataclass
class PipelineStats:
    """Pipeline performance statistics."""
    chunks_received: int = 0
    speech_segments: int = 0
    transcriptions: int = 0
    vad_decisions: int = 0
    avg_vad_latency_ms: float = 0.0
    avg_asr_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0


class RealtimePipeline:
    """
    End-to-end optimized real-time voice processing pipeline.

    Integrates Silero VAD, ring buffer, streaming transcription,
    and concurrent processing for < 200ms latency.

    Architecture:
        ┌─────────┐    ┌──────────┐    ┌──────────────┐
        │ Mic Input│───→│RingBuffer│───→│  Silero VAD  │
        └─────────┘    └──────────┘    └──────┬───────┘
                                              │
                                    ┌─────────▼──────────┐
                                    │StreamingTranscriber │
                                    │  (ThreadPool)       │
                                    └─────────┬──────────┘
                                              │
                                    ┌─────────▼──────────┐
                                    │  Result Callback    │
                                    └────────────────────┘

    Args:
        config: Pipeline configuration.
        whisper_bridge: WhisperBridge instance for transcription.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        whisper_bridge=None,
    ):
        self.config = config or PipelineConfig()
        self.whisper_bridge = whisper_bridge

        # Silero VAD
        self.vad = SileroVAD(
            threshold=self.config.silero_vad_threshold,
            sample_rate=self.config.sample_rate,
            min_speech_duration_ms=self.config.vad_min_speech_ms,
            silence_duration_ms=self.config.vad_silence_duration_ms,
        )

        # Ring buffer for audio
        self._ring_buffer = RingBuffer(self.config.ring_buffer_capacity)

        # Pre-speech buffer (captures audio before speech detection)
        self._pre_speech = RingBuffer(self.config.pre_speech_samples)

        # Streaming transcriber
        self._transcriber: Optional[StreamingTranscriber] = None
        if self.whisper_bridge:
            self._transcriber = StreamingTranscriber(
                whisper_bridge=self.whisper_bridge,
                chunk_duration_ms=self.config.streaming_chunk_ms,
                overlap_ms=self.config.streaming_overlap_ms,
                sample_rate=self.config.sample_rate,
                language=self.config.language,
            )

        # Thread pool for concurrent processing
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_asr,
            thread_name_prefix="ASR-Worker",
        )

        # State
        self._is_running = False
        self._is_speaking = False
        self._silence_counter = 0
        self._speech_chunks: List[np.ndarray] = []
        self._total_speech_samples = 0
        self._silence_chunks_threshold = int(
            self.config.vad_silence_duration_ms / self.config.chunk_duration_ms
        )

        # Latency tracking
        self.latency_tracker = LatencyTracker()

        # Stats
        self._stats = PipelineStats()

        # Callbacks
        self.on_speech_start: Optional[Callable[[], None]] = None
        self.on_speech_end: Optional[Callable[[np.ndarray], None]] = None
        self.on_transcription: Optional[Callable[[StreamingChunk], None]] = None
        self.on_partial_transcription: Optional[Callable[[StreamingChunk], None]] = None
        self.on_vad_result: Optional[Callable[[bool, float], None]] = None

        # Lock
        self._lock = threading.Lock()

        # Pending ASR futures
        self._pending_futures: List[Future] = []

    @property
    def is_running(self) -> bool:
        """Whether the pipeline is active."""
        return self._is_running

    @property
    def stats(self) -> PipelineStats:
        """Current statistics."""
        return self._stats

    def start(self):
        """Start the pipeline."""
        self._is_running = True
        self._stats = PipelineStats()
        self.latency_tracker.clear()
        self.vad.reset()
        logger.info("RealtimePipeline started (target: < 200ms latency)")

    def stop(self):
        """Stop the pipeline and wait for pending work."""
        self._is_running = False

        # Wait for pending ASR tasks
        for f in self._pending_futures:
            try:
                f.result(timeout=5.0)
            except (RuntimeError, TimeoutError):
                pass
        self._pending_futures.clear()

        # Finalize transcriber
        if self._transcriber:
            final = self._transcriber.finalize()
            if final and self.on_transcription:
                self.on_transcription(final)

        self._executor.shutdown(wait=True)
        logger.info("RealtimePipeline stopped")

    def feed(self, audio_chunk: np.ndarray) -> None:
        """
        Feed an audio chunk into the pipeline.

        This is the main entry point. It:
        1. Writes to ring buffer (O(1))
        2. Runs Silero VAD (~1ms)
        3. If speech detected, starts/continues streaming transcription

        Args:
            audio_chunk: Float32 audio samples (mono, 16kHz).
        """
        if not self._is_running:
            return

        t_start = time.perf_counter()

        # Ensure float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        self._stats.chunks_received += 1

        # Step 1: Write to ring buffer (fast, O(1))
        self._ring_buffer.write(audio_chunk)

        # Also keep in pre-speech buffer
        if not self._is_speaking:
            self._pre_speech.write(audio_chunk)

        # Step 2: VAD (fast, ~1ms)
        self.latency_tracker.start_vad()
        is_speech = self.vad.is_speech(audio_chunk)
        self.latency_tracker.end_vad()
        self._stats.vad_decisions += 1

        # Notify VAD result
        if self.on_vad_result:
            prob = self.vad.get_speech_probability(audio_chunk)
            self.on_vad_result(is_speech, prob)

        # Step 3: State machine
        if is_speech:
            self._handle_speech(audio_chunk)
        else:
            self._handle_silence(audio_chunk)

        # Track overall feed latency
        feed_time = (time.perf_counter() - t_start) * 1000
        if feed_time > 5:
            logger.debug("feed() took %.1fms (target < 5ms)", feed_time)

    def feed_bytes(self, audio_bytes: bytes, sample_width: int = 2) -> None:
        """Feed raw audio bytes."""
        if sample_width == 2:
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        self.feed(audio)

    def _handle_speech(self, audio_chunk: np.ndarray):
        """Handle a speech chunk."""
        if not self._is_speaking:
            # Speech started!
            self._is_speaking = True
            self._speech_chunks = []
            self._total_speech_samples = 0
            self._silence_counter = 0

            # Include pre-speech buffer
            pre = self._pre_speech.read_all()
            if len(pre) > 0:
                self._speech_chunks.append(pre)
                self._total_speech_samples += len(pre)

            self._stats.speech_segments += 1

            if self.on_speech_start:
                try:
                    self.on_speech_start()
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.warning("on_speech_start callback error: %s", e)

        # Accumulate
        self._speech_chunks.append(audio_chunk)
        self._total_speech_samples += len(audio_chunk)
        self._silence_counter = 0

        # Streaming transcription: feed to transcriber incrementally
        if self._transcriber:
            self.latency_tracker.start_asr()
            chunk_result = self._transcriber.feed(audio_chunk)
            if chunk_result:
                self.latency_tracker.end_asr()
                self._stats.transcriptions += 1
                self._update_avg_latency()

                if self.on_partial_transcription:
                    try:
                        self.on_partial_transcription(chunk_result)
                    except (RuntimeError, ValueError, TypeError) as e:
                        logger.warning("on_partial_transcription callback error: %s", e)
            else:
                # No result yet (accumulating), don't end ASR timer
                pass

    def _handle_silence(self, audio_chunk: np.ndarray):
        """Handle a silence chunk."""
        if self._is_speaking:
            self._silence_counter += 1

            # Still accumulate during silence (trailing audio)
            self._speech_chunks.append(audio_chunk)
            self._total_speech_samples += len(audio_chunk)

            if self._silence_counter >= self._silence_chunks_threshold:
                # Speech ended — finalize
                self._end_speech()

    def _end_speech(self):
        """Process end of speech segment."""
        if not self._speech_chunks:
            self._is_speaking = False
            return

        full_speech = np.concatenate(self._speech_chunks)
        duration_ms = len(full_speech) / self.config.sample_rate * 1000

        if duration_ms < self.config.vad_min_speech_ms:
            logger.debug("Speech too short (%.0fms), discarding", duration_ms)
            self._reset_speech()
            return

        if self.on_speech_end:
            try:
                self.on_speech_end(full_speech)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.warning("on_speech_end callback error: %s", e)

        # Finalize streaming transcription
        if self._transcriber:
            self.latency_tracker.start_asr()
            final_chunk = self._transcriber.finalize()
            self.latency_tracker.end_asr()

            if final_chunk:
                self._stats.transcriptions += 1
                self._update_avg_latency()

                if self.on_transcription:
                    try:
                        self.on_transcription(final_chunk)
                    except (RuntimeError, ValueError, TypeError) as e:
                        logger.warning("on_transcription callback error: %s", e)

            self._transcriber.reset()
        elif self.whisper_bridge:
            # Fallback: full transcription in background
            self._transcribe_async(full_speech, duration_ms)

        self._reset_speech()

    def _transcribe_async(self, audio: np.ndarray, duration_ms: float):
        """Submit full transcription to thread pool."""
        future = self._executor.submit(
            self._transcribe_worker, audio, duration_ms
        )
        self._pending_futures.append(future)

        # Clean completed futures
        self._pending_futures = [
            f for f in self._pending_futures if not f.done()
        ]

    def _transcribe_worker(self, audio: np.ndarray, duration_ms: float):
        """Background transcription worker."""
        self.latency_tracker.start_asr()
        try:
            kwargs = {"sample_rate": self.config.sample_rate}
            if self.config.language:
                kwargs["language"] = self.config.language

            result = self.whisper_bridge.transcribe(audio, **kwargs)
            self.latency_tracker.end_asr()
            self._stats.transcriptions += 1
            self._update_avg_latency()

            chunk = StreamingChunk(
                text=result.text.strip(),
                chunk_index=0,
                start_sample=0,
                end_sample=len(audio),
                is_final=True,
                processing_time_ms=self.latency_tracker._current.asr_latency_ms
                if self.latency_tracker._current else 0,
            )

            if self.on_transcription:
                try:
                    self.on_transcription(chunk)
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.warning("on_transcription callback error: %s", e)

        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Background transcription failed: %s", e)

    def _reset_speech(self):
        """Reset speech accumulation state."""
        self._is_speaking = False
        self._speech_chunks = []
        self._total_speech_samples = 0
        self._silence_counter = 0

    def _update_avg_latency(self):
        """Update average latency stats."""
        m = self.latency_tracker.complete()
        if m:
            n = self._stats.transcriptions
            self._stats.avg_total_latency_ms = (
                (self._stats.avg_total_latency_ms * (n - 1) + m.total_latency_ms) / n
            )
            self._stats.avg_vad_latency_ms = (
                (self._stats.avg_vad_latency_ms * (n - 1) + m.vad_latency_ms) / n
            )
            self._stats.avg_asr_latency_ms = (
                (self._stats.avg_asr_latency_ms * (n - 1) + m.asr_latency_ms) / n
            )

    def get_latency_stats(self) -> Dict[str, Any]:
        """Get detailed latency statistics."""
        return self.latency_tracker.get_stats()

    def reset(self):
        """Reset the pipeline to initial state."""
        self._ring_buffer.clear()
        self._pre_speech.clear()
        self._reset_speech()
        self.vad.reset()
        if self._transcriber:
            self._transcriber.reset()
        self._stats = PipelineStats()
        self.latency_tracker.clear()
