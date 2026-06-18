"""
Real-Time Audio Stream Processor for vram_core
================================================

Handles real-time audio streaming with chunk-based processing,
Voice Activity Detection (VAD), and low-latency transcription
pipeline integration.

Target: < 200ms end-to-end latency on RTX 3060.

Architecture:
    - StreamProcessor: Main class for real-time audio processing
    - CircularBuffer: Lock-free ring buffer for audio chunks
    - VADProcessor: Simple energy-based Voice Activity Detection
"""

import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, List, Tuple

import numpy as np

from vram_core.audio_utils import AudioProcessor
from vram_core.noise_reduction import NoiseReducer

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """Stream processing states."""
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    PROCESSING = "processing"
    ERROR = "error"


@dataclass
class StreamConfig:
    """Configuration for stream processing."""
    sample_rate: int = 16000
    chunk_duration_ms: int = 100          # Chunk size in milliseconds
    vad_threshold: float = 0.02           # Energy threshold for VAD
    vad_silence_duration_ms: int = 800    # Silence duration to end speech
    vad_min_speech_ms: int = 200          # Minimum speech duration to process
    max_buffer_duration_s: float = 30.0   # Maximum buffer duration
    pre_speech_buffer_ms: int = 200       # Pre-speech context buffer
    overlap_ms: int = 50                  # Overlap between chunks for continuity

    @property
    def chunk_size(self) -> int:
        """Chunk size in samples."""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)

    @property
    def silence_chunks(self) -> int:
        """Number of consecutive silent chunks to trigger end of speech."""
        return int(self.vad_silence_duration_ms / self.chunk_duration_ms)

    @property
    def min_speech_chunks(self) -> int:
        """Minimum number of speech chunks to process."""
        return int(self.vad_min_speech_ms / self.chunk_duration_ms)

    @property
    def pre_speech_samples(self) -> int:
        """Number of pre-speech buffer samples."""
        return int(self.sample_rate * self.pre_speech_buffer_ms / 1000)


class CircularBuffer:
    """
    Thread-safe circular buffer for audio samples.

    Uses a deque with maxlen for automatic old-sample eviction.
    """

    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self._buffer: deque = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> int:
        """
        Write audio samples to the buffer.

        Args:
            data: Audio samples to write.

        Returns:
            Number of samples written.
        """
        with self._lock:
            samples = data.flatten().tolist()
            self._buffer.extend(samples)
            return len(samples)

    def read(self, n_samples: int) -> np.ndarray:
        """
        Read and remove n samples from the buffer.

        Args:
            n_samples: Number of samples to read.

        Returns:
            Audio samples as numpy array.
        """
        with self._lock:
            n = min(n_samples, len(self._buffer))
            if n == 0:
                return np.array([], dtype=np.float32)
            samples = [self._buffer.popleft() for _ in range(n)]
            return np.array(samples, dtype=np.float32)

    def peek(self, n_samples: int) -> np.ndarray:
        """
        Read n samples without removing them.

        Args:
            n_samples: Number of samples to peek.

        Returns:
            Audio samples as numpy array.
        """
        with self._lock:
            n = min(n_samples, len(self._buffer))
            if n == 0:
                return np.array([], dtype=np.float32)
            samples = list(self._buffer)[-n:]
            return np.array(samples, dtype=np.float32)

    @property
    def size(self) -> int:
        """Current number of samples in buffer."""
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()


class VADProcessor:
    """
    Simple energy-based Voice Activity Detection.

    Computes short-time energy and zero-crossing rate to detect
    speech segments in real-time.
    """

    def __init__(
        self,
        threshold: float = 0.02,
        sample_rate: int = 16000,
        frame_size_ms: int = 25,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_size_ms / 1000)

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Determine if audio chunk contains speech.

        Uses short-time energy analysis.

        Args:
            audio: Audio chunk (float32).

        Returns:
            True if speech is detected.
        """
        if len(audio) == 0:
            return False

        energy = self.compute_energy(audio)
        return energy > self.threshold

    def compute_energy(self, audio: np.ndarray) -> float:
        """
        Compute RMS energy of audio chunk.

        Args:
            audio: Audio chunk (float32).

        Returns:
            RMS energy value.
        """
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))

    def compute_zero_crossing_rate(self, audio: np.ndarray) -> float:
        """
        Compute zero-crossing rate.

        Args:
            audio: Audio chunk (float32).

        Returns:
            Zero-crossing rate (0.0 - 1.0).
        """
        if len(audio) < 2:
            return 0.0
        crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
        return crossings / (len(audio) - 1)

    def detect_speech_segments(
        self,
        audio: np.ndarray,
        min_duration_ms: float = 200,
    ) -> List[Tuple[int, int]]:
        """
        Detect speech segments in a longer audio buffer.

        Args:
            audio: Full audio buffer (float32).
            min_duration_ms: Minimum segment duration in ms.

        Returns:
            List of (start_sample, end_sample) tuples.
        """
        frame_size = self.frame_size
        min_frames = int(self.sample_rate * min_duration_ms / 1000 / frame_size)

        segments = []
        in_speech = False
        speech_start = 0

        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i : i + frame_size]
            if self.is_speech(frame):
                if not in_speech:
                    speech_start = i
                    in_speech = True
            else:
                if in_speech:
                    duration_frames = (i - speech_start) // frame_size
                    if duration_frames >= min_frames:
                        segments.append((speech_start, i))
                    in_speech = False

        # Handle case where speech continues to end
        if in_speech:
            end = min(speech_start + frame_size * min_frames, len(audio))
            segments.append((speech_start, end))

        return segments


@dataclass
class StreamEvent:
    """Event emitted by the stream processor."""
    event_type: str  # "speech_start", "speech_end", "transcription", "error"
    timestamp: float = field(default_factory=time.time)
    data: Optional[object] = None
    audio: Optional[np.ndarray] = None


class StreamProcessor:
    """
    Real-time audio stream processor with VAD and transcription.

    Handles audio input in chunks, performs Voice Activity Detection,
    and triggers transcription when speech ends.

    Target latency: < 200ms on RTX 3060.

    Usage:
        processor = StreamProcessor(config=StreamConfig())
        processor.on_transcription = lambda result: print(result.text)

        # Feed audio chunks
        processor.feed(audio_chunk)
    """

    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        whisper_bridge: Optional[object] = None,
    ):
        """
        Initialize stream processor.

        Args:
            config: Stream processing configuration.
            whisper_bridge: Optional WhisperBridge instance for transcription.
        """
        self.config = config or StreamConfig()
        self.whisper_bridge = whisper_bridge

        # Audio processor for format conversion
        self.audio_processor = AudioProcessor(
            target_sample_rate=self.config.sample_rate
        )

        # Noise reduction (pre-VAD preprocessing)
        self.noise_reducer = NoiseReducer(strength="medium")

        # VAD processor
        self.vad = VADProcessor(
            threshold=self.config.vad_threshold,
            sample_rate=self.config.sample_rate,
        )

        # Circular buffers
        max_buffer_samples = int(
            self.config.sample_rate * self.config.max_buffer_duration_s
        )
        self._audio_buffer = CircularBuffer(max_buffer_samples)
        self._pre_speech_buffer = CircularBuffer(self.config.pre_speech_samples)

        # State
        self._state = StreamState.IDLE
        self._silence_counter = 0
        self._speech_chunks: List[np.ndarray] = []
        self._total_speech_samples = 0

        # Callbacks
        self.on_speech_start: Optional[Callable[[], None]] = None
        self.on_speech_end: Optional[Callable[[np.ndarray], None]] = None
        self.on_transcription: Optional[Callable[[object], None]] = None
        self.on_state_change: Optional[Callable[[StreamState], None]] = None
        self.on_event: Optional[Callable[[StreamEvent], None]] = None

        # Threading
        self._lock = threading.Lock()
        self._processing_thread: Optional[threading.Thread] = None

        # Statistics
        self._stats = {
            "chunks_processed": 0,
            "speech_segments": 0,
            "total_speech_duration_s": 0.0,
            "total_processing_time_s": 0.0,
            "avg_latency_ms": 0.0,
        }

    @property
    def state(self) -> StreamState:
        """Current stream state."""
        return self._state

    @property
    def stats(self) -> dict:
        """Processing statistics."""
        return self._stats.copy()

    def _set_state(self, new_state: StreamState) -> None:
        """Update state and notify callback. Thread-safe."""
        with self._lock:
            if self._state != new_state:
                self._state = new_state
                logger.debug("State -> %s", new_state.value)
                callback = self.on_state_change
        # Invoke callback outside lock to avoid deadlocks
        if callback:
            try:
                callback(new_state)
            except Exception as e:
                logger.warning("State change callback error: %s", e)

    def _emit_event(self, event: StreamEvent) -> None:
        """Emit event to callback."""
        if self.on_event:
            try:
                self.on_event(event)
            except Exception as e:
                logger.warning("Event callback error: %s", e)

    # ------------------------------------------------------------------
    # Audio Input
    # ------------------------------------------------------------------

    def feed(self, audio_chunk: np.ndarray) -> None:
        """
        Feed an audio chunk into the stream processor.

        The chunk is processed through VAD and routed to the
        appropriate handler based on current state.

        Args:
            audio_chunk: Audio samples (float32, mono, 16kHz).
        """
        with self._lock:
            self._stats["chunks_processed"] += 1

            # Ensure float32
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)

            # Noise reduction (pre-VAD preprocessing)
            if self.noise_reducer is not None:
                audio_chunk = self.noise_reducer.process(
                    audio_chunk, sample_rate=self.config.sample_rate
                )

            # VAD analysis
            is_speech = self.vad.is_speech(audio_chunk)

            if is_speech:
                self._handle_speech(audio_chunk)
            else:
                self._handle_silence(audio_chunk)

    def feed_bytes(self, audio_bytes: bytes, sample_width: int = 2) -> None:
        """
        Feed raw audio bytes (e.g. from microphone stream).

        Args:
            audio_bytes: Raw audio bytes.
            sample_width: Bytes per sample (2 for int16, 4 for float32).
        """
        if sample_width == 2:
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        self.feed(audio)

    def _handle_speech(self, audio_chunk: np.ndarray) -> None:
        """Handle a chunk detected as speech."""
        if self._state == StreamState.IDLE:
            # Speech started
            self._set_state(StreamState.SPEAKING)
            self._speech_chunks = []
            self._total_speech_samples = 0
            self._silence_counter = 0

            # Include pre-speech buffer for context
            pre_samples = self._pre_speech_buffer.read(
                self._pre_speech_buffer.size
            )
            if len(pre_samples) > 0:
                self._speech_chunks.append(pre_samples)
                self._total_speech_samples += len(pre_samples)

            if self.on_speech_start:
                try:
                    self.on_speech_start()
                except Exception as e:
                    logger.warning("Speech start callback error: %s", e)

            self._emit_event(StreamEvent(event_type="speech_start"))

        # Accumulate speech chunk
        self._speech_chunks.append(audio_chunk)
        self._total_speech_samples += len(audio_chunk)
        self._silence_counter = 0

    def _handle_silence(self, audio_chunk: np.ndarray) -> None:
        """Handle a chunk detected as silence."""
        # Always keep recent audio in pre-speech buffer
        self._pre_speech_buffer.write(audio_chunk)

        if self._state == StreamState.SPEAKING:
            self._silence_counter += 1

            # Still accumulate during silence (for trailing audio)
            self._speech_chunks.append(audio_chunk)
            self._total_speech_samples += len(audio_chunk)

            if self._silence_counter >= self.config.silence_chunks:
                # Speech ended
                self._end_speech()

    def _end_speech(self) -> None:
        """Process end of speech segment."""
        self._set_state(StreamState.PROCESSING)

        # Concatenate all speech chunks
        if len(self._speech_chunks) == 0:
            self._set_state(StreamState.IDLE)
            return

        full_speech = np.concatenate(self._speech_chunks)
        speech_duration = len(full_speech) / self.config.sample_rate

        # Check minimum speech duration
        min_duration = self.config.vad_min_speech_ms / 1000.0
        if speech_duration < min_duration:
            logger.debug(
                "Speech too short (%.2fs < %.2fs), discarding",
                speech_duration, min_duration,
            )
            self._reset_speech_state()
            return

        self._stats["speech_segments"] += 1
        self._stats["total_speech_duration_s"] += speech_duration

        logger.info(
            "Speech segment: %.2fs (%d samples)",
            speech_duration, len(full_speech),
        )

        if self.on_speech_end:
            try:
                self.on_speech_end(full_speech)
            except Exception as e:
                logger.warning("Speech end callback error: %s", e)

        self._emit_event(
            StreamEvent(event_type="speech_end", audio=full_speech)
        )

        # Transcribe if bridge is available
        if self.whisper_bridge:
            self._transcribe_async(full_speech)
        else:
            self._set_state(StreamState.IDLE)

    def _transcribe_async(self, audio: np.ndarray) -> None:
        """Run transcription in a background thread."""
        self._processing_thread = threading.Thread(
            target=self._transcribe_worker,
            args=(audio,),
            daemon=True,
        )
        self._processing_thread.start()

    def _transcribe_worker(self, audio: np.ndarray) -> None:
        """Background transcription worker."""
        start_time = time.time()

        try:
            result = self.whisper_bridge.transcribe(
                audio,
                sample_rate=self.config.sample_rate,
            )

            processing_time = time.time() - start_time

            with self._lock:
                self._stats["total_processing_time_s"] += processing_time

                # Update average latency
                n = self._stats["speech_segments"]
                avg = self._stats["avg_latency_ms"]
                self._stats["avg_latency_ms"] = (
                    (avg * (n - 1) + processing_time * 1000) / n
                )

            logger.info(
                "Transcription: '%s' (%.2fs processing)",
                result.text[:50], processing_time,
            )

            if self.on_transcription:
                try:
                    self.on_transcription(result)
                except Exception as e:
                    logger.warning("Transcription callback error: %s", e)

            self._emit_event(
                StreamEvent(event_type="transcription", data=result)
            )

        except Exception as e:
            logger.error("Transcription failed: %s", e)
            self._set_state(StreamState.ERROR)
            self._emit_event(
                StreamEvent(event_type="error", data=str(e))
            )
            return
        finally:
            self._reset_speech_state()

    def _reset_speech_state(self) -> None:
        """Reset speech accumulation state. Thread-safe."""
        with self._lock:
            self._speech_chunks = []
            self._total_speech_samples = 0
            self._silence_counter = 0
        self._set_state(StreamState.IDLE)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the processor to initial state."""
        with self._lock:
            self._audio_buffer.clear()
            self._pre_speech_buffer.clear()
            self._reset_speech_state()
            self._stats = {
                "chunks_processed": 0,
                "speech_segments": 0,
                "total_speech_duration_s": 0.0,
                "total_processing_time_s": 0.0,
                "avg_latency_ms": 0.0,
            }
            logger.info("Stream processor reset")

    def update_threshold(self, threshold: float) -> None:
        """
        Update VAD energy threshold.

        Args:
            threshold: New energy threshold (0.0 - 1.0).
        """
        self.vad.threshold = threshold
        logger.info("VAD threshold updated to %.4f", threshold)

    def get_buffered_audio(self) -> np.ndarray:
        """
        Get all audio currently in the pre-speech buffer.

        Returns:
            Numpy array of buffered audio samples.
        """
        return self._pre_speech_buffer.peek(self._pre_speech_buffer.size)