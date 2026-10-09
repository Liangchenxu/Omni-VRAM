"""
Real-Time Full-Duplex Audio Stream Processor for vram_core
==========================================================

Handles real-time audio streaming with chunk-based processing,
Voice Activity Detection (VAD), low-latency transcription pipeline
integration and full-duplex barge-in (interruption) handling.

Target: < 200ms end-to-end latency on RTX 3060,
sub-millisecond barge-in signal dispatch.

Architecture:
    - StreamProcessor: Main class for real-time audio processing
    - CircularBuffer: Pre-allocated numpy ring buffer (zero-growth audio memory)
    - VADProcessor: Energy-based Voice Activity Detection (frame level)
    - StreamState: Speech-segment state machine (backward compatible)
    - DuplexState: Full-duplex conversational state machine (barge-in aware)

Full-duplex flow:
    LISTENING --(user speech ends)--> THINKING --(TTS starts)--> SPEAKING
    SPEAKING --(barge-in detected)--> LISTENING   # playback must flush/mute

The barge-in detector runs at the very head of :meth:`StreamProcessor.feed`,
before noise reduction / VAD / ASR work, so the external playback pipeline is
notified (``on_interrupt``) with sub-millisecond latency.
"""

from __future__ import annotations

import time
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy as np

from vram_core.audio_utils import AudioProcessor
from vram_core.noise_reduction import NoiseReducer

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """Stream processing states (speech segment state machine)."""
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    PROCESSING = "processing"
    ERROR = "error"


class DuplexState(Enum):
    """
    Full-duplex conversational states.

    Attributes:
        IDLE: Processor is not bound to an active conversation yet.
        LISTENING: Capturing user audio (playback muted / stopped).
        THINKING: User finished speaking, ASR / LLM pipeline is running.
        SPEAKING: System (TTS) playback in progress -- barge-in is armed.
    """
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


# Minimum length (samples) of a microphone chunk that may be correlated against
# the playback reference. Shorter chunks make the NCC estimate too noisy to veto
# a genuine interruption.
_MIN_ECHO_NCC_SAMPLES = 64


@dataclass
class BargeInEvent:
    """
    Evidence payload emitted when the user interrupts system playback.

    Attributes:
        energy: RMS energy of the triggering frame.
        threshold: Effective (dynamically weighted) threshold that was exceeded.
        frames: Number of consecutive high-energy frames observed.
        playback_elapsed_ms: Milliseconds elapsed since playback started.
        detection_latency_ms: Milliseconds from ``feed()`` entry to dispatch --
            this is the "signal truncation" latency the playback pipeline sees.
        duplex_state: Duplex state at detection time (``SPEAKING``).
        reason: Human readable trigger reason.
        timestamp: UNIX timestamp of the detection.
    """
    energy: float
    threshold: float
    frames: int
    playback_elapsed_ms: float
    detection_latency_ms: float
    duplex_state: DuplexState = DuplexState.SPEAKING
    reason: str = "energy_vad"
    timestamp: float = field(default_factory=time.time)



@dataclass
class StreamConfig:
    """
    Configuration for stream processing.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        chunk_duration_ms: Chunk size in milliseconds.
        vad_threshold: Energy threshold for VAD (0.0 - 1.0).
        vad_silence_duration_ms: Silence duration to end speech.
        vad_min_speech_ms: Minimum speech duration to process.
        max_buffer_duration_s: Maximum segment buffer duration
            (pre-allocated ring buffer capacity -- no reallocation ever).
        pre_speech_buffer_ms: Pre-speech context buffer duration.
        overlap_ms: Overlap between chunks for continuity (informational).

        enable_barge_in: Arm the barge-in detector while playback is active.
        interrupt_energy_thresh: Base RMS energy required to accept a barge-in
            frame (before sensitivity / echo weighting).
        interrupt_min_frames: Consecutive high-energy frames (K) required
            before the interrupt signal is raised.
        barge_in_sensitivity: User sensitivity multiplier; values > 1.0 lower the
            effective threshold (easier to interrupt), < 1.0 raise it.
        echo_suppression_factor: Extra threshold multiplier applied right after
            playback starts, to avoid triggering on the speaker echo of the TTS.
        echo_suppression_decay_ms: Time (ms) over which ``echo_suppression_factor``
            decays linearly back to 1.0 (floating-weight suppression).
        vad_frame_ms: Frame length used for frame-level VAD / barge-in scanning.
        echo_ncc_threshold: Peak normalised cross-correlation (NCC) against the
            playback reference above which a candidate barge-in is vetoed as our
            own speaker echo (v2.6.1). Raise it for aggressive interruption, lower
            it when the speaker/microphone coupling is loud.
        echo_ncc_max_lag_ms: Acoustic propagation delay search range for the NCC
            echo test (0 .. ``echo_ncc_max_lag_ms`` milliseconds).
        playback_reference_duration_s: Capacity of the playback reference ring
            buffer (how much TTS audio is kept for echo correlation).
        async_upload: Mount the pinned-memory asynchronous upload channel so that
            audio intake overlaps GPU work (v2.6.0).
    """
    sample_rate: int = 16000
    chunk_duration_ms: int = 100          # Chunk size in milliseconds
    vad_threshold: float = 0.02           # Energy threshold for VAD
    vad_silence_duration_ms: int = 800    # Silence duration to end speech
    vad_min_speech_ms: int = 200          # Minimum speech duration to process
    max_buffer_duration_s: float = 30.0   # Maximum buffer duration
    pre_speech_buffer_ms: int = 200       # Pre-speech context buffer
    overlap_ms: int = 50                  # Overlap between chunks for continuity

    # ---- Full-duplex / barge-in ----
    enable_barge_in: bool = True
    interrupt_energy_thresh: float = 0.06
    interrupt_min_frames: int = 3
    barge_in_sensitivity: float = 1.0
    echo_suppression_factor: float = 2.0
    echo_suppression_decay_ms: int = 400
    vad_frame_ms: int = 25

    # ---- Acoustic echo cancellation / NCC veto (v2.6.1) ----
    # Energy gating alone cannot separate the user from the TTS speaker: a loud
    # passage of our own playback looks exactly like a barge-in. The detector
    # therefore cross-correlates the microphone chunk with the PCM it is
    # currently playing and vetoes the interruption when the two are coherent.
    echo_ncc_threshold: float = 0.55
    echo_ncc_max_lag_ms: int = 400
    playback_reference_duration_s: float = 2.0

    # ---- GPU pipeline (v2.6.0) ----
    # Pinned-memory host->device upload channel (CUDA stream isolated). Off by
    # default: enabling it costs a pinned buffer, so it is opt-in for GPU setups.
    async_upload: bool = False

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

    @property
    def max_buffer_samples(self) -> int:
        """Capacity (in samples) of the segment ring buffer."""
        return int(self.sample_rate * self.max_buffer_duration_s)

    @property
    def vad_frame_size(self) -> int:
        """VAD frame length in samples."""
        return max(1, int(self.sample_rate * self.vad_frame_ms / 1000))


class PinnedUploadChannel:
    """
    Pinned-memory asynchronous host->device upload channel (v2.6.0).

    Audio chunks arrive on the CPU (microphone / WebSocket) while the GPU is
    still busy with the previous chunk. Uploading through a *page-locked*
    (pinned) staging buffer on a **dedicated CUDA stream** lets the DMA engine
    move the data without paging the host buffer, so the H2D copy overlaps the
    GPU front-end instead of serialising with it::

        channel = PinnedUploadChannel(buffer_size=1600)
        device_chunk = channel.upload(chunk)   # returns immediately
        ... GPU work for the previous chunk ...
        channel.synchronize()                  # data is now ready

    Every chunk reuses the same staging buffer, so a long stream performs no
    per-chunk allocation.

    Degradation: without torch (or without a CUDA device) the channel runs in
    ``simulated`` mode -- it keeps the staging semantics (single reused buffer,
    no per-chunk allocation, timing statistics) but performs the copy in host
    memory, so profiling code and tests work everywhere.
    """

    def __init__(self, buffer_size: int = 16000, device: int = 0, enabled: bool = True):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")

        self.buffer_size = int(buffer_size)
        self.device_index = int(device)
        self.enabled = bool(enabled)
        self.backend = "simulated"
        self._torch = None
        self._stream = None
        self._device = None
        self._staging_tensor = None
        self._device_buffer = None
        self._staging = np.zeros(self.buffer_size, dtype=np.float32)
        self._lock = threading.Lock()

        self._uploads = 0
        self._bytes = 0
        self._last_upload_ms = 0.0
        self._total_upload_ms = 0.0

        if self.enabled:
            self._try_enable_cuda()

    def _try_enable_cuda(self) -> None:
        """Enable pinned memory + a private CUDA stream when possible."""
        try:
            import torch
        except ImportError:
            logger.info(
                "PinnedUploadChannel: torch unavailable, using simulated upload backend"
            )
            return
        if not torch.cuda.is_available():
            logger.info(
                "PinnedUploadChannel: no CUDA device, using simulated upload backend"
            )
            return
        try:
            device = torch.device(f"cuda:{self.device_index}")
            self._staging_tensor = torch.zeros(
                self.buffer_size, dtype=torch.float32, pin_memory=True
            )
            self._device_buffer = torch.zeros(
                self.buffer_size, dtype=torch.float32, device=device
            )
            self._stream = torch.cuda.Stream(device=device)
            self._torch = torch
            self._device = device
            self.backend = "cuda"
            logger.info(
                "PinnedUploadChannel: pinned-memory upload on a dedicated CUDA stream"
            )
        except (RuntimeError, OSError) as error:
            logger.warning(
                "PinnedUploadChannel: CUDA init failed (%s), using simulated backend",
                error,
            )
            self.backend = "simulated"

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def is_cuda(self) -> bool:
        """True when pinned memory + a CUDA stream are used."""
        return self.backend == "cuda"

    @property
    def staging_buffer(self) -> np.ndarray:
        """Reusable host staging buffer."""
        return self._staging

    # ── Upload ────────────────────────────────────────────────────────────
    def upload(self, chunk: np.ndarray):
        """
        Queue one audio chunk for upload (non-blocking on the CUDA backend).

        Args:
            chunk: Audio samples (float32 preferred). Longer chunks are
                truncated to ``buffer_size``.

        Returns:
            A device tensor view (CUDA backend) or None (simulated backend).
        """
        if not self.enabled or chunk is None:
            return None

        samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
        count = min(samples.size, self.buffer_size)
        if count == 0:
            return None

        start = time.perf_counter()
        result = None
        if self.backend == "cuda":
            with self._lock:
                self._staging_tensor[:count].copy_(
                    self._torch.from_numpy(np.ascontiguousarray(samples[:count]))
                )
                # Order the private copy stream against the work already queued
                # on the caller's stream. The device buffer is (re)allocated and
                # zero-initialised on that stream when the channel is created --
                # without this handshake the memset can race the H2D copy and
                # silently leave the buffer zeroed.
                current = self._torch.cuda.current_stream(self._device)
                self._stream.wait_stream(current)
                with self._torch.cuda.stream(self._stream):
                    self._device_buffer[:count].copy_(
                        self._staging_tensor[:count], non_blocking=True
                    )
                # ...and make the caller's stream observe the finished copy, so
                # a subsequent read (``.cpu()`` / a front-end kernel) is ordered
                # after it instead of running concurrently.
                current.wait_stream(self._stream)
                result = self._device_buffer[:count]
        else:
            self._staging[:count] = samples[:count]

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        with self._lock:
            self._uploads += 1
            self._bytes += int(count * samples.dtype.itemsize)
            self._last_upload_ms = elapsed_ms
            self._total_upload_ms += elapsed_ms
        return result

    def synchronize(self) -> None:
        """Block until every queued H2D copy has completed."""
        if self.backend == "cuda" and self._stream is not None:
            self._stream.synchronize()

    def stats(self) -> dict:
        """Upload channel statistics (backend, volume, last/average latency)."""
        average_ms = self._total_upload_ms / self._uploads if self._uploads else 0.0
        return {
            "backend": self.backend,
            "buffer_size": self.buffer_size,
            "uploads": self._uploads,
            "bytes": self._bytes,
            "last_upload_ms": self._last_upload_ms,
            "avg_upload_ms": average_ms,
        }

    def close(self) -> None:
        """Flush pending copies and release the device buffers."""
        self.synchronize()
        self._staging_tensor = None
        self._device_buffer = None
        self._stream = None
        self.backend = "simulated"

    def __enter__(self) -> "PinnedUploadChannel":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown safety
            pass



class CircularBuffer:
    """
    Thread-safe, pre-allocated ring buffer for ``float32`` audio samples.

    Unlike the previous ``deque``-based implementation this buffer owns a single
    contiguous ``numpy`` array, so writing audio never allocates per sample and
    never triggers ``np.concatenate`` / ``np.append`` growth: memory usage stays
    constant regardless of how long the stream runs (real-time safe).

    All operations are O(1) amortised; the only copies are the ``n`` samples
    returned by :meth:`read` / :meth:`peek`. When the buffer is full the oldest
    samples are silently evicted (sliding-window semantics).

    Args:
        max_samples: Capacity in samples (must be > 0).
    """

    __slots__ = ("capacity", "_data", "_head", "_tail", "_count",
                 "_total_written", "_lock")

    def __init__(self, max_samples: int):
        if max_samples is None or max_samples <= 0:
            raise ValueError("max_samples must be a positive integer")
        self.capacity = int(max_samples)
        self._data = np.zeros(self.capacity, dtype=np.float32)
        self._head = 0
        self._tail = 0
        self._count = 0
        self._total_written = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers (callers must hold ``self._lock``)
    # ------------------------------------------------------------------

    def _copy_from(self, start: int, n_samples: int) -> np.ndarray:
        """Copy ``n_samples`` starting at physical index ``start`` (wrapping)."""
        end = start + n_samples
        if end <= self.capacity:
            return self._data[start:end].copy()
        first = self.capacity - start
        out = np.empty(n_samples, dtype=np.float32)
        out[:first] = self._data[start:]
        out[first:] = self._data[:n_samples - first]
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, data: np.ndarray) -> int:
        """
        Write audio samples to the buffer (evicting oldest on overflow).

        Args:
            data: Audio samples to write.

        Returns:
            Number of samples written.
        """
        if data is None:
            return 0
        samples = np.asarray(data, dtype=np.float32).reshape(-1)
        n_samples = int(samples.size)
        if n_samples == 0:
            return 0
        samples = np.ascontiguousarray(samples)

        with self._lock:
            self._total_written += n_samples
            if n_samples >= self.capacity:
                # Only the newest ``capacity`` samples are relevant.
                self._data[:] = samples[-self.capacity:]
                self._head = 0
                self._count = self.capacity
                self._tail = 0
                return n_samples

            space_to_end = self.capacity - self._head
            if n_samples <= space_to_end:
                self._data[self._head:self._head + n_samples] = samples
            else:
                self._data[self._head:] = samples[:space_to_end]
                self._data[:n_samples - space_to_end] = samples[space_to_end:]

            self._head = (self._head + n_samples) % self.capacity
            self._count += n_samples
            if self._count >= self.capacity:
                self._count = self.capacity
                self._tail = self._head
            return n_samples

    def read(self, n_samples: int) -> np.ndarray:
        """
        Read and remove n samples from the buffer.

        Args:
            n_samples: Number of samples to read.

        Returns:
            Audio samples as numpy array (float32).
        """
        with self._lock:
            n = min(int(n_samples), self._count)
            if n <= 0:
                return np.array([], dtype=np.float32)
            out = self._copy_from(self._tail, n)
            self._tail = (self._tail + n) % self.capacity
            self._count -= n
            return out

    def read_all(self) -> np.ndarray:
        """
        Read and remove every buffered sample (chronological order).

        Returns:
            Audio samples as numpy array (float32).
        """
        return self.read(self.size)

    def peek(self, n_samples: int) -> np.ndarray:
        """
        Read n samples without removing them.

        This is the sliding-window extractor: ``peek(window_samples)`` always
        returns the most recent window of audio at O(n) cost using the
        pre-allocated storage (no growth, no ``np.concatenate``).

        Args:
            n_samples: Number of samples to peek.

        Returns:
            Audio samples as numpy array (float32).
        """
        with self._lock:
            n = min(int(n_samples), self._count)
            if n <= 0:
                return np.array([], dtype=np.float32)
            start = (self._tail + self._count - n) % self.capacity
            return self._copy_from(start, n)

    def extract_window(self, n_samples: int) -> np.ndarray:
        """Sliding-window extraction alias of :meth:`peek`."""
        return self.peek(n_samples)

    def clear(self) -> None:
        """Clear the buffer (keeps the pre-allocated memory)."""
        with self._lock:
            self._head = 0
            self._tail = 0
            self._count = 0

    @property
    def size(self) -> int:
        """Current number of samples in buffer."""
        with self._lock:
            return self._count

    @property
    def available(self) -> int:
        """Remaining free space in samples."""
        with self._lock:
            return self.capacity - self._count

    @property
    def total_written(self) -> int:
        """Total number of samples ever written (diagnostics only)."""
        with self._lock:
            return self._total_written

    @property
    def memory_bytes(self) -> int:
        """Bytes of pre-allocated storage owned by this buffer (constant)."""
        return int(self._data.nbytes)

    @property
    def is_empty(self) -> bool:
        """True when no samples are buffered."""
        return self.size == 0

    @property
    def is_full(self) -> bool:
        """True when the buffer is at capacity."""
        return self.size >= self.capacity

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"CircularBuffer(capacity={self.capacity}, size={self.size}, "
                f"dtype=float32)")


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

    def frame_energies(
        self,
        audio: np.ndarray,
        frame_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute per-frame RMS energies (vectorised, allocation friendly).

        Used by the barge-in detector to count *consecutive* high-energy frames
        without any Python-level per-sample loop.

        Args:
            audio: Audio chunk (float32).
            frame_size: Frame length in samples (defaults to ``self.frame_size``).

        Returns:
            float32 array with one RMS energy per full frame. If the chunk is
            shorter than one frame a single energy covering the whole chunk is
            returned; an empty chunk yields an empty array.
        """
        if audio is None or len(audio) == 0:
            return np.array([], dtype=np.float32)
        size = int(frame_size or self.frame_size)
        if size <= 0:
            size = self.frame_size
        usable = (len(audio) // size) * size
        if usable < size:
            return np.array([self.compute_energy(audio)], dtype=np.float32)
        frames = np.asarray(audio, dtype=np.float32)[:usable].reshape(-1, size)
        return np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32)

    def speech_confidence(self, audio: np.ndarray) -> float:
        """
        Map RMS energy onto a 0..1 speech confidence.

        The mapping is smooth (``0.5`` at the VAD threshold) so callers can
        weigh decisions instead of relying on a hard boolean.

        Args:
            audio: Audio chunk (float32).

        Returns:
            Confidence in the range 0.0 - 1.0.
        """
        energy = self.compute_energy(audio)
        if energy <= 0.0:
            return 0.0
        if self.threshold <= 0.0:
            return 1.0
        return float(energy / (energy + self.threshold))

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
    """Event emitted by the stream processor.

    ``event_type`` is one of ``"speech_start"``, ``"speech_end"``,
    ``"transcription"``, ``"interrupt"``, ``"playback_start"``,
    ``"playback_stop"`` or ``"error"``.
    """
    event_type: str  # "speech_start", "speech_end", "transcription", "error"
    timestamp: float = field(default_factory=time.time)
    data: Optional[object] = None
    audio: Optional[np.ndarray] = None


@dataclass
class _FeedOutcome:
    """
    Internal container for work produced while the state lock was held.

    Callbacks and events are queued here instead of being invoked inline so the
    lock is always released before user code runs (prevents re-entrancy
    deadlocks and keeps the audio thread free of user latency).

    Attributes:
        callbacks: Zero-argument callables to run after unlocking.
        events: ``StreamEvent`` objects to publish after unlocking.
        segment: Completed speech segment ready for transcription.
    """
    callbacks: List[Callable[[], None]] = field(default_factory=list)
    events: List[StreamEvent] = field(default_factory=list)
    segment: Optional[np.ndarray] = None


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
        upload_channel: Optional[PinnedUploadChannel] = None,
        async_upload: Optional[bool] = None,
    ):
        """
        Initialize stream processor.

        Args:
            config: Stream processing configuration.
            whisper_bridge: Optional WhisperBridge instance for transcription.
            upload_channel: Optional pre-built :class:`PinnedUploadChannel`.
            async_upload: Override ``config.async_upload``; when enabled (and no
                channel is injected) a pinned-memory upload channel is created so
                that audio intake overlaps GPU work.
        """
        self.config = config or StreamConfig()
        self.whisper_bridge = whisper_bridge

        # Audio processor for format conversion
        self.audio_processor = AudioProcessor(
            target_sample_rate=self.config.sample_rate
        )

        # Noise reduction (pre-VAD preprocessing)
        self.noise_reducer = NoiseReducer(strength="medium")

        # VAD processor (frame level, shared with the barge-in detector)
        self.vad = VADProcessor(
            threshold=self.config.vad_threshold,
            sample_rate=self.config.sample_rate,
            frame_size_ms=self.config.vad_frame_ms,
        )

        # Pinned-memory asynchronous upload channel (v2.6.0, opt-in)
        self._upload_channel = upload_channel
        enable_upload = (
            bool(self.config.async_upload) if async_upload is None else bool(async_upload)
        )
        if self._upload_channel is None and enable_upload:
            self._upload_channel = PinnedUploadChannel(
                buffer_size=max(self.config.chunk_size, self.config.vad_frame_size),
            )

        # Pre-allocated ring buffers (fixed memory, no concatenation growth)
        self._audio_buffer = CircularBuffer(self.config.max_buffer_samples)
        self._pre_speech_buffer = CircularBuffer(self.config.pre_speech_samples)

        # Speech segment state
        self._state = StreamState.IDLE
        self._duplex_state = DuplexState.IDLE
        self._silence_counter = 0
        self._total_speech_samples = 0

        # Full-duplex / barge-in state
        self._playback_active = False
        self._playback_started_at: Optional[float] = None
        self._interrupt_frames = 0

        # Playback reference ring buffer (v2.6.1 AEC). The TTS playback thread
        # pushes every chunk it sends to the speaker through
        # :meth:`register_playback_chunk`; the barge-in detector correlates the
        # microphone signal against this window to reject its own echo.
        self._playback_reference_buffer = CircularBuffer(
            max(
                self.config.vad_frame_size,
                int(self.config.sample_rate * self.config.playback_reference_duration_s),
            )
        )

        # Callbacks
        self.on_speech_start: Optional[Callable[[], None]] = None
        self.on_speech_end: Optional[Callable[[np.ndarray], None]] = None
        self.on_transcription: Optional[Callable[[object], None]] = None
        self.on_state_change: Optional[Callable[[StreamState], None]] = None
        self.on_event: Optional[Callable[[StreamEvent], None]] = None
        self.on_interrupt: Optional[Callable[[BargeInEvent], None]] = None
        self.on_duplex_state_change: Optional[Callable[[DuplexState], None]] = None

        # Threading -- re-entrant lock: helpers are called from locked sections
        # and callbacks are always dispatched after the lock is released.
        self._lock = threading.RLock()
        self._processing_thread: Optional[threading.Thread] = None

        # Statistics
        self._stats = {
            "chunks_processed": 0,
            "speech_segments": 0,
            "total_speech_duration_s": 0.0,
            "total_processing_time_s": 0.0,
            "avg_latency_ms": 0.0,
            "interrupts": 0,
            "last_interrupt_latency_ms": 0.0,
            "echo_vetoes": 0,
        }

    @property
    def state(self) -> StreamState:
        """Current speech-segment state."""
        return self._state

    @property
    def duplex_state(self) -> DuplexState:
        """Current full-duplex conversational state."""
        return self._duplex_state

    @property
    def is_playback_active(self) -> bool:
        """True while external playback (TTS) is running."""
        return self._playback_active

    @property
    def memory_footprint_bytes(self) -> int:
        """Bytes of pre-allocated audio memory (constant over time)."""
        return (
            self._audio_buffer.memory_bytes
            + self._pre_speech_buffer.memory_bytes
            + self._playback_reference_buffer.memory_bytes
        )

    @property
    def upload_channel(self) -> Optional[PinnedUploadChannel]:
        """The pinned-memory upload channel (None when disabled)."""
        return self._upload_channel

    @property
    def upload_stats(self) -> dict:
        """
        Statistics of the asynchronous upload channel.

        Returns ``{"enabled": False}`` when the channel is not mounted, so that
        monitoring code can call it unconditionally.
        """
        if self._upload_channel is None:
            return {"enabled": False, "backend": "disabled"}
        stats = dict(self._upload_channel.stats())
        stats["enabled"] = True
        return stats

    @property
    def stats(self) -> dict:
        """Processing statistics."""
        return self._stats.copy()

    def _set_state(
        self,
        new_state: StreamState,
        outcome: Optional["_FeedOutcome"] = None,
    ) -> None:
        """
        Update the speech state and notify ``on_state_change``.

        Thread-safe: the callback is always invoked *after* the state lock is
        released. When ``outcome`` is provided (feed path) the notification is
        queued so the caller can dispatch it once it owns no lock.

        Args:
            new_state: New ``StreamState`` value.
            outcome: Optional notification collector for deferred dispatch.
        """
        with self._lock:
            if self._state == new_state:
                return
            self._state = new_state
            callback = self.on_state_change
        logger.debug("State -> %s", new_state.value)
        if callback is not None:
            if outcome is not None:
                outcome.callbacks.append(
                    lambda cb=callback, st=new_state: self._safe_call(cb, st)
                )
            else:
                self._safe_call(callback, new_state)

    def _set_duplex_state(
        self,
        new_state: DuplexState,
        outcome: Optional["_FeedOutcome"] = None,
    ) -> None:
        """
        Update the full-duplex state and notify ``on_duplex_state_change``.

        Args:
            new_state: New ``DuplexState`` value.
            outcome: Optional notification collector for deferred dispatch.
        """
        with self._lock:
            if self._duplex_state == new_state:
                return
            self._duplex_state = new_state
            callback = self.on_duplex_state_change
        logger.debug("Duplex state -> %s", new_state.value)
        if callback is not None:
            if outcome is not None:
                outcome.callbacks.append(
                    lambda cb=callback, st=new_state: self._safe_call(cb, st)
                )
            else:
                self._safe_call(callback, new_state)

    @staticmethod
    def _safe_call(callback: Callable, *args) -> None:
        """Invoke a user callback, swallowing (and logging) exceptions."""
        try:
            callback(*args)
        except Exception as e:  # noqa: BLE001 - user callback must never break audio
            logger.warning("Callback error: %s", e)

    def _emit_event(
        self,
        event: StreamEvent,
        outcome: Optional["_FeedOutcome"] = None,
    ) -> None:
        """
        Emit a stream event to ``on_event``.

        Args:
            event: Event to publish.
            outcome: Optional notification collector for deferred dispatch.
        """
        if outcome is not None:
            outcome.events.append(event)
            return
        if self.on_event:
            self._safe_call(self.on_event, event)

    # ------------------------------------------------------------------
    # Full-Duplex / Barge-In
    # ------------------------------------------------------------------

    def set_playback_state(self, is_playing: bool) -> None:
        """
        Notify the processor that external playback (TTS) starts or stops.

        While playback is active the processor enters ``DuplexState.SPEAKING``
        and arms the barge-in detector. The effective interrupt threshold is
        weighted by ``echo_suppression_factor`` for the first
        ``echo_suppression_decay_ms`` milliseconds (floating weight) so the
        speaker echo of the TTS cannot trigger a false interruption; on top of
        that the detector cross-correlates every candidate chunk against the
        playback reference (see :meth:`register_playback_chunk`) and vetoes the
        interruption outright when the two are coherent.

        Args:
            is_playing: True when playback starts, False when it stops.
        """
        now = time.perf_counter()
        with self._lock:
            self._playback_active = bool(is_playing)
            self._playback_started_at = now if is_playing else None
            self._interrupt_frames = 0
            # A new playback session invalidates the previous reference: fresh
            # TTS audio is registered chunk by chunk from here on.
            self._playback_reference_buffer.clear()

        if is_playing:
            self._set_duplex_state(DuplexState.SPEAKING)
            self._emit_event(StreamEvent(event_type="playback_start"))
        else:
            self._set_duplex_state(DuplexState.LISTENING)
            self._emit_event(StreamEvent(event_type="playback_stop"))
        logger.info("Playback state -> %s", "playing" if is_playing else "stopped")

    def _is_barge_in_armed(self) -> bool:
        """True when barge-in detection should scan incoming chunks."""
        return (
            self.config.enable_barge_in
            and self._playback_active
            and self._duplex_state is DuplexState.SPEAKING
        )

    # ---- Playback reference & acoustic echo veto (v2.6.1) ---------------

    def register_playback_chunk(self, chunk: np.ndarray) -> int:
        """
        Record PCM that is about to be sent to the speaker.

        The TTS output thread calls this for every chunk it hands to the playback
        device, so the processor owns a ~2 s ring buffer of exactly what is
        currently audible. The barge-in detector correlates the microphone signal
        against that window to reject *its own* playback as an interruption.

        Args:
            chunk: Mono PCM of the emitted chunk (float32 preferred).

        Returns:
            Number of samples written (0 for empty/None input).
        """
        if chunk is None:
            return 0
        samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return 0
        with self._lock:
            self._playback_reference_buffer.write(samples)
        return int(samples.size)

    def clear_playback_reference(self) -> None:
        """Forget the playback reference (call when TTS audio is flushed)."""
        with self._lock:
            self._playback_reference_buffer.clear()

    @property
    def playback_reference_buffer(self) -> "CircularBuffer":
        """Ring buffer holding the most recent playback (speaker) audio."""
        return self._playback_reference_buffer

    def _echo_ncc_peak(self, audio_chunk: np.ndarray) -> float:
        """
        Peak normalised cross-correlation between the microphone and playback.

        For every acoustic propagation delay ``tau`` in
        ``[0, echo_ncc_max_lag_ms]`` the normalised cross-correlation

        .. math::

            R(\\tau) = \\frac{\\sum (x[t] - \\mu_x)(y[t + \\tau] - \\mu_y)}
                       {\\sqrt{\\sum (x[t] - \\mu_x)^2 \\sum (y[t + \\tau] - \\mu_y)^2}}

        is evaluated, where ``x`` is the microphone chunk and ``y`` the reference
        (speaker) signal. The lag sweep is a single vectorised
        ``np.correlate(..., mode="valid")`` plus a cumulative-sum energy
        normalisation, so no per-lag Python loop or allocation is involved. On a
        16 kHz CPU stream the full sweep costs a few milliseconds (~7 ms for a
        2 s reference and a 400 ms lag range) and is only paid once the
        frame-level energy gate has already fired.

        Args:
            audio_chunk: Raw microphone chunk.

        Returns:
            Highest ``|R(tau)|`` found, in ``[0, 1]``; ``0.0`` when no playback
            reference is available (the common, echoless case).
        """
        reference = self._playback_reference_buffer.peek(
            self._playback_reference_buffer.size
        )
        if reference.size == 0:
            return 0.0

        x = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if x.size < _MIN_ECHO_NCC_SAMPLES or reference.size < x.size:
            return 0.0
        # Only the newest ``len(x) + max_lag`` reference samples can echo into
        # this chunk, which also bounds the correlation cost.
        max_lag = int(
            self.config.sample_rate * max(0, self.config.echo_ncc_max_lag_ms) / 1000.0
        )
        max_lag = int(np.clip(max_lag, 0, reference.size - x.size))
        # ``max_lag == 0`` degenerates to the single (zero-delay) lag, which is
        # still a valid echo test; the reference is never shorter than the chunk.
        segment_length = x.size + max_lag
        ref = reference[-segment_length:]

        x_zm = x - x.mean()
        ref_zm = ref - ref.mean()
        x_energy = float(np.dot(x_zm, x_zm))
        if x_energy <= 1e-12:
            return 0.0

        # cross[j] = sum_i ref[j + i] * x[i]  ->  tau = max_lag - j
        cross = np.correlate(ref_zm.astype(np.float64), x_zm.astype(np.float64), mode="valid")
        cumulative = np.concatenate(([0.0], np.cumsum(ref_zm.astype(np.float64) ** 2)))
        window_energy = (
            cumulative[x.size: x.size + max_lag + 1] - cumulative[: max_lag + 1]
        )
        denominator = np.sqrt(np.maximum(window_energy, 1e-12) * x_energy)
        rho = np.abs(cross / denominator)
        peak = float(np.max(rho)) if rho.size else 0.0
        return peak if np.isfinite(peak) else 0.0

    def _effective_interrupt_threshold(self, now: float) -> float:
        """
        Compute the dynamically weighted barge-in threshold.

        ``threshold = interrupt_energy_thresh / barge_in_sensitivity`` with an
        additional damping factor that decays linearly from
        ``echo_suppression_factor`` (right after playback start) back to ``1.0``
        after ``echo_suppression_decay_ms``.

        Args:
            now: ``time.perf_counter()`` value of the current scan.

        Returns:
            Effective RMS energy threshold.
        """
        sensitivity = max(self.config.barge_in_sensitivity, 1e-6)
        threshold = self.config.interrupt_energy_thresh / sensitivity

        started = self._playback_started_at
        decay_ms = max(self.config.echo_suppression_decay_ms, 0)
        if started is None or decay_ms <= 0:
            return threshold

        elapsed_ms = (now - started) * 1000.0
        progress = min(1.0, max(0.0, elapsed_ms / decay_ms))
        damping = self.config.echo_suppression_factor - (
            self.config.echo_suppression_factor - 1.0
        ) * progress
        return threshold * damping

    def _scan_barge_in(
        self,
        audio_chunk: np.ndarray,
        enter: float,
    ) -> Optional[BargeInEvent]:
        """
        Scan one chunk for barge-in evidence (allocation-light fast path).

        The chunk is split into ``vad_frame_ms`` frames (strided numpy view, no
        copy) and consecutive frames above the weighted threshold are counted.
        Counting continues across chunk boundaries, so the signal is raised as
        soon as ``interrupt_min_frames`` consecutive frames are observed -- a few
        milliseconds of sustained user speech.

        Once the energy gate is satisfied the candidate is confirmed acoustically
        (v2.6.1): the microphone chunk is cross-correlated with the playback
        reference and a peak above ``echo_ncc_threshold`` vetoes the interruption
        as our own speaker echo. Only genuinely uncorrelated (user) audio is
        allowed through.

        Args:
            audio_chunk: Raw (unprocessed) audio chunk.
            enter: ``time.perf_counter()`` captured at :meth:`feed` entry.

        Returns:
            A :class:`BargeInEvent` when the frame gate is satisfied and the
            chunk is not a playback echo, else None.
        """
        if audio_chunk is None or audio_chunk.size == 0:
            return None

        now = time.perf_counter()
        threshold = self._effective_interrupt_threshold(now)
        energies = self.vad.frame_energies(audio_chunk)

        min_frames = max(1, int(self.config.interrupt_min_frames))
        with self._lock:
            # Frame counting continues across chunk boundaries, so a barge-in
            # that starts just before a chunk edge is still confirmed.
            frames = self._interrupt_frames

        triggered = False
        trigger_energy = 0.0
        for energy in energies:
            if float(energy) > threshold:
                frames += 1
                trigger_energy = float(energy)
                if frames >= min_frames:
                    triggered = True
                    break
            else:
                frames = 0

        with self._lock:
            self._interrupt_frames = min(frames, min_frames)

        if not triggered:
            return None

        # ---- Acoustic echo veto (v2.6.1) --------------------------------
        # A loud TTS passage re-entering through the microphone satisfies the
        # energy gate, so the candidate is only accepted when it does *not*
        # correlate with what we are currently playing.
        ncc_peak = self._echo_ncc_peak(audio_chunk)
        if ncc_peak >= self.config.echo_ncc_threshold:
            with self._lock:
                # The run is discarded: an echo must not arm the next real chunk.
                self._interrupt_frames = 0
                self._stats["echo_vetoes"] += 1
            logger.debug(
                "Barge-in vetoed as playback echo (NCC=%.3f >= %.3f)",
                ncc_peak, self.config.echo_ncc_threshold,
            )
            return None

        started = self._playback_started_at or now
        return BargeInEvent(
            energy=trigger_energy,
            threshold=threshold,
            frames=frames,
            playback_elapsed_ms=(now - started) * 1000.0,
            detection_latency_ms=(time.perf_counter() - enter) * 1000.0,
            duplex_state=DuplexState.SPEAKING,
        )

    def _trigger_interrupt(self, event: BargeInEvent) -> None:
        """
        Raise the barge-in signal and drop the interrupted turn's audio.

        Ordering matters: ``on_interrupt`` is invoked *first* (so the playback
        pipeline can mute/flush immediately), then the stale input of the
        interrupted turn is discarded and the machine returns to
        ``DuplexState.LISTENING``. Nothing here blocks the audio thread.

        The detector is disarmed as part of the interrupt (``is_playback_active``
        becomes False), because the truncated utterance is no longer playing; the
        next playback session re-arms it through
        :meth:`set_playback_state`.

        Args:
            event: Detection evidence produced by :meth:`_scan_barge_in`.
        """
        with self._lock:
            if self._duplex_state is not DuplexState.SPEAKING:
                return
            self._interrupt_frames = 0
            self._playback_active = False
            self._playback_started_at = None
            self._pre_speech_buffer.clear()  # drop TTS tail / echo context
            self._playback_reference_buffer.clear()  # ... and its reference
            self._stats["interrupts"] += 1
            self._stats["last_interrupt_latency_ms"] = event.detection_latency_ms
            callback = self.on_interrupt

        logger.info(
            "Barge-in detected (energy=%.4f > thr=%.4f, frames=%d, latency=%.3f ms)",
            event.energy, event.threshold, event.frames,
            event.detection_latency_ms,
        )

        # 1) Truncation signal -- the most latency critical notification
        if callback is not None:
            self._safe_call(callback, event)
        else:
            logger.debug("Barge-in detected but no on_interrupt callback registered")

        # 2) Discard the interrupted turn, re-enter listening mode
        self._reset_speech_state()
        self._emit_event(StreamEvent(event_type="interrupt", data=event))

    # ------------------------------------------------------------------
    # Audio Input
    # ------------------------------------------------------------------

    def feed(self, audio_chunk: np.ndarray) -> None:
        """
        Feed an audio chunk into the stream processor.

        Processing order (latency critical first):

        1. **Barge-in fast path** -- a raw, frame-level energy scan that can
           raise ``on_interrupt`` synchronously (sub-millisecond) while the
           system is speaking.
        2. Noise reduction (pre-VAD preprocessing).
        3. VAD analysis + speech state machine (ring-buffer accumulation).

        All user callbacks are dispatched *after* the internal state lock has
        been released, so a callback may safely call back into the processor.

        Args:
            audio_chunk: Audio samples (float32, mono, 16kHz).
        """
        if audio_chunk is None:
            return
        enter = time.perf_counter()
        chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)

        # ---- Priority path: barge-in detection -------------------------
        if self._is_barge_in_armed():
            event = self._scan_barge_in(chunk, enter)
            if event is not None:
                self._trigger_interrupt(event)

        # ---- Overlapped host->device upload (pinned memory + CUDA stream) ----
        # Queued here, before the CPU-bound noise reduction / VAD work, so the
        # DMA transfer overlaps the current chunk's processing.
        if self._upload_channel is not None:
            self._upload_channel.upload(chunk)

        # ---- Regular pipeline -----------------------------------------
        if self.noise_reducer is not None and chunk.size:
            chunk = self.noise_reducer.process(
                chunk, sample_rate=self.config.sample_rate
            )

        is_speech = self.vad.is_speech(chunk)

        outcome = _FeedOutcome()
        with self._lock:
            self._stats["chunks_processed"] += 1
            if self._duplex_state is DuplexState.IDLE:
                self._set_duplex_state(DuplexState.LISTENING, outcome)
            if is_speech:
                self._handle_speech(chunk, outcome)
            else:
                self._handle_silence(chunk, outcome)

        # Dispatch notifications outside the lock (callbacks may re-enter)
        self._dispatch(outcome)
        if outcome.segment is not None:
            self._finish_segment(outcome.segment)

    def _dispatch(self, outcome: "_FeedOutcome") -> None:
        """Publish queued events/callbacks. Never called while holding a lock."""
        for event in outcome.events:
            if self.on_event:
                self._safe_call(self.on_event, event)
        for callback in outcome.callbacks:
            callback()

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

    def _handle_speech(
        self,
        audio_chunk: np.ndarray,
        outcome: "_FeedOutcome",
    ) -> None:
        """
        Handle a chunk detected as speech (state lock held).

        Args:
            audio_chunk: Speech chunk (already noise reduced).
            outcome: Notification collector for deferred dispatch.
        """
        if self._state is StreamState.IDLE:
            self._start_speech(outcome)
        elif self._duplex_state is DuplexState.THINKING:
            # User is talking again while ASR/LLM is still running
            self._set_duplex_state(DuplexState.LISTENING, outcome)

        self._accumulate(audio_chunk)
        self._silence_counter = 0

    def _start_speech(self, outcome: "_FeedOutcome") -> None:
        """
        Open a new speech segment: reset the ring buffer, prepend pre-speech
        context and fire ``on_speech_start`` (state lock held).
        """
        self._set_state(StreamState.SPEAKING, outcome)
        if self._duplex_state in (DuplexState.IDLE, DuplexState.THINKING):
            self._set_duplex_state(DuplexState.LISTENING, outcome)

        self._silence_counter = 0
        self._total_speech_samples = 0
        self._audio_buffer.clear()

        # Include pre-speech context for better ASR accuracy
        pre_samples = self._pre_speech_buffer.read(self._pre_speech_buffer.size)
        if pre_samples.size:
            self._accumulate(pre_samples)

        logger.debug("Speech started")
        if self.on_speech_start is not None:
            outcome.callbacks.append(lambda: self._safe_call(self.on_speech_start))
        outcome.events.append(StreamEvent(event_type="speech_start"))

    def _accumulate(self, audio_chunk: np.ndarray) -> None:
        """
        Append audio to the segment ring buffer (state lock held).

        Uses pre-allocated contiguous memory: no ``np.concatenate`` /
        ``np.append`` growth, so memory usage is bounded by
        ``max_buffer_duration_s`` regardless of segment length.

        Args:
            audio_chunk: Audio samples to append.
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return
        self._audio_buffer.write(audio_chunk)
        self._total_speech_samples = min(
            self._total_speech_samples + int(len(audio_chunk)),
            self._audio_buffer.capacity,
        )

    def _handle_silence(
        self,
        audio_chunk: np.ndarray,
        outcome: "_FeedOutcome",
    ) -> None:
        """
        Handle a chunk detected as silence (state lock held).

        Args:
            audio_chunk: Silent chunk (already noise reduced).
            outcome: Notification collector for deferred dispatch.
        """
        # Always keep recent audio in the pre-speech ring buffer
        if audio_chunk is not None and len(audio_chunk) > 0:
            self._pre_speech_buffer.write(audio_chunk)

        if self._state is not StreamState.SPEAKING:
            return

        self._silence_counter += 1

        # Still accumulate during silence (keeps the trailing audio)
        self._accumulate(audio_chunk)

        if self._silence_counter >= self.config.silence_chunks:
            self._end_speech(outcome)

    def _end_speech(self, outcome: "_FeedOutcome") -> None:
        """
        Close the current speech segment (state lock held).

        Extracts the segment from the ring buffer in a single copy, applies the
        minimum-duration gate and queues the end-of-speech notifications. The
        heavy transcription work is *not* done here (see
        :meth:`_finish_segment`).

        Args:
            outcome: Notification collector; receives ``segment`` when valid.
        """
        self._silence_counter = 0

        if self._total_speech_samples <= 0:
            self._set_state(StreamState.IDLE, outcome)
            return

        full_speech = self._audio_buffer.read(self._total_speech_samples)
        self._audio_buffer.clear()
        self._total_speech_samples = 0
        speech_duration = len(full_speech) / self.config.sample_rate

        # Check minimum speech duration
        min_duration = self.config.vad_min_speech_ms / 1000.0
        if speech_duration < min_duration:
            logger.debug(
                "Speech too short (%.2fs < %.2fs), discarding",
                speech_duration, min_duration,
            )
            self._set_state(StreamState.IDLE, outcome)
            self._set_duplex_state(
                DuplexState.SPEAKING if self._playback_active
                else DuplexState.LISTENING,
                outcome,
            )
            return

        self._stats["speech_segments"] += 1
        self._stats["total_speech_duration_s"] += speech_duration
        self._set_state(StreamState.PROCESSING, outcome)
        self._set_duplex_state(DuplexState.THINKING, outcome)

        logger.info(
            "Speech segment: %.2fs (%d samples)",
            speech_duration, len(full_speech),
        )

        if self.on_speech_end is not None:
            outcome.callbacks.append(
                lambda audio=full_speech: self._safe_call(self.on_speech_end, audio)
            )
        outcome.events.append(
            StreamEvent(event_type="speech_end", audio=full_speech)
        )
        outcome.segment = full_speech

    def _finish_segment(self, audio: np.ndarray) -> None:
        """
        Start transcription for a finished segment (called without the lock).

        Args:
            audio: The completed speech segment.
        """
        if self.whisper_bridge is not None:
            self._transcribe_async(audio)
        else:
            self._reset_speech_state()

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

            if self.on_transcription is not None:
                self._safe_call(self.on_transcription, result)

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
        """
        Reset speech accumulation state. Thread-safe.

        Clears the segment ring buffer (keeping its pre-allocated memory) and
        returns the processor to ``IDLE`` -- or straight back to
        ``DuplexState.SPEAKING`` when TTS playback is still active.
        """
        with self._lock:
            self._audio_buffer.clear()
            self._total_speech_samples = 0
            self._silence_counter = 0
            self._interrupt_frames = 0
        self._set_state(StreamState.IDLE)
        self._set_duplex_state(
            DuplexState.SPEAKING if self._playback_active else DuplexState.LISTENING
        )

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the processor to its initial state (keeps playback armed state)."""
        with self._lock:
            self._audio_buffer.clear()
            self._pre_speech_buffer.clear()
            self._playback_reference_buffer.clear()
            self._total_speech_samples = 0
            self._silence_counter = 0
            self._interrupt_frames = 0
            self._stats = {
                "chunks_processed": 0,
                "speech_segments": 0,
                "total_speech_duration_s": 0.0,
                "total_processing_time_s": 0.0,
                "avg_latency_ms": 0.0,
                "interrupts": 0,
                "last_interrupt_latency_ms": 0.0,
                "echo_vetoes": 0,
            }
        self._set_state(StreamState.IDLE)
        self._set_duplex_state(DuplexState.IDLE)
        if self._upload_channel is not None:
            self._upload_channel.synchronize()
        logger.info("Stream processor reset")

    def flush(self) -> None:
        """
        Drop all buffered input audio without touching statistics.

        Useful after a barge-in or after the playback pipeline has muted, to
        guarantee that echoed TTS audio never reaches the recogniser.
        """
        with self._lock:
            self._audio_buffer.clear()
            self._pre_speech_buffer.clear()
            self._playback_reference_buffer.clear()
            self._total_speech_samples = 0
            self._silence_counter = 0
            self._interrupt_frames = 0
        logger.debug("Input buffers flushed")

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