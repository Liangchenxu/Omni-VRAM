"""
Unit Tests for Full-Duplex Barge-In & Ring-Buffer Audio Memory (v2.6.0)
=======================================================================

Covers:
    1. DuplexState machine (IDLE / LISTENING / THINKING / SPEAKING)
    2. Barge-in detection: sub-millisecond dispatch, consecutive frame gating,
       dynamic threshold weighting and speaker-echo suppression
    3. Pre-allocated circular audio memory (fixed footprint, slice accuracy)
    4. Thread-safety: callbacks may re-enter the processor without deadlocking
"""

import os
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vram_core.stream_processor import (
    BargeInEvent,
    CircularBuffer,
    DuplexState,
    StreamConfig,
    StreamProcessor,
    StreamState,
    VADProcessor,
)

CHUNK = 1600  # 100 ms @ 16 kHz


def speech_chunk(amplitude: float = 0.5) -> np.ndarray:
    """Return a deterministic loud chunk (DC level -> RMS == amplitude)."""
    return np.full(CHUNK, amplitude, dtype=np.float32)


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    """Poll ``predicate`` until it is true or the timeout elapses."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class BlockingBridge:
    """Whisper bridge stub that blocks until ``release`` is set."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def transcribe(self, audio, sample_rate: int = 16000):
        """Signal that transcription started, then wait for the test."""
        from vram_core.whisper import WhisperResult

        self.started.set()
        self.release.wait(timeout=5.0)
        return WhisperResult(text="hello", language="en", confidence=0.9)


class TestDuplexStateMachine:
    """IDLE / LISTENING / THINKING / SPEAKING transitions."""

    def test_initial_state_is_idle(self):
        """A fresh processor is not bound to a conversation."""
        processor = StreamProcessor()
        assert processor.duplex_state is DuplexState.IDLE
        assert processor.is_playback_active is False

    def test_playback_transitions(self):
        """set_playback_state drives SPEAKING <-> LISTENING."""
        processor = StreamProcessor()
        states = []
        processor.on_duplex_state_change = states.append

        processor.set_playback_state(True)
        assert processor.duplex_state is DuplexState.SPEAKING
        assert processor.is_playback_active is True

        processor.set_playback_state(False)
        assert processor.duplex_state is DuplexState.LISTENING
        assert processor.is_playback_active is False

        assert states == [DuplexState.SPEAKING, DuplexState.LISTENING]

    def test_listening_on_first_audio(self):
        """Feeding audio while IDLE moves the machine to LISTENING."""
        processor = StreamProcessor()
        processor.feed(np.zeros(CHUNK, dtype=np.float32))
        assert processor.duplex_state is DuplexState.LISTENING

    def test_thinking_while_transcribing(self):
        """End of speech puts the machine in THINKING while ASR is running."""
        config = StreamConfig(
            chunk_duration_ms=100,
            vad_silence_duration_ms=200,
            vad_min_speech_ms=100,
        )
        bridge = BlockingBridge()
        processor = StreamProcessor(config=config, whisper_bridge=bridge)

        for _ in range(3):
            processor.feed(speech_chunk())
        assert processor.duplex_state is DuplexState.LISTENING

        for _ in range(2):
            processor.feed(np.zeros(CHUNK, dtype=np.float32))

        assert bridge.started.wait(timeout=5.0), "transcription never started"
        assert processor.state is StreamState.PROCESSING
        assert processor.duplex_state is DuplexState.THINKING

        bridge.release.set()
        assert _wait_for(lambda: processor.duplex_state is DuplexState.LISTENING)


class TestBargeInDetection:
    """Barge-in detection, dynamic threshold weighting and interrupt handling."""

    def test_barge_in_fires_with_sub_millisecond_signal(self):
        """A loud chunk during playback raises on_interrupt immediately."""
        config = StreamConfig(
            interrupt_energy_thresh=0.06,
            interrupt_min_frames=2,
            echo_suppression_factor=1.0,
            echo_suppression_decay_ms=0,
        )
        processor = StreamProcessor(config=config)
        events = []
        processor.on_interrupt = events.append

        processor.set_playback_state(True)
        started = time.perf_counter()
        processor.feed(speech_chunk(0.5))
        elapsed_ms = (time.perf_counter() - started) * 1000

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, BargeInEvent)
        assert event.energy > event.threshold > 0.0
        assert event.frames >= 2
        # The truncation signal itself must be sub-millisecond
        assert event.detection_latency_ms < 1.0, event.detection_latency_ms
        # ... and the whole feed call must stay far below the audio budget
        assert elapsed_ms < 100.0, elapsed_ms

    def test_barge_in_requires_consecutive_frames(self):
        """One or two loud frames are not enough; K consecutive frames are."""
        config = StreamConfig(
            interrupt_energy_thresh=0.06,
            interrupt_min_frames=3,
            echo_suppression_factor=1.0,
            echo_suppression_decay_ms=0,
            vad_frame_ms=25,  # 400 samples per frame
        )
        processor = StreamProcessor(config=config)
        events = []
        processor.on_interrupt = events.append
        processor.set_playback_state(True)

        chunk = np.zeros(CHUNK, dtype=np.float32)
        chunk[:800] = 0.5          # 2 loud frames, then quiet resets the run
        processor.feed(chunk)
        assert events == []

        chunk = np.zeros(CHUNK, dtype=np.float32)
        chunk[:1200] = 0.5         # 3 consecutive loud frames
        processor.feed(chunk)
        assert len(events) == 1
        assert events[0].frames >= 3

    def test_consecutive_frames_span_chunk_boundaries(self):
        """Frame counting continues across chunks (no detection blind spot)."""
        config = StreamConfig(
            interrupt_energy_thresh=0.06,
            interrupt_min_frames=3,
            echo_suppression_factor=1.0,
            echo_suppression_decay_ms=0,
            vad_frame_ms=25,  # 400 samples per frame
        )
        processor = StreamProcessor(config=config)
        events = []
        processor.on_interrupt = events.append
        processor.set_playback_state(True)

        chunk = np.zeros(CHUNK, dtype=np.float32)
        chunk[800:] = 0.5          # chunk ends with 2 loud frames -> not enough
        processor.feed(chunk)
        assert events == []

        chunk = np.zeros(CHUNK, dtype=np.float32)
        chunk[:400] = 0.5          # 1 more frame -> 3 consecutive in total
        processor.feed(chunk)
        assert len(events) == 1
        assert events[0].frames >= 3

    def test_speaker_echo_does_not_trigger(self):
        """Floating threshold weighting suppresses TTS echo right after start."""
        config = StreamConfig(
            interrupt_energy_thresh=0.06,
            interrupt_min_frames=2,
            barge_in_sensitivity=1.0,
            echo_suppression_factor=2.0,
            echo_suppression_decay_ms=500,
        )
        processor = StreamProcessor(config=config)
        events = []
        processor.on_interrupt = events.append
        processor.set_playback_state(True)

        # RMS 0.09 > base threshold (0.06) but < damped threshold (0.12)
        processor.feed(speech_chunk(0.09))
        assert events == []

        # Real user speech easily clears the damped threshold
        processor.feed(speech_chunk(0.4))
        assert len(events) == 1
        assert 0.11 < events[0].threshold < 0.13

    def test_sensitivity_multiplier(self):
        """barge_in_sensitivity > 1 lowers the effective threshold."""
        config = StreamConfig(
            interrupt_energy_thresh=0.06,
            barge_in_sensitivity=2.0,
            echo_suppression_factor=1.0,
            echo_suppression_decay_ms=0,
        )
        processor = StreamProcessor(config=config)
        processor.set_playback_state(True)
        threshold = processor._effective_interrupt_threshold(time.perf_counter())
        assert threshold == pytest.approx(0.03, rel=1e-3)

        # Lower sensitivity raises the threshold instead
        processor.config.barge_in_sensitivity = 0.5
        threshold = processor._effective_interrupt_threshold(time.perf_counter())
        assert threshold == pytest.approx(0.12, rel=1e-3)

    def test_barge_in_disabled(self):
        """enable_barge_in=False keeps the detector silent."""
        config = StreamConfig(enable_barge_in=False, interrupt_min_frames=1)
        processor = StreamProcessor(config=config)
        events = []
        processor.on_interrupt = events.append
        processor.set_playback_state(True)
        processor.feed(speech_chunk(0.5))
        assert events == []

    def test_barge_in_only_while_speaking(self):
        """Without playback the detector stays disarmed."""
        processor = StreamProcessor()
        events = []
        processor.on_interrupt = events.append
        processor.feed(speech_chunk(0.5))
        assert events == []
        assert processor.stats["interrupts"] == 0

    def test_interrupt_truncates_and_restarts(self):
        """Barge-in drops the interrupted turn and starts a new one."""
        processor = StreamProcessor()
        interrupts = []
        processor.on_interrupt = interrupts.append
        events = []
        processor.on_event = lambda e: events.append(e.event_type)
        processor.set_playback_state(True)
        processor.feed(speech_chunk(0.5))

        assert len(interrupts) == 1
        assert processor.is_playback_active is False
        assert processor.duplex_state is DuplexState.LISTENING
        assert processor.stats["interrupts"] == 1
        assert processor.stats["last_interrupt_latency_ms"] < 1.0
        # The very chunk that interrupted becomes the start of the next turn
        assert processor.state is StreamState.SPEAKING
        assert "interrupt" in events

    def test_interrupt_callback_errors_are_contained(self):
        """A raising callback must not break the audio thread."""
        processor = StreamProcessor()

        def boom(_event):
            raise RuntimeError("callback failure")

        processor.on_interrupt = boom
        processor.set_playback_state(True)
        processor.feed(speech_chunk(0.5))
        assert processor.stats["interrupts"] == 1
        assert processor.state is StreamState.SPEAKING

    def test_resumes_speaking_when_playback_active(self):
        """After transcription the machine resumes SPEAKING if TTS is running."""
        config = StreamConfig(
            chunk_duration_ms=100,
            vad_silence_duration_ms=200,
            vad_min_speech_ms=100,
        )
        bridge = BlockingBridge()
        processor = StreamProcessor(config=config, whisper_bridge=bridge)

        for _ in range(3):
            processor.feed(speech_chunk())
        processor.set_playback_state(True)
        for _ in range(2):
            processor.feed(np.zeros(CHUNK, dtype=np.float32))

        assert bridge.started.wait(timeout=5.0), "transcription never started"
        assert processor.duplex_state is DuplexState.THINKING

        bridge.release.set()
        assert _wait_for(lambda: processor.duplex_state is DuplexState.SPEAKING)


class TestCircularAudioMemory:
    """Pre-allocated, contiguous ring buffer for streaming audio."""

    def test_memory_is_preallocated_and_constant(self):
        """2000 writes must not grow the buffer's memory footprint."""
        buffer = CircularBuffer(16000)
        initial_bytes = buffer.memory_bytes
        initial_capacity = buffer.capacity

        for _ in range(2000):
            buffer.write(np.random.randn(CHUNK).astype(np.float32))

        assert buffer.capacity == initial_capacity
        assert buffer.memory_bytes == initial_bytes
        assert buffer.memory_bytes == initial_capacity * 4  # float32
        assert buffer.size == initial_capacity
        assert buffer.total_written == 2000 * CHUNK

    def test_sliding_window_extraction(self):
        """peek/extract_window return the newest samples in order."""
        buffer = CircularBuffer(10)
        for value in range(25):
            buffer.write(np.array([value], dtype=np.float32))

        np.testing.assert_array_equal(buffer.extract_window(5), [20, 21, 22, 23, 24])
        np.testing.assert_array_equal(buffer.peek(3), [22, 23, 24])
        assert buffer.size == 10
        np.testing.assert_array_equal(buffer.read_all(), list(range(15, 25)))
        assert buffer.size == 0

    def test_wrapped_write_and_read_consistency(self):
        """Wrap-around never reorders or corrupts samples."""
        buffer = CircularBuffer(7)
        values = np.arange(100, dtype=np.float32)
        for value in values:
            buffer.write(np.array([value], dtype=np.float32))
            if buffer.size > 3:
                buffer.read(2)
        expected = list(values[-buffer.size:])
        np.testing.assert_array_equal(buffer.read_all(), expected)

    def test_large_write_larger_than_capacity(self):
        """A write bigger than the capacity keeps only the newest samples."""
        buffer = CircularBuffer(4)
        buffer.write(np.arange(10, dtype=np.float32))
        assert buffer.size == 4
        np.testing.assert_array_equal(buffer.read_all(), [6, 7, 8, 9])

    def test_invalid_capacity(self):
        """A non-positive capacity is rejected."""
        with pytest.raises(ValueError):
            CircularBuffer(0)

    def test_processor_footprint_is_stable(self):
        """Long streams never allocate new audio memory inside the processor."""
        processor = StreamProcessor()
        footprint = processor.memory_footprint_bytes
        assert footprint > 0

        speech = np.random.randn(CHUNK).astype(np.float32) * 0.5
        silence = np.zeros(CHUNK, dtype=np.float32)
        for index in range(500):
            processor.feed(speech if index % 5 else silence)

        assert processor.memory_footprint_bytes == footprint
        assert processor.stats["chunks_processed"] == 500

    def test_slice_accuracy_of_accumulated_segment(self):
        """The ring buffer returns the exact chronological segment audio."""
        processor = StreamProcessor(
            config=StreamConfig(chunk_duration_ms=100, vad_silence_duration_ms=300)
        )
        processor.noise_reducer = None  # keep levels untouched
        received = []
        processor.on_speech_end = received.append

        levels = (0.1, 0.2, 0.3, 0.4, 0.5)
        for level in levels:
            processor.feed(np.full(CHUNK, level, dtype=np.float32))
        for _ in range(3):
            processor.feed(np.zeros(CHUNK, dtype=np.float32))

        assert len(received) == 1
        segment = received[0]
        assert len(segment) == 8 * CHUNK
        rows = segment.reshape(-1, CHUNK).mean(axis=1)
        np.testing.assert_allclose(rows[:5], levels, atol=1e-6)
        np.testing.assert_allclose(rows[5:], 0.0, atol=1e-6)

    def test_vad_frame_energies(self):
        """Frame energies are computed without a Python sample loop."""
        vad = VADProcessor(frame_size_ms=25)  # 400 samples
        chunk = np.zeros(CHUNK, dtype=np.float32)
        chunk[:400] = 0.5
        energies = vad.frame_energies(chunk)
        assert len(energies) == 4
        assert energies[0] == pytest.approx(0.5)
        assert energies[1] == pytest.approx(0.0)

    def test_vad_short_chunk_fallback(self):
        """Chunks shorter than one frame fall back to a single energy."""
        vad = VADProcessor(frame_size_ms=25)
        energies = vad.frame_energies(np.full(100, 0.25, dtype=np.float32))
        assert len(energies) == 1
        assert energies[0] == pytest.approx(0.25)

    def test_speech_confidence_mapping(self):
        """Confidence is 0 for silence and approaches 1 for loud audio."""
        vad = VADProcessor(threshold=0.02)
        assert vad.speech_confidence(np.zeros(CHUNK, dtype=np.float32)) == 0.0
        assert vad.speech_confidence(np.full(CHUNK, 0.02, dtype=np.float32)) == pytest.approx(0.5)
        assert vad.speech_confidence(np.full(CHUNK, 0.5, dtype=np.float32)) > 0.9


class TestDuplexThreadSafety:
    """Callbacks may re-enter the processor; the audio thread never deadlocks."""

    def test_feed_does_not_deadlock_with_reentrant_callbacks(self):
        """State/start callbacks that call back into the processor are safe."""
        processor = StreamProcessor()
        observed = []

        def on_state(state):
            processor.flush()          # re-enters the processor
            observed.append(state)

        processor.on_state_change = on_state
        processor.on_speech_start = lambda: processor.update_threshold(0.03)

        def worker():
            processor.feed(speech_chunk())

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=5.0)

        assert not thread.is_alive(), "feed() deadlocked (re-entrant lock)"
        assert StreamState.SPEAKING in observed

    def test_flush_and_reset(self):
        """flush() keeps statistics, reset() clears them."""
        processor = StreamProcessor()
        processor.feed(np.zeros(CHUNK, dtype=np.float32))
        assert processor.get_buffered_audio().size > 0

        processor.flush()
        assert processor.get_buffered_audio().size == 0
        assert processor.stats["chunks_processed"] == 1

        processor.reset()
        assert processor.stats["chunks_processed"] == 0
        assert processor.duplex_state is DuplexState.IDLE

    def test_barge_in_clears_pre_speech_context(self):
        """Echo/TTS tail must not leak into the turn that interrupted it."""
        processor = StreamProcessor()
        processor.set_playback_state(True)
        # Simulate TTS tail captured by the microphone (quiet -> pre-speech buffer)
        processor.feed(np.zeros(CHUNK, dtype=np.float32))
        assert processor.get_buffered_audio().size > 0

        processor.on_interrupt = lambda event: None
        processor.feed(speech_chunk(0.6))
        # Barge-in dropped the echo context before the new turn started
        assert processor.state is StreamState.SPEAKING
        assert processor.stats["interrupts"] == 1
