"""
Unit Tests for Acoustic Echo Veto of Barge-In Detection (v2.6.1)
================================================================

Covers the v2.6.1 full-duplex anti-self-interruption upgrade of
``vram_core/stream_processor.py``:

    1. ``StreamProcessor.register_playback_chunk()`` -- the playback reference
       ring buffer (what the speaker is emitting right now)
    2. ``_echo_ncc_peak()`` -- time-domain normalised cross-correlation (NCC)
       against that reference over a 0..echo_ncc_max_lag_ms delay search range
    3. Echo veto: a loud but *correlated* TTS passage never raises ``on_interrupt``
    4. Genuine (uncorrelated) user speech still interrupts immediately
    5. Backwards compatibility: with no reference the legacy energy-only detector
       behaves exactly as before
    6. Budget: the veto scan stays far below the audio chunk budget
"""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vram_core.stream_processor import StreamConfig, StreamProcessor

CHUNK = 1600   # 100 ms @ 16 kHz
DELAY = 1200   # 75 ms acoustic propagation delay (inside the 400 ms range)
PLAYBACK_AMPLITUDE = 0.5


def make_config(**overrides) -> StreamConfig:
    """Barge-in config with the echo damping neutralised (NCC does the work)."""
    params = dict(
        interrupt_energy_thresh=0.06,
        interrupt_min_frames=2,
        echo_suppression_factor=1.0,
        echo_suppression_decay_ms=0,
    )
    params.update(overrides)
    return StreamConfig(**params)


def make_playback(seed: int = 0, chunks: int = 5) -> np.ndarray:
    """Deterministic 'TTS' signal: shaped noise, 100 ms per chunk."""
    rng = np.random.default_rng(seed)
    return (PLAYBACK_AMPLITUDE * rng.standard_normal(CHUNK * chunks)).astype(np.float32)


def make_voice(f0: float = 173.0) -> np.ndarray:
    """Deterministic voiced signal uncorrelated with the noise playback."""
    t = np.arange(CHUNK) / 16000.0
    signal = PLAYBACK_AMPLITUDE * np.sin(2 * np.pi * f0 * t)
    signal = signal + 0.2 * np.sin(2 * np.pi * (f0 * 2.3) * t)
    return signal.astype(np.float32)


def delayed_chunks(playback: np.ndarray, count: int = 1, delay: int = DELAY):
    """
    Microphone captures of ``playback``, each lagging the speaker by ``delay``.

    Windows are taken backwards from the end of the registered playback so every
    capture is a full ``CHUNK``-long copy of audio that is really in the
    reference buffer.
    """
    captures = []
    for index in range(count):
        end = playback.size - delay - index * CHUNK
        if end - CHUNK < 0:
            raise ValueError("playback stream is too short for the requested captures")
        captures.append(playback[end - CHUNK:end].copy())
    return captures


def register_playback(processor: StreamProcessor, playback: np.ndarray) -> None:
    """Stream ``playback`` into the processor chunk by chunk (TTS thread)."""
    for start in range(0, playback.size, CHUNK):
        processor.register_playback_chunk(playback[start:start + CHUNK])


class TestPlaybackReference:
    """``register_playback_chunk`` maintains the speaker reference buffer."""

    def test_chunks_are_accumulated_in_the_ring_buffer(self):
        """Every emitted chunk lands in the playback reference buffer."""
        processor = StreamProcessor()
        assert processor.playback_reference_buffer.size == 0

        written = processor.register_playback_chunk(np.ones(CHUNK, dtype=np.float32))
        assert written == CHUNK
        assert processor.playback_reference_buffer.size == CHUNK

    def test_none_and_empty_chunks_are_ignored(self):
        """Degenerate input never raises and never writes."""
        processor = StreamProcessor()
        assert processor.register_playback_chunk(None) == 0
        assert processor.register_playback_chunk(np.array([], dtype=np.float32)) == 0
        assert processor.playback_reference_buffer.size == 0

    def test_reference_buffer_covers_two_seconds(self):
        """The buffer capacity follows ``playback_reference_duration_s``."""
        processor = StreamProcessor(config=StreamConfig(playback_reference_duration_s=2.0))
        capacity = processor.playback_reference_buffer.capacity
        assert capacity == 2 * 16000

        processor.register_playback_chunk(np.ones(10 * 16000, dtype=np.float32))
        assert processor.playback_reference_buffer.size == capacity  # oldest evicted

    def test_starting_playback_invalidates_the_old_reference(self):
        """A new TTS turn must not correlate against the previous one."""
        processor = StreamProcessor()
        register_playback(processor, make_playback())
        assert processor.playback_reference_buffer.size > 0

        processor.set_playback_state(True)
        assert processor.playback_reference_buffer.size == 0

    def test_reset_and_flush_clear_the_reference(self):
        """``reset()`` / ``flush()`` drop stale speaker audio."""
        processor = StreamProcessor()
        register_playback(processor, make_playback())
        processor.flush()
        assert processor.playback_reference_buffer.size == 0

        register_playback(processor, make_playback())
        processor.reset()
        assert processor.playback_reference_buffer.size == 0

    def test_explicit_clear_helper(self):
        """``clear_playback_reference()`` is the public flush of the reference."""
        processor = StreamProcessor()
        register_playback(processor, make_playback())
        processor.clear_playback_reference()
        assert processor.playback_reference_buffer.size == 0


class TestEchoNccPeak:
    """Normalised cross-correlation between microphone and playback."""

    def test_delayed_copy_correlates_almost_perfectly(self):
        """A mic signal that is a delayed copy of the playback peaks near 1.0."""
        processor = StreamProcessor()
        playback = make_playback()
        register_playback(processor, playback)

        peak = processor._echo_ncc_peak(delayed_chunks(playback)[0])
        assert peak > 0.9, peak

    def test_uncorrelated_speech_stays_below_the_threshold(self):
        """Unrelated voiced audio does not correlate with the noise playback."""
        processor = StreamProcessor()
        register_playback(processor, make_playback())
        assert processor._echo_ncc_peak(make_voice()) < 0.55

    def test_no_reference_returns_zero(self):
        """Without registered playback there is nothing to correlate against."""
        processor = StreamProcessor()
        assert processor._echo_ncc_peak(make_voice()) == 0.0

    def test_short_chunks_are_rejected(self):
        """Chunks below the minimum correlation length are ignored."""
        processor = StreamProcessor()
        register_playback(processor, make_playback())
        assert processor._echo_ncc_peak(np.ones(16, dtype=np.float32)) == 0.0

    def test_silent_microphone_chunk_returns_zero(self):
        """A silent chunk has no energy to correlate (guarded division)."""
        processor = StreamProcessor()
        register_playback(processor, make_playback())
        assert processor._echo_ncc_peak(np.zeros(CHUNK, dtype=np.float32)) == 0.0

    def test_reference_shorter_than_the_chunk_returns_zero(self):
        """A partially filled reference cannot be correlated yet."""
        processor = StreamProcessor()
        processor.register_playback_chunk(np.ones(200, dtype=np.float32))
        assert processor._echo_ncc_peak(np.ones(CHUNK, dtype=np.float32)) == 0.0


class TestEchoVetoDecision:
    """The veto itself: echo is suppressed, real speech is not."""

    def test_playback_echo_never_interrupts(self):
        """A loud correlated passage is vetoed instead of interrupting."""
        processor = StreamProcessor(config=make_config())
        interrupts = []
        processor.on_interrupt = interrupts.append
        playback = make_playback()
        processor.set_playback_state(True)
        register_playback(processor, playback)

        processor.feed(delayed_chunks(playback)[0])

        assert interrupts == []
        assert processor.stats["interrupts"] == 0
        assert processor.stats["echo_vetoes"] == 1
        # A veto is not an interruption: playback keeps running
        assert processor.is_playback_active is True

    def test_veto_resets_the_consecutive_frame_run(self):
        """An echo must not arm the frame counter for the next chunk."""
        processor = StreamProcessor(config=make_config())
        playback = make_playback()
        processor.set_playback_state(True)
        register_playback(processor, playback)

        processor.feed(delayed_chunks(playback)[0])
        assert processor._interrupt_frames == 0

    def test_repeated_echo_chunks_are_all_vetoed(self):
        """Every captured chunk of the same passage is recognised as echo."""
        processor = StreamProcessor(config=make_config())
        interrupts = []
        processor.on_interrupt = interrupts.append
        playback = make_playback()
        processor.set_playback_state(True)
        register_playback(processor, playback)

        for capture in delayed_chunks(playback, count=3):
            processor.feed(capture)

        assert interrupts == []
        assert processor.stats["echo_vetoes"] == 3

    def test_uncorrelated_user_speech_still_interrupts(self):
        """Real speech is unaffected by the echo veto."""
        processor = StreamProcessor(config=make_config())
        interrupts = []
        processor.on_interrupt = interrupts.append
        processor.set_playback_state(True)
        register_playback(processor, make_playback())

        processor.feed(make_voice())

        assert len(interrupts) == 1
        assert processor.stats["interrupts"] == 1
        assert processor.stats["echo_vetoes"] == 0
        assert processor.is_playback_active is False

    def test_threshold_can_disable_the_veto(self):
        """``echo_ncc_threshold > 1`` restores pure energy-based interruption."""
        processor = StreamProcessor(config=make_config(echo_ncc_threshold=1.01))
        interrupts = []
        processor.on_interrupt = interrupts.append
        playback = make_playback()
        processor.set_playback_state(True)
        register_playback(processor, playback)

        processor.feed(delayed_chunks(playback)[0])

        assert len(interrupts) == 1
        assert processor.stats["echo_vetoes"] == 0

    def test_veto_requires_playback_to_be_active(self):
        """Without playback the detector is disarmed, so nothing is vetoed."""
        processor = StreamProcessor(config=make_config())
        playback = make_playback()
        register_playback(processor, playback)
        processor.feed(delayed_chunks(playback)[0])
        assert processor.stats["echo_vetoes"] == 0

    def test_statistics_are_reset_with_the_processor(self):
        """``reset()`` clears the echo-veto counter along with the rest."""
        processor = StreamProcessor(config=make_config())
        playback = make_playback()
        processor.set_playback_state(True)
        register_playback(processor, playback)
        processor.feed(delayed_chunks(playback)[0])
        assert processor.stats["echo_vetoes"] == 1

        processor.reset()
        assert processor.stats["echo_vetoes"] == 0


class TestEchoVetoBudget:
    """The veto scan must fit comfortably inside the audio chunk budget."""

    def test_ncc_scan_is_fast_enough_for_realtime(self):
        """A 2 s reference with a 400 ms lag range costs a few milliseconds."""
        processor = StreamProcessor()
        register_playback(processor, make_playback(chunks=20))  # 2 s of playback
        chunk = make_voice()

        processor._echo_ncc_peak(chunk)  # warm-up
        started = time.perf_counter()
        for _ in range(20):
            processor._echo_ncc_peak(chunk)
        average_ms = (time.perf_counter() - started) / 20 * 1000

        assert average_ms < 50.0, average_ms

    def test_feed_with_veto_stays_within_the_chunk_budget(self):
        """The whole vetoed feed call stays well below the 100 ms chunk."""
        processor = StreamProcessor(config=make_config())
        processor.set_playback_state(True)
        register_playback(processor, make_playback())
        capture = delayed_chunks(make_playback())[0]

        processor.feed(capture)  # warm-up
        started = time.perf_counter()
        for _ in range(5):
            processor.feed(capture)
        average_ms = (time.perf_counter() - started) / 5 * 1000

        assert average_ms < 100.0, average_ms
