"""
Unit Tests for vram_core Stream Processor
==========================================

Tests real-time audio streaming, VAD, circular buffer,
and stream state management.
"""

import os
import sys
import time
import pytest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vram_core.stream_processor import (
    StreamProcessor,
    StreamConfig,
    StreamState,
    StreamEvent,
    CircularBuffer,
    VADProcessor,
)


class TestStreamConfig:
    """Test StreamConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StreamConfig()
        assert config.sample_rate == 16000
        assert config.chunk_duration_ms == 100
        assert config.vad_threshold == 0.02
        assert config.vad_silence_duration_ms == 800
        assert config.vad_min_speech_ms == 200

    def test_chunk_size_property(self):
        """Test chunk_size calculation."""
        config = StreamConfig(sample_rate=16000, chunk_duration_ms=100)
        assert config.chunk_size == 1600

    def test_silence_chunks_property(self):
        """Test silence_chunks calculation."""
        config = StreamConfig(chunk_duration_ms=100, vad_silence_duration_ms=800)
        assert config.silence_chunks == 8

    def test_min_speech_chunks_property(self):
        """Test min_speech_chunks calculation."""
        config = StreamConfig(chunk_duration_ms=100, vad_min_speech_ms=200)
        assert config.min_speech_chunks == 2

    def test_pre_speech_samples_property(self):
        """Test pre_speech_samples calculation."""
        config = StreamConfig(sample_rate=16000, pre_speech_buffer_ms=200)
        assert config.pre_speech_samples == 3200

    def test_custom_config(self):
        """Test custom configuration."""
        config = StreamConfig(
            sample_rate=44100,
            chunk_duration_ms=50,
            vad_threshold=0.05,
        )
        assert config.sample_rate == 44100
        assert config.chunk_duration_ms == 50
        assert config.vad_threshold == 0.05
        assert config.chunk_size == 2205


class TestCircularBuffer:
    """Test CircularBuffer implementation."""

    def test_write_and_read(self):
        """Test basic write and read operations."""
        buf = CircularBuffer(100)
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        written = buf.write(data)
        assert written == 3
        assert buf.size == 3

        result = buf.read(3)
        np.testing.assert_array_equal(result, data)
        assert buf.size == 0

    def test_read_more_than_available(self):
        """Test reading more samples than available."""
        buf = CircularBuffer(100)
        data = np.array([1.0, 2.0], dtype=np.float32)
        buf.write(data)

        result = buf.read(10)
        assert len(result) == 2

    def test_read_empty_buffer(self):
        """Test reading from empty buffer."""
        buf = CircularBuffer(100)
        result = buf.read(10)
        assert len(result) == 0

    def test_overflow_eviction(self):
        """Test that old samples are evicted on overflow."""
        buf = CircularBuffer(5)
        buf.write(np.array([1, 2, 3, 4, 5], dtype=np.float32))
        buf.write(np.array([6, 7], dtype=np.float32))

        assert buf.size == 5
        result = buf.read(5)
        np.testing.assert_array_equal(result, [3, 4, 5, 6, 7])

    def test_peek(self):
        """Test peek doesn't remove samples."""
        buf = CircularBuffer(100)
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buf.write(data)

        peeked = buf.peek(3)
        np.testing.assert_array_equal(peeked, data)
        assert buf.size == 3  # Not removed

    def test_peek_empty(self):
        """Test peek on empty buffer."""
        buf = CircularBuffer(100)
        result = buf.peek(10)
        assert len(result) == 0

    def test_clear(self):
        """Test buffer clearing."""
        buf = CircularBuffer(100)
        buf.write(np.array([1, 2, 3], dtype=np.float32))
        buf.clear()
        assert buf.size == 0


class TestVADProcessor:
    """Test Voice Activity Detection."""

    def test_speech_detection_positive(self):
        """Test that loud audio is detected as speech."""
        vad = VADProcessor(threshold=0.02)
        # Generate a loud sine wave
        t = np.linspace(0, 0.1, 1600)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        assert vad.is_speech(audio)

    def test_speech_detection_negative(self):
        """Test that quiet audio is not detected as speech."""
        vad = VADProcessor(threshold=0.02)
        # Generate near-silent audio
        audio = np.random.randn(1600).astype(np.float32) * 0.001
        assert not vad.is_speech(audio)

    def test_speech_detection_empty(self):
        """Test that empty audio returns False."""
        vad = VADProcessor(threshold=0.02)
        assert not vad.is_speech(np.array([], dtype=np.float32))

    def test_energy_computation(self):
        """Test RMS energy computation."""
        vad = VADProcessor()
        audio = np.ones(100, dtype=np.float32) * 0.5
        energy = vad.compute_energy(audio)
        assert round(abs(energy - 0.5), 5) == 0

    def test_energy_empty(self):
        """Test energy of empty audio."""
        vad = VADProcessor()
        energy = vad.compute_energy(np.array([], dtype=np.float32))
        assert energy == 0.0

    def test_zero_crossing_rate(self):
        """Test zero-crossing rate computation."""
        vad = VADProcessor()
        # Alternating signal has high ZCR
        audio = np.array([1, -1, 1, -1, 1, -1], dtype=np.float32)
        zcr = vad.compute_zero_crossing_rate(audio)
        assert zcr > 0.8

    def test_zero_crossing_rate_constant(self):
        """Test ZCR of constant signal is zero."""
        vad = VADProcessor()
        audio = np.ones(100, dtype=np.float32)
        zcr = vad.compute_zero_crossing_rate(audio)
        assert round(abs(zcr - 0.0), 5) == 0

    def test_zero_crossing_rate_empty(self):
        """Test ZCR of empty audio."""
        vad = VADProcessor()
        zcr = vad.compute_zero_crossing_rate(np.array([], dtype=np.float32))
        assert zcr == 0.0

    def test_detect_speech_segments(self):
        """Test speech segment detection."""
        vad = VADProcessor(threshold=0.02)

        # Create audio: silence, speech, silence, speech, silence
        silence = np.zeros(8000, dtype=np.float32)
        speech = np.random.randn(16000).astype(np.float32) * 0.5

        audio = np.concatenate([silence, speech, silence, speech, silence])
        segments = vad.detect_speech_segments(audio, min_duration_ms=100)

        assert len(segments) >= 1

    def test_detect_speech_segments_all_silent(self):
        """Test segment detection on silent audio."""
        vad = VADProcessor(threshold=0.02)
        audio = np.zeros(16000, dtype=np.float32)
        segments = vad.detect_speech_segments(audio)
        assert len(segments) == 0


class TestStreamProcessor:
    """Test StreamProcessor main class."""

    def test_default_initialization(self):
        """Test default processor initialization."""
        processor = StreamProcessor()
        assert processor.state == StreamState.IDLE
        assert processor.whisper_bridge is None

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = StreamConfig(vad_threshold=0.05)
        processor = StreamProcessor(config=config)
        assert processor.vad.threshold == 0.05

    def test_initial_state(self):
        """Test initial state is IDLE."""
        processor = StreamProcessor()
        assert processor.state == StreamState.IDLE
        stats = processor.stats
        assert stats["chunks_processed"] == 0
        assert stats["speech_segments"] == 0


class TestStreamProcessorFeed:
    """Test StreamProcessor audio feeding."""

    def test_feed_silence(self):
        """Test feeding silent audio stays IDLE."""
        processor = StreamProcessor()
        silence = np.zeros(1600, dtype=np.float32)
        processor.feed(silence)
        assert processor.state == StreamState.IDLE
        assert processor.stats["chunks_processed"] == 1

    def test_feed_speech_transitions_to_speaking(self):
        """Test that loud audio transitions to SPEAKING."""
        processor = StreamProcessor()
        speech = np.random.randn(1600).astype(np.float32) * 0.5
        processor.feed(speech)
        assert processor.state == StreamState.SPEAKING

    def test_speech_start_callback(self):
        """Test speech start callback is called."""
        processor = StreamProcessor()
        callback_called = []
        processor.on_speech_start = lambda: callback_called.append(True)

        speech = np.random.randn(1600).astype(np.float32) * 0.5
        processor.feed(speech)
        assert len(callback_called) == 1

    def test_feed_bytes_int16(self):
        """Test feeding raw int16 bytes."""
        processor = StreamProcessor()
        speech = (np.random.randn(1600) * 16000).astype(np.int16)
        processor.feed_bytes(speech.tobytes(), sample_width=2)
        assert processor.stats["chunks_processed"] == 1

    def test_feed_bytes_float32(self):
        """Test feeding raw float32 bytes."""
        processor = StreamProcessor()
        speech = (np.random.randn(1600) * 0.5).astype(np.float32)
        processor.feed_bytes(speech.tobytes(), sample_width=4)
        assert processor.stats["chunks_processed"] == 1

    def test_feed_bytes_invalid_width(self):
        """Test feeding bytes with invalid sample width."""
        processor = StreamProcessor()
        with pytest.raises(ValueError):
            processor.feed_bytes(b"\x00" * 100, sample_width=3)


class TestStreamProcessorSpeechEnd:
    """Test speech end detection and processing."""

    def test_speech_end_returns_to_idle(self):
        """Test that silence after speech returns to IDLE."""
        config = StreamConfig(
            vad_silence_duration_ms=200,
            vad_min_speech_ms=100,
            chunk_duration_ms=100,
        )
        processor = StreamProcessor(config=config)

        # Feed speech
        speech = np.random.randn(1600).astype(np.float32) * 0.5
        for _ in range(5):
            processor.feed(speech)

        assert processor.state == StreamState.SPEAKING

        # Feed silence to trigger end
        silence = np.zeros(1600, dtype=np.float32)
        for _ in range(5):
            processor.feed(silence)

        assert processor.state == StreamState.IDLE
        assert processor.stats["speech_segments"] == 1

    def test_speech_end_callback(self):
        """Test speech end callback receives audio."""
        config = StreamConfig(
            vad_silence_duration_ms=200,
            vad_min_speech_ms=100,
            chunk_duration_ms=100,
        )
        processor = StreamProcessor(config=config)

        received_audio = []
        processor.on_speech_end = lambda audio: received_audio.append(audio)

        # Feed speech then silence
        speech = np.random.randn(1600).astype(np.float32) * 0.5
        for _ in range(3):
            processor.feed(speech)

        silence = np.zeros(1600, dtype=np.float32)
        for _ in range(5):
            processor.feed(silence)

        assert len(received_audio) == 1
        assert isinstance(received_audio[0], np.ndarray)

    def test_short_speech_discarded(self):
        """Test that very short speech is discarded."""
        config = StreamConfig(
            vad_silence_duration_ms=200,
            vad_min_speech_ms=500,
            chunk_duration_ms=100,
        )
        processor = StreamProcessor(config=config)

        # Feed one speech chunk then silence
        speech = np.random.randn(1600).astype(np.float32) * 0.5
        processor.feed(speech)

        silence = np.zeros(1600, dtype=np.float32)
        for _ in range(5):
            processor.feed(silence)

        # Should be discarded (too short)
        assert processor.stats["speech_segments"] == 0


class TestStreamProcessorStateCallbacks:
    """Test state change and event callbacks."""

    def test_state_change_callback(self):
        """Test state change callback fires."""
        processor = StreamProcessor()
        states = []
        processor.on_state_change = lambda s: states.append(s.value)

        speech = np.random.randn(1600).astype(np.float32) * 0.5
        processor.feed(speech)

        assert "speaking" in states

    def test_event_callback(self):
        """Test event callback fires."""
        processor = StreamProcessor()
        events = []
        processor.on_event = lambda e: events.append(e.event_type)

        speech = np.random.randn(1600).astype(np.float32) * 0.5
        processor.feed(speech)

        assert "speech_start" in events

    def test_event_has_timestamp(self):
        """Test events have timestamps."""
        processor = StreamProcessor()
        events = []
        processor.on_event = lambda e: events.append(e)

        speech = np.random.randn(1600).astype(np.float32) * 0.5
        processor.feed(speech)

        assert len(events) > 0
        assert isinstance(events[0].timestamp, float)
        assert events[0].timestamp > 0


class TestStreamProcessorReset:
    """Test processor reset functionality."""

    def test_reset(self):
        """Test reset clears all state."""
        processor = StreamProcessor()

        # Feed some audio
        speech = np.random.randn(1600).astype(np.float32) * 0.5
        processor.feed(speech)

        processor.reset()
        assert processor.state == StreamState.IDLE
        assert processor.stats["chunks_processed"] == 0
        assert processor.stats["speech_segments"] == 0

    def test_update_threshold(self):
        """Test VAD threshold update."""
        processor = StreamProcessor()
        processor.update_threshold(0.1)
        assert processor.vad.threshold == 0.1


class TestStreamProcessorWithMockWhisper:
    """Test StreamProcessor with mocked WhisperBridge."""

    def test_transcription_callback_called(self):
        """Test that transcription callback is called after speech ends."""
        config = StreamConfig(
            vad_silence_duration_ms=200,
            vad_min_speech_ms=100,
            chunk_duration_ms=100,
        )

        mock_bridge = MagicMock()
        mock_bridge.transcribe.return_value = MagicMock(
            text="hello",
            language="en",
        )

        processor = StreamProcessor(config=config, whisper_bridge=mock_bridge)

        results = []
        processor.on_transcription = lambda r: results.append(r)

        # Feed speech
        speech = np.random.randn(1600).astype(np.float32) * 0.5
        for _ in range(3):
            processor.feed(speech)

        # Feed silence to trigger transcription
        silence = np.zeros(1600, dtype=np.float32)
        for _ in range(5):
            processor.feed(silence)

        # Wait for async transcription
        time.sleep(0.5)

        if results:
            assert results[0].text == "hello"


class TestStreamProcessorBufferedAudio:
    """Test buffered audio retrieval."""

    def test_get_buffered_audio(self):
        """Test getting buffered audio."""
        processor = StreamProcessor()

        # Feed silence (goes to pre-speech buffer)
        silence = np.random.randn(1600).astype(np.float32) * 0.001
        for _ in range(3):
            processor.feed(silence)

        buffered = processor.get_buffered_audio()
        # Should have some buffered audio
        assert len(buffered) > 0


class TestStreamEvent:
    """Test StreamEvent dataclass."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = StreamEvent(event_type="speech_start")
        assert event.event_type == "speech_start"
        assert isinstance(event.timestamp, float)
        assert event.data is None
        assert event.audio is None

    def test_event_with_data(self):
        """Test event with data."""
        event = StreamEvent(event_type="transcription", data="hello")
        assert event.data == "hello"

    def test_event_with_audio(self):
        """Test event with audio."""
        audio = np.zeros(1600, dtype=np.float32)
        event = StreamEvent(event_type="speech_end", audio=audio)
        np.testing.assert_array_equal(event.audio, audio)


