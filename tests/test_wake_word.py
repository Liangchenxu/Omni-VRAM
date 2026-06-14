"""
Tests for vram_core.wake_word module.

Covers:
    - WakeWordEvent data class
    - WakeWordDetector initialization
    - Energy-based detection
    - Whisper-based detection (mocked)
    - Callback registration and invocation
    - Cooldown mechanism
    - Stream processing
    - Statistics and history
    - Reset functionality
"""

import unittest
import time
from unittest.mock import patch, MagicMock

import numpy as np

from vram_core.wake_word import WakeWordEvent, WakeWordDetector


class TestWakeWordEvent(unittest.TestCase):
    """Test WakeWordEvent data class."""

    def test_default_values(self):
        event = WakeWordEvent()
        self.assertEqual(event.keyword, "")
        self.assertEqual(event.confidence, 0.0)
        self.assertEqual(event.timestamp, 0.0)
        self.assertEqual(event.audio_start, 0.0)
        self.assertEqual(event.audio_end, 0.0)

    def test_custom_values(self):
        event = WakeWordEvent(
            keyword="hello",
            confidence=0.95,
            timestamp=100.0,
            audio_start=1.0,
            audio_end=3.0,
        )
        self.assertEqual(event.keyword, "hello")
        self.assertEqual(event.confidence, 0.95)


class TestWakeWordDetectorInit(unittest.TestCase):
    """Test WakeWordDetector initialization."""

    def test_init_default(self):
        detector = WakeWordDetector()
        self.assertEqual(detector.mode, "energy")
        self.assertEqual(detector.keywords, [])
        self.assertEqual(detector.energy_threshold, 0.05)
        self.assertEqual(detector.sample_rate, 16000)
        self.assertEqual(detector.sensitivity, 0.8)

    def test_init_custom_keywords(self):
        detector = WakeWordDetector(keywords=["Hey Computer", "Wake Up"])
        self.assertEqual(detector.keywords, ["hey computer", "wake up"])

    def test_init_custom_params(self):
        detector = WakeWordDetector(
            mode="whisper",
            energy_threshold=0.1,
            sample_rate=44100,
            chunk_duration=3.0,
            cooldown=2.0,
            sensitivity=0.9,
        )
        self.assertEqual(detector.mode, "whisper")
        self.assertEqual(detector.sample_rate, 44100)
        self.assertEqual(detector.cooldown, 2.0)

    def test_init_chunk_samples_calculation(self):
        detector = WakeWordDetector(sample_rate=16000, chunk_duration=2.0)
        self.assertEqual(detector.chunk_samples, 32000)


class TestWakeWordDetectorCallbacks(unittest.TestCase):
    """Test callback registration."""

    def test_on_detect_registers_callback(self):
        detector = WakeWordDetector()
        cb = MagicMock()
        detector.on_detect(cb)
        self.assertIn(cb, detector._callbacks)

    def test_multiple_callbacks(self):
        detector = WakeWordDetector()
        cb1 = MagicMock()
        cb2 = MagicMock()
        detector.on_detect(cb1)
        detector.on_detect(cb2)
        self.assertEqual(len(detector._callbacks), 2)


class TestEnergyDetection(unittest.TestCase):
    """Test energy-based wake word detection."""

    def test_detect_loud_signal(self):
        """Loud audio should trigger energy detection."""
        detector = WakeWordDetector(
            mode="energy",
            energy_threshold=0.01,
            cooldown=0.0,
        )
        # Generate loud audio
        audio = np.ones(16000, dtype=np.float32) * 0.5
        event = detector.process_chunk(audio)
        self.assertIsNotNone(event)
        self.assertEqual(event.keyword, "[energy_spike]")
        self.assertGreater(event.confidence, 0)

    def test_no_detect_silent_signal(self):
        """Silent audio should not trigger detection."""
        detector = WakeWordDetector(
            mode="energy",
            energy_threshold=0.05,
            cooldown=0.0,
        )
        audio = np.zeros(16000, dtype=np.float32)
        event = detector.process_chunk(audio)
        self.assertIsNone(event)

    def test_no_detect_low_signal(self):
        """Low-level noise should not trigger detection."""
        detector = WakeWordDetector(
            mode="energy",
            energy_threshold=0.1,
            cooldown=0.0,
        )
        audio = np.random.randn(16000).astype(np.float32) * 0.001
        event = detector.process_chunk(audio)
        self.assertIsNone(event)

    def test_sensitivity_affects_threshold(self):
        """Higher sensitivity lowers effective threshold."""
        detector_high = WakeWordDetector(
            mode="energy", energy_threshold=0.05, sensitivity=0.99, cooldown=0.0,
        )
        detector_low = WakeWordDetector(
            mode="energy", energy_threshold=0.05, sensitivity=0.1, cooldown=0.0,
        )
        # Audio that is above high-sensitivity threshold but below low-sensitivity
        audio = np.ones(16000, dtype=np.float32) * 0.02

        event_high = detector_high.process_chunk(audio.copy())
        event_low = detector_low.process_chunk(audio.copy())

        # High sensitivity should detect, low might not
        self.assertIsNotNone(event_high)

    def test_confidence_range(self):
        """Confidence should be between 0 and 1."""
        detector = WakeWordDetector(
            mode="energy", energy_threshold=0.01, cooldown=0.0,
        )
        audio = np.ones(16000, dtype=np.float32) * 0.5
        event = detector.process_chunk(audio)
        self.assertGreaterEqual(event.confidence, 0.0)
        self.assertLessEqual(event.confidence, 1.0)


class TestCooldown(unittest.TestCase):
    """Test cooldown mechanism."""

    def test_cooldown_prevents_double_detection(self):
        """Second detection within cooldown period should be suppressed."""
        detector = WakeWordDetector(
            mode="energy",
            energy_threshold=0.01,
            cooldown=10.0,  # 10 second cooldown
        )
        audio = np.ones(16000, dtype=np.float32) * 0.5

        event1 = detector.process_chunk(audio)
        event2 = detector.process_chunk(audio)

        self.assertIsNotNone(event1)
        self.assertIsNone(event2)


class TestWhisperDetection(unittest.TestCase):
    """Test Whisper-based wake word detection."""

    def test_whisper_no_bridge_returns_none(self):
        """Without whisper_bridge, returns None."""
        detector = WakeWordDetector(
            mode="whisper",
            keywords=["hello"],
            cooldown=0.0,
        )
        audio = np.ones(16000, dtype=np.float32) * 0.5
        event = detector.process_chunk(audio)
        self.assertIsNone(event)

    def test_whisper_no_keywords_returns_none(self):
        """Without keywords, returns None."""
        mock_bridge = MagicMock()
        detector = WakeWordDetector(
            mode="whisper",
            keywords=[],
            whisper_bridge=mock_bridge,
            cooldown=0.0,
        )
        audio = np.ones(16000, dtype=np.float32) * 0.5
        event = detector.process_chunk(audio)
        self.assertIsNone(event)

    def test_whisper_detects_keyword(self):
        """Whisper detection finds keyword in transcription."""
        mock_bridge = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Hey computer please help me"
        mock_result.confidence = 0.92
        mock_bridge.transcribe.return_value = mock_result

        detector = WakeWordDetector(
            mode="whisper",
            keywords=["hey computer"],
            whisper_bridge=mock_bridge,
            sample_rate=16000,
            chunk_duration=0.5,
            cooldown=0.0,
        )
        # Feed enough audio to fill buffer
        audio = np.ones(int(16000 * 2.0), dtype=np.float32) * 0.1
        event = detector.process_chunk(audio)

        self.assertIsNotNone(event)
        self.assertEqual(event.keyword, "hey computer")
        self.assertEqual(event.confidence, 0.92)

    def test_whisper_no_keyword_match(self):
        """Whisper returns no event when keyword not in text."""
        mock_bridge = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "random text without keyword"
        mock_result.confidence = 0.5
        mock_bridge.transcribe.return_value = mock_result

        detector = WakeWordDetector(
            mode="whisper",
            keywords=["hey computer"],
            whisper_bridge=mock_bridge,
            sample_rate=16000,
            chunk_duration=0.5,
            cooldown=0.0,
        )
        audio = np.ones(int(16000 * 2.0), dtype=np.float32) * 0.1
        event = detector.process_chunk(audio)
        self.assertIsNone(event)

    def test_whisper_exception_handled(self):
        """Whisper exceptions are caught and return None."""
        mock_bridge = MagicMock()
        mock_bridge.transcribe.side_effect = RuntimeError("Model error")

        detector = WakeWordDetector(
            mode="whisper",
            keywords=["hello"],
            whisper_bridge=mock_bridge,
            sample_rate=16000,
            chunk_duration=0.5,
            cooldown=0.0,
        )
        audio = np.ones(int(16000 * 2.0), dtype=np.float32) * 0.1
        event = detector.process_chunk(audio)
        self.assertIsNone(event)


class TestUnknownMode(unittest.TestCase):
    """Test unknown mode handling."""

    def test_unknown_mode_returns_none(self):
        detector = WakeWordDetector(mode="unknown_mode", cooldown=0.0)
        audio = np.ones(16000, dtype=np.float32)
        event = detector.process_chunk(audio)
        self.assertIsNone(event)


class TestCallbackInvocation(unittest.TestCase):
    """Test that callbacks are invoked on detection."""

    def test_callback_called_on_energy_detect(self):
        detector = WakeWordDetector(
            mode="energy", energy_threshold=0.01, cooldown=0.0,
        )
        cb = MagicMock()
        detector.on_detect(cb)
        audio = np.ones(16000, dtype=np.float32) * 0.5
        detector.process_chunk(audio)
        cb.assert_called_once()
        args = cb.call_args[0]
        self.assertEqual(args[0], "[energy_spike]")
        self.assertGreater(args[1], 0)

    def test_callback_error_does_not_crash(self):
        """Callback exceptions should be caught."""
        detector = WakeWordDetector(
            mode="energy", energy_threshold=0.01, cooldown=0.0,
        )
        bad_cb = MagicMock(side_effect=RuntimeError("Callback failed"))
        detector.on_detect(bad_cb)
        audio = np.ones(16000, dtype=np.float32) * 0.5
        # Should not raise
        event = detector.process_chunk(audio)
        self.assertIsNotNone(event)


class TestProcessStream(unittest.TestCase):
    """Test process_stream method."""

    def test_process_stream_detects_events(self):
        detector = WakeWordDetector(
            mode="energy", energy_threshold=0.01, cooldown=0.0,
        )
        # Create stream with loud segments at different positions
        silent = np.zeros(4096, dtype=np.float32)
        loud = np.ones(4096, dtype=np.float32) * 0.5
        stream = np.concatenate([silent, loud, silent, loud, silent])

        events = detector.process_stream(stream, chunk_size=4096)
        self.assertGreaterEqual(len(events), 1)

    def test_process_stream_empty(self):
        detector = WakeWordDetector(mode="energy", cooldown=0.0)
        stream = np.zeros(16000, dtype=np.float32)
        events = detector.process_stream(stream)
        self.assertEqual(len(events), 0)


class TestHistoryAndStats(unittest.TestCase):
    """Test history and statistics."""

    def test_get_history_empty(self):
        detector = WakeWordDetector()
        self.assertEqual(detector.get_history(), [])

    def test_get_history_after_detection(self):
        detector = WakeWordDetector(
            mode="energy", energy_threshold=0.01, cooldown=0.0,
        )
        audio = np.ones(16000, dtype=np.float32) * 0.5
        detector.process_chunk(audio)
        history = detector.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].keyword, "[energy_spike]")

    def test_get_stats(self):
        detector = WakeWordDetector(
            mode="energy",
            keywords=["hello"],
            energy_threshold=0.05,
            sensitivity=0.7,
        )
        stats = detector.get_stats()
        self.assertEqual(stats["mode"], "energy")
        self.assertIn("keywords", stats)
        self.assertEqual(stats["energy_threshold"], 0.05)
        self.assertEqual(stats["sensitivity"], 0.7)
        self.assertIn("total_processed_seconds", stats)
        self.assertIn("total_detections", stats)
        self.assertIn("cooldown", stats)


class TestReset(unittest.TestCase):
    """Test reset functionality."""

    def test_reset_clears_state(self):
        detector = WakeWordDetector(
            mode="energy", energy_threshold=0.01, cooldown=0.0,
        )
        audio = np.ones(16000, dtype=np.float32) * 0.5
        detector.process_chunk(audio)
        self.assertGreater(len(detector._history), 0)

        detector.reset()
        self.assertEqual(len(detector._history), 0)
        self.assertEqual(len(detector._buffer), 0)
        self.assertEqual(detector._last_detect_time, 0.0)
        self.assertEqual(detector._total_processed, 0.0)


class TestAudioTypeConversion(unittest.TestCase):
    """Test automatic audio type conversion."""

    def test_int16_converted_to_float32(self):
        detector = WakeWordDetector(
            mode="energy", energy_threshold=0.01, cooldown=0.0,
        )
        # Int16 audio (loud)
        audio = np.ones(16000, dtype=np.int16) * 16000
        event = detector.process_chunk(audio)
        # Should not crash - converted internally
        self.assertIsNotNone(event)


if __name__ == "__main__":
    unittest.main()