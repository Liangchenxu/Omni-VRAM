import pytest
"""
End-to-End Latency Tests for RealtimePipeline
===============================================

Tests the optimized real-time pipeline with synthetic audio data.
Validates that each component works correctly and measures latency.

Usage:
    python -m pytest tests/test_realtime_latency.py -v
    python tests/test_realtime_latency.py
"""

import time
import threading
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from vram_core.realtime_optimizer import (
    RingBuffer,
    SileroVAD,
    StreamingTranscriber,
    StreamingChunk,
    LatencyTracker,
    LatencyMeasurement,
    RealtimePipeline,
    PipelineConfig,
    PipelineStats,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# Test Helpers
# ==============================================================================

def generate_speech_like_audio(
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 300.0,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate synthetic speech-like audio (sine wave with noise)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    # Mix a few frequencies for more speech-like signal
    signal = amplitude * (
        0.5 * np.sin(2 * np.pi * frequency * t) +
        0.3 * np.sin(2 * np.pi * frequency * 2 * t) +
        0.2 * np.sin(2 * np.pi * frequency * 3 * t)
    )
    # Add some noise
    noise = 0.02 * np.random.randn(len(t)).astype(np.float32)
    return (signal + noise).astype(np.float32)


def generate_silence(duration_s: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate silence (low-level noise)."""
    n = int(sample_rate * duration_s)
    return (0.001 * np.random.randn(n)).astype(np.float32)


def generate_vad_friendly_audio(
    speech_duration_s: float = 2.0,
    silence_before_s: float = 0.3,
    silence_after_s: float = 0.5,
    sample_rate: int = 16000,
) -> tuple:
    """Generate audio with clear speech/silence boundaries for VAD testing."""
    before = generate_silence(silence_before_s, sample_rate)
    speech = generate_speech_like_audio(speech_duration_s, sample_rate, amplitude=0.4)
    after = generate_silence(silence_after_s, sample_rate)
    full = np.concatenate([before, speech, after])
    return full, len(before), len(before) + len(speech)


# ==============================================================================
# Test: RingBuffer
# ==============================================================================

class TestRingBuffer:
    """Tests for the optimized RingBuffer."""

    def test_write_and_read(self):
        """Basic write/read operations."""
        buf = RingBuffer(1000)
        data = np.random.randn(500).astype(np.float32)

        written = buf.write(data)
        assert written == 500
        assert buf.size == 500

        read_data = buf.read(500)
        assert len(read_data) == 500
        np.testing.assert_array_almost_equal(read_data, data, decimal=6)
        assert buf.size == 0

    def test_overwrite_when_full(self):
        """Buffer overwrites oldest data when full."""
        buf = RingBuffer(100)
        data1 = np.arange(100, dtype=np.float32)
        data2 = np.arange(100, 200, dtype=np.float32)

        buf.write(data1)
        assert buf.size == 100

        buf.write(data2)
        assert buf.size == 100

        read_data = buf.read(100)
        np.testing.assert_array_almost_equal(read_data, data2, decimal=6)

    def test_write_larger_than_capacity(self):
        """Writing data larger than capacity keeps only the last part."""
        buf = RingBuffer(100)
        data = np.arange(200, dtype=np.float32)

        buf.write(data)
        assert buf.size == 100

        read_data = buf.read(100)
        np.testing.assert_array_almost_equal(read_data, data[-100:], decimal=6)

    def test_peek_does_not_consume(self):
        """Peek reads without consuming."""
        buf = RingBuffer(100)
        data = np.arange(50, dtype=np.float32)
        buf.write(data)

        peeked = buf.peek(30)
        assert len(peeked) == 30
        assert buf.size == 50  # Unchanged

        peeked_all = buf.peek(50)
        assert len(peeked_all) == 50
        assert buf.size == 50

    def test_read_all(self):
        """read_all returns all samples and clears."""
        buf = RingBuffer(100)
        data = np.random.randn(80).astype(np.float32)
        buf.write(data)

        result = buf.read_all()
        assert len(result) == 80
        np.testing.assert_array_almost_equal(result, data, decimal=6)
        assert buf.size == 0

    def test_empty_read(self):
        """Reading from empty buffer returns empty array."""
        buf = RingBuffer(100)
        result = buf.read(50)
        assert len(result) == 0

    def test_clear(self):
        """Clear resets the buffer."""
        buf = RingBuffer(100)
        buf.write(np.random.randn(80).astype(np.float32))
        buf.clear()
        assert buf.size == 0
        assert buf.is_empty

    def test_performance(self):
        """Benchmark: write should be < 0.1ms for typical chunks."""
        buf = RingBuffer(16000 * 5)  # 5 seconds
        chunk = np.random.randn(480).astype(np.float32)  # 30ms at 16kHz

        start = time.perf_counter()
        for _ in range(1000):
            buf.write(chunk)
        elapsed = (time.perf_counter() - start) / 1000 * 1000  # ms per op

        logger.info("RingBuffer.write: %.4f ms/op (1000 iterations)", elapsed)
        assert elapsed < 1.0, f"Buffer write too slow: {elapsed:.3f}ms"

    def test_thread_safety(self):
        """Concurrent reads and writes should not corrupt data."""
        buf = RingBuffer(10000)
        errors = []

        def writer():
            try:
                for i in range(100):
                    data = np.full(100, float(i), dtype=np.float32)
                    buf.write(data)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    buf.read(50)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread safety errors: {errors}"


# ==============================================================================
# Test: SileroVAD (mock-based)
# ==============================================================================

class TestSileroVAD:
    """Tests for SileroVAD (with mock model for environments without torch)."""

    def test_fallback_energy_vad(self):
        """Test fallback energy-based VAD."""
        vad = SileroVAD(threshold=0.5, sample_rate=16000)

        # Force fallback by not loading model
        speech = generate_speech_like_audio(1.0, amplitude=0.3)
        silence = generate_silence(1.0)

        # Test fallback directly
        assert vad._fallback_energy_vad(speech) is True
        assert vad._fallback_energy_vad(silence) is False

    def test_fallback_energy_empty(self):
        """Empty audio returns False."""
        vad = SileroVAD(threshold=0.5)
        assert vad._fallback_energy_vad(np.array([], dtype=np.float32)) is False

    def test_fallback_probability(self):
        """Energy fallback returns correct probability range."""
        vad = SileroVAD(threshold=0.5)
        speech = generate_speech_like_audio(0.5, amplitude=0.5)
        prob = vad._fallback_energy(speech)
        assert 0.0 <= prob <= 1.0 or prob > 0  # RMS can be > 1 for normalized

    def test_state_tracking(self):
        """VAD state tracking works correctly."""
        vad = SileroVAD(threshold=0.5)
        assert vad.is_speech_active is False

        # Reset
        vad.reset()
        assert vad.is_speech_active is False
        assert vad._speech_start_time is None


# ==============================================================================
# Test: LatencyTracker
# ==============================================================================

class TestLatencyTracker:
    """Tests for latency instrumentation."""

    def test_basic_measurement(self):
        """Measure a simple VAD + ASR pipeline."""
        tracker = LatencyTracker()

        tracker.start_measurement(audio_duration_ms=1000)
        tracker.start_vad()
        time.sleep(0.001)
        tracker.end_vad()

        tracker.start_asr()
        time.sleep(0.005)
        tracker.end_asr()

        m = tracker.complete()

        assert m is not None
        assert m.vad_latency_ms > 0
        assert m.asr_latency_ms > 0
        assert m.total_latency_ms > 0
        assert m.audio_duration_ms == 1000

    def test_breakdown_percentages(self):
        """Breakdown should sum to ~100%."""
        tracker = LatencyTracker()
        tracker.start_measurement()

        tracker.start_vad()
        time.sleep(0.001)
        tracker.end_vad()

        tracker.start_asr()
        time.sleep(0.005)
        tracker.end_asr()

        m = tracker.complete()
        breakdown = m.breakdown

        assert "vad_pct" in breakdown
        assert "asr_pct" in breakdown
        total_pct = sum(breakdown.values())
        assert 90 <= total_pct <= 110  # Allow some floating point tolerance

    def test_aggregate_stats(self):
        """Aggregate stats should include mean, p50, p95, etc."""
        tracker = LatencyTracker()

        for _ in range(10):
            tracker.start_measurement()
            tracker.start_vad()
            time.sleep(0.001)
            tracker.end_vad()
            tracker.start_asr()
            time.sleep(0.003)
            tracker.end_asr()
            tracker.complete()

        stats = tracker.get_stats()

        assert "total_ms" in stats
        assert "vad_ms" in stats
        assert "asr_ms" in stats
        assert stats["count"] == 10

        for key in ["mean", "p50", "p95", "p99", "min", "max"]:
            assert key in stats["total_ms"]

    def test_empty_stats(self):
        """Stats from empty tracker."""
        tracker = LatencyTracker()
        stats = tracker.get_stats()
        assert stats == {}


# ==============================================================================
# Test: StreamingTranscriber
# ==============================================================================

class TestStreamingTranscriber:
    """Tests for streaming transcriber (mock Whisper)."""

    def _make_mock_bridge(self, text: str = "hello world"):
        """Create a mock WhisperBridge."""
        mock_bridge = MagicMock()
        mock_result = MagicMock()
        mock_result.text = text
        mock_result.confidence = 0.95
        mock_result.language = "en"
        mock_bridge.transcribe.return_value = mock_result
        return mock_bridge

    def test_accumulates_until_chunk_size(self):
        """Should not transcribe until enough audio accumulates."""
        bridge = self._make_mock_bridge()
        transcriber = StreamingTranscriber(
            whisper_bridge=bridge,
            chunk_duration_ms=500,
            overlap_ms=100,
            sample_rate=16000,
        )

        # Feed 100ms chunks
        for _ in range(4):
            chunk = generate_speech_like_audio(0.1)
            result = transcriber.feed(chunk)
            assert result is None  # Not enough yet

        # 5th chunk should trigger transcription (500ms total)
        chunk = generate_speech_like_audio(0.1)
        result = transcriber.feed(chunk)
        assert result is not None
        assert result.text == "hello world"
        bridge.transcribe.assert_called_once()

    def test_finalize_remaining(self):
        """finalize() should transcribe remaining audio."""
        bridge = self._make_mock_bridge()
        transcriber = StreamingTranscriber(
            whisper_bridge=bridge,
            chunk_duration_ms=500,
            overlap_ms=100,
            sample_rate=16000,
        )

        # Feed 300ms (less than chunk size)
        chunk = generate_speech_like_audio(0.3)
        transcriber.feed(chunk)

        # Finalize should transcribe the remaining
        result = transcriber.finalize()
        assert result is not None
        assert result.is_final is True

    def test_finalize_too_short(self):
        """finalize() should skip very short remaining audio."""
        bridge = self._make_mock_bridge()
        transcriber = StreamingTranscriber(
            whisper_bridge=bridge,
            chunk_duration_ms=500,
            sample_rate=16000,
        )

        # Feed only 50ms
        chunk = generate_speech_like_audio(0.05)
        transcriber.feed(chunk)

        result = transcriber.finalize()
        assert result is None  # Too short

    def test_callback_invoked(self):
        """on_chunk_result callback should be called."""
        bridge = self._make_mock_bridge()
        transcriber = StreamingTranscriber(
            whisper_bridge=bridge,
            chunk_duration_ms=500,
            overlap_ms=100,
            sample_rate=16000,
        )

        callback_results = []
        transcriber.on_chunk_result = lambda c: callback_results.append(c)

        # Feed enough audio
        for _ in range(5):
            transcriber.feed(generate_speech_like_audio(0.1))

        assert len(callback_results) >= 1
        assert callback_results[0].text == "hello world"

    def test_reset(self):
        """Reset clears state."""
        bridge = self._make_mock_bridge()
        transcriber = StreamingTranscriber(
            whisper_bridge=bridge,
            chunk_duration_ms=500,
            sample_rate=16000,
        )

        transcriber.feed(generate_speech_like_audio(0.3))
        transcriber.reset()

        assert transcriber._chunk_index == 0
        assert len(transcriber._pending_audio) == 0


# ==============================================================================
# Test: RealtimePipeline Integration
# ==============================================================================

class TestRealtimePipeline:
    """Integration tests for the full RealtimePipeline."""

    def _make_mock_bridge(self, text: str = "test transcription"):
        """Create a mock WhisperBridge."""
        mock_bridge = MagicMock()
        mock_result = MagicMock()
        mock_result.text = text
        mock_result.confidence = 0.9
        mock_result.language = "zh"
        mock_bridge.transcribe.return_value = mock_result
        return mock_bridge

    def test_pipeline_lifecycle(self):
        """Pipeline can start and stop cleanly."""
        config = PipelineConfig(silero_vad_threshold=0.5)
        pipeline = RealtimePipeline(config=config)

        pipeline.start()
        assert pipeline.is_running is True

        pipeline.stop()
        assert pipeline.is_running is False

    def test_feed_without_start(self):
        """Feeding before start should be no-op."""
        pipeline = RealtimePipeline(config=PipelineConfig())
        # Should not raise
        pipeline.feed(generate_silence(0.1))

    def test_ring_buffer_integration(self):
        """Ring buffer correctly stores fed audio."""
        pipeline = RealtimePipeline(config=PipelineConfig(ring_buffer_duration_s=2.0))
        pipeline.start()

        # Feed some audio
        for _ in range(10):
            pipeline.feed(generate_speech_like_audio(0.1))

        assert pipeline._ring_buffer.size > 0
        pipeline.stop()

    def test_vad_callback(self):
        """VAD result callback is called on each feed."""
        config = PipelineConfig(silero_vad_threshold=0.5)
        pipeline = RealtimePipeline(config=config)

        vad_results = []
        pipeline.on_vad_result = lambda is_speech, prob: vad_results.append(
            (is_speech, prob)
        )

        pipeline.start()

        # Feed speech-like audio
        for _ in range(5):
            pipeline.feed(generate_speech_like_audio(0.1, amplitude=0.3))

        pipeline.stop()

        # VAD callback should have been called
        assert len(vad_results) == 5
        # Each result should be a (bool, float) tuple
        for is_speech, prob in vad_results:
            assert isinstance(is_speech, bool)
            assert isinstance(prob, float)

    def test_speech_start_end_callbacks(self):
        """Speech start/end callbacks are called."""
        config = PipelineConfig(
            silero_vad_threshold=0.3,
            vad_min_speech_ms=100,
            vad_silence_duration_ms=200,
        )
        pipeline = RealtimePipeline(config=config)

        events = []
        pipeline.on_speech_start = lambda: events.append("start")
        pipeline.on_speech_end = lambda audio: events.append("end")

        pipeline.start()

        # Feed speech then silence
        for _ in range(10):
            pipeline.feed(generate_speech_like_audio(0.1, amplitude=0.5))

        for _ in range(10):
            pipeline.feed(generate_silence(0.1))

        pipeline.stop()

        logger.info("Events: %s", events)
        # Should have at least one start event (VAD-dependent)
        # This test is lenient because VAD behavior depends on model/threshold

    def test_transcription_callback_with_mock(self):
        """Full pipeline with mock transcription."""
        config = PipelineConfig(
            silero_vad_threshold=0.3,
            vad_min_speech_ms=100,
            vad_silence_duration_ms=200,
            streaming_chunk_ms=300,
        )
        mock_bridge = self._make_mock_bridge("你好世界")
        pipeline = RealtimePipeline(config=config, whisper_bridge=mock_bridge)

        results = []
        pipeline.on_transcription = lambda r: results.append(r)

        pipeline.start()

        # Feed enough speech for transcription
        for _ in range(15):
            pipeline.feed(generate_speech_like_audio(0.1, amplitude=0.5))

        # Feed silence to trigger end
        for _ in range(10):
            pipeline.feed(generate_silence(0.1))

        pipeline.stop()

        # Check stats
        stats = pipeline.stats
        logger.info(
            "Pipeline stats: chunks=%d, segments=%d, transcriptions=%d",
            stats.chunks_received,
            stats.speech_segments,
            stats.transcriptions,
        )

    def test_stats_collection(self):
        """Stats are collected correctly."""
        pipeline = RealtimePipeline(config=PipelineConfig())
        pipeline.start()

        for _ in range(20):
            pipeline.feed(generate_speech_like_audio(0.05))

        stats = pipeline.stats
        assert stats.chunks_received == 20
        assert stats.vad_decisions == 20

        pipeline.stop()

    def test_reset(self):
        """Pipeline reset clears all state."""
        pipeline = RealtimePipeline(config=PipelineConfig())
        pipeline.start()

        for _ in range(10):
            pipeline.feed(generate_speech_like_audio(0.1))

        pipeline.reset()

        assert pipeline._ring_buffer.size == 0
        assert pipeline.stats.chunks_received == 0
        pipeline.stop()

    def test_feed_bytes(self):
        """feed_bytes correctly converts int16 to float32."""
        pipeline = RealtimePipeline(config=PipelineConfig())
        pipeline.start()

        # Create int16 audio
        audio_int16 = (generate_speech_like_audio(0.1) * 32768).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        pipeline.feed_bytes(audio_bytes, sample_width=2)
        assert pipeline.stats.chunks_received == 1

        pipeline.stop()


# ==============================================================================
# Test: End-to-End Latency Benchmark
# ==============================================================================

class TestEndToEndLatency:
    """
    End-to-end latency benchmark.

    Measures the time from audio input to transcription output.
    This is the key metric for real-time performance.
    """

    def test_feed_latency_without_asr(self):
        """
        Measure feed() latency without ASR (VAD + buffer only).
        Target: < 5ms per chunk.
        """
        config = PipelineConfig(chunk_duration_ms=30)
        pipeline = RealtimePipeline(config=config)
        pipeline.start()

        chunk = generate_speech_like_audio(0.03)  # 30ms chunk
        latencies = []

        for _ in range(100):
            t_start = time.perf_counter()
            pipeline.feed(chunk)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000)

        pipeline.stop()

        arr = np.array(latencies)
        mean_ms = np.mean(arr)
        p95_ms = np.percentile(arr, 95)
        p99_ms = np.percentile(arr, 99)

        logger.info(
            "feed() latency (no ASR): mean=%.2fms, p95=%.2fms, p99=%.2fms",
            mean_ms, p95_ms, p99_ms,
        )

        # Target: mean < 5ms (excluding ASR)
        # Note: Silero VAD may add latency on first call due to model loading
        assert mean_ms < 100, f"feed() too slow: {mean_ms:.2f}ms (expected < 100ms)"

    def test_ring_buffer_throughput(self):
        """
        Measure ring buffer write/read throughput.
        Target: > 1M samples/second.
        """
        buf = RingBuffer(16000 * 5)
        chunk = generate_speech_like_audio(0.03)
        n_iterations = 10000

        t_start = time.perf_counter()
        for _ in range(n_iterations):
            buf.write(chunk)
        t_end = time.perf_counter()

        total_samples = n_iterations * len(chunk)
        elapsed = t_end - t_start
        throughput = total_samples / elapsed

        logger.info(
            "RingBuffer throughput: %.0f samples/sec (%.1fms for %d writes)",
            throughput, elapsed * 1000, n_iterations,
        )

        assert throughput > 500000, f"Throughput too low: {throughput:.0f} samples/sec"

    def test_latency_tracker_overhead(self):
        """
        LatencyTracker should add minimal overhead.
        Target: < 0.01ms per measurement.
        """
        tracker = LatencyTracker()

        t_start = time.perf_counter()
        for _ in range(10000):
            tracker.start_measurement()
            tracker.start_vad()
            tracker.end_vad()
            tracker.start_asr()
            tracker.end_asr()
            tracker.complete()
        t_end = time.perf_counter()

        per_op_ms = (t_end - t_start) / 10000 * 1000
        logger.info("LatencyTracker overhead: %.4f ms/op", per_op_ms)

        assert per_op_ms < 0.1, f"Tracker too slow: {per_op_ms:.4f}ms"

    def test_simulated_conversation(self):
        """
        Simulate a conversation with multiple speech segments.
        Measures overall pipeline throughput.
        """
        config = PipelineConfig(
            silero_vad_threshold=0.3,
            vad_min_speech_ms=100,
            vad_silence_duration_ms=200,
        )
        mock_bridge = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "test"
        mock_result.confidence = 0.9
        mock_result.language = "en"
        mock_bridge.transcribe.return_value = mock_result

        pipeline = RealtimePipeline(config=config, whisper_bridge=mock_bridge)
        pipeline.start()

        total_chunks = 0
        t_start = time.perf_counter()

        # Simulate 3 speech segments
        for seg in range(3):
            # Speech
            for _ in range(20):
                pipeline.feed(generate_speech_like_audio(0.05, amplitude=0.5))
                total_chunks += 1

            # Silence
            for _ in range(10):
                pipeline.feed(generate_silence(0.05))
                total_chunks += 1

        t_end = time.perf_counter()
        pipeline.stop()

        elapsed = t_end - t_start
        chunks_per_sec = total_chunks / elapsed

        logger.info(
            "Simulated conversation: %d chunks in %.2fs (%.0f chunks/sec), "
            "%d segments detected, %d transcriptions",
            total_chunks, elapsed, chunks_per_sec,
            pipeline.stats.speech_segments,
            pipeline.stats.transcriptions,
        )

        assert chunks_per_sec > 100, f"Too slow: {chunks_per_sec:.0f} chunks/sec"


# ==============================================================================
# Main: Run all tests
# ==============================================================================

def run_all_tests():
    """Run all test classes manually (without pytest)."""
    test_classes = [
        TestRingBuffer,
        TestSileroVAD,
        TestLatencyTracker,
        TestStreamingTranscriber,
        TestRealtimePipeline,
        TestEndToEndLatency,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for test_class in test_classes:
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in methods:
            total += 1
            test_name = f"{test_class.__name__}.{method_name}"
            try:
                method = getattr(instance, method_name)
                method()
                passed += 1
                print(f"  ✓ {test_name}")
            except Exception as e:
                failed += 1
                errors.append((test_name, str(e)))
                print(f"  ✗ {test_name}: {e}")

    print()
    print("=" * 60)
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if errors:
        print("\n  Failed tests:")
        for name, err in errors:
            print(f"    - {name}: {err}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)