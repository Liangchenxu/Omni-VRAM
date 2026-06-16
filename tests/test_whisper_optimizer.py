"""
Unit & Performance Tests for WhisperOptimizer
===============================================
"""

import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

from vram_core.whisper.optimizer import (
    WhisperOptimizer,
    BatchResult,
    CacheStats,
    StreamChunk,
    TranscriptionCache,
)
from vram_core.whisper.result import WhisperResult


class TestTranscriptionCache:
    """Test the SHA256-based transcription cache."""

    def test_cache_miss_then_hit(self):
        cache = TranscriptionCache(max_entries=10, enable_disk=False)
        audio = np.random.randn(16000).astype(np.float32)  # 1s audio
        result = WhisperResult(text="hello world", language="en", confidence=0.9)

        # Miss
        assert cache.get(audio, "base", "en", "int8") is None

        # Put
        cache.put(audio, "base", "en", "int8", result)

        # Hit
        cached = cache.get(audio, "base", "en", "int8")
        assert cached is not None
        assert cached.text == "hello world"

    def test_cache_different_model_misses(self):
        cache = TranscriptionCache(max_entries=10, enable_disk=False)
        audio = np.random.randn(16000).astype(np.float32)
        result = WhisperResult(text="test", language="en", confidence=0.8)

        cache.put(audio, "base", "en", "int8", result)

        # Different model should miss
        assert cache.get(audio, "small", "en", "int8") is None
        # Different language should miss
        assert cache.get(audio, "base", "zh", "int8") is None

    def test_cache_eviction(self):
        cache = TranscriptionCache(max_entries=2, enable_disk=False)

        for i in range(3):
            audio = np.array([float(i)] * 16000, dtype=np.float32)
            result = WhisperResult(text=f"result {i}", language="en", confidence=0.9)
            cache.put(audio, "base", "en", "int8", result)

        stats = cache.stats()
        assert stats.total_entries == 2

    def test_cache_clear(self):
        cache = TranscriptionCache(max_entries=10, enable_disk=False)
        audio = np.random.randn(16000).astype(np.float32)
        cache.put(audio, "base", "en", "int8", WhisperResult(text="x", language="en"))

        cache.clear()
        assert cache.get(audio, "base", "en", "int8") is None
        assert cache.stats().total_entries == 0

    def test_cache_stats_hit_rate(self):
        cache = TranscriptionCache(max_entries=10, enable_disk=False)
        audio = np.random.randn(16000).astype(np.float32)
        cache.put(audio, "base", "en", "int8", WhisperResult(text="x", language="en"))

        # 1 hit, 0 miss
        cache.get(audio, "base", "en", "int8")
        assert cache.stats().hit_rate == 1.0

        # 1 hit, 1 miss
        cache.get(np.zeros(16000, dtype=np.float32), "base", "en", "int8")
        assert cache.stats().hit_rate == pytest.approx(0.5)


class TestBatchResult:
    """Test BatchResult dataclass."""

    def test_empty_batch(self):
        br = BatchResult()
        assert br.avg_rtf == 0.0
        assert br.texts == []
        assert br.summary()["total_files"] == 0

    def test_batch_summary(self):
        br = BatchResult(
            results=[
                WhisperResult(text="a", language="en"),
                WhisperResult(text="b", language="en"),
            ],
            total_files=3,
            successful=2,
            failed=1,
            total_audio_duration=10.0,
            total_processing_time=5.0,
            errors={"bad.wav": "not found"},
        )
        assert br.avg_rtf == pytest.approx(0.5)
        assert br.texts == ["a", "b"]
        s = br.summary()
        assert s["total_files"] == 3
        assert s["successful"] == 2


class TestStreamChunk:
    """Test StreamChunk dataclass."""

    def test_chunk_fields(self):
        sc = StreamChunk(text="hello", start=0.0, end=30.0, is_final=True, chunk_index=2)
        assert sc.text == "hello"
        assert sc.is_final
        assert sc.chunk_index == 2


class TestWhisperOptimizerInit:
    """Test WhisperOptimizer initialization."""

    def test_default_init(self):
        opt = WhisperOptimizer()
        assert opt.model_name == "base"
        assert opt.device == "cpu"
        assert opt.compute_type == "int8"
        assert opt.language is None
        assert opt.beam_size == 5
        assert opt.vad_filter
        assert not opt.is_warm
        assert opt._cache is not None

    def test_custom_init(self):
        opt = WhisperOptimizer(
            model_name="small",
            device="cuda",
            compute_type="float16",
            language="zh",
            enable_cache=False,
            beam_size=3,
            vad_filter=False,
        )
        assert opt.model_name == "small"
        assert opt.device == "cuda"
        assert opt.language == "zh"
        assert opt.beam_size == 3
        assert not opt.vad_filter
        assert opt._cache is None

    def test_quantization_presets(self):
        assert "int8" in WhisperOptimizer.QUANTIZE_PRESETS
        assert "float16" in WhisperOptimizer.QUANTIZE_PRESETS
        assert "float32" in WhisperOptimizer.QUANTIZE_PRESETS


class TestWhisperOptimizerQuantization:
    """Test quantization preset switching."""

    def test_set_quantization_valid(self):
        opt = WhisperOptimizer(compute_type="float16", device="cuda")
        opt.set_quantization("int8")
        assert opt.compute_type == "int8"

    def test_set_quantization_invalid(self):
        opt = WhisperOptimizer()
        with pytest.raises(ValueError):
            opt.set_quantization("nonexistent")

    def test_cpu_forces_int8_for_gpu_presets(self):
        opt = WhisperOptimizer(device="cpu", compute_type="int8")
        opt.set_quantization("float16")
        assert opt.compute_type == "int8"

    def test_same_quantization_no_change(self):
        opt = WhisperOptimizer(compute_type="int8")
        opt.set_quantization("int8")
        assert opt.compute_type == "int8"


class TestWhisperOptimizerAudioPreparation:
    """Test audio preparation utilities."""

    def test_prepare_float32_audio(self):
        opt = WhisperOptimizer()
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = opt._prepare_audio(audio)
        assert result.dtype == np.float32
        assert len(result) == 16000

    def test_prepare_int16_audio_normalized(self):
        opt = WhisperOptimizer()
        audio = np.random.randint(-32768, 32767, size=16000, dtype=np.int16).astype(np.float32)
        result = opt._prepare_audio(audio)
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_audio_to_wav_bytes_roundtrip(self):
        opt = WhisperOptimizer()
        audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32) * 0.5
        wav_bytes = opt._audio_to_wav_bytes(audio, 16000)
        assert wav_bytes.startswith(b"RIFF")
        assert b"WAVE" in wav_bytes
        assert len(wav_bytes) > 44

    def test_prepare_nonexistent_file(self):
        opt = WhisperOptimizer()
        with pytest.raises(FileNotFoundError):
            opt._prepare_audio("/nonexistent/path/audio.wav")


class TestWhisperOptimizerStats:
    """Test statistics and diagnostics."""

    def test_initial_stats(self):
        opt = WhisperOptimizer(model_name="base", device="cpu", compute_type="int8")
        stats = opt.get_stats()
        assert stats["model"] == "base"
        assert stats["device"] == "cpu"
        assert stats["transcribe_count"] == 0
        assert stats["avg_rtf"] == 0.0
        assert not stats["warmup_done"]
        assert not stats["model_loaded"]

    def test_stats_with_cache(self):
        opt = WhisperOptimizer(enable_cache=True)
        stats = opt.get_stats()
        assert "cache" in stats
        assert stats["cache"]["entries"] == 0

    def test_stats_without_cache(self):
        opt = WhisperOptimizer(enable_cache=False)
        stats = opt.get_stats()
        assert "cache" not in stats


class TestWhisperOptimizerMockedTranscription:
    """Test transcription flow with mocked faster-whisper."""

    @patch("vram_core.whisper.optimizer.WhisperOptimizer._ensure_model")
    def test_warmup_sets_flag(self, mock_ensure):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            iter([MagicMock(start=0.0, end=1.0, text=" warmup", no_speech_prob=0.1)]),
            MagicMock(language="en"),
        )
        mock_ensure.return_value = mock_model

        opt = WhisperOptimizer()
        elapsed = opt.warmup(duration_s=1.0)
        assert opt.is_warm
        assert elapsed > 0

    @patch("vram_core.whisper.optimizer.WhisperOptimizer._ensure_model")
    def test_transcribe_uses_cache(self, mock_ensure):
        mock_model = MagicMock()
        mock_seg = MagicMock(start=0.0, end=1.0, text=" hello", no_speech_prob=0.1)
        mock_model.transcribe.return_value = (
            iter([mock_seg]),
            MagicMock(language="en"),
        )
        mock_ensure.return_value = mock_model

        opt = WhisperOptimizer(enable_cache=True)
        audio = np.random.randn(16000).astype(np.float32) * 0.1

        # First call - cache miss, should call model
        r1 = opt.transcribe(audio, sample_rate=16000)
        assert mock_model.transcribe.call_count == 1

        # Second call - cache hit, should NOT call model again
        r2 = opt.transcribe(audio, sample_rate=16000)
        assert mock_model.transcribe.call_count == 1
        assert r1.text == r2.text

    @patch("vram_core.whisper.optimizer.WhisperOptimizer._ensure_model")
    def test_streaming_yields_chunks(self, mock_ensure):
        mock_model = MagicMock()
        chunk_counter = {"n": 0}

        def side_effect(*args, **kwargs):
            chunk_counter["n"] += 1
            seg = MagicMock(
                start=0.0, end=10.0,
                text=f" chunk {chunk_counter['n']}",
                no_speech_prob=0.1,
            )
            return iter([seg]), MagicMock(language="en")

        mock_model.transcribe.side_effect = side_effect
        mock_ensure.return_value = mock_model

        opt = WhisperOptimizer()
        # 70s audio with 30s chunks should yield 3 chunks
        audio = np.random.randn(70 * 16000).astype(np.float32) * 0.1

        chunks = list(opt.transcribe_streaming(audio, chunk_duration_s=30.0, overlap_s=2.0))
        assert len(chunks) >= 2
        for chunk in chunks:
            assert isinstance(chunk, StreamChunk)
            assert chunk.start > -1
        # Last chunk should be marked as final
        assert chunks[-1].is_final

    def test_release_vram_returns_stats(self):
        opt = WhisperOptimizer()
        stats = opt.release_vram()
        assert "before_mb" in stats
        assert "after_mb" in stats


class TestWhisperOptimizerBenchmark:
    """Test benchmark functionality."""

    @patch("pathlib.Path.exists", return_value=True)
    @patch("vram_core.whisper.optimizer.WhisperOptimizer._prepare_audio")
    @patch("vram_core.whisper.optimizer.WhisperOptimizer._transcribe_internal")
    def test_benchmark_returns_metrics(self, mock_transcribe, mock_prepare, mock_exists):
        mock_prepare.return_value = np.random.randn(16000).astype(np.float32) * 0.1
        mock_transcribe.return_value = WhisperResult(
            text="test result", language="en", confidence=0.9
        )

        opt = WhisperOptimizer()
        result = opt.benchmark("test.wav", iterations=2, warmup=False)

        assert result["iterations"] == 2
        assert "avg_time_s" in result
        assert "avg_rtf" in result
        assert "throughput_x_realtime" in result


