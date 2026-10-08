"""
Tests for the GPU pipeline operators (v2.6.0).

Covers:
    - ``WhisperOptimizer`` audio front-end: eager mel extraction (torch and the
      NumPy fallback) plus the CUDA-graph capture / replay contract
    - ``PinnedUploadChannel``: pinned-memory staging, non-blocking upload
      statistics, truncation, lifecycle and the simulated (CPU) backend
    - ``StreamProcessor`` integration of the async upload channel

All tests are hardware agnostic: the CUDA-only assertions are skipped when no
device is present, while the graceful-degradation contracts (status objects and
the simulated upload backend) are always verified.
"""

import numpy as np
import pytest

from vram_core.stream_processor import (
    PinnedUploadChannel,
    StreamConfig,
    StreamProcessor,
)
from vram_core.whisper import WhisperOptimizer
from vram_core.whisper.optimizer import FrontendGraphStatus


def sine(num_samples: int = 1600, frequency: float = 440.0, amplitude: float = 0.3):
    t = np.arange(num_samples) / 16000.0
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


# ─── Audio front-end: eager path ────────────────────────────────────────────

class TestFrontendMel:
    """Fixed-shape STFT + mel extraction used by the CUDA graph."""

    def test_shape_and_dtype(self):
        optimizer = WhisperOptimizer(model_name="base", device="cpu")
        mel = optimizer.frontend_mel(sine(1600))
        assert mel.ndim == 2
        assert mel.shape[0] == 80
        assert mel.shape[1] > 0
        assert mel.dtype == np.float32
        assert np.isfinite(mel).all()

    def test_custom_resolution(self):
        optimizer = WhisperOptimizer(model_name="base", device="cpu")
        mel = optimizer.frontend_mel(sine(1600), n_fft=256, hop_length=128, n_mels=32)
        assert mel.shape[0] == 32

    def test_empty_audio(self):
        optimizer = WhisperOptimizer(model_name="base", device="cpu")
        mel = optimizer.frontend_mel(np.array([], dtype=np.float32), n_mels=40)
        assert mel.shape == (40, 0)

    def test_short_chunk_is_padded(self):
        optimizer = WhisperOptimizer(model_name="base", device="cpu")
        mel = optimizer.frontend_mel(sine(64), n_fft=512, hop_length=256, n_mels=20)
        assert mel.shape[0] == 20
        assert np.isfinite(mel).all()

    def test_torch_path_follows_numpy_fallback_pattern(self):
        pytest.importorskip("torch")
        optimizer = WhisperOptimizer(model_name="base", device="cpu")
        audio = sine(1600)
        n_fft, hop_length, n_mels = 512, 256, 40

        torch_mel = optimizer.frontend_mel(
            audio, use_graph=False, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        numpy_mel = WhisperOptimizer._frontend_mel_numpy(audio, n_fft, hop_length, n_mels)

        # scipy normalises the STFT by the window sum while torch does not, so the
        # log-mel matrices differ by a constant per frame: compare after removing
        # that offset.
        frames = min(torch_mel.shape[1], numpy_mel.shape[1])
        assert frames > 0
        a = torch_mel[:, :frames] - torch_mel[:, :frames].mean(axis=0, keepdims=True)
        b = numpy_mel[:, :frames] - numpy_mel[:, :frames].mean(axis=0, keepdims=True)
        correlation = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        assert correlation > 0.9


# ─── Audio front-end: CUDA graph ────────────────────────────────────────────

class TestFrontendGraph:
    """Capture / replay contract, including the CPU-only degradation path."""

    def test_capture_returns_status_object(self):
        optimizer = WhisperOptimizer(model_name="base", device="cpu")
        status = optimizer.capture_frontend_graph(1600, n_mels=80)
        assert isinstance(status, FrontendGraphStatus)
        assert status.sample_chunk_size == 1600
        assert status.n_mels == 80
        assert status.n_fft == 512
        assert status.hop_length == 256
        assert isinstance(status.captured, bool)

    def test_cpu_only_reports_reason_instead_of_raising(self):
        if cuda_available():
            pytest.skip("CUDA device present; degradation path not applicable")
        optimizer = WhisperOptimizer(model_name="base", device="cpu")
        status = optimizer.capture_frontend_graph(1600)
        assert status.captured is False
        assert status.reason
        assert optimizer.is_frontend_graph_ready is False
        assert optimizer.frontend_graph_status is status

    def test_replay_without_graph_returns_none(self):
        optimizer = WhisperOptimizer(model_name="base", device="cpu")
        assert optimizer.run_frontend_graph(sine(1600)) is None

    def test_status_serialises(self):
        optimizer = WhisperOptimizer(model_name="base", device="cpu")
        optimizer.capture_frontend_graph(800)
        data = optimizer.frontend_graph_status.to_dict()
        assert set(data) >= {"captured", "reason", "sample_chunk_size", "device"}

    def test_invalid_capture_arguments(self):
        optimizer = WhisperOptimizer(model_name="base", device="cpu")
        with pytest.raises(ValueError):
            optimizer.capture_frontend_graph(0)
        with pytest.raises(ValueError):
            optimizer.capture_frontend_graph(1600, n_fft=0)

    def test_release_frontend_graph(self):
        optimizer = WhisperOptimizer(model_name="base", device="cpu")
        optimizer.capture_frontend_graph(1600)
        optimizer.release_frontend_graph()
        assert optimizer.is_frontend_graph_ready is False
        assert optimizer.frontend_graph_status.captured is False
        assert optimizer.frontend_graph_status.reason == "released"

    def test_frontend_mel_falls_back_to_eager(self):
        optimizer = WhisperOptimizer(model_name="base", device="cpu")
        optimizer.capture_frontend_graph(1600)          # no-op without CUDA
        mel = optimizer.frontend_mel(sine(1600), use_graph=True)
        assert mel.shape[0] == 80
        assert np.isfinite(mel).all()

    def test_graph_replay_matches_eager_path(self):
        if not cuda_available():
            pytest.skip("CUDA device required for graph replay")
        optimizer = WhisperOptimizer(model_name="base", device="cuda:0")
        status = optimizer.capture_frontend_graph(1600, n_mels=64)
        assert status.captured is True

        audio = sine(1600)
        replayed = optimizer.run_frontend_graph(audio)
        assert replayed is not None
        assert replayed.shape[0] == 64

        eager = optimizer.frontend_mel(audio, use_graph=False, n_mels=64)
        np.testing.assert_allclose(replayed.numpy(), eager, rtol=1e-3, atol=1e-3)
        optimizer.release_frontend_graph()

    def test_wrong_chunk_length_falls_back(self):
        if not cuda_available():
            pytest.skip("CUDA device required for graph replay")
        optimizer = WhisperOptimizer(model_name="base", device="cuda:0")
        optimizer.capture_frontend_graph(1600)
        assert optimizer.run_frontend_graph(sine(800)) is None
        optimizer.release_frontend_graph()


# ─── Pinned-memory upload channel ───────────────────────────────────────────

class TestPinnedUploadChannel:
    """Staging semantics and statistics, on CUDA and on the simulated backend."""

    def test_backend_is_reported(self):
        channel = PinnedUploadChannel(buffer_size=1600)
        assert channel.backend in ("simulated", "cuda")
        assert channel.is_cuda is (channel.backend == "cuda")
        assert channel.staging_buffer.shape == (1600,)

    def test_invalid_buffer_size_raises(self):
        with pytest.raises(ValueError):
            PinnedUploadChannel(buffer_size=0)

    def test_upload_counts_samples_and_bytes(self):
        channel = PinnedUploadChannel(buffer_size=1600)
        channel.upload(sine(1600))
        channel.upload(sine(800))
        stats = channel.stats()
        assert stats["uploads"] == 2
        assert stats["bytes"] == (1600 + 800) * 4
        assert stats["last_upload_ms"] >= 0.0
        assert stats["avg_upload_ms"] >= 0.0

    def test_upload_truncates_overlong_chunks(self):
        channel = PinnedUploadChannel(buffer_size=100)
        channel.upload(sine(1000))
        assert channel.stats()["bytes"] == 100 * 4
        if not channel.is_cuda:      # staged data is observable on the CPU backend
            np.testing.assert_allclose(channel.staging_buffer, sine(1000)[:100])

    def test_empty_chunk_is_ignored(self):
        channel = PinnedUploadChannel(buffer_size=256)
        assert channel.upload(np.array([], dtype=np.float32)) is None
        assert channel.upload(None) is None
        assert channel.stats()["uploads"] == 0

    def test_disabled_channel_is_inert(self):
        channel = PinnedUploadChannel(buffer_size=256, enabled=False)
        assert channel.backend == "simulated"
        assert channel.upload(sine(256)) is None
        assert channel.stats()["uploads"] == 0

    def test_synchronize_and_close(self):
        channel = PinnedUploadChannel(buffer_size=256)
        channel.upload(sine(256))
        channel.synchronize()                     # must not raise on either backend
        channel.close()
        assert channel.backend == "simulated"

    def test_context_manager_closes(self):
        with PinnedUploadChannel(buffer_size=128) as channel:
            channel.upload(sine(128))
        assert channel.backend == "simulated"

    def test_cuda_upload_returns_device_tensor(self):
        if not cuda_available():
            pytest.skip("CUDA device required")
        channel = PinnedUploadChannel(buffer_size=512)
        assert channel.is_cuda is True
        tensor = channel.upload(sine(512))
        assert tensor is not None
        assert tensor.is_cuda
        channel.synchronize()
        np.testing.assert_allclose(
            tensor.detach().cpu().numpy(), sine(512), rtol=1e-5, atol=1e-6
        )
        channel.close()


# ─── StreamProcessor integration ────────────────────────────────────────────

class TestStreamProcessorUploadIntegration:
    """The upload channel is mounted only when requested and is fed per chunk."""

    def test_disabled_by_default(self):
        processor = StreamProcessor()
        assert processor.upload_channel is None
        stats = processor.upload_stats
        assert stats["enabled"] is False

    def test_config_flag_mounts_channel(self):
        processor = StreamProcessor(config=StreamConfig(async_upload=True))
        assert processor.upload_channel is not None
        assert processor.upload_stats["enabled"] is True

    def test_config_flag_defaults_to_off(self):
        assert StreamConfig().async_upload is False

    def test_explicit_override_mounts_channel(self):
        processor = StreamProcessor(config=StreamConfig(), async_upload=True)
        assert processor.upload_channel is not None

    def test_feed_pushes_every_chunk(self):
        processor = StreamProcessor(config=StreamConfig(async_upload=True))
        processor.feed(sine(1600))
        processor.feed(sine(1600))
        stats = processor.upload_stats
        assert stats["uploads"] == 2
        assert stats["bytes"] == 2 * 1600 * 4

    def test_injected_channel_is_used(self):
        channel = PinnedUploadChannel(buffer_size=1600)
        processor = StreamProcessor(upload_channel=channel)
        processor.feed(sine(1600))
        assert channel.stats()["uploads"] == 1
        assert processor.upload_stats["uploads"] == 1

    def test_reset_synchronizes_without_error(self):
        processor = StreamProcessor(config=StreamConfig(async_upload=True))
        processor.feed(sine(1600))
        processor.reset()
        assert processor.upload_stats["uploads"] == 1

    def test_upload_stats_is_a_copy(self):
        processor = StreamProcessor(config=StreamConfig(async_upload=True))
        stats = processor.upload_stats
        stats["uploads"] = 999
        assert processor.upload_stats["uploads"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


