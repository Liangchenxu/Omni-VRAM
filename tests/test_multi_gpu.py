"""
Tests for vram_core.multi_gpu module.

Covers:
    - GPUInfo dataclass
    - TaskAssignment dataclass
    - MultiGPUManager initialization and detection
    - GPU info retrieval (mocked torch/nvml)
    - Load balancing (get_best_gpu)
    - Parallel transcription
    - Memory management (cache clearing)
    - Status formatting
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from concurrent.futures import Future

import numpy as np

from vram_core.multi_gpu import (
    GPUInfo,
    TaskAssignment,
    MultiGPUManager,
)


class TestGPUInfo:
    """Test GPUInfo dataclass."""

    def test_default_values(self):
        info = GPUInfo(
            device_id=0, name="RTX 4090",
            total_memory_mb=24000, free_memory_mb=20000, used_memory_mb=4000,
        )
        assert info.device_id == 0
        assert info.is_available
        assert info.utilization_pct == 0.0
        assert info.temperature_c == 0

    def test_free_memory_gb(self):
        info = GPUInfo(
            device_id=0, name="Test", total_memory_mb=24576,
            free_memory_mb=12288, used_memory_mb=12288,
        )
        assert round(abs(info.free_memory_gb - 12.0), 1) == 0

    def test_total_memory_gb(self):
        info = GPUInfo(
            device_id=0, name="Test", total_memory_mb=24576,
            free_memory_mb=12288, used_memory_mb=12288,
        )
        assert round(abs(info.total_memory_gb - 24.0), 1) == 0

    def test_usage_pct(self):
        info = GPUInfo(
            device_id=0, name="Test", total_memory_mb=10000,
            free_memory_mb=3000, used_memory_mb=7000,
        )
        assert round(abs(info.usage_pct - 70.0), 1) == 0

    def test_usage_pct_zero_total(self):
        info = GPUInfo(
            device_id=0, name="Test", total_memory_mb=0,
            free_memory_mb=0, used_memory_mb=0,
        )
        assert info.usage_pct == 0.0


class TestTaskAssignment:
    """Test TaskAssignment dataclass."""

    def test_creation(self):
        assignment = TaskAssignment(
            device_id=1, gpu_name="RTX 3090", free_memory_mb=18000, task_id="t1",
        )
        assert assignment.device_id == 1
        assert assignment.task_id == "t1"


class TestMultiGPUManagerInit:
    """Test MultiGPUManager initialization."""

    @patch.object(MultiGPUManager, '_detect_gpu_count', return_value=0)
    def test_init_no_gpus(self, mock_detect):
        manager = MultiGPUManager()
        assert manager.gpu_count == 0
        assert not manager.is_available

    @patch.object(MultiGPUManager, '_detect_gpu_count', return_value=4)
    def test_init_with_gpus(self, mock_detect):
        manager = MultiGPUManager()
        assert manager.gpu_count == 4
        assert manager.is_available

    @patch.object(MultiGPUManager, '_detect_gpu_count', return_value=2)
    def test_init_custom_params(self, mock_detect):
        manager = MultiGPUManager(min_free_memory_mb=2048, prefer_device=1)
        assert manager.min_free_memory_mb == 2048
        assert manager.prefer_device == 1


class TestMultiGPUManagerDetection:
    """Test GPU detection logic."""

    @patch('vram_core.multi_gpu._CUDA_AVAILABLE', False)
    @patch('vram_core.multi_gpu._NVML_AVAILABLE', False)
    @patch('vram_core.multi_gpu._TORCH_AVAILABLE', False)
    def test_detect_no_backends(self):
        count = MultiGPUManager._detect_gpu_count()
        assert count == 0

    @patch('vram_core.multi_gpu._NVML_AVAILABLE', False)
    def test_detect_zero_gpus(self):
        # Manager init calls _detect_gpu_count
        with patch.object(MultiGPUManager, '_detect_gpu_count', return_value=0):
            manager = MultiGPUManager()
        assert manager.gpu_count == 0


class TestGetStatus:
    """Test status reporting."""

    @patch.object(MultiGPUManager, '_detect_gpu_count', return_value=0)
    def test_status_no_gpus(self, mock_detect):
        manager = MultiGPUManager()
        status = manager.get_status()
        assert status == "No GPUs available"

    @patch.object(MultiGPUManager, '_detect_gpu_count', return_value=2)
    def test_status_with_gpus(self, mock_detect):
        manager = MultiGPUManager()
        mock_info = GPUInfo(
            device_id=0, name="RTX 4090",
            total_memory_mb=24576, free_memory_mb=20000, used_memory_mb=4576,
            utilization_pct=30.0, temperature_c=45,
        )
        with patch.object(manager, 'get_all_gpu_info', return_value=[mock_info, mock_info]):
            status = manager.get_status()
        assert "2 GPU(s)" in status
        assert "RTX 4090" in status


class TestGetBestGPU:
    """Test load balancing logic."""

    def _make_manager(self, gpu_count=2):
        with patch.object(MultiGPUManager, '_detect_gpu_count', return_value=gpu_count):
            manager = MultiGPUManager(min_free_memory_mb=512)
        return manager

    def test_best_gpu_no_gpus(self):
        manager = self._make_manager(gpu_count=0)
        result = manager.get_best_gpu()
        assert result is None

    def test_best_gpu_returns_most_free(self):
        manager = self._make_manager(gpu_count=2)
        gpu0 = GPUInfo(device_id=0, name="GPU0", total_memory_mb=16000,
                        free_memory_mb=8000, used_memory_mb=8000)
        gpu1 = GPUInfo(device_id=1, name="GPU1", total_memory_mb=16000,
                        free_memory_mb=12000, used_memory_mb=4000)
        with patch.object(manager, 'get_gpu_info', side_effect=lambda i: [gpu0, gpu1][i]):
            result = manager.get_best_gpu()
        assert result is not None
        assert result.device_id == 1

    def test_best_gpu_insufficient_memory(self):
        manager = self._make_manager(gpu_count=1)
        gpu = GPUInfo(device_id=0, name="GPU0", total_memory_mb=16000,
                       free_memory_mb=100, used_memory_mb=15900)
        with patch.object(manager, 'get_gpu_info', return_value=gpu):
            result = manager.get_best_gpu(required_memory_mb=5000)
        assert result is None

    def test_best_gpu_preferred_device(self):
        manager = self._make_manager(gpu_count=2)
        manager.prefer_device = 0
        gpu0 = GPUInfo(device_id=0, name="GPU0", total_memory_mb=16000,
                        free_memory_mb=8000, used_memory_mb=8000)
        with patch.object(manager, 'get_gpu_info', return_value=gpu0):
            result = manager.get_best_gpu()
        assert result is not None
        assert result.device_id == 0


class TestGetGPUInfo:
    """Test GPU info retrieval."""

    def test_invalid_device_id(self):
        with patch.object(MultiGPUManager, '_detect_gpu_count', return_value=1):
            manager = MultiGPUManager()
        with pytest.raises(ValueError):
            manager.get_gpu_info(5)

    def test_fallback_no_backends(self):
        with patch.object(MultiGPUManager, '_detect_gpu_count', return_value=1):
            manager = MultiGPUManager()
        with patch('vram_core.multi_gpu._TORCH_AVAILABLE', False), \
             patch('vram_core.multi_gpu._NVML_AVAILABLE', False):
            info = manager.get_gpu_info(0)
        assert info.name == "Unknown"
        assert info.total_memory_mb == 0


class TestTranscribeParallel:
    """Test parallel transcription."""

    def test_empty_input(self):
        with patch.object(MultiGPUManager, '_detect_gpu_count', return_value=1):
            manager = MultiGPUManager()
        results = manager.transcribe_parallel([])
        assert results == []

    def test_parallel_with_custom_fn(self):
        with patch.object(MultiGPUManager, '_detect_gpu_count', return_value=1):
            manager = MultiGPUManager()

        def mock_transcribe(device_id, audio, sr):
            return "transcribed text"

        audio_files = [("test.wav", np.zeros(16000, dtype=np.float32))]
        with patch.object(manager, 'get_best_gpu', return_value=TaskAssignment(
            device_id=0, gpu_name="GPU0", free_memory_mb=16000,
        )):
            results = manager.transcribe_parallel(
                audio_files, transcribe_fn=mock_transcribe, max_workers=1,
            )
        assert len(results) == 1
        assert results[0]["status"] == "success"
        assert results[0]["result"] == "transcribed text"

    def test_parallel_with_error(self):
        with patch.object(MultiGPUManager, '_detect_gpu_count', return_value=1):
            manager = MultiGPUManager()

        def failing_fn(device_id, audio, sr):
            raise RuntimeError("GPU error")

        audio_files = [("fail.wav", np.zeros(16000, dtype=np.float32))]
        with patch.object(manager, 'get_best_gpu', return_value=TaskAssignment(
            device_id=0, gpu_name="GPU0", free_memory_mb=16000,
        )):
            results = manager.transcribe_parallel(
                audio_files, transcribe_fn=failing_fn, max_workers=1,
            )
        assert len(results) == 1
        assert results[0]["status"] == "error"


class TestClearGPUCache:
    """Test cache clearing."""

    @patch.object(MultiGPUManager, '_detect_gpu_count', return_value=0)
    def test_clear_no_torch(self, mock_detect):
        manager = MultiGPUManager()
        with patch('vram_core.multi_gpu._TORCH_AVAILABLE', False):
            manager.clear_gpu_cache()  # Should not raise


class TestTotalFreeMemory:
    """Test total free memory calculation."""

    @patch.object(MultiGPUManager, '_detect_gpu_count', return_value=2)
    def test_total_free_memory(self, mock_detect):
        manager = MultiGPUManager()
        gpu0 = GPUInfo(device_id=0, name="G0", total_memory_mb=16000,
                        free_memory_mb=8000, used_memory_mb=8000)
        gpu1 = GPUInfo(device_id=1, name="G1", total_memory_mb=16000,
                        free_memory_mb=6000, used_memory_mb=10000)
        with patch.object(manager, 'get_gpu_info', side_effect=lambda i: [gpu0, gpu1][i]):
            total = manager.get_total_free_memory()
        assert total == 14000

    @patch.object(MultiGPUManager, '_detect_gpu_count', return_value=0)
    def test_total_free_memory_no_gpus(self, mock_detect):
        manager = MultiGPUManager()
        assert manager.get_total_free_memory() == 0


