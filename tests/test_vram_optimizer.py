"""
Tests for vram_core.vram_optimizer module.

Covers:
    - MemoryPressure enum
    - VRAMStatus dataclass
    - KVCacheEstimate dataclass
    - VRAMOptimizer initialization
    - get_status (mocked torch/nvml)
    - _compute_pressure
    - estimate_kv_cache
    - recommend_dtype
    - auto_optimize
    - cleanup_cache / force_cleanup
    - can_allocate
    - get_cleanup_stats
    - get_model_size_estimate
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from vram_core.vram_optimizer import (
    MemoryPressure,
    VRAMStatus,
    KVCacheEstimate,
    VRAMOptimizer,
)


class TestMemoryPressure:
    """Test MemoryPressure enum."""

    def test_values(self):
        assert MemoryPressure.LOW.value == "low"
        assert MemoryPressure.MEDIUM.value == "medium"
        assert MemoryPressure.HIGH.value == "high"
        assert MemoryPressure.CRITICAL.value == "critical"

    def test_all_members(self):
        assert len(MemoryPressure) == 4


class TestVRAMStatus:
    """Test VRAMStatus dataclass."""

    def test_creation(self):
        status = VRAMStatus(
            device_id=0, gpu_name="RTX 4090",
            total_mb=24000, used_mb=12000, free_mb=12000,
            usage_pct=50.0, pressure=MemoryPressure.LOW,
        )
        assert status.device_id == 0
        assert status.pressure == MemoryPressure.LOW

    def test_total_gb(self):
        status = VRAMStatus(
            device_id=0, gpu_name="Test",
            total_mb=24576, used_mb=0, free_mb=24576,
            usage_pct=0.0, pressure=MemoryPressure.LOW,
        )
        assert round(abs(status.total_gb - 24.0), 1) == 0

    def test_free_gb(self):
        status = VRAMStatus(
            device_id=0, gpu_name="Test",
            total_mb=16384, used_mb=8192, free_mb=8192,
            usage_pct=50.0, pressure=MemoryPressure.LOW,
        )
        assert round(abs(status.free_gb - 8.0), 1) == 0


class TestKVCacheEstimate:
    """Test KVCacheEstimate dataclass."""

    def test_creation(self):
        est = KVCacheEstimate(
            total_mb=512.0, per_layer_mb=16.0,
            n_layers=32, seq_length=2048,
            batch_size=1, dtype_bytes=2,
        )
        assert est.total_mb == 512.0
        assert est.n_layers == 32


class TestVRAMOptimizerInit:
    """Test VRAMOptimizer initialization."""

    def test_default_init(self):
        opt = VRAMOptimizer()
        assert opt.device_id == 0
        assert opt.cleanup_threshold_pct == 85.0
        assert opt.target_usage_pct == 60.0

    def test_custom_init(self):
        opt = VRAMOptimizer(device_id=1, cleanup_threshold_pct=90.0, target_usage_pct=50.0)
        assert opt.device_id == 1
        assert opt.cleanup_threshold_pct == 90.0


class TestComputePressure:
    """Test _compute_pressure static method."""

    def test_low(self):
        assert VRAMOptimizer._compute_pressure(25.0) == MemoryPressure.LOW
        assert VRAMOptimizer._compute_pressure(49.9) == MemoryPressure.LOW

    def test_medium(self):
        assert VRAMOptimizer._compute_pressure(50.0) == MemoryPressure.MEDIUM
        assert VRAMOptimizer._compute_pressure(69.9) == MemoryPressure.MEDIUM

    def test_high(self):
        assert VRAMOptimizer._compute_pressure(70.0) == MemoryPressure.HIGH
        assert VRAMOptimizer._compute_pressure(84.9) == MemoryPressure.HIGH

    def test_critical(self):
        assert VRAMOptimizer._compute_pressure(85.0) == MemoryPressure.CRITICAL
        assert VRAMOptimizer._compute_pressure(99.0) == MemoryPressure.CRITICAL

    def test_zero(self):
        assert VRAMOptimizer._compute_pressure(0.0) == MemoryPressure.LOW


class TestEstimateKVCache:
    """Test KV-Cache estimation."""

    def test_default_estimate(self):
        est = VRAMOptimizer.estimate_kv_cache()
        assert est.total_mb > 0
        assert est.n_layers == 32
        assert est.seq_length == 2048
        assert est.dtype_bytes == 2

    def test_fp16_vs_fp32(self):
        est_fp16 = VRAMOptimizer.estimate_kv_cache(dtype_bytes=2)
        est_fp32 = VRAMOptimizer.estimate_kv_cache(dtype_bytes=4)
        assert round(abs(est_fp32.total_mb - est_fp16.total_mb * 2), 1) == 0

    def test_batch_size_scaling(self):
        est1 = VRAMOptimizer.estimate_kv_cache(batch_size=1)
        est4 = VRAMOptimizer.estimate_kv_cache(batch_size=4)
        assert round(abs(est4.total_mb - est1.total_mb * 4), 1) == 0

    def test_seq_length_scaling(self):
        est_short = VRAMOptimizer.estimate_kv_cache(seq_length=1024)
        est_long = VRAMOptimizer.estimate_kv_cache(seq_length=4096)
        assert round(abs(est_long.total_mb - est_short.total_mb * 4), 1) == 0

    def test_per_layer_mb(self):
        est = VRAMOptimizer.estimate_kv_cache(n_layers=32)
        expected_total = est.per_layer_mb * 32
        assert round(abs(est.total_mb - expected_total), 1) == 0


class TestRecommendDtype:
    """Test quantization recommendation."""

    def _make_optimizer_with_free(self, free_mb):
        opt = VRAMOptimizer()
        mock_status = VRAMStatus(
            device_id=0, gpu_name="Test", total_mb=24000,
            used_mb=24000 - free_mb, free_mb=free_mb,
            usage_pct=(24000 - free_mb) / 24000 * 100,
            pressure=MemoryPressure.LOW,
        )
        return opt, mock_status

    def test_plenty_memory_returns_float32(self):
        opt, mock_status = self._make_optimizer_with_free(16000)
        with patch.object(opt, 'get_status', return_value=mock_status):
            assert opt.recommend_dtype() == "float32"

    def test_moderate_memory_returns_float16(self):
        opt, mock_status = self._make_optimizer_with_free(5000)
        with patch.object(opt, 'get_status', return_value=mock_status):
            assert opt.recommend_dtype() == "float16"

    def test_low_memory_returns_int8(self):
        opt, mock_status = self._make_optimizer_with_free(2500)
        with patch.object(opt, 'get_status', return_value=mock_status):
            assert opt.recommend_dtype() == "int8"

    def test_very_low_memory_returns_none(self):
        opt, mock_status = self._make_optimizer_with_free(500)
        with patch.object(opt, 'get_status', return_value=mock_status):
            assert opt.recommend_dtype() == "none"

    def test_with_required_mb_enough_for_fp32(self):
        opt, mock_status = self._make_optimizer_with_free(8000)
        with patch.object(opt, 'get_status', return_value=mock_status):
            assert opt.recommend_dtype(required_mb=3000) == "float32"

    def test_with_required_mb_enough_for_fp16(self):
        opt, mock_status = self._make_optimizer_with_free(5000)
        with patch.object(opt, 'get_status', return_value=mock_status):
            assert opt.recommend_dtype(required_mb=4000) == "float16"

    def test_with_required_mb_enough_for_int8(self):
        opt, mock_status = self._make_optimizer_with_free(3000)
        with patch.object(opt, 'get_status', return_value=mock_status):
            assert opt.recommend_dtype(required_mb=5000) == "int8"

    def test_with_required_mb_not_enough(self):
        opt, mock_status = self._make_optimizer_with_free(500)
        with patch.object(opt, 'get_status', return_value=mock_status):
            assert opt.recommend_dtype(required_mb=5000) == "none"


class TestGetStatus:
    """Test get_status with no GPU available."""

    def test_status_no_gpu(self):
        opt = VRAMOptimizer()
        with patch('vram_core.vram_optimizer._TORCH_AVAILABLE', False), \
             patch('vram_core.vram_optimizer._NVML_AVAILABLE', False):
            status = opt.get_status()
        assert status.gpu_name == "No GPU"
        assert status.total_mb == 0
        assert status.free_mb == 0


class TestAutoOptimize:
    """Test auto_optimize logic."""

    def test_critical_pressure_triggers_force_cleanup(self):
        opt = VRAMOptimizer()
        mock_status = VRAMStatus(
            device_id=0, gpu_name="Test", total_mb=10000,
            used_mb=9000, free_mb=1000, usage_pct=90.0,
            pressure=MemoryPressure.CRITICAL,
        )
        with patch.object(opt, 'get_status', return_value=mock_status), \
             patch.object(opt, 'force_cleanup') as mock_force:
            result = opt.auto_optimize()
        assert result
        mock_force.assert_called_once()

    def test_high_pressure_above_threshold_triggers_cleanup(self):
        opt = VRAMOptimizer(cleanup_threshold_pct=85.0)
        mock_status = VRAMStatus(
            device_id=0, gpu_name="Test", total_mb=10000,
            used_mb=8600, free_mb=1400, usage_pct=86.0,
            pressure=MemoryPressure.HIGH,
        )
        with patch.object(opt, 'get_status', return_value=mock_status), \
             patch.object(opt, 'cleanup_cache') as mock_clean:
            result = opt.auto_optimize()
        assert result
        mock_clean.assert_called_once()

    def test_high_pressure_below_threshold_no_cleanup(self):
        opt = VRAMOptimizer(cleanup_threshold_pct=90.0)
        mock_status = VRAMStatus(
            device_id=0, gpu_name="Test", total_mb=10000,
            used_mb=7500, free_mb=2500, usage_pct=75.0,
            pressure=MemoryPressure.HIGH,
        )
        with patch.object(opt, 'get_status', return_value=mock_status):
            result = opt.auto_optimize()
        assert not result

    def test_low_pressure_no_cleanup(self):
        opt = VRAMOptimizer()
        mock_status = VRAMStatus(
            device_id=0, gpu_name="Test", total_mb=10000,
            used_mb=3000, free_mb=7000, usage_pct=30.0,
            pressure=MemoryPressure.LOW,
        )
        with patch.object(opt, 'get_status', return_value=mock_status):
            result = opt.auto_optimize()
        assert not result

    def test_medium_pressure_no_cleanup(self):
        opt = VRAMOptimizer()
        mock_status = VRAMStatus(
            device_id=0, gpu_name="Test", total_mb=10000,
            used_mb=6000, free_mb=4000, usage_pct=60.0,
            pressure=MemoryPressure.MEDIUM,
        )
        with patch.object(opt, 'get_status', return_value=mock_status):
            result = opt.auto_optimize()
        assert not result


class TestCleanupCache:
    """Test cleanup_cache and force_cleanup."""

    def test_cleanup_no_torch(self):
        opt = VRAMOptimizer()
        with patch('vram_core.vram_optimizer._TORCH_AVAILABLE', False):
            opt.cleanup_cache()  # Should not raise
        assert opt._cleanup_count == 0

    def test_force_cleanup_no_torch(self):
        opt = VRAMOptimizer()
        with patch('vram_core.vram_optimizer._TORCH_AVAILABLE', False):
            opt.force_cleanup()  # Should not raise


class TestCanAllocate:
    """Test can_allocate method."""

    def test_can_allocate_true(self):
        opt = VRAMOptimizer()
        mock_status = VRAMStatus(
            device_id=0, gpu_name="Test", total_mb=24000,
            used_mb=12000, free_mb=12000, usage_pct=50.0,
            pressure=MemoryPressure.MEDIUM,
        )
        with patch.object(opt, 'get_status', return_value=mock_status):
            assert opt.can_allocate(8000)

    def test_can_allocate_false(self):
        opt = VRAMOptimizer()
        mock_status = VRAMStatus(
            device_id=0, gpu_name="Test", total_mb=24000,
            used_mb=23000, free_mb=1000, usage_pct=95.8,
            pressure=MemoryPressure.CRITICAL,
        )
        with patch.object(opt, 'get_status', return_value=mock_status):
            assert not opt.can_allocate(8000)


class TestCleanupStats:
    """Test get_cleanup_stats."""

    def test_initial_stats(self):
        opt = VRAMOptimizer()
        stats = opt.get_cleanup_stats()
        assert stats["cleanup_count"] == 0
        assert stats["last_cleanup_time"] == 0.0
        assert stats["cleanup_threshold_pct"] == 85.0
        assert stats["target_usage_pct"] == 60.0


class TestGetModelSizeEstimate:
    """Test get_model_size_estimate."""

    def test_7b_model_fp16(self):
        # 7B params * 2 bytes = 14GB 锟?14336 MB
        size_mb = VRAMOptimizer.get_model_size_estimate(7.0, dtype_bytes=2)
        assert size_mb == pytest.approx(14336.0, abs=1.0)

    def test_7b_model_fp32(self):
        size_mb = VRAMOptimizer.get_model_size_estimate(7.0, dtype_bytes=4)
        assert size_mb == pytest.approx(28672.0, abs=1.0)

    def test_scaling(self):
        size_fp16 = VRAMOptimizer.get_model_size_estimate(1.0, dtype_bytes=2)
        size_fp32 = VRAMOptimizer.get_model_size_estimate(1.0, dtype_bytes=4)
        assert size_fp32 == pytest.approx(size_fp16 * 2, abs=1.0)


