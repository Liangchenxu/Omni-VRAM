"""
VRAM Optimizer for vram_core
==============================

Intelligent GPU memory management with KV-Cache optimization,
memory monitoring, automatic cleanup, and dynamic quantization.

Features:
    - Real-time VRAM usage monitoring
    - KV-Cache size estimation and management
    - Automatic memory cleanup on threshold breach
    - Dynamic quantization (FP16/INT8) based on available memory
    - Memory pressure levels (low/medium/high/critical)

Usage:
    from vram_core.vram_optimizer import VRAMOptimizer

    optimizer = VRAMOptimizer(device_id=0)
    status = optimizer.get_status()
    optimizer.auto_optimize()

    # Dynamic quantization recommendation
    dtype = optimizer.recommend_dtype(required_mb=2000)
"""

import gc
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


class MemoryPressure(Enum):
    LOW = "low"          # < 50% used
    MEDIUM = "medium"    # 50-70% used
    HIGH = "high"        # 70-85% used
    CRITICAL = "critical"  # > 85% used


@dataclass
class VRAMStatus:
    """Current VRAM status."""
    device_id: int
    gpu_name: str
    total_mb: int
    used_mb: int
    free_mb: int
    usage_pct: float
    pressure: MemoryPressure
    kv_cache_est_mb: float = 0.0
    temperature_c: int = 0

    @property
    def total_gb(self) -> float:
        return self.total_mb / 1024.0

    @property
    def free_gb(self) -> float:
        return self.free_mb / 1024.0


@dataclass
class KVCacheEstimate:
    """KV-Cache memory estimate for a transformer model."""
    total_mb: float
    per_layer_mb: float
    n_layers: int
    seq_length: int
    batch_size: int
    dtype_bytes: int  # 2 for fp16, 4 for fp32


class VRAMOptimizer:
    """
    Intelligent VRAM optimizer with KV-Cache management.

    Features:
        - Real-time memory monitoring
        - Memory pressure detection
        - Automatic cache clearing on high pressure
        - Dynamic quantization recommendations
        - KV-Cache size estimation

    Args:
        device_id: GPU device ID.
        cleanup_threshold_pct: Memory usage % to trigger cleanup (default 85).
        target_usage_pct: Target memory usage after cleanup (default 60).

    Usage:
        optimizer = VRAMOptimizer(device_id=0)
        print(optimizer.get_status())

        # Auto-optimize memory
        optimizer.auto_optimize()

        # Get quantization recommendation
        dtype = optimizer.recommend_dtype(required_mb=2000)
        # Returns 'float16', 'int8', or 'float32'
    """

    def __init__(
        self,
        device_id: int = 0,
        cleanup_threshold_pct: float = 85.0,
        target_usage_pct: float = 60.0,
    ):
        self.device_id = device_id
        self.cleanup_threshold_pct = cleanup_threshold_pct
        self.target_usage_pct = target_usage_pct
        self._last_cleanup_time = 0.0
        self._cleanup_count = 0

    def get_status(self) -> VRAMStatus:
        """Get current VRAM status."""
        total, used, free = 0, 0, 0
        gpu_name = "No GPU"
        temp = 0

        if _TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(self.device_id)
                gpu_name = props.name
                mem_info = torch.cuda.mem_get_info(self.device_id)
                free = mem_info[0] // (1024 * 1024)
                total = mem_info[1] // (1024 * 1024)
                used = total - free
            except Exception:
                pass
        elif _NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                name = pynvml.nvmlDeviceGetName(handle)
                gpu_name = name.decode("utf-8") if isinstance(name, bytes) else name
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total = mem.total // (1024 * 1024)
                used = mem.used // (1024 * 1024)
                free = mem.free // (1024 * 1024)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass

        usage_pct = (used / total * 100.0) if total > 0 else 0.0
        pressure = self._compute_pressure(usage_pct)

        return VRAMStatus(
            device_id=self.device_id,
            gpu_name=gpu_name,
            total_mb=total,
            used_mb=used,
            free_mb=free,
            usage_pct=usage_pct,
            pressure=pressure,
            temperature_c=temp,
        )

    @staticmethod
    def _compute_pressure(usage_pct: float) -> MemoryPressure:
        if usage_pct < 50:
            return MemoryPressure.LOW
        elif usage_pct < 70:
            return MemoryPressure.MEDIUM
        elif usage_pct < 85:
            return MemoryPressure.HIGH
        else:
            return MemoryPressure.CRITICAL

    # 鈹€鈹€ KV-Cache Estimation 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @staticmethod
    def estimate_kv_cache(
        n_layers: int = 32,
        n_heads: int = 32,
        head_dim: int = 128,
        seq_length: int = 2048,
        batch_size: int = 1,
        dtype_bytes: int = 2,
    ) -> KVCacheEstimate:
        """
        Estimate KV-Cache memory usage for a transformer model.

        Formula: 2 (K+V) 脳 n_layers 脳 n_heads 脳 head_dim 脳 seq_length 脳 batch_size 脳 dtype_bytes

        Args:
            n_layers: Number of transformer layers.
            n_heads: Number of attention heads.
            head_dim: Dimension per head.
            seq_length: Sequence length.
            batch_size: Batch size.
            dtype_bytes: Bytes per element (2=fp16, 4=fp32).

        Returns:
            KVCacheEstimate with memory breakdown.
        """
        per_element = 2 * n_layers * n_heads * head_dim * seq_length * batch_size * dtype_bytes
        total_bytes = per_element
        total_mb = total_bytes / (1024 * 1024)
        per_layer_mb = (2 * n_heads * head_dim * seq_length * batch_size * dtype_bytes) / (1024 * 1024)

        return KVCacheEstimate(
            total_mb=total_mb,
            per_layer_mb=per_layer_mb,
            n_layers=n_layers,
            seq_length=seq_length,
            batch_size=batch_size,
            dtype_bytes=dtype_bytes,
        )

    # 鈹€鈹€ Quantization Recommendation 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def recommend_dtype(self, required_mb: int = 0) -> str:
        """
        Recommend quantization dtype based on available memory.

        Returns:
            "float32" (if plenty of memory),
            "float16" (if moderate memory),
            "int8" (if tight memory),
            "none" (if not enough for any inference)
        """
        status = self.get_status()
        free_mb = status.free_mb

        if required_mb > 0:
            if free_mb >= required_mb * 2:
                return "float32"
            elif free_mb >= required_mb:
                return "float16"
            elif free_mb >= required_mb * 0.5:
                return "int8"
            else:
                return "none"

        # No specific requirement: use thresholds
        if free_mb >= 8000:
            return "float32"
        elif free_mb >= 4000:
            return "float16"
        elif free_mb >= 2000:
            return "int8"
        else:
            return "none"

    # 鈹€鈹€ Memory Management 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def auto_optimize(self) -> bool:
        """
        Automatically optimize VRAM usage.

        Returns True if cleanup was performed.
        """
        status = self.get_status()

        if status.pressure == MemoryPressure.CRITICAL:
            logger.warning("VRAM CRITICAL (%.1f%%) 锟?forcing cleanup", status.usage_pct)
            self.force_cleanup()
            return True
        elif status.pressure == MemoryPressure.HIGH:
            if status.usage_pct >= self.cleanup_threshold_pct:
                logger.info("VRAM HIGH (%.1f%%) 锟?running cleanup", status.usage_pct)
                self.cleanup_cache()
                return True

        return False

    def cleanup_cache(self):
        """Clear PyTorch CUDA cache and run garbage collection."""
        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._last_cleanup_time = time.time()
            self._cleanup_count += 1
            logger.info("GPU cache cleared (cleanup #%d)", self._cleanup_count)

    def force_cleanup(self):
        """Aggressive cleanup: clear all caches and synchronize."""
        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device_id)
            self._last_cleanup_time = time.time()
            self._cleanup_count += 1
            logger.info("Forced GPU cleanup (cleanup #%d)", self._cleanup_count)

    # 鈹€鈹€ Utility 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def can_allocate(self, required_mb: int) -> bool:
        """Check if we can allocate required MB of VRAM."""
        status = self.get_status()
        return status.free_mb >= required_mb

    def get_cleanup_stats(self) -> Dict:
        """Get cleanup statistics."""
        return {
            "cleanup_count": self._cleanup_count,
            "last_cleanup_time": self._last_cleanup_time,
            "cleanup_threshold_pct": self.cleanup_threshold_pct,
            "target_usage_pct": self.target_usage_pct,
        }

    @staticmethod
    def get_model_size_estimate(
        n_params_billion: float,
        dtype_bytes: int = 2,
    ) -> float:
        """Estimate model VRAM in MB given parameter count and dtype."""
        return n_params_billion * 1e9 * dtype_bytes / (1024 * 1024)