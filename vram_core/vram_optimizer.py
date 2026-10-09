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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

# Compiled CUDA extension (vram_hacker.cu) -- optional
_CUDA_EXT = None
try:
    import vram_core._vram_hacker as _CUDA_EXT  # type: ignore[no-redef]
except (ImportError, AttributeError):  # pragma: no cover - CPU-only deployments
    _CUDA_EXT = None

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except (ImportError, OSError, RuntimeError):
    _NVML_AVAILABLE = False


class MemoryPressure(Enum):
    LOW = "low"          # < 50% used
    MEDIUM = "medium"    # 50-70% used
    HIGH = "high"        # 70-85% used
    CRITICAL = "critical"  # > 85% used


# ── Named Constants ──────────────────────────────────────────────
_MB_TO_GB: float = 1 / 1024.0
_PRESSURE_LOW_THRESHOLD_PCT: float = 50.0
_PRESSURE_MEDIUM_THRESHOLD_PCT: float = 70.0
_PRESSURE_HIGH_THRESHOLD_PCT: float = 85.0
_DTYPE_BYTES_FP16: int = 2
_DTYPE_BYTES_FP32: int = 4
_MB_BYTES: int = 1024 * 1024
_GB_BYTES: int = 1024 ** 3


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
                free = mem_info[0] // _MB_BYTES
                total = mem_info[1] // _MB_BYTES
                used = total - free
            except (RuntimeError, OSError) as e:
                logger.debug("torch mem_get_info failed: %s", e)
        elif _NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                name = pynvml.nvmlDeviceGetName(handle)
                gpu_name = name.decode("utf-8") if isinstance(name, bytes) else name
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total = mem.total // _MB_BYTES
                used = mem.used // _MB_BYTES
                free = mem.free // _MB_BYTES
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except (pynvml.NVMLError, RuntimeError, OSError) as e:
                logger.debug("NVML mem_get_info failed: %s", e)

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
        if usage_pct < _PRESSURE_LOW_THRESHOLD_PCT:
            return MemoryPressure.LOW
        elif usage_pct < _PRESSURE_MEDIUM_THRESHOLD_PCT:
            return MemoryPressure.MEDIUM
        elif usage_pct < _PRESSURE_HIGH_THRESHOLD_PCT:
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
        # Thresholds based on typical model requirements
        _FREE_THRESHOLD_FLOAT32_MB: int = 8000
        _FREE_THRESHOLD_FLOAT16_MB: int = 4000
        _FREE_THRESHOLD_INT8_MB: int = 2000

        if free_mb >= _FREE_THRESHOLD_FLOAT32_MB:
            return "float32"
        elif free_mb >= _FREE_THRESHOLD_FLOAT16_MB:
            return "float16"
        elif free_mb >= _FREE_THRESHOLD_INT8_MB:
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
            logger.warning("VRAM CRITICAL (%.1f%%) — forcing cleanup", status.usage_pct)
            self.force_cleanup()
            return True
        elif status.pressure == MemoryPressure.HIGH:
            if status.usage_pct >= self.cleanup_threshold_pct:
                logger.info("VRAM HIGH (%.1f%%) — running cleanup", status.usage_pct)
                self.cleanup_cache()
                return True

        return False

    def cleanup_cache(self) -> None:
        """Clear PyTorch CUDA cache and run garbage collection."""
        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError as e:
                logger.warning("Failed to clear CUDA cache: %s", e)
                return
            self._last_cleanup_time = time.time()
            self._cleanup_count += 1
            logger.info("GPU cache cleared (cleanup #%d)", self._cleanup_count)

    def force_cleanup(self) -> None:
        """Aggressive cleanup: clear all caches and synchronize."""
        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device_id)
            except RuntimeError as e:
                logger.warning("Failed during forced GPU cleanup: %s", e)
                return
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
        dtype_bytes: int = _DTYPE_BYTES_FP16,
    ) -> float:
        """Estimate model VRAM in MB given parameter count and dtype.

        Uses binary convention: 1 billion = 1024^3, 1 MB = 1024^2 bytes.
        So 7B FP16 = 7 * 1024 * 2 = 14336 MB.
        """
        return n_params_billion * _GB_BYTES * dtype_bytes / _MB_BYTES

    # ── Paged KV-Cache (v2.6.0) ───────────────────────────────────────────

    def create_paged_cache(
        self,
        model_params_billion: float = 7.0,
        n_layers: int = 32,
        n_heads: int = 32,
        head_dim: int = 128,
        seq_length: int = 2048,
        block_size: int = 16,
        max_sequences: int = 16,
        dtype_bytes: int = _DTYPE_BYTES_FP16,
        **kwargs,
    ) -> "PagedKVCacheManager":
        """
        Size a :class:`PagedKVCacheManager` from the current free VRAM.

        The pool is sized so that ``max_sequences`` concurrent conversations fit
        into the *free* VRAM reported by :meth:`get_status`, leaving the model
        weights untouched.

        Returns:
            A ready-to-use :class:`PagedKVCacheManager`.
        """
        per_token_bytes = 2 * n_heads * head_dim * dtype_bytes  # K + V
        blocks_per_seq = max(1, (seq_length + block_size - 1) // block_size)
        min_blocks = max_sequences * blocks_per_seq

        status = self.get_status()
        free_bytes = max(status.free_mb, 0) * _MB_BYTES
        per_block_bytes = max(per_token_bytes * block_size, 1)
        affordable = int(free_bytes * 0.5 / per_block_bytes)  # keep 50% head-room
        total_blocks = max(min_blocks, min(affordable, min_blocks * 8)) if affordable else min_blocks

        logger.info(
            "create_paged_cache: %d blocks x %d tokens (block_size=%d) for %d sequences",
            total_blocks, block_size, block_size, max_sequences,
        )
        return PagedKVCacheManager(
            num_blocks=total_blocks,
            block_size=block_size,
            num_heads=n_heads,
            head_dim=head_dim,
            max_sequences=max_sequences,
            device_id=self.device_id,
            **kwargs,
        )


class BlockAllocator:
    """
    Free-list physical block allocator for a paged KV-Cache.

    Blocks are handed out in O(1) from a LIFO free list, mirroring the block
    manager of PagedAttention servers: physical blocks are never required to be
    contiguous, so a long conversation no longer reserves ``max_seq_len`` of
    memory up-front.
    """

    def __init__(self, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = int(num_blocks)
        self._free: List[int] = list(range(self.num_blocks - 1, -1, -1))
        self._in_use: set = set()

    # ── Allocation ────────────────────────────────────────────────────────
    def allocate(self, count: int = 1) -> List[int]:
        """
        Allocate ``count`` physical blocks.

        Raises:
            RuntimeError: when the pool is exhausted (callers should evict or
                free a sequence and retry).
        """
        if count <= 0:
            return []
        if count > len(self._free):
            raise RuntimeError(
                f"Paged KV-Cache exhausted: requested {count} block(s), "
                f"only {len(self._free)} free"
            )
        blocks = [self._free.pop() for _ in range(count)]
        self._in_use.update(blocks)
        return blocks

    def free(self, blocks: Iterable[int]) -> None:
        """Return blocks to the free list (idempotent for unknown blocks)."""
        for block in blocks:
            if block in self._in_use:
                self._in_use.discard(block)
                self._free.append(int(block))

    def reset(self) -> None:
        """Return the allocator to its pristine state."""
        self._free = list(range(self.num_blocks - 1, -1, -1))
        self._in_use.clear()

    # ── Introspection ─────────────────────────────────────────────────────
    @property
    def free_block_count(self) -> int:
        return len(self._free)

    @property
    def used_block_count(self) -> int:
        return len(self._in_use)

    @property
    def usage_pct(self) -> float:
        return 100.0 * self.used_block_count / self.num_blocks

    def stats(self) -> Dict[str, float]:
        """Allocator statistics."""
        return {
            "num_blocks": float(self.num_blocks),
            "free_blocks": float(self.free_block_count),
            "used_blocks": float(self.used_block_count),
            "usage_pct": float(self.usage_pct),
        }


class PagedKVCacheManager:
    """
    Paged KV-Cache manager driven by a physical block table.

    Layout
    ------
    ``pool``        : ``[num_blocks, num_heads, block_size, head_dim]`` float32
    ``block_table`` : ``[max_sequences, max_blocks_per_seq]`` int32 (-1 = unmapped)
    ``seq_lens``    : ``[max_sequences]`` int32

    Logical token ``position`` of the sequence living in slot ``s`` maps to::

        physical_block = block_table[s, position // block_size]
        offset_in_block = position % block_size

    so physical blocks may be scattered anywhere in the pool: no fragmentation,
    no ``max_seq_len`` over-allocation, and sequences grow block by block.

    Backends
    --------
    * ``cuda``  - the ``paged_kv_cache_append`` kernel from ``vram_hacker.cu``
      (coalesced writes along ``head_dim``); enabled automatically when the
      compiled extension and a CUDA device are both present. When the extension
      also ships ``fused_paged_kv_cache_scale_append`` (v2.6.1),
      :meth:`append_scaled` performs its scale/truncation *inside* that kernel and
      saves one full global-memory round-trip over the new tokens.
    * ``numpy`` - vectorised scatter with *identical* indexing semantics, used on
      CPU-only installs; the unit tests exercise this path.

    Usage::

        cache = PagedKVCacheManager(num_blocks=64, block_size=16, num_heads=8, head_dim=64)
        cache.allocate_sequence("session-1")
        cache.append("session-1", tokens)          # (num_new_tokens, num_heads, head_dim)
        cache.append_scaled("session-1", tokens, scale=0.125, clamp_limit=8.0)
        kv = cache.gather("session-1")             # (seq_len, num_heads, head_dim)
        cache.free_sequence("session-1")
    """

    def __init__(
        self,
        num_blocks: int = 256,
        block_size: int = 16,
        num_heads: int = 8,
        head_dim: int = 64,
        max_sequences: int = 8,
        max_blocks_per_seq: Optional[int] = None,
        device_id: int = 0,
        use_cuda: bool = True,
    ):
        if num_blocks <= 0 or block_size <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_blocks, block_size, num_heads and head_dim must be positive")

        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.max_sequences = int(max_sequences)
        self.max_blocks_per_seq = int(
            max_blocks_per_seq
            or max(1, self.num_blocks // max(self.max_sequences, 1))
        )

        self._allocator = BlockAllocator(self.num_blocks)
        self._pool = np.zeros(
            (self.num_blocks, self.num_heads, self.block_size, self.head_dim),
            dtype=np.float32,
        )
        self._block_table = np.full(
            (self.max_sequences, self.max_blocks_per_seq), -1, dtype=np.int32
        )
        self._seq_lens = np.zeros(self.max_sequences, dtype=np.int32)

        self._seq_slots: Dict[object, int] = {}               # seq_id -> slot
        self._slot_blocks: Dict[int, List[int]] = {}          # slot -> physical blocks
        self._free_slots: List[int] = list(range(self.max_sequences))

        # Prefix caching (v2.7.0): every physical block carries a reference
        # count. A block is released to the allocator only when the last owner
        # (the prefix registry or a sharing sequence) drops its reference, and
        # any append that would write into a shared block copies it first
        # (copy-on-write), so a prefix is read-only by construction.
        self.block_ref_counts: Dict[int, int] = {}           # physical block -> refs
        self.prefix_blocks: Dict[object, List[int]] = {}     # prefix_id -> blocks
        self._prefix_lengths: Dict[object, int] = {}         # prefix_id -> tokens

        # CUDA backend (optional, transparently disabled when unavailable)
        self.device_id = int(device_id)
        self.backend = "numpy"
        self._device = None
        self._torch_pool = None
        self._torch_block_table = None
        self._torch_seq_lens = None
        self._fused_kernel = False
        if use_cuda:
            self._try_enable_cuda()

    # ── Backend selection ─────────────────────────────────────────────────
    def _try_enable_cuda(self) -> None:
        """Enable the compiled CUDA kernel when possible, else stay on NumPy."""
        if _CUDA_EXT is None or not hasattr(_CUDA_EXT, "paged_kv_cache_append"):
            logger.info("PagedKV: CUDA extension unavailable, using vectorised NumPy backend")
            return
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            logger.info("PagedKV: no CUDA device, using vectorised NumPy backend")
            return
        try:
            device = torch.device(f"cuda:{self.device_id}")
            self._torch_pool = torch.from_numpy(self._pool).to(device)
            self._torch_block_table = torch.from_numpy(self._block_table).to(device)
            self._torch_seq_lens = torch.from_numpy(self._seq_lens).to(device)
            self._device = device
            self.backend = "cuda"
            self._fused_kernel = hasattr(_CUDA_EXT, "fused_paged_kv_cache_scale_append")
            logger.info(
                "PagedKV: CUDA paged kernel enabled (%d blocks, fused scale+append=%s)",
                self.num_blocks, self._fused_kernel,
            )
        except (RuntimeError, OSError, ValueError) as error:
            logger.warning("PagedKV: CUDA backend init failed (%s), using NumPy", error)
            self.backend = "numpy"

    def _sync_device_block_table(self, slot: int) -> None:
        """Mirror a host block-table row (and the length) to the CUDA backend."""
        if self.backend != "cuda":
            return
        self._torch_block_table[slot].copy_(
            torch.from_numpy(self._block_table[slot]).to(self._device)
        )
        self._torch_seq_lens[slot] = int(self._seq_lens[slot])

    def _sync_host_pool(self) -> None:
        """Pull the device pool / lengths back into the host mirrors."""
        if self.backend != "cuda":
            return
        self._pool[:] = self._torch_pool.detach().cpu().numpy()
        self._seq_lens[:] = self._torch_seq_lens.detach().cpu().numpy()

    def _push_pool_blocks(self, blocks: Iterable[int]) -> None:
        """Mirror the host copy of ``blocks`` back to the CUDA pool."""
        if self.backend != "cuda":
            return
        for block in blocks:
            block = int(block)
            self._torch_pool[block].copy_(
                torch.from_numpy(np.ascontiguousarray(self._pool[block])).to(self._device)
            )

    # ── Reference counting (v2.7.0 prefix caching) ────────────────────────
    def _allocate_blocks(self, count: int) -> List[int]:
        """Allocate physical blocks and initialise their reference count to 1."""
        blocks = [int(block) for block in self._allocator.allocate(count)]
        for block in blocks:
            self.block_ref_counts[block] = 1
        return blocks

    def _reference_block(self, block: int) -> None:
        """Add one owner to a physical block (used when sharing a prefix)."""
        block = int(block)
        self.block_ref_counts[block] = self.block_ref_counts.get(block, 0) + 1

    def _release_block(self, block: int) -> None:
        """
        Drop one owner from a physical block.

        The block only returns to the free list when the last owner lets go
        (``ref_count == 0``), which is what keeps a shared prefix alive while
        any sequence still references it.
        """
        block = int(block)
        remaining = self.block_ref_counts.get(block, 1) - 1
        if remaining > 0:
            self.block_ref_counts[block] = remaining
            return
        self.block_ref_counts.pop(block, None)
        self._allocator.free([block])

    def reference_count(self, physical_block: int) -> int:
        """Number of owners of a physical block (0 when it is free)."""
        return int(self.block_ref_counts.get(int(physical_block), 0))

    @property
    def shared_block_count(self) -> int:
        """Blocks currently referenced by more than one owner (read-only)."""
        return sum(1 for count in self.block_ref_counts.values() if count > 1)

    @property
    def prefix_ids(self) -> List[object]:
        """Ids of the registered shared prefixes."""
        return list(self.prefix_blocks.keys())

    # ── Prefix registry (v2.7.0) ──────────────────────────────────────────
    def register_prefix(self, prefix_id: object, tokens) -> List[int]:
        """
        Register a shared, read-only prefix (v2.7.0).

        The tokens are written into freshly allocated physical blocks exactly
        once; every :meth:`allocate_sequence` naming this prefix then *shares*
        those blocks (their reference count is incremented) instead of copying
        them. N conversations that start from the same system prompt / few-shot
        header therefore pay for a single copy of it in VRAM.

        The blocks stay immutable while shared: :meth:`append` performs
        copy-on-write on the block it is about to touch, so a sequence can keep
        growing without corrupting the prefix (or its siblings).

        Args:
            prefix_id: Identifier for the prefix (re-registering frees the old
                one first).
            tokens: ``(num_tokens, num_heads, head_dim)`` float32 array/tensor.

        Returns:
            The physical blocks backing the prefix, in logical order.
        """
        tokens_array = self._prepare_tokens(tokens)

        if prefix_id in self.prefix_blocks:
            self.free_prefix(prefix_id)

        count = int(tokens_array.shape[0])
        num_blocks = (count + self.block_size - 1) // self.block_size
        if num_blocks == 0:
            self.prefix_blocks[prefix_id] = []
            self._prefix_lengths[prefix_id] = 0
            return []

        blocks = self._allocate_blocks(num_blocks)
        positions = np.arange(count)
        logical = positions // self.block_size
        offsets = positions % self.block_size
        physical = np.asarray(blocks, dtype=np.int64)[logical]
        self._pool[physical, :, offsets, :] = tokens_array
        self._push_pool_blocks(blocks)

        self.prefix_blocks[prefix_id] = blocks
        self._prefix_lengths[prefix_id] = count
        logger.debug(
            "PagedKV: registered prefix '%s' (%d tokens, %d blocks)",
            prefix_id, count, num_blocks,
        )
        return list(blocks)

    def free_prefix(self, prefix_id: object) -> None:
        """
        Drop the registry's reference to a prefix's blocks.

        Sequences still sharing the prefix keep their own references, so the
        physical blocks survive until the last one is freed as well.
        """
        blocks = self.prefix_blocks.pop(prefix_id, None)
        if blocks is None:
            return
        self._prefix_lengths.pop(prefix_id, None)
        for block in blocks:
            self._release_block(block)

    def _attach_prefix(self, slot: int, prefix_id: object) -> None:
        """Point a freshly allocated sequence slot at a registered prefix."""
        if prefix_id not in self.prefix_blocks:
            raise KeyError(
                f"Unknown prefix '{prefix_id}'; call register_prefix() first"
            )
        blocks = list(self.prefix_blocks[prefix_id])
        for block in blocks:
            self._reference_block(block)
        self._slot_blocks[slot] = blocks
        self._block_table[slot, :] = -1
        if blocks:
            self._block_table[slot, :len(blocks)] = np.asarray(blocks, dtype=np.int32)
        self._seq_lens[slot] = int(self._prefix_lengths.get(prefix_id, 0))
        self._sync_device_block_table(slot)

    def _cow_shared_blocks(self, seq_id: object, start: int, count: int) -> None:
        """
        Copy-on-write every shared block the next append is about to touch.

        A shared block (``ref_count > 1``) belongs to a registered prefix; the
        pending write would silently rewrite the other owners' context, so the
        block is cloned into a private one and the sequence's block table is
        re-pointed at the clone. Only the *last* (partially filled) prefix block
        is ever affected, so an append that starts on a block boundary clones
        nothing.
        """
        if count <= 0:
            return
        slot = self._seq_slots[seq_id]
        first = start // self.block_size
        last = (start + count - 1) // self.block_size
        cloned: List[int] = []
        for logical in range(first, last + 1):
            physical = int(self._block_table[slot, logical])
            if physical < 0 or self.block_ref_counts.get(physical, 1) <= 1:
                continue
            self._sync_host_pool()
            replacement = self._allocate_blocks(1)[0]
            self._pool[replacement] = self._pool[physical]
            self._block_table[slot, logical] = replacement
            self._slot_blocks[slot][logical] = replacement
            self._release_block(physical)
            cloned.append(replacement)
        if cloned:
            self._push_pool_blocks(cloned)
            self._sync_device_block_table(slot)
            logger.debug(
                "PagedKV: copy-on-write for '%s' cloned %d shared block(s)",
                seq_id, len(cloned),
            )

    # ── Introspection ─────────────────────────────────────────────────────
    @property
    def is_cuda(self) -> bool:
        """True when the compiled CUDA kernel drives the pool."""
        return self.backend == "cuda"

    @property
    def has_fused_kernel(self) -> bool:
        """True when the fused scale+append kernel drives :meth:`append_scaled`."""
        return bool(self._fused_kernel)

    @property
    def pool(self) -> np.ndarray:
        """Host view of the physical block pool (synced from device when needed)."""
        self._sync_host_pool()
        return self._pool

    @property
    def block_table(self) -> np.ndarray:
        """The full ``[max_sequences, max_blocks_per_seq]`` block table."""
        return self._block_table

    @property
    def seq_lens(self) -> np.ndarray:
        """Current logical length of every slot."""
        return self._seq_lens

    @property
    def sequence_ids(self) -> List[object]:
        """Registered sequence ids."""
        return list(self._seq_slots.keys())

    @property
    def free_sequence_slots(self) -> int:
        """Number of unused sequence slots."""
        return len(self._free_slots)

    @property
    def pool_bytes(self) -> int:
        """Size of the physical pool in bytes."""
        return int(self._pool.nbytes)

    def stats(self) -> Dict[str, float]:
        """Manager statistics (blocks, sequences, backend)."""
        stats = self._allocator.stats()
        stats.update({
            "block_size": float(self.block_size),
            "num_heads": float(self.num_heads),
            "head_dim": float(self.head_dim),
            "max_sequences": float(self.max_sequences),
            "active_sequences": float(len(self._seq_slots)),
            "pool_bytes": float(self.pool_bytes),
            "backend_cuda": 1.0 if self.is_cuda else 0.0,
            "fused_scale_append": 1.0 if self._fused_kernel else 0.0,
            # Prefix caching (v2.7.0)
            "prefixes": float(len(self.prefix_blocks)),
            "shared_blocks": float(self.shared_block_count),
        })
        return stats

    # ── Sequence lifecycle ────────────────────────────────────────────────
    def allocate_sequence(self, seq_id: object = None, prefix_id: object = None) -> int:
        """
        Register a sequence and return its slot index.

        Idempotent: registering an already known ``seq_id`` returns its slot.

        Args:
            seq_id: Sequence identifier (auto-generated when omitted).
            prefix_id: Optional registered prefix to share (v2.7.0); the
                sequence starts life holding that prefix's tokens and shares its
                physical blocks by reference instead of copying them.
        """
        if seq_id is None:
            seq_id = f"seq-{len(self._seq_slots)}"
        if seq_id in self._seq_slots:
            return self._seq_slots[seq_id]
        if not self._free_slots:
            raise RuntimeError(
                f"No free sequence slot ({self.max_sequences} in use); "
                "free a sequence before allocating another one"
            )
        slot = self._free_slots.pop(0)
        self._seq_slots[seq_id] = slot
        self._slot_blocks[slot] = []
        self._block_table[slot, :] = -1
        self._seq_lens[slot] = 0
        self._sync_device_block_table(slot)
        if prefix_id is not None:
            self._attach_prefix(slot, prefix_id)
        return slot

    def free_sequence(self, seq_id: object) -> None:
        """
        Release the sequence's reference to every physical block it holds.

        Shared blocks (a registered prefix) are only returned to the allocator
        once the last owner releases them, so freeing one conversation never
        invalidates a prefix another one is still using.
        """
        slot = self._seq_slots.pop(seq_id, None)
        if slot is None:
            return
        for block in self._slot_blocks.pop(slot, []):
            self._release_block(block)
        self._block_table[slot, :] = -1
        self._seq_lens[slot] = 0
        self._free_slots.append(slot)
        self._sync_device_block_table(slot)

    def _require_slot(self, seq_id: object) -> int:
        """Return the slot of ``seq_id``, creating the sequence when needed."""
        if seq_id not in self._seq_slots:
            return self.allocate_sequence(seq_id)
        return self._seq_slots[seq_id]

    def _ensure_capacity(self, seq_id: object, num_new_tokens: int) -> int:
        """
        Make sure ``num_new_tokens`` extra tokens fit, allocating blocks when the
        sequence crosses a block boundary.

        Returns:
            The first token position of the append.
        """
        slot = self._require_slot(seq_id)
        start = int(self._seq_lens[slot])
        required_blocks = (start + num_new_tokens + self.block_size - 1) // self.block_size
        if required_blocks > self.max_blocks_per_seq:
            raise RuntimeError(
                f"Sequence '{seq_id}' would need {required_blocks} blocks, "
                f"exceeding max_blocks_per_seq={self.max_blocks_per_seq}"
            )
        row = self._block_table[slot]
        mapped = int(np.count_nonzero(row >= 0))
        if required_blocks > mapped:
            blocks = self._allocate_blocks(required_blocks - mapped)
            row[mapped:required_blocks] = np.asarray(blocks, dtype=np.int32)
            self._slot_blocks.setdefault(slot, []).extend(blocks)
            self._sync_device_block_table(slot)
        return start

    # ── Writes ────────────────────────────────────────────────────────────
    def _append_numpy(self, slot: int, tokens: np.ndarray, start: int) -> None:
        """Vectorised scatter of ``tokens`` into their physical blocks."""
        count = tokens.shape[0]
        positions = np.arange(start, start + count)
        logical = positions // self.block_size
        offsets = positions % self.block_size
        physical = self._block_table[slot, logical]
        if not np.all(physical >= 0):
            raise RuntimeError("append() reached an unmapped physical block")
        # Advanced indexing puts the token axis first: (count, num_heads, head_dim)
        self._pool[physical, :, offsets, :] = tokens

    def _append_cuda(self, slot: int, tokens: np.ndarray, start: int) -> None:
        """Launch ``paged_kv_cache_append`` for a single sequence (batch = 1)."""
        # (num_new_tokens, num_heads, head_dim) -> [batch=1, heads, tokens, dim]
        batched = np.ascontiguousarray(tokens.transpose(1, 0, 2))[np.newaxis, ...]
        new_kv = torch.from_numpy(batched).to(self._device)
        row = self._torch_block_table[slot:slot + 1]
        lengths = self._torch_seq_lens[slot:slot + 1]
        _CUDA_EXT.paged_kv_cache_append(
            new_kv, self._torch_pool, row, lengths, int(self.block_size)
        )
        # The kernel extends seq_lens in place on the device
        self._seq_lens[slot] = int(self._torch_seq_lens[slot].item())

    def _append_scaled_numpy(
        self,
        slot: int,
        tokens: np.ndarray,
        start: int,
        scale: float,
        clamp_limit: Optional[float],
    ) -> None:
        """Vectorised scale (+ clamp) followed by the paged scatter."""
        scaled = tokens * np.float32(scale)
        if clamp_limit is not None and clamp_limit > 0.0:
            np.clip(scaled, -clamp_limit, clamp_limit, out=scaled)
        self._append_numpy(slot, np.ascontiguousarray(scaled, dtype=np.float32), start)

    def _append_scaled_cuda(
        self,
        slot: int,
        tokens: np.ndarray,
        start: int,
        scale: float,
        clamp_limit: Optional[float],
    ) -> None:
        """Launch the fused scale+append kernel for a single sequence."""
        batched = np.ascontiguousarray(tokens.transpose(1, 0, 2))[np.newaxis, ...]
        new_kv = torch.from_numpy(batched).to(self._device)
        row = self._torch_block_table[slot:slot + 1]
        lengths = self._torch_seq_lens[slot:slot + 1]
        _CUDA_EXT.fused_paged_kv_cache_scale_append(
            new_kv, self._torch_pool, row, lengths, int(self.block_size),
            float(scale), float(clamp_limit or 0.0),
        )
        self._seq_lens[slot] = int(self._torch_seq_lens[slot].item())

    def _prepare_tokens(self, new_kv) -> np.ndarray:
        """
        Validate and normalise a ``(num_new_tokens, num_heads, head_dim)`` block.

        Args:
            new_kv: float32 array / torch tensor of new K/V tokens.

        Returns:
            Contiguous float32 array shaped like the pool's per-token layout.

        Raises:
            ValueError: On a wrong rank or a head layout mismatching the pool.
        """
        tokens = new_kv
        if _TORCH_AVAILABLE and isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()
        tokens = np.ascontiguousarray(tokens, dtype=np.float32)

        if tokens.ndim != 3:
            raise ValueError(
                "new_kv must have shape (num_new_tokens, num_heads, head_dim), "
                f"got {tokens.shape}"
            )
        if tokens.shape[1:] != (self.num_heads, self.head_dim):
            raise ValueError(
                f"new_kv head layout {tokens.shape[1:]} does not match the pool "
                f"({self.num_heads}, {self.head_dim})"
            )
        return tokens

    def append(self, seq_id: object, new_kv) -> int:
        """
        Append new K/V tokens to a sequence.

        Args:
            seq_id: Sequence identifier (auto-registered on first use).
            new_kv: ``(num_new_tokens, num_heads, head_dim)`` float32 array or
                torch tensor.

        Returns:
            The sequence length after the append.
        """
        tokens = self._prepare_tokens(new_kv)
        if tokens.shape[0] == 0:
            slot = self._require_slot(seq_id)
            return int(self._seq_lens[slot])

        start = self._ensure_capacity(seq_id, tokens.shape[0])
        slot = self._seq_slots[seq_id]
        # A prefix-backed sequence owns its blocks read-only: clone the block the
        # append lands in before writing to it (v2.7.0).
        self._cow_shared_blocks(seq_id, start, tokens.shape[0])

        if self.backend == "cuda":
            self._append_cuda(slot, tokens, start)
        else:
            self._append_numpy(slot, tokens, start)
            self._seq_lens[slot] = start + tokens.shape[0]
        return int(self._seq_lens[slot])

    def append_scaled(
        self,
        seq_id: object,
        new_kv,
        scale: float = 1.0,
        clamp_limit: Optional[float] = None,
    ) -> int:
        """
        Append K/V tokens after an in-flight ``scale`` (+ optional truncation).

        Semantically ``append(seq_id, new_kv * scale)``, i.e.::

            out = clamp(new_kv * scale, -clamp_limit, +clamp_limit)

        On the CUDA backend the transform happens inside the append kernel
        (``fused_paged_kv_cache_scale_append_kernel`` in ``vram_hacker.cu``): the
        new tokens are read from global memory exactly once and the scaled values
        are written straight into their physical blocks, so the extra
        read-modify-write pass of a naive "scale then append" pipeline disappears.
        Without the compiled extension (or on CPU) the identical arithmetic is
        performed by a vectorised NumPy path, so both backends stay semantically
        equivalent.

        Args:
            seq_id: Sequence identifier (auto-registered on first use).
            new_kv: ``(num_new_tokens, num_heads, head_dim)`` float32 array or
                torch tensor.
            scale: Multiplicative factor applied to every element (e.g. an
                attention temperature or a RoPE rescaling factor).
            clamp_limit: Optional symmetric magnitude bound (dynamic truncation);
                ``None`` or ``<= 0`` disables it.

        Returns:
            The sequence length after the append.
        """
        scale = float(scale)
        limit: Optional[float] = None if clamp_limit is None else float(clamp_limit)
        tokens = self._prepare_tokens(new_kv)
        if tokens.shape[0] == 0:
            slot = self._require_slot(seq_id)
            return int(self._seq_lens[slot])

        start = self._ensure_capacity(seq_id, tokens.shape[0])
        slot = self._seq_slots[seq_id]
        # Copy-on-write before the fused/numpy transform touches a shared block
        # (v2.7.0 prefix caching).
        self._cow_shared_blocks(seq_id, start, tokens.shape[0])

        if self.backend == "cuda" and self._fused_kernel:
            self._append_scaled_cuda(slot, tokens, start, scale, limit)
        else:
            self._append_scaled_numpy(slot, tokens, start, scale, limit)
            self._seq_lens[slot] = start + tokens.shape[0]
        return int(self._seq_lens[slot])

    # ── Reads ─────────────────────────────────────────────────────────────
    def sequence_length(self, seq_id: object) -> int:
        """Current logical length of ``seq_id`` (0 when unknown)."""
        slot = self._seq_slots.get(seq_id)
        if slot is None:
            return 0
        return int(self._seq_lens[slot])

    def get_block_table(self, seq_id: object) -> np.ndarray:
        """
        Physical block table of one sequence (mapped entries only).

        Returns:
            int32 array of ``ceil(seq_len / block_size)`` physical block ids.
        """
        slot = self._seq_slots.get(seq_id)
        if slot is None:
            return np.zeros(0, dtype=np.int32)
        mapped = int(np.count_nonzero(self._block_table[slot] >= 0))
        return self._block_table[slot, :mapped].copy()

    def gather(self, seq_id: object, upto: Optional[int] = None) -> np.ndarray:
        """
        Materialise the logical K/V tensor of a sequence.

        Args:
            seq_id: Sequence identifier.
            upto: Optional exclusive upper bound on the token position.

        Returns:
            ``(length, num_heads, head_dim)`` float32 array in logical token
            order, reconstructed through the physical block table.
        """
        length = self.sequence_length(seq_id)
        if upto is not None:
            length = min(length, int(upto))
        if length <= 0:
            return np.zeros((0, self.num_heads, self.head_dim), dtype=np.float32)

        slot = self._seq_slots[seq_id]
        positions = np.arange(length)
        logical = positions // self.block_size
        offsets = positions % self.block_size
        physical = self._block_table[slot, logical]
        if not np.all(physical >= 0):
            raise RuntimeError("gather() found an unmapped physical block")
        self._sync_host_pool()
        return np.ascontiguousarray(self._pool[physical, :, offsets, :])

    # ── Reset ─────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Drop every sequence and prefix, returning all blocks to the allocator."""
        self._allocator.reset()
        self._pool[:] = 0.0
        self._block_table[:] = -1
        self._seq_lens[:] = 0
        self._seq_slots.clear()
        self._slot_blocks.clear()
        self.block_ref_counts.clear()
        self.prefix_blocks.clear()
        self._prefix_lengths.clear()
        self._free_slots = list(range(self.max_sequences))
        if self.backend == "cuda":
            self._torch_pool.zero_()
            self._torch_block_table.copy_(
                torch.from_numpy(self._block_table).to(self._device)
            )
            self._torch_seq_lens.zero_()

    @property
    def block_allocator(self) -> BlockAllocator:
        """The underlying free-list block allocator."""
        return self._allocator

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"PagedKVCacheManager(backend={self.backend}, blocks={self.num_blocks}, "
            f"block_size={self.block_size}, active={len(self._seq_slots)}, "
            f"free_blocks={self._allocator.free_block_count})"
        )





