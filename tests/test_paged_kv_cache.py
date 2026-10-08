"""
Tests for the paged KV-Cache subsystem (v2.6.0).

Covers:
    - BlockAllocator: free-list allocation, release, exhaustion, reset, stats
    - PagedKVCacheManager: physical paging, block-table growth, vectorised
      writes, gather/read-back, sequence lifecycle, error handling, backends
    - VRAMOptimizer.create_paged_cache sizing helper

The suite runs against the NumPy backend (always available) and exercises the
exact indexing semantics implemented by ``paged_kv_cache_append_kernel`` in
``vram_hacker.cu``: token -> (physical_block, offset_in_block) -> coalesced
write along ``head_dim``. The same tests therefore validate the paging logic on
CPU-only machines; the CUDA path is exercised additionally when a device and the
compiled extension are present.
"""

import numpy as np
import pytest
from unittest.mock import patch

from vram_core.vram_optimizer import (
    BlockAllocator,
    MemoryPressure,
    PagedKVCacheManager,
    VRAMOptimizer,
    VRAMStatus,
)


def make_tokens(num_tokens: int, num_heads: int = 2, head_dim: int = 3, seed: int = 0):
    """Random float32 token block shaped (num_tokens, num_heads, head_dim)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_tokens, num_heads, head_dim)).astype(np.float32)


# ─── BlockAllocator ─────────────────────────────────────────────────────────

class TestBlockAllocator:
    """Physical block free-list behaviour."""

    def test_allocate_returns_unique_blocks(self):
        allocator = BlockAllocator(4)
        blocks = allocator.allocate(2)
        assert len(blocks) == 2
        assert len(set(blocks)) == 2
        assert allocator.used_block_count == 2
        assert allocator.free_block_count == 2

    def test_free_returns_blocks_to_the_pool(self):
        allocator = BlockAllocator(3)
        blocks = allocator.allocate(3)
        allocator.free(blocks)
        assert allocator.free_block_count == 3
        assert allocator.used_block_count == 0
        # Freed blocks become available again
        assert sorted(allocator.allocate(3)) == sorted(blocks)

    def test_exhaustion_raises(self):
        allocator = BlockAllocator(2)
        allocator.allocate(2)
        with pytest.raises(RuntimeError):
            allocator.allocate(1)

    def test_allocate_zero_returns_empty(self):
        allocator = BlockAllocator(2)
        assert allocator.allocate(0) == []

    def test_free_unknown_block_is_noop(self):
        allocator = BlockAllocator(2)
        allocator.free([999])
        assert allocator.free_block_count == 2

    def test_double_free_is_idempotent(self):
        allocator = BlockAllocator(2)
        blocks = allocator.allocate(1)
        allocator.free(blocks)
        allocator.free(blocks)
        assert allocator.free_block_count == 2

    def test_reset_restores_full_pool(self):
        allocator = BlockAllocator(5)
        allocator.allocate(4)
        allocator.reset()
        assert allocator.free_block_count == 5
        assert allocator.used_block_count == 0

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            BlockAllocator(0)

    def test_stats_keys_and_usage(self):
        allocator = BlockAllocator(4)
        allocator.allocate(1)
        stats = allocator.stats()
        assert stats["num_blocks"] == 4.0
        assert stats["used_blocks"] == 1.0
        assert stats["usage_pct"] == pytest.approx(25.0)


# ─── Manager: construction ──────────────────────────────────────────────────

class TestPagedKVCacheInit:
    """Construction, defaults and backend selection."""

    def test_defaults(self):
        cache = PagedKVCacheManager(num_blocks=32, block_size=8, num_heads=2, head_dim=4)
        assert cache.num_blocks == 32
        assert cache.block_size == 8
        assert cache.max_blocks_per_seq == 4          # 32 blocks / 8 sequences
        assert cache.block_table.shape == (8, 4)
        assert cache.pool.shape == (32, 2, 8, 4)
        assert cache.pool.dtype == np.float32

    def test_backend_is_known(self):
        cache = PagedKVCacheManager(num_blocks=8, block_size=4, num_heads=1, head_dim=2)
        assert cache.backend in ("numpy", "cuda")
        assert cache.is_cuda is (cache.backend == "cuda")

    def test_numpy_backend_when_cuda_disabled(self):
        cache = PagedKVCacheManager(
            num_blocks=8, block_size=4, num_heads=1, head_dim=2, use_cuda=False
        )
        assert cache.backend == "numpy"
        assert cache.is_cuda is False

    def test_invalid_dimensions_raise(self):
        with pytest.raises(ValueError):
            PagedKVCacheManager(num_blocks=0)
        with pytest.raises(ValueError):
            PagedKVCacheManager(num_blocks=4, block_size=0)
        with pytest.raises(ValueError):
            PagedKVCacheManager(num_blocks=4, num_heads=0)
        with pytest.raises(ValueError):
            PagedKVCacheManager(num_blocks=4, head_dim=0)

    def test_pool_starts_zeroed(self):
        cache = PagedKVCacheManager(num_blocks=4, block_size=2, num_heads=1, head_dim=2)
        assert np.all(cache.pool == 0.0)
        assert cache.pool_bytes == 4 * 1 * 2 * 2 * 4
        assert cache.sequence_ids == []
        assert cache.free_sequence_slots == cache.max_sequences

    def test_stats_shape(self):
        cache = PagedKVCacheManager(num_blocks=16, block_size=4, num_heads=2, head_dim=2)
        stats = cache.stats()
        for key in ("num_blocks", "free_blocks", "used_blocks", "block_size",
                    "active_sequences", "pool_bytes", "backend_cuda"):
            assert key in stats
        assert stats["free_blocks"] == 16.0

    def test_repr_mentions_backend(self):
        cache = PagedKVCacheManager(num_blocks=4, block_size=2, num_heads=1, head_dim=2)
        assert "PagedKVCacheManager" in repr(cache)
        assert cache.backend in repr(cache)


# ─── Manager: writes / reads ────────────────────────────────────────────────

class TestPagedKVCacheWrite:
    """Token append, block-table growth and read-back."""

    def test_append_records_length(self):
        # 8 blocks / 4 sequences => 2 blocks per sequence (enough for a 5-token seq)
        cache = PagedKVCacheManager(
            num_blocks=8, block_size=4, num_heads=2, head_dim=3, max_sequences=4,
        )
        assert cache.append("s1", make_tokens(3)) == 3
        assert cache.append("s1", make_tokens(2, seed=1)) == 5
        assert cache.sequence_length("s1") == 5

    def test_append_and_gather_roundtrip_across_blocks(self):
        cache = PagedKVCacheManager(num_blocks=16, block_size=4, num_heads=2, head_dim=3)
        first = make_tokens(3, seed=1)
        second = make_tokens(5, seed=2)      # crosses two block boundaries
        cache.append("s1", first)
        cache.append("s1", second)

        gathered = cache.gather("s1")
        expected = np.concatenate([first, second], axis=0)
        assert gathered.shape == expected.shape
        np.testing.assert_allclose(gathered, expected, rtol=1e-6, atol=1e-6)

    def test_block_table_grows_only_on_boundary(self):
        cache = PagedKVCacheManager(num_blocks=16, block_size=4, num_heads=1, head_dim=2)
        assert cache.append("s1", make_tokens(3, num_heads=1, head_dim=2)) == 3
        assert len(cache.get_block_table("s1")) == 1
        cache.append("s1", make_tokens(1, num_heads=1, head_dim=2, seed=1))
        assert len(cache.get_block_table("s1")) == 1     # exactly filled the block
        cache.append("s1", make_tokens(1, num_heads=1, head_dim=2, seed=2))
        assert len(cache.get_block_table("s1")) == 2     # spilled into a new block

    def test_physical_blocks_are_distinct_per_sequence(self):
        cache = PagedKVCacheManager(num_blocks=16, block_size=4, num_heads=1, head_dim=2)
        cache.append("a", make_tokens(6, num_heads=1, head_dim=2))
        cache.append("b", make_tokens(6, num_heads=1, head_dim=2, seed=3))
        blocks_a = set(cache.get_block_table("a").tolist())
        blocks_b = set(cache.get_block_table("b").tolist())
        assert blocks_a and blocks_b
        assert blocks_a.isdisjoint(blocks_b)

    def test_unmapped_block_table_entries_stay_minus_one(self):
        cache = PagedKVCacheManager(num_blocks=16, block_size=4, num_heads=1, head_dim=2)
        cache.append("s1", make_tokens(2, num_heads=1, head_dim=2))
        row = cache.block_table[cache._seq_slots["s1"]]
        assert row[0] >= 0
        assert np.all(row[1:] == -1)

    def test_gather_with_upper_bound(self):
        # 8 blocks / 4 sequences => 2 blocks per sequence (a 6-token seq spans two)
        cache = PagedKVCacheManager(
            num_blocks=8, block_size=4, num_heads=1, head_dim=2, max_sequences=4,
        )
        tokens = make_tokens(6, num_heads=1, head_dim=2)
        cache.append("s1", tokens)
        partial = cache.gather("s1", upto=4)
        assert partial.shape == (4, 1, 2)
        np.testing.assert_allclose(partial, tokens[:4], rtol=1e-6, atol=1e-6)

    def test_gather_unknown_sequence_is_empty(self):
        cache = PagedKVCacheManager(num_blocks=4, block_size=2, num_heads=1, head_dim=2)
        assert cache.gather("ghost").shape == (0, 1, 2)
        assert cache.sequence_length("ghost") == 0
        assert cache.get_block_table("ghost").shape == (0,)

    def test_empty_append_keeps_length(self):
        cache = PagedKVCacheManager(num_blocks=4, block_size=2, num_heads=1, head_dim=2)
        cache.append("s1", make_tokens(2, num_heads=1, head_dim=2))
        assert cache.append("s1", np.zeros((0, 1, 2), dtype=np.float32)) == 2

    def test_wrong_rank_raises(self):
        cache = PagedKVCacheManager(num_blocks=4, block_size=2, num_heads=1, head_dim=2)
        with pytest.raises(ValueError):
            cache.append("s1", np.zeros((2, 2), dtype=np.float32))

    def test_wrong_head_layout_raises(self):
        cache = PagedKVCacheManager(num_blocks=4, block_size=2, num_heads=2, head_dim=2)
        with pytest.raises(ValueError):
            cache.append("s1", np.zeros((2, 3, 2), dtype=np.float32))

    def test_max_blocks_per_sequence_enforced(self):
        cache = PagedKVCacheManager(
            num_blocks=8, block_size=2, num_heads=1, head_dim=2,
            max_sequences=2, max_blocks_per_seq=2,
        )
        cache.append("s1", make_tokens(4, num_heads=1, head_dim=2))
        with pytest.raises(RuntimeError):
            cache.append("s1", make_tokens(1, num_heads=1, head_dim=2))

    def test_slot_exhaustion_raises(self):
        cache = PagedKVCacheManager(
            num_blocks=8, block_size=2, num_heads=1, head_dim=2, max_sequences=1
        )
        cache.allocate_sequence("s1")
        with pytest.raises(RuntimeError):
            cache.allocate_sequence("s2")

    def test_pool_exhaustion_raises(self):
        cache = PagedKVCacheManager(
            num_blocks=2, block_size=2, num_heads=1, head_dim=2,
            max_sequences=1, max_blocks_per_seq=4,
        )
        cache.append("s1", make_tokens(4, num_heads=1, head_dim=2))
        with pytest.raises(RuntimeError):
            cache.append("s1", make_tokens(2, num_heads=1, head_dim=2))

    def test_torch_tensor_input(self):
        torch = pytest.importorskip("torch")
        cache = PagedKVCacheManager(num_blocks=8, block_size=4, num_heads=2, head_dim=3)
        tokens = torch.randn(3, 2, 3)
        assert cache.append("s1", tokens) == 3
        np.testing.assert_allclose(
            cache.gather("s1"), tokens.numpy(), rtol=1e-5, atol=1e-5
        )


# ─── Manager: lifecycle ─────────────────────────────────────────────────────

class TestPagedKVCacheLifecycle:
    """Sequence registration / release / reuse."""

    def test_free_sequence_returns_blocks(self):
        # 8 blocks / 4 sequences => 2 blocks per sequence (an 8-token seq spans two)
        cache = PagedKVCacheManager(
            num_blocks=8, block_size=4, num_heads=1, head_dim=2, max_sequences=4,
        )
        cache.append("s1", make_tokens(8, num_heads=1, head_dim=2))
        assert cache.stats()["free_blocks"] == 6.0
        cache.free_sequence("s1")
        assert cache.stats()["free_blocks"] == 8.0
        assert cache.sequence_length("s1") == 0
        assert cache.get_block_table("s1").shape == (0,)

    def test_freed_blocks_are_reused(self):
        cache = PagedKVCacheManager(num_blocks=4, block_size=4, num_heads=1, head_dim=2)
        cache.append("s1", make_tokens(4, num_heads=1, head_dim=2))
        first_blocks = cache.get_block_table("s1").tolist()
        cache.free_sequence("s1")
        cache.append("s2", make_tokens(4, num_heads=1, head_dim=2, seed=5))
        assert cache.get_block_table("s2").tolist() == first_blocks

    def test_allocate_sequence_is_idempotent(self):
        cache = PagedKVCacheManager(num_blocks=4, block_size=2, num_heads=1, head_dim=2)
        slot_a = cache.allocate_sequence("s1")
        slot_b = cache.allocate_sequence("s1")
        assert slot_a == slot_b
        assert cache.free_sequence_slots == cache.max_sequences - 1

    def test_free_unknown_sequence_is_noop(self):
        cache = PagedKVCacheManager(num_blocks=4, block_size=2, num_heads=1, head_dim=2)
        cache.free_sequence("ghost")
        assert cache.stats()["free_blocks"] == 4.0

    def test_reset_clears_pool_and_sequences(self):
        # 8 blocks / 4 sequences => 2 blocks per sequence (a 6-token seq spans two)
        cache = PagedKVCacheManager(
            num_blocks=8, block_size=4, num_heads=1, head_dim=2, max_sequences=4,
        )
        cache.append("s1", make_tokens(6, num_heads=1, head_dim=2))
        cache.reset()
        assert cache.sequence_ids == []
        assert cache.stats()["free_blocks"] == 8.0
        assert np.all(cache.pool == 0.0)
        assert np.all(cache.block_table == -1)
        assert np.all(cache.seq_lens == 0)

    def test_append_auto_registers_sequence(self):
        cache = PagedKVCacheManager(num_blocks=4, block_size=2, num_heads=1, head_dim=2)
        cache.append("auto", make_tokens(2, num_heads=1, head_dim=2))
        assert "auto" in cache.sequence_ids

    def test_block_allocator_is_exposed(self):
        cache = PagedKVCacheManager(num_blocks=4, block_size=2, num_heads=1, head_dim=2)
        assert isinstance(cache.block_allocator, BlockAllocator)


# ─── VRAMOptimizer integration ──────────────────────────────────────────────

class TestVRAMOptimizerPagedCache:
    """The sizing helper must never over-commit the reported free VRAM."""

    def test_create_paged_cache_uses_free_vram(self):
        optimizer = VRAMOptimizer()
        status = VRAMStatus(
            device_id=0, gpu_name="Test", total_mb=24000,
            used_mb=4000, free_mb=20000, usage_pct=16.7,
            pressure=MemoryPressure.LOW,
        )
        with patch.object(optimizer, "get_status", return_value=status):
            cache = optimizer.create_paged_cache(
                n_heads=8, head_dim=64, seq_length=512, block_size=16,
                max_sequences=4,
            )
        assert isinstance(cache, PagedKVCacheManager)
        assert cache.num_blocks >= 4 * 32          # 4 sequences x ceil(512/16)
        assert cache.stats()["free_blocks"] == float(cache.num_blocks)
        assert cache.pool_bytes <= 20000 * 1024 * 1024

    def test_create_paged_cache_without_gpu_still_returns_manager(self):
        optimizer = VRAMOptimizer()
        status = VRAMStatus(
            device_id=0, gpu_name="No GPU", total_mb=0, used_mb=0, free_mb=0,
            usage_pct=0.0, pressure=MemoryPressure.LOW,
        )
        with patch.object(optimizer, "get_status", return_value=status):
            cache = optimizer.create_paged_cache(
                seq_length=64, block_size=16, max_sequences=2
            )
        assert cache.num_blocks >= 2 * 4
        assert cache.backend in ("numpy", "cuda")


