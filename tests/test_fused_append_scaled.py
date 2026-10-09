"""
Tests for the fused paged KV-Cache scale+append path (v2.6.1)
============================================================

Covers ``PagedKVCacheManager.append_scaled`` -- the Python face of the
``fused_paged_kv_cache_scale_append_kernel`` added to ``vram_hacker.cu``:

    1. ``append_scaled(scale=...)`` is arithmetically identical to appending an
       already-scaled block (fusing only removes a memory round-trip)
    2. Dynamic truncation through ``clamp_limit``
    3. Paging semantics: block-table growth, block crossings, ``gather``
       round-trip and multi-sequence isolation
    4. Backend transparency: the vectorised NumPy fallback mirrors the CUDA path
       (these tests run on CPU-only machines)
    5. Validation/error handling shared with ``append``
"""

import numpy as np
import pytest

from vram_core.vram_optimizer import PagedKVCacheManager


def make_tokens(num_tokens: int, num_heads: int = 2, head_dim: int = 3, seed: int = 0):
    """Random float32 token block shaped (num_tokens, num_heads, head_dim)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_tokens, num_heads, head_dim)).astype(np.float32)


def make_cache(**kwargs) -> PagedKVCacheManager:
    """Small CPU cache with per-test overrides."""
    params = dict(num_blocks=16, block_size=4, num_heads=2, head_dim=3, max_sequences=4)
    params.update(kwargs)
    return PagedKVCacheManager(**params)


class TestAppendScaledArithmetic:
    """The fused path must match an explicit scale-then-append pipeline."""

    def test_scale_matches_pre_scaled_append(self):
        """``append_scaled(x, s)`` == ``append(x * s)``."""
        tokens = make_tokens(4, seed=1)
        fused = make_cache()
        reference = make_cache()

        fused.append_scaled("s1", tokens, scale=0.5)
        reference.append("s1", tokens * np.float32(0.5))

        np.testing.assert_allclose(fused.gather("s1"), reference.gather("s1"), atol=1e-6)

    def test_default_scale_is_an_identity(self):
        """``scale=1.0`` and no clamp reproduce ``append`` exactly."""
        tokens = make_tokens(3, seed=2)
        fused = make_cache()
        reference = make_cache()

        fused.append_scaled("s1", tokens)
        reference.append("s1", tokens)

        np.testing.assert_allclose(fused.gather("s1"), reference.gather("s1"), atol=1e-6)

    def test_negative_and_fractional_scales(self):
        """Scale factors are applied element-wise, sign included."""
        tokens = make_tokens(2, seed=3)
        cache = make_cache()
        cache.append_scaled("s1", tokens, scale=-2.5)
        np.testing.assert_allclose(
            cache.gather("s1"), (tokens * np.float32(-2.5)), atol=1e-5
        )

    def test_clamp_limit_truncates_the_magnitude(self):
        """``clamp_limit`` bounds every scaled value symmetrically."""
        tokens = make_tokens(4, seed=4) * 10.0
        cache = make_cache()
        cache.append_scaled("s1", tokens, scale=5.0, clamp_limit=1.0)

        gathered = cache.gather("s1")
        assert float(np.max(np.abs(gathered))) <= 1.0 + 1e-6

    def test_non_positive_clamp_is_disabled(self):
        """``clamp_limit <= 0`` behaves like no truncation at all."""
        tokens = make_tokens(4, seed=5)
        cache = make_cache()
        cache.append_scaled("s1", tokens, scale=3.0, clamp_limit=0.0)
        np.testing.assert_allclose(
            cache.gather("s1"), (tokens * np.float32(3.0)), atol=1e-5
        )

    def test_returns_the_resulting_sequence_length(self):
        """The return value mirrors ``append`` (length after the write)."""
        cache = make_cache()
        assert cache.append_scaled("s1", make_tokens(3, seed=6)) == 3
        assert cache.append_scaled("s1", make_tokens(2, seed=7), scale=0.5) == 5
        assert cache.sequence_length("s1") == 5


class TestAppendScaledPaging:
    """Physical paging must be unaffected by the fused transformation."""

    def test_blocks_are_allocated_as_the_sequence_grows(self):
        """Crossing a block boundary allocates another physical block."""
        cache = make_cache(block_size=4)
        cache.append_scaled("s1", make_tokens(3, seed=8))
        assert cache.get_block_table("s1").size == 1

        cache.append_scaled("s1", make_tokens(3, seed=9), scale=0.5)
        assert cache.get_block_table("s1").size == 2
        assert cache.gather("s1").shape == (6, 2, 3)

    def test_values_are_reconstructed_through_the_block_table(self):
        """Scattered blocks read back in logical token order."""
        cache = make_cache(block_size=2, num_blocks=16)
        first = make_tokens(2, seed=10)
        second = make_tokens(2, seed=11)
        cache.append_scaled("s1", first, scale=0.5)
        cache.append_scaled("s1", second, scale=2.0)

        gathered = cache.gather("s1")
        np.testing.assert_allclose(gathered[:2], first * np.float32(0.5), atol=1e-5)
        np.testing.assert_allclose(gathered[2:], second * np.float32(2.0), atol=1e-5)

    def test_sequences_stay_isolated(self):
        """Two sequences keep their own scaled content."""
        cache = make_cache()
        tokens_a = make_tokens(2, seed=12)
        tokens_b = make_tokens(2, seed=13)
        cache.append_scaled("a", tokens_a, scale=0.25)
        cache.append_scaled("b", tokens_b, scale=4.0)

        np.testing.assert_allclose(
            cache.gather("a"), tokens_a * np.float32(0.25), atol=1e-5
        )
        np.testing.assert_allclose(
            cache.gather("b"), tokens_b * np.float32(4.0), atol=1e-5
        )
        assert cache.sequence_length("a") == cache.sequence_length("b") == 2

    def test_empty_block_is_a_noop(self):
        """A zero-length append registers the sequence but writes nothing."""
        cache = make_cache()
        empty = np.zeros((0, 2, 3), dtype=np.float32)
        assert cache.append_scaled("s1", empty, scale=2.0) == 0
        assert cache.sequence_length("s1") == 0
        assert "s1" in cache.sequence_ids

    def test_capacity_overflow_still_raises(self):
        """The block budget guard is shared with ``append``."""
        cache = make_cache(num_blocks=2, block_size=2, max_sequences=1)
        cache.append_scaled("s1", make_tokens(4, seed=14))
        with pytest.raises(RuntimeError):
            cache.append_scaled("s1", make_tokens(2, seed=15))


class TestAppendScaledValidation:
    """Validation mirrors ``append`` exactly."""

    def test_wrong_rank_raises(self):
        """A 2-D block is rejected."""
        cache = make_cache()
        with pytest.raises(ValueError):
            cache.append_scaled("s1", np.zeros((4, 3), dtype=np.float32))

    def test_wrong_head_layout_raises(self):
        """A block whose heads/dims do not match the pool is rejected."""
        cache = make_cache()
        with pytest.raises(ValueError):
            cache.append_scaled("s1", make_tokens(2, num_heads=1, head_dim=3))

    def test_torch_tensor_input_is_accepted(self):
        """Torch tensors are converted exactly like ``append`` does."""
        torch = pytest.importorskip("torch")
        cache = make_cache()
        tokens = make_tokens(2, seed=16)
        cache.append_scaled("s1", torch.from_numpy(tokens), scale=0.5)
        np.testing.assert_allclose(
            cache.gather("s1"), tokens * np.float32(0.5), atol=1e-6
        )


class TestAppendScaledBackend:
    """Backend reporting for the fused kernel."""

    def test_cpu_cache_reports_the_numpy_backend(self):
        """Without the compiled extension the manager stays on NumPy."""
        cache = make_cache(use_cuda=False)
        assert cache.backend == "numpy"
        assert cache.has_fused_kernel is False

    def test_stats_expose_the_fused_flag(self):
        """``stats()`` documents whether the fused kernel is active."""
        stats = make_cache(use_cuda=False).stats()
        assert stats["fused_scale_append"] == 0.0
        assert stats["backend_cuda"] == 0.0
