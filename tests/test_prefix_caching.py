"""
Tests for Paged KV-Cache prefix caching (v2.7.0).

Covers the ref-counted, shared, read-only prefix blocks layered on top of
``PagedKVCacheManager``:

    - ``register_prefix``: one physical copy of a shared prompt / few-shot header
    - ``allocate_sequence(..., prefix_id=...)``: sequences *share* those blocks
      (reference count + 1) instead of copying them
    - copy-on-write: appending into a partially filled shared block clones it,
      so the prefix (and every sibling that still shares it) stays pristine
    - refcount-aware ``free_sequence`` / ``free_prefix``: blocks only return to
      the allocator at ``ref_count == 0``
    - statistics / reset / error handling

The suite runs against the vectorised NumPy backend (always available); the CUDA
backend executes the identical paging logic when the compiled extension and a
device are present.
"""

import numpy as np
import pytest

from vram_core.vram_optimizer import PagedKVCacheManager


def make_tokens(num_tokens: int, num_heads: int = 1, head_dim: int = 2, seed: int = 0):
    """Random float32 token block shaped (num_tokens, num_heads, head_dim)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_tokens, num_heads, head_dim)).astype(np.float32)


def as_pool(tokens: np.ndarray) -> np.ndarray:
    """(tokens, heads, dim) -> the pool's per-block (heads, tokens, dim) layout."""
    return np.transpose(tokens, (1, 0, 2))


def make_cache(num_blocks: int = 16, block_size: int = 4, max_sequences: int = 4):
    return PagedKVCacheManager(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=1,
        head_dim=2,
        max_sequences=max_sequences,
    )


# ─── Registration ───────────────────────────────────────────────────────────

class TestRegisterPrefix:
    """A prefix is written once and then shared by reference."""

    def test_register_writes_the_tokens_into_physical_blocks(self):
        cache = make_cache()
        tokens = make_tokens(4, seed=1)
        blocks = cache.register_prefix("sys", tokens)
        assert len(blocks) == 1
        assert cache.prefix_ids == ["sys"]
        np.testing.assert_allclose(cache.pool[blocks[0]], as_pool(tokens), rtol=1e-6, atol=1e-6)

    def test_prefix_spans_multiple_blocks_in_logical_order(self):
        cache = make_cache()
        tokens = make_tokens(6, seed=2)          # 6 tokens / block_size 4 -> 2 blocks
        blocks = cache.register_prefix("sys", tokens)
        assert len(blocks) == 2
        np.testing.assert_allclose(
            cache.pool[blocks[0]][:, :4, :], as_pool(tokens[:4]), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            cache.pool[blocks[1]][:, :2, :], as_pool(tokens[4:]), rtol=1e-6, atol=1e-6
        )

    def test_registration_holds_one_reference_per_block(self):
        cache = make_cache()
        blocks = cache.register_prefix("sys", make_tokens(6, seed=3))
        assert [cache.reference_count(block) for block in blocks] == [1, 1]

    def test_empty_prefix_is_allowed(self):
        cache = make_cache()
        assert cache.register_prefix("empty", np.zeros((0, 1, 2), dtype=np.float32)) == []
        assert cache.prefix_ids == ["empty"]

    def test_re_registering_releases_the_previous_blocks(self):
        cache = make_cache()
        cache.register_prefix("sys", make_tokens(4, seed=4))
        assert cache.stats()["used_blocks"] == 1.0

        cache.register_prefix("sys", make_tokens(4, seed=5))
        # the old block went back to the free list (and is likely reused)
        assert cache.prefix_ids == ["sys"]
        assert cache.stats()["used_blocks"] == 1.0
        assert cache.reference_count(cache.prefix_blocks["sys"][0]) == 1

    def test_prefix_rejects_wrong_head_layout(self):
        cache = make_cache()
        with pytest.raises(ValueError):
            cache.register_prefix("bad", make_tokens(4, num_heads=2, head_dim=2))


# ─── Sharing ────────────────────────────────────────────────────────────────

class TestPrefixSharing:
    """Sequences attach to a prefix without duplicating its blocks."""

    def test_sequence_starts_with_the_prefix_content(self):
        cache = make_cache()
        tokens = make_tokens(6, seed=6)
        cache.register_prefix("sys", tokens)
        cache.allocate_sequence("a", prefix_id="sys")
        assert cache.sequence_length("a") == 6
        np.testing.assert_allclose(cache.gather("a"), tokens, rtol=1e-6, atol=1e-6)

    def test_blocks_are_shared_not_copied(self):
        cache = make_cache()
        prefix_blocks = cache.register_prefix("sys", make_tokens(6, seed=7))
        cache.allocate_sequence("a", prefix_id="sys")
        cache.allocate_sequence("b", prefix_id="sys")
        assert cache.get_block_table("a").tolist() == prefix_blocks
        assert cache.get_block_table("b").tolist() == prefix_blocks
        # 2 shared prefix blocks only, not 2 + 2 + 2
        assert cache.stats()["used_blocks"] == 2.0
        assert [cache.reference_count(block) for block in prefix_blocks] == [3, 3]

    def test_shared_block_count_reports_shared_blocks(self):
        cache = make_cache()
        cache.register_prefix("sys", make_tokens(6, seed=8))
        assert cache.shared_block_count == 0            # only the registry holds it
        cache.allocate_sequence("a", prefix_id="sys")
        assert cache.shared_block_count == 2
        assert cache.stats()["shared_blocks"] == 2.0

    def test_unknown_prefix_raises(self):
        cache = make_cache()
        with pytest.raises(KeyError):
            cache.allocate_sequence("a", prefix_id="missing")

    def test_prefixes_are_listed_in_stats(self):
        cache = make_cache()
        cache.register_prefix("a", make_tokens(4, seed=9))
        cache.register_prefix("b", make_tokens(4, seed=10))
        assert cache.stats()["prefixes"] == 2.0


# ─── Copy-on-write ──────────────────────────────────────────────────────────

class TestCopyOnWrite:
    """Appending into a shared block clones it before writing."""

    def test_append_clones_the_shared_tail_block(self):
        cache = make_cache()
        prefix = make_tokens(6, seed=11)                 # 2 blocks, tail half full
        prefix_blocks = cache.register_prefix("sys", prefix)
        cache.allocate_sequence("a", prefix_id="sys")

        extra = make_tokens(2, seed=12)
        assert cache.append("a", extra) == 8

        table = cache.get_block_table("a").tolist()
        assert table[0] == prefix_blocks[0]              # untouched head is still shared
        assert table[1] != prefix_blocks[1]              # tail was cloned
        expected = np.concatenate([prefix, extra], axis=0)
        np.testing.assert_allclose(cache.gather("a"), expected, rtol=1e-6, atol=1e-6)

    def test_prefix_stays_pristine_after_a_write(self):
        cache = make_cache()
        prefix = make_tokens(6, seed=13)
        prefix_blocks = cache.register_prefix("sys", prefix)
        cache.allocate_sequence("a", prefix_id="sys")
        cache.append("a", make_tokens(2, seed=14))

        np.testing.assert_allclose(cache.pool[prefix_blocks[1]][:, :2, :],
                                   as_pool(prefix[4:]), rtol=1e-6, atol=1e-6)
        # ...and a sibling that still shares the prefix is unaffected
        cache.allocate_sequence("b", prefix_id="sys")
        np.testing.assert_allclose(cache.gather("b"), prefix, rtol=1e-6, atol=1e-6)

    def test_later_sequences_see_the_original_prefix(self):
        cache = make_cache()
        prefix = make_tokens(6, seed=15)
        cache.register_prefix("sys", prefix)
        cache.allocate_sequence("a", prefix_id="sys")
        cache.append("a", make_tokens(2, seed=16))
        cache.allocate_sequence("c", prefix_id="sys")
        np.testing.assert_allclose(cache.gather("c"), prefix, rtol=1e-6, atol=1e-6)

    def test_append_at_a_block_boundary_does_not_clone(self):
        cache = make_cache()
        prefix = make_tokens(4, seed=17)                 # exactly one full block
        prefix_blocks = cache.register_prefix("sys", prefix)
        cache.allocate_sequence("a", prefix_id="sys")
        cache.append("a", make_tokens(3, seed=18))

        table = cache.get_block_table("a").tolist()
        assert table[0] == prefix_blocks[0]
        assert cache.reference_count(prefix_blocks[0]) == 2

    def test_append_scaled_also_copies_on_write(self):
        cache = make_cache()
        prefix = make_tokens(6, seed=19)
        prefix_blocks = cache.register_prefix("sys", prefix)
        cache.allocate_sequence("a", prefix_id="sys")

        extra = make_tokens(2, seed=20)
        assert cache.append_scaled("a", extra, scale=0.5, clamp_limit=0.2) == 8
        expected = np.concatenate([prefix, np.clip(extra * 0.5, -0.2, 0.2)], axis=0)
        np.testing.assert_allclose(cache.gather("a"), expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(cache.pool[prefix_blocks[1]][:, :2, :],
                                   as_pool(prefix[4:]), rtol=1e-6, atol=1e-6)

    def test_multiple_writes_keep_cloning_only_the_boundary_block(self):
        cache = make_cache(num_blocks=32)
        cache.register_prefix("sys", make_tokens(6, seed=21))
        cache.allocate_sequence("a", prefix_id="sys")
        cache.append("a", make_tokens(2, seed=22))
        cache.append("a", make_tokens(2, seed=23))
        assert cache.sequence_length("a") == 10
        assert len(cache.get_block_table("a")) == 3


# ─── Reference-counted release ──────────────────────────────────────────────

class TestReferenceCounting:
    """Blocks survive until the last owner lets go."""

    def test_free_sequence_only_releases_one_reference(self):
        cache = make_cache(num_blocks=8, max_sequences=2)
        blocks = cache.register_prefix("p", make_tokens(4, seed=24))
        cache.allocate_sequence("a", prefix_id="p")
        cache.allocate_sequence("b", prefix_id="p")
        assert cache.reference_count(blocks[0]) == 3

        cache.free_sequence("a")
        assert cache.reference_count(blocks[0]) == 2
        assert cache.gather("b").shape[0] == 4

        cache.free_sequence("b")
        assert cache.reference_count(blocks[0]) == 1
        cache.free_prefix("p")
        assert cache.reference_count(blocks[0]) == 0
        assert cache.stats()["free_blocks"] == 8.0

    def test_free_prefix_keeps_blocks_alive_for_sharing_sequences(self):
        cache = make_cache(num_blocks=8, max_sequences=2)
        tokens = make_tokens(4, seed=25)
        blocks = cache.register_prefix("p", tokens)
        cache.allocate_sequence("a", prefix_id="p")

        cache.free_prefix("p")
        assert "p" not in cache.prefix_ids
        assert cache.reference_count(blocks[0]) == 1
        np.testing.assert_allclose(cache.gather("a"), tokens, rtol=1e-6, atol=1e-6)

        cache.free_sequence("a")
        assert cache.reference_count(blocks[0]) == 0

    def test_a_sequence_keeps_its_clone_after_the_prefix_is_gone(self):
        cache = make_cache(num_blocks=8, max_sequences=2)
        prefix = make_tokens(6, seed=26)
        cache.register_prefix("p", prefix)
        cache.allocate_sequence("a", prefix_id="p")
        extra = make_tokens(2, seed=27)
        cache.append("a", extra)
        cache.free_prefix("p")

        assert cache.sequence_length("a") == 8
        np.testing.assert_allclose(
            cache.gather("a"), np.concatenate([prefix, extra], axis=0),
            rtol=1e-6, atol=1e-6,
        )

    def test_reference_count_of_an_unused_block_is_zero(self):
        cache = make_cache(num_blocks=4)
        assert cache.reference_count(0) == 0


# ─── Lifecycle ──────────────────────────────────────────────────────────────

class TestPrefixLifecycle:
    """Reset and interaction with the non-prefix API."""

    def test_reset_clears_prefixes_and_reference_counts(self):
        cache = make_cache(num_blocks=8)
        cache.register_prefix("p", make_tokens(6, seed=28))
        cache.allocate_sequence("a", prefix_id="p")
        cache.reset()

        assert cache.prefix_ids == []
        assert cache.block_ref_counts == {}
        assert cache.shared_block_count == 0
        assert cache.stats()["free_blocks"] == 8.0
        assert np.all(cache.pool == 0.0)

    def test_plain_sequences_are_unaffected_by_prefix_support(self):
        cache = make_cache()
        tokens = make_tokens(5, seed=31)
        cache.append("plain", tokens)
        np.testing.assert_allclose(cache.gather("plain"), tokens, rtol=1e-6, atol=1e-6)
        assert cache.shared_block_count == 0
        assert cache.prefix_ids == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
