"""
Unit Tests for Word-Boundary Aware Overlap Alignment (v2.6.1)
============================================================

Covers the v2.6.1 streaming-ASR upgrade in ``vram_core/streaming_asr.py``:

    1. ``word_spans()`` -- character spans of trustworthy multi-character words
       (``ChineseTokenizer`` based; jieba is optional, unsegmented CJK runs are
       rejected as boundary anchors)
    2. ``snap_to_word_boundary()`` -- moving a cut off a word interior onto the
       nearest word or punctuation boundary
    3. ``align_overlap_text(..., word_aware=True)`` -- no broken/duplicated
       characters at the sliding-window boundary for the *approximate* overlap
       paths (punctuation drift, LCS jitter)
    4. Backwards compatibility: exact character overlaps are never adjusted and
       the v2.6.0 alignment results are reproduced
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vram_core.streaming_asr import (
    OverlapAligner,
    align_overlap_text,
    get_alignment_tokenizer,
    snap_to_word_boundary,
    word_spans,
)


class StubTokenizer:
    """``ChineseTokenizer`` stand-in returning scripted character spans."""

    def __init__(self, spans):
        self.spans = [(int(start), int(end)) for start, end in spans]
        self.calls = 0

    def tokenize(self, text):
        """Return one token per scripted span (word text extracted from input)."""
        self.calls += 1
        return [
            SimpleNamespace(word=text[start:end], start=start, end=end)
            for start, end in self.spans
        ]


class BrokenTokenizer:
    """Tokenizer that always raises (exercises the defensive path)."""

    def tokenize(self, text):
        """Raise unconditionally."""
        raise RuntimeError("segmentation failure")


class TestWordSpans:
    """Trustworthy word spans extracted from the project tokenizer."""

    def test_single_characters_are_not_boundaries(self):
        """A one-character token can never be 'cut in half'."""
        tokenizer = StubTokenizer([(0, 1), (2, 4)])
        assert word_spans("a bc", tokenizer) == [(2, 4)]

    def test_over_long_cjk_runs_are_ignored(self):
        """Without jieba a whole CJK run comes back as one token: not a word."""
        unsegmented = "进行深度学习训练"          # 8 chars -> fallback lump
        assert word_spans(unsegmented, StubTokenizer([(0, 8)])) == []

        plausible = "进行深度学习"                # 6 chars -> still word-like
        assert word_spans(plausible, StubTokenizer([(0, 6)])) == [(0, 6)]

    def test_latin_words_are_kept_without_jieba(self):
        """Latin/digit runs are word-delimited even in the fallback tokenizer."""
        spans = [(0, 4), (5, 13)]
        assert word_spans("deep learning", StubTokenizer(spans)) == [(0, 4), (5, 13)]

    def test_broken_tokenizer_yields_no_spans(self):
        """A raising segmenter degrades to 'no boundary information'."""
        assert word_spans("abc def", BrokenTokenizer()) == []

    def test_empty_text_short_circuits(self):
        """Empty input never reaches the tokenizer."""
        tokenizer = StubTokenizer([(0, 3)])
        assert word_spans("", tokenizer) == []
        assert tokenizer.calls == 0

    def test_default_tokenizer_is_shared_and_usable(self):
        """The lazily built project tokenizer is reused across calls."""
        first = get_alignment_tokenizer()
        assert first is get_alignment_tokenizer()
        if first is not None:
            assert isinstance(word_spans("hello world", first), list)


class TestSnapToWordBoundary:
    """The cut-nudging primitive."""

    def test_safe_index_is_unchanged(self):
        """An index that is not inside a word is already safe."""
        tokenizer = StubTokenizer([(0, 4), (5, 9)])
        assert snap_to_word_boundary("deep learn", 4, tokenizer) == 4

    def test_cut_inside_a_word_snaps_to_the_word_edge(self):
        """A cut inside ``learning`` moves to its end."""
        tokenizer = StubTokenizer([(0, 8)])
        assert snap_to_word_boundary("learning now", 4, tokenizer) == 8

    def test_closest_boundary_wins(self):
        """The nearest word edge is chosen, not the farthest one."""
        tokenizer = StubTokenizer([(0, 4), (5, 13)])
        assert snap_to_word_boundary("deep learning", 6, tokenizer) == 5

    def test_punctuation_is_used_when_no_word_edge_is_close(self):
        """Punctuation inside the span acts as a boundary anchor."""
        tokenizer = StubTokenizer([(0, 12)])          # one long latin token
        assert snap_to_word_boundary("hello, world", 5, tokenizer) == 5

    def test_max_shift_limits_the_adjustment(self):
        """A boundary farther away than ``max_shift`` is not used."""
        tokenizer = StubTokenizer([(0, 10)])
        assert snap_to_word_boundary("abcdefghij", 4, tokenizer, max_shift=2) == 4

    def test_edge_indices_are_never_snapped(self):
        """Cutting at the very start/end of the text is always safe."""
        tokenizer = StubTokenizer([(0, 8)])
        assert snap_to_word_boundary("learning", 0, tokenizer) == 0
        assert snap_to_word_boundary("learning", 8, tokenizer) == 8


class TestWordAwareAlignment:
    """``align_overlap_text`` / ``OverlapAligner`` word-boundary behaviour."""

    def test_approximate_cut_is_snapped_off_the_word_interior(self):
        """The LCS-jitter cut inside ``alphabeta`` is moved to its edge."""
        previous, new = "zzz alphaq", "alphabeta gamma"
        raw = align_overlap_text(previous, new, word_aware=False)
        fixed = align_overlap_text(previous, new, word_aware=True)

        assert raw != fixed, "word-aware snapping did not adjust the cut"
        assert "alphaqeta" in raw              # unsnapped: broken word at the joint
        assert fixed == "zzz alphaq gamma"     # snapped: clean word boundary

    def test_default_tokenizer_snaps_latin_words(self):
        """jieba is optional -- latin runs are segmented by the fallback too."""
        assert align_overlap_text("zzz alphaq", "alphabeta gamma") == "zzz alphaq gamma"

    def test_exact_overlap_is_never_snapped(self):
        """A character-exact overlap reassembles a mid-word cut correctly."""
        merged = align_overlap_text("we are cod", "coding now")
        assert merged == "we are coding now"

    def test_disabling_word_awareness_restores_the_raw_cut(self):
        """``word_aware=False`` reproduces the pre-2.6.1 behaviour exactly."""
        assert (
            align_overlap_text("zzz alphaq", "alphabeta gamma", word_aware=False)
            == "zzz alphaqeta gamma"
        )

    def test_aligner_forwards_tokenizer_and_flag(self):
        """``OverlapAligner`` accepts an injected segmenter."""
        aligner = OverlapAligner(word_aware=True, tokenizer=StubTokenizer([(0, 9)]))
        assert aligner.merge("zzz alphaq") == "zzz alphaq"
        assert aligner.merge("alphabeta gamma") == "zzz alphaq gamma"

    def test_aligner_without_word_awareness(self):
        """Opting out keeps the raw concatenation."""
        aligner = OverlapAligner(word_aware=False)
        assert aligner.merge("zzz alphaq") == "zzz alphaq"
        assert aligner.merge("alphabeta gamma") == "zzz alphaqeta gamma"

    def test_broken_tokenizer_never_breaks_alignment(self):
        """Segmentation failures fall back to the unsnapped result."""
        merged = align_overlap_text(
            "zzz alphaq", "alphabeta gamma", tokenizer=BrokenTokenizer()
        )
        assert merged == "zzz alphaqeta gamma"

    def test_v260_chinese_results_are_preserved(self):
        """The documented v2.6.0 alignment examples still hold."""
        assert align_overlap_text("今天天气", "天气真好") == "今天天气真好"
        assert align_overlap_text("今天天气", "今天天气真好") == "今天天气真好"
        assert align_overlap_text("今天天气真好", "天气真好") == "今天天气真好"
        assert align_overlap_text("今天天气，", "今天天气真好") == "今天天气，真好"
        assert (
            align_overlap_text("今天天气真好", "今天天气真棒我们走吧")
            == "今天天气真好我们走吧"
        )

    def test_aligner_sequence_with_word_awareness(self):
        """A multi-window session still produces one clean transcript."""
        aligner = OverlapAligner()
        assert aligner.merge("今天天气") == "今天天气"
        assert aligner.merge("天气真好") == "今天天气真好"
        assert (
            aligner.merge("今天天气真好我们出去")
            == "今天天气真好我们出去"
        )

    def test_mixed_language_overlap(self):
        """Chinese/latin mixed windows keep their words intact."""
        merged = align_overlap_text("我们使用 GPU", "GPU 训练模型")
        assert merged == "我们使用 GPU 训练模型"
