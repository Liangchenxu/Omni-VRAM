"""
Unit Tests for Dynamic Overlap Alignment & Whisper Hallucination Suppression
============================================================================

Covers the v2.6.0 streaming ASR hardening:

    1. ``align_overlap_text`` / ``OverlapAligner`` -- dynamic overlap-add
       (LCS) alignment that removes duplicated window boundaries
    2. ``filter_hallucinations`` / ``truncate_repetitions`` / ``clean_transcript``
       -- suppression of silence hallucinations and pathological n-gram loops
    3. ``StreamASR`` integration: energy gate, aligned partials, filtered finals,
       ``flush()`` and status reporting
"""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vram_core.streaming_asr import (
    HALLUCINATION_PHRASES,
    OverlapAligner,
    StreamASR,
    StreamASRConfig,
    TranscriptFilter,
    align_overlap_text,
    clean_transcript,
    filter_hallucinations,
    find_exact_overlap,
    find_lcs_overlap,
    is_hallucination,
    lcs_length,
    tokenize_text,
    truncate_repetitions,
)
from vram_core.whisper import WhisperResult


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    """Poll ``predicate`` until it is true or the timeout elapses."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class FakeBridge:
    """Deterministic Whisper stand-in returning a scripted list of texts."""

    def __init__(self, texts, language: str = "zh"):
        self.texts = list(texts)
        self.language = language
        self.calls = 0

    def transcribe(self, audio, sample_rate: int = 16000) -> WhisperResult:
        """Return the next scripted transcript (last one repeats)."""
        self.calls += 1
        index = min(self.calls - 1, len(self.texts) - 1)
        return WhisperResult(
            text=self.texts[index], language=self.language, confidence=0.9
        )


def speech_chunk(samples: int = 1600, amplitude: float = 0.3) -> np.ndarray:
    """Return a deterministic speech-like chunk."""
    return (amplitude * np.sin(np.arange(samples) * 0.1)).astype(np.float32)


class TestAlignOverlapText:
    """Dynamic overlap-add merging of consecutive window transcripts."""

    def test_removes_duplicated_boundary(self):
        """[今天天气] + [天气真好] must not repeat "天气"."""
        assert align_overlap_text("今天天气", "天气真好") == "今天天气真好"

    def test_superset_window(self):
        """A window that already contains the transcript replaces it."""
        assert align_overlap_text("今天天气", "今天天气真好") == "今天天气真好"

    def test_regression_keeps_longer_text(self):
        """A shorter re-recognition never truncates the transcript."""
        assert align_overlap_text("今天天气真好", "天气真好") == "今天天气真好"

    def test_punctuation_drift_is_tolerated(self):
        """Punctuation added by ASR does not break the overlap match."""
        assert align_overlap_text("今天天气，", "今天天气真好") == "今天天气，真好"

    def test_lcs_absorbs_single_character_jitter(self):
        """One substituted character inside the overlap is still aligned."""
        merged = align_overlap_text("今天天气真好", "今天天气真棒我们走吧")
        assert merged == "今天天气真好我们走吧"

    def test_no_overlap_concatenates(self):
        """Unrelated windows are concatenated without losing content."""
        assert align_overlap_text("今天天气", "我们走吧") == "今天天气我们走吧"

    def test_empty_inputs(self):
        """Empty sides are handled gracefully."""
        assert align_overlap_text("", "你好") == "你好"
        assert align_overlap_text("你好", "") == "你好"
        assert align_overlap_text("", "") == ""

    def test_english_words(self):
        """Word-level overlap works for latin text."""
        assert align_overlap_text("I think we", "we should go") == "I think we should go"


class TestAlignmentHelpers:
    """Low level helpers used by the aligner."""

    def test_tokenize_mixed_text(self):
        """CJK is split per character, latin per word."""
        assert tokenize_text("今天天气 hello") == ["今", "天", "天", "气", " ", "hello"]

    def test_lcs_length(self):
        """Rolling-row LCS matches the classic definition."""
        assert lcs_length("abcde", "ace") == 3
        assert lcs_length("abc", "abc") == 3
        assert lcs_length("abc", "xyz") == 0
        assert lcs_length("", "abc") == 0

    def test_find_exact_overlap(self):
        """Exact suffix/prefix overlap length."""
        assert find_exact_overlap("今天天气", "天气真好") == 2
        assert find_exact_overlap("今天天气", "今天天气真好") == 4
        assert find_exact_overlap("今天天气", "我们走吧") == 0

    def test_find_lcs_overlap_requires_minimum(self):
        """Short accidental overlaps are rejected."""
        k, _ = find_lcs_overlap("abc", "xyz", min_overlap=4)
        assert k == 0

    def test_overlap_budget_is_respected(self):
        """max_overlap_chars caps the removable overlap."""
        assert find_exact_overlap("abcabcabc", "abcabcxyz", max_overlap_chars=9) == 6
        assert find_exact_overlap("abcabcabc", "abcabcxyz", max_overlap_chars=5) == 3


class TestOverlapAligner:
    """Stateful aligner used by StreamASR."""

    def test_sequential_merges(self):
        """Consecutive window results build one clean transcript."""
        aligner = OverlapAligner()
        assert aligner.merge("今天天气") == "今天天气"
        assert aligner.merge("天气真好") == "今天天气真好"
        assert aligner.merge("今天天气真好我们去公园吧") == "今天天气真好我们去公园吧"
        assert aligner.text == "今天天气真好我们去公园吧"

    def test_reset_and_set_text(self):
        """reset() clears state, set_text() re-syncs with a final result."""
        aligner = OverlapAligner()
        aligner.merge("你好")
        aligner.reset()
        assert aligner.text == ""
        aligner.set_text("你好")
        assert aligner.text == "你好"

    def test_sliding_window_does_not_lose_words(self):
        """A window that slides forward keeps the earlier head text."""
        aligner = OverlapAligner()
        aligner.merge("今天天气真好")
        assert aligner.merge("天气真好我们出去") == "今天天气真好我们出去"


class TestHallucinationSuppression:
    """Whisper hallucination filtering and repetition truncation."""

    def test_known_phrases_are_registered(self):
        """The artefact phrase table is populated."""
        assert len(HALLUCINATION_PHRASES) > 10

    def test_whole_output_is_hallucination(self):
        """Credits-only output is dropped entirely."""
        for text in ("感谢观看", "谢谢观看", "字幕由 Amara.org 社区提供",
                     "Thanks for watching", "。。。", "♪♪♪", " "):
            assert is_hallucination(text), text
            assert filter_hallucinations(text) == ""

    def test_trailing_artefact_is_stripped(self):
        """An artefact appended to real speech is removed."""
        assert filter_hallucinations("今天天气不错感谢观看") == "今天天气不错"
        assert filter_hallucinations("hello world Thanks for watching") == "hello world"

    def test_real_content_is_preserved(self):
        """Normal speech passes through untouched."""
        for text in (
            "你好，今天天气不错，我们出去走走吧",
            "please subscribe to my channel later",   # mid-sentence: keep
            "人工智能在医疗领域的应用越来越广泛",
        ):
            cleaned = filter_hallucinations(text)
            assert cleaned, text
            assert not is_hallucination(text), text

    def test_repetition_loop_is_truncated(self):
        """A phrase loop repeated 3+ times is cut back to one occurrence."""
        assert truncate_repetitions("谢谢大家谢谢大家谢谢大家") == "谢谢大家"
        assert truncate_repetitions("我们明天见我们明天见我们明天见") == "我们明天见"

    def test_natural_interjections_survive(self):
        """Laughter and short reduplications are not pathological loops."""
        assert truncate_repetitions("哈哈哈哈") == "哈哈哈哈"
        assert truncate_repetitions("好的好的") == "好的好的"

    def test_normal_sentence_untouched(self):
        """Text without loops is returned unchanged."""
        text = "今天天气不错我们一起去公园散步吧"
        assert truncate_repetitions(text) == text

    def test_clean_transcript_chain(self):
        """Filter -> loop truncation -> filter."""
        assert clean_transcript("谢谢大家谢谢大家谢谢大家") == "谢谢大家"
        assert clean_transcript("感谢观看") == ""
        assert clean_transcript("你好你好吗") == "你好你好吗"

    def test_transcript_filter_can_be_disabled(self):
        """enable_hallucination_filter=False passes raw text through."""
        config = StreamASRConfig(enable_hallucination_filter=False)
        filt = TranscriptFilter(config)
        assert filt.process("感谢观看") == "感谢观看"
        assert filt.process("  ") == ""


class TestStreamASRIntegration:
    """StreamASR end-to-end behaviour with a scripted bridge."""

    @staticmethod
    def _config(**overrides) -> StreamASRConfig:
        """Fast, deterministic streaming config for tests."""
        params = dict(
            window_duration=0.5,
            step_duration=0.05,
            min_audio_duration=0.2,
            silence_timeout=0.3,
        )
        params.update(overrides)
        return StreamASRConfig(**params)

    def test_defaults_expose_new_options(self):
        """The v2.6.0 options have sensible defaults."""
        config = StreamASRConfig()
        assert config.enable_overlap_alignment is True
        assert config.enable_hallucination_filter is True
        assert config.max_overlap_chars > 0
        assert config.min_speech_energy > 0.0
        assert config.min_repeat_count >= 2
        assert config.max_transcript_chars > 0

    def test_partials_are_overlap_aligned(self):
        """Consecutive windows are merged instead of being concatenated."""
        asr = StreamASR(
            config=self._config(),
            whisper_bridge=FakeBridge(["今天天气", "天气真好"]),
        )
        partials = []
        asr.on_partial_result = partials.append
        asr.start()
        try:
            for _ in range(6):
                asr.feed(speech_chunk())
            assert _wait_for(lambda: len(partials) >= 2), partials
        finally:
            asr.stop()

        assert partials[0] == "今天天气"
        assert partials[-1] == "今天天气真好"
        assert all("天气天气" not in text for text in partials)

    def test_alignment_can_be_disabled(self):
        """With alignment off the raw window text is emitted."""
        asr = StreamASR(
            config=self._config(enable_overlap_alignment=False),
            whisper_bridge=FakeBridge(["今天天气", "天气真好"]),
        )
        partials = []
        asr.on_partial_result = partials.append
        asr.start()
        try:
            for _ in range(6):
                asr.feed(speech_chunk())
            assert _wait_for(lambda: len(partials) >= 2), partials
        finally:
            asr.stop()

        assert partials[-1] == "天气真好"

    def test_final_result_is_emitted_and_filtered(self):
        """The final transcript is filtered and emitted once."""
        asr = StreamASR(
            config=self._config(),
            whisper_bridge=FakeBridge(["今天天气", "天气真好"]),
        )
        finals = []
        partials = []
        asr.on_final_result = finals.append
        asr.on_partial_result = partials.append
        asr.start()
        for _ in range(6):
            asr.feed(speech_chunk())
        assert _wait_for(lambda: bool(partials)), partials
        result = asr.stop()

        assert result is not None
        assert result.is_final is True
        assert result.text == "今天天气真好"
        assert [r.text for r in finals] == ["今天天气真好"]

    def test_silence_never_reaches_whisper(self):
        """Quiet audio is gated before recognition (hallucination source)."""
        bridge = FakeBridge(["感谢观看"])
        asr = StreamASR(config=self._config(), whisper_bridge=bridge)
        partials = []
        asr.on_partial_result = partials.append
        asr.start()
        try:
            noise = (np.random.randn(1600) * 0.001).astype(np.float32)
            for _ in range(8):
                asr.feed(noise)
            time.sleep(0.3)
            assert bridge.calls == 0, "silence was sent to Whisper"
        finally:
            final = asr.stop()

        # Even the final pass is filtered out -> no result emitted
        assert final is None
        assert partials == []

    def test_repetition_loop_is_truncated_in_partials(self):
        """A loop hallucination is truncated before reaching callbacks."""
        asr = StreamASR(
            config=self._config(),
            whisper_bridge=FakeBridge(["谢谢大家谢谢大家谢谢大家"]),
        )
        partials = []
        asr.on_partial_result = partials.append
        asr.start()
        try:
            for _ in range(6):
                asr.feed(speech_chunk())
            assert _wait_for(lambda: bool(partials)), partials
        finally:
            asr.stop()

        assert all(text == "谢谢大家" for text in partials), partials

    def test_flush_clears_audio_and_transcript(self):
        """flush() drops buffered audio and the pending partial text."""
        config = self._config(step_duration=0.4)  # slow steps: no race with flush
        bridge = FakeBridge(["今天天气"])
        asr = StreamASR(config=config, whisper_bridge=bridge)
        asr.start()
        try:
            for _ in range(8):
                asr.feed(speech_chunk())
            assert _wait_for(lambda: asr.get_status()["last_text"] != "")
            assert asr.buffer_duration > 0.0

            asr.flush()
            assert asr.get_status()["last_text"] == ""
            assert asr.buffer_duration == 0.0
        finally:
            asr.stop()

    def test_status_reports_new_options(self):
        """get_status() exposes alignment/filter flags."""
        asr = StreamASR(config=self._config(), whisper_bridge=FakeBridge(["x"]))
        status = asr.get_status()
        assert status["config"]["overlap_alignment"] is True
        assert status["config"]["hallucination_filter"] is True
        assert status["is_running"] is False

    def test_feed_requires_start(self):
        """The public contract of feed() is unchanged."""
        asr = StreamASR(config=self._config(), whisper_bridge=FakeBridge(["x"]))
        with pytest.raises(RuntimeError):
            asr.feed(speech_chunk())


