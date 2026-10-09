"""
Tests for the streaming sentence-level TTS pipeline (v2.7.0).

Covers:
    - ``SentenceStreamBuffer``: sentence segmentation for streamed LLM text,
      including the decimal-point guard ("3.14" must never be split), CJK
      terminators, trailing closers, ``min_chars`` merging and ``max_chars``
      cutting
    - ``TTSEngine.stream_synthesize``: string (v2.6) input plus sync/async
      fragment iterators, per-sentence synthesis and tail flushing

Audio is produced through the injectable ``synthesize`` hook so the suite runs
without edge-tts / network access; the injected hook mirrors the real backend
signature (``str -> AsyncIterator[bytes]``).
"""

import asyncio

import pytest
from unittest.mock import MagicMock, patch

from vram_core.tts_engine import SentenceStreamBuffer, TTSEngine


def make_engine() -> TTSEngine:
    """A TTS engine whose active backend is edge-tts (never actually called)."""
    with patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", True):
        return TTSEngine(backend="edge-tts")


def recording_synth(sink):
    """Build a ``str -> AsyncIterator[bytes]`` hook that records its input."""
    async def _synth(sentence: str):
        sink.append(sentence)
        yield f"<{sentence}>".encode("utf-8")
    return _synth


def collect(agen_factory) -> list:
    """Run an async generator factory to completion and return its items."""
    async def _drain():
        return [item async for item in agen_factory()]

    return asyncio.run(_drain())


# ─── SentenceStreamBuffer ───────────────────────────────────────────────────

class TestSentenceBoundaries:
    """Streamed fragments are released as complete sentences."""

    def test_waits_for_a_terminator(self):
        buffer = SentenceStreamBuffer()
        assert buffer.feed("Hello there") == []
        assert buffer.pending == "Hello there"

    def test_splits_on_ascii_period(self):
        buffer = SentenceStreamBuffer()
        assert buffer.feed("Hello there. How are you? ") == [
            "Hello there.", "How are you?",
        ]

    def test_splits_on_cjk_terminators(self):
        buffer = SentenceStreamBuffer()
        assert buffer.feed("你好。今天天气很好！是吗？") == ["你好。", "今天天气很好！", "是吗？"]

    def test_splits_on_newline(self):
        buffer = SentenceStreamBuffer()
        assert buffer.feed("first line\nsecond line\n") == ["first line", "second line"]

    def test_closing_quotes_stay_with_the_sentence(self):
        buffer = SentenceStreamBuffer()
        assert buffer.feed('He said "go home." Then he left. ') == [
            'He said "go home."', "Then he left.",
        ]

    def test_min_chars_merges_tiny_fragments(self):
        buffer = SentenceStreamBuffer(min_chars=4)
        assert buffer.feed("Hi. Hello there. ") == ["Hi. Hello there."]

    def test_max_chars_cuts_a_punctuation_free_run(self):
        buffer = SentenceStreamBuffer(max_chars=12)
        sentences = buffer.feed("alpha beta gamma delta epsilon")
        # cut at the last whitespace before the 12-character cap
        assert sentences == ["alpha beta"]
        assert buffer.pending == "gamma delta epsilon"

    def test_flush_returns_the_tail(self):
        buffer = SentenceStreamBuffer()
        buffer.feed("no terminator here")
        assert buffer.flush() == ["no terminator here"]
        assert buffer.pending == ""
        assert buffer.flush() == []

    def test_reset_drops_the_buffer(self):
        buffer = SentenceStreamBuffer()
        buffer.feed("pending text")
        buffer.reset()
        assert buffer.pending == ""
        assert len(buffer) == 0


class TestDecimalPointProtection:
    """A decimal point is never treated as a sentence boundary."""

    def test_decimal_in_one_fragment(self):
        buffer = SentenceStreamBuffer()
        assert buffer.feed("The value is 3.14 exactly. ") == ["The value is 3.14 exactly."]

    def test_decimal_split_across_fragments(self):
        buffer = SentenceStreamBuffer()
        assert buffer.feed("The value is 3.") == []
        assert buffer.feed("14 exactly. ") == ["The value is 3.14 exactly."]

    def test_thousands_separator_and_exponent(self):
        buffer = SentenceStreamBuffer()
        assert buffer.feed("It cost 1,000.50 dollars. ") == ["It cost 1,000.50 dollars."]

    def test_version_string_is_not_a_boundary(self):
        buffer = SentenceStreamBuffer()
        assert buffer.feed("Omni-VRAM v2.7.0 is out. ") == ["Omni-VRAM v2.7.0 is out."]

    def test_ip_address_is_not_a_boundary(self):
        buffer = SentenceStreamBuffer()
        assert buffer.feed("Connect to 192.168.0.1 now. ") == ["Connect to 192.168.0.1 now."]

    def test_trailing_number_dot_is_held_until_resolved(self):
        buffer = SentenceStreamBuffer()
        assert buffer.feed("The answer is 42.") == []      # decimal? unknown yet
        assert buffer.feed(" Next topic.") == ["The answer is 42."]
        assert buffer.flush() == ["Next topic."]

    def test_flush_resolves_a_trailing_dot(self):
        buffer = SentenceStreamBuffer()
        assert buffer.feed("Done.") == []
        assert buffer.flush() == ["Done."]


# ─── TTSEngine.stream_synthesize ────────────────────────────────────────────

class TestStreamSynthesize:
    """The engine turns a fragment stream into per-sentence audio."""

    def test_string_input_is_synthesized_as_one_utterance(self):
        engine = make_engine()
        spoken = []
        chunks = collect(lambda: engine.stream_synthesize(
            "Hello world", synthesize=recording_synth(spoken)
        ))
        assert spoken == ["Hello world"]
        assert chunks == [b"<Hello world>"]

    def test_fragment_iterable_produces_one_chunk_per_sentence(self):
        engine = make_engine()
        spoken = []
        fragments = ["Hello there. ", "How are you? ", "Fine, thanks! "]
        chunks = collect(lambda: engine.stream_synthesize(
            iter(fragments), synthesize=recording_synth(spoken)
        ))
        assert spoken == ["Hello there.", "How are you?", "Fine, thanks!"]
        assert chunks == [b"<Hello there.>", b"<How are you?>", b"<Fine, thanks!>"]

    def test_async_iterator_input(self):
        engine = make_engine()
        spoken = []

        async def fragments():
            for fragment in ["你好。", "世界！"]:
                yield fragment

        chunks = collect(lambda: engine.stream_synthesize(
            fragments(), synthesize=recording_synth(spoken)
        ))
        assert spoken == ["你好。", "世界！"]
        assert chunks == ["<你好。>".encode("utf-8"), "<世界！>".encode("utf-8")]

    def test_tail_is_flushed_at_the_end(self):
        engine = make_engine()
        spoken = []
        collect(lambda: engine.stream_synthesize(
            iter(["One sentence. ", "no terminator"]),
            synthesize=recording_synth(spoken),
        ))
        assert spoken == ["One sentence.", "no terminator"]

    def test_decimals_are_not_split_across_chunks(self):
        engine = make_engine()
        spoken = []
        collect(lambda: engine.stream_synthesize(
            iter(["圆周率是 3.", "14，很精确。"]),
            synthesize=recording_synth(spoken),
        ))
        assert spoken == ["圆周率是 3.14，很精确。"]

    def test_synthesis_starts_before_the_stream_is_exhausted(self):
        engine = make_engine()
        consumed = []
        synthesized_at = []

        def fragments():
            for fragment in ["First sentence. ", "Second sentence. ", "Third one. "]:
                consumed.append(fragment)
                yield fragment

        async def synth(sentence):
            synthesized_at.append(len(consumed))
            yield sentence.encode("utf-8")

        chunks = collect(lambda: engine.stream_synthesize(fragments(), synthesize=synth))
        # sentence N is spoken as soon as fragment N lands -- not after the LLM
        # has produced the whole answer
        assert synthesized_at == [1, 2, 3]
        assert len(chunks) == 3

    def test_caller_supplied_buffer_is_reused(self):
        engine = make_engine()
        buffer = SentenceStreamBuffer()
        collect(lambda: engine.stream_synthesize(
            iter(["Dangling tail"]), sentence_buffer=buffer, synthesize=recording_synth([])
        ))
        assert buffer.pending == ""

    def test_streaming_still_requires_edge_tts_without_a_hook(self):
        with patch("vram_core.tts_engine._PYTTSX3_AVAILABLE", True):
            with patch("vram_core.tts_engine.pyttsx3") as mock_pyttsx3:
                mock_pyttsx3.init.return_value = MagicMock()
                engine = TTSEngine(backend="pyttsx3")

                async def run():
                    async for _ in engine.stream_synthesize("test"):
                        pass

                with pytest.raises(RuntimeError):
                    asyncio.run(run())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
