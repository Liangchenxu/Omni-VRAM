"""
Text-to-Speech Engine for vram_core
=====================================

Multi-backend TTS with automatic fallback:

1. **edge-tts** (preferred): Microsoft Edge TTS, free, high quality, 300+ voices
   - Requires: pip install edge-tts
   - Features: 300+ voices, 50+ languages, SSML support

2. **pyttsx3** (fallback): Offline TTS via system speech engine
   - Requires: pip install pyttsx3
   - Features: Offline, cross-platform, basic voice control

Usage:
    from vram_core.tts_engine import TTSEngine

    engine = TTSEngine()
    engine.speak("Hello, world!")
    engine.synthesize_to_file("Hello", "output.mp3")

    # List available voices
    voices = engine.list_voices(language="zh")

    # Streaming synthesis
    async for chunk in engine.stream_synthesize("Long text..."):
        process_audio_chunk(chunk)

    # Sentence-level streaming pipeline (v2.7.0): feed the LLM token stream in,
    # get finished-sentence audio out as soon as each sentence closes.
    async for chunk in engine.stream_synthesize(llm_token_stream()):
        process_audio_chunk(chunk)
"""

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Callable, Iterable, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# 鈹€鈹€ Backend Detection 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_EDGE_TTS_AVAILABLE = False
try:
    import edge_tts
    _EDGE_TTS_AVAILABLE = True
    logger.info("edge-tts detected 锟?high-quality TTS available")
except ImportError:
    pass

_PYTTSX3_AVAILABLE = False
try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
    if not _EDGE_TTS_AVAILABLE:
        logger.info("pyttsx3 detected 锟?offline TTS available")
except ImportError:
    pass

# Optional: soundfile for audio I/O
_SOUNDFILE_AVAILABLE = False
try:
    import soundfile as sf
    _SOUNDFILE_AVAILABLE = True
except ImportError:
    pass


@dataclass
class TTSVoice:
    """TTS voice metadata."""
    voice_id: str
    name: str
    language: str
    gender: str = "unknown"
    provider: str = "unknown"


@dataclass
class TTSResult:
    """TTS synthesis result."""
    audio: Optional[np.ndarray] = None
    sample_rate: int = 24000
    duration_seconds: float = 0.0
    voice_id: str = ""
    text: str = ""
    file_path: Optional[str] = None


class SentenceStreamBuffer:
    """
    Accumulates a streamed text fragment feed and emits complete sentences.

    LLM output arrives token by token; feeding every token straight into a TTS
    backend is both slow (one synthesis per token) and unnatural (no prosody
    across a sentence). This buffer holds the tail back until a *real* sentence
    boundary is seen, then releases the finished sentence so the synthesizer can
    start on it while the LLM is still generating the next one.

    Boundary rules
    --------------
    * CJK terminators ``。！？；…`` and newlines end a sentence immediately.
    * ASCII ``! ? ;`` end a sentence; trailing closing quotes/brackets
      (``"'\u201d\u2019)]}``) stay attached to it.
    * ``.`` is only a boundary when it is *not* part of a number or an internal
      token: ``3.14``, ``1,000.50``, ``v2.7.0`` and ``192.168.0.1`` stay whole.
    * A ``.`` that is still the last character received is undecidable (the next
      fragment may turn it into ``3.`` + ``14``), so it is buffered until more
      text -- or :meth:`flush` -- resolves it.

    Example:
        >>> buffer = SentenceStreamBuffer()
        >>> buffer.feed("The value is 3.")
        []
        >>> buffer.feed("14 exactly. Done. ")
        ['The value is 3.14 exactly.', 'Done.']
        >>> buffer.flush()
        []
    """

    CJK_TERMINATORS = "。！？；…\n"
    ASCII_TERMINATORS = ".!?;"
    CLOSERS = "\"'\u201d\u2019)]}】」』"

    def __init__(self, min_chars: int = 1, max_chars: int = 0):
        """
        Args:
            min_chars: Minimum sentence length (in characters) that may be
                emitted; shorter fragments stay buffered.
            max_chars: When > 0, a punctuation-free run longer than this is cut
                at the last whitespace before the cap, so a wall of text still
                streams instead of stalling until the very end.
        """
        self.min_chars = max(1, int(min_chars))
        self.max_chars = max(0, int(max_chars))
        self._buffer = ""

    # ── State ─────────────────────────────────────────────────────────────
    @property
    def pending(self) -> str:
        """Text held back until its sentence is complete."""
        return self._buffer

    def __len__(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        """Drop any buffered text."""
        self._buffer = ""

    # ── Feeding ───────────────────────────────────────────────────────────
    def feed(self, fragment: str) -> List[str]:
        """
        Add a fragment and return every sentence it completed.

        Args:
            fragment: Newly arrived text (typically one LLM token/chunk).

        Returns:
            The complete sentences released by this fragment, in order.
        """
        if not fragment:
            return []
        self._buffer += str(fragment)

        sentences: List[str] = []
        while True:
            boundary = self._find_boundary()
            if boundary is None:
                break
            sentences.append(self._buffer[:boundary].strip())
            self._buffer = self._buffer[boundary:].lstrip()

        if self.max_chars and len(self._buffer) > self.max_chars:
            cut = self._buffer.rfind(" ", 0, self.max_chars)
            if cut <= 0:
                cut = self.max_chars
            sentences.append(self._buffer[:cut].strip())
            self._buffer = self._buffer[cut:].lstrip()

        return [sentence for sentence in sentences if sentence]

    def flush(self) -> List[str]:
        """Release the remaining buffered text as a final sentence."""
        tail = self._buffer.strip()
        self._buffer = ""
        return [tail] if tail else []

    # ── Boundary detection ────────────────────────────────────────────────
    def _find_boundary(self) -> Optional[int]:
        """Index just past the first committed terminator, or ``None``."""
        text = self._buffer
        length = len(text)
        index = 0
        while index < length:
            char = text[index]
            if char in self.CJK_TERMINATORS:
                end = index + 1
            elif char in self.ASCII_TERMINATORS:
                if char == ".":
                    if self._dot_is_internal(text, index):
                        index += 1
                        continue
                    if index + 1 == length:
                        # "3." may still become "3.14" in the next fragment.
                        return None
                end = index + 1
            else:
                index += 1
                continue

            while end < length and text[end] in self.CLOSERS:
                end += 1
            if len(text[:end].strip()) < self.min_chars:
                index = end
                continue
            return end
        return None

    def _dot_is_internal(self, text: str, index: int) -> bool:
        """
        True when the ``.`` at ``index`` belongs to a number or a token rather
        than to a sentence boundary (``3.14``, ``1,000.50``, ``v2.7.0``,
        ``e.g.``, ``192.168.0.1``).
        """
        previous = text[index - 1] if index > 0 else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if previous.isalnum() and following.isalnum():
            return True
        # A trailing "3." at the very end of the buffer is ambiguous: wait.
        if previous.isalnum() and not following and index + 1 == len(text):
            return True
        return False


class TTSEngine:
    """
    Multi-backend Text-to-Speech engine.

    Features:
        - edge-tts: Free, high-quality, 300+ voices
        - pyttsx3: Offline fallback via system engine
        - Auto backend selection
        - File output (mp3, wav, ogg)
        - Async streaming synthesis
        - Voice listing and selection

    Args:
        backend: Backend to use ("auto", "edge-tts", "pyttsx3").
        voice: Voice ID (e.g. "en-US-AriaNeural", "zh-CN-XiaoxiaoNeural").
        rate: Speech rate adjustment (e.g. "+20%", "-10%").
        volume: Volume adjustment (e.g. "+0%", "+50%").
        pitch: Pitch adjustment (e.g. "+0Hz", "-5Hz").
    """

    DEFAULT_VOICES = {
        "en": "en-US-AriaNeural",
        "zh": "zh-CN-XiaoxiaoNeural",
        "ja": "ja-JP-NanamiNeural",
        "ko": "ko-KR-SunHiNeural",
        "es": "es-ES-ElviraNeural",
        "fr": "fr-FR-DeniseNeural",
        "de": "de-DE-KatjaNeural",
    }

    def __init__(
        self,
        backend: str = "auto",
        voice: Optional[str] = None,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
    ):
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self._voice = voice or self.DEFAULT_VOICES["en"]

        self._edge_tts = None
        self._pyttsx3_engine = None
        self._active_backend = "none"
        self._init_backend(backend)

    def _init_backend(self, backend: str):
        """Initialize TTS backend."""
        if backend in ("auto", "edge-tts") and _EDGE_TTS_AVAILABLE:
            self._active_backend = "edge-tts"
            logger.info("Using edge-tts backend, voice: %s", self._voice)
        elif backend in ("auto", "pyttsx3") and _PYTTSX3_AVAILABLE:
            try:
                self._pyttsx3_engine = pyttsx3.init()
                self._pyttsx3_engine.setProperty('rate', 150)
                self._pyttsx3_engine.setProperty('volume', 1.0)
                self._active_backend = "pyttsx3"
                logger.info("Using pyttsx3 backend")
            except (RuntimeError, OSError) as e:
                logger.warning("pyttsx3 init failed: %s", e)
                self._active_backend = "none"
        else:
            logger.warning("No TTS backend available. Install edge-tts or pyttsx3.")
            self._active_backend = "none"

    @property
    def backend(self) -> str:
        return self._active_backend

    @property
    def voice(self) -> str:
        return self._voice

    @voice.setter
    def voice(self, value: str):
        self._voice = value

    # 鈹€鈹€ Synthesis 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
    ) -> TTSResult:
        """
        Synthesize text to audio.

        Args:
            text: Text to synthesize.
            output_path: If set, save audio to this file path.

        Returns:
            TTSResult with audio data or file path.
        """
        if not text.strip():
            return TTSResult(text=text, voice_id=self._voice)

        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(), f"omni_vram_tts_{id(text)}.mp3"
            )

        if self._active_backend == "edge-tts":
            return self._synthesize_edge_tts(text, output_path)
        elif self._active_backend == "pyttsx3":
            return self._synthesize_pyttsx3(text, output_path)
        else:
            raise RuntimeError("No TTS backend available")

    def _synthesize_edge_tts(self, text: str, output_path: str) -> TTSResult:
        """Synthesize using edge-tts."""
        try:
            communicate = edge_tts.Communicate(
                text,
                voice=self._voice,
                rate=self.rate,
                volume=self.volume,
                pitch=self.pitch,
            )
            asyncio.run(communicate.save(output_path))

            # Try to load audio data
            audio = None
            sr = 24000
            if _SOUNDFILE_AVAILABLE:
                try:
                    audio, sr = sf.read(output_path, dtype='float32')
                except Exception:
                    pass

            duration = 0.0
            if audio is not None:
                duration = len(audio) / sr

            return TTSResult(
                audio=audio,
                sample_rate=sr,
                duration_seconds=duration,
                voice_id=self._voice,
                text=text,
                file_path=output_path,
            )
        except (RuntimeError, OSError, ConnectionError) as e:
            logger.error("edge-tts synthesis failed: %s", e)
            raise

    def _synthesize_pyttsx3(self, text: str, output_path: str) -> TTSResult:
        """Synthesize using pyttsx3."""
        try:
            wav_path = output_path if output_path.endswith('.wav') else output_path + '.wav'
            self._pyttsx3_engine.save_to_file(text, wav_path)
            self._pyttsx3_engine.runAndWait()

            audio = None
            sr = 22050
            if _SOUNDFILE_AVAILABLE:
                try:
                    audio, sr = sf.read(wav_path, dtype='float32')
                except Exception:
                    pass

            duration = 0.0
            if audio is not None:
                duration = len(audio) / sr

            return TTSResult(
                audio=audio,
                sample_rate=sr,
                duration_seconds=duration,
                voice_id=self._voice,
                text=text,
                file_path=wav_path,
            )
        except (RuntimeError, OSError) as e:
            logger.error("pyttsx3 synthesis failed: %s", e)
            raise

    def speak(self, text: str):
        """Synthesize and play text directly (blocking)."""
        if self._active_backend == "pyttsx3" and self._pyttsx3_engine:
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()
        else:
            result = self.synthesize(text)
            # Try to play with sounddevice
            try:
                import sounddevice as sd
                if result.audio is not None:
                    sd.play(result.audio, result.sample_rate)
                    sd.wait()
            except ImportError:
                logger.warning("Install sounddevice to play audio: pip install sounddevice")

    def synthesize_to_file(self, text: str, path: str) -> TTSResult:
        """Synthesize text directly to a specified file."""
        return self.synthesize(text, output_path=path)

    # 鈹€鈹€ Async Streaming 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    async def stream_synthesize(
        self,
        text: Union[str, AsyncIterator[str], Iterable[str]],
        *,
        sentence_buffer: Optional["SentenceStreamBuffer"] = None,
        synthesize: Optional[Callable[[str], AsyncIterator[bytes]]] = None,
    ) -> AsyncIterator[bytes]:
        """
        Stream synthesis using edge-tts (yields audio chunks).

        Two input modes (v2.7.0):

        * a complete ``str`` -- synthesised as one utterance (original API);
        * an **async or sync iterator of text fragments** (an LLM token stream).
          The fragments are buffered into complete sentences by
          :class:`SentenceStreamBuffer` and each sentence is synthesized the
          moment it closes, so the first audio chunk leaves long before the LLM
          has finished generating -- and, because a decimal point never splits a
          sentence, ``"3.14"`` is spoken as one number rather than two.

        Args:
            text: Text to synthesize, or an iterable/async-iterable of
                fragments.
            sentence_buffer: Optional buffer to reuse (its state is visible to
                the caller after the stream ends).
            synthesize: Optional override for the per-sentence synthesizer
                (``str -> AsyncIterator[bytes]``); defaults to the active
                backend. Useful for testing and for plugging in a local model.

        Yields:
            Audio data chunks as bytes.
        """
        if isinstance(text, str):
            async for chunk in self._stream_sentence_audio(text, synthesize):
                yield chunk
            return

        buffer = sentence_buffer if sentence_buffer is not None else SentenceStreamBuffer()
        async for fragment in self._aiter_fragments(text):
            for sentence in buffer.feed(fragment):
                async for chunk in self._stream_sentence_audio(sentence, synthesize):
                    yield chunk
        for sentence in buffer.flush():
            async for chunk in self._stream_sentence_audio(sentence, synthesize):
                yield chunk

    async def _stream_sentence_audio(
        self, sentence: str, synthesize: Optional[Callable[[str], AsyncIterator[bytes]]]
    ) -> AsyncIterator[bytes]:
        """Synthesize one sentence through the injected hook or edge-tts."""
        if synthesize is not None:
            async for chunk in synthesize(sentence):
                yield chunk
            return

        if self._active_backend != "edge-tts":
            raise RuntimeError("Streaming requires edge-tts backend")
        if not sentence.strip():
            return

        communicate = edge_tts.Communicate(
            sentence,
            voice=self._voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    @staticmethod
    async def _aiter_fragments(source) -> AsyncIterator[str]:
        """Normalise a sync/async iterable of fragments into async iteration."""
        if hasattr(source, "__aiter__"):
            async for fragment in source:
                yield fragment
            return
        for fragment in source:
            yield fragment

    # 鈹€鈹€ Voice Listing 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    async def _list_edge_voices(self, language: Optional[str] = None) -> List[TTSVoice]:
        """List available edge-tts voices."""
        voices = []
        edge_voices = await edge_tts.list_voices()
        for v in edge_voices:
            lang = v.get("Locale", "")
            if language and not lang.startswith(language):
                continue
            voices.append(TTSVoice(
                voice_id=v.get("ShortName", ""),
                name=v.get("FriendlyName", ""),
                language=lang,
                gender=v.get("Gender", "unknown"),
                provider="edge-tts",
            ))
        return voices

    def list_voices(self, language: Optional[str] = None) -> List[TTSVoice]:
        """List available voices for the active backend."""
        if self._active_backend == "edge-tts":
            return asyncio.run(self._list_edge_voices(language))
        elif self._active_backend == "pyttsx3" and self._pyttsx3_engine:
            voices = []
            for v in self._pyttsx3_engine.getProperty('voices'):
                lang = ""
                if hasattr(v, 'languages'):
                    lang = v.languages[0] if v.languages else ""
                voices.append(TTSVoice(
                    voice_id=v.id,
                    name=v.name,
                    language=lang,
                    provider="pyttsx3",
                ))
            return voices
        return []

    # 鈹€鈹€ Utility 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @staticmethod
    def available_backends() -> List[str]:
        """List available TTS backends."""
        backends = []
        if _EDGE_TTS_AVAILABLE:
            backends.append("edge-tts")
        if _PYTTSX3_AVAILABLE:
            backends.append("pyttsx3")
        return backends

    @staticmethod
    def available_languages() -> List[str]:
        """List languages with default voices."""
        return list(TTSEngine.DEFAULT_VOICES.keys())

    def close(self):
        """Release resources."""
        if self._pyttsx3_engine:
            try:
                self._pyttsx3_engine.stop()
            except Exception:
                pass
            self._pyttsx3_engine = None

    def __del__(self):
        self.close()