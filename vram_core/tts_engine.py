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
"""

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Callable, List, Optional

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

    async def stream_synthesize(self, text: str) -> AsyncIterator[bytes]:
        """
        Stream synthesis using edge-tts (yields audio chunks).

        Args:
            text: Text to synthesize.

        Yields:
            Audio data chunks as bytes.
        """
        if self._active_backend != "edge-tts":
            raise RuntimeError("Streaming requires edge-tts backend")

        communicate = edge_tts.Communicate(
            text,
            voice=self._voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

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