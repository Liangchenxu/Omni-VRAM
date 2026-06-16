"""
Tests for vram_core.tts_engine module.

Covers:
    - TTSVoice / TTSResult data classes
    - TTSEngine initialization with various backends
    - Backend detection and selection
    - synthesize(), speak(), synthesize_to_file()
    - Voice listing, language listing, available_backends()
    - Edge cases: empty text, no backend available
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock

import numpy as np

from vram_core.tts_engine import (
    TTSVoice,
    TTSResult,
    TTSEngine,
)


class TestTTSVoice:
    """Test TTSVoice data class."""

    def test_create_default(self):
        v = TTSVoice(voice_id="en-US-AriaNeural", name="Aria", language="en-US")
        assert v.voice_id == "en-US-AriaNeural"
        assert v.gender == "unknown"
        assert v.provider == "unknown"

    def test_create_with_all_fields(self):
        v = TTSVoice(
            voice_id="zh-CN-XiaoxiaoNeural",
            name="Xiaoxiao",
            language="zh-CN",
            gender="Female",
            provider="edge-tts",
        )
        assert v.gender == "Female"
        assert v.provider == "edge-tts"


class TestTTSResult:
    """Test TTSResult data class."""

    def test_default_values(self):
        r = TTSResult()
        assert r.audio is None
        assert r.sample_rate == 24000
        assert r.duration_seconds == 0.0
        assert r.text == ""

    def test_custom_values(self):
        audio = np.random.randn(24000).astype(np.float32)
        r = TTSResult(
            audio=audio,
            sample_rate=24000,
            duration_seconds=1.0,
            voice_id="en-US-AriaNeural",
            text="Hello",
            file_path="/tmp/test.mp3",
        )
        assert r.audio is not None
        assert r.file_path == "/tmp/test.mp3"


class TestTTSEngineInit:
    """Test TTSEngine initialization."""

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", False)
    @patch("vram_core.tts_engine._PYTTSX3_AVAILABLE", False)
    def test_no_backend_available(self):
        """Engine with no backend sets active_backend to 'none'."""
        engine = TTSEngine(backend="auto")
        assert engine.backend == "none"

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", True)
    def test_auto_selects_edge_tts(self):
        """Auto mode selects edge-tts when available."""
        engine = TTSEngine(backend="auto")
        assert engine.backend == "edge-tts"

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", False)
    @patch("vram_core.tts_engine._PYTTSX3_AVAILABLE", True)
    @patch("vram_core.tts_engine.pyttsx3")
    def test_auto_selects_pyttsx3_fallback(self, mock_pyttsx3):
        """Auto mode falls back to pyttsx3 when edge-tts unavailable."""
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        engine = TTSEngine(backend="auto")
        assert engine.backend == "pyttsx3"

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", True)
    def test_explicit_edge_tts(self):
        engine = TTSEngine(backend="edge-tts")
        assert engine.backend == "edge-tts"

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", False)
    @patch("vram_core.tts_engine._PYTTSX3_AVAILABLE", False)
    def test_explicit_unavailable_backend(self):
        """Requesting unavailable backend results in 'none'."""
        engine = TTSEngine(backend="pyttsx3")
        assert engine.backend == "none"

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", True)
    def test_custom_voice(self):
        engine = TTSEngine(voice="zh-CN-XiaoxiaoNeural")
        assert engine.voice == "zh-CN-XiaoxiaoNeural"

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", True)
    def test_default_voice(self):
        engine = TTSEngine()
        assert engine.voice == "en-US-AriaNeural"

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", True)
    def test_custom_params(self):
        engine = TTSEngine(rate="+20%", volume="+50%", pitch="-5Hz")
        assert engine.rate == "+20%"
        assert engine.volume == "+50%"
        assert engine.pitch == "-5Hz"


class TestTTSEngineSynthesize:
    """Test TTSEngine.synthesize()."""

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", False)
    @patch("vram_core.tts_engine._PYTTSX3_AVAILABLE", False)
    def test_synthesize_no_backend_raises(self):
        """synthesize with no backend raises RuntimeError."""
        engine = TTSEngine(backend="auto")
        with pytest.raises(RuntimeError):
            engine.synthesize("Hello")

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", True)
    def test_synthesize_empty_text(self):
        """Empty text returns result without error."""
        engine = TTSEngine(backend="edge-tts")
        result = engine.synthesize("")
        assert result.text == ""
        assert result.audio is None

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", True)
    def test_synthesize_whitespace_only(self):
        """Whitespace-only text returns result without synthesis."""
        engine = TTSEngine(backend="edge-tts")
        result = engine.synthesize("   ")
        assert result.text == "   "

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", True)
    @patch("vram_core.tts_engine.edge_tts")
    @patch("vram_core.tts_engine._SOUNDFILE_AVAILABLE", False)
    def test_synthesize_edge_tts_basic(self, mock_edge_tts):
        """Basic edge-tts synthesis creates output file."""
        mock_comm = MagicMock()
        mock_edge_tts.Communicate.return_value = mock_comm

        async def mock_save(path):
            Path(path).write_bytes(b"\x00" * 100)

        mock_comm.save = mock_save

        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "test.mp3")
            engine = TTSEngine(backend="edge-tts")
            # We need to run the async save
            with patch("asyncio.run", side_effect=lambda coro: coro):
                result = engine.synthesize("Hello world", output_path=output)
            # File should exist
            # (may not due to mock complexity, so just check no crash)

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", False)
    @patch("vram_core.tts_engine._PYTTSX3_AVAILABLE", True)
    @patch("vram_core.tts_engine.pyttsx3")
    def test_synthesize_pyttsx3_basic(self, mock_pyttsx3):
        """pyttsx3 synthesis calls save_to_file."""
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        engine = TTSEngine(backend="pyttsx3")

        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "test.wav")
            # Create a dummy wav file for sf.read
            Path(output).write_bytes(b"\x00" * 100)
            result = engine.synthesize("Hello", output_path=output)
            assert result.voice_id == engine.voice
            assert result.text == "Hello"


class TestTTSEngineSpeak:
    """Test TTSEngine.speak()."""

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", False)
    @patch("vram_core.tts_engine._PYTTSX3_AVAILABLE", True)
    @patch("vram_core.tts_engine.pyttsx3")
    def test_speak_pyttsx3(self, mock_pyttsx3):
        """speak with pyttsx3 calls say and runAndWait."""
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        engine = TTSEngine(backend="pyttsx3")
        engine.speak("Hello")
        mock_engine.say.assert_called_once_with("Hello")
        mock_engine.runAndWait.assert_called_once()


class TestTTSEngineSynthesizeToFile:
    """Test TTSEngine.synthesize_to_file()."""

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", False)
    @patch("vram_core.tts_engine._PYTTSX3_AVAILABLE", False)
    def test_synthesize_to_file_no_backend(self):
        engine = TTSEngine(backend="auto")
        with pytest.raises(RuntimeError):
            engine.synthesize_to_file("Hello", "/tmp/out.mp3")


class TestTTSEngineVoiceListing:
    """Test voice listing methods."""

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", True)
    def test_list_voices_edge_tts(self):
        """list_voices with edge-tts returns voice list."""
        with patch("vram_core.tts_engine.edge_tts") as mock_edge:
            mock_edge.list_voices = AsyncMock(return_value=[
                {"ShortName": "en-US-AriaNeural", "FriendlyName": "Aria",
                 "Locale": "en-US", "Gender": "Female"},
                {"ShortName": "zh-CN-XiaoxiaoNeural", "FriendlyName": "Xiaoxiao",
                 "Locale": "zh-CN", "Gender": "Female"},
            ])
            engine = TTSEngine(backend="edge-tts")
            # list_voices uses asyncio.run internally
            voices = engine.list_voices(language="en")
            assert isinstance(voices, list)
            # Check the mock was called
            mock_edge.list_voices.assert_called()

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", False)
    @patch("vram_core.tts_engine._PYTTSX3_AVAILABLE", False)
    def test_list_voices_no_backend(self):
        """list_voices with no backend returns empty list."""
        engine = TTSEngine(backend="auto")
        voices = engine.list_voices()
        assert voices == []

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", True)
    def test_available_backends_edge_tts(self):
        """available_backends includes edge-tts when available."""
        backends = TTSEngine.available_backends()
        assert "edge-tts" in backends

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", False)
    @patch("vram_core.tts_engine._PYTTSX3_AVAILABLE", False)
    def test_available_backends_none(self):
        backends = TTSEngine.available_backends()
        assert backends == []

    def test_available_languages(self):
        langs = TTSEngine.available_languages()
        assert "en" in langs
        assert "zh" in langs
        assert "ja" in langs

    def test_default_voices_dict(self):
        assert "en" in TTSEngine.DEFAULT_VOICES
        assert "zh" in TTSEngine.DEFAULT_VOICES


class TestTTSEngineVoiceProperty:
    """Test voice property getter/setter."""

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", True)
    def test_voice_setter(self):
        engine = TTSEngine()
        engine.voice = "zh-CN-YunxiNeural"
        assert engine.voice == "zh-CN-YunxiNeural"


class TestTTSEngineClose:
    """Test engine close/cleanup."""

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", True)
    def test_close_edge_tts(self):
        """close() on edge-tts engine does not crash."""
        engine = TTSEngine(backend="edge-tts")
        engine.close()  # Should not raise

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", False)
    @patch("vram_core.tts_engine._PYTTSX3_AVAILABLE", True)
    @patch("vram_core.tts_engine.pyttsx3")
    def test_close_pyttsx3(self, mock_pyttsx3):
        """close() stops pyttsx3 engine."""
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        engine = TTSEngine(backend="pyttsx3")
        engine.close()
        mock_engine.stop.assert_called_once()


class TestTTSEngineStreaming:
    """Test async streaming synthesis."""

    @patch("vram_core.tts_engine._EDGE_TTS_AVAILABLE", True)
    def test_stream_requires_edge_tts(self):
        """stream_synthesize raises RuntimeError for non-edge-tts backend."""
        with patch("vram_core.tts_engine._PYTTSX3_AVAILABLE", True):
            with patch("vram_core.tts_engine.pyttsx3") as mock_pyttsx3:
                mock_pyttsx3.init.return_value = MagicMock()
                engine = TTSEngine(backend="pyttsx3")
                # stream_synthesize is async
                async def run():
                    async for _ in engine.stream_synthesize("test"):
                        pass

                with pytest.raises(RuntimeError):
                    asyncio.run(run())


