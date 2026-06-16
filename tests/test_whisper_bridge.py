"""
Unit Tests for vram_core Whisper Bridge
========================================

Tests Whisper backend integration, transcription, language detection,
and backend auto-selection logic.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vram_core.whisper_bridge import (
    WhisperBridge,
    WhisperBackend,
    TranscriptionResult,
)


class TestWhisperBackendEnum:
    """Test WhisperBackend enum values."""

    def test_backend_values(self):
        """Test enum values are correct."""
        assert WhisperBackend.WHISPER_CPP.value == "whisper_cpp"
        assert WhisperBackend.OPENAI_API.value == "openai_api"
        assert WhisperBackend.AUTO.value == "auto"

    def test_backend_from_value(self):
        """Test creating backend from string value."""
        assert WhisperBackend("whisper_cpp") == WhisperBackend.WHISPER_CPP
        assert WhisperBackend("openai_api") == WhisperBackend.OPENAI_API


class TestTranscriptionResult:
    """Test TranscriptionResult data class."""

    def test_creation(self):
        """Test basic creation."""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            segments=[{"start": 0.0, "end": 1.0, "text": "Hello world"}],
            backend=WhisperBackend.OPENAI_API,
            duration=1.0,
            processing_time=0.5,
        )
        assert result.text == "Hello world"
        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.backend == WhisperBackend.OPENAI_API
        assert result.duration == 1.0
        assert result.processing_time == 0.5

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = TranscriptionResult(
            text="Test",
            language="zh",
            backend=WhisperBackend.WHISPER_CPP,
        )
        d = result.to_dict()
        assert d["text"] == "Test"
        assert d["language"] == "zh"
        assert d["backend"] == "whisper_cpp"
        assert d["segments"] == []
        assert d["duration"] == 0.0

    def test_repr(self):
        """Test string representation."""
        result = TranscriptionResult(text="Hello", language="en")
        repr_str = repr(result)
        assert "TranscriptionResult" in repr_str
        assert "Hello" in repr_str
        assert "en" in repr_str

    def test_default_segments(self):
        """Test default empty segments."""
        result = TranscriptionResult(text="", language="")
        assert result.segments == []


class TestWhisperBridgeInit:
    """Test WhisperBridge initialization."""

    def test_default_init(self):
        """Test default initialization."""
        with patch.object(WhisperBridge, '_auto_detect_backend', return_value=WhisperBackend.WHISPER_CPP):
            bridge = WhisperBridge(backend=WhisperBackend.WHISPER_CPP)
            assert bridge.backend == WhisperBackend.WHISPER_CPP
            assert bridge.whisper_model == "base"
            assert bridge.device == "cuda"

    def test_init_with_openai_key(self):
        """Test initialization with OpenAI API key."""
        bridge = WhisperBridge(
            backend=WhisperBackend.OPENAI_API,
            openai_api_key="test-key-123",
        )
        assert bridge.backend == WhisperBackend.OPENAI_API
        assert bridge.openai_api_key == "test-key-123"

    def test_init_with_env_api_key(self):
        """Test initialization picks up environment API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-456"}):
            bridge = WhisperBridge(
                backend=WhisperBackend.OPENAI_API,
            )
            assert bridge.openai_api_key == "env-key-456"

    def test_init_with_language(self):
        """Test initialization with forced language."""
        bridge = WhisperBridge(
            backend=WhisperBackend.OPENAI_API,
            openai_api_key="test",
            language="zh",
        )
        assert bridge.language == "zh"

    def test_supported_languages(self):
        """Test that supported languages are defined."""
        assert "zh" in WhisperBridge.SUPPORTED_LANGUAGES
        assert "en" in WhisperBridge.SUPPORTED_LANGUAGES
        assert "ja" in WhisperBridge.SUPPORTED_LANGUAGES
        assert WhisperBridge.SUPPORTED_LANGUAGES["zh"] == "Chinese"
        assert WhisperBridge.SUPPORTED_LANGUAGES["en"] == "English"
        assert WhisperBridge.SUPPORTED_LANGUAGES["ja"] == "Japanese"


class TestWhisperBridgeAutoDetect:
    """Test backend auto-detection logic."""

    def test_auto_detect_prefers_faster_whisper(self):
        """Test that auto-detect prefers faster-whisper when available."""
        with patch.object(WhisperBridge, '_check_faster_whisper', return_value=True), \
             patch.object(WhisperBridge, '_check_whisper_cpp', return_value=False):
            bridge = WhisperBridge(backend=WhisperBackend.AUTO)
            assert bridge.backend == WhisperBackend.FASTER_WHISPER

    def test_auto_detect_falls_back_to_openai(self):
        """Test that auto-detect falls back to OpenAI API."""
        with patch.object(WhisperBridge, '_check_faster_whisper', return_value=False), \
             patch.object(WhisperBridge, '_check_whisper_cpp', return_value=False):
            bridge = WhisperBridge(
                backend=WhisperBackend.AUTO,
                openai_api_key="test-key",
            )
            assert bridge.backend == WhisperBackend.OPENAI_API

    def test_auto_detect_defaults_to_whisper_cpp(self):
        """Test that auto-detect defaults to whisper.cpp when nothing available."""
        with patch.object(WhisperBridge, '_check_faster_whisper', return_value=False), \
             patch.object(WhisperBridge, '_check_whisper_cpp', return_value=False):
            bridge = WhisperBridge(backend=WhisperBackend.AUTO)
            assert bridge.backend == WhisperBackend.WHISPER_CPP

    def test_check_whisper_cpp_with_path(self):
        """Test whisper.cpp detection with explicit path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake binary
            exe_path = Path(tmpdir) / "main.exe"
            exe_path.touch()

            bridge = WhisperBridge(
                backend=WhisperBackend.WHISPER_CPP,
                whisper_cpp_path=tmpdir,
            )
            assert bridge._check_whisper_cpp()

    def test_check_whisper_cpp_missing_path(self):
        """Test whisper.cpp detection with missing path."""
        bridge = WhisperBridge(
            backend=WhisperBackend.WHISPER_CPP,
            whisper_cpp_path="/nonexistent/path",
        )
        with patch('shutil.which', return_value=None):
            assert not bridge._check_whisper_cpp()


class TestWhisperBridgeGetAvailableBackends:
    """Test available backends listing."""

    def test_no_backends_available(self):
        """Test when no backends are available."""
        with patch.object(WhisperBridge, '_check_faster_whisper', return_value=False), \
             patch.object(WhisperBridge, '_check_whisper_cpp', return_value=False):
            bridge = WhisperBridge(
                backend=WhisperBackend.WHISPER_CPP,
                openai_api_key=None,
            )
            available = bridge.get_available_backends()
            assert len(available) == 0

    def test_only_openai_available(self):
        """Test when only OpenAI API is available."""
        with patch.object(WhisperBridge, '_check_faster_whisper', return_value=False), \
             patch.object(WhisperBridge, '_check_whisper_cpp', return_value=False):
            bridge = WhisperBridge(
                backend=WhisperBackend.OPENAI_API,
                openai_api_key="test-key",
            )
            available = bridge.get_available_backends()
            assert len(available) == 1
            assert WhisperBackend.OPENAI_API in available

    def test_all_backends_available(self):
        """Test when all backends are available."""
        with patch.object(WhisperBridge, '_check_faster_whisper', return_value=True), \
             patch.object(WhisperBridge, '_check_whisper_cpp', return_value=True):
            bridge = WhisperBridge(
                backend=WhisperBackend.WHISPER_CPP,
                openai_api_key="test-key",
            )
            available = bridge.get_available_backends()
            assert len(available) == 3
            assert WhisperBackend.FASTER_WHISPER in available
            assert WhisperBackend.WHISPER_CPP in available
            assert WhisperBackend.OPENAI_API in available


class TestWhisperBridgeSegmentParsing:
    """Test whisper.cpp segment parsing."""

    def test_parse_segments(self):
        """Test parsing segments from stderr output."""
        stderr = (
            "whisper_init_from_file: loading model from 'models/ggml-base.bin'\n"
            "[00:00:00.000 --> 00:00:02.500] Hello, how are you?\n"
            "[00:00:02.500 --> 00:00:05.100] I'm doing well, thank you.\n"
            "[00:00:05.100 --> 00:00:08.000] That's great to hear.\n"
        )
        segments = WhisperBridge._parse_whisper_cpp_segments(stderr)
        assert len(segments) == 3
        assert segments[0]["start"] == "00:00:00.000"
        assert segments[0]["end"] == "00:00:02.500"
        assert segments[0]["text"] == "Hello, how are you?"

    def test_parse_empty_stderr(self):
        """Test parsing empty stderr."""
        segments = WhisperBridge._parse_whisper_cpp_segments("")
        assert len(segments) == 0

    def test_parse_no_segments_stderr(self):
        """Test parsing stderr with no segment lines."""
        stderr = "whisper_init_from_file: loading model\nsome other output\n"
        segments = WhisperBridge._parse_whisper_cpp_segments(stderr)
        assert len(segments) == 0


class TestWhisperBridgeTranscribeOpenAI:
    """Test OpenAI API transcription."""

    def test_transcribe_openai_success(self):
        """Test successful OpenAI API transcription."""
        # Create a mock openai module
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Hello, this is a test."
        mock_response.language = "en"
        mock_response.segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello, this is a test."}
        ]
        mock_client.audio.transcriptions.create.return_value = mock_response

        # Temporarily inject mock into sys.modules
        import sys
        old_openai = sys.modules.get('openai')
        sys.modules['openai'] = mock_openai
        try:
            bridge = WhisperBridge(
                backend=WhisperBackend.OPENAI_API,
                openai_api_key="test-key",
            )

            audio = np.random.randn(16000).astype(np.float32) * 0.1
            result = bridge.transcribe(audio, sample_rate=16000)

            assert result.text == "Hello, this is a test."
            assert result.language == "en"
            assert result.backend == WhisperBackend.OPENAI_API
            assert len(result.segments) == 1
        finally:
            if old_openai is not None:
                sys.modules['openai'] = old_openai
            else:
                sys.modules.pop('openai', None)

    def test_transcribe_openai_no_key(self):
        """Test OpenAI API transcription fails without key (ImportError or ValueError)."""
        import sys
        # Create a mock openai module so ImportError is not raised
        mock_openai = MagicMock()
        old_openai = sys.modules.get('openai')
        sys.modules['openai'] = mock_openai
        try:
            bridge = WhisperBridge(
                backend=WhisperBackend.OPENAI_API,
                openai_api_key=None,
            )
            bridge.openai_api_key = None

            audio = np.random.randn(16000).astype(np.float32) * 0.1
            with pytest.raises(ValueError):
                bridge.transcribe(audio, sample_rate=16000)
        finally:
            if old_openai is not None:
                sys.modules['openai'] = old_openai
            else:
                sys.modules.pop('openai', None)


class TestWhisperBridgeTranscribeWithFileInput:
    """Test transcription with file input."""

    def _create_temp_wav(self, sample_rate=16000, duration_s=1.0):
        """Create a temporary WAV file."""
        import struct as st
        n_samples = int(sample_rate * duration_s)
        samples = np.random.randint(-32768, 32767, size=n_samples, dtype=np.int16)
        data_size = len(samples) * 2
        header = st.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + data_size, b"WAVE", b"fmt ",
            16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
            b"data", data_size,
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(header)
        tmp.write(samples.tobytes())
        tmp.close()
        return tmp.name

    @patch.object(WhisperBridge, '_transcribe_openai_api')
    def test_transcribe_from_file(self, mock_transcribe):
        """Test transcription from file path."""
        mock_transcribe.return_value = TranscriptionResult(
            text="test result",
            language="en",
        )

        bridge = WhisperBridge(
            backend=WhisperBackend.OPENAI_API,
            openai_api_key="test",
        )

        wav_path = self._create_temp_wav()
        try:
            result = bridge.transcribe(wav_path)
            assert result.text == "test result"
            # Verify audio was loaded (mock was called)
            mock_transcribe.assert_called_once()
        finally:
            os.unlink(wav_path)


class TestWhisperBridgeTranscribeStream:
    """Test stream transcription."""

    @patch.object(WhisperBridge, 'transcribe')
    def test_transcribe_stream(self, mock_transcribe):
        """Test transcribing from multiple chunks."""
        mock_transcribe.return_value = TranscriptionResult(
            text="stream result",
            language="en",
        )

        bridge = WhisperBridge(
            backend=WhisperBackend.OPENAI_API,
            openai_api_key="test",
        )

        chunks = [
            np.random.randn(8000).astype(np.float32),
            np.random.randn(8000).astype(np.float32),
        ]

        result = bridge.transcribe_stream(chunks, sample_rate=16000)
        assert result.text == "stream result"
        # Verify transcribe was called with concatenated audio
        call_args = mock_transcribe.call_args
        audio_arg = call_args[0][0]
        assert len(audio_arg) == 16000


class TestWhisperBridgeLanguageDetection:
    """Test language detection."""

    @patch.object(WhisperBridge, 'transcribe')
    def test_detect_language(self, mock_transcribe):
        """Test language detection from audio."""
        mock_transcribe.return_value = TranscriptionResult(
            text="浣犲ソ涓栫晫",
            language="zh",
        )

        bridge = WhisperBridge(
            backend=WhisperBackend.OPENAI_API,
            openai_api_key="test",
        )

        audio = np.random.randn(32000).astype(np.float32) * 0.1
        lang = bridge.detect_language(audio, sample_rate=16000)
        assert lang == "zh"

    @patch.object(WhisperBridge, 'transcribe')
    def test_detect_language_truncates_long_audio(self, mock_transcribe):
        """Test that long audio is truncated for language detection."""
        mock_transcribe.return_value = TranscriptionResult(
            text="test",
            language="en",
        )

        bridge = WhisperBridge(
            backend=WhisperBackend.OPENAI_API,
            openai_api_key="test",
        )

        # 60 seconds of audio
        audio = np.random.randn(960000).astype(np.float32) * 0.1
        bridge.detect_language(audio, sample_rate=16000)

        # Verify audio was truncated to 30 seconds
        call_args = mock_transcribe.call_args
        audio_arg = call_args[0][0]
        assert len(audio_arg) <= 30 * 16000


