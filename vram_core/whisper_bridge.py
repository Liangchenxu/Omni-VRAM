"""
Whisper Backend Bridge for Omni-VRAM
=====================================

Provides a unified interface for speech-to-text transcription using
either local whisper.cpp or OpenAI's Whisper API. Automatically selects
the best available backend.

Architecture:
    - WhisperBackend: Enum for backend selection
    - WhisperBridge: Main class for transcription
"""

import os
import time
import logging
import subprocess
import shutil
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import numpy as np

from vram_core.audio_utils import AudioProcessor

logger = logging.getLogger(__name__)


class WhisperBackend(Enum):
    """Supported Whisper backends."""
    WHISPER_CPP = "whisper_cpp"
    OPENAI_API = "openai_api"
    AUTO = "auto"


class TranscriptionResult:
    """Container for transcription results."""

    def __init__(
        self,
        text: str,
        language: str,
        segments: Optional[List[Dict[str, Any]]] = None,
        backend: Optional[WhisperBackend] = None,
        duration: float = 0.0,
        processing_time: float = 0.0,
    ):
        self.text = text
        self.language = language
        self.segments = segments or []
        self.backend = backend
        self.duration = duration
        self.processing_time = processing_time

    def __repr__(self) -> str:
        return (
            f"TranscriptionResult(text='{self.text[:50]}...', "
            f"language='{self.language}', "
            f"backend={self.backend}, "
            f"duration={self.duration:.2f}s, "
            f"processing_time={self.processing_time:.2f}s)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "text": self.text,
            "language": self.language,
            "segments": self.segments,
            "backend": self.backend.value if self.backend else None,
            "duration": self.duration,
            "processing_time": self.processing_time,
        }


class WhisperBridge:
    """
    Unified Whisper transcription bridge.

    Supports multiple backends:
        1. whisper.cpp (local, GPU-accelerated)
        2. OpenAI Whisper API (cloud)

    Usage:
        bridge = WhisperBridge()
        result = bridge.transcribe("audio.wav")
        print(result.text)
    """

    # Supported languages for auto-detection
    SUPPORTED_LANGUAGES = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "pt": "Portuguese",
        "ru": "Russian",
        "ar": "Arabic",
    }

    def __init__(
        self,
        backend: WhisperBackend = WhisperBackend.AUTO,
        whisper_cpp_path: Optional[str] = None,
        whisper_model: str = "base",
        openai_api_key: Optional[str] = None,
        openai_model: str = "whisper-1",
        language: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize WhisperBridge.

        Args:
            backend: Which backend to use (AUTO will select best available).
            whisper_cpp_path: Path to whisper.cpp binary directory.
            whisper_model: Model size for whisper.cpp (tiny/base/small/medium/large).
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var).
            openai_model: OpenAI Whisper model name.
            language: Force specific language (None for auto-detect).
            device: Device for whisper.cpp (cuda/cpu).
        """
        self.backend = backend
        self.whisper_cpp_path = whisper_cpp_path
        self.whisper_model = whisper_model
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.openai_model = openai_model
        self.language = language
        self.device = device

        # Audio processor for format conversion
        self.audio_processor = AudioProcessor(target_sample_rate=16000)

        # Auto-detect backend
        if self.backend == WhisperBackend.AUTO:
            self.backend = self._auto_detect_backend()
            logger.info("Auto-detected backend: %s", self.backend.value)

    def _auto_detect_backend(self) -> WhisperBackend:
        """
        Automatically detect the best available backend.

        Priority:
            1. whisper.cpp (local, faster for GPU)
            2. OpenAI API (cloud, requires API key)

        Returns:
            Best available WhisperBackend.
        """
        # Check for whisper.cpp
        if self._check_whisper_cpp():
            return WhisperBackend.WHISPER_CPP

        # Check for OpenAI API key
        if self.openai_api_key:
            return WhisperBackend.OPENAI_API

        # Default to whisper.cpp (will fail gracefully if not found)
        logger.warning(
            "No Whisper backend found. Install whisper.cpp or set OPENAI_API_KEY."
        )
        return WhisperBackend.WHISPER_CPP

    def _check_whisper_cpp(self) -> bool:
        """Check if whisper.cpp is available."""
        if self.whisper_cpp_path:
            main_exe = Path(self.whisper_cpp_path) / "main.exe"
            if main_exe.exists():
                return True

        # Check PATH
        if shutil.which("whisper") or shutil.which("main"):
            return True

        return False

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        sample_rate: int = 16000,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio_input: File path or numpy array (float32, mono, 16kHz).
            sample_rate: Sample rate if audio_input is numpy array.
            **kwargs: Additional backend-specific options.

        Returns:
            TranscriptionResult with text, language, segments, etc.
        """
        start_time = time.time()

        # Load audio if file path
        if isinstance(audio_input, (str, Path)):
            audio_data, sr = self.audio_processor.load(audio_input)
            audio_duration = len(audio_data) / sr
        else:
            audio_data = audio_input
            sr = sample_rate
            audio_duration = len(audio_data) / sr

        # Ensure float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Route to backend
        if self.backend == WhisperBackend.WHISPER_CPP:
            result = self._transcribe_whisper_cpp(audio_data, sr, **kwargs)
        elif self.backend == WhisperBackend.OPENAI_API:
            result = self._transcribe_openai_api(audio_data, sr, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        result.duration = audio_duration
        result.processing_time = time.time() - start_time
        result.backend = self.backend

        logger.info(
            "Transcribed %.2fs audio in %.2fs using %s",
            result.duration, result.processing_time, self.backend.value,
        )

        return result

    def transcribe_stream(
        self,
        audio_chunks: List[np.ndarray],
        sample_rate: int = 16000,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe from a stream of audio chunks.

        Args:
            audio_chunks: List of audio chunks (float32 numpy arrays).
            sample_rate: Sample rate of chunks.
            **kwargs: Additional backend-specific options.

        Returns:
            TranscriptionResult with aggregated transcription.
        """
        # Concatenate chunks
        full_audio = np.concatenate(audio_chunks)
        return self.transcribe(full_audio, sample_rate=sample_rate, **kwargs)

    # ------------------------------------------------------------------
    # Whisper.cpp Backend
    # ------------------------------------------------------------------

    def _transcribe_whisper_cpp(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe using whisper.cpp.

        Args:
            audio: Float32 mono audio array.
            sample_rate: Sample rate.
            **kwargs: Additional whisper.cpp options.

        Returns:
            TranscriptionResult.
        """
        import tempfile

        # Find whisper.cpp binary
        main_exe = self._find_whisper_cpp_binary()

        # Write audio to temporary WAV file
        wav_bytes = AudioProcessor.to_wav_bytes(audio, sample_rate)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        try:
            # Build command
            cmd = [
                main_exe,
                "-m", self._get_model_path(),
                "-f", tmp_path,
                "--output-txt",
                "--no-prints",
            ]

            if self.language:
                cmd.extend(["-l", self.language])

            if self.device == "cuda":
                cmd.extend(["--gpu-layers", "1"])

            # Add extra args
            for key, value in kwargs.items():
                cmd.extend([f"--{key}", str(value)])

            logger.debug("Running: %s", " ".join(cmd))

            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"whisper.cpp failed with code {result.returncode}: {result.stderr}"
                )

            # Parse output
            text = result.stdout.strip()
            language = self.language or "unknown"

            # Try to detect language from output
            for lang_code, lang_name in self.SUPPORTED_LANGUAGES.items():
                if lang_name.lower() in result.stderr.lower():
                    language = lang_code
                    break

            return TranscriptionResult(
                text=text,
                language=language,
                segments=self._parse_whisper_cpp_segments(result.stderr),
            )

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _find_whisper_cpp_binary(self) -> str:
        """Find the whisper.cpp main executable."""
        if self.whisper_cpp_path:
            main_exe = Path(self.whisper_cpp_path) / "main.exe"
            if main_exe.exists():
                return str(main_exe)
            # Try without .exe
            main_exe = Path(self.whisper_cpp_path) / "main"
            if main_exe.exists():
                return str(main_exe)

        # Check PATH
        for name in ["whisper", "main", "whisper-cli"]:
            path = shutil.which(name)
            if path:
                return path

        raise FileNotFoundError(
            "whisper.cpp binary not found. "
            "Set whisper_cpp_path or add to PATH. "
            "See: https://github.com/ggerganov/whisper.cpp"
        )

    def _get_model_path(self) -> str:
        """Get path to whisper model file."""
        # Check common model locations
        model_names = [
            f"ggml-{self.whisper_model}.bin",
            f"ggml-{self.whisper_model}.en.bin",
        ]

        # Check whisper_cpp_path/models/
        if self.whisper_cpp_path:
            for name in model_names:
                model_path = Path(self.whisper_cpp_path) / "models" / name
                if model_path.exists():
                    return str(model_path)

        # Check environment variable
        env_path = os.environ.get("WHISPER_MODEL_PATH")
        if env_path:
            return env_path

        # Check current directory
        for name in model_names:
            if Path(name).exists():
                return name

        raise FileNotFoundError(
            f"Whisper model '{self.whisper_model}' not found. "
            f"Expected: {model_names}"
        )

    @staticmethod
    def _parse_whisper_cpp_segments(stderr: str) -> List[Dict[str, Any]]:
        """Parse segments from whisper.cpp stderr output."""
        segments = []
        import re

        # Pattern: [HH:MM:SS.mmm --> HH:MM:SS.mmm] text
        pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)'

        for line in stderr.split('\n'):
            match = re.search(pattern, line)
            if match:
                segments.append({
                    "start": match.group(1),
                    "end": match.group(2),
                    "text": match.group(3).strip(),
                })

        return segments

    # ------------------------------------------------------------------
    # OpenAI API Backend
    # ------------------------------------------------------------------

    def _transcribe_openai_api(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe using OpenAI Whisper API.

        Args:
            audio: Float32 mono audio array.
            sample_rate: Sample rate.
            **kwargs: Additional API options.

        Returns:
            TranscriptionResult.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI API backend. "
                "Install with: pip install openai"
            )

        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key not set. "
                "Set OPENAI_API_KEY environment variable or pass openai_api_key."
            )

        client = openai.OpenAI(api_key=self.openai_api_key)

        # Encode audio to WAV bytes
        wav_bytes = AudioProcessor.to_wav_bytes(audio, sample_rate)

        # Prepare request
        request_kwargs = {
            "model": self.openai_model,
            "file": ("audio.wav", wav_bytes, "audio/wav"),
            "response_format": "verbose_json",
        }

        if self.language:
            request_kwargs["language"] = self.language

        # Add extra kwargs
        for key, value in kwargs.items():
            if key not in request_kwargs:
                request_kwargs[key] = value

        logger.debug("Calling OpenAI Whisper API...")

        # Make API call
        response = client.audio.transcriptions.create(**request_kwargs)

        # Parse response
        text = response.text
        language = getattr(response, "language", self.language or "unknown")

        segments = []
        if hasattr(response, "segments") and response.segments:
            for seg in response.segments:
                segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                })

        return TranscriptionResult(
            text=text,
            language=language,
            segments=segments,
        )

    # ------------------------------------------------------------------
    # Language Detection
    # ------------------------------------------------------------------

    def detect_language(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Detect the language of audio content.

        Uses whisper.cpp's language detection or falls back to
        OpenAI API's auto-detection.

        Args:
            audio: Float32 mono audio array.
            sample_rate: Sample rate.

        Returns:
            ISO 639-1 language code (e.g. 'zh', 'en', 'ja').
        """
        # Use a short segment for detection
        max_duration = 30  # seconds
        max_samples = max_duration * sample_rate
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        result = self.transcribe(
            audio,
            sample_rate=sample_rate,
            language=None,  # Force auto-detect
        )

        return result.language

    def get_available_backends(self) -> List[WhisperBackend]:
        """
        List all available backends.

        Returns:
            List of available WhisperBackend values.
        """
        available = []

        if self._check_whisper_cpp():
            available.append(WhisperBackend.WHISPER_CPP)

        if self.openai_api_key:
            available.append(WhisperBackend.OPENAI_API)

        return available