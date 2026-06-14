"""
Whisper Backend Bridge for Omni-VRAM
=====================================

Provides a unified interface for speech-to-text transcription using
either local whisper.cpp or OpenAI's Whisper API. Automatically selects
the best available backend.

Architecture:
    - WhisperBackend: Enum for backend selection
    - WhisperResult: Dataclass for transcription results with SRT export
    - WhisperBridge: Main class for transcription

Audio Preprocessing:
    Uses pydub for format conversion, supporting wav/mp3/flac/ogg/m4a.
    Automatically converts to 16kHz mono WAV for whisper consumption.
"""

import os
import re
import time
import logging
import subprocess
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field

import numpy as np

from vram_core.config import config

logger = logging.getLogger(__name__)

# ── Supported audio formats ──────────────────────────────────────
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac"}


class WhisperBackend(Enum):
    """Supported Whisper backends."""
    FASTER_WHISPER = "faster_whisper"
    WHISPER_CPP = "whisper_cpp"
    OPENAI_API = "openai_api"
    DISTIL_WHISPER = "distil_whisper"
    AUTO = "auto"


# All supported Whisper model sizes with metadata
WHISPER_MODELS = {
    "tiny": {"params": "39M", "vram": "~1GB", "speed": "fastest", "quality": "low"},
    "base": {"params": "74M", "vram": "~1GB", "speed": "very fast", "quality": "basic"},
    "small": {"params": "244M", "vram": "~2GB", "speed": "fast", "quality": "good"},
    "medium": {"params": "769M", "vram": "~5GB", "speed": "moderate", "quality": "very good"},
    "large": {"params": "1550M", "vram": "~10GB", "speed": "slow", "quality": "excellent"},
    "large-v2": {"params": "1550M", "vram": "~10GB", "speed": "slow", "quality": "excellent"},
    "large-v3": {"params": "1550M", "vram": "~10GB", "speed": "slow", "quality": "best"},
    "turbo": {"params": "809M", "vram": "~6GB", "speed": "very fast", "quality": "excellent"},
}

# Distil-Whisper models (faster, smaller, English-focused)
DISTIL_WHISPER_MODELS = {
    "distil-small.en": {"base": "small", "vram": "~1.5GB", "speed": "6x faster", "lang": "en"},
    "distil-medium.en": {"base": "medium", "vram": "~3.5GB", "speed": "6x faster", "lang": "en"},
    "distil-large-v2": {"base": "large-v2", "vram": "~6GB", "speed": "6x faster", "lang": "en"},
    "distil-large-v3": {"base": "large-v3", "vram": "~6GB", "speed": "6x faster", "lang": "en+multi"},
}

# Compute type precision options
COMPUTE_TYPES = {
    "float32": {"precision": "full", "speed": "slowest", "quality": "best"},
    "float16": {"precision": "half", "speed": "fast", "quality": "best", "gpu_only": True},
    "int8": {"precision": "integer", "speed": "fastest", "quality": "good"},
    "int4": {"precision": "4-bit", "speed": "ultra-fast", "quality": "acceptable", "experimental": True},
}


@dataclass
class WhisperResult:
    """
    Container for Whisper transcription results.

    Attributes:
        text: Full transcription text.
        language: Detected or specified language code.
        confidence: Average confidence score (0.0-1.0, if available).
        segments: List of segments with timestamps.
        backend: Which backend was used.
        audio_duration: Duration of input audio in seconds.
        processing_time: Time taken for transcription in seconds.
    """
    text: str = ""
    language: str = "unknown"
    confidence: float = 0.0
    segments: List[Dict[str, Any]] = field(default_factory=list)
    backend: Optional[WhisperBackend] = None
    audio_duration: float = 0.0
    processing_time: float = 0.0

    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (
            f"WhisperResult(text='{preview}', "
            f"language='{self.language}', "
            f"confidence={self.confidence:.2f}, "
            f"segments={len(self.segments)}, "
            f"backend={self.backend})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "segments": self.segments,
            "backend": self.backend.value if self.backend else None,
            "audio_duration": self.audio_duration,
            "processing_time": self.processing_time,
        }

    def export_srt(self, output_path: Union[str, Path]) -> Path:
        """
        Export transcription segments as SRT subtitle file.

        Args:
            output_path: Path to write the .srt file.

        Returns:
            Path to the created file.

        Raises:
            ValueError: If no segments with timestamps are available.
        """
        if not self.segments:
            raise ValueError(
                "No segments available for SRT export. "
                "Segments with timestamps are required."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for i, seg in enumerate(self.segments, start=1):
            start = self._format_srt_time(seg.get("start", 0.0))
            end = self._format_srt_time(seg.get("end", 0.0))
            text = seg.get("text", "").strip()
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")  # blank line separator

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"SRT exported to {output_path} ({len(self.segments)} segments)")
        return output_path

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """
        Format seconds to SRT time format: HH:MM:SS,mmm

        Args:
            seconds: Time in seconds (float).

        Returns:
            Formatted time string.
        """
        # Handle string input (already formatted)
        if isinstance(seconds, str):
            # Convert "HH:MM:SS.mmm" to "HH:MM:SS,mmm"
            return seconds.replace(".", ",")

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# Backward compatibility alias
TranscriptionResult = WhisperResult


class AudioPreprocessor:
    """
    Audio preprocessing using pydub.

    Converts various audio formats to 16kHz mono WAV suitable for
    whisper consumption.

    Supported formats: wav, mp3, flac, ogg, m4a, wma, aac
    """

    @staticmethod
    def load_and_convert(
        audio_path: Union[str, Path],
        target_sample_rate: int = 16000,
    ) -> tuple:
        """
        Load audio file and convert to 16kHz mono WAV.

        Args:
            audio_path: Path to audio file.
            target_sample_rate: Target sample rate (default: 16000).

        Returns:
            Tuple of (audio_data as float32 numpy array, sample_rate).

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            ValueError: If format is not supported.
        """
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                "pydub package required for audio preprocessing. "
                "Install with: pip install pydub\n"
                "Note: MP3/OGG/M4A also require ffmpeg."
            )

        audio_path = Path(audio_path)

        # Validate file exists
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Validate format
        suffix = audio_path.suffix.lower()
        if suffix not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported audio format: '{suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}"
            )

        logger.debug(f"Loading audio: {audio_path} (format: {suffix})")

        # Load with pydub (handles format detection automatically)
        if suffix == ".wav":
            audio_segment = AudioSegment.from_wav(str(audio_path))
        elif suffix == ".mp3":
            audio_segment = AudioSegment.from_mp3(str(audio_path))
        elif suffix == ".flac":
            audio_segment = AudioSegment.from_file(str(audio_path), format="flac")
        elif suffix == ".ogg":
            audio_segment = AudioSegment.from_ogg(str(audio_path))
        elif suffix == ".m4a":
            audio_segment = AudioSegment.from_file(str(audio_path), format="m4a")
        else:
            # Generic fallback
            audio_segment = AudioSegment.from_file(str(audio_path))

        # Convert to mono
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
            logger.debug("Converted stereo to mono")

        # Resample to target sample rate
        if audio_segment.frame_rate != target_sample_rate:
            audio_segment = audio_segment.set_frame_rate(target_sample_rate)
            logger.debug(f"Resampled to {target_sample_rate}Hz")

        # Convert to 16-bit PCM
        audio_segment = audio_segment.set_sample_width(2)  # 16-bit

        # Convert to numpy float32
        samples = np.frombuffer(audio_segment.raw_data, dtype=np.int16)
        audio_data = samples.astype(np.float32) / 32768.0

        logger.debug(
            f"Audio loaded: {len(audio_data)} samples, "
            f"{len(audio_data) / target_sample_rate:.2f}s, "
            f"{target_sample_rate}Hz mono"
        )

        return audio_data, target_sample_rate

    @staticmethod
    def to_wav_bytes(
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ) -> bytes:
        """
        Convert numpy audio array to WAV bytes.

        Args:
            audio_data: Float32 audio array.
            sample_rate: Sample rate.

        Returns:
            WAV file bytes.
        """
        try:
            from pydub import AudioSegment
            from pydub.utils import make_chunks

            # Convert float32 to int16
            int16_data = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)

            # Create AudioSegment
            segment = AudioSegment(
                data=int16_data.tobytes(),
                sample_width=2,
                frame_rate=sample_rate,
                channels=1,
            )

            # Export to bytes
            buffer = tempfile.SpooledTemporaryFile(max_size=10 * 1024 * 1024)
            segment.export(buffer, format="wav")
            buffer.seek(0)
            return buffer.read()

        except ImportError:
            # Fallback: manual WAV encoding
            return AudioPreprocessor._manual_wav_bytes(audio_data, sample_rate)

    @staticmethod
    def _manual_wav_bytes(audio_data: np.ndarray, sample_rate: int) -> bytes:
        """Manual WAV encoding fallback when pydub is not available."""
        import struct

        int16_data = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
        data = int16_data.tobytes()
        data_size = len(data)

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            1,
            sample_rate,
            sample_rate * 2,
            2,
            16,
            b"data",
            data_size,
        )

        return header + data


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

        # Export subtitles
        result.export_srt("output.srt")
    """

    # Supported languages
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

    # Common whisper.cpp installation paths (Windows)
    COMMON_WHISPER_PATHS = [
        r"C:\whisper.cpp\build\bin\Release",
        r"C:\whisper.cpp\build\bin",
        r"C:\Program Files\whisper.cpp\bin",
        r"C:\tools\whisper.cpp\build\bin\Release",
        os.path.expanduser(r"~\whisper.cpp\build\bin\Release"),
        os.path.expanduser(r"~\whisper.cpp\build\bin"),
    ]

    # Transcription timeout in seconds
    TRANSCRIPTION_TIMEOUT = 300

    def __init__(
        self,
        backend: WhisperBackend = WhisperBackend.AUTO,
        whisper_cpp_path: Optional[str] = None,
        whisper_model: str = "base",
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        """
        Initialize WhisperBridge.

        Args:
            backend: Which backend to use (AUTO will select best available).
            whisper_cpp_path: Path to whisper.cpp binary directory.
            whisper_model: Model size (tiny/base/small/medium/large).
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var).
            openai_model: OpenAI Whisper model name.
            language: Force specific language (None for auto-detect).
            device: Device for whisper.cpp (cuda/cpu).
            compute_type: CTranslate2 compute type for faster-whisper
                          (int8/float16/float32, default: float16).

        Configuration priority:
            1. Constructor arguments (highest)
            2. Environment variables / .env file (via config)
            3. Default values (lowest)
        """
        # Use config as fallback for unset values
        self.backend = backend
        self.whisper_cpp_path = whisper_cpp_path or (
            str(config.whisper_cpp_path) if config.whisper_cpp_path else None
        )
        self.whisper_model = whisper_model
        self.openai_api_key = openai_api_key or config.openai_api_key
        self.openai_model = openai_model or config.openai_model
        self.language = language or config.language
        self.device = device or config.whisper_device
        self.compute_type = compute_type or config.whisper_compute_type

        # Lazy-loaded faster-whisper model cache
        self._fw_model = None
        self._fw_model_size = None
        self._fw_model_compute_type = None

        # Model cache directory
        self._cache_dir = Path.home() / ".cache" / "omni-vram" / "models"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Audio preprocessor
        self.audio_preprocessor = AudioPreprocessor()

        # Resolve backend from config if AUTO
        if self.backend == WhisperBackend.AUTO:
            config_backend = config.whisper_backend
            if config_backend != "auto":
                try:
                    self.backend = WhisperBackend(config_backend)
                    logger.info(f"Backend from config: {self.backend.value}")
                except ValueError:
                    pass

        # Auto-detect backend
        if self.backend == WhisperBackend.AUTO:
            self.backend = self._auto_detect_backend()
            logger.info(f"Auto-detected backend: {self.backend.value}")

    def _auto_detect_backend(self) -> WhisperBackend:
        """
        Automatically detect the best available backend.

        Priority:
            1. faster-whisper (GPU-accelerated, CTranslate2, fastest)
            2. Distil-Whisper (ultra-fast distilled models)
            3. whisper.cpp (local, GPU-accelerated)
            4. OpenAI API (cloud, requires API key)

        Returns:
            Best available WhisperBackend.
        """
        # 1. Check faster-whisper (highest priority)
        if self._check_faster_whisper():
            return WhisperBackend.FASTER_WHISPER

        # 2. Check Distil-Whisper (uses faster-whisper under the hood)
        if self._check_faster_whisper() and self.whisper_model.startswith("distil-"):
            return WhisperBackend.DISTIL_WHISPER

        # 3. Check whisper.cpp
        if self._check_whisper_cpp():
            return WhisperBackend.WHISPER_CPP

        # 4. Check OpenAI API
        if self.openai_api_key:
            return WhisperBackend.OPENAI_API

        # Default to faster-whisper (will fail gracefully with helpful message)
        logger.warning(
            "No Whisper backend found. "
            "Install faster-whisper, whisper.cpp, or set OPENAI_API_KEY in .env"
        )
        return WhisperBackend.FASTER_WHISPER

    def _check_faster_whisper(self) -> bool:
        """
        Check if faster-whisper is available.

        Returns:
            True if faster-whisper package is installed.
        """
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _check_whisper_cpp(self) -> bool:
        """
        Check if whisper.cpp is available.

        Checks:
            1. Explicitly configured path
            2. Common installation paths
            3. System PATH

        Returns:
            True if whisper.cpp binary is found.
        """
        # Check explicit path
        if self.whisper_cpp_path:
            main_exe = Path(self.whisper_cpp_path) / "main.exe"
            if main_exe.exists():
                return True
            main_exe = Path(self.whisper_cpp_path) / "main"
            if main_exe.exists():
                return True

        # Check common installation paths
        for path_str in self.COMMON_WHISPER_PATHS:
            path = Path(path_str)
            if (path / "main.exe").exists() or (path / "main").exists():
                return True

        # Check system PATH
        for name in ["whisper", "main", "whisper-cli", "whisper.exe"]:
            if shutil.which(name):
                return True

        return False

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        sample_rate: int = 16000,
        language: Optional[str] = None,
        **kwargs,
    ) -> WhisperResult:
        """
        Transcribe audio to text.

        Args:
            audio_input: File path (wav/mp3/flac/ogg/m4a) or numpy array.
            sample_rate: Sample rate if audio_input is numpy array.
            language: Override language for this transcription only (thread-safe).
                      If None, uses self.language (set at init time).
            **kwargs: Additional backend-specific options.

        Returns:
            WhisperResult with text, language, confidence, segments, etc.

        Raises:
            FileNotFoundError: If whisper.cpp is not installed.
            ValueError: If audio file doesn't exist or format unsupported.
            subprocess.TimeoutExpired: If transcription exceeds 300s.
        """
        start_time = time.time()

        # Determine effective language for this call (thread-safe: no shared state mutation)
        effective_language = language or self.language

        # Load audio
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            logger.info(f"Transcribing file: {audio_path}")
            audio_data, sr = self.audio_preprocessor.load_and_convert(
                audio_path, target_sample_rate=16000
            )
            audio_duration = len(audio_data) / sr
        else:
            audio_data = audio_input
            sr = sample_rate
            audio_duration = len(audio_data) / sr

            # Ensure float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

        # Pass effective_language to backend methods via kwargs
        kwargs["language"] = effective_language

        # Route to backend
        if self.backend == WhisperBackend.FASTER_WHISPER:
            result = self._transcribe_faster_whisper(audio_data, sr, **kwargs)
        elif self.backend == WhisperBackend.DISTIL_WHISPER:
            result = self._transcribe_distil_whisper(audio_data, sr, **kwargs)
        elif self.backend == WhisperBackend.WHISPER_CPP:
            result = self._transcribe_whisper_cpp(audio_data, sr, **kwargs)
        elif self.backend == WhisperBackend.OPENAI_API:
            result = self._transcribe_openai_api(audio_data, sr, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        result.audio_duration = audio_duration
        result.processing_time = time.time() - start_time
        result.backend = self.backend

        logger.info(
            f"Transcribed {result.audio_duration:.2f}s audio in "
            f"{result.processing_time:.2f}s using {self.backend.value}"
        )

        return result

    def transcribe_stream(
        self,
        audio_chunks: List[np.ndarray],
        sample_rate: int = 16000,
        **kwargs,
    ) -> WhisperResult:
        """
        Transcribe from a stream of audio chunks.

        Args:
            audio_chunks: List of audio chunks (float32 numpy arrays).
            sample_rate: Sample rate of chunks.
            **kwargs: Additional backend-specific options.

        Returns:
            WhisperResult with aggregated transcription.
        """
        full_audio = np.concatenate(audio_chunks)
        return self.transcribe(full_audio, sample_rate=sample_rate, **kwargs)

    # ------------------------------------------------------------------
    # Faster-Whisper Backend (GPU-Accelerated, CTranslate2)
    # ------------------------------------------------------------------

    def _get_faster_whisper_model(self):
        """
        Get or load the faster-whisper model (cached).

        Returns:
            Tuple of (model, model_info) where model is a faster-whisper
            WhisperModel instance.

        Raises:
            ImportError: If faster-whisper is not installed.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper package required for GPU-accelerated transcription.\n"
                "Install with: pip install faster-whisper\n"
                "\n"
                "This provides CTranslate2-based inference, ~5x faster than "
                "native whisper."
            )

        model_size = self.whisper_model

        # Return cached model if same size
        if self._fw_model is not None and self._fw_model_size == model_size:
            return self._fw_model

        device = self.device if self.device == "cuda" else "cpu"
        compute_type = self.compute_type

        # Auto-adjust compute type for CPU
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"
            logger.info("CPU mode: compute_type adjusted to int8")

        logger.info(
            f"Loading faster-whisper model: {model_size} "
            f"(device={device}, compute_type={compute_type})"
        )

        self._fw_model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        self._fw_model_size = model_size

        return self._fw_model

    def _transcribe_faster_whisper(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs,
    ) -> WhisperResult:
        """
        Transcribe using faster-whisper (CTranslate2 GPU engine).

        Performance: ~5x faster than native whisper on GPU.

        Args:
            audio: Float32 mono audio array.
            sample_rate: Sample rate.
            **kwargs: Additional faster-whisper options
                      (beam_size, vad_filter, etc.).

        Returns:
            WhisperResult.
        """
        import tempfile
        import os

        model = self._get_faster_whisper_model()

        # Write audio to temp WAV for faster-whisper file-based input
        wav_bytes = self.audio_preprocessor.to_wav_bytes(audio, sample_rate)
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name

            # Use per-call language if provided (thread-safe), fallback to instance language
            effective_language = kwargs.pop("language", None) or self.language

            # Build transcription kwargs
            transcribe_kwargs = {
                "beam_size": kwargs.pop("beam_size", 5),
                "vad_filter": kwargs.pop("vad_filter", True),
                "vad_parameters": kwargs.pop("vad_parameters", {
                    "min_silence_duration_ms": 500,
                }),
            }

            if effective_language:
                transcribe_kwargs["language"] = effective_language

            transcribe_kwargs.update(kwargs)

            logger.debug(f"Running faster-whisper with: {transcribe_kwargs}")

            segments_iter, info = model.transcribe(tmp_path, **transcribe_kwargs)

            # Collect segments
            segments = []
            full_text_parts = []
            for seg in segments_iter:
                segment_dict = {
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                    "confidence": round(
                        1.0 - seg.no_speech_prob, 3
                    ) if seg.no_speech_prob is not None else None,
                    "avg_logprob": round(seg.avg_logprob, 4)
                    if hasattr(seg, "avg_logprob") else None,
                }
                segments.append(segment_dict)
                full_text_parts.append(seg.text.strip())

            full_text = " ".join(full_text_parts)

            # Language info from detection
            detected_lang = getattr(info, "language", self.language or "unknown")
            lang_prob = getattr(info, "language_probability", 0.0)

            # Average confidence from segments
            confidence = 0.0
            confidences = [
                s["confidence"] for s in segments
                if s.get("confidence") is not None
            ]
            if confidences:
                confidence = sum(confidences) / len(confidences)

            result = WhisperResult(
                text=full_text,
                language=detected_lang,
                confidence=confidence,
                segments=segments,
            )

            logger.info(
                f"faster-whisper: detected language={detected_lang} "
                f"(prob={lang_prob:.2f}), "
                f"{len(segments)} segments"
            )

            return result

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Whisper.cpp Backend (Local)
    # ------------------------------------------------------------------

    def _transcribe_whisper_cpp(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs,
    ) -> WhisperResult:
        """
        Transcribe using local whisper.cpp.

        Args:
            audio: Float32 mono audio array.
            sample_rate: Sample rate.
            **kwargs: Additional whisper.cpp options.

        Returns:
            WhisperResult.

        Raises:
            FileNotFoundError: If whisper.cpp binary or model not found.
            subprocess.TimeoutExpired: If transcription exceeds timeout.
        """
        # Find whisper.cpp binary
        main_exe = self._find_whisper_cpp_binary()
        logger.info(f"Using whisper.cpp: {main_exe}")

        # Get model path
        model_path = self._get_model_path()
        logger.info(f"Using model: {model_path}")

        # Write audio to temporary WAV file
        wav_bytes = self.audio_preprocessor.to_wav_bytes(audio, sample_rate)
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name

            # Use per-call language if provided (thread-safe), fallback to instance language
            effective_language = kwargs.pop("language", None) or self.language

            # Build command
            cmd = [
                main_exe,
                "-m", model_path,
                "-f", tmp_path,
                "--output-txt",
                "--no-prints",
            ]

            # Language
            if effective_language:
                cmd.extend(["-l", effective_language])

            # GPU acceleration
            if self.device == "cuda":
                cmd.extend(["--gpu-layers", "1"])

            # Extra user kwargs
            for key, value in kwargs.items():
                cmd.extend([f"--{key}", str(value)])

            logger.debug(f"Running: {' '.join(cmd)}")

            # Execute with timeout
            try:
                proc_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.TRANSCRIPTION_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                raise subprocess.TimeoutExpired(
                    cmd=cmd,
                    timeout=self.TRANSCRIPTION_TIMEOUT,
                    output=None,
                    stderr=None,
                )

            if proc_result.returncode != 0:
                raise RuntimeError(
                    f"whisper.cpp failed (exit code {proc_result.returncode}): "
                    f"{proc_result.stderr}"
                )

            # Parse output
            text = proc_result.stdout.strip()
            language = effective_language or "unknown"
            segments = self._parse_whisper_cpp_segments(proc_result.stderr)

            # Try to detect language from stderr output
            for lang_code, lang_name in self.SUPPORTED_LANGUAGES.items():
                if lang_name.lower() in proc_result.stderr.lower():
                    language = lang_code
                    break

            # Calculate confidence from segments if available
            confidence = 0.0
            if segments:
                confidences = [
                    s.get("confidence", 0.0) for s in segments
                    if s.get("confidence") is not None
                ]
                if confidences:
                    confidence = sum(confidences) / len(confidences)

            return WhisperResult(
                text=text,
                language=language,
                confidence=confidence,
                segments=segments,
            )

        finally:
            # Clean up temp file
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _find_whisper_cpp_binary(self) -> str:
        """
        Find the whisper.cpp main executable.

        Search order:
            1. Configured whisper_cpp_path
            2. Common installation paths
            3. System PATH

        Returns:
            Path to whisper.cpp executable.

        Raises:
            FileNotFoundError: If not found with helpful installation message.
        """
        candidates = []

        # 1. Explicit path
        if self.whisper_cpp_path:
            base = Path(self.whisper_cpp_path)
            candidates.extend([
                base / "main.exe",
                base / "Release" / "main.exe",
                base / "main",
            ])

        # 2. Common paths
        for path_str in self.COMMON_WHISPER_PATHS:
            base = Path(path_str)
            candidates.extend([
                base / "main.exe",
                base / "main",
            ])

        # Check candidates
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        # 3. System PATH
        for name in ["whisper", "main", "whisper-cli", "whisper.exe"]:
            found = shutil.which(name)
            if found:
                return found

        # Not found - provide helpful message
        raise FileNotFoundError(
            "whisper.cpp not found! Please install whisper.cpp:\n"
            "\n"
            "  1. Clone:  git clone https://github.com/ggerganov/whisper.cpp.git\n"
            "  2. Build:   cd whisper.cpp && cmake -B build && cmake --build build --config Release\n"
            "  3. Model:   bash models/download-ggml-model.sh base\n"
            "  4. Configure: Set WHISPER_CPP_PATH in .env file\n"
            "\n"
            f"Searched in:\n"
            + "\n".join(f"  - {p}" for p in [self.whisper_cpp_path] + self.COMMON_WHISPER_PATHS)
        )

    def _get_model_path(self) -> str:
        """
        Get path to whisper model file.

        Search order:
            1. Config (WHISPER_MODEL_PATH)
            2. whisper_cpp_path/models/
            3. Current directory

        Returns:
            Path to model file.

        Raises:
            FileNotFoundError: If model not found.
        """
        model_names = [
            f"ggml-{self.whisper_model}.bin",
            f"ggml-{self.whisper_model}.en.bin",
        ]

        # 1. Check config
        if config.whisper_model_path and config.whisper_model_path.exists():
            return str(config.whisper_model_path)

        # 2. Check whisper_cpp_path/models/
        if self.whisper_cpp_path:
            for name in model_names:
                model_path = Path(self.whisper_cpp_path) / "models" / name
                if model_path.exists():
                    return str(model_path)

        # 3. Check current directory
        for name in model_names:
            if Path(name).exists():
                return name

        raise FileNotFoundError(
            f"Whisper model '{self.whisper_model}' not found.\n"
            f"Expected one of: {model_names}\n"
            f"\n"
            f"Download a model:\n"
            f"  bash models/download-ggml-model.sh {self.whisper_model}\n"
            f"\n"
            f"Or set WHISPER_MODEL_PATH in .env file."
        )

    @staticmethod
    def _parse_whisper_cpp_segments(stderr: str) -> List[Dict[str, Any]]:
        """
        Parse segments from whisper.cpp stderr output.

        Expected format: [HH:MM:SS.mmm --> HH:MM:SS.mmm] text

        Args:
            stderr: stderr output from whisper.cpp.

        Returns:
            List of segment dicts with start, end, text, confidence.
        """
        segments = []

        # Pattern: [HH:MM:SS.mmm --> HH:MM:SS.mmm] text
        pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)'

        for line in stderr.split('\n'):
            match = re.search(pattern, line)
            if match:
                text = match.group(3).strip()
                # Extract confidence if present (whisper.cpp sometimes outputs it)
                confidence = None
                conf_match = re.search(r'\(p\s*=\s*([\d.]+)\)', text)
                if conf_match:
                    confidence = float(conf_match.group(1))
                    text = re.sub(r'\s*\(p\s*=\s*[\d.]+\)', '', text).strip()

                segments.append({
                    "start": match.group(1),
                    "end": match.group(2),
                    "text": text,
                    "confidence": confidence,
                })

        return segments

    # ------------------------------------------------------------------
    # OpenAI API Backend (Cloud)
    # ------------------------------------------------------------------

    def _transcribe_openai_api(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs,
    ) -> WhisperResult:
        """
        Transcribe using OpenAI Whisper API.

        Args:
            audio: Float32 mono audio array.
            sample_rate: Sample rate.
            **kwargs: Additional API options.

        Returns:
            WhisperResult.
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
                "Set OPENAI_API_KEY in .env file or pass openai_api_key."
            )

        client = openai.OpenAI(api_key=self.openai_api_key)

        # Encode audio to WAV bytes
        wav_bytes = self.audio_preprocessor.to_wav_bytes(audio, sample_rate)

        # Use per-call language if provided (thread-safe), fallback to instance language
        effective_language = kwargs.pop("language", None) or self.language

        # Prepare request
        request_kwargs = {
            "model": self.openai_model,
            "file": ("audio.wav", wav_bytes, "audio/wav"),
            "response_format": "verbose_json",
        }

        if effective_language:
            request_kwargs["language"] = effective_language

        for key, value in kwargs.items():
            if key not in request_kwargs:
                request_kwargs[key] = value

        logger.debug("Calling OpenAI Whisper API...")

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
                    "confidence": seg.get("avg_logprob", None),
                })

        # Calculate average confidence
        confidence = 0.0
        if segments:
            confs = [s["confidence"] for s in segments if s.get("confidence") is not None]
            if confs:
                confidence = sum(confs) / len(confs)

        return WhisperResult(
            text=text,
            language=language,
            confidence=confidence,
            segments=segments,
        )

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def detect_language(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Detect the language of audio content.

        Args:
            audio: Float32 mono audio array.
            sample_rate: Sample rate.

        Returns:
            ISO 639-1 language code (e.g. 'zh', 'en', 'ja').
        """
        max_duration = 30  # seconds
        max_samples = max_duration * sample_rate
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        result = self.transcribe(audio, sample_rate=sample_rate, language=None)
        return result.language

    def get_available_backends(self) -> List[WhisperBackend]:
        """
        List all available backends.

        Returns:
            List of available WhisperBackend values.
        """
        available = []

        if self._check_faster_whisper():
            available.append(WhisperBackend.FASTER_WHISPER)

        if self._check_whisper_cpp():
            available.append(WhisperBackend.WHISPER_CPP)

        if self.openai_api_key:
            available.append(WhisperBackend.OPENAI_API)

        return available

    def get_status(self) -> Dict[str, Any]:
        """
        Get current bridge status and configuration.

        Returns:
            Dictionary with status information.
        """
        return {
            "backend": self.backend.value,
            "whisper_cpp_path": self.whisper_cpp_path,
            "whisper_model": self.whisper_model,
            "language": self.language,
            "device": self.device,
            "compute_type": self.compute_type,
            "available_backends": [b.value for b in self.get_available_backends()],
            "has_openai_key": bool(self.openai_api_key),
            "has_faster_whisper": self._check_faster_whisper(),
            "model_cache_dir": str(self._cache_dir),
            "cached_models": self.list_cached_models(),
            "config": config.to_dict(),
        }

    # ------------------------------------------------------------------
    # Model Cache Management & Auto-Download
    # ------------------------------------------------------------------

    @staticmethod
    def list_available_models() -> Dict[str, Any]:
        """
        List all available Whisper models with metadata.

        Returns:
            Dict mapping model name to metadata (params, vram, speed, quality).
        """
        all_models = {}
        all_models.update(WHISPER_MODELS)
        all_models.update(DISTIL_WHISPER_MODELS)
        return all_models

    def list_cached_models(self) -> List[str]:
        """
        List models already downloaded in cache directory.

        Returns:
            List of cached model directory names.
        """
        cached = []
        if self._cache_dir.exists():
            for item in self._cache_dir.iterdir():
                if item.is_dir():
                    cached.append(item.name)
        # Also check CTranslate2 cache
        ct2_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if ct2_cache.exists():
            for item in ct2_cache.iterdir():
                if item.is_dir() and "whisper" in item.name.lower():
                    cached.append(item.name)
        return cached

    def download_model(self, model_name: str, force: bool = False) -> Path:
        """
        Download a Whisper model to local cache.

        For faster-whisper / Distil-Whisper models, triggers CTranslate2
        auto-download by running a dummy transcription.
        For whisper.cpp, downloads the GGML model file.

        Args:
            model_name: Model name (e.g. 'large-v3', 'distil-large-v3').
            force: Force re-download even if cached.

        Returns:
            Path to the downloaded model or cache directory.
        """
        cache_path = self._cache_dir / model_name
        if cache_path.exists() and not force:
            logger.info(f"Model '{model_name}' already cached at {cache_path}")
            return cache_path

        logger.info(f"Downloading model: {model_name}")

        if model_name in DISTIL_WHISPER_MODELS or model_name in WHISPER_MODELS:
            # For faster-whisper models, trigger CTranslate2 auto-download
            try:
                from faster_whisper import WhisperModel
                device = self.device if self.device == "cuda" else "cpu"
                compute_type = self.compute_type
                if device == "cpu" and compute_type == "float16":
                    compute_type = "int8"

                actual_model = model_name
                if model_name in DISTIL_WHISPER_MODELS:
                    actual_model = DISTIL_WHISPER_MODELS[model_name]["base"]
                    # For distil models, use the full HuggingFace model ID
                    if model_name == "distil-large-v3":
                        actual_model = "Systran/fdistil-whisper-large-v3"
                    elif model_name == "distil-large-v2":
                        actual_model = "Systran/fdistil-whisper-large-v2"
                    elif model_name == "distil-medium.en":
                        actual_model = "Systran/fdistil-whisper-medium.en"
                    elif model_name == "distil-small.en":
                        actual_model = "Systran/fdistil-whisper-small.en"

                logger.info(f"Downloading faster-whisper model: {actual_model}")
                model = WhisperModel(actual_model, device=device, compute_type=compute_type)
                # Clear from cache since we're just downloading
                self._fw_model = None
                self._fw_model_size = None
                logger.info(f"Model '{model_name}' downloaded successfully")
                cache_path.mkdir(parents=True, exist_ok=True)
                return cache_path
            except Exception as e:
                logger.error(f"Failed to download model '{model_name}': {e}")
                raise

        elif self._check_whisper_cpp():
            # Download GGML model for whisper.cpp
            model_url = (
                f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
                f"ggml-{model_name}.bin"
            )
            cache_path.mkdir(parents=True, exist_ok=True)
            target = cache_path / f"ggml-{model_name}.bin"
            logger.info(f"Downloading GGML model from {model_url}")
            try:
                import urllib.request
                urllib.request.urlretrieve(model_url, str(target))
                logger.info(f"GGML model downloaded to {target}")
                return target
            except Exception as e:
                logger.error(f"Failed to download GGML model: {e}")
                raise
        else:
            raise RuntimeError(
                "No backend available for model download. "
                "Install faster-whisper or whisper.cpp."
            )

    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """
        Clear cached models.

        Args:
            model_name: Specific model to clear. If None, clears all.
        """
        import shutil as _shutil

        if model_name:
            target = self._cache_dir / model_name
            if target.exists():
                _shutil.rmtree(target)
                logger.info(f"Cleared cache for model: {model_name}")
            else:
                logger.warning(f"Model '{model_name}' not found in cache")
        else:
            if self._cache_dir.exists():
                _shutil.rmtree(self._cache_dir)
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all model cache")

        # Reset loaded model if it was cleared
        if model_name is None or model_name == self._fw_model_size:
            self._fw_model = None
            self._fw_model_size = None

    def switch_model(self, new_model: str, auto_download: bool = True) -> None:
        """
        Switch to a different model. Clears the previous model from memory.

        Args:
            new_model: New model name (e.g. 'large-v3', 'distil-large-v3').
            auto_download: If True, auto-download the model if not cached.
        """
        logger.info(f"Switching model: {self.whisper_model} -> {new_model}")

        # Validate model name
        all_models = self.list_available_models()
        if new_model not in all_models:
            logger.warning(
                f"Unknown model '{new_model}'. Known models: {list(all_models.keys())}"
            )

        # Clear current model from memory
        self._fw_model = None
        self._fw_model_size = None
        self._fw_model_compute_type = None

        # Update model name
        self.whisper_model = new_model

        # Auto-download if requested
        if auto_download:
            try:
                self.download_model(new_model)
            except Exception as e:
                logger.warning(f"Auto-download failed: {e}. Model will be downloaded on first use.")

    def set_precision(self, compute_type: str) -> None:
        """
        Set the compute precision for inference.

        Args:
            compute_type: One of 'float32', 'float16', 'int8', 'int4'.

        Raises:
            ValueError: If compute_type is not supported.
        """
        if compute_type not in COMPUTE_TYPES:
            raise ValueError(
                f"Unsupported compute_type '{compute_type}'. "
                f"Supported: {list(COMPUTE_TYPES.keys())}"
            )

        ct_info = COMPUTE_TYPES[compute_type]
        if ct_info.get("gpu_only") and self.device == "cpu":
            logger.warning(
                f"compute_type '{compute_type}' is GPU-only. "
                f"Switching to 'int8' for CPU."
            )
            compute_type = "int8"

        if ct_info.get("experimental"):
            logger.warning(f"compute_type '{compute_type}' is experimental")

        self.compute_type = compute_type
        # Clear cached model to force reload with new precision
        self._fw_model = None
        self._fw_model_size = None
        logger.info(f"Precision set to {compute_type}")

    # ------------------------------------------------------------------
    # Distil-Whisper Backend
    # ------------------------------------------------------------------

    def _transcribe_distil_whisper(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs,
    ) -> WhisperResult:
        """
        Transcribe using Distil-Whisper (6x faster, English-optimized).

        Distil-Whisper is a distilled version of Whisper that maintains
        similar quality while being 6x faster.

        Args:
            audio: Float32 mono audio array.
            sample_rate: Sample rate.
            **kwargs: Additional options.

        Returns:
            WhisperResult.
        """
        import tempfile
        import os

        # Map distil model names to HuggingFace model IDs
        distil_model_map = {
            "distil-small.en": "Systran/fdistil-whisper-small.en",
            "distil-medium.en": "Systran/fdistil-whisper-medium.en",
            "distil-large-v2": "Systran/fdistil-whisper-large-v2",
            "distil-large-v3": "Systran/fdistil-whisper-large-v3",
        }

        model_id = distil_model_map.get(
            self.whisper_model,
            f"Systran/fdistil-{self.whisper_model}"
        )

        logger.info(f"Loading Distil-Whisper model: {model_id}")

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper required for Distil-Whisper. "
                "Install with: pip install faster-whisper"
            )

        device = self.device if self.device == "cuda" else "cpu"
        compute_type = self.compute_type
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"

        model = WhisperModel(model_id, device=device, compute_type=compute_type)

        wav_bytes = self.audio_preprocessor.to_wav_bytes(audio, sample_rate)
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name

            effective_language = kwargs.pop("language", None) or self.language

            transcribe_kwargs = {
                "beam_size": kwargs.pop("beam_size", 5),
                "vad_filter": kwargs.pop("vad_filter", True),
            }
            if effective_language:
                transcribe_kwargs["language"] = effective_language
            transcribe_kwargs.update(kwargs)

            segments_iter, info = model.transcribe(tmp_path, **transcribe_kwargs)

            segments = []
            full_text_parts = []
            for seg in segments_iter:
                segment_dict = {
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                    "confidence": round(1.0 - seg.no_speech_prob, 3)
                    if seg.no_speech_prob is not None else None,
                }
                segments.append(segment_dict)
                full_text_parts.append(seg.text.strip())

            full_text = " ".join(full_text_parts)
            detected_lang = getattr(info, "language", "en")

            confidences = [s["confidence"] for s in segments if s.get("confidence") is not None]
            confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return WhisperResult(
                text=full_text,
                language=detected_lang,
                confidence=confidence,
                segments=segments,
            )

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a model.

        Args:
            model_name: Model name.

        Returns:
            Dict with model info, or None if unknown.
        """
        if model_name in WHISPER_MODELS:
            info = WHISPER_MODELS[model_name].copy()
            info["type"] = "standard"
            info["name"] = model_name
            return info
        elif model_name in DISTIL_WHISPER_MODELS:
            info = DISTIL_WHISPER_MODELS[model_name].copy()
            info["type"] = "distil"
            info["name"] = model_name
            return info
        return None
