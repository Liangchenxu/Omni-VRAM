"""
WhisperBridge
==============

Main Whisper transcription engine supporting multiple backends:
  - faster-whisper  (CTranslate2, default)
  - whisper.cpp      (C++ binary)
  - OpenAI API       (cloud)
  - Distil-Whisper    (optimized variant)
"""

import asyncio
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from vram_core.config import config
from vram_core.whisper.models import (
    COMPUTE_TYPES,
    DISTIL_WHISPER_MODELS,
    SUPPORTED_AUDIO_FORMATS,
    WHISPER_MODELS,
    WhisperBackend,
)
from vram_core.whisper.preprocessor import AudioPreprocessor
from vram_core.whisper.result import WhisperResult, TranscriptionResult

logger = logging.getLogger("vram_core.whisper.bridge")


@dataclass
class TranscriptionJob:
    """Tracks a single transcription job."""

    job_id: str
    file_path: str
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[WhisperResult] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress: float = 0.0

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class WhisperBridge:
    """
    Main Whisper transcription bridge supporting multiple backends.

    Auto-selects the best available backend on initialization.
    Thread-safe for concurrent transcription requests.
    """

    SUPPORTED_LANGUAGES: Dict[str, str] = {
        "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
        "fr": "French", "de": "German", "es": "Spanish", "ru": "Russian",
        "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
        "sv": "Swedish", "da": "Danish", "fi": "Finnish", "no": "Norwegian",
        "hu": "Hungarian", "cs": "Czech", "ro": "Romanian", "bg": "Bulgarian",
        "hr": "Croatian", "sk": "Slovak", "sl": "Slovenian", "et": "Estonian",
        "lv": "Latvian", "lt": "Lithuanian", "mt": "Maltese", "ga": "Irish",
        "cy": "Welsh", "is": "Icelandic", "ms": "Malay", "id": "Indonesian",
        "tl": "Filipino", "vi": "Vietnamese", "th": "Thai", "hi": "Hindi",
        "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
        "ar": "Arabic", "he": "Hebrew", "tr": "Turkish", "el": "Greek",
        "uk": "Ukrainian", "mk": "Macedonian", "sq": "Albanian", "bs": "Bosnian",
        "sr": "Serbian", "sw": "Swahili",
    }

    def __init__(
        self,
        whisper_cpp_path: Optional[str] = None,
        whisper_model: Optional[str] = None,
        language: Optional[str] = None,
        backend: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        proxy: Optional[str] = None,
    ):
        """
        Initialize the WhisperBridge.

        Args:
            whisper_cpp_path: Path to whisper.cpp directory.
            whisper_model:    Model name (default: config or 'base').
            language:         Default language code.
            backend:          Backend name (default: auto-detect).
            device:           'cuda' or 'cpu' (default: config).
            compute_type:     Precision type (default: config).
            openai_api_key:   API key for OpenAI backend.
            openai_model:     OpenAI model name (default: 'whisper-1').
            proxy:            HTTP proxy URL.
        """
        self.whisper_cpp_path = whisper_cpp_path or (str(config.whisper_cpp_path) if config.whisper_cpp_path else None)
        self.whisper_model = whisper_model or (str(config.whisper_model_path).split("/")[-1].replace("ggml-","").replace(".bin","") if config.whisper_model_path else "base")
        self.language = language or config.language
        self.openai_api_key = openai_api_key or config.openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.openai_model = openai_model or config.openai_model
        self.proxy = proxy or None
        self.device = device or config.device
        self.compute_type = compute_type or config.whisper_compute_type

        self.audio_preprocessor = AudioPreprocessor()
        if backend == WhisperBackend.AUTO or (isinstance(backend, str) and backend == "auto"):
            self._backend = self._auto_detect_backend()
        else:
            self._backend = self._resolve_backend(backend)
        self._fw_model: Optional[Any] = None
        self._fw_model_size: Optional[str] = None
        self._fw_model_compute_type: Optional[str] = None
        self._jobs: Dict[str, TranscriptionJob] = {}
        self._cache_dir = Path.home() / ".cache" / "vram_core" / "models"
        self._thread_pool = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=4)

        self._validate_language()

        logger.info(
            f"WhisperBridge initialized "
            f"(backend={self._backend.value}, model={self.whisper_model}, "
            f"device={self.device}, compute_type={self.compute_type})"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backend(self) -> WhisperBackend:
        return self._backend

    @backend.setter
    def backend(self, value: Union[str, WhisperBackend]) -> None:
        if isinstance(value, str):
            value = WhisperBackend(value)
        self._backend = value

    @property
    def model(self) -> str:
        return self._fw_model_size or self.whisper_model

    # ------------------------------------------------------------------
    # Backend Resolution
    # ------------------------------------------------------------------

    def _auto_detect_backend(self) -> WhisperBackend:
        """Auto-detect the best available backend."""
        if self._check_faster_whisper():
            return WhisperBackend.FASTER_WHISPER
        if self._check_whisper_cpp():
            return WhisperBackend.WHISPER_CPP
        if self.openai_api_key:
            return WhisperBackend.OPENAI_API
        # Default to whisper.cpp even if not installed
        return WhisperBackend.WHISPER_CPP

    def _resolve_backend(self, backend: Optional[str] = None) -> WhisperBackend:
        """Resolve which backend to use."""
        if backend and backend != "auto":
            try:
                return WhisperBackend(backend)
            except ValueError:
                logger.warning(
                    f"Unknown backend '{backend}', auto-detecting. "
                    f"Valid: {[b.value for b in WhisperBackend]}"
                )

        if self._check_faster_whisper():
            return WhisperBackend.FASTER_WHISPER
        if self._check_whisper_cpp():
            return WhisperBackend.WHISPER_CPP
        if self.openai_api_key:
            return WhisperBackend.OPENAI_API

        logger.warning(
            "No Whisper backend available! Install faster-whisper (pip install faster-whisper) "
            "or build whisper.cpp. Falling back to OpenAI API."
        )
        return WhisperBackend.OPENAI_API

    @staticmethod
    def _check_faster_whisper() -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _check_whisper_cpp(self) -> bool:
        if not self.whisper_cpp_path:
            return False
        binary = Path(self.whisper_cpp_path) / "main"
        if not binary.exists():
            binary = Path(self.whisper_cpp_path) / "main.exe"
        return binary.exists()

    def _validate_language(self) -> None:
        """Validate the configured language code."""
        if not self.language:
            return
        valid_languages = {
            "zh", "en", "ja", "ko", "fr", "de", "es", "ru", "it", "pt",
            "nl", "pl", "sv", "da", "fi", "no", "hu", "cs", "ro", "bg",
            "hr", "sk", "sl", "et", "lv", "lt", "mt", "ga", "cy", "is",
            "ms", "id", "tl", "vi", "th", "hi", "bn", "ta", "te", "ur",
            "ar", "he", "tr", "el", "uk", "mk", "sq", "bs", "sr", "sw",
        }
        if self.language not in valid_languages:
            logger.warning(
                f"Language '{self.language}' may not be supported by Whisper. "
                f"Common codes: zh, en, ja, ko, fr, de, es, ru"
            )

    # ------------------------------------------------------------------
    # Core Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray, bytes],
        sample_rate: int = 16000,
        language: Optional[str] = None,
        task: str = "transcribe",
        output_format: str = "text",
        **kwargs,
    ) -> WhisperResult:
        """
        Transcribe audio to text (unified entry point).

        Args:
            audio:        File path, numpy array, or bytes.
            sample_rate:  Sample rate (for numpy array input).
            language:     Override language (default: instance language).
            task:         'transcribe' or 'translate'.
            output_format: 'text' or 'json'.
            **kwargs:     Backend-specific options.

        Returns:
            WhisperResult with transcription.
        """
        start_time = time.time()

        if self._backend == WhisperBackend.OPENAI_API:
            effective_language = language
        else:
            effective_language = language or self.language

        audio_array = self._prepare_audio(audio, sample_rate)

        logger.info(
            f"Transcribing {len(audio_array) / sample_rate:.1f}s audio "
            f"(backend={self._backend.value}, lang={effective_language or 'auto'})"
        )

        # Route to backend-specific method
        backend_map: Dict[WhisperBackend, Callable[..., WhisperResult]] = {
            WhisperBackend.FASTER_WHISPER: self._transcribe_faster_whisper,
            WhisperBackend.WHISPER_CPP:    self._transcribe_whisper_cpp,
            WhisperBackend.OPENAI_API:     self._transcribe_openai_api,
        }

        # Support distil-whisper variant
        if (
            self._backend == WhisperBackend.FASTER_WHISPER
            and self.whisper_model in DISTIL_WHISPER_MODELS
            and "distil" in self.whisper_model
        ):
            transcribe_fn = self._transcribe_distil_whisper
        else:
            transcribe_fn = backend_map.get(self._backend)
            if transcribe_fn is None:
                raise ValueError(f"Unsupported backend: {self._backend}")

        result = transcribe_fn(
            audio_array, sample_rate,
            language=effective_language, task=task, **kwargs,
        )
        result.backend = self._backend.value
        result.model = self.whisper_model

        elapsed = time.time() - start_time
        audio_duration = len(audio_array) / sample_rate
        rtf = elapsed / audio_duration if audio_duration > 0 else 0

        logger.info(
            f"Transcription completed in {elapsed:.2f}s "
            f"(audio={audio_duration:.1f}s, RTF={rtf:.2f}, "
            f"lang={result.language}, conf={result.confidence:.2f})"
        )

        return result

    async def async_transcribe(
        self,
        audio: Union[str, Path, np.ndarray, bytes],
        sample_rate: int = 16000,
        language: Optional[str] = None,
        task: str = "transcribe",
        output_format: str = "text",
        **kwargs,
    ) -> WhisperResult:
        """
        Async wrapper for :meth:`transcribe`.

        Runs the synchronous transcription in a thread pool executor
        so the event loop is not blocked.

        Args:
            audio:         File path, numpy array, or bytes.
            sample_rate:   Sample rate (for numpy array input).
            language:      Override language.
            task:          'transcribe' or 'translate'.
            output_format: 'text' or 'json'.
            **kwargs:      Backend-specific options.

        Returns:
            WhisperResult with transcription.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.transcribe(
                audio,
                sample_rate=sample_rate,
                language=language,
                task=task,
                output_format=output_format,
                **kwargs,
            ),
        )

    def transcribe_file(
        self,
        file_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        output_format: str = "text",
        **kwargs,
    ) -> WhisperResult:
        """
        Transcribe an audio file with auto-format detection.

        Args:
            file_path:     Path to audio file.
            language:      Override language.
            task:          'transcribe' or 'translate'.
            output_format: 'text' or 'json'.

        Returns:
            WhisperResult with transcription.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        if not self.audio_preprocessor.check_format_support(str(file_path)):
            raise ValueError(
                f"Unsupported audio format: {file_path.suffix}. "
                f"Supported: {SUPPORTED_AUDIO_FORMATS}"
            )

        audio_info = self.audio_preprocessor.get_audio_info(str(file_path))
        duration = audio_info.get("duration", 0)
        logger.info(
            f"Loading audio: {file_path.name} "
            f"({duration:.1f}s, {audio_info.get('sample_rate', '?')}Hz, "
            f"{audio_info.get('channels', '?')}ch)"
        )

        max_duration = 600  # 10 minutes
        if duration > max_duration:
            logger.info(f"Audio too long ({duration:.1f}s), using chunked transcription")
            return self._transcribe_long_audio(
                str(file_path), language=language, task=task, **kwargs
            )

        audio_array = self.audio_preprocessor.load_audio_pydub(str(file_path))
        return self.transcribe(
            audio_array, sample_rate=16000, language=language, task=task,
            output_format=output_format, **kwargs,
        )

    def transcribe_stream(
        self,
        chunks: List[np.ndarray],
        sample_rate: int = 16000,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe multiple audio chunks as a single stream.

        Concatenates all chunks and transcribes the combined audio.

        Args:
            chunks:       List of numpy audio arrays.
            sample_rate:  Sample rate for all chunks.
            **kwargs:     Additional options passed to transcribe().

        Returns:
            TranscriptionResult with combined transcription.
        """
        combined = np.concatenate(chunks)
        result = self.transcribe(combined, sample_rate=sample_rate, **kwargs)
        return TranscriptionResult(
            text=result.text,
            language=result.language,
            segments=result.segments,
            backend=result.backend,
        )

    def _prepare_audio(
        self,
        audio: Union[str, Path, np.ndarray, bytes],
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Convert various input types to a normalized float32 array."""
        if isinstance(audio, np.ndarray):
            audio_array = audio.astype(np.float32)
            if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                audio_array = audio_array / 32768.0
            return audio_array

        if isinstance(audio, bytes):
            return self.audio_preprocessor.load_audio_pydub(audio)

        audio_path = Path(audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return self.audio_preprocessor.load_audio_pydub(str(audio_path))

    # ------------------------------------------------------------------
    # faster-whisper Backend (Primary)
    # ------------------------------------------------------------------

    def _transcribe_faster_whisper(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        **kwargs,
    ) -> WhisperResult:
        """Transcribe using faster-whisper (CTranslate2)."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Install with:\n"
                "  pip install faster-whisper\n\n"
                "For GPU support:\n"
                "  pip install faster-whisper[gpu]"
            )

        model = self._get_faster_whisper_model()
        language = kwargs.pop("language", None)
        task = kwargs.pop("task", "transcribe")

        beam_size = kwargs.pop("beam_size", 5)
        best_of = kwargs.pop("best_of", 5)
        vad_filter = kwargs.pop("vad_filter", True)
        vad_parameters = kwargs.pop("vad_parameters", None)

        transcribe_kwargs: Dict[str, Any] = {
            "beam_size": beam_size,
            "best_of": best_of,
            "vad_filter": vad_filter,
            "language": language,
            "task": task,
        }
        if vad_parameters:
            transcribe_kwargs["vad_parameters"] = vad_parameters
        transcribe_kwargs.update(kwargs)

        import tempfile
        import os
        wav_bytes = self.audio_preprocessor.to_wav_bytes(audio, sample_rate)
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name

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
                    "tokens": seg.tokens if hasattr(seg, "tokens") else None,
                    "temperature": seg.temperature if hasattr(seg, "temperature") else None,
                    "avg_logprob": seg.avg_logprob if hasattr(seg, "avg_logprob") else None,
                    "compression_ratio": seg.compression_ratio if hasattr(seg, "compression_ratio") else None,
                    "no_speech_prob": seg.no_speech_prob if hasattr(seg, "no_speech_prob") else None,
                    "seek": seg.seek if hasattr(seg, "seek") else None,
                }
                segments.append(segment_dict)
                full_text_parts.append(seg.text.strip())

            full_text = " ".join(full_text_parts)

            confidences = [s["confidence"] for s in segments if s.get("confidence") is not None]
            confidence = sum(confidences) / len(confidences) if confidences else 0.0

            detected_lang = getattr(info, "language", language or "unknown")
            language_probability = getattr(info, "language_probability", 0.0)

            result = WhisperResult(
                text=full_text,
                language=detected_lang,
                confidence=confidence,
                segments=segments,
                raw={"info": info, "language_probability": language_probability},
            )

            logger.info(
                f"faster-whisper transcription: {len(segments)} segments, "
                f"lang={detected_lang} (prob={language_probability:.2f})"
            )

            return result

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _get_faster_whisper_model(self):
        """Get or load the faster-whisper model (cached)."""
        from faster_whisper import WhisperModel

        compute_type = self.compute_type
        if self.device == "cpu" and compute_type == "float16":
            compute_type = "int8"
            logger.info("CPU detected, using int8 instead of float16")

        model_key = f"{self.whisper_model}_{self.device}_{compute_type}"

        if (
            self._fw_model is not None
            and self._fw_model_size == self.whisper_model
            and self._fw_model_compute_type == compute_type
        ):
            logger.debug(f"Reusing cached model: {model_key}")
            return self._fw_model

        if self._fw_model is not None:
            logger.info(
                f"Model config changed ({self._fw_model_size} -> {self.whisper_model}), "
                f"reloading..."
            )
            self._fw_model = None

        actual_model = self.whisper_model
        if self.whisper_model in DISTIL_WHISPER_MODELS:
            distil_info = DISTIL_WHISPER_MODELS[self.whisper_model]
            actual_model = distil_info["base"]
            logger.info(
                f"Using Distil-Whisper '{self.whisper_model}' "
                f"(base: {actual_model}, {distil_info['speed']})"
            )

        logger.info(
            f"Loading faster-whisper model: {actual_model} "
            f"(device={self.device}, compute_type={compute_type})"
        )

        t0 = time.time()
        model_kwargs: Dict[str, Any] = {
            "device": self.device,
            "compute_type": compute_type,
        }

        # cpu_threads not in config, use reasonable default
        cpu_threads = 4
        if cpu_threads > 0:
            model_kwargs["cpu_threads"] = cpu_threads

        try:
            model = WhisperModel(actual_model, **model_kwargs)
        except Exception as e:
            if "CUDA" in str(e) or "gpu" in str(e).lower():
                logger.warning(f"GPU load failed ({e}), falling back to CPU")
                model_kwargs["device"] = "cpu"
                model_kwargs["compute_type"] = "int8"
                self.device = "cpu"
                self.compute_type = "int8"
                model = WhisperModel(actual_model, **model_kwargs)
            else:
                raise

        elapsed = time.time() - t0
        logger.info(f"Model loaded in {elapsed:.2f}s")

        self._fw_model = model
        self._fw_model_size = self.whisper_model
        self._fw_model_compute_type = compute_type
        return model

    # ------------------------------------------------------------------
    # whisper.cpp Backend (Legacy)
    # ------------------------------------------------------------------

    def _transcribe_whisper_cpp(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        **kwargs,
    ) -> WhisperResult:
        """Transcribe using whisper.cpp binary."""
        import subprocess
        import tempfile

        model_path = self._find_whisper_cpp_model()
        binary = self._find_whisper_cpp_binary()

        wav_bytes = self.audio_preprocessor.to_wav_bytes(audio, sample_rate)
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name

            cmd = [
                str(binary),
                "-m", str(model_path),
                "-f", tmp_path,
                "-t", str(4),
                "--no-gpu",
                "-l", kwargs.get("language", self.language or "auto"),
            ]

            beam_size = kwargs.get("beam_size")
            if beam_size:
                cmd.extend(["--beam-size", str(beam_size)])

            logger.debug(f"whisper.cpp command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise RuntimeError(f"whisper.cpp failed (code {result.returncode}): {error_msg}")

            segments = self._parse_whisper_cpp_segments(result.stderr)
            full_text = " ".join(seg["text"] for seg in segments)

            confidences = [seg["confidence"] for seg in segments if seg.get("confidence") is not None]
            confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return WhisperResult(
                text=full_text,
                language=self.language or "unknown",
                confidence=confidence,
                segments=segments,
                raw={"stdout": result.stdout, "stderr": result.stderr},
            )

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _find_whisper_cpp_binary(self) -> Path:
        """Locate the whisper.cpp main binary."""
        if not self.whisper_cpp_path:
            raise FileNotFoundError(
                "whisper.cpp path not set. "
                "Set WHISPER_CPP_PATH in .env or pass whisper_cpp_path."
            )

        base = Path(self.whisper_cpp_path)
        for name in ["main", "main.exe", "whisper-cli", "whisper-cli.exe"]:
            binary = base / name
            if binary.exists():
                return binary

        build_paths = [
            base / "build" / "bin" / "Release" / "main.exe",
            base / "build" / "bin" / "main",
            base / "build" / "main",
        ]
        for path in build_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"whisper.cpp binary not found in {self.whisper_cpp_path}.\n"
            f"Build whisper.cpp first:\n"
            f"  cd {self.whisper_cpp_path}\n"
            f"  make"
        )

    def _find_whisper_cpp_model(self) -> str:
        """Locate the GGML model file."""
        model_names = [
            f"ggml-{self.whisper_model}.bin",
            f"ggml-{self.whisper_model}.en.bin",
        ]

        if config.whisper_model_path and config.whisper_model_path.exists():
            return str(config.whisper_model_path)

        if self.whisper_cpp_path:
            for name in model_names:
                model_path = Path(self.whisper_cpp_path) / "models" / name
                if model_path.exists():
                    return str(model_path)

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
        """Parse segments from whisper.cpp stderr output."""
        segments = []
        pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)'

        for line in stderr.split('\n'):
            match = re.search(pattern, line)
            if match:
                text = match.group(3).strip()
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
        sample_rate: int = 16000,
        **kwargs,
    ) -> WhisperResult:
        """Transcribe using OpenAI Whisper API."""
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
        wav_bytes = self.audio_preprocessor.to_wav_bytes(audio, sample_rate)
        effective_language = kwargs.pop("language", None) or self.language

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
        """Detect the language of audio content."""
        max_duration = 30
        max_samples = max_duration * sample_rate
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        result = self.transcribe(audio, sample_rate=sample_rate, language=None)
        return result.language

    def get_available_backends(self) -> List[WhisperBackend]:
        """List all available backends."""
        available = []
        if self._check_faster_whisper():
            available.append(WhisperBackend.FASTER_WHISPER)
        if self._check_whisper_cpp():
            available.append(WhisperBackend.WHISPER_CPP)
        if self.openai_api_key:
            available.append(WhisperBackend.OPENAI_API)
        return available

    def get_status(self) -> Dict[str, Any]:
        """Get current bridge status and configuration."""
        return {
            "backend": self._backend.value,
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
        """List all available Whisper models with metadata."""
        all_models = {}
        all_models.update(WHISPER_MODELS)
        all_models.update(DISTIL_WHISPER_MODELS)
        return all_models

    def list_cached_models(self) -> List[str]:
        """List models already downloaded in cache directory."""
        cached = []
        if self._cache_dir.exists():
            for item in self._cache_dir.iterdir():
                if item.is_dir():
                    cached.append(item.name)
        ct2_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if ct2_cache.exists():
            for item in ct2_cache.iterdir():
                if item.is_dir() and "whisper" in item.name.lower():
                    cached.append(item.name)
        return cached

    def download_model(self, model_name: str, force: bool = False) -> Path:
        """Download a Whisper model to local cache."""
        cache_path = self._cache_dir / model_name
        if cache_path.exists() and not force:
            logger.info(f"Model '{model_name}' already cached at {cache_path}")
            return cache_path

        logger.info(f"Downloading model: {model_name}")

        if model_name in DISTIL_WHISPER_MODELS or model_name in WHISPER_MODELS:
            try:
                from faster_whisper import WhisperModel
                device = self.device if self.device == "cuda" else "cpu"
                compute_type = self.compute_type
                if device == "cpu" and compute_type == "float16":
                    compute_type = "int8"

                actual_model = model_name
                if model_name in DISTIL_WHISPER_MODELS:
                    actual_model = DISTIL_WHISPER_MODELS[model_name]["base"]
                    distil_hf_map = {
                        "distil-large-v3": "Systran/fdistil-whisper-large-v3",
                        "distil-large-v2": "Systran/fdistil-whisper-large-v2",
                        "distil-medium.en": "Systran/fdistil-whisper-medium.en",
                        "distil-small.en": "Systran/fdistil-whisper-small.en",
                    }
                    actual_model = distil_hf_map.get(model_name, f"Systran/fdistil-{model_name}")

                logger.info(f"Downloading faster-whisper model: {actual_model}")
                model = WhisperModel(actual_model, device=device, compute_type=compute_type)
                self._fw_model = None
                self._fw_model_size = None
                logger.info(f"Model '{model_name}' downloaded successfully")
                cache_path.mkdir(parents=True, exist_ok=True)
                return cache_path
            except Exception as e:
                logger.error(f"Failed to download model '{model_name}': {e}")
                raise

        elif self._check_whisper_cpp():
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
        """Clear cached models."""
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

        if model_name is None or model_name == self._fw_model_size:
            self._fw_model = None
            self._fw_model_size = None

    def switch_model(self, new_model: str, auto_download: bool = True) -> None:
        """Switch to a different model. Clears the previous model from memory."""
        logger.info(f"Switching model: {self.whisper_model} -> {new_model}")

        all_models = self.list_available_models()
        if new_model not in all_models:
            logger.warning(
                f"Unknown model '{new_model}'. Known models: {list(all_models.keys())}"
            )

        self._fw_model = None
        self._fw_model_size = None
        self._fw_model_compute_type = None
        self.whisper_model = new_model

        if auto_download:
            try:
                self.download_model(new_model)
            except Exception as e:
                logger.warning(f"Auto-download failed: {e}. Model will be downloaded on first use.")

    def set_precision(self, compute_type: str) -> None:
        """Set the compute precision for inference."""
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
        self._fw_model = None
        self._fw_model_size = None
        logger.info(f"Precision set to {compute_type}")

    # ------------------------------------------------------------------
    # Distil-Whisper Backend
    # ------------------------------------------------------------------

    def _transcribe_distil_whisper(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        **kwargs,
    ) -> WhisperResult:
        """Transcribe using Distil-Whisper (6x faster, English-optimized)."""
        import tempfile
        import os

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
        """Get detailed information about a model."""
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

    def _transcribe_long_audio(
        self,
        file_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        chunk_duration: int = 30,
        overlap: float = 2.0,
        **kwargs,
    ) -> WhisperResult:
        """
        Transcribe long audio files by splitting into chunks.

        Args:
            file_path:       Path to audio file.
            language:        Override language.
            task:            'transcribe' or 'translate'.
            chunk_duration:  Duration of each chunk in seconds.
            overlap:         Overlap between chunks in seconds.
            **kwargs:        Backend-specific options.

        Returns:
            WhisperResult with combined transcription.
        """
        audio_array = self.audio_preprocessor.load_audio_pydub(file_path)
        sample_rate = 16000
        chunk_samples = chunk_duration * sample_rate
        overlap_samples = int(overlap * sample_rate)

        total_samples = len(audio_array)
        segments = []
        full_text_parts = []
        offset = 0.0

        pos = 0
        while pos < total_samples:
            end = min(pos + chunk_samples, total_samples)
            chunk = audio_array[pos:end]

            if len(chunk) < sample_rate:
                break

            chunk_result = self.transcribe(
                chunk, sample_rate=sample_rate,
                language=language, task=task, **kwargs,
            )

            for seg in chunk_result.segments:
                adj_seg = seg.copy()
                adj_seg["start"] = round(seg["start"] + offset, 3)
                adj_seg["end"] = round(seg["end"] + offset, 3)
                segments.append(adj_seg)

            full_text_parts.append(chunk_result.text)

            chunk_len = len(chunk) / sample_rate
            offset += chunk_len - overlap
            pos += chunk_samples - overlap_samples

        full_text = " ".join(full_text_parts)
        confidences = [s["confidence"] for s in segments if s.get("confidence") is not None]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return WhisperResult(
            text=full_text,
            language=language or self.language or "unknown",
            confidence=confidence,
            segments=segments,
            audio_duration=total_samples / sample_rate,
        )

    def transcribe_async(
        self,
        audio: Union[str, Path, np.ndarray, bytes],
        sample_rate: int = 16000,
        language: Optional[str] = None,
        callback: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """
        Submit an async transcription task to a thread pool.

        Args:
            audio:      File path, numpy array, or bytes.
            sample_rate: Sample rate.
            language:   Override language.
            callback:   Optional callback(WhisperResult) called on completion.
            **kwargs:   Additional arguments passed to transcribe().

        Returns:
            concurrent.futures.Future that resolves to WhisperResult.
        """
        import concurrent.futures

        future = self._thread_pool.submit(
            self.transcribe, audio, sample_rate=sample_rate,
            language=language, **kwargs
        )

        if callback:
            def _on_done(fut):
                try:
                    result = fut.result()
                    callback(result)
                except Exception as e:
                    logger.error(f"Async transcribe callback error: {e}")
            future.add_done_callback(_on_done)

        return future
