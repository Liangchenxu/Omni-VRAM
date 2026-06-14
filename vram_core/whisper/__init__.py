"""
Whisper Package for Omni-VRAM
==============================

Submodules:
    models      - Backend enum, model metadata constants
    result      - WhisperResult dataclass with SRT export
    preprocessor - AudioPreprocessor (pydub format conversion)
    bridge      - WhisperBridge main transcription engine

Backward Compatibility:
    Import from vram_core.whisper_bridge still works (re-export shim).
"""

from vram_core.whisper.models import (
    WhisperBackend,
    WHISPER_MODELS,
    DISTIL_WHISPER_MODELS,
    COMPUTE_TYPES,
    SUPPORTED_AUDIO_FORMATS,
)
from vram_core.whisper.result import WhisperResult
from vram_core.whisper.preprocessor import AudioPreprocessor
from vram_core.whisper.bridge import WhisperBridge

__all__ = [
    "WhisperBackend",
    "WHISPER_MODELS",
    "DISTIL_WHISPER_MODELS",
    "COMPUTE_TYPES",
    "SUPPORTED_AUDIO_FORMATS",
    "WhisperResult",
    "AudioPreprocessor",
    "WhisperBridge",
]