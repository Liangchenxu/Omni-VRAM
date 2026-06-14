"""
Whisper Bridge — Backward Compatibility Shim
=============================================

This module re-exports everything from ``vram_core.whisper`` so that
existing imports like::

    from vram_core.whisper_bridge import WhisperBridge

continue to work unchanged.

For new code, prefer importing from the package directly::

    from vram_core.whisper import WhisperBridge, WhisperResult
"""

# Re-export all public symbols from the whisper package
from vram_core.whisper import (
    AudioPreprocessor,
    COMPUTE_TYPES,
    DISTIL_WHISPER_MODELS,
    SUPPORTED_AUDIO_FORMATS,
    WHISPER_MODELS,
    WhisperBackend,
    WhisperBridge,
    WhisperResult,
)

# Also re-export TranscriptionJob which lived in the old bridge module
from vram_core.whisper.bridge import TranscriptionJob

__all__ = [
    "AudioPreprocessor",
    "COMPUTE_TYPES",
    "DISTIL_WHISPER_MODELS",
    "SUPPORTED_AUDIO_FORMATS",
    "WHISPER_MODELS",
    "WhisperBackend",
    "WhisperBridge",
    "WhisperResult",
    "TranscriptionJob",
]