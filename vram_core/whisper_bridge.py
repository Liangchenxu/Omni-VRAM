"""
Whisper Bridge - Backward Compatibility Shim (DEPRECATED)
=========================================================

.. deprecated:: 2.1.0
    This module is a compatibility shim. Use ``vram_core.whisper`` instead.
    Will be removed in v3.0.0.

This module re-exports everything from ``vram_core.whisper`` so that
existing imports like::

    from vram_core.whisper_bridge import WhisperBridge

continue to work unchanged.

For new code, prefer importing from the package directly::

    from vram_core.whisper import WhisperBridge, WhisperResult
"""

import warnings
warnings.warn(
    "vram_core.whisper_bridge is deprecated since v2.1.0. "
    "Use 'from vram_core.whisper import ...' instead. "
    "This shim will be removed in v3.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

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
    TranscriptionResult,
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
    "TranscriptionResult",
    "TranscriptionJob",
]