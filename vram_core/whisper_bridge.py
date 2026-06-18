"""
Backward-compatibility shim for vram_core.whisper_bridge.
=========================================================

.. deprecated::
    This module is deprecated. Use ``vram_core.whisper`` instead.

    All classes have been moved to the ``vram_core.whisper`` subpackage:
    - WhisperBridge -> vram_core.whisper.bridge
    - WhisperBackend, WhisperResult, TranscriptionResult -> vram_core.whisper.models / result
    - AudioPreprocessor -> vram_core.whisper.preprocessor

This shim re-exports everything so existing code continues to work.
"""

import warnings

warnings.warn(
    "vram_core.whisper_bridge is deprecated. Use vram_core.whisper instead.",
    DeprecationWarning,
    stacklevel=2,
)

from vram_core.whisper import (  # noqa: F401
    WhisperBridge,
    WhisperBackend,
    WhisperResult,
    TranscriptionResult,
    AudioPreprocessor,
)

__all__ = [
    "WhisperBridge",
    "WhisperBackend",
    "WhisperResult",
    "TranscriptionResult",
    "AudioPreprocessor",
]