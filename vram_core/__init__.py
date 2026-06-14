"""
Omni-VRAM Core: Zero-Copy CUDA Audio-to-LLM Bridge
====================================================

High-performance audio processing and LLM inference bridge with
zero-copy VRAM memory injection.

Modules:
    - audio_utils: Audio format detection, loading, conversion
    - whisper_bridge: Whisper model backend integration
    - stream_processor: Real-time audio stream processing with VAD
"""

__version__ = "0.2.0"

# CUDA extension (built from vram_hacker.cu)
try:
    import vram_core._vram_hacker as _backend
    CUDA_AVAILABLE = True
    # Expose CUDA functions if available
    stress_test = _backend.stress_test
    launch_dynamic_kernel = _backend.launch_dynamic_kernel
    inject_into_model = _backend.inject_into_model
    query_memory = _backend.query_memory
except (ImportError, AttributeError):
    CUDA_AVAILABLE = False

from vram_core.audio_utils import AudioProcessor
from vram_core.whisper_bridge import (
    WhisperBridge, WhisperBackend, WhisperResult,
    TranscriptionResult, AudioPreprocessor,
)
from vram_core.stream_processor import StreamProcessor, StreamConfig, StreamState
from vram_core.config import config, OmniConfig, setup_logging

__all__ = [
    # Core
    "AudioProcessor",
    "WhisperBridge",
    "WhisperBackend",
    "WhisperResult",
    "TranscriptionResult",
    "AudioPreprocessor",
    "StreamProcessor",
    "StreamConfig",
    "StreamState",
    # Configuration
    "config",
    "OmniConfig",
    "setup_logging",
    "CUDA_AVAILABLE",
]
