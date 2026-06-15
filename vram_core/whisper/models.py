"""
Whisper Models & Constants
===========================

Defines backend enum, model metadata tables, compute types, and
supported audio formats for the Whisper transcription engine.
"""

from enum import Enum
from typing import Any, Dict


class WhisperBackend(str, Enum):
    """Available Whisper transcription backends."""

    FASTER_WHISPER = "faster-whisper"  # CTranslate2 (default)
    WHISPER_CPP = "whisper_cpp"        # C++ inference
    OPENAI_API = "openai_api"          # Cloud API
    AUTO = "auto"                       # Auto-detect best backend


# ── Model Metadata ────────────────────────────────────────────────
# vram_mb: approximate GPU memory at float16 (large-v3 ~2.9 GB)

WHISPER_MODELS: Dict[str, Dict[str, Any]] = {
    "tiny":    {"params": "39M",  "vram_mb": 100,  "speed": "~32x", "quality": "***"},
    "base":    {"params": "74M",  "vram_mb": 150,  "speed": "~16x", "quality": "****"},
    "small":   {"params": "244M", "vram_mb": 500,  "speed": "~6x",  "quality": "****"},
    "medium":  {"params": "769M", "vram_mb": 1500, "speed": "~2x",  "quality": "*****"},
    "large":   {"params": "1550M","vram_mb": 2900, "speed": "1x",   "quality": "*****"},
    "large-v2":{"params": "1550M","vram_mb": 2900, "speed": "1x",   "quality": "*****"},
    "large-v3":{"params": "1550M","vram_mb": 2900, "speed": "1x",   "quality": "*****"},
}

DISTIL_WHISPER_MODELS: Dict[str, Dict[str, Any]] = {
    "distil-large-v2": {"params": "756M", "vram_mb": 1500, "speed": "~6x", "quality": "*****", "base": "large-v2"},
    "distil-large-v3": {"params": "756M", "vram_mb": 1500, "speed": "~6x", "quality": "*****", "base": "large-v3"},
    "distil-medium.en":{"params": "394M", "vram_mb": 800,  "speed": "~8x", "quality": "*****", "base": "medium.en"},
    "distil-small.en": {"params": "166M", "vram_mb": 400,  "speed": "~12x","quality": "****",  "base": "small.en"},
}

COMPUTE_TYPES: Dict[str, Dict[str, Any]] = {
    "float32": {"type": "float32", "desc": "Full precision (slowest)"},
    "float16": {"type": "float16", "desc": "Half precision (recommended for GPU)"},
    "int8":    {"type": "int8",    "desc": "Quantized (good for CPU)"},
    "int4":    {"type": "int4",    "desc": "4-bit quantized (experimental, needs GPU)"},
    "bfloat16":{"type": "bfloat16","desc": "Brain floating point (Ampere+ GPU)"},
}

SUPPORTED_AUDIO_FORMATS = [
    ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4", ".aac", ".wma", ".opus",
]