"""
Omni-VRAM Core: Zero-Copy CUDA Audio-to-LLM Bridge
====================================================

High-performance audio processing and LLM inference bridge with
zero-copy VRAM memory injection.

Modules:
    - audio_utils: Audio format detection, loading, conversion
    - whisper_bridge: Whisper model backend integration (faster-whisper / whisper.cpp / OpenAI API)
    - stream_processor: Real-time audio stream processing with VAD
    - streaming_asr: Real-time streaming speech recognition with sliding window
    - api_server: FastAPI-based REST + WebSocket transcription API
    - noise_reduction: WebRTC / noisereduce / RNNoise multi-backend noise suppression
    - emotion_recognition: wav2vec2 / openSMILE emotion recognition
    - speaker_diarization: pyannote-audio / resemblyzer speaker diarization
    - multi_gpu: Multi-GPU support with pipeline/data/tensor parallelism
    - vram_optimizer: KV-Cache memory management and VRAM optimization
    - tts_engine: Multi-backend TTS (edge-tts / pyttsx3)
    - voice_translator: Speech-to-speech translation (MarianMT / Google)
    - audio_event_detection: Audio event detection (YAMNet / energy-based)
    - grpc_server: gRPC + HTTP REST API server
    - plugin_manager: Extensible plugin system
"""

__version__ = "0.6.0"

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
from vram_core.streaming_asr import StreamASR, StreamASRConfig, StreamASRResult
from vram_core.noise_reduction import NoiseReducer
from vram_core.emotion_recognition import EmotionRecognizer
from vram_core.speaker_diarization import SpeakerDiarizer
from vram_core.multi_gpu import MultiGPUManager, DevicePool, DeviceStatus
from vram_core.vram_optimizer import VRAMOptimizer, MemoryPressure, VRAMStatus
from vram_core.tts_engine import TTSEngine
from vram_core.voice_translator import VoiceTranslator
from vram_core.audio_event_detection import AudioEventDetector
from vram_core.plugin_manager import PluginManager, PluginBase, PluginInfo
from vram_core.config import config, OmniConfig, setup_logging

# gRPC server (optional, may fail if grpcio not installed)
try:
    from vram_core.grpc_server import OmniVRAMServicer
except ImportError:
    OmniVRAMServicer = None

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
    # Streaming ASR
    "StreamASR",
    "StreamASRConfig",
    "StreamASRResult",
    # Audio Enhancement
    "NoiseReducer",
    # Emotion & Speaker
    "EmotionRecognizer",
    "SpeakerDiarizer",
    # Multi-GPU
    "MultiGPUManager",
    "DevicePool",
    "DeviceStatus",
    # VRAM Optimization
    "VRAMOptimizer",
    "MemoryPressure",
    "VRAMStatus",
    # TTS
    "TTSEngine",
    # Translation
    "VoiceTranslator",
    # Audio Event Detection
    "AudioEventDetector",
    # Plugin System
    "PluginManager",
    "PluginBase",
    "PluginInfo",
    # Configuration
    "config",
    "OmniConfig",
    "setup_logging",
    "CUDA_AVAILABLE",
]