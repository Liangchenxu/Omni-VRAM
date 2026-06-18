"""
vram_core Core: Zero-Copy CUDA Audio-to-LLM Bridge
====================================================

High-performance audio processing and LLM inference bridge with
zero-copy VRAM memory injection.

Modules:
    - audio_utils: Audio format detection, loading, conversion
    - whisper_bridge: Whisper model backend integration (faster-whisper / whisper.cpp / OpenAI API / Distil-Whisper)
    - stream_processor: Real-time audio stream processing with VAD
    - streaming_asr: Real-time streaming speech recognition with sliding window
    - api_server: FastAPI-based REST + WebSocket transcription API
    - noise_reduction: WebRTC / noisereduce / RNNoise multi-backend noise suppression
    - emotion_recognition: wav2vec2 / openSMILE emotion recognition
    - speaker_diarization: pyannote-audio / resemblyzer speaker diarization
    - speaker_verification: MFCC-based speaker identity verification (1:1 / 1:N)
    - multi_gpu: Multi-GPU support with pipeline/data/tensor parallelism
    - vram_optimizer: KV-Cache memory management and VRAM optimization
    - tts_engine: Multi-backend TTS (edge-tts / pyttsx3)
    - voice_translator: Speech-to-speech translation (MarianMT / Google)
    - audio_event_detection: Audio event detection (YAMNet / energy-based)
    - grpc_server: gRPC + HTTP REST API server
    - plugin_manager: Extensible plugin system
    - distributed_transcriber: Multi-GPU / multi-machine parallel transcription
    - monitoring: Production metrics, Prometheus export, Grafana dashboards
    - wake_word: Wake word / keyword detection (energy-based / Whisper-based)
    - chinese: Chinese text processing (punctuation, normalization, tokenization, domain dicts)
    - meeting_summarizer: AI-powered meeting summarization with topic/decision/action extraction
"""

__version__ = "2.5.0"

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
from vram_core.whisper import (
    WhisperBridge, WhisperBackend, WhisperResult,
    TranscriptionResult, AudioPreprocessor,
)
from vram_core.stream_processor import StreamProcessor, StreamConfig, StreamState
from vram_core.streaming_asr import StreamASR, StreamASRConfig, StreamASRResult
from vram_core.noise_reduction import NoiseReducer
from vram_core.emotion_recognition import EmotionRecognizer
from vram_core.speaker_diarization import SpeakerDiarizer
from vram_core.multi_gpu import MultiGPUManager
from vram_core.vram_optimizer import VRAMOptimizer, MemoryPressure
from vram_core.tts_engine import TTSEngine
from vram_core.voice_translator import VoiceTranslator
from vram_core.audio_event_detection import AudioEventDetector
from vram_core.plugin_manager import PluginManager, PluginBase, PluginInfo
from vram_core.config import config, OmniConfig, setup_logging
from vram_core.meeting_summarizer import MeetingSummarizer, MeetingMinutes, ActionItem, Decision, TopicSegment
from vram_core.chinese.punctuation import PunctuationRestorer
from vram_core.chinese.normalizer import TextNormalizer
from vram_core.chinese.tokenizer import ChineseTokenizer as Tokenizer
from vram_core.chinese.domain_dict import DomainDictionary as DomainDict
DOMAIN_DICTS = {}

# v2.5.0: Audio Enhancement & Quality
from vram_core.audio_enhancer import AudioEnhancer
from vram_core.speech_quality import SpeechQualityAssessor, QualityReport

# v2.5.0: LLM Meeting Assistant
from vram_core.llm_client import LLMClient, LLMConfig
from vram_core.meeting_analyzer import MeetingAnalyzer, MeetingAnalysis, ActionItem as MeetingActionItem

# New modules (v1.0.0)
from vram_core.speaker_verification import SpeakerVerifier, Voiceprint, VerificationResult
from vram_core.distributed_transcriber import DistributedTranscriber, DistributedResult
from vram_core.monitoring import MetricsCollector, SystemHealth
from vram_core.wake_word import WakeWordDetector, WakeWordEvent

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
    # VRAM Optimization
    "VRAMOptimizer",
    "MemoryPressure",
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
    # Speaker Verification
    "SpeakerVerifier",
    "Voiceprint",
    "VerificationResult",
    # Distributed Transcription
    "DistributedTranscriber",
    "DistributedResult",
    # Monitoring
    "MetricsCollector",
    "SystemHealth",
    # Wake Word Detection
    "WakeWordDetector",
    "WakeWordEvent",
    # Meeting Summarization
    "MeetingSummarizer",
    "MeetingMinutes",
    "ActionItem",
    "Decision",
    "TopicSegment",
    # Chinese Text Processing
    "PunctuationRestorer",
    "TextNormalizer",
    "Tokenizer",
    "DomainDict",
    "DOMAIN_DICTS",
    # v2.5.0: Audio Enhancement & Quality
    "AudioEnhancer",
    "SpeechQualityAssessor",
    "QualityReport",
    # v2.5.0: LLM Meeting Assistant
    "LLMClient",
    "LLMConfig",
    "MeetingAnalyzer",
    "MeetingAnalysis",
    "MeetingActionItem",
    # Configuration
    "config",
    "OmniConfig",
    "setup_logging",
    "CUDA_AVAILABLE",
]
