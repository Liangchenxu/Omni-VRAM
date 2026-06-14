# Changelog

All notable changes to Omni-VRAM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- GPU-accelerated Whisper (faster-whisper / CTranslate2)
- Noise suppression (RNNoise / WebRTC APM)
- Streaming ASR (chunked decoding)
- Docker deployment support
- Web UI (Gradio / Streamlit)
- Multi-GPU support

---

## [0.4.0] - 2024-01-XX

### Added
- Complete documentation suite:
  - `docs/installation.md` — Full installation guide (Windows/Linux/macOS)
  - `docs/quickstart.md` — Quick start tutorial with step-by-step examples
  - `docs/api_reference.md` — Comprehensive API reference for all modules
  - `docs/examples.md` — Detailed guide for all example applications
  - `docs/faq.md` — Frequently asked questions and troubleshooting
- Technical blog post (`docs/blog_omni_vram.md`)
- Updated README.md with badges, quick start, and contribution links

---

## [0.3.0] - 2024-01-XX

### Added
- Example application: Real-time Voice Assistant (`examples/realtime_voice_assistant.py`)
  - PyAudio microphone input with device selection
  - Configurable VAD threshold
  - Audio recording save support
  - Session summary on exit
- Example application: Meeting Transcriber (`examples/meeting_transcriber.py`)
  - Long-duration recording with auto-segmentation
  - Export to TXT and JSON formats
  - Offline file transcription mode
- Example application: Voice Chat Bot (`examples/voice_chat_bot.py`)
  - Multi-turn voice conversation
  - Chat history management with context
  - Export conversation logs
  - LLM API integration point (echo mode placeholder)
- Example application: Benchmark Suite (`examples/benchmark_suite.py`)
  - Hardware info collection (GPU/CUDA/CPU/RAM)
  - KV-Cache performance benchmark (8 configs, torch.cat vs zero-copy)
  - Audio processing benchmark
  - Whisper transcription speed benchmark
  - Markdown report generation
- Example: Whisper local test script (`examples/test_whisper_local.py`)
- Unit tests for AudioProcessor (`tests/test_audio_utils.py`, 20 test cases)
- Unit tests for WhisperBridge (`tests/test_whisper_bridge.py`, 16 test cases)
- Unit tests for StreamProcessor (`tests/test_stream_processor.py`, 16 test cases)

### Changed
- Improved error handling across all modules
- Enhanced logging with structured messages

---

## [0.2.0] - 2024-01-XX

### Added
- Whisper bridge module (`vram_core/whisper_bridge.py`)
  - Multi-backend support: OpenAI API, whisper.cpp CLI, Python whisper
  - Automatic backend detection and fallback (API → CLI → Python → None)
  - Audio preprocessing pipeline for Whisper compatibility
  - Segment-level timestamps and confidence scores
  - `WhisperBackend` enum for backend selection
  - `WhisperResult` data class for structured output
- Configuration management module (`vram_core/config.py`)
  - `OmniConfig` singleton with `.env` file loading
  - 20+ configuration parameters (API keys, paths, model settings)
  - Configuration validation and error reporting
  - Runtime update support
  - Sensitive information masking in logs
- `AudioProcessor` class in `vram_core/audio_utils.py`
  - Format detection (WAV, MP3, FLAC, OGG, RAW)
  - Stereo-to-mono conversion
  - Sample rate conversion (linear interpolation)
  - Audio normalization (peak)
  - WAV byte encoding
  - Duration calculation
  - Support for loading from file path or bytes
- `StreamProcessor` class in `vram_core/stream_processor.py`
  - Energy-based VAD (Voice Activity Detection)
  - Speech segment collection with silence detection
  - Auto-segmentation on silence (configurable threshold)
  - Force segmentation on max duration
  - Callback-driven architecture (`on_transcription`, `on_state_change`)
  - State machine: IDLE → SPEAKING → PROCESSING
- Package-level exports in `vram_core/__init__.py`
  - Unified API: `from vram_core import AudioProcessor, WhisperBridge, ...`
  - CUDA availability detection with graceful fallback
  - Version constant: `vram_core.__version__`
- `.env.example` configuration template with all parameters
- Updated `README.md` with v0.2.0 documentation

### Changed
- Reorganized project structure into `vram_core/` package
- Moved audio processing from inline code to `AudioProcessor` class

---

## [0.1.0] - 2024-01-XX

### Added
- Initial release of Omni-VRAM
- CUDA kernel: Zero-copy KV-Cache injection (`vram_hacker.cu`)
  - `append_kv_kernel` — O(1) atomic append with pointer offset
  - Pre-allocated contiguous VRAM, no `torch.cat` overhead
  - Up to 11x faster than `torch.cat` on repeated updates
- CUDA kernel: Fused audio front-end (`vram_hacker.cu`)
  - VAD energy calculation + pre-emphasis + Hann windowing in single kernel
  - Shared memory optimization, 6.7x faster than separate NumPy operations
- CUDA kernel: Hardware DNA scanner
  - GPU compute capability detection
  - SM count, CUDA cores, VRAM capacity
  - L2 cache size and shared memory limits
- CUDA kernel: Dynamic kernel dispatcher
  - Runtime kernel selection based on hardware capabilities
- CUDA kernel: VRAM stress test utility
- Build system (`setup.py`)
  - CUDA extension compilation with setuptools
  - Automatic NVCC detection
  - Graceful fallback when CUDA is unavailable
- Integration test (`test_run.py`)
  - CUDA availability check
  - Config loading verification
  - KV-Cache benchmark (100 iterations)