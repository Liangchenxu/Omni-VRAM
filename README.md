# Omni-VRAM: LLM 语音交互框架
### 让大模型长出耳朵和嘴巴

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA: 11.0+](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)
![Platform: Windows/Linux](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)
![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
[![Tests](https://github.com/Liangchenxu/Omni-VRAM/actions/workflows/test.yml/badge.svg)](https://github.com/Liangchenxu/Omni-VRAM/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/omni-vram.svg)](https://pypi.org/project/omni-vram/)
[![Version](https://img.shields.io/badge/Version-2.5.0-orange.svg)](https://github.com/Liangchenxu/Omni-VRAM/releases)

[**English**](#english-documentation) | [**中文文档**](#chinese-documentation) | [**Docs**](docs/)

---

<a id="english-documentation"></a>
## 📖 Overview

**Omni-VRAM** is a production-ready **LLM voice interaction framework** that lets large language models hear and speak. Built on CUDA zero-copy technology, it provides **28 core modules** covering the entire audio AI pipeline — from speech recognition to synthesis, from single GPU to distributed clusters.

> **v2.5.0**: Major release with 28 modules, new Audio Enhancer (7-stage pipeline), Speech Quality Assessment (SNR/PESQ-lite), LLM Meeting Assistant (multi-provider AI analysis), Edge Deployment Backends (ONNX/TensorRT/Lite), and 85+ integration tests.

Traditional Python audio pipelines and PyTorch operations (e.g., `torch.cat` for KV-Cache) introduce significant overhead. Omni-VRAM implements **Operator Fusion** and **Zero-Copy Memory Injection** at the hardware level, enabling consumer-grade GPUs (RTX 30/40 series) to achieve sub-millisecond latency for real-time voice agents.

### ✅ Core Features (28 Modules)

| # | Module | Description |
|---|--------|-------------|
| 1 | **Whisper Transcription** | Multi-backend (faster-whisper / whisper.cpp / API / Distil-Whisper), tiny → large-v3.5, GPU 5× speedup |
| 2 | **Real-Time Streaming ASR** | Sliding-window VAD, partial/final callbacks, <500ms latency |
| 3 | **Noise Reduction** | Spectral subtraction, adaptive Wiener filter, multi-stage pipeline, silence detection |
| 4 | **Emotion Recognition** | MFCC + energy + ZCR features, 6 emotions (neutral/happy/sad/angry/surprised/fearful) |
| 5 | **Speaker Diarization** | MFCC feature extraction + cosine similarity clustering, identifies "who spoke when" |
| 6 | **Speaker Verification** | MFCC voiceprint, 1:1 verification & 1:N identification, persistent voiceprint library |
| 7 | **Wake Word Detection** | Energy-based & phoneme-level fuzzy matching, custom vocabulary |
| 8 | **TTS Engine** | Multi-backend (pyttsx3 / edge-tts / gTTS), 300+ voices, async synthesis |
| 9 | **Voice Translation** | Speech-to-speech pipeline, 50+ language pairs |
| 10 | **Audio Event Detection** | Energy threshold + spectral analysis, detects cough/laughter/applause and more |
| 11 | **Multi-GPU Support** | Auto device discovery, load balancing (round-robin / least-used / VRAM-priority) |
| 12 | **Distributed Transcription** | Multi-machine parallel batch processing, auto load balancing |
| 13 | **KV-Cache VRAM Optimizer** | Memory pressure detection (LOW/MEDIUM/HIGH/CRITICAL), KV-Cache estimation, quantization recommendation |
| 14 | **Production Monitoring** | Real-time GPU monitoring (memory/temp/power/utilization), `@gpu_monitor` decorator |
| 15 | **REST API** | FastAPI async HTTP + WebSocket streaming |
| 16 | **gRPC Server** | High-performance dual-protocol (gRPC + REST) server |
| 17 | **Plugin System** | Extensible architecture with discovery, lifecycle & hook events |
| 18 | **CUDA Kernels** | Zero-Copy KV-Cache (11× faster), Fused Audio Frontend (28× faster) |
| 19 | **Stream Processor** | Real-time audio stream processing with VAD, buffering, and segmentation |
| 20 | **Whisper Bridge** | Modular Whisper integration with model management, CUDA bridge, and preprocessing |
| 21 | **Audio Utilities** | Audio format detection, conversion, resampling, spectral computation |
| 22 | **Configuration System** | YAML/JSON config files, environment variable overrides, hot-reload |
| 23 | **Chinese Meeting Transcription** | Punctuation restoration, text normalization (numbers/currency/dates), Chinese tokenization, domain dictionaries (medical/tech/finance/legal/education/government), dialect support |
| 24 | **Meeting Summarizer** | AI-powered meeting summarization with topic segmentation, decision detection, action item extraction, speaker contribution analysis, multi-format export (Markdown/JSON/Dict) |
| 25 | **Audio Enhancer** | 7-stage enhancement pipeline (noise reduction → dereverb → normalization → AGC → high-pass → speech EQ → noise gate), quality presets (fast/broadcast/studio) |
| 26 | **Speech Quality Assessment** | SNR estimation, spectral clarity, PESQ-lite estimate, clipping detection, quality grading (excellent/good/fair/poor) |
| 27 | **LLM Meeting Assistant** | Multi-provider LLM client (OpenAI/Claude/Ollama/custom), AI meeting analysis with topic/decision/action extraction, sentiment analysis, structured JSON output |
| 28 | **Edge Deployment Backends** | ONNX Runtime (CPU/GPU, INT8/INT4), TensorRT (FP16/INT8), Lite backend for mobile/embedded (Raspberry Pi, Jetson Nano) |

### 📁 Project Structure

```
Omni-VRAM/
├── app.py                      # Gradio Web Demo (transcription/emotion/diarization/mic)
├── vram_hacker.cu              # CUDA kernel source (KV-Cache injection)
├── setup.py                    # Build & install script
├── pyproject.toml              # Modern Python project configuration
├── requirements.txt            # Python dependencies
├── test_run.py                 # Quick integration test
├── run_tests.py                # Unified test runner
├── .env.example                # Configuration template
├── Dockerfile                  # GPU Docker image (CUDA + audio libs)
├── Dockerfile.cpu              # CPU-only Docker image
├── docker-compose.yml          # One-click Docker deployment
│
├── vram_core/                  # Python core library (24 modules)
│   ├── __init__.py             # Package exports (v2.5.0)
│   ├── config.py               # Configuration management
│   ├── utils.py                # General utility functions
│   ├── audio_utils.py          # Audio format detection & conversion
│   ├── whisper_bridge.py       # Whisper multi-backend integration (legacy)
│   ├── whisper/                # Whisper sub-module (v2.0)
│   │   ├── bridge.py           # CUDA Whisper bridge
│   │   ├── models.py           # Model management
│   │   ├── optimizer.py        # Whisper CUDA Graph & quantization optimizer
│   │   ├── preprocessor.py     # Audio preprocessor
│   │   └── result.py           # Transcription result dataclass
│   ├── stream_processor.py     # Real-time stream processor + VAD
│   ├── streaming_asr.py        # Real-time streaming ASR engine
│   ├── realtime_optimizer.py   # Real-time latency optimizer (auto-tune chunk size)
│   ├── api_server.py           # FastAPI REST + WebSocket API
│   ├── noise_reduction.py      # STFT spectral subtraction noise reduction
│   ├── emotion_recognition.py  # Acoustic feature-based emotion recognition
│   ├── speaker_diarization.py  # MFCC speaker diarization & clustering
│   ├── speaker_verification.py # Speaker voiceprint verification (1:1 & 1:N)
│   ├── wake_word.py            # Wake word / keyword detection
│   ├── multi_gpu.py            # Multi-GPU management & parallelism
│   ├── vram_optimizer.py       # KV-Cache VRAM optimization & OOM recovery
│   ├── tts_engine.py           # Multi-backend text-to-speech
│   ├── voice_translator.py     # Speech-to-speech translation pipeline
│   ├── audio_event_detection.py # Audio event detection
│   ├── distributed_transcriber.py # Multi-GPU/machine parallel transcription
│   ├── monitoring.py           # GPU monitoring & Prometheus metrics
│   ├── grpc_server.py          # gRPC + HTTP REST dual-protocol server
│   ├── plugin_manager.py       # Plugin discovery, loading & lifecycle
│   ├── meeting_summarizer.py   # AI meeting summarization (topics/decisions/actions)
│   ├── audio_enhancer.py       # 7-stage audio enhancement pipeline
│   ├── speech_quality.py       # Speech quality assessment (SNR/PESQ-lite)
│   ├── llm_client.py           # Multi-provider LLM client
│   ├── meeting_analyzer.py     # AI meeting analysis with structured output
│   ├── backends/               # Edge deployment backends
│   │   ├── onnx_backend.py     # ONNX Runtime inference
│   │   ├── tensorrt_backend.py # TensorRT optimized inference
│   │   └── lite_backend.py     # Lightweight mobile/embedded inference
│   └── chinese/                # Chinese NLP pipeline
│       ├── punctuation.py      # Chinese punctuation restoration
│       ├── normalizer.py       # Text normalization (numbers/currency/dates)
│       ├── tokenizer.py        # Chinese word segmentation
│       ├── domain_dict.py      # Domain dictionaries (medical/tech/finance/legal)
│       └── dialect.py          # Cantonese dialect normalization
│
├── examples/                   # Example applications
│   ├── realtime_voice_assistant.py  # Real-time voice assistant
│   ├── meeting_transcriber.py       # Meeting transcription & summary
│   ├── voice_chat_bot.py            # Multi-turn voice chat bot
│   ├── benchmark_suite.py           # Performance benchmark suite
│   ├── benchmark_v3.py              # v2.1.0 benchmark comparison
│   ├── api_demo.py                  # API server demo client
│   ├── test_whisper_local.py        # Whisper local test script
│   └── test_emotion.py              # Emotion recognition test
│
├── tests/                      # Unit & integration tests (28 test files, 85+ test cases)
│   ├── test_audio_utils.py
│   ├── test_emotion_recognition.py
│   ├── test_meeting_transcription.py # Meeting transcription & summarization tests
│   ├── test_monitoring.py
│   ├── test_multi_gpu.py
│   ├── test_noise_reduction.py
│   ├── test_plugin_manager.py
│   ├── test_speaker_diarization.py
│   ├── test_speaker_verification.py
│   ├── test_stream_processor.py
│   ├── test_tts_engine.py
│   ├── test_vram_optimizer.py
│   ├── test_wake_word.py
│   ├── test_whisper_bridge.py
│   ├── test_whisper_optimizer.py    # Whisper optimization tests
│   ├── test_integration.py          # Full pipeline integration tests
│   ├── test_websocket.py            # WebSocket API tests
│   ├── test_realtime_latency.py     # Real-time latency tests
│   ├── test_v250.py                 # v2.5.0 feature tests (16 cases)
│   └── benchmark_comparison.py      # Benchmark comparison
│
└── docs/                       # Documentation
    ├── installation.md
    ├── quickstart.md
    ├── api_reference.md
    ├── examples.md
    ├── faq.md
    └── blog_omni_vram.md
```

### 🐳 Docker Deployment

```bash
# GPU version (with CUDA support)
docker build -t omni-vram:gpu .
docker run --gpus all -p 8000:8000 omni-vram:gpu

# CPU-only version (no CUDA required)
docker build -f Dockerfile.cpu -t omni-vram:cpu .
docker run -p 8000:8000 omni-vram:cpu

# One-click with docker-compose
docker-compose up -d

# Run with environment variables
docker run --gpus all \
  -e WHISPER_MODEL=base \
  -e DEFAULT_LANGUAGE=zh \
  -p 8000:8000 \
  omni-vram:gpu
```

### 🧪 Examples

| Example | Description | Command |
|---------|-------------|---------|
| **Gradio Web Demo** | Web UI with transcription, emotion, diarization & mic recording | `python app.py` |
| **Real-time Voice Assistant** | Microphone → VAD → Whisper → Display, with file recording | `python examples/realtime_voice_assistant.py` |
| **Meeting Transcriber** | Long-form recording with silence auto-segmentation and export | `python examples/meeting_transcriber.py --output meeting.txt` |
| **Voice Chat Bot** | Multi-turn dialogue with history tracking and LLM-ready architecture | `python examples/voice_chat_bot.py` |
| **Benchmark Suite** | Performance testing for all modules with Markdown report | `python examples/benchmark_suite.py --skip-whisper` |
| **Emotion Recognition** | Speech emotion analysis demo | `python examples/test_emotion.py` |
| **Whisper Local Test** | Local Whisper transcription test | `python examples/test_whisper_local.py` |

### 🌐 Gradio Web Demo

Launch the interactive web UI with one command:

```bash
# Install Gradio (if not already installed)
pip install gradio

# Start the demo (default: http://localhost:7860)
python app.py

# Options
python app.py --port 8080        # Custom port
python app.py --share            # Create public link
python app.py --debug            # Debug mode
```

**Features:**
- 📝 **Speech Transcription** — Upload audio → get text (with model/language/noise reduction options)
- 🎭 **Emotion Recognition** — Upload audio → detect emotion (6 emotions with probability bars)
- 👥 **Speaker Diarization** — Upload conversation → identify who spoke when
- 🎙️ **Live Microphone** — Record voice → instant transcription
- 📥 **Download Results** — Export as JSON / TXT / SRT subtitle files

---

## 📊 Performance Benchmarks

*Hardware: NVIDIA RTX 3060 (12GB) | Platform: Windows WDDM | CUDA: 12.1*

### 1. KV-Cache Memory Injection
*Task: Appending 100 updates (50 tokens each) to a 100,000-capacity KV-Cache tensor (Dimension: 4096).*

| Engine / Method | Latency | Complexity | OOM Risk |
| :--- | :--- | :--- | :--- |
| PyTorch Native (`torch.cat`) | 90.32 ms | $O(N)$ (Reallocation) | High (VRAM Fragmentation) |
| **Omni-VRAM (Zero-Copy)** | **8.07 ms** | **$O(1)$ (Pointer Offset)** | **None** |
| **Improvement** | **11.19x** | - | - |

### 2. Audio Processing Pipeline
| Pipeline Stage | Input Size | PyTorch / CPU Baseline | Omni-VRAM C++ Kernel | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Concurrent VAD** | 10 Minutes (16kHz) | 9.45 ms (CPU `unfold`) | **0.33 ms** | **~28x** |
| **Fused Frontend** | 60 Seconds (16kHz) | 20.33 ms (VRAM Stacking)| **1.05 ms** | **~19x** |

### 3. Whisper Transcription (CPU)
| Model | 1s Audio | 5s Audio | 10s Audio |
| :--- | :--- | :--- | :--- |
| tiny | ~200ms | ~500ms | ~900ms |
| base | ~400ms | ~1200ms | ~2200ms |

> Run `python examples/benchmark_suite.py` for automated benchmarks on your hardware.

---

## 🛠️ Installation

```bash
# Quick install (Python package only, no CUDA kernels)
pip install omni-vram

# Full install (with CUDA kernels for 11x/28x speedup)
git clone https://github.com/Liangchenxu/Omni-VRAM.git
cd Omni-VRAM
pip install -r requirements.txt

# Build and install the CUDA extension
# Note: Ensure NVCC and Visual Studio C++ Build Tools are properly configured.
python setup.py install

# (Optional) Install Web API server dependencies
pip install fastapi uvicorn python-multipart

# (Optional) Install whisper.cpp for local transcription
# See docs/installation.md for detailed instructions
```

### Configuration

```bash
# Copy the configuration template
cp .env.example .env

# Edit .env with your settings
# At minimum, set WHISPER_CPP_PATH and WHISPER_MODEL_PATH for local transcription
```

> See [docs/installation.md](docs/installation.md) for detailed installation guide.

## 💡 Quick Start

### Whisper Transcription

```python
from vram_core.whisper_bridge import WhisperBridge
from vram_core.whisper_bridge import WhisperBackend

# Initialize with automatic backend detection
whisper = WhisperBridge(
    backend=WhisperBackend.AUTO,
    whisper_model="base",
    language="zh",
)

# Transcribe an audio file
result = whisper.transcribe("audio.wav")
print(f"Text: {result.text}")
print(f"Confidence: {result.confidence}")
print(f"Duration: {result.audio_duration}s")
```

### Real-Time Stream Processing

```python
import numpy as np
from vram_core.stream_processor import StreamProcessor, StreamConfig
from vram_core.whisper_bridge import WhisperBridge, WhisperBackend

# Initialize components
whisper = WhisperBridge(backend=WhisperBackend.AUTO, whisper_model="base")
config = StreamConfig(sample_rate=16000, chunk_duration_ms=100, vad_threshold=0.02)
processor = StreamProcessor(config=config, whisper_bridge=whisper)

# Set up callbacks
processor.on_transcription = lambda result: print(f"Transcribed: {result.text}")

# Feed audio chunks (e.g., from microphone)
audio_chunk = np.random.randn(1600).astype(np.float32)
processor.feed(audio_chunk)
```

### Streaming ASR (Real-time Microphone Transcription)

```python
import numpy as np
from vram_core.whisper_bridge import WhisperBridge, WhisperBackend
from vram_core.streaming_asr import StreamASR, StreamASRConfig

# Initialize whisper
whisper = WhisperBridge(backend=WhisperBackend.AUTO, whisper_model="base")

# Configure streaming ASR
config = StreamASRConfig(
    sample_rate=16000,
    vad_threshold=0.015,
    language="zh",
)
asr = StreamASR(config=config, whisper_bridge=whisper)

# Set up callbacks
asr.on_partial_result = lambda text: print(f"[Partial] {text}")
asr.on_final_result = lambda result: print(f"[Final] {result.text}")

# Start and feed audio
asr.start()
audio_chunk = np.random.randn(3200).astype(np.float32)  # from microphone
asr.feed(audio_chunk)
```

### Chinese Meeting Transcription

```python
from vram_core.chinese.punctuation import PunctuationRestorer
from vram_core.chinese.normalizer import TextNormalizer
from vram_core.chinese.tokenizer import ChineseTokenizer
from vram_core.meeting_summarizer import MeetingSummarizer

# Restore punctuation to raw ASR text
restorer = PunctuationRestorer()
text = restorer.restore("今天下午三点开会讨论项目进度请各位准时参加")
# → "今天下午三点开会讨论项目进度，请各位准时参加。"

# Normalize numbers, currency, dates
normalizer = TextNormalizer()
text = normalizer.normalize("一共花了3500元购买了50台设备")
# → "一共花了三千五百元购买了五十台设备"

# Chinese word segmentation
tokenizer = ChineseTokenizer()
tokens = tokenizer.tokenize("语音识别技术发展迅速")
# → ["语音识别", "技术", "发展", "迅速"]

# Meeting summarization
summarizer = MeetingSummarizer()
minutes = summarizer.summarize(transcript_text, speaker_segments=segments)
print(minutes.summary)
for topic in minutes.topics:
    print(f"Topic: {topic.title}")
for decision in minutes.decisions:
    print(f"Decision: {decision.content}")
for action in minutes.action_items:
    print(f"Action: {action.description} → {action.assignee}")
```

### Speaker Diarization

```python
import numpy as np
from vram_core.speaker_diarization import SpeakerDiarizer

# Initialize diarizer
diarizer = SpeakerDiarizer(n_mfcc=13, similarity_threshold=0.7)

# Load audio (float32, mono, 16kHz)
audio = np.fromfile("audio.raw", dtype=np.float32)

# Perform diarization
result = diarizer.diarize(audio, sample_rate=16000)

for segment in result.segments:
    print(f"[{segment.start_time:.1f}s - {segment.end_time:.1f}s] Speaker: {segment.speaker_id}")

print(f"Total speakers: {diarizer.get_speaker_count()}")
```

### Speaker Verification

```python
import numpy as np
from vram_core.speaker_verification import SpeakerVerifier

# Initialize verifier
verifier = SpeakerVerifier(threshold=0.75, storage_path="voiceprints.json")

# Register a speaker
audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
verifier.register("alice", audio, sample_rate=16000)

# Verify identity
test_audio = np.random.randn(16000).astype(np.float32)
result = verifier.verify("alice", test_audio)
print(f"Verified: {result.verified}, Confidence: {result.confidence:.3f}")

# 1:N identification
best = verifier.verify_any(test_audio)
if best:
    print(f"Identified: {best.speaker_id} ({best.confidence:.3f})")
```

### Emotion Recognition

```python
import numpy as np
from vram_core.emotion_recognition import EmotionRecognizer

# Initialize
recognizer = EmotionRecognizer(sample_rate=16000)

# Analyze emotion from audio
audio = np.random.randn(32000).astype(np.float32)  # 2 seconds
result = recognizer.recognize(audio)
print(f"Emotion: {result['emotion']}, Confidence: {result['confidence']:.2f}")
```

### Noise Reduction

```python
import numpy as np
from vram_core.noise_reduction import NoiseReducer

# Initialize
reducer = NoiseReducer(sample_rate=16000)

# Reduce noise
noisy_audio = np.random.randn(16000).astype(np.float32)
clean_audio = reducer.reduce_noise(noisy_audio, aggressiveness=0.7)
```

### VRAM Optimization

```python
from vram_core.vram_optimizer import VRAMOptimizer

# Initialize
optimizer = VRAMOptimizer(device_id=0)

# Check VRAM status
status = optimizer.get_status()
print(f"GPU: {status.gpu_name}, Usage: {status.usage_pct:.1f}%, Pressure: {status.pressure.value}")

# Estimate KV-Cache memory
estimate = VRAMOptimizer.estimate_kv_cache(n_layers=32, seq_length=2048, batch_size=1)
print(f"KV-Cache: {estimate.total_mb:.1f} MB")

# Get quantization recommendation
dtype = optimizer.recommend_dtype(required_mb=4000)
print(f"Recommended dtype: {dtype}")

# Auto-optimize (cleanup if pressure is high)
optimizer.auto_optimize()
```

### TTS (Text-to-Speech)

```python
from vram_core.tts_engine import TTSEngine

# Initialize with edge-tts backend
engine = TTSEngine(backend="edge-tts")

# Synthesize speech
engine.synthesize("Hello, world!", output_path="output.mp3")
```

### Web API Server

```bash
# Start the API server
python -m vram_core.api_server --model base --language zh --port 8000
```

```python
# Client: File upload transcription
import requests
with open("audio.wav", "rb") as f:
    resp = requests.post("http://localhost:8000/transcribe", files={"file": f})
    print(resp.json()["text"])

# Client: WebSocket streaming
import websockets, asyncio
async def stream():
    async with websockets.connect("ws://localhost:8000/stream") as ws:
        await ws.send(audio_bytes)  # 16-bit PCM, 16kHz mono
        result = await ws.recv()
        print(result)
```

### Plugin System

```python
from vram_core.plugin_manager import PluginManager

# Initialize plugin manager
pm = PluginManager(plugin_dir="./plugins")

# Load a plugin
pm.load_plugin("my_plugin")

# List loaded plugins
for plugin in pm.list_plugins():
    print(f"Plugin: {plugin['name']} v{plugin['version']}")

# Register hooks
pm.register_hook("on_transcription", my_callback)
```

> See [docs/quickstart.md](docs/quickstart.md) for more examples.

---

## 🔧 Troubleshooting (故障排除)

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ImportError: No module named 'vram_core._vram_hacker'` | CUDA extension not built | Run `python setup.py install` or use CPU-only mode (the library works without CUDA) |
| `CUDA_HOME not found` | NVCC not in PATH | Set `CUDA_HOME` env variable or install CUDA Toolkit |
| `No module named 'faster_whisper'` | Optional dependency missing | `pip install faster-whisper` |
| `torch.cuda.is_available() returns False` | PyTorch CPU-only installed | Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `Port already in use` when starting API server | Port 8000 is occupied | Use `--port 8080` or kill the existing process |
| `ModuleNotFoundError: No module named 'gradio'` | Gradio not installed | `pip install gradio` |
| `RuntimeError: CUDA out of memory` | GPU VRAM exhausted | Use `VRAMOptimizer.auto_optimize()` or switch to smaller Whisper model |
| `PermissionError` on Windows when building CUDA | Missing admin/elevated permissions | Run terminal as Administrator |
| `resemblyzer not found` for speaker diarization | Optional speaker dependency | `pip install resemblyzer` |
| Tests fail with `pytest` | Missing test dependencies | `pip install pytest numpy` then `pytest tests/ -v` |

### Diagnostic Commands

```bash
# Check Python environment
python -c "import vram_core; print(vram_core.__version__, vram_core.CUDA_AVAILABLE)"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Run quick integration test
python test_run.py

# Run full test suite
pytest tests/ -v --tb=short

# Run only integration tests
pytest tests/test_integration.py -v

# Check VRAM status
python -c "from vram_core.vram_optimizer import VRAMOptimizer; o = VRAMOptimizer(); print(o.get_status())"
```

### Getting Help

- 📖 Check the [FAQ](docs/faq.md) for frequently asked questions
- 🐛 Report bugs via [GitHub Issues](https://github.com/Liangchenxu/Omni-VRAM/issues)
- 💬 Join discussions in [GitHub Discussions](https://github.com/Liangchenxu/Omni-VRAM/discussions)
- 📧 Contact: [Liangchenxu](https://github.com/Liangchenxu)

---

## 🤝 Contributing (English)

We welcome contributions of all kinds!

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Liangchenxu/Omni-VRAM.git
cd Omni-VRAM

# Install in development mode
pip install -e ".[dev]"

# Install test dependencies
pip install pytest pytest-cov numpy
```

### Contribution Workflow

1. **Fork** the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with tests
4. Run the test suite: `pytest tests/ -v`
5. Commit: `git commit -m 'feat: add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a **Pull Request**

### Code Standards

- All new modules must have corresponding unit tests in `tests/`
- Integration tests go in `tests/test_integration.py`
- Follow PEP 8 style guidelines
- Add docstrings for all public classes and methods
- Use type hints for function signatures
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation
  - `test:` for tests
  - `refactor:` for refactoring

### Project Architecture

```
vram_core/
├── __init__.py          # Public API exports + version
├── config.py            # Singleton config (YAML/env/hot-reload)
├── audio_utils.py       # Audio format detection & conversion
├── noise_reduction.py   # Spectral subtraction noise reduction
├── emotion_recognition.py # MFCC/energy emotion recognition
├── speaker_*.py         # Speaker diarization & verification
├── wake_word.py         # Wake word detection
├── streaming_asr.py     # Real-time ASR engine
├── meeting_summarizer.py # Meeting summarization
├── whisper/             # Whisper integration subpackage
├── chinese/             # Chinese NLP pipeline subpackage
├── plugin_manager.py    # Plugin system with hooks
└── monitoring.py        # GPU monitoring & metrics
```

---

## ⚠️ Disclaimer & Liability Waiver
**Hardware Interaction Warning:** Omni-VRAM interfaces directly with physical GPU hardware at the CUDA C++ level, employing aggressive zero-copy pointer manipulation to maximize throughput. 
While extensively tested, this software is provided **"as is"**, without warranty of any kind. The authors shall NOT be held liable for any kernel panics, system freezes, data loss, or hardware instability resulting from the use of this engine. **Use in production environments at your own risk.**

## 📜 License
Released under the [**MIT License**](https://opensource.org/licenses/MIT). 
You are free to use, modify, and distribute this software in both commercial and non-commercial projects, provided that the original copyright notice and this permission notice are included.

---
---

<a id="chinese-documentation"></a>
## 📖 简介 (Overview)

**Omni-VRAM** 是一个生产级的 **LLM 语音交互框架**，让大模型长出耳朵和嘴巴。基于 CUDA 零拷贝技术构建，提供 **28 个核心模块**，覆盖完整的语音 AI 管线——从语音识别到语音合成，从单 GPU 到分布式集群。

> **v2.5.0**：重大版本更新，新增音频增强器（7 级处理管线）、语音质量评估（SNR/PESQ-lite）、LLM 会议助手（多供应商 AI 分析）、边缘部署后端（ONNX/TensorRT/Lite）、85+ 集成测试等。

传统的 Python 音频处理管线和 PyTorch 操作（如 `torch.cat` 更新 KV-Cache）会引入严重的性能开销。Omni-VRAM 在硬件层面实现**算子融合**和**零拷贝内存注入**，使消费级显卡（RTX 30/40 系列）能够为实时语音助手提供亚毫秒级延迟。

### ✅ 核心功能（28 个模块）

| # | 模块 | 说明 |
|---|------|------|
| 1 | **Whisper 语音转写** | 多后端（faster-whisper / whisper.cpp / API / Distil-Whisper），tiny → large-v3.5，GPU 加速 5 倍 |
| 2 | **实时流式 ASR** | 滑动窗口 VAD，部分/最终结果回调，延迟 <500ms |
| 3 | **噪声消除** | 频谱减法、自适应维纳滤波、多级降噪管道、静音检测 |
| 4 | **情绪识别** | MFCC + 能量 + 过零率特征，6 种情绪（中性/开心/悲伤/愤怒/惊讶/恐惧） |
| 5 | **说话人分离** | MFCC 特征提取 + 余弦相似度聚类，自动识别"谁在什么时间说话" |
| 6 | **声纹验证** | MFCC 声纹提取，1:1 验证 & 1:N 识别，声纹持久化存储 |
| 7 | **唤醒词检测** | 能量检测 & 音素级模糊匹配，自定义唤醒词 |
| 8 | **TTS 语音合成** | 多后端（pyttsx3 / edge-tts / gTTS），300+ 音色，异步合成 |
| 9 | **语音翻译** | 语音到语音翻译管线，50+ 语言对 |
| 10 | **音频事件检测** | 能量阈值 + 频谱分析，检测咳嗽/笑声/掌声等事件 |
| 11 | **多 GPU 支持** | 自动设备发现，负载均衡（轮询/最少使用/显存优先） |
| 12 | **分布式转写** | 多机多卡并行批量处理，自动负载均衡 |
| 13 | **KV-Cache 显存优化** | 显存压力检测（LOW/MEDIUM/HIGH/CRITICAL），KV-Cache 估算，量化精度推荐 |
| 14 | **生产监控** | 实时 GPU 监控（显存/温度/功耗/利用率），`@gpu_monitor` 装饰器 |
| 15 | **REST API** | FastAPI 异步 HTTP + WebSocket 流式传输 |
| 16 | **gRPC 服务** | 高性能双协议（gRPC + REST）服务器 |
| 17 | **插件系统** | 可扩展架构，支持发现、生命周期与钩子事件 |
| 18 | **CUDA 内核** | 零拷贝 KV-Cache（11 倍加速），融合音频前端（28 倍加速） |
| 19 | **流式处理器** | 实时音频流处理，支持 VAD、缓冲区管理和分段处理 |
| 20 | **Whisper 桥接** | 模块化 Whisper 集成，含模型管理、CUDA 桥接和预处理 |
| 21 | **音频工具集** | 音频格式检测、转换、重采样、频谱计算 |
| 22 | **配置系统** | YAML/JSON 配置文件，环境变量覆盖，热重载 |
| 23 | **中文会议转写** | 标点恢复、文本规范化（数字/货币/日期）、中文分词、领域词典（医疗/科技/金融/法律/教育/政府）、方言支持 |
| 24 | **会议摘要生成** | AI 会议摘要：议题分段、决策检测、行动项提取、说话人贡献分析、多格式导出（Markdown/JSON/Dict） |
| 25 | **音频增强器** | 7 级增强管线（降噪→去混响→归一化→AGC→高通滤波→语音 EQ→噪声门），质量预设（快速/广播/录音室） |
| 26 | **语音质量评估** | SNR 估计、频谱清晰度、PESQ-lite 估计、削波检测、质量分级（优秀/良好/一般/较差） |
| 27 | **LLM 会议助手** | 多供应商 LLM 客户端（OpenAI/Claude/Ollama/自定义），AI 会议分析含议题/决策/行动项提取、情感分析、结构化 JSON 输出 |
| 28 | **边缘部署后端** | ONNX Runtime（CPU/GPU，INT8/INT4）、TensorRT（FP16/INT8）、Lite 后端（树莓派/Jetson Nano/移动端） |

### 📁 目录结构

```
Omni-VRAM/
├── app.py                      # Gradio Web Demo（语音转写/情绪/分离/麦克风）
├── vram_hacker.cu              # CUDA 核函数源码（KV-Cache 注入）
├── setup.py                    # 编译安装脚本
├── pyproject.toml              # 现代 Python 项目配置
├── requirements.txt            # Python 依赖清单
├── test_run.py                 # 快速集成测试
├── run_tests.py                # 统一测试运行器
├── .env.example                # 配置模板
├── Dockerfile                  # GPU Docker 镜像（CUDA + 音频库）
├── Dockerfile.cpu              # 纯 CPU Docker 镜像
├── docker-compose.yml          # 一键 Docker 部署
│
├── vram_core/                  # Python 核心库（24 个模块）
│   ├── __init__.py             # 包导出（v2.5.0）
│   ├── config.py               # 配置管理
│   ├── utils.py                # 通用工具函数
│   ├── audio_utils.py          # 音频格式检测与转换
│   ├── whisper_bridge.py       # Whisper 多后端集成（旧版）
│   ├── whisper/                # Whisper 子模块（v2.0）
│   │   ├── bridge.py           # CUDA Whisper 桥接
│   │   ├── models.py           # 模型管理
│   │   ├── optimizer.py        # Whisper CUDA Graph & 量化优化器
│   │   ├── preprocessor.py     # 音频预处理器
│   │   └── result.py           # 转录结果数据结构
│   ├── stream_processor.py     # 实时流处理器 + VAD
│   ├── streaming_asr.py        # 实时流式语音识别引擎
│   ├── realtime_optimizer.py   # 实时延迟优化器（自动调整分块大小）
│   ├── api_server.py           # FastAPI REST + WebSocket API
│   ├── noise_reduction.py      # STFT 谱减法噪声消除
│   ├── emotion_recognition.py  # 声学特征情绪识别
│   ├── speaker_diarization.py  # MFCC 说话人分离与聚类
│   ├── speaker_verification.py # 声纹验证（1:1 验证 & 1:N 识别）
│   ├── wake_word.py            # 唤醒词 / 关键词检测
│   ├── multi_gpu.py            # 多 GPU 管理与并行
│   ├── vram_optimizer.py       # KV-Cache 显存优化与 OOM 恢复
│   ├── tts_engine.py           # 多后端语音合成
│   ├── voice_translator.py     # 语音到语音翻译管线
│   ├── audio_event_detection.py # 音频事件检测
│   ├── distributed_transcriber.py # 多GPU/多机并行转写
│   ├── monitoring.py           # GPU 监控与 Prometheus 指标
│   ├── grpc_server.py          # gRPC + HTTP REST 双协议服务器
│   ├── plugin_manager.py       # 插件发现、加载与生命周期管理
│   ├── meeting_summarizer.py   # AI 会议摘要（议题/决策/行动项）
│   ├── audio_enhancer.py       # 7 级音频增强管线
│   ├── speech_quality.py       # 语音质量评估（SNR/PESQ-lite）
│   ├── llm_client.py           # 多供应商 LLM 客户端
│   ├── meeting_analyzer.py     # AI 会议分析（结构化输出）
│   ├── backends/               # 边缘部署后端
│   │   ├── onnx_backend.py     # ONNX Runtime 推理
│   │   ├── tensorrt_backend.py # TensorRT 优化推理
│   │   └── lite_backend.py     # 轻量级移动端/嵌入式推理
│   └── chinese/                # 中文 NLP 管线
│       ├── punctuation.py      # 中文标点恢复
│       ├── normalizer.py       # 文本规范化（数字/货币/日期）
│       ├── tokenizer.py        # 中文分词
│       ├── domain_dict.py      # 领域词典（医疗/科技/金融/法律）
│       └── dialect.py          # 粤语方言规范化
│
├── examples/                   # 示例应用
│   ├── realtime_voice_assistant.py  # 实时语音助手
│   ├── meeting_transcriber.py       # 会议录音转写与摘要
│   ├── voice_chat_bot.py            # 多轮语音对话机器人
│   ├── benchmark_suite.py           # 性能基准测试套件
│   ├── benchmark_v3.py              # v2.1.0 基准对比
│   ├── api_demo.py                  # API 服务端示例客户端
│   ├── test_whisper_local.py        # Whisper 本地测试
│   └── test_emotion.py              # 情绪识别测试
│
├── tests/                      # 单元 & 集成测试（28 个测试文件，85+ 测试用例）
│   ├── test_audio_utils.py
│   ├── test_emotion_recognition.py
│   ├── test_meeting_transcription.py # 会议转写与摘要测试
│   ├── test_monitoring.py
│   ├── test_multi_gpu.py
│   ├── test_noise_reduction.py
│   ├── test_plugin_manager.py
│   ├── test_speaker_diarization.py
│   ├── test_speaker_verification.py
│   ├── test_stream_processor.py
│   ├── test_tts_engine.py
│   ├── test_vram_optimizer.py
│   ├── test_wake_word.py
│   ├── test_whisper_bridge.py
│   ├── test_whisper_optimizer.py    # Whisper 优化测试
│   ├── test_integration.py          # 全管线集成测试
│   ├── test_websocket.py            # WebSocket API 测试
│   ├── test_realtime_latency.py     # 实时延迟测试
│   ├── test_v250.py                 # v2.5.0 功能测试（16 个用例）
│   └── benchmark_comparison.py      # 基准对比
│
└── docs/                       # 文档
    ├── installation.md
    ├── quickstart.md
    ├── api_reference.md
    ├── examples.md
    ├── faq.md
    └── blog_omni_vram.md
```

### 🐳 Docker 部署

```bash
# GPU 版本（含 CUDA 支持）
docker build -t omni-vram:gpu .
docker run --gpus all -p 8000:8000 omni-vram:gpu

# 纯 CPU 版本（无需 CUDA）
docker build -f Dockerfile.cpu -t omni-vram:cpu .
docker run -p 8000:8000 omni-vram:cpu

# 一键 docker-compose 部署
docker-compose up -d

# 带环境变量运行
docker run --gpus all \
  -e WHISPER_MODEL=base \
  -e DEFAULT_LANGUAGE=zh \
  -p 8000:8000 \
  omni-vram:gpu
```

### 🧪 示例目录

| 示例 | 说明 | 运行命令 |
|------|------|----------|
| **Gradio Web Demo** | Web 界面：转写、情绪、分离、麦克风录音 | `python app.py` |
| **实时语音助手** | 麦克风 → VAD → Whisper → 显示，支持文件录音 | `python examples/realtime_voice_assistant.py` |
| **会议录音转写** | 长时间录音，自动静音分段，导出文字结果 | `python examples/meeting_transcriber.py --output meeting.txt` |
| **语音对话机器人** | 多轮对话，对话历史跟踪，LLM 可接入架构 | `python examples/voice_chat_bot.py` |
| **性能基准测试** | 全模块性能测试，自动生成 Markdown 报告 | `python examples/benchmark_suite.py --skip-whisper` |
| **情绪识别** | 语音情绪分析演示 | `python examples/test_emotion.py` |
| **Whisper 本地测试** | 本地 Whisper 转写测试 | `python examples/test_whisper_local.py` |

### 🌐 Gradio Web Demo

一键启动交互式 Web 界面：

```bash
# 安装 Gradio（如尚未安装）
pip install gradio

# 启动演示（默认：http://localhost:7860）
python app.py

# 可选参数
python app.py --port 8080        # 自定义端口
python app.py --share            # 创建公网链接
python app.py --debug            # 调试模式
```

**功能：**
- 📝 **语音转写** — 上传音频 → 转写文字（支持模型/语言/降噪选项）
- 🎭 **情绪识别** — 上传音频 → 分析情绪（6 种情绪，概率条展示）
- 👥 **说话人分离** — 上传对话 → 识别谁在什么时间说话
- 🎙️ **实时麦克风** — 录音 → 即时转写
- 📥 **下载结果** — 导出为 JSON / TXT / SRT 字幕文件

---

## 📊 性能基准测试 (Benchmarks)

*硬件环境：NVIDIA RTX 3060 (12GB) | 平台：Windows WDDM | CUDA 版本：12.1*

### 1. KV-Cache 显存注入
*任务：在一个容量为 100,000、维度为 4096 的 KV-Cache 张量中，连续追加 100 次（每次 50 个 token）的新特征。*

| 引擎 / 方法 | 延迟 | 复杂度 | 爆显存(OOM) 风险 |
| :--- | :--- | :--- | :--- |
| PyTorch 原生 (`torch.cat`) | 90.32 ms | $O(N)$ (显存重新分配) | 极高 (显存碎片化) |
| **Omni-VRAM (零拷贝)** | **8.07 ms** | **$O(1)$ (底层指针偏移)** | **无** |
| **性能提升** | **11.19 倍** | - | - |

### 2. 音频处理管线
| 管线阶段 | 输入数据规模 | PyTorch / CPU 基准 | Omni-VRAM C++ 算子 | 加速比 |
| :--- | :--- | :--- | :--- | :--- |
| **并发 VAD 检测** | 10 分钟 (16kHz) | 9.45 ms (CPU `unfold`) | **0.33 ms** | **约 28 倍** |
| **融合特征提取** | 60 秒(16kHz) | 20.33 ms (VRAM 堆叠)| **1.05 ms** | **约 19 倍** |

### 3. Whisper 语音转写 (CPU)
| 模型 | 1 秒音频 | 5 秒音频 | 10 秒音频 |
| :--- | :--- | :--- | :--- |
| tiny | ~200ms | ~500ms | ~900ms |
| base | ~400ms | ~1200ms | ~2200ms |

> 运行 `python examples/benchmark_suite.py` 在你的硬件上进行自动化基准测试。

---

## 🛠️ 安装 (Installation)

```bash
# 快速安装（只装 Python 包，无 CUDA 内核）
pip install omni-vram

# 完整安装（含 CUDA 内核，享受 11 倍 / 28 倍加速）
git clone https://github.com/Liangchenxu/Omni-VRAM.git
cd Omni-VRAM
pip install -r requirements.txt

# 编译并安装 CUDA 扩展模块
# 注意：请确保已正确配置 NVCC 和 Visual Studio C++ 编译工具
python setup.py install

# (可选) 安装 Web API 服务器依赖
pip install fastapi uvicorn python-multipart

# (可选) 安装 whisper.cpp 用于本地语音转写
# 详见 docs/installation.md
```

### 配置文件

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件设置你的配置
# 至少需要设置 WHISPER_CPP_PATH 和 WHISPER_MODEL_PATH 用于本地转写
```

> 详细安装指南请参阅 [docs/installation.md](docs/installation.md)。

## 💡 快速开始 (Quick Start)

### Whisper 语音转写

```python
from vram_core.whisper_bridge import WhisperBridge, WhisperBackend

# 自动后端检测初始化
whisper = WhisperBridge(
    backend=WhisperBackend.AUTO,
    whisper_model="base",
    language="zh",
)

# 转写音频文件
result = whisper.transcribe("audio.wav")
print(f"文本: {result.text}")
print(f"置信度: {result.confidence}")
print(f"时长: {result.audio_duration}秒")
```

### 实时流处理

```python
import numpy as np
from vram_core.stream_processor import StreamProcessor, StreamConfig
from vram_core.whisper_bridge import WhisperBridge, WhisperBackend

# 初始化组件
whisper = WhisperBridge(backend=WhisperBackend.AUTO, whisper_model="base")
config = StreamConfig(sample_rate=16000, chunk_duration_ms=100, vad_threshold=0.02)
processor = StreamProcessor(config=config, whisper_bridge=whisper)

# 设置回调
processor.on_transcription = lambda result: print(f"转写结果: {result.text}")

# 喂入音频分块（如来自麦克风）
audio_chunk = np.random.randn(1600).astype(np.float32)
processor.feed(audio_chunk)
```

### 中文会议转写

```python
from vram_core.chinese.punctuation import PunctuationRestorer
from vram_core.chinese.normalizer import TextNormalizer
from vram_core.chinese.tokenizer import ChineseTokenizer
from vram_core.meeting_summarizer import MeetingSummarizer

# 为 ASR 原始文本恢复标点
restorer = PunctuationRestorer()
text = restorer.restore("今天下午三点开会讨论项目进度请各位准时参加")
# → "今天下午三点开会讨论项目进度，请各位准时参加。"

# 规范化数字、货币、日期
normalizer = TextNormalizer()
text = normalizer.normalize("一共花了3500元购买了50台设备")
# → "一共花了三千五百元购买了五十台设备"

# 中文分词
tokenizer = ChineseTokenizer()
tokens = tokenizer.tokenize("语音识别技术发展迅速")
# → ["语音识别", "技术", "发展", "迅速"]

# 会议摘要生成
summarizer = MeetingSummarizer()
minutes = summarizer.summarize(transcript_text, speaker_segments=segments)
print(minutes.summary)
for topic in minutes.topics:
    print(f"议题: {topic.title}")
for decision in minutes.decisions:
    print(f"决策: {decision.content}")
for action in minutes.action_items:
    print(f"行动: {action.description} → {action.assignee}")
```

### 说话人分离

```python
import numpy as np
from vram_core.speaker_diarization import SpeakerDiarizer

# 初始化分离器
diarizer = SpeakerDiarizer(n_mfcc=13, similarity_threshold=0.7)

# 加载音频（float32, 单声道, 16kHz）
audio = np.fromfile("audio.raw", dtype=np.float32)

# 执行说话人分离
result = diarizer.diarize(audio, sample_rate=16000)

for segment in result.segments:
    print(f"[{segment.start_time:.1f}s - {segment.end_time:.1f}s] 说话人: {segment.speaker_id}")

print(f"总说话人数: {diarizer.get_speaker_count()}")
```

### 声纹验证

```python
import numpy as np
from vram_core.speaker_verification import SpeakerVerifier

# 初始化验证器
verifier = SpeakerVerifier(threshold=0.75, storage_path="voiceprints.json")

# 注册声纹
audio = np.random.randn(16000).astype(np.float32)  # 1 秒音频
verifier.register("alice", audio, sample_rate=16000)

# 验证身份
test_audio = np.random.randn(16000).astype(np.float32)
result = verifier.verify("alice", test_audio)
print(f"验证结果: {result.verified}, 置信度: {result.confidence:.3f}")

# 1:N 识别
best = verifier.verify_any(test_audio)
if best:
    print(f"识别结果: {best.speaker_id} ({best.confidence:.3f})")
```

### 情绪识别

```python
import numpy as np
from vram_core.emotion_recognition import EmotionRecognizer

# 初始化
recognizer = EmotionRecognizer(sample_rate=16000)

# 分析情绪
audio = np.random.randn(32000).astype(np.float32)  # 2 秒音频
result = recognizer.recognize(audio)
print(f"情绪: {result['emotion']}, 置信度: {result['confidence']:.2f}")
```

### 噪声消除

```python
import numpy as np
from vram_core.noise_reduction import NoiseReducer

# 初始化
reducer = NoiseReducer(sample_rate=16000)

# 降噪处理
noisy_audio = np.random.randn(16000).astype(np.float32)
clean_audio = reducer.reduce_noise(noisy_audio, aggressiveness=0.7)
```

### VRAM 显存优化

```python
from vram_core.vram_optimizer import VRAMOptimizer

# 初始化
optimizer = VRAMOptimizer(device_id=0)

# 查看显存状态
status = optimizer.get_status()
print(f"GPU: {status.gpu_name}, 使用率: {status.usage_pct:.1f}%, 压力: {status.pressure.value}")

# 估算 KV-Cache 显存
estimate = VRAMOptimizer.estimate_kv_cache(n_layers=32, seq_length=2048, batch_size=1)
print(f"KV-Cache: {estimate.total_mb:.1f} MB")

# 获取量化精度推荐
dtype = optimizer.recommend_dtype(required_mb=4000)
print(f"推荐精度: {dtype}")

# 自动优化（高压力时自动清理）
optimizer.auto_optimize()
```

### TTS 语音合成

```python
from vram_core.tts_engine import TTSEngine

# 初始化（使用 edge-tts 后端）
engine = TTSEngine(backend="edge-tts")

# 合成语音
engine.synthesize("你好，世界！", output_path="output.mp3")
```

### Web API 服务

```bash
# 启动 API 服务
python -m vram_core.api_server --model base --language zh --port 8000
```

```python
# 客户端：文件上传转写
import requests
with open("audio.wav", "rb") as f:
    resp = requests.post("http://localhost:8000/transcribe", files={"file": f})
    print(resp.json()["text"])

# 客户端：WebSocket 流式转写
import websockets, asyncio
async def stream():
    async with websockets.connect("ws://localhost:8000/stream") as ws:
        await ws.send(audio_bytes)  # 16-bit PCM, 16kHz 单声道
        result = await ws.recv()
        print(result)
```

### 插件系统

```python
from vram_core.plugin_manager import PluginManager

# 初始化插件管理器
pm = PluginManager(plugin_dir="./plugins")

# 加载插件
pm.load_plugin("my_plugin")

# 列出已加载插件
for plugin in pm.list_plugins():
    print(f"插件: {plugin['name']} v{plugin['version']}")

# 注册钩子
pm.register_hook("on_transcription", my_callback)
```

> 更多示例请参阅 [docs/quickstart.md](docs/quickstart.md)。

---

## ⚠️ 免责声明 (Disclaimer)
**硬件交互警告：** Omni-VRAM 在 CUDA C++ 层级直接与物理 GPU 硬件交互，将采用激进的零拷贝指针操作以追求极限吞吐。
尽管已经过充分测试，但本软件仍按 *"原样 (as is)"* 提供，不作任何形式的保证。对于因使用本引擎而导致的任何内核崩溃、系统死锁、数据丢失或硬件不稳定，作者概不负责。**在生产环境中使用本软件，请自行承担一切风险。**

## 📜 协议 (License)
本项目基于 [**MIT License**](https://opensource.org/licenses/MIT) 开源。
您可以自由地在商业或非商业项目中使用、修改和分发本软件，但前提是必须保留原始版权声明及本许可声明。

---

## 🤝 贡献指南 (Contributing)

我们欢迎任何形式的贡献。

1. **Fork** 本仓库
2. 创建你的特性分支：`git checkout -b feature/amazing-feature`
3. 提交你的更改：`git commit -m 'feat: add amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 提交 **Pull Request**

请确保：
- 所有单元测试通过：`pytest tests/ -v`
- 新功能附带相应的测试用例
- 遵循项目代码风格

> 详细信息请参阅 [CHANGELOG.md](CHANGELOG.md) 了解版本历史，[docs/faq.md](docs/faq.md) 了解常见问题。

---

## ⭐ Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=Liangchenxu/Omni-VRAM&type=Date)](https://star-history.com/#Liangchenxu/Omni-VRAM&Date)

---

<div align="center">

**[⬅ 回到顶部](#omni-vram-llm-语音交互框架)**

Made with ❤️ by [Liangchenxu](https://github.com/Liangchenxu)

</div>