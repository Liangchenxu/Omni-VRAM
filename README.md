# Omni-VRAM: Zero-Copy CUDA Audio-to-LLM Bridge
### 零拷贝跨硬件语音大模型底层直通桥

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA: 11.0+](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)
![Platform: Windows/Linux](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)
![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
[![Tests](https://github.com/Liangchenxu/Omni-VRAM/actions/workflows/test.yml/badge.svg)](https://github.com/Liangchenxu/Omni-VRAM/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/omni-vram.svg)](https://pypi.org/project/omni-vram/)
[![Version](https://img.shields.io/badge/Version-1.0.0-orange.svg)](https://github.com/Liangchenxu/Omni-VRAM/releases)

[**English**](#english-documentation) | [**中文文档**](#chinese-documentation) | [**Docs**](docs/)

---

<a id="english-documentation"></a>
## 📖 Overview

**Omni-VRAM** is a production-ready, high-performance audio AI platform built on CUDA zero-copy technology. It eliminates VRAM fragmentation and memory transfer bottlenecks for real-time LLM voice applications, providing 20 core modules covering the entire audio AI pipeline — from speech recognition to synthesis, from single GPU to distributed clusters.

Traditional Python audio pipelines and PyTorch operations (e.g., `torch.cat` for KV-Cache) introduce significant overhead. Omni-VRAM implements **Operator Fusion** and **Zero-Copy Memory Injection** at the hardware level, enabling consumer-grade GPUs (RTX 30/40 series) to achieve sub-millisecond latency for real-time voice agents.

### ✨ Core Features

| Module | Description |
|--------|-------------|
| **Whisper Transcription** | Multi-backend (faster-whisper / whisper.cpp / API / Distil-Whisper), tiny → large-v3.5, GPU 5× speedup |
| **Real-Time Streaming ASR** | Sliding-window VAD, partial/final callbacks, <500ms latency |
| **Noise Reduction** | WebRTC / RNNoise / noisereduce — three backends, auto-applied in pipeline |
| **Emotion Recognition** | wav2vec2 model, 7 emotions (happy/sad/angry/neutral/surprised/fear/disgust) |
| **Speaker Diarization** | pyannote-audio / resemblyzer, identifies "who spoke when" |
| **Speaker Verification** | MFCC voiceprint, 1:1 verification & 1:N identification, voiceprint library |
| **Wake Word Detection** | Energy-based & Whisper keyword detection, custom vocabulary |
| **TTS Engine** | edge-tts (300+ voices) / pyttsx3 (offline) |
| **Voice Translation** | Speech-to-speech pipeline, MarianMT + Google, 50+ language pairs |
| **Audio Event Detection** | YAMNet / energy-based, detects speech/music/alarm/silence |
| **Multi-GPU** | Pipeline / data / tensor parallelism, NVLink detection, fault tolerance |
| **Distributed Transcription** | Multi-machine parallel batch processing, auto load balancing |
| **KV-Cache VRAM Optimizer** | NF4/FP4 4-bit quantization, LRU eviction, OOM auto-recovery |
| **Production Monitoring** | Prometheus metrics, Grafana dashboards, health checks, p95/p99 latency |
| **REST API** | FastAPI async HTTP + WebSocket streaming |
| **gRPC Server** | High-performance dual-protocol (gRPC + REST) server |
| **Plugin System** | Extensible architecture with discovery, lifecycle & hook events |
| **CUDA Kernels** | Zero-Copy KV-Cache (11× faster), Fused Audio Frontend (28× faster) |

### 📁 Project Structure

```
Omni-VRAM/
├── app.py                      # Gradio Web Demo (语音转写/情绪/分离/麦克风)
├── vram_hacker.cu              # CUDA kernel source (KV-Cache injection)
├── setup.py                    # Build & install script
├── test_run.py                 # Quick integration test
├── .env.example                # Configuration template
│
├── vram_core/                  # Python core library
│   ├── __init__.py             # Package exports (v1.0.0)
│   ├── config.py               # Configuration management (.env loader)
│   ├── audio_utils.py          # Audio format detection & conversion
│   ├── whisper_bridge.py       # Whisper multi-backend integration
│   ├── stream_processor.py     # Real-time stream processor + VAD
│   ├── streaming_asr.py        # Real-time streaming ASR engine
│   ├── api_server.py           # FastAPI REST + WebSocket API
│   ├── noise_reduction.py      # STFT spectral subtraction noise reduction
│   ├── emotion_recognition.py  # Acoustic feature-based emotion recognition
│   ├── speaker_diarization.py  # MFCC speaker diarization & clustering
│   ├── speaker_verification.py # Speaker voiceprint verification (1:1 & 1:N)
│   ├── wake_word.py            # Wake word / keyword detection
│   ├── multi_gpu.py            # Multi-GPU management & parallelism
│   ├── vram_optimizer.py       # KV-Cache VRAM optimization & OOM recovery
│   ├── tts_engine.py           # Multi-backend text-to-speech (edge-tts / pyttsx3)
│   ├── voice_translator.py     # Speech-to-speech translation pipeline
│   ├── audio_event_detection.py # Audio event detection (YAMNet / energy-based)
│   ├── distributed_transcriber.py # Multi-GPU/machine parallel transcription
│   ├── monitoring.py           # Prometheus metrics & Grafana dashboards
│   ├── grpc_server.py          # gRPC + HTTP REST dual-protocol server
│   └── plugin_manager.py       # Plugin discovery, loading & lifecycle
│
├── examples/                   # Example applications
│   ├── realtime_voice_assistant.py  # Real-time voice assistant
│   ├── meeting_transcriber.py       # Meeting transcription & summary
│   ├── voice_chat_bot.py            # Multi-turn voice chat bot
│   ├── benchmark_suite.py           # Performance benchmark suite
│   ├── api_demo.py                  # API server demo client
│   ├── test_whisper_local.py        # Whisper local test script
│   ├── test_emotion.py              # Emotion recognition test
│   └── test_tts_translator.py       # TTS & translator test
│
├── tests/                      # Unit tests
│   ├── test_audio_utils.py
│   ├── test_whisper_bridge.py
│   ├── test_stream_processor.py
│   ├── test_noise_reduction.py
│   ├── test_emotion_recognition.py
│   └── test_speaker_diarization.py
│
└── docs/                       # Documentation
    ├── installation.md
    ├── quickstart.md
    ├── api_reference.md
    ├── examples.md
    └── faq.md
```

### 🧩 Examples

| Example | Description | Command |
|---------|-------------|---------|
| **Gradio Web Demo** | Web UI with transcription, emotion, diarization & mic recording | `python app.py` |
| **Real-time Voice Assistant** | Microphone → VAD → Whisper → Display, with file recording | `python examples/realtime_voice_assistant.py` |
| **Meeting Transcriber** | Long-form recording with silence auto-segmentation and export | `python examples/meeting_transcriber.py --output meeting.txt` |
| **Voice Chat Bot** | Multi-turn dialogue with history tracking and LLM-ready architecture | `python examples/voice_chat_bot.py` |
| **Benchmark Suite** | Performance testing for all modules with Markdown report | `python examples/benchmark_suite.py --skip-whisper` |
| **TTS & Translation** | Text-to-speech and speech-to-speech translation test | `python examples/test_tts_translator.py` |
| **Emotion Recognition** | Speech emotion analysis demo | `python examples/test_emotion.py` |

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
- 🎭 **Emotion Recognition** — Upload audio → detect emotion (7 emotions with probability bars)
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

## 💻 Quick Start

### Basic CUDA Operations

```python
import torch
import vram_core

# 1. Hardware Initialization
print(vram_core.scan_hardware_dna())

# 2. Fused Audio Processing
audio_stream = torch.randn(960000, device='cuda', dtype=torch.float32)
# Performs VAD, pre-emphasis, and windowing in ~1 ms
is_speaking, features = vram_core.smart_audio_listen(audio_stream, threshold=0.5)

# 3. Zero-Copy LLM KV-Cache Update
hidden_dim = 4096
max_seq_len = 100000
# Pre-allocate VRAM once
kv_cache = torch.zeros((max_seq_len, hidden_dim), device='cuda', dtype=torch.float32)
current_pos = torch.tensor([0], device='cuda', dtype=torch.int32)

if is_speaking.item():
    # Direct memory injection (0 reallocation overhead)
    new_tokens = torch.randn((50, hidden_dim), device='cuda', dtype=torch.float32)
    vram_core.append_to_kv_cache(kv_cache, new_tokens, current_pos)
```

### Whisper Transcription

```python
from vram_core import WhisperBridge, WhisperBackend

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
from vram_core import StreamProcessor, StreamConfig, WhisperBridge, WhisperBackend

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
from vram_core import WhisperBridge, WhisperBackend, StreamASR, StreamASRConfig

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

### Web API Server

```bash
# Start the API server
python vram_core/api_server.py --model base --language zh --port 8000
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

> See [docs/quickstart.md](docs/quickstart.md) for more examples.

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

**Omni-VRAM** 是一个生产级高性能语音 AI 平台，基于 CUDA 零拷贝技术构建。它消除了实时 LLM 语音应用中的显存碎片化与数据搬运瓶颈，提供 20 个核心模块，覆盖完整的语音 AI 管线——从语音识别到语音合成，从单卡到分布式集群。

传统的 Python 音频处理流和 PyTorch 操作（如 `torch.cat` 更新 KV-Cache）会引入严重的开销。Omni-VRAM 在硬件底层实现**算子融合**与**零拷贝内存注入**，使消费级显卡（RTX 30/40 系列）能够为实时语音助手提供亚毫秒级延迟。

### ✨ 核心功能

| 模块 | 说明 |
|------|------|
| **Whisper 语音转写** | 多后端（faster-whisper / whisper.cpp / API / Distil-Whisper），tiny → large-v3.5，GPU 加速 5 倍 |
| **实时流式 ASR** | 滑动窗口 VAD，部分/最终结果回调，延迟 <500ms |
| **噪声抑制** | WebRTC / RNNoise / noisereduce 三后端，自动集成到管线 |
| **情绪识别** | wav2vec2 模型，7 种情绪（开心/悲伤/愤怒/中性/惊讶/恐惧/厌恶） |
| **说话人分离** | pyannote-audio / resemblyzer，自动识别"谁在什么时候说话" |
| **声纹验证** | MFCC 声纹提取，1:1 验证 & 1:N 识别，声纹库管理 |
| **唤醒词检测** | 能量检测 & Whisper 关键词检测，自定义词汇 |
| **语音合成 TTS** | edge-tts（300+ 语音）/ pyttsx3（离线） |
| **语音翻译** | 语音到语音翻译管线，MarianMT + Google，50+ 语言对 |
| **音频事件检测** | YAMNet / 能量分析，检测语音/音乐/警报/静音 |
| **多 GPU 支持** | 管线/数据/张量并行，NVLink 检测，故障容错 |
| **分布式转录** | 多机多卡并行批量处理，自动负载均衡 |
| **KV-Cache 显存优化** | NF4/FP4 4-bit 量化，LRU 淘汰，OOM 自动恢复 |
| **生产监控** | Prometheus 指标，Grafana 仪表盘，健康检查，p95/p99 延迟 |
| **REST API** | FastAPI 异步 HTTP + WebSocket 流式服务 |
| **gRPC 服务** | 高性能双协议（gRPC + REST）服务器 |
| **插件系统** | 可扩展架构，支持发现、生命周期与钩子事件 |
| **CUDA 内核** | 零拷贝 KV-Cache（11 倍加速），融合音频前端（28 倍加速） |

### 📁 目录结构

```
Omni-VRAM/
├── app.py                      # Gradio Web Demo（语音转写/情绪/分离/麦克风）
├── vram_hacker.cu              # CUDA 核函数源码（KV-Cache 注入）
├── setup.py                    # 编译安装脚本
├── test_run.py                 # 快速集成测试
├── .env.example                # 配置模板
│
├── vram_core/                  # Python 核心库
│   ├── __init__.py             # 包导出（v1.0.0）
│   ├── config.py               # 配置管理（.env 加载）
│   ├── audio_utils.py          # 音频格式检测与转换
│   ├── whisper_bridge.py       # Whisper 多后端集成
│   ├── stream_processor.py     # 实时流处理器 + VAD
│   ├── streaming_asr.py        # 实时流式语音识别引擎
│   ├── api_server.py           # FastAPI REST + WebSocket API
│   ├── noise_reduction.py      # STFT 谱减法噪声抑制
│   ├── emotion_recognition.py  # 声学特征情绪识别
│   ├── speaker_diarization.py  # MFCC 说话人识别与聚类
│   ├── speaker_verification.py # 声纹验证（1:1 验证 & 1:N 识别）
│   ├── wake_word.py            # 唤醒词 / 关键词检测
│   ├── multi_gpu.py            # 多 GPU 管理与并行
│   ├── vram_optimizer.py       # KV-Cache 显存优化与 OOM 恢复
│   ├── tts_engine.py           # 多后端语音合成（edge-tts / pyttsx3）
│   ├── voice_translator.py     # 语音到语音翻译管线
│   ├── audio_event_detection.py # 音频事件检测（YAMNet / 能量分析）
│   ├── distributed_transcriber.py # 多 GPU/多机并行转录
│   ├── monitoring.py           # Prometheus 指标与 Grafana 仪表盘
│   ├── grpc_server.py          # gRPC + HTTP REST 双协议服务器
│   └── plugin_manager.py       # 插件发现、加载与生命周期管理
│
├── examples/                   # 示例应用
│   ├── realtime_voice_assistant.py  # 实时语音助手
│   ├── meeting_transcriber.py       # 会议录音转写与摘要
│   ├── voice_chat_bot.py            # 多轮语音对话机器人
│   ├── benchmark_suite.py           # 性能基准测试套件
│   ├── api_demo.py                  # API 服务演示客户端
│   ├── test_whisper_local.py        # Whisper 本地测试
│   ├── test_emotion.py              # 情绪识别测试
│   └── test_tts_translator.py       # 语音合成与翻译测试
│
├── tests/                      # 单元测试
│   ├── test_audio_utils.py
│   ├── test_whisper_bridge.py
│   ├── test_stream_processor.py
│   ├── test_noise_reduction.py
│   ├── test_emotion_recognition.py
│   └── test_speaker_diarization.py
│
└── docs/                       # 文档
    ├── installation.md
    ├── quickstart.md
    ├── api_reference.md
    ├── examples.md
    └── faq.md
```

### 🧩 示例项目

| 示例 | 说明 | 运行命令 |
|------|------|----------|
| **Gradio Web Demo** | Web 界面：转写、情绪、分离、麦克风录音 | `python app.py` |
| **实时语音助手** | 麦克风 → VAD → Whisper → 显示，支持文件录制 | `python examples/realtime_voice_assistant.py` |
| **会议录音转写** | 长时间录音，自动静音分段，导出文字记录 | `python examples/meeting_transcriber.py --output meeting.txt` |
| **语音对话机器人** | 多轮对话，对话历史追踪，LLM 可接入架构 | `python examples/voice_chat_bot.py` |
| **性能基准测试** | 全模块性能测试，自动生成 Markdown 报告 | `python examples/benchmark_suite.py --skip-whisper` |
| **语音合成与翻译** | TTS 语音合成和语音到语音翻译测试 | `python examples/test_tts_translator.py` |
| **情绪识别** | 语音情绪分析演示 | `python examples/test_emotion.py` |

### 🌐 Gradio Web Demo

一条命令启动交互式 Web 界面：

```bash
# 安装 Gradio（如尚未安装）
pip install gradio

# 启动演示（默认: http://localhost:7860）
python app.py

# 可选参数
python app.py --port 8080        # 自定义端口
python app.py --share            # 创建公网链接
python app.py --debug            # 调试模式
```

**功能：**
- 📝 **语音转写** — 上传音频 → 转写文字（支持模型/语言/噪声抑制选项）
- 🎭 **情绪识别** — 上传音频 → 分析情绪（7 种情绪，带概率条形图）
- 👥 **说话人分离** — 上传对话音频 → 识别谁在什么时候说话
- 🎙️ **实时麦克风** — 录制语音 → 即时转写
- 📥 **下载结果** — 导出为 JSON / TXT / SRT 字幕文件

---

## 📊 性能基准测试 (Benchmarks)

*硬件环境: NVIDIA RTX 3060 (12GB) | 平台: Windows WDDM | CUDA 版本: 12.1*

### 1. KV-Cache 显存注入
*任务：在一个容量为 100,000、维度为 4096 的 KV-Cache 张量中，连续追加 100 次（每次 50 个 token）的新特征。*

| 引擎 / 方法 | 延迟 | 复杂度 | 爆显存 (OOM) 风险 |
| :--- | :--- | :--- | :--- |
| PyTorch 原生 (`torch.cat`) | 90.32 ms | $O(N)$ (显存重新分配) | 极高 (显存碎片化) |
| **Omni-VRAM (零拷贝)** | **8.07 ms** | **$O(1)$ (底层指针偏移)** | **无** |
| **性能提升** | **11.19 倍** | - | - |

### 2. 音频处理管线
| 管线阶段 | 输入数据规模 | PyTorch / CPU 基准线 | Omni-VRAM C++ 算子 | 加速比 |
| :--- | :--- | :--- | :--- | :--- |
| **并发 VAD 检测** | 10 分钟 (16kHz) | 9.45 ms (CPU `unfold`) | **0.33 ms** | **约 28 倍** |
| **融合特征提取** | 60 秒 (16kHz) | 20.33 ms (VRAM 堆叠)| **1.05 ms** | **约 19 倍** |

### 3. Whisper 语音转写 (CPU)
| 模型 | 1 秒音频 | 5 秒音频 | 10 秒音频 |
| :--- | :--- | :--- | :--- |
| tiny | ~200ms | ~500ms | ~900ms |
| base | ~400ms | ~1200ms | ~2200ms |

> 运行 `python examples/benchmark_suite.py` 在你的硬件上进行自动化基准测试。

---

## 🛠️ 安装 (Installation)

```bash
# 快速安装（仅 Python 包，无 CUDA 内核）
pip install omni-vram

# 完整安装（含 CUDA 内核，获得 11 倍 / 28 倍加速）
git clone https://github.com/Liangchenxu/Omni-VRAM.git
cd Omni-VRAM
pip install -r requirements.txt

# 编译并安装 CUDA 扩展模块
# 注意：请确保已正确配置 NVCC 与 Visual Studio C++ 编译工具
python setup.py install

# (可选) 安装 Web API 服务依赖
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

## 💻 快速开始 (Quick Start)

### 基本 CUDA 操作

```python
import torch
import vram_core

# 1. 硬件底层雷达初始化
print(vram_core.scan_hardware_dna())

# 2. 算子融合音频处理
audio_stream = torch.randn(960000, device='cuda', dtype=torch.float32)
# 1毫秒内并发完成 VAD 检测、预加重与加窗
is_speaking, features = vram_core.smart_audio_listen(audio_stream, threshold=0.5)

# 3. 零拷贝大模型 KV-Cache 更新
hidden_dim = 4096
max_seq_len = 100000
# 仅进行一次物理显存预分配
kv_cache = torch.zeros((max_seq_len, hidden_dim), device='cuda', dtype=torch.float32)
current_pos = torch.tensor([0], device='cuda', dtype=torch.int32)

if is_speaking.item():
    # 物理级显存直通注入（0 内存重新分配开销）
    new_tokens = torch.randn((50, hidden_dim), device='cuda', dtype=torch.float32)
    vram_core.append_to_kv_cache(kv_cache, new_tokens, current_pos)
```

### Whisper 语音转写

```python
from vram_core import WhisperBridge, WhisperBackend

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
from vram_core import StreamProcessor, StreamConfig, WhisperBridge, WhisperBackend

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

### 实时流式语音识别 (Streaming ASR)

```python
import numpy as np
from vram_core import WhisperBridge, WhisperBackend, StreamASR, StreamASRConfig

# 初始化 Whisper
whisper = WhisperBridge(backend=WhisperBackend.AUTO, whisper_model="base")

# 配置流式 ASR
config = StreamASRConfig(
    sample_rate=16000,
    vad_threshold=0.015,
    language="zh",
)
asr = StreamASR(config=config, whisper_bridge=whisper)

# 设置回调
asr.on_partial_result = lambda text: print(f"[部分] {text}")
asr.on_final_result = lambda result: print(f"[最终] {result.text}")

# 启动并喂入音频
asr.start()
audio_chunk = np.random.randn(3200).astype(np.float32)  # 来自麦克风
asr.feed(audio_chunk)
```

### Web API 服务

```bash
# 启动 API 服务
python vram_core/api_server.py --model base --language zh --port 8000
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

> 更多示例请参阅 [docs/quickstart.md](docs/quickstart.md)。

---

## ⚠️ 免责声明 (Disclaimer)
**硬件交互警告：** Omni-VRAM 在 CUDA C++ 级别直接与物理 GPU 硬件交互，并采用激进的零拷贝指针操作以压榨极限吞吐量。
尽管经过了测试，但本软件按**"原样 (as is)"**提供，不作任何形式的担保。对于因使用本底层引擎而导致的任何内核崩溃、系统死锁、数据丢失或硬件不稳定，作者概不负责。**在生产环境中使用本软件，请自行承担一切风险。**

## 📜 协议 (License)
本项目基于 [**MIT License**](https://opensource.org/licenses/MIT) 开源。
您可以自由地在商业或非商业项目中使用、修改和分发本软件，但前提是必须保留原始版权声明及本许可声明。

---

## 🤝 贡献指南 (Contributing)

我们欢迎任何形式的贡献！

1. **Fork** 本仓库
2. 创建你的特性分支：`git checkout -b feature/amazing-feature`
3. 提交你的修改：`git commit -m 'feat: add amazing feature'`
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

**[⬆ 回到顶部](#omni-vram-zero-copy-cuda-audio-to-llm-bridge)**

Made with ❤️ by [Liangchenxu](https://github.com/Liangchenxu)

</div>
