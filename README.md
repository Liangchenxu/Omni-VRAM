# Omni-VRAM: Zero-Copy CUDA Audio-to-LLM Bridge
### 零拷贝跨硬件语音大模型底层直通桥

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA: 11.0+](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)
![Platform: Windows/Linux](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)
![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
[![Tests](https://github.com/Liangchenxu/Omni-VRAM/actions/workflows/test.yml/badge.svg)](https://github.com/Liangchenxu/Omni-VRAM/actions/workflows/test.yml)
[![Version](https://img.shields.io/badge/Version-1.0.0-orange.svg)](https://github.com/Liangchenxu/Omni-VRAM/releases)

[**English**](#english-documentation) | [**中文文档**](#chinese-documentation) | [**Docs**](docs/)

---

<a id="english-documentation"></a>
## 📖 Overview

**Omni-VRAM** is a high-performance, lightweight CUDA extension designed to eliminate VRAM fragmentation and memory transfer bottlenecks in real-time LLM (Large Language Model) audio applications.

Traditional Python-based audio processing pipelines and PyTorch native operations (such as `torch.cat` for KV-Cache updates) introduce significant overhead and non-deterministic latency. Omni-VRAM solves this by implementing **Operator Fusion** and **Zero-Copy Memory Injection** directly at the hardware level, enabling consumer-grade GPUs (e.g., RTX 30/40 series) to achieve sub-millisecond end-to-end latency for real-time voice agents.

### ✨ Core Features

* **Zero-Copy KV-Cache Appender:** Bypasses PyTorch's dynamic memory reallocation (`torch.cat`) by pre-allocating continuous VRAM blocks and directly injecting hardware-level token embeddings ($O(1)$ complexity).
* **Fused Audio Frontend:** Performs Voice Activity Detection (VAD), Pre-emphasis, and Windowing (Hann) in a single CUDA kernel execution, eliminating intermediate VRAM allocations.
* **Hardware-Aware Radar:** Dynamically scans GPU architecture (`sm_XX`) and SM counts at runtime to dispatch the most optimal computation strategy.
* **Whisper Multi-Backend:** Supports **faster-whisper** (CTranslate2, recommended), whisper.cpp CLI, OpenAI API, and legacy Python whisper with automatic fallback chain.
* **Real-Time Streaming ASR:** Sliding-window VAD-based speech recognition with concurrent worker threads, partial/final result callbacks, and GPU batch acceleration.
* **Web API Server:** FastAPI-based REST + WebSocket server for transcription — file upload, base64 input, and real-time streaming endpoint.
* **Stream Processing:** Chunk-based audio stream processor with built-in VAD, segment extraction, and callback-driven architecture.
* **Audio Format Utilities:** Automatic format detection, sample rate conversion, stereo-to-mono, normalization, and WAV encoding.

### 📁 Project Structure

```
Omni-VRAM/
├── vram_hacker.cu              # CUDA kernel source (KV-Cache injection)
├── setup.py                    # Build & install script
├── test_run.py                 # Quick integration test
├── .env.example                # Configuration template
│
├── vram_core/                  # Python core library
│   ├── __init__.py             # Package exports
│   ├── config.py               # Configuration management (.env loader)
│   ├── audio_utils.py          # Audio format detection & conversion
│   ├── whisper_bridge.py       # Whisper multi-backend integration
│   ├── stream_processor.py     # Real-time stream processor + VAD
│   ├── streaming_asr.py        # Real-time streaming ASR engine
│   └── api_server.py           # FastAPI REST + WebSocket API
│
├── examples/                   # Example applications
│   ├── realtime_voice_assistant.py  # Real-time voice assistant
│   ├── meeting_transcriber.py       # Meeting transcription & summary
│   ├── voice_chat_bot.py            # Multi-turn voice chat bot
│   ├── benchmark_suite.py           # Performance benchmark suite
│   ├── api_demo.py                  # API server demo client
│   └── test_whisper_local.py        # Whisper local test script
│
├── tests/                      # Unit tests
│   ├── test_audio_utils.py
│   ├── test_whisper_bridge.py
│   └── test_stream_processor.py
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
| **Real-time Voice Assistant** | Microphone → VAD → Whisper → Display, with file recording | `python examples/realtime_voice_assistant.py` |
| **Meeting Transcriber** | Long-form recording with silence auto-segmentation and export | `python examples/meeting_transcriber.py --output meeting.txt` |
| **Voice Chat Bot** | Multi-turn dialogue with history tracking and LLM-ready architecture | `python examples/voice_chat_bot.py` |
| **Benchmark Suite** | Performance testing for all modules with Markdown report | `python examples/benchmark_suite.py --skip-whisper` |

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
# Clone the repository
git clone https://github.com/Liangchenxu/Omni-VRAM.git
cd Omni-VRAM

# Install all dependencies (core + audio + faster-whisper)
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

**Omni-VRAM** 是一款高性能、轻量级的 CUDA 底层扩展库，专为解决大语言模型（LLM）实时语音应用中的显存碎片化与数据搬运瓶颈而设计。

传统的基于 Python 的音频处理流以及 PyTorch 原生操作（例如使用 `torch.cat` 更新 KV-Cache）会引发严重的内存重新分配开销和不可控的延迟。Omni-VRAM 通过在硬件底层实现**算子融合（Operator Fusion）**与**零拷贝内存注入（Zero-Copy Memory Injection）**，使得消费级显卡（如 RTX 30/40 系列）能够为实时语音助手提供亚毫秒级的端到端计算延迟。

### ✨ 核心特性

* **零拷贝 KV-Cache 注入器:** 完全绕过 PyTorch 的动态内存分配（`torch.cat`），通过预分配连续的物理显存块，以硬件指针偏移的方式直接写入 Token 向量（$O(1)$ 时间复杂度）。
* **融合音频前处理核心:** 在单一 CUDA 核函数中并行完成语音活动检测（VAD）、预加重（Pre-emphasis）与汉宁窗（Hann Window）处理，彻底消除中间显存开销。
* **跨硬件自适应雷达:** 运行时动态扫描 GPU 架构（`sm_XX`）与流处理器簇（SM）数量，自动调度最优级别的计算策略。
* **Whisper 语音转写集成:** 多后端支持——**faster-whisper**（CTranslate2，推荐）、whisper.cpp 命令行、OpenAI API、Python whisper 库，自动回退链。
* **实时流式语音识别:** 基于滑动窗口 VAD 的流式 ASR，支持并发 Worker 线程、部分/最终结果回调、GPU 批处理加速。
* **Web API 服务:** 基于 FastAPI 的 REST + WebSocket 转写服务——文件上传、Base64 输入、实时流式端点。
* **实时流处理引擎:** 基于分块的音频流处理器，内置 VAD 检测、语音片段提取，支持回调驱动架构。
* **音频格式工具链:** 自动格式检测、采样率转换、立体声转单声道、归一化、WAV 编码。

### 📁 目录结构

```
Omni-VRAM/
├── vram_hacker.cu              # CUDA 核函数源码（KV-Cache 注入）
├── setup.py                    # 编译安装脚本
├── test_run.py                 # 快速集成测试
├── .env.example                # 配置模板
│
├── vram_core/                  # Python 核心库
│   ├── __init__.py             # 包导出
│   ├── config.py               # 配置管理（.env 加载）
│   ├── audio_utils.py          # 音频格式检测与转换
│   ├── whisper_bridge.py       # Whisper 多后端集成
│   ├── stream_processor.py     # 实时流处理器 + VAD
│   ├── streaming_asr.py        # 实时流式语音识别引擎
│   └── api_server.py           # FastAPI REST + WebSocket API
│
├── examples/                   # 示例应用
│   ├── realtime_voice_assistant.py  # 实时语音助手
│   ├── meeting_transcriber.py       # 会议录音转写与摘要
│   ├── voice_chat_bot.py            # 多轮语音对话机器人
│   ├── benchmark_suite.py           # 性能基准测试套件
│   ├── api_demo.py                  # API 服务演示客户端
│   └── test_whisper_local.py        # Whisper 本地测试
│
├── tests/                      # 单元测试
│   ├── test_audio_utils.py
│   ├── test_whisper_bridge.py
│   └── test_stream_processor.py
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
| **实时语音助手** | 麦克风 → VAD → Whisper → 显示，支持文件录制 | `python examples/realtime_voice_assistant.py` |
| **会议录音转写** | 长时间录音，自动静音分段，导出文字记录 | `python examples/meeting_transcriber.py --output meeting.txt` |
| **语音对话机器人** | 多轮对话，对话历史追踪，LLM 可接入架构 | `python examples/voice_chat_bot.py` |
| **性能基准测试** | 全模块性能测试，自动生成 Markdown 报告 | `python examples/benchmark_suite.py --skip-whisper` |

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
# 克隆项目仓库
git clone https://github.com/Liangchenxu/Omni-VRAM.git
cd Omni-VRAM

# 安装所有依赖（核心 + 音频 + faster-whisper）
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
