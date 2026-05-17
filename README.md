# Omni-VRAM: Zero-Copy CUDA Audio-to-LLM Bridge
### 零拷贝跨硬件语音大模型底层直通桥

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![CUDA: 11.0+](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)
![Platform: Windows/Linux](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)
![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)

[**English**](#english-documentation) | [**中文文档**](#chinese-documentation)

---

<a id="english-documentation"></a>
## 📖 Overview

**Omni-VRAM** is a high-performance, lightweight CUDA extension designed to eliminate VRAM fragmentation and memory transfer bottlenecks in real-time LLM (Large Language Model) audio applications. 

Traditional Python-based audio processing pipelines and PyTorch native operations (such as `torch.cat` for KV-Cache updates) introduce significant overhead and non-deterministic latency. Omni-VRAM solves this by implementing **Operator Fusion** and **Zero-Copy Memory Injection** directly at the hardware level, enabling consumer-grade GPUs (e.g., RTX 30/40 series) to achieve sub-millisecond end-to-end latency for real-time voice agents.

### ✨ Core Features

* **Zero-Copy KV-Cache Appender:** Bypasses PyTorch's dynamic memory reallocation (`torch.cat`) by pre-allocating continuous VRAM blocks and directly injecting hardware-level token embeddings ($O(1)$ complexity).
* **Fused Audio Frontend:** Performs Voice Activity Detection (VAD), Pre-emphasis, and Windowing (Hann) in a single CUDA kernel execution, eliminating intermediate VRAM allocations.
* **Hardware-Aware Radar:** Dynamically scans GPU architecture (`sm_XX`) and SM counts at runtime to dispatch the most optimal computation strategy.

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

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Liangchenxu/Omni-VRAM.git
cd Omni-VRAM

# Build and install the CUDA extension
# Note: Ensure NVCC and Visual Studio C++ Build Tools are properly configured.
python setup.py install
```

## 💻 Quick Start

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

---

## 🛠️ 安装 (Installation)

```bash
# 克隆项目仓库
git clone https://github.com/Liangchenxu/Omni-VRAM.git
cd Omni-VRAM

# 编译并安装 CUDA 扩展模块
# 注意：请确保已正确配置 NVCC 与 Visual Studio C++ 编译工具
python setup.py install
```

## 💻 快速开始 (Quick Start)

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

## ⚠️ 免责声明 (Disclaimer)
**硬件交互警告：** Omni-VRAM 在 CUDA C++ 级别直接与物理 GPU 硬件交互，并采用激进的零拷贝指针操作以压榨极限吞吐量。
尽管经过了测试，但本软件按**“原样 (as is)”**提供，不作任何形式的担保。对于因使用本底层引擎而导致的任何内核崩溃、系统死锁、数据丢失或硬件不稳定，作者概不负责。**在生产环境中使用本软件，请自行承担一切风险。**

## 📜 协议 (License)
本项目基于 [**MIT License**](https://opensource.org/licenses/MIT) 开源。
您可以自由地在商业或非商业项目中使用、修改和分发本软件，但前提是必须保留原始版权声明及本许可声明。