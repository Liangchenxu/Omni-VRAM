# Omni-VRAM 安装指南 (Installation Guide)

## 系统要求

| 项目 | 最低要�?| 推荐配置 |
|------|----------|----------|
| **操作系统** | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| **GPU** | NVIDIA GPU (Compute Capability �?6.0) | RTX 3060+ / RTX 4060+ |
| **CUDA** | CUDA Toolkit 11.0 | CUDA Toolkit 12.x |
| **Python** | Python 3.8 | Python 3.10+ |
| **编译�?* | MSVC 2019+ (Windows) / GCC 9+ (Linux) | MSVC 2022 / GCC 11+ |
| **RAM** | 8 GB | 16 GB+ |
| **VRAM** | 4 GB | 8 GB+ |

## 1. 基础安装

### 1.1 克隆项目

```bash
git clone https://github.com/Liangchenxu/Omni-VRAM.git
cd Omni-VRAM
```

### 1.2 安装 Python 依赖

```bash
# 核心依赖
pip install numpy torch python-dotenv

# 音频处理依赖（用于实时语音功能）
pip install pyaudio pydub soundfile

# Whisper Python 库（可选，用于本地转写�?
pip install openai-whisper
```

### 1.3 编译 CUDA 扩展

#### Windows

```powershell
# 确保已安装：
# 1. Visual Studio 2019/2022（含 C++ 桌面开发工作负载）
# 2. CUDA Toolkit（匹配你�?GPU 驱动版本�?

# 设置环境变量（如�?NVCC 不在 PATH 中）
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
$env:PATH += ";$env:CUDA_HOME\bin"

# 编译安装
python setup.py install
```

#### Linux

```bash
# 确保已安装：
# sudo apt install build-essential
# CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

python setup.py install
```

### 1.4 验证安装

```python
import Omni-VRAM

print(f"Version: {Omni-VRAM.__version__}")
print(f"CUDA Extension: {Omni-VRAM.CUDA_AVAILABLE}")

if Omni-VRAM.CUDA_AVAILABLE:
    print(Omni-VRAM.query_memory())
```

## 2. 配置 Whisper

Omni-VRAM 支持三种 Whisper 后端，推荐按以下优先级配置：

### 2.1 方案 A：whisper.cpp（推�?�?本地、高性能�?

```bash
# 1. 编译 whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make

# 2. 下载模型
bash ./models/download-ggml-model.sh base
# 模型会下载到 models/ggml-base.bin

# 3. 配置 .env
cp .env.example .env
# 编辑 .env，设置：
#   WHISPER_CPP_PATH=/path/to/whisper.cpp/build/bin
#   WHISPER_MODEL_PATH=/path/to/whisper.cpp/models/ggml-base.bin
```

**模型大小参考：**

| 模型 | 文件大小 | 内存占用 | 速度 (CPU) | 准确�?|
|------|----------|----------|------------|--------|
| tiny | ~75 MB | ~400 MB | 最�?| 一�?|
| base | ~142 MB | ~500 MB | �?| 良好 |
| small | ~466 MB | ~1.0 GB | 中等 | 较好 |
| medium | ~1.5 GB | ~2.6 GB | 较慢 | �?|
| large | ~3.1 GB | ~4.7 GB | �?| 最�?|

### 2.2 方案 B：OpenAI Whisper API（云服务�?

```bash
# �?.env 中设置：
#   OPENAI_API_KEY=sk-your-api-key-here
```

需要联网，�?token 计费。延迟取决于网络�?

### 2.3 方案 C：Python whisper 库（本地、简易）

```bash
pip install openai-whisper

# .env 中无需额外配置，WhisperBridge 会自动检�?
```

首次使用会自动下载模型。推理速度�?whisper.cpp 慢�?

### 2.4 自动后端选择

当设�?`WHISPER_BACKEND=AUTO`（默认）时，系统按以下顺序尝试：

```
OpenAI API �?whisper.cpp CLI �?Python whisper �?None（无转写�?
```

只要配置了对应的路径/API Key，系统会自动选择可用的后端�?

## 3. 音频设备配置

### PyAudio 安装

```bash
pip install pyaudio
```

**Windows 常见问题�?* 如果 `pip install pyAudio` 失败，需要安装预编译�?wheel�?

```bash
pip install pipwin
pipwin install pyaudio
```

**Linux�?*

```bash
sudo apt install portaudio19-dev
pip install pyaudio
```

**macOS�?*

```bash
brew install portaudio
pip install pyaudio
```

### 检测音频设�?

```python
import pyaudio

pa = pyaudio.PyAudio()
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"[{i}] {info['name']} (inputs: {info['maxInputChannels']})")
pa.terminate()
```

## 4. 环境变量参�?

完整配置项见 `.env.example`，以下是关键配置�?

| 变量 | 说明 | 默认�?|
|------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | �?|
| `WHISPER_CPP_PATH` | whisper.cpp 可执行文件目�?| �?|
| `WHISPER_MODEL_PATH` | GGML 模型文件路径 | �?|
| `WHISPER_MODEL_SIZE` | Python whisper 模型大小 | `base` |
| `WHISPER_BACKEND` | 后端选择 | `AUTO` |
| `WHISPER_LANGUAGE` | 语言代码 | `zh` |
| `VRAM_SAMPLE_RATE` | 目标采样�?(Hz) | `16000` |
| `VAD_THRESHOLD` | VAD 能量阈�?| `0.02` |
| `VRAM_LOG_LEVEL` | 日志级别 | `INFO` |

## 5. 故障排除

### NVCC 找不�?

```
error: command 'nvcc' failed
```

**解决�?* 确保 CUDA Toolkit 已安装且 `nvcc` �?PATH 中：

```bash
nvcc --version
```

### 编译失败：找不到 Visual Studio

```
error: Microsoft Visual C++ 14.0 or greater is required
```

**解决�?* 安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，勾�?"C++ 桌面开�?�?

### PyTorch CUDA 版本不匹�?

```
CUDA error: no kernel image is available for execution on the device
```

**解决�?* 确保 PyTorch �?CUDA 版本与系�?CUDA Toolkit 版本一致：

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

### �?GPU 环境

如果没有 NVIDIA GPU，Omni-VRAM 的核�?CUDA 功能不可用，�?Whisper 转写、音频处理等 Python 模块仍可正常使用（CPU 模式）�?

## 下一�?

- [快速上手教程](quickstart.md) �?运行第一个示�?
- [API 参考文档](api_reference.md) �?查看完整 API
- [示例项目](examples.md) �?了解实际应用场景