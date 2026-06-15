# Omni-VRAM 快速上手教�?(Quick Start)

本教程将引导你从零开始使�?Omni-VRAM 的核心功能�?

## 前提条件

- 已完�?[安装](installation.md)
- 已配�?`.env` 文件（至少设置了 Whisper 后端�?

## 1. 验证安装

```python
import Omni-VRAM

print(f"版本: {Omni-VRAM.__version__}")
print(f"CUDA 扩展: {'可用' if Omni-VRAM.CUDA_AVAILABLE else '不可�?}")
```

## 2. 音频格式工具

`AudioProcessor` 提供完整的音频格式处理工具链�?

### 2.1 加载与格式检�?

```python
from Omni-VRAM import AudioProcessor
import numpy as np

# 从文件加载音频（自动检测格式和采样率）
audio, sr = AudioProcessor.load("speech.wav")
print(f"采样�? {sr}Hz, 时长: {len(audio)/sr:.1f}s")

# 从原始字节加�?
with open("speech.wav", "rb") as f:
    audio_bytes = f.read()
audio, sr = AudioProcessor.load_from_bytes(audio_bytes, "wav")
```

### 2.2 格式转换

```python
# 立体声转单声�?
if AudioProcessor.is_stereo(audio):
    mono = AudioProcessor.stereo_to_mono(audio)

# 采样率转�?
if sr != 16000:
    resampled = AudioProcessor.resample(audio, sr, 16000)

# 归一�?
normalized = AudioProcessor.normalize(audio)

# 导出 WAV
wav_bytes = AudioProcessor.to_wav_bytes(audio, sample_rate=16000)
with open("output.wav", "wb") as f:
    f.write(wav_bytes)
```

## 3. Whisper 语音转写

### 3.1 基本转写

```python
from Omni-VRAM import WhisperBridge, WhisperBackend

# 初始化（自动选择最佳后端）
whisper = WhisperBridge(
    backend=WhisperBackend.AUTO,
    whisper_model="base",
    language="zh",
)

# 查看当前后端
status = whisper.get_status()
print(f"后端: {status['backend']}")

# 转写音频文件
result = whisper.transcribe("meeting.wav")
print(f"转写文本: {result.text}")
print(f"置信�? {result.confidence:.2f}")
print(f"音频时长: {result.audio_duration:.1f}s")
```

### 3.2 转写 NumPy 数组

```python
# 直接转写内存中的音频数据
audio = np.random.randn(16000 * 5).astype(np.float32)  # 5 �?16kHz 音频
result = whisper.transcribe(audio, sample_rate=16000)
print(result.text)
```

### 3.3 获取时间段信�?

```python
result = whisper.transcribe("speech.wav")

# 带时间戳的转写段�?
for seg in result.segments:
    print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
```

## 4. 实时流处�?

`StreamProcessor` 是实时语音处理的核心组件，它将音频流分为小块，自动进�?VAD 检测和语音转写�?

### 4.1 基本设置

```python
import numpy as np
from Omni-VRAM import (
    StreamProcessor, StreamConfig,
    WhisperBridge, WhisperBackend,
)

# 初始�?Whisper
whisper = WhisperBridge(
    backend=WhisperBackend.AUTO,
    whisper_model="base",
    language="zh",
)

# 配置流处理器
config = StreamConfig(
    sample_rate=16000,
    chunk_duration_ms=100,      # 每块 100ms
    vad_threshold=0.02,          # VAD 能量阈�?
    min_speech_duration_ms=300,  # 最短语�?300ms
    silence_duration_ms=500,     # 500ms 静音视为结束
)

processor = StreamProcessor(config=config, whisper_bridge=whisper)
```

### 4.2 设置回调

```python
# 转写结果回调
def on_transcription(result):
    print(f"📝 [{result.audio_duration:.1f}s] {result.text}")
    print(f"   置信�? {result.confidence:.2f}")

# 状态变化回�?
def on_state_change(old, new):
    print(f"状�? {old.value} �?{new.value}")

processor.on_transcription = on_transcription
processor.on_state_change = on_state_change
```

### 4.3 喂入音频数据

```python
# 模拟从麦克风读取音频�?
chunk_size = config.chunk_size  # 1600 samples (100ms @ 16kHz)

for i in range(1000):
    # 实际场景中从 PyAudio 读取
    audio_chunk = microphone.read(chunk_size)
    processor.feed(audio_chunk)
```

### 4.4 从麦克风实时处理

```python
import pyaudio

pa = pyaudio.PyAudio()
stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=config.chunk_size,
)

print("🎤 开始录音，�?Ctrl+C 停止...")

try:
    while True:
        data = stream.read(config.chunk_size, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        processor.feed(audio)
except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
```

## 5. CUDA 加速功�?

> 需要已编译 CUDA 扩展且有 NVIDIA GPU�?

### 5.1 硬件信息扫描

```python
import Omni-VRAM

if Omni-VRAM.CUDA_AVAILABLE:
    info = Omni-VRAM.query_memory()
    print(f"GPU 显存: {info}")
```

### 5.2 零拷�?KV-Cache 注入

```python
import torch
import Omni-VRAM

if not Omni-VRAM.CUDA_AVAILABLE:
    print("CUDA 不可�?)
    exit()

# 预分�?KV-Cache 显存
hidden_dim = 4096
max_seq = 100000
kv_cache = torch.zeros(max_seq, hidden_dim, device='cuda', dtype=torch.float32)
current_pos = torch.tensor([0], device='cuda', dtype=torch.int32)

# 模拟多轮 token 注入
for step in range(10):
    new_tokens = torch.randn(50, hidden_dim, device='cuda')
    Omni-VRAM.append_to_kv_cache(kv_cache, new_tokens, current_pos)
    print(f"步骤 {step}: 当前位置 {current_pos.item()}")
```

### 5.3 融合音频处理

```python
import torch
import Omni-VRAM

# 60 �?16kHz 音频
audio = torch.randn(960000, device='cuda', dtype=torch.float32)

# 一步完�?VAD + 预加�?+ 汉宁�?
is_speaking, features = Omni-VRAM.smart_audio_listen(audio, threshold=0.5)
print(f"是否在说�? {is_speaking.item()}")
print(f"特征形状: {features.shape}")
```

## 6. 完整示例：录音并转写

```python
import time
import numpy as np
from Omni-VRAM import (
    StreamProcessor, StreamConfig, StreamState,
    WhisperBridge, WhisperBackend, AudioProcessor,
)

def main():
    # 初始�?
    whisper = WhisperBridge(backend=WhisperBackend.AUTO, whisper_model="base")
    config = StreamConfig(sample_rate=16000, chunk_duration_ms=100)
    processor = StreamProcessor(config=config, whisper_bridge=whisper)

    # 收集结果
    results = []
    processor.on_transcription = lambda r: results.append(r)

    # 读取 WAV 文件并分块处�?
    audio, sr = AudioProcessor.load("speech.wav")
    if sr != config.sample_rate:
        audio = AudioProcessor.resample(audio, sr, config.sample_rate)
    audio = AudioProcessor.normalize(audio)

    chunk_size = config.chunk_size
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        processor.feed(chunk)

    # 输出结果
    print(f"\n共识�?{len(results)} 段语�?")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.audio_duration:.1f}s] {r.text}")

if __name__ == "__main__":
    main()
```

## 下一�?

- [API 参考文档](api_reference.md) �?查看完整 API 接口
- [示例项目](examples.md) �?了解更复杂的应用场景
- [常见问题](faq.md) �?遇到问题来这