# Omni-VRAM API 参考文�?(API Reference)

本文档涵�?Omni-VRAM 所有公开 API。版�? `v0.2.0`

## 顶层导入

```python
import Omni-VRAM

# 或按模块导入
from Omni-VRAM import AudioProcessor, WhisperBridge, WhisperBackend
from Omni-VRAM import StreamProcessor, StreamConfig, StreamState
from Omni-VRAM import config, OmniConfig, setup_logging
```

---

## 1. 全局常量

| 名称 | 类型 | 说明 |
|------|------|------|
| `Omni-VRAM.__version__` | `str` | 版本�?|
| `Omni-VRAM.CUDA_AVAILABLE` | `bool` | CUDA 扩展是否可用 |

---

## 2. 配置管理 (`Omni-VRAM.config`)

### `OmniConfig`

全局配置单例，自动从 `.env` 文件加载环境变量�?

```python
from Omni-VRAM import config  # 全局单例实例

# 访问配置�?
print(config.openai_api_key)
print(config.whisper_cpp_path)
print(config.whisper_model_path)
print(config.whisper_language)
print(config.vram_sample_rate)
print(config.vram_log_level)

# 验证配置
errors = config.validate()
if errors:
    for err in errors:
        print(f"配置错误: {err}")

# 查看所有配�?
config.print_config()

# 运行时修�?
config.update(whisper_language="en", vram_sample_rate=44100)

# 重新加载 .env
config.reload()
```

#### 属性列�?

| 属�?| 类型 | 环境变量 | 默认�?| 说明 |
|------|------|----------|--------|------|
| `openai_api_key` | `Optional[str]` | `OPENAI_API_KEY` | `None` | OpenAI API 密钥 |
| `openai_model` | `str` | `OPENAI_MODEL` | `"whisper-1"` | API 模型�?|
| `whisper_cpp_path` | `Optional[str]` | `WHISPER_CPP_PATH` | `None` | whisper.cpp 目录 |
| `whisper_model_path` | `Optional[str]` | `WHISPER_MODEL_PATH` | `None` | GGML 模型路径 |
| `whisper_model_size` | `str` | `WHISPER_MODEL_SIZE` | `"base"` | Python whisper 模型大小 |
| `whisper_backend` | `str` | `WHISPER_BACKEND` | `"AUTO"` | 后端选择 |
| `whisper_output_format` | `str` | `WHISPER_OUTPUT_FORMAT` | `"json"` | 输出格式 |
| `whisper_threads` | `int` | `WHISPER_THREADS` | `0` | 推理线程�?|
| `whisper_language` | `Optional[str]` | `WHISPER_LANGUAGE` | `None` | 语言代码 |
| `vram_sample_rate` | `int` | `VRAM_SAMPLE_RATE` | `16000` | 目标采样�?|
| `vram_device` | `str` | `VRAM_DEVICE` | `"cuda"` | 计算设备 |
| `vram_log_level` | `str` | `VRAM_LOG_LEVEL` | `"INFO"` | 日志级别 |

#### 方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `validate()` | `() -> List[str]` | 验证配置，返回错误列�?|
| `reload()` | `() -> None` | 重新加载 .env 文件 |
| `update(**kwargs)` | `(Any) -> None` | 运行时更新配�?|
| `print_config()` | `() -> None` | 打印所有配置（隐藏敏感信息�?|

### `setup_logging`

```python
from Omni-VRAM import setup_logging

setup_logging("INFO")   # 设置全局日志级别
setup_logging("DEBUG")  # 调试模式
```

---

## 3. 音频处理 (`Omni-VRAM.audio_utils`)

### `AudioProcessor`

音频格式处理工具类。所有方法均支持 CPU NumPy 数组，兼�?CUDA 零拷贝管道�?

```python
from Omni-VRAM import AudioProcessor

# 实例方法（使用实例配置）
processor = AudioProcessor(target_sample_rate=16000)
result = processor.process_file("input.wav")

# 静�?类方法（直接调用�?
mono = AudioProcessor.stereo_to_mono(stereo_audio)
resampled = AudioProcessor.resample(audio, 44100, 16000)
```

#### 构造函�?

```python
AudioProcessor(target_sample_rate: int = 16000, target_channels: int = 1)
```

| 参数 | 类型 | 默认�?| 说明 |
|------|------|--------|------|
| `target_sample_rate` | `int` | `16000` | 目标采样�?|
| `target_channels` | `int` | `1` | 目标通道�?(1=mono, 2=stereo) |

#### 实例方法

| 方法 | 签名 | 返回�?| 说明 |
|------|------|--------|------|
| `process_file` | `(path, **opts) -> Tuple[ndarray, int]` | `(audio, sample_rate)` | 加载并处理音频文�?|
| `process_bytes` | `(data, format, **opts) -> Tuple[ndarray, int]` | `(audio, sample_rate)` | 处理原始字节数据 |
| `convert_to_target` | `(audio, sr) -> Tuple[ndarray, int]` | `(audio, sample_rate)` | 转换为目标格�?|

#### 静�?类方�?

| 方法 | 签名 | 返回�?| 说明 |
|------|------|--------|------|
| `load` | `(path, **opts) -> Tuple[ndarray, int]` | `(audio, sample_rate)` | 从文件加载音�?|
| `load_from_bytes` | `(data, fmt, **opts) -> Tuple[ndarray, int]` | `(audio, sample_rate)` | 从字节加载音�?|
| `detect_format` | `(path) -> str` | `"wav"/"mp3"/...` | 检测文件格�?|
| `stereo_to_mono` | `(audio) -> ndarray` | `mono_audio` | 立体声转单声�?|
| `resample` | `(audio, orig_sr, target_sr) -> ndarray` | `resampled` | 采样率转�?|
| `normalize` | `(audio) -> ndarray` | `normalized` | 峰值归一�?|
| `is_stereo` | `(audio) -> bool` | `True/False` | 判断是否立体�?|
| `to_wav_bytes` | `(audio, sample_rate) -> bytes` | `wav_bytes` | 导出 WAV 字节 |
| `get_duration` | `(audio, sr) -> float` | `seconds` | 计算音频时长 |

#### 加载选项 (`**opts`)

| 选项 | 类型 | 默认�?| 说明 |
|------|------|--------|------|
| `target_sample_rate` | `int` | `None` | 自动重采样到目标采样�?|
| `target_channels` | `int` | `None` | 自动转换通道�?|
| `dtype` | `str` | `"float32"` | 输出数据类型 |
| `mono` | `bool` | `True` | 是否自动转单声道 |

---

## 4. Whisper 转写 (`Omni-VRAM.whisper_bridge`)

### `WhisperBackend` (枚举)

```python
from Omni-VRAM import WhisperBackend

WhisperBackend.AUTO          # 自动选择最佳可用后�?
WhisperBackend.WHISPER_CPP   # whisper.cpp CLI
WhisperBackend.WHISPER_API   # OpenAI Cloud API
WhisperBackend.WHISPER_PYTHON # Python whisper �?
WhisperBackend.NONE          # 无后端（禁用转写�?
```

自动选择优先�? `API �?CLI �?Python �?None`

### `WhisperResult`

转写结果数据类�?

| 属�?| 类型 | 说明 |
|------|------|------|
| `text` | `str` | 转写文本 |
| `confidence` | `float` | 置信�?(0.0-1.0) |
| `language` | `str` | 检测到的语言 |
| `segments` | `List[dict]` | 带时间戳的分段列�?|
| `audio_duration` | `float` | 输入音频时长（秒�?|

`segments` 中每项结�?
```python
{
    "start": 0.0,   # 开始时间（秒）
    "end": 2.5,     # 结束时间（秒�?
    "text": "你好"   # 分段文本
}
```

### `TranscriptionResult`

`WhisperResult` 的别名，用于流处理回调�?

### `AudioPreprocessor`

音频预处理工具，用于转写前的格式标准化�?

```python
from Omni-VRAM.whisper_bridge import AudioPreprocessor

result = AudioPreprocessor.prepare_for_whisper(
    audio_data,         # np.ndarray �?bytes
    source_sr=44100,    # 原始采样�?
    target_sr=16000,    # 目标采样�?
)
# 返回: {"audio": ndarray, "sample_rate": 16000, "duration": float}
```

### `WhisperBridge`

核心转写桥接类�?

```python
from Omni-VRAM import WhisperBridge, WhisperBackend

whisper = WhisperBridge(
    backend=WhisperBackend.AUTO,    # 后端选择
    whisper_model="base",           # 模型大小/路径
    whisper_path=None,              # whisper.cpp 路径（覆�?.env�?
    model_path=None,                # 模型文件路径（覆�?.env�?
    device="cpu",                   # 计算设备
    language="zh",                  # 语言
)
```

#### 方法

| 方法 | 签名 | 返回�?| 说明 |
|------|------|--------|------|
| `transcribe` | `(audio, **kw) -> WhisperResult` | 转写结果 | 转写音频 |
| `get_status` | `() -> dict` | 状态字�?| 获取后端状�?|

#### `transcribe` 参数

```python
result = whisper.transcribe(
    audio,                   # str (文件路径) / np.ndarray / bytes
    sample_rate=16000,       # �?audio �?ndarray 时必须指�?
    language="zh",           # 覆盖默认语言
    task="transcribe",       # "transcribe" �?"translate"
)
```

#### `get_status` 返回�?

```python
{
    "backend": "whisper_cpp",        # 当前后端�?
    "backend_type": "WHISPER_CPP",   # 枚举值名
    "is_ready": True,                # 是否就绪
    "model": "base",                 # 模型标识
    "device": "cpu",                 # 设备
}
```

---

## 5. 流处理器 (`Omni-VRAM.stream_processor`)

### `StreamConfig`

流处理器配置�?

```python
from Omni-VRAM import StreamConfig

config = StreamConfig(
    sample_rate=16000,           # 采样�?(Hz)
    chunk_duration_ms=100,       # 每块时长 (ms)
    vad_threshold=0.02,          # VAD 能量阈�?(0.0-1.0)
    min_speech_duration_ms=300,  # 最短语音时�?(ms)
    silence_duration_ms=500,     # 判定结束的静音时�?(ms)
    max_segment_duration_s=30.0, # 最大分段时�?(s)
)
```

| 属�?| 类型 | 默认�?| 说明 |
|------|------|--------|------|
| `sample_rate` | `int` | `16000` | 采样�?|
| `chunk_duration_ms` | `int` | `100` | 每块时长 (ms) |
| `vad_threshold` | `float` | `0.02` | VAD 能量阈�?|
| `min_speech_duration_ms` | `int` | `300` | 最短语音时�?|
| `silence_duration_ms` | `int` | `500` | 结束判定静音时长 |
| `max_segment_duration_s` | `float` | `30.0` | 最大分段时�?|
| `chunk_size` | `int` | (自动计算) | 每块样本�?|
| `min_speech_chunks` | `int` | (自动计算) | 最短语音块�?|
| `silence_chunks` | `int` | (自动计算) | 结束判定块数 |

### `StreamState` (枚举)

```python
from Omni-VRAM import StreamState

StreamState.IDLE     # 空闲
StreamState.SPEAKING # 正在说话
StreamState.PROCESSING  # 正在转写
```

### `StreamProcessor`

实时音频流处理器�?

```python
from Omni-VRAM import StreamProcessor, StreamConfig

processor = StreamProcessor(
    config=config,               # StreamConfig 实例
    whisper_bridge=whisper,      # WhisperBridge 实例
)
```

#### 回调

| 回调 | 签名 | 触发时机 |
|------|------|----------|
| `on_transcription` | `(WhisperResult) -> None` | 转写完成�?|
| `on_state_change` | `(StreamState, StreamState) -> None` | 状态变化时 |

```python
processor.on_transcription = lambda result: print(result.text)
processor.on_state_change = lambda old, new: print(f"{old} �?{new}")
```

#### 方法

| 方法 | 签名 | 返回�?| 说明 |
|------|------|--------|------|
| `feed` | `(chunk: ndarray) -> None` | `None` | 喂入音频�?|
| `get_state` | `() -> StreamState` | `StreamState` | 获取当前状�?|
| `reset` | `() -> None` | `None` | 重置处理器状�?|

#### `feed` 方法

输入: `np.ndarray`，shape �?`(chunk_size,)`，dtype �?`float32`，值域 `[-1.0, 1.0]`�?

如果输入�?int16 格式，需先转�?
```python
audio_float = audio_int16.astype(np.float32) / 32768.0
processor.feed(audio_float)
```

---

## 6. CUDA 扩展 (`Omni-VRAM._vram_hacker`)

> 仅在 CUDA 扩展编译成功且有 NVIDIA GPU 时可用�? 
> 检�? `Omni-VRAM.CUDA_AVAILABLE`

### `scan_hardware_dna`

```python
info = Omni-VRAM.scan_hardware_dna()
# 返回 dict: {"compute_capability": "8.6", "sm_count": 28, ...}
```

### `append_to_kv_cache`

零拷�?KV-Cache 追加�?

```python
Omni-VRAM.append_to_kv_cache(kv_cache, new_tokens, current_pos)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `kv_cache` | `torch.Tensor` | 预分配的 KV-Cache 张量 `(max_seq, hidden_dim)` |
| `new_tokens` | `torch.Tensor` | �?token 向量 `(n_tokens, hidden_dim)` |
| `current_pos` | `torch.Tensor` | 当前位置指量 `int32`，原地更�?|

### `smart_audio_listen`

融合音频前端处理�?

```python
is_speaking, features = Omni-VRAM.smart_audio_listen(audio, threshold=0.5)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `audio` | `torch.Tensor` | GPU 上的 float32 音频 |
| `threshold` | `float` | VAD 阈�?|

| 返回�?| 类型 | 说明 |
|--------|------|------|
| `is_speaking` | `torch.Tensor` | 标量，是否检测到语音 |
| `features` | `torch.Tensor` | 提取的音频特�?|

### `query_memory`

```python
info = Omni-VRAM.query_memory()
# 返回 dict: {"total_mb": 12288, "used_mb": 1024, "free_mb": 11264}
```

### `inject_into_model`

底层模型注入（高级用法）�?

```python
Omni-VRAM.inject_into_model(model_handle, tensor_data, offset)
```

### `launch_dynamic_kernel`

动态内核调度（高级用法）�?

```python
Omni-VRAM.launch_dynamic_kernel(kernel_id, grid, block, *args)
```

### `stress_test`

显存压力测试�?

```python
Omni-VRAM.stress_test(size_mb=1024, iterations=100)
```

---

## 类型参�?

### numpy 数组约定

所有音频数据统一使用:
- **dtype**: `float32`
- **值域**: `[-1.0, 1.0]`
- **通道**: 单通道 (mono) �?1D 数组 `(n_samples,)`
- **立体�?*: 2D 数组 `(n_samples, 2)`

### 文件路径

接受 `str` �?`pathlib.Path` 对象