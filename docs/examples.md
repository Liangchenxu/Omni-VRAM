# Omni-VRAM 示例项目说明 (Examples)

本目录包�?4 个完整示例应用，展示 Omni-VRAM 在不同场景下的使用方式�?

## 快速导�?

| 示例 | 场景 | 难度 | 依赖 |
|------|------|------|------|
| [实时语音助手](#实时语音助手) | 实时麦克风录�?+ VAD + 转写 | ⭐⭐ | PyAudio |
| [会议录音转写](#会议录音转写) | 长时间录�?+ 自动分段 + 导出 | ⭐⭐ | PyAudio, pydub |
| [语音对话机器人](#语音对话机器�? | 多轮语音对话 + 历史管理 | ⭐⭐�?| PyAudio |
| [基准测试套件](#基准测试套件) | 全模块性能测试 | �?| NumPy, PyTorch |

---

## 实时语音助手

**文件:** `examples/realtime_voice_assistant.py`

### 功能

- 从麦克风实时采集音频�?
- 基于能量阈值的 VAD 语音活动检�?
- 自动收集语音片段，静音时触发 Whisper 转写
- 实时显示转写结果和置信度
- 支持录制音频保存到文�?
- 优雅退出（Ctrl+C）并显示会话摘要

### 使用方法

```bash
# 基本使用
python examples/realtime_voice_assistant.py

# 指定麦克风设�?
python examples/realtime_voice_assistant.py --device 2

# 保存录音到文�?
python examples/realtime_voice_assistant.py --save recording.wav

# 调整 VAD 灵敏度（值越小越灵敏�?
python examples/realtime_voice_assistant.py --threshold 0.01

# 指定 Whisper 模型
python examples/realtime_voice_assistant.py --model small

# 调试模式
python examples/realtime_voice_assistant.py --verbose
```

### 命令行参�?

| 参数 | 类型 | 默认�?| 说明 |
|------|------|--------|------|
| `--device` | `int` | 默认设备 | PyAudio 麦克风设备索�?|
| `--save` | `str` | 不保�?| 录音保存文件路径 (WAV) |
| `--threshold` | `float` | `0.02` | VAD 能量阈�?|
| `--model` | `str` | `base` | Whisper 模型大小 |
| `--language` | `str` | `zh` | 语言代码 |
| `--backend` | `str` | `AUTO` | Whisper 后端 |
| `--verbose` | flag | `False` | 调试模式 |

### 工作流程

```
麦克�?�?PyAudio �?int16 �?float32 �?StreamProcessor.feed()
                                              �?
                                     VAD 能量检�?
                                      �?          �?
                                    静音         语音
                                     �?            �?
                                   忽略      缓冲语音片段
                                               �?(静音 500ms)
                                         WhisperBridge.transcribe()
                                               �?
                                          显示转写结果
```

---

## 会议录音转写

**文件:** `examples/meeting_transcriber.py`

### 功能

- 从麦克风录制长时间会议音�?
- 静音自动分段（每段最�?30 秒强制切分）
- 每段实时转写并显�?
- 会话结束时导出完整转写记录（纯文�?/ JSON�?
- 支持从文件导入音频进行离线转�?
- 自动生成摘要统计

### 使用方法

```bash
# 从麦克风录制并转�?
python examples/meeting_transcriber.py

# 导出到文�?
python examples/meeting_transcriber.py --output meeting.txt

# JSON 格式导出（含时间戳）
python examples/meeting_transcriber.py --output meeting.json --format json

# 从已有音频文件转�?
python examples/meeting_transcriber.py --input recording.wav --output minutes.txt

# 使用更大�?Whisper 模型（更准确�?
python examples/meeting_transcriber.py --model medium
```

### 命令行参�?

| 参数 | 类型 | 默认�?| 说明 |
|------|------|--------|------|
| `--output` / `-o` | `str` | `meeting_minutes.txt` | 输出文件路径 |
| `--format` / `-f` | `str` | `txt` | 输出格式: `txt` / `json` |
| `--input` / `-i` | `str` | 麦克�?| 输入音频文件路径 |
| `--model` | `str` | `base` | Whisper 模型大小 |
| `--language` | `str` | `zh` | 语言代码 |
| `--threshold` | `float` | `0.02` | VAD 阈�?|
| `--verbose` | flag | `False` | 调试模式 |

### 输出格式

**纯文�?(`--format txt`):**
```
=== Omni-VRAM Meeting Transcription ===
Date: 2024-01-15
Total segments: 5
Total duration: 125.3s

[Segment 1] (12.3s, confidence: 0.85)
大家好，今天我们讨论项目进展...

[Segment 2] (8.7s, confidence: 0.92)
首先看一下上周的完成情况...

=== End of Transcription ===
```

**JSON (`--format json`):**
```json
{
  "metadata": {
    "date": "2024-01-15T14:30:00",
    "total_segments": 5,
    "total_duration": 125.3
  },
  "segments": [
    {
      "index": 0,
      "duration": 12.3,
      "confidence": 0.85,
      "text": "大家好，今天我们讨论项目进展..."
    }
  ]
}
```

---

## 语音对话机器�?

**文件:** `examples/voice_chat_bot.py`

### 功能

- 语音输入 + 文字输出的多轮对�?
- 完整的对话历史管理（上下文保持）
- 对话导出（纯文本 / JSON�?
- 预留 LLM API 接入点（当前使用回显模式�?
- 会话统计和摘�?

### 使用方法

```bash
# 基本使用
python examples/voice_chat_bot.py

# 导出对话记录
python examples/voice_chat_bot.py --output chat_log.json --format json

# 设置助手名称
python examples/voice_chat_bot.py --name "小助�?

# 使用更大模型
python examples/voice_chat_bot.py --model small

# 调试模式
python examples/voice_chat_bot.py --verbose
```

### 命令行参�?

| 参数 | 类型 | 默认�?| 说明 |
|------|------|--------|------|
| `--output` / `-o` | `str` | `chat_log.txt` | 对话记录输出路径 |
| `--format` / `-f` | `str` | `txt` | 输出格式: `txt` / `json` |
| `--model` | `str` | `base` | Whisper 模型大小 |
| `--name` | `str` | `"VRAM Assistant"` | 助手名称 |
| `--language` | `str` | `zh` | 语言代码 |
| `--backend` | `str` | `AUTO` | Whisper 后端 |
| `--threshold` | `float` | `0.02` | VAD 阈�?|
| `--verbose` | flag | `False` | 调试模式 |

### LLM 集成

当前机器人使用回显模式（将用户输入原样返回）。要接入真实 LLM，修�?`_generate_response` 方法�?

```python
def _generate_response(self, user_input: str) -> str:
    # 方式 1: 使用 OpenAI API
    import openai
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手�?},
            *[
                {"role": m.role, "content": m.content}
                for m in self.history.get_history()
            ],
            {"role": "user", "content": user_input},
        ],
    )
    return response.choices[0].message.content

    # 方式 2: 使用本地模型
    # return local_model.generate(user_input, context=self.history)
```

### 对话历史数据结构

```python
@dataclass
class ChatMessage:
    role: str         # "user" / "assistant"
    content: str      # 文本内容
    timestamp: str    # ISO 格式时间�?
    duration: float   # 语音时长（仅用户消息�?
    confidence: float # 转写置信度（仅用户消息）
```

---

## 基准测试套件

**文件:** `examples/benchmark_suite.py`

### 功能

- 自动收集硬件信息（GPU/CUDA/CPU/RAM�?
- KV-Cache 注入性能测试�? 种配置，torch.cat vs 零拷贝）
- 音频处理性能测试（float32 转换、重采样、归一化、VAD�?
- Whisper 转写速度测试（多模型 × 多时长）
- 自动生成 Markdown 格式的性能报告

### 使用方法

```bash
# 完整测试（包�?Whisper，耗时较长�?
python examples/benchmark_suite.py

# 跳过 Whisper 测试（无需音频文件�?
python examples/benchmark_suite.py --skip-whisper

# 只测�?KV-Cache
python examples/benchmark_suite.py --skip-whisper --skip-audio

# 自定义迭代次数和输出
python examples/benchmark_suite.py --iterations 20 --output my_report.md

# 调试模式
python examples/benchmark_suite.py --verbose
```

### 命令行参�?

| 参数 | 类型 | 默认�?| 说明 |
|------|------|--------|------|
| `--output` | `str` | `benchmark_report.md` | 报告输出文件 |
| `--iterations` | `int` | `10` | 测试迭代次数 |
| `--skip-whisper` | flag | `False` | 跳过 Whisper 测试 |
| `--skip-audio` | flag | `False` | 跳过音频处理测试 |
| `--skip-kvcache` | flag | `False` | 跳过 KV-Cache 测试 |
| `--verbose` | flag | `False` | 调试模式 |

### 生成报告结构

运行后会生成 `benchmark_report.md`，包含以下章节：

1. **Environment** �?系统�?GPU 环境信息
2. **KV-Cache Performance** �?8 种配置的对比表（torch.cat vs 直接注入�?
3. **Audio Processing** �?AudioProcessor 各操作耗时 + StreamProcessor VAD 性能
4. **Whisper Performance** �?各模型各时长的转写速度和实时倍率
5. **Summary** �?自动生成的性能要点总结

### 测试覆盖的模�?

| 模块 | 测试�?| 指标 |
|------|--------|------|
| `Omni-VRAM._vram_hacker` | `append_to_kv_cache` | 延迟(ms)、加速比(x) |
| `Omni-VRAM.audio_utils` | `_to_float32`, `stereo_to_mono`, `resample`, `normalize`, `to_wav_bytes` | 处理时间(ms) |
| `Omni-VRAM.stream_processor` | `feed` (VAD 静音/语音), 60s 全流�?| 延迟(ms)、实时因�?|
| `Omni-VRAM.whisper_bridge` | `transcribe` (tiny/base × 1s/5s/10s) | 处理时间(ms)、实时倍率 |

---

## 其他测试文件

### `examples/test_whisper_local.py`

Whisper 本地转写的独立测试脚本。用于快速验�?whisper.cpp �?Python whisper 后端是否正常工作�?

```bash
python examples/test_whisper_local.py
```

---

## 自定义示�?

基于现有示例创建自定义应用的基本模式�?

```python
from Omni-VRAM import (
    StreamProcessor, StreamConfig,
    WhisperBridge, WhisperBackend,
    AudioProcessor, setup_logging,
)

# 1. 初始化日�?
setup_logging("INFO")

# 2. 创建 Whisper 实例
whisper = WhisperBridge(
    backend=WhisperBackend.AUTO,
    whisper_model="base",
    language="zh",
)

# 3. 配置流处理器
config = StreamConfig(sample_rate=16000, chunk_duration_ms=100)

# 4. 创建处理器并设置回调
processor = StreamProcessor(config=config, whisper_bridge=whisper)
processor.on_transcription = lambda r: print(r.text)

# 5. 喂入音频数据
# processor.feed(audio_chunk)