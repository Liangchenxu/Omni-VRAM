# Omni-VRAM 常见问题 (FAQ)

## 安装与编�?

### Q: 安装时报�?`error: command 'nvcc' failed`

**A:** CUDA Toolkit 未正确安装或不在系统 PATH 中�?

```bash
# 检�?nvcc 是否可用
nvcc --version

# 如果未找到，手动添加�?PATH
# Windows:
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
$env:PATH += ";$env:CUDA_HOME\bin"

# Linux:
export PATH=/usr/local/cuda/bin:$PATH
```

### Q: 编译时报�?`Microsoft Visual C++ 14.0 or greater is required`

**A:** Windows 上编�?CUDA 扩展需�?Visual Studio Build Tools�?

1. 下载安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. 安装时勾�?"使用 C++ 的桌面开�? 工作负载
3. 重新打开终端后再编译

### Q: `pip install pyaudio` 失败

**A:** PyAudio 需�?PortAudio 依赖�?

- **Windows:** `pip install pipwin && pipwin install pyaudio`
- **Linux:** `sudo apt install portaudio19-dev && pip install pyaudio`
- **macOS:** `brew install portaudio && pip install pyaudio`

### Q: 没有 NVIDIA GPU 能用吗？

**A:** 可以。CUDA 扩展（KV-Cache 注入、融合音频前端）不可用，但以下功能完全正常：
- Whisper 语音转写（所有后端）
- 音频格式处理（AudioProcessor�?
- 实时流处理（StreamProcessor）�?VAD 使用 CPU

`Omni-VRAM.CUDA_AVAILABLE` 会显�?`False`，程序会自动回退�?CPU 实现�?

### Q: PyTorch �?CUDA 版本与系�?CUDA Toolkit 不匹�?

**A:** 两者版本必须匹配�?

```bash
# 查看 PyTorch �?CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 查看系统 CUDA 版本
nvcc --version
```

如需安装特定版本�?PyTorch�?
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Whisper 转写

### Q: `TranscriptionError: Transcription failed (exit code: -1073740791)`

**A:** whisper.cpp CLI 调用失败。常见原因：

1. **whisper.cpp 路径配置错误** �?检�?`WHISPER_CPP_PATH` 指向的是可执行文件所在目�?
2. **模型文件路径错误** �?检�?`WHISPER_MODEL_PATH` 指向�?`.bin` 文件存在
3. **音频格式问题** �?whisper.cpp 要求 16kHz 单声�?WAV，系统会自动转换但偶有兼容性问�?
4. **DLL 缺失 (Windows)** �?whisper.cpp 可能依赖 Visual C++ Redistributable

建议运行 `python examples/test_whisper_local.py` 进行诊断�?

### Q: Whisper 转写很慢怎么办？

**A:** 按优先级尝试以下方案�?

1. **换用更小的模型：** `tiny` > `base` > `small` > `medium` > `large`
2. **换用 whisper.cpp 后端�?* �?Python whisper 库快 5-10 �?
3. **设置 `WHISPER_THREADS`�?* 设置�?CPU 核心�?
4. **使用 OpenAI API�?* 将计算卸载到云端，需�?API Key

### Q: 如何切换 Whisper 后端�?

**A:** �?`.env` 中设�?`WHISPER_BACKEND`�?

| �?| 后端 | 说明 |
|------|------|------|
| `AUTO` | 自动 | �?API→CLI→Python→None 顺序尝试 |
| `WHISPER_API` | OpenAI API | 需�?`OPENAI_API_KEY` |
| `WHISPER_CPP` | whisper.cpp CLI | 需�?`WHISPER_CPP_PATH` �?`WHISPER_MODEL_PATH` |
| `WHISPER_PYTHON` | Python whisper | 需�?`pip install openai-whisper` |
| `NONE` | 禁用 | 不使用转写功�?|

### Q: 自动后端选择的逻辑是什么？

**A:** `WhisperBackend.AUTO` 按以下顺序逐个尝试�?

1. **OpenAI API** �?如果 `OPENAI_API_KEY` 已设置且格式有效
2. **whisper.cpp** �?如果 `WHISPER_CPP_PATH` �?`WHISPER_MODEL_PATH` 已设置且文件存在
3. **Python whisper** �?如果 `import whisper` 成功（已安装 `openai-whisper`�?
4. **None** �?如果以上均不可用，转写功能被禁用

---

## 流处理器

### Q: VAD 不灵�?/ 太灵�?

**A:** 调整 `StreamConfig.vad_threshold`�?

- **太灵敏（误触发多）：** 增大阈值，�?`0.05` �?`0.1`
- **不灵敏（漏检多）�?* 减小阈值，�?`0.02` �?`0.005`

环境噪音大的场景建议值：`0.05 - 0.1`
安静环境建议值：`0.01 - 0.02`

### Q: 语音片段被截�?

**A:** 检查以下参数：

- `min_speech_duration_ms`：语音短于此值会被忽略，默认 300ms
- `silence_duration_ms`：静音超过此值触发转写，默认 500ms
- `max_segment_duration_s`：强制截断时长，默认 30s

### Q: `feed` 方法报错 `ValueError`

**A:** 输入数据必须满足�?
- dtype: `float32`
- shape: `(chunk_size,)` �?�?`config.chunk_size` 一�?
- 值域: `[-1.0, 1.0]`

�?PyAudio 读取�?int16 数据需先转换：
```python
audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
```

---

## CUDA 扩展

### Q: `append_to_kv_cache` �?O(1) 是如何实现的�?

**A:** 传统 `torch.cat` 需要分配新内存、复制旧数据、追加新数据。零拷贝方案�?

1. 启动时预分配一块足够大的连续显存（�?100,000 × 4096�?
2. 使用原子计数�?`current_pos` 追踪写入位置
3. 每次追加直接将新 token 写入 `kv_cache[current_pos:current_pos+n]` 的位�?
4. `current_pos` 原子递增，无需重新分配或拷�?

### Q: `smart_audio_listen` 返回�?features 是什么？

**A:** 融合 kernel 输出的音频特征向量，包含预加重（pre-emphasis）和汉宁窗（Hann window）处理后的频域特征。主要用于下游语音模型的输入�?

---

## 音频处理

### Q: 如何列出可用的麦克风设备�?

**A:**
```python
import pyaudio
pa = pyaudio.PyAudio()
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"[{i}] {info['name']}")
pa.terminate()
```

### Q: 录音有回声或噪声

**A:** 这是硬件/环境问题，Omni-VRAM 不含回声消除（AEC）或噪声抑制（NS）。建议：
- 使用耳机避免扬声器回�?
- 在安静环境中录音
- 后续版本可能集成 RNNoise �?WebRTC APM

---

## 性能与调�?

### Q: 如何查看详细日志�?

**A:** �?`.env` 中设�?`VRAM_LOG_LEVEL=DEBUG`，或在代码中�?

```python
from Omni-VRAM import setup_logging
setup_logging("DEBUG")
```

### Q: 如何运行基准测试�?

**A:**
```bash
# 跳过 Whisper 的快速测�?
python examples/benchmark_suite.py --skip-whisper

# 完整测试（需要音频文件和 Whisper 模型�?
python examples/benchmark_suite.py
```

### Q: 内存泄漏或显存不释放

**A:** 
1. 确保 `StreamProcessor` 在使用完毕后调用 `processor.reset()` 或释放引�?
2. CUDA 显存�?PyTorch 管理，使�?`torch.cuda.empty_cache()` 可手动释放缓�?
3. 确保没有意外保留�?GPU tensor 的引�?

---

## 项目与社�?

### Q: 如何报告 Bug�?

**A:** 请在 [GitHub Issues](https://github.com/Liangchenxu/Omni-VRAM/issues) 提交，包含：
- 操作系统和版�?
- Python、CUDA、PyTorch 版本
- 完整错误日志
- 复现步骤

### Q: 如何贡献代码�?

**A:**
1. Fork 项目
2. 创建功能分支：`git checkout -b feature/my-feature`
3. 提交更改并编写测�?
4. 发起 Pull Request

### Q: 许可证是什么？

**A:** MIT License。可自由用于商业和非商业项目