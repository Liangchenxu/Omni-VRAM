"""
Omni-VRAM Gradio Web Demo
==========================

语音 AI 平台 Web 演示界面

功能：
- 上传音频 → 语音转写
- 上传音频 → 情绪识别
- 上传音频 → 说话人分离
- 实时麦克风转写
- 下载结果（JSON / TXT / SRT）

启动方式：
    pip install gradio
    python app.py
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omni-vram-demo")

# ── Import vram_core modules ─────────────────────────────────────
try:
    from vram_core import (
        WhisperBridge,
        WhisperBackend,
        EmotionRecognizer,
        SpeakerDiarizer,
        NoiseReducer,
        AudioProcessor,
        __version__,
    )
except ImportError as e:
    logger.error(f"Failed to import vram_core: {e}")
    raise SystemExit(
        "请先安装 vram_core: pip install -r requirements.txt\n"
        f"错误信息: {e}"
    )

# ── Gradio import ────────────────────────────────────────────────
try:
    import gradio as gr
except ImportError:
    raise SystemExit(
        "请先安装 Gradio: pip install gradio\n"
        "安装后重新运行: python app.py"
    )


# ═══════════════════════════════════════════════════════════════════
# 初始化模块（懒加载模式，首次调用时初始化）
# ═══════════════════════════════════════════════════════════════════

_whisper = None
_emotion = None
_diarizer = None
_noise_reducer = None


def get_whisper(model_size: str = "base", language: str = "zh"):
    """懒加载 Whisper 模型"""
    global _whisper
    if _whisper is None:
        logger.info(f"初始化 WhisperBridge (model={model_size}, lang={language})...")
        _whisper = WhisperBridge(
            backend=WhisperBackend.AUTO,
            whisper_model=model_size,
            language=language,
        )
    return _whisper


def get_emotion():
    """懒加载情绪识别器"""
    global _emotion
    if _emotion is None:
        logger.info("初始化 EmotionRecognizer...")
        _emotion = EmotionRecognizer()
    return _emotion


def get_diarizer():
    """懒加载说话人分离器"""
    global _diarizer
    if _diarizer is None:
        logger.info("初始化 SpeakerDiarizer...")
        _diarizer = SpeakerDiarizer()
    return _diarizer


def get_noise_reducer():
    """懒加载噪声抑制器"""
    global _noise_reducer
    if _noise_reducer is None:
        logger.info("初始化 NoiseReducer...")
        _noise_reducer = NoiseReducer()
    return _noise_reducer


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def load_audio_from_file(filepath: str) -> tuple:
    """从文件路径加载音频数据，返回 (numpy_array, sample_rate)"""
    if filepath is None:
        raise ValueError("请先上传或录制音频文件")
    processor = AudioProcessor()
    audio_data = processor.load(filepath)
    sr = audio_data.sample_rate
    audio = audio_data.audio
    # 确保是单声道
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def format_timestamp(seconds: float) -> str:
    """将秒数转换为 SRT 时间戳格式 HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def to_srt(segments: list) -> str:
    """将转写结果转换为 SRT 字幕格式"""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg.get("start", 0))
        end = format_timestamp(seg.get("end", 0))
        text = seg.get("text", "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def to_txt(result) -> str:
    """将转写结果转换为纯文本格式"""
    if hasattr(result, "text"):
        return result.text
    return str(result)


def to_json(data) -> str:
    """将数据转换为格式化 JSON 字符串"""
    if hasattr(data, "__dict__"):
        # dataclass 或对象，转为 dict
        try:
            import dataclasses
            if dataclasses.is_dataclass(data):
                return json.dumps(dataclasses.asdict(data), ensure_ascii=False, indent=2)
        except Exception:
            pass
        # 通用对象
        clean = {}
        for k, v in data.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, np.ndarray):
                continue
            clean[k] = v
        return json.dumps(clean, ensure_ascii=False, indent=2)
    return json.dumps(data, ensure_ascii=False, indent=2)


def save_temp_file(content: str, suffix: str = ".txt") -> str:
    """保存内容到临时文件，返回路径"""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="omni_vram_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ═══════════════════════════════════════════════════════════════════
# 核心处理函数
# ═══════════════════════════════════════════════════════════════════

def transcribe_audio(audio_path, model_size, language, enable_noise_reduction):
    """
    语音转写：上传音频 → 转写文字
    返回: (转写文本, 细节信息, JSON路径, TXT路径, SRT路径)
    """
    if audio_path is None:
        return "❌ 请先上传或录制音频", "", None, None, None

    try:
        t0 = time.time()
        audio, sr = load_audio_from_file(audio_path)

        # 可选：噪声抑制
        if enable_noise_reduction:
            reducer = get_noise_reducer()
            audio = reducer.reduce(audio, sample_rate=sr)

        # 转写
        whisper = get_whisper(model_size=model_size, language=language)
        result = whisper.transcribe(audio_path if not enable_noise_reduction else audio, sample_rate=sr)
        elapsed = time.time() - t0

        # 构造输出
        text = result.text if hasattr(result, "text") else str(result)
        segments = getattr(result, "segments", []) or []

        detail_lines = [
            f"⏱️ 耗时: {elapsed:.2f} 秒",
            f"📊 置信度: {getattr(result, 'confidence', 'N/A')}",
            f"🎤 音频时长: {getattr(result, 'audio_duration', len(audio)/sr):.1f} 秒",
            f"📝 模型: {model_size} | 语言: {language}",
        ]
        if enable_noise_reduction:
            detail_lines.append("🔇 已启用噪声抑制")
        detail = "\n".join(detail_lines)

        # 生成下载文件
        json_content = json.dumps({
            "text": text,
            "language": language,
            "model": model_size,
            "duration_seconds": getattr(result, "audio_duration", len(audio)/sr),
            "processing_time_seconds": round(elapsed, 2),
            "segments": [
                {
                    "start": getattr(s, "start", 0),
                    "end": getattr(s, "end", 0),
                    "text": getattr(s, "text", ""),
                    "confidence": getattr(s, "confidence", 0),
                }
                for s in segments
            ] if segments else [],
        }, ensure_ascii=False, indent=2)

        seg_dicts = [
            {"start": getattr(s, "start", 0), "end": getattr(s, "end", 0), "text": getattr(s, "text", "")}
            for s in segments
        ]

        json_path = save_temp_file(json_content, ".json")
        txt_path = save_temp_file(text, ".txt")
        srt_path = save_temp_file(to_srt(seg_dicts) if seg_dicts else text, ".srt")

        return text, detail, json_path, txt_path, srt_path

    except Exception as e:
        logger.exception("转写失败")
        return f"❌ 转写失败: {e}", "", None, None, None


def recognize_emotion(audio_path):
    """
    情绪识别：上传音频 → 分析情绪
    返回: (主情绪, 详细分析, JSON路径)
    """
    if audio_path is None:
        return "❌ 请先上传或录制音频", "", None

    try:
        t0 = time.time()
        audio, sr = load_audio_from_file(audio_path)
        recognizer = get_emotion()
        result = recognizer.analyze(audio, sample_rate=sr)
        elapsed = time.time() - t0

        emotion = result.emotion if hasattr(result, "emotion") else str(result)
        confidence = getattr(result, "confidence", 0)
        all_scores = getattr(result, "all_scores", {})

        # 主要输出
        emoji_map = {
            "happy": "😊", "sad": "😢", "angry": "😠",
            "neutral": "😐", "surprised": "😮", "surprise": "😮",
            "fear": "😨", "disgust": "🤢",
        }
        emoji = emoji_map.get(emotion.lower(), "🎭")
        main_output = f"{emoji} **{emotion}**（置信度 {confidence:.1%}）"

        # 详细分析
        detail_lines = [
            f"⏱️ 耗时: {elapsed:.2f} 秒",
            f"🎤 音频时长: {len(audio)/sr:.1f} 秒",
            "",
            "**各情绪概率:**",
        ]
        if all_scores:
            for emo, score in sorted(all_scores.items(), key=lambda x: -x[1]):
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                e = emoji_map.get(emo.lower(), "🎭")
                detail_lines.append(f"  {e} {emo}: {bar} {score:.1%}")
        detail = "\n".join(detail_lines)

        # JSON 下载
        json_data = {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in all_scores.items()} if all_scores else {},
            "audio_duration_seconds": round(len(audio)/sr, 2),
            "processing_time_seconds": round(elapsed, 2),
        }
        json_path = save_temp_file(json.dumps(json_data, ensure_ascii=False, indent=2), ".json")

        return main_output, detail, json_path

    except Exception as e:
        logger.exception("情绪识别失败")
        return f"❌ 情绪识别失败: {e}", "", None


def diarize_speakers(audio_path):
    """
    说话人分离：上传音频 → 识别谁在说话
    返回: (主结果, 详细信息, JSON路径, TXT路径, SRT路径)
    """
    if audio_path is None:
        return "❌ 请先上传或录制音频", "", None, None, None

    try:
        t0 = time.time()
        audio, sr = load_audio_from_file(audio_path)
        diarizer = get_diarizer()
        segments = diarizer.diarize(audio, sample_rate=sr)
        elapsed = time.time() - t0

        if not segments:
            return "🔇 未检测到语音活动", "", None, None, None

        # 统计信息
        speakers = set()
        seg_list = []
        for seg in segments:
            speakers.add(seg.speaker_id)
            seg_list.append({
                "start": round(seg.start_time, 2),
                "end": round(seg.end_time, 2),
                "speaker": seg.speaker_id,
                "confidence": round(getattr(seg, "confidence", 0), 3),
            })

        # 主输出（表格化）
        lines = [f"🎤 检测到 **{len(speakers)}** 位说话人，共 **{len(segments)}** 个片段\n"]
        lines.append("| 时间段 | 说话人 | 时长 |")
        lines.append("|--------|--------|------|")
        for seg in seg_list:
            lines.append(
                f"| {seg['start']:.1f}s - {seg['end']:.1f}s "
                f"| {seg['speaker']} "
                f"| {seg['end'] - seg['start']:.1f}s |"
            )
        main_output = "\n".join(lines)

        # 详细信息
        detail_lines = [
            f"⏱️ 耗时: {elapsed:.2f} 秒",
            f"🎤 音频时长: {len(audio)/sr:.1f} 秒",
            f"👥 说话人数量: {len(speakers)}",
            f"📝 片段数量: {len(segments)}",
            "",
            "**说话人分布:**",
        ]
        speaker_durations = {}
        for seg in seg_list:
            spk = seg["speaker"]
            dur = seg["end"] - seg["start"]
            speaker_durations[spk] = speaker_durations.get(spk, 0) + dur
        total_dur = sum(speaker_durations.values())
        for spk, dur in sorted(speaker_durations.items()):
            pct = dur / total_dur * 100 if total_dur > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            detail_lines.append(f"  {spk}: {bar} {dur:.1f}s ({pct:.0f}%)")
        detail = "\n".join(detail_lines)

        # 下载文件
        json_path = save_temp_file(json.dumps({
            "speakers": list(speakers),
            "total_segments": len(segments),
            "segments": seg_list,
        }, ensure_ascii=False, indent=2), ".json")

        txt_lines = [f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['speaker']}" for s in seg_list]
        txt_path = save_temp_file("\n".join(txt_lines), ".txt")

        srt_lines = []
        for i, s in enumerate(seg_list, 1):
            start_ts = format_timestamp(s["start"])
            end_ts = format_timestamp(s["end"])
            srt_lines.append(f"{i}\n{start_ts} --> {end_ts}\n{s['speaker']}\n")
        srt_path = save_temp_file("\n".join(srt_lines), ".srt")

        return main_output, detail, json_path, txt_path, srt_path

    except Exception as e:
        logger.exception("说话人分离失败")
        return f"❌ 说话人分离失败: {e}", "", None, None, None


def mic_transcribe(audio, model_size, language):
    """
    实时麦克风转写
    Gradio 的 Audio(type="numpy") 返回 (sample_rate, numpy_array)
    """
    if audio is None:
        return "❌ 请录制音频", ""

    try:
        # Gradio 返回 (sample_rate, data) 元组
        if isinstance(audio, tuple):
            sr, data = audio
            audio_np = np.array(data, dtype=np.float32)
            # 如果是多声道，转单声道
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
        else:
            audio_np = np.array(audio, dtype=np.float32)
            sr = 16000

        # 保存到临时文件供 Whisper 使用
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            # 写入 WAV
            import wave
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                # 归一化到 int16
                if audio_np.max() <= 1.0:
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_np.astype(np.int16)
                wf.writeframes(audio_int16.tobytes())

        whisper = get_whisper(model_size=model_size, language=language)
        result = whisper.transcribe(tmp_path)
        text = result.text if hasattr(result, "text") else str(result)

        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        duration = len(audio_np) / sr
        detail = f"⏱️ 音频时长: {duration:.1f} 秒 | 📝 模型: {model_size}"
        return text, detail

    except Exception as e:
        logger.exception("麦克风转写失败")
        return f"❌ 转写失败: {e}", ""


# ═══════════════════════════════════════════════════════════════════
# Gradio 界面
# ═══════════════════════════════════════════════════════════════════

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
)

TITLE = """
# 🎙️ Omni-VRAM 语音 AI 平台
**基于 CUDA 零拷贝技术的高性能语音 AI 演示** · v{version}
""".format(version=__version__)

DESCRIPTION = """
> 上传音频文件或录制语音，体验语音转写、情绪识别、说话人分离等 AI 能力。
> 
> 支持格式：WAV、MP3、FLAC、OGG 等常见音频格式。
"""


def build_ui():
    with gr.Blocks(theme=THEME, title="Omni-VRAM Demo", css="""
        .footer { text-align: center; margin-top: 20px; opacity: 0.6; }
    """) as demo:

        gr.Markdown(TITLE)
        gr.Markdown(DESCRIPTION)

        # ── Tab 1: 语音转写 ──────────────────────────────────────
        with gr.Tab("📝 语音转写", id="transcribe"):
            gr.Markdown("### 上传音频文件，自动转写为文字")
            with gr.Row():
                with gr.Column(scale=1):
                    trans_audio = gr.Audio(
                        label="🎤 上传音频",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    with gr.Accordion("⚙️ 转写设置", open=False):
                        trans_model = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                            value="base",
                            label="模型大小",
                            info="tiny 最快，large 最准",
                        )
                        trans_lang = gr.Dropdown(
                            choices=["zh", "en", "ja", "ko", "auto"],
                            value="zh",
                            label="语言",
                            info="auto 为自动检测",
                        )
                        trans_denoise = gr.Checkbox(
                            label="启用噪声抑制",
                            value=False,
                            info="对含噪声的音频效果更好",
                        )
                    trans_btn = gr.Button("🚀 开始转写", variant="primary", size="lg")

                with gr.Column(scale=1):
                    trans_text = gr.Textbox(
                        label="📝 转写结果",
                        lines=8,
                        show_copy_button=True,
                    )
                    trans_detail = gr.Markdown(label="📊 详情")
                    with gr.Row():
                        trans_json_dl = gr.File(label="📥 JSON 下载")
                        trans_txt_dl = gr.File(label="📥 TXT 下载")
                        trans_srt_dl = gr.File(label="📥 SRT 字幕下载")

            trans_btn.click(
                fn=transcribe_audio,
                inputs=[trans_audio, trans_model, trans_lang, trans_denoise],
                outputs=[trans_text, trans_detail, trans_json_dl, trans_txt_dl, trans_srt_dl],
            )

        # ── Tab 2: 情绪识别 ──────────────────────────────────────
        with gr.Tab("🎭 情绪识别", id="emotion"):
            gr.Markdown("### 上传音频，分析说话人的情绪状态")
            gr.Markdown("*支持 7 种情绪：开心、悲伤、愤怒、中性、惊讶、恐惧、厌恶*")
            with gr.Row():
                with gr.Column(scale=1):
                    emo_audio = gr.Audio(
                        label="🎤 上传音频",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    emo_btn = gr.Button("🔍 分析情绪", variant="primary", size="lg")

                with gr.Column(scale=1):
                    emo_main = gr.Markdown(label="🎭 识别结果")
                    emo_detail = gr.Markdown(label="📊 详细分析")
                    emo_json_dl = gr.File(label="📥 JSON 下载")

            emo_btn.click(
                fn=recognize_emotion,
                inputs=[emo_audio],
                outputs=[emo_main, emo_detail, emo_json_dl],
            )

        # ── Tab 3: 说话人分离 ────────────────────────────────────
        with gr.Tab("👥 说话人分离", id="diarize"):
            gr.Markdown("### 上传多人对话音频，识别「谁在什么时候说话」")
            with gr.Row():
                with gr.Column(scale=1):
                    diar_audio = gr.Audio(
                        label="🎤 上传音频",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    diar_btn = gr.Button("🔍 分析说话人", variant="primary", size="lg")

                with gr.Column(scale=1):
                    diar_main = gr.Markdown(label="👥 分离结果")
                    diar_detail = gr.Markdown(label="📊 详细信息")
                    with gr.Row():
                        diar_json_dl = gr.File(label="📥 JSON 下载")
                        diar_txt_dl = gr.File(label="📥 TXT 下载")
                        diar_srt_dl = gr.File(label="📥 SRT 字幕下载")

            diar_btn.click(
                fn=diarize_speakers,
                inputs=[diar_audio],
                outputs=[diar_main, diar_detail, diar_json_dl, diar_txt_dl, diar_srt_dl],
            )

        # ── Tab 4: 实时麦克风转写 ────────────────────────────────
        with gr.Tab("🎙️ 实时麦克风转写", id="mic"):
            gr.Markdown("### 录制语音，实时转写为文字")
            gr.Markdown("> 💡 点击录音按钮开始，录制完成后自动转写")
            with gr.Row():
                with gr.Column(scale=1):
                    mic_audio = gr.Audio(
                        label="🎤 录制语音",
                        type="numpy",
                        sources=["microphone"],
                    )
                    with gr.Accordion("⚙️ 设置", open=False):
                        mic_model = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium"],
                            value="base",
                            label="模型大小",
                        )
                        mic_lang = gr.Dropdown(
                            choices=["zh", "en", "ja", "ko", "auto"],
                            value="zh",
                            label="语言",
                        )
                    mic_btn = gr.Button("🚀 开始转写", variant="primary", size="lg")

                with gr.Column(scale=1):
                    mic_text = gr.Textbox(
                        label="📝 转写结果",
                        lines=8,
                        show_copy_button=True,
                    )
                    mic_detail = gr.Markdown(label="📊 详情")

            mic_btn.click(
                fn=mic_transcribe,
                inputs=[mic_audio, mic_model, mic_lang],
                outputs=[mic_text, mic_detail],
            )

        # ── 底部信息 ─────────────────────────────────────────────
        gr.Markdown("""
        ---
        <div class="footer">
        
        **Omni-VRAM** v{version} · [GitHub](https://github.com/Liangchenxu/Omni-VRAM) · 
        [文档](https://github.com/Liangchenxu/Omni-VRAM/tree/main/docs) · 
        Made with ❤️ by [Liangchenxu](https://github.com/Liangchenxu)
        
        </div>
        """.format(version=__version__))

    return demo


# ═══════════════════════════════════════════════════════════════════
# 启动入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Omni-VRAM Gradio Web Demo")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="端口号 (默认: 7860)")
    parser.add_argument("--share", action="store_true", help="创建公网链接")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    logger.info(f"启动 Omni-VRAM Web Demo (v{__version__})...")
    logger.info(f"地址: http://{args.host}:{args.port}")

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )