"""
vram_core Gradio Web Demo
==========================

璇煶 AI 骞冲彴 Web 婕旂ず鐣岄潰

鍔熻兘锛?
- 涓婁紶闊抽 鈫?璇煶杞啓
- 涓婁紶闊抽 鈫?鎯呯华璇嗗埆
- 涓婁紶闊抽 鈫?璇磋瘽浜哄垎绂?
- 瀹炴椂楹﹀厠椋庤浆鍐?
- 涓嬭浇缁撴灉锛圝SON / TXT / SRT锛?

鍚姩鏂瑰紡锛?
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
logger = logging.getLogger("vram_core-demo")

# 鈹€鈹€ Import vram_core modules 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
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
        "璇峰厛瀹夎 vram_core: pip install -r requirements.txt\n"
        f"閿欒淇℃伅: {e}"
    )

# 鈹€鈹€ Gradio import 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
try:
    import gradio as gr
except ImportError:
    raise SystemExit(
        "璇峰厛瀹夎 Gradio: pip install gradio\n"
        "瀹夎鍚庨噸鏂拌繍琛岋細python app.py"
    )


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# 鍒濆鍖栨ā鍧楋紙鎳掑姞杞芥ā寮忥紝棣栨璋冪敤鏃跺垵濮嬪寲锛?
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

_whisper = None
_emotion = None
_diarizer = None
_noise_reducer = None


def get_whisper(model_size: str = "base", language: str = "zh"):
    """鎳掑姞杞?Whisper 妯″瀷"""
    global _whisper
    if _whisper is None:
        logger.info(f"鍒濆鍖?WhisperBridge (model={model_size}, lang={language})...")
        _whisper = WhisperBridge(
            backend=WhisperBackend.AUTO,
            whisper_model=model_size,
            language=language,
        )
    return _whisper


def get_emotion():
    """鎳掑姞杞芥儏缁瘑鍒櫒"""
    global _emotion
    if _emotion is None:
        logger.info("鍒濆鍖?EmotionRecognizer...")
        _emotion = EmotionRecognizer()
    return _emotion


def get_diarizer():
    """鎳掑姞杞借璇濅汉鍒嗙鍣?""
    global _diarizer
    if _diarizer is None:
        logger.info("鍒濆鍖?SpeakerDiarizer...")
        _diarizer = SpeakerDiarizer()
    return _diarizer


def get_noise_reducer():
    """鎳掑姞杞藉櫔澹版姂鍒跺櫒"""
    global _noise_reducer
    if _noise_reducer is None:
        logger.info("鍒濆鍖?NoiseReducer...")
        _noise_reducer = NoiseReducer()
    return _noise_reducer


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# 宸ュ叿鍑芥暟
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

def load_audio_from_file(filepath: str) -> tuple:
    """浠庢枃浠惰矾寰勫姞杞介煶棰戞暟鎹紝杩斿洖 (numpy_array, sample_rate)"""
    if filepath is None:
        raise ValueError("璇峰厛涓婁紶鎴栧綍鍒堕煶棰戞枃浠?)
    processor = AudioProcessor()
    audio_data = processor.load(filepath)
    sr = audio_data.sample_rate
    audio = audio_data.audio
    # 纭繚鏄崟澹伴亾
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def format_timestamp(seconds: float) -> str:
    """灏嗙鏁拌浆鎹负 SRT 鏃堕棿鎴虫牸寮?HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def to_srt(segments: list) -> str:
    """灏嗚浆鍐欑粨鏋滆浆鎹负 SRT 瀛楀箷鏍煎紡"""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg.get("start", 0))
        end = format_timestamp(seg.get("end", 0))
        text = seg.get("text", "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def to_txt(result) -> str:
    """灏嗚浆鍐欑粨鏋滆浆鎹负绾枃鏈牸寮?""
    if hasattr(result, "text"):
        return result.text
    return str(result)


def to_json(data) -> str:
    """灏嗘暟鎹浆鎹负鏍煎紡鍖栫殑 JSON 瀛楃涓?""
    if hasattr(data, "__dict__"):
        # dataclass 鎴栧璞★紝杞负 dict
        try:
            import dataclasses
            if dataclasses.is_dataclass(data):
                return json.dumps(dataclasses.asdict(data), ensure_ascii=False, indent=2)
        except Exception:
            pass
        # 閫氱敤瀵硅薄
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
    """淇濆瓨鍐呭鍒颁复鏃舵枃浠讹紝杩斿洖璺緞"""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="omni_vram_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# 鏍稿績澶勭悊鍑芥暟
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

def transcribe_audio(audio_path, model_size, language, enable_noise_reduction):
    """
    璇煶杞啓锛氫笂浼犻煶棰?鈫?杞啓鏂囧瓧
    杩斿洖: (杞啓鏂囨湰, 缁嗚妭淇℃伅, JSON璺緞, TXT璺緞, SRT璺緞)
    """
    if audio_path is None:
        return "鈿狅笍 璇峰厛涓婁紶鎴栧綍鍒堕煶棰?, "", None, None, None

    try:
        t0 = time.time()
        audio, sr = load_audio_from_file(audio_path)

        # 鍙€夛細鍣０鎶戝埗
        if enable_noise_reduction:
            reducer = get_noise_reducer()
            audio = reducer.reduce(audio, sample_rate=sr)

        # 杞啓
        whisper = get_whisper(model_size=model_size, language=language)
        result = whisper.transcribe(audio_path if not enable_noise_reduction else audio, sample_rate=sr)
        elapsed = time.time() - t0

        # 鏋勯€犺緭鍑?
        text = result.text if hasattr(result, "text") else str(result)
        segments = getattr(result, "segments", []) or []

        detail_lines = [
            f"鈴憋笍 鑰楁椂: {elapsed:.2f} 绉?,
            f"馃搳 缃俊搴? {getattr(result, 'confidence', 'N/A')}",
            f"馃帳 闊抽鏃堕暱: {getattr(result, 'audio_duration', len(audio)/sr):.1f} 绉?,
            f"馃摑 妯″瀷: {model_size} | 璇█: {language}",
        ]
        if enable_noise_reduction:
            detail_lines.append("馃攪 宸插惎鐢ㄥ櫔澹版姂鍒?)
        detail = "\n".join(detail_lines)

        # 鐢熸垚涓嬭浇鏂囦欢
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
        logger.exception("杞啓澶辫触")
        return f"鉂?杞啓澶辫触: {e}", "", None, None, None


def recognize_emotion(audio_path):
    """
    鎯呯华璇嗗埆锛氫笂浼犻煶棰?鈫?鍒嗘瀽鎯呯华
    杩斿洖: (涓绘儏缁? 璇︾粏鍒嗘瀽, JSON璺緞)
    """
    if audio_path is None:
        return "鈿狅笍 璇峰厛涓婁紶鎴栧綍鍒堕煶棰?, "", None

    try:
        t0 = time.time()
        audio, sr = load_audio_from_file(audio_path)
        recognizer = get_emotion()
        result = recognizer.analyze(audio, sample_rate=sr)
        elapsed = time.time() - t0

        emotion = result.emotion if hasattr(result, "emotion") else str(result)
        confidence = getattr(result, "confidence", 0)
        all_scores = getattr(result, "all_scores", {})

        # 涓昏杈撳嚭
        emoji_map = {
            "happy": "馃槉", "sad": "馃槩", "angry": "馃槧",
            "neutral": "馃槓", "surprised": "馃槷", "surprise": "馃槷",
            "fear": "馃槰", "disgust": "馃あ",
        }
        emoji = emoji_map.get(emotion.lower(), "馃幁")
        main_output = f"{emoji} **{emotion}**锛堢疆淇″害 {confidence:.1%}锛?

        # 璇︾粏鍒嗘瀽
        detail_lines = [
            f"鈴憋笍 鑰楁椂: {elapsed:.2f} 绉?,
            f"馃帳 闊抽鏃堕暱: {len(audio)/sr:.1f} 绉?,
            "",
            "**鍚勬儏缁鐜?*",
        ]
        if all_scores:
            for emo, score in sorted(all_scores.items(), key=lambda x: -x[1]):
                bar = "鈻? * int(score * 20) + "鈻? * (20 - int(score * 20))
                e = emoji_map.get(emo.lower(), "馃幁")
                detail_lines.append(f"  {e} {emo}: {bar} {score:.1%}")
        detail = "\n".join(detail_lines)

        # JSON 涓嬭浇
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
        logger.exception("鎯呯华璇嗗埆澶辫触")
        return f"鉂?鎯呯华璇嗗埆澶辫触: {e}", "", None


def diarize_speakers(audio_path):
    """
    璇磋瘽浜哄垎绂伙細涓婁紶闊抽 鈫?璇嗗埆璋佸湪璇磋瘽
    杩斿洖: (涓荤粨鏋? 璇︾粏淇℃伅, JSON璺緞, TXT璺緞, SRT璺緞)
    """
    if audio_path is None:
        return "鈿狅笍 璇峰厛涓婁紶鎴栧綍鍒堕煶棰?, "", None, None, None

    try:
        t0 = time.time()
        audio, sr = load_audio_from_file(audio_path)
        diarizer = get_diarizer()
        segments = diarizer.diarize(audio, sample_rate=sr)
        elapsed = time.time() - t0

        if not segments:
            return "馃攪 鏈娴嬪埌璇煶娲诲姩", "", None, None, None

        # 缁熻淇℃伅
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

        # 涓昏緭鍑猴紙琛ㄦ牸鍖栵級
        lines = [f"馃帳 妫€娴嬪埌 **{len(speakers)}** 浣嶈璇濅汉锛屽叡 **{len(segments)}** 涓墖娈礬n"]
        lines.append("| 鏃堕棿娈?| 璇磋瘽浜?| 鏃堕暱 |")
        lines.append("|--------|--------|------|")
        for seg in seg_list:
            lines.append(
                f"| {seg['start']:.1f}s - {seg['end']:.1f}s "
                f"| {seg['speaker']} "
                f"| {seg['end'] - seg['start']:.1f}s |"
            )
        main_output = "\n".join(lines)

        # 璇︾粏淇℃伅
        detail_lines = [
            f"鈴憋笍 鑰楁椂: {elapsed:.2f} 绉?,
            f"馃帳 闊抽鏃堕暱: {len(audio)/sr:.1f} 绉?,
            f"馃懃 璇磋瘽浜烘暟: {len(speakers)}",
            f"馃摑 鐗囨鏁伴噺: {len(segments)}",
            "",
            "**璇磋瘽浜哄垎甯?*",
        ]
        speaker_durations = {}
        for seg in seg_list:
            spk = seg["speaker"]
            dur = seg["end"] - seg["start"]
            speaker_durations[spk] = speaker_durations.get(spk, 0) + dur
        total_dur = sum(speaker_durations.values())
        for spk, dur in sorted(speaker_durations.items()):
            pct = dur / total_dur * 100 if total_dur > 0 else 0
            bar = "鈻? * int(pct / 5) + "鈻? * (20 - int(pct / 5))
            detail_lines.append(f"  {spk}: {bar} {dur:.1f}s ({pct:.0f}%)")
        detail = "\n".join(detail_lines)

        # 涓嬭浇鏂囦欢
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
        logger.exception("璇磋瘽浜哄垎绂诲け璐?)
        return f"鉂?璇磋瘽浜哄垎绂诲け璐? {e}", "", None, None, None


def mic_transcribe(audio, model_size, language):
    """
    瀹炴椂楹﹀厠椋庤浆鍐?
    Gradio 鐨?Audio(type="numpy") 杩斿洖 (sample_rate, numpy_array)
    """
    if audio is None:
        return "鈿狅笍 璇峰綍鍒堕煶棰?, ""

    try:
        # Gradio 杩斿洖 (sample_rate, data) 鍏冪粍
        if isinstance(audio, tuple):
            sr, data = audio
            audio_np = np.array(data, dtype=np.float32)
            # 濡傛灉鏄澹伴亾锛岃浆鍗曞０閬?
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
        else:
            audio_np = np.array(audio, dtype=np.float32)
            sr = 16000

        # 淇濆瓨鍒颁复鏃舵枃浠朵緵 Whisper 浣跨敤
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            # 鍐欏叆 WAV
            import wave
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                # 褰掍竴鍖栧埌 int16
                if audio_np.max() <= 1.0:
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_np.astype(np.int16)
                wf.writeframes(audio_int16.tobytes())

        whisper = get_whisper(model_size=model_size, language=language)
        result = whisper.transcribe(tmp_path)
        text = result.text if hasattr(result, "text") else str(result)

        # 娓呯悊涓存椂鏂囦欢
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        duration = len(audio_np) / sr
        detail = f"鈴憋笍 闊抽鏃堕暱: {duration:.1f} 绉?| 馃摑 妯″瀷: {model_size}"
        return text, detail

    except Exception as e:
        logger.exception("楹﹀厠椋庤浆鍐欏け璐?)
        return f"鉂?杞啓澶辫触: {e}", ""


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# Gradio 鐣岄潰
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
)

TITLE = """
# 馃帣锔?vram_core 璇煶 AI 骞冲彴
**鍩轰簬 CUDA 闆舵嫹璐濇妧鏈殑楂樻€ц兘璇煶 AI 婕旂ず** 路 v{version}
""".format(version=__version__)

DESCRIPTION = """
> 涓婁紶闊抽鏂囦欢鎴栧綍鍒惰闊筹紝浣撻獙璇煶杞啓銆佹儏缁瘑鍒€佽璇濅汉鍒嗙绛?AI 鑳藉姏銆?
> 
> 鏀寔鏍煎紡锛歐AV銆丮P3銆丗LAC銆丱GG 绛夊父瑙侀煶棰戞牸寮忋€?
"""


def build_ui():
    with gr.Blocks(theme=THEME, title="vram_core Demo", css="""
        .footer { text-align: center; margin-top: 20px; opacity: 0.6; }
    """) as demo:

        gr.Markdown(TITLE)
        gr.Markdown(DESCRIPTION)

        # 鈹€鈹€ Tab 1: 璇煶杞啓 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        with gr.Tab("馃摑 璇煶杞啓", id="transcribe"):
            gr.Markdown("### 涓婁紶闊抽鏂囦欢锛岃嚜鍔ㄨ浆鍐欎负鏂囧瓧")
            with gr.Row():
                with gr.Column(scale=1):
                    trans_audio = gr.Audio(
                        label="馃帳 涓婁紶闊抽",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    with gr.Accordion("鈿欙笍 杞啓璁剧疆", open=False):
                        trans_model = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                            value="base",
                            label="妯″瀷澶у皬",
                            info="tiny 鏈€蹇紝large 鏈€鍑?,
                        )
                        trans_lang = gr.Dropdown(
                            choices=["zh", "en", "ja", "ko", "auto"],
                            value="zh",
                            label="璇█",
                            info="auto 涓鸿嚜鍔ㄦ娴?,
                        )
                        trans_denoise = gr.Checkbox(
                            label="鍚敤鍣０鎶戝埗",
                            value=False,
                            info="瀵瑰惈鍣０鐨勯煶棰戞晥鏋滄洿濂?,
                        )
                    trans_btn = gr.Button("馃殌 寮€濮嬭浆鍐?, variant="primary", size="lg")

                with gr.Column(scale=1):
                    trans_text = gr.Textbox(
                        label="馃摑 杞啓缁撴灉",
                        lines=8,
                        show_copy_button=True,
                    )
                    trans_detail = gr.Markdown(label="馃搳 璇︽儏")
                    with gr.Row():
                        trans_json_dl = gr.File(label="馃摜 JSON 涓嬭浇")
                        trans_txt_dl = gr.File(label="馃摜 TXT 涓嬭浇")
                        trans_srt_dl = gr.File(label="馃摜 SRT 瀛楀箷涓嬭浇")

            trans_btn.click(
                fn=transcribe_audio,
                inputs=[trans_audio, trans_model, trans_lang, trans_denoise],
                outputs=[trans_text, trans_detail, trans_json_dl, trans_txt_dl, trans_srt_dl],
            )

        # 鈹€鈹€ Tab 2: 鎯呯华璇嗗埆 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        with gr.Tab("馃幁 鎯呯华璇嗗埆", id="emotion"):
            gr.Markdown("### 涓婁紶闊抽锛屽垎鏋愯璇濅汉鐨勬儏缁姸鎬?)
            gr.Markdown("*鏀寔 7 绉嶆儏缁細寮€蹇冦€佹偛浼ゃ€佹劋鎬掋€佷腑鎬с€佹儕璁躲€佹亹鎯с€佸帉鎭?")
            with gr.Row():
                with gr.Column(scale=1):
                    emo_audio = gr.Audio(
                        label="馃帳 涓婁紶闊抽",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    emo_btn = gr.Button("馃攳 鍒嗘瀽鎯呯华", variant="primary", size="lg")

                with gr.Column(scale=1):
                    emo_main = gr.Markdown(label="馃幁 璇嗗埆缁撴灉")
                    emo_detail = gr.Markdown(label="馃搳 璇︾粏鍒嗘瀽")
                    emo_json_dl = gr.File(label="馃摜 JSON 涓嬭浇")

            emo_btn.click(
                fn=recognize_emotion,
                inputs=[emo_audio],
                outputs=[emo_main, emo_detail, emo_json_dl],
            )

        # 鈹€鈹€ Tab 3: 璇磋瘽浜哄垎绂?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        with gr.Tab("馃懃 璇磋瘽浜哄垎绂?, id="diarize"):
            gr.Markdown("### 涓婁紶澶氫汉瀵硅瘽闊抽锛岃瘑鍒€岃皝鍦ㄤ粈涔堟椂鍊欒璇濄€?)
            with gr.Row():
                with gr.Column(scale=1):
                    diar_audio = gr.Audio(
                        label="馃帳 涓婁紶闊抽",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    diar_btn = gr.Button("馃攳 鍒嗘瀽璇磋瘽浜?, variant="primary", size="lg")

                with gr.Column(scale=1):
                    diar_main = gr.Markdown(label="馃懃 鍒嗙缁撴灉")
                    diar_detail = gr.Markdown(label="馃搳 璇︾粏淇℃伅")
                    with gr.Row():
                        diar_json_dl = gr.File(label="馃摜 JSON 涓嬭浇")
                        diar_txt_dl = gr.File(label="馃摜 TXT 涓嬭浇")
                        diar_srt_dl = gr.File(label="馃摜 SRT 瀛楀箷涓嬭浇")

            diar_btn.click(
                fn=diarize_speakers,
                inputs=[diar_audio],
                outputs=[diar_main, diar_detail, diar_json_dl, diar_txt_dl, diar_srt_dl],
            )

        # 鈹€鈹€ Tab 4: 瀹炴椂楹﹀厠椋庤浆鍐?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        with gr.Tab("馃帣锔?瀹炴椂楹﹀厠椋庤浆鍐?, id="mic"):
            gr.Markdown("### 褰曞埗璇煶锛屽疄鏃惰浆鍐欎负鏂囧瓧")
            gr.Markdown("> 馃挕 鐐瑰嚮褰曢煶鎸夐挳寮€濮嬶紝褰曞埗瀹屾垚鍚庤嚜鍔ㄨ浆鍐?)
            with gr.Row():
                with gr.Column(scale=1):
                    mic_audio = gr.Audio(
                        label="馃帳 褰曞埗璇煶",
                        type="numpy",
                        sources=["microphone"],
                    )
                    with gr.Accordion("鈿欙笍 璁剧疆", open=False):
                        mic_model = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium"],
                            value="base",
                            label="妯″瀷澶у皬",
                        )
                        mic_lang = gr.Dropdown(
                            choices=["zh", "en", "ja", "ko", "auto"],
                            value="zh",
                            label="璇█",
                        )
                    mic_btn = gr.Button("馃殌 寮€濮嬭浆鍐?, variant="primary", size="lg")

                with gr.Column(scale=1):
                    mic_text = gr.Textbox(
                        label="馃摑 杞啓缁撴灉",
                        lines=8,
                        show_copy_button=True,
                    )
                    mic_detail = gr.Markdown(label="馃搳 璇︽儏")

            mic_btn.click(
                fn=mic_transcribe,
                inputs=[mic_audio, mic_model, mic_lang],
                outputs=[mic_text, mic_detail],
            )

        # 鈹€鈹€ 搴曢儴淇℃伅 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        gr.Markdown("""
        ---
        <div class="footer">
        
        **vram_core** v{version} 路 [GitHub](https://github.com/Liangchenxu/vram_core) 路 
        [鏂囨。](https://github.com/Liangchenxu/vram_core/tree/main/docs) 路 
        Made with 鉂わ笍 by [Liangchenxu](https://github.com/Liangchenxu)
        
        </div>
        """.format(version=__version__))

    return demo


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# 鍚姩鍏ュ彛
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vram_core Gradio Web Demo")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="鐩戝惉鍦板潃 (榛樿: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="绔彛鍙?(榛樿: 7860)")
    parser.add_argument("--share", action="store_true", help="鍒涘缓鍏綉閾炬帴")
    parser.add_argument("--debug", action="store_true", help="璋冭瘯妯″紡")
    args = parser.parse_args()

    logger.info(f"鍚姩 vram_core Web Demo (v{__version__})...")
    logger.info(f"鍦板潃: http://{args.host}:{args.port}")

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )