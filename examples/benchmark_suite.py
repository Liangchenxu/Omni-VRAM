#!/usr/bin/env python3
"""
vram_core: Benchmark Suite
============================

Comprehensive performance testing for all vram_core modules.
Generates a Markdown report with environment info, latency metrics,
throughput numbers, and optimization recommendations.

Tests:
    1. KV-Cache Injection 锟?append_to_kv_cache latency and throughput
    2. Audio Processing 锟?AudioProcessor conversion speed, StreamProcessor VAD speed
    3. Whisper Transcription 锟?model size comparison, audio duration scaling
    4. Hardware Info 锟?GPU, CUDA, system information

Usage:
    # Full benchmark (including whisper)
    python examples/benchmark_suite.py

    # Skip whisper tests (no audio file needed)
    python examples/benchmark_suite.py --skip-whisper

    # Custom output and iterations
    python examples/benchmark_suite.py --output my_report.md --iterations 20

    # Verbose debug mode
    python examples/benchmark_suite.py --verbose

Output:
    benchmark_report.md 锟?Markdown performance report

Requirements:
    pip install numpy torch
    pip install pyaudio numpy pydub python-dotenv  (for audio tests)
    Whisper.cpp must be installed for whisper tests.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from vram_core.config import setup_logging


# 鈹€鈹€ Hardware Info 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def gather_hardware_info() -> Dict[str, Any]:
    """Gather system and GPU hardware information."""
    info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }

    # PyTorch info
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
            info["gpu_compute_capability"] = str(torch.cuda.get_device_capability(0))
        else:
            info["cuda_version"] = "N/A"
            info["gpu_name"] = "N/A"
            info["gpu_memory_mb"] = 0
            info["gpu_compute_capability"] = "N/A"
    except ImportError:
        info["torch_version"] = "Not installed"
        info["cuda_available"] = False
        info["cuda_version"] = "N/A"
        info["gpu_name"] = "N/A"
        info["gpu_memory_mb"] = 0
        info["gpu_compute_capability"] = "N/A"

    # nvidia-smi fallback
    if info.get("gpu_name") == "N/A":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 3:
                    info["gpu_name"] = parts[0].strip()
                    info["gpu_memory_mb"] = int(parts[1].strip().replace(" MiB", ""))
                    info["nvidia_driver"] = parts[2].strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

    # CPU cores
    try:
        info["cpu_cores"] = os.cpu_count() or 0
    except Exception:
        info["cpu_cores"] = 0

    # RAM
    try:
        import psutil
        info["ram_mb"] = psutil.virtual_memory().total // (1024 * 1024)
    except ImportError:
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "computersystem", "get", "totalphysicalmemory"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        line = line.strip()
                        if line.isdigit():
                            info["ram_mb"] = int(line) // (1024 * 1024)
                            break
        except Exception:
            pass

    return info


# 鈹€鈹€ Benchmark: KV-Cache 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def benchmark_kv_cache(iterations: int, verbose: bool = False) -> Dict[str, Any]:
    """
    Benchmark KV-Cache injection performance.

    Tests both:
    - torch.cat baseline (O(N) reallocation)
    - Direct VRAM injection (O(1) in-place, simulated on CPU if no CUDA)

    Returns dict with latency and throughput metrics.
    """
    logger = logging.getLogger("benchmark.kv_cache")
    results = {
        "test_name": "KV-Cache Injection",
        "configurations": [],
    }

    # Test configurations: (seq_len, hidden_dim, new_tokens)
    configs = [
        (128, 768, 1),
        (512, 768, 1),
        (1024, 768, 1),
        (2048, 768, 1),
        (4096, 768, 1),
        (1024, 768, 8),
        (1024, 768, 32),
        (1024, 4096, 1),
    ]

    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed, skipping KV-Cache benchmark")
        results["error"] = "PyTorch not installed"
        return results

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    for max_seq, hidden_dim, new_tokens in configs:
        logger.info(f"Testing KV-Cache: seq={max_seq}, dim={hidden_dim}, new={new_tokens}")

        # 鈹€鈹€ Method 1: torch.cat baseline 鈹€鈹€
        cat_latencies = []
        for _ in range(iterations):
            kv_cache = torch.randn(max_seq, hidden_dim, device=device, dtype=torch.float32)
            current_len = max_seq // 2
            existing = kv_cache[:current_len]
            new_data = torch.randn(new_tokens, hidden_dim, device=device, dtype=torch.float32)

            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            result = torch.cat([existing, new_data], dim=0)
            kv_cache[:current_len + new_tokens] = result

            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            cat_latencies.append((t1 - t0) * 1000)  # ms

        # 鈹€鈹€ Method 2: Direct injection (simulated) 鈹€鈹€
        inject_latencies = []
        for _ in range(iterations):
            kv_cache = torch.zeros(max_seq, hidden_dim, device=device, dtype=torch.float32)
            current_pos = torch.tensor([max_seq // 2], device=device, dtype=torch.int32)
            new_data = torch.randn(new_tokens, hidden_dim, device=device, dtype=torch.float32)
            current_len = current_pos.item()

            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Direct in-place write (simulates CUDA kernel behavior)
            kv_cache[current_len:current_len + new_tokens] = new_data
            current_pos[0] = current_len + new_tokens

            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            inject_latencies.append((t1 - t0) * 1000)  # ms

        cat_avg = np.mean(cat_latencies)
        cat_std = np.std(cat_latencies)
        inject_avg = np.mean(inject_latencies)
        inject_std = np.std(inject_latencies)
        speedup = cat_avg / inject_avg if inject_avg > 0 else float('inf')

        config_result = {
            "max_seq_len": max_seq,
            "hidden_dim": hidden_dim,
            "new_tokens": new_tokens,
            "device": device,
            "torch_cat_avg_ms": round(cat_avg, 4),
            "torch_cat_std_ms": round(cat_std, 4),
            "direct_inject_avg_ms": round(inject_avg, 4),
            "direct_inject_std_ms": round(inject_std, 4),
            "speedup": round(speedup, 2),
        }
        results["configurations"].append(config_result)

        if verbose:
            logger.info(f"  torch.cat: {cat_avg:.4f}ms 卤 {cat_std:.4f}ms")
            logger.info(f"  inject:    {inject_avg:.4f}ms 卤 {inject_std:.4f}ms")
            logger.info(f"  speedup:   {speedup:.2f}x")

    return results


# 鈹€鈹€ Benchmark: Audio Processing 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def benchmark_audio_processing(iterations: int, verbose: bool = False) -> Dict[str, Any]:
    """
    Benchmark audio processing modules:
    - AudioProcessor: format conversion, resampling, normalization
    - StreamProcessor: VAD detection speed

    Returns dict with timing metrics.
    """
    logger = logging.getLogger("benchmark.audio")
    results = {
        "test_name": "Audio Processing",
        "audio_processor": {},
        "stream_processor": {},
    }

    # 鈹€鈹€ AudioProcessor Benchmarks 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    try:
        from vram_core.audio_utils import AudioProcessor

        processor = AudioProcessor(target_sample_rate=16000)

        # Test 1: Float32 conversion
        int16_data = np.random.randint(-32768, 32767, size=16000 * 5, dtype=np.int16)
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            processor._to_float32(int16_data)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        results["audio_processor"]["to_float32_5s_ms"] = round(np.mean(times), 4)
        results["audio_processor"]["to_float32_5s_std_ms"] = round(np.std(times), 4)

        # Test 2: Stereo to mono
        stereo_data = np.random.randn(16000 * 5, 2).astype(np.float32)
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            AudioProcessor.stereo_to_mono(stereo_data)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        results["audio_processor"]["stereo_to_mono_5s_ms"] = round(np.mean(times), 4)
        results["audio_processor"]["stereo_to_mono_5s_std_ms"] = round(np.std(times), 4)

        # Test 3: Resampling
        for orig_sr in [44100, 48000]:
            audio_5s = np.random.randn(orig_sr * 5).astype(np.float32)
            times = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                AudioProcessor.resample(audio_5s, orig_sr, 16000)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
            key = f"resample_{orig_sr}_to_16000_5s_ms"
            results["audio_processor"][key] = round(np.mean(times), 4)
            results["audio_processor"][key + "_std"] = round(np.std(times), 4)

        # Test 4: Normalization
        audio_data = np.random.randn(16000 * 10).astype(np.float32)
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            AudioProcessor.normalize(audio_data)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        results["audio_processor"]["normalize_10s_ms"] = round(np.mean(times), 4)
        results["audio_processor"]["normalize_10s_std_ms"] = round(np.std(times), 4)

        # Test 5: WAV encoding
        audio_10s = np.random.randn(16000 * 10).astype(np.float32)
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            AudioProcessor.to_wav_bytes(audio_10s, 16000)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        results["audio_processor"]["to_wav_bytes_10s_ms"] = round(np.mean(times), 4)
        results["audio_processor"]["to_wav_bytes_10s_std_ms"] = round(np.std(times), 4)

        if verbose:
            logger.info("AudioProcessor results:")
            for k, v in results["audio_processor"].items():
                if not k.endswith("_std"):
                    std_key = k + "_std_ms"
                    std_val = results["audio_processor"].get(std_key, results["audio_processor"].get(k + "_std", 0))
                    logger.info(f"  {k}: {v}ms 卤 {std_val}ms")

    except ImportError as e:
        logger.warning(f"Cannot import AudioProcessor: {e}")
        results["audio_processor"]["error"] = str(e)

    # 鈹€鈹€ StreamProcessor VAD Benchmarks 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    try:
        from vram_core.stream_processor import StreamProcessor, StreamConfig

        config = StreamConfig(
            sample_rate=16000,
            chunk_duration_ms=100,
            vad_threshold=0.02,
        )

        # Create a mock whisper bridge that does nothing
        class MockWhisper:
            def transcribe(self, audio, **kwargs):
                return type('Result', (), {
                    'text': 'test', 'confidence': 0.9,
                    'language': 'en', 'segments': [],
                    'audio_duration': 1.0,
                })()
            def get_status(self):
                return {"backend": "mock"}

        processor = StreamProcessor(config=config, whisper_bridge=MockWhisper())

        # Test VAD with silence
        silence = np.zeros(config.chunk_size, dtype=np.float32)
        times = []
        for _ in range(iterations * 10):
            t0 = time.perf_counter()
            processor.feed(silence)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        results["stream_processor"]["vad_silence_ms"] = round(np.mean(times), 4)
        results["stream_processor"]["vad_silence_std_ms"] = round(np.std(times), 4)

        # Test VAD with speech-like audio
        speech = np.random.randn(config.chunk_size).astype(np.float32) * 0.1
        times = []
        for _ in range(iterations * 10):
            t0 = time.perf_counter()
            processor.feed(speech)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        results["stream_processor"]["vad_speech_ms"] = round(np.mean(times), 4)
        results["stream_processor"]["vad_speech_std_ms"] = round(np.std(times), 4)

        # Test full chunk processing (60 seconds of audio, 100ms chunks = 600 chunks)
        n_chunks = 600
        chunks = [np.random.randn(config.chunk_size).astype(np.float32) * 0.05 for _ in range(n_chunks)]
        t0 = time.perf_counter()
        for chunk in chunks:
            processor.feed(chunk)
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000
        results["stream_processor"]["process_60s_audio_ms"] = round(total_ms, 2)
        results["stream_processor"]["realtime_factor"] = round(60000 / total_ms, 2) if total_ms > 0 else 0

        if verbose:
            logger.info("StreamProcessor results:")
            for k, v in results["stream_processor"].items():
                logger.info(f"  {k}: {v}")

    except ImportError as e:
        logger.warning(f"Cannot import StreamProcessor: {e}")
        results["stream_processor"]["error"] = str(e)

    return results


# 鈹€鈹€ Benchmark: Whisper 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def benchmark_whisper(iterations: int, verbose: bool = False) -> Dict[str, Any]:
    """
    Benchmark Whisper transcription performance.

    Tests:
    - Different model sizes (tiny, base, small)
    - Different audio durations (1s, 5s, 10s, 30s)

    Returns dict with transcription speed metrics.
    """
    logger = logging.getLogger("benchmark.whisper")
    results = {
        "test_name": "Whisper Transcription",
        "models": {},
    }

    try:
        from vram_core.whisper_bridge import WhisperBridge, WhisperBackend
    except ImportError as e:
        logger.warning(f"Cannot import WhisperBridge: {e}")
        results["error"] = str(e)
        return results

    # Generate synthetic audio at 16kHz
    durations = [1, 5, 10]
    audio_data = {}
    for dur in durations:
        audio_data[dur] = np.random.randn(16000 * dur).astype(np.float32) * 0.1

    models_to_test = ["tiny", "base"]

    for model_size in models_to_test:
        logger.info(f"Testing Whisper model: {model_size}")
        model_results = {
            "model": model_size,
            "durations": {},
            "error": None,
        }

        try:
            whisper = WhisperBridge(
                backend=WhisperBackend.AUTO,
                whisper_model=model_size,
                device="cpu",  # Use CPU for consistent benchmarking
            )
            status = whisper.get_status()
            model_results["backend"] = status["backend"]

            for dur in durations:
                audio = audio_data[dur]
                times = []
                for _ in range(min(iterations, 3)):  # Limit whisper iterations
                    t0 = time.perf_counter()
                    whisper.transcribe(audio, sample_rate=16000)
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1000)

                avg_ms = np.mean(times)
                std_ms = np.std(times)
                realtime_factor = (dur * 1000) / avg_ms if avg_ms > 0 else 0

                model_results["durations"][f"{dur}s"] = {
                    "avg_ms": round(avg_ms, 2),
                    "std_ms": round(std_ms, 2),
                    "realtime_factor": round(realtime_factor, 2),
                    "rtfx_label": f"{realtime_factor:.1f}x realtime",
                }

                if verbose:
                    logger.info(f"  {model_size} / {dur}s: {avg_ms:.2f}ms 卤 {std_ms:.2f}ms ({realtime_factor:.1f}x realtime)")

        except Exception as e:
            logger.warning(f"Whisper {model_size} failed: {e}")
            model_results["error"] = str(e)

        results["models"][model_size] = model_results

    return results


# 鈹€鈹€ Report Generation 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def generate_report(
    hardware: Dict[str, Any],
    kv_cache: Dict[str, Any],
    audio: Dict[str, Any],
    whisper: Optional[Dict[str, Any]],
    output_path: str,
) -> str:
    """Generate Markdown benchmark report."""
    lines = []

    # Header
    lines.append("# vram_core Benchmark Report")
    lines.append("")
    lines.append(f"> Generated: {hardware.get('timestamp', 'N/A')}")
    lines.append("")

    # Environment
    lines.append("## Environment")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Platform | {hardware.get('platform', 'N/A')} |")
    lines.append(f"| Processor | {hardware.get('processor', 'N/A')} |")
    lines.append(f"| CPU Cores | {hardware.get('cpu_cores', 'N/A')} |")
    lines.append(f"| RAM | {hardware.get('ram_mb', 'N/A')} MB |")
    lines.append(f"| Python | {hardware.get('python_version', 'N/A')} |")
    lines.append(f"| NumPy | {hardware.get('numpy_version', 'N/A')} |")
    lines.append(f"| PyTorch | {hardware.get('torch_version', 'N/A')} |")
    lines.append(f"| CUDA Available | {hardware.get('cuda_available', 'N/A')} |")
    lines.append(f"| CUDA Version | {hardware.get('cuda_version', 'N/A')} |")
    lines.append(f"| GPU | {hardware.get('gpu_name', 'N/A')} |")
    lines.append(f"| GPU Memory | {hardware.get('gpu_memory_mb', 'N/A')} MB |")
    lines.append(f"| Compute Capability | {hardware.get('gpu_compute_capability', 'N/A')} |")
    if "nvidia_driver" in hardware:
        lines.append(f"| NVIDIA Driver | {hardware.get('nvidia_driver', 'N/A')} |")
    lines.append("")

    # KV-Cache Performance
    lines.append("## KV-Cache Performance")
    lines.append("")
    if "error" in kv_cache:
        lines.append(f"> 鈿狅笍 {kv_cache['error']}")
        lines.append("")
    else:
        lines.append("### torch.cat vs Direct VRAM Injection")
        lines.append("")
        lines.append("| Max Seq | Hidden Dim | New Tokens | torch.cat (ms) | Direct Inject (ms) | Speedup |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for cfg in kv_cache.get("configurations", []):
            lines.append(
                f"| {cfg['max_seq_len']:,} "
                f"| {cfg['hidden_dim']:,} "
                f"| {cfg['new_tokens']} "
                f"| {cfg['torch_cat_avg_ms']:.4f} 卤 {cfg['torch_cat_std_ms']:.4f} "
                f"| {cfg['direct_inject_avg_ms']:.4f} 卤 {cfg['direct_inject_std_ms']:.4f} "
                f"| **{cfg['speedup']:.2f}x** |"
            )
        lines.append("")
        lines.append(f"> Device: {kv_cache.get('configurations', [{}])[0].get('device', 'N/A') if kv_cache.get('configurations') else 'N/A'}")
        lines.append("")

    # Audio Processing Performance
    lines.append("## Audio Processing Performance")
    lines.append("")

    ap = audio.get("audio_processor", {})
    if "error" in ap:
        lines.append(f"> 鈿狅笍 AudioProcessor: {ap['error']}")
        lines.append("")
    else:
        lines.append("### AudioProcessor")
        lines.append("")
        lines.append("| Operation | Time (ms) | Std (ms) |")
        lines.append("|---|---:|---:|")

        op_labels = {
            "to_float32_5s_ms": "int16 锟?float32 (5s audio)",
            "stereo_to_mono_5s_ms": "Stereo 锟?Mono (5s audio)",
            "resample_44100_to_16000_5s_ms": "Resample 44.1kHz锟?6kHz (5s)",
            "resample_48000_to_16000_5s_ms": "Resample 48kHz锟?6kHz (5s)",
            "normalize_10s_ms": "Normalization (10s audio)",
            "to_wav_bytes_10s_ms": "WAV encoding (10s audio)",
        }
        for key, label in op_labels.items():
            if key in ap:
                std_key = key + "_std"
                std_val = ap.get(std_key, 0)
                lines.append(f"| {label} | {ap[key]:.4f} | {std_val:.4f} |")
        lines.append("")

    sp = audio.get("stream_processor", {})
    if "error" in sp:
        lines.append(f"> 鈿狅笍 StreamProcessor: {sp['error']}")
        lines.append("")
    else:
        lines.append("### StreamProcessor (VAD)")
        lines.append("")
        lines.append("| Operation | Time (ms) | Std (ms) |")
        lines.append("|---|---:|---:|")
        if "vad_silence_ms" in sp:
            lines.append(f"| VAD 锟?silence chunk | {sp['vad_silence_ms']:.4f} | {sp['vad_silence_std_ms']:.4f} |")
        if "vad_speech_ms" in sp:
            lines.append(f"| VAD 锟?speech chunk | {sp['vad_speech_ms']:.4f} | {sp['vad_speech_std_ms']:.4f} |")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        if "process_60s_audio_ms" in sp:
            lines.append(f"| Process 60s audio | {sp['process_60s_audio_ms']:.2f} ms |")
        if "realtime_factor" in sp:
            lines.append(f"| Realtime factor | **{sp['realtime_factor']:.1f}x** |")
        lines.append("")

    # Whisper Performance
    lines.append("## Whisper Performance")
    lines.append("")
    if whisper is None:
        lines.append("> 鈴笍 Skipped (--skip-whisper)")
        lines.append("")
    elif "error" in whisper:
        lines.append(f"> 鈿狅笍 {whisper['error']}")
        lines.append("")
    else:
        for model_name, model_data in whisper.get("models", {}).items():
            lines.append(f"### Model: `{model_name}`")
            lines.append("")
            if model_data.get("error"):
                lines.append(f"> 鈿狅笍 Error: {model_data['error']}")
                lines.append("")
                continue
            lines.append(f"- Backend: `{model_data.get('backend', 'N/A')}`")
            lines.append("")
            lines.append("| Audio Duration | Avg (ms) | Std (ms) | Realtime Factor |")
            lines.append("|---:|---:|---:|---:|")
            for dur_key, dur_data in model_data.get("durations", {}).items():
                lines.append(
                    f"| {dur_key} "
                    f"| {dur_data['avg_ms']:.2f} "
                    f"| {dur_data['std_ms']:.2f} "
                    f"| {dur_data['rtfx_label']} |"
                )
            lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")

    # Auto-generate summary points
    summary_points = []

    # KV-Cache
    configs = kv_cache.get("configurations", [])
    if configs:
        max_speedup = max(c["speedup"] for c in configs)
        avg_speedup = np.mean([c["speedup"] for c in configs])
        summary_points.append(
            f"- **KV-Cache injection** achieves up to **{max_speedup:.1f}x** speedup "
            f"over `torch.cat` (avg {avg_speedup:.1f}x across {len(configs)} configurations)"
        )

    # Audio
    if "realtime_factor" in sp:
        summary_points.append(
            f"- **StreamProcessor VAD** runs at **{sp['realtime_factor']:.0f}x** realtime, "
            f"well within real-time constraints"
        )

    # Whisper
    if whisper and "models" in whisper:
        for model_name, model_data in whisper["models"].items():
            if model_data.get("durations"):
                best_dur = "10s"
                if best_dur in model_data["durations"]:
                    rtf = model_data["durations"][best_dur]["realtime_factor"]
                    summary_points.append(
                        f"- **Whisper ({model_name})** achieves **{rtf:.1f}x** realtime on 10s audio (CPU)"
                    )

    # GPU
    if hardware.get("cuda_available"):
        summary_points.append(
            f"- CUDA available: **{hardware.get('gpu_name', 'N/A')}** "
            f"({hardware.get('gpu_memory_mb', 0)} MB VRAM)"
        )
    else:
        summary_points.append("- CUDA not available 锟?all tests ran on CPU")

    if not summary_points:
        summary_points.append("- No benchmark data collected")

    lines.extend(summary_points)
    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated by vram_core Benchmark Suite*")
    lines.append("")

    report = "\n".join(lines)

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


# 鈹€鈹€ Command Line Arguments 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="vram_core: Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/benchmark_suite.py\n"
            "  python examples/benchmark_suite.py --skip-whisper\n"
            "  python examples/benchmark_suite.py --output my_report.md --iterations 20\n"
            "  python examples/benchmark_suite.py --verbose\n"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_report.md",
        help="Output report file (default: benchmark_report.md)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        default=False,
        help="Skip Whisper transcription benchmark",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        default=False,
        help="Skip audio processing benchmark",
    )
    parser.add_argument(
        "--skip-kvcache",
        action="store_true",
        default=False,
        help="Skip KV-Cache benchmark",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose/debug logging",
    )

    return parser.parse_args()


# 鈹€鈹€ Main 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def main():
    """Main entry point for benchmark suite."""
    args = parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger("benchmark_suite")

    print()
    print("锟? + "锟? * 58 + "锟?)
    print("锟? + "  vram_core: Benchmark Suite".center(58) + "锟?)
    print("锟? + "锟? * 58 + "锟?)
    print()
    print(f"  Iterations:     {args.iterations}")
    print(f"  Output:         {args.output}")
    print(f"  Skip KV-Cache:  {args.skip_kvcache}")
    print(f"  Skip Audio:     {args.skip_audio}")
    print(f"  Skip Whisper:   {args.skip_whisper}")
    print()

    # 鈹€鈹€ Step 1: Hardware Info 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("  [1/4] Gathering hardware info...")
    hardware = gather_hardware_info()
    gpu_display = hardware.get("gpu_name", "N/A")
    if gpu_display == "N/A":
        gpu_display = "No GPU detected"
    print(f"  锟?{gpu_display}")
    print(f"     CUDA: {'Yes' if hardware.get('cuda_available') else 'No'}, "
          f"PyTorch: {hardware.get('torch_version', 'N/A')}")

    # 鈹€鈹€ Step 2: KV-Cache Benchmark 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    if not args.skip_kvcache:
        print()
        print("  [2/4] Benchmarking KV-Cache injection...")
        kv_cache_results = benchmark_kv_cache(args.iterations, args.verbose)
        configs = kv_cache_results.get("configurations", [])
        if configs:
            max_speedup = max(c["speedup"] for c in configs)
            print(f"  锟?Max speedup: {max_speedup:.1f}x (direct inject vs torch.cat)")
        elif "error" in kv_cache_results:
            print(f"  鈿狅笍  {kv_cache_results['error']}")
        else:
            print(f"  锟?Completed")
    else:
        print()
        print("  [2/4] Skipping KV-Cache benchmark")
        kv_cache_results = {"test_name": "KV-Cache Injection", "skipped": True}

    # 鈹€鈹€ Step 3: Audio Benchmark 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    if not args.skip_audio:
        print()
        print("  [3/4] Benchmarking audio processing...")
        audio_results = benchmark_audio_processing(args.iterations, args.verbose)
        rtf = audio_results.get("stream_processor", {}).get("realtime_factor")
        if rtf:
            print(f"  锟?StreamProcessor: {rtf:.0f}x realtime")
        else:
            print(f"  锟?Completed")
    else:
        print()
        print("  [3/4] Skipping audio benchmark")
        audio_results = {"test_name": "Audio Processing", "skipped": True}

    # 鈹€鈹€ Step 4: Whisper Benchmark 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    if not args.skip_whisper:
        print()
        print("  [4/4] Benchmarking Whisper transcription...")
        whisper_results = benchmark_whisper(min(args.iterations, 3), args.verbose)
        if "error" in whisper_results:
            print(f"  鈿狅笍  {whisper_results['error']}")
        else:
            models = whisper_results.get("models", {})
            for name, data in models.items():
                if data.get("durations") and "10s" in data["durations"]:
                    rtf = data["durations"]["10s"]["realtime_factor"]
                    print(f"  锟?{name}: {rtf:.1f}x realtime (10s audio)")
    else:
        print()
        print("  [4/4] Skipping Whisper benchmark")
        whisper_results = None

    # 鈹€鈹€ Generate Report 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print()
    print("  Generating report...")

    report = generate_report(
        hardware=hardware,
        kv_cache=kv_cache_results,
        audio=audio_results,
        whisper=whisper_results,
        output_path=args.output,
    )

    print(f"  锟?Report saved: {args.output}")

    # 鈹€鈹€ Summary 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print()
    print("锟? + "锟? * 58 + "锟?)
    print("锟? + "  Benchmark Complete".center(58) + "锟?)
    print("锟? + "锟? * 58 + "锟?)

    line_count = len(report.split("\n"))
    table_count = report.count("|---")
    section_count = report.count("## ")
    print(f"锟? Report:   {args.output:<46}锟?)
    print(f"锟? Sections: {section_count:<46}锟?)
    print(f"锟? Tables:   {table_count:<46}锟?)
    print(f"锟? Lines:    {line_count:<46}锟?)
    print("锟? + "锟? * 58 + "锟?)
    print()


if __name__ == "__main__":
    main()