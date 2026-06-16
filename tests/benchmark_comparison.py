import pytest
"""
Omni-VRAM vs faster-whisper Performance Comparison
===================================================

Compares transcription speed, latency, and VRAM usage between
Omni-VRAM (WhisperBridge) and faster-whisper on identical audio.

Usage:
    python tests/benchmark_comparison.py
    python tests/benchmark_comparison.py --model base --runs 3
"""

import argparse
import os
import sys
import time
import warnings
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")

# ── Add project root to path ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_RATE = 16000


# ── Data Classes ──────────────────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    engine: str
    model: str
    audio_duration_s: float
    transcription_time_s: float
    rtf: float  # Real-Time Factor (lower = faster)
    first_token_latency_ms: float
    vram_peak_mb: float
    vram_delta_mb: float
    transcript_length: int = 0
    transcript_preview: str = ""


@dataclass
class ComparisonReport:
    """Full comparison report."""
    timestamp: str = ""
    hardware_info: Dict = field(default_factory=dict)
    results: List[BenchmarkResult] = field(default_factory=list)


# ── Audio Generation ─────────────────────────────────────────────────────
def generate_test_audio(duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate synthetic speech-like audio for benchmarking."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 400 * t)
        + 0.1 * np.sin(2 * np.pi * 800 * t)
        + 0.05 * np.sin(2 * np.pi * 1600 * t)
        + 0.02 * np.random.randn(len(t)).astype(np.float32)
    )
    signal = signal / np.max(np.abs(signal)) * 0.8
    return signal


def save_test_audio(audio: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE):
    """Save audio array to WAV file."""
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


# ── GPU Monitoring ───────────────────────────────────────────────────────
class VRAMMonitor:
    """Monitor GPU VRAM usage."""

    def __init__(self):
        self.baseline_mb = 0.0
        self._available = False
        try:
            import torch
            if torch.cuda.is_available():
                self._available = True
                torch.cuda.reset_peak_memory_stats()
                self.baseline_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        except ImportError:
            pass

    def get_peak_mb(self) -> float:
        if not self._available:
            return 0.0
        import torch
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    def get_peak_delta(self) -> float:
        if not self._available:
            return 0.0
        return self.get_peak_mb() - self.baseline_mb


# ── Hardware Info ─────────────────────────────────────────────────────────
def collect_hardware_info() -> Dict:
    """Collect hardware information."""
    import platform

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or "Unknown",
        "cpu_cores": os.cpu_count(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
            )
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["torch_version"] = "Not installed"
        info["cuda_available"] = False

    try:
        import faster_whisper
        info["faster_whisper_version"] = getattr(faster_whisper, "__version__", "installed")
    except ImportError:
        info["faster_whisper_version"] = "Not installed"

    return info


# ── Omni-VRAM Benchmark ──────────────────────────────────────────────────
def benchmark_omni_vram(
    audio: np.ndarray,
    model_name: str,
    runs: int,
    audio_durations: List[float],
) -> List[BenchmarkResult]:
    """Benchmark Omni-VRAM WhisperBridge."""
    results = []
    try:
        from vram_core.whisper_bridge import WhisperBridge, WhisperBackend
    except ImportError:
        print("  ⚠  vram_core.whisper_bridge not available, skipping")
        return results

    bridge = None
    for backend in [WhisperBackend.PYTHON_WHISPER, WhisperBackend.CLI]:
        try:
            bridge = WhisperBridge(backend=backend)
            if bridge.is_ready:
                print(f"  ✓  WhisperBridge ready (backend={backend.name})")
                break
        except Exception:
            continue

    if bridge is None or not bridge.is_ready:
        print("  ⚠  No working Whisper backend found for Omni-VRAM")
        return results

    for dur in audio_durations:
        test_audio = audio[: int(dur * SAMPLE_RATE)] if dur <= len(audio) / SAMPLE_RATE else generate_test_audio(dur)
        monitor = VRAMMonitor()
        times, ftls = [], []
        transcript = ""

        for run in range(runs):
            start = time.perf_counter()
            try:
                result = bridge.transcribe(test_audio)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                ftls.append(elapsed * 1000)
                transcript = result.text if hasattr(result, "text") else str(result)
            except Exception as e:
                print(f"  ⚠  Omni-VRAM run {run + 1} failed: {e}")

        if times:
            avg_t = np.mean(times)
            results.append(BenchmarkResult(
                engine="Omni-VRAM", model=model_name,
                audio_duration_s=dur, transcription_time_s=round(avg_t, 4),
                rtf=round(avg_t / dur, 4) if dur > 0 else 0,
                first_token_latency_ms=round(np.mean(ftls), 2),
                vram_peak_mb=round(monitor.get_peak_mb(), 1),
                vram_delta_mb=round(monitor.get_peak_delta(), 1),
                transcript_length=len(transcript),
                transcript_preview=transcript[:80],
            ))
    return results


# ── faster-whisper Benchmark ─────────────────────────────────────────────
def benchmark_faster_whisper(
    audio: np.ndarray,
    model_name: str,
    runs: int,
    audio_durations: List[float],
) -> List[BenchmarkResult]:
    """Benchmark faster-whisper."""
    results = []
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("  ⚠  faster-whisper not installed. pip install faster-whisper")
        return results

    device, compute_type = "cuda", "float16"
    try:
        import torch
        if not torch.cuda.is_available():
            device, compute_type = "cpu", "int8"
    except ImportError:
        device, compute_type = "cpu", "int8"

    try:
        print(f"  Loading faster-whisper '{model_name}' on {device}...")
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"  ⚠  Failed to load faster-whisper: {e}")
        return results

    for dur in audio_durations:
        test_audio = audio[: int(dur * SAMPLE_RATE)] if dur <= len(audio) / SAMPLE_RATE else generate_test_audio(dur)
        monitor = VRAMMonitor()
        times, ftls = [], []
        transcript = ""

        for run in range(runs):
            start = time.perf_counter()
            try:
                segments, _ = model.transcribe(test_audio, beam_size=5, vad_filter=True)
                full_text, first_token_time = "", None
                for seg in segments:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    full_text += seg.text
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                if first_token_time:
                    ftls.append((first_token_time - start) * 1000)
                transcript = full_text
            except Exception as e:
                print(f"  ⚠  faster-whisper run {run + 1} failed: {e}")

        if times:
            avg_t = np.mean(times)
            results.append(BenchmarkResult(
                engine="faster-whisper", model=model_name,
                audio_duration_s=dur, transcription_time_s=round(avg_t, 4),
                rtf=round(avg_t / dur, 4) if dur > 0 else 0,
                first_token_latency_ms=round(np.mean(ftls), 2) if ftls else round(avg_t * 1000, 2),
                vram_peak_mb=round(monitor.get_peak_mb(), 1),
                vram_delta_mb=round(monitor.get_peak_delta(), 1),
                transcript_length=len(transcript),
                transcript_preview=transcript[:80],
            ))
    return results


# ── Streaming Latency Benchmark ──────────────────────────────────────────
def benchmark_streaming_latency(audio: np.ndarray, model_name: str, runs: int) -> Dict:
    """Benchmark real-time streaming latency for both engines."""
    results = {}

    # Omni-VRAM streaming
    try:
        from vram_core.stream_processor import StreamProcessor
        sp = StreamProcessor()
        chunk_size = int(SAMPLE_RATE * 0.5)
        latencies = []
        for _ in range(runs):
            chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
            start = time.perf_counter()
            for chunk in chunks:
                sp.feed(chunk)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed / max(len(chunks), 1))
        results["Omni-VRAM"] = {
            "avg_chunk_latency_ms": round(np.mean(latencies), 2),
            "p95_latency_ms": round(np.percentile(latencies, 95), 2),
            "p99_latency_ms": round(np.percentile(latencies, 99), 2),
            "chunk_size_ms": 500,
        }
    except Exception as e:
        results["Omni-VRAM"] = {"error": str(e)}

    # faster-whisper chunked
    try:
        from faster_whisper import WhisperModel
        device, compute_type = "cuda", "float16"
        try:
            import torch
            if not torch.cuda.is_available():
                device, compute_type = "cpu", "int8"
        except ImportError:
            device, compute_type = "cpu", "int8"
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        chunk_size = int(SAMPLE_RATE * 0.5)
        latencies = []
        for _ in range(runs):
            chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
            chunk_latencies = []
            for chunk in chunks:
                start = time.perf_counter()
                segments, _ = model.transcribe(chunk, vad_filter=True)
                for _ in segments:
                    pass
                chunk_latencies.append((time.perf_counter() - start) * 1000)
            latencies.append(np.mean(chunk_latencies))
        results["faster-whisper"] = {
            "avg_chunk_latency_ms": round(np.mean(latencies), 2),
            "p95_latency_ms": round(np.percentile(latencies, 95), 2),
            "p99_latency_ms": round(np.percentile(latencies, 99), 2),
            "chunk_size_ms": 500,
        }
    except ImportError:
        results["faster-whisper"] = {"error": "faster-whisper not installed"}
    except Exception as e:
        results["faster-whisper"] = {"error": str(e)}

    return results


# ── Report Generation ────────────────────────────────────────────────────
def rtf_str(rtf: float) -> str:
    if rtf < 0.3:
        return f"**{rtf}** ⚡"
    elif rtf < 1.0:
        return f"{rtf} ✅"
    return f"{rtf} 🐌"


def generate_markdown_report(
    report: ComparisonReport,
    streaming_results: Dict,
    output_path: str,
) -> str:
    """Generate Markdown comparison report."""
    L = []
    L.append("# Omni-VRAM vs faster-whisper Performance Comparison\n")
    L.append(f"**Generated**: {report.timestamp}  ")
    L.append(f"**Omni-VRAM Version**: 2.1.1\n")

    hw = report.hardware_info
    L.append("## Hardware Information\n")
    L.append("| Item | Value |")
    L.append("|------|-------|")
    for key, label in [
        ("platform", "Platform"), ("cpu", "CPU"), ("cpu_cores", "CPU Cores"),
        ("gpu_name", "GPU"), ("gpu_memory_gb", "GPU Memory"),
        ("cuda_version", "CUDA"), ("torch_version", "PyTorch"),
        ("faster_whisper_version", "faster-whisper"), ("python", "Python"),
    ]:
        L.append(f"| {label} | {hw.get(key, 'N/A')} |")
    L.append("")

    # Transcription Speed
    speed = [r for r in report.results]
    if speed:
        L.append("## Transcription Speed Comparison\n")
        L.append("| Engine | Audio Duration | Time | RTF | First Token Latency | VRAM Peak |")
        L.append("|--------|---------------|-----:|-----|--------------------:|----------:|")
        for r in speed:
            L.append(
                f"| {r.engine} | {r.audio_duration_s}s | "
                f"{r.transcription_time_s}s | {rtf_str(r.rtf)} | "
                f"{r.first_token_latency_ms}ms | {r.vram_peak_mb}MB |"
            )
        L.append("")

        L.append("### Speed Analysis\n")
        durations = sorted(set(r.audio_duration_s for r in speed))
        for dur in durations:
            omni = [r for r in speed if r.engine == "Omni-VRAM" and r.audio_duration_s == dur]
            fw = [r for r in speed if r.engine == "faster-whisper" and r.audio_duration_s == dur]
            if omni and fw:
                ratio = omni[0].transcription_time_s / fw[0].transcription_time_s if fw[0].transcription_time_s > 0 else 999
                if ratio < 1:
                    L.append(f"- **{dur}s audio**: Omni-VRAM is **{1 / ratio:.1f}x faster**")
                else:
                    L.append(f"- **{dur}s audio**: faster-whisper is **{ratio:.1f}x faster**")
        L.append("")

    # Streaming
    if streaming_results:
        L.append("## Real-time Streaming Latency\n")
        L.append("| Engine | Avg Latency | P95 | P99 | Chunk Size |")
        L.append("|--------|------------:|----:|----:|----------:|")
        for engine, data in streaming_results.items():
            if "error" in data:
                L.append(f"| {engine} | Error | — | — | — |")
            else:
                L.append(
                    f"| {engine} | {data['avg_chunk_latency_ms']}ms | "
                    f"{data['p95_latency_ms']}ms | {data['p99_latency_ms']}ms | "
                    f"{data['chunk_size_ms']}ms |"
                )
        L.append("")

    # VRAM
    vram_results = [r for r in speed if r.vram_peak_mb > 0]
    if vram_results:
        L.append("## VRAM Usage\n")
        L.append("| Engine | Duration | Peak VRAM | Delta |")
        L.append("|--------|---------:|----------:|------:|")
        for r in vram_results:
            L.append(f"| {r.engine} | {r.audio_duration_s}s | {r.vram_peak_mb}MB | {r.vram_delta_mb}MB |")
        L.append("")

    # Summary
    L.append("## Summary\n")
    if speed:
        fastest = min(speed, key=lambda r: r.rtf)
        L.append(f"- **Fastest RTF**: {fastest.engine} (RTF={fastest.rtf}, {fastest.audio_duration_s}s audio)")
        lowest_ftl = min(speed, key=lambda r: r.first_token_latency_ms)
        L.append(f"- **Lowest First-Token Latency**: {lowest_ftl.engine} ({lowest_ftl.first_token_latency_ms}ms)")
        vram_c = [r for r in speed if r.vram_peak_mb > 0]
        if vram_c:
            lowest = min(vram_c, key=lambda r: r.vram_peak_mb)
            L.append(f"- **Lowest VRAM**: {lowest.engine} ({lowest.vram_peak_mb}MB peak)")
    L.append("")
    L.append("---")
    L.append("*Benchmark generated by Omni-VRAM v2.1.1 benchmark_comparison.py*")

    text = "\n".join(L)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Omni-VRAM vs faster-whisper Benchmark")
    parser.add_argument("--model", default="base", help="Whisper model size (default: base)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test")
    parser.add_argument("--durations", nargs="+", type=float, default=[10.0, 60.0])
    parser.add_argument("--output", default=None, help="Output report path")
    parser.add_argument("--skip-streaming", action="store_true")
    args = parser.parse_args()

    output_path = args.output or str(Path(__file__).resolve().parent.parent / "benchmark_comparison.md")

    print("=" * 60)
    print("  Omni-VRAM vs faster-whisper Benchmark")
    print("=" * 60)
    print(f"  Model: {args.model} | Runs: {args.runs} | Durations: {args.durations}s")
    print(f"  Output: {output_path}")
    print("=" * 60)

    print("\n[1/4] Hardware info...")
    hw_info = collect_hardware_info()
    print(f"  GPU: {hw_info.get('gpu_name', 'N/A')}")

    print("\n[2/4] Generating test audio...")
    max_dur = max(args.durations)
    test_audio = generate_test_audio(max_dur)
    audio_path = str(Path(output_path).parent / "test_audio_benchmark.wav")
    save_test_audio(test_audio, audio_path)
    print(f"  Saved: {audio_path}")

    print("\n[3/4] Running benchmarks...")
    all_results = []

    print("\n  ── Omni-VRAM ──")
    all_results.extend(benchmark_omni_vram(test_audio, args.model, args.runs, args.durations))

    print("\n  ── faster-whisper ──")
    all_results.extend(benchmark_faster_whisper(test_audio, args.model, args.runs, args.durations))

    streaming = {}
    if not args.skip_streaming:
        print("\n  ── Streaming Latency ──")
        streaming = benchmark_streaming_latency(test_audio, args.model, min(args.runs, 3))

    print("\n[4/4] Generating report...")
    report = ComparisonReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        hardware_info=hw_info,
        results=all_results,
    )
    generate_markdown_report(report, streaming, output_path)
    print(f"\n  ✓ Report saved to: {output_path}")

    print("\n" + "=" * 60)
    for r in all_results:
        print(
            f"  {r.engine:<15} | {r.audio_duration_s:>5.0f}s | "
            f"Time: {r.transcription_time_s:.3f}s | RTF: {rtf_str(r.rtf)}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()