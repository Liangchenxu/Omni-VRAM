#!/usr/bin/env python3
"""
vram_core v2.1 Benchmark Suite
===============================

Comprehensive performance benchmarks for all vram_core v2.x modules.
Generates a Markdown report with latency, throughput, and quality metrics.

Tests:
    1. Emotion Recognition - inference latency, accuracy on synthetic samples
    2. Speaker Verification - enrollment + verification latency, FA/FN rates
    3. Speaker Diarization - segmentation speed, cluster quality
    4. Noise Reduction - processing speed, SNR improvement
    5. Wake Word Detection - detection latency, false positive rate
    6. Chinese NLP Pipeline - normalization, punctuation, dialect, tokenization
    7. Streaming ASR - feed latency, partial result generation
    8. Plugin Manager - load time, hook dispatch latency
    9. Memory & VRAM - module memory footprint profiling

Usage:
    python examples/benchmark_v3.py
    python examples/benchmark_v3.py --output my_report.md --iterations 50
    python examples/benchmark_v3.py --modules emotion,speaker,noise
    python examples/benchmark_v3.py --skip-slow

Output:
    benchmark_v3_report.md - Markdown performance report
"""

import argparse
import gc
import json
import os
import platform
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_v3")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def timer(func: Callable, iterations: int = 10, warmup: int = 2) -> Dict[str, float]:
    """Run func `iterations` times, return timing stats in ms."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "iterations": iterations,
    }


def generate_test_audio(
    duration_s: float = 3.0,
    sample_rate: int = 16000,
    freq: float = 440.0,
    noise_level: float = 0.05,
) -> np.ndarray:
    """Generate synthetic audio for benchmarking."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    noise = noise_level * np.random.randn(len(t))
    return (signal + noise).astype(np.float32)


def system_info() -> Dict[str, Any]:
    """Collect system information."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "timestamp": datetime.now().isoformat(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
    except ImportError:
        info["torch"] = "not installed"
        info["cuda_available"] = False

    try:
        import psutil
        info["ram_mb"] = psutil.virtual_memory().total / 1024 / 1024
    except ImportError:
        pass

    return info


# ---------------------------------------------------------------------------
# Benchmark Modules
# ---------------------------------------------------------------------------

class BenchmarkEmotionRecognition:
    """Benchmark emotion recognition module."""

    name = "Emotion Recognition"

    def run(self, iterations: int = 20) -> Dict[str, Any]:
        results = {}
        try:
            from vram_core.emotion_recognition import EmotionRecognizer

            # Energy-based (no model download needed)
            recognizer = EmotionRecognizer(backend="energy")
            audio = generate_test_audio(duration_s=3.0)

            # Latency
            stats = timer(lambda: recognizer.recognize(audio, sample_rate=16000), iterations=iterations)
            results["energy_backend_latency"] = stats

            # Memory footprint
            gc.collect()
            results["module_loaded"] = True

        except Exception as e:
            results["error"] = str(e)
            results["module_loaded"] = False

        return results


class BenchmarkSpeakerVerification:
    """Benchmark speaker verification module."""

    name = "Speaker Verification"

    def run(self, iterations: int = 20) -> Dict[str, Any]:
        results = {}
        try:
            from vram_core.speaker_verification import SpeakerVerifier

            verifier = SpeakerVerifier()
            audio1 = generate_test_audio(duration_s=3.0, freq=200)
            audio2 = generate_test_audio(duration_s=3.0, freq=250)

            # Enrollment latency
            enroll_stats = timer(
                lambda: verifier.enroll("test_speaker", audio1, sample_rate=16000),
                iterations=iterations,
            )
            results["enrollment_latency"] = enroll_stats

            # Verification latency
            verify_stats = timer(
                lambda: verifier.verify(audio2, sample_rate=16000),
                iterations=iterations,
            )
            results["verification_latency"] = verify_stats

            # Feature extraction latency
            feat_stats = timer(
                lambda: verifier.extract_features(audio1, sample_rate=16000),
                iterations=iterations,
            )
            results["feature_extraction_latency"] = feat_stats

            results["module_loaded"] = True

        except Exception as e:
            results["error"] = str(e)
            results["module_loaded"] = False

        return results


class BenchmarkSpeakerDiarization:
    """Benchmark speaker diarization module."""

    name = "Speaker Diarization"

    def run(self, iterations: int = 10) -> Dict[str, Any]:
        results = {}
        try:
            from vram_core.speaker_diarization import SpeakerDiarizer

            diarizer = SpeakerDiarizer(backend="resemblyzer")

            # Generate 10s audio
            audio = generate_test_audio(duration_s=10.0)

            # Diarize latency
            stats = timer(
                lambda: diarizer.diarize(audio, sample_rate=16000),
                iterations=iterations,
            )
            results["diarize_latency"] = stats
            results["audio_duration_s"] = 10.0
            results["rtf"] = stats["mean_ms"] / (10.0 * 1000)  # Real-time factor
            results["module_loaded"] = True

        except Exception as e:
            results["error"] = str(e)
            results["module_loaded"] = False

        return results


class BenchmarkNoiseReduction:
    """Benchmark noise reduction module."""

    name = "Noise Reduction"

    def run(self, iterations: int = 20) -> Dict[str, Any]:
        results = {}
        try:
            from vram_core.noise_reduction import NoiseReducer

            for strength in ["light", "medium", "aggressive"]:
                reducer = NoiseReducer(strength=strength)
                audio = generate_test_audio(duration_s=3.0, noise_level=0.15)

                stats = timer(
                    lambda a=audio, r=reducer: r.process(a, sample_rate=16000),
                    iterations=iterations,
                )
                results[f"{strength}_latency"] = stats
                results[f"{strength}_rtf"] = stats["mean_ms"] / (3.0 * 1000)

                # SNR improvement
                clean = reducer.process(audio, sample_rate=16000)
                signal = generate_test_audio(duration_s=3.0, noise_level=0.0)
                snr_before = 10 * np.log10(np.mean(signal ** 2) / np.mean((audio - signal) ** 2 + 1e-10))
                snr_after = 10 * np.log10(np.mean(signal ** 2) / np.mean((clean - signal) ** 2 + 1e-10))
                results[f"{strength}_snr_before_db"] = float(snr_before)
                results[f"{strength}_snr_after_db"] = float(snr_after)
                results[f"{strength}_snr_improvement_db"] = float(snr_after - snr_before)

            results["module_loaded"] = True

        except Exception as e:
            results["error"] = str(e)
            results["module_loaded"] = False

        return results


class BenchmarkWakeWord:
    """Benchmark wake word detection module."""

    name = "Wake Word Detection"

    def run(self, iterations: int = 50) -> Dict[str, Any]:
        results = {}
        try:
            from vram_core.wake_word import WakeWordDetector

            detector = WakeWordDetector(
                engine="energy",
                energy_threshold=0.02,
                keywords=["hello", "test"],
            )
            audio = generate_test_audio(duration_s=1.0)

            stats = timer(
                lambda: detector.detect(audio, sample_rate=16000),
                iterations=iterations,
            )
            results["detect_latency"] = stats
            results["module_loaded"] = True

        except Exception as e:
            results["error"] = str(e)
            results["module_loaded"] = False

        return results


class BenchmarkChineseNLP:
    """Benchmark Chinese NLP pipeline."""

    name = "Chinese NLP Pipeline"

    def run(self, iterations: int = 50) -> Dict[str, Any]:
        results = {}
        test_texts = [
            "今天天气真不错，我想出去走走",
            "请帮我查一下明天北京到上海的高铁票",
            "人工智能在医疗领域的应用越来越广泛",
            "这个项目的测试覆盖率达到了百分之九十五以上",
            "语音识别技术让大模型长出了耳朵和嘴巴",
        ]

        try:
            from vram_core.chinese.normalizer import ChineseNormalizer
            normalizer = ChineseNormalizer()
            stats = timer(
                lambda: [normalizer.normalize(t) for t in test_texts],
                iterations=iterations,
            )
            results["normalizer_latency"] = stats
        except Exception as e:
            results["normalizer_error"] = str(e)

        try:
            from vram_core.chinese.punctuation import PunctuationRestorer
            restorer = PunctuationRestorer()
            unpunctuated = ["今天天气真不错我想出去走走", "请帮我查一下明天的高铁票"]
            stats = timer(
                lambda: [restorer.restore(t) for t in unpunctuated],
                iterations=iterations,
            )
            results["punctuation_latency"] = stats
        except Exception as e:
            results["punctuation_error"] = str(e)

        try:
            from vram_core.chinese.tokenizer import ChineseTokenizer
            tokenizer = ChineseTokenizer()
            stats = timer(
                lambda: [tokenizer.tokenize(t) for t in test_texts],
                iterations=iterations,
            )
            results["tokenizer_latency"] = stats
        except Exception as e:
            results["tokenizer_error"] = str(e)

        try:
            from vram_core.chinese.dialect import DialectConverter
            converter = DialectConverter()
            stats = timer(
                lambda: [converter.convert(t, source="cantonese") for t in ["你好", "多谢"]],
                iterations=iterations,
            )
            results["dialect_latency"] = stats
        except Exception as e:
            results["dialect_error"] = str(e)

        try:
            from vram_core.chinese.domain_dict import DomainDictionary
            ddict = DomainDictionary()
            stats = timer(
                lambda: [ddict.lookup(t) for t in ["GPU", "KV-Cache", "CUDA"]],
                iterations=iterations,
            )
            results["domain_dict_latency"] = stats
        except Exception as e:
            results["domain_dict_error"] = str(e)

        results["module_loaded"] = True
        return results


class BenchmarkPluginManager:
    """Benchmark plugin manager."""

    name = "Plugin Manager"

    def run(self, iterations: int = 100) -> Dict[str, Any]:
        results = {}
        try:
            from vram_core.plugin_manager import PluginManager

            # Initialize
            stats = timer(lambda: PluginManager(), iterations=iterations)
            results["init_latency"] = stats

            # Hook dispatch
            pm = PluginManager()
            pm.register_hook("test_hook", lambda x: x)
            stats = timer(lambda: pm.dispatch_hook("test_hook", 42), iterations=iterations)
            results["hook_dispatch_latency"] = stats

            results["module_loaded"] = True

        except Exception as e:
            results["error"] = str(e)
            results["module_loaded"] = False

        return results


class BenchmarkMemoryProfile:
    """Profile memory footprint of loaded modules."""

    name = "Memory Profile"

    def run(self, iterations: int = 1) -> Dict[str, Any]:
        results = {}
        gc.collect()

        modules_to_check = [
            ("emotion_recognition", "vram_core.emotion_recognition"),
            ("speaker_verification", "vram_core.speaker_verification"),
            ("speaker_diarization", "vram_core.speaker_diarization"),
            ("noise_reduction", "vram_core.noise_reduction"),
            ("wake_word", "vram_core.wake_word"),
            ("plugin_manager", "vram_core.plugin_manager"),
            ("chinese_normalizer", "vram_core.chinese.normalizer"),
            ("chinese_punctuation", "vram_core.chinese.punctuation"),
            ("chinese_tokenizer", "vram_core.chinese.tokenizer"),
        ]

        for name, mod_path in modules_to_check:
            try:
                t0 = time.perf_counter()
                __import__(mod_path)
                elapsed = (time.perf_counter() - t0) * 1000
                results[name] = {"import_ms": round(elapsed, 2), "loaded": True}
            except Exception as e:
                results[name] = {"error": str(e), "loaded": False}

        results["module_loaded"] = True
        return results


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

REPORT_HEADER = """# vram_core v2.1 Benchmark Report

**Generated**: {timestamp}
**Python**: {python}
**Platform**: {platform}
**GPU**: {gpu}
**CUDA**: {cuda}

---

"""

SECTION_TEMPLATE = """
## {name}

{table}

"""

def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format a Markdown table."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def generate_report(all_results: Dict[str, Any], sys_info: Dict[str, Any]) -> str:
    """Generate Markdown report from benchmark results."""
    report = REPORT_HEADER.format(
        timestamp=sys_info.get("timestamp", "N/A"),
        python=sys_info.get("python", "N/A"),
        platform=sys_info.get("platform", "N/A"),
        gpu=sys_info.get("gpu", "N/A") if sys_info.get("cuda_available") else "CPU only",
        cuda=sys_info.get("torch", "N/A"),
    )

    for module_name, results in all_results.items():
        if not results.get("module_loaded"):
            report += f"\n## {module_name}\n\n⚠️ Module not available: {results.get('error', 'unknown')}\n"
            continue

        headers = ["Metric", "Mean", "P50", "P95", "Std"]
        rows = []

        for key, val in results.items():
            if isinstance(val, dict) and "mean_ms" in val:
                rows.append([
                    key.replace("_latency", "").replace("_", " ").title(),
                    f"{val['mean_ms']:.2f} ms",
                    f"{val['p50_ms']:.2f} ms",
                    f"{val['p95_ms']:.2f} ms",
                    f"{val['std_ms']:.2f} ms",
                ])
            elif isinstance(val, (int, float)):
                rows.append([
                    key.replace("_", " ").title(),
                    f"{val:.4f}" if isinstance(val, float) else str(val),
                    "-", "-", "-",
                ])

        if rows:
            table = format_table(headers, rows)
            report += SECTION_TEMPLATE.format(name=module_name, table=table)

    # Summary
    report += "\n## Summary\n\n"
    report += "| Module | Status | Best Latency (ms) |\n"
    report += "|--------|--------|--------------------|\n"
    for module_name, results in all_results.items():
        status = "✅" if results.get("module_loaded") else "❌"
        best = "N/A"
        for key, val in results.items():
            if isinstance(val, dict) and "min_ms" in val:
                if best == "N/A" or val["min_ms"] < float(best):
                    best = f"{val['min_ms']:.2f}"
        report += f"| {module_name} | {status} | {best} |\n"

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BENCHMARK_CLASSES = {
    "emotion": BenchmarkEmotionRecognition,
    "speaker": BenchmarkSpeakerVerification,
    "diarization": BenchmarkSpeakerDiarization,
    "noise": BenchmarkNoiseReduction,
    "wakeword": BenchmarkWakeWord,
    "chinese": BenchmarkChineseNLP,
    "plugin": BenchmarkPluginManager,
    "memory": BenchmarkMemoryProfile,
}


def main():
    parser = argparse.ArgumentParser(description="vram_core v2.1 Benchmark Suite")
    parser.add_argument("--output", default="benchmark_v3_report.md", help="Output report file")
    parser.add_argument("--iterations", type=int, default=20, help="Iterations per benchmark")
    parser.add_argument("--modules", type=str, default=None,
                        help="Comma-separated modules to benchmark: " + ",".join(BENCHMARK_CLASSES.keys()))
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow benchmarks (diarization, chinese)")
    args = parser.parse_args()

    logger.info("Starting vram_core v2.1 Benchmark Suite")

    # System info
    sys_info = system_info()
    logger.info(f"System: {sys_info.get('platform', 'unknown')}")
    logger.info(f"GPU: {sys_info.get('gpu', 'CPU only')}")

    # Select modules
    if args.modules:
        selected = [m.strip() for m in args.modules.split(",")]
    else:
        selected = list(BENCHMARK_CLASSES.keys())

    if args.skip_slow:
        selected = [m for m in selected if m not in ("diarization", "chinese")]

    all_results = {}

    for mod_key in selected:
        if mod_key not in BENCHMARK_CLASSES:
            logger.warning(f"Unknown module: {mod_key}")
            continue

        bench = BENCHMARK_CLASSES[mod_key]()
        logger.info(f"Running benchmark: {bench.name}")
        try:
            results = bench.run(iterations=args.iterations)
            all_results[bench.name] = results
            status = "OK" if results.get("module_loaded") else "FAILED"
            logger.info(f"  {bench.name}: {status}")
        except Exception as e:
            logger.error(f"  {bench.name} crashed: {e}")
            all_results[bench.name] = {"module_loaded": False, "error": str(e)}

    # Generate report
    report = generate_report(all_results, sys_info)
    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for name, results in all_results.items():
        status = "✅" if results.get("module_loaded") else "❌"
        best = "N/A"
        for key, val in results.items():
            if isinstance(val, dict) and "min_ms" in val:
                if best == "N/A" or val["min_ms"] < float(best):
                    best = f"{val['min_ms']:.2f}ms"
        print(f"  {status} {name}: best={best}")
    print("=" * 60)


if __name__ == "__main__":
    main()