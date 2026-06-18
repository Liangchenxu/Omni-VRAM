"""
WhisperOptimizer - High-Performance Whisper Transcription Engine
=================================================================

Performance optimizations over the base WhisperBridge:
  1. Model preloading & warmup (eliminate first-request latency)
  2. Batch transcription (multi-file with auto-merge of short clips)
  3. Result caching (hash-based, avoids redundant transcription)
  4. Streaming output (long audio yields partial results)
  5. GPU VRAM optimization (int8/int4 quantization, auto-release)

Goal: Faster than vanilla faster-whisper in every benchmark.
"""

import hashlib
import io
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from vram_core.whisper.result import WhisperResult

logger = logging.getLogger("vram_core.whisper.optimizer")

# ── Named Constants ──────────────────────────────────────────────
_DEFAULT_SAMPLE_RATE: int = 16000
_INT16_MAX: float = 32768.0
_INT16_CLIP_LOW: int = -32768
_INT16_CLIP_HIGH: int = 32767


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """Result container for batch transcription."""

    results: List[WhisperResult] = field(default_factory=list)
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    total_audio_duration: float = 0.0
    total_processing_time: float = 0.0
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def avg_rtf(self) -> float:
        """Average Real-Time Factor (lower is better)."""
        if self.total_audio_duration <= 0:
            return 0.0
        return self.total_processing_time / self.total_audio_duration

    @property
    def texts(self) -> List[str]:
        """List of transcription texts."""
        return [r.text for r in self.results]

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the batch result."""
        return {
            "total_files": self.total_files,
            "successful": self.successful,
            "failed": self.failed,
            "total_audio_duration": round(self.total_audio_duration, 2),
            "total_processing_time": round(self.total_processing_time, 3),
            "avg_rtf": round(self.avg_rtf, 4),
            "errors": self.errors,
        }


@dataclass
class StreamChunk:
    """A single chunk in a streaming transcription."""

    text: str
    start: float
    end: float
    is_final: bool = False
    chunk_index: int = 0


@dataclass
class CacheStats:
    """Transcription cache statistics."""

    total_entries: int = 0
    total_size_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Transcription Cache
# ---------------------------------------------------------------------------

class TranscriptionCache:
    """
    File-hash based transcription result cache.

    Stores results in memory with optional disk persistence.
    Cache key = SHA256(audio_bytes) + model + language + compute_type.
    """

    def __init__(
        self,
        max_entries: int = 256,
        cache_dir: Optional[Path] = None,
        enable_disk: bool = False,
    ):
        self._cache: Dict[str, WhisperResult] = {}
        self._max_entries = max_entries
        self._cache_dir = cache_dir or (Path.home() / ".cache" / "vram_core" / "transcription")
        self._enable_disk = enable_disk
        self._lock = threading.Lock()
        self._stats = CacheStats()

        if enable_disk:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # -- public API ---------------------------------------------------------

    def get(
        self,
        audio: Union[np.ndarray, bytes],
        model: str,
        language: str,
        compute_type: str,
    ) -> Optional[WhisperResult]:
        """Look up cached result. Returns None on miss."""
        key = self._make_key(audio, model, language, compute_type)
        with self._lock:
            if key in self._cache:
                self._stats.hit_count += 1
                logger.debug("Cache hit: %s...", key[:16])
                return self._cache[key]

        # Try disk cache
        if self._enable_disk:
            disk_result = self._load_from_disk(key)
            if disk_result is not None:
                with self._lock:
                    self._cache[key] = disk_result
                    self._stats.hit_count += 1
                return disk_result

        self._stats.miss_count += 1
        return None

    def put(
        self,
        audio: Union[np.ndarray, bytes],
        model: str,
        language: str,
        compute_type: str,
        result: WhisperResult,
    ) -> None:
        """Store a transcription result in the cache."""
        key = self._make_key(audio, model, language, compute_type)
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_entries:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[key] = result
            self._stats.total_entries = len(self._cache)

        if self._enable_disk:
            self._save_to_disk(key, result)

    def clear(self) -> None:
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()

        if self._enable_disk and self._cache_dir.exists():
            import shutil
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def stats(self) -> CacheStats:
        with self._lock:
            self._stats.total_entries = len(self._cache)
            return self._stats

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _audio_to_bytes(audio: Union[np.ndarray, bytes]) -> bytes:
        if isinstance(audio, bytes):
            return audio
        return audio.tobytes()

    def _make_key(
        self,
        audio: Union[np.ndarray, bytes],
        model: str,
        language: str,
        compute_type: str,
    ) -> str:
        audio_bytes = self._audio_to_bytes(audio)
        h = hashlib.sha256(audio_bytes).hexdigest()[:32]
        return f"{h}_{model}_{language}_{compute_type}"

    def _disk_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _save_to_disk(self, key: str, result: WhisperResult) -> None:
        import json
        try:
            data = {
                "text": result.text,
                "language": result.language,
                "confidence": result.confidence,
                "segments": result.segments,
                "backend": result.backend,
                "model": result.model,
            }
            self._disk_path(key).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except (OSError, TypeError, ValueError) as e:
            logger.warning("Disk cache save failed: %s", e)

    def _load_from_disk(self, key: str) -> Optional[WhisperResult]:
        import json
        path = self._disk_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return WhisperResult(
                text=data["text"],
                language=data.get("language", "unknown"),
                confidence=data.get("confidence", 0.0),
                segments=data.get("segments", []),
                backend=data.get("backend"),
                model=data.get("model"),
            )
        except (OSError, TypeError, ValueError, KeyError) as e:
            logger.warning("Disk cache load failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# WhisperOptimizer  (main class)
# ---------------------------------------------------------------------------

class WhisperOptimizer:
    """
    High-performance Whisper transcription engine.

    Wraps faster-whisper with:
      - Model preloading & warmup
      - Batch transcription (multi-file, parallel)
      - SHA256 result caching
      - Streaming chunk output
      - int8 / int4 quantization + VRAM auto-release
    """

    QUANTIZE_PRESETS: Dict[str, Dict[str, Any]] = {
        "int8": {"compute_type": "int8", "description": "8-bit integer (2x smaller, ~1% accuracy loss)"},
        "int8_float16": {"compute_type": "int8_float16", "description": "8-bit weights, float16 compute (GPU only)"},
        "float16": {"compute_type": "float16", "description": "Half precision (GPU only, default)"},
        "float32": {"compute_type": "float32", "description": "Full precision (slowest, most accurate)"},
        "int16": {"compute_type": "int16", "description": "16-bit integer"},
    }

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        enable_cache: bool = True,
        cache_max_entries: int = 256,
        enable_disk_cache: bool = False,
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        """
        Initialize the optimized Whisper engine.

        Args:
            model_name:        Model size (tiny/base/small/medium/large-v3).
            device:            'cuda' or 'cpu'.
            compute_type:      Precision (int8/float16/float32/int8_float16).
            language:          Default language code.
            enable_cache:      Enable in-memory result caching.
            cache_max_entries: Max cached transcription results.
            enable_disk_cache: Persist cache to disk.
            beam_size:         Beam search width (lower = faster).
            vad_filter:        Enable VAD to skip silence (recommended).
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter

        self._model = None
        self._model_lock = threading.Lock()
        self._warmup_done = False

        # Cache
        self._cache: Optional[TranscriptionCache] = None
        if enable_cache:
            self._cache = TranscriptionCache(
                max_entries=cache_max_entries,
                enable_disk=enable_disk_cache,
            )

        # Stats
        self._transcribe_count = 0
        self._total_audio_duration = 0.0
        self._total_processing_time = 0.0

        _cache_state = 'on' if enable_cache else 'off'
        logger.info(
            "WhisperOptimizer initialized "
            "(model=%s, device=%s, compute_type=%s, "
            "beam_size=%s, vad=%s, cache=%s)",
            model_name, device, compute_type, beam_size, vad_filter, _cache_state,
        )

    # ------------------------------------------------------------------
    # 1. Model Preloading & Warmup
    # ------------------------------------------------------------------

    def preload(self) -> None:
        """
        Preload the model into memory. Call at startup to eliminate
        first-request latency.
        """
        logger.info("Preloading model '%s'...", self.model_name)
        t0 = time.perf_counter()
        self._ensure_model()
        elapsed = time.perf_counter() - t0
        logger.info("Model preloaded in %.2fs", elapsed)

    def warmup(self, duration_s: float = 3.0) -> float:
        """
        Run a warmup inference on dummy audio to prime all CUDA kernels
        and compilation caches. Returns warmup time in seconds.

        Args:
            duration_s: Duration of dummy audio in seconds.

        Returns:
            Time taken for warmup in seconds.
        """
        logger.info("Warming up model with %ss dummy audio...", duration_s)
        t0 = time.perf_counter()

        self._ensure_model()

        # Generate silent dummy audio
        sample_rate = 16000
        dummy = np.zeros(int(duration_s * sample_rate), dtype=np.float32)

        # Run a throwaway inference
        self._transcribe_internal(dummy, sample_rate, skip_cache=True)

        elapsed = time.perf_counter() - t0
        self._warmup_done = True
        logger.info("Warmup completed in %.2fs", elapsed)
        return elapsed

    @property
    def is_warm(self) -> bool:
        """Whether the model has been warmed up."""
        return self._warmup_done

    # ------------------------------------------------------------------
    # 2. Single Transcription (with cache)
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray, bytes],
        sample_rate: int = 16000,
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs,
    ) -> WhisperResult:
        """
        Transcribe audio with all optimizations applied:
        - Auto model load (or use preloaded)
        - Cache lookup (skip inference on hit)
        - Optimal beam_size & VAD settings

        Args:
            audio:       File path, numpy array, or bytes.
            sample_rate: Sample rate for numpy input.
            language:    Override language.
            task:        'transcribe' or 'translate'.
            **kwargs:    Additional faster-whisper options.

        Returns:
            WhisperResult with transcription.
        """
        t0 = time.perf_counter()

        audio_array = self._prepare_audio(audio, sample_rate)
        effective_language = language or self.language

        # Cache lookup
        if self._cache:
            cached = self._cache.get(
                audio_array, self.model_name, effective_language or "auto", self.compute_type
            )
            if cached is not None:
                elapsed = time.perf_counter() - t0
                logger.info("Cache hit! Transcription served in %.1fms", elapsed * 1000)
                return cached

        result = self._transcribe_internal(
            audio_array, sample_rate,
            language=effective_language, task=task, **kwargs
        )

        # Cache store
        if self._cache:
            self._cache.put(
                audio_array, self.model_name, effective_language or "auto",
                self.compute_type, result
            )

        elapsed = time.perf_counter() - t0
        audio_dur = len(audio_array) / sample_rate
        rtf = elapsed / audio_dur if audio_dur > 0 else 0

        self._transcribe_count += 1
        self._total_audio_duration += audio_dur
        self._total_processing_time += elapsed

        logger.info(
            "Transcribed %.1fs audio in %.2fs "
            "(RTF=%.3f, lang=%s, conf=%.2f)",
            audio_dur, elapsed, rtf, result.language, result.confidence,
        )

        return result

    # ------------------------------------------------------------------
    # 3. Batch Transcription
    # ------------------------------------------------------------------

    def transcribe_batch(
        self,
        audio_files: List[Union[str, Path]],
        language: Optional[str] = None,
        max_workers: int = 2,
        merge_short: bool = True,
        merge_threshold_s: float = 5.0,
        **kwargs,
    ) -> BatchResult:
        """
        Transcribe multiple audio files efficiently.

        Optimization strategies:
        - Short audio files (< threshold) are concatenated into a single
          inference call to reduce per-call overhead.
        - Parallel processing for files that can't be merged.
        - Shared model instance across all files.

        Args:
            audio_files:      List of audio file paths.
            language:         Override language.
            max_workers:      Max parallel workers for non-merged files.
            merge_short:      Whether to merge short audio files.
            merge_threshold_s: Threshold in seconds for merging.
            **kwargs:         Additional transcription options.

        Returns:
            BatchResult with all transcription results.
        """
        t0 = time.perf_counter()
        batch = BatchResult(total_files=len(audio_files))

        if not audio_files:
            return batch

        effective_language = language or self.language

        # Load all audio and classify by duration
        short_files: List[Tuple[str, np.ndarray]] = []
        long_files: List[Tuple[str, np.ndarray]] = []

        for fpath in audio_files:
            fpath = Path(fpath)
            if not fpath.exists():
                batch.errors[str(fpath)] = f"File not found: {fpath}"
                batch.failed += 1
                continue
            try:
                audio_array = self._prepare_audio(str(fpath))
                duration = len(audio_array) / 16000
                batch.total_audio_duration += duration

                if merge_short and duration < merge_threshold_s:
                    short_files.append((str(fpath), audio_array))
                else:
                    long_files.append((str(fpath), audio_array))
            except Exception as e:
                batch.errors[str(fpath)] = str(e)
                batch.failed += 1

        # Process merged short files
        if short_files:
            logger.info(
                "Merging %d short audio files "
                "(threshold=%ss)...",
                len(short_files), merge_threshold_s,
            )
            # Group into chunks that won't exceed a reasonable total length
            max_merge_duration = 60.0  # 1 minute per merged batch
            current_batch: List[Tuple[str, np.ndarray]] = []
            current_duration = 0.0

            for fpath, audio_array in short_files:
                dur = len(audio_array) / 16000
                if current_duration + dur > max_merge_duration and current_batch:
                    # Process current batch
                    self._process_merged_batch(
                        current_batch, batch, effective_language, **kwargs
                    )
                    current_batch = []
                    current_duration = 0.0
                current_batch.append((fpath, audio_array))
                current_duration += dur

            if current_batch:
                self._process_merged_batch(
                    current_batch, batch, effective_language, **kwargs
                )

        # Process long files (optionally in parallel)
        if long_files:
            if max_workers > 1 and len(long_files) > 1:
                self._process_parallel(
                    long_files, batch, effective_language, max_workers, **kwargs
                )
            else:
                for fpath, audio_array in long_files:
                    try:
                        result = self.transcribe(
                            audio_array, language=effective_language, **kwargs
                        )
                        batch.results.append(result)
                        batch.successful += 1
                    except Exception as e:
                        batch.errors[fpath] = str(e)
                        batch.failed += 1

        batch.total_processing_time = time.perf_counter() - t0
        logger.info(
            "Batch complete: %d/%d succeeded "
            "(%.2fs, avg RTF=%.3f)",
            batch.successful, batch.total_files,
            batch.total_processing_time, batch.avg_rtf,
        )
        return batch

    def _process_merged_batch(
        self,
        batch_files: List[Tuple[str, np.ndarray]],
        batch_result: BatchResult,
        language: Optional[str],
        **kwargs,
    ) -> None:
        """Concatenate short audio files and transcribe as one."""
        try:
            # Concatenate with 0.5s silence gap between files
            gap = np.zeros(int(0.5 * 16000), dtype=np.float32)
            parts = []
            for _, audio in batch_files:
                parts.append(audio)
                parts.append(gap)
            merged = np.concatenate(parts[:-1])  # drop last gap

            result = self.transcribe(merged, language=language, **kwargs)

            # Split result back into individual file results
            # Use duration-based segment assignment
            file_durations = [
                len(audio) / 16000 for _, audio in batch_files
            ]
            split_results = self._split_result_by_duration(
                result, file_durations
            )

            for i, (fpath, _) in enumerate(batch_files):
                if i < len(split_results):
                    batch_result.results.append(split_results[i])
                else:
                    batch_result.results.append(
                        WhisperResult(text="", language=result.language)
                    )
                batch_result.successful += 1

        except Exception as e:
            logger.warning(f"Merged batch failed: {e}, falling back to individual")
            for fpath, audio in batch_files:
                try:
                    result = self.transcribe(audio, language=language, **kwargs)
                    batch_result.results.append(result)
                    batch_result.successful += 1
                except Exception as e2:
                    batch_result.errors[fpath] = str(e2)
                    batch_result.failed += 1

    def _split_result_by_duration(
        self,
        result: WhisperResult,
        file_durations: List[float],
    ) -> List[WhisperResult]:
        """Split a merged transcription result by expected durations."""
        if not result.segments:
            return [result] * len(file_durations)

        split_results = []
        offset = 0.0
        gap = 0.5  # silence gap between merged files

        for i, dur in enumerate(file_durations):
            start_time = offset
            end_time = offset + dur

            file_segments = [
                seg for seg in result.segments
                if seg.get("start", 0) >= start_time
                and seg.get("start", 0) < end_time
            ]

            # Adjust segment times relative to file start
            adjusted_segments = []
            for seg in file_segments:
                adj = dict(seg)
                adj["start"] = max(0, round(seg["start"] - offset, 3))
                adj["end"] = round(min(dur, seg["end"] - offset), 3)
                adjusted_segments.append(adj)

            file_text = " ".join(seg.get("text", "").strip() for seg in adjusted_segments)

            file_confidences = [
                s["confidence"] for s in adjusted_segments
                if s.get("confidence") is not None
            ]
            avg_conf = (
                sum(file_confidences) / len(file_confidences)
                if file_confidences else 0.0
            )

            split_results.append(WhisperResult(
                text=file_text.strip(),
                language=result.language,
                confidence=avg_conf,
                segments=adjusted_segments,
            ))

            offset += dur + gap

        return split_results

    def _process_parallel(
        self,
        files: List[Tuple[str, np.ndarray]],
        batch_result: BatchResult,
        language: Optional[str],
        max_workers: int,
        **kwargs,
    ) -> None:
        """Process long audio files in parallel."""

        def _transcribe_one(item: Tuple[str, np.ndarray]) -> Tuple[str, Optional[WhisperResult], Optional[str]]:
            fpath, audio = item
            try:
                result = self.transcribe(audio, language=language, **kwargs)
                return fpath, result, None
            except Exception as e:
                return fpath, None, str(e)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_transcribe_one, item): item
                for item in files
            }
            for future in as_completed(futures):
                fpath, result, error = future.result()
                if result:
                    batch_result.results.append(result)
                    batch_result.successful += 1
                else:
                    batch_result.errors[fpath] = error or "Unknown error"
                    batch_result.failed += 1

    # ------------------------------------------------------------------
    # 4. Streaming Transcription
    # ------------------------------------------------------------------

    def transcribe_streaming(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: int = 16000,
        chunk_duration_s: float = 30.0,
        overlap_s: float = 2.0,
        language: Optional[str] = None,
        **kwargs,
    ) -> Generator[StreamChunk, None, None]:
        """
        Transcribe long audio with streaming output.

        Splits audio into overlapping chunks and yields partial results
        as each chunk completes. Overlap ensures no words are lost at
        chunk boundaries.

        Args:
            audio:           File path or numpy array.
            sample_rate:     Sample rate.
            chunk_duration_s: Duration of each chunk in seconds.
            overlap_s:       Overlap between consecutive chunks.
            language:        Override language.
            **kwargs:        Additional transcription options.

        Yields:
            StreamChunk with partial transcription text.
        """
        audio_array = self._prepare_audio(audio, sample_rate)
        total_samples = len(audio_array)
        chunk_samples = int(chunk_duration_s * sample_rate)
        overlap_samples = int(overlap_s * sample_rate)
        step_samples = chunk_samples - overlap_samples

        effective_language = language or self.language
        chunk_index = 0

        _total_s = total_samples / sample_rate
        logger.info(
            "Streaming transcription: %.1fs audio, "
            "chunk=%ss, overlap=%ss",
            _total_s, chunk_duration_s, overlap_s,
        )

        pos = 0
        while pos < total_samples:
            end = min(pos + chunk_samples, total_samples)
            chunk_audio = audio_array[pos:end]
            chunk_start_time = pos / sample_rate
            chunk_end_time = end / sample_rate

            result = self._transcribe_internal(
                chunk_audio, sample_rate,
                language=effective_language, **kwargs
            )

            # Filter segments that fall in the overlap region
            # (only yield segments past the overlap to avoid duplicates)
            meaningful_text = result.text.strip()
            if chunk_index > 0 and overlap_s > 0:
                # Skip text from the overlap region to avoid duplication
                overlap_boundary = overlap_s
                filtered_segments = [
                    seg for seg in result.segments
                    if seg.get("start", 0) >= overlap_boundary
                ]
                meaningful_text = " ".join(
                    seg.get("text", "").strip() for seg in filtered_segments
                )

            is_final = end >= total_samples

            yield StreamChunk(
                text=meaningful_text,
                start=chunk_start_time,
                end=chunk_end_time,
                is_final=is_final,
                chunk_index=chunk_index,
            )

            chunk_index += 1
            pos += step_samples

            if end >= total_samples:
                break

    # ------------------------------------------------------------------
    # 5. GPU VRAM Optimization
    # ------------------------------------------------------------------

    def set_quantization(self, preset: str) -> None:
        """
        Set quantization level for the model.

        Presets:
          'int8'         - 8-bit integer, 2x smaller, ~1% accuracy loss
          'int8_float16' - 8-bit weights, float16 compute (GPU only)
          'float16'      - Half precision (GPU only, default)
          'float32'      - Full precision (slowest, most accurate)
          'int16'        - 16-bit integer

        Args:
            preset: Quantization preset name.
        """
        if preset not in self.QUANTIZE_PRESETS:
            raise ValueError(
                f"Unknown quantization preset '{preset}'. "
                f"Available: {list(self.QUANTIZE_PRESETS.keys())}"
            )

        new_compute_type = self.QUANTIZE_PRESETS[preset]["compute_type"]

        if self.device == "cpu" and new_compute_type in ("float16", "int8_float16"):
            logger.warning(
                f"Preset '{preset}' requires GPU. Using 'int8' for CPU."
            )
            new_compute_type = "int8"

        if new_compute_type != self.compute_type:
            logger.info("Quantization: %s -> %s", self.compute_type, new_compute_type)
            self.compute_type = new_compute_type
            self._unload_model()

    def release_vram(self) -> Dict[str, float]:
        """
        Release GPU VRAM by unloading the model.
        Returns VRAM stats before and after release.

        Returns:
            Dict with 'before_mb' and 'after_mb' VRAM usage.
        """
        stats = {"before_mb": 0.0, "after_mb": 0.0}

        try:
            import torch
            if torch.cuda.is_available():
                stats["before_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        except ImportError:
            pass

        self._unload_model()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                stats["after_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        except ImportError:
            pass

        freed = stats["before_mb"] - stats["after_mb"]
        logger.info("VRAM released: %.1fMB (%.1f -> %.1fMB)", freed, stats['before_mb'], stats['after_mb'])
        return stats

    def get_vram_usage(self) -> Dict[str, float]:
        """Get current GPU VRAM usage in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return {
                    "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
                    "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 1),
                    "total_mb": round(props.total_memory / 1024 / 1024, 1),
                }
        except (ImportError, AttributeError):
            pass
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0}

    # ------------------------------------------------------------------
    # Internal Model Management
    # ------------------------------------------------------------------

    def _ensure_model(self) -> Any:
        """Get or load the faster-whisper model (thread-safe)."""
        if self._model is not None:
            return self._model

        with self._model_lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return self._model

            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError(
                    "faster-whisper not installed. Install with:\n"
                    "  pip install faster-whisper"
                )

            device = self.device
            compute_type = self.compute_type

            if device == "cpu" and compute_type in ("float16", "int8_float16"):
                compute_type = "int8"
                logger.info("CPU detected, forcing compute_type=int8")

            logger.info(
                "Loading model '%s' "
                "(device=%s, compute_type=%s)...",
                self.model_name, device, compute_type,
            )

            t0 = time.perf_counter()
            try:
                model = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type=compute_type,
                )
            except Exception as e:
                if "CUDA" in str(e) or "gpu" in str(e).lower():
                    logger.warning(f"GPU load failed ({e}), falling back to CPU/int8")
                    device = "cpu"
                    compute_type = "int8"
                    model = WhisperModel(
                        self.model_name, device=device, compute_type=compute_type
                    )
                    self.device = device
                    self.compute_type = compute_type
                else:
                    raise

            elapsed = time.perf_counter() - t0
            logger.info(f"Model loaded in {elapsed:.2f}s")

            self._model = model
            # Sync back actual device/compute_type used (may differ from requested)
            self.device = device
            self.compute_type = compute_type
            return model

    def _unload_model(self) -> None:
        """Unload model from memory."""
        with self._model_lock:
            if self._model is not None:
                del self._model
                self._model = None
                self._warmup_done = False
                logger.info("Model unloaded from memory")

                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

    def _transcribe_internal(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        skip_cache: bool = False,
        **kwargs,
    ) -> WhisperResult:
        """Core transcription using faster-whisper."""
        model = self._ensure_model()

        language = kwargs.pop("language", None)
        task = kwargs.pop("task", "transcribe")
        beam_size = kwargs.pop("beam_size", self.beam_size)
        vad_filter = kwargs.pop("vad_filter", self.vad_filter)

        transcribe_kwargs: Dict[str, Any] = {
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "language": language,
            "task": task,
        }
        transcribe_kwargs.update(kwargs)

        # Write audio to temp WAV file
        import struct

        wav_bytes = self._audio_to_wav_bytes(audio, sample_rate)
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name

            segments_iter, info = model.transcribe(tmp_path, **transcribe_kwargs)

            segments = []
            full_text_parts = []
            for seg in segments_iter:
                segment_dict = {
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                    "confidence": round(1.0 - seg.no_speech_prob, 3)
                    if seg.no_speech_prob is not None else None,
                }
                segments.append(segment_dict)
                full_text_parts.append(seg.text.strip())

            full_text = " ".join(full_text_parts)

            confidences = [
                s["confidence"] for s in segments
                if s.get("confidence") is not None
            ]
            confidence = sum(confidences) / len(confidences) if confidences else 0.0

            detected_lang = getattr(info, "language", language or "unknown")

            return WhisperResult(
                text=full_text,
                language=detected_lang,
                confidence=confidence,
                segments=segments,
                backend="faster-whisper",
                model=self.model_name,
            )

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Audio Preparation Utilities
    # ------------------------------------------------------------------

    def _prepare_audio(
        self,
        audio: Union[str, Path, np.ndarray, bytes],
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Convert various input types to float32 numpy array."""
        if isinstance(audio, np.ndarray):
            arr = audio.astype(np.float32)
            if arr.max() > 1.0 or arr.min() < -1.0:
                arr = arr / 32768.0
            return arr

        if isinstance(audio, bytes):
            return self._load_audio_from_bytes(audio)

        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return self._load_audio_file(str(path))

    @staticmethod
    def _load_audio_file(file_path: str) -> np.ndarray:
        """Load audio file to float32 numpy array using pydub."""
        from pydub import AudioSegment

        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(_DEFAULT_SAMPLE_RATE).set_channels(1).set_sample_width(2)
        samples = np.frombuffer(audio.raw_data, dtype=np.int16)
        return samples.astype(np.float32) / _INT16_MAX

    @staticmethod
    def _load_audio_from_bytes(data: bytes) -> np.ndarray:
        """Load audio from bytes."""
        from pydub import AudioSegment

        audio = AudioSegment.from_file(io.BytesIO(data))
        audio = audio.set_frame_rate(_DEFAULT_SAMPLE_RATE).set_channels(1).set_sample_width(2)
        samples = np.frombuffer(audio.raw_data, dtype=np.int16)
        return samples.astype(np.float32) / _INT16_MAX

    @staticmethod
    def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert float32 numpy array to WAV bytes."""
        import struct

        audio_int16 = (audio * _INT16_CLIP_HIGH).clip(_INT16_CLIP_LOW, _INT16_CLIP_HIGH).astype(np.int16)
        data_bytes = audio_int16.tobytes()

        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(data_bytes)

        buf = io.BytesIO()
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + data_size))
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(struct.pack("<I", 16))
        buf.write(struct.pack("<HHIIHH", 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample))
        buf.write(b"data")
        buf.write(struct.pack("<I", data_size))
        buf.write(data_bytes)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Statistics & Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_rtf = 0.0
        if self._total_audio_duration > 0:
            avg_rtf = self._total_processing_time / self._total_audio_duration

        stats: Dict[str, Any] = {
            "model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
            "warmup_done": self._warmup_done,
            "model_loaded": self._model is not None,
            "transcribe_count": self._transcribe_count,
            "total_audio_duration_s": round(self._total_audio_duration, 2),
            "total_processing_time_s": round(self._total_processing_time, 3),
            "avg_rtf": round(avg_rtf, 4),
        }

        if self._cache:
            cs = self._cache.stats()
            stats["cache"] = {
                "entries": cs.total_entries,
                "hit_count": cs.hit_count,
                "miss_count": cs.miss_count,
                "hit_rate": round(cs.hit_rate, 4),
            }

        vram = self.get_vram_usage()
        if vram["total_mb"] > 0:
            stats["vram"] = vram

        return stats

    def benchmark(
        self,
        audio_path: str,
        iterations: int = 3,
        warmup: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a benchmark on a single audio file.

        Args:
            audio_path: Path to test audio file.
            iterations: Number of runs (first is excluded if warmup=True).
            warmup:     Whether to do a warmup run first.

        Returns:
            Dict with benchmark results.
        """
        from pathlib import Path

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        audio_array = self._prepare_audio(str(path))
        duration = len(audio_array) / 16000

        times: List[float] = []

        if warmup:
            logger.info("Benchmark warmup run...")
            self._transcribe_internal(audio_array, 16000, skip_cache=True)

        logger.info(f"Benchmark: {iterations} iterations on {path.name} ({duration:.1f}s)")
        for i in range(iterations):
            t0 = time.perf_counter()
            result = self._transcribe_internal(audio_array, 16000, skip_cache=True)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            rtf = elapsed / duration
            logger.info(f"  Run {i+1}: {elapsed:.3f}s (RTF={rtf:.3f})")

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        avg_rtf = avg_time / duration

        return {
            "file": str(path),
            "audio_duration_s": round(duration, 2),
            "model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "iterations": iterations,
            "warmup": warmup,
            "avg_time_s": round(avg_time, 3),
            "min_time_s": round(min_time, 3),
            "max_time_s": round(max_time, 3),
            "avg_rtf": round(avg_rtf, 4),
            "throughput_x_realtime": round(1.0 / avg_rtf, 2) if avg_rtf > 0 else 0,
            "result_text_preview": result.text[:200] if result.text else "",
        }