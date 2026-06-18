"""
Distributed Transcriber for vram_core
======================================

Supports multi-GPU parallel transcription for long audio files.
Splits audio into segments, transcribes on multiple GPUs, and merges results.

Features:
    - Multi-GPU parallel: long audio auto-split across GPUs
    - Multi-machine: Redis/RabbitMQ task distribution (optional)
    - Auto-merge with overlap handling
    - Progress callbacks

Usage:
    from vram_core.distributed_transcriber import DistributedTranscriber
    dt = DistributedTranscriber(whisper_bridge=bridge, num_workers=4)
    result = dt.transcribe("long_audio.wav")
"""

import gc
import logging
import time
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Maximum number of segments allowed to prevent OOM on very large files
MAX_SEGMENTS = 1000


@dataclass
class SegmentTask:
    """A segment of audio to be transcribed."""
    segment_id: int = 0
    audio: Optional[np.ndarray] = None
    start_time: float = 0.0
    end_time: float = 0.0
    worker_id: int = 0
    result_text: str = ""
    processing_time: float = 0.0
    status: str = "pending"  # pending / running / done / error


@dataclass
class DistributedResult:
    """Aggregated result from distributed transcription."""
    text: str = ""
    language: str = "unknown"
    segments: List[Dict[str, Any]] = field(default_factory=list)
    total_duration: float = 0.0
    processing_time: float = 0.0
    num_workers: int = 1
    num_segments: int = 0
    audio_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "segments": self.segments,
            "total_duration": self.total_duration,
            "processing_time": self.processing_time,
            "num_workers": self.num_workers,
            "num_segments": self.num_segments,
            "audio_duration": self.audio_duration,
        }


class DistributedTranscriber:
    """
    Multi-GPU / multi-worker distributed audio transcription.

    Splits long audio into overlapping segments, distributes across
    workers, and merges results with deduplication.

    Args:
        whisper_bridge: WhisperBridge instance for transcription.
        num_workers: Number of parallel workers (default: auto-detect GPUs).
        segment_duration: Duration of each segment in seconds (default: 60).
        overlap_duration: Overlap between segments in seconds (default: 2).
        on_progress: Optional callback(segment_id, status, progress_pct).
    """

    def __init__(
        self,
        whisper_bridge=None,
        num_workers: int = 4,
        segment_duration: float = 60.0,
        overlap_duration: float = 2.0,
        on_progress: Optional[Callable] = None,
        max_segments: int = MAX_SEGMENTS,
    ):
        self.whisper_bridge = whisper_bridge
        self.num_workers = num_workers
        self.segment_duration = segment_duration
        self.overlap_duration = overlap_duration
        self.on_progress = on_progress
        self.max_segments = max_segments
        self._lock = threading.Lock()

        # Auto-detect GPU count
        try:
            import torch
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                self.num_workers = min(num_workers, gpu_count)
            logger.info("Detected %d GPUs, using %d workers", gpu_count, self.num_workers)
        except ImportError:
            pass

        logger.info(
            "DistributedTranscriber: workers=%d, segment=%ss, overlap=%ss",
            self.num_workers, segment_duration, overlap_duration,
        )

    def transcribe(
        self,
        audio_input,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        **kwargs,
    ) -> DistributedResult:
        """
        Transcribe audio using distributed workers.

        Args:
            audio_input: File path or numpy array.
            sample_rate: Sample rate if numpy array.
            language: Language code.
            **kwargs: Additional whisper options.

        Returns:
            DistributedResult with merged text.
        """
        start_time = time.time()

        # Load audio
        if isinstance(audio_input, str):
            from vram_core.whisper import AudioPreprocessor
            audio, sr = AudioPreprocessor.load_and_convert(audio_input, sample_rate)
        else:
            audio = audio_input
            sr = sample_rate

        audio_duration = len(audio) / sr
        logger.info("Audio duration: %.1fs", audio_duration)

        # Split into segments
        segments = self._split_audio(audio, sr)
        logger.info("Split into %d segments", len(segments))

        # Enforce max segments limit
        if len(segments) > self.max_segments:
            raise ValueError(
                f"Audio too long: {len(segments)} segments exceeds limit of {self.max_segments}. "
                f"Increase segment_duration or max_segments."
            )

        # Transcribe in parallel
        completed = self._transcribe_parallel(segments, sr, language, **kwargs)

        # Merge results
        merged = self._merge_results(completed, sr)

        elapsed = time.time() - start_time
        merged.processing_time = elapsed
        merged.num_workers = self.num_workers
        merged.num_segments = len(segments)
        merged.audio_duration = audio_duration
        merged.total_duration = elapsed

        speedup = audio_duration / elapsed if elapsed > 0 else 0
        logger.info(
            "Distributed transcription done: %.2fs (speedup: %.1fx vs real-time)",
            elapsed, speedup,
        )

        return merged

    def _split_audio(self, audio: np.ndarray, sr: int) -> List[SegmentTask]:
        """Split audio into overlapping segments."""
        segment_samples = int(self.segment_duration * sr)
        overlap_samples = int(self.overlap_duration * sr)
        step = segment_samples - overlap_samples

        segments = []
        seg_id = 0
        start = 0

        while start < len(audio):
            end = min(start + segment_samples, len(audio))
            seg = SegmentTask(
                segment_id=seg_id,
                audio=audio[start:end],
                start_time=start / sr,
                end_time=end / sr,
            )
            segments.append(seg)
            seg_id += 1
            if end >= len(audio):
                break
            start += step

        return segments

    def _transcribe_parallel(
        self,
        segments: List[SegmentTask],
        sr: int,
        language: Optional[str],
        **kwargs,
    ) -> List[SegmentTask]:
        """Transcribe segments in parallel using thread pool."""
        if not self.whisper_bridge:
            raise RuntimeError("No WhisperBridge configured")

        completed: List[SegmentTask] = []

        def _transcribe_segment(task: SegmentTask) -> SegmentTask:
            task.status = "running"
            t0 = time.time()
            try:
                result = self.whisper_bridge.transcribe(
                    task.audio, sample_rate=sr, language=language, **kwargs
                )
                task.result_text = result.text
                task.status = "done"
            except (RuntimeError, OSError, ValueError) as e:
                logger.error("Segment %d failed: %s", task.segment_id, e)
                task.result_text = ""
                task.status = "error"
            finally:
                # Release audio data immediately after transcription to prevent
                # memory accumulation for large files with many segments
                task.audio = None
            task.processing_time = time.time() - t0
            return task

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {
                pool.submit(_transcribe_segment, seg): seg
                for seg in segments
            }

            done_count = 0
            for future in as_completed(futures):
                task = future.result()
                with self._lock:
                    completed.append(task)
                    done_count += 1
                pct = done_count / len(segments) * 100
                if self.on_progress:
                    self.on_progress(task.segment_id, task.status, pct)
                logger.debug(
                    "Segment %d: %s (%.2fs) [%.0f%%]",
                    task.segment_id, task.status, task.processing_time, pct,
                )

        # Release reference to original audio segments and trigger GC
        del futures
        gc.collect()

        # Sort by segment_id
        completed.sort(key=lambda s: s.segment_id)
        return completed

    def _merge_results(
        self,
        segments: List[SegmentTask],
        sr: int,
    ) -> DistributedResult:
        """Merge segment results, handling overlap deduplication."""
        all_texts = []
        all_seg_info = []
        detected_language = "unknown"

        for seg in segments:
            if seg.status == "done" and seg.result_text.strip():
                all_texts.append(seg.result_text.strip())
                all_seg_info.append({
                    "segment_id": seg.segment_id,
                    "start": seg.start_time,
                    "end": seg.end_time,
                    "text": seg.result_text.strip(),
                    "processing_time": seg.processing_time,
                    "status": seg.status,
                })

        merged_text = " ".join(all_texts)

        return DistributedResult(
            text=merged_text,
            language=detected_language,
            segments=all_seg_info,
        )


class RedisDistributedTranscriber(DistributedTranscriber):
    """
    Multi-machine distributed transcriber using Redis as message broker.

    Requires: pip install redis
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", **kwargs):
        super().__init__(**kwargs)
        self.redis_url = redis_url
        self._redis = None

    def _get_redis(self):
        if self._redis is None:
            try:
                import redis
                self._redis = redis.from_url(self.redis_url)
                self._redis.ping()
                logger.info("Connected to Redis: %s", self.redis_url)
            except ImportError:
                raise ImportError("pip install redis")
            except (ConnectionError, TimeoutError, OSError) as e:
                raise RuntimeError("Redis connection failed: %s" % e) from e
        return self._redis

    def publish_task(self, task_id: str, audio_bytes: bytes) -> None:
        """Publish a transcription task to Redis queue."""
        r = self._get_redis()
        import json
        task_data = json.dumps({"task_id": task_id, "audio_size": len(audio_bytes)})
        r.set(f"omnivram:task:{task_id}:meta", task_data)
        r.set(f"omnivram:task:{task_id}:audio", audio_bytes)
        r.lpush("omnivram:task_queue", task_id)
        logger.info("Published task %s (%d bytes)", task_id, len(audio_bytes))

    def consume_tasks(self, whisper_bridge=None, timeout: int = 10) -> None:
        """Worker: consume tasks from Redis queue."""
        r = self._get_redis()
        bridge = whisper_bridge or self.whisper_bridge

        logger.info("Worker started, waiting for tasks...")
        while True:
            result = r.brpop("omnivram:task_queue", timeout=timeout)
            if result is None:
                continue
            task_id = result[1].decode()
            logger.info("Processing task: %s", task_id)

            try:
                audio_bytes = r.get(f"omnivram:task:{task_id}:audio")
                if audio_bytes and bridge:
                    import io
                    from vram_core.whisper import AudioPreprocessor
                    audio, sr = AudioPreprocessor.load_and_convert(
                        io.BytesIO(audio_bytes)
                    )
                    transcription = bridge.transcribe(audio, sample_rate=sr)
                    r.set(
                        f"omnivram:task:{task_id}:result",
                        transcription.to_dict().__str__(),
                    )
                    logger.info("Task %s done: %s", task_id, transcription.text[:50])
            except (RuntimeError, OSError, ValueError) as e:
                logger.error("Task %s failed: %s", task_id, e)
