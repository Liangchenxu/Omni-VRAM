"""
Production Monitoring Module for vram_core
===========================================

Metrics collection, Prometheus endpoint, health checks, and Grafana dashboard export.

Usage:
    from vram_core.monitoring import MetricsCollector
    collector = MetricsCollector()
    collector.record_transcription(latency=0.5, success=True)
    print(collector.get_metrics())
"""

import time
import logging
import threading
import json
from contextlib import contextmanager
from typing import Optional, Dict, Any, Iterator, List, Tuple
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

#: Canonical stage order of the end-to-end voice pipeline (v2.7.0).
#: Every timestamp is taken on the same monotonic clock, in microseconds.
PIPELINE_STAGES: Tuple[str, ...] = (
    "vad_cutoff",
    "asr_transcribed",
    "llm_first_token",
    "tts_first_chunk",
)


def _now_us() -> float:
    """Monotonic clock in microseconds (``perf_counter_ns`` based)."""
    return time.perf_counter_ns() / 1000.0


def _percentile(values: List[float], pct: float) -> float:
    """Nearest-rank percentile (0.0 for an empty sample)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    index = max(0, min(len(ordered) - 1, int(round(rank))))
    return float(ordered[index])


@dataclass
class TranscriptionMetric:
    """Single transcription metric."""
    timestamp: float = 0.0
    latency: float = 0.0
    audio_duration: float = 0.0
    success: bool = True
    backend: str = ""
    error: str = ""


@dataclass
class SystemHealth:
    """System health snapshot."""
    status: str = "healthy"
    uptime: float = 0.0
    total_requests: int = 0
    success_rate: float = 100.0
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization: float = 0.0
    active_workers: int = 0
    queue_depth: int = 0
    requests_per_second: float = 0.0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "uptime_seconds": round(self.uptime, 1),
            "total_requests": self.total_requests,
            "success_rate": round(self.success_rate, 2),
            "avg_latency_ms": round(self.avg_latency * 1000, 1),
            "p95_latency_ms": round(self.p95_latency * 1000, 1),
            "p99_latency_ms": round(self.p99_latency * 1000, 1),
            "gpu_memory_used_mb": round(self.gpu_memory_used_mb, 0),
            "gpu_memory_total_mb": round(self.gpu_memory_total_mb, 0),
            "gpu_utilization_pct": round(self.gpu_utilization, 1),
            "active_workers": self.active_workers,
            "queue_depth": self.queue_depth,
            "requests_per_second": round(self.requests_per_second, 2),
            "error_count": self.error_count,
        }


class MetricsCollector:
    """
    Production metrics collector for vram_core.

    Features:
        - Record transcription latency, success/failure
        - Percentile latency (p50, p95, p99)
        - GPU memory monitoring
        - Prometheus text format export
        - Grafana dashboard JSON export
        - Health check endpoint data

    Usage:
        collector = MetricsCollector()

        # Record metrics
        collector.record_transcription(latency=0.5, success=True, backend="faster_whisper")
        collector.record_error("connection_timeout")

        # Get metrics
        health = collector.get_health()
        prometheus_text = collector.export_prometheus()
        grafana_json = collector.export_grafana_dashboard()
    """

    def __init__(self, max_history: int = 10000):
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._max_history = max_history

        # Metrics storage
        self._latencies: deque = deque(maxlen=max_history)
        self._audio_durations: deque = deque(maxlen=max_history)
        self._success_count = 0
        self._failure_count = 0
        self._error_counts: Dict[str, int] = {}
        self._backend_counts: Dict[str, int] = {}

        # Throughput tracking
        self._request_timestamps: deque = deque(maxlen=max_history)

        # Custom gauges
        self._gauges: Dict[str, float] = {}
        self._counters: Dict[str, int] = {}

        logger.info("MetricsCollector initialized")

    def record_transcription(
        self,
        latency: float,
        audio_duration: float = 0.0,
        success: bool = True,
        backend: str = "unknown",
        error: str = "",
    ) -> None:
        """Record a transcription request."""
        with self._lock:
            now = time.time()
            self._latencies.append(latency)
            self._audio_durations.append(audio_duration)
            self._request_timestamps.append(now)

            if success:
                self._success_count += 1
            else:
                self._failure_count += 1
                if error:
                    self._error_counts[error] = self._error_counts.get(error, 0) + 1

            self._backend_counts[backend] = self._backend_counts.get(backend, 0) + 1

    def record_error(self, error_type: str) -> None:
        """Record an error."""
        with self._lock:
            self._failure_count += 1
            self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1

    def set_gauge(self, name: str, value: float) -> None:
        """Set a custom gauge value."""
        with self._lock:
            self._gauges[name] = value

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a custom counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    # 鈹€鈹€ Query Methods 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def get_health(self) -> SystemHealth:
        """Get current system health snapshot."""
        with self._lock:
            total = self._success_count + self._failure_count
            success_rate = (self._success_count / total * 100) if total > 0 else 100.0

            # Latency percentiles
            sorted_lat = sorted(self._latencies) if self._latencies else [0]
            avg_lat = sum(sorted_lat) / len(sorted_lat) if sorted_lat else 0
            p95 = sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) > 1 else sorted_lat[0]
            p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 1 else sorted_lat[0]

            # Throughput (requests in last 60s)
            now = time.time()
            recent = [t for t in self._request_timestamps if now - t < 60]
            rps = len(recent) / 60.0 if recent else 0

            # GPU memory
            gpu_mem_used = 0.0
            gpu_mem_total = 0.0
            gpu_util = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem_used = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    gpu_util = (gpu_mem_used / gpu_mem_total * 100) if gpu_mem_total > 0 else 0
            except ImportError:
                pass

            # Status
            status = "healthy"
            if success_rate < 95:
                status = "degraded"
            if success_rate < 80:
                status = "unhealthy"

            return SystemHealth(
                status=status,
                uptime=now - self._start_time,
                total_requests=total,
                success_rate=success_rate,
                avg_latency=avg_lat,
                p95_latency=p95,
                p99_latency=p99,
                gpu_memory_used_mb=gpu_mem_used,
                gpu_memory_total_mb=gpu_mem_total,
                gpu_utilization=gpu_util,
                requests_per_second=rps,
                error_count=self._failure_count,
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dict."""
        health = self.get_health()
        with self._lock:
            return {
                **health.to_dict(),
                "backend_distribution": dict(self._backend_counts),
                "error_distribution": dict(self._error_counts),
                "custom_gauges": dict(self._gauges),
                "custom_counters": dict(self._counters),
            }

    # 鈹€鈹€ Prometheus Export 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus exposition format string.
        """
        health = self.get_health()
        lines = [
            "# HELP omnivram_requests_total Total transcription requests",
            "# TYPE omnivram_requests_total counter",
            f"omnivram_requests_total {health.total_requests}",
            "",
            "# HELP omnivram_requests_success_total Successful requests",
            "# TYPE omnivram_requests_success_total counter",
            f"omnivram_requests_success_total {self._success_count}",
            "",
            "# HELP omnivram_requests_failed_total Failed requests",
            "# TYPE omnivram_requests_failed_total counter",
            f"omnivram_requests_failed_total {health.error_count}",
            "",
            "# HELP omnivram_latency_seconds Average transcription latency",
            "# TYPE omnivram_latency_seconds gauge",
            f"omnivram_latency_seconds {health.avg_latency}",
            "",
            "# HELP omnivram_latency_p95_seconds 95th percentile latency",
            "# TYPE omnivram_latency_p95_seconds gauge",
            f"omnivram_latency_p95_seconds {health.p95_latency}",
            "",
            "# HELP omnivram_latency_p99_seconds 99th percentile latency",
            "# TYPE omnivram_latency_p99_seconds gauge",
            f"omnivram_latency_p99_seconds {health.p99_latency}",
            "",
            "# HELP omnivram_gpu_memory_used_mb GPU memory used in MB",
            "# TYPE omnivram_gpu_memory_used_mb gauge",
            f"omnivram_gpu_memory_used_mb {health.gpu_memory_used_mb}",
            "",
            "# HELP omnivram_gpu_utilization_pct GPU utilization percentage",
            "# TYPE omnivram_gpu_utilization_pct gauge",
            f"omnivram_gpu_utilization_pct {health.gpu_utilization}",
            "",
            "# HELP omnivram_requests_per_second Current throughput",
            "# TYPE omnivram_requests_per_second gauge",
            f"omnivram_requests_per_second {health.requests_per_second}",
            "",
            "# HELP omnivram_uptime_seconds Uptime in seconds",
            "# TYPE omnivram_uptime_seconds gauge",
            f"omnivram_uptime_seconds {health.uptime}",
            "",
        ]

        # Backend distribution
        with self._lock:
            for backend, count in self._backend_counts.items():
                lines.append(f'omnivram_backend_requests{{backend="{backend}"}} {count}')

        return "\n".join(lines)

    # 鈹€鈹€ Grafana Dashboard Export 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def export_grafana_dashboard(self) -> Dict[str, Any]:
        """
        Export a Grafana dashboard JSON configuration.

        Returns:
            Grafana dashboard JSON dict.
        """
        return {
            "dashboard": {
                "title": "vram_core Production Dashboard",
                "tags": ["vram_core", "audio", "transcription"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Request Rate (req/s)",
                        "type": "graph",
                        "targets": [{"expr": "omnivram_requests_per_second"}],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    },
                    {
                        "id": 2,
                        "title": "Latency (ms)",
                        "type": "graph",
                        "targets": [
                            {"expr": "omnivram_latency_seconds * 1000", "legendFormat": "avg"},
                            {"expr": "omnivram_latency_p95_seconds * 1000", "legendFormat": "p95"},
                            {"expr": "omnivram_latency_p99_seconds * 1000", "legendFormat": "p99"},
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    },
                    {
                        "id": 3,
                        "title": "GPU Memory (MB)",
                        "type": "gauge",
                        "targets": [{"expr": "omnivram_gpu_memory_used_mb"}],
                        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 8},
                    },
                    {
                        "id": 4,
                        "title": "Success Rate (%)",
                        "type": "gauge",
                        "targets": [{"expr": "omnivram_requests_success_total / omnivram_requests_total * 100"}],
                        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 8},
                    },
                    {
                        "id": 5,
                        "title": "Error Count",
                        "type": "stat",
                        "targets": [{"expr": "omnivram_requests_failed_total"}],
                        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 8},
                    },
                    {
                        "id": 6,
                        "title": "Uptime",
                        "type": "stat",
                        "targets": [{"expr": "omnivram_uptime_seconds"}],
                        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 8},
                    },
                ],
                "refresh": "10s",
                "time": {"from": "now-1h", "to": "now"},
            }
        }

    def save_grafana_dashboard(self, path: str) -> None:
        """Save Grafana dashboard JSON to file."""
        dashboard = self.export_grafana_dashboard()
        Path(path).write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
        logger.info(f"Grafana dashboard saved to {path}")

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._latencies.clear()
            self._audio_durations.clear()
            self._request_timestamps.clear()
            self._success_count = 0
            self._failure_count = 0
            self._error_counts.clear()
            self._backend_counts.clear()
            self._gauges.clear()
            self._counters.clear()
            self._start_time = time.time()
            logger.info("Metrics reset")


@dataclass
class LatencyTrace:
    """
    One end-to-end pipeline run, timestamped with microsecond resolution.

    Timestamps are stored as absolute ``perf_counter_ns`` microseconds; the
    helpers below expose them relative to :attr:`start_us`, which is what a
    waterfall chart needs.
    """

    trace_id: str = ""
    start_us: float = field(default_factory=_now_us)
    marks: Dict[str, float] = field(default_factory=dict)
    completed: bool = False

    # ── Recording ─────────────────────────────────────────────────────────
    def mark(self, stage: str) -> float:
        """
        Timestamp a pipeline stage.

        Returns:
            The stage offset from the trace start, in microseconds.
        """
        self.marks[str(stage)] = _now_us()
        return self.offset_us(stage)

    @contextmanager
    def span(self, stage: str) -> Iterator["LatencyTrace"]:
        """Time a block of code as ``stage`` (marks it on exit)."""
        try:
            yield self
        finally:
            self.mark(stage)

    def offset_us(self, stage: str) -> float:
        """Microseconds between the trace start and ``stage`` (0.0 if unmarked)."""
        if stage not in self.marks:
            return 0.0
        return self.marks[stage] - self.start_us

    def duration_us(self, stage: str) -> float:
        """
        Incremental cost of ``stage``: the gap from the previous marked pipeline
        stage (``vad_cutoff`` is measured from the trace start).
        """
        if stage not in self.marks:
            return 0.0
        order: List[str] = list(PIPELINE_STAGES)
        index = order.index(stage) if stage in order else len(order)
        previous = 0.0
        for earlier in order[:index]:
            if earlier in self.marks:
                previous = self.marks[earlier] - self.start_us
        return self.offset_us(stage) - previous

    # ── Views ─────────────────────────────────────────────────────────────
    @property
    def total_us(self) -> float:
        """End-to-end latency of the trace (µs)."""
        if not self.marks:
            return 0.0
        return max(value - self.start_us for value in self.marks.values())

    def offsets(self) -> Dict[str, float]:
        """Stage name -> offset from the trace start (µs)."""
        return {stage: self.offset_us(stage) for stage in self.marks}

    def stage_durations(self) -> Dict[str, float]:
        """Stage name -> incremental microsecond cost."""
        return {stage: self.duration_us(stage) for stage in self.marks}

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation of the trace."""
        return {
            "trace_id": self.trace_id,
            "completed": self.completed,
            "total_us": round(self.total_us, 2),
            "stages": {
                stage: {
                    "offset_us": round(self.offset_us(stage), 2),
                    "duration_us": round(self.duration_us(stage), 2),
                }
                for stage in self.marks
            },
        }


class LatencyProfiler:
    """
    End-to-end, microsecond-resolution latency profiler for the voice pipeline.

    It follows the canonical stages of a spoken turn::

        vad_cutoff -------- the VAD decided the user stopped speaking
        asr_transcribed --- the final transcript is available
        llm_first_token --- the first token arrived from the LLM
        tts_first_chunk --- the first synthesized audio chunk is ready to play

    Every boundary is captured with ``time.perf_counter_ns`` so the per-stage
    budget of a real-time turn (target: well under a second end-to-end, with the
    ASR/LLM/TTS stages individually in the tens of milliseconds) is measurable
    without a sampling profiler.

    Typical use::

        profiler = LatencyProfiler()
        trace = profiler.start_trace("turn-1")
        trace.mark("vad_cutoff")
        ... asr work ...
        trace.mark("asr_transcribed")
        ... llm work ...
        trace.mark("llm_first_token")
        ... tts work ...
        trace.mark("tts_first_chunk")
        profiler.finish_trace(trace)

        print(profiler.waterfall())
        profiler.export_json("latency.json")

    Args:
        max_history: Completed traces retained for percentile statistics.
        stages: Stage order to use (defaults to :data:`PIPELINE_STAGES`).
        collector: Optional :class:`MetricsCollector`; when given, the stage
            durations of every finished trace are published as gauges
            (``latency.<stage>.us``).
    """

    def __init__(
        self,
        max_history: int = 1000,
        stages: Tuple[str, ...] = PIPELINE_STAGES,
        collector: Optional[MetricsCollector] = None,
    ):
        self.stages: Tuple[str, ...] = tuple(stages)
        self.collector = collector
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=max_history)
        self._active: Dict[str, LatencyTrace] = {}

    # ── Trace lifecycle ───────────────────────────────────────────────────
    def start_trace(self, trace_id: Optional[str] = None) -> LatencyTrace:
        """Begin a new trace and register it as active."""
        trace = LatencyTrace(trace_id=trace_id or f"trace-{time.time_ns()}")
        with self._lock:
            self._active[trace.trace_id] = trace
        return trace

    def trace(self, trace_id: Optional[str] = None):
        """
        Context manager that starts, yields and finishes a trace::

            with profiler.trace("turn-1") as t:
                t.mark("vad_cutoff")
        """
        return _TraceContext(self, trace_id)

    def _current(self, trace_id: Optional[str] = None) -> LatencyTrace:
        with self._lock:
            if trace_id is not None:
                trace = self._active.get(trace_id)
            else:
                trace = next(reversed(self._active.values()), None) if self._active else None
        if trace is None:
            raise RuntimeError(
                "LatencyProfiler has no active trace; call start_trace() first"
            )
        return trace

    def mark(self, stage: str, trace_id: Optional[str] = None) -> float:
        """Timestamp ``stage`` on the active (or named) trace."""
        return self._current(trace_id).mark(stage)

    def finish_trace(self, trace: Optional[LatencyTrace] = None) -> LatencyTrace:
        """
        Complete a trace, store it and publish its stage gauges.

        When ``trace`` is omitted the most recently started active trace is used.
        """
        if trace is None:
            trace = self._current()
        trace.completed = True
        with self._lock:
            self._active.pop(trace.trace_id, None)
            self._history.append(trace)
        if self.collector is not None:
            for stage, duration in trace.stage_durations().items():
                self.collector.set_gauge(f"latency.{stage}.us", duration)
            self.collector.set_gauge("latency.e2e.us", trace.total_us)
        return trace

    # ── History ───────────────────────────────────────────────────────────
    @property
    def traces(self) -> List[LatencyTrace]:
        """Completed traces, oldest first."""
        with self._lock:
            return list(self._history)

    @property
    def last_trace(self) -> Optional[LatencyTrace]:
        """The most recently completed trace, if any."""
        with self._lock:
            return self._history[-1] if self._history else None

    # ── Statistics ────────────────────────────────────────────────────────
    def stage_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Per-stage microsecond statistics over the completed traces.

        Returns:
            ``{stage: {"count", "min", "mean", "p50", "p95", "max"}}`` with every
            value in microseconds.
        """
        traces = self.traces
        stats: Dict[str, Dict[str, float]] = {}
        for stage in self.stages:
            samples = [
                trace.duration_us(stage)
                for trace in traces
                if stage in trace.marks
            ]
            if not samples:
                continue
            stats[stage] = {
                "count": float(len(samples)),
                "min": min(samples),
                "mean": sum(samples) / len(samples),
                "p50": _percentile(samples, 50),
                "p95": _percentile(samples, 95),
                "max": max(samples),
            }
        return stats

    def summary(self) -> Dict[str, Any]:
        """Compact JSON-friendly overview of the profiler state."""
        traces = self.traces
        totals = [trace.total_us for trace in traces]
        return {
            "traces": len(traces),
            "stages": list(self.stages),
            "end_to_end_us": {
                "min": min(totals) if totals else 0.0,
                "mean": (sum(totals) / len(totals)) if totals else 0.0,
                "p95": _percentile(totals, 95),
                "max": max(totals) if totals else 0.0,
            },
            "stage_stats": self.stage_stats(),
        }

    # ── Rendering ─────────────────────────────────────────────────────────
    def waterfall(self, trace: Optional[LatencyTrace] = None, width: int = 32) -> str:
        """
        Render an ASCII waterfall chart of one trace.

        Each row is a pipeline stage: the bar starts at the stage's offset from
        the turn start and its length is that stage's incremental duration, so
        serialisation gaps are visible at a glance.

        Args:
            trace: Trace to render (defaults to the most recent one).
            width: Character width of the timeline.

        Returns:
            Multi-line string, or a placeholder when no trace is available.
        """
        trace = trace or self.last_trace
        if trace is None:
            return "LatencyProfiler: no completed trace"
        width = max(8, int(width))
        total = trace.total_us or 1.0
        lines = [
            f"trace {trace.trace_id} - total {total:.1f} us "
            f"(0 - {total / 1000.0:.3f} ms)"
        ]
        for stage in self.stages:
            if stage not in trace.marks:
                continue
            offset = trace.offset_us(stage)
            duration = trace.duration_us(stage)
            start_col = min(width - 1, int(round(offset / total * width)))
            span_col = max(1, int(round(duration / total * width)))
            span_col = min(span_col, width - start_col)
            bar = [" "] * width
            for column in range(start_col, start_col + span_col):
                bar[column] = "#"
            lines.append(
                f"{stage:<18} |{''.join(bar)}| "
                f"{duration:10.1f} us @{offset:10.1f} us"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot (summary + every completed trace)."""
        return {
            "summary": self.summary(),
            "traces": [trace.to_dict() for trace in self.traces],
        }

    def export_json(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export the profile as JSON.

        Args:
            path: Optional destination; when given the payload is written there.

        Returns:
            The exported payload (also returned when nothing is written).
        """
        payload = self.to_dict()
        if path:
            Path(path).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("Latency profile exported to %s", path)
        return payload

    def reset(self) -> None:
        """Drop the history and any active traces."""
        with self._lock:
            self._history.clear()
            self._active.clear()
        logger.info("LatencyProfiler reset")


class _TraceContext:
    """Helper implementing ``LatencyProfiler.trace()`` as a context manager."""

    def __init__(self, profiler: LatencyProfiler, trace_id: Optional[str]):
        self._profiler = profiler
        self._trace_id = trace_id
        self.trace: Optional[LatencyTrace] = None

    def __enter__(self) -> LatencyTrace:
        self.trace = self._profiler.start_trace(self._trace_id)
        return self.trace

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._profiler.finish_trace(self.trace)
        return False


# Health check HTTP handler (for integration with FastAPI/Flask)
def create_health_endpoint(collector: MetricsCollector) -> Dict[str, Any]:
    """
    Create a health check response dict.

    Returns:
        Dict with health status for HTTP response.
    """
    health = collector.get_health()
    status_code = 200 if health.status == "healthy" else 503
    return {
        "status_code": status_code,
        "body": health.to_dict(),
    }