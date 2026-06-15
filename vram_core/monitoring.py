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
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


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