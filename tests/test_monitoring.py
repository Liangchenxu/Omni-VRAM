"""
Tests for vram_core.monitoring module.

Covers:
    - MetricsCollector initialization, recording, querying
    - TranscriptionMetric / SystemHealth data classes
    - Prometheus export format
    - Grafana dashboard export
    - Health endpoint helper
    - Thread safety, reset, edge cases
"""

import pytest
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from vram_core.monitoring import (
    TranscriptionMetric,
    SystemHealth,
    MetricsCollector,
    create_health_endpoint,
)


class TestTranscriptionMetric:
    """Test TranscriptionMetric data class."""

    def test_default_values(self):
        m = TranscriptionMetric()
        assert m.timestamp == 0.0
        assert m.latency == 0.0
        assert m.success
        assert m.backend == ""

    def test_custom_values(self):
        m = TranscriptionMetric(
            timestamp=100.0, latency=0.5, audio_duration=10.0,
            success=False, backend="faster_whisper", error="timeout",
        )
        assert not m.success
        assert m.backend == "faster_whisper"


class TestSystemHealth:
    """Test SystemHealth data class."""

    def test_default_values(self):
        h = SystemHealth()
        assert h.status == "healthy"
        assert h.total_requests == 0
        assert h.success_rate == 100.0

    def test_to_dict(self):
        h = SystemHealth(
            status="healthy", uptime=120.5, total_requests=100,
            success_rate=99.0, avg_latency=0.3, p95_latency=0.8,
            p99_latency=1.2, gpu_memory_used_mb=2048.0,
            gpu_memory_total_mb=8192.0, gpu_utilization=25.0,
            active_workers=4, queue_depth=2, requests_per_second=5.5,
            error_count=1,
        )
        d = h.to_dict()
        assert d["status"] == "healthy"
        assert d["total_requests"] == 100
        assert "avg_latency_ms" in d
        assert "gpu_utilization_pct" in d


class TestMetricsCollector:
    """Test MetricsCollector core functionality."""

    def test_init_default(self):
        collector = MetricsCollector()
        assert collector._success_count == 0
        assert collector._failure_count == 0

    def test_init_custom_max_history(self):
        collector = MetricsCollector(max_history=500)
        assert collector._max_history == 500

    def test_record_transcription_success(self):
        collector = MetricsCollector()
        collector.record_transcription(latency=0.5, success=True, backend="faster_whisper")
        assert collector._success_count == 1
        assert collector._failure_count == 0
        assert "faster_whisper" in collector._backend_counts

    def test_record_transcription_failure(self):
        collector = MetricsCollector()
        collector.record_transcription(
            latency=1.0, success=False, backend="openai", error="timeout"
        )
        assert collector._failure_count == 1
        assert collector._error_counts["timeout"] == 1

    def test_record_multiple_transcriptions(self):
        collector = MetricsCollector()
        for i in range(10):
            collector.record_transcription(latency=0.1 * i, success=i % 3 != 0)
        assert collector._success_count + collector._failure_count == 10

    def test_record_error(self):
        collector = MetricsCollector()
        collector.record_error("connection_refused")
        collector.record_error("connection_refused")
        assert collector._failure_count == 2
        assert collector._error_counts["connection_refused"] == 2

    def test_set_gauge(self):
        collector = MetricsCollector()
        collector.set_gauge("gpu_temp", 72.5)
        assert collector._gauges["gpu_temp"] == 72.5

    def test_set_gauge_overwrite(self):
        collector = MetricsCollector()
        collector.set_gauge("cpu", 50.0)
        collector.set_gauge("cpu", 80.0)
        assert collector._gauges["cpu"] == 80.0

    def test_increment_counter(self):
        collector = MetricsCollector()
        collector.increment_counter("requests")
        collector.increment_counter("requests", value=5)
        assert collector._counters["requests"] == 6

    def test_get_health_no_data(self):
        """Health with no data returns healthy with 100% success rate."""
        collector = MetricsCollector()
        health = collector.get_health()
        assert health.status == "healthy"
        assert health.success_rate == 100.0
        assert health.total_requests == 0

    def test_get_health_healthy(self):
        """High success rate returns healthy status."""
        collector = MetricsCollector()
        for _ in range(100):
            collector.record_transcription(latency=0.1, success=True)
        health = collector.get_health()
        assert health.status == "healthy"
        assert health.success_rate == 100.0

    def test_get_health_degraded(self):
        """Success rate < 95% returns degraded."""
        collector = MetricsCollector()
        for i in range(100):
            collector.record_transcription(latency=0.1, success=i < 90)
        health = collector.get_health()
        assert health.status == "degraded"

    def test_get_health_unhealthy(self):
        """Success rate < 80% returns unhealthy."""
        collector = MetricsCollector()
        for i in range(100):
            collector.record_transcription(latency=0.1, success=i < 50)
        health = collector.get_health()
        assert health.status == "unhealthy"

    def test_latency_percentiles(self):
        """Percentile latencies are computed correctly."""
        collector = MetricsCollector()
        for i in range(100):
            collector.record_transcription(latency=i * 0.01, success=True)
        health = collector.get_health()
        assert health.avg_latency > 0
        assert health.p95_latency >= health.avg_latency
        assert health.p99_latency >= health.p95_latency

    def test_get_metrics_includes_all_sections(self):
        """get_metrics returns all expected sections."""
        collector = MetricsCollector()
        collector.record_transcription(latency=0.5, success=True, backend="fw")
        collector.set_gauge("test_gauge", 1.0)
        collector.increment_counter("test_counter")
        metrics = collector.get_metrics()
        assert "status" in metrics
        assert "backend_distribution" in metrics
        assert "error_distribution" in metrics
        assert "custom_gauges" in metrics
        assert "custom_counters" in metrics
        assert metrics["custom_gauges"]["test_gauge"] == 1.0

    def test_export_prometheus_format(self):
        """Prometheus export contains expected metric names."""
        collector = MetricsCollector()
        collector.record_transcription(latency=0.3, success=True, backend="faster_whisper")
        prom = collector.export_prometheus()
        assert "omnivram_requests_total" in prom
        assert "omnivram_latency_seconds" in prom
        assert "omnivram_gpu_memory_used_mb" in prom
        assert "# HELP" in prom
        assert "# TYPE" in prom

    def test_export_prometheus_backend_labels(self):
        """Prometheus export includes backend labels."""
        collector = MetricsCollector()
        collector.record_transcription(latency=0.1, success=True, backend="faster_whisper")
        collector.record_transcription(latency=0.2, success=True, backend="openai_api")
        prom = collector.export_prometheus()
        assert 'backend="faster_whisper"' in prom
        assert 'backend="openai_api"' in prom

    def test_export_grafana_dashboard_structure(self):
        """Grafana dashboard has expected structure."""
        collector = MetricsCollector()
        dashboard = collector.export_grafana_dashboard()
        assert "dashboard" in dashboard
        assert dashboard["dashboard"]["title"] == "vram_core Production Dashboard"
        panels = dashboard["dashboard"]["panels"]
        assert len(panels) > 0

    def test_save_grafana_dashboard(self):
        """Dashboard JSON can be saved to file."""
        collector = MetricsCollector()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "dashboard.json")
            collector.save_grafana_dashboard(path)
            assert Path(path).exists()
            import json
            data = json.loads(Path(path).read_text())
            assert "dashboard" in data

    def test_reset_clears_all(self):
        """reset() clears all metrics."""
        collector = MetricsCollector()
        collector.record_transcription(latency=0.5, success=True)
        collector.record_error("test")
        collector.set_gauge("g", 1.0)
        collector.increment_counter("c")
        collector.reset()
        assert collector._success_count == 0
        assert collector._failure_count == 0
        assert len(collector._latencies) == 0
        assert len(collector._gauges) == 0
        assert len(collector._counters) == 0

    def test_health_endpoint_healthy(self):
        """create_health_endpoint returns 200 for healthy."""
        collector = MetricsCollector()
        for _ in range(10):
            collector.record_transcription(latency=0.1, success=True)
        endpoint = create_health_endpoint(collector)
        assert endpoint["status_code"] == 200
        assert endpoint["body"]["status"] == "healthy"

    def test_health_endpoint_unhealthy(self):
        """create_health_endpoint returns 503 for unhealthy."""
        collector = MetricsCollector()
        for i in range(100):
            collector.record_transcription(latency=0.1, success=i < 30)
        endpoint = create_health_endpoint(collector)
        assert endpoint["status_code"] == 503

    def test_thread_safety(self):
        """Concurrent recording does not crash."""
        import threading
        collector = MetricsCollector()
        errors = []

        def record_batch(n):
            try:
                for _ in range(n):
                    collector.record_transcription(latency=0.01, success=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_batch, args=(100,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert collector._success_count == 400

    def test_uptime_increases(self):
        """Uptime increases over time."""
        collector = MetricsCollector()
        h1 = collector.get_health()
        time.sleep(0.05)
        h2 = collector.get_health()
        assert h2.uptime > h1.uptime


