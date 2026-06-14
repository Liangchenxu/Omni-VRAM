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

import unittest
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


class TestTranscriptionMetric(unittest.TestCase):
    """Test TranscriptionMetric data class."""

    def test_default_values(self):
        m = TranscriptionMetric()
        self.assertEqual(m.timestamp, 0.0)
        self.assertEqual(m.latency, 0.0)
        self.assertTrue(m.success)
        self.assertEqual(m.backend, "")

    def test_custom_values(self):
        m = TranscriptionMetric(
            timestamp=100.0, latency=0.5, audio_duration=10.0,
            success=False, backend="faster_whisper", error="timeout",
        )
        self.assertFalse(m.success)
        self.assertEqual(m.backend, "faster_whisper")


class TestSystemHealth(unittest.TestCase):
    """Test SystemHealth data class."""

    def test_default_values(self):
        h = SystemHealth()
        self.assertEqual(h.status, "healthy")
        self.assertEqual(h.total_requests, 0)
        self.assertEqual(h.success_rate, 100.0)

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
        self.assertEqual(d["status"], "healthy")
        self.assertEqual(d["total_requests"], 100)
        self.assertIn("avg_latency_ms", d)
        self.assertIn("gpu_utilization_pct", d)


class TestMetricsCollector(unittest.TestCase):
    """Test MetricsCollector core functionality."""

    def test_init_default(self):
        collector = MetricsCollector()
        self.assertEqual(collector._success_count, 0)
        self.assertEqual(collector._failure_count, 0)

    def test_init_custom_max_history(self):
        collector = MetricsCollector(max_history=500)
        self.assertEqual(collector._max_history, 500)

    def test_record_transcription_success(self):
        collector = MetricsCollector()
        collector.record_transcription(latency=0.5, success=True, backend="faster_whisper")
        self.assertEqual(collector._success_count, 1)
        self.assertEqual(collector._failure_count, 0)
        self.assertIn("faster_whisper", collector._backend_counts)

    def test_record_transcription_failure(self):
        collector = MetricsCollector()
        collector.record_transcription(
            latency=1.0, success=False, backend="openai", error="timeout"
        )
        self.assertEqual(collector._failure_count, 1)
        self.assertEqual(collector._error_counts["timeout"], 1)

    def test_record_multiple_transcriptions(self):
        collector = MetricsCollector()
        for i in range(10):
            collector.record_transcription(latency=0.1 * i, success=i % 3 != 0)
        self.assertEqual(collector._success_count + collector._failure_count, 10)

    def test_record_error(self):
        collector = MetricsCollector()
        collector.record_error("connection_refused")
        collector.record_error("connection_refused")
        self.assertEqual(collector._failure_count, 2)
        self.assertEqual(collector._error_counts["connection_refused"], 2)

    def test_set_gauge(self):
        collector = MetricsCollector()
        collector.set_gauge("gpu_temp", 72.5)
        self.assertEqual(collector._gauges["gpu_temp"], 72.5)

    def test_set_gauge_overwrite(self):
        collector = MetricsCollector()
        collector.set_gauge("cpu", 50.0)
        collector.set_gauge("cpu", 80.0)
        self.assertEqual(collector._gauges["cpu"], 80.0)

    def test_increment_counter(self):
        collector = MetricsCollector()
        collector.increment_counter("requests")
        collector.increment_counter("requests", value=5)
        self.assertEqual(collector._counters["requests"], 6)

    def test_get_health_no_data(self):
        """Health with no data returns healthy with 100% success rate."""
        collector = MetricsCollector()
        health = collector.get_health()
        self.assertEqual(health.status, "healthy")
        self.assertEqual(health.success_rate, 100.0)
        self.assertEqual(health.total_requests, 0)

    def test_get_health_healthy(self):
        """High success rate returns healthy status."""
        collector = MetricsCollector()
        for _ in range(100):
            collector.record_transcription(latency=0.1, success=True)
        health = collector.get_health()
        self.assertEqual(health.status, "healthy")
        self.assertEqual(health.success_rate, 100.0)

    def test_get_health_degraded(self):
        """Success rate < 95% returns degraded."""
        collector = MetricsCollector()
        for i in range(100):
            collector.record_transcription(latency=0.1, success=i < 90)
        health = collector.get_health()
        self.assertEqual(health.status, "degraded")

    def test_get_health_unhealthy(self):
        """Success rate < 80% returns unhealthy."""
        collector = MetricsCollector()
        for i in range(100):
            collector.record_transcription(latency=0.1, success=i < 50)
        health = collector.get_health()
        self.assertEqual(health.status, "unhealthy")

    def test_latency_percentiles(self):
        """Percentile latencies are computed correctly."""
        collector = MetricsCollector()
        for i in range(100):
            collector.record_transcription(latency=i * 0.01, success=True)
        health = collector.get_health()
        self.assertGreater(health.avg_latency, 0)
        self.assertGreaterEqual(health.p95_latency, health.avg_latency)
        self.assertGreaterEqual(health.p99_latency, health.p95_latency)

    def test_get_metrics_includes_all_sections(self):
        """get_metrics returns all expected sections."""
        collector = MetricsCollector()
        collector.record_transcription(latency=0.5, success=True, backend="fw")
        collector.set_gauge("test_gauge", 1.0)
        collector.increment_counter("test_counter")
        metrics = collector.get_metrics()
        self.assertIn("status", metrics)
        self.assertIn("backend_distribution", metrics)
        self.assertIn("error_distribution", metrics)
        self.assertIn("custom_gauges", metrics)
        self.assertIn("custom_counters", metrics)
        self.assertEqual(metrics["custom_gauges"]["test_gauge"], 1.0)

    def test_export_prometheus_format(self):
        """Prometheus export contains expected metric names."""
        collector = MetricsCollector()
        collector.record_transcription(latency=0.3, success=True, backend="faster_whisper")
        prom = collector.export_prometheus()
        self.assertIn("omnivram_requests_total", prom)
        self.assertIn("omnivram_latency_seconds", prom)
        self.assertIn("omnivram_gpu_memory_used_mb", prom)
        self.assertIn("# HELP", prom)
        self.assertIn("# TYPE", prom)

    def test_export_prometheus_backend_labels(self):
        """Prometheus export includes backend labels."""
        collector = MetricsCollector()
        collector.record_transcription(latency=0.1, success=True, backend="faster_whisper")
        collector.record_transcription(latency=0.2, success=True, backend="openai_api")
        prom = collector.export_prometheus()
        self.assertIn('backend="faster_whisper"', prom)
        self.assertIn('backend="openai_api"', prom)

    def test_export_grafana_dashboard_structure(self):
        """Grafana dashboard has expected structure."""
        collector = MetricsCollector()
        dashboard = collector.export_grafana_dashboard()
        self.assertIn("dashboard", dashboard)
        self.assertEqual(dashboard["dashboard"]["title"], "Omni-VRAM Production Dashboard")
        panels = dashboard["dashboard"]["panels"]
        self.assertGreater(len(panels), 0)

    def test_save_grafana_dashboard(self):
        """Dashboard JSON can be saved to file."""
        collector = MetricsCollector()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "dashboard.json")
            collector.save_grafana_dashboard(path)
            self.assertTrue(Path(path).exists())
            import json
            data = json.loads(Path(path).read_text())
            self.assertIn("dashboard", data)

    def test_reset_clears_all(self):
        """reset() clears all metrics."""
        collector = MetricsCollector()
        collector.record_transcription(latency=0.5, success=True)
        collector.record_error("test")
        collector.set_gauge("g", 1.0)
        collector.increment_counter("c")
        collector.reset()
        self.assertEqual(collector._success_count, 0)
        self.assertEqual(collector._failure_count, 0)
        self.assertEqual(len(collector._latencies), 0)
        self.assertEqual(len(collector._gauges), 0)
        self.assertEqual(len(collector._counters), 0)

    def test_health_endpoint_healthy(self):
        """create_health_endpoint returns 200 for healthy."""
        collector = MetricsCollector()
        for _ in range(10):
            collector.record_transcription(latency=0.1, success=True)
        endpoint = create_health_endpoint(collector)
        self.assertEqual(endpoint["status_code"], 200)
        self.assertEqual(endpoint["body"]["status"], "healthy")

    def test_health_endpoint_unhealthy(self):
        """create_health_endpoint returns 503 for unhealthy."""
        collector = MetricsCollector()
        for i in range(100):
            collector.record_transcription(latency=0.1, success=i < 30)
        endpoint = create_health_endpoint(collector)
        self.assertEqual(endpoint["status_code"], 503)

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
        self.assertEqual(len(errors), 0)
        self.assertEqual(collector._success_count, 400)

    def test_uptime_increases(self):
        """Uptime increases over time."""
        collector = MetricsCollector()
        h1 = collector.get_health()
        time.sleep(0.05)
        h2 = collector.get_health()
        self.assertGreater(h2.uptime, h1.uptime)


if __name__ == "__main__":
    unittest.main()