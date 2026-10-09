"""
Tests for the end-to-end microsecond latency profiler (v2.7.0).

Covers:
    - ``LatencyTrace``: microsecond marks/offsets, incremental stage durations,
      the ``span`` context manager and JSON serialisation
    - ``LatencyProfiler``: trace lifecycle, the active-trace API, waterfall
      rendering, per-stage percentile statistics, JSON export and the optional
      ``MetricsCollector`` gauge integration

Timestamps are injected directly (a trace is just a start time plus marks), so
the assertions are exact rather than wall-clock dependent; a couple of tests use
the real clock only to prove that measurement itself is monotonic.
"""

import json

import pytest

from vram_core.monitoring import (
    PIPELINE_STAGES,
    LatencyProfiler,
    LatencyTrace,
    MetricsCollector,
)


def fixed_trace(trace_id: str = "t", scale: float = 1.0) -> LatencyTrace:
    """A trace with deterministic microsecond timings."""
    return LatencyTrace(
        trace_id=trace_id,
        start_us=1_000.0,
        marks={
            "vad_cutoff": 1_000.0 + 100.0 * scale,
            "asr_transcribed": 1_000.0 + 1_100.0 * scale,
            "llm_first_token": 1_000.0 + 1_600.0 * scale,
            "tts_first_chunk": 1_000.0 + 2_100.0 * scale,
        },
    )


# ─── Stages ─────────────────────────────────────────────────────────────────

class TestPipelineStages:
    def test_canonical_order(self):
        assert PIPELINE_STAGES == (
            "vad_cutoff", "asr_transcribed", "llm_first_token", "tts_first_chunk",
        )


# ─── LatencyTrace ───────────────────────────────────────────────────────────

class TestLatencyTrace:
    """Microsecond accounting of a single turn."""

    def test_offsets_are_relative_to_the_start(self):
        trace = fixed_trace()
        assert trace.offset_us("vad_cutoff") == pytest.approx(100.0)
        assert trace.offset_us("tts_first_chunk") == pytest.approx(2_100.0)

    def test_durations_are_incremental(self):
        trace = fixed_trace()
        assert trace.duration_us("vad_cutoff") == pytest.approx(100.0)
        assert trace.duration_us("asr_transcribed") == pytest.approx(1_000.0)
        assert trace.duration_us("llm_first_token") == pytest.approx(500.0)
        assert trace.duration_us("tts_first_chunk") == pytest.approx(500.0)

    def test_total_is_the_end_to_end_latency(self):
        assert fixed_trace().total_us == pytest.approx(2_100.0)

    def test_unmarked_stage_reports_zero(self):
        trace = LatencyTrace(trace_id="t", start_us=0.0, marks={"vad_cutoff": 50.0})
        assert trace.offset_us("tts_first_chunk") == 0.0
        assert trace.duration_us("tts_first_chunk") == 0.0

    def test_empty_trace_has_zero_total(self):
        assert LatencyTrace(trace_id="t").total_us == 0.0

    def test_mark_uses_the_monotonic_clock(self):
        trace = LatencyTrace(trace_id="live")
        first = trace.mark("vad_cutoff")
        second = trace.mark("asr_transcribed")
        assert second >= first >= 0.0

    def test_span_context_manager_marks_the_stage(self):
        trace = LatencyTrace(trace_id="span")
        with trace.span("vad_cutoff"):
            pass
        assert "vad_cutoff" in trace.marks

    def test_stage_views(self):
        trace = fixed_trace()
        assert trace.offsets()["llm_first_token"] == pytest.approx(1_600.0)
        assert trace.stage_durations()["llm_first_token"] == pytest.approx(500.0)

    def test_to_dict_is_json_serialisable(self):
        payload = fixed_trace("t1").to_dict()
        assert payload["trace_id"] == "t1"
        assert payload["total_us"] == pytest.approx(2_100.0)
        assert payload["stages"]["tts_first_chunk"]["duration_us"] == pytest.approx(500.0)
        json.dumps(payload)


# ─── LatencyProfiler ────────────────────────────────────────────────────────

class TestLatencyProfilerLifecycle:
    """Starting, marking and finishing traces."""

    def test_start_and_finish_records_the_trace(self):
        profiler = LatencyProfiler()
        trace = profiler.start_trace("turn-1")
        trace.mark("vad_cutoff")
        profiler.finish_trace(trace)

        assert trace.completed is True
        assert [t.trace_id for t in profiler.traces] == ["turn-1"]
        assert profiler.last_trace is trace

    def test_mark_uses_the_active_trace(self):
        profiler = LatencyProfiler()
        trace = profiler.start_trace("turn-1")
        offset = profiler.mark("vad_cutoff")
        assert offset >= 0.0
        assert "vad_cutoff" in trace.marks

    def test_marking_without_a_trace_raises(self):
        profiler = LatencyProfiler()
        with pytest.raises(RuntimeError):
            profiler.mark("vad_cutoff")

    def test_context_manager_finishes_the_trace(self):
        profiler = LatencyProfiler()
        with profiler.trace("turn-1") as trace:
            trace.mark("vad_cutoff")
        assert profiler.last_trace is trace
        assert trace.completed is True

    def test_history_is_bounded(self):
        profiler = LatencyProfiler(max_history=2)
        for index in range(3):
            profiler.finish_trace(fixed_trace(f"t{index}"))
        assert [t.trace_id for t in profiler.traces] == ["t1", "t2"]

    def test_reset_clears_everything(self):
        profiler = LatencyProfiler()
        profiler.start_trace("turn-1")
        profiler.finish_trace(fixed_trace("t1"))
        profiler.reset()
        assert profiler.traces == []
        assert profiler.last_trace is None
        with pytest.raises(RuntimeError):
            profiler.mark("vad_cutoff")

    def test_finish_without_arguments_uses_the_active_trace(self):
        profiler = LatencyProfiler()
        profiler.start_trace("turn-1")
        finished = profiler.finish_trace()
        assert finished.trace_id == "turn-1"
        assert profiler.last_trace is finished


# ─── Statistics ─────────────────────────────────────────────────────────────

class TestLatencyStatistics:
    """Per-stage percentiles over the completed traces."""

    def test_stage_stats_are_reported_in_microseconds(self):
        profiler = LatencyProfiler()
        for index, scale in enumerate([1.0, 2.0, 3.0]):
            profiler.finish_trace(fixed_trace(f"t{index}", scale=scale))

        stats = profiler.stage_stats()
        assert stats["vad_cutoff"]["count"] == 3.0
        assert stats["vad_cutoff"]["min"] == pytest.approx(100.0)
        assert stats["vad_cutoff"]["max"] == pytest.approx(300.0)
        assert stats["vad_cutoff"]["mean"] == pytest.approx(200.0)
        assert stats["asr_transcribed"]["mean"] == pytest.approx(2_000.0)

    def test_percentiles_follow_the_nearest_rank(self):
        profiler = LatencyProfiler()
        for scale in [1.0, 2.0, 3.0, 4.0]:
            profiler.finish_trace(fixed_trace("t", scale=scale))
        # tts_first_chunk durations are 500 / 1000 / 1500 / 2000 us
        stats = profiler.stage_stats()["tts_first_chunk"]
        assert stats["p50"] == pytest.approx(1_500.0)
        assert stats["p95"] == pytest.approx(2_000.0)

    def test_stage_stats_ignore_unmarked_stages(self):
        profiler = LatencyProfiler()
        profiler.finish_trace(LatencyTrace(
            trace_id="partial", start_us=0.0, marks={"vad_cutoff": 10.0},
        ))
        stats = profiler.stage_stats()
        assert "vad_cutoff" in stats
        assert "asr_transcribed" not in stats

    def test_summary_reports_end_to_end_percentiles(self):
        profiler = LatencyProfiler()
        for scale in [1.0, 2.0, 3.0]:
            profiler.finish_trace(fixed_trace("t", scale=scale))
        summary = profiler.summary()
        assert summary["traces"] == 3
        assert summary["stages"] == list(PIPELINE_STAGES)
        assert summary["end_to_end_us"]["mean"] == pytest.approx(4_200.0)
        assert summary["end_to_end_us"]["max"] == pytest.approx(6_300.0)

    def test_summary_of_an_empty_profiler_is_zeroed(self):
        summary = LatencyProfiler().summary()
        assert summary["traces"] == 0
        assert summary["end_to_end_us"]["mean"] == 0.0
        assert summary["stage_stats"] == {}


# ─── Rendering & export ─────────────────────────────────────────────────────

class TestLatencyReporting:
    """Waterfall rendering, JSON export and metrics integration."""

    def test_waterfall_lists_every_marked_stage(self):
        profiler = LatencyProfiler()
        profiler.finish_trace(fixed_trace("turn-1"))
        chart = profiler.waterfall()
        assert "turn-1" in chart
        for stage in PIPELINE_STAGES:
            assert stage in chart
        assert chart.count("\n") == len(PIPELINE_STAGES)      # header + 4 rows

    def test_waterfall_without_a_trace_is_a_placeholder(self):
        assert "no completed trace" in LatencyProfiler().waterfall()

    def test_waterfall_accepts_an_explicit_trace(self):
        profiler = LatencyProfiler()
        chart = profiler.waterfall(fixed_trace("explicit"))
        assert "explicit" in chart

    def test_export_json_writes_a_readable_file(self, tmp_path):
        profiler = LatencyProfiler()
        profiler.finish_trace(fixed_trace("turn-1"))
        target = tmp_path / "latency.json"
        payload = profiler.export_json(str(target))

        assert target.exists()
        on_disk = json.loads(target.read_text(encoding="utf-8"))
        assert on_disk == payload
        assert on_disk["summary"]["traces"] == 1
        assert on_disk["traces"][0]["trace_id"] == "turn-1"

    def test_export_json_without_a_path_returns_the_payload(self):
        profiler = LatencyProfiler()
        profiler.finish_trace(fixed_trace("turn-1"))
        payload = profiler.export_json()
        assert payload["summary"]["traces"] == 1
        assert isinstance(payload["traces"], list)

    def test_stage_durations_are_published_to_the_metrics_collector(self):
        collector = MetricsCollector()
        profiler = LatencyProfiler(collector=collector)
        profiler.finish_trace(fixed_trace("turn-1"))

        gauges = collector.get_metrics()["custom_gauges"]
        assert gauges["latency.vad_cutoff.us"] == pytest.approx(100.0)
        assert gauges["latency.tts_first_chunk.us"] == pytest.approx(500.0)
        assert gauges["latency.e2e.us"] == pytest.approx(2_100.0)

    def test_profiler_without_a_collector_is_still_usable(self):
        profiler = LatencyProfiler(collector=None)
        assert profiler.finish_trace(fixed_trace("turn-1")).trace_id == "turn-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
