import pytest
"""Quick validation tests for realtime_optimizer module."""
import sys
import time
import threading
import numpy as np
sys.path.insert(0, '.')

from vram_core.realtime_optimizer import RingBuffer, SileroVAD, LatencyTracker, RealtimePipeline, PipelineConfig

results = []

def test(name, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"  PASS: {name}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  FAIL: {name}: {e}")

# RingBuffer
def t1():
    buf = RingBuffer(1000)
    data = np.random.randn(500).astype(np.float32)
    assert buf.write(data) == 500
    assert buf.size == 500
    rd = buf.read(500)
    assert len(rd) == 500
    np.testing.assert_array_almost_equal(rd, data)
    buf.clear()
    assert buf.is_empty

def t2():
    buf = RingBuffer(100)
    buf.write(np.arange(100, dtype=np.float32))
    buf.write(np.arange(100, 200, dtype=np.float32))
    rd = buf.read(100)
    np.testing.assert_array_almost_equal(rd, np.arange(100, 200, dtype=np.float32))

def t3():
    buf = RingBuffer(10000)
    errors = []
    def w():
        try:
            for i in range(50):
                buf.write(np.full(100, float(i), dtype=np.float32))
        except Exception as e:
            errors.append(e)
    def r():
        try:
            for _ in range(50):
                buf.read(50)
        except Exception as e:
            errors.append(e)
    ts = [threading.Thread(target=w), threading.Thread(target=r), threading.Thread(target=w)]
    for t in ts: t.start()
    for t in ts: t.join(timeout=10)
    assert len(errors) == 0

def t4():
    buf = RingBuffer(16000 * 5)
    chunk = np.random.randn(480).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(1000):
        buf.write(chunk)
    ms = (time.perf_counter() - t0) / 1000 * 1000
    assert ms < 1.0, f"Too slow: {ms:.4f} ms/op"

# SileroVAD fallback
def t5():
    vad = SileroVAD(threshold=0.5)
    speech = (0.3 * np.sin(np.linspace(0, 6.28, 16000))).astype(np.float32)
    silence = (0.001 * np.random.randn(16000)).astype(np.float32)
    assert vad._fallback_energy_vad(speech) is True
    assert vad._fallback_energy_vad(silence) is False
    assert vad._fallback_energy_vad(np.array([], dtype=np.float32)) is False
    vad.reset()
    assert vad.is_speech_active is False

# LatencyTracker
def t6():
    tracker = LatencyTracker()
    tracker.start_measurement(audio_duration_ms=1000)
    tracker.start_vad(); time.sleep(0.001); tracker.end_vad()
    tracker.start_asr(); time.sleep(0.005); tracker.end_asr()
    m = tracker.complete()
    assert m is not None
    assert m.vad_latency_ms > 0
    assert m.asr_latency_ms > 0
    assert m.total_latency_ms > 0
    assert m.audio_duration_ms == 1000
    bd = m.breakdown
    assert "vad_pct" in bd and "asr_pct" in bd

def t7():
    tracker = LatencyTracker()
    for _ in range(10):
        tracker.start_measurement()
        tracker.start_vad(); time.sleep(0.001); tracker.end_vad()
        tracker.start_asr(); time.sleep(0.003); tracker.end_asr()
        tracker.complete()
    stats = tracker.get_stats()
    assert stats["count"] == 10
    for k in ["mean", "p50", "p95", "p99", "min", "max"]:
        assert k in stats["total_ms"]

# Pipeline
def t8():
    config = PipelineConfig(silero_vad_threshold=0.5)
    pipeline = RealtimePipeline(config=config)
    pipeline.start()
    assert pipeline.is_running
    pipeline.stop()
    assert not pipeline.is_running

def t9():
    pipeline = RealtimePipeline(config=PipelineConfig())
    pipeline.start()
    for _ in range(20):
        pipeline.feed(np.random.randn(480).astype(np.float32))
    assert pipeline.stats.chunks_received == 20
    assert pipeline.stats.vad_decisions == 20
    pipeline.stop()

def t10():
    config = PipelineConfig(chunk_duration_ms=30)
    pipeline = RealtimePipeline(config=config)
    pipeline.start()
    chunk = np.random.randn(480).astype(np.float32)
    lats = []
    for _ in range(100):
        t0 = time.perf_counter()
        pipeline.feed(chunk)
        lats.append((time.perf_counter() - t0) * 1000)
    pipeline.stop()
    arr = np.array(lats)
    mean_ms = np.mean(arr)
    print(f"    Latency: mean={mean_ms:.2f}ms, p95={np.percentile(arr,95):.2f}ms, p99={np.percentile(arr,99):.2f}ms")

print("=" * 50)
print("RealtimeOptimizer Quick Tests")
print("=" * 50)

test("RingBuffer basic", t1)
test("RingBuffer overwrite", t2)
test("RingBuffer thread-safety", t3)
test("RingBuffer performance", t4)
test("SileroVAD fallback", t5)
test("LatencyTracker basic", t6)
test("LatencyTracker stats", t7)
test("Pipeline lifecycle", t8)
test("Pipeline feed", t9)
test("Pipeline latency benchmark", t10)

passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print("=" * 50)
print(f"Results: {passed}/{passed+failed} passed, {failed} failed")
print("=" * 50)
if failed > 0:
    for name, ok, err in results:
        if not ok:
            print(f"  FAILED: {name}: {err}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)