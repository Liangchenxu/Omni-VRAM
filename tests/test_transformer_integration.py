"""Tests for the v2.8.0 Transformer decode hook (``wddm_bypass.integration``).

The suite has the same two halves as ``test_engine_suite.py``:

Pure Python (always runs)
    Plan validation, static-buffer views, capture/replay bookkeeping against an
    injected fake backend, the ``accelerate_decode`` decorator and the
    HuggingFace-shaped ``HookedDecoder`` driven with numpy arrays.

GPU (needs the built extension and a CUDA device)
    A simulated Llama/Qwen decode step - 12 layers x 4 launches plus residual
    folding, 51 kernels per step - is captured into one graph. Its outputs are
    compared with a numpy reference, and the submission cost of 51 eager
    launches is compared with a single graph launch.
"""

import json
import sys
import sysconfig
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from wddm_bypass import Engine, ShapeKey, load_backend  # noqa: E402
from wddm_bypass.backend import BackendUnavailable  # noqa: E402
from wddm_bypass.integration import (  # noqa: E402
    BufferSpec,
    DecodeBuffers,
    DecodeHook,
    DecodePlan,
    HookClosed,
    HookedDecoder,
    accelerate_decode,
    data_pointer,
    dtype_size,
    tensor_pointer,
    torch_stream,
)

# ---------------------------------------------------------------------------
# Fake native backend (mirrors the native signatures the facade calls)
# ---------------------------------------------------------------------------

LAUNCH_CALLS = []


def _launch_scale(dst, count, alpha, stream_token=0):
    LAUNCH_CALLS.append(("scale", dst, count, float(alpha), int(stream_token)))


def _launch_add(dst, lhs, rhs, count, stream_token=0):
    LAUNCH_CALLS.append(("add", dst, lhs, rhs, count, int(stream_token)))


def _launch_relu(dst, count, stream_token=0):
    LAUNCH_CALLS.append(("relu", dst, count, int(stream_token)))


class FakeNativeEngine:
    """Records every native call the facade makes; no CUDA involved."""

    def __init__(self, device_index=-1):
        self.device_index = device_index
        self.calls = []
        self.slots = {}
        self.buffers = {}
        self.capturing = {}
        self.replays = 0
        self.synchronize_calls = 0
        self._address = 0x1000

    # -- capture -----------------------------------------------------------
    def begin_capture(self, graph_id, shape_key="", hot_update=False):
        self.calls.append(("begin_capture", graph_id, hot_update))
        if self.capturing.get(graph_id):
            raise RuntimeError("already capturing")
        self.capturing[graph_id] = True
        self.slots[graph_id] = {"shape_key": shape_key, "nodes": 0}

    def end_capture(self, graph_id):
        self.calls.append(("end_capture", graph_id))
        if not self.capturing.get(graph_id):
            raise RuntimeError("not capturing")
        self.capturing[graph_id] = False
        self.slots[graph_id]["nodes"] = 1

    def is_capturing(self, graph_id):
        return bool(self.capturing.get(graph_id))

    # -- execution ---------------------------------------------------------
    def replay(self, graph_id, stream_token=0):
        self.calls.append(("replay", graph_id, stream_token))
        if not self.has_graph(graph_id):
            raise RuntimeError("graph has no executable")
        self.replays += 1

    def update_graph(self, graph_id):
        self.calls.append(("update_graph", graph_id))

    def reset(self, graph_id):
        self.calls.append(("reset", graph_id))
        self.slots.pop(graph_id, None)
        self.capturing.pop(graph_id, None)

    def reset_all(self):
        self.calls.append(("reset_all",))
        self.slots.clear()
        self.capturing.clear()

    def has_graph(self, graph_id):
        slot = self.slots.get(graph_id)
        return bool(slot and slot["nodes"] > 0)

    def list_graphs(self):
        return sorted(self.slots)

    def graph_id_for_shape(self, shape_key):
        for identifier, slot in self.slots.items():
            if slot["shape_key"] == shape_key:
                return identifier
        return -1

    # -- buffers -----------------------------------------------------------
    def register_static_buffer(self, name, bytes):
        self.calls.append(("register_static_buffer", name, bytes))
        self._address += 0x100
        self.buffers[name] = (self._address, bytes)
        return self._address

    def static_buffer_ptr(self, name):
        if name not in self.buffers:
            raise RuntimeError(f"unknown buffer {name}")
        return self.buffers[name][0]

    def static_buffer_bytes(self, name):
        if name not in self.buffers:
            raise RuntimeError(f"unknown buffer {name}")
        return self.buffers[name][1]

    def release_static_buffer(self, name):
        self.calls.append(("release_static_buffer", name))
        self.buffers.pop(name, None)

    def copy_to_static_input(self, name, source, bytes, stream_token=0):
        self.calls.append(("copy_to_static_input", name, source, bytes, stream_token))

    def copy_from_static(self, name, destination, bytes, stream_token=0):
        self.calls.append(("copy_from_static", name, destination, bytes, stream_token))

    def zero_static_buffer(self, name, stream_token=0):
        self.calls.append(("zero_static_buffer", name, stream_token))

    # -- introspection -----------------------------------------------------
    def describe(self, graph_id):
        return json.dumps({"graph_id": graph_id, "has_exec": self.has_graph(graph_id)})

    def capture_stream_ptr(self):
        return 0xCAFE

    def synchronize(self):
        self.synchronize_calls += 1

    def stats(self):
        return {"graphs": len(self.slots), "replays": self.replays}


FAKE_BACKEND = SimpleNamespace(
    WDDMBypassEngine=FakeNativeEngine,
    launch_scale=_launch_scale,
    launch_add=_launch_add,
    launch_relu=_launch_relu,
    __version__="1.0.0-fake",
    build_info=lambda: {"version": "1.0.0-fake"},
)

NO_OPS_BACKEND = SimpleNamespace(WDDMBypassEngine=FakeNativeEngine)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine():
    """Engine wired to the fake backend, closed at the end of the test."""
    LAUNCH_CALLS.clear()
    instance = Engine(backend=FAKE_BACKEND)
    try:
        yield instance
    finally:
        instance.close()


def decode_plan(key=(1, 1, 8), size=8, dtype="float32"):
    """Five-buffer geometry that looks like a one-token decode step."""
    return DecodePlan.build(
        key,
        {"x": size, "tmp": size, "tmp2": size, "acc": size, "out": size},
        inputs=("x",),
        outputs=("out",),
        dtype=dtype,
        name="unit-test-decode",
    )


@pytest.fixture()
def plan():
    return decode_plan()


def fake_body(buffers):
    """Four launches: scale -> relu -> add -> add."""
    buffers.scale("tmp", 0.5)
    buffers.relu("tmp")
    buffers.add("tmp2", "tmp", "x")
    buffers.add("acc", "x", "x")
    return buffers.launches


# ---------------------------------------------------------------------------
# BufferSpec and DecodePlan
# ---------------------------------------------------------------------------


class TestBufferSpec:
    def test_nbytes_uses_the_dtype_size(self):
        assert dtype_size("fp16") == 2
        spec = BufferSpec(name="hidden", count=16, dtype="float16")
        assert spec.nbytes == 32
        assert spec.shape == ()

    def test_shape_is_kept_and_must_match_count(self):
        spec = BufferSpec.of("hidden", (2, 4, 8), "float32")
        assert (spec.count, spec.shape, spec.nbytes) == (64, (2, 4, 8), 256)
        with pytest.raises(ValueError):
            BufferSpec(name="hidden", count=8, shape=(2, 8))

    @pytest.mark.parametrize("count", [0, -4])
    def test_non_positive_counts_rejected(self, count):
        with pytest.raises(ValueError):
            BufferSpec(name="x", count=count)

    def test_invalid_names_counts_and_dtypes(self):
        with pytest.raises(TypeError):
            BufferSpec(name="x", count=1.5)
        with pytest.raises(ValueError):
            BufferSpec(name="x", count=4, dtype="float8")
        with pytest.raises(ValueError):
            BufferSpec(name="   ", count=4)
        with pytest.raises(ValueError):
            BufferSpec.of("x", (2, 0))
        with pytest.raises(TypeError):
            BufferSpec.of("x", (2, 2.0))

    def test_dtype_size_rejects_unknown_input(self):
        with pytest.raises(ValueError):
            dtype_size("complex256")
        with pytest.raises(TypeError):
            dtype_size(4)


class TestDecodePlan:
    def test_build_from_shapes(self, plan):
        assert plan.key == ShapeKey(1, 1, 8)
        assert plan.nbytes == 5 * 8 * 4
        assert plan.dtype == "float32"
        assert plan.buffers["x"].count == 8
        assert plan.describe()["inputs"] == ["x"]
        assert plan.describe()["buffers"]["out"] == 32

    def test_unknown_buffer_names_rejected(self):
        with pytest.raises(ValueError):
            DecodePlan.build((1, 1, 8), {"x": 8}, inputs=("missing",))
        with pytest.raises(ValueError):
            DecodePlan.build((1, 1, 8), {"x": 8}, outputs=("missing",))

    def test_duplicate_buffer_names_rejected(self):
        with pytest.raises(ValueError):
            DecodePlan(
                key=ShapeKey(1, 1, 8),
                specs=(BufferSpec("x", 8), BufferSpec("x", 8)),
            )

    def test_key_must_be_a_shape_key(self):
        with pytest.raises(TypeError):
            DecodePlan(key=(1, 1, 8), specs=(BufferSpec("x", 8),))

    def test_empty_specs_rejected(self):
        with pytest.raises(TypeError):
            DecodePlan(key=ShapeKey(1, 1, 8), specs=())

    def test_spec_lookup(self, plan):
        assert plan.spec("x").count == 8
        with pytest.raises(KeyError):
            plan.spec("nope")


# ---------------------------------------------------------------------------
# DecodeBuffers
# ---------------------------------------------------------------------------


class TestDecodeBuffers:
    def test_pointers_come_from_the_engine(self, engine, plan):
        hook = DecodeHook(engine, plan=plan)
        try:
            hook.register()
            view = DecodeBuffers(engine, plan)
            assert view.ptr("x") == engine.buffer_ptr("x")
            assert (view.count("x"), view.nbytes("x")) == (8, 32)
            assert view.names()[:2] == ("x", "tmp")
            assert len(view) == 5
        finally:
            hook.close()

    def test_launches_count_and_carry_the_stream(self, engine, plan):
        hook = DecodeHook(engine, plan=plan)
        try:
            hook.register()
            view = DecodeBuffers(engine, plan, engine.capture_stream())
            view.scale("tmp", 0.5).relu("tmp").add("tmp2", "tmp", "x")
            assert view.launches == 3
            assert view.captured is True
            assert [call[0] for call in LAUNCH_CALLS] == ["scale", "relu", "add"]
            assert all(call[-1] == 0xCAFE for call in LAUNCH_CALLS)
        finally:
            hook.close()

    def test_add_requires_equal_element_counts(self, engine):
        mixed = DecodePlan.build((1, 1, 8), {"x": 8, "y": 16, "z": 8})
        hook = DecodeHook(engine, plan=mixed)
        try:
            view = DecodeBuffers(engine, mixed)
            with pytest.raises(ValueError):
                view.add("x", "y", "z")
        finally:
            hook.close()

    def test_missing_launcher_is_reported(self, plan):
        engine = Engine(backend=NO_OPS_BACKEND)
        try:
            view = DecodeBuffers(engine, plan)
            with pytest.raises(BackendUnavailable):
                view.relu("tmp")
        finally:
            engine.close()

    def test_unknown_buffer_name_raises(self, engine, plan):
        view = DecodeBuffers(engine, plan)
        with pytest.raises(KeyError):
            view.scale("nope", 1.0)

    def test_bools_and_null_are_not_addresses(self):
        with pytest.raises(TypeError):
            data_pointer(True)
        with pytest.raises(ValueError):
            data_pointer(0)
        with pytest.raises(TypeError):
            data_pointer(object())


# ---------------------------------------------------------------------------
# DecodeHook bookkeeping (fake backend)
# ---------------------------------------------------------------------------


class TestDecodeHook:
    def test_first_call_warms_captures_then_replays(self, engine, plan):
        hook = DecodeHook(engine, plan=plan, warmup=1)
        try:
            first = hook.execute(fake_body)
            assert first["path"] == "capture"
            assert first["launches"] == 4
            stats = hook.stats()
            assert stats["captures"] == 1
            assert stats["warmup_steps"] == 1
            assert stats["replays"] == 1

            second = hook.execute(fake_body)
            assert second["path"] == "replay"
            assert second["graph_id"] == first["graph_id"]
            stats = hook.stats()
            assert stats["captures"] == 1
            assert stats["replays"] == 2
            assert stats["steps"] == 2
            assert stats["graphs"] == 1
            assert stats["registered_bytes"] == plan.nbytes
        finally:
            hook.close()

    def test_capture_runs_on_the_capture_stream(self, engine, plan):
        hook = DecodeHook(engine, plan=plan, warmup=0)
        try:
            hook.execute(fake_body)
            captured = [call[-1] for call in LAUNCH_CALLS]
            assert captured, "the body issued no launches"
            assert all(stream == 0xCAFE for stream in captured)
            LAUNCH_CALLS.clear()
            hook.execute(fake_body, eager=True)
            assert all(call[-1] == 0 for call in LAUNCH_CALLS)
        finally:
            hook.close()

    def test_eager_path_never_captures(self, engine, plan):
        hook = DecodeHook(engine, plan=plan)
        try:
            info = hook.execute(fake_body, eager=True)
            assert info["path"] == "eager"
            assert info["launches"] == 4
            stats = hook.stats()
            assert stats["captures"] == 0
            assert stats["eager_steps"] == 1
            assert stats["replays"] == 0
        finally:
            hook.close()

    def test_feed_and_fetch_bracket_the_replay(self, engine, plan):
        hook = DecodeHook(engine, plan=plan, warmup=0)
        source = np.arange(8, dtype=np.float32)
        destination = np.zeros(8, dtype=np.float32)
        try:
            for _ in range(2):
                hook.execute(fake_body, feed={"x": source}, fetch={"out": destination})
            names = [call[0] for call in engine._raw.calls]
            assert "copy_to_static_input" in names
            assert "copy_from_static" in names
            last_replay = max(index for index, name in enumerate(names) if name == "replay")
            last_copy_in = max(
                index for index, name in enumerate(names) if name == "copy_to_static_input"
            )
            last_copy_out = max(
                index for index, name in enumerate(names) if name == "copy_from_static"
            )
            assert last_copy_in < last_replay < last_copy_out, names
            assert hook.stats()["feed_copies"] == 2
            assert hook.stats()["fetch_copies"] == 2
        finally:
            hook.close()

    def test_replay_before_capture_raises(self, engine, plan):
        hook = DecodeHook(engine, plan=plan)
        try:
            with pytest.raises(KeyError):
                hook.replay()
        finally:
            hook.close()

    def test_graph_budget_falls_back_to_eager(self, engine):
        first = decode_plan((1, 1, 8))
        second = decode_plan((1, 2, 8))
        hook = DecodeHook(engine, plan=first, warmup=0, max_graphs=1)
        try:
            assert hook.execute(fake_body)["path"] == "capture"
            info = hook.execute(fake_body, plan=second)
            assert info["path"] == "eager"
            assert "budget" in info["reason"]
            assert hook.stats()["eager_fallbacks"] == 1
            assert hook.stats()["graphs"] == 1
        finally:
            hook.close()

    def test_resizing_a_registered_buffer_is_refused(self, engine):
        small = decode_plan((1, 1, 8), size=8)
        large = decode_plan((1, 1, 16), size=16)
        hook = DecodeHook(engine, plan=small, warmup=0)
        try:
            hook.register()
            with pytest.raises(ValueError):
                hook.register(large)
        finally:
            hook.close()

    def test_body_with_no_launches_is_refused(self, engine, plan):
        hook = DecodeHook(engine, plan=plan, warmup=0)
        try:
            with pytest.raises(ValueError):
                hook.execute(lambda buffers: None)
        finally:
            hook.close()

    def test_closed_hook_rejects_every_call(self, engine, plan):
        hook = DecodeHook(engine, plan=plan)
        hook.close()
        hook.close()  # idempotent
        assert hook.closed is True
        with pytest.raises(HookClosed):
            hook.stats()
        with pytest.raises(HookClosed):
            hook.execute(fake_body)

    def test_plan_is_required(self, engine):
        hook = DecodeHook(engine)
        try:
            with pytest.raises(ValueError):
                hook.execute(fake_body)
        finally:
            hook.close()


# ---------------------------------------------------------------------------
# The decorator and the HuggingFace-shaped mount (fake backend, real numpy)
# ---------------------------------------------------------------------------


class TestAccelerateDecode:
    def test_captures_once_and_stops_calling_the_body(self, engine, plan):
        hook = DecodeHook(engine, plan=plan, warmup=1)
        seen = []

        @accelerate_decode(hook)
        def decode_step(buffers, layers=1):
            seen.append(layers)
            for _ in range(layers):
                buffers.scale("tmp", 0.5)
            return buffers.launches

        try:
            first = decode_step(layers=2)
            assert first["path"] == "capture"
            assert first["launches"] == 2
            # Warm-up runs the body, and so does the capture itself (recording
            # means issuing the launches once). Nothing else may run it.
            assert seen == [2, 2], "warm-up and capture execute the body once each"

            second = decode_step(layers=2)
            assert second["path"] == "replay"
            assert seen == [2, 2], "a replay must not execute the body again"
            assert decode_step.hook is hook
            assert decode_step.frame is plan
            assert decode_step.__name__ == "decode_step"
            assert hook.stats()["captures"] == 1
        finally:
            hook.close()

    def test_non_hook_and_non_callable_are_rejected(self, engine, plan):
        with pytest.raises(TypeError):
            accelerate_decode(engine)
        decorator = accelerate_decode(DecodeHook(engine, plan=plan))
        try:
            with pytest.raises(TypeError):
                decorator("not callable")
        finally:
            decorator(lambda buffers: buffers).hook.close()


class TestHookedDecoder:
    def test_forward_feeds_and_fetches_the_declared_buffers(self, engine, plan):
        decoder = HookedDecoder(
            fake_body, plan, hook=DecodeHook(engine, plan=plan, warmup=0)
        )
        try:
            source = np.arange(8, dtype=np.float32)
            outputs = decoder.forward({"x": source})
            assert set(outputs) == {"out"}
            assert outputs["out"].shape == (8,)
            assert outputs["out"].dtype == np.float32
            assert decoder.last["path"] == "capture"
            names = [call[0] for call in engine._raw.calls]
            assert "copy_to_static_input" in names
            assert "copy_from_static" in names
            second = decoder.forward({"x": source})
            assert decoder.last["path"] == "replay"
            assert set(second) == {"out"}
            assert decoder.stats()["captures"] == 1
        finally:
            decoder.close()

    def test_missing_input_is_named(self, engine, plan):
        decoder = HookedDecoder(
            fake_body, plan, hook=DecodeHook(engine, plan=plan, warmup=0)
        )
        try:
            with pytest.raises(KeyError):
                decoder.forward({})
        finally:
            decoder.close()

    def test_wrong_input_size_is_refused(self, engine, plan):
        decoder = HookedDecoder(
            fake_body, plan, hook=DecodeHook(engine, plan=plan, warmup=0)
        )
        try:
            with pytest.raises(ValueError):
                decoder.forward({"x": np.zeros(7, dtype=np.float32)})
        finally:
            decoder.close()

    def test_install_replaces_forward_and_returns_the_old_one(self, engine, plan):
        class StandInModel:
            """Duck-typed LlamaForCausalLM: only ``forward`` is needed."""

            def __init__(self):
                self.eager_calls = 0

            def forward(self, inputs):
                self.eager_calls += 1
                return "eager"

        decoder = HookedDecoder(
            fake_body, plan, hook=DecodeHook(engine, plan=plan, warmup=0)
        )
        model = StandInModel()
        previous = decoder.install(model)
        try:
            assert previous({}) == "eager"
            assert model.eager_calls == 1
            outputs = model.forward({"x": np.zeros(8, dtype=np.float32)})
            assert set(outputs) == {"out"}
            assert model.eager_calls == 1, "the graph path must not call the model"
        finally:
            decoder.close()

    def test_bfloat16_outputs_explain_the_limitation(self, engine):
        bf16_plan = decode_plan(dtype="bfloat16")
        decoder = HookedDecoder(
            fake_body, bf16_plan, hook=DecodeHook(engine, plan=bf16_plan, warmup=0)
        )
        try:
            source = np.zeros(8, dtype=np.uint16)  # 16 bytes, same as 8 x bf16
            with pytest.raises(ValueError):
                decoder.forward({"x": source})
        finally:
            decoder.close()


# ---------------------------------------------------------------------------
# Torch adapters and pointer helpers
# ---------------------------------------------------------------------------


class TestTorchAdapters:
    def test_tensor_pointer_rejects_non_tensors(self):
        with pytest.raises(TypeError):
            tensor_pointer(object())
        with pytest.raises(TypeError):
            tensor_pointer(np.zeros(4, dtype=np.float32))

    def test_data_pointer_accepts_arrays_and_buffers(self):
        array = np.zeros(4, dtype=np.float32)
        assert data_pointer(array) == array.ctypes.data
        assert data_pointer(0x1000) == 0x1000

    def test_torch_stream_wraps_the_requested_stream(self, engine, plan):
        hook = DecodeHook(engine, plan=plan)
        try:
            try:
                import torch
            except ImportError:
                with pytest.raises(BackendUnavailable):
                    with torch_stream(engine):
                        pass
                return
            if not torch.cuda.is_available():
                with pytest.raises(BackendUnavailable):
                    with torch_stream(engine):
                        pass
                return
            token = int(torch.cuda.Stream().cuda_stream)
            assert token != 0, "a fresh torch stream must have a real handle"
            with torch_stream(engine, stream_token=token) as stream:
                assert int(stream.cuda_stream) == token
                assert int(torch.cuda.current_stream().cuda_stream) == token
        finally:
            hook.close()


# ---------------------------------------------------------------------------
# GPU half: a simulated Llama/Qwen decode step, captured and replayed
# ---------------------------------------------------------------------------

BATCH, SEQUENCE, HIDDEN = 1, 1, 2048
LAYERS = 12
ALPHA = 0.5
FINAL_SCALE = 0.25
ELEMENTS = BATCH * SEQUENCE * HIDDEN
KERNELS_PER_STEP = 4 * LAYERS + 3


def gpu_plan(key=(BATCH, SEQUENCE, HIDDEN)):
    """Fixed geometry of one decode step at hidden=2048."""
    return DecodePlan.build(
        key,
        {
            "x": (BATCH, SEQUENCE, HIDDEN),
            "acc": ELEMENTS,
            "tmp": ELEMENTS,
            "out": (BATCH, SEQUENCE, HIDDEN),
        },
        inputs=("x",),
        outputs=("out",),
        name="simulated-llama-decode",
    )


def llama_decode_body(buffers, layers=LAYERS):
    """Twelve decode blocks: scale, activation, and two residual folds each."""
    buffers.add("acc", "x", "x")
    for _ in range(layers):
        buffers.scale("acc", ALPHA)
        buffers.relu("acc")
        buffers.add("tmp", "acc", "x")
        buffers.add("acc", "tmp", "x")
    buffers.scale("acc", FINAL_SCALE)
    buffers.add("out", "acc", "x")


def numpy_reference(x, layers=LAYERS):
    """float64 reference for :func:`llama_decode_body`."""
    acc = x.astype(np.float64) + x
    for _ in range(layers):
        acc = np.maximum(acc * ALPHA, 0.0)
        tmp = acc + x
        acc = tmp + x
    acc = acc * FINAL_SCALE
    return acc + x


def extension_present():
    """True when a build for *this* interpreter exists (no CUDA import here)."""
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"
    package = PROJECT_DIR / "wddm_bypass"
    return (package / f"_wddm_bypass{suffix}").exists()


BACKEND_PRESENT = extension_present()
requires_gpu = pytest.mark.skipif(
    not BACKEND_PRESENT, reason="the compiled CUDA graph engine is not built"
)


@pytest.fixture()
def gpu_engine():
    """A real engine per test, closed afterwards so its graphs and buffers go."""
    engine = Engine()
    try:
        yield engine
    finally:
        engine.close()


def decode_input():
    """Deterministic one-token hidden state, bounded to [0, 1)."""
    return ((np.arange(ELEMENTS, dtype=np.float32) % 97) / 97.0).astype(np.float32)


@requires_gpu
class TestTransformerDecodeOnGPU:
    def test_captured_step_matches_the_numpy_reference(self, gpu_engine):
        assert bool(load_backend().cuda_graph_supported()) is True
        plan = gpu_plan()
        decoder = HookedDecoder(llama_decode_body, plan, engine=gpu_engine, warmup=1)
        try:
            x = decode_input()
            captured = decoder.forward({"x": x})
            assert decoder.last["path"] == "capture"
            replay = decoder.forward({"x": x})
            assert decoder.last["path"] == "replay"
            eager = decoder.forward({"x": x}, eager=True)

            expected = numpy_reference(x)
            for label, result in (
                ("capture", captured), ("replay", replay), ("eager", eager)
            ):
                assert result["out"].shape == (BATCH, SEQUENCE, HIDDEN), label
                np.testing.assert_allclose(
                    result["out"].ravel(), expected, rtol=1e-4, atol=1e-3,
                    err_msg=f"{label} output diverged from the reference",
                )

            stats = decoder.stats()
            assert stats["captures"] == 1
            assert stats["eager_fallbacks"] == 0
            assert stats["kernels_per_step"] == {str(plan.key): KERNELS_PER_STEP}
        finally:
            decoder.close()

    def test_graph_replay_cuts_submission_cost(self, gpu_engine):
        plan = gpu_plan()
        hook = DecodeHook(gpu_engine, plan=plan, warmup=1)
        try:
            hook.execute(llama_decode_body)
            steps = 20
            rounds = 3
            eager_samples = []
            replay_samples = []
            for _ in range(rounds):
                start = time.perf_counter()
                for _ in range(steps):
                    hook.execute(llama_decode_body, eager=True, synchronize=False)
                gpu_engine.synchronize()
                eager_samples.append(time.perf_counter() - start)

                start = time.perf_counter()
                for _ in range(steps):
                    hook.execute(llama_decode_body, synchronize=False)
                gpu_engine.synchronize()
                replay_samples.append(time.perf_counter() - start)

            eager_best = min(eager_samples)
            replay_best = min(replay_samples)
            stats = hook.stats()
            assert stats["kernels_per_step"][str(plan.key)] == KERNELS_PER_STEP
            assert stats["captures"] == 1
            assert stats["eager_fallbacks"] == 0
            assert replay_best < eager_best, (
                f"replay {replay_best * 1e6:.1f} us was not faster than "
                f"eager {eager_best * 1e6:.1f} us"
            )
            assert replay_best * 1.5 < eager_best, (
                f"expected at least 1.5x lower submission cost for "
                f"{KERNELS_PER_STEP} kernels/step, got {eager_best * 1e6:.1f} us "
                f"eager vs {replay_best * 1e6:.1f} us replay"
            )
        finally:
            hook.close()

    def test_hot_update_merges_instead_of_rebuilding(self, gpu_engine):
        plan = gpu_plan()
        hook = DecodeHook(gpu_engine, plan=plan, warmup=0)
        try:
            hook.execute(llama_decode_body)
            before = gpu_engine.stats()
            hook.capture(llama_decode_body, hot_update=True)
            after = gpu_engine.stats()

            assert after["updates"] == before["updates"] + 1
            assert after["rebuilds"] == before["rebuilds"]

            x = decode_input()
            out = np.zeros(ELEMENTS, dtype=np.float32)
            hook.replay(feed={"x": x}, fetch={"out": out})
            np.testing.assert_allclose(
                out, numpy_reference(x), rtol=1e-4, atol=1e-3
            )
        finally:
            hook.close()
