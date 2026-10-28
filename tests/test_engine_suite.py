"""Tests for the WDDM-Bypass execution subsystem (``wddm_bypass``).

Since v2.6.0 the engine is part of the Omni-VRAM tree: the Python package lives
at the repository root (``wddm_bypass/``), the native sources and their build
script stay in ``WDDM-Bypass/``, and the joint vram_core workflow is covered at
the bottom of this file.

The suite is split in two halves:

Pure Python (always runs)
    Bucketing, routing, the ``Engine`` facade against an injected fake backend,
    the loader's error surface, and an ASCII-only gate over ``csrc/``.

GPU (skipped with an explicit reason when unavailable)
    Capture, replay, hot update, buffer round trips and reset against the real
    pybind11 extension. They skip - never silently pass - when the extension is
    not built or no CUDA device can instantiate graphs.

The fake backend is a deliberate copy of the native API surface: it makes the
facade's contract (which native call gets which arguments, in which order)
testable on a machine without a GPU or a CUDA toolchain.
"""

import importlib
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

# The suite lives in the repository's own ``tests/`` directory, so the project
# root - the directory that holds ``wddm_bypass/`` - is one level up.  The native
# half (``csrc/`` and ``setup.py``) stays under ``WDDM-Bypass/``.
PROJECT_DIR = Path(__file__).resolve().parents[1]
NATIVE_DIR = PROJECT_DIR / "WDDM-Bypass"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from wddm_bypass import (  # noqa: E402  (after the sys.path bootstrap)
    BUILD_HINT,
    EXTENSION_NAME,
    BucketPolicy,
    Engine,
    EngineClosed,
    ShapeKey,
    ShapeRouter,
    __version__,
    backend_available,
    build_info,
    capture_context,
    coerce_key,
    load_backend,
)
from wddm_bypass.backend import BackendUnavailable  # noqa: E402
from wddm_bypass.bucketing import power_of_two_at_least, round_up  # noqa: E402

# ---------------------------------------------------------------------------
# Fake native backend
# ---------------------------------------------------------------------------


class FakeNativeEngine:
    """Records every call the facade makes; mirrors the native signatures."""

    def __init__(self, device_index=-1):
        self.device_index = device_index
        self.calls = []
        self.slots = {}
        self.buffers = {}
        self.capturing = {}
        self.replays = 0
        self.reset_all_calls = 0
        self.synchronize_calls = 0
        self._address = 0x1000

    # -- capture -----------------------------------------------------------
    def begin_capture(self, graph_id, shape_key="", hot_update=False):
        self.calls.append(("begin_capture", graph_id, shape_key, hot_update))
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
        self.reset_all_calls += 1
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
        payload = {"graph_id": graph_id, "has_exec": self.has_graph(graph_id)}
        return json.dumps(payload)

    def capture_stream_ptr(self):
        return 0xCAFE

    def synchronize(self):
        self.synchronize_calls += 1

    def stats(self):
        return {"graphs": len(self.slots), "replays": self.replays, "torch_interop": 0}


FAKE_BACKEND = SimpleNamespace(
    WDDMBypassEngine=FakeNativeEngine,
    __version__="1.0.0-fake",
    build_info=lambda: {"version": "1.0.0-fake"},
)


@pytest.fixture()
def engine():
    """Engine wired to the fake backend, closed at the end of the test."""
    instance = Engine(backend=FAKE_BACKEND)
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture()
def exact_engine():
    """Fake-backed engine whose policy gives every shape its own graph."""
    instance = Engine(backend=FAKE_BACKEND, policy=BucketPolicy.exact())
    try:
        yield instance
    finally:
        instance.close()


def call_names(engine_instance):
    """Sequence of native call names recorded by the fake backend."""
    return [call[0] for call in engine_instance._raw.calls]


# ---------------------------------------------------------------------------
# ShapeKey and the rounding helpers
# ---------------------------------------------------------------------------


class TestShapeKey:
    def test_fields_and_element_count(self):
        key = ShapeKey(2, 128, 768, "float16")
        assert (key.batch, key.sequence, key.hidden, key.dtype) == (2, 128, 768, "float16")
        assert key.elements == 2 * 128 * 768
        assert key.as_tuple() == (2, 128, 768, "float16")

    def test_dtype_is_normalised(self):
        assert ShapeKey(1, 8, 8, " FP16 ").dtype == "fp16"
        assert ShapeKey(1, 8, 8, "FP16") == ShapeKey(1, 8, 8, "fp16")

    def test_hashable_and_distinct_by_dtype(self):
        assert len({ShapeKey(1, 8, 8), ShapeKey(1, 8, 8, "float16")}) == 2

    @pytest.mark.parametrize("field", ["batch", "sequence", "hidden"])
    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_dimensions_rejected(self, field, value):
        arguments = {"batch": 1, "sequence": 1, "hidden": 1}
        arguments[field] = value
        with pytest.raises(ValueError):
            ShapeKey(**arguments)

    def test_non_integer_dimensions_rejected(self):
        with pytest.raises(TypeError):
            ShapeKey(1.0, 1, 1)
        with pytest.raises(TypeError):
            ShapeKey(True, 1, 1)

    def test_invalid_dtype_rejected(self):
        with pytest.raises(ValueError):
            ShapeKey(1, 1, 1, "   ")
        with pytest.raises(TypeError):
            ShapeKey(1, 1, 1, None)

    def test_str_and_parse_round_trip(self):
        key = ShapeKey(4, 256, 1024, "float16")
        assert str(key) == "4x256x1024:float16"
        assert ShapeKey.parse(str(key)) == key
        assert ShapeKey.parse("4x256x1024") == ShapeKey(4, 256, 1024)

    @pytest.mark.parametrize("text", ["4x256", "axbxc", "4x-1x8", ""])
    def test_parse_rejects_garbage(self, text):
        with pytest.raises(ValueError):
            ShapeKey.parse(text)


class TestRoundingHelpers:
    def test_power_of_two_at_least(self):
        values = [1, 2, 3, 5, 8, 9, 1000]
        expected = [1, 2, 4, 8, 8, 16, 1024]
        assert [power_of_two_at_least(value) for value in values] == expected

    def test_round_up(self):
        assert round_up(1, 64) == 64
        assert round_up(64, 64) == 64
        assert round_up(65, 64) == 128

    def test_helpers_reject_non_positive_input(self):
        with pytest.raises(ValueError):
            power_of_two_at_least(0)
        with pytest.raises(ValueError):
            round_up(1, 0)
        with pytest.raises(ValueError):
            round_up(0, 8)


class TestCoerceKey:
    def test_accepts_every_documented_form(self):
        expected = ShapeKey(1, 8, 8, "float16")
        assert coerce_key(expected) is expected
        assert coerce_key((1, 8, 8, "float16")) == expected
        assert coerce_key([1, 8, 8]) == ShapeKey(1, 8, 8)
        assert coerce_key("1x8x8:float16") == expected

    def test_rejects_wrong_shapes_and_types(self):
        with pytest.raises(ValueError):
            coerce_key((1, 8))
        with pytest.raises(TypeError):
            coerce_key(3.5)


# ---------------------------------------------------------------------------
# BucketPolicy
# ---------------------------------------------------------------------------


class TestBucketPolicy:
    def test_default_rounds_each_axis_to_a_power_of_two(self):
        policy = BucketPolicy()
        assert policy.bucket((3, 7, 700)) == ShapeKey(4, 8, 1024)

    def test_multiple_rounding(self):
        policy = BucketPolicy(batch_step=4, sequence_step=64, hidden_step=128)
        assert policy.bucket((1, 1, 1)) == ShapeKey(4, 64, 128)
        assert policy.bucket((5, 65, 129)) == ShapeKey(8, 128, 256)

    def test_exact_policy_is_the_identity(self):
        policy = BucketPolicy.exact()
        key = ShapeKey(3, 7, 700, "float16")
        assert policy.bucket(key) == key

    def test_dtype_survives_bucketing(self):
        policy = BucketPolicy.exact()
        half = policy.bucket((1, 8, 8, "float16"))
        single = policy.bucket((1, 8, 8, "float32"))
        assert half != single
        assert half.dtype == "float16"

    def test_bucket_accepts_tuples_and_strings(self):
        policy = BucketPolicy.exact()
        assert policy.bucket((1, 8, 8)) == policy.bucket("1x8x8:float32")

    def test_invalid_steps_rejected(self):
        with pytest.raises(ValueError):
            BucketPolicy(sequence_step=-2)
        with pytest.raises(TypeError):
            BucketPolicy(sequence_step=1.5)
        with pytest.raises(TypeError):
            BucketPolicy(batch_step=True)

    def test_describe_mentions_every_step(self):
        text = BucketPolicy(2, 64, 128).describe()
        assert "batch_step=2" in text
        assert "sequence_step=64" in text
        assert "hidden_step=128" in text


# ---------------------------------------------------------------------------
# ShapeRouter
# ---------------------------------------------------------------------------


class TestShapeRouter:
    def test_ids_follow_first_seen_order(self):
        router = ShapeRouter(BucketPolicy.exact())
        assert router.graph_id((1, 128, 768)) == 0
        assert router.graph_id((1, 256, 768)) == 1
        assert router.graph_id((1, 128, 768)) == 0

    def test_default_policy_shares_an_octave(self):
        router = ShapeRouter()
        first = router.graph_id((1, 100, 700))
        assert router.graph_id((1, 128, 1024)) == first
        assert router.graph_id((1, 200, 2000)) != first

    def test_stats_counts_hits_and_misses(self):
        router = ShapeRouter(BucketPolicy.exact())
        router.graph_id((1, 8, 8))
        router.graph_id((1, 8, 8))
        router.graph_id((2, 8, 8))
        assert router.stats() == {"shapes": 2, "shape_hits": 1, "shape_misses": 2}

    def test_lookup_never_allocates(self):
        router = ShapeRouter(BucketPolicy.exact())
        assert router.lookup((1, 8, 8)) is None
        assert len(router) == 0
        assert router.stats()["shape_misses"] == 0

    def test_bucket_for(self):
        router = ShapeRouter(BucketPolicy.exact())
        identifier = router.graph_id((1, 8, 8, "float16"))
        assert router.bucket_for(identifier) == ShapeKey(1, 8, 8, "float16")
        with pytest.raises(KeyError):
            router.bucket_for(identifier + 1)
        with pytest.raises(TypeError):
            router.bucket_for("0")

    def test_contains_and_iteration(self):
        router = ShapeRouter(BucketPolicy.exact())
        router.graph_id((1, 8, 8))
        router.graph_id((2, 8, 8))
        assert (1, 8, 8) in router
        assert (3, 8, 8) not in router
        assert 42 not in router
        assert [identifier for identifier, _ in router] == [0, 1]

    def test_malformed_string_keys_are_rejected_not_ignored(self):
        router = ShapeRouter(BucketPolicy.exact())
        with pytest.raises(ValueError):
            "not a shape" in router

    def test_clear_resets_ids_and_counters(self):
        router = ShapeRouter(BucketPolicy.exact())
        router.graph_id((1, 8, 8))
        router.clear()
        assert len(router) == 0
        assert router.graph_id((2, 4, 4)) == 0
        assert router.stats() == {"shapes": 1, "shape_hits": 0, "shape_misses": 1}

    def test_invalid_policy_type_rejected(self):
        with pytest.raises(TypeError):
            ShapeRouter(policy="exact")


# ---------------------------------------------------------------------------
# Engine: capture
# ---------------------------------------------------------------------------


class TestEngineCapture:
    def test_capture_forwards_the_bucketed_key(self, engine):
        with engine.capture((1, 100, 700)) as graph_id:
            assert graph_id == 0
        assert call_names(engine) == ["begin_capture", "end_capture"]
        _, identifier, shape_key, hot_update = engine._raw.calls[0]
        assert (identifier, shape_key, hot_update) == (0, "1x128x1024:float32", False)

    def test_capture_state_is_visible_inside_the_region(self, engine):
        with engine.capture((1, 8, 8)):
            assert engine.is_capturing((1, 8, 8))
            assert not engine.has((1, 8, 8))
        assert not engine.is_capturing((1, 8, 8))
        assert engine.has((1, 8, 8))

    def test_hot_update_is_forwarded(self, engine):
        with engine.capture((1, 8, 8), hot_update=True):
            pass
        assert engine._raw.calls[0][3] is True

    def test_body_exception_finishes_and_discards_the_capture(self, engine):
        with pytest.raises(ZeroDivisionError):
            with engine.capture((1, 8, 8)):
                raise ZeroDivisionError("boom")
        assert call_names(engine) == ["begin_capture", "end_capture", "reset"]
        assert engine.graphs() == []
        assert not engine.has((1, 8, 8))

    def test_engine_stays_usable_after_an_abandoned_capture(self, engine):
        with pytest.raises(RuntimeError):
            with engine.capture((1, 8, 8)):
                raise RuntimeError("body failed")
        with engine.capture((1, 8, 8)) as graph_id:
            assert graph_id == 0
        assert engine.has((1, 8, 8))

    def test_explicit_begin_end_capture(self, engine):
        assert engine.begin_capture("1x8x8:float16") == 0
        assert engine.is_capturing((1, 8, 8, "float16"))
        assert engine.end_capture((1, 8, 8, "float16")) == 0
        assert engine.has((1, 8, 8, "float16"))

    def test_capture_stream_is_exposed_for_the_captured_launches(self, engine):
        """Work has to be launched on the capture stream to be recorded."""
        assert engine.capture_stream() == 0xCAFE

    def test_capture_stream_is_refused_once_the_engine_is_closed(self, engine):
        engine.close()
        with pytest.raises(EngineClosed):
            engine.capture_stream()

    def test_two_shapes_get_two_graphs_under_exact_policy(self, exact_engine):
        with exact_engine.capture((1, 128, 768)):
            pass
        with exact_engine.capture((1, 256, 768)):
            pass
        assert exact_engine.graphs() == [0, 1]

    def test_one_bucket_serves_a_whole_octave(self, engine):
        with engine.capture((1, 100, 700)) as first:
            pass
        with engine.capture((1, 128, 1024)) as second:
            pass
        assert first == second
        assert engine.graphs() == [0]


# ---------------------------------------------------------------------------
# Engine: replay, update, describe
# ---------------------------------------------------------------------------


class TestEngineExecution:
    def test_replay_forwards_the_stream_token(self, engine):
        with engine.capture((1, 8, 8)):
            pass
        engine.replay((1, 8, 8), stream_token=0x2A)
        assert engine._raw.calls[-1] == ("replay", 0, 0x2A)
        assert engine._raw.replays == 1

    def test_run_is_an_alias_for_replay(self, engine):
        with engine.capture((1, 8, 8)):
            pass
        engine.run((1, 8, 8))
        engine.replay((1, 8, 8))
        assert engine._raw.replays == 2

    def test_replay_without_a_capture_surfaces_the_native_error(self, engine):
        with pytest.raises(RuntimeError):
            engine.replay((1, 8, 8))

    def test_update_merges_the_staging_graph(self, engine):
        with engine.capture((1, 8, 8)):
            pass
        engine.update((1, 8, 8))
        assert ("update_graph", 0) in engine._raw.calls

    def test_describe_returns_json_for_the_routed_graph(self, engine):
        with engine.capture((1, 8, 8)):
            pass
        assert json.loads(engine.describe((1, 8, 8))) == {"graph_id": 0, "has_exec": True}

    def test_synchronize_delegates(self, engine):
        engine.synchronize()
        assert engine._raw.synchronize_calls == 1


# ---------------------------------------------------------------------------
# Engine: static buffers
# ---------------------------------------------------------------------------


class TestEngineBuffers:
    def test_register_buffer_returns_the_native_address(self, engine):
        address = engine.register_buffer("hidden", 4096)
        assert address == engine.buffer_ptr("hidden")
        assert engine.buffer_bytes("hidden") == 4096
        assert engine.buffers() == ["hidden"]
        assert engine._raw.calls[0] == ("register_static_buffer", "hidden", 4096)

    def test_register_buffer_validates_arguments(self, engine):
        with pytest.raises(ValueError):
            engine.register_buffer("", 16)
        with pytest.raises(ValueError):
            engine.register_buffer("x", 0)
        with pytest.raises(ValueError):
            engine.register_buffer("x", True)
        with pytest.raises(ValueError):
            engine.register_buffer("x", -8)

    def test_copy_helpers_forward_the_registered_length(self, engine):
        engine.register_buffer("hidden", 4096)
        engine.copy_into("hidden", 0xABCD)
        engine.read_buffer("hidden", 0xBCDE)
        engine.copy_into("hidden", 0xABCD, bytes=128)
        copies = [call for call in engine._raw.calls if call[0].startswith("copy_")]
        assert copies == [
            ("copy_to_static_input", "hidden", 0xABCD, 4096, 0),
            ("copy_from_static", "hidden", 0xBCDE, 4096, 0),
            ("copy_to_static_input", "hidden", 0xABCD, 128, 0),
        ]

    def test_copy_rejects_bad_buffers_addresses_and_sizes(self, engine):
        engine.register_buffer("hidden", 64)
        with pytest.raises(KeyError):
            engine.copy_into("missing", 0x10)
        with pytest.raises(ValueError):
            engine.copy_into("hidden", 0)
        with pytest.raises(ValueError):
            engine.read_buffer("hidden", False)
        with pytest.raises(ValueError):
            engine.copy_into("hidden", 0x10, bytes=65)
        with pytest.raises(TypeError):
            engine.copy_into("hidden", 0x10, bytes=1.5)

    def test_zero_and_release(self, engine):
        engine.register_buffer("hidden", 64)
        engine.zero_buffer("hidden", stream_token=7)
        assert engine._raw.calls[-1] == ("zero_static_buffer", "hidden", 7)
        engine.release_buffer("hidden")
        assert engine.buffers() == []
        engine.release_buffer("hidden")
        assert call_names(engine).count("release_static_buffer") == 1

    def test_released_buffers_are_no_longer_readable(self, engine):
        engine.register_buffer("hidden", 64)
        engine.release_buffer("hidden")
        with pytest.raises(KeyError):
            engine.read_buffer("hidden", 0x10)


# ---------------------------------------------------------------------------
# Engine: lifecycle and stats
# ---------------------------------------------------------------------------


class TestEngineLifecycle:
    def test_stats_merge_native_and_facade_counters(self, engine):
        engine.register_buffer("hidden", 16)
        with engine.capture((1, 8, 8)):
            pass
        engine.replay((1, 8, 8))
        engine.replay((1, 8, 8))
        stats = engine.stats()
        assert stats["replays"] == 2
        assert stats["graphs"] == 1
        assert stats["shapes"] == 1
        assert stats["shape_hits"] == 2
        assert stats["shape_misses"] == 1
        assert stats["buffers"] == 1
        assert stats["device_index"] == -1
        assert stats["backend_version"] == "1.0.0-fake"
        assert "sequence_step" in stats["bucket_policy"]

    def test_reset_keeps_the_shape_mapping(self, engine):
        with engine.capture((1, 8, 8)):
            pass
        engine.reset((1, 8, 8))
        assert engine.graphs() == []
        assert not engine.has((1, 8, 8))
        with engine.capture((1, 8, 8)) as graph_id:
            assert graph_id == 0

    def test_reset_of_an_unknown_shape_is_a_no_op(self, engine):
        before = len(engine._raw.calls)
        engine.reset((9, 9, 9))
        assert len(engine._raw.calls) == before

    def test_reset_all_clears_every_graph(self, exact_engine):
        with exact_engine.capture((1, 8, 8)):
            pass
        with exact_engine.capture((2, 8, 8)):
            pass
        exact_engine.reset_all()
        assert exact_engine.graphs() == []
        assert exact_engine._raw.reset_all_calls == 1

    def test_repr_reports_state(self, engine):
        assert "open" in repr(engine)
        engine.close()
        assert "closed" in repr(engine)

    def test_close_is_idempotent_and_blocks_further_use(self, engine):
        engine.close()
        engine.close()
        assert engine._raw.reset_all_calls == 1
        with pytest.raises(EngineClosed):
            engine.replay((1, 8, 8))
        with pytest.raises(EngineClosed):
            engine.register_buffer("x", 8)
        with pytest.raises(EngineClosed):
            engine.stats()

    def test_context_manager_closes_the_engine(self):
        with Engine(backend=FAKE_BACKEND) as instance:
            instance.register_buffer("hidden", 8)
            assert instance.buffers() == ["hidden"]
        assert instance.closed
        assert instance._raw.reset_all_calls == 1
        with pytest.raises(EngineClosed):
            instance.graphs()

    def test_injected_backend_is_exposed(self, engine):
        assert engine.backend is FAKE_BACKEND
        assert engine.device_index == -1


# ---------------------------------------------------------------------------
# Loader and module-level helpers
# ---------------------------------------------------------------------------


class TestBackendLoader:
    def test_missing_extension_raises_with_the_build_hint(self):
        with pytest.raises(BackendUnavailable) as error:
            load_backend("wddm_bypass._module_that_does_not_exist")
        message = str(error.value)
        assert BUILD_HINT in message
        assert "build_ext" in message

    def test_backend_available_matches_importability(self):
        assert backend_available("json") is True
        assert backend_available("wddm_bypass._module_that_does_not_exist") is False

    def test_load_backend_returns_the_imported_module(self):
        assert load_backend("json").__name__ == "json"

    def test_engine_without_a_backend_surfaces_the_build_hint(self):
        """``Engine()`` surfaces the build hint when the extension cannot load.

        A built checkout would otherwise have to skip this: the missing-backend
        path is reached by making ``importlib.import_module`` fail for the
        extension name, which is exactly the ImportError the loader sees when
        the ``.pyd`` has never been built. No test-only hook is added to the
        loader, so the real message-building code is exercised either way.
        """
        real_import_module = importlib.import_module

        def import_module(name, *args, **kwargs):
            if name == EXTENSION_NAME:
                raise ImportError("simulated missing extension")
            return real_import_module(name, *args, **kwargs)

        with mock.patch.object(importlib, "import_module", import_module):
            with pytest.raises(BackendUnavailable) as error:
                Engine()

        message = str(error.value)
        assert EXTENSION_NAME in message
        assert BUILD_HINT in message
        assert "build_ext" in message

    def test_build_info_reports_the_extension_state(self):
        if backend_available():
            info = build_info()
            assert isinstance(info, dict)
            assert info.get("version")
        else:
            with pytest.raises(BackendUnavailable):
                build_info()


def test_module_level_capture_context_matches_the_method():
    instance = Engine(backend=FAKE_BACKEND)
    try:
        with capture_context(instance, (1, 8, 8), hot_update=True) as graph_id:
            assert instance.is_capturing((1, 8, 8))
        assert graph_id == 0
        assert instance._raw.calls[0][3] is True
    finally:
        instance.close()


# ---------------------------------------------------------------------------
# Packaging and source hygiene
# ---------------------------------------------------------------------------

CSRC_DIR = NATIVE_DIR / "csrc"
SOURCE_SUFFIXES = {".cu", ".cuh", ".h", ".hpp", ".cpp"}


def test_cuda_sources_and_header_exist():
    expected = [
        "csrc/bypass_core.cpp",
        "csrc/graph_engine.cu",
        "csrc/torch_interop.cpp",
        "csrc/include/graph_manager.h",
        "setup.py",
    ]
    for relative in expected:
        assert (NATIVE_DIR / relative).is_file(), relative


def test_cuda_sources_are_ascii_only():
    """MSVC + nvcc on Windows mis-parse non-ASCII sources without /utf-8."""
    assert CSRC_DIR.is_dir()
    offenders = []
    for path in sorted(CSRC_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        try:
            path.read_bytes().decode("ascii")
        except UnicodeDecodeError as error:
            offenders.append(f"{path.relative_to(NATIVE_DIR)} at byte {error.start}")
    assert not offenders, offenders


def test_version_is_declared_consistently():
    """``setup.py``, the package and the extension must agree on v1.0.0."""
    setup_py = (NATIVE_DIR / "setup.py").read_text(encoding="utf-8")
    declared = re.search(r'^\s*version="([^"]+)"', setup_py, re.MULTILINE)
    assert declared is not None, "setup.py has no version= keyword"
    assert declared.group(1) == __version__
    core = (CSRC_DIR / "bypass_core.cpp").read_text(encoding="ascii")
    assert f'"{__version__}"' in core


def test_every_bound_method_is_declared_in_the_header():
    """pybind11 binds members by pointer, so a missing declaration is a build error.

    Lambdas (``__repr__``, ``build_info``) have no C++ declaration to look for.
    """
    core = (CSRC_DIR / "bypass_core.cpp").read_text(encoding="ascii")
    header = (CSRC_DIR / "include" / "graph_manager.h").read_text(encoding="ascii")
    bound = sorted(set(re.findall(r'\.def\(\s*"([A-Za-z_][A-Za-z0-9_]*)"', core)))
    lambdas = set(re.findall(r'\.def\(\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*,\s*\[', core))
    assert bound, "no pybind11 method bindings found"
    missing = [
        name
        for name in bound
        if not name.startswith("__") and name not in lambdas and f"{name}(" not in header
    ]
    assert not missing, f"bound but not declared in graph_manager.h: {missing}"


# ---------------------------------------------------------------------------
# Real extension: capture, replay, buffers
# ---------------------------------------------------------------------------
# Everything below needs the compiled pybind11 module, and most of it needs a
# device that can instantiate graphs. The skips carry the reason, so a green
# run on a machine without CUDA never hides an untested engine.

NATIVE_SURFACE = (
    "begin_capture",
    "end_capture",
    "is_capturing",
    "replay",
    "update_graph",
    "reset",
    "reset_all",
    "has_graph",
    "list_graphs",
    "graph_id_for_shape",
    "register_static_buffer",
    "static_buffer_ptr",
    "static_buffer_bytes",
    "release_static_buffer",
    "copy_to_static_input",
    "zero_static_buffer",
    "copy_from_static",
    "launch_fused_affine_relu",
    "describe",
    "capture_stream_ptr",
    "stats",
    "synchronize",
)

try:
    NATIVE_BACKEND = load_backend()
    NATIVE_ERROR = ""
except BackendUnavailable as error:
    NATIVE_BACKEND = None
    NATIVE_ERROR = str(error).splitlines()[0]


def _native_status():
    """``(ready, reason)`` for the GPU half of the suite."""
    if NATIVE_BACKEND is None:
        return False, NATIVE_ERROR or "compiled extension is not importable"
    try:
        if not bool(NATIVE_BACKEND.cuda_graph_supported()):
            return False, "this device or driver cannot instantiate CUDA graphs"
    except Exception as error:  # noqa: BLE001 - any probe failure means "unusable"
        return False, f"capability probe failed: {error}"
    return True, ""


NATIVE_READY, NATIVE_REASON = _native_status()

requires_extension = pytest.mark.skipif(
    NATIVE_BACKEND is None, reason=f"compiled extension missing ({NATIVE_ERROR})"
)
requires_gpu = pytest.mark.skipif(
    not NATIVE_READY, reason=f"CUDA graph engine unavailable ({NATIVE_REASON})"
)


def native_engine_error():
    """The extension's ``EngineError`` type, or a broad fallback."""
    return getattr(NATIVE_BACKEND, "EngineError", Exception)


@pytest.fixture()
def native_engine():
    instance = Engine(backend=NATIVE_BACKEND)
    try:
        yield instance
    finally:
        instance.close()


def upload(native, name, values):
    """Copy a NumPy array into a registered static buffer."""
    np = pytest.importorskip("numpy")
    native.register_buffer(name, values.nbytes)
    native.copy_into(name, values.ctypes.data, stream_token=0)
    native.synchronize()
    return np


def download(native, name, count):
    """Copy a static buffer back into a fresh float32 array."""
    np = pytest.importorskip("numpy")
    out = np.zeros(count, dtype=np.float32)
    native.read_buffer(name, out.ctypes.data, stream_token=0)
    native.synchronize()
    return out


@requires_extension
def test_extension_exposes_the_documented_surface():
    missing = [
        name for name in NATIVE_SURFACE if not hasattr(NATIVE_BACKEND.WDDMBypassEngine, name)
    ]
    assert not missing, missing
    for name in (
        "launch_scale",
        "launch_add",
        "launch_relu",
        "launch_fused_affine_relu",
        "build_info",
    ):
        assert hasattr(NATIVE_BACKEND, name), name
    assert NATIVE_BACKEND.build_info()["version"] == __version__
    assert NATIVE_BACKEND.__version__ == __version__


@requires_gpu
def test_capture_covers_the_work_and_replay_repeats_it(native_engine):
    np = pytest.importorskip("numpy")
    initial = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    upload(native_engine, "data", initial)

    stream = native_engine.capture_stream()
    assert stream != 0, "the capture stream must exist before the capture begins"
    with native_engine.capture((1, 4, 1)):
        NATIVE_BACKEND.launch_scale(native_engine.buffer_ptr("data"), 4, 2.0, stream)
    native_engine.synchronize()

    # Stream capture records the work; it does not execute it.
    np.testing.assert_allclose(download(native_engine, "data", 4), [1, 2, 3, 4])

    # One replay runs the whole recorded graph: the same scale, exactly once.
    native_engine.replay((1, 4, 1))
    native_engine.synchronize()
    np.testing.assert_allclose(download(native_engine, "data", 4), [2, 4, 6, 8])

    assert native_engine.has((1, 4, 1))
    assert native_engine.graphs() == [0]
    assert json.loads(native_engine.describe((1, 4, 1)))


@requires_gpu
def test_static_buffer_zero_and_release_round_trip(native_engine):
    np = pytest.importorskip("numpy")
    upload(native_engine, "data", np.array([5.0, 6.0], dtype=np.float32))
    native_engine.zero_buffer("data")
    native_engine.synchronize()
    np.testing.assert_allclose(download(native_engine, "data", 2), [0.0, 0.0])
    native_engine.release_buffer("data")
    assert native_engine.buffers() == []
    with pytest.raises(KeyError):
        native_engine.read_buffer("data", 0x10)


@requires_gpu
def test_reset_discards_the_executable(native_engine):
    address = native_engine.register_buffer("probe", 16)
    with native_engine.capture((1, 8, 1)):
        NATIVE_BACKEND.launch_relu(address, 4, native_engine.capture_stream())
    native_engine.reset((1, 8, 1))
    assert native_engine.graphs() == []
    assert not native_engine.has((1, 8, 1))
    with pytest.raises(native_engine_error()):
        native_engine.replay((1, 8, 1))


@requires_gpu
def test_buffer_bounds_are_enforced_before_the_native_call(native_engine):
    native_engine.register_buffer("small", 16)
    with pytest.raises(ValueError):
        native_engine.copy_into("small", 0x1000, bytes=17)
    with pytest.raises(ValueError):
        native_engine.copy_into("small", 0)


@requires_gpu
def test_replay_submits_a_chain_in_one_launch(native_engine):
    """R8 (minimal benchmark): one graph submission versus N kernel launches.

    Timings are printed, and the only assertion is that replay did not become an
    order of magnitude slower, so a noisy machine cannot fail the suite while a
    real regression still shows up.
    """
    np = pytest.importorskip("numpy")
    depth = 8
    iterations = 200
    upload(native_engine, "data", np.ones(4096, dtype=np.float32))
    destination = native_engine.buffer_ptr("data")
    stream = native_engine.capture_stream()

    with native_engine.capture((1, 4096, 1)):
        for _ in range(depth):
            NATIVE_BACKEND.launch_scale(destination, 4096, 1.0, stream)
    native_engine.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        native_engine.replay((1, 4096, 1))
    native_engine.synchronize()
    graph_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        for _ in range(depth):
            NATIVE_BACKEND.launch_scale(destination, 4096, 1.0, 0)
    native_engine.synchronize()
    direct_seconds = time.perf_counter() - start

    print(
        f"\n  replay {iterations} x {depth} kernels: {graph_seconds:.4f}s "
        f"({graph_seconds / iterations * 1e6:.1f} us per submission)"
    )
    print(
        f"  direct {iterations * depth} launches: {direct_seconds:.4f}s "
        f"({direct_seconds / (iterations * depth) * 1e6:.1f} us per launch)"
    )

    assert graph_seconds > 0.0 and direct_seconds > 0.0
    assert graph_seconds <= direct_seconds * 10.0, "graph replay regressed by more than 10x"


# ---------------------------------------------------------------------------
# vram_core integration (v2.6.0): the memory side drives the submission side
# ---------------------------------------------------------------------------
# The engine suite above exercises ``wddm_bypass`` on its own.  The cases below
# are the joint ones: a real ``PagedKVCacheManager`` owns the blocks, the bridge
# registers its pool as a static buffer, and the pool geometry becomes the graph
# shape key, so one capture serves every sequence that fits the pool.

from vram_core.vram_optimizer import PagedKVCacheManager  # noqa: E402
from vram_core.wddm_bridge import (  # noqa: E402
    BridgeClosed,
    InsufficientVRAMError,
    WddmExecutionBridge,
    device_pointer,
    nbytes_of,
    wddm_available,
)


class PressureProbe:
    """Stand-in for ``VRAMOptimizer.can_allocate`` - the same one-argument contract."""

    def __init__(self, allow=True):
        self.allow = allow
        self.requests = []

    def can_allocate(self, required_mb):
        self.requests.append(int(required_mb))
        return self.allow


def make_paged_cache(**overrides):
    """A CPU-backed paged cache, so the bridge is testable without a GPU."""
    kwargs = dict(
        num_blocks=32,
        block_size=8,
        num_heads=4,
        head_dim=16,
        max_sequences=4,
        use_cuda=False,
    )
    kwargs.update(overrides)
    return PagedKVCacheManager(**kwargs)


def test_bridge_binds_a_vram_core_pool_as_a_wddm_static_buffer():
    """The pool geometry becomes the graph key, and every replay routes by it."""
    cache = make_paged_cache()
    cache.allocate_sequence("session-1")
    cache.append("session-1", np.arange(2 * 4 * 16, dtype=np.float32).reshape(2, 4, 16))

    engine = Engine(backend=FAKE_BACKEND, policy=BucketPolicy.exact())
    bridge = WddmExecutionBridge(engine=engine)
    try:
        bound = bridge.bind_paged_kv_cache(cache)
        pool_bytes = cache.num_blocks * cache.num_heads * cache.block_size * cache.head_dim * 4

        assert bound.name == "kv_pool"
        assert bound.source == "PagedKVCacheManager"
        assert bound.nbytes == cache.pool_bytes == pool_bytes
        assert ("register_static_buffer", "kv_pool", pool_bytes) in engine._raw.calls
        assert engine.buffer_ptr("kv_pool") == bound.address

        # the key *is* the pool geometry: elements x 4 bytes == pool bytes
        assert bound.shape_key == ShapeKey(
            cache.num_blocks, cache.num_heads * cache.block_size, cache.head_dim
        )
        assert bound.shape_key.elements * 4 == pool_bytes

        # one graph per geometry, recorded once, replayed once per decode step
        with bridge.capture(bound.shape_key) as graph_id:
            engine.buffer_ptr("kv_pool")
        assert graph_id == 0
        begin = [call for call in engine._raw.calls if call[0] == "begin_capture"]
        assert begin and begin[0][3] is False          # not a hot update
        assert bridge.route(bound.shape_key) == 0
        assert bridge.has(bound.shape_key)

        for _ in range(3):
            assert bridge.replay_for_cache(cache) == bound.shape_key
        bridge.synchronize()
        assert engine._raw.replays == 3

        measured = bridge.stats()
        assert measured["bound_buffers"] == 1
        assert measured["static_bytes"] == pool_bytes
        assert measured["kv_pool_bytes"] == cache.pool_bytes
        assert measured["kv_backend"] == "numpy"
        assert measured["kv_used_blocks"] == 1
        assert measured["kv_free_blocks"] == cache.num_blocks - 1
        assert measured["kv_shape"] == bound.shape_key.as_tuple()
        assert measured["replays"] == 3
    finally:
        bridge.close()
        engine.close()


def test_bridge_gates_bindings_and_merges_a_hot_update():
    """Budget + pressure gates refuse before allocation; hot update keeps routing."""
    cache = make_paged_cache(num_blocks=16, block_size=4, num_heads=2, head_dim=8)
    probe = PressureProbe(allow=True)
    engine = Engine(backend=FAKE_BACKEND, policy=BucketPolicy.exact())
    bridge = WddmExecutionBridge(
        engine=engine,
        optimizer=probe,
        budget_bytes=cache.pool_bytes + 64,      # room for the pool and one tensor
    )
    try:
        bound = bridge.bind_paged_kv_cache(cache)
        assert probe.requests == [1]                      # 4 KiB rounds up to 1 MiB
        assert bridge.committed_bytes == cache.pool_bytes

        # a second pool cannot fit the byte budget, and nothing is allocated
        with pytest.raises(InsufficientVRAMError):
            bridge.bind_paged_kv_cache(cache, name="kv_pool_2")
        assert "kv_pool_2" not in engine._raw.buffers
        assert bridge.committed_bytes == cache.pool_bytes

        # an explicit VRAM refusal is reported through the same gate
        refused = WddmExecutionBridge(engine=engine, optimizer=PressureProbe(allow=False))
        try:
            with pytest.raises(InsufficientVRAMError):
                refused.bind_paged_kv_cache(cache, name="kv_pool_3")
            assert "kv_pool_3" not in engine._raw.buffers
        finally:
            refused.close()

        # a host tensor is sized and registered, but never copied as a device seed
        activation = bridge.bind_tensor("activations", np.zeros(16, dtype=np.float32))
        assert activation.nbytes == 64
        assert activation.shape_key is None
        assert ("register_static_buffer", "activations", 64) in engine._raw.calls
        assert not any(call[0] == "copy_to_static_input" for call in engine._raw.calls)
        with pytest.raises(TypeError):
            bridge.upload("activations", np.zeros(16, dtype=np.float32))
        bridge.upload("activations", 0x4000)
        assert ("copy_to_static_input", "activations", 0x4000, 64, 0) in engine._raw.calls

        # hot update merges the same topology instead of rebuilding the slot
        with bridge.capture(bound.shape_key, hot_update=True):
            engine.buffer_ptr("kv_pool")
        begins = [call for call in engine._raw.calls if call[0] == "begin_capture"]
        assert begins[-1][3] is True
        assert bridge.has(bound.shape_key)

        # release is idempotent and frees exactly what was committed
        assert bridge.release("kv_pool") is True
        assert bridge.release("kv_pool") is False
        assert bridge.committed_bytes == activation.nbytes
        assert "kv_pool" not in engine._raw.buffers
    finally:
        bridge.close()
        engine.close()

    with pytest.raises(BridgeClosed):
        bridge.stats()


def test_bridge_helpers_measure_buffers_and_refuse_host_pointers():
    """The helpers are the only place that touches raw sizes and addresses."""
    assert nbytes_of(np.zeros(8, dtype=np.float32)) == 32
    assert nbytes_of(bytearray(7)) == 7
    assert device_pointer(0x2000) == 0x2000
    with pytest.raises(TypeError):
        device_pointer(np.zeros(4, dtype=np.float32))   # host address, not a device one
    with pytest.raises(TypeError):
        nbytes_of(None)
    assert isinstance(wddm_available(), bool)


# ---------------------------------------------------------------------------
# v2.9.0: vectorised fused affine + ReLU (one kernel, 12 bytes per element)
# ---------------------------------------------------------------------------


def fused_reference(a, b, scale):
    """``max(0, a * scale + b)`` in float32 - the definition of the fused op."""
    return np.maximum(np.float32(0.0), a * np.float32(scale) + b)


@requires_gpu
def test_fused_affine_relu_matches_the_reference(native_engine):
    rng = np.random.default_rng(7)
    # 4099 is deliberately not a multiple of 4: it exercises the float4 tail.
    count = 4099
    scale = 1.75
    a = rng.standard_normal(count).astype(np.float32)
    b = rng.standard_normal(count).astype(np.float32)
    upload(native_engine, "a", a)
    upload(native_engine, "b", b)
    upload(native_engine, "out", np.zeros(count, dtype=np.float32))

    native_engine.launch_fused_affine_relu("out", "a", "b", scale, count)
    native_engine.synchronize()

    assert native_engine.stats()["fused_launches"] == 1
    np.testing.assert_allclose(
        download(native_engine, "out", count),
        fused_reference(a, b, scale),
        rtol=1e-5,
        atol=1e-5,
    )


@requires_gpu
def test_fused_affine_relu_defaults_to_the_shared_buffer_length(native_engine):
    values = np.arange(1, 257, dtype=np.float32)
    offsets = np.full(256, 0.5, dtype=np.float32)
    upload(native_engine, "a", values)
    upload(native_engine, "b", offsets)
    upload(native_engine, "out", np.zeros(256, dtype=np.float32))

    native_engine.launch_fused_affine_relu("out", "a", "b", 2.0)

    native_engine.synchronize()
    np.testing.assert_allclose(
        download(native_engine, "out", 256),
        fused_reference(values, offsets, 2.0),
        rtol=1e-5,
        atol=1e-5,
    )


@requires_gpu
def test_fused_affine_relu_rejects_unknown_or_too_small_buffers(native_engine):
    upload(native_engine, "a", np.ones(8, dtype=np.float32))
    upload(native_engine, "b", np.ones(8, dtype=np.float32))
    upload(native_engine, "out", np.ones(8, dtype=np.float32))

    with pytest.raises(native_engine_error()):
        native_engine.launch_fused_affine_relu("out", "a", "missing", 1.0, 8)
    with pytest.raises(native_engine_error()):
        native_engine.launch_fused_affine_relu("out", "a", "b", 1.0, 9)  # 36 > 32 bytes
    with pytest.raises(ValueError):
        native_engine.launch_fused_affine_relu("out", "a", "b", 1.0, 0)


@requires_gpu
def test_fused_affine_relu_is_captured_as_a_single_graph_node(native_engine):
    count = 1024
    a = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    b = np.full(count, 0.25, dtype=np.float32)
    upload(native_engine, "a", a)
    upload(native_engine, "b", b)
    upload(native_engine, "out", np.zeros(count, dtype=np.float32))
    key = (1, count, 1)

    # No stream token: inside a capture, 0 means the stream being recorded.
    with native_engine.capture(key):
        native_engine.launch_fused_affine_relu("out", "a", "b", 3.0, count)

    description = json.loads(native_engine.describe(key))
    assert description["node_count"] == 1
    assert description["has_exec"] is True

    native_engine.zero_buffer("out")
    native_engine.replay(key)
    native_engine.synchronize()
    np.testing.assert_allclose(
        download(native_engine, "out", count),
        fused_reference(a, b, 3.0),
        rtol=1e-5,
        atol=1e-5,
    )


def _median_ms(call, iters=40, warmup=10):
    """Wall-clock milliseconds per call, device-synchronised every round.

    CUDA events are not used on purpose: an event recorded on the default stream
    does not order against work submitted to another stream, so it would measure
    submission only and report a few microseconds for a 4 MiB kernel.
    """
    torch = pytest.importorskip("torch")
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iters):
            call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e3 / iters)
    samples.sort()
    return samples[len(samples) // 2]


@requires_gpu
def test_fused_kernel_breaks_the_4mib_bandwidth_wall(native_engine):
    """1,048,576 float32 (4 MiB): one fused kernel against three eager kernels.

    The v2.8.0 measurement for this size was 1.10x, because cutting the launch
    count alone does not move the HBM traffic. Fusing the three ops into one
    register-resident kernel drops the traffic from 28 to 12 bytes per element,
    so the same bandwidth carries 2.33x more work and the measured speedup
    leaves 1.10x behind.
    """
    torch = pytest.importorskip("torch")
    count = 1 << 20
    nbytes = count * 4
    scale = 1.75

    a = torch.randn(count, device="cuda", dtype=torch.float32)
    b = torch.randn(count, device="cuda", dtype=torch.float32)
    for name in ("a", "b", "out"):
        native_engine.register_buffer(name, nbytes)
    native_engine.copy_into("a", a.data_ptr())
    native_engine.copy_into("b", b.data_ptr())
    native_engine.synchronize()

    def eager_three():
        """Three independent launches: 28 bytes per element through HBM."""
        return torch.relu(torch.add(torch.mul(a, scale), b))

    stream = torch.cuda.Stream()
    stream_ptr = int(stream.cuda_stream)

    # Same three kernels, captured: this isolates the launch-count win only.
    three_key = (1, count, 2)
    with torch.cuda.stream(stream):
        native_engine.begin_capture(three_key, stream_ptr=stream_ptr)
        eager_three()
        native_engine.end_capture(three_key)
    torch.cuda.synchronize()

    # One fused kernel, captured: 12 bytes per element.
    fused_key = (1, count, 3)
    with torch.cuda.stream(stream):
        native_engine.begin_capture(fused_key, stream_ptr=stream_ptr)
        native_engine.launch_fused_affine_relu(
            "out", "a", "b", scale, count, stream_token=stream_ptr
        )
        native_engine.end_capture(fused_key)
    torch.cuda.synchronize()

    eager_ms = _median_ms(eager_three)
    three_ms = _median_ms(lambda: native_engine.replay(three_key, stream_token=stream_ptr))
    fused_ms = _median_ms(lambda: native_engine.replay(fused_key, stream_token=stream_ptr))

    # > 1.10x against the unfused eager path: the old ceiling is broken.
    assert fused_ms * 1.10 < eager_ms, (fused_ms, eager_ms)
    # ... and the win survives when the launch count is already amortised.
    assert fused_ms < three_ms, (fused_ms, three_ms)

    native_engine.replay(fused_key, stream_token=stream_ptr)
    native_engine.synchronize()
    np.testing.assert_allclose(
        download(native_engine, "out", count),
        torch.relu(a * scale + b).cpu().numpy(),
        rtol=1e-4,
        atol=1e-4,
    )


# ---------------------------------------------------------------------------
# v2.9.0: Windows DLL self-healing
# ---------------------------------------------------------------------------


class TestWindowsDllHealing:
    """The import surface is repaired once, before the first extension import."""

    def test_returns_a_list_and_never_repeats_itself(self):
        from wddm_bypass.backend import heal_windows_dll_environment

        first = heal_windows_dll_environment()
        assert isinstance(first, list)
        assert heal_windows_dll_environment() == []

    def test_platform_guard_is_a_no_op_off_windows(self, monkeypatch):
        from wddm_bypass import backend as backend_module

        monkeypatch.setattr(sys, "platform", "linux")
        assert backend_module.heal_windows_dll_environment() == []

    def test_registers_existing_prefixes_and_puts_them_on_path(self, monkeypatch):
        import os

        from wddm_bypass import backend as backend_module

        # Fresh "never healed" state; the handles stay in the real list so the
        # registered directories outlive this test.
        monkeypatch.setattr(backend_module, "_HEALED_DIRS", set())

        added = backend_module.heal_windows_dll_environment()
        if sys.platform != "win32":      # that branch is covered above
            assert added == []
            return

        assert added, "a Windows Conda prefix exposes at least one existing directory"
        assert all(os.path.isdir(directory) for directory in added)
        path_parts = {
            os.path.normcase(os.path.normpath(part))
            for part in os.environ.get("PATH", "").split(os.pathsep)
            if part
        }
        for directory in added:
            assert os.path.normcase(os.path.abspath(directory)) in path_parts
        assert backend_module.heal_windows_dll_environment() == []

    def test_load_backend_heals_before_importing(self, monkeypatch):
        from wddm_bypass import backend as backend_module

        calls = []
        monkeypatch.setattr(
            backend_module,
            "heal_windows_dll_environment",
            lambda: calls.append(1) or [],
        )
        backend_module.load_backend("wddm_bypass.bucketing")
        assert calls, "load_backend() must heal the DLL surface first"


# ---------------------------------------------------------------------------
# v2.9.2: warp-shuffle fused RMSNorm (one kernel, 12 bytes per element)
# ---------------------------------------------------------------------------


def rmsnorm_reference(x, weight, eps=1e-6):
    """``x * rsqrt(mean(x^2) + eps) * weight`` evaluated in float64."""
    values = np.asarray(x, dtype=np.float64)
    scale = np.asarray(weight, dtype=np.float64)
    mean_square = float(np.mean(values * values))
    return (values * (1.0 / np.sqrt(mean_square + eps)) * scale).astype(np.float32)


@requires_gpu
def test_fused_rmsnorm_matches_the_reference(native_engine):
    rng = np.random.default_rng(11)
    count = 4096  # a real hidden dimension (Llama-7B / Qwen-7B)
    x = rng.standard_normal(count).astype(np.float32)
    weight = rng.standard_normal(count).astype(np.float32)
    upload(native_engine, "x", x)
    upload(native_engine, "w", weight)
    upload(native_engine, "out", np.zeros(count, dtype=np.float32))

    native_engine.launch_fused_rmsnorm("out", "x", "w", 1e-6, count)
    native_engine.synchronize()

    assert native_engine.stats()["rmsnorm_launches"] == 1
    np.testing.assert_allclose(
        download(native_engine, "out", count),
        rmsnorm_reference(x, weight),
        rtol=1e-5,
        atol=1e-6,
    )


@requires_gpu
def test_fused_rmsnorm_handles_short_and_unaligned_rows(native_engine):
    """n below one warp, and n with a tail past the block, both stay exact."""
    for count in (7, 33, 1000, 4099):
        rng = np.random.default_rng(count)
        x = rng.standard_normal(count).astype(np.float32)
        weight = np.linspace(0.5, 1.5, count, dtype=np.float32)
        upload(native_engine, "x", x)
        upload(native_engine, "w", weight)
        upload(native_engine, "out", np.zeros(count, dtype=np.float32))

        native_engine.launch_fused_rmsnorm("out", "x", "w", 1e-6, count)
        native_engine.synchronize()

        np.testing.assert_allclose(
            download(native_engine, "out", count),
            rmsnorm_reference(x, weight),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"count={count}",
        )


@requires_gpu
def test_fused_rmsnorm_defaults_to_the_shared_buffer_length(native_engine):
    x = np.arange(1, 257, dtype=np.float32)
    weight = np.full(256, 0.75, dtype=np.float32)
    upload(native_engine, "x", x)
    upload(native_engine, "w", weight)
    upload(native_engine, "out", np.zeros(256, dtype=np.float32))

    native_engine.launch_fused_rmsnorm("out", "x", "w", 1e-6)

    native_engine.synchronize()
    np.testing.assert_allclose(
        download(native_engine, "out", 256),
        rmsnorm_reference(x, weight),
        rtol=1e-5,
        atol=1e-6,
    )


@requires_gpu
def test_fused_rmsnorm_rejects_unknown_or_too_small_buffers(native_engine):
    upload(native_engine, "x", np.ones(8, dtype=np.float32))
    upload(native_engine, "w", np.ones(8, dtype=np.float32))
    upload(native_engine, "out", np.ones(8, dtype=np.float32))

    with pytest.raises(native_engine_error()):
        native_engine.launch_fused_rmsnorm("out", "x", "missing", 1e-6, 8)
    with pytest.raises(native_engine_error()):
        native_engine.launch_fused_rmsnorm("out", "x", "w", 1e-6, 9)  # 36 > 32 bytes
    with pytest.raises(ValueError):
        native_engine.launch_fused_rmsnorm("out", "x", "w", 1e-6, 0)


@requires_gpu
def test_fused_rmsnorm_is_captured_as_a_single_graph_node(native_engine):
    count = 2048
    x = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    weight = np.linspace(0.5, 1.5, count, dtype=np.float32)
    upload(native_engine, "x", x)
    upload(native_engine, "w", weight)
    upload(native_engine, "out", np.zeros(count, dtype=np.float32))
    key = (1, count, 7)

    with native_engine.capture(key):
        native_engine.launch_fused_rmsnorm("out", "x", "w", 1e-6, count)

    description = json.loads(native_engine.describe(key))
    assert description["node_count"] == 1
    assert description["has_exec"] is True

    native_engine.zero_buffer("out")
    native_engine.replay(key)
    native_engine.synchronize()
    np.testing.assert_allclose(
        download(native_engine, "out", count),
        rmsnorm_reference(x, weight),
        rtol=1e-5,
        atol=1e-6,
    )


@requires_gpu
def test_fused_rmsnorm_cuts_the_eager_chain_latency(native_engine):
    """Hidden=4096: one fused launch against the eager four-launch chain.

    The eager chain is launch-bound at this size (four submissions for 16 KB of
    data), which is exactly the shape every transformer block pays twice.
    """
    torch = pytest.importorskip("torch")
    count = 4096
    nbytes = count * 4
    eps = 1e-6

    x = torch.randn(count, device="cuda", dtype=torch.float32)
    weight = torch.randn(count, device="cuda", dtype=torch.float32)
    for name in ("x", "w", "out"):
        native_engine.register_buffer(name, nbytes)
    native_engine.copy_into("x", x.data_ptr())
    native_engine.copy_into("w", weight.data_ptr())
    native_engine.synchronize()

    def eager_chain():
        return (x * torch.rsqrt(x.pow(2).mean() + eps)) * weight

    fused_key = (1, count, 8)
    with native_engine.capture(fused_key):
        native_engine.launch_fused_rmsnorm("out", "x", "w", eps, count)

    eager_ms = _median_ms(eager_chain)
    fused_ms = _median_ms(lambda: native_engine.replay(fused_key))

    assert fused_ms * 1.2 < eager_ms, (fused_ms, eager_ms)

    native_engine.replay(fused_key)
    native_engine.synchronize()
    np.testing.assert_allclose(
        download(native_engine, "out", count),
        rmsnorm_reference(x.cpu().numpy(), weight.cpu().numpy(), eps),
        rtol=1e-4,
        atol=1e-5,
    )



