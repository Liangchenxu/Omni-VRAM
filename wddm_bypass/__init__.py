"""WDDM-Bypass v1.0.0 - pre-instantiated CUDA graph replay for Windows.

What this package actually does
-------------------------------
On Windows every kernel launch costs a user-mode driver round trip that WDDM
batches before it reaches the GPU. The only lever a user-mode process has over
that cost is to stop issuing N launches: capture N kernels into one
``cudaGraph_t`` once, instantiate it, and then submit the whole batch with a
single ``cudaGraphLaunch``. That is the "bypass" this project implements. It
does not patch the driver or the OS scheduler - that is not reachable from user
mode - and nothing here claims to.

Layers
------
``wddm_bypass._wddm_bypass``
    pybind11 extension: the engine, RAII handle wrappers, static device buffers.
``wddm_bypass.backend``
    Import isolation with a precise build hint.
``wddm_bypass.bucketing``
    Shape -> bucket -> graph id routing (pure Python).
``wddm_bypass.Engine``
    Backend-agnostic facade: ``capture()`` context manager, buffered replay and
    one place for the lifecycle rules.

Typical use::

    from wddm_bypass import Engine, ShapeKey

    with Engine() as engine:
        engine.register_buffer("hidden", 1 << 20)
        with engine.capture(ShapeKey(1, 128, 768)):
            ops.launch_relu(engine.buffer_ptr("hidden"), 262144)
        engine.replay((1, 128, 768))
        engine.synchronize()
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, List, Optional

from .backend import (
    BUILD_HINT,
    EXTENSION_NAME,
    BackendUnavailable,
    backend_available,
    load_backend,
)
from .bucketing import BucketPolicy, ShapeKey, ShapeRouter, coerce_key

#: Version of the native ``wddm_bypass._wddm_bypass`` extension - its ABI
#: version, not the Omni-VRAM release version (that is ``vram_core.__version__``,
#: currently 2.8.0). The string is compiled into ``build_info()["version"]`` and
#: ``WDDM-Bypass/setup.py``, and ``tests/test_engine_suite.py`` asserts all three
#: agree, so it only moves when the C++ side is rebuilt.
__version__ = "1.0.0"

__all__ = [
    "BUILD_HINT",
    "EXTENSION_NAME",
    "BackendUnavailable",
    "BucketPolicy",
    "Engine",
    "EngineClosed",
    "ShapeKey",
    "ShapeRouter",
    "__version__",
    "backend_available",
    "build_info",
    "capture_context",
    "coerce_key",
    "load_backend",
]


class EngineClosed(RuntimeError):
    """Raised when a closed :class:`Engine` is used."""


class Engine:
    """Facade over the native ``WDDMBypassEngine``.

    Args:
        device_index: CUDA device to pin the engine to; ``-1`` keeps the caller's
            current device.
        backend: An injected backend module. Tests pass a fake; production code
            leaves this as ``None`` and the compiled extension is imported.
        policy: Bucketing policy for shape -> graph id routing.

    The engine is not thread safe on the Python side, and does not need to be:
    every native entry point serialises on its own mutex, and the routing tables
    here are plain dicts guarded by the GIL.
    """

    def __init__(
        self,
        device_index: int = -1,
        *,
        backend: Any = None,
        policy: Optional[BucketPolicy] = None,
    ) -> None:
        self._backend = backend if backend is not None else load_backend()
        self._device_index = int(device_index)
        self._raw = self._backend.WDDMBypassEngine(self._device_index)
        self._router = ShapeRouter(policy)
        self._buffers: Dict[str, int] = {}
        self._closed = False

    # -- introspection -----------------------------------------------------
    @property
    def backend(self) -> Any:
        """The backend module this engine was built against."""
        return self._backend

    @property
    def device_index(self) -> int:
        """CUDA device the native engine is pinned to."""
        return self._device_index

    @property
    def router(self) -> ShapeRouter:
        """The shape -> graph id router in use."""
        return self._router

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has run."""
        return self._closed

    @property
    def version(self) -> str:
        """Version reported by the loaded backend, if it reports one."""
        return str(getattr(self._backend, "__version__", "unknown"))

    def _ensure_open(self) -> None:
        if self._closed:
            raise EngineClosed("WDDM-Bypass: this Engine has been closed")

    def graph_id(self, key: Any) -> int:
        """Graph id that serves ``key``, allocating one on first sight."""
        self._ensure_open()
        return self._router.graph_id(key)

    def __len__(self) -> int:
        return len(self._router)

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return (
            f"<Engine {state} device={self._device_index} "
            f"shapes={len(self._router)} buffers={len(self._buffers)}>"
        )

    # -- capture -----------------------------------------------------------
    def begin_capture(self, key: Any, hot_update: bool = False) -> int:
        """Start capturing GPU work for ``key``; returns the graph id.

        Every kernel launched between this call and :meth:`end_capture` becomes
        one node of the graph. With ``hot_update=True`` and an already
        instantiated graph, :meth:`end_capture` merges the new capture into the
        existing executable instead of re-instantiating it.
        """
        self._ensure_open()
        graph_id = self._router.graph_id(key)
        self._raw.begin_capture(graph_id, str(self._router.bucket(key)), bool(hot_update))
        return graph_id

    def end_capture(self, key: Any) -> int:
        """Finish the capture for ``key``; returns the graph id."""
        self._ensure_open()
        graph_id = self._router.graph_id(key)
        self._raw.end_capture(graph_id)
        return graph_id

    def capture_stream(self) -> int:
        """Raw ``cudaStream_t`` (as an int) that carries the work being captured.

        Stream capture records launches; it does not execute them. A launch has
        to be issued on this stream, otherwise it is neither recorded nor
        replayed - it simply runs eagerly once::

            stream = engine.capture_stream()
            with engine.capture(key):
                ops.launch_scale(pointer, count, 2.0, stream)

        The value is also what ``torch.cuda.ExternalStream`` needs when the work
        is submitted through PyTorch inside a capture region.
        """
        self._ensure_open()
        return int(self._raw.capture_stream_ptr())


    @contextlib.contextmanager
    def capture(self, key: Any, hot_update: bool = False) -> Iterator[int]:
        """Context manager form of :meth:`begin_capture` / :meth:`end_capture`.

        If the body raises, the pending capture is finished and discarded so the
        engine is never left with a stream stuck in capture mode::

            with engine.capture(ShapeKey(1, 128, 768)) as graph_id:
                ops.launch_scale(pointer, 98304, 0.5)

        The native slot for ``graph_id`` is erased, so the next capture for the
        same key starts clean.
        """
        graph_id = self.begin_capture(key, hot_update=hot_update)
        try:
            yield graph_id
        except BaseException:
            self._abandon_capture(graph_id)
            raise
        self._raw.end_capture(graph_id)

    def _abandon_capture(self, graph_id: int) -> None:
        """Best-effort teardown after an exception inside a capture region.

        Deliberately quiet: the caller's exception is the one that matters. If
        ``end_capture`` fails here, the native engine marks its capture stream
        unusable and the next capture asks for :meth:`reset_all`.
        """
        with contextlib.suppress(Exception):
            self._raw.end_capture(graph_id)
        with contextlib.suppress(Exception):
            self._raw.reset(graph_id)

    def is_capturing(self, key: Any) -> bool:
        """True while a capture is open for ``key``."""
        self._ensure_open()
        graph_id = self._router.lookup(key)
        if graph_id is None:
            return False
        return bool(self._raw.is_capturing(graph_id))

    # -- execution ---------------------------------------------------------
    def replay(self, key: Any, stream_token: int = 0) -> None:
        """Submit the graph that serves ``key`` in a single launch.

        ``stream_token`` is a raw ``cudaStream_t`` as an integer; ``0`` means the
        caller's current stream (the current PyTorch stream in a torch build).
        """
        self._ensure_open()
        self._raw.replay(self._router.graph_id(key), int(stream_token))

    def run(self, key: Any, stream_token: int = 0) -> None:
        """Alias for :meth:`replay`, for call sites that read better as ``run``."""
        self.replay(key, stream_token=stream_token)

    def update(self, key: Any) -> None:
        """Merge the pending staging graph into the executable for ``key``."""
        self._ensure_open()
        self._raw.update_graph(self._router.graph_id(key))

    def synchronize(self) -> None:
        """Block until the engine's device is idle."""
        self._ensure_open()
        self._raw.synchronize()

    def describe(self, key: Any) -> str:
        """One-line JSON description of the graph that serves ``key``."""
        self._ensure_open()
        return str(self._raw.describe(self._router.graph_id(key)))

    # -- static buffers ----------------------------------------------------
    def register_buffer(self, name: str, bytes: int) -> int:
        """Allocate (or replace) a named device buffer; returns its address."""
        self._ensure_open()
        if not isinstance(name, str) or not name:
            raise ValueError("buffer name must be a non-empty string")
        if isinstance(bytes, bool) or not isinstance(bytes, int) or bytes <= 0:
            raise ValueError(f"buffer size must be a positive int, got {bytes!r}")
        address = int(self._raw.register_static_buffer(name, bytes))
        self._buffers[name] = int(bytes)
        return address

    def buffer_ptr(self, name: str) -> int:
        """Device address of a registered buffer, as an integer."""
        self._ensure_open()
        return int(self._raw.static_buffer_ptr(name))

    def buffer_bytes(self, name: str) -> int:
        """Byte size of a registered buffer."""
        self._ensure_open()
        return int(self._raw.static_buffer_bytes(name))

    def buffers(self) -> List[str]:
        """Names of the buffers registered through this engine, sorted."""
        return sorted(self._buffers)

    def _buffer_length(self, name: str, bytes: Optional[int]) -> int:
        length = self._buffers.get(name)
        if length is None:
            raise KeyError(f"no buffer named {name!r} is registered")
        if bytes is None:
            return length
        if isinstance(bytes, bool) or not isinstance(bytes, int):
            raise TypeError(f"bytes must be an int, got {type(bytes).__name__}")
        if bytes < 0 or bytes > length:
            raise ValueError(f"bytes must be in [0, {length}], got {bytes}")
        return bytes

    def copy_into(
        self,
        name: str,
        source: int,
        bytes: Optional[int] = None,
        stream_token: int = 0,
    ) -> None:
        """Async copy ``source`` (host or device address) into a buffer."""
        self._ensure_open()
        if isinstance(source, bool) or not isinstance(source, int) or source == 0:
            raise ValueError(f"source must be a non-zero address, got {source!r}")
        length = self._buffer_length(name, bytes)
        self._raw.copy_to_static_input(name, source, length, int(stream_token))

    def read_buffer(
        self,
        name: str,
        destination: int,
        bytes: Optional[int] = None,
        stream_token: int = 0,
    ) -> None:
        """Async copy a buffer out to ``destination`` (host or device address)."""
        self._ensure_open()
        if isinstance(destination, bool) or not isinstance(destination, int) or destination == 0:
            raise ValueError(f"destination must be a non-zero address, got {destination!r}")
        length = self._buffer_length(name, bytes)
        self._raw.copy_from_static(name, destination, length, int(stream_token))

    def zero_buffer(self, name: str, stream_token: int = 0) -> None:
        """Fill a buffer with zeros asynchronously."""
        self._ensure_open()
        self._raw.zero_static_buffer(name, int(stream_token))

    def release_buffer(self, name: str) -> None:
        """Free a buffer; a name this engine never registered is ignored."""
        self._ensure_open()
        if name not in self._buffers:
            return
        self._raw.release_static_buffer(name)
        del self._buffers[name]

    # -- lifecycle ---------------------------------------------------------
    def graphs(self) -> List[int]:
        """Graph ids the native engine currently holds a slot for."""
        self._ensure_open()
        return [int(identifier) for identifier in self._raw.list_graphs()]

    def has(self, key: Any) -> bool:
        """True when ``key`` has a captured graph with a live executable."""
        self._ensure_open()
        graph_id = self._router.lookup(key)
        if graph_id is None:
            return False
        return bool(self._raw.has_graph(graph_id))

    def reset(self, key: Any) -> None:
        """Destroy the graph (and executable) that serves ``key``.

        The shape -> id mapping survives, so the next capture for the same key
        reuses the same id instead of leaking a new one.
        """
        self._ensure_open()
        graph_id = self._router.lookup(key)
        if graph_id is None:
            return
        self._raw.reset(graph_id)

    def reset_all(self) -> None:
        """Destroy every graph and executable; buffers are kept."""
        self._ensure_open()
        self._raw.reset_all()

    def stats(self) -> Dict[str, Any]:
        """Native counters merged with this facade's shape and buffer counters."""
        self._ensure_open()
        measured: Dict[str, Any] = dict(self._raw.stats())
        router_stats = self._router.stats()
        measured["shapes"] = router_stats["shapes"]
        measured["shape_hits"] = router_stats["shape_hits"]
        measured["shape_misses"] = router_stats["shape_misses"]
        measured["buffers"] = len(self._buffers)
        measured["device_index"] = self._device_index
        measured["bucket_policy"] = self._router.policy.describe()
        measured["backend_version"] = self.version
        return measured

    def close(self) -> None:
        """Release every native resource. Idempotent, and safe in a ``finally``."""
        if self._closed:
            return
        try:
            self._raw.reset_all()
        finally:
            self._buffers.clear()
            self._closed = True

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def build_info() -> Dict[str, Any]:
    """Build and capability information reported by the native module.

    Raises:
        BackendUnavailable: The compiled extension is not importable.
    """
    return dict(load_backend().build_info())


@contextlib.contextmanager
def capture_context(engine: Engine, key: Any, hot_update: bool = False) -> Iterator[int]:
    """Module-level spelling of :meth:`Engine.capture`."""
    with engine.capture(key, hot_update=hot_update) as graph_id:
        yield graph_id
