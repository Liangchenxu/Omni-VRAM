"""Cooperative bridge between :mod:`vram_core` and the WDDM-Bypass engine.

``vram_core`` owns the **memory** side of the stack: paged KV-cache pools, the
physical block allocator, pressure-aware budgets.  :mod:`wddm_bypass` owns the
**submission** side: multi-slot CUDA graphs that replay a whole step with one
``cudaGraphLaunch``.  Neither package should learn about the other's internals,
so this module is the single seam between them:

* a vram_core pool is registered as a WDDM-Bypass **static buffer** - a capture
  must not allocate, and a replay must not either, so the address set has to be
  fixed before ``begin_capture``;
* the graph's :class:`wddm_bypass.ShapeKey` is derived from that pool's
  geometry, so one captured graph serves every sequence that fits the pool
  instead of one graph per sequence length;
* the optional budget / pressure gate refuses a binding that the current VRAM
  state cannot afford, before the driver ever sees an allocation.

Nothing here imports the native extension at import time: every ``wddm_bypass``
lookup is lazy, so ``import vram_core`` keeps working on a machine that never
built the engine.

Typical use::

    from vram_core.vram_optimizer import PagedKVCacheManager
    from vram_core.wddm_bridge import WddmExecutionBridge

    cache = PagedKVCacheManager(num_blocks=256, block_size=16)
    cache.allocate_sequence("session-1")

    with WddmExecutionBridge() as bridge:          # one engine, one device
        bound = bridge.bind_paged_kv_cache(cache)  # static buffer + shape key
        with bridge.capture(bound.shape_key):      # record the step once
            launch_paged_append(bridge.engine.buffer_ptr(bound.name), ...)
        bridge.replay(bound.shape_key)             # one submission per step
        bridge.synchronize()
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any, Dict, Iterator, NamedTuple, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "BoundBuffer",
    "BridgeClosed",
    "BridgeUnavailable",
    "DEFAULT_POOL_BUFFER",
    "InsufficientVRAMError",
    "WddmExecutionBridge",
    "device_pointer",
    "nbytes_of",
    "wddm_available",
]

#: Buffer name used by :meth:`WddmExecutionBridge.bind_paged_kv_cache`.
DEFAULT_POOL_BUFFER = "kv_pool"

_MIB = 1024 * 1024


class BridgeUnavailable(RuntimeError):
    """The ``wddm_bypass`` subsystem (or its native backend) is not importable."""


class BridgeClosed(RuntimeError):
    """Raised when a closed :class:`WddmExecutionBridge` is used."""


class InsufficientVRAMError(RuntimeError):
    """A binding was refused because the VRAM budget or pressure check failed."""


class BoundBuffer(NamedTuple):
    """One vram_core allocation registered as a WDDM-Bypass static buffer."""

    name: str
    nbytes: int
    address: int
    shape_key: Any
    source: str


def _subsystem() -> Any:
    """Import ``wddm_bypass`` lazily and translate a missing package."""
    try:
        import wddm_bypass  # noqa: PLC0415  (deliberately lazy)
    except ImportError as error:  # pragma: no cover - depends on the build
        raise BridgeUnavailable(
            "wddm_bypass is not importable; build the native subsystem with "
            "`python WDDM-Bypass/setup.py build_ext --inplace`"
        ) from error
    return wddm_bypass


def wddm_available() -> bool:
    """True when the ``wddm_bypass`` Python layer can be imported."""
    try:
        _subsystem()
    except BridgeUnavailable:
        return False
    return True


def nbytes_of(tensor: Any) -> int:
    """Byte size of a tensor / array / buffer, duck-typed across libraries.

    Torch and NumPy both expose ``nbytes``; a bare ``memoryview`` does not, so
    the buffer protocol is the last resort.  A zero-sized buffer is refused
    because a static buffer of zero bytes would capture an empty step.
    """
    if tensor is None:
        raise TypeError("tensor must be an array-like object, got None")

    size = getattr(tensor, "nbytes", None)
    if isinstance(size, int) and not isinstance(size, bool) and size > 0:
        return size

    numel = getattr(tensor, "numel", None)
    element_size = getattr(tensor, "element_size", None)
    if callable(numel) and callable(element_size):
        measured = int(numel()) * int(element_size())
        if measured > 0:
            return measured

    try:
        measured = memoryview(tensor).nbytes
    except TypeError as error:
        raise TypeError(f"cannot measure the byte size of {type(tensor).__name__}") from error
    if measured <= 0:
        raise ValueError("tensor must not be empty")
    return measured


def device_pointer(source: Any) -> int:
    """Device address of ``source`` as an int.

    ``source`` may be an int (already a pointer) or an object exposing
    ``data_ptr()`` - a CUDA tensor, or a vram_core handle.  A CPU tensor is
    refused explicitly: handing a host address to ``cudaMemcpyAsync`` is a
    crash, not an error code.
    """
    if isinstance(source, int) and not isinstance(source, bool):
        if source <= 0:
            raise ValueError(f"pointer must be positive, got {source}")
        return source

    if getattr(source, "is_cuda", None) is False:
        raise ValueError("device_pointer needs a CUDA tensor, got a CPU tensor")

    data_ptr = getattr(source, "data_ptr", None)
    if callable(data_ptr):
        address = int(data_ptr())
        if address <= 0:
            raise ValueError("tensor reports a null device pointer")
        return address

    # A NumPy array is always a host buffer: ``__array_interface__`` hands out a
    # host address, and handing that to ``cudaMemcpyAsync`` is a crash rather than
    # an error code, so it is refused instead of guessed at.
    raise TypeError(f"cannot resolve a device pointer from {type(source).__name__}")


class WddmExecutionBridge:
    """Binds vram_core allocations to a :class:`wddm_bypass.Engine`.

    Args:
        engine: An existing engine to borrow.  When ``None`` one is created
            (and owned) from ``backend`` / ``policy`` / ``device_index``.
        optimizer: Optional VRAM gate.  Any object exposing
            ``can_allocate(required_mb) -> bool`` works; :class:`VRAMOptimizer`
            is the intended one.
        budget_bytes: Optional hard cap on the bytes this bridge binds.
        policy: Bucketing policy forwarded to a newly created engine.
        device_index: CUDA device for a newly created engine (``-1`` = current).
        backend: Injected backend module, i.e. the test seam of
            :class:`wddm_bypass.Engine`.
        dtype: Element type that labels the derived graph shape keys.
    """

    def __init__(
        self,
        engine: Any = None,
        *,
        optimizer: Any = None,
        budget_bytes: Optional[int] = None,
        policy: Any = None,
        device_index: int = -1,
        backend: Any = None,
        dtype: str = "float32",
    ) -> None:
        if budget_bytes is not None:
            if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int):
                raise TypeError("budget_bytes must be an int")
            if budget_bytes <= 0:
                raise ValueError(f"budget_bytes must be positive, got {budget_bytes}")
        if not isinstance(dtype, str) or not dtype.strip():
            raise ValueError("dtype must be a non-empty string")

        self._owns_engine = engine is None
        if engine is None:
            module = _subsystem()
            engine = module.Engine(device_index, backend=backend, policy=policy)
        self._engine = engine
        self._optimizer = optimizer
        self._budget_bytes = budget_bytes
        self._dtype = dtype.strip().lower()
        self._bound: Dict[str, BoundBuffer] = {}
        self._committed_bytes = 0
        self._cache: Any = None
        self._closed = False

    # -- introspection -----------------------------------------------------
    @property
    def engine(self) -> Any:
        """The underlying :class:`wddm_bypass.Engine`."""
        return self._engine

    @property
    def bound_buffers(self) -> Dict[str, BoundBuffer]:
        """Registered buffers, keyed by name (a copy, safe to iterate)."""
        return dict(self._bound)

    @property
    def committed_bytes(self) -> int:
        """Bytes this bridge has registered as static buffers so far."""
        return self._committed_bytes

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise BridgeClosed("this WddmExecutionBridge is closed")

    # -- shape plumbing ----------------------------------------------------
    def shape_key_for_cache(self, cache: Any) -> Any:
        """Derive the graph :class:`~wddm_bypass.ShapeKey` of a paged pool.

        The key is the pool's own geometry, so ``key.elements`` times the element
        size is *exactly* the pool footprint::

            ShapeKey(num_blocks, num_heads * block_size, head_dim, dtype)

        ``batch`` is therefore the block count and ``sequence`` the token span a
        single block covers.  Two pools of the same geometry share one graph.
        """
        self._ensure_open()
        module = _subsystem()
        values = {}
        for field in ("num_blocks", "num_heads", "block_size", "head_dim"):
            raw = getattr(cache, field, None)
            if raw is None:
                raise TypeError(f"paged cache has no {field!r} attribute")
            value = int(raw)
            if value <= 0:
                raise ValueError(f"{field} must be positive, got {value}")
            values[field] = value
        return module.ShapeKey(
            values["num_blocks"],
            values["num_heads"] * values["block_size"],
            values["head_dim"],
            self._dtype,
        )

    # -- budget gate -------------------------------------------------------
    def _reserve(self, nbytes: int) -> None:
        """Refuse a binding the configured budget or VRAM gate cannot afford."""
        if nbytes <= 0:
            raise ValueError(f"nbytes must be positive, got {nbytes}")
        if self._budget_bytes is not None and self._committed_bytes + nbytes > self._budget_bytes:
            raise InsufficientVRAMError(
                f"binding {nbytes} bytes would exceed the bridge budget "
                f"({self._committed_bytes} of {self._budget_bytes} bytes already committed)"
            )
        can_allocate = getattr(self._optimizer, "can_allocate", None)
        if callable(can_allocate):
            required_mb = math.ceil(nbytes / _MIB)
            if not can_allocate(required_mb):
                raise InsufficientVRAMError(
                    f"the VRAM gate refused {required_mb} MiB "
                    f"({self._committed_bytes} bytes already bound by this bridge)"
                )

    # -- bindings ----------------------------------------------------------
    def bind_tensor(self, name: str, tensor: Any, *, stream_token: int = 0) -> BoundBuffer:
        """Register a tensor-sized static buffer, uploading it when device-resident."""
        self._ensure_open()
        nbytes = nbytes_of(tensor)
        self._reserve(nbytes)
        address = int(self._engine.register_buffer(name, nbytes))
        bound = BoundBuffer(
            name=name,
            nbytes=nbytes,
            address=address,
            shape_key=None,
            source=type(tensor).__name__,
        )
        self._bound[name] = bound
        self._committed_bytes += nbytes

        if getattr(tensor, "is_cuda", None) is True:
            # A host array registered only for its size is left to the caller;
            # only a device tensor can seed a device buffer without a crash.
            self.upload(name, tensor, stream_token=stream_token)
        return bound

    def bind_paged_kv_cache(
        self,
        cache: Any,
        name: str = DEFAULT_POOL_BUFFER,
    ) -> BoundBuffer:
        """Register ``cache``'s block pool as a static buffer.

        The pool address becomes fixed for the lifetime of the graph, which is
        exactly the precondition ``cudaStreamBeginCapture`` needs.  The returned
        :class:`BoundBuffer` carries the derived shape key, so the same object
        can be handed straight to :meth:`capture` and :meth:`replay`.
        """
        self._ensure_open()
        nbytes = int(getattr(cache, "pool_bytes", 0))
        if nbytes <= 0:
            raise ValueError(
                "paged cache reports no pool bytes; pass a PagedKVCacheManager "
                "with num_blocks/block_size/num_heads/head_dim > 0"
            )
        shape_key = self.shape_key_for_cache(cache)
        self._reserve(nbytes)
        address = int(self._engine.register_buffer(name, nbytes))
        bound = BoundBuffer(
            name=name,
            nbytes=nbytes,
            address=address,
            shape_key=shape_key,
            source="PagedKVCacheManager",
        )
        self._bound[name] = bound
        self._committed_bytes += nbytes
        self._cache = cache
        logger.debug("bound %s (%d bytes) to graph shape %s", name, nbytes, shape_key)
        return bound

    def release(self, name: str) -> bool:
        """Free a static buffer; a name this bridge never bound is ignored."""
        self._ensure_open()
        bound = self._bound.pop(name, None)
        if bound is None:
            return False
        self._engine.release_buffer(name)
        self._committed_bytes -= bound.nbytes
        return True

    # -- data movement -----------------------------------------------------
    def upload(
        self,
        name: str,
        source: Any,
        *,
        nbytes: Optional[int] = None,
        stream_token: int = 0,
    ) -> None:
        """Copy ``source`` into the static buffer ``name`` (device to device)."""
        self._ensure_open()
        if name not in self._bound:
            raise KeyError(f"no buffer named {name!r} is bound by this bridge")
        length = self._bound[name].nbytes if nbytes is None else int(nbytes)
        self._engine.copy_into(name, device_pointer(source), length, int(stream_token))

    def zero(self, name: str, stream_token: int = 0) -> None:
        """Zero a bound static buffer asynchronously."""
        self._ensure_open()
        if name not in self._bound:
            raise KeyError(f"no buffer named {name!r} is bound by this bridge")
        self._engine.zero_buffer(name, int(stream_token))

    # -- execution ---------------------------------------------------------
    def route(self, key: Any) -> Optional[int]:
        """Graph id ``key`` routes to, or ``None`` if it was never seen."""
        self._ensure_open()
        return self._engine.router.lookup(key)

    def has(self, key: Any) -> bool:
        """True when ``key`` has a live executable graph."""
        self._ensure_open()
        return bool(self._engine.has(key))

    @contextlib.contextmanager
    def capture(self, key: Any, *, hot_update: bool = False) -> Iterator[int]:
        """Record the step for ``key``; see :meth:`wddm_bypass.Engine.capture`."""
        self._ensure_open()
        with self._engine.capture(key, hot_update=hot_update) as graph_id:
            yield graph_id

    def replay(self, key: Any, stream_token: int = 0) -> None:
        """Submit the graph that serves ``key`` (one launch for the whole step)."""
        self._ensure_open()
        self._engine.replay(key, stream_token=int(stream_token))

    def replay_for_cache(self, cache: Any, stream_token: int = 0) -> Any:
        """Replay the graph derived from ``cache``'s geometry; returns the key."""
        key = self.shape_key_for_cache(cache)
        if not self._engine.has(key):
            raise KeyError(f"no captured graph for paged geometry {key}; capture it first")
        self.replay(key, stream_token=stream_token)
        return key

    def synchronize(self) -> None:
        """Block until the engine's device is idle."""
        self._ensure_open()
        self._engine.synchronize()

    def reset(self, key: Any) -> None:
        """Destroy the graph for ``key``; the routing entry survives."""
        self._ensure_open()
        self._engine.reset(key)

    # -- reporting ---------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Engine counters merged with this bridge's vram_core view."""
        self._ensure_open()
        measured: Dict[str, Any] = dict(self._engine.stats())
        measured["bound_buffers"] = len(self._bound)
        measured["static_bytes"] = self._committed_bytes
        measured["bridge_budget_bytes"] = self._budget_bytes
        measured["bridge_dtype"] = self._dtype
        cache = self._cache
        if cache is not None:
            # ``block_allocator`` is a property on PagedKVCacheManager; accept a
            # method spelling too so a future refactor cannot break the report.
            allocator = cache.block_allocator
            if callable(allocator):
                allocator = allocator()
            measured["kv_pool_bytes"] = int(cache.pool_bytes)
            measured["kv_backend"] = str(getattr(cache, "backend", "unknown"))
            measured["kv_used_blocks"] = int(allocator.used_block_count)
            measured["kv_free_blocks"] = int(allocator.free_block_count)
            measured["kv_shape"] = self.shape_key_for_cache(cache).as_tuple()
        return measured

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        """Release every buffer, then the engine if this bridge created it."""
        if self._closed:
            return
        try:
            for name in list(self._bound):
                self._engine.release_buffer(name)
            self._bound.clear()
            self._committed_bytes = 0
            self._cache = None
        finally:
            if self._owns_engine:
                self._engine.close()
            self._closed = True

    def __enter__(self) -> "WddmExecutionBridge":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"WddmExecutionBridge(buffers={len(self._bound)}, "
            f"bytes={self._committed_bytes}, closed={self._closed})"
        )
