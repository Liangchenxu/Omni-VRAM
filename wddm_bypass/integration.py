"""wddm_bypass.integration - decode-step mounting for Transformer inference.

Where this fits (v2.7.0)
------------------------
``Engine`` captures and replays CUDA graphs. This module turns that into a
mount point for an autoregressive decoder: one decode step becomes one graph,
and every later step of the same geometry is a single ``cudaGraphLaunch``
instead of N submissions that WDDM has to batch first.

What it accelerates, honestly
-----------------------------
* **Submission.** N kernel launches per step become one graph launch, which is
  the half a user-mode process can control on Windows. The engine's own
  reference benchmark reports 17.3x lower submission cost for a 64-kernel step.
* **Nothing else.** The kernels, their occupancy and the memory system are
  untouched, so a GPU-bound step gains little (the same benchmark reports 1.10x
  for a 4 MiB buffer). A hook that claims more than that is measuring noise.

Scope limits
------------
* Fixed geometry only. A step whose :class:`~wddm_bypass.ShapeKey` was never
  captured runs eager and is counted in ``stats()["eager_fallbacks"]``; it is
  never silently re-captured mid-flight.
* The captured body only issues launches. Data movement is explicit:
  ``execute(..., feed=..., fetch=...)`` copies tensors into the static buffers
  before the replay and out of them afterwards, because a captured graph keeps
  the buffer addresses it was recorded with.
* No tensor library is imported at module import time. ``torch`` is touched only
  by :func:`torch_stream` / :func:`tensor_pointer`, ``numpy`` only by
  :class:`HookedDecoder`.

Typical use::

    from wddm_bypass import Engine
    from wddm_bypass.integration import DecodeHook, DecodePlan

    plan = DecodePlan.build(
        (1, 1, 2048),
        {"hidden": 2048, "tmp": 2048, "logits": 2048},
        inputs=("hidden",),
        outputs=("logits",),
    )
    hook = DecodeHook(Engine(), plan=plan)
    hook.execute(decode_body)   # warm-up, capture, replay
    hook.execute(decode_body)   # one graph launch per step from here on
"""

from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

from .backend import BackendUnavailable
from .bucketing import ShapeKey, coerce_key

__all__ = [
    "BufferSpec",
    "DecodeBuffers",
    "DecodeHook",
    "DecodePlan",
    "HookClosed",
    "HookedDecoder",
    "accelerate_decode",
    "data_pointer",
    "dtype_size",
    "tensor_pointer",
    "torch_stream",
]

#: Element sizes for the dtypes a decode step actually uses.
_DTYPE_SIZES: Dict[str, int] = {
    "float32": 4,
    "fp32": 4,
    "float": 4,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int8": 1,
    "uint8": 1,
    "int32": 4,
    "int64": 8,
}


def dtype_size(dtype: str) -> int:
    """Bytes per element for ``dtype``.

    Raises:
        TypeError: ``dtype`` is not a string.
        ValueError: ``dtype`` is not one of the sizes this module knows.
    """
    if not isinstance(dtype, str):
        raise TypeError(f"dtype must be a string, got {type(dtype).__name__}")
    key = dtype.strip().lower()
    if key not in _DTYPE_SIZES:
        known = ", ".join(sorted(_DTYPE_SIZES))
        raise ValueError(f"unsupported dtype {dtype!r}; known dtypes: {known}")
    return _DTYPE_SIZES[key]


@dataclass(frozen=True)
class BufferSpec:
    """One static buffer a captured step reads or writes.

    Attributes:
        name: Buffer name, unique inside a :class:`DecodePlan`.
        count: Element count (not bytes).
        dtype: Element type used to turn ``count`` into bytes.
    """

    name: str
    count: int
    dtype: str = "float32"
    shape: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("BufferSpec.name must be a non-empty string")
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError(f"BufferSpec.count must be an int, got {type(self.count).__name__}")
        if self.count <= 0:
            raise ValueError(f"BufferSpec.count must be positive, got {self.count}")
        if not isinstance(self.dtype, str):
            raise TypeError(f"BufferSpec.dtype must be a string, got {type(self.dtype).__name__}")
        # Validates the dtype eagerly; the frozen dataclass cannot store a copy.
        dtype_size(self.dtype)
        if self.shape:
            if not isinstance(self.shape, tuple):
                raise TypeError("BufferSpec.shape must be a tuple of ints")
            elements = 1
            for dim in self.shape:
                if isinstance(dim, bool) or not isinstance(dim, int):
                    raise TypeError(f"BufferSpec.shape dimensions must be ints, got {dim!r}")
                if dim <= 0:
                    raise ValueError(f"BufferSpec.shape dimensions must be positive, got {dim!r}")
                elements *= dim
            if elements != self.count:
                raise ValueError(
                    f"BufferSpec({self.name!r}) declares shape {self.shape} = "
                    f"{elements} elements but count={self.count}"
                )

    @property
    def nbytes(self) -> int:
        """Size of the buffer in bytes."""
        return self.count * dtype_size(self.dtype)

    @classmethod
    def of(cls, name: str, shape: Union[int, Sequence[int]], dtype: str = "float32") -> "BufferSpec":
        """Build a spec from an element count or a tensor shape."""
        if isinstance(shape, bool):
            raise TypeError("shape must be an int or a sequence of ints, not a bool")
        if isinstance(shape, int):
            count = shape
            dims: Tuple[int, ...] = ()
        else:
            dims = tuple(shape)
            if not dims:
                raise ValueError(f"BufferSpec.of({name!r}, ...) needs at least one dimension")
            count = 1
            for dim in dims:
                if isinstance(dim, bool) or not isinstance(dim, int):
                    raise TypeError(f"shape dimensions must be ints, got {dim!r}")
                if dim <= 0:
                    raise ValueError(f"shape dimensions must be positive, got {dim!r}")
                count *= dim
        return cls(name=name, count=count, dtype=dtype, shape=dims)


@dataclass(frozen=True)
class DecodePlan:
    """Fixed-shape geometry of one decode step: buffers in, buffers out.

    Attributes:
        key: Bucketed execution shape; the engine routes replays by it.
        specs: Every static buffer the captured body touches.
        inputs: Buffer names the hook copies caller data into before replay.
        outputs: Buffer names the hook copies back after replay.
        name: Human-readable label used in reports.
    """

    key: ShapeKey
    specs: Tuple[BufferSpec, ...]
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()
    name: str = "decode"

    def __post_init__(self) -> None:
        if not isinstance(self.key, ShapeKey):
            raise TypeError(f"DecodePlan.key must be a ShapeKey, got {type(self.key).__name__}")
        if not isinstance(self.specs, tuple) or not self.specs:
            raise TypeError("DecodePlan.specs must be a non-empty tuple of BufferSpec")
        names = []
        for spec in self.specs:
            if not isinstance(spec, BufferSpec):
                raise TypeError(
                    f"DecodePlan.specs must contain BufferSpec, got {type(spec).__name__}"
                )
            names.append(spec.name)
        if len(set(names)) != len(names):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"duplicate buffer names in DecodePlan: {duplicates}")
        known = set(names)
        for group, values in (("inputs", self.inputs), ("outputs", self.outputs)):
            for value in values:
                if value not in known:
                    raise ValueError(
                        f"{group} names buffer {value!r}, which the plan does not declare"
                    )
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("DecodePlan.name must be a non-empty string")

    @property
    def buffers(self) -> Dict[str, BufferSpec]:
        """Buffer specs keyed by name."""
        return {spec.name: spec for spec in self.specs}

    @property
    def nbytes(self) -> int:
        """Bytes reserved by every buffer in this plan."""
        return sum(spec.nbytes for spec in self.specs)

    @property
    def dtype(self) -> str:
        """Dtype of the first buffer (the plan-level label)."""
        return self.specs[0].dtype

    def spec(self, name: str) -> BufferSpec:
        """The spec called ``name``.

        Raises:
            KeyError: The plan has no such buffer.
        """
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(f"plan {self.name!r} has no buffer named {name!r}")

    @classmethod
    def build(
        cls,
        key: Union[ShapeKey, Sequence[Any]],
        shapes: Mapping[str, Union[int, Sequence[int]]],
        *,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
        dtype: str = "float32",
        name: str = "decode",
    ) -> "DecodePlan":
        """Build a plan from ``{buffer name: element count or shape}``."""
        resolved = coerce_key(key)
        specs = tuple(BufferSpec.of(buffer, shape, dtype) for buffer, shape in shapes.items())
        return cls(
            key=resolved,
            specs=specs,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            name=name,
        )

    def describe(self) -> Dict[str, Any]:
        """Plain dict for reports and test assertions."""
        return {
            "name": self.name,
            "key": self.key.as_tuple(),
            "dtype": self.dtype,
            "buffers": {spec.name: spec.nbytes for spec in self.specs},
            "nbytes": self.nbytes,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
        }


class DecodeBuffers:
    """Name -> static device buffer view handed to a captured decode body.

    The body issues launches through this object and never allocates: an
    allocation inside a capture region is exactly what
    ``cudaStreamBeginCapture`` refuses, and it is why the engine's buffers are
    registered before the capture starts.

    Args:
        engine: A :class:`wddm_bypass.Engine` (or a compatible facade).
        plan: Geometry whose buffers are being addressed.
        stream_token: Raw stream handle the launches go to. Inside a capture
            region this must be ``engine.capture_stream()``, otherwise the
            launches are neither recorded nor replayed.
    """

    __slots__ = ("_engine", "_plan", "_stream_token", "launches")

    def __init__(self, engine: Any, plan: DecodePlan, stream_token: int = 0) -> None:
        if not isinstance(plan, DecodePlan):
            raise TypeError(f"plan must be a DecodePlan, got {type(plan).__name__}")
        self._engine = engine
        self._plan = plan
        self._stream_token = int(stream_token)
        #: Kernels issued through this view (one graph node each, when capturing).
        self.launches = 0

    # -- introspection -----------------------------------------------------
    @property
    def plan(self) -> DecodePlan:
        """The geometry this view addresses."""
        return self._plan

    @property
    def stream_token(self) -> int:
        """Raw stream handle every launch is issued on."""
        return self._stream_token

    @property
    def captured(self) -> bool:
        """True when launches are being recorded into a graph."""
        return self._stream_token != 0

    def names(self) -> Tuple[str, ...]:
        """Buffer names in plan order."""
        return tuple(spec.name for spec in self._plan.specs)

    def spec(self, name: str) -> BufferSpec:
        """The buffer spec called ``name``."""
        return self._plan.spec(name)

    def ptr(self, name: str) -> int:
        """Device address of a static buffer, as an integer."""
        return int(self._engine.buffer_ptr(name))

    def count(self, name: str) -> int:
        """Element count of a static buffer."""
        return self.spec(name).count

    def nbytes(self, name: str) -> int:
        """Byte size of a static buffer."""
        return self.spec(name).nbytes

    # -- launchers ---------------------------------------------------------
    def scale(self, name: str, alpha: float = 1.0) -> "DecodeBuffers":
        """In-place ``x *= alpha`` over the whole buffer."""
        spec = self.spec(name)
        self._launcher("launch_scale")(
            self.ptr(name), spec.count, float(alpha), self._stream_token
        )
        self.launches += 1
        return self

    def add(self, destination: str, lhs: str, rhs: str) -> "DecodeBuffers":
        """``destination = lhs + rhs`` elementwise."""
        spec = self.spec(destination)
        other = self.spec(lhs)
        third = self.spec(rhs)
        if not (spec.count == other.count == third.count):
            raise ValueError(
                f"launch_add needs equal element counts, got "
                f"{destination}={spec.count}, {lhs}={other.count}, {rhs}={third.count}"
            )
        self._launcher("launch_add")(
            self.ptr(destination), self.ptr(lhs), self.ptr(rhs), spec.count,
            self._stream_token,
        )
        self.launches += 1
        return self

    def relu(self, name: str) -> "DecodeBuffers":
        """In-place ``x = max(x, 0)`` over the whole buffer."""
        spec = self.spec(name)
        self._launcher("launch_relu")(self.ptr(name), spec.count, self._stream_token)
        self.launches += 1
        return self

    def zero(self, name: str) -> "DecodeBuffers":
        """Asynchronously zero a static buffer (a memset, not a kernel)."""
        self._engine.zero_buffer(name, self._stream_token)
        return self

    # -- data movement -----------------------------------------------------
    def copy_in(self, name: str, source: Any, nbytes: Optional[int] = None) -> "DecodeBuffers":
        """Copy ``source`` (tensor, array or raw pointer) into a static buffer."""
        length = self.nbytes(name) if nbytes is None else int(nbytes)
        self._engine.copy_into(name, data_pointer(source), length, self._stream_token)
        return self

    def copy_out(self, name: str, destination: Any, nbytes: Optional[int] = None) -> "DecodeBuffers":
        """Copy a static buffer out to ``destination`` (tensor, array or pointer)."""
        length = self.nbytes(name) if nbytes is None else int(nbytes)
        self._engine.read_buffer(name, data_pointer(destination), length, self._stream_token)
        return self

    # -- internals ---------------------------------------------------------
    def _launcher(self, attribute: str) -> Callable[..., Any]:
        backend = getattr(self._engine, "backend", None)
        if backend is None:
            raise BackendUnavailable(
                "the engine exposes no backend module, so no kernel launcher is "
                "reachable; pass a real wddm_bypass.Engine or a fake backend that "
                "provides launch_scale / launch_add / launch_relu"
            )
        function = getattr(backend, attribute, None)
        if not callable(function):
            raise BackendUnavailable(
                f"the backend module has no {attribute}() launcher; the decode "
                f"hook needs launch_scale, launch_add and launch_relu"
            )
        return function

    def __len__(self) -> int:
        return len(self._plan.specs)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        mode = "capture" if self.captured else "eager"
        return (
            f"DecodeBuffers({self._plan.name}, buffers={len(self._plan.specs)}, "
            f"launches={self.launches}, mode={mode})"
        )


class HookClosed(RuntimeError):
    """Raised when a closed :class:`DecodeHook` is used."""


class DecodeHook:
    """Captures one decode step and replays it as a single graph launch.

    Args:
        engine: Engine to drive. ``None`` creates one lazily on first use from
            ``backend`` / ``device_index`` / ``policy``.
        plan: Default geometry, so callers can omit it per call.
        backend: Injected backend module for a lazily created engine (the same
            test seam :class:`wddm_bypass.Engine` uses).
        device_index: CUDA device for a lazily created engine (``-1`` =
            whatever device is current).
        policy: Bucketing policy for a lazily created engine.
        warmup: Eager executions before the capture. They exist so that
            one-time allocations (math library handles, allocator growth) can
            never land inside the capture region.
        max_graphs: Upper bound on captured geometries. When a new shape arrives
            after the budget is spent, the step runs eager and is counted in
            ``stats()["eager_fallbacks"]`` instead of silently growing the pool.
        owns_engine: Whether :meth:`close` also closes ``engine``. Defaults to
            ``True`` only when this hook created it.
    """

    def __init__(
        self,
        engine: Any = None,
        *,
        plan: Optional[DecodePlan] = None,
        backend: Any = None,
        device_index: int = -1,
        policy: Any = None,
        warmup: int = 1,
        max_graphs: int = 4,
        owns_engine: Optional[bool] = None,
    ) -> None:
        if plan is not None and not isinstance(plan, DecodePlan):
            raise TypeError(f"plan must be a DecodePlan, got {type(plan).__name__}")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError(f"warmup must be a non-negative int, got {warmup!r}")
        if isinstance(max_graphs, bool) or not isinstance(max_graphs, int) or max_graphs < 1:
            raise ValueError(f"max_graphs must be a positive int, got {max_graphs!r}")
        self._engine = engine
        self._plan = plan
        self._backend = backend
        self._device_index = int(device_index)
        self._policy = policy
        self._warmup = warmup
        self._max_graphs = max_graphs
        self._owns_engine = (engine is None) if owns_engine is None else bool(owns_engine)
        self._closed = False
        self._registered: Dict[str, BufferSpec] = {}
        self._plans: Dict[ShapeKey, DecodePlan] = {}
        self._graph_ids: Dict[ShapeKey, int] = {}
        self._kernels: Dict[ShapeKey, int] = {}
        self._counters = {
            "steps": 0,
            "captures": 0,
            "replays": 0,
            "eager_steps": 0,
            "warmup_steps": 0,
            "eager_fallbacks": 0,
            "feed_copies": 0,
            "fetch_copies": 0,
        }

    # -- introspection -----------------------------------------------------
    @property
    def engine(self) -> Any:
        """The engine this hook drives, created on first use when needed."""
        self._ensure_open()
        if self._engine is None:
            from . import Engine  # local import: keeps the module free of cycles

            self._engine = Engine(
                self._device_index, backend=self._backend, policy=self._policy
            )
        return self._engine

    @property
    def plan(self) -> Optional[DecodePlan]:
        """The default geometry, or ``None``."""
        return self._plan

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise HookClosed("this DecodeHook is closed")

    # -- buffer registration ----------------------------------------------
    def register(self, plan: Optional[DecodePlan] = None) -> DecodePlan:
        """Register a plan's static buffers; idempotent per name.

        A second plan may reuse a buffer name only when the existing buffer is
        at least as large, because a captured graph keeps the address it was
        recorded with. Sharing a larger scratch buffer across octave buckets is
        the intended way to serve several geometries with one pool.
        """
        self._ensure_open()
        target = self._resolve_plan(plan)
        engine = self.engine
        for spec in target.specs:
            existing = self._engine_buffer_bytes(engine, spec.name)
            if existing is None:
                engine.register_buffer(spec.name, spec.nbytes)
                self._registered[spec.name] = spec
                continue
            if existing < spec.nbytes:
                raise ValueError(
                    f"buffer {spec.name!r} is already registered with "
                    f"{existing} bytes but plan {target.name!r} needs "
                    f"{spec.nbytes}; register the largest geometry first or give "
                    f"each geometry its own buffer names"
                )
            # The engine already owns a buffer that is big enough. Re-registering
            # would free it and dangle every pointer a captured graph holds, so
            # the existing allocation is reused as-is.
            known = self._registered.get(spec.name)
            if known is None or known.nbytes < existing:
                self._registered[spec.name] = BufferSpec(
                    name=spec.name, count=existing // dtype_size(spec.dtype),
                    dtype=spec.dtype, shape=(),
                )
        self._plans[target.key] = target
        return target

    @staticmethod
    def _engine_buffer_bytes(engine: Any, name: str) -> Optional[int]:
        """Bytes the engine already holds for ``name``, or ``None``.

        The name list is consulted first: ``buffer_bytes`` on an unknown name
        raises the native ``EngineError``, which pybind11 registers as a plain
        ``Exception`` subclass, so probing by exception alone would be brittle.
        """
        listing = getattr(engine, "buffers", None)
        if callable(listing):
            try:
                names = listing()
            except Exception:  # pragma: no cover - defensive probe
                names = None
            if names is not None and name not in names:
                return None
        getter = getattr(engine, "buffer_bytes", None)
        if not callable(getter):
            return None
        try:
            return int(getter(name))
        except Exception:  # pragma: no cover - the name list already gated this
            return None

    def _resolve_plan(self, plan: Optional[DecodePlan]) -> DecodePlan:
        target = plan if plan is not None else self._plan
        if target is None:
            raise ValueError(
                "no DecodePlan was given: pass plan= to DecodeHook or to this call"
            )
        if not isinstance(target, DecodePlan):
            raise TypeError(f"plan must be a DecodePlan, got {type(target).__name__}")
        return target

    # -- capture -----------------------------------------------------------
    def capture(
        self,
        body: Callable[[DecodeBuffers], Any],
        plan: Optional[DecodePlan] = None,
        *,
        hot_update: bool = False,
    ) -> int:
        """Record ``body`` into the graph that serves the plan's geometry.

        Returns the graph id. ``hot_update=True`` merges the fresh capture into
        an existing executable with ``cudaGraphExecUpdate`` instead of
        re-instantiating it; both outcomes are visible in ``engine.stats()``.
        """
        self._ensure_open()
        target = self.register(plan)
        engine = self.engine
        key = target.key
        if engine.has(key) and not hot_update:
            return self._graph_ids.get(key, engine.graph_id(key))

        with engine.capture(key, hot_update=hot_update) as graph_id:
            buffers = DecodeBuffers(engine, target, engine.capture_stream())
            body(buffers)

        if buffers.launches == 0:
            raise ValueError(
                "the decode body issued no kernels; a capture with no GPU work "
                "produces no graph. Launch through the DecodeBuffers argument."
            )
        self._graph_ids[key] = graph_id
        self._kernels[key] = buffers.launches
        self._counters["captures"] += 1
        return graph_id

    # -- execution ---------------------------------------------------------
    def kernels_per_step(self, plan: Optional[DecodePlan] = None) -> int:
        """Kernels the captured graph for ``plan`` holds (0 when uncaptured)."""
        target = plan if plan is not None else self._plan
        if target is None:
            return 0
        return int(self._kernels.get(target.key, 0))

    def _submit(
        self,
        body: Optional[Callable[[DecodeBuffers], Any]],
        plan: DecodePlan,
        *,
        feed: Optional[Mapping[str, Any]],
        fetch: Optional[Mapping[str, Any]],
        stream_token: int,
        replay: bool,
        synchronize: bool,
    ) -> DecodeBuffers:
        engine = self.engine
        buffers = DecodeBuffers(engine, plan, int(stream_token))

        for name, source in (feed or {}).items():
            buffers.copy_in(name, source)
            self._counters["feed_copies"] += 1

        if replay:
            engine.replay(plan.key, stream_token=int(stream_token))
            self._counters["replays"] += 1
        else:
            if body is None:
                raise ValueError("an eager submission needs the decode body")
            body(buffers)

        for name, destination in (fetch or {}).items():
            buffers.copy_out(name, destination)
            self._counters["fetch_copies"] += 1

        if synchronize:
            engine.synchronize()
        return buffers

    def execute(
        self,
        body: Callable[[DecodeBuffers], Any],
        plan: Optional[DecodePlan] = None,
        *,
        feed: Optional[Mapping[str, Any]] = None,
        fetch: Optional[Mapping[str, Any]] = None,
        stream_token: int = 0,
        eager: bool = False,
        synchronize: bool = True,
    ) -> Dict[str, Any]:
        """Run one decode step: warm-up + capture + replay, or eager.

        The first call for a geometry executes the body eagerly ``warmup`` times
        (correct output, one-time allocations done), captures it once - which
        records but does not execute - and replays it once so this call also
        returns real outputs. Every later call is a single graph launch.

        Args:
            body: ``body(buffers)`` - issues the step's kernel launches.
            plan: Geometry to use; defaults to the hook's plan.
            feed: ``{buffer name: tensor/array/pointer}`` copied in before use.
            fetch: ``{buffer name: tensor/array/pointer}`` copied out afterwards.
            stream_token: Raw stream for replay; ``0`` means "current stream".
            eager: Force the uncaptured path (for A/B measurements).
            synchronize: Block until the device is idle before returning.

        Returns:
            ``{"path": ..., "graph_id": ..., "key": ..., "launches": ...}``.
        """
        self._ensure_open()
        target = self.register(self._resolve_plan(plan))
        engine = self.engine
        key = target.key
        self._counters["steps"] += 1

        if eager:
            self._counters["eager_steps"] += 1
            buffers = self._submit(
                body, target, feed=feed, fetch=fetch,
                stream_token=stream_token, replay=False, synchronize=synchronize,
            )
            return {
                "path": "eager",
                "graph_id": self._graph_ids.get(key),
                "key": key,
                "launches": buffers.launches,
                "reason": "eager requested",
            }

        captured = engine.has(key)

        if not captured and len(self._graph_ids) >= self._max_graphs:
            # Graph budget spent: run eager and say so instead of growing the
            # pool behind the caller's back.
            self._counters["eager_steps"] += 1
            self._counters["eager_fallbacks"] += 1
            buffers = self._submit(
                body, target, feed=feed, fetch=fetch,
                stream_token=stream_token, replay=False, synchronize=synchronize,
            )
            return {
                "path": "eager",
                "graph_id": None,
                "key": key,
                "launches": buffers.launches,
                "reason": f"graph budget spent ({self._max_graphs})",
            }

        if not captured:
            for _ in range(self._warmup):
                self._submit(
                    body, target, feed=feed, fetch=fetch,
                    stream_token=stream_token, replay=False, synchronize=True,
                )
                self._counters["warmup_steps"] += 1
                self._counters["eager_steps"] += 1
            graph_id = self.capture(body, target)
            # Capture records work, it does not run it: replay once so the
            # caller receives outputs from this very call.
            self._submit(
                None, target, feed=feed, fetch=fetch,
                stream_token=stream_token, replay=True, synchronize=synchronize,
            )
            return {
                "path": "capture",
                "graph_id": graph_id,
                "key": key,
                "launches": self._kernels.get(key, 0),
                "reason": "first call for this geometry",
            }

        self._submit(
            None, target, feed=feed, fetch=fetch,
            stream_token=stream_token, replay=True, synchronize=synchronize,
        )
        return {
            "path": "replay",
            "graph_id": self._graph_ids.get(key),
            "key": key,
            "launches": self._kernels.get(key, 0),
            "reason": "captured",
        }

    def replay(
        self,
        plan: Optional[DecodePlan] = None,
        *,
        feed: Optional[Mapping[str, Any]] = None,
        fetch: Optional[Mapping[str, Any]] = None,
        stream_token: int = 0,
        synchronize: bool = True,
    ) -> Dict[str, Any]:
        """Replay an already captured geometry without running the body.

        Raises:
            KeyError: Nothing has been captured for this geometry yet.
        """
        self._ensure_open()
        target = self.register(self._resolve_plan(plan))
        if not self.engine.has(target.key):
            raise KeyError(
                f"no captured graph for {target.key} yet; call execute() or capture() first"
            )
        self._counters["steps"] += 1
        buffers = self._submit(
            None, target, feed=feed, fetch=fetch,
            stream_token=stream_token, replay=True, synchronize=synchronize,
        )
        return {
            "path": "replay",
            "graph_id": self._graph_ids.get(target.key),
            "key": target.key,
            "launches": buffers.launches,
        }

    # -- reporting and lifecycle -------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Counters plus what the hook currently holds."""
        self._ensure_open()
        payload: Dict[str, Any] = dict(self._counters)
        payload["graphs"] = len(self._graph_ids)
        payload["max_graphs"] = self._max_graphs
        payload["buffers"] = len(self._registered)
        payload["registered_bytes"] = sum(
            spec.nbytes for spec in self._registered.values()
        )
        payload["kernels_per_step"] = {
            str(key): count for key, count in self._kernels.items()
        }
        payload["plan"] = self._plan.describe() if self._plan is not None else None
        payload["warmup"] = self._warmup
        return payload

    def close(self) -> None:
        """Release the engine this hook owns; idempotent and safe in ``finally``."""
        if self._closed:
            return
        try:
            if self._owns_engine and self._engine is not None:
                self._engine.close()
        finally:
            self._engine = None
            self._registered.clear()
            self._plans.clear()
            self._graph_ids.clear()
            self._kernels.clear()
            self._closed = True

    def __enter__(self) -> "DecodeHook":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        state = "closed" if self._closed else "open"
        return (
            f"DecodeHook({state}, graphs={len(self._graph_ids)}, "
            f"buffers={len(self._registered)}, warmup={self._warmup})"
        )


def accelerate_decode(
    hook: DecodeHook,
    *,
    plan: Optional[DecodePlan] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a decode body so it is captured once and replayed afterwards.

    The decorated function keeps its own signature - ``body(buffers, *args,
    **kwargs)`` - and its arguments are forwarded on the warm-up runs. The
    replay itself never calls the body again, which is the whole point.

    Example::

        hook = DecodeHook(engine, plan=plan)

        @accelerate_decode(hook)
        def decode_step(buffers, layer_count=12):
            for _ in range(layer_count):
                buffers.scale("tmp", 0.5).relu("tmp").add("tmp2", "tmp", "x")

        decode_step()             # warm-up + capture + replay
        decode_step()             # one graph launch

    The wrapper exposes ``.hook`` and ``.frame`` (the plan it replays).
    """
    if not isinstance(hook, DecodeHook):
        raise TypeError(f"hook must be a DecodeHook, got {type(hook).__name__}")

    def decorator(body: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(body):
            raise TypeError(f"accelerate_decode needs a callable body, got {type(body).__name__}")

        @functools.wraps(body)
        def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return hook.execute(
                lambda buffers: body(buffers, *args, **kwargs), plan=plan
            )

        wrapper.hook = hook
        wrapper.frame = plan if plan is not None else hook.plan
        return wrapper

    return decorator


class HookedDecoder:
    """HuggingFace-shaped mount: ``forward(inputs) -> outputs`` over a graph.

    The class hides the feed/replay/fetch dance behind a mapping-in,
    mapping-out call, which is the shape a ``LlamaForCausalLM`` /
    ``Qwen2ForCausalLM`` decode step has once its tensors are pinned to fixed
    geometry.

    Args:
        body: ``body(buffers)`` issuing the step's kernel launches.
        plan: Geometry; ``inputs`` and ``outputs`` declare the tensor names.
        hook: Reuse an existing :class:`DecodeHook` instead of building one.
        engine: Engine to attach a new hook to (ignored when ``hook`` is given).
        backend / device_index / warmup / max_graphs: Forwarded to the hook.

    Example::

        decoder = HookedDecoder(llama_decode_body, plan, engine=engine)
        outputs = decoder.forward({"hidden": hidden})   # {"logits": ndarray}
    """

    def __init__(
        self,
        body: Callable[[DecodeBuffers], Any],
        plan: DecodePlan,
        *,
        hook: Optional[DecodeHook] = None,
        engine: Any = None,
        backend: Any = None,
        device_index: int = -1,
        warmup: int = 1,
        max_graphs: int = 4,
    ) -> None:
        if not callable(body):
            raise TypeError(f"body must be callable, got {type(body).__name__}")
        if not isinstance(plan, DecodePlan):
            raise TypeError(f"plan must be a DecodePlan, got {type(plan).__name__}")
        self._body = body
        self._plan = plan
        self._hook = (
            hook
            if hook is not None
            else DecodeHook(
                engine,
                plan=plan,
                backend=backend,
                device_index=device_index,
                warmup=warmup,
                max_graphs=max_graphs,
            )
        )
        #: Result of the last :meth:`forward` call (path, graph id, launches).
        self.last: Dict[str, Any] = {}

    # -- introspection -----------------------------------------------------
    @property
    def hook(self) -> DecodeHook:
        return self._hook

    @property
    def plan(self) -> DecodePlan:
        return self._plan

    @property
    def engine(self) -> Any:
        return self._hook.engine

    # -- execution ---------------------------------------------------------
    def forward(
        self,
        inputs: Mapping[str, Any],
        *,
        eager: bool = False,
        synchronize: bool = True,
        stream_token: int = 0,
    ) -> Dict[str, Any]:
        """Run one decode step and return the declared outputs.

        Args:
            inputs: ``{buffer name: tensor/array/pointer}`` for every input the
                plan declares.
            eager: Force the uncaptured path.
            synchronize: Block until the device is idle before returning.

        Returns:
            ``{output buffer name: numpy array}`` - flat arrays, reshaped to the
            buffer's declared shape when it has one.

        Raises:
            KeyError: A declared input is missing.
            ValueError: An input's byte size does not match its static buffer.
            BackendUnavailable: numpy is not importable.
        """
        numpy = self._numpy()
        if not isinstance(inputs, Mapping):
            raise TypeError(f"forward() needs a mapping of inputs, got {type(inputs).__name__}")
        missing = [name for name in self._plan.inputs if name not in inputs]
        if missing:
            raise KeyError(f"forward() is missing the plan inputs {missing}")

        feed: Dict[str, Any] = {}
        for name in self._plan.inputs:
            source = inputs[name]
            self._check_size(name, source)
            feed[name] = source

        outputs: Dict[str, Any] = {}
        for name in self._plan.outputs:
            spec = self._plan.spec(name)
            array = numpy.empty(spec.count, dtype=self._numpy_dtype(spec.dtype))
            outputs[name] = array.reshape(spec.shape) if spec.shape else array
        fetch = {name: array.ctypes.data for name, array in outputs.items()}

        info = self._hook.execute(
            self._body,
            self._plan,
            feed=feed,
            fetch=fetch,
            eager=eager,
            synchronize=synchronize,
            stream_token=stream_token,
        )
        self.last = dict(info)
        return outputs

    def install(self, target: Any, attribute: str = "forward") -> Callable[..., Any]:
        """Replace ``target.forward`` with :meth:`forward`; returns the old one.

        A ``LlamaForCausalLM``-shaped object only needs a ``forward`` attribute
        for this to succeed, so the mount is testable without transformers.
        """
        previous = getattr(target, attribute)
        setattr(target, attribute, self.forward)
        return previous

    def stats(self) -> Dict[str, Any]:
        """The underlying hook's counters."""
        return self._hook.stats()

    def close(self) -> None:
        """Close the hook (and, when it owns it, the engine)."""
        self._hook.close()

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _numpy() -> Any:
        try:
            import numpy
        except ImportError as error:  # pragma: no cover - numpy is a hard dep
            raise BackendUnavailable(
                "HookedDecoder needs numpy to allocate its output arrays; install "
                "numpy or drive DecodeHook.execute(fetch=...) directly"
            ) from error
        return numpy

    @staticmethod
    def _numpy_dtype(dtype: str) -> str:
        key = dtype.strip().lower()
        mapping = {
            "float32": "float32", "fp32": "float32", "float": "float32",
            "float16": "float16", "fp16": "float16", "half": "float16",
            "int8": "int8", "uint8": "uint8", "int32": "int32", "int64": "int64",
        }
        if key not in mapping:
            raise ValueError(
                f"numpy has no equivalent for {dtype!r}; pass your own destination "
                f"buffer through DecodeHook.execute(fetch=...) instead"
            )
        return mapping[key]

    def _check_size(self, name: str, source: Any) -> None:
        spec = self._plan.spec(name)
        nbytes = getattr(source, "nbytes", None)
        if callable(nbytes):
            nbytes = nbytes()
        if isinstance(nbytes, int) and nbytes != spec.nbytes:
            raise ValueError(
                f"input {name!r} is {nbytes} bytes but its static buffer is "
                f"{spec.nbytes}; a decode graph is only valid for exact geometry"
            )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"HookedDecoder({self._plan.name}, outputs={list(self._plan.outputs)})"


def data_pointer(source: Any) -> int:
    """Address of a tensor, array or raw pointer, without importing a framework.

    Accepted forms, in order:

    * ``int`` - a device or host address (``0`` is rejected)
    * an object with ``data_ptr()`` - a ``torch.Tensor``
    * an object with ``__array_interface__`` - a numpy array
    * an object with ``buffer_info()`` - ``array.array``, ``mmap``

    Host addresses are accepted because every engine copy runs through
    ``cudaMemcpyDefault`` under unified addressing, which resolves the direction
    from the pointer itself. They are not accepted where a kernel needs a device
    pointer - there, resolve the address through :meth:`DecodeBuffers.ptr`.

    Raises:
        TypeError: The object exposes no address.
        ValueError: The resolved address is null.
    """
    if isinstance(source, bool):
        raise TypeError("a bool is not an address")
    if isinstance(source, int):
        address = source
    else:
        data_ptr = getattr(source, "data_ptr", None)
        if callable(data_ptr):
            address = int(data_ptr())
        else:
            array_interface = getattr(source, "__array_interface__", None)
            if isinstance(array_interface, dict) and "data" in array_interface:
                address = int(array_interface["data"][0])
            else:
                buffer_info = getattr(source, "buffer_info", None)
                if callable(buffer_info):
                    address = int(buffer_info()[0])
                else:
                    raise TypeError(
                        f"cannot resolve an address from {type(source).__name__}; "
                        f"pass an int, a torch.Tensor, a numpy array or a buffer"
                    )
    if address <= 0:
        raise ValueError(f"{type(source).__name__} reports a null address")
    return address


def tensor_pointer(tensor: Any) -> int:
    """Device address of a torch tensor, or a precise error otherwise."""
    data_ptr = getattr(tensor, "data_ptr", None)
    if not callable(data_ptr):
        raise TypeError(
            f"tensor_pointer expects a torch.Tensor, got {type(tensor).__name__}"
        )
    address = int(data_ptr())
    if address <= 0:
        raise ValueError("tensor reports a null device pointer")
    return address


@contextlib.contextmanager
def torch_stream(engine: Any, stream_token: Optional[int] = None) -> Iterator[Any]:
    """Run torch work on the engine's capture stream.

    Yields the ``torch.cuda.ExternalStream`` so a body can issue torch ops that
    the capture actually records::

        with torch_stream(engine) as stream:
            with engine.capture(key):
                torch.matmul(hidden, weight, out=projected)

    A torch op issued on any other stream is not recorded, so the replayed graph
    would silently omit it - which is the failure this helper exists to prevent.

    Raises:
        BackendUnavailable: torch is missing or reports no usable CUDA device.
    """
    try:
        import torch
    except ImportError as error:
        raise BackendUnavailable(
            "torch_stream needs PyTorch; install a CUDA build of torch, or issue "
            "the step through DecodeBuffers (launch_scale/add/relu) instead"
        ) from error
    if not torch.cuda.is_available():
        raise BackendUnavailable(
            "torch_stream needs a usable CUDA device; torch.cuda.is_available() is False"
        )
    token = engine.capture_stream() if stream_token is None else int(stream_token)
    stream = torch.cuda.ExternalStream(token)
    with torch.cuda.stream(stream):
        yield stream
