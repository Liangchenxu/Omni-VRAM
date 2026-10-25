"""Deterministic shape bucketing and graph-id routing.

A captured CUDA graph is only replayable for the exact shapes it was captured
with, so a real workload needs one graph per shape. This module turns that
requirement into a stable mapping::

    ShapeKey(batch, sequence, hidden, dtype) -> coarse bucket -> graph id

Graph ids are handed out in first-seen order, so the same sequence of shapes
always produces the same ids across processes and runs. That determinism is what
makes routing testable and makes engine logs readable.

Pure Python on purpose: no torch, no numpy, no CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Union

__all__ = [
    "BucketPolicy",
    "ShapeKey",
    "ShapeRouter",
    "coerce_key",
    "power_of_two_at_least",
    "round_up",
]


def round_up(value: int, step: int) -> int:
    """Round ``value`` up to the next multiple of ``step`` (``step`` > 0)."""
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    return ((value + step - 1) // step) * step


def power_of_two_at_least(value: int) -> int:
    """Smallest power of two that is greater than or equal to ``value``."""
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    result = 1
    while result < value:
        result <<= 1
    return result


@dataclass(frozen=True)
class ShapeKey:
    """One execution shape, e.g. ``ShapeKey(1, 128, 768, "float16")``.

    Instances are hashable and comparable, so they can be used directly as
    dictionary keys in the router.
    """

    batch: int
    sequence: int
    hidden: int
    dtype: str = "float32"

    def __post_init__(self) -> None:
        for name in ("batch", "sequence", "hidden"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if not isinstance(self.dtype, str):
            raise TypeError(f"dtype must be a str, got {type(self.dtype).__name__}")
        normalised = self.dtype.strip().lower()
        if not normalised:
            raise ValueError("dtype must not be empty")
        if normalised != self.dtype:
            # Normalise once, at construction, so equality stays shape-based.
            object.__setattr__(self, "dtype", normalised)

    @property
    def elements(self) -> int:
        """Number of scalar elements in one tensor of this shape."""
        return self.batch * self.sequence * self.hidden

    def as_tuple(self) -> Tuple[int, int, int, str]:
        """The four fields as a plain tuple."""
        return (self.batch, self.sequence, self.hidden, self.dtype)

    def __str__(self) -> str:
        return f"{self.batch}x{self.sequence}x{self.hidden}:{self.dtype}"

    @classmethod
    def parse(cls, text: str) -> "ShapeKey":
        """Inverse of :meth:`__str__`; raises ``ValueError`` on a bad string."""
        if not isinstance(text, str):
            raise TypeError(f"expected a str, got {type(text).__name__}")
        shape_part, separator, dtype = text.partition(":")
        if not separator:
            shape_part, dtype = text, "float32"
        pieces = shape_part.lower().split("x")
        if len(pieces) != 3:
            raise ValueError(f"expected '<batch>x<sequence>x<hidden>', got {text!r}")
        try:
            batch, sequence, hidden = (int(piece) for piece in pieces)
        except ValueError as error:
            raise ValueError(f"non-numeric dimension in {text!r}") from error
        return cls(batch, sequence, hidden, dtype)


KeyLike = Union[ShapeKey, Tuple[int, int, int], Tuple[int, int, int, str]]


def coerce_key(value: KeyLike) -> ShapeKey:
    """Accept a :class:`ShapeKey`, a 3-tuple or a 4-tuple and return a key."""
    if isinstance(value, ShapeKey):
        return value
    if isinstance(value, (tuple, list)):
        if len(value) == 3:
            return ShapeKey(value[0], value[1], value[2])
        if len(value) == 4:
            return ShapeKey(value[0], value[1], value[2], value[3])
        raise ValueError(f"expected a 3- or 4-tuple, got {len(value)} items")
    if isinstance(value, str):
        return ShapeKey.parse(value)
    raise TypeError(f"cannot interpret {type(value).__name__} as a ShapeKey")


@dataclass(frozen=True)
class BucketPolicy:
    """How a shape is rounded before it is mapped to a graph id.

    Each ``*_step`` field accepts three kinds of value:

    ``step > 0``
        Round up to the next multiple of ``step`` - use this to align to
        hardware-friendly tiles, e.g. ``sequence_step=64``.
    ``step == 0``
        Round up to the next power of two. This is the default: one graph serves
        a whole octave of shapes, so the graph count stays logarithmic.
    ``step == -1``
        No rounding, one graph per distinct shape - see :meth:`exact`.
    """

    batch_step: int = 0
    sequence_step: int = 0
    hidden_step: int = 0

    def __post_init__(self) -> None:
        for name in ("batch_step", "sequence_step", "hidden_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < -1:
                raise ValueError(f"{name} must be -1, 0 or positive, got {value}")

    @classmethod
    def exact(cls) -> "BucketPolicy":
        """A policy that buckets nothing: one graph per distinct shape."""
        return cls(-1, -1, -1)

    def _round(self, value: int, step: int) -> int:
        if step == -1:
            return value
        if step == 0:
            return power_of_two_at_least(value)
        return round_up(value, step)

    def bucket(self, key: KeyLike) -> ShapeKey:
        """The bucket key that serves ``key``."""
        shape = coerce_key(key)
        return ShapeKey(
            self._round(shape.batch, self.batch_step),
            self._round(shape.sequence, self.sequence_step),
            self._round(shape.hidden, self.hidden_step),
            shape.dtype,
        )

    def describe(self) -> str:
        """Human readable summary, used in logs and ``Engine.stats()``."""
        return (
            f"batch_step={self.batch_step} sequence_step={self.sequence_step} "
            f"hidden_step={self.hidden_step}"
        )


class ShapeRouter:
    """Maps shapes onto dense, first-seen graph ids.

    >>> router = ShapeRouter(BucketPolicy.exact())
    >>> router.graph_id(ShapeKey(1, 128, 768))
    0
    >>> router.graph_id((1, 128, 768))
    0
    >>> router.graph_id(ShapeKey(1, 256, 768))
    1
    """

    def __init__(self, policy: Optional[BucketPolicy] = None) -> None:
        if policy is not None and not isinstance(policy, BucketPolicy):
            raise TypeError(f"policy must be a BucketPolicy, got {type(policy).__name__}")
        self._policy = policy if policy is not None else BucketPolicy()
        self._ids: Dict[ShapeKey, int] = {}
        self._buckets: List[ShapeKey] = []
        self._hits = 0
        self._misses = 0

    # -- configuration -----------------------------------------------------
    @property
    def policy(self) -> BucketPolicy:
        """The active rounding policy (never ``None``)."""
        return self._policy

    def bucket(self, key: KeyLike) -> ShapeKey:
        """The :class:`ShapeKey` bucket that ``key`` routes to."""
        return self._policy.bucket(key)

    # -- routing -----------------------------------------------------------
    def graph_id(self, key: KeyLike) -> int:
        """Stable graph id for ``key``; allocates a new id on first sight."""
        bucket = self._policy.bucket(key)
        known = self._ids.get(bucket)
        if known is not None:
            self._hits += 1
            return known

        identifier = len(self._buckets)
        self._ids[bucket] = identifier
        self._buckets.append(bucket)
        self._misses += 1
        return identifier

    def lookup(self, key: KeyLike) -> Optional[int]:
        """The graph id already assigned to ``key``, or ``None``."""
        return self._ids.get(self._policy.bucket(key))

    def bucket_for(self, graph_id: int) -> ShapeKey:
        """The bucket that owns ``graph_id``; raises ``KeyError`` if unknown."""
        if isinstance(graph_id, bool) or not isinstance(graph_id, int):
            raise TypeError(f"graph_id must be an int, got {type(graph_id).__name__}")
        if graph_id < 0 or graph_id >= len(self._buckets):
            raise KeyError(f"no shape is routed to graph id {graph_id}")
        return self._buckets[graph_id]

    # -- bookkeeping -------------------------------------------------------
    def __len__(self) -> int:
        return len(self._buckets)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, (ShapeKey, tuple, list, str)) and self.lookup(key) is not None

    def __iter__(self) -> Iterator[Tuple[int, ShapeKey]]:
        return iter(list(enumerate(self._buckets)))

    def buckets(self) -> List[ShapeKey]:
        """Snapshot of the assigned buckets, in id order."""
        return list(self._buckets)

    def stats(self) -> Dict[str, int]:
        """Hit/miss counters, shaped like the engine's own stats dict."""
        return {
            "shapes": len(self._buckets),
            "shape_hits": self._hits,
            "shape_misses": self._misses,
        }

    def clear(self) -> None:
        """Forget every mapping; ids restart at zero."""
        self._ids.clear()
        self._buckets.clear()
        self._hits = 0
        self._misses = 0
