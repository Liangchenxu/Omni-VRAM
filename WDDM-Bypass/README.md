# WDDM-Bypass

**Pre-instantiated CUDA graph replay for Windows inference: turn a model step of _N_ kernel launches into _one_ submission.**

<p align="center">
  <img alt="platform" src="https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-0078D6">
  <img alt="cuda" src="https://img.shields.io/badge/CUDA-12.1-76B900">
  <img alt="python" src="https://img.shields.io/badge/python-3.8%2B-3776AB">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="tests" src="https://img.shields.io/badge/tests-83%20passed-brightgreen">
</p>

`WDDM-Bypass` is a small C++17/CUDA runtime that captures a block of GPU work once
(`cudaStreamBeginCapture`), instantiates it (`cudaGraphInstantiate`), and then
replays that whole block forever with a single `cudaGraphLaunch`. It exists because
on Windows the *submission* side of that equation — not the GPU — is frequently the
bottleneck, and it is also the side you cannot fix inside a kernel.

- **Package:** `wddm_bypass` (Python 3.8+) at the **repository root**, native
  module `wddm_bypass._wddm_bypass` built from `WDDM-Bypass/csrc/`
- **Native version:** 1.0.0 — MIT licensed (the release version is Omni-VRAM's)
- **Part of:** [Omni-VRAM](https://github.com/Liangchenxu/Omni-VRAM) v2.6.0, as
  its execution subsystem; `vram_core.wddm_bridge` is the seam that feeds
  vram_core pools into this engine

```python
import wddm_bypass as wb

ops = wb.load_backend()                     # the compiled extension
with wb.Engine() as engine:                 # one device, one static buffer pool
    ptr = engine.register_buffer("x", 1 << 22)
    stream = engine.capture_stream()
    with engine.capture(wb.ShapeKey(1, 128, 768)):
        ops.launch_scale(ptr, 1 << 20, 1.5, stream)
        ops.launch_relu(ptr, 1 << 20, stream)
    engine.replay((1, 128, 768))            # N kernels, one submission, repeatable
    engine.synchronize()
```

---

## Table of contents

1. [Why Windows needs this](#why-windows-needs-this)
2. [Scope: what this is and is not](#scope-what-this-is-and-is-not)
3. [Architecture: Multi-Graph Slots and Hot Update](#architecture-multi-graph-slots-and-hot-update)
4. [Benchmark on an RTX 3060](#benchmark-on-an-rtx-3060)
5. [Quick start](#quick-start)
6. [Python API](#python-api)
7. [Native API](#native-api)
8. [Testing](#testing)
9. [Build requirements and troubleshooting](#build-requirements-and-troubleshooting)
10. [Design decisions](#design-decisions)
11. [Limitations](#limitations)
12. [License](#license)

---

## Why Windows needs this

On Linux a CUDA launch is close to a write into a mapped ring buffer. On Windows a
launch is a **submission batch** handed to WDDM (the Windows Display Driver Model),
which the driver schedules as it sees fit. The consequences that matter for
inference:

| Symptom | Cause |
|---|---|
| A step made of many tiny kernels runs far below its arithmetic intensity | Each launch is a separate submission; the GPU is idle between batches |
| Throughput collapses when a loop issues dozens of launches back to back | The pending-batch queue is finite, so the driver **blocks the calling thread** until the GPU drains it |
| Per-launch cost grows with queue depth | Measured here: ~2.6 µs/launch at 8 launches, ~5.9 µs/launch at 64 launches for identical kernels |
| Profilers report "GPU idle" while a thread shows 100 % CPU | The time is in the user-mode driver, not on the device |

CUDA graphs attack this directly: capture the launch sequence once, then submit it
as a single graph launch. The driver sees one batch instead of _N_, and the
per-launch cost — and the queue-depth cliff — disappears from the hot path.

> **This is not a driver hack.** Kernel-mode patching, scheduler replacement and
> "bypassing the OS" are not reachable from a user-mode process. This project does
> the only thing a user-mode process *can* do about WDDM batching: submit less
> often. Nothing in this repository touches the driver, the registry or the OS
> scheduler.

## Scope: what this is and is not

| It is | It is not |
|---|---|
| A replayable, hot-updatable CUDA graph engine with multi-graph slots | A patched driver or a WDDM scheduler bypass |
| A named static device buffer pool that makes captures allocation-free | A general-purpose caching allocator |
| A shape → graph-id router with deterministic, first-seen ids | An autotuner or a kernel fusion compiler |
| A pybind11 module you can import without PyTorch | A PyTorch-only extension (torch is optional, see `WDDM_BYPASS_WITH_TORCH`) |
| Fail-loud: a missing extension raises `BackendUnavailable` with the build command | A silent Python fallback that pretends to replay graphs |

## Architecture: Multi-Graph Slots and Hot Update

```
            user code
                |  engine.capture(key) / engine.replay(key)
                v
  wddm_bypass.Engine            facade: lifecycle rules, buffer bookkeeping
    |-- ShapeRouter ----------> BucketPolicy: shape -> bucket (octave / tile / exact)
    |                              shape -> dense graph id, deterministic on first sight
    `-- _wddm_bypass (pybind11)
            |  begin_capture / end_capture / replay / update_graph / reset
            v
  WDDMBypassEngine              std::unordered_map<int, GraphSlot>
    |-- GraphSlot[0]            GraphHandle + GraphExecHandle + shape_key + counters
    |-- GraphSlot[1]            ... one slot per shape bucket, replayed in place
    |-- StaticBuffer pool       cudaMalloc once, never inside a capture
    `-- capture stream          private, non-blocking; RAII for every handle
```

### Multi-Graph Slots

A captured graph is only replayable for the exact addresses and the exact topology
it was captured with, so a real workload needs one graph per shape *and* a policy
for keeping that set small.

- **Slots.** Each graph id owns a `GraphSlot`: the `cudaGraph_t`, its
  `cudaGraphExec_t`, the bucket key it was captured for, a one-line topology
  summary, and counters (captures, replays, updates, rebuilds).
- **Routing.** `ShapeKey(batch, sequence, hidden, dtype)` is rounded by a
  `BucketPolicy` and mapped to a **dense, first-seen** integer id. The same
  sequence of shapes therefore produces the same ids in every process and every
  run, which makes logs readable and tests deterministic.
- **Bucketing** is per dimension and chooses between three regimes: `step == 0`
  rounds up to the next power of two (default — one graph serves a whole octave of
  shapes), `step > 0` rounds up to a hardware tile, `step == -1` buckets nothing
  (`BucketPolicy.exact()`, one graph per distinct shape).
- **Static buffers.** Every buffer a replay touches is registered up front with
  `register_buffer(name, bytes)` and addressed by `buffer_ptr(name)`. Captures are
  allocation-free by construction, which also removes the most common cause of
  "operation not permitted during capture": a `cudaMalloc` inside the capture
  region.
- **Reset keeps routing.** `reset(id)` destroys the graph and its executable but
  leaves the shape → id mapping intact, so a re-capture reuses the slot instead of
  leaking a new id.

### Hot Update (`cudaGraphExecUpdate`)

Re-instantiating an executable graph is the expensive part of a shape change:
capture is cheap, `cudaGraphInstantiate` is not. `end_capture` therefore has two
paths, selected per capture:

```
with engine.capture(key, hot_update=True):   # 1. record into a staging graph
    ...                                      #    topology may differ from the live one

# 2. end_capture():
#    a. cudaGraphExecUpdate(live_exec, staging_graph)   -> in-place merge
#    b. on failure (topology too different, as CUDA reports it):
#       destroy live_exec, cudaGraphInstantiate(staging_graph)
```

The engine counts both outcomes (`stats()["updates"]`, `stats()["rebuilds"]`), so
the fallback is visible rather than silent: a workload that starts rebuilding on
every step is telling you that its topology no longer matches the slot it is
routed to. Verified on the reference box: re-capturing the same two-kernel topology
yields `updates: +1, rebuilds: +0`, while adding a third kernel to the region yields
`updates: +0, rebuilds: +1` — `cudaGraphExecUpdate` cannot add nodes to an
executable, so the engine re-instantiates it.

### Lifecycle and failure rules

- **RAII everywhere.** `GraphHandle`, `GraphExecHandle` and the buffer ownership
  types destroy exactly once; a thrown exception cannot leak a CUDA handle.
- **`CUDA_CHECK` on every call**, reporting the failed API, `file:line` and the raw
  `cudaError_t`, which surfaces in Python as `CudaError.status`.
- **Abandoned captures are cleaned up.** If the body of `Engine.capture()` raises,
  the facade finishes and discards the capture and erases the native slot, so the
  stream can never be left stuck in capture mode.
- **A closed engine is unusable, loudly.** `close()` is idempotent and safe in a
  `finally`; afterwards every method raises `EngineClosed`.
- **No silent degradation.** If the extension is missing, `Engine()` raises
  `BackendUnavailable` carrying `BUILD_HINT` — the exact command to build it.
- **One engine, one mutex.** Every native entry point serialises on an engine-level
  `std::mutex`; the Python routing tables are plain dicts guarded by the GIL.

## Benchmark on an RTX 3060

Measured on the reference box: **NVIDIA GeForce RTX 3060 (12 GB), Windows 11,
CUDA 12.1 runtime (12010), Release build, `with_torch: false`**. Three float32
buffers, each step a chain of `scale`/`add`/`relu` kernels — 8 kernels for a decode
step, 32 and 64 kernels to show what happens when a step has more nodes — driven
from CPython.

Two different things are measured, because they answer different questions:

- **Step latency** — issue the step, then `cudaDeviceSynchronize()`. This is what a
  serving loop experiences.
- **Submission cost** — issue the step and synchronise only at the end, so the GPU
  runs concurrently. This isolates the CPU-side cost of talking to the driver, which
  is exactly the part WDDM batching inflates.

| Buffer size | Kernels / step | Step latency: eager launches | Step latency: graph replay | Gain | Submission: eager | Submission: graph | Gain |
|---|---|---|---|---|---|---|---|
| 4,096 (16 KiB) | 8 | 83.3 µs | 52.9 µs | **1.57×** | 20.8 µs | 8.9 µs | **2.34×** |
| 4,096 (16 KiB) | 32 | 151.2 µs | 88.7 µs | **1.70×** | 182.1 µs | 15.1 µs | **12.1×** |
| 4,096 (16 KiB) | 64 | 406.0 µs | 131.8 µs | **3.08×** | 380.6 µs | 21.9 µs | **17.3×** |
| 65,536 (256 KiB) | 8 | 107.4 µs | 62.5 µs | **1.72×** | 45.0 µs | 14.4 µs | **3.14×** |
| 1,048,576 (4 MiB) | 8 | 318.8 µs | 288.9 µs | **1.10×** | 28.8 µs | 9.7 µs | **2.98×** |

Median of 400 (8-kernel), 300 (32-kernel) and 200 (64-kernel) iterations after warm
up. Reading the table:

- **Small kernels are where graphs pay.** The GPU work is trivial, so the step time
  is almost entirely submission. 64 launches cost 381 µs to *issue* eagerly versus
  22 µs for one graph launch — a **17×** CPU-side saving — and the caller never sees
  the queue-depth cliff that turns 2.6 µs/launch into 5.9 µs/launch.
- **Big kernels are GPU-bound** (1.10× at 4 MiB): the memory system, not WDDM, sets
  the pace, and a graph can only remove the gaps between submissions. Reporting this
  honestly is the point of the last row.
- **Graph submission cost is flat** (8.9 → 21.9 µs across 8 → 64 nodes) while eager
  submission scales with the launch count, which is the whole argument for capture:
  the code path does not depend on how many kernels the model has.

Capture-path costs on the same box (8-kernel graph over 65,536-element buffers):

| Operation | Median |
|---|---|
| Cold capture + `cudaGraphInstantiate` (new graph id) | 78.7 µs |
| `cudaGraphExecUpdate` merge of an identical topology | 70.3 µs |
| `reset` + capture + `cudaGraphInstantiate` (rebuild) | 77.9 µs |

100 consecutive hot updates ran without a single rebuild (`updates: +100`,
`rebuilds: +0`), i.e. the merge path is exercised in practice, not just in tests.
For an 8-node graph the merge saves ~10 % over re-instantiating; the mechanism is
the same one that dominates for large graphs, where instantiation, not capture,
is the expensive half.

> These numbers are from one machine and one run. Absolute values move with driver
> version and clock state; the ratios are the durable part, and the harness that
> produced them is a throwaway script that is deliberately not part of this tree
> (see [Testing](#testing)).

## Quick start

### 1. Build the extension in place

```powershell
# from the repository root
cd WDDM-Bypass
python setup.py build_ext --inplace     # -> ../wddm_bypass/_wddm_bypass*.pyd
```

The build is self-diagnosing: `setup.py` locates `nvcc`, imports the MSVC
environment from `vcvars64.bat`, repairs a stale Windows SDK `INCLUDE`/`LIB`, and
adds the two flags CUDA 12.1 needs for a modern MSVC. If the toolchain is
incomplete it prints the reason and installs the pure-Python layer only — it never
produces a half-linked module.

```powershell
python -c "import wddm_bypass as wb; print(wb.build_info())"
# {'version': '1.0.0', 'with_torch': False, 'cuda_runtime_version': 12010,
#  'compiled_cuda_version': 12010, 'build_type': 'Release'}
```

### 2. Capture a step and replay it

```python
import wddm_bypass as wb

ops = wb.load_backend()          # wddm_bypass._wddm_bypass
key = wb.ShapeKey(1, 128, 768)   # batch, sequence, hidden, dtype='float32'

with wb.Engine() as engine:
    ptr = engine.register_buffer("hidden", 1 << 22)      # 4 MiB of float32
    count = (1 << 22) // 4

    stream = engine.capture_stream()                     # capture happens here
    with engine.capture(key) as graph_id:
        ops.launch_scale(ptr, count, 1.5, stream)        # recorded, not executed
        ops.launch_relu(ptr, count, stream)
    print("graph", graph_id, engine.describe(key))

    for _ in range(1000):
        engine.replay(key)                               # one submission per step
    engine.synchronize()

    print(engine.stats())
```

Two rules make capture work:

1. **Issue every launch on `engine.capture_stream()`.** Capture records what the
   stream sees; a launch on another stream is not part of the graph (it just runs
   eagerly once). For PyTorch work the raw handle is what
   `torch.cuda.ExternalStream` expects, so the same region can wrap tensor ops.
2. **Allocate nothing inside the region.** Register buffers first, then capture.
   `cudaMalloc` during capture is illegal and is the usual cause of
   `cudaErrorStreamCaptureUnsupported`.

### 3. Re-shape without re-instantiating

```python
with engine.capture(key, hot_update=True):   # same topology: merge in place
    ops.launch_scale(ptr, count, 1.5, stream)
    ops.launch_relu(ptr, count, stream)
engine.stats()["updates"], engine.stats()["rebuilds"]   # -> 1, 0

# Changing the node count is not mergeable, so CUDA reports it and the engine
# re-instantiates instead of merging. The counters say which path ran.
with engine.capture(key, hot_update=True):
    ops.launch_scale(ptr, count, 1.5, stream)
    ops.launch_relu(ptr, count, stream)
    ops.launch_relu(ptr, count, stream)
engine.stats()["updates"], engine.stats()["rebuilds"]   # -> 1, 1
```

## Python API

Everything below lives in `wddm_bypass` and is importable on any machine; only
`Engine()` and `load_backend()` need the compiled module.

| Member | Purpose |
|---|---|
| `Engine(device_index=-1, *, backend=None, policy=None)` | Facade over one device. `-1` keeps the caller's current device; `backend` is the test injection point |
| `engine.capture(key, hot_update=False)` | Context manager: `begin_capture` … `end_capture`, and discards the capture if the body raises |
| `engine.begin_capture(key, hot_update=False)` / `end_capture(key)` | Explicit form of the same lifecycle |
| `engine.capture_stream()` | Raw `cudaStream_t` (as an int) that captures must be issued on |
| `engine.is_capturing(key)` | True while a capture region is open |
| `engine.replay(key)` / `engine.run(key)` | Submit the graph for `key` — one launch for the whole chain |
| `engine.update(key)` | Merge the pending staging graph into the executable |
| `engine.synchronize()` | Block until the device is idle |
| `engine.register_buffer(name, bytes)` | Allocate or replace a static device buffer; returns its address |
| `engine.buffer_ptr(name)` / `buffer_bytes(name)` / `buffers()` | Address, size, and the sorted buffer names |
| `engine.copy_into(name, src)` / `read_buffer(name, dst)` / `zero_buffer(name)` | Async H2D/D2D/D2H copies and zero fill |
| `engine.release_buffer(name)` | Free a buffer (unknown names are ignored) |
| `engine.graphs()` / `has(key)` / `reset(key)` / `reset_all()` | Slot inventory and teardown; `reset` keeps the shape → id mapping |
| `engine.stats()` | Native counters merged with routing and buffer counters |
| `engine.describe(key)` | One-line JSON topology summary for a slot |
| `engine.close()` / `with Engine() as e:` | Idempotent teardown; afterwards everything raises `EngineClosed` |
| `ShapeKey(batch, sequence, hidden, dtype="float32")`, `coerce_key(x)` | Shape identity; tuples, strings and keys are interchangeable |
| `BucketPolicy()` / `.exact()` / `BucketPolicy(64, 0, 0)` | Octave (default), per-shape, or tile-aligned routing |
| `ShapeRouter(policy)`, `engine.router` | Shape → graph id, deterministic and first-seen |
| `load_backend()`, `backend_available()`, `build_info()` | Native module access; `BackendUnavailable` + `BUILD_HINT` on failure |

`stats()` reports `graphs`, `static_buffers`, `static_bytes`, `captures`, `replays`,
`updates`, `rebuilds`, `allocations`, `capturing`, `device_index`, `torch_interop`,
plus `shapes`, `shape_hits`, `shape_misses`, `buffers`, `bucket_policy` and
`backend_version` from the facade.

## Native API

`wddm_bypass._wddm_bypass` is a self-contained pybind11 module: no PyTorch import,
one mutex, integer/string arguments only.

| Symbol | Notes |
|---|---|
| `WDDMBypassEngine(device_index=-1)` | Owns graphs, executables and the buffer pool |
| `.begin_capture(graph_id, shape_key="", hot_update=False)` | Records onto the engine's private non-blocking stream |
| `.end_capture(graph_id)` | Instantiates, or merges when `hot_update` and an executable exist |
| `.replay(graph_id, stream_token=0)` | `stream_token=0` means the caller's current stream (PyTorch's when built with `WDDM_BYPASS_WITH_TORCH=1`) |
| `.update_graph(id)`, `.reset(id)`, `.reset_all()`, `.has_graph(id)`, `.list_graphs()`, `.graph_id_for_shape(key)`, `.is_capturing(id)`, `.describe(id)` | Slot management |
| `.register_static_buffer(name, bytes)`, `.static_buffer_ptr(name)`, `.static_buffer_bytes(name)`, `.release_static_buffer(name)` | Buffer pool; addresses are plain integers |
| `.copy_to_static_input(name, src, bytes, stream_token=0)`, `.copy_from_static(name, dst, bytes, stream_token=0)`, `.zero_static_buffer(name, stream_token=0)` | Async copies; `src`/`dst` are host or device addresses |
| `.capture_stream_ptr()`, `.stats()`, `.synchronize()` | Introspection |
| `launch_scale(dst, count, alpha, stream_token=0)`, `launch_add(dst, lhs, rhs, count, stream_token=0)`, `launch_relu(dst, count, stream_token=0)` | Grid-stride kernels, 256 threads/block, ≤ 65,535 blocks; `count` is elements |
| `cuda_graph_supported()`, `device_name()`, `runtime_version()`, `build_info()` | Capability probe |
| `EngineError`, `CudaError` | `CudaError` subclasses `EngineError` and carries `.status` (the raw `cudaError_t`) |

## Testing

```powershell
# from the repository root (the directory holding vram_core/ and wddm_bypass/)
python -m pytest tests/test_engine_suite.py -v
# 86 passed, 0 skipped (1.9 s)
```

86 tests, zero skips on a built checkout: the engine cases below plus the joint
`vram_core` ones (pool -> static buffer -> routed replay, the budget/pressure
gates, and the bridge helpers). The engine cases come in two halves:

| Half | What it covers |
|---|---|
| Pure Python, against an injected fake backend | `ShapeKey` validation, rounding helpers, bucketing and first-seen routing, and the facade's exact native call sequence (method, arguments, order) for capture, replay, update, buffers and lifecycle |
| GPU, against the compiled extension | The binding surface matches the documented list, capture covers the work and replay repeats it bit-exactly, static buffer zero/release round trips, `reset` discards the executable, buffer bounds are rejected before the native call, and one graph launch carries the whole chain |

Also gated in the same suite: every source under `csrc/` must be **ASCII-only**
(nvcc + MSVC mis-parse non-ASCII without `/utf-8`), every bound method must be
declared in `graph_manager.h`, and `setup.py`, the package and the extension must
agree on the version string.

The GPU tests `skip` with an explicit reason only when the extension is not built
or no device can instantiate graphs; they never pass silently. The benchmark tables
above came from a throwaway harness kept out of the tree — the suite asserts
behaviour, not timings, so CI never fails on a noisy clock.

## Build requirements and troubleshooting

| Requirement | Detail |
|---|---|
| CUDA Toolkit 12.1 | `setup.py` probes `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe` **before** the environment, then falls back to `CUDA_HOME`, `CUDA_PATH` and `PATH` |
| MSVC 2022 | 19.3x is whitelisted by CUDA 12.1; 19.4x is not, so `-allow-unsupported-compiler` plus `/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` are added unconditionally on Windows |
| Windows SDK 10 | `vcvars64.bat` resolves the kit through the registry key `KitsRoot10`; a stale key is detected and `INCLUDE`/`LIB` are rebuilt from the kit that actually exists |
| PyTorch, build time only | Supplies the pybind11 headers through `torch.utils.cpp_extension`. libtorch is **not** linked unless `WDDM_BYPASS_WITH_TORCH=1`, so the built module only needs `cudart64_12.dll` at import |
| CPython 3.8+ | Built and tested with CPython 3.10 (the extension is ABI-tagged `cp310`) |

| Symptom | Cause / fix |
|---|---|
| `unsupported Microsoft Visual Studio version` | CUDA 12.1 does not know MSVC 19.4x; the flags above already override it (or move to CUDA 12.4+) |
| `cannot open include file: 'corecrt.h'` | The Windows SDK is missing or `KitsRoot10` points at a deleted folder; install the SDK (`winget install Microsoft.WindowsSDK.10.0.22621`) |
| `ImportError: DLL load failed ... cudart64_12.dll` | Python 3.8+ loads extensions with restricted search, so `PATH` alone is not enough; `wddm_bypass.backend` registers the toolkit `bin` directory with `os.add_dll_directory` — set `CUDA_HOME`/`CUDA_PATH` if it still fails |
| `wddm_bypass.BackendUnavailable` | The extension was never built on this machine: run `python setup.py build_ext --inplace` |
| `torch CUDA 11.8` vs `nvcc 12.1` warning | torch's major.minor guard is neutralised for this header-only build and the mismatch is reported in the setup log; the module links only `cudart` |

## Design decisions

- **No Python fallback.** A graph engine that silently degrades to eager launches
  would be worse than an import error, because the caller would never learn that the
  latency win is gone. Missing backend ⇒ `BackendUnavailable` with the build command.
- **PyTorch is a build-time dependency, not a runtime one.** `CUDAExtension` links
  libtorch into everything it describes; those libraries are stripped so the `.pyd`
  can be imported by a bare CPython. Opt in with `WDDM_BYPASS_WITH_TORCH=1` when you
  want `replay()` to follow torch's current stream natively.
- **Static buffers instead of an allocator.** A capture must not allocate, and a
  replay must not either. `register_static_buffer` makes the address set fixed for
  the lifetime of the graph, which is what makes capture safe and replay
  deterministic.
- **Deterministic ids.** First-seen ordering means two runs that see the same shapes
  produce the same graph ids, so logs, tests and `describe()` output are comparable.
- **ASCII-only CUDA sources.** Enforced by a test: nvcc plus MSVC on a non-UTF-8
  code page will mis-parse anything else, and the failure mode is a confusing syntax
  error inside a header.
- **Timings are not tests.** The suite asserts behaviour; the benchmark table is a
  measurement on one machine, not a regression gate.

## Limitations

- **Windows is the target and the only platform validated here.** The engine is
  plain CUDA and builds elsewhere, but the latency story (and the MSVC/SDK part of
  `setup.py`) is Windows-specific.
- **One device per `Engine`.** `device_index` is fixed at construction; use one
  engine per device.
- **A graph is replayable only for what it captured.** Pointer and topology changes
  belong to `reset` + re-capture, or to `hot_update` when the topology is close
  enough for `cudaGraphExecUpdate`.
- **Capture is a stream property.** While a region is open, other threads must not
  submit work that the capture stream would pick up; the engine uses a private
  non-blocking stream to keep that window small, but it is not a multi-threaded
  capture API.
- **Static buffers are caller-sized, not a general allocator.** There is no
  fragmentation handling and no aliasing analysis.
- **Graph launch is not free.** A ~9–22 µs graph launch dominates a step of one or
  two tiny kernels; capture pays off when a step has many kernels or when the
  submission queue is the bottleneck.

## License

MIT — see [LICENSE](../LICENSE). The project is part of
[Omni-VRAM](https://github.com/Liangchenxu/Omni-VRAM).



