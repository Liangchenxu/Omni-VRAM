# WDDM-Bypass v1.0.0 - Filtered Execution Plan (Step 1 deliverable)

Scope: turn the single-file `vram_hacker.cu` POC into a reusable, module-level
WDDM scheduling-bypass / dynamic CUDA-graph engine with its own CMake-free build,
Python API and unit tests.

---

## 1. Requirement filter (what survives the cut)

| # | Raw requirement | Verdict | Engineering form |
|---|-----------------|---------|------------------|
| R1 | "Bypass WDDM scheduling" | KEEP, renamed | The only user-mode lever over WDDM batch-submission latency is a **single, pre-instantiated graph submission** instead of N kernel launches. Implemented as `cudaGraphLaunch` on one exec handle, replayable indefinitely. No kernel-mode / driver patching is attempted (impossible and unsafe from user mode). |
| R2 | Manage several graphs | KEEP | `std::unordered_map<int, GraphSlot>` keyed by a stable int32 graph id. |
| R3 | Manage a memory pool | KEEP, narrowed | Named **static device buffers** (`cudaMalloc` once, reused by every replay) so a captured graph never allocates. Real caching allocators are out of scope for v1.0.0. |
| R4 | `update_graph(int)` hot update | KEEP | `cudaGraphExecUpdate` against a freshly captured staging graph, with the documented destroy-and-reinstantiate fallback. |
| R5 | Zero-copy static buffers | KEEP | `register_static_buffer` + `copy_to_static_input` (async D2D/H2D). |
| R6 | Python bindings | KEEP | pybind11 module `wddm_bypass._wddm_bypass`. |
| R7 | Unit tests | KEEP | `tests/test_engine_suite.py` (pytest, mirrors repo conventions). |
| R8 | Benchmarks | KEEP, minimal | One in-suite latency test (graph replay vs. N separate launches). |
| R9 | "Hack the driver / bypass the OS scheduler" | DROP | Out of process reach; not implementable in user mode. Documented in README rather than faked. |

## 2. Module layering

```
WDDM-Bypass/
  csrc/include/graph_manager.h   GraphSlot / StaticBuffer / RAII handles / WDDMBypassEngine API + CUDA_CHECK
  csrc/graph_engine.cu           engine implementation + 3 elementwise kernels
  csrc/bypass_core.cpp           pybind11 module, engine class, free-function launchers
  csrc/torch_interop.cpp         optional c10 stream + at::Tensor conveniences (WDDM_BYPASS_WITH_TORCH)
  wddm_bypass/__init__.py        public API: Engine wrapper, capture_context, bucketed routing
  wddm_bypass/bucketing.py       ShapeKey + deterministic shape -> graph id routing
  tests/test_engine_suite.py     pytest suite (pure-Python logic + GPU-gated CUDA tests)
```

## 3. Mandated implementation rules - compliance map

| Rule | Where it is honoured |
|------|----------------------|
| Top-level `os.environ['CUDA_HOME']` | `setup.py` line 1 block, before any `import torch` |
| `os.environ['DISTUTILS_USE_SDK'] = "1"` | `setup.py`, paired with an explicit `vcvars64.bat` environment import so MSVC headers/libs are actually reachable |
| `BuildExtension.with_options(use_ninja=False)` | `setup.py` `cmdclass["build_ext"]` |
| `.cu` ASCII-only | every file under `csrc/` (verified by a gate test) |
| `CUDA_CHECK` on every CUDA call | `graph_manager.h`; every runtime call in `graph_engine.cu` is wrapped, except destructor paths which are `noexcept` best-effort |
| RAII for graph / exec handles | `GraphHandle`, `GraphExecHandle`, `StaticBuffer::DeviceBuffer` (`unique_ptr` + `cudaFree`) |
| `c10::cuda::getCurrentCUDAStream().stream()` | `detail::stream_from_token(0)` when `WDDM_BYPASS_WITH_TORCH` is defined |

## 4. Build feasibility gate - RESULT: BLOCKED on this machine

Two independent, reproduced failures (nvcc 12.1 + MSVC 19.44.35207):

1. `crt/host_config.h(153): fatal error C1189: #error: -- unsupported Microsoft
   Visual Studio version! Only the versions between 2017 and 2022 (inclusive)
   are supported!`
2. With `-allow-unsupported-compiler`:
   `D:\vs\VC\Tools\MSVC\14.44.35207\include\crtdefs.h(10): fatal error C1083:
   cannot open include file: 'corecrt.h': No such file or directory`

Root causes:

- **Windows SDK (UCRT) is not installed.** `HKLM\SOFTWARE\Microsoft\Windows Kits\Installed Roots\KitsRoot10`
  still points at `D:\Windows Kits\10\`, but that directory does not exist. `vcvars64.bat`
  therefore exports only 2 `INCLUDE` entries (MSVC + VS aux) and 1 `LIB` entry, so
  `corecrt.h`, `windows.h` and `kernel32.lib` are all missing. No host compilation is
  possible at all, for `.cu`, `.cpp` or pybind11 alike.
- **MSVC 19.44 is newer than CUDA 12.1 knows about** (`_MSC_VER` table ends at VS 2022 17.9).
  Needs `-allow-unsupported-compiler`, or a newer toolkit.
- **torch is `2.7.1+cu118` while the installed nvcc is 12.1**, so torch's
  `_check_cuda_version` would reject the extension link step even after the SDK is installed.

Remediation the user must pick (cannot be resolved from inside the repo):

```powershell
# 1) Windows SDK (required, no way around it)
winget install --id Microsoft.WindowsSDK.10.0.22621 --accept-source-agreements --accept-package-agreements
# 2) make torch and nvcc agree (cu121 build matches the installed toolkit)
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.7.1
```

Consequence for this drop: the C++/CUDA sources are written and gated by tests that
`pytest.skip` with the precise reason when the extension is absent. The pure-Python
layer (routing, bucketing, context-manager contract, lifecycle bookkeeping, error
surface) is fully exercised today through an injected fake backend, so the suite is
green on this machine and becomes a full GPU suite the moment the two packages land.
