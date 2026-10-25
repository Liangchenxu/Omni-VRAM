// WDDM-Bypass v1.0.0 - engine implementation.
//
// Everything here is host-side CUDA Runtime work plus three trivial elementwise
// kernels that exist so a graph can be built and replayed without any tensor
// framework. ASCII only.

#include "graph_manager.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef WDDM_BYPASS_WITH_TORCH
#include <c10/cuda/CUDAStream.h>
#endif

namespace wddm_bypass {

namespace detail {

void throw_cuda_error(cudaError_t status,
                      const char* expression,
                      const char* file,
                      int line) {
    if (status == cudaSuccess) {
        return;
    }

    const char* name = cudaGetErrorName(status);
    const char* text = cudaGetErrorString(status);

    std::string message;
    message.reserve(192);
    message.append("CUDA error ");
    message.append(name != nullptr ? name : "unknown");
    message.append(": ");
    message.append(text != nullptr ? text : "no description available");
    message.append("\n  at ");
    message.append(file != nullptr ? file : "<unknown file>");
    message.append(":");
    message.append(std::to_string(line));
    message.append("\n  in ");
    message.append(expression != nullptr ? expression : "<unknown expression>");

    // Clear the sticky per-thread error so the next call observes its own
    // status rather than this one's.
    (void)cudaGetLastError();

    throw CudaError(message, status);
}

cudaStream_t stream_from_token(std::uintptr_t token) noexcept {
    if (token != 0) {
        return reinterpret_cast<cudaStream_t>(token);
    }
#ifdef WDDM_BYPASS_WITH_TORCH
    // Mandated interop: follow whatever stream PyTorch has made current.
    try {
        return c10::cuda::getCurrentCUDAStream().stream();
    } catch (...) {
        return static_cast<cudaStream_t>(nullptr);
    }
#else
    return static_cast<cudaStream_t>(nullptr);
#endif
}

bool built_with_torch() noexcept {
#ifdef WDDM_BYPASS_WITH_TORCH
    return true;
#else
    return false;
#endif
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 65535;

int block_count(std::int64_t count) {
    if (count <= 0) {
        return 0;
    }
    const std::int64_t blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return static_cast<int>(std::min<std::int64_t>(blocks, kMaxBlocks));
}

__global__ void scale_kernel(float* dst, float alpha, long long count) {
    const long long stride =
        static_cast<long long>(gridDim.x) * static_cast<long long>(blockDim.x);
    for (long long index =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        dst[index] *= alpha;
    }
}

__global__ void add_kernel(float* dst,
                           const float* lhs,
                           const float* rhs,
                           long long count) {
    const long long stride =
        static_cast<long long>(gridDim.x) * static_cast<long long>(blockDim.x);
    for (long long index =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        dst[index] = lhs[index] + rhs[index];
    }
}

__global__ void relu_kernel(float* dst, long long count) {
    const long long stride =
        static_cast<long long>(gridDim.x) * static_cast<long long>(blockDim.x);
    for (long long index =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const float value = dst[index];
        dst[index] = value > 0.0f ? value : 0.0f;
    }
}

void require_pointer(std::uintptr_t pointer, const char* what) {
    if (pointer == 0) {
        std::string message("WDDM-Bypass: null device pointer passed to ");
        message.append(what);
        throw EngineError(message);
    }
}

}  // namespace

namespace ops {

void launch_scale(std::uintptr_t dst,
                  std::int64_t count,
                  float alpha,
                  std::uintptr_t stream_token) {
    require_pointer(dst, "launch_scale");
    if (count <= 0) {
        return;
    }
    const cudaStream_t stream = detail::stream_from_token(stream_token);
    scale_kernel<<<block_count(count), kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<float*>(dst), alpha, static_cast<long long>(count));
}

void launch_add(std::uintptr_t dst,
                std::uintptr_t lhs,
                std::uintptr_t rhs,
                std::int64_t count,
                std::uintptr_t stream_token) {
    require_pointer(dst, "launch_add");
    require_pointer(lhs, "launch_add");
    require_pointer(rhs, "launch_add");
    if (count <= 0) {
        return;
    }
    const cudaStream_t stream = detail::stream_from_token(stream_token);
    add_kernel<<<block_count(count), kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<float*>(dst),
        reinterpret_cast<const float*>(lhs),
        reinterpret_cast<const float*>(rhs),
        static_cast<long long>(count));
}

void launch_relu(std::uintptr_t dst,
                 std::int64_t count,
                 std::uintptr_t stream_token) {
    require_pointer(dst, "launch_relu");
    if (count <= 0) {
        return;
    }
    const cudaStream_t stream = detail::stream_from_token(stream_token);
    relu_kernel<<<block_count(count), kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<float*>(dst), static_cast<long long>(count));
}

}  // namespace ops

// ---------------------------------------------------------------------------
// Capability probes
// ---------------------------------------------------------------------------

bool cuda_graph_supported() {
    // Real probe rather than a guess: capture and discard an empty graph.
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }

    bool supported = false;
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) ==
        cudaSuccess) {
        cudaGraph_t graph = nullptr;
        if (cudaStreamEndCapture(stream, &graph) == cudaSuccess) {
            supported = true;
            if (graph != nullptr) {
                (void)cudaGraphDestroy(graph);
            }
        }
    }

    (void)cudaStreamDestroy(stream);
    (void)cudaGetLastError();
    return supported;
}

std::string device_name() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return std::string();
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        (void)cudaGetLastError();
        return std::string();
    }
    return std::string(properties.name);
}

int runtime_version() {
    int version = 0;
    if (cudaRuntimeGetVersion(&version) != cudaSuccess) {
        (void)cudaGetLastError();
        return CUDART_VERSION;
    }
    return version;
}

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------

const char* WDDMBypassEngine::node_type_name(cudaGraphNodeType type) noexcept {
    switch (type) {
        case cudaGraphNodeTypeKernel:
            return "Kernel";
        case cudaGraphNodeTypeMemcpy:
            return "Memcpy";
        case cudaGraphNodeTypeMemset:
            return "Memset";
        case cudaGraphNodeTypeHost:
            return "Host";
        case cudaGraphNodeTypeGraph:
            return "Graph";
        case cudaGraphNodeTypeEmpty:
            return "Empty";
        case cudaGraphNodeTypeWaitEvent:
            return "WaitEvent";
        case cudaGraphNodeTypeEventRecord:
            return "EventRecord";
        case cudaGraphNodeTypeExtSemaphoreSignal:
            return "ExtSemaphoreSignal";
        case cudaGraphNodeTypeExtSemaphoreWait:
            return "ExtSemaphoreWait";
        case cudaGraphNodeTypeMemAlloc:
            return "MemAlloc";
        case cudaGraphNodeTypeMemFree:
            return "MemFree";
        default:
            return "Unknown";
    }
}

WDDMBypassEngine::WDDMBypassEngine(int device_index) {
    if (device_index < 0) {
        CUDA_CHECK(cudaGetDevice(&device_index_));
    } else {
        device_index_ = device_index;
        CUDA_CHECK(cudaSetDevice(device_index_));
    }

    // Created up front so capture_stream_ptr() is already valid before the first
    // capture begins: callers need that value to launch the work being recorded.
    ensure_capture_stream();
}

WDDMBypassEngine::~WDDMBypassEngine() {
    try {
        reset_all();
    } catch (...) {
        // Destructors must not throw; the CUDA context may already be gone.
    }
    if (capture_stream_ != nullptr) {
        (void)cudaStreamDestroy(capture_stream_);
        capture_stream_ = nullptr;
    }
}

void WDDMBypassEngine::ensure_capture_stream() {
    if (capture_stream_ == nullptr) {
        // Non-blocking so the capture stream never implicitly synchronizes with
        // the legacy default stream, which is illegal during a capture.
        CUDA_CHECK(
            cudaStreamCreateWithFlags(&capture_stream_, cudaStreamNonBlocking));
    }
}

GraphSlot& WDDMBypassEngine::require_slot(int graph_id) {
    auto iterator = slots_.find(graph_id);
    if (iterator == slots_.end()) {
        std::ostringstream message;
        message << "WDDM-Bypass: no graph registered for id " << graph_id;
        throw EngineError(message.str());
    }
    return iterator->second;
}

const GraphSlot& WDDMBypassEngine::require_slot(int graph_id) const {
    auto iterator = slots_.find(graph_id);
    if (iterator == slots_.end()) {
        std::ostringstream message;
        message << "WDDM-Bypass: no graph registered for id " << graph_id;
        throw EngineError(message.str());
    }
    return iterator->second;
}

void WDDMBypassEngine::describe_topology(GraphSlot& slot) const {
    slot.node_count = 0;
    slot.node_types.clear();
    slot.topology_digest.clear();

    if (!slot.graph) {
        return;
    }

    std::size_t count = 0;
    CUDA_CHECK(cudaGraphGetNodes(slot.graph.get(), nullptr, &count));

    std::vector<cudaGraphNode_t> nodes(count);
    if (count > 0) {
        CUDA_CHECK(cudaGraphGetNodes(slot.graph.get(), nodes.data(), &count));
        nodes.resize(count);
    }

    slot.node_count = count;
    slot.node_types.reserve(count);

    std::ostringstream digest;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        cudaGraphNodeType type = cudaGraphNodeTypeCount;
        CUDA_CHECK(cudaGraphNodeGetType(nodes[index], &type));
        const char* name = node_type_name(type);
        slot.node_types.emplace_back(name);
        digest << name << ';';
    }
    slot.topology_digest = digest.str();
}

// ---------------------------------------------------------------------------
// Capture lifecycle
// ---------------------------------------------------------------------------

void WDDMBypassEngine::activate_device() {
    // PyTorch (or any other library) may have switched the current device.
    int current = -1;
    CUDA_CHECK(cudaGetDevice(&current));
    if (current != device_index_) {
        CUDA_CHECK(cudaSetDevice(device_index_));
    }
}

void WDDMBypassEngine::begin_capture(int graph_id,
                                     const std::string& shape_key,
                                     bool hot_update) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (capture_broken_) {
        throw EngineError(
            "WDDM-Bypass: the capture stream is unusable after a failed "
            "end_capture(); call reset_all() before capturing again");
    }
    if (active_capture_id_ >= 0) {
        std::ostringstream message;
        message << "WDDM-Bypass: graph " << active_capture_id_
                << " is already capturing; CUDA captures cannot nest";
        throw EngineError(message.str());
    }

    activate_device();
    ensure_capture_stream();

    // Capture always goes to the staging slot, so a capture that throws at any
    // point leaves the previous executable untouched and replayable.
    GraphSlot& slot = slots_[graph_id];
    slot.graph_id = graph_id;
    if (!shape_key.empty()) {
        slot.shape_key = shape_key;
    }
    slot.staging.reset();
    slot.capturing_into_staging = hot_update;

    // The capture stream must be idle before capture begins.
    CUDA_CHECK(cudaStreamSynchronize(capture_stream_));
    CUDA_CHECK(cudaStreamBeginCapture(capture_stream_,
                                      cudaStreamCaptureModeThreadLocal));

    slot.capturing = true;
    active_capture_id_ = graph_id;
}

void WDDMBypassEngine::end_capture(int graph_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    GraphSlot& slot = require_slot(graph_id);
    if (!slot.capturing || active_capture_id_ != graph_id) {
        std::ostringstream message;
        message << "WDDM-Bypass: end_capture() called for graph " << graph_id
                << " but it is not capturing";
        throw EngineError(message.str());
    }

    cudaGraph_t captured = nullptr;
    const cudaError_t status = cudaStreamEndCapture(capture_stream_, &captured);

    // Clear capture bookkeeping before any throw so the engine stays coherent.
    slot.capturing = false;
    active_capture_id_ = -1;

    if (status != cudaSuccess) {
        capture_broken_ = true;
        detail::throw_cuda_error(status,
                                 "cudaStreamEndCapture(capture_stream_, ...)",
                                 __FILE__,
                                 __LINE__);
    }
    if (captured == nullptr) {
        throw EngineError(
            "WDDM-Bypass: the capture contained no GPU work, so no graph was "
            "produced; launch at least one kernel inside the capture region");
    }

    slot.staging.reset(captured);

    // A capture records work; it does not run it. An empty graph is therefore
    // almost always a launch that went to the default stream instead of
    // capture_stream_ptr(), and instantiating it would make every replay a
    // silent no-op. Report it while the cause is still obvious.
    std::size_t recorded_nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(slot.staging.get(), nullptr, &recorded_nodes));
    if (recorded_nodes == 0) {
        slot.staging.reset();
        throw EngineError(
            "WDDM-Bypass: the capture recorded no GPU work; launch the kernels "
            "on capture_stream_ptr() between begin_capture() and end_capture()");
    }

    ++slot.capture_count;
    ++total_captures_;

    if (!slot.has_exec()) {
        // First capture for this id: promote the staging graph and instantiate.
        slot.graph.reset(slot.staging.release());
        instantiate_slot(slot);
        return;
    }

    if (slot.capturing_into_staging) {
        hot_update_slot(slot);
        return;
    }

    // Explicit re-capture: the topology may be unrelated to the live
    // executable, so rebuild instead of trying an in-place update.
    slot.graph.reset(slot.staging.release());
    slot.exec.reset();
    instantiate_slot(slot);
    ++slot.rebuild_count;
    ++total_rebuilds_;
}

bool WDDMBypassEngine::is_capturing(int graph_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iterator = slots_.find(graph_id);
    if (iterator == slots_.end()) {
        return false;
    }
    return iterator->second.capturing;
}

// ---------------------------------------------------------------------------
// Instantiation and hot update
// ---------------------------------------------------------------------------

void WDDMBypassEngine::instantiate_slot(GraphSlot& slot) {
    if (!slot.graph) {
        throw EngineError("WDDM-Bypass: cannot instantiate an empty graph");
    }
    cudaGraphExec_t exec = nullptr;
    // CUDA 12 signature: (cudaGraphExec_t*, cudaGraph_t, unsigned long long).
    CUDA_CHECK(cudaGraphInstantiate(&exec, slot.graph.get(), 0ULL));
    slot.exec.reset(exec);
    describe_topology(slot);
}

void WDDMBypassEngine::hot_update_slot(GraphSlot& slot) {
    if (!slot.staging) {
        throw EngineError(
            "WDDM-Bypass: no staging graph to merge; capture with "
            "hot_update=True before calling update_graph()");
    }

    if (!slot.has_exec()) {
        slot.graph.reset(slot.staging.release());
        instantiate_slot(slot);
        ++slot.rebuild_count;
        ++total_rebuilds_;
        return;
    }

    cudaGraphExecUpdateResultInfo info{};
    const cudaError_t status =
        cudaGraphExecUpdate(slot.exec.get(), slot.staging.get(), &info);

    const bool merged =
        (status == cudaSuccess) && (info.result == cudaGraphExecUpdateSuccess);

    if (merged) {
        // The executable was updated in place: same handle, same driver-side
        // launch state, only the kernel parameters moved.
        slot.graph.reset(slot.staging.release());
        describe_topology(slot);
        ++slot.update_count;
        ++total_updates_;
        return;
    }

    // A failed update can leave the executable unusable - the runtime
    // documentation requires destroying it - so rebuild from the staging graph
    // instead of replaying a half-updated handle.
    (void)cudaGetLastError();
    slot.exec.reset();
    slot.graph.reset(slot.staging.release());
    instantiate_slot(slot);
    ++slot.rebuild_count;
    ++total_rebuilds_;
}

void WDDMBypassEngine::update_graph(int graph_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    GraphSlot& slot = require_slot(graph_id);
    if (slot.capturing) {
        throw EngineError(
            "WDDM-Bypass: update_graph() called while a capture is still in "
            "progress for this graph id");
    }
    hot_update_slot(slot);
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

void WDDMBypassEngine::replay(int graph_id, std::uintptr_t stream_token) {
    std::lock_guard<std::mutex> lock(mutex_);
    activate_device();

    GraphSlot& slot = require_slot(graph_id);
    if (!slot.has_exec()) {
        std::ostringstream message;
        message << "WDDM-Bypass: graph " << graph_id
                << " has no instantiated executable yet; finish the capture "
                   "first";
        throw EngineError(message.str());
    }

    const cudaStream_t stream = detail::stream_from_token(stream_token);
    CUDA_CHECK(cudaGraphLaunch(slot.exec.get(), stream));

    ++slot.replay_count;
    ++total_replays_;
}

void WDDMBypassEngine::synchronize() {
    std::lock_guard<std::mutex> lock(mutex_);
    activate_device();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Resource management
// ---------------------------------------------------------------------------

void WDDMBypassEngine::reset(int graph_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iterator = slots_.find(graph_id);
    if (iterator == slots_.end()) {
        return;
    }
    if (iterator->second.capturing) {
        std::ostringstream message;
        message << "WDDM-Bypass: cannot reset graph " << graph_id
                << " while it is capturing";
        throw EngineError(message.str());
    }
    // RAII: erasing the slot destroys the executable and both graphs.
    slots_.erase(iterator);
}

void WDDMBypassEngine::reset_all() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (active_capture_id_ >= 0 && capture_stream_ != nullptr) {
        // Best effort: leave the stream out of capture mode so it can either be
        // reused or safely destroyed.
        cudaGraph_t orphan = nullptr;
        (void)cudaStreamEndCapture(capture_stream_, &orphan);
        if (orphan != nullptr) {
            (void)cudaGraphDestroy(orphan);
        }
        active_capture_id_ = -1;
    }

    if (capture_broken_ && capture_stream_ != nullptr) {
        (void)cudaStreamDestroy(capture_stream_);
        capture_stream_ = nullptr;
    }
    capture_broken_ = false;

    slots_.clear();
    buffers_.clear();

    total_captures_ = 0;
    total_replays_ = 0;
    total_updates_ = 0;
    total_rebuilds_ = 0;
    total_allocations_ = 0;
    total_allocated_bytes_ = 0;

    (void)cudaGetLastError();
}

bool WDDMBypassEngine::has_graph(int graph_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return slots_.find(graph_id) != slots_.end();
}

std::vector<int> WDDMBypassEngine::list_graphs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> ids;
    ids.reserve(slots_.size());
    for (const auto& entry : slots_) {
        ids.push_back(entry.first);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

int WDDMBypassEngine::graph_id_for_shape(const std::string& shape_key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shape_key.empty()) {
        return -1;
    }
    for (const auto& entry : slots_) {
        if (entry.second.shape_key == shape_key) {
            return entry.first;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Static device buffers
// ---------------------------------------------------------------------------

std::uintptr_t WDDMBypassEngine::register_static_buffer(const std::string& name,
                                                        std::size_t bytes) {
    if (name.empty()) {
        throw EngineError("WDDM-Bypass: static buffer name must not be empty");
    }
    if (bytes == 0) {
        throw EngineError("WDDM-Bypass: static buffer size must be positive");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    activate_device();

    void* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, bytes));
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(raw);

    auto existing = buffers_.find(name);
    if (existing != buffers_.end()) {
        total_allocated_bytes_ -= existing->second.bytes;
    } else {
        ++total_allocations_;
    }

    // Replacing the entry frees the previous allocation through the deleter.
    buffers_[name] =
        StaticBuffer(name, bytes,
                     std::unique_ptr<void, DeviceMemoryDeleter>(raw));
    total_allocated_bytes_ += bytes;
    return address;
}

std::uintptr_t WDDMBypassEngine::static_buffer_ptr(
    const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iterator = buffers_.find(name);
    if (iterator == buffers_.end()) {
        std::string message("WDDM-Bypass: unknown static buffer '");
        message.append(name);
        message.append("'");
        throw EngineError(message);
    }
    return reinterpret_cast<std::uintptr_t>(iterator->second.ptr());
}

std::size_t WDDMBypassEngine::static_buffer_bytes(
    const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iterator = buffers_.find(name);
    if (iterator == buffers_.end()) {
        std::string message("WDDM-Bypass: unknown static buffer '");
        message.append(name);
        message.append("'");
        throw EngineError(message);
    }
    return iterator->second.bytes;
}

void WDDMBypassEngine::release_static_buffer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iterator = buffers_.find(name);
    if (iterator == buffers_.end()) {
        return;
    }
    total_allocated_bytes_ -= iterator->second.bytes;
    if (total_allocations_ > 0) {
        --total_allocations_;
    }
    buffers_.erase(iterator);
}

void WDDMBypassEngine::copy_to_static_input(const std::string& name,
                                            std::uintptr_t source,
                                            std::size_t bytes,
                                            std::uintptr_t stream_token) {
    std::lock_guard<std::mutex> lock(mutex_);
    activate_device();

    auto iterator = buffers_.find(name);
    if (iterator == buffers_.end()) {
        std::string message("WDDM-Bypass: unknown static buffer '");
        message.append(name);
        message.append("'");
        throw EngineError(message);
    }
    if (source == 0) {
        throw EngineError(
            "WDDM-Bypass: null source pointer in copy_to_static_input");
    }
    if (bytes > iterator->second.bytes) {
        std::ostringstream message;
        message << "WDDM-Bypass: copy of " << bytes
                << " bytes overflows static buffer '" << name << "' ("
                << iterator->second.bytes << " bytes)";
        throw EngineError(message.str());
    }
    if (bytes == 0) {
        return;
    }

    const cudaStream_t stream = detail::stream_from_token(stream_token);
    // cudaMemcpyDefault resolves host/device direction from the pointers under
    // unified addressing, which is always available on 64-bit Windows.
    CUDA_CHECK(cudaMemcpyAsync(iterator->second.ptr(),
                               reinterpret_cast<const void*>(source),
                               bytes,
                               cudaMemcpyDefault,
                               stream));
}

void WDDMBypassEngine::copy_from_static(const std::string& name,
                                        std::uintptr_t destination,
                                        std::size_t bytes,
                                        std::uintptr_t stream_token) {
    std::lock_guard<std::mutex> lock(mutex_);
    activate_device();

    auto iterator = buffers_.find(name);
    if (iterator == buffers_.end()) {
        std::string message("WDDM-Bypass: unknown static buffer '");
        message.append(name);
        message.append("'");
        throw EngineError(message);
    }
    if (destination == 0) {
        throw EngineError(
            "WDDM-Bypass: null destination pointer in copy_from_static");
    }
    if (bytes > iterator->second.bytes) {
        std::ostringstream message;
        message << "WDDM-Bypass: copy of " << bytes
                << " bytes overflows static buffer '" << name << "' ("
                << iterator->second.bytes << " bytes)";
        throw EngineError(message.str());
    }
    if (bytes == 0) {
        return;
    }

    const cudaStream_t stream = detail::stream_from_token(stream_token);
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(destination),
                               iterator->second.ptr(),
                               bytes,
                               cudaMemcpyDefault,
                               stream));
}

void WDDMBypassEngine::zero_static_buffer(const std::string& name,
                                          std::uintptr_t stream_token) {
    std::lock_guard<std::mutex> lock(mutex_);
    activate_device();

    auto iterator = buffers_.find(name);
    if (iterator == buffers_.end()) {
        std::string message("WDDM-Bypass: unknown static buffer '");
        message.append(name);
        message.append("'");
        throw EngineError(message);
    }

    const cudaStream_t stream = detail::stream_from_token(stream_token);
    CUDA_CHECK(cudaMemsetAsync(iterator->second.ptr(),
                               0,
                               iterator->second.bytes,
                               stream));
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

std::string WDDMBypassEngine::describe(int graph_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const GraphSlot& slot = require_slot(graph_id);

    std::ostringstream text;
    text << "{\"graph_id\":" << slot.graph_id;
    text << ",\"shape_key\":\"" << slot.shape_key << "\"";
    text << ",\"node_count\":" << slot.node_count;
    text << ",\"topology\":\"" << slot.topology_digest << "\"";
    text << ",\"has_exec\":" << (slot.has_exec() ? "true" : "false");
    text << ",\"capturing\":" << (slot.capturing ? "true" : "false");
    text << ",\"captures\":" << slot.capture_count;
    text << ",\"replays\":" << slot.replay_count;
    text << ",\"updates\":" << slot.update_count;
    text << ",\"rebuilds\":" << slot.rebuild_count;
    text << "}";
    return text.str();
}

std::uintptr_t WDDMBypassEngine::capture_stream_ptr() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reinterpret_cast<std::uintptr_t>(capture_stream_);
}

std::map<std::string, std::uint64_t> WDDMBypassEngine::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::map<std::string, std::uint64_t> values;
    values["graphs"] = static_cast<std::uint64_t>(slots_.size());
    values["static_buffers"] = static_cast<std::uint64_t>(buffers_.size());
    values["static_bytes"] = static_cast<std::uint64_t>(total_allocated_bytes_);
    values["captures"] = total_captures_;
    values["replays"] = total_replays_;
    values["updates"] = total_updates_;
    values["rebuilds"] = total_rebuilds_;
    values["allocations"] = total_allocations_;
    values["capturing"] = active_capture_id_ >= 0 ? 1u : 0u;
    values["device_index"] = static_cast<std::uint64_t>(device_index_);
    values["torch_interop"] = detail::built_with_torch() ? 1u : 0u;
    return values;
}

}  // namespace wddm_bypass
