// WDDM-Bypass v1.0.0 - CUDA graph manager public interface.
//
// The engine owns every CUDA object it creates (graphs, executable graphs and
// static device buffers) through RAII wrappers, so a failed capture or a thrown
// exception can never leak a handle. All CUDA Runtime calls on the capture and
// replay paths go through CUDA_CHECK, which reports the API name and source
// location and clears the sticky error flag before throwing.
//
// ASCII only: this header is compiled on Windows with /source-charset:utf-8.

#ifndef WDDM_BYPASS_GRAPH_MANAGER_H
#define WDDM_BYPASS_GRAPH_MANAGER_H

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace wddm_bypass {

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Base class for every error raised by the native engine.
class EngineError : public std::runtime_error {
public:
    explicit EngineError(const std::string& what) : std::runtime_error(what) {}
};

/// Raised when a CUDA Runtime entry point returns anything but cudaSuccess.
class CudaError : public EngineError {
public:
    CudaError(const std::string& what, cudaError_t status)
        : EngineError(what), status_(status) {}

    /// The raw cudaError_t value, mirrored to Python as `CudaError.status`.
    cudaError_t status() const noexcept { return status_; }

private:
    cudaError_t status_;
};

namespace detail {

/// Builds the message for a failed CUDA call and throws CudaError.
/// Declared in one place so CUDA_CHECK evaluates its argument exactly once.
void throw_cuda_error(cudaError_t status,
                      const char* expression,
                      const char* file,
                      int line);

}  // namespace detail

/// Wraps a CUDA Runtime call, throwing CudaError with call-site context.
#define CUDA_CHECK(call) \
    ::wddm_bypass::detail::throw_cuda_error((call), #call, __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// RAII handle wrappers
// ---------------------------------------------------------------------------

/// Owns a cudaGraph_t and destroys it exactly once.
class GraphHandle {
public:
    GraphHandle() noexcept = default;
    explicit GraphHandle(cudaGraph_t handle) noexcept : handle_(handle) {}
    ~GraphHandle() { reset(); }

    GraphHandle(GraphHandle&& other) noexcept : handle_(other.release()) {}
    GraphHandle& operator=(GraphHandle&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }

    GraphHandle(const GraphHandle&) = delete;
    GraphHandle& operator=(const GraphHandle&) = delete;

    cudaGraph_t get() const noexcept { return handle_; }
    explicit operator bool() const noexcept { return handle_ != nullptr; }

    /// Gives up ownership without destroying the handle.
    cudaGraph_t release() noexcept {
        cudaGraph_t handle = handle_;
        handle_ = nullptr;
        return handle;
    }

    /// Destroys the current handle (if any) and adopts `handle`.
    void reset(cudaGraph_t handle = nullptr) noexcept {
        if (handle_ != nullptr && handle_ != handle) {
            // Destructors must not throw; a shutdown-time failure is not
            // actionable because the context is already going away.
            (void)cudaGraphDestroy(handle_);
        }
        handle_ = handle;
    }

private:
    cudaGraph_t handle_ = nullptr;
};

/// Owns a cudaGraphExec_t and destroys it exactly once.
class GraphExecHandle {
public:
    GraphExecHandle() noexcept = default;
    explicit GraphExecHandle(cudaGraphExec_t handle) noexcept : handle_(handle) {}
    ~GraphExecHandle() { reset(); }

    GraphExecHandle(GraphExecHandle&& other) noexcept : handle_(other.release()) {}
    GraphExecHandle& operator=(GraphExecHandle&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }

    GraphExecHandle(const GraphExecHandle&) = delete;
    GraphExecHandle& operator=(const GraphExecHandle&) = delete;

    cudaGraphExec_t get() const noexcept { return handle_; }
    explicit operator bool() const noexcept { return handle_ != nullptr; }

    cudaGraphExec_t release() noexcept {
        cudaGraphExec_t handle = handle_;
        handle_ = nullptr;
        return handle;
    }

    void reset(cudaGraphExec_t handle = nullptr) noexcept {
        if (handle_ != nullptr && handle_ != handle) {
            (void)cudaGraphExecDestroy(handle_);
        }
        handle_ = handle;
    }

private:
    cudaGraphExec_t handle_ = nullptr;
};

/// cudaFree as a std::unique_ptr deleter, so device memory is released on
/// every path including exceptions during a capture.
struct DeviceMemoryDeleter {
    void operator()(void* pointer) const noexcept {
        if (pointer != nullptr) {
            (void)cudaFree(pointer);
        }
    }
};

/// A named cudaMalloc allocation owned by the engine; never resized, never
/// reallocated during a capture, which is what keeps graphs allocation-free.
struct StaticBuffer {
    std::string name;
    std::size_t bytes = 0;
    std::unique_ptr<void, DeviceMemoryDeleter> data;

    StaticBuffer() = default;
    StaticBuffer(std::string buffer_name, std::size_t byte_count,
                 std::unique_ptr<void, DeviceMemoryDeleter> allocation)
        : name(std::move(buffer_name)),
          bytes(byte_count),
          data(std::move(allocation)) {}

    StaticBuffer(StaticBuffer&&) noexcept = default;
    StaticBuffer& operator=(StaticBuffer&&) noexcept = default;
    StaticBuffer(const StaticBuffer&) = delete;
    StaticBuffer& operator=(const StaticBuffer&) = delete;

    void* ptr() const noexcept { return data.get(); }
};

// ---------------------------------------------------------------------------
// GraphSlot: one managed graph
// ---------------------------------------------------------------------------

/// A single captured graph together with its instantiated executable.
///
/// `graph` is the authoritative source graph the executable was built from.
/// `staging` holds a freshly captured graph that has not been merged yet; it is
/// what makes cudaGraphExecUpdate possible - the executable is updated in place
/// from `staging`, then `staging` is promoted to `graph`.
struct GraphSlot {
    int graph_id = -1;
    std::string shape_key;

    GraphHandle graph;
    GraphHandle staging;
    GraphExecHandle exec;

    bool capturing = false;
    bool capturing_into_staging = false;

    std::size_t node_count = 0;
    std::vector<std::string> node_types;
    std::string topology_digest;

    std::uint64_t capture_count = 0;
    std::uint64_t replay_count = 0;
    std::uint64_t update_count = 0;
    std::uint64_t rebuild_count = 0;

    bool has_exec() const noexcept { return static_cast<bool>(exec); }
};

namespace detail {

/// Maps a stream token coming from Python to a cudaStream_t.
///
/// 0 means "whatever stream is current for the caller": the current PyTorch
/// CUDA stream when built with WDDM_BYPASS_WITH_TORCH (via
/// c10::cuda::getCurrentCUDAStream().stream()), otherwise the legacy default
/// stream. Any non-zero token is used verbatim.
cudaStream_t stream_from_token(std::uintptr_t token) noexcept;

/// True when this build links PyTorch's c10 CUDA stream API.
bool built_with_torch() noexcept;

}  // namespace detail

namespace ops {

/// Elementwise kernels used to build and exercise graphs. Every launcher takes
/// raw device pointers as integers so the native layer never depends on a
/// tensor type, and a stream token (0 = current stream, see
/// detail::stream_from_token) so it can be captured by PyTorch or standalone.
void launch_scale(std::uintptr_t dst,
                  std::int64_t count,
                  float alpha,
                  std::uintptr_t stream_token);

void launch_add(std::uintptr_t dst,
                std::uintptr_t lhs,
                std::uintptr_t rhs,
                std::int64_t count,
                std::uintptr_t stream_token);

void launch_relu(std::uintptr_t dst,
                 std::int64_t count,
                 std::uintptr_t stream_token);

}  // namespace ops

/// Capability probes; all are cheap and cache nothing across CUDA errors.
bool cuda_graph_supported();
std::string device_name();
int runtime_version();

// ---------------------------------------------------------------------------
// WDDMBypassEngine
// ---------------------------------------------------------------------------

/// Owns a set of captured CUDA graphs, their executables and the static device
/// buffers they read from.
///
/// Runtime shape changes are handled by re-capturing into a staging graph and
/// merging it with cudaGraphExecUpdate, which keeps the executable handle and
/// therefore the driver-side launch state alive across shape changes.
class WDDMBypassEngine {
public:
    /// Pins the engine to `device_index`; -1 keeps the caller's current device.
    explicit WDDMBypassEngine(int device_index = -1);

    /// Destroys every graph, executable and buffer, then the capture stream.
    ~WDDMBypassEngine();

    WDDMBypassEngine(const WDDMBypassEngine&) = delete;
    WDDMBypassEngine& operator=(const WDDMBypassEngine&) = delete;

    // -- capture lifecycle --------------------------------------------------
    /// Starts capturing into the engine's internal non-blocking stream.
    /// With `hot_update` and an existing executable the capture goes to the
    /// staging graph and end_capture() merges it instead of rebuilding.
    void begin_capture(int graph_id,
                       const std::string& shape_key = std::string(),
                       bool hot_update = false);

    /// Ends the capture and instantiates or hot-updates the executable.
    void end_capture(int graph_id);

    bool is_capturing(int graph_id) const;

    // -- execution ----------------------------------------------------------
    /// Launches the graph's executable on the resolved stream.
    void replay(int graph_id, std::uintptr_t stream_token = 0);

    /// Merges the staging graph into the live executable with
    /// cudaGraphExecUpdate; falls back to destroy + instantiate when the
    /// topology changed too much for an in-place update.
    void update_graph(int graph_id);

    // -- resource management ------------------------------------------------
    void reset(int graph_id);
    void reset_all();

    bool has_graph(int graph_id) const;
    std::vector<int> list_graphs() const;
    int graph_id_for_shape(const std::string& shape_key) const;

    std::uintptr_t register_static_buffer(const std::string& name,
                                          std::size_t bytes);
    std::uintptr_t static_buffer_ptr(const std::string& name) const;
    std::size_t static_buffer_bytes(const std::string& name) const;
    void release_static_buffer(const std::string& name);

    void copy_to_static_input(const std::string& name,
                              std::uintptr_t source,
                              std::size_t bytes,
                              std::uintptr_t stream_token = 0);

    /// Async copy out of a static buffer into `destination` (host or device).
    /// Provided so a replay can be verified without a tensor library.
    void copy_from_static(const std::string& name,
                          std::uintptr_t destination,
                          std::size_t bytes,
                          std::uintptr_t stream_token = 0);

    void zero_static_buffer(const std::string& name,
                            std::uintptr_t stream_token = 0);

    // -- introspection ------------------------------------------------------
    std::string describe(int graph_id) const;
    std::uintptr_t capture_stream_ptr() const;
    std::map<std::string, std::uint64_t> stats() const;
    void synchronize();

private:
    GraphSlot& require_slot(int graph_id);
    const GraphSlot& require_slot(int graph_id) const;
    void ensure_capture_stream();
    void activate_device();
    void instantiate_slot(GraphSlot& slot);
    void hot_update_slot(GraphSlot& slot);
    void describe_topology(GraphSlot& slot) const;
    static const char* node_type_name(cudaGraphNodeType type) noexcept;

    mutable std::mutex mutex_;
    int device_index_ = 0;
    cudaStream_t capture_stream_ = nullptr;

    std::unordered_map<int, GraphSlot> slots_;
    std::unordered_map<std::string, StaticBuffer> buffers_;

    int active_capture_id_ = -1;
    bool capture_broken_ = false;

    std::uint64_t total_captures_ = 0;
    std::uint64_t total_replays_ = 0;
    std::uint64_t total_updates_ = 0;
    std::uint64_t total_rebuilds_ = 0;
    std::uint64_t total_allocations_ = 0;
    std::size_t total_allocated_bytes_ = 0;
};

}  // namespace wddm_bypass

#endif  // WDDM_BYPASS_GRAPH_MANAGER_H
