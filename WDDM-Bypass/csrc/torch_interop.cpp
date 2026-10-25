// WDDM-Bypass v1.0.0 - PyTorch interop helpers.
//
// These helpers reach torch through Python attributes instead of linking
// libtorch, so they work with whatever torch the interpreter can import and
// never pin the extension to one CUDA version. Building with
// WDDM_BYPASS_WITH_TORCH additionally links c10, which lets replay() follow the
// current PyTorch stream without a round trip through Python.

#include "graph_manager.h"

#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace py = pybind11;

namespace wddm_bypass {
namespace python {

namespace {

py::object import_torch() {
    PyObject* module = PyImport_ImportModule("torch");
    if (module == nullptr) {
        // Drop the ImportError: callers see EngineError instead, which keeps the
        // failure inside one exception hierarchy.
        PyErr_Clear();
        throw EngineError(
            "WDDM-Bypass: PyTorch is not importable; install torch to use the "
            "interop helpers");
    }
    return py::reinterpret_steal<py::object>(module);
}

py::object cuda_module() {
    py::object torch = import_torch();
    if (!py::hasattr(torch, "cuda")) {
        throw EngineError("WDDM-Bypass: this torch build has no torch.cuda");
    }
    py::object cuda = torch.attr("cuda");
    const bool available = py::cast<bool>(cuda.attr("is_available")());
    if (!available) {
        throw EngineError("WDDM-Bypass: torch.cuda is not available");
    }
    return cuda;
}

std::uintptr_t current_stream_token() {
    py::object cuda = cuda_module();
    py::object stream = cuda.attr("current_stream")();
    // torch stores the raw cudaStream_t as the `cuda_stream` integer.
    return py::cast<std::uintptr_t>(stream.attr("cuda_stream"));
}

int current_device_index() {
    py::object cuda = cuda_module();
    return py::cast<int>(cuda.attr("current_device")());
}

bool cuda_available() {
    py::object torch = import_torch();
    if (!py::hasattr(torch, "cuda")) {
        return false;
    }
    return py::cast<bool>(torch.attr("cuda").attr("is_available")());
}

void synchronize_device() {
    py::object cuda = cuda_module();
    cuda.attr("synchronize")();
}

std::uintptr_t tensor_data_ptr(const py::object& tensor) {
    if (!py::hasattr(tensor, "data_ptr")) {
        throw EngineError(
            "WDDM-Bypass: tensor_data_ptr expects a torch.Tensor");
    }
    return py::cast<std::uintptr_t>(tensor.attr("data_ptr")());
}

std::size_t tensor_nbytes(const py::object& tensor) {
    if (!py::hasattr(tensor, "numel") || !py::hasattr(tensor, "element_size")) {
        throw EngineError("WDDM-Bypass: tensor_nbytes expects a torch.Tensor");
    }
    const auto elements = py::cast<std::int64_t>(tensor.attr("numel")());
    const auto element_size = py::cast<std::int64_t>(tensor.attr("element_size")());
    if (elements <= 0 || element_size <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(elements) *
           static_cast<std::size_t>(element_size);
}

py::object external_stream(std::uintptr_t stream_token) {
    if (stream_token == 0) {
        throw EngineError(
            "WDDM-Bypass: external_stream needs a non-zero stream handle");
    }
    py::object cuda = cuda_module();
    return cuda.attr("ExternalStream")(stream_token);
}

}  // namespace

void bind_torch_interop(py::module_& module) {
    module.def("torch_cuda_available",
               &cuda_available,
               "True when torch is importable and CUDA is usable.");
    module.def("torch_stream_token",
               &current_stream_token,
               "Raw handle of the current PyTorch stream, for "
               "replay(stream_token=...).");
    module.def("torch_device_index",
               &current_device_index,
               "Index of the current torch CUDA device.");
    module.def("torch_synchronize",
               &synchronize_device,
               "Blocks until the current torch CUDA device is idle.");
    module.def("tensor_data_ptr",
               &tensor_data_ptr,
               py::arg("tensor"),
               "Device address of a torch.Tensor as an integer.");
    module.def("tensor_nbytes",
               &tensor_nbytes,
               py::arg("tensor"),
               "Byte size of a torch.Tensor.");
    module.def("external_stream",
               &external_stream,
               py::arg("stream_token"),
               "Wrap an integer stream handle as torch.cuda.ExternalStream.");
}

}  // namespace python
}  // namespace wddm_bypass
