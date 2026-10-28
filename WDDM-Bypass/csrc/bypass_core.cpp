// WDDM-Bypass v1.0.0 - pybind11 bindings.
//
// One translation unit, no templates in the public surface: every call takes
// integers, strings or the engine object, so the module stays cheap to compile
// and does not force a libtorch dependency on callers that only need graphs.

#include "graph_manager.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <exception>
#include <string>

namespace py = pybind11;

namespace wddm_bypass {
namespace python {

/// Defined in torch_interop.cpp: PyTorch stream and tensor helpers.
void bind_torch_interop(py::module_& module);

}  // namespace python
}  // namespace wddm_bypass


namespace {

// Borrowed pointer to the Python CudaError class; the module keeps it alive for
// as long as this extension can raise. Only consulted while translating.
PyObject* g_cuda_error_type = nullptr;

void bind_module(py::module_& module) {
    using wddm_bypass::CudaError;
    using wddm_bypass::EngineError;
    using wddm_bypass::WDDMBypassEngine;

    module.doc() =
        "WDDM-Bypass v1.0.0 - pre-instantiated CUDA graph replay engine";

    auto& engine_error =
        py::register_exception<EngineError>(module, "EngineError");

    // CudaError is created by hand (rather than with register_exception) so the
    // handler below can attach the raw cudaError_t as `CudaError.status`.
    const std::string qualified =
        std::string(py::str(module.attr("__name__"))) + ".CudaError";
    PyObject* raw_type = PyErr_NewExceptionWithDoc(
        qualified.c_str(),
        "Raised when a CUDA Runtime call fails.",
        engine_error.ptr(),
        nullptr);
    if (raw_type == nullptr) {
        throw py::error_already_set();
    }
    py::object cuda_error = py::reinterpret_steal<py::object>(raw_type);
    module.attr("CudaError") = cuda_error;
    g_cuda_error_type = cuda_error.ptr();

    // Registered after the EngineError handler, so it is tried first.
    py::register_exception_translator([](std::exception_ptr pointer) {
        try {
            if (pointer) {
                std::rethrow_exception(pointer);
            }
        } catch (const CudaError& error) {
            if (g_cuda_error_type == nullptr) {
                return;
            }
            py::object type =
                py::reinterpret_borrow<py::object>(g_cuda_error_type);
            py::object instance = type(py::str(error.what()));
            instance.attr("status") = static_cast<int>(error.status());
            PyErr_SetObject(type.ptr(), instance.ptr());
        }
    });

    py::class_<WDDMBypassEngine>(module,
                                 "WDDMBypassEngine",
                                 "Owns captured CUDA graphs, their executable "
                                 "handles and the static device buffers they "
                                 "read from.")
        .def(py::init<int>(), py::arg("device_index") = -1)
        .def("begin_capture",
             &WDDMBypassEngine::begin_capture,
             py::arg("graph_id"),
             py::arg("shape_key") = std::string(),
             py::arg("hot_update") = false,
             py::arg("stream_ptr") = 0,
             "Start capturing GPU work. stream_ptr=0 captures on the engine's "
             "internal stream; a non-zero token is a caller-owned "
             "cudaStream_t (torch.cuda.Stream().cuda_stream) that is captured "
             "in Relaxed mode and never destroyed by the engine. With "
             "hot_update=True and an existing executable, end_capture() merges "
             "the result with cudaGraphExecUpdate instead of re-instantiating.")
        .def("end_capture",
             &WDDMBypassEngine::end_capture,
             py::arg("graph_id"),
             "End the capture and instantiate or hot-update the executable.")
        .def("is_capturing",
             &WDDMBypassEngine::is_capturing,
             py::arg("graph_id"))
        .def("replay",
             &WDDMBypassEngine::replay,
             py::arg("graph_id"),
             py::arg("stream_token") = 0,
             "Launch the graph's executable. stream_token=0 means the current "
             "stream (the current PyTorch stream when built with torch); a "
             "non-zero token is any cudaStream_t, including a caller-owned "
             "PyTorch stream.")
        .def("update_graph",
             &WDDMBypassEngine::update_graph,
             py::arg("graph_id"),
             "Merge the staging graph into the live executable.")
        .def("reset", &WDDMBypassEngine::reset, py::arg("graph_id"))
        .def("reset_all", &WDDMBypassEngine::reset_all)
        .def("has_graph", &WDDMBypassEngine::has_graph, py::arg("graph_id"))
        .def("list_graphs", &WDDMBypassEngine::list_graphs)
        .def("graph_id_for_shape",
             &WDDMBypassEngine::graph_id_for_shape,
             py::arg("shape_key"))
        .def("register_static_buffer",
             &WDDMBypassEngine::register_static_buffer,
             py::arg("name"),
             py::arg("bytes"),
             "Allocate (or replace) a named device buffer and return its "
             "address as an integer.")
        .def("static_buffer_ptr",
             &WDDMBypassEngine::static_buffer_ptr,
             py::arg("name"))
        .def("static_buffer_bytes",
             &WDDMBypassEngine::static_buffer_bytes,
             py::arg("name"))
        .def("release_static_buffer",
             &WDDMBypassEngine::release_static_buffer,
             py::arg("name"))
        .def("copy_to_static_input",
             &WDDMBypassEngine::copy_to_static_input,
             py::arg("name"),
             py::arg("source"),
             py::arg("bytes"),
             py::arg("stream_token") = 0,
             "Async copy into a static buffer; source is a host or device "
             "address.")
        .def("zero_static_buffer",
             &WDDMBypassEngine::zero_static_buffer,
             py::arg("name"),
             py::arg("stream_token") = 0)
        .def("launch_fused_affine_relu",
             &WDDMBypassEngine::launch_fused_affine_relu,
             py::arg("dst"),
             py::arg("a"),
             py::arg("b"),
             py::arg("scale"),
             py::arg("count"),
             py::arg("stream_token") = 0,
             "Launch dst = max(0, a * scale + b) over three registered static "
             "buffers, count floats each. Inside an open capture a token of 0 "
             "means the active capture stream, so the kernel becomes a graph "
             "node instead of running eagerly.")
        .def("launch_fused_rmsnorm",
             &WDDMBypassEngine::launch_fused_rmsnorm,
             py::arg("dst"),
             py::arg("x"),
             py::arg("weight"),
             py::arg("eps"),
             py::arg("count"),
             py::arg("stream_token") = 0,
             "Launch dst = x * rsqrt(mean(x^2) + eps) * weight over three "
             "registered static buffers. One block, warp-shuffle reduction, one "
             "submission; count is the element count of one normalised row.")
        .def("copy_from_static",
             &WDDMBypassEngine::copy_from_static,
             py::arg("name"),
             py::arg("destination"),
             py::arg("bytes"),
             py::arg("stream_token") = 0,
             "Async copy out of a static buffer; destination is a host or "
             "device address.")
        .def("describe",
             &WDDMBypassEngine::describe,
             py::arg("graph_id"),
             "One-line JSON description of a graph slot.")
        .def("capture_stream_ptr",
             &WDDMBypassEngine::capture_stream_ptr,
             "The internal capture stream, for torch.cuda.ExternalStream.")
        .def("stats", &WDDMBypassEngine::stats)
        .def("synchronize", &WDDMBypassEngine::synchronize)
        .def("__repr__", [](const WDDMBypassEngine& engine) {
            const auto measured = engine.stats();
            std::string text = "<WDDMBypassEngine graphs=";
            text += std::to_string(measured.at("graphs"));
            text += " replays=";
            text += std::to_string(measured.at("replays"));
            text += ">";
            return text;
        });

    module.def("launch_scale",
               &wddm_bypass::ops::launch_scale,
               py::arg("dst"),
               py::arg("count"),
               py::arg("alpha"),
               py::arg("stream_token") = 0);
    module.def("launch_add",
               &wddm_bypass::ops::launch_add,
               py::arg("dst"),
               py::arg("lhs"),
               py::arg("rhs"),
               py::arg("count"),
               py::arg("stream_token") = 0);
    module.def("launch_relu",
               &wddm_bypass::ops::launch_relu,
               py::arg("dst"),
               py::arg("count"),
               py::arg("stream_token") = 0);
    module.def("launch_fused_affine_relu",
               &wddm_bypass::ops::launch_fused_affine_relu,
               py::arg("dst"),
               py::arg("a"),
               py::arg("b"),
               py::arg("scale"),
               py::arg("count"),
               py::arg("stream_token") = 0);
    module.def("launch_fused_rmsnorm",
               &wddm_bypass::ops::launch_fused_rmsnorm,
               py::arg("dst"),
               py::arg("x"),
               py::arg("weight"),
               py::arg("eps"),
               py::arg("count"),
               py::arg("stream_token") = 0);

    module.def("cuda_graph_supported", &wddm_bypass::cuda_graph_supported);
    module.def("device_name", &wddm_bypass::device_name);
    module.def("runtime_version", &wddm_bypass::runtime_version);

    module.def("build_info", []() {
        py::dict info;
        info["version"] = "1.0.0";
        info["with_torch"] = wddm_bypass::detail::built_with_torch();
        info["cuda_runtime_version"] = wddm_bypass::runtime_version();
        info["compiled_cuda_version"] = CUDART_VERSION;
#ifdef NDEBUG
        info["build_type"] = "Release";
#else
        info["build_type"] = "Debug";
#endif
        return info;
    });

    // PyTorch helpers import torch lazily and raise EngineError when it is
    // missing, so the core module never depends on torch at import time.
    wddm_bypass::python::bind_torch_interop(module);

    module.attr("__version__") = "1.0.0";
}

}  // namespace

#if defined(TORCH_EXTENSION_NAME)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { bind_module(module); }
#else
PYBIND11_MODULE(_wddm_bypass, module) { bind_module(module); }
#endif
