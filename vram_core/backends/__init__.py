"""
Edge Deployment Backends
=======================

Backends for optimized inference on various hardware:
    - onnx_backend: ONNX Runtime inference (CPU/GPU)
    - tensorrt_backend: TensorRT optimized inference (NVIDIA GPU)
    - lite_backend: Lightweight inference for mobile/embedded
"""

try:
    from vram_core.backends.onnx_backend import ONNXBackend, ONNXConfig
except ImportError:
    ONNXBackend = None
    ONNXConfig = None

try:
    from vram_core.backends.tensorrt_backend import TensorRTBackend, TensorRTConfig
except ImportError:
    TensorRTBackend = None
    TensorRTConfig = None

try:
    from vram_core.backends.lite_backend import LiteBackend, LiteConfig
except ImportError:
    LiteBackend = None
    LiteConfig = None

__all__ = [
    "ONNXBackend", "ONNXConfig",
    "TensorRTBackend", "TensorRTConfig",
    "LiteBackend", "LiteConfig",
]