"""
TensorRT Backend: NVIDIA TensorRT optimized inference.
Supports FP16/INT8 quantization, CUDA Graph optimization.
"""

import logging
import time
import os
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    CUDA_PY_AVAILABLE = True
except ImportError:
    CUDA_PY_AVAILABLE = False


@dataclass
class TensorRTConfig:
    """TensorRT backend configuration."""
    engine_path: str = ""
    onnx_path: str = ""
    fp16: bool = True
    int8: bool = False
    max_batch_size: int = 1
    max_workspace_size: int = 1 << 30  # 1GB
    input_shapes: Dict[str, tuple] = None
    dynamic_axes: bool = False

    def __post_init__(self):
        if self.input_shapes is None:
            self.input_shapes = {}


class TensorRTBackend:
    """
    TensorRT inference backend for maximum GPU performance.
    
    Features:
        - FP16/INT8 quantization
        - CUDA Graph optimization
        - Dynamic batch support
        - ONNX -> TensorRT engine conversion
    
    Example:
        >>> backend = TensorRTBackend(TensorRTConfig(engine_path="model.trt"))
        >>> result = backend.infer({"audio": audio_array})
    """
    
    def __init__(self, config: Optional[TensorRTConfig] = None):
        if not TRT_AVAILABLE:
            raise ImportError("tensorrt required: pip install tensorrt")
        
        self.config = config or TensorRTConfig()
        self._engine = None
        self._context = None
        self._bindings = []
        self._stream = None
        
        if self.config.engine_path and os.path.exists(self.config.engine_path):
            self._load_engine()
    
    def _load_engine(self):
        """Load TensorRT engine from file."""
        logger.info(f"Loading TensorRT engine: {self.config.engine_path}")
        
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        
        with open(self.config.engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        
        self._context = self._engine.create_execution_context()
        
        if CUDA_PY_AVAILABLE:
            self._stream = cuda.Stream()
        
        logger.info("TensorRT engine loaded successfully")
    
    def build_engine(self, onnx_path: str, output_path: str) -> str:
        """Build TensorRT engine from ONNX model."""
        if not TRT_AVAILABLE:
            raise ImportError("tensorrt required")
        
        logger.info(f"Building TensorRT engine from: {onnx_path}")
        
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt_logger)
        
        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")
        
        # Build config
        config = builder.create_builder_config()
        config.max_workspace_size = self.config.max_workspace_size
        
        if self.config.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled")
        
        if self.config.int8 and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("INT8 mode enabled")
        
        # Build engine
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save
        with open(output_path, "wb") as f:
            f.write(engine)
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"TensorRT engine saved: {output_path} ({size_mb:.1f}MB)")
        
        # Reload
        self.config.engine_path = output_path
        self._load_engine()
        
        return output_path
    
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference with TensorRT."""
        if self._engine is None:
            raise RuntimeError("No engine loaded")
        
        if not CUDA_PY_AVAILABLE:
            raise ImportError("pycuda required for TensorRT inference")
        
        start = time.time()
        
        # Allocate device memory
        d_inputs = {}
        d_outputs = {}
        outputs = {}
        
        for name, arr in inputs.items():
            arr = np.ascontiguousarray(arr.astype(np.float32))
            d_input = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(d_input, arr)
            d_inputs[name] = d_input
        
        # Get output shapes and allocate
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = self._engine.get_tensor_shape(name)
                dtype = trt.nptype(self._engine.get_tensor_dtype(name))
                output = np.empty(shape, dtype=dtype)
                d_output = cuda.mem_alloc(output.nbytes)
                d_outputs[name] = (d_output, output)
        
        # Set tensor addresses
        for name, d_input in d_inputs.items():
            self._context.set_tensor_address(name, int(d_input))
        for name, (d_output, _) in d_outputs.items():
            self._context.set_tensor_address(name, int(d_output))
        
        # Execute
        self._context.execute_async_v3(stream_handle=self._stream.handle)
        self._stream.synchronize()
        
        # Copy outputs
        for name, (d_output, output) in d_outputs.items():
            cuda.memcpy_dtoh(output, d_output)
            outputs[name] = output
        
        latency_ms = (time.time() - start) * 1000
        logger.debug(f"TensorRT inference: {latency_ms:.2f}ms")
        
        return outputs
    
    @property
    def is_loaded(self) -> bool:
        return self._engine is not None
    
    def __del__(self):
        """Cleanup GPU resources."""
        self._engine = None
        self._context = None