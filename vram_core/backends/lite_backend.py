"""
Lite Backend: Lightweight inference for mobile and embedded devices.
Supports model quantization, memory optimization, and minimal API surface.
"""

import logging
import time
import os
import json
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LiteConfig:
    """Lite backend configuration."""
    model_dir: str = ""
    quantize: bool = True
    max_memory_mb: int = 256
    num_threads: int = 2
    use_fp16: bool = False
    batch_size: int = 1
    cache_dir: str = ".lite_cache"


class LiteBackend:
    """
    Lightweight inference backend for mobile/embedded deployment.
    
    Designed for:
        - iOS/Android (via ONNX Runtime Mobile)
        - Raspberry Pi
        - Jetson Nano
        - ESP32 (wake word only)
    
    Features:
        - Minimal memory footprint
        - INT8 quantization
        - Model caching
        - Simple API
    
    Example:
        >>> backend = LiteBackend(LiteConfig(model_dir="./models"))
        >>> result = backend.infer(audio_data)
    """
    
    def __init__(self, config: Optional[LiteConfig] = None):
        self.config = config or LiteConfig()
        self._models = {}
        self._cache = {}
        
        os.makedirs(self.config.cache_dir, exist_ok=True)
    
    def load_model(self, model_name: str, model_path: str = "") -> bool:
        """Load a lightweight model."""
        if not model_path:
            model_path = os.path.join(self.config.model_dir, f"{model_name}.onnx")
        
        if not os.path.exists(model_path):
            logger.warning("Model not found: %s", model_path)
            return False
        
        try:
            import onnxruntime as ort
            
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.config.num_threads
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.enable_mem_pattern = True
            
            # Use lightweight providers
            providers = ["CPUExecutionProvider"]
            # QNN (Qualcomm) provider — always available as string,
            # but may fail at session creation if not installed
            providers.insert(0, "QNNExecutionProvider")
            
            session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            self._models[model_name] = session
            
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info("Lite model loaded: %s (%.1fMB)", model_name, size_mb)
            return True
            
        except ImportError:
            logger.warning("onnxruntime not available for lite backend")
            return False
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Failed to load lite model: %s", e)
            return False
    
    def infer(self, model_name: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference with a loaded model."""
        if model_name not in self._models:
            raise ValueError("Model '%s' not loaded" % model_name)

        # Check cache
        cache_key = self._cache_key(model_name, inputs)
        if cache_key in self._cache:
            logger.debug("Cache hit for %s", model_name)
            return self._cache[cache_key]
        
        session = self._models[model_name]
        
        start = time.time()
        output_names = [o.name for o in session.get_outputs()]
        outputs = session.run(output_names, inputs)
        latency_ms = (time.time() - start) * 1000
        
        result = dict(zip(output_names, outputs))
        
        # Cache small results
        if self._should_cache(result):
            self._cache[cache_key] = result
        
        logger.debug("Lite inference: %s in %.2fms", model_name, latency_ms)
        return result
    
    def _cache_key(self, model_name: str, inputs: Dict[str, np.ndarray]) -> str:
        """Generate cache key from inputs."""
        import hashlib
        h = hashlib.md5(model_name.encode())
        for name, arr in sorted(inputs.items()):
            h.update(arr.tobytes()[:100])  # First 100 bytes for speed
        return h.hexdigest()
    
    def _should_cache(self, result: Dict[str, np.ndarray]) -> bool:
        """Check if result is small enough to cache."""
        total_bytes = sum(arr.nbytes for arr in result.values())
        return total_bytes < 1024 * 1024  # < 1MB
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata."""
        if model_name not in self._models:
            return {}
        
        session = self._models[model_name]
        return {
            "name": model_name,
            "inputs": [{"name": i.name, "shape": i.shape, "type": str(i.type)} for i in session.get_inputs()],
            "outputs": [{"name": o.name, "shape": o.shape, "type": str(o.type)} for o in session.get_outputs()],
            "providers": session.get_providers(),
        }
    
    def benchmark(self, model_name: str, inputs: Dict[str, np.ndarray], iterations: int = 100) -> Dict[str, float]:
        """Benchmark model inference."""
        if model_name not in self._models:
            raise ValueError("Model '%s' not loaded" % model_name)

        latencies = []
        for _ in range(iterations):
            start = time.time()
            self.infer(model_name, inputs)
            latencies.append((time.time() - start) * 1000)
        
        latencies = np.array(latencies)
        return {
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "iterations": iterations,
        }
    
    @staticmethod
    def prepare_for_mobile(model_path: str, output_path: str = "") -> str:
        """Prepare model for mobile deployment (quantize + optimize)."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            raise ImportError("onnxruntime quantization required")
        
        if not output_path:
            output_path = model_path.replace(".onnx", "_mobile.onnx")
        
        quantize_dynamic(
            model_path,
            output_path,
            weight_type=QuantType.QUInt8,
            per_channel=True,
        )
        
        orig_mb = os.path.getsize(model_path) / (1024 * 1024)
        new_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info("Mobile model: %.1fMB -> %.1fMB (%.0f%%)", orig_mb, new_mb, new_mb / orig_mb * 100)
        
        return output_path
    
    @property
    def loaded_models(self) -> List[str]:
        return list(self._models.keys())
    
    @property
    def memory_usage_mb(self) -> float:
        """Estimate memory usage."""
        total = 0
        for session in self._models.values():
            for inp in session.get_inputs():
                shape = inp.shape
                total += np.prod([s if isinstance(s, int) else 1 for s in shape]) * 4
        return total / (1024 * 1024)