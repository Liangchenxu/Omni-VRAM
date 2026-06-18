"""
Multi-GPU Manager for vram_core
================================

Automatic multi-GPU detection, load balancing, and parallel transcription
for systems with multiple NVIDIA GPUs.

Features:
    - Auto-detect all available GPUs
    - Monitor VRAM usage per GPU in real-time
    - Load-balanced task assignment (least-loaded GPU)
    - Parallel transcription across multiple GPUs
    - Thread-safe GPU allocation

Usage:
    from vram_core.multi_gpu import MultiGPUManager

    manager = MultiGPUManager()
    print(manager.gpu_count, "GPUs available")
    print(manager.get_status())

    # Parallel transcription
    results = manager.transcribe_parallel(audio_files)
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

logger = logging.getLogger(__name__)


# 鈹€鈹€ CUDA Detection 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_CUDA_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        _CUDA_AVAILABLE = True
except ImportError:
    pass

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except (ImportError, OSError, RuntimeError):
    _NVML_AVAILABLE = False


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    device_id: int
    name: str
    total_memory_mb: int
    free_memory_mb: int
    used_memory_mb: int
    utilization_pct: float = 0.0
    temperature_c: int = 0
    is_available: bool = True

    @property
    def free_memory_gb(self) -> float:
        return self.free_memory_mb / 1024.0

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_mb / 1024.0

    @property
    def usage_pct(self) -> float:
        if self.total_memory_mb == 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100.0


@dataclass
class TaskAssignment:
    """Result of a task assignment to a GPU."""
    device_id: int
    gpu_name: str
    free_memory_mb: int
    task_id: str = ""


class MultiGPUManager:
    """
    Multi-GPU manager with load balancing and parallel processing.

    Features:
        - Auto-detect all NVIDIA GPUs
        - Real-time VRAM monitoring per GPU
        - Load-balanced assignment (least-loaded strategy)
        - Parallel transcription with ThreadPoolExecutor
        - Thread-safe GPU allocation

    Args:
        min_free_memory_mb: Minimum free memory to consider GPU available.
        prefer_device: Preferred GPU device ID (-1 for auto).

    Usage:
        manager = MultiGPUManager()

        # Get GPU status
        for gpu in manager.get_all_gpu_info():
            print(f"GPU {gpu.device_id}: {gpu.name} ({gpu.free_memory_gb:.1f}GB free)")

        # Get best GPU for a task
        assignment = manager.get_best_gpu()
        print(f"Use GPU {assignment.device_id}")

        # Parallel processing
        results = manager.transcribe_parallel(audio_list, model_loader=my_loader)
    """

    def __init__(
        self,
        min_free_memory_mb: int = 1024,
        prefer_device: int = -1,
    ):
        self.min_free_memory_mb = min_free_memory_mb
        self.prefer_device = prefer_device
        self._lock = threading.Lock()
        self._allocated_tasks: Dict[int, List[str]] = {}

        # Detect GPUs
        self._gpu_count = self._detect_gpu_count()
        logger.info("MultiGPUManager: %d GPU(s) detected", self._gpu_count)

    @staticmethod
    def _detect_gpu_count() -> int:
        """Detect number of available GPUs."""
        if _TORCH_AVAILABLE and _CUDA_AVAILABLE:
            return torch.cuda.device_count()
        if _NVML_AVAILABLE:
            try:
                return pynvml.nvmlDeviceGetCount()
            except (pynvml.NVMLError, RuntimeError):
                pass
        return 0

    @property
    def gpu_count(self) -> int:
        """Number of detected GPUs."""
        return self._gpu_count

    @property
    def is_available(self) -> bool:
        """Whether multi-GPU is available."""
        return self._gpu_count > 0

    # 鈹€鈹€ GPU Info 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def get_gpu_info(self, device_id: int) -> GPUInfo:
        """Get information about a specific GPU."""
        if device_id >= self._gpu_count:
            raise ValueError(f"Invalid device_id {device_id}, only {self._gpu_count} GPUs")

        if _TORCH_AVAILABLE and _CUDA_AVAILABLE:
            return self._get_gpu_info_torch(device_id)
        if _NVML_AVAILABLE:
            return self._get_gpu_info_nvml(device_id)
        return GPUInfo(
            device_id=device_id, name="Unknown",
            total_memory_mb=0, free_memory_mb=0, used_memory_mb=0,
        )

    def _get_gpu_info_torch(self, device_id: int) -> GPUInfo:
        """Get GPU info using PyTorch."""
        try:
            props = torch.cuda.get_device_properties(device_id)
            mem_info = torch.cuda.mem_get_info(device_id)
            free_mem = mem_info[0] // (1024 * 1024)
            total_mem = mem_info[1] // (1024 * 1024)
            used_mem = total_mem - free_mem

            return GPUInfo(
                device_id=device_id,
                name=props.name,
                total_memory_mb=total_mem,
                free_memory_mb=free_mem,
                used_memory_mb=used_mem,
            )
        except (RuntimeError, OSError) as e:
            logger.warning("Failed to get GPU %d info via torch: %s", device_id, e)
            return GPUInfo(
                device_id=device_id, name="Error",
                total_memory_mb=0, free_memory_mb=0, used_memory_mb=0,
            )

    def _get_gpu_info_nvml(self, device_id: int) -> GPUInfo:
        """Get GPU info using NVML."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = mem.total // (1024 * 1024)
            used_mb = mem.used // (1024 * 1024)
            free_mb = mem.free // (1024 * 1024)

            util = 0.0
            try:
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util = float(util_info.gpu)
            except (pynvml.NVMLError, RuntimeError):
                pass

            temp = 0
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except (pynvml.NVMLError, RuntimeError):
                pass

            return GPUInfo(
                device_id=device_id, name=name,
                total_memory_mb=total_mb, free_memory_mb=free_mb,
                used_memory_mb=used_mb, utilization_pct=util,
                temperature_c=temp,
            )
        except (pynvml.NVMLError, RuntimeError, OSError) as e:
            logger.warning("Failed to get GPU %d info via NVML: %s", device_id, e)
            return GPUInfo(
                device_id=device_id, name="Error",
                total_memory_mb=0, free_memory_mb=0, used_memory_mb=0,
            )

    def get_all_gpu_info(self) -> List[GPUInfo]:
        """Get information about all GPUs."""
        return [self.get_gpu_info(i) for i in range(self._gpu_count)]

    def get_status(self) -> str:
        """Get a formatted status string for all GPUs."""
        if self._gpu_count == 0:
            return "No GPUs available"

        lines = [f"=== {self._gpu_count} GPU(s) ==="]
        for gpu in self.get_all_gpu_info():
            lines.append(
                f"  GPU {gpu.device_id}: {gpu.name} | "
                f"{gpu.free_memory_gb:.1f}/{gpu.total_memory_gb:.1f} GB free | "
                f"Util: {gpu.utilization_pct:.0f}% | "
                f"Temp: {gpu.temperature_c}掳C"
            )
        return "\n".join(lines)

    # 鈹€鈹€ Load Balancing 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def get_best_gpu(
        self,
        required_memory_mb: int = 0,
    ) -> Optional[TaskAssignment]:
        """
        Get the best GPU for a new task (least-loaded strategy).

        Args:
            required_memory_mb: Minimum memory required for the task.

        Returns:
            TaskAssignment or None if no suitable GPU found.
        """
        with self._lock:
            if self._gpu_count == 0:
                return None

            # If preferred device specified, try it first
            if 0 <= self.prefer_device < self._gpu_count:
                gpu = self.get_gpu_info(self.prefer_device)
                if gpu.free_memory_mb >= max(required_memory_mb, self.min_free_memory_mb):
                    return TaskAssignment(
                        device_id=gpu.device_id,
                        gpu_name=gpu.name,
                        free_memory_mb=gpu.free_memory_mb,
                    )

            # Find GPU with most free memory
            best_gpu = None
            best_free = -1

            for i in range(self._gpu_count):
                gpu = self.get_gpu_info(i)
                if gpu.free_memory_mb >= max(required_memory_mb, self.min_free_memory_mb):
                    if gpu.free_memory_mb > best_free:
                        best_free = gpu.free_memory_mb
                        best_gpu = gpu

            if best_gpu is None:
                logger.warning("No GPU with sufficient memory found")
                return None

            return TaskAssignment(
                device_id=best_gpu.device_id,
                gpu_name=best_gpu.name,
                free_memory_mb=best_gpu.free_memory_mb,
            )

    # 鈹€鈹€ Parallel Transcription 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def transcribe_parallel(
        self,
        audio_files: List[Tuple[str, np.ndarray]],
        transcribe_fn=None,
        max_workers: Optional[int] = None,
        sample_rate: int = 16000,
    ) -> List[Dict]:
        """
        Transcribe multiple audio files in parallel across GPUs.

        Args:
            audio_files: List of (filename, audio_array) tuples.
            transcribe_fn: Function(device_id, audio, sample_rate) -> result.
                If None, uses WhisperBridge as default.
            max_workers: Max parallel workers (default: number of GPUs).
            sample_rate: Audio sample rate.

        Returns:
            List of result dicts with filename, text, device_id, etc.
        """
        if not audio_files:
            return []

        if transcribe_fn is None:
            transcribe_fn = self._default_transcribe

        if max_workers is None:
            max_workers = max(1, self._gpu_count)

        results = []
        errors = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for filename, audio in audio_files:
                # Get best GPU for this task
                assignment = self.get_best_gpu(required_memory_mb=512)
                device_id = assignment.device_id if assignment else -1

                future = executor.submit(
                    self._run_transcription,
                    transcribe_fn, device_id, filename, audio, sample_rate,
                )
                future_map[future] = filename

            for future in as_completed(future_map):
                filename = future_map[future]
                try:
                    result = future.result()
                    results.append(result)
                except (RuntimeError, OSError, ValueError) as e:
                    logger.error("Transcription failed for %s: %s", filename, e)
                    errors.append({"filename": filename, "error": str(e)})

        if errors:
            logger.warning("%d transcription(s) failed", len(errors))

        return results

    @staticmethod
    def _run_transcription(
        transcribe_fn, device_id: int, filename: str,
        audio: np.ndarray, sample_rate: int,
    ) -> Dict:
        """Run a single transcription task with OOM fallback."""
        start_time = time.time()
        try:
            result = transcribe_fn(device_id, audio, sample_rate)
            elapsed = time.time() - start_time
            return {
                "filename": filename,
                "device_id": device_id,
                "result": result,
                "elapsed_seconds": elapsed,
                "status": "success",
            }
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU %d OOM for %s, clearing cache and retrying on CPU", device_id, filename)
            if _TORCH_AVAILABLE:
                torch.cuda.empty_cache()
            try:
                result = transcribe_fn(-1, audio, sample_rate)
                return {
                    "filename": filename,
                    "device_id": -1,
                    "result": result,
                    "elapsed_seconds": time.time() - start_time,
                    "status": "fallback_cpu",
                }
            except (RuntimeError, OSError) as cpu_err:
                return {
                    "filename": filename,
                    "device_id": device_id,
                    "error": f"OOM then CPU fallback failed: {cpu_err}",
                    "elapsed_seconds": time.time() - start_time,
                    "status": "error",
                }
        except (RuntimeError, OSError, ValueError) as e:
            return {
                "filename": filename,
                "device_id": device_id,
                "error": str(e),
                "elapsed_seconds": time.time() - start_time,
                "status": "error",
            }

    @staticmethod
    def _default_transcribe(device_id: int, audio: np.ndarray, sample_rate: int) -> str:
        """Default transcription using WhisperBridge with OOM fallback."""
        try:
            from vram_core.whisper import WhisperBridge
            bridge = WhisperBridge(device_id=device_id if device_id >= 0 else None)
            result = bridge.transcribe(audio, sample_rate=sample_rate)
            return result.text if hasattr(result, 'text') else str(result)
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU %d OOM in default transcription, falling back to CPU", device_id)
            if _TORCH_AVAILABLE:
                torch.cuda.empty_cache()
            bridge = WhisperBridge(device="cpu")
            result = bridge.transcribe(audio, sample_rate=sample_rate)
            return result.text if hasattr(result, 'text') else str(result)
        except (RuntimeError, OSError) as e:
            logger.error("Default transcription failed on GPU %d: %s", device_id, e)
            raise

    # 鈹€鈹€ Memory Management 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def clear_gpu_cache(self, device_id: Optional[int] = None):
        """Clear CUDA cache on specified or all GPUs."""
        if not _TORCH_AVAILABLE:
            return

        if device_id is not None:
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
                logger.info("Cleared GPU %d cache", device_id)
        else:
            for i in range(self._gpu_count):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            logger.info("Cleared all GPU caches")

    def get_total_free_memory(self) -> int:
        """Get total free memory across all GPUs in MB."""
        return sum(
            self.get_gpu_info(i).free_memory_mb
            for i in range(self._gpu_count)
        )