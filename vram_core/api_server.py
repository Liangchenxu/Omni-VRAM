"""
vram_core Web API Server
========================

FastAPI-based REST + WebSocket API for speech transcription.

Endpoints:
    POST /transcribe           - File upload transcription
    POST /transcribe/base64    - Base64 audio transcription
    POST /transcribe/async     - Async submit transcription task
    GET  /task/{task_id}       - Query async task status/result
    DELETE /task/{task_id}     - Cancel async task
    WebSocket /stream          - Real-time streaming ASR (16-bit PCM)
    WebSocket /ws/stream       - Real-time streaming ASR (Float32 PCM)
    WebSocket /ws/transcribe   - Enhanced real-time streaming ASR
    GET  /health               - Health check

Usage:
    python -m vram_core.api_server --host 0.0.0.0 --port 8000

Dependencies:
    pip install fastapi uvicorn python-multipart
"""

import os
import sys
import time
import json
import wave
import uuid
import base64
import logging
import argparse
import tempfile
import threading
import queue
import numpy as np
from typing import Optional, Dict, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".aac", ".wma"}
VALID_LANGUAGE_CODES = {
    "zh", "en", "ja", "ko", "fr", "de", "es", "ru", "pt", "it",
    "ar", "hi", "th", "vi", "nl", "pl", "sv", "tr", "uk", "cs",
    "ro", "hu", "el", "he", "id", "ms", "tl", "fi", "da", "nb",
    "auto", None,
}


# ─── Rate Limiter ───────────────────────────────────────────────────────────

class RateLimiter:
    """Simple in-memory sliding-window rate limiter per IP address."""

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> bool:
        """Check if a request from client_ip is allowed."""
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            if client_ip not in self._requests:
                self._requests[client_ip] = []

            # Prune old entries
            timestamps = self._requests[client_ip]
            self._requests[client_ip] = [t for t in timestamps if t > cutoff]

            if len(self._requests[client_ip]) >= self.max_requests:
                return False

            self._requests[client_ip].append(now)
            return True

    def get_retry_after(self, client_ip: str) -> float:
        """Get seconds until the client can make another request."""
        with self._lock:
            timestamps = self._requests.get(client_ip, [])
            if not timestamps:
                return 0.0
            oldest = min(timestamps)
            return max(0.0, self.window_seconds - (time.time() - oldest))

# ─── Lazy imports for optional dependencies ─────────────────────────────────

def _check_fastapi():
    """Check if FastAPI dependencies are installed."""
    missing = []
    for pkg in ["fastapi", "uvicorn", "multipart"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Missing packages for API server: {', '.join(missing)}\n"
            "Install with: pip install fastapi uvicorn python-multipart"
        )


# ─── Task Queue (in-memory) ────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AsyncTask:
    """Represents an async transcription task."""

    def __init__(self, task_id: str, audio_bytes: bytes, filename: str,
                 language: Optional[str] = None):
        self.task_id = task_id
        self.audio_bytes = audio_bytes
        self.filename = filename
        self.language = language
        self.status = TaskStatus.PENDING
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self._callback: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "task_id": self.task_id,
            "status": self.status.value,
            "created_at": self.created_at,
        }
        if self.started_at:
            d["started_at"] = self.started_at
        if self.completed_at:
            d["completed_at"] = self.completed_at
        if self.result:
            d["result"] = self.result
        if self.error:
            d["error"] = self.error
        return d


class AsyncTaskQueue:
    """In-memory task queue with worker thread pool."""

    def __init__(self, whisper_bridge, max_workers: int = 4, max_queue_size: int = 100):
        self._bridge = whisper_bridge
        self._tasks: Dict[str, AsyncTask] = {}
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._max_workers = max_workers
        self._active_count = 0
        self._shutdown = False
        self._max_queue_size = max_queue_size
        self._workers = []
        for i in range(max_workers):
            t = threading.Thread(target=self._worker, daemon=True, name=f"async-task-worker-{i}")
            t.start()
            self._workers.append(t)
        logger.info("AsyncTaskQueue started (max_workers=%s, max_queue_size=%s)", max_workers, max_queue_size)

    def submit(self, audio_bytes: bytes, filename: str,
               language: Optional[str] = None,
               callback: Optional[Callable] = None) -> str:
        """Submit an async transcription task. Returns task_id."""
        if self._queue.full():
            raise RuntimeError(f"Task queue is full ({self._max_queue_size}). Try again later.")

        task_id = str(uuid.uuid4())
        task = AsyncTask(task_id, audio_bytes, filename, language)
        task._callback = callback

        with self._lock:
            self._tasks[task_id] = task

        self._queue.put(task_id)
        logger.info("Task %s submitted (queue size: %s)", task_id, self._queue.qsize())
        return task_id

    def get_task(self, task_id: str) -> Optional[AsyncTask]:
        """Get task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task. Returns True if cancelled."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                task.completed_at = time.time()
                return True
        return False

    def get_all_tasks(self) -> list:
        """Get all tasks."""
        with self._lock:
            return [t.to_dict() for t in self._tasks.values()]

    def cleanup(self, max_age: float = 3600.0):
        """Remove completed/failed tasks older than max_age seconds."""
        cutoff = time.time() - max_age
        with self._lock:
            to_remove = [
                tid for tid, t in self._tasks.items()
                if t.completed_at and t.completed_at < cutoff
            ]
            for tid in to_remove:
                del self._tasks[tid]
        if to_remove:
            logger.info("Cleaned up %s old tasks", len(to_remove))

    def shutdown(self):
        """Shutdown the task queue."""
        self._shutdown = True
        # Signal all workers to stop
        for _ in self._workers:
            self._queue.put(None)
        for t in self._workers:
            if t.is_alive():
                t.join(timeout=5.0)

    def _worker(self):
        """Worker thread that processes tasks from the queue."""
        while not self._shutdown:
            try:
                task_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task_id is None:
                break

            with self._lock:
                task = self._tasks.get(task_id)
                if not task or task.status == TaskStatus.CANCELLED:
                    continue
                task.status = TaskStatus.PROCESSING
                task.started_at = time.time()
                self._active_count += 1

            try:
                # Decode audio using temp directory for safe cleanup
                suffix = os.path.splitext(task.filename)[1] or ".wav"
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = os.path.join(tmp_dir, f"audio{suffix}")
                    with open(tmp_path, "wb") as tmp:
                        tmp.write(task.audio_bytes)

                    audio_data, sr = self._bridge.audio_preprocessor.load_and_convert(
                        tmp_path, target_sample_rate=16000
                    )

                # Transcribe
                result = self._bridge.transcribe(
                    audio_data, sample_rate=sr, language=task.language
                )

                task.status = TaskStatus.COMPLETED
                task.result = {
                    "text": result.text,
                    "language": result.language,
                    "duration": result.audio_duration,
                    "confidence": result.confidence,
                    "segments": result.segments,
                    "backend": result.backend.value if result.backend else "unknown",
                }
                task.completed_at = time.time()
                logger.info(
                    "Task %s completed in %.2fs",
                    task_id, task.completed_at - task.started_at,
                )

                # Fire callback
                if task._callback:
                    try:
                        task._callback(task)
                    except Exception as cb_err:
                        logger.error("Task callback error: %s", cb_err)

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = time.time()
                logger.error("Task %s failed: %s", task_id, e)
            finally:
                with self._lock:
                    self._active_count -= 1


def create_app(
    whisper_model: str = "base",
    language: Optional[str] = None,
    backend: Optional[str] = None,
):
    """
    Create and configure the FastAPI application.

    Args:
        whisper_model: Whisper model size.
        language: Default language.
        backend: Whisper backend name.

    Returns:
        FastAPI application instance.
    """
    _check_fastapi()

    from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    from vram_core.whisper import WhisperBridge, WhisperBackend
    from vram_core.streaming_asr import StreamASR, StreamASRConfig
    from vram_core.config import config

    # Resolve backend
    whisper_backend = WhisperBackend.AUTO
    if backend:
        try:
            whisper_backend = WhisperBackend(backend)
        except ValueError:
            logger.warning("Invalid backend '%s', using AUTO", backend)

    # Initialize whisper bridge (singleton for the app)
    whisper = WhisperBridge(
        backend=whisper_backend,
        whisper_model=whisper_model,
        language=language,
    )

    # Initialize async task queue
    task_queue = AsyncTaskQueue(whisper_bridge=whisper)

    app = FastAPI(
        title="vram_core Transcription API",
        description="High-performance speech-to-text API powered by vram_core",
        version="2.2.1",
    )

    # ─── API Key Authentication ───────────────────────────────────────────────
    API_KEY = os.environ.get("API_KEY", "")

    @app.middleware("http")
    async def auth_middleware(request, call_next):
        """Require X-API-Key header if API_KEY env var is set."""
        # Skip auth for health check and docs
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc", "/"):
            return await call_next(request)

        # If API_KEY is not set, allow all requests (dev mode)
        if not API_KEY:
            return await call_next(request)

        # Check header
        provided_key = request.headers.get("X-API-Key", "")
        if provided_key != API_KEY:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized. Provide a valid X-API-Key header."},
            )

        return await call_next(request)

    # ─── Rate Limiting ───────────────────────────────────────────────────────
    rate_limiter = RateLimiter(max_requests=60, window_seconds=60.0)

    @app.middleware("http")
    async def rate_limit_middleware(request, call_next):
        """Rate limit: 60 requests per minute per IP."""
        # Skip rate limiting for health check and docs
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc", "/"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        if not rate_limiter.is_allowed(client_ip):
            retry_after = rate_limiter.get_retry_after(client_ip)
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded. Try again later.",
                    "retry_after_seconds": round(retry_after, 1),
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        return await call_next(request)

    # ─── Request/Response Models ─────────────────────────────────────────────

    class Base64Request(BaseModel):
        audio_base64: str
        language: Optional[str] = None

    class TranscribeResponse(BaseModel):
        text: str
        language: str
        duration: float
        confidence: float
        segments: list
        backend: str
        processing_time: float

    class AsyncSubmitResponse(BaseModel):
        task_id: str
        status: str
        message: str

    class TaskStatusResponse(BaseModel):
        task_id: str
        status: str
        created_at: float
        started_at: Optional[float] = None
        completed_at: Optional[float] = None
        result: Optional[dict] = None
        error: Optional[str] = None

    class HealthResponse(BaseModel):
        status: str
        version: str
        gpu: bool
        backend: str
        available_backends: list

    # ─── Helper: decode audio bytes to numpy ─────────────────────────────────

    def _decode_audio_bytes(
        audio_bytes: bytes,
        filename: str = "audio.wav",
    ) -> tuple:
        """
        Decode audio bytes to float32 numpy array.

        Args:
            audio_bytes: Raw audio file bytes.
            filename: Original filename (for format detection).

        Returns:
            Tuple of (audio_data, sample_rate).
        """
        # Write to temp file for processing using TemporaryDirectory for safe cleanup
        suffix = os.path.splitext(filename)[1] or ".wav"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, f"audio{suffix}")
            with open(tmp_path, "wb") as tmp:
                tmp.write(audio_bytes)

            audio_data, sr = whisper.audio_preprocessor.load_and_convert(
                tmp_path, target_sample_rate=16000
            )
            return audio_data, sr

    # ─── POST /transcribe ────────────────────────────────────────────────────

    @app.post("/transcribe", response_model=TranscribeResponse)
    async def transcribe_file(
        file: UploadFile = File(...),
        language: Optional[str] = Form(None),
    ):
        """
        Transcribe an uploaded audio file.

        Supports: WAV, MP3, FLAC, OGG, M4A
        """
        start = time.time()

        # Read file bytes with size limit
        audio_bytes = await file.read()
        if not audio_bytes:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty audio file"},
            )
        if len(audio_bytes) > MAX_UPLOAD_SIZE_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": f"File too large. Maximum size is {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB"},
            )

        # Validate file format
        filename = file.filename or "audio.wav"
        ext = os.path.splitext(filename)[1].lower()
        if ext and ext not in SUPPORTED_AUDIO_EXTENSIONS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported audio format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"},
            )

        # Validate language
        if language and language not in VALID_LANGUAGE_CODES:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported language code '{language}'"},
            )

        # Decode audio
        try:
            audio_data, sr = _decode_audio_bytes(
                audio_bytes, filename
            )
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to decode audio: {str(e)}"},
            )

        # Transcribe (pass language per-call to avoid shared state mutation)
        try:
            result = whisper.transcribe(audio_data, sample_rate=sr, language=language)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Transcription failed: {str(e)}"},
            )

        return TranscribeResponse(
            text=result.text,
            language=result.language,
            duration=result.audio_duration,
            confidence=result.confidence,
            segments=result.segments,
            backend=result.backend.value if result.backend else "unknown",
            processing_time=time.time() - start,
        )

    # ─── POST /transcribe/base64 ─────────────────────────────────────────────

    @app.post("/transcribe/base64", response_model=TranscribeResponse)
    async def transcribe_base64(request: Base64Request):
        """
        Transcribe audio from base64-encoded data.

        Input JSON:
            {
                "audio_base64": "<base64 encoded wav/mp3>",
                "language": "zh"  // optional
            }
        """
        start = time.time()

        # Decode base64
        try:
            audio_bytes = base64.b64decode(request.audio_base64)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid base64 data: {str(e)}"},
            )

        if not audio_bytes:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty audio data"},
            )
        if len(audio_bytes) > MAX_UPLOAD_SIZE_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": f"Audio data too large. Maximum size is {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB"},
            )

        # Validate language
        if request.language and request.language not in VALID_LANGUAGE_CODES:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported language code '{request.language}'"},
            )

        # Decode audio
        try:
            audio_data, sr = _decode_audio_bytes(audio_bytes, "audio.wav")
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to decode audio: {str(e)}"},
            )

        # Transcribe (pass language per-call to avoid shared state mutation)
        try:
            result = whisper.transcribe(audio_data, sample_rate=sr, language=request.language)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Transcription failed: {str(e)}"},
            )

        return TranscribeResponse(
            text=result.text,
            language=result.language,
            duration=result.audio_duration,
            confidence=result.confidence,
            segments=result.segments,
            backend=result.backend.value if result.backend else "unknown",
            processing_time=time.time() - start,
        )

    # ─── POST /transcribe/async ──────────────────────────────────────────────

    @app.post("/transcribe/async", response_model=AsyncSubmitResponse)
    async def transcribe_async(
        file: UploadFile = File(...),
        language: Optional[str] = Form(None),
    ):
        """
        Submit an async transcription task.

        Returns a task_id that can be used to query status via GET /task/{task_id}.
        """
        audio_bytes = await file.read()
        if not audio_bytes:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty audio file"},
            )
        if len(audio_bytes) > MAX_UPLOAD_SIZE_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": f"File too large. Maximum size is {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB"},
            )

        # Validate file format
        filename = file.filename or "audio.wav"
        ext = os.path.splitext(filename)[1].lower()
        if ext and ext not in SUPPORTED_AUDIO_EXTENSIONS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported audio format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"},
            )

        # Validate language
        if language and language not in VALID_LANGUAGE_CODES:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported language code '{language}'"},
            )

        task_id = task_queue.submit(
            audio_bytes=audio_bytes,
            filename=filename,
            language=language,
        )

        return AsyncSubmitResponse(
            task_id=task_id,
            status="pending",
            message=f"Task submitted. Query status at GET /task/{task_id}",
        )

    # ─── GET /task/{task_id} ─────────────────────────────────────────────────

    @app.get("/task/{task_id}", response_model=TaskStatusResponse)
    async def get_task_status(task_id: str):
        """Query async task status and result."""
        task = task_queue.get_task(task_id)
        if not task:
            return JSONResponse(
                status_code=404,
                content={"error": f"Task {task_id} not found"},
            )
        return TaskStatusResponse(**task.to_dict())

    # ─── DELETE /task/{task_id} ──────────────────────────────────────────────

    @app.delete("/task/{task_id}")
    async def cancel_task(task_id: str):
        """Cancel a pending async task."""
        success = task_queue.cancel_task(task_id)
        if success:
            return {"task_id": task_id, "status": "cancelled", "message": "Task cancelled"}
        task = task_queue.get_task(task_id)
        if not task:
            return JSONResponse(
                status_code=404,
                content={"error": f"Task {task_id} not found"},
            )
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Task {task_id} cannot be cancelled (status: {task.status.value})"
            },
        )

    # ─── WebSocket /stream ───────────────────────────────────────────────────

    @app.websocket("/stream")
    async def websocket_stream(websocket: WebSocket):
        """
        Real-time streaming transcription via WebSocket.

        Protocol:
            Client sends: Binary audio chunks (16-bit PCM, 16kHz, mono)
                          or JSON text messages with config

            Server sends: JSON messages with partial/final results
                {"type": "partial", "text": "..."}
                {"type": "final", "text": "...", "start": 0.0, "end": 2.5}
                {"type": "error", "message": "..."}
                {"type": "ready"}
        """
        await websocket.accept()

        # Create per-connection StreamASR
        asr_config = StreamASRConfig(
            language=whisper.language,
            whisper_model=whisper.whisper_model,
        )
        asr = StreamASR(
            config=asr_config,
            whisper_bridge=whisper,
        )

        async def send_json(data: dict):
            await websocket.send_text(json.dumps(data, ensure_ascii=False))

        # Thread-safe queues for cross-thread communication
        # ASR worker thread produces, async event loop consumes
        partial_results = queue.Queue()
        final_results = queue.Queue()

        # Set up callbacks (run in ASR worker thread)
        def on_partial(text):
            partial_results.put({"type": "partial", "text": text})

        def on_final(result):
            final_results.put({
                "type": "final",
                "text": result.text,
                "start": result.start_time,
                "end": result.end_time,
                "confidence": result.confidence,
                "language": result.language,
            })

        asr.on_partial_result = on_partial
        asr.on_final_result = on_final

        await send_json({"type": "ready"})
        logger.info("WebSocket client connected")

        try:
            asr.start()

            while True:
                message = await websocket.receive()

                if message.get("type") == "websocket.receive":
                    if "bytes" in message and message["bytes"]:
                        # Binary audio data (16-bit PCM, 16kHz, mono)
                        raw_bytes = message["bytes"]
                        audio_chunk = np.frombuffer(raw_bytes, dtype=np.int16)
                        audio_float = audio_chunk.astype(np.float32) / 32768.0
                        asr.feed(audio_float)

                        # Drain all pending results from thread-safe queues
                        while not partial_results.empty():
                            try:
                                await send_json(partial_results.get_nowait())
                            except queue.Empty:
                                break
                        while not final_results.empty():
                            try:
                                await send_json(final_results.get_nowait())
                            except queue.Empty:
                                break

                    elif "text" in message and message["text"]:
                        # Text message — could be config or command
                        try:
                            cmd = json.loads(message["text"])
                            if cmd.get("action") == "stop":
                                # Finalize and send remaining
                                final = asr.stop()
                                # Drain any remaining queued results after stop
                                while not partial_results.empty():
                                    try:
                                        await send_json(partial_results.get_nowait())
                                    except queue.Empty:
                                        break
                                while not final_results.empty():
                                    try:
                                        await send_json(final_results.get_nowait())
                                    except queue.Empty:
                                        break
                                if final:
                                    await send_json({
                                        "type": "final",
                                        "text": final.text,
                                        "start": final.start_time,
                                        "end": final.end_time,
                                        "confidence": final.confidence,
                                        "language": final.language,
                                    })
                                await send_json({"type": "stopped"})
                                break
                            elif cmd.get("action") == "config":
                                # Update language for this connection's ASR only
                                if "language" in cmd:
                                    asr.config.language = cmd["language"]
                                await send_json({"type": "config_updated"})
                        except json.JSONDecodeError:
                            await send_json({
                                "type": "error",
                                "message": "Invalid JSON command",
                            })

                elif message.get("type") == "websocket.disconnect":
                    break

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error("WebSocket error: %s", e, exc_info=True)
            try:
                await send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            if asr.is_running:
                asr.stop()
            logger.info("WebSocket session ended")

    # ─── WebSocket /ws/stream (browser real-time recording) ──────────────────

    @app.websocket("/ws/stream")
    async def websocket_ws_stream(websocket: WebSocket):
        """
        WebSocket endpoint for browser real-time recording.

        Accepts binary Float32 PCM audio (as sent by browser RecordingWorklet)
        and returns partial/final JSON results.

        Protocol:
            Client sends: Binary Float32 PCM chunks (16kHz, mono)
                          or JSON text messages with config
            Server sends: JSON messages
                {"type": "partial", "text": "..."}
                {"type": "final", "text": "...", "start": 0.0, "end": 2.5}
                {"type": "error", "message": "..."}
                {"type": "ready"}
        """
        await websocket.accept()

        asr_config = StreamASRConfig(
            language=whisper.language,
            whisper_model=whisper.whisper_model,
        )
        asr = StreamASR(
            config=asr_config,
            whisper_bridge=whisper,
        )

        async def send_json(data: dict):
            await websocket.send_text(json.dumps(data, ensure_ascii=False))

        partial_results = queue.Queue()
        final_results = queue.Queue()

        def on_partial(text):
            partial_results.put({"type": "partial", "text": text})

        def on_final(result):
            final_results.put({
                "type": "final",
                "text": result.text,
                "start": result.start_time,
                "end": result.end_time,
                "confidence": result.confidence,
                "language": result.language,
            })

        asr.on_partial_result = on_partial
        asr.on_final_result = on_final

        await send_json({"type": "ready"})
        logger.info("WebSocket /ws/stream client connected")

        try:
            asr.start()

            while True:
                message = await websocket.receive()

                if message.get("type") == "websocket.receive":
                    if "bytes" in message and message["bytes"]:
                        raw_bytes = message["bytes"]
                        audio_chunk = np.frombuffer(raw_bytes, dtype=np.float32)
                        asr.feed(audio_chunk)

                        while not partial_results.empty():
                            try:
                                await send_json(partial_results.get_nowait())
                            except queue.Empty:
                                break
                        while not final_results.empty():
                            try:
                                await send_json(final_results.get_nowait())
                            except queue.Empty:
                                break

                    elif "text" in message and message["text"]:
                        try:
                            cmd = json.loads(message["text"])
                            if cmd.get("action") == "stop":
                                final = asr.stop()
                                while not partial_results.empty():
                                    try:
                                        await send_json(partial_results.get_nowait())
                                    except queue.Empty:
                                        break
                                while not final_results.empty():
                                    try:
                                        await send_json(final_results.get_nowait())
                                    except queue.Empty:
                                        break
                                if final:
                                    await send_json({
                                        "type": "final",
                                        "text": final.text,
                                        "start": final.start_time,
                                        "end": final.end_time,
                                        "confidence": final.confidence,
                                        "language": final.language,
                                    })
                                await send_json({"type": "stopped"})
                                break
                            elif cmd.get("action") == "config":
                                if "language" in cmd:
                                    asr.config.language = cmd["language"]
                                await send_json({"type": "config_updated"})
                        except json.JSONDecodeError:
                            await send_json({
                                "type": "error",
                                "message": "Invalid JSON command",
                            })

                elif message.get("type") == "websocket.disconnect":
                    break

        except WebSocketDisconnect:
            logger.info("WebSocket /ws/stream client disconnected")
        except Exception as e:
            logger.error("WebSocket /ws/stream error: %s", e, exc_info=True)
            try:
                await send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            if asr.is_running:
                asr.stop()
            logger.info("WebSocket /ws/stream session ended")

    # ─── WebSocket /ws/transcribe (enhanced real-time) ───────────────────────

    @app.websocket("/ws/transcribe")
    async def websocket_ws_transcribe(websocket: WebSocket):
        """
        Enhanced WebSocket endpoint for real-time streaming transcription.

        Supports multiple audio formats and provides richer results.

        Protocol:
            Client sends:
                Binary audio chunks (16-bit PCM or Float32, default 16kHz mono)
                JSON text messages:
                    {"action": "start", "language": "zh", "sample_rate": 16000,
                     "encoding": "pcm_s16le" | "pcm_f32le"}
                    {"action": "stop"}
                    {"action": "config", "language": "en"}

            Server sends:
                {"type": "ready"}
                {"type": "started", "config": {...}}
                {"type": "partial", "text": "...", "timestamp": 1234567890.123}
                {"type": "final", "text": "...", "start": 0.0, "end": 2.5,
                 "confidence": 0.95, "language": "zh", "segments": [...]}
                {"type": "stopped"}
                {"type": "error", "message": "..."}
                {"type": "config_updated", "config": {...}}
        """
        await websocket.accept()

        # Connection state
        conn_language = whisper.language
        conn_sample_rate = 16000
        conn_encoding = "pcm_s16le"  # pcm_s16le or pcm_f32le
        asr = None

        async def send_json(data: dict):
            data["timestamp"] = time.time()
            await websocket.send_text(json.dumps(data, ensure_ascii=False))

        partial_results = queue.Queue()
        final_results = queue.Queue()

        def on_partial(text):
            partial_results.put({"type": "partial", "text": text})

        def on_final(result):
            final_results.put({
                "type": "final",
                "text": result.text,
                "start": result.start_time,
                "end": result.end_time,
                "confidence": result.confidence,
                "language": result.language,
                "segments": getattr(result, "segments", []),
            })

        await send_json({"type": "ready"})
        logger.info("WebSocket /ws/transcribe client connected")

        try:
            while True:
                message = await websocket.receive()

                if message.get("type") == "websocket.receive":
                    if "bytes" in message and message["bytes"]:
                        raw_bytes = message["bytes"]

                        # Must have started first
                        if asr is None:
                            await send_json({
                                "type": "error",
                                "message": "Send {\"action\": \"start\"} first",
                            })
                            continue

                        # Decode based on encoding
                        if conn_encoding == "pcm_f32le":
                            audio_chunk = np.frombuffer(raw_bytes, dtype=np.float32)
                        else:
                            audio_chunk = np.frombuffer(raw_bytes, dtype=np.int16)
                            audio_chunk = audio_chunk.astype(np.float32) / 32768.0

                        asr.feed(audio_chunk)

                        # Drain pending results
                        while not partial_results.empty():
                            try:
                                await send_json(partial_results.get_nowait())
                            except queue.Empty:
                                break
                        while not final_results.empty():
                            try:
                                await send_json(final_results.get_nowait())
                            except queue.Empty:
                                break

                    elif "text" in message and message["text"]:
                        try:
                            cmd = json.loads(message["text"])
                            action = cmd.get("action")

                            if action == "start":
                                # Start a new ASR session
                                conn_language = cmd.get("language", whisper.language)
                                conn_sample_rate = cmd.get("sample_rate", 16000)
                                conn_encoding = cmd.get("encoding", "pcm_s16le")

                                asr_config = StreamASRConfig(
                                    language=conn_language,
                                    whisper_model=whisper.whisper_model,
                                )
                                asr = StreamASR(
                                    config=asr_config,
                                    whisper_bridge=whisper,
                                )
                                asr.on_partial_result = on_partial
                                asr.on_final_result = on_final
                                asr.start()

                                await send_json({
                                    "type": "started",
                                    "config": {
                                        "language": conn_language,
                                        "sample_rate": conn_sample_rate,
                                        "encoding": conn_encoding,
                                    },
                                })
                                logger.info(
                                    "WS transcribe started: lang=%s, sr=%s, enc=%s",
                                    conn_language, conn_sample_rate, conn_encoding,
                                )

                            elif action == "stop":
                                if asr and asr.is_running:
                                    final = asr.stop()
                                    # Drain remaining
                                    while not partial_results.empty():
                                        try:
                                            await send_json(partial_results.get_nowait())
                                        except queue.Empty:
                                            break
                                    while not final_results.empty():
                                        try:
                                            await send_json(final_results.get_nowait())
                                        except queue.Empty:
                                            break
                                    if final:
                                        await send_json({
                                            "type": "final",
                                            "text": final.text,
                                            "start": final.start_time,
                                            "end": final.end_time,
                                            "confidence": final.confidence,
                                            "language": final.language,
                                        })
                                    asr = None
                                await send_json({"type": "stopped"})

                            elif action == "config":
                                new_lang = cmd.get("language")
                                if new_lang and asr:
                                    asr.config.language = new_lang
                                    conn_language = new_lang
                                await send_json({
                                    "type": "config_updated",
                                    "config": {
                                        "language": conn_language,
                                        "sample_rate": conn_sample_rate,
                                        "encoding": conn_encoding,
                                    },
                                })

                            else:
                                await send_json({
                                    "type": "error",
                                    "message": f"Unknown action: {action}",
                                })

                        except json.JSONDecodeError:
                            await send_json({
                                "type": "error",
                                "message": "Invalid JSON command",
                            })

                elif message.get("type") == "websocket.disconnect":
                    break

        except WebSocketDisconnect:
            logger.info("WebSocket /ws/transcribe client disconnected")
        except Exception as e:
            logger.error("WebSocket /ws/transcribe error: %s", e, exc_info=True)
            try:
                await send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            if asr and asr.is_running:
                asr.stop()
            logger.info("WebSocket /ws/transcribe session ended")

    # ─── GET /health ─────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """
        Health check endpoint.

        Returns server status, version, GPU availability, and
        active whisper backend.
        """
        # Check GPU availability
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            pass

        return HealthResponse(
            status="ok",
            version="2.2.1",
            gpu=gpu_available,
            backend=whisper.backend.value,
            available_backends=[b.value for b in whisper.get_available_backends()],
        )

    # ─── GET / ───────────────────────────────────────────────────────────────

    @app.get("/")
    async def root():
        """API root — redirects to docs."""
        return {
            "name": "vram_core Transcription API",
            "version": "2.2.1",
            "docs": "/docs",
            "health": "/health",
            "endpoints": {
                "POST /transcribe": "Upload audio file for transcription",
                "POST /transcribe/base64": "Send base64-encoded audio",
                "POST /transcribe/async": "Submit async transcription task",
                "GET /task/{task_id}": "Query async task status/result",
                "DELETE /task/{task_id}": "Cancel async task",
                "WebSocket /stream": "Real-time streaming (16-bit PCM)",
                "WebSocket /ws/stream": "Real-time streaming (Float32 PCM)",
                "WebSocket /ws/transcribe": "Enhanced real-time streaming",
                "GET /health": "Health check",
            },
        }

    return app


def main():
    """CLI entry point for the API server."""
    parser = argparse.ArgumentParser(
        description="vram_core Transcription API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m vram_core.api_server
    python -m vram_core.api_server --host 0.0.0.0 --port 8000
    python -m vram_core.api_server --model small --language zh
    python -m vram_core.api_server --backend faster_whisper --workers 4
        """,
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to bind (default: 8000)",
    )
    parser.add_argument(
        "--model", default="base",
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--language", default=None,
        help="Default language code (default: auto-detect)",
    )
    parser.add_argument(
        "--backend", default=None,
        choices=["auto", "faster_whisper", "whisper_cpp", "openai_api"],
        help="Whisper backend (default: auto)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of uvicorn workers (default: 1)",
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting vram_core API server...")
    logger.info("  Host: %s:%s", args.host, args.port)
    logger.info("  Model: %s", args.model)
    logger.info("  Language: %s", args.language or 'auto-detect')
    logger.info("  Backend: %s", args.backend or 'auto')
    logger.info("  Workers: %s", args.workers)

    # Create app
    app = create_app(
        whisper_model=args.model,
        language=args.language,
        backend=args.backend,
    )

    # Run server
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()