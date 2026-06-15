"""
vram_core Web API Server
========================

FastAPI-based REST + WebSocket API for speech transcription.

Endpoints:
    POST /transcribe         - File upload transcription
    POST /transcribe/base64  - Base64 audio transcription
    WebSocket /stream        - Real-time streaming ASR
    GET  /health             - Health check

Usage:
    python -m vram_core.api_server --host 0.0.0.0 --port 8000

Dependencies:
    pip install fastapi uvicorn python-multipart
"""

import io
import os
import sys
import time
import json
import wave
import base64
import logging
import argparse
import tempfile
import threading
import queue
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# 鈹€鈹€ Lazy imports for optional dependencies 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

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

    from vram_core.whisper_bridge import WhisperBridge, WhisperBackend
    from vram_core.streaming_asr import StreamASR, StreamASRConfig
    from vram_core.config import config

    # Resolve backend
    whisper_backend = WhisperBackend.AUTO
    if backend:
        try:
            whisper_backend = WhisperBackend(backend)
        except ValueError:
            logger.warning(f"Invalid backend '{backend}', using AUTO")

    # Initialize whisper bridge (singleton for the app)
    whisper = WhisperBridge(
        backend=whisper_backend,
        whisper_model=whisper_model,
        language=language,
    )

    app = FastAPI(
        title="vram_core Transcription API",
        description="High-performance speech-to-text API powered by vram_core",
        version="1.0.0",
    )

    # 鈹€鈹€ Request/Response Models 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

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

    class HealthResponse(BaseModel):
        status: str
        version: str
        gpu: bool
        backend: str
        available_backends: list

    # 鈹€鈹€ Helper: decode audio bytes to numpy 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

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
        from vram_core.audio_utils import AudioProcessor

        # Write to temp file for processing
        suffix = os.path.splitext(filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            audio_data, sr = whisper.audio_preprocessor.load_and_convert(
                tmp_path, target_sample_rate=16000
            )
            return audio_data, sr
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # 鈹€鈹€ POST /transcribe 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

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

        # Read file bytes
        audio_bytes = await file.read()
        if not audio_bytes:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty audio file"},
            )

        # Decode audio
        try:
            audio_data, sr = _decode_audio_bytes(
                audio_bytes, file.filename or "audio.wav"
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

    # 鈹€鈹€ POST /transcribe/base64 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

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

    # 鈹€鈹€ WebSocket /stream 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

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
                        # Text message 锟?could be config or command
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
                                # Update language if provided
                                if "language" in cmd:
                                    asr.config.language = cmd["language"]
                                    whisper.language = cmd["language"]
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
            logger.error(f"WebSocket error: {e}", exc_info=True)
            try:
                await send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            if asr.is_running:
                asr.stop()
            logger.info("WebSocket session ended")

    # 鈹€鈹€ GET /health 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

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
            version="1.0.0",
            gpu=gpu_available,
            backend=whisper.backend.value,
            available_backends=[b.value for b in whisper.get_available_backends()],
        )

    # 鈹€鈹€ GET / 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @app.get("/")
    async def root():
        """API root 锟?redirects to docs."""
        return {
            "name": "vram_core Transcription API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "endpoints": {
                "POST /transcribe": "Upload audio file for transcription",
                "POST /transcribe/base64": "Send base64-encoded audio",
                "WebSocket /stream": "Real-time streaming transcription",
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

    logger.info(f"Starting vram_core API server...")
    logger.info(f"  Host: {args.host}:{args.port}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Language: {args.language or 'auto-detect'}")
    logger.info(f"  Backend: {args.backend or 'auto'}")
    logger.info(f"  Workers: {args.workers}")

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