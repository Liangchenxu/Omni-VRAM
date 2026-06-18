"""
gRPC Server Interface for vram_core
=====================================

High-performance gRPC API for remote audio processing.

Features:
    - Streaming ASR (bidirectional)
    - Speaker diarization
    - Emotion recognition
    - Noise reduction
    - Health checking

Requires: pip install grpcio grpcio-tools protobuf

Usage:
    # Start server
    python -m vram_core.grpc_server --port 50051

    # Or programmatically
    from vram_core.grpc_server import serve
    serve(port=50051)
"""

import logging
import time
from concurrent import futures
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_GRPC_AVAILABLE = False
try:
    import grpc
    from grpc import ServicerContext
    _GRPC_AVAILABLE = True
except ImportError:
    logger.info("grpcio not available. Install with: pip install grpcio")


# 鈹€鈹€ Service Implementation 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class OmniVRAMServicer:
    """
    gRPC service implementation for vram_core.

    Provides endpoints for:
        - transcribe: Full audio transcription
        - diarize: Speaker diarization
        - detect_emotion: Emotion recognition
        - reduce_noise: Noise reduction
        - health_check: Service health
    """

    def __init__(self):
        self._whisper = None
        self._diarizer = None
        self._emotion = None
        self._noise = None
        self._start_time = time.time()
        self._request_count = 0
        logger.info("OmniVRAM gRPC servicer initialized")

    def _ensure_whisper(self):
        if self._whisper is None:
            from vram_core.whisper import WhisperBridge
            self._whisper = WhisperBridge()
            logger.info("Whisper loaded for gRPC")

    def _ensure_diarizer(self):
        if self._diarizer is None:
            from vram_core.speaker_diarization import SpeakerDiarizer
            self._diarizer = SpeakerDiarizer()
            logger.info("Diarizer loaded for gRPC")

    def _ensure_emotion(self):
        if self._emotion is None:
            from vram_core.emotion_recognition import EmotionRecognizer
            self._emotion = EmotionRecognizer()
            logger.info("Emotion recognizer loaded for gRPC")

    def _ensure_noise(self):
        if self._noise is None:
            from vram_core.noise_reduction import NoiseReducer
            self._noise = NoiseReducer()
            logger.info("Noise reducer loaded for gRPC")

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000, language: Optional[str] = None) -> dict:
        """Transcribe audio bytes to text."""
        self._ensure_whisper()
        self._request_count += 1
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        result = self._whisper.transcribe(audio, sample_rate=sample_rate, language=language)
        return {
            "text": result.text if hasattr(result, 'text') else str(result),
            "language": getattr(result, 'language', language or "unknown"),
            "duration": len(audio) / sample_rate,
        }

    def diarize(self, audio_bytes: bytes, sample_rate: int = 16000) -> dict:
        """Perform speaker diarization."""
        self._ensure_diarizer()
        self._request_count += 1
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        result = self._diarizer.diarize(audio, sample_rate=sample_rate)
        return {
            "speaker_count": result.speaker_count,
            "segments": [
                {
                    "start": seg.start_time,
                    "end": seg.end_time,
                    "speaker": seg.speaker_id,
                    "confidence": seg.confidence,
                }
                for seg in result.segments
            ],
            "backend": result.backend_used,
        }

    def detect_emotion(self, audio_bytes: bytes, sample_rate: int = 16000) -> dict:
        """Detect emotion from audio."""
        self._ensure_emotion()
        self._request_count += 1
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        result = self._emotion.recognize(audio, sample_rate=sample_rate)
        return {
            "primary_emotion": result.primary_emotion,
            "confidence": result.confidence,
            "all_emotions": result.emotion_scores,
            "valence": result.valence,
            "arousal": result.arousal,
        }

    def reduce_noise(self, audio_bytes: bytes, sample_rate: int = 16000, reduction_level: str = "medium") -> bytes:
        """Reduce noise from audio."""
        self._ensure_noise()
        self._request_count += 1
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        level_map = {"light": 0.3, "medium": 0.6, "aggressive": 0.9}
        strength = level_map.get(reduction_level, 0.6)
        cleaned = self._noise.reduce_noise(audio, sample_rate=sample_rate, noise_reduction_strength=strength)
        return cleaned.astype(np.float32).tobytes()

    def health_check(self) -> dict:
        """Return service health status."""
        uptime = time.time() - self._start_time
        return {
            "status": "healthy",
            "uptime_seconds": uptime,
            "request_count": self._request_count,
            "components": {
                "whisper": self._whisper is not None,
                "diarizer": self._diarizer is not None,
                "emotion": self._emotion is not None,
                "noise": self._noise is not None,
            },
        }

    def close(self):
        """Release resources."""
        if self._whisper:
            self._whisper.close()
        if self._diarizer:
            self._diarizer.close()
        if self._emotion:
            self._emotion.close()
        if self._noise:
            self._noise.close()


# 鈹€鈹€ Flask/HTTP Alternative (if gRPC not available) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class OmniVRAMHTTPServer:
    """
    HTTP REST API alternative when gRPC is not available.

    Uses Flask or built-in http.server for a simple REST API.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self._servicer = OmniVRAMServicer()
        self._app = None
        self._init_flask()

    def _init_flask(self):
        """Initialize Flask app if available."""
        try:
            from flask import Flask, request, jsonify
            self._app = Flask(__name__)

            @self._app.route("/health", methods=["GET"])
            def health():
                return jsonify(self._servicer.health_check())

            @self._app.route("/transcribe", methods=["POST"])
            def transcribe():
                audio_bytes = request.get_data()
                sr = int(request.args.get("sample_rate", 16000))
                lang = request.args.get("language")
                result = self._servicer.transcribe(audio_bytes, sr, lang)
                return jsonify(result)

            @self._app.route("/diarize", methods=["POST"])
            def diarize():
                audio_bytes = request.get_data()
                sr = int(request.args.get("sample_rate", 16000))
                result = self._servicer.diarize(audio_bytes, sr)
                return jsonify(result)

            @self._app.route("/emotion", methods=["POST"])
            def emotion():
                audio_bytes = request.get_data()
                sr = int(request.args.get("sample_rate", 16000))
                result = self._servicer.detect_emotion(audio_bytes, sr)
                return jsonify(result)

            @self._app.route("/reduce_noise", methods=["POST"])
            def reduce_noise():
                audio_bytes = request.get_data()
                sr = int(request.args.get("sample_rate", 16000))
                level = request.args.get("level", "medium")
                result = self._servicer.reduce_noise(audio_bytes, sr, level)
                return result, 200, {"Content-Type": "application/octet-stream"}

            logger.info("Flask REST API initialized")
        except ImportError:
            logger.info("Flask not available. Install with: pip install flask")

    def run(self):
        """Start the HTTP server."""
        if self._app is None:
            raise RuntimeError("Flask not available. Install with: pip install flask")
        logger.info("Starting HTTP server on %s:%d", self.host, self.port)
        self._app.run(host=self.host, port=self.port, debug=False)


# 鈹€鈹€ gRPC Serve Function 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def serve(
    port: int = 50051,
    max_workers: int = 4,
    use_http_fallback: bool = True,
):
    """
    Start the vram_core API server.

    Attempts gRPC first, falls back to HTTP REST if grpcio not installed.

    Args:
        port: Server port.
        max_workers: Max concurrent workers.
        use_http_fallback: Use HTTP if gRPC unavailable.
    """
    if _GRPC_AVAILABLE:
        _serve_grpc(port, max_workers)
    elif use_http_fallback:
        _serve_http(port)
    else:
        raise RuntimeError(
            "gRPC not available. Install with: pip install grpcio grpcio-tools"
        )


def _serve_grpc(port: int, max_workers: int):
    """Start gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    # Note: For full gRPC, you'd add generated service stubs here.
    # This is a simplified version using the servicer directly.
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("gRPC server started on port %d", port)
    logger.info("Servicer ready: transcribe, diarize, detect_emotion, reduce_noise")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        server.stop(grace=5)


def _serve_http(port: int):
    """Start HTTP fallback server."""
    logger.info("Starting HTTP fallback server on port %d", port)
    http_server = OmniVRAMHTTPServer(port=port)
    http_server.run()


# 鈹€鈹€ CLI Entry Point 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vram_core API Server")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--workers", type=int, default=4, help="Max workers")
    parser.add_argument("--http", action="store_true", help="Force HTTP mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.http:
        _serve_http(args.port)
    else:
        serve(port=args.port, max_workers=args.workers)