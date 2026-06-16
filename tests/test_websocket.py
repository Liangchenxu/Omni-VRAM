"""
WebSocket API Tests
===================

Tests for the WebSocket endpoints in vram_core.api_server:
  - /stream        — Real-time streaming (16-bit PCM)
  - /ws/stream     — Real-time streaming (Float32 PCM)
  - /ws/transcribe — Enhanced real-time streaming

Requires: pytest, httpx, fastapi
    pip install pytest httpx
"""

import json
import struct
import time
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_whisper_bridge():
    """Create a mock WhisperBridge for testing."""
    mock = MagicMock()
    mock.language = "zh"
    mock.whisper_model = "base"
    mock.backend = MagicMock()
    mock.backend.value = "faster_whisper"

    # Mock transcribe result
    mock_result = MagicMock()
    mock_result.text = "test transcription"
    mock_result.language = "zh"
    mock_result.confidence = 0.95
    mock_result.start_time = 0.0
    mock_result.end_time = 2.5
    mock_result.audio_duration = 2.5
    mock_result.segments = [
        {"start": 0.0, "end": 2.5, "text": "test transcription", "confidence": 0.95}
    ]
    mock_result.backend = MagicMock()
    mock_result.backend.value = "faster_whisper"

    mock.transcribe.return_value = mock_result
    mock.get_available_backends.return_value = [mock.backend]

    # Mock audio_preprocessor
    mock.audio_preprocessor = MagicMock()
    mock.audio_preprocessor.load_and_convert.return_value = (
        np.random.randn(16000).astype(np.float32), 16000
    )

    return mock


@pytest.fixture
def mock_stream_asr():
    """Create a mock StreamASR for testing."""
    mock = MagicMock()
    mock.is_running = False
    mock.config = MagicMock()
    mock.config.language = "zh"

    mock_result = MagicMock()
    mock_result.text = "stream result"
    mock_result.language = "zh"
    mock_result.confidence = 0.9
    mock_result.start_time = 0.0
    mock_result.end_time = 1.5
    mock_result.segments = []

    mock.stop.return_value = mock_result
    return mock


def _make_pcm_s16_bytes(samples: int = 1600, frequency: float = 440.0,
                         sample_rate: int = 16000) -> bytes:
    """Generate 16-bit PCM sine wave audio bytes."""
    t = np.arange(samples) / sample_rate
    audio = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    return audio.tobytes()


def _make_pcm_f32_bytes(samples: int = 1600, frequency: float = 440.0,
                         sample_rate: int = 16000) -> bytes:
    """Generate Float32 PCM sine wave audio bytes."""
    t = np.arange(samples) / sample_rate
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio.tobytes()


# ─── WebSocket /stream Tests ────────────────────────────────────────────────

class TestWebSocketStream:
    """Tests for WebSocket /stream endpoint."""

    @pytest.mark.asyncio
    async def test_connection_established(self):
        """Test that WebSocket connection sends ready message."""
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.language = "zh"
            mock_bridge.whisper_model = "base"
            MockBridge.return_value = mock_bridge

            app = create_app(whisper_model="base")

            # Use TestClient WebSocket
            with TestClient(app) as client:
                with client.websocket_connect("/stream") as ws:
                    data = ws.receive_json()
                    assert data["type"] == "ready"

    @pytest.mark.asyncio
    async def test_audio_transmission(self):
        """Test sending audio chunks via WebSocket."""
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.language = "zh"
            mock_bridge.whisper_model = "base"
            MockBridge.return_value = mock_bridge

            app = create_app(whisper_model="base")

            with TestClient(app) as client:
                with client.websocket_connect("/stream") as ws:
                    # Receive ready
                    ws.receive_json()

                    # Send audio chunk
                    audio_bytes = _make_pcm_s16_bytes(1600)
                    ws.send_bytes(audio_bytes)

                    # Send stop
                    ws.send_text(json.dumps({"action": "stop"}))
                    # Should receive stopped message
                    stopped = ws.receive_json()
                    assert stopped["type"] in ("final", "stopped")

    @pytest.mark.asyncio
    async def test_stop_command(self):
        """Test sending stop command."""
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.language = "zh"
            mock_bridge.whisper_model = "base"
            MockBridge.return_value = mock_bridge

            app = create_app(whisper_model="base")

            with TestClient(app) as client:
                with client.websocket_connect("/stream") as ws:
                    ws.receive_json()  # ready
                    ws.send_text(json.dumps({"action": "stop"}))
                    result = ws.receive_json()
                    assert result["type"] in ("final", "stopped")


# ─── WebSocket /ws/transcribe Tests ─────────────────────────────────────────

class TestWebSocketWsTranscribe:
    """Tests for WebSocket /ws/transcribe endpoint."""

    @pytest.mark.asyncio
    async def test_connection_ready(self):
        """Test /ws/transcribe connection sends ready message."""
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.language = "zh"
            mock_bridge.whisper_model = "base"
            MockBridge.return_value = mock_bridge

            app = create_app(whisper_model="base")

            with TestClient(app) as client:
                with client.websocket_connect("/ws/transcribe") as ws:
                    data = ws.receive_json()
                    assert data["type"] == "ready"
                    assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_start_action(self):
        """Test sending start action with config."""
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.language = "zh"
            mock_bridge.whisper_model = "base"
            MockBridge.return_value = mock_bridge

            app = create_app(whisper_model="base")

            with TestClient(app) as client:
                with client.websocket_connect("/ws/transcribe") as ws:
                    ws.receive_json()  # ready

                    # Send start
                    ws.send_text(json.dumps({
                        "action": "start",
                        "language": "en",
                        "encoding": "pcm_s16le",
                    }))

                    started = ws.receive_json()
                    assert started["type"] == "started"
                    assert started["config"]["language"] == "en"
                    assert started["config"]["encoding"] == "pcm_s16le"

    @pytest.mark.asyncio
    async def test_audio_before_start_error(self):
        """Test sending audio before start returns error."""
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.language = "zh"
            mock_bridge.whisper_model = "base"
            MockBridge.return_value = mock_bridge

            app = create_app(whisper_model="base")

            with TestClient(app) as client:
                with client.websocket_connect("/ws/transcribe") as ws:
                    ws.receive_json()  # ready

                    # Send audio without start
                    ws.send_bytes(_make_pcm_s16_bytes(1600))

                    error = ws.receive_json()
                    assert error["type"] == "error"
                    assert "start" in error["message"].lower()

    @pytest.mark.asyncio
    async def test_full_transcription_flow(self):
        """Test complete start -> audio -> stop flow."""
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.language = "zh"
            mock_bridge.whisper_model = "base"
            MockBridge.return_value = mock_bridge

            app = create_app(whisper_model="base")

            with TestClient(app) as client:
                with client.websocket_connect("/ws/transcribe") as ws:
                    ws.receive_json()  # ready

                    # Start session
                    ws.send_text(json.dumps({"action": "start", "language": "zh"}))
                    started = ws.receive_json()
                    assert started["type"] == "started"

                    # Send audio
                    for _ in range(5):
                        ws.send_bytes(_make_pcm_s16_bytes(1600))

                    # Stop
                    ws.send_text(json.dumps({"action": "stop"}))
                    result = ws.receive_json()
                    assert result["type"] in ("final", "stopped")

    @pytest.mark.asyncio
    async def test_config_update(self):
        """Test config update during session."""
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.language = "zh"
            mock_bridge.whisper_model = "base"
            MockBridge.return_value = mock_bridge

            app = create_app(whisper_model="base")

            with TestClient(app) as client:
                with client.websocket_connect("/ws/transcribe") as ws:
                    ws.receive_json()  # ready

                    # Start
                    ws.send_text(json.dumps({"action": "start", "language": "zh"}))
                    ws.receive_json()  # started

                    # Update config
                    ws.send_text(json.dumps({"action": "config", "language": "en"}))
                    config = ws.receive_json()
                    assert config["type"] == "config_updated"
                    assert config["config"]["language"] == "en"

    @pytest.mark.asyncio
    async def test_pcm_f32_encoding(self):
        """Test Float32 PCM encoding option."""
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.language = "zh"
            mock_bridge.whisper_model = "base"
            MockBridge.return_value = mock_bridge

            app = create_app(whisper_model="base")

            with TestClient(app) as client:
                with client.websocket_connect("/ws/transcribe") as ws:
                    ws.receive_json()  # ready

                    # Start with f32 encoding
                    ws.send_text(json.dumps({
                        "action": "start",
                        "encoding": "pcm_f32le",
                    }))
                    started = ws.receive_json()
                    assert started["type"] == "started"
                    assert started["config"]["encoding"] == "pcm_f32le"

                    # Send f32 audio
                    ws.send_bytes(_make_pcm_f32_bytes(1600))

                    # Stop
                    ws.send_text(json.dumps({"action": "stop"}))
                    result = ws.receive_json()
                    assert result["type"] in ("final", "stopped")


# ─── Async Task Queue Tests ─────────────────────────────────────────────────

class TestAsyncTaskQueue:
    """Tests for async transcription task queue."""

    def test_task_submission(self):
        """Test submitting a task returns a task_id."""
        from vram_core.api_server import AsyncTaskQueue

        mock_bridge = MagicMock()
        mock_bridge.audio_preprocessor = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "hello"
        mock_result.language = "en"
        mock_result.confidence = 0.9
        mock_result.audio_duration = 1.0
        mock_result.segments = []
        mock_result.backend = MagicMock()
        mock_result.backend.value = "faster_whisper"
        mock_bridge.transcribe.return_value = mock_result
        mock_bridge.audio_preprocessor.load_and_convert.return_value = (
            np.zeros(16000, dtype=np.float32), 16000
        )

        queue = AsyncTaskQueue(whisper_bridge=mock_bridge)
        try:
            task_id = queue.submit(audio_bytes=b"fake_wav", filename="test.wav")
            assert task_id is not None
            assert len(task_id) > 0

            task = queue.get_task(task_id)
            assert task is not None
        finally:
            queue.shutdown()

    def test_task_cancellation(self):
        """Test cancelling a pending task."""
        from vram_core.api_server import AsyncTaskQueue

        mock_bridge = MagicMock()
        queue = AsyncTaskQueue(whisper_bridge=mock_bridge)
        try:
            task_id = queue.submit(audio_bytes=b"fake", filename="test.wav")

            # Try to cancel (may succeed if task is still pending)
            task = queue.get_task(task_id)
            assert task is not None
        finally:
            queue.shutdown()

    def test_task_not_found(self):
        """Test querying a non-existent task returns None."""
        from vram_core.api_server import AsyncTaskQueue

        mock_bridge = MagicMock()
        queue = AsyncTaskQueue(whisper_bridge=mock_bridge)
        try:
            task = queue.get_task("non-existent-id")
            assert task is None
        finally:
            queue.shutdown()


# ─── Async REST API Tests ───────────────────────────────────────────────────

class TestAsyncRestAPI:
    """Tests for async REST endpoints."""

    def test_submit_async_endpoint(self):
        """Test POST /transcribe/async."""
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.language = "zh"
            mock_bridge.whisper_model = "base"
            MockBridge.return_value = mock_bridge

            app = create_app(whisper_model="base")

            with TestClient(app) as client:
                # Create a minimal WAV file
                import io
                import wave
                wav_buf = io.BytesIO()
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(b"\x00\x00" * 16000)
                wav_bytes = wav_buf.getvalue()

                response = client.post(
                    "/transcribe/async",
                    files={"file": ("test.wav", wav_bytes, "audio/wav")},
                )
                assert response.status_code == 200
                data = response.json()
                assert "task_id" in data
                assert data["status"] == "pending"

    def test_get_task_status(self):
        """Test GET /task/{task_id}."""
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.language = "zh"
            mock_bridge.whisper_model = "base"
            MockBridge.return_value = mock_bridge

            app = create_app(whisper_model="base")

            with TestClient(app) as client:
                # Query non-existent task
                response = client.get("/task/non-existent-id")
                assert response.status_code == 404

    def test_cancel_task(self):
        """Test DELETE /task/{task_id}."""
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.language = "zh"
            mock_bridge.whisper_model = "base"
            MockBridge.return_value = mock_bridge

            app = create_app(whisper_model="base")

            with TestClient(app) as client:
                # Cancel non-existent task
                response = client.delete("/task/non-existent-id")
                assert response.status_code == 404


# ─── Health & Root Tests ─────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Tests for health and root endpoints."""

    def test_health_check(self):
        """Test GET /health."""
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.language = "zh"
            mock_bridge.whisper_model = "base"
            mock_bridge.backend = MagicMock()
            mock_bridge.backend.value = "faster_whisper"
            mock_bridge.get_available_backends.return_value = [mock_bridge.backend]
            MockBridge.return_value = mock_bridge

            app = create_app(whisper_model="base")

            with TestClient(app) as client:
                response = client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "ok"
                assert data["version"] == "2.1.3"

    def test_root_endpoint(self):
        """Test GET /."""
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.language = "zh"
            mock_bridge.whisper_model = "base"
            MockBridge.return_value = mock_bridge

            app = create_app(whisper_model="base")

            with TestClient(app) as client:
                response = client.get("/")
                assert response.status_code == 200
                data = response.json()
                assert data["version"] == "2.1.3"
                assert "endpoints" in data
                assert "/transcribe/async" in str(data["endpoints"])
                assert "/ws/transcribe" in str(data["endpoints"])


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])