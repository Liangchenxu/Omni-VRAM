"""
Tests for :mod:`vram_core.realtime_server` - the OpenAI Realtime API compatible
``/v1/realtime`` WebSocket gateway.

The suite is split in three layers, mirroring the module itself:

* pure helpers (PCM16/base64 codecs, resampling, session configuration),
* the :class:`RealtimeSession` state machine driven through wire frames
  (handshake, server VAD turn detection, push-to-talk commit, barge-in),
* the FastAPI ``/v1/realtime`` mount used by ``api_server.create_app``.

Events are asserted on the session's own replay log (``event_types`` /
``events``, populated synchronously by ``_emit``) while delivery to the socket
sink is asserted through an ``asyncio`` wait helper, so the tests never race
against the outbound pump task.
"""

import asyncio
import base64
import json
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vram_core.realtime_server import (
    REALTIME_MODEL,
    SUPPORTED_TURN_DETECTION,
    WIRE_AUDIO_FORMAT,
    RealtimeSession,
    RealtimeSessionConfig,
    base64_to_pcm16,
    float32_to_pcm16,
    pcm16_to_base64,
    pcm16_to_float32,
    resample_pcm16,
)

SAMPLE_RATE = 16000


# ─── Test helpers ───────────────────────────────────────────────────────────

def sine(seconds: float, freq: float = 220.0, amplitude: float = 0.5,
         rate: int = SAMPLE_RATE) -> np.ndarray:
    """Build a mono float32 sine tone (used as synthetic "speech")."""
    timeline = np.arange(int(seconds * rate), dtype=np.float32) / float(rate)
    return (amplitude * np.sin(2.0 * np.pi * freq * timeline)).astype(np.float32)


def silence(seconds: float, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Build a mono float32 silence buffer."""
    return np.zeros(int(seconds * rate), dtype=np.float32)


def append_frames(samples: np.ndarray, chunk_ms: int = 100) -> list:
    """Encode ``samples`` as ``input_audio_buffer.append`` wire frames."""
    step = max(1, int(SAMPLE_RATE * chunk_ms / 1000))
    frames = []
    for start in range(0, len(samples), step):
        payload = float32_to_pcm16(samples[start:start + step])
        frames.append(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": pcm16_to_base64(payload),
        }))
    return frames


def collector():
    """Return ``(sent, send)`` where ``sent`` records every delivered event."""
    sent = []

    async def send(event):
        sent.append(event)

    return sent, send


async def feed(session: RealtimeSession, samples: np.ndarray, chunk_ms: int = 100) -> None:
    """Push ``samples`` through the real ``input_audio_buffer.append`` path."""
    for frame in append_frames(samples, chunk_ms=chunk_ms):
        await session.handle_message(frame)


async def wait_for(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    """Poll ``predicate`` until it is true or ``timeout`` seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return bool(predicate())


def events_of(session: RealtimeSession, event_type: str) -> list:
    """All logged events of ``event_type``."""
    return [event for event in session.events if event["type"] == event_type]


def finished(session: RealtimeSession) -> bool:
    """True once the active response has been closed out."""
    return any(event["type"] == "response.done" for event in session.events)


def fast_config(**overrides) -> RealtimeSessionConfig:
    """Session config with tiny endpointing delays so tests stay fast."""
    config = RealtimeSessionConfig()
    config.input_sample_rate = SAMPLE_RATE
    config.processing_sample_rate = SAMPLE_RATE
    config.tts_sample_rate = SAMPLE_RATE
    config.output_sample_rate = SAMPLE_RATE
    config.chunk_duration_ms = 100
    config.min_speech_ms = 100
    config.silence_duration_ms = 200
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def sine_synth(chunks: int = 3, seconds: float = 0.05):
    """Async TTS stub yielding float32 chunks at the configured TTS rate."""
    async def synthesize(_text):
        for _ in range(chunks):
            await asyncio.sleep(0)
            yield sine(seconds)

    return synthesize


def gated_synth(gate: asyncio.Event, chunks: int = 2):
    """Async TTS stub that stalls on ``gate`` after the first chunk."""
    async def synthesize(_text):
        for index in range(chunks):
            if index:
                await gate.wait()
            yield sine(0.05)

    return synthesize


# ─── Protocol constants ─────────────────────────────────────────────────────

class TestProtocolConstants:
    """The wire contract advertised to OpenAI-compatible clients."""

    def test_model_and_audio_format(self):
        assert REALTIME_MODEL == "omni-vram-realtime"
        assert WIRE_AUDIO_FORMAT == "pcm16"

    def test_only_server_side_vad_is_supported(self):
        assert SUPPORTED_TURN_DETECTION == ("server_vad",)

    def test_session_config_is_exported_from_the_package(self):
        import vram_core

        assert vram_core.RealtimeSession is RealtimeSession
        assert vram_core.RealtimeSessionConfig is RealtimeSessionConfig
        assert vram_core.REALTIME_MODEL == REALTIME_MODEL


# ─── Audio codecs ───────────────────────────────────────────────────────────

class TestPcm16Codec:
    """PCM16 <-> float32 <-> base64 conversions used on the audio path."""

    def test_round_trip_preserves_samples(self):
        samples = sine(0.01)
        decoded = pcm16_to_float32(float32_to_pcm16(samples))
        assert decoded.shape == samples.shape
        assert np.allclose(decoded, samples, atol=2e-3)

    def test_odd_trailing_byte_is_dropped(self):
        assert pcm16_to_float32(b"\x01\x02\x03").size == 1

    def test_empty_inputs_are_empty(self):
        assert pcm16_to_float32(b"").size == 0
        assert float32_to_pcm16(np.zeros(0, dtype=np.float32)) == b""
        assert pcm16_to_base64(b"") == ""
        assert base64_to_pcm16("") == b""

    def test_hot_signal_is_clipped_not_wrapped(self):
        raw = float32_to_pcm16(np.array([2.0, -2.0], dtype=np.float32))
        assert np.frombuffer(raw, dtype="<i2").tolist() == [32767, -32767]

    def test_base64_round_trip(self):
        payload = float32_to_pcm16(sine(0.02))
        assert base64_to_pcm16(pcm16_to_base64(payload)) == payload

    def test_base64_payload_tolerates_real_world_encodings(self):
        payload = b"\x00\x01\xfa\xfb"
        encoded = base64.b64encode(payload).decode("ascii")
        assert base64_to_pcm16(encoded) == payload              # canonical
        assert base64_to_pcm16(encoded.rstrip("=")) == payload  # missing padding
        assert base64_to_pcm16(encoded.encode("ascii")) == payload  # bytes input
        assert base64_to_pcm16("  " + encoded + "\n") == payload    # whitespace
        urlsafe = base64.urlsafe_b64encode(payload).decode("ascii")
        assert base64_to_pcm16(urlsafe) == payload              # urlsafe alphabet

    def test_base64_missing_payload_is_rejected(self):
        with pytest.raises(ValueError):
            base64_to_pcm16(None)


# ─── Resampling ─────────────────────────────────────────────────────────────

class TestResampling:
    """``resample_pcm16`` keeps the torch-free AudioProcessor path."""

    def test_identity_when_rates_match(self):
        payload = float32_to_pcm16(sine(0.2))
        assert resample_pcm16(payload, SAMPLE_RATE, SAMPLE_RATE) == payload

    def test_empty_payload_stays_empty(self):
        assert resample_pcm16(b"", SAMPLE_RATE, 24000) == b""

    def test_upsampling_roughly_doubles_the_sample_count(self):
        payload = float32_to_pcm16(sine(0.25))
        resampled = pcm16_to_float32(resample_pcm16(payload, 16000, 32000))
        expected = pcm16_to_float32(payload).size * 2
        assert abs(resampled.size - expected) <= 2

    def test_downsampling_roughly_halves_the_sample_count(self):
        payload = float32_to_pcm16(sine(0.25, rate=32000))
        resampled = pcm16_to_float32(resample_pcm16(payload, 32000, 16000))
        expected = pcm16_to_float32(payload).size // 2
        assert abs(resampled.size - expected) <= 2

    def test_resampled_signal_keeps_its_energy(self):
        payload = float32_to_pcm16(sine(0.3))
        resampled = pcm16_to_float32(resample_pcm16(payload, 16000, 24000))
        assert resampled.size > 0
        assert np.max(np.abs(resampled)) > 0.05


# ─── Session configuration ──────────────────────────────────────────────────

class TestSessionConfig:
    """Configuration defaults and OpenAI-shaped patch application."""

    def test_defaults_describe_the_wire_contract(self):
        view = RealtimeSessionConfig().as_dict()
        assert view["object"] == "realtime.session"
        assert view["model"] == REALTIME_MODEL
        assert view["input_audio_format"] == WIRE_AUDIO_FORMAT
        assert view["output_audio_format"] == WIRE_AUDIO_FORMAT
        assert view["input_audio_sample_rate"] == 24000
        assert view["output_audio_sample_rate"] == 24000
        assert view["turn_detection"]["type"] == "server_vad"
        assert view["turn_detection"]["silence_duration_ms"] == 800

    def test_flat_patch_reports_only_real_changes(self):
        config = RealtimeSessionConfig()
        changed = config.apply({
            "voice": "verse",
            "input_audio_sample_rate": 16000,
            "vad_threshold": 0.05,
        })
        assert set(changed) == {"voice", "input_sample_rate", "vad_threshold"}
        assert config.voice == "verse"
        assert config.input_sample_rate == 16000
        assert config.vad_threshold == 0.05

    def test_nested_openai_patch_is_understood(self):
        config = RealtimeSessionConfig()
        changed = config.apply({
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.04,
                "silence_duration_ms": 500,
            },
            "audio": {
                "input": {"format": {"type": "pcm16", "rate": 16000}},
                "output": {"format": {"rate": 22050}},
            },
        })
        assert {"vad_threshold", "silence_duration_ms",
                "input_sample_rate", "output_sample_rate"} <= set(changed)
        assert config.vad_threshold == 0.04
        assert config.silence_duration_ms == 500
        assert config.input_sample_rate == 16000
        assert config.output_sample_rate == 22050

    def test_unknown_and_null_fields_are_ignored(self):
        config = RealtimeSessionConfig()
        assert config.apply({"voice": None, "not_a_field": 1}) == []
        assert config.voice == RealtimeSessionConfig().voice

    def test_non_mapping_patch_is_ignored(self):
        config = RealtimeSessionConfig()
        assert config.apply("nonsense") == []


# ─── Handshake / session.update ─────────────────────────────────────────────

class TestHandshake:
    """``session.created`` handshake and ``session.update`` patches."""

    @pytest.mark.asyncio
    async def test_start_emits_session_created_and_returns_the_view(self):
        sent, send = collector()
        session = RealtimeSession(send=send, session_id="sess_unit", config=fast_config())

        view = await session.start()

        assert view["id"] == "sess_unit"
        assert view["model"] == REALTIME_MODEL
        created = events_of(session, "session.created")
        assert len(created) == 1
        assert created[0]["session"]["id"] == "sess_unit"
        assert created[0]["event_id"].startswith("evt_")
        assert await wait_for(lambda: [e["type"] for e in sent] == ["session.created"])
        assert session.event_types() == ["session.created"]

        await session.close()

    @pytest.mark.asyncio
    async def test_session_update_applies_and_acknowledges(self):
        sent, send = collector()
        session = RealtimeSession(send=send, config=fast_config())
        await session.start()

        await session.handle_message(json.dumps({
            "type": "session.update",
            "session": {"voice": "shimmer",
                        "turn_detection": {"silence_duration_ms": 350}},
        }))

        assert await wait_for(lambda: len(sent) == 2)
        updated = events_of(session, "session.updated")
        assert len(updated) == 1
        assert updated[0]["session"]["voice"] == "shimmer"
        assert updated[0]["session"]["turn_detection"]["silence_duration_ms"] == 350
        assert session.config.voice == "shimmer"
        assert session.config.silence_duration_ms == 350

        await session.close()


# ─── Conversation turns ─────────────────────────────────────────────────────

class TestServerVadTurn:
    """End-to-end turn driven purely by ``input_audio_buffer.append``."""

    @pytest.mark.asyncio
    async def test_vad_turn_produces_a_complete_response(self):
        sent, send = collector()
        session = RealtimeSession(
            send=send,
            config=fast_config(),
            transcriber=lambda _audio: "hello world",
            synthesize=sine_synth(chunks=3),
        )
        await session.start()

        await feed(session, sine(0.5))
        assert await wait_for(
            lambda: "input_audio_buffer.speech_started" in session.event_types()
        ), "server VAD never reported speech"

        await feed(session, silence(0.6))
        assert await wait_for(lambda: finished(session)), "the turn never completed"

        types = session.event_types()
        assert types[0] == "session.created"
        assert types.index("input_audio_buffer.speech_started") < types.index(
            "input_audio_buffer.speech_stopped")
        assert types.index("input_audio_buffer.speech_stopped") < types.index(
            "response.created")
        assert types.count("response.created") == 1
        assert types.count("response.audio.delta") == 3
        assert "response.audio_transcript.delta" in types
        assert "response.audio_transcript.done" in types

        done = events_of(session, "response.done")[-1]
        assert done["response"]["status"] == "completed"
        transcript = events_of(session, "response.audio_transcript.done")[-1]
        assert transcript["transcript"] == "hello world"

        delta = events_of(session, "response.audio.delta")[0]
        assert delta["item_id"] == transcript["item_id"]
        # 0.05 s of PCM16 at 16 kHz == 800 samples == 1600 bytes, no resampling
        assert len(base64_to_pcm16(delta["delta"])) == 1600

        assert session.pending_events() == 0
        await session.close()

    @pytest.mark.asyncio
    async def test_tts_bytes_are_forwarded_as_pcm16(self):
        sent, send = collector()

        async def bytes_synth(_text):
            yield float32_to_pcm16(sine(0.05))

        session = RealtimeSession(
            send=send,
            config=fast_config(),
            transcriber=lambda _audio: "bytes",
            synthesize=bytes_synth,
        )
        await session.start()
        await feed(session, sine(0.3))
        await session.commit_audio()
        assert await wait_for(lambda: finished(session))

        delta = events_of(session, "response.audio.delta")[-1]
        assert len(base64_to_pcm16(delta["delta"])) == 1600
        await session.close()


class TestPushToTalk:
    """``input_audio_buffer.commit`` bypasses server VAD entirely."""

    @pytest.mark.asyncio
    async def test_commit_answers_without_any_silence(self):
        sent, send = collector()
        session = RealtimeSession(
            send=send,
            config=fast_config(),
            transcriber=lambda _audio: "push to talk",
            synthesize=sine_synth(chunks=1),
        )
        await session.start()

        await feed(session, sine(0.3))
        await session.handle_message(json.dumps({"type": "input_audio_buffer.commit"}))
        assert await wait_for(lambda: finished(session))

        types = session.event_types()
        assert "input_audio_buffer.committed" in types
        assert types.count("response.created") == 1
        assert events_of(session, "response.done")[-1]["response"]["status"] == "completed"
        await session.close()

    @pytest.mark.asyncio
    async def test_clear_drops_buffered_audio(self):
        sent, send = collector()
        session = RealtimeSession(
            send=send,
            config=fast_config(),
            transcriber=lambda _audio: "should not happen",
            synthesize=sine_synth(chunks=1),
        )
        await session.start()

        await feed(session, sine(0.3))
        await session.handle_message(json.dumps({"type": "input_audio_buffer.clear"}))
        await session.handle_message(json.dumps({"type": "input_audio_buffer.commit"}))
        await asyncio.sleep(0.05)

        assert events_of(session, "response.created") == []
        assert "input_audio_buffer.committed" in session.event_types()
        assert events_of(session, "error") == []
        await session.close()


# ─── Interruption / single response in flight ───────────────────────────────

class TestInterruption:
    """Barge-in truncation and the one-response-at-a-time rule."""

    @staticmethod
    async def streaming_session(send, gate):
        """Start a push-to-talk turn whose TTS stalls on ``gate``."""
        session = RealtimeSession(
            send=send,
            config=fast_config(),
            transcriber=lambda _audio: "an answer",
            synthesize=gated_synth(gate),
        )
        await session.start()
        await feed(session, sine(0.3))
        await session.handle_message(json.dumps({"type": "input_audio_buffer.commit"}))
        assert await wait_for(
            lambda: "response.audio.delta" in session.event_types()
        ), "the response never started streaming"
        return session

    @pytest.mark.asyncio
    async def test_cancel_truncates_and_closes_the_response(self):
        sent, send = collector()
        gate = asyncio.Event()
        session = await self.streaming_session(send, gate)
        before = len(events_of(session, "response.audio.delta"))

        flushed = session.cancel_response(reason="barge_in")

        assert flushed >= 0
        assert await wait_for(lambda: session.pending_events() == 0), \
            "the truncation frames never reached the socket"
        assert [event["type"] for event in sent][-2:] == [
            "response.cancelled", "response.done"]
        assert session.event_types()[-2:] == ["response.cancelled", "response.done"]
        assert events_of(session, "response.cancelled")[-1]["reason"] == "barge_in"
        done = events_of(session, "response.done")[-1]
        assert done["response"]["status"] == "cancelled"
        assert done["response"]["reason"] == "barge_in"

        gate.set()
        await asyncio.sleep(0.15)
        assert len(events_of(session, "response.audio.delta")) == before
        assert all(
            event["response"]["status"] != "completed"
            for event in events_of(session, "response.done")
        )
        await session.close()

    @pytest.mark.asyncio
    async def test_vad_barge_in_callback_truncates_the_turn(self):
        sent, send = collector()
        gate = asyncio.Event()
        session = await self.streaming_session(send, gate)

        session.processor.on_interrupt()  # what StreamProcessor calls on barge-in

        assert any(e["type"] == "response.cancelled" for e in session.events)
        assert events_of(session, "response.done")[-1]["response"]["status"] == "cancelled"
        gate.set()
        await session.close()

    @pytest.mark.asyncio
    async def test_response_cancel_frame_is_honoured(self):
        sent, send = collector()
        gate = asyncio.Event()
        session = await self.streaming_session(send, gate)

        await session.handle_message(json.dumps({"type": "response.cancel"}))

        cancelled = events_of(session, "response.cancelled")[-1]
        assert cancelled["reason"] == "client_request"
        gate.set()
        await session.close()

    @pytest.mark.asyncio
    async def test_cancel_is_idempotent_when_nothing_is_running(self):
        sent, send = collector()
        session = RealtimeSession(send=send, config=fast_config())
        await session.start()
        # Let the pump deliver the handshake before probing the idle cancel.
        assert await wait_for(lambda: session.pending_events() == 0)

        assert session.cancel_response() == 0
        await session.handle_message(json.dumps({"type": "response.cancel"}))

        assert "response.cancelled" not in session.event_types()
        assert events_of(session, "response.done") == []
        await session.close()

    @pytest.mark.asyncio
    async def test_a_second_turn_is_dropped_while_answering(self):
        sent, send = collector()
        gate = asyncio.Event()
        session = await self.streaming_session(send, gate)

        await feed(session, sine(0.3))
        await session.handle_message(json.dumps({"type": "input_audio_buffer.commit"}))
        await asyncio.sleep(0.1)

        types = session.event_types()
        assert types.count("input_audio_buffer.committed") == 2
        assert types.count("response.created") == 1

        gate.set()
        assert await wait_for(lambda: finished(session))
        await session.close()


# ─── Error handling ─────────────────────────────────────────────────────────

class TestErrorHandling:
    """Malformed frames become ``error`` events, never crashed sessions."""

    @pytest.mark.asyncio
    async def test_unparsable_json_is_reported(self):
        sent, send = collector()
        session = RealtimeSession(send=send, config=fast_config())
        await session.start()

        await session.handle_message("this is not json")

        error = events_of(session, "error")[-1]
        assert error["error"]["type"] == "invalid_request_error"
        assert "JSON" in error["error"]["message"]
        await session.close()

    @pytest.mark.asyncio
    async def test_non_object_json_is_reported(self):
        sent, send = collector()
        session = RealtimeSession(send=send, config=fast_config())
        await session.start()

        await session.handle_message("[1, 2, 3]")

        assert "JSON object" in events_of(session, "error")[-1]["error"]["message"]
        await session.close()

    @pytest.mark.asyncio
    async def test_unknown_event_type_is_reported(self):
        sent, send = collector()
        session = RealtimeSession(send=send, config=fast_config())
        await session.start()

        await session.handle_message(json.dumps({"type": "conversation.item.create"}))
        error = events_of(session, "error")[-1]
        assert "Unsupported event type" in error["error"]["message"]

        await session.handle_message(json.dumps({"payload": "no type"}))
        assert "<missing>" in events_of(session, "error")[-1]["error"]["message"]
        await session.close()

    @pytest.mark.asyncio
    async def test_invalid_base64_audio_starts_no_turn(self):
        sent, send = collector()
        session = RealtimeSession(
            send=send,
            config=fast_config(),
            transcriber=lambda _audio: "never",
            synthesize=sine_synth(chunks=1),
        )
        await session.start()

        await session.handle_message(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": "abcde",
        }))

        error = events_of(session, "error")[-1]
        assert error["error"]["type"] == "invalid_request_error"
        assert "base64" in error["error"]["message"]
        assert events_of(session, "response.created") == []
        await session.close()

    @pytest.mark.asyncio
    async def test_failing_transcriber_reports_a_server_error(self):
        sent, send = collector()

        def boom(_audio):
            raise RuntimeError("asr exploded")

        session = RealtimeSession(
            send=send,
            config=fast_config(),
            transcriber=boom,
            synthesize=sine_synth(chunks=1),
        )
        await session.start()
        await feed(session, sine(0.3))
        await session.handle_message(json.dumps({"type": "input_audio_buffer.commit"}))
        assert await wait_for(lambda: finished(session))

        assert events_of(session, "error")[-1]["error"]["type"] == "server_error"
        assert events_of(session, "response.done")[-1]["response"]["status"] == "failed"
        assert await wait_for(lambda: session.pending_events() == 0)
        await session.close()

    @pytest.mark.asyncio
    async def test_empty_transcript_ends_the_turn_without_audio(self):
        sent, send = collector()
        session = RealtimeSession(
            send=send,
            config=fast_config(),
            transcriber=lambda _audio: "",
            synthesize=sine_synth(chunks=1),
        )
        await session.start()
        await feed(session, sine(0.3))
        await session.handle_message(json.dumps({"type": "input_audio_buffer.commit"}))
        assert await wait_for(lambda: finished(session))

        assert "response.audio_transcript.done" in session.event_types()
        assert events_of(session, "response.audio.delta") == []
        assert events_of(session, "response.done")[-1]["response"]["status"] == "completed"
        await session.close()


# ─── Binary frames ──────────────────────────────────────────────────────────

class TestBinaryTransport:
    """Raw PCM16 binary frames are accepted next to JSON text frames."""

    @pytest.mark.asyncio
    async def test_raw_pcm16_frame_feeds_the_vad(self):
        sent, send = collector()
        session = RealtimeSession(send=send, config=fast_config())
        await session.start()

        chunk = float32_to_pcm16(sine(0.1))
        for _ in range(4):
            await session.handle_message(chunk)

        assert await wait_for(
            lambda: "input_audio_buffer.speech_started" in session.event_types()
        )
        assert events_of(session, "error") == []
        await session.close()

    @pytest.mark.asyncio
    async def test_binary_json_frame_is_still_parsed_as_json(self):
        sent, send = collector()
        session = RealtimeSession(send=send, config=fast_config())
        await session.start()

        await session.handle_message(b'{"type": "input_audio_buffer.clear"}')

        assert session.event_types() == ["session.created"]
        await session.close()


# ─── Latency profiling ──────────────────────────────────────────────────────

class TestLatencyProfiling:
    """An injected ``LatencyProfiler`` records one trace per answered turn."""

    @pytest.mark.asyncio
    async def test_turn_marks_are_published(self):
        from vram_core.monitoring import LatencyProfiler

        sent, send = collector()
        profiler = LatencyProfiler()
        session = RealtimeSession(
            send=send,
            config=fast_config(),
            transcriber=lambda _audio: "profile me",
            synthesize=sine_synth(chunks=2),
            profiler=profiler,
        )
        await session.start()
        await feed(session, sine(0.3))
        await session.handle_message(json.dumps({"type": "input_audio_buffer.commit"}))
        assert await wait_for(lambda: finished(session))
        assert await wait_for(lambda: len(profiler.traces) == 1)

        marks = profiler.traces[-1].marks
        assert {"vad_cutoff", "asr_transcribed", "tts_first_chunk"} <= set(marks)
        assert marks["tts_first_chunk"] >= marks["vad_cutoff"]

        stats = profiler.stage_stats()
        assert "asr_transcribed" in stats
        assert stats["asr_transcribed"]["count"] == 1.0
        await session.close()


# ─── FastAPI mount (/v1/realtime) ───────────────────────────────────────────

class TestRealtimeVoiceAliases:
    """OpenAI voice names map onto the edge-tts voice ids."""

    def test_openai_names_are_translated(self):
        try:
            from vram_core.api_server import (
                REALTIME_VOICE_ALIASES,
                _resolve_realtime_voice,
            )
        except ImportError:  # pragma: no cover - fastapi missing
            pytest.skip("fastapi/httpx not installed")

        assert {"alloy", "echo", "fable", "onyx", "nova", "shimmer",
                "verse"} <= set(REALTIME_VOICE_ALIASES)
        for name, mapped in REALTIME_VOICE_ALIASES.items():
            assert _resolve_realtime_voice(name) == mapped
            assert _resolve_realtime_voice(name.upper()) == mapped

    def test_unknown_and_missing_voices_pass_through(self):
        try:
            from vram_core.api_server import _resolve_realtime_voice
        except ImportError:  # pragma: no cover - fastapi missing
            pytest.skip("fastapi/httpx not installed")

        assert _resolve_realtime_voice(None) is None
        assert _resolve_realtime_voice("zh-CN-XiaoxiaoNeural") == "zh-CN-XiaoxiaoNeural"


class TestRealtimeWebSocketRoute:
    """The route is a thin transport over :class:`RealtimeSession`."""

    @pytest.mark.asyncio
    async def test_handshake_and_session_update(self):
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:  # pragma: no cover - fastapi missing
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge, \
                patch("vram_core.tts_engine.TTSEngine") as MockTTS:
            MockBridge.return_value = MagicMock(language="en", whisper_model="base")
            MockTTS.return_value = MagicMock()
            app = create_app(whisper_model="base")

            with TestClient(app) as client:
                with client.websocket_connect(
                    "/v1/realtime?voice=nova&language=en"
                ) as ws:
                    created = ws.receive_json()
                    assert created["type"] == "session.created"
                    session_view = created["session"]
                    assert session_view["id"].startswith("sess_")
                    assert session_view["voice"] == "nova"
                    assert session_view["language"] == "en"
                    assert session_view["turn_detection"]["type"] == "server_vad"
                    assert session_view["input_audio_format"] == "pcm16"
                    assert session_view["output_audio_format"] == "pcm16"
                    assert session_view["input_audio_sample_rate"] == 24000
                    assert session_view["output_audio_sample_rate"] == 24000

                    ws.send_text(json.dumps({
                        "type": "session.update",
                        "session": {
                            "voice": "alloy",
                            "turn_detection": {"silence_duration_ms": 400},
                        },
                    }))
                    updated = ws.receive_json()
                    assert updated["type"] == "session.updated"
                    assert updated["session"]["voice"] == "alloy"
                    assert updated["session"]["turn_detection"][
                        "silence_duration_ms"] == 400

    @pytest.mark.asyncio
    async def test_full_voice_turn_over_the_route(self):
        try:
            from fastapi.testclient import TestClient
            from vram_core.api_server import create_app
        except ImportError:  # pragma: no cover - fastapi missing
            pytest.skip("fastapi/httpx not installed")

        with patch("vram_core.api_server.WhisperBridge") as MockBridge, \
                patch("vram_core.tts_engine.TTSEngine") as MockTTS:
            bridge = MagicMock(language="en", whisper_model="base")
            bridge.transcribe.return_value = "hello from the route"
            MockBridge.return_value = bridge

            async def fake_stream(_text):
                yield sine(0.05)

            MockTTS.return_value = MagicMock(
                stream_synthesize=lambda text: fake_stream(text)
            )
            app = create_app(whisper_model="base")

            with TestClient(app) as client:
                with client.websocket_connect("/v1/realtime") as ws:
                    assert ws.receive_json()["type"] == "session.created"

                    for frame in append_frames(sine(0.3)):
                        ws.send_text(frame)
                    ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))

                    seen = []
                    transcript = None
                    for _ in range(30):
                        event = ws.receive_json()
                        seen.append(event["type"])
                        if event["type"] == "response.audio_transcript.done":
                            transcript = event["transcript"]
                        if event["type"] == "response.done":
                            done = event
                            break
                    else:  # pragma: no cover - the turn must always close
                        pytest.fail(f"the turn never completed: {seen}")

                    assert "input_audio_buffer.committed" in seen
                    assert seen.index("input_audio_buffer.committed") < seen.index(
                        "response.created")
                    assert seen.count("response.created") == 1
                    assert transcript == "hello from the route"
                    assert seen.count("response.audio.delta") >= 1
                    assert seen[-1] == "response.done"
                    assert done["response"]["status"] == "completed"
