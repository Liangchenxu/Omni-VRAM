"""
vram_core OpenAI Realtime API Compatible Gateway (v2.7.1)
=========================================================

An industrial-grade **OpenAI Realtime API** WebSocket protocol gateway that
stitches the existing Omni-VRAM building blocks into a standard real-time
speech-to-speech server::

    client audio --┐
                   ├─► StreamProcessor   (VAD + barge-in + NCC echo veto)
    base64 PCM16 --┘        │
                            ├─► ASR   (WhisperBridge / injected transcriber)
                            │
                            └─► TTSEngine (sentence-level streaming synthesis)
                                        │
    response.audio.delta ◄──────────────┘

The wire protocol mirrors ``wss://api.openai.com/v1/realtime``.

**Client events (accepted)**

===============================  ==================================================
``session.update``               reconfigure sample rate / VAD / voice / prompt
``input_audio_buffer.append``    base64 PCM16 audio chunk -> ``StreamProcessor.feed``
``input_audio_buffer.commit``    manual turn commit (VAD bypass)
``input_audio_buffer.clear``     drop the pending input buffer
``response.cancel``              client-initiated interruption
===============================  ==================================================

**Server events (emitted)**

========================================  =======================================
``session.created`` / ``session.updated`` handshake / acknowledgement
``input_audio_buffer.speech_started``     VAD speech start *or* barge-in truncation
``input_audio_buffer.speech_stopped``     VAD end of user speech
``input_audio_buffer.committed``          manual commit acknowledgement
``response.created``                      a new assistant turn begins
``response.audio_transcript.delta``       streaming ASR transcript increment
``response.audio_transcript.done``        final transcript of the turn
``response.audio.delta``                  base64 PCM16 TTS audio increment
``response.done``                         turn finished (``completed``/``cancelled``)
``response.cancelled``                    instant interruption frame
``error``                                 protocol / payload error
========================================  =======================================

**Instant interruption (flush on barge-in)**
    :attr:`StreamProcessor.on_interrupt` is bound to
    :meth:`RealtimeSession._on_interrupt`: the running TTS async generator is
    cancelled, the outbound send queue is flushed and the truncation frames
    (``input_audio_buffer.speech_started`` + ``response.cancelled`` +
    ``response.done`` with ``status="cancelled"``) are pushed downstream
    immediately -- no stale audio survives the interruption.

**Audio normalisation**
    Everything is normalised to **PCM16 mono**; sample-rate adaptation uses
    :meth:`vram_core.audio_utils.AudioProcessor.resample` (fast linear
    interpolation), so 24 kHz and 16 kHz clients interoperate transparently.

Usage::

    session = RealtimeSession(send=websocket_send, whisper_bridge=bridge)
    await session.start()                     # -> session.created
    await session.handle_message(raw_json)    # client events
    await session.close()
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import numpy as np

from vram_core.audio_utils import AudioProcessor
from vram_core.stream_processor import BargeInEvent, StreamConfig, StreamProcessor
from vram_core.tts_engine import SentenceStreamBuffer, TTSEngine

logger = logging.getLogger(__name__)

# ─── Protocol constants ─────────────────────────────────────────────────────

#: Default Realtime model identifier reported in ``session.created``.
REALTIME_MODEL = "omni-vram-realtime"

#: ``server_vad`` is the only turn-detection mode supported natively.
SUPPORTED_TURN_DETECTION = ("server_vad",)

#: PCM16 is the only wire sample format accepted/emitted by the gateway.
WIRE_AUDIO_FORMAT = "pcm16"

#: Maximum number of server events retained in memory for introspection/tests.
_EVENT_HISTORY_LIMIT = 2000

#: Safety cap for the manual-commit input buffer (seconds at processing rate).
_MANUAL_BUFFER_SECONDS = 60.0


# ─── Audio helpers (PCM16 <-> float32, base64, resampling) ──────────────────

def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """
    Decode little-endian PCM16 bytes to float32 samples in ``[-1.0, 1.0]``.

    A trailing odd byte (truncated frame) is dropped instead of raising, so a
    byte-splitting transport can never break the audio path.

    Args:
        pcm_bytes: Raw little-endian signed 16-bit PCM.

    Returns:
        Mono float32 array (empty when ``pcm_bytes`` is empty).
    """
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)
    if len(pcm_bytes) % 2:
        pcm_bytes = pcm_bytes[:-1]
    return np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(samples) -> bytes:
    """
    Encode float32 samples in ``[-1.0, 1.0]`` to little-endian PCM16 bytes.

    Values are clipped (never wrapped) so a hot signal cannot invert phase.

    Args:
        samples: Array-like of float samples.

    Returns:
        Raw little-endian signed 16-bit PCM.
    """
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return b""
    clipped = np.clip(array, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def pcm16_to_base64(pcm_bytes: bytes) -> str:
    """Base64-encode raw PCM16 bytes for a ``*.delta`` wire frame."""
    return base64.b64encode(pcm_bytes).decode("ascii")


def base64_to_pcm16(data: Any) -> bytes:
    """
    Decode a base64 audio payload with maximum transport tolerance.

    Accepts missing ``=`` padding, embedded whitespace/newlines, the URL-safe
    alphabet (``-``/``_``) and ``bytes`` input, because different client SDKs
    encode ``input_audio_buffer.append`` differently.

    Args:
        data: Base64 text (or raw ASCII bytes).

    Returns:
        Decoded bytes (empty string -> ``b""``).

    Raises:
        ValueError: When the payload is missing or not decodable base64.
    """
    if data is None:
        raise ValueError("audio payload is missing")
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data).decode("ascii", errors="ignore")
    else:
        raw = str(data)
    raw = "".join(raw.split())
    if not raw:
        return b""
    raw = raw.replace("-", "+").replace("_", "/")
    raw += "=" * (-len(raw) % 4)
    try:
        return base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError) as error:  # pragma: no cover - defensive
        raise ValueError(f"invalid base64 audio payload: {error}") from error


def resample_pcm16(pcm_bytes: bytes, orig_sr: int, target_sr: int) -> bytes:
    """
    Resample raw PCM16 bytes to ``target_sr``.

    Delegates to :meth:`vram_core.audio_utils.AudioProcessor.resample` so the
    torch-free path stays identical across the codebase.

    Args:
        pcm_bytes: Raw PCM16 bytes.
        orig_sr: Source sample rate in Hz.
        target_sr: Destination sample rate in Hz.

    Returns:
        Resampled PCM16 bytes (unchanged when the rates already match).
    """
    if not pcm_bytes or int(orig_sr) == int(target_sr):
        return pcm_bytes or b""
    samples = pcm16_to_float32(pcm_bytes)
    if samples.size == 0:
        return b""
    resampled = AudioProcessor.resample(samples, int(orig_sr), int(target_sr))
    return float32_to_pcm16(np.asarray(resampled, dtype=np.float32))


def _as_transcript_text(result: Any) -> str:
    """
    Coerce a transcription result (``str`` or ``WhisperResult``-like) to text.

    Any non-string ``.text`` (e.g. an unconfigured ``MagicMock`` in a test) is
    treated as "no transcript" rather than being stringified into garbage.
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    return ""


def _looks_like_json(raw: bytes) -> bool:
    """Heuristic: a binary frame starting with ``{``/``[`` is JSON, else audio."""
    stripped = bytes(raw).lstrip()
    return stripped[:1] in (b"{", b"[")


# ─── Session configuration ──────────────────────────────────────────────────

@dataclass
class RealtimeSessionConfig:
    """
    Configuration of one Realtime session (the wire ``session`` object).

    Attributes:
        model: Model identifier echoed back in ``session.created``.
        voice: TTS voice name requested by the client.
        instructions: System prompt / style instructions (informational; the
            gateway is ASR->TTS, so it is echoed rather than executed).
        language: Optional BCP-47 language hint for the ASR/TTS layer.
        input_audio_format: Wire format of inbound audio (``pcm16`` only).
        output_audio_format: Wire format of outbound audio (``pcm16`` only).
        input_sample_rate: Sample rate of the client's PCM16 stream (Hz).
        output_sample_rate: Sample rate of the server's PCM16 stream (Hz).
        processing_sample_rate: Internal StreamProcessor rate (Hz).
        vad_threshold: RMS energy threshold for the server-side VAD.
        silence_duration_ms: Silence needed to close a user turn.
        min_speech_ms: Minimum user speech duration that opens a turn.
        turn_detection: ``server_vad`` (the only supported mode).
        enable_barge_in: Arm instant interruption detection.
        chunk_duration_ms: Internal chunk granularity of the processor.
        tts_sample_rate: Native sample rate of the TTS backend output.
    """

    model: str = REALTIME_MODEL
    voice: str = "alloy"
    instructions: str = ""
    language: Optional[str] = None
    input_audio_format: str = WIRE_AUDIO_FORMAT
    output_audio_format: str = WIRE_AUDIO_FORMAT
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000
    processing_sample_rate: int = 16000
    vad_threshold: float = 0.02
    silence_duration_ms: int = 800
    min_speech_ms: int = 200
    turn_detection: str = "server_vad"
    enable_barge_in: bool = True
    chunk_duration_ms: int = 100
    tts_sample_rate: int = 24000

    def as_dict(self) -> Dict[str, Any]:
        """Serialisable view used by ``session.created`` / ``session.updated``."""
        return {
            "object": "realtime.session",
            "model": self.model,
            "voice": self.voice,
            "instructions": self.instructions,
            "language": self.language,
            "input_audio_format": self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "input_audio_sample_rate": int(self.input_sample_rate),
            "output_audio_sample_rate": int(self.output_sample_rate),
            "processing_sample_rate": int(self.processing_sample_rate),
            "turn_detection": {
                "type": self.turn_detection,
                "threshold": float(self.vad_threshold),
                "silence_duration_ms": int(self.silence_duration_ms),
                "min_speech_ms": int(self.min_speech_ms),
            },
            "enable_barge_in": bool(self.enable_barge_in),
        }

    def apply(self, payload: Dict[str, Any]) -> List[str]:
        """
        Merge a client ``session`` patch into this configuration.

        Both the flat Omni-VRAM dialect (``input_audio_sample_rate``,
        ``turn_detection.threshold``) and the nested OpenAI shape
        (``audio.input.format.rate``) are understood.

        Args:
            payload: The ``session`` object of a ``session.update`` event
                (anything that is not an object is ignored).

        Returns:
            The attribute names that actually changed.
        """
        changed: List[str] = []

        if not isinstance(payload, dict):
            logger.debug(
                "Ignoring session patch of type %s", type(payload).__name__
            )
            return changed

        def _set(attr: str, value: Any) -> None:
            if value is None:
                return
            if getattr(self, attr) != value:
                setattr(self, attr, value)
                changed.append(attr)

        for key in ("model", "voice", "instructions", "language"):
            if key in payload:
                _set(key, payload[key])

        _set("input_sample_rate", payload.get("input_audio_sample_rate"))
        _set("output_sample_rate", payload.get("output_audio_sample_rate"))
        _set("vad_threshold", payload.get("vad_threshold"))
        _set("silence_duration_ms", payload.get("silence_duration_ms"))
        _set("min_speech_ms", payload.get("min_speech_ms"))
        _set("enable_barge_in", payload.get("enable_barge_in"))

        turn = payload.get("turn_detection")
        if isinstance(turn, dict):
            _set("turn_detection", turn.get("type") or self.turn_detection)
            _set("vad_threshold", turn.get("threshold"))
            _set("silence_duration_ms", turn.get("silence_duration_ms"))
            _set("min_speech_ms", turn.get("min_speech_ms"))

        audio = payload.get("audio")
        if isinstance(audio, dict):
            for direction in ("input", "output"):
                branch = audio.get(direction)
                fmt = branch.get("format") if isinstance(branch, dict) else None
                if isinstance(fmt, dict):
                    _set(f"{direction}_sample_rate", fmt.get("rate"))
                    if fmt.get("type"):
                        _set(f"{direction}_audio_format", fmt["type"])

        return changed


# ─── Realtime session ───────────────────────────────────────────────────────

class RealtimeSession:
    """
    One OpenAI Realtime API conversation (speech-to-speech over a message sink).

    The session orchestrates the existing Omni-VRAM components:

    * :class:`vram_core.stream_processor.StreamProcessor` -- server-side VAD and
      the arm-able barge-in detector (with the v2.6.1 NCC echo veto);
    * an ASR callable (an injected ``transcriber`` or
      ``whisper_bridge.transcribe``);
    * a streaming TTS source (an injected ``synthesize`` hook or
      :meth:`vram_core.tts_engine.TTSEngine.stream_synthesize`);
    * an optional :class:`vram_core.monitoring.LatencyProfiler`.

    It is transport agnostic: ``send`` is any callable accepting a JSON-ready
    ``dict`` and may be a coroutine function. The WebSocket route in
    :mod:`vram_core.api_server` is a thin adapter around it.

    Args:
        send: Sink for server events (sync or async callable).
        config: Optional :class:`RealtimeSessionConfig`.
        whisper_bridge: ASR bridge exposing ``transcribe(audio, sample_rate=...)``.
        tts_engine: :class:`~vram_core.tts_engine.TTSEngine` used when no
            ``synthesize`` hook is injected.
        transcriber: Optional ``audio -> str | result`` override (sync or async).
        synthesize: Optional ``text -> Iterable[bytes] | AsyncIterator[bytes]``
            override.
        profiler: Optional latency profiler (``start_trace`` / ``mark`` /
            ``finish_trace``).
        session_id: Explicit session id (generated when omitted).
        loop: Event loop owning the outbound queue (inferred in :meth:`start`).
    """

    #: Event types this gateway is able to handle.
    CLIENT_EVENTS = (
        "session.update",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "input_audio_buffer.clear",
        "response.cancel",
    )

    def __init__(
        self,
        send: Callable[[Dict[str, Any]], Any],
        *,
        config: Optional[RealtimeSessionConfig] = None,
        whisper_bridge: Optional[object] = None,
        tts_engine: Optional[TTSEngine] = None,
        transcriber: Optional[Callable[[np.ndarray], Any]] = None,
        synthesize: Optional[Callable[[str], Any]] = None,
        profiler: Optional[object] = None,
        session_id: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self._send = send
        self.config = config or RealtimeSessionConfig()
        self.whisper_bridge = whisper_bridge
        self.tts_engine = tts_engine
        self.profiler = profiler
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:24]}"
        self._transcriber = transcriber
        self._synthesize = synthesize
        self._loop = loop

        self._out_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        self._pump_task: Optional[asyncio.Task] = None
        self._response_task: Optional[asyncio.Future] = None
        self._response_id: Optional[str] = None
        self._item_id: Optional[str] = None
        self._response_cancelled = False
        self._closed = False
        self._lock = threading.RLock()

        # Manual-commit input accumulation (VAD bypass path).
        self._manual_buffer: List[np.ndarray] = []
        self._manual_samples = 0

        # Bounded history of every event the server produced (introspection/tests).
        self._event_log: List[Dict[str, Any]] = []

        # StreamProcessor is deliberately created WITHOUT a whisper bridge: the
        # session owns the single ASR pass per turn, so audio is never
        # transcribed twice.
        self.processor = StreamProcessor(config=self._build_stream_config())
        self.processor.on_speech_start = self._on_speech_start
        self.processor.on_speech_end = self._on_speech_end
        self.processor.on_interrupt = self._on_interrupt

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def events(self) -> List[Dict[str, Any]]:
        """Every server event emitted so far (bounded history, copy)."""
        with self._lock:
            return list(self._event_log)

    def event_types(self) -> List[str]:
        """The ``type`` of every event emitted so far, in order."""
        return [event["type"] for event in self.events]

    def pending_events(self) -> int:
        """Number of events queued but not yet handed to ``send``."""
        return self._out_queue.qsize()

    def session_view(self) -> Dict[str, Any]:
        """The serialisable ``session`` object (with the live ``id``)."""
        view = self.config.as_dict()
        view["id"] = self.session_id
        return view

    def _build_stream_config(self) -> StreamConfig:
        """Map the Realtime session configuration onto a ``StreamConfig``."""
        return StreamConfig(
            sample_rate=int(self.config.processing_sample_rate),
            chunk_duration_ms=int(self.config.chunk_duration_ms),
            vad_threshold=float(self.config.vad_threshold),
            vad_silence_duration_ms=int(self.config.silence_duration_ms),
            vad_min_speech_ms=int(self.config.min_speech_ms),
            enable_barge_in=bool(self.config.enable_barge_in),
        )

    # ── Outbound event plumbing ──────────────────────────────────────────

    def _emit(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        """
        Build, record and queue a server event.

        Args:
            event_type: OpenAI Realtime event ``type`` string.
            **fields: Event payload.

        Returns:
            The complete event dict.
        """
        event: Dict[str, Any] = {
            "event_id": f"evt_{uuid.uuid4().hex[:24]}",
            "type": event_type,
        }
        event.update(fields)
        with self._lock:
            self._event_log.append(event)
            if len(self._event_log) > _EVENT_HISTORY_LIMIT:
                del self._event_log[: len(self._event_log) - _EVENT_HISTORY_LIMIT]
        self._enqueue(event)
        return event

    def _enqueue(self, event: Dict[str, Any]) -> None:
        """
        Queue an event for the pump.

        Thread safe: callbacks may fire from an ASR worker thread, in which case
        the put is marshalled onto the owning event loop.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            self._out_queue.put_nowait(event)
            return
        try:
            running: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._out_queue.put_nowait(event)
        else:
            loop.call_soon_threadsafe(self._out_queue.put_nowait, event)

    def _flush_out_queue(self) -> int:
        """
        Drop every queued-but-unsent event -- the "instant" half of a barge-in.

        Returns:
            Number of events discarded.
        """
        flushed = 0
        while True:
            try:
                self._out_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            flushed += 1
        return flushed

    def _error(self, code: str, message: str) -> Dict[str, Any]:
        """Emit an OpenAI-shaped ``error`` event."""
        return self._emit(
            "error", error={"type": code, "code": code, "message": message}
        )

    async def _pump(self) -> None:
        """Drain the outbound queue into ``send``, one event at a time."""
        while True:
            event = await self._out_queue.get()
            try:
                result = self._send(event)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - a dead sink must not hang
                logger.warning("Realtime send sink failed: %s", error)
                return

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> Dict[str, Any]:
        """
        Start the outbound pump and perform the ``session.created`` handshake.

        Returns:
            The serialised session object that was sent to the client.
        """
        self._loop = asyncio.get_running_loop()
        self._closed = False
        if self._pump_task is None or self._pump_task.done():
            self._pump_task = self._loop.create_task(self._pump())
        view = self.session_view()
        self._emit("session.created", session=view)
        logger.info(
            "Realtime session %s started (model=%s)", self.session_id, view["model"]
        )
        return view

    async def close(self) -> None:
        """Cancel in-flight work and stop the outbound pump."""
        self._closed = True
        self._response_cancelled = True
        task = self._response_task
        self._response_task = None
        if task is not None and not task.done():
            task.cancel()
        pump = self._pump_task
        self._pump_task = None
        if pump is not None and not pump.done():
            pump.cancel()
            try:
                await pump
            except asyncio.CancelledError:  # pragma: no cover - normal shutdown
                pass
            except Exception:  # noqa: BLE001 - shutdown must never raise
                pass

    async def __aenter__(self) -> "RealtimeSession":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ── Client protocol routing ──────────────────────────────────────────

    async def handle_message(self, raw: Any) -> None:
        """
        Handle one client frame (JSON text, or raw PCM16 binary audio).

        Binary frames are a transport convenience: a frame that does not start
        with ``{``/``[`` is treated as raw PCM16 at ``config.input_sample_rate``.

        Args:
            raw: ``str``/``bytes`` JSON payload, or raw PCM16 ``bytes``.
        """
        if isinstance(raw, (bytes, bytearray)) and not _looks_like_json(raw):
            self.feed_pcm16(bytes(raw))
            return
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            self._error(
                "invalid_request_error",
                "Could not parse the client message as JSON.",
            )
            return
        if not isinstance(data, dict):
            self._error(
                "invalid_request_error", "Client message must be a JSON object."
            )
            return
        await self.handle_event(data)

    async def handle_event(self, event: Dict[str, Any]) -> None:
        """Dispatch one decoded client event to its handler."""
        etype = str(event.get("type") or "")
        if etype == "session.update":
            session = event.get("session")
            self.update_session(session if isinstance(session, dict) else {})
        elif etype == "input_audio_buffer.append":
            self.append_audio(event.get("audio"))
        elif etype == "input_audio_buffer.commit":
            await self.commit_audio()
        elif etype == "input_audio_buffer.clear":
            self.clear_input()
        elif etype == "response.cancel":
            self.cancel_response(reason="client_request")
        else:
            self._error(
                "invalid_request_error",
                f"Unsupported event type: {etype or '<missing>'}",
            )

    # ── session.update ───────────────────────────────────────────────────

    def update_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a ``session.update`` patch and acknowledge it.

        Unsupported values are clamped (PCM16 is the only wire format, and
        ``server_vad`` the only turn detector); each clamp is reported back
        through the ``changed`` list instead of failing the turn.

        Args:
            payload: The ``session`` object of the client event.

        Returns:
            The updated session view that was acknowledged.
        """
        changed = self.config.apply(payload) if isinstance(payload, dict) else []
        for attr in ("input_audio_format", "output_audio_format"):
            if str(getattr(self.config, attr)).lower() != WIRE_AUDIO_FORMAT:
                setattr(self.config, attr, WIRE_AUDIO_FORMAT)
                changed.append(attr)
        if str(self.config.turn_detection) not in SUPPORTED_TURN_DETECTION:
            self.config.turn_detection = "server_vad"
            changed.append("turn_detection")

        # Re-derive the live VAD / barge-in knobs from the new configuration.
        self.processor.config = self._build_stream_config()
        self.processor.vad.threshold = float(self.config.vad_threshold)

        view = self.session_view()
        self._emit("session.updated", session=view, changed=sorted(set(changed)))
        logger.info(
            "Realtime session %s updated: %s", self.session_id, sorted(set(changed))
        )
        return view

    # ── Audio input ──────────────────────────────────────────────────────

    def append_audio(self, audio_b64: Any) -> None:
        """
        Handle ``input_audio_buffer.append``.

        Args:
            audio_b64: Base64 PCM16 payload (decoded tolerantly).
        """
        try:
            pcm = base64_to_pcm16(audio_b64)
        except ValueError as error:
            self._error("invalid_request_error", str(error))
            return
        if pcm:
            self.feed_pcm16(pcm)

    def feed_pcm16(self, pcm_bytes: bytes) -> None:
        """
        Feed raw PCM16 audio taken at ``config.input_sample_rate``.

        The bytes are decoded, resampled to the internal processing rate and
        handed to :meth:`StreamProcessor.feed`; the same samples are retained for
        the manual ``commit`` path.

        Args:
            pcm_bytes: Raw little-endian mono PCM16 audio.
        """
        samples = pcm16_to_float32(pcm_bytes)
        if samples.size == 0:
            return
        input_rate = int(self.config.input_sample_rate)
        proc_rate = int(self.config.processing_sample_rate)
        if input_rate != proc_rate:
            samples = np.asarray(
                AudioProcessor.resample(samples, input_rate, proc_rate),
                dtype=np.float32,
            )
        self.feed_audio(samples)

    def feed_audio(self, samples: np.ndarray) -> None:
        """
        Feed float32 mono samples already at the processing rate.

        Drives both the server-side VAD (``speech_started`` / ``speech_stopped``)
        and the barge-in detector through :class:`StreamProcessor`.

        Args:
            samples: Array-like float samples in ``[-1.0, 1.0]``.
        """
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return
        self._append_manual(chunk)
        self.processor.feed(chunk)

    def _append_manual(self, chunk: np.ndarray) -> None:
        """Accumulate input for the manual ``commit`` path, bounded in memory."""
        limit = max(
            1, int(_MANUAL_BUFFER_SECONDS * int(self.config.processing_sample_rate))
        )
        excess = self._manual_samples + int(chunk.size) - limit
        while self._manual_buffer and excess > 0:
            head = self._manual_buffer[0]
            if head.size <= excess:
                excess -= int(head.size)
                self._manual_samples -= int(head.size)
                self._manual_buffer.pop(0)
            else:
                self._manual_buffer[0] = head[excess:]
                self._manual_samples -= excess
                excess = 0
        self._manual_buffer.append(chunk)
        self._manual_samples += int(chunk.size)

    async def commit_audio(self) -> None:
        """
        Handle ``input_audio_buffer.commit`` -- start a turn without VAD.

        The accumulated input is closed and the response pipeline is driven
        directly, which lets a client that does its own endpointing
        (push-to-talk) reuse the whole ASR/TTS chain. An empty buffer is
        acknowledged but never starts a response.
        """
        audio = (
            np.concatenate(self._manual_buffer)
            if self._manual_buffer
            else np.zeros(0, dtype=np.float32)
        )
        self._manual_buffer = []
        self._manual_samples = 0
        self._item_id = f"item_{uuid.uuid4().hex[:24]}"
        self._emit("input_audio_buffer.committed", item_id=self._item_id)
        if audio.size == 0:
            return
        rate = max(1, int(self.config.processing_sample_rate))
        self._emit(
            "input_audio_buffer.speech_started",
            item_id=self._item_id,
            audio_start_ms=0,
        )
        self._emit(
            "input_audio_buffer.speech_stopped",
            item_id=self._item_id,
            audio_end_ms=int(audio.size / rate * 1000),
        )
        self._schedule_response(audio)

    def clear_input(self) -> None:
        """
        Handle ``input_audio_buffer.clear``.

        Drops the manual-commit buffer and every input frame buffered by the
        processor (:meth:`StreamProcessor.flush`), so an aborted push-to-talk
        turn can never leak into the next one.
        """
        self._manual_buffer = []
        self._manual_samples = 0
        self.processor.flush()

    # ── StreamProcessor callbacks (VAD + barge-in) ───────────────────────

    def _on_speech_start(self) -> None:
        """VAD opened a user turn: emit ``input_audio_buffer.speech_started``."""
        self._item_id = f"item_{uuid.uuid4().hex[:24]}"
        self._emit(
            "input_audio_buffer.speech_started",
            item_id=self._item_id,
            audio_start_ms=0,
        )

    def _on_speech_end(self, audio: np.ndarray) -> None:
        """VAD closed a user turn: emit the stop frame and answer it."""
        rate = max(1, int(self.config.processing_sample_rate))
        self._emit(
            "input_audio_buffer.speech_stopped",
            item_id=self._item_id,
            audio_end_ms=int(len(audio) / rate * 1000),
        )
        self._schedule_response(np.asarray(audio, dtype=np.float32))

    def _on_interrupt(self, event: Optional[BargeInEvent] = None) -> None:
        """
        Barge-in hook: truncate the assistant turn *now*.

        Invoked synchronously from :meth:`StreamProcessor.feed` (the audio fast
        path), so it must never block: the TTS generator is cancelled and the
        outbound queue flushed, which is what makes the interruption audibly
        instant. ``event`` carries the detection evidence (energy, threshold,
        detection latency) for logging.

        Args:
            event: Barge-in evidence from the detector (unused beyond logging).
        """
        if event is not None:
            logger.debug(
                "Barge-in drives truncation (energy=%.4f, latency=%.3f ms)",
                event.energy,
                event.detection_latency_ms,
            )
        self.cancel_response(reason="barge_in")

    # ── Interruption ─────────────────────────────────────────────────────

    def cancel_response(self, reason: str = "client_request") -> int:
        """
        Interrupt the in-flight assistant turn immediately.

        Cancels the streaming-response task, discards every queued-but-unsent
        event, disarms playback and emits the truncation frames
        (``response.cancelled`` followed by ``response.done`` with
        ``status="cancelled"``). Idempotent, and safe when nothing is running.

        Args:
            reason: ``barge_in`` (VAD detection) or ``client_request``.

        Returns:
            Number of stale events that were dropped from the outbound queue.
        """
        self._response_cancelled = True
        task = self._response_task
        if task is not None and not task.done():
            task.cancel()
        flushed = self._flush_out_queue()

        response_id = self._response_id
        self._response_id = None
        # Mute the playback pipeline first: no queued TTS audio may survive.
        self.processor.set_playback_state(False)
        self.processor.clear_playback_reference()

        if response_id is not None:
            self._emit("response.cancelled", response_id=response_id, reason=reason)
            self._emit(
                "response.done",
                response={"id": response_id, "status": "cancelled", "reason": reason},
            )
        logger.info(
            "Realtime response cancelled (reason=%s, flushed=%d events)",
            reason,
            flushed,
        )
        return flushed

    # ── Response pipeline (ASR -> TTS) ───────────────────────────────────

    def _schedule_response(self, audio: np.ndarray) -> Optional[asyncio.Future]:
        """
        Start the ASR -> TTS response pipeline for one closed user turn.

        Exactly one response may be in flight: a turn arriving while the previous
        answer is still streaming is dropped, because the ongoing answer owns the
        output (matching OpenAI server-VAD behaviour).

        Args:
            audio: User speech segment (float32, processing rate).

        Returns:
            The scheduled task/future, or the still-running one, or None when
            nothing could be scheduled.
        """
        existing = self._response_task
        if existing is not None and not existing.done():
            logger.debug("Response already in flight; dropping the new turn")
            return existing
        if self._closed:
            return None
        loop = self._loop
        if loop is None or loop.is_closed():
            logger.debug("No event loop bound; response not scheduled")
            return None
        coro = self._run_response(np.asarray(audio, dtype=np.float32))
        try:
            running: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._response_task = loop.create_task(coro)
        else:  # pragma: no cover - only hit when feed() is driven off-loop
            self._response_task = asyncio.run_coroutine_threadsafe(coro, loop)
        return self._response_task

    async def _run_response(self, audio: np.ndarray) -> None:
        """
        Transcribe one user turn and stream the synthesized answer back.

        Emits, in order: ``response.created`` -> ``response.audio_transcript.delta``
        -> ``response.audio_transcript.done`` -> one ``response.audio.delta`` per
        TTS chunk -> ``response.done``. A cancellation (barge-in or explicit
        ``response.cancel``) is announced by :meth:`cancel_response`.

        Args:
            audio: User speech segment (float32, processing rate).
        """
        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        item_id = f"item_{uuid.uuid4().hex[:24]}"
        self._response_id = response_id
        self._response_cancelled = False

        trace = None
        if self.profiler is not None:
            try:
                trace = self.profiler.start_trace(f"{self.session_id}:{response_id}")
            except Exception:  # noqa: BLE001 - profiling is best effort
                trace = None
        self._mark(trace, "vad_cutoff")

        self._emit(
            "response.created",
            response={
                "id": response_id,
                "object": "realtime.response",
                "status": "in_progress",
            },
        )
        try:
            text = await self._transcribe(audio)
            self._mark(trace, "asr_transcribed")
            if self._response_cancelled:
                return

            transcript = text.strip()
            if transcript:
                self._emit(
                    "response.audio_transcript.delta",
                    response_id=response_id,
                    item_id=item_id,
                    delta=transcript,
                )
            self._emit(
                "response.audio_transcript.done",
                response_id=response_id,
                item_id=item_id,
                transcript=transcript,
            )
            if not transcript:
                self._emit(
                    "response.done",
                    response={
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "completed",
                    },
                )
                return

            await self._stream_tts(transcript, response_id, item_id, trace)
            if self._response_cancelled:
                return
            self._emit(
                "response.done",
                response={
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "completed",
                },
            )
        except asyncio.CancelledError:
            # The truncation frames were emitted by cancel_response().
            logger.debug("Realtime response %s cancelled", response_id)
            raise
        except Exception as error:  # noqa: BLE001 - one bad turn must not kill the socket
            logger.warning("Realtime response failed: %s", error)
            self._error("server_error", f"Response generation failed: {error}")
            self._emit(
                "response.done",
                response={"id": response_id, "status": "failed"},
            )
        finally:
            if not self._response_cancelled:
                self.processor.set_playback_state(False)
                self.processor.clear_playback_reference()
            if self._response_id == response_id:
                self._response_id = None
            self._finish_trace(trace)

    # ── ASR / TTS plumbing ───────────────────────────────────────────────

    async def _transcribe(self, audio: np.ndarray) -> str:
        """
        Run automatic speech recognition for one user turn.

        The injected ``transcriber`` hook wins (sync or async); otherwise the
        ``whisper_bridge`` runs in a worker thread so the event loop (and hence
        the barge-in fast path) is never blocked.

        Args:
            audio: User speech segment (float32, processing rate).

        Returns:
            The recognised text (``""`` when no ASR backend is configured).
        """
        if self._transcriber is not None:
            result = self._transcriber(audio)
            if inspect.isawaitable(result):
                result = await result
            return _as_transcript_text(result)
        if self.whisper_bridge is not None:
            loop = asyncio.get_running_loop()
            sample_rate = int(self.config.processing_sample_rate)
            result = await loop.run_in_executor(
                None,
                lambda: self.whisper_bridge.transcribe(audio, sample_rate=sample_rate),
            )
            return _as_transcript_text(result)
        logger.debug("No ASR backend configured; the turn has no transcript")
        return ""

    async def _stream_tts(
        self,
        text: str,
        response_id: str,
        item_id: str,
        trace: Any = None,
    ) -> int:
        """
        Synthesize ``text`` and emit one ``response.audio.delta`` per chunk.

        Every emitted chunk is also registered as the playback reference, so the
        v2.6.1 NCC echo veto recognises the speaker's own output instead of
        treating it as a barge-in.

        Args:
            text: Text to speak (the turn's transcript).
            response_id: Owning response id.
            item_id: Owning conversation item id.
            trace: Optional latency trace for the ``tts_first_chunk`` mark.

        Returns:
            Number of audio chunks emitted.
        """
        emitted = 0
        # Arm barge-in for this playback session (DuplexState.SPEAKING).
        self.processor.set_playback_state(True)
        async for chunk in self._synthesize_chunks(text):
            if self._response_cancelled:
                break
            pcm = self._to_wire_pcm16(chunk)
            if not pcm:
                continue
            self.processor.register_playback_chunk(pcm16_to_float32(pcm))
            self._emit(
                "response.audio.delta",
                response_id=response_id,
                item_id=item_id,
                delta=pcm16_to_base64(pcm),
            )
            emitted += 1
            if emitted == 1:
                self._mark(trace, "tts_first_chunk")
        return emitted

    async def _synthesize_chunks(self, text: str) -> AsyncIterator[bytes]:
        """
        Normalise the configured TTS source into one async chunk iterator.

        Accepts an injected ``synthesize`` hook returning an async iterator, an
        awaitable or a plain iterable, and otherwise falls back to
        :meth:`vram_core.tts_engine.TTSEngine.stream_synthesize`.
        """
        if self._synthesize is not None:
            source = self._synthesize(text)
            if inspect.isawaitable(source):
                source = await source
            if hasattr(source, "__aiter__"):
                async for chunk in source:
                    yield chunk
            else:
                for chunk in source:
                    yield chunk
            return
        if self.tts_engine is None:
            logger.debug("No TTS backend configured; the turn carries no audio")
            return
        async for chunk in self.tts_engine.stream_synthesize(text):
            yield chunk

    def _to_wire_pcm16(self, chunk: Any) -> bytes:
        """
        Normalise one TTS chunk to PCM16 at ``config.output_sample_rate``.

        ``bytes`` are taken as PCM16 at the backend's native rate (resampled when
        it differs from the wire rate); float numpy arrays are encoded directly.
        Anything unusable yields ``b""`` -- the chunk is skipped, never fatal.

        Args:
            chunk: Raw TTS output (``bytes`` or a numpy array).

        Returns:
            PCM16 bytes ready for base64 encoding.
        """
        if chunk is None:
            return b""
        if isinstance(chunk, np.ndarray):
            array = np.asarray(chunk).reshape(-1)
            if array.dtype in (np.float32, np.float64):
                raw = float32_to_pcm16(array)
            else:
                raw = (
                    np.clip(array.astype(np.int32), -32768, 32767)
                    .astype("<i2")
                    .tobytes()
                )
        elif isinstance(chunk, (bytes, bytearray, memoryview)):
            raw = bytes(chunk)
        else:
            return b""
        if len(raw) % 2:
            raw = raw[:-1]
        if not raw:
            return b""
        tts_rate = int(self.config.tts_sample_rate)
        out_rate = int(self.config.output_sample_rate)
        if tts_rate != out_rate:
            raw = resample_pcm16(raw, tts_rate, out_rate)
        return raw

    # ── Latency profiling ────────────────────────────────────────────────

    @staticmethod
    def _mark(trace: Any, stage: str) -> None:
        """Timestamp a pipeline stage on the per-turn trace (best effort)."""
        if trace is None:
            return
        try:
            trace.mark(stage)
        except Exception:  # noqa: BLE001 - profiling must never break audio
            logger.debug("Latency mark(%s) failed", stage, exc_info=True)

    def _finish_trace(self, trace: Any) -> None:
        """Close the per-turn latency trace and publish its stage gauges."""
        if trace is None or self.profiler is None:
            return
        try:
            self.profiler.finish_trace(trace)
        except Exception:  # noqa: BLE001 - profiling must never break audio
            logger.debug("Latency finish_trace failed", exc_info=True)


__all__ = [
    "REALTIME_MODEL",
    "SUPPORTED_TURN_DETECTION",
    "WIRE_AUDIO_FORMAT",
    "RealtimeSession",
    "RealtimeSessionConfig",
    "base64_to_pcm16",
    "float32_to_pcm16",
    "pcm16_to_base64",
    "pcm16_to_float32",
    "resample_pcm16",
]

