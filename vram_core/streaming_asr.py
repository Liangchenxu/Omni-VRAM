"""
vram_core Streaming ASR (Automatic Speech Recognition)
======================================================

Real-time streaming speech recognition with sliding window + incremental
recognition strategy. Achieves first-word latency < 500ms.

Architecture:
    - Sliding window: 2s window, 0.5s step (overlap ~300-500ms of audio)
    - Incremental recognition: re-process only new audio each step
    - Dynamic overlap-add alignment (LCS): consecutive window transcripts are
      merged so a word is never emitted twice at the window boundary
    - Whisper hallucination suppression: known artefact phrases, punctuation
      only output, low-energy (silence) gating and pathological n-gram loops
    - Callback-driven: on_partial_result / on_final_result
    - Supports Chinese-English mixed recognition

Usage:
    from vram_core.streaming_asr import StreamASR

    asr = StreamASR(language="zh")
    asr.on_partial_result = lambda text: print(f"Partial: {text}")
    asr.on_final_result = lambda result: print(f"Final: {result.text}")
    asr.start()
    # ... feed audio via asr.feed(chunk)
    asr.stop()

Thread Safety:
    StreamASR is designed to be used from a single thread or with
    external synchronization.
"""

import re
import time
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from dataclasses import dataclass, field

from vram_core.whisper import WhisperBridge, WhisperBackend, WhisperResult

logger = logging.getLogger(__name__)


@dataclass
class StreamASRConfig:
    """
    Configuration for streaming ASR.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        window_duration: Sliding window duration in seconds.
        step_duration: Step size in seconds (how often to re-recognize).
        min_audio_duration: Minimum audio duration before first recognition.
        language: Language code (zh, en, etc.) or None for auto-detect.
        whisper_model: Whisper model size (tiny/base/small/medium/large).
        backend: Whisper backend to use.
        vad_threshold: Energy threshold for voice activity detection.
        silence_timeout: Seconds of silence before finalizing a segment.
        overlap_duration: Overlap between windows to avoid cutting words.

        enable_overlap_alignment: Merge consecutive window transcripts with the
            dynamic overlap-add aligner (removes duplicated boundary words).
        max_overlap_chars: Maximum number of characters that may be treated as
            overlap between two consecutive transcripts.
        normalize_punctuation: Ignore punctuation/space drift when matching the
            overlap (ASR often punctuates the same words differently).
        use_lcs_alignment: Enable the LCS fallback that absorbs single character
            recognition jitter (e.g. homophone substitutions).
        enable_hallucination_filter: Enable Whisper hallucination suppression.
        min_speech_energy: Window RMS below this value is treated as silence and
            never sent to Whisper (silence is the main hallucination source).
        max_ngram_size: Longest repeated token block (n-gram) considered when
            detecting pathological loops.
        min_repeat_count: Number of consecutive repeats that marks a loop as
            pathological (e.g. "谢谢大家" x3).
        repeat_keep: How many repetitions are kept when a loop is truncated.
        max_transcript_chars: Safety cap for a single utterance transcript
            (finalizes the segment when exceeded).
    """
    sample_rate: int = 16000
    window_duration: float = 2.0
    step_duration: float = 0.5
    min_audio_duration: float = 0.5
    language: Optional[str] = "zh"
    whisper_model: str = "base"
    backend: WhisperBackend = WhisperBackend.AUTO
    vad_threshold: float = 0.01
    silence_timeout: float = 1.5
    overlap_duration: float = 0.2

    # ---- Overlap alignment ----
    enable_overlap_alignment: bool = True
    max_overlap_chars: int = 40
    normalize_punctuation: bool = True
    use_lcs_alignment: bool = True

    # ---- Hallucination suppression ----
    enable_hallucination_filter: bool = True
    min_speech_energy: float = 0.005
    max_ngram_size: int = 8
    min_repeat_count: int = 3
    repeat_keep: int = 1
    max_transcript_chars: int = 4000


@dataclass
class StreamASRResult:
    """
    Result from streaming ASR.

    Attributes:
        text: Transcribed text.
        is_final: Whether this is a final (committed) result.
        timestamp: Timestamp when result was generated.
        start_time: Start time of the audio segment (seconds from stream start).
        end_time: End time of the audio segment.
        confidence: Confidence score.
        language: Detected language.
    """
    text: str = ""
    is_final: bool = False
    timestamp: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0
    language: str = "unknown"


# ---------------------------------------------------------------------------
# Dynamic overlap-add (LCS) alignment
# ---------------------------------------------------------------------------

# Token pattern: latin words / numbers, whitespace runs, and single characters
# (CJK ideographs and punctuation are tokenised one character at a time).
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|\s+|[^\sA-Za-z0-9]")


def tokenize_text(text: str) -> List[str]:
    """
    Split text into word / CJK-character / whitespace / punctuation tokens.

    Args:
        text: Input text.

    Returns:
        List of tokens (whitespace preserved as its own tokens).
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text)


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """
    Longest common subsequence length between two token sequences.

    Rolling-row dynamic programming: O(len(left) * len(right)) time and
    O(len(right)) extra memory.

    Args:
        left: First token sequence.
        right: Second token sequence.

    Returns:
        LCS length (0 when either side is empty).
    """
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for l_token in left:
        current = [0]
        for j, r_token in enumerate(right, start=1):
            if l_token == r_token:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def _normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """
    Strip punctuation/whitespace (lower-casing latin text) and keep an index map.

    Args:
        text: Input text.

    Returns:
        Tuple of (normalized string, list mapping normalized index -> raw index).
    """
    chars: List[str] = []
    index_map: List[int] = []
    for raw_index, char in enumerate(text):
        if char.isspace() or not (char.isalnum() or char.isalpha()):
            continue
        chars.append(char.lower())
        index_map.append(raw_index)
    return "".join(chars), index_map


def find_exact_overlap(previous: str, new: str, max_overlap_chars: int = 40) -> int:
    """
    Longest k where ``previous[-k:]`` equals ``new[:k]`` (character exact).

    Args:
        previous: Running transcript (already emitted text).
        new: Newly recognised text of the current window.
        max_overlap_chars: Upper bound for the overlap length.

    Returns:
        Overlap length in characters (0 when there is no overlap).
    """
    limit = min(len(previous), len(new), max(0, int(max_overlap_chars)))
    for k in range(limit, 0, -1):
        if previous[-k:] == new[:k]:
            return k
    return 0


def find_lcs_overlap(
    previous: str,
    new: str,
    max_overlap_chars: int = 24,
    min_overlap: int = 4,
    max_edits: int = 1,
) -> Tuple[int, int]:
    """
    Largest k where the LCS of ``previous[-k:]`` and ``new[:k]`` is ``k - max_edits``.

    This absorbs the single-character jitter Whisper introduces at a sliding
    window boundary (homophones, dropped/added particles) which would otherwise
    defeat the exact suffix/prefix match.

    Args:
        previous: Running transcript.
        new: Newly recognised text.
        max_overlap_chars: Upper bound for the overlap length.
        min_overlap: Overlaps shorter than this are rejected as coincidence.
        max_edits: Tolerated mismatching characters inside the overlap.

    Returns:
        Tuple ``(k, lcs)``; ``k`` is 0 when no robust overlap exists.
    """
    limit = min(len(previous), len(new), max(0, int(max_overlap_chars)))
    for k in range(limit, min_overlap - 1, -1):
        score = lcs_length(previous[-k:], new[:k])
        if score >= k - max_edits:
            return k, score
    return 0, 0


def align_overlap_text(
    previous: str,
    new: str,
    max_overlap_chars: int = 40,
    normalize_punctuation: bool = True,
    use_lcs: bool = True,
) -> str:
    """
    Merge two consecutive transcripts, removing the boundary duplication.

    Example::

        align_overlap_text("今天天气", "天气真好")     -> "今天天气真好"
        align_overlap_text("今天天气", "今天天气真好") -> "今天天气真好"

    Args:
        previous: Running transcript (text already shown to the user).
        new: Newly recognised (sliding window) transcript.
        max_overlap_chars: Upper bound for the overlap length in characters.
        normalize_punctuation: Ignore punctuation/space drift inside the overlap.
        use_lcs: Enable the LCS fallback for jitter inside the overlap.

    Returns:
        The merged transcript. Content is never dropped: when no overlap can be
        identified the two strings are simply concatenated.
    """
    prev = (previous or "").strip()
    cur = (new or "").strip()
    if not prev:
        return cur
    if not cur:
        return prev
    if prev == cur or prev in cur:
        # The new window already contains everything recognised so far.
        return cur
    if cur in prev:
        # Regression (partial re-recognition): keep the longer transcript.
        return prev

    # 1) Exact suffix/prefix overlap -- the common case
    k_exact = find_exact_overlap(prev, cur, max_overlap_chars)
    if k_exact > 0:
        return prev + cur[k_exact:]

    # 2) Punctuation / whitespace tolerant overlap
    if normalize_punctuation:
        prev_norm, _ = _normalize_with_map(prev)
        cur_norm, cur_map = _normalize_with_map(cur)
        if prev_norm and cur_norm:
            if cur_norm == prev_norm:
                return cur
            if cur_norm in prev_norm:
                return prev
            if cur_norm.startswith(prev_norm):
                # New window is a superset: keep prev's exact rendering
                return prev + cur[cur_map[len(prev_norm)]:]
            if prev_norm in cur_norm:
                return cur
            k_norm = find_exact_overlap(prev_norm, cur_norm, max_overlap_chars)
            if 0 < k_norm < len(cur_map):
                return prev + cur[cur_map[k_norm]:]

    # 3) LCS fallback (single-character recognition jitter)
    if use_lcs:
        k_lcs, _ = find_lcs_overlap(prev, cur, max_overlap_chars)
        if k_lcs > 0:
            return prev + cur[k_lcs:]

    # 4) No overlap detected: plain concatenation (never drop content)
    return prev + cur


class OverlapAligner:
    """
    Stateful dynamic overlap-add (LCS) aligner for sliding-window transcripts.

    Keeps the running transcript of the current utterance and merges every new
    window result into it, so a word never appears twice at the window
    boundary::

        aligner = OverlapAligner()
        aligner.merge("今天天气")      # -> "今天天气"
        aligner.merge("天气真好")      # -> "今天天气真好"

    Args:
        max_overlap_chars: Upper bound for the removable overlap (characters).
        normalize_punctuation: Ignore punctuation/space drift in the overlap.
        use_lcs: Enable the LCS jitter-tolerant fallback.
    """

    def __init__(
        self,
        max_overlap_chars: int = 40,
        normalize_punctuation: bool = True,
        use_lcs: bool = True,
    ) -> None:
        self.max_overlap_chars = int(max_overlap_chars)
        self.normalize_punctuation = bool(normalize_punctuation)
        self.use_lcs = bool(use_lcs)
        self._text = ""

    @property
    def text(self) -> str:
        """Running (merged) transcript."""
        return self._text

    def reset(self) -> None:
        """Forget the running transcript (start of a new utterance)."""
        self._text = ""

    def set_text(self, text: str) -> None:
        """Replace the running transcript (e.g. when re-syncing with a final)."""
        self._text = (text or "").strip()

    def merge(self, text: str) -> str:
        """
        Merge a new window transcript into the running transcript.

        Args:
            text: Text recognised for the current sliding window.

        Returns:
            The merged transcript.
        """
        self._text = align_overlap_text(
            self._text,
            text,
            self.max_overlap_chars,
            normalize_punctuation=self.normalize_punctuation,
            use_lcs=self.use_lcs,
        )
        return self._text


# ---------------------------------------------------------------------------
# Whisper hallucination suppression
# ---------------------------------------------------------------------------

# Classic artefacts Whisper emits when fed silence, noise or the tail of a
# media file: subtitle credits, transcript watermarks, "thanks for watching"
# loops and channel intros. Stored lower-case for case-insensitive matching.
HALLUCINATION_PHRASES: Tuple[str, ...] = (
    # Chinese subtitle / video credits
    "字幕由", "字幕组", "字幕志愿者", "中文字幕", "字幕制作", "翻译：", "校对：",
    "字幕提供", "感谢观看", "感谢您的观看", "谢谢观看", "谢谢大家观看",
    "请订阅", "请点赞", "请关注", "不吝点赞", "订阅频道", "点赞订阅",
    "明镜与点点", "栏目", "下期再见", "大家好，欢迎收看",
    # English / international
    "subtitles by", "subtitle", "amara.org", "www.", "http://", "https://",
    "thanks for watching", "thank you for watching", "please subscribe",
    "please like and subscribe", "subscribe to my channel", "transcription by",
    "audio by", "copyright", "all rights reserved",
)

# Text made only of whitespace / punctuation / music notes is not speech.
_PUNCTUATION_ONLY_RE = re.compile(r"^[\s\W_♪♫♩~～]+$", re.UNICODE)

# Upper bound for the "whole output is a known artefact" check.
_HALLUCINATION_MAX_LEN = 24

# Fraction of a short transcript that must be covered by artefact phrases
# before the whole transcript is discarded (instead of just the suffix).
_HALLUCINATION_COVERAGE = 0.6


def is_hallucination(text: str) -> bool:
    """
    Decide whether a transcript is entirely a known Whisper artefact.

    A short text counts as an artefact when only punctuation/music notes remain,
    or when known artefact phrases cover most of it -- so a real sentence that
    merely ends with an artefact (e.g. "今天天气不错感谢观看") is *not* dropped.

    Args:
        text: Raw transcript.

    Returns:
        True when the text is empty, punctuation-only, or essentially made of a
        known artefact phrase (e.g. "感谢观看").
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    if _PUNCTUATION_ONLY_RE.match(stripped):
        return True
    if len(stripped) > _HALLUCINATION_MAX_LEN:
        return False

    lowered = stripped.lower()
    matched = 0
    for phrase in HALLUCINATION_PHRASES:
        if phrase in lowered:
            matched += len(phrase)
    return matched >= _HALLUCINATION_COVERAGE * len(stripped)


def filter_hallucinations(text: str) -> str:
    """
    Remove known Whisper hallucination artefacts.

    The whole transcript is dropped when it is an artefact; otherwise artefacts
    are stripped from the *end* of the text only (never mid-sentence, so real
    content is preserved).

    Args:
        text: Raw transcript.

    Returns:
        Cleaned transcript (possibly an empty string).
    """
    stripped = (text or "").strip()
    if not stripped or is_hallucination(stripped):
        return ""

    cleaned = stripped
    for _ in range(4):
        lowered = cleaned.lower()
        removed = False
        for phrase in HALLUCINATION_PHRASES:
            if lowered.endswith(phrase):
                cleaned = cleaned[: len(cleaned) - len(phrase)]
                cleaned = cleaned.rstrip(" \t，,。.、！!？?；;：:")
                removed = True
        if not removed:
            break
    return cleaned.strip()


def truncate_repetitions(
    text: str,
    max_period: int = 8,
    min_repeats: int = 3,
    keep: int = 1,
) -> str:
    """
    Truncate pathological n-gram loops produced by Whisper on silence.

    Detects a token block (period ``p <= max_period``) repeated consecutively
    ``min_repeats`` times or more -- e.g. ``"谢谢大家谢谢大家谢谢大家"`` -- and
    keeps only the first ``keep`` repetitions. Single-token periods are ignored
    so natural interjections such as ``"哈哈哈"`` survive.

    Args:
        text: Transcript to clean.
        max_period: Longest repeating block (in tokens) to look for.
        min_repeats: Number of consecutive repetitions considered pathological.
        keep: How many repetitions to keep when truncating.

    Returns:
        Transcript with abnormal repetition removed.
    """
    if not text:
        return text
    tokens = tokenize_text(text)
    if len(tokens) < max(4, int(min_repeats) * 2):
        return text

    max_period = max(2, int(max_period))
    min_repeats = max(2, int(min_repeats))
    keep = max(1, int(keep))

    span: Optional[Tuple[int, int, int]] = None  # (start, period, repeats)
    total = len(tokens)

    for period in range(2, max_period + 1):
        run_start: Optional[int] = None
        index = period
        while index <= total:
            if index < total and tokens[index] == tokens[index - period]:
                if run_start is None:
                    run_start = index - period
            elif run_start is not None:
                span = _better_loop(span, run_start, index - 1, period, min_repeats)
                run_start = None
            index += 1
        if run_start is not None:
            span = _better_loop(span, run_start, total - 1, period, min_repeats)

    if span is None:
        return text

    start, period, repeats = span
    logger.debug(
        "Truncating repeated n-gram loop: period=%d, repeats=%d at token %d",
        period, repeats, start,
    )
    kept = tokens[: start + period * keep] + tokens[start + period * repeats:]
    return "".join(kept)


def _better_loop(
    current: Optional[Tuple[int, int, int]],
    run_start: int,
    run_end: int,
    period: int,
    min_repeats: int,
) -> Optional[Tuple[int, int, int]]:
    """
    Keep the longest qualifying repetition loop found so far.

    Args:
        current: Best loop so far as ``(start, period, repeats)``.
        run_start: Token index where the periodic run starts.
        run_end: Token index where the periodic run ends (inclusive).
        period: Candidate period in tokens.
        min_repeats: Minimum number of repetitions to qualify.

    Returns:
        The (possibly updated) best loop, or ``current`` when the candidate does
        not qualify or is shorter than the current best.
    """
    span_len = run_end - run_start + 1
    repeats = span_len // period
    if repeats < min_repeats or span_len < period * min_repeats:
        return current
    if current is None or repeats * period > current[2] * current[1]:
        return (run_start, period, repeats)
    return current


def clean_transcript(
    text: str,
    max_period: int = 8,
    min_repeats: int = 3,
    keep: int = 1,
) -> str:
    """
    Full hallucination-suppression chain (filter -> loop truncation -> filter).

    Args:
        text: Raw transcript.
        max_period: Longest repeating block (tokens) to detect.
        min_repeats: Repetitions that mark a loop as pathological.
        keep: Repetitions kept when truncating a loop.

    Returns:
        Cleaned transcript (empty string when everything was an artefact).
    """
    cleaned = filter_hallucinations(text)
    if not cleaned:
        return ""
    cleaned = truncate_repetitions(
        cleaned, max_period=max_period, min_repeats=min_repeats, keep=keep
    )
    return filter_hallucinations(cleaned)


class TranscriptFilter:
    """
    Configurable hallucination-suppression chain for streaming ASR output.

    Applies, in order:

        1. Known-artefact filtering (subtitle credits, "thanks for watching", ...).
        2. Punctuation-only / empty rejection.
        3. Pathological n-gram loop truncation.

    Args:
        config: ``StreamASRConfig`` providing the filter parameters.
    """

    def __init__(self, config: Optional[StreamASRConfig] = None) -> None:
        cfg = config or StreamASRConfig()
        self.enabled = bool(cfg.enable_hallucination_filter)
        self.max_period = int(cfg.max_ngram_size)
        self.min_repeats = int(cfg.min_repeat_count)
        self.keep = int(cfg.repeat_keep)

    def process(self, text: str) -> str:
        """
        Clean a raw transcript.

        Args:
            text: Raw transcript from Whisper.

        Returns:
            Cleaned transcript (empty string when filtered out entirely).
        """
        raw = (text or "").strip()
        if not raw:
            return ""
        if not self.enabled:
            return raw
        return clean_transcript(
            raw,
            max_period=self.max_period,
            min_repeats=self.min_repeats,
            keep=self.keep,
        )


class StreamASR:
    """
    Real-time streaming ASR engine.

    Implements sliding window + incremental recognition:
    - Feeds audio chunks into an internal buffer
    - Every step_duration seconds, runs whisper on the sliding window
    - Compares new partial result with previous to emit incremental updates
    - Detects silence to finalize segments

    Callbacks:
        on_partial_result(text: str): Called with partial (in-progress) text.
        on_final_result(result: StreamASRResult): Called when a segment is finalized.

    Usage:
        asr = StreamASR(language="zh")
        asr.on_partial_result = lambda t: print(f"Partial: {t}")
        asr.on_final_result = lambda r: print(f"Final: {r.text}")

        asr.start()
        asr.feed(audio_chunk_1)
        asr.feed(audio_chunk_2)
        # ...
        asr.stop()
    """

    def __init__(
        self,
        config: Optional[StreamASRConfig] = None,
        whisper_bridge: Optional[WhisperBridge] = None,
        language: Optional[str] = None,
        whisper_model: Optional[str] = None,
        backend: Optional[WhisperBackend] = None,
    ):
        """
        Initialize StreamASR.

        Args:
            config: Full configuration object (overrides individual params).
            whisper_bridge: Pre-configured WhisperBridge instance.
            language: Language code (overrides config).
            whisper_model: Model size (overrides config).
            backend: Whisper backend (overrides config).
        """
        self.config = config or StreamASRConfig()

        # Override config with explicit params
        if language is not None:
            self.config.language = language
        if whisper_model is not None:
            self.config.whisper_model = whisper_model
        if backend is not None:
            self.config.backend = backend

        # Whisper bridge
        self._bridge = whisper_bridge or WhisperBridge(
            backend=self.config.backend,
            whisper_model=self.config.whisper_model,
            language=self.config.language,
        )

        # Internal audio buffer
        self._buffer = np.array([], dtype=np.float32)
        self._buffer_lock = threading.Lock()

        # Recognition state
        self._is_running = False
        self._last_text = ""
        self._segment_start_time = 0.0
        self._total_audio_time = 0.0
        self._step_samples = int(self.config.step_duration * self.config.sample_rate)
        self._window_samples = int(self.config.window_duration * self.config.sample_rate)
        self._min_samples = int(self.config.min_audio_duration * self.config.sample_rate)
        self._silence_start: Optional[float] = None
        self._last_feed_time: Optional[float] = None

        # Hallucination suppression + dynamic overlap-add alignment
        self._filter = TranscriptFilter(self.config)
        self._aligner = OverlapAligner(
            max_overlap_chars=self.config.max_overlap_chars,
            normalize_punctuation=self.config.normalize_punctuation,
            use_lcs=self.config.use_lcs_alignment,
        )
        self._partial_text = ""

        # Callbacks
        self.on_partial_result: Optional[Callable[[str], None]] = None
        self.on_final_result: Optional[Callable[[StreamASRResult], None]] = None

        # Worker thread
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        logger.info(
            "StreamASR initialized: window=%ss, step=%ss, model=%s, language=%s",
            self.config.window_duration,
            self.config.step_duration,
            self.config.whisper_model,
            self.config.language,
        )

    @property
    def is_running(self) -> bool:
        """Whether the ASR engine is actively processing."""
        return self._is_running

    @property
    def buffer_duration(self) -> float:
        """Current buffer duration in seconds."""
        with self._buffer_lock:
            return len(self._buffer) / self.config.sample_rate

    def start(self):
        """
        Start the streaming ASR engine.

        Begins the internal processing loop that periodically checks
        the audio buffer and runs recognition.
        """
        if self._is_running:
            logger.warning("StreamASR is already running")
            return

        self._is_running = True
        self._stop_event.clear()
        self._last_text = ""
        self._total_audio_time = 0.0
        self._silence_start = None
        self._last_feed_time = time.time()

        # Reset alignment / hallucination state for the new session
        self._aligner.reset()
        self._partial_text = ""

        # Start worker thread
        self._worker_thread = threading.Thread(
            target=self._processing_loop,
            name="StreamASR-Worker",
            daemon=True,
        )
        self._worker_thread.start()

        logger.info("StreamASR started")

    def stop(self) -> Optional[StreamASRResult]:
        """
        Stop the streaming ASR engine.

        Returns:
            Final result if there's remaining audio in the buffer.
        """
        if not self._is_running:
            logger.warning("StreamASR is not running")
            return None

        self._is_running = False
        self._stop_event.set()

        # Wait for worker to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        # Finalize remaining audio
        final_result = self._finalize_segment()

        logger.info("StreamASR stopped")
        return final_result

    def feed(self, audio_chunk: np.ndarray):
        """
        Feed an audio chunk into the ASR engine.

        Args:
            audio_chunk: Float32 audio array. Will be converted to mono
                         if multi-channel. Sample rate should match config.

        Raises:
            RuntimeError: If ASR engine is not started.
        """
        if not self._is_running:
            raise RuntimeError("StreamASR is not running. Call start() first.")

        # Ensure float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Flatten to mono if needed
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)

        with self._buffer_lock:
            self._buffer = np.concatenate([self._buffer, audio_chunk])

            # Cap buffer to max 2x window to prevent unbounded growth
            max_samples = self._window_samples * 2
            if len(self._buffer) > max_samples:
                # Keep only the most recent max_samples
                self._buffer = self._buffer[-max_samples:]

        self._last_feed_time = time.time()

    def _processing_loop(self):
        """Main processing loop running in worker thread."""
        logger.debug("Processing loop started")

        while not self._stop_event.is_set():
            try:
                # Sleep for step duration
                self._stop_event.wait(timeout=self.config.step_duration)

                if self._stop_event.is_set():
                    break

                # Check if we have enough audio
                with self._buffer_lock:
                    buffer_len = len(self._buffer)

                if buffer_len < self._min_samples:
                    continue

                # Run recognition on sliding window
                self._recognize_step()

            except (RuntimeError, OSError, ValueError) as e:
                logger.error("Error in processing loop: %s", e, exc_info=True)
                # Continue processing despite errors
                continue

        logger.debug("Processing loop ended")

    def _recognize_step(self):
        """
        Run one recognition step on the current sliding window.

        Order of operations (each stage suppresses a class of hallucination):

        1. Extract the most recent ``window_duration`` seconds of audio.
        2. **Energy gate** -- silence is never sent to Whisper.
        3. Whisper recognition.
        4. **Hallucination filter** -- artefact phrases, punctuation-only output
           and pathological n-gram loops are removed.
        5. **Dynamic overlap alignment** -- the window transcript is merged into
           the running utterance transcript (LCS overlap-add) and emitted as a
           partial result when it changed.
        """
        with self._buffer_lock:
            buffer_len = len(self._buffer)
            if buffer_len == 0:
                return

            # Extract sliding window (most recent window_duration seconds)
            window_samples = min(self._window_samples, buffer_len)
            audio_window = self._buffer[-window_samples:].copy()

        # --- Energy gate: silence is the main hallucination source ---
        energy = self._compute_energy(audio_window)
        if energy < self.config.min_speech_energy:
            logger.debug(
                "Skipping recognition (window energy %.5f < %.5f)",
                energy, self.config.min_speech_energy,
            )
            self._check_silence()
            return

        # Run whisper recognition
        try:
            result = self._bridge.transcribe(
                audio_window,
                sample_rate=self.config.sample_rate,
            )
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning("Recognition failed: %s", e)
            return

        # --- Hallucination suppression ---
        raw_text = (result.text or "").strip()
        text = self._filter.process(raw_text)

        if not text:
            # Empty result 锟?might be silence
            self._check_silence()
            return

        # Real content recognised: reset silence detection
        self._silence_start = None

        # --- Dynamic overlap-add alignment with the previous window ---
        if self.config.enable_overlap_alignment:
            new_text = self._aligner.merge(text)
        else:
            new_text = text
        self._partial_text = new_text

        # Compare with previous text to determine incremental update
        if new_text != self._last_text:
            self._last_text = new_text

            # Emit partial result
            if self.on_partial_result:
                try:
                    self.on_partial_result(new_text)
                except (RuntimeError, ValueError) as e:
                    logger.error("Error in on_partial_result callback: %s", e)

            logger.debug("Partial: %s", new_text)

        # Finalize when the utterance transcript outgrows the safety cap
        if len(new_text) >= self.config.max_transcript_chars:
            logger.info(
                "Transcript reached %d chars, finalizing segment",
                len(new_text),
            )
            self._finalize_segment()

    @staticmethod
    def _compute_energy(audio: np.ndarray) -> float:
        """
        Compute the RMS energy of an audio window.

        Args:
            audio: Audio samples (float32).

        Returns:
            RMS energy (0.0 for empty input).
        """
        if audio is None or len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    def _check_silence(self):
        """Check for silence and finalize segment if silence timeout reached."""
        now = time.time()

        if self._silence_start is None:
            self._silence_start = now
            return

        silence_duration = now - self._silence_start

        if silence_duration >= self.config.silence_timeout:
            # Silence timeout reached 锟?finalize current segment
            self._finalize_segment()
            self._silence_start = None

    def _finalize_segment(self) -> Optional[StreamASRResult]:
        """
        Finalize the current audio segment.

        Runs a final recognition on the buffered audio, emits
        on_final_result callback, and clears the buffer.

        Returns:
            StreamASRResult or None if buffer is empty.
        """
        with self._buffer_lock:
            if len(self._buffer) == 0:
                return None

            audio_data = self._buffer.copy()
            self._buffer = np.array([], dtype=np.float32)

        audio_duration = len(audio_data) / self.config.sample_rate
        segment_start = self._total_audio_time
        self._total_audio_time += audio_duration

        # Energy gate on the final pass too: silence must never be decoded,
        # otherwise Whisper invents credits/loops for it.
        energy = self._compute_energy(audio_data)
        if energy < self.config.min_speech_energy:
            logger.debug(
                "Final segment is silence (energy %.5f < %.5f), not decoding",
                energy, self.config.min_speech_energy,
            )
            self._reset_utterance_state()
            return None

        # Run final recognition
        try:
            result = self._bridge.transcribe(
                audio_data,
                sample_rate=self.config.sample_rate,
            )
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning("Final recognition failed: %s", e)
            return None

        if not result.text.strip():
            return None

        # Hallucination suppression on the final transcript as well
        final_text = self._filter.process(result.text)
        if not final_text:
            logger.debug("Final transcript filtered out (hallucination/silence)")
            self._reset_utterance_state()
            return None

        # Merge with the streaming (aligned) transcript to avoid losing the
        # head of the utterance when the ring buffer has slid forward.
        if self.config.enable_overlap_alignment and self._partial_text:
            final_text = align_overlap_text(
                self._partial_text, final_text, self.config.max_overlap_chars
            )

        asr_result = StreamASRResult(
            text=final_text,
            is_final=True,
            timestamp=time.time(),
            start_time=segment_start,
            end_time=self._total_audio_time,
            confidence=result.confidence,
            language=result.language,
        )

        self._reset_utterance_state()

        # Emit final result
        if self.on_final_result:
            try:
                self.on_final_result(asr_result)
            except (RuntimeError, ValueError) as e:
                logger.error("Error in on_final_result callback: %s", e)

        logger.info(
            "Final segment [%.1fs - %.1fs]: %s...",
            segment_start, self._total_audio_time, asr_result.text[:80],
        )

        return asr_result

    def _reset_utterance_state(self) -> None:
        """Clear per-utterance alignment/partial state (thread safe)."""
        self._aligner.reset()
        self._partial_text = ""
        self._last_text = ""

    def flush(self) -> None:
        """
        Drop buffered audio and the in-progress transcript.

        Intended for barge-in handling: when the system playback is interrupted,
        both the pending partial text and the audio buffer of the interrupted
        turn must be discarded so they cannot leak into the next utterance.
        """
        with self._buffer_lock:
            self._buffer = np.array([], dtype=np.float32)
        self._reset_utterance_state()
        self._silence_start = None
        logger.debug("StreamASR flushed")

    def get_status(self) -> dict:
        """
        Get current ASR engine status.

        Returns:
            Dictionary with status information.
        """
        return {
            "is_running": self._is_running,
            "buffer_duration": self.buffer_duration,
            "total_audio_time": self._total_audio_time,
            "config": {
                "window_duration": self.config.window_duration,
                "step_duration": self.config.step_duration,
                "language": self.config.language,
                "whisper_model": self.config.whisper_model,
                "sample_rate": self.config.sample_rate,
                "overlap_alignment": self.config.enable_overlap_alignment,
                "hallucination_filter": self.config.enable_hallucination_filter,
            },
            "last_text": self._last_text[:100] if self._last_text else "",
        }