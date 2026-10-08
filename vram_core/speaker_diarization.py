"""
Speaker Diarization Module for vram_core
=========================================

Multi-backend speaker diarization 锟?identifies "who spoke when":

1. **pyannote-audio** (preferred): State-of-the-art neural diarization
   - Requires: pip install pyannote-audio
   - Requires: HuggingFace token (free, set via HF_TOKEN env var)
   - Features: Real-time separation, auto speaker count, speaker profiles
   - Output: Timestamped speaker labels with confidence

2. **MFCC + Cosine Similarity** (fallback): Lightweight feature-based
   - No external dependencies beyond numpy/scipy
   - Features: MFCC embeddings, cosine similarity clustering

Usage:
    from vram_core.speaker_diarization import SpeakerDiarizer

    # Auto-detect best backend
    diarizer = SpeakerDiarizer()
    segments = diarizer.diarize(audio_array, sample_rate=16000)
    for seg in segments:
        print(f"[{seg.start_time:.1f}s-{seg.end_time:.1f}s] {seg.speaker_id}")

    # Force pyannote backend
    diarizer = SpeakerDiarizer(backend="pyannote", hf_token="hf_xxx")

    # Force MFCC fallback
    diarizer = SpeakerDiarizer(backend="mfcc")
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import stft

from vram_core.utils import ensure_float32, merge_adjacent_events

logger = logging.getLogger(__name__)


# 鈹€鈹€ Backend Detection 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_PYANNOTE_AVAILABLE = False
try:
    from pyannote.audio import Pipeline as PyannotePipeline
    _PYANNOTE_AVAILABLE = True
    logger.info("pyannote-audio detected — neural diarization backend available")
except (ImportError, OSError):
    logger.info(
        "pyannote-audio not available, using MFCC fallback. "
        "Install with: pip install pyannote-audio"
    )


# 鈹€鈹€ Data Classes 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
@dataclass
class SpeakerSegment:
    """A diarized audio segment with speaker identity."""
    start_time: float
    end_time: float
    speaker_id: str
    audio: Optional[np.ndarray] = None
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        return (
            f"SpeakerSegment(speaker='{self.speaker_id}', "
            f"{self.start_time:.2f}s-{self.end_time:.2f}s, "
            f"conf={self.confidence:.3f})"
        )


@dataclass
class SpeakerProfile:
    """Stored profile for an identified speaker."""
    speaker_id: str
    embedding: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    total_duration: float = 0.0
    segment_count: int = 0


@dataclass
class DiarizationResult:
    """Full diarization result with metadata."""
    segments: List[SpeakerSegment]
    speaker_count: int
    total_duration: float
    backend_used: str = "unknown"


# 鈹€鈹€ pyannote Backend 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class BaseVoiceprintExtractor(ABC):
    """
    Pluggable voiceprint embedding interface (v2.6.0).

    Any speaker-embedding front-end can be attached to the diarization /
    verification pipeline through this interface, as long as it maps a mono
    audio buffer to a fixed-length, L2-normalisable vector::

        class MyExtractor(BaseVoiceprintExtractor):
            @property
            def dim(self): return 192
            def embed(self, audio, sample_rate=16000): ...

    Built-in implementations: :class:`MFCCExtractor` (pure NumPy, always
    available) and :class:`ONNXEmbeddingExtractor` (ECAPA-TDNN / CAM++ style
    ONNX graphs, with graceful fallback to MFCC).
    """

    #: Human readable identifier, used in logs and diagnostics
    name: str = "base"

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the produced embedding vector."""

    @abstractmethod
    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Compute a speaker embedding for a mono audio buffer.

        Args:
            audio: Audio samples (float32 preferred, int16 accepted).
            sample_rate: Sample rate in Hz.

        Returns:
            float32 embedding vector of length :attr:`dim`.
        """

    @staticmethod
    def l2_normalize(vector: np.ndarray) -> np.ndarray:
        """Return the L2-normalised copy of ``vector`` (zeros stay zeros)."""
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-10:
            vector = vector / norm
        return vector.astype(np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors (0.0 for degenerate input)."""
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"{type(self).__name__}(dim={self.dim})"


class MFCCExtractor(BaseVoiceprintExtractor):
    """
    Lightweight pure-NumPy MFCC voiceprint extractor.

    Pipeline: STFT -> power spectrum -> mel filterbank -> log -> DCT-II ->
    per-frame energy normalisation -> (mean[, std]) statistics -> L2 norm.

    The mel filterbank and the DCT basis are cached per shape, so repeated calls
    on short analysis segments no longer rebuild the full ``(n_mels, n_bins)``
    matrix on every call.
    """

    name = "mfcc"

    def __init__(
        self,
        n_mfcc: int = 13,
        frame_length: int = 512,
        hop_length: int = 256,
        sample_rate: int = 16000,
        n_mels: int = 26,
        include_std: bool = True,
        energy_normalize: bool = True,
    ):
        self.n_mfcc = int(n_mfcc)
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.include_std = bool(include_std)
        self.energy_normalize = bool(energy_normalize)
        self._filterbank_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._dct_cache: Dict[Tuple[int, int], np.ndarray] = {}

    @property
    def dim(self) -> int:
        """``2 * n_mfcc`` with the std statistics, ``n_mfcc`` without."""
        return self.n_mfcc * 2 if self.include_std else self.n_mfcc

    # ── Front-end ─────────────────────────────────────────────────────────
    def _mel_filterbank(self, n_filters: int, n_fft: int, sample_rate: int) -> np.ndarray:
        """Cached mel-spaced triangular filterbank (``n_fft`` = bin count)."""
        key = (n_filters, n_fft)
        cached = self._filterbank_cache.get(key)
        if cached is not None:
            return cached

        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_points = np.linspace(
            hz_to_mel(0.0), hz_to_mel(sample_rate / 2.0), n_filters + 2
        )
        hz_points = mel_to_hz(mel_points)
        freq_bins = np.fft.rfftfreq(n_fft * 2 - 1, d=1.0 / sample_rate)
        n_freq_bins = len(freq_bins)

        filterbank = np.zeros((n_filters, n_freq_bins), dtype=np.float32)
        for i in range(n_filters):
            f_low, f_center, f_high = hz_points[i], hz_points[i + 1], hz_points[i + 2]
            for j, freq in enumerate(freq_bins):
                if f_low <= freq <= f_center and f_center > f_low:
                    filterbank[i, j] = (freq - f_low) / (f_center - f_low)
                elif f_center < freq <= f_high and f_high > f_center:
                    filterbank[i, j] = (f_high - freq) / (f_high - f_center)

        self._filterbank_cache[key] = filterbank
        return filterbank

    def _dct(self, x: np.ndarray, n_coeffs: int) -> np.ndarray:
        """Cached Type-II DCT basis applied to ``x`` (rows = features)."""
        n_features = x.shape[0]
        n = min(n_coeffs, n_features)
        key = (n, n_features)
        basis = self._dct_cache.get(key)
        if basis is None:
            k = np.arange(n).reshape(-1, 1)
            n_idx = np.arange(n_features).reshape(1, -1)
            basis = np.cos(np.pi * k * (2 * n_idx + 1) / (2 * n_features))
            self._dct_cache[key] = basis
        return (basis @ x).astype(np.float32)

    def extract_mfcc(
        self, audio: np.ndarray, sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """Extract MFCC features ``(n_mfcc, n_frames)`` from a mono buffer."""
        sample_rate = int(sample_rate or self.sample_rate)
        if len(audio) == 0:
            return np.zeros((self.n_mfcc, 0), dtype=np.float32)

        audio = ensure_float32(np.asarray(audio).reshape(-1))
        min_len = self.frame_length * 2
        if len(audio) < min_len:
            audio = np.pad(audio, (0, min_len - len(audio)), mode="constant")

        _freqs, _times, Zxx = stft(
            audio, fs=sample_rate, nperseg=self.frame_length,
            noverlap=self.frame_length - self.hop_length,
        )
        power = np.abs(Zxx) ** 2
        mel_fb = self._mel_filterbank(self.n_mels, power.shape[0], sample_rate)
        mel_spectrum = mel_fb @ power
        log_mel = np.log(mel_spectrum + 1e-10)
        mfcc = self._dct(log_mel, self.n_mfcc)

        # Energy normalisation: remove the per-frame offset so that loudness
        # differences do not dominate speaker similarity.
        if self.energy_normalize and mfcc.shape[1] > 0:
            mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
        return mfcc.astype(np.float32)

    # ── Embedding ─────────────────────────────────────────────────────────
    def embed_from_mfcc(self, mfcc: np.ndarray) -> np.ndarray:
        """(mean[, std]) statistics of an MFCC matrix, L2 normalised."""
        mfcc = np.asarray(mfcc, dtype=np.float32)
        if mfcc.ndim != 2 or mfcc.shape[1] == 0:
            return np.zeros(self.dim, dtype=np.float32)
        features = [np.mean(mfcc, axis=1)]
        if self.include_std:
            features.append(np.std(mfcc, axis=1))
        return self.l2_normalize(np.concatenate(features))

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """MFCC-statistics voiceprint embedding."""
        return self.embed_from_mfcc(self.extract_mfcc(audio, sample_rate))


class ONNXEmbeddingExtractor(BaseVoiceprintExtractor):
    """
    Pluggable ONNX voiceprint front-end (ECAPA-TDNN / CAM++ / x-vector ...).

    The graph is treated as a black box: input tensor name, output tensor name
    and the expected number of samples are auto-detected from the model metadata
    and can all be overridden, so any exported speaker-embedding network can be
    attached without touching the diarization pipeline::

        diarizer = SpeakerDiarizer(
            embedding_extractor=ONNXEmbeddingExtractor("ecapa_tdnn.onnx"),
        )

    Degradation strategy: when ``onnxruntime`` is missing, the model file does
    not exist or inference fails, the extractor transparently falls back to
    :class:`MFCCExtractor` (``is_fallback`` is then True) so that CPU-only or
    dependency-free deployments keep working.
    """

    name = "onnx"

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        sample_rate: int = 16000,
        fallback_n_mfcc: int = 13,
        normalize: bool = True,
        providers: Optional[List[str]] = None,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.sample_rate = int(sample_rate)
        self.normalize = bool(normalize)
        self._fallback = MFCCExtractor(n_mfcc=fallback_n_mfcc, sample_rate=sample_rate)
        self._session = None
        self._input_name = input_name
        self._output_name = output_name
        self._expected_samples: Optional[int] = None
        self._dim: Optional[int] = None
        self.is_fallback = True
        self._load(providers)

    def _load(self, providers: Optional[List[str]]) -> None:
        """Try to load the ONNX session; keep the MFCC fallback on any failure."""
        if self.model_path is None:
            logger.info("ONNXEmbeddingExtractor: no model path, using MFCC fallback")
            return
        if not self.model_path.exists():
            logger.warning(
                "ONNXEmbeddingExtractor: model %s not found, using MFCC fallback",
                self.model_path,
            )
            return
        try:
            import onnxruntime
        except ImportError:
            logger.info(
                "ONNXEmbeddingExtractor: onnxruntime not installed "
                "(pip install onnxruntime), using MFCC fallback"
            )
            return

        try:
            session = onnxruntime.InferenceSession(
                str(self.model_path), providers=providers or None
            )
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            self._input_name = self._input_name or inputs[0].name
            self._output_name = self._output_name or outputs[0].name

            in_shape = list(inputs[0].shape)
            if in_shape and isinstance(in_shape[-1], int) and in_shape[-1] > 0:
                self._expected_samples = int(in_shape[-1])

            out_shape = list(outputs[0].shape)
            if out_shape and isinstance(out_shape[-1], int) and out_shape[-1] > 0:
                self._dim = int(out_shape[-1])

            self._session = session
            self.is_fallback = False
            logger.info(
                "ONNXEmbeddingExtractor loaded %s (input=%s, output=%s, samples=%s)",
                self.model_path.name, self._input_name, self._output_name,
                self._expected_samples,
            )
        except Exception as error:  # noqa: BLE001 - any backend failure -> fallback
            logger.warning(
                "ONNXEmbeddingExtractor: failed to load %s (%s), using MFCC fallback",
                self.model_path, error,
            )

    @property
    def dim(self) -> int:
        """Embedding dimensionality (MFCC dimension while in fallback mode)."""
        if self._dim is not None:
            return int(self._dim)
        return self._fallback.dim

    @property
    def expected_samples(self) -> Optional[int]:
        """Statically known input length of the ONNX graph (None when dynamic)."""
        return self._expected_samples

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """ONNX speaker embedding with transparent MFCC fallback."""
        if self._session is None:
            return self._fallback.embed(audio, sample_rate)

        try:
            samples = ensure_float32(np.asarray(audio).reshape(-1))
            if self._expected_samples and len(samples) != self._expected_samples:
                if len(samples) > self._expected_samples:
                    samples = samples[: self._expected_samples]
                else:
                    samples = np.pad(
                        samples,
                        (0, self._expected_samples - len(samples)),
                        mode="constant",
                    )
            batch = samples[np.newaxis, :].astype(np.float32)
            outputs = self._session.run([self._output_name], {self._input_name: batch})
            embedding = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            return self.l2_normalize(embedding) if self.normalize else embedding
        except Exception as error:  # noqa: BLE001 - runtime failure -> fallback
            logger.warning(
                "ONNXEmbeddingExtractor inference failed (%s), using MFCC", error
            )
            self._session = None
            self.is_fallback = True
            return self._fallback.embed(audio, sample_rate)


def create_voiceprint_extractor(
    backend: str = "auto",
    model_path: Optional[str] = None,
    n_mfcc: int = 13,
    sample_rate: int = 16000,
    **kwargs,
) -> BaseVoiceprintExtractor:
    """
    Factory for voiceprint extractors.

    Args:
        backend: "auto" (ONNX when a model path is usable, else MFCC), "onnx"
            or "mfcc".
        model_path: Path to an ONNX speaker-embedding model.
        n_mfcc: MFCC coefficient count for the MFCC extractor / ONNX fallback.
        sample_rate: Sample rate in Hz.
        **kwargs: Forwarded to the concrete extractor constructor.

    Returns:
        A :class:`BaseVoiceprintExtractor` instance.
    """
    backend = (backend or "auto").lower()
    if backend in ("onnx", "auto") and model_path:
        extractor = ONNXEmbeddingExtractor(
            model_path=model_path, sample_rate=sample_rate,
            fallback_n_mfcc=n_mfcc, **kwargs,
        )
        if not extractor.is_fallback:
            return extractor
        if backend == "onnx":
            logger.warning("Requested ONNX extractor is unavailable, using MFCC")
    return MFCCExtractor(n_mfcc=n_mfcc, sample_rate=sample_rate, **kwargs)


def available_voiceprint_backends() -> List[str]:
    """List usable voiceprint backends (``onnx`` only when onnxruntime exists)."""
    backends = ["mfcc"]
    try:
        import onnxruntime  # noqa: F401
        backends.insert(0, "onnx")
    except ImportError:
        pass
    return backends


class AdaptiveCosineClusterer:
    """
    Online speaker clustering with dynamically calibrated cosine thresholds.

    A fixed cosine threshold produces spurious speaker switches on short
    segments, because a 200-500 ms utterance yields a noisy embedding. This
    clusterer therefore:

      * raises the acceptance threshold for short segments
        (``threshold = base + short_penalty * (1 - duration / min_segment_s)``),
      * smooths the similarity against a per-speaker exponential average so that
        a single outlier segment cannot flip the speaker,
      * validates accepted merges with a Gaussian BIC likelihood-ratio test
        (Chen & Gopalakrishnan): a merge is vetoed while the merged model is
        statistically worse than the two separate models, which prevents
        collapsing two genuinely different voices into one cluster.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        min_segment_s: float = 1.0,
        short_penalty: float = 0.12,
        smoothing: float = 0.5,
        bic_penalty: float = 1.0,
        max_centroid_samples: int = 24,
        use_bic: bool = True,
    ):
        self.similarity_threshold = float(similarity_threshold)
        self.min_segment_s = max(float(min_segment_s), 1e-3)
        self.short_penalty = float(short_penalty)
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self.bic_penalty = float(bic_penalty)
        self.max_centroid_samples = max(int(max_centroid_samples), 2)
        self.use_bic = bool(use_bic)
        self.reset()

    def reset(self) -> None:
        """Forget every cluster."""
        self._centroids: Dict[str, np.ndarray] = {}
        self._samples: Dict[str, List[np.ndarray]] = {}
        self._counts: Dict[str, int] = {}
        self._similarity_ema: Dict[str, float] = {}
        self._next_speaker_id = 1

    # ── Threshold calibration ─────────────────────────────────────────────
    def dynamic_threshold(self, duration_s: float = 1.0) -> float:
        """
        Acceptance threshold for a segment of ``duration_s`` seconds.

        Short segments get a stricter threshold; long segments converge to the
        configured base threshold.
        """
        ratio = min(max(float(duration_s), 0.0) / self.min_segment_s, 1.0)
        threshold = self.similarity_threshold + self.short_penalty * (1.0 - ratio)
        return float(min(threshold, 0.999))

    # ── BIC merge test ────────────────────────────────────────────────────
    @staticmethod
    def bic_score(
        samples_a: np.ndarray,
        samples_b: np.ndarray,
        penalty: float = 1.0,
    ) -> float:
        """
        Penalised Gaussian BIC difference for merging two clusters.

        Returns ``BIC_merged - BIC_separate``; a **negative** value means the
        merged model is preferred (i.e. the two clusters should be merged).
        """
        a = np.asarray(samples_a, dtype=np.float64)
        b = np.asarray(samples_b, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        if a.size == 0 or b.size == 0:
            return 0.0

        dim = a.shape[1]
        n_a, n_b = a.shape[0], b.shape[0]
        n = n_a + n_b

        def _log_det(samples: np.ndarray) -> float:
            var = np.maximum(samples.var(axis=0), 1e-8)
            return float(np.sum(np.log(var)))

        log_det_a = _log_det(a)
        log_det_b = _log_det(b)
        log_det_merged = _log_det(np.vstack([a, b]))

        # -2 * log-likelihood difference (constants that cancel are dropped)
        delta_ll = n * log_det_merged - (n_a * log_det_a + n_b * log_det_b)
        # Parameter difference: merging removes dim + dim*(dim+1)/2 parameters
        free_dim = dim + dim * (dim + 1) / 2.0
        return float(delta_ll - penalty * free_dim * np.log(max(n, 2)))

    # ── Assignment ────────────────────────────────────────────────────────
    def assign(self, embedding: np.ndarray, duration_s: float = 1.0) -> Tuple[str, float]:
        """
        Assign an embedding to a speaker cluster (creating one when needed).

        Args:
            embedding: Embedding vector (normalised internally).
            duration_s: Duration of the originating segment in seconds.

        Returns:
            ``(speaker_id, confidence)`` where confidence is the smoothed cosine
            similarity (1.0 for a freshly created cluster).
        """
        embedding = BaseVoiceprintExtractor.l2_normalize(embedding)
        threshold = self.dynamic_threshold(duration_s)

        best_id: Optional[str] = None
        best_similarity = -1.0
        for speaker_id, centroid in self._centroids.items():
            raw = BaseVoiceprintExtractor.cosine_similarity(embedding, centroid)
            previous = self._similarity_ema.get(speaker_id, raw)
            smoothed = self.smoothing * previous + (1.0 - self.smoothing) * raw
            self._similarity_ema[speaker_id] = smoothed
            if smoothed > best_similarity:
                best_similarity = smoothed
                best_id = speaker_id

        if best_id is not None:
            if best_similarity >= threshold and not self._bic_veto(best_id, embedding):
                self._update_cluster(best_id, embedding)
                return best_id, float(best_similarity)
            return self._create_cluster(embedding, best_similarity)

        return self._create_cluster(embedding, 1.0)

    def _bic_veto(self, speaker_id: str, embedding: np.ndarray) -> bool:
        """True when the BIC test rejects folding ``embedding`` into the cluster."""
        if not self.use_bic:
            return False
        history = self._samples.get(speaker_id, [])
        if len(history) < 3:
            # Too little evidence to overrule the cosine decision
            return False
        samples = np.asarray(history, dtype=np.float64)
        score = self.bic_score(
            samples, embedding[np.newaxis, :], penalty=self.bic_penalty
        )
        return score > 0.0

    def _update_cluster(self, speaker_id: str, embedding: np.ndarray) -> None:
        """Update the running centroid / sample history of a cluster."""
        count = self._counts.get(speaker_id, 0) + 1
        self._counts[speaker_id] = count
        alpha = 1.0 / count
        centroid = self._centroids[speaker_id]
        updated = (1.0 - alpha) * centroid + alpha * embedding
        self._centroids[speaker_id] = BaseVoiceprintExtractor.l2_normalize(updated)

        history = self._samples.setdefault(speaker_id, [])
        history.append(embedding)
        if len(history) > self.max_centroid_samples:
            del history[0]

    def _create_cluster(self, embedding: np.ndarray, confidence: float) -> Tuple[str, float]:
        """Create a new cluster and return its id with the reference confidence."""
        speaker_id = f"Speaker_{self._next_speaker_id}"
        self._next_speaker_id += 1
        self._centroids[speaker_id] = embedding
        self._samples[speaker_id] = [embedding]
        self._counts[speaker_id] = 1
        self._similarity_ema[speaker_id] = 1.0
        if confidence <= 0.0:
            confidence = 1.0
        return speaker_id, float(min(max(confidence, 0.0), 1.0))

    # ── Diagnostics ───────────────────────────────────────────────────────
    @property
    def centroids(self) -> Dict[str, np.ndarray]:
        """Current cluster centroid per speaker id."""
        return self._centroids

    @property
    def speaker_count(self) -> int:
        """Number of clusters."""
        return len(self._centroids)

    def cluster_sizes(self) -> Dict[str, int]:
        """Number of segments assigned to each cluster."""
        return dict(self._counts)


class PyannoteDiarizer:
    """
    Neural speaker diarization using pyannote-audio.

    Uses the pyannote/speaker-diarization pipeline for state-of-the-art
    speaker diarization with automatic speaker count detection.

    Args:
        model_name: HuggingFace model name for diarization pipeline.
        hf_token: HuggingFace token (or set HF_TOKEN env var).
        device: Device for inference ("cpu", "cuda", "auto").
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        hf_token: Optional[str] = None,
        device: str = "auto",
    ):
        if not _PYANNOTE_AVAILABLE:
            raise RuntimeError(
                "pyannote-audio not available. "
                "Install with: pip install pyannote-audio"
            )

        self.model_name = model_name
        self._token = hf_token or os.environ.get("HF_TOKEN", "")
        if not self._token:
            logger.warning(
                "No HuggingFace token provided. Set HF_TOKEN env var or pass hf_token. "
                "Some models require authentication."
            )

        self._pipeline = None
        self._device_str = device
        self._load_pipeline()

    def _load_pipeline(self):
        """Load the pyannote diarization pipeline."""
        try:
            logger.info("Loading pyannote diarization: %s", self.model_name)
            self._pipeline = PyannotePipeline.from_pretrained(
                self.model_name,
                use_auth_token=self._token if self._token else None,
            )

            # Move to device
            if self._device_str == "cuda" or (
                self._device_str == "auto" and self._has_cuda()
            ):
                try:
                    import torch
                    self._pipeline.to(torch.device("cuda"))
                    logger.info("pyannote pipeline moved to GPU")
                except (RuntimeError, OSError):
                    logger.info("Could not move pipeline to GPU, using CPU")

            logger.info("pyannote diarization pipeline loaded successfully")
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Failed to load pyannote pipeline: %s", e)
            raise RuntimeError(f"pyannote pipeline load failed: {e}") from e

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[SpeakerSegment]:
        """
        Perform neural speaker diarization.

        Args:
            audio: Audio signal (float32, mono).
            sample_rate: Sample rate in Hz.
            num_speakers: Exact number of speakers (optional).
            min_speakers: Minimum number of speakers.
            max_speakers: Maximum number of speakers.

        Returns:
            List of SpeakerSegment with timestamps and speaker IDs.
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline not loaded")

        audio = ensure_float32(audio)

        try:
            import torch

            # pyannote expects a dict with "waveform" and "sample_rate"
            waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, n_samples)
            input_dict = {"waveform": waveform, "sample_rate": sample_rate}

            # Apply speaker count constraints
            params = {}
            if num_speakers is not None:
                params["num_speakers"] = num_speakers
            if min_speakers is not None:
                params["min_speakers"] = min_speakers
            if max_speakers is not None:
                params["max_speakers"] = max_speakers

            diarization = self._pipeline(input_dict, **params)

            # Convert pyannote Annotation to SpeakerSegment list
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = float(turn.start)
                end = float(turn.end)
                if end > start:
                    segments.append(SpeakerSegment(
                        start_time=start,
                        end_time=end,
                        speaker_id=str(speaker),
                        confidence=0.95,  # pyannote doesn't output per-segment confidence
                    ))

            return segments

        except (RuntimeError, OSError, ValueError, TypeError) as e:
            logger.error("pyannote diarization failed: %s", e)
            raise

    def close(self):
        """Release resources."""
        self._pipeline = None


# 鈹€鈹€ MFCC Fallback Backend 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class MFCCDiarizer:
    """
    MFCC-based speaker diarization with cosine similarity clustering.

    Lightweight fallback that uses MFCC features as speaker embeddings.
    """

    def __init__(
        self,
        n_mfcc: int = 13,
        similarity_threshold: float = 0.7,
        segment_duration_ms: float = 1000.0,
        frame_length: int = 512,
        hop_length: int = 256,
    ):
        self.n_mfcc = n_mfcc
        self.similarity_threshold = similarity_threshold
        self.segment_duration_ms = segment_duration_ms
        self.frame_length = frame_length
        self.hop_length = hop_length

        self._speakers: Dict[str, SpeakerProfile] = {}
        self._next_speaker_id = 1

    def extract_mfcc(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract MFCC features using scipy STFT."""
        if len(audio) == 0:
            return np.zeros((self.n_mfcc, 0), dtype=np.float32)
        audio = ensure_float32(audio)

        min_len = self.frame_length * 2
        if len(audio) < min_len:
            audio = np.pad(audio, (0, min_len - len(audio)), mode='constant')

        freqs, times, Zxx = stft(
            audio, fs=sample_rate, nperseg=self.frame_length,
            noverlap=self.frame_length - self.hop_length,
        )

        power = np.abs(Zxx) ** 2
        mel_fb = self._mel_filterbank(26, power.shape[0], sample_rate)
        mel_spectrum = mel_fb @ power
        log_mel = np.log(mel_spectrum + 1e-10)
        mfcc = self._dct(log_mel, self.n_mfcc)
        return mfcc.astype(np.float32)

    def _mel_filterbank(self, n_filters: int, n_fft: int, sample_rate: int) -> np.ndarray:
        """Create mel-spaced triangular filterbank."""
        def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(0)
        mel_max = hz_to_mel(sample_rate / 2)
        mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
        hz_points = mel_to_hz(mel_points)
        freq_bins = np.fft.rfftfreq(n_fft * 2 - 1, d=1.0 / sample_rate)
        n_freq_bins = len(freq_bins)

        filterbank = np.zeros((n_filters, n_freq_bins), dtype=np.float32)
        for i in range(n_filters):
            f_low, f_center, f_high = hz_points[i], hz_points[i + 1], hz_points[i + 2]
            for j, freq in enumerate(freq_bins):
                if f_low <= freq <= f_center and f_center > f_low:
                    filterbank[i, j] = (freq - f_low) / (f_center - f_low)
                elif f_center < freq <= f_high and f_high > f_center:
                    filterbank[i, j] = (f_high - freq) / (f_high - f_center)
        return filterbank

    def _dct(self, x: np.ndarray, n_coeffs: int) -> np.ndarray:
        """Compute Type-II DCT."""
        n_features = x.shape[0]
        n = min(n_coeffs, n_features)
        k = np.arange(n).reshape(-1, 1)
        n_idx = np.arange(n_features).reshape(1, -1)
        dct_basis = np.cos(np.pi * k * (2 * n_idx + 1) / (2 * n_features))
        return (dct_basis @ x).astype(np.float32)

    def compute_embedding(self, mfcc: np.ndarray) -> np.ndarray:
        """Compute speaker embedding from MFCC (mean + std)."""
        if mfcc.shape[1] == 0:
            return np.zeros(self.n_mfcc * 2, dtype=np.float32)
        mean_feat = np.mean(mfcc, axis=1)
        std_feat = np.std(mfcc, axis=1)
        embedding = np.concatenate([mean_feat, std_feat])
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _assign_speaker(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Assign embedding to known speaker or create new one."""
        if not self._speakers:
            sid = f"Speaker_{self._next_speaker_id}"
            self._next_speaker_id += 1
            self._speakers[sid] = SpeakerProfile(speaker_id=sid, embedding=embedding)
            return sid, 1.0

        best_speaker, best_sim = None, -1.0
        for sid, profile in self._speakers.items():
            sim = self.cosine_similarity(embedding, profile.embedding)
            if sim > best_sim:
                best_sim, best_speaker = sim, sid

        if best_sim >= self.similarity_threshold and best_speaker:
            profile = self._speakers[best_speaker]
            alpha = 1.0 / (profile.segment_count + 1)
            profile.embedding = (1 - alpha) * profile.embedding + alpha * embedding
            norm = np.linalg.norm(profile.embedding)
            if norm > 1e-10:
                profile.embedding /= norm
            profile.segment_count += 1
            return best_speaker, best_sim
        else:
            sid = f"Speaker_{self._next_speaker_id}"
            self._next_speaker_id += 1
            self._speakers[sid] = SpeakerProfile(speaker_id=sid, embedding=embedding)
            return sid, 1.0

    def reset(self):
        """Reset speaker registry."""
        self._speakers.clear()
        self._next_speaker_id = 1

    def diarize(self, audio: np.ndarray, sample_rate: int = 16000) -> List[SpeakerSegment]:
        """Perform MFCC-based speaker diarization."""
        if len(audio) == 0:
            return []
        audio = ensure_float32(audio)

        self.reset()
        segment_samples = int(sample_rate * self.segment_duration_ms / 1000)
        total_segments = max(1, int(np.ceil(len(audio) / segment_samples)))
        segments: List[SpeakerSegment] = []

        for i in range(total_segments):
            start = i * segment_samples
            end = min(start + segment_samples, len(audio))
            seg_audio = audio[start:end]
            if len(seg_audio) < self.frame_length:
                continue

            mfcc = self.extract_mfcc(seg_audio, sample_rate)
            embedding = self.compute_embedding(mfcc)
            speaker_id, confidence = self._assign_speaker(embedding)

            seg_start = start / sample_rate
            seg_end = end / sample_rate

            # Update speaker profile total_duration
            if speaker_id in self._speakers:
                self._speakers[speaker_id].total_duration += (seg_end - seg_start)

            segments.append(SpeakerSegment(
                start_time=seg_start,
                end_time=seg_end,
                speaker_id=speaker_id,
                audio=seg_audio,
                confidence=confidence,
            ))

        return self._merge_consecutive(segments)

    def _merge_consecutive(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Merge consecutive segments from the same speaker.

        Audio data is dropped during merge to reduce memory usage.
        If raw audio per segment is needed, use segments before merging.
        """
        if not segments:
            return []
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg.speaker_id == merged[-1].speaker_id:
                prev = merged[-1]
                merged[-1] = SpeakerSegment(
                    start_time=prev.start_time,
                    end_time=seg.end_time,
                    speaker_id=prev.speaker_id,
                    audio=None,  # Drop audio to save memory during merge
                    confidence=min(prev.confidence, seg.confidence),
                )
            else:
                merged.append(seg)
        return merged

    @property
    def speakers(self) -> Dict[str, SpeakerProfile]:
        return self._speakers


# 鈹€鈹€ Main SpeakerDiarizer 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class SpeakerDiarizer:
    """
    Multi-backend speaker diarization.

    Features:
        - pyannote-audio neural diarization (preferred, state-of-the-art)
        - MFCC + cosine similarity clustering (fallback, lightweight)
        - Auto backend selection
        - Speaker profiles and statistics
        - Timestamped output

    Args:
        backend: Backend to use ("auto", "pyannote", "mfcc").
        hf_token: HuggingFace token for pyannote (or set HF_TOKEN env var).
        model_name: pyannote model name.
        n_mfcc: Number of MFCC coefficients (MFCC backend).
        similarity_threshold: Cosine similarity threshold (MFCC backend).
        segment_duration_ms: Analysis segment duration in ms (MFCC backend).
    """

    def __init__(
        self,
        backend: str = "auto",
        hf_token: Optional[str] = None,
        model_name: str = "pyannote/speaker-diarization-3.1",
        n_mfcc: int = 13,
        similarity_threshold: float = 0.7,
        segment_duration_ms: float = 1000.0,
    ):
        self._backend_type = backend
        self._pyannote: Optional[PyannoteDiarizer] = None
        self._mfcc: Optional[MFCCDiarizer] = None
        self._active_backend = "mfcc"

        # Init MFCC always (used as fallback)
        self._mfcc = MFCCDiarizer(
            n_mfcc=n_mfcc,
            similarity_threshold=similarity_threshold,
            segment_duration_ms=segment_duration_ms,
        )

        # Try pyannote
        if backend == "pyannote" or (backend == "auto" and _PYANNOTE_AVAILABLE):
            try:
                self._pyannote = PyannoteDiarizer(
                    model_name=model_name, hf_token=hf_token,
                )
                self._active_backend = "pyannote"
                logger.info("Using pyannote diarization backend")
            except (RuntimeError, OSError, ImportError, ValueError) as e:
                logger.warning("pyannote init failed (%s), falling back to MFCC", e)
                self._active_backend = "mfcc"
        else:
            self._active_backend = "mfcc"
            logger.info("Using MFCC diarization backend")

    @property
    def backend(self) -> str:
        return self._active_backend

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """
        Perform speaker diarization.

        Args:
            audio: Audio signal (float32, mono).
            sample_rate: Sample rate in Hz.
            num_speakers: Exact number of speakers (pyannote only).
            min_speakers: Minimum speakers (pyannote only).
            max_speakers: Maximum speakers (pyannote only).

        Returns:
            DiarizationResult with segments, speaker count, and metadata.
        """
        if self._active_backend == "pyannote" and self._pyannote is not None:
            segments = self._pyannote.diarize(
                audio, sample_rate, num_speakers, min_speakers, max_speakers,
            )
        else:
            segments = self._mfcc.diarize(audio, sample_rate)

        # Compute speaker count
        speaker_ids = set(seg.speaker_id for seg in segments)
        total_duration = sum(seg.duration for seg in segments)

        return DiarizationResult(
            segments=segments,
            speaker_count=len(speaker_ids),
            total_duration=total_duration,
            backend_used=self._active_backend,
        )

    def diarize_segments(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        **kwargs,
    ) -> List[SpeakerSegment]:
        """Convenience method: return just the segments list."""
        result = self.diarize(audio, sample_rate, **kwargs)
        return result.segments

    def get_speaker_count(self) -> int:
        """Get number of identified speakers (from last diarization)."""
        if self._active_backend == "mfcc" and self._mfcc:
            return len(self._mfcc.speakers)
        return 0

    def get_speaker_ids(self) -> List[str]:
        """Get list of speaker IDs (from last diarization)."""
        if self._active_backend == "mfcc" and self._mfcc:
            return list(self._mfcc.speakers.keys())
        return []

    def get_speaker_profile(self, speaker_id: str) -> Optional[SpeakerProfile]:
        """Get profile for a specific speaker."""
        if self._active_backend == "mfcc" and self._mfcc:
            return self._mfcc.speakers.get(speaker_id)
        return None

    @staticmethod
    def available_backends() -> List[str]:
        """List available diarization backends."""
        backends = ["mfcc"]
        if _PYANNOTE_AVAILABLE:
            backends.insert(0, "pyannote")
        return backends

    def close(self):
        """Release resources."""
        if self._pyannote is not None:
            self._pyannote.close()
            self._pyannote = None

    def __del__(self):
        self.close()