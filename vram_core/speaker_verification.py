"""
Speaker Verification Module for vram_core
==========================================

Provides 1:1 speaker identity verification using MFCC + cosine similarity.
Supports: register voiceprint, verify voiceprint, delete voiceprint.

v2.6.0 additions:
    - Pluggable voiceprint embeddings (``BaseVoiceprintExtractor`` with ONNX or
      MFCC backends) usable instead of the built-in MFCC statistics
    - Duration-adaptive verification thresholds for short probes
    - Dynamic cosine pruning of inconsistent enrolment templates
    - Optional Gaussian BIC veto for statistically inconsistent probes

Architecture:
    - Voiceprint: Dataclass storing MFCC feature template
    - SpeakerVerifier: Main verification engine

Usage:
    verifier = SpeakerVerifier()
    verifier.register("alice", audio_array)
    result = verifier.verify("alice", test_audio)
    print(result.verified, result.confidence)
"""

import logging
import json
import hashlib
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field

import numpy as np

from vram_core.utils import ensure_float32, compute_zero_crossing_rate

logger = logging.getLogger(__name__)


@dataclass
class Voiceprint:
    """
    Stored voiceprint template for a speaker.

    Attributes:
        speaker_id: Unique speaker identifier.
        mfcc_mean: Mean MFCC feature vector (template).
        mfcc_std: Standard deviation of MFCC features.
        num_samples: Number of audio samples used to create template.
        created_at: Timestamp of creation.
        updated_at: Timestamp of last update.
        metadata: Optional metadata dict.
    """
    speaker_id: str = ""
    mfcc_mean: Optional[np.ndarray] = None
    mfcc_std: Optional[np.ndarray] = None
    num_samples: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (numpy arrays to lists)."""
        return {
            "speaker_id": self.speaker_id,
            "mfcc_mean": self.mfcc_mean.tolist() if self.mfcc_mean is not None else None,
            "mfcc_std": self.mfcc_std.tolist() if self.mfcc_std is not None else None,
            "num_samples": self.num_samples,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Voiceprint":
        """Deserialize from dict."""
        vp = cls()
        vp.speaker_id = data.get("speaker_id", "")
        if data.get("mfcc_mean") is not None:
            vp.mfcc_mean = np.array(data["mfcc_mean"], dtype=np.float32)
        if data.get("mfcc_std") is not None:
            vp.mfcc_std = np.array(data["mfcc_std"], dtype=np.float32)
        vp.num_samples = data.get("num_samples", 0)
        vp.created_at = data.get("created_at", 0.0)
        vp.updated_at = data.get("updated_at", 0.0)
        vp.metadata = data.get("metadata", {})
        return vp


@dataclass
class VerificationResult:
    """
    Result of a speaker verification attempt.

    Attributes:
        speaker_id: The speaker being verified.
        verified: Whether verification passed.
        confidence: Similarity score (0.0-1.0).
        threshold: Threshold used for decision.
        processing_time: Time taken in seconds.
    """
    speaker_id: str = ""
    verified: bool = False
    confidence: float = 0.0
    threshold: float = 0.75
    processing_time: float = 0.0

    def __repr__(self) -> str:
        status = "锟?VERIFIED" if self.verified else "锟?REJECTED"
        return (
            f"VerificationResult({self.speaker_id}: {status}, "
            f"confidence={self.confidence:.3f}, "
            f"threshold={self.threshold:.3f})"
        )


class SpeakerVerifier:
    """
    Speaker identity verification engine.

    Uses MFCC feature extraction + cosine similarity for 1:1 verification.

    Features:
        - Register voiceprint from audio samples
        - Verify speaker identity against stored voiceprint
        - Delete / update voiceprints
        - Persistent storage (JSON file)
        - Multi-sample enrollment (averages multiple recordings)
        - Pluggable voiceprint extractors (``extractor=`` /
          ``extractor_backend=``, v2.6.0): ONNX speaker-embedding models with a
          transparent MFCC fallback
        - Duration-adaptive thresholds (``adaptive_threshold=True``) and dynamic
          cosine pruning (``dynamic_pruning=True``) for short / noisy probes
        - Optional Gaussian BIC veto (``use_bic=True``) rejecting probes that are
          statistically inconsistent with the enrollment templates

    Usage:
        verifier = SpeakerVerifier(threshold=0.75)

        # Register
        verifier.register("alice", alice_audio, sample_rate=16000)
        verifier.register("bob", bob_audio, sample_rate=16000)

        # Verify
        result = verifier.verify("alice", test_audio, sample_rate=16000)
        if result.verified:
            print(f"Welcome, Alice! (confidence: {result.confidence:.2f})")

    v2.6.0 hardening (all opt-in, legacy defaults are unchanged):
        verifier = SpeakerVerifier(
            extractor_backend="auto", model_path="ecapa.onnx",
            adaptive_threshold=True, dynamic_pruning=True, use_bic=True,
        )
    """

    def __init__(
        self,
        threshold: float = 0.75,
        n_mfcc: int = 20,
        sample_rate: int = 16000,
        storage_path: Optional[Union[str, Path]] = None,
        extractor: Optional[Any] = None,
        extractor_backend: str = "builtin",
        model_path: Optional[str] = None,
        enhanced_mfcc: bool = False,
        adaptive_threshold: bool = False,
        min_segment_s: float = 1.0,
        short_penalty: float = 0.12,
        max_templates: int = 8,
        dynamic_pruning: bool = False,
        template_prune_gap: float = 0.15,
        bic_penalty: float = 1.0,
        use_bic: bool = False,
    ):
        """
        Initialize SpeakerVerifier.

        Args:
            threshold: Cosine similarity threshold for verification (0.0-1.0).
            n_mfcc: Number of MFCC coefficients to extract.
            sample_rate: Expected audio sample rate.
            storage_path: Path to persist voiceprints (JSON file).
            extractor: Optional voiceprint extractor object exposing
                ``embed(audio, sample_rate)`` (e.g. a
                :class:`BaseVoiceprintExtractor`). When given it replaces the
                built-in MFCC statistics for scoring.
            extractor_backend: ``"builtin"`` (legacy MFCC path, default) or a
                backend understood by ``create_voiceprint_extractor()``
                (``"auto"`` / ``"onnx"`` / ``"mfcc"``).
            model_path: ONNX speaker-embedding model used by the ONNX backend.
            enhanced_mfcc: Build the *enhanced* MFCC track (Δ/ΔΔ with CMVN, see
                :class:`MFCCExtractor`) for the configured backend and for the
                ONNX fallback. Off by default so pre-2.6.1 behaviour (and
                embeddings) are reproduced exactly.
            adaptive_threshold: Raise the threshold for short probes
                (see :meth:`dynamic_threshold`), mirroring
                ``AdaptiveCosineClusterer``.
            min_segment_s: Probe duration (s) at which the base threshold is used.
            short_penalty: Extra threshold applied to a zero-length probe.
            max_templates: Enrollment vectors kept per speaker (FIFO).
            dynamic_pruning: Prune enrollment vectors that disagree with the best
                match before averaging, so a single bad enrollment cannot drag
                the score down.
            template_prune_gap: Cosine gap below the best match that still counts
                as agreement (used when ``dynamic_pruning`` is enabled).
            bic_penalty: Penalty of the Gaussian BIC likelihood-ratio test.
            use_bic: Reject probes the BIC test considers statistically
                inconsistent with the enrollment vectors.
        """
        self.threshold = threshold
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self._voiceprints: Dict[str, Voiceprint] = {}
        self._storage_path = Path(storage_path) if storage_path else None

        # ── v2.6.0: pluggable embeddings + adaptive decision rules ──────────
        self.adaptive_threshold = bool(adaptive_threshold)
        self.min_segment_s = max(float(min_segment_s), 1e-3)
        self.short_penalty = max(float(short_penalty), 0.0)
        self.max_templates = max(int(max_templates), 1)
        self.dynamic_pruning = bool(dynamic_pruning)
        self.template_prune_gap = max(float(template_prune_gap), 0.0)
        self.bic_penalty = float(bic_penalty)
        self.use_bic = bool(use_bic)
        self._templates: Dict[str, List[np.ndarray]] = {}
        self.extractor_backend = (extractor_backend or "builtin").lower()
        self.enhanced_mfcc = bool(enhanced_mfcc)
        self.extractor = extractor or self._create_extractor(model_path)

        # Load existing voiceprints if storage exists
        if self._storage_path and self._storage_path.exists():
            self._load()

        logger.info(
            f"SpeakerVerifier initialized: threshold={threshold}, "
            f"n_mfcc={n_mfcc}, extractor={self.extractor_name}, "
            f"registered_speakers={len(self._voiceprints)}"
        )

    # ── Extractor plumbing (v2.6.0) ───────────────────────────────────────

    def _create_extractor(self, model_path: Optional[str]) -> Optional[Any]:
        """Build the configured voiceprint extractor (``None`` = builtin MFCC)."""
        if self.extractor_backend in ("", "builtin", "default", "none"):
            return None
        try:
            from vram_core.speaker_diarization import create_voiceprint_extractor

            return create_voiceprint_extractor(
                backend=self.extractor_backend,
                model_path=model_path,
                n_mfcc=self.n_mfcc,
                sample_rate=self.sample_rate,
                enhanced_mfcc=self.enhanced_mfcc,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                f"Voiceprint extractor '{self.extractor_backend}' unavailable "
                f"({exc}); falling back to built-in MFCC features"
            )
            return None

    @property
    def extractor_name(self) -> str:
        """Identifier of the active extractor (``"builtin"`` for MFCC stats)."""
        if self.extractor is None:
            return "builtin"
        return str(getattr(self.extractor, "name", type(self.extractor).__name__))

    # ── Probe scoring & adaptive decision rules (v2.6.0) ──────────────────

    def _probe(self, audio: np.ndarray, sample_rate: int):
        """
        Extract ``(mfcc_mean, probe_vector)`` from an audio buffer.

        ``mfcc_mean`` keeps the historical MFCC-statistics behaviour, while
        ``probe_vector`` is the L2-normalised vector compared against the
        enrollment pool (extractor embedding when configured, MFCC mean
        otherwise).
        """
        if np.asarray(audio).reshape(-1).size == 0:
            # Zero-length probes are rejected deterministically instead of relying
            # on the feature front-end to cope with empty input.
            probe_dim = self.n_mfcc
            if self.extractor is not None:
                probe_dim = int(getattr(self.extractor, "dim", self.n_mfcc))
            return (
                np.zeros(self.n_mfcc, dtype=np.float32),
                np.zeros(probe_dim, dtype=np.float32),
            )

        mfcc = np.asarray(self._extract_mfcc(audio, sample_rate), dtype=np.float32)
        if mfcc.ndim != 2 or mfcc.shape[0] == 0:
            mean = np.zeros(self.n_mfcc, dtype=np.float32)
        else:
            mean = np.mean(mfcc, axis=0).astype(np.float32)

        if self.extractor is not None:
            vector = np.asarray(
                self.extractor.embed(audio, sample_rate), dtype=np.float32
            )
        else:
            vector = mean
        return mean, self._l2_normalize(vector)

    def _probe_vector(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """L2-normalised comparison vector for ``audio`` (see :meth:`_probe`)."""
        return self._probe(audio, sample_rate)[1]

    @staticmethod
    def _l2_normalize(vector: np.ndarray) -> np.ndarray:
        """L2-normalised copy of ``vector`` (degenerate vectors are returned)."""
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if np.isfinite(norm) and norm > 1e-10:
            vector = vector / norm
        return vector.astype(np.float32)

    def _store_template(self, speaker_id: str, vector: np.ndarray) -> None:
        """Append a normalised enrollment vector, keeping the newest templates."""
        pool = self._templates.setdefault(speaker_id, [])
        pool.append(self._l2_normalize(vector))
        # FIFO bound: keep only the ``max_templates`` most recent enrollments so
        # the pool cannot grow without limit across long sessions.
        excess = len(pool) - self.max_templates
        if excess > 0:
            del pool[:excess]

    def _pool_similarity(self, pool: List[np.ndarray], probe: np.ndarray) -> float:
        """
        Mean cosine similarity over an enrollment pool.

        Pool entries and ``probe`` are L2-normalised, so the dot product is the
        cosine similarity. With ``dynamic_pruning`` enabled, templates whose
        similarity is more than ``template_prune_gap`` below the best match are
        dropped as outliers (dynamic cosine pruning) before averaging.
        """
        if not pool:
            return 0.0
        sims = np.array([float(np.dot(t, probe)) for t in pool], dtype=np.float64)
        if self.dynamic_pruning and sims.size > 1:
            keep = sims >= (float(np.max(sims)) - self.template_prune_gap)
            if not np.any(keep):  # never prune the best match itself
                keep = np.zeros_like(sims, dtype=bool)
                keep[int(np.argmax(sims))] = True
            sims = sims[keep]
        return float(np.mean(sims))

    def _score(
        self,
        speaker_id: str,
        mfcc_mean: np.ndarray,
        probe: np.ndarray,
    ) -> float:
        """
        Similarity between a probe and a registered speaker.

        The enrollment pool is used when a voiceprint extractor is configured or
        when ``dynamic_pruning`` is enabled; otherwise the stored MFCC template
        is compared directly (identical to pre-2.6.0 behaviour).
        """
        pool = self._templates.get(speaker_id) or []
        if pool and (self.extractor is not None or self.dynamic_pruning):
            return self._pool_similarity(pool, probe)

        voiceprint = self._voiceprints.get(speaker_id)
        if voiceprint is None or voiceprint.mfcc_mean is None:
            return 0.0
        return self._cosine_similarity(voiceprint.mfcc_mean, mfcc_mean)

    def _duration_s(self, audio: np.ndarray, sample_rate: int) -> float:
        """Duration of ``audio`` in seconds (0.0 for degenerate input)."""
        length = int(np.asarray(audio).reshape(-1).shape[0])
        return length / float(sample_rate or self.sample_rate or 16000)

    def dynamic_threshold(self, duration_s: float = 1.0) -> float:
        """
        Decision threshold for a probe of ``duration_s`` seconds.

        With ``adaptive_threshold`` disabled this is simply the configured
        threshold. When enabled, short probes get a stricter threshold,
        mirroring ``AdaptiveCosineClusterer.dynamic_threshold``.
        """
        base = float(self.threshold)
        if not self.adaptive_threshold:
            return base
        ratio = min(max(float(duration_s), 0.0) / self.min_segment_s, 1.0)
        return float(min(base + self.short_penalty * (1.0 - ratio), 0.999))

    def _bic_veto(self, speaker_id: str, probe: np.ndarray) -> bool:
        """True when the Gaussian BIC test rejects ``probe`` for ``speaker_id``."""
        if not self.use_bic:
            return False
        pool = self._templates.get(speaker_id) or []
        if len(pool) < 3:
            # A covariance test on fewer than three enrollment vectors is noise.
            return False
        try:
            from vram_core.speaker_diarization import AdaptiveCosineClusterer
        except Exception:  # pragma: no cover - defensive
            return False
        score = AdaptiveCosineClusterer.bic_score(
            np.asarray(pool, dtype=np.float64),
            np.asarray(probe, dtype=np.float64)[np.newaxis, :],
            penalty=self.bic_penalty,
        )
        return bool(score > 0.0)

    def templates_count(self, speaker_id: Optional[str] = None) -> int:
        """Number of enrollment vectors stored for a speaker (or in total)."""
        if speaker_id is None:
            return sum(len(pool) for pool in self._templates.values())
        return len(self._templates.get(speaker_id, []))

    def reset_templates(self) -> None:
        """Forget every stored enrollment vector (voiceprints are untouched)."""
        self._templates.clear()


    # 鈹€鈹€ MFCC Feature Extraction 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _extract_mfcc(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract MFCC features from audio.

        Args:
            audio: Float32 mono audio array.
            sr: Sample rate.

        Returns:
            MFCC feature matrix (n_frames, n_mfcc).
        """
        audio = ensure_float32(audio)

        # Try librosa first
        try:
            import librosa
            mfcc = librosa.feature.mfcc(
                y=audio, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=512, hop_length=256,
            )
            return mfcc.T  # (n_frames, n_mfcc)
        except ImportError:
            pass

        # Fallback: manual MFCC using numpy FFT
        return self._manual_mfcc(audio, sr)

    def _manual_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Manual MFCC extraction using numpy (fallback when librosa not available).

        Simplified MFCC: pre-emphasis 锟?framing 锟?FFT 锟?mel filterbank 锟?DCT.
        """
        n_fft = 512
        hop_length = 256
        n_mels = self.n_mfcc

        # Pre-emphasis
        emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        # Framing
        frame_len = n_fft
        num_frames = max(1, (len(emphasized) - frame_len) // hop_length + 1)
        frames = np.zeros((num_frames, frame_len), dtype=np.float32)
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_len
            if end <= len(emphasized):
                frames[i] = emphasized[start:end]
            else:
                frames[i, :len(emphasized) - start] = emphasized[start:]

        # Hamming window
        hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_len) / (frame_len - 1))
        frames *= hamming

        # FFT
        mag = np.abs(np.fft.rfft(frames, n=n_fft))
        pow_spec = (mag ** 2) / n_fft

        # Mel filterbank
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700.0)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595.0) - 1)

        low_mel = hz_to_mel(0)
        high_mel = hz_to_mel(sr / 2)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        n_filters = n_mels
        fbank = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)
        for m in range(1, n_filters + 1):
            f_left = bin_points[m - 1]
            f_center = bin_points[m]
            f_right = bin_points[m + 1]
            for k in range(f_left, f_center):
                if f_center != f_left:
                    fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                if f_right != f_center:
                    fbank[m - 1, k] = (f_right - k) / (f_right - f_center)

        mel_spec = np.dot(pow_spec, fbank.T)
        mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)
        log_mel = np.log(mel_spec)

        # DCT (Type-II)
        n = log_mel.shape[1]
        dct_matrix = np.zeros((self.n_mfcc, n), dtype=np.float32)
        for i in range(self.n_mfcc):
            for j in range(n):
                dct_matrix[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * n))
        mfcc = np.dot(log_mel, dct_matrix.T)

        return mfcc  # (n_frames, n_mfcc)

    # 鈹€鈹€ Registration 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def register(
        self,
        speaker_id: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Voiceprint:
        """
        Register a voiceprint for a speaker.

        If the speaker already exists, the new sample is merged with the
        existing template using exponential moving average.

        Args:
            speaker_id: Unique speaker identifier.
            audio: Float32 mono audio array.
            sample_rate: Audio sample rate.
            metadata: Optional metadata dict.

        Returns:
            Voiceprint object.
        """
        start_time = time.time()

        # Extract MFCC features
        mfcc = self._extract_mfcc(audio, sample_rate)

        # Compute statistics
        mfcc_mean = np.mean(mfcc, axis=0).astype(np.float32)
        mfcc_std = np.std(mfcc, axis=0).astype(np.float32)

        if speaker_id in self._voiceprints:
            # Update existing voiceprint (exponential moving average)
            existing = self._voiceprints[speaker_id]
            alpha = 0.3  # weight for new sample
            if existing.mfcc_mean is not None:
                existing.mfcc_mean = (1 - alpha) * existing.mfcc_mean + alpha * mfcc_mean
                existing.mfcc_std = (1 - alpha) * existing.mfcc_std + alpha * mfcc_std
            else:
                existing.mfcc_mean = mfcc_mean
                existing.mfcc_std = mfcc_std
            existing.num_samples += 1
            existing.updated_at = time.time()
            if metadata:
                existing.metadata.update(metadata)
            voiceprint = existing
            logger.info(
                f"Updated voiceprint for '{speaker_id}' "
                f"(samples={existing.num_samples})"
            )
        else:
            # Create new voiceprint
            voiceprint = Voiceprint(
                speaker_id=speaker_id,
                mfcc_mean=mfcc_mean,
                mfcc_std=mfcc_std,
                num_samples=1,
                created_at=time.time(),
                updated_at=time.time(),
                metadata=metadata or {},
            )
            self._voiceprints[speaker_id] = voiceprint
            logger.info(
                f"Registered voiceprint for '{speaker_id}' "
                f"(features={mfcc_mean.shape[0]})"
            )

        # v2.6.0: keep the enrollment pool used by the pruning / BIC decision
        # rules and by pluggable voiceprint extractors.
        enrollment = (
            np.asarray(self.extractor.embed(audio, sample_rate), dtype=np.float32)
            if self.extractor is not None
            else mfcc_mean
        )
        self._store_template(speaker_id, enrollment)

        # Persist
        self._save()

        elapsed = time.time() - start_time
        logger.debug(f"Registration took {elapsed:.3f}s")

        return voiceprint

    def register_from_samples(
        self,
        speaker_id: str,
        audio_samples: List[np.ndarray],
        sample_rate: int = 16000,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Voiceprint:
        """
        Register from multiple audio samples for better accuracy.

        Args:
            speaker_id: Unique speaker identifier.
            audio_samples: List of float32 mono audio arrays.
            sample_rate: Audio sample rate.
            metadata: Optional metadata.

        Returns:
            Voiceprint object.
        """
        all_mfcc = []
        enrollments = []
        for sample in audio_samples:
            mfcc = self._extract_mfcc(sample, sample_rate)
            all_mfcc.append(mfcc)
            if self.extractor is not None:
                enrollments.append(
                    np.asarray(self.extractor.embed(sample, sample_rate), dtype=np.float32)
                )
            elif getattr(mfcc, "ndim", 0) == 2 and mfcc.shape[0]:
                enrollments.append(np.mean(mfcc, axis=0).astype(np.float32))

        # Concatenate all MFCC frames
        combined = np.concatenate(all_mfcc, axis=0)
        mfcc_mean = np.mean(combined, axis=0).astype(np.float32)
        mfcc_std = np.std(combined, axis=0).astype(np.float32)

        voiceprint = Voiceprint(
            speaker_id=speaker_id,
            mfcc_mean=mfcc_mean,
            mfcc_std=mfcc_std,
            num_samples=len(audio_samples),
            created_at=time.time(),
            updated_at=time.time(),
            metadata=metadata or {},
        )
        # v2.6.0: enrollment pool built from every provided sample
        for enrollment in enrollments:
            self._store_template(speaker_id, enrollment)

        self._voiceprints[speaker_id] = voiceprint
        self._save()

        logger.info(
            f"Registered voiceprint for '{speaker_id}' from "
            f"{len(audio_samples)} samples"
        )
        return voiceprint

    # 鈹€鈹€ Verification 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def verify(
        self,
        speaker_id: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
        threshold: Optional[float] = None,
    ) -> VerificationResult:
        """
        Verify speaker identity against stored voiceprint.

        Args:
            speaker_id: Speaker to verify against.
            audio: Test audio (float32 mono).
            sample_rate: Audio sample rate.
            threshold: Override verification threshold.

        Returns:
            VerificationResult with verified flag and confidence.

        Raises:
            KeyError: If speaker_id not registered.
        """
        start_time = time.time()

        if speaker_id not in self._voiceprints:
            raise KeyError(
                f"Speaker '{speaker_id}' not registered. "
                f"Registered: {list(self._voiceprints.keys())}"
            )

        # Probe features: legacy MFCC statistics + normalised comparison vector
        mfcc_mean, probe = self._probe(audio, sample_rate)

        # Threshold: explicit override > duration-adaptive > configured value
        if threshold:
            effective_threshold = float(threshold)
        elif self.adaptive_threshold:
            effective_threshold = self.dynamic_threshold(
                self._duration_s(audio, sample_rate)
            )
        else:
            effective_threshold = float(self.threshold)

        # Similarity against the enrollment pool / stored template
        similarity = self._score(speaker_id, mfcc_mean, probe)

        # Decision (the BIC veto only applies when explicitly enabled)
        verified = similarity >= effective_threshold and not self._bic_veto(
            speaker_id, probe
        )

        elapsed = time.time() - start_time

        result = VerificationResult(
            speaker_id=speaker_id,
            verified=verified,
            confidence=float(similarity),
            threshold=effective_threshold,
            processing_time=elapsed,
        )

        logger.info(
            f"Verification '{speaker_id}': "
            f"{'VERIFIED' if verified else 'REJECTED'} "
            f"(confidence={similarity:.3f}, threshold={effective_threshold:.3f})"
        )

        return result

    def verify_any(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        threshold: Optional[float] = None,
    ) -> Optional[VerificationResult]:
        """
        Identify which registered speaker matches the audio (1:N).

        Args:
            audio: Test audio (float32 mono).
            sample_rate: Audio sample rate.
            threshold: Override verification threshold.

        Returns:
            Best matching VerificationResult, or None if no match.
        """
        mfcc_mean, probe = self._probe(audio, sample_rate)

        # Threshold: explicit override > duration-adaptive > configured value
        if threshold:
            effective_threshold = float(threshold)
        elif self.adaptive_threshold:
            effective_threshold = self.dynamic_threshold(
                self._duration_s(audio, sample_rate)
            )
        else:
            effective_threshold = float(self.threshold)

        best_result = None
        best_confidence = 0.0

        for speaker_id, voiceprint in self._voiceprints.items():
            similarity = self._score(speaker_id, mfcc_mean, probe)
            if similarity > best_confidence:
                best_confidence = similarity
                best_result = VerificationResult(
                    speaker_id=speaker_id,
                    verified=similarity >= effective_threshold,
                    confidence=float(similarity),
                    threshold=effective_threshold,
                )

        if (
            best_result
            and best_result.verified
            and self._bic_veto(best_result.speaker_id, probe)
        ):
            logger.info(
                f"BIC veto rejected '{best_result.speaker_id}' "
                f"(confidence={best_result.confidence:.3f})"
            )
            return None

        if best_result and best_result.verified:
            logger.info(
                f"Identified speaker: '{best_result.speaker_id}' "
                f"(confidence={best_result.confidence:.3f})"
            )
            return best_result

        logger.info("No matching speaker found")
        return None

    # 鈹€鈹€ Management 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def delete(self, speaker_id: str) -> bool:
        """
        Delete a registered voiceprint.

        Args:
            speaker_id: Speaker to delete.

        Returns:
            True if deleted, False if not found.
        """
        if speaker_id in self._voiceprints:
            del self._voiceprints[speaker_id]
            self._templates.pop(speaker_id, None)
            self._save()
            logger.info(f"Deleted voiceprint for '{speaker_id}'")
            return True
        return False

    def list_speakers(self) -> List[Dict[str, Any]]:
        """
        List all registered speakers.

        Returns:
            List of dicts with speaker info.
        """
        speakers = []
        for sid, vp in self._voiceprints.items():
            speakers.append({
                "speaker_id": sid,
                "num_samples": vp.num_samples,
                "created_at": vp.created_at,
                "updated_at": vp.updated_at,
                "metadata": vp.metadata,
            })
        return speakers

    def get_speaker(self, speaker_id: str) -> Optional[Voiceprint]:
        """Get a specific voiceprint."""
        return self._voiceprints.get(speaker_id)

    def set_threshold(self, threshold: float) -> None:
        """Update verification threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        logger.info(f"Threshold set to {threshold}")

    # 鈹€鈹€ Utility 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    # 鈹€鈹€ Persistence 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _save(self) -> None:
        """Save voiceprints to JSON file."""
        if not self._storage_path:
            return

        data = {}
        for sid, vp in self._voiceprints.items():
            data[sid] = vp.to_dict()

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug(f"Saved {len(data)} voiceprints to {self._storage_path}")

    def _load(self) -> None:
        """Load voiceprints from JSON file."""
        try:
            data = json.loads(self._storage_path.read_text(encoding="utf-8"))
            for sid, vp_dict in data.items():
                self._voiceprints[sid] = Voiceprint.from_dict(vp_dict)
            logger.info(f"Loaded {len(self._voiceprints)} voiceprints from {self._storage_path}")
        except Exception as e:
            logger.error(f"Failed to load voiceprints: {e}")