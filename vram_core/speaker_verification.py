"""
Speaker Verification Module for vram_core
==========================================

Provides 1:1 speaker identity verification using MFCC + cosine similarity.
Supports: register voiceprint, verify voiceprint, delete voiceprint.

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

    Usage:
        verifier = SpeakerVerifier(threshold=0.75)

        # Register
        verifier.register("alice", alice_audio, sample_rate=16000)
        verifier.register("bob", bob_audio, sample_rate=16000)

        # Verify
        result = verifier.verify("alice", test_audio, sample_rate=16000)
        if result.verified:
            print(f"Welcome, Alice! (confidence: {result.confidence:.2f})")
    """

    def __init__(
        self,
        threshold: float = 0.75,
        n_mfcc: int = 20,
        sample_rate: int = 16000,
        storage_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize SpeakerVerifier.

        Args:
            threshold: Cosine similarity threshold for verification (0.0-1.0).
            n_mfcc: Number of MFCC coefficients to extract.
            sample_rate: Expected audio sample rate.
            storage_path: Path to persist voiceprints (JSON file).
        """
        self.threshold = threshold
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self._voiceprints: Dict[str, Voiceprint] = {}
        self._storage_path = Path(storage_path) if storage_path else None

        # Load existing voiceprints if storage exists
        if self._storage_path and self._storage_path.exists():
            self._load()

        logger.info(
            f"SpeakerVerifier initialized: threshold={threshold}, "
            f"n_mfcc={n_mfcc}, registered_speakers={len(self._voiceprints)}"
        )

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
        for sample in audio_samples:
            mfcc = self._extract_mfcc(sample, sample_rate)
            all_mfcc.append(mfcc)

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
        effective_threshold = threshold or self.threshold

        if speaker_id not in self._voiceprints:
            raise KeyError(
                f"Speaker '{speaker_id}' not registered. "
                f"Registered: {list(self._voiceprints.keys())}"
            )

        voiceprint = self._voiceprints[speaker_id]

        # Extract MFCC from test audio
        test_mfcc = self._extract_mfcc(audio, sample_rate)
        test_mean = np.mean(test_mfcc, axis=0).astype(np.float32)

        # Compute cosine similarity
        similarity = self._cosine_similarity(voiceprint.mfcc_mean, test_mean)

        # Decision
        verified = similarity >= effective_threshold

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
        effective_threshold = threshold or self.threshold

        test_mfcc = self._extract_mfcc(audio, sample_rate)
        test_mean = np.mean(test_mfcc, axis=0).astype(np.float32)

        best_result = None
        best_confidence = 0.0

        for speaker_id, voiceprint in self._voiceprints.items():
            similarity = self._cosine_similarity(voiceprint.mfcc_mean, test_mean)
            if similarity > best_confidence:
                best_confidence = similarity
                best_result = VerificationResult(
                    speaker_id=speaker_id,
                    verified=similarity >= effective_threshold,
                    confidence=float(similarity),
                    threshold=effective_threshold,
                )

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