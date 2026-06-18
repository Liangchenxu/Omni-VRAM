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
from dataclasses import dataclass, field
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