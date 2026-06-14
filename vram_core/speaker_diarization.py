"""
Speaker Diarization Module for Omni-VRAM
==========================================

Lightweight speaker diarization based on MFCC feature extraction
and cosine similarity clustering. Identifies "who spoke when" by
comparing acoustic embeddings across audio segments.

No external speaker embedding models required — uses MFCC statistics
as speaker embeddings with scipy for signal processing.

Algorithm:
    1. Extract MFCC features from each audio segment
    2. Compute mean MFCC vector as speaker embedding
    3. Compare embeddings via cosine similarity
    4. Dynamic clustering: assign to known speaker (similarity > threshold)
       or create new speaker identity

Usage:
    from vram_core.speaker_diarization import SpeakerDiarizer

    diarizer = SpeakerDiarizer()
    segments = diarizer.diarize(audio_array, sample_rate=16000)
    for seg in segments:
        print(f"[{seg.start_time:.1f}s-{seg.end_time:.1f}s] Speaker {seg.speaker_id}")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import stft

logger = logging.getLogger(__name__)


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
    embedding: np.ndarray
    total_duration: float = 0.0
    segment_count: int = 0


class SpeakerDiarizer:
    """
    MFCC-based speaker diarization with cosine similarity clustering.

    Extracts MFCC features from audio segments and uses them as
    speaker embeddings. New segments are compared against known
    speakers using cosine similarity.

    Args:
        n_mfcc: Number of MFCC coefficients to extract.
        similarity_threshold: Cosine similarity threshold for same-speaker
            decision (default 0.7).
        segment_duration_ms: Duration of analysis segments in milliseconds.
        frame_length: FFT frame length in samples.
        hop_length: STFT hop length in samples.

    Usage:
        diarizer = SpeakerDiarizer(similarity_threshold=0.7)
        segments = diarizer.diarize(audio, sample_rate=16000)
        print(diarizer.get_speaker_count(), "speakers found")
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

        # Speaker registry
        self._speakers: Dict[str, SpeakerProfile] = {}
        self._next_speaker_id = 1

    def extract_mfcc(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Extract MFCC features from audio using scipy STFT.

        Computes mel-frequency cepstral coefficients by:
        1. STFT → power spectrum
        2. Mel filterbank application
        3. Log compression
        4. DCT to get cepstral coefficients

        Args:
            audio: Audio signal (float32, mono).
            sample_rate: Sample rate in Hz.

        Returns:
            MFCC matrix (n_mfcc, n_frames).
        """
        if len(audio) == 0:
            return np.zeros((self.n_mfcc, 0), dtype=np.float32)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure minimum length for STFT
        min_len = self.frame_length * 2
        if len(audio) < min_len:
            audio = np.pad(audio, (0, min_len - len(audio)), mode='constant')

        # STFT
        freqs, times, Zxx = stft(
            audio,
            fs=sample_rate,
            nperseg=self.frame_length,
            noverlap=self.frame_length - self.hop_length,
        )

        # Power spectrum
        power = np.abs(Zxx) ** 2

        # Mel filterbank
        mel_fb = self._mel_filterbank(
            n_filters=26,
            n_fft=power.shape[0],
            sample_rate=sample_rate,
            fmin=0,
            fmax=sample_rate / 2,
        )

        # Apply mel filterbank
        mel_spectrum = mel_fb @ power

        # Log compression (avoid log(0))
        log_mel = np.log(mel_spectrum + 1e-10)

        # DCT to get MFCC
        mfcc = self._dct(log_mel, n_coeffs=self.n_mfcc)

        return mfcc.astype(np.float32)

    def _mel_filterbank(
        self,
        n_filters: int,
        n_fft: int,
        sample_rate: int,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ) -> np.ndarray:
        """
        Create a mel-spaced triangular filterbank.

        Args:
            n_filters: Number of mel filters.
            n_fft: FFT size (number of frequency bins).
            sample_rate: Audio sample rate.
            fmin: Minimum frequency.
            fmax: Maximum frequency.

        Returns:
            Filterbank matrix (n_filters, n_fft//2 + 1).
        """
        if fmax is None:
            fmax = sample_rate / 2.0

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        # Mel-spaced center frequencies
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
        hz_points = mel_to_hz(mel_points)

        # FFT bin indices
        freq_bins = np.fft.rfftfreq(n_fft * 2 - 1, d=1.0 / sample_rate)
        n_freq_bins = len(freq_bins)

        # Build filterbank
        filterbank = np.zeros((n_filters, n_freq_bins), dtype=np.float32)

        for i in range(n_filters):
            f_low = hz_points[i]
            f_center = hz_points[i + 1]
            f_high = hz_points[i + 2]

            for j, freq in enumerate(freq_bins):
                if f_low <= freq <= f_center and f_center > f_low:
                    filterbank[i, j] = (freq - f_low) / (f_center - f_low)
                elif f_center < freq <= f_high and f_high > f_center:
                    filterbank[i, j] = (f_high - freq) / (f_high - f_center)

        return filterbank

    def _dct(
        self,
        x: np.ndarray,
        n_coeffs: int,
    ) -> np.ndarray:
        """
        Compute Type-II DCT (Discrete Cosine Transform).

        Args:
            x: Input matrix (n_features, n_frames).
            n_coeffs: Number of DCT coefficients to keep.

        Returns:
            DCT coefficients matrix (n_coeffs, n_frames).
        """
        n_features = x.shape[0]
        n = min(n_coeffs, n_features)

        # DCT basis
        k = np.arange(n).reshape(-1, 1)
        n_idx = np.arange(n_features).reshape(1, -1)

        dct_basis = np.cos(np.pi * k * (2 * n_idx + 1) / (2 * n_features))

        # Apply DCT to each frame
        dct_result = dct_basis @ x

        return dct_result.astype(np.float32)

    def compute_embedding(self, mfcc: np.ndarray) -> np.ndarray:
        """
        Compute speaker embedding from MFCC features.

        Uses the mean and standard deviation of MFCC coefficients
        across time frames as the speaker embedding vector.

        Args:
            mfcc: MFCC matrix (n_mfcc, n_frames).

        Returns:
            Speaker embedding vector (n_mfcc * 2,).
        """
        if mfcc.shape[1] == 0:
            return np.zeros(self.n_mfcc * 2, dtype=np.float32)

        mean_feat = np.mean(mfcc, axis=1)
        std_feat = np.std(mfcc, axis=1)
        embedding = np.concatenate([mean_feat, std_feat])

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Args:
            a: First embedding vector.
            b: Second embedding vector.

        Returns:
            Cosine similarity in [-1, 1].
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _assign_speaker(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Assign an embedding to an existing speaker or create a new one.

        Args:
            embedding: Speaker embedding vector.

        Returns:
            Tuple of (speaker_id, similarity_score).
        """
        if not self._speakers:
            speaker_id = f"Speaker_{self._next_speaker_id}"
            self._next_speaker_id += 1
            self._speakers[speaker_id] = SpeakerProfile(
                speaker_id=speaker_id,
                embedding=embedding,
            )
            return speaker_id, 1.0

        # Compare against all known speakers
        best_speaker = None
        best_similarity = -1.0

        for sid, profile in self._speakers.items():
            sim = self.cosine_similarity(embedding, profile.embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_speaker = sid

        if best_similarity >= self.similarity_threshold and best_speaker:
            # Update speaker embedding (running average)
            profile = self._speakers[best_speaker]
            count = profile.segment_count
            alpha = 1.0 / (count + 1)
            profile.embedding = (
                (1 - alpha) * profile.embedding + alpha * embedding
            )
            # Re-normalize
            norm = np.linalg.norm(profile.embedding)
            if norm > 1e-10:
                profile.embedding = profile.embedding / norm
            profile.segment_count += 1
            return best_speaker, best_similarity
        else:
            # New speaker
            speaker_id = f"Speaker_{self._next_speaker_id}"
            self._next_speaker_id += 1
            self._speakers[speaker_id] = SpeakerProfile(
                speaker_id=speaker_id,
                embedding=embedding,
            )
            return speaker_id, 1.0

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on an audio signal.

        Splits audio into fixed-duration segments, extracts MFCC
        embeddings for each, and clusters them into speakers.

        Args:
            audio: Audio signal (float32, mono).
            sample_rate: Sample rate in Hz.

        Returns:
            List of SpeakerSegment with speaker IDs and time ranges.
        """
        if len(audio) == 0:
            return []

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Reset speaker registry for fresh diarization
        self._speakers.clear()
        self._next_speaker_id = 1

        # Compute segment size in samples
        segment_samples = int(sample_rate * self.segment_duration_ms / 1000)
        total_segments = max(1, int(np.ceil(len(audio) / segment_samples)))

        segments: List[SpeakerSegment] = []

        for i in range(total_segments):
            start_sample = i * segment_samples
            end_sample = min(start_sample + segment_samples, len(audio))
            segment_audio = audio[start_sample:end_sample]

            # Skip very short segments
            if len(segment_audio) < self.frame_length:
                continue

            # Extract MFCC and compute embedding
            mfcc = self.extract_mfcc(segment_audio, sample_rate)
            embedding = self.compute_embedding(mfcc)

            # Assign to speaker
            speaker_id, confidence = self._assign_speaker(embedding)

            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate

            segments.append(SpeakerSegment(
                start_time=start_time,
                end_time=end_time,
                speaker_id=speaker_id,
                audio=segment_audio,
                confidence=confidence,
            ))

        # Merge consecutive segments with same speaker
        merged = self._merge_consecutive(segments)

        # Update speaker profiles total duration
        for seg in merged:
            if seg.speaker_id in self._speakers:
                self._speakers[seg.speaker_id].total_duration += seg.duration

        return merged

    def _merge_consecutive(
        self,
        segments: List[SpeakerSegment],
    ) -> List[SpeakerSegment]:
        """
        Merge consecutive segments assigned to the same speaker.

        Args:
            segments: Original segments list.

        Returns:
            Merged segments list.
        """
        if not segments:
            return []

        merged: List[SpeakerSegment] = []
        current = segments[0]

        for seg in segments[1:]:
            if seg.speaker_id == current.speaker_id:
                # Extend current segment
                current = SpeakerSegment(
                    start_time=current.start_time,
                    end_time=seg.end_time,
                    speaker_id=current.speaker_id,
                    audio=(
                        np.concatenate([current.audio, seg.audio])
                        if current.audio is not None and seg.audio is not None
                        else None
                    ),
                    confidence=min(current.confidence, seg.confidence),
                )
            else:
                merged.append(current)
                current = seg

        merged.append(current)
        return merged

    def get_speaker_count(self) -> int:
        """Get number of identified speakers."""
        return len(self._speakers)

    def get_speaker_ids(self) -> List[str]:
        """Get list of all identified speaker IDs."""
        return list(self._speakers.keys())

    def get_speaker_profile(self, speaker_id: str) -> Optional[SpeakerProfile]:
        """
        Get profile for a specific speaker.

        Args:
            speaker_id: Speaker identifier.

        Returns:
            SpeakerProfile or None if not found.
        """
        return self._speakers.get(speaker_id)