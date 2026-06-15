"""
AudioPreprocessor
=================

Format conversion and normalization for Whisper input.
Uses pydub as the audio I/O backend.
"""

import io
import logging
import wave
from typing import Optional

import numpy as np

from vram_core.whisper.models import SUPPORTED_AUDIO_FORMATS

logger = logging.getLogger("vram_core.whisper.preprocessor")


class AudioPreprocessor:
    """Handle audio format conversion and preprocessing for Whisper."""

    @staticmethod
    def check_pydub():
        """Ensure pydub is available."""
        try:
            import pydub  # noqa: F401
        except ImportError:
            raise ImportError(
                "pydub is required for audio processing.\n"
                "Install with: pip install pydub\n"
                "Also install ffmpeg: https://ffmpeg.org/download.html"
            )

    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        if audio.size == 0:
            return audio
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    @staticmethod
    def get_audio_info(file_path: str) -> dict:
        """Get audio file metadata without loading the full file."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            return {
                "duration": len(audio) / 1000.0,
                "channels": audio.channels,
                "sample_rate": audio.frame_rate,
                "sample_width": audio.sample_width,
                "format": file_path.rsplit(".", 1)[-1].lower() if "." in file_path else "unknown",
            }
        except Exception as e:
            logger.warning(f"Could not get audio info for {file_path}: {e}")
            return {"error": str(e)}

    @staticmethod
    def load_audio_pydub(
        source,
        target_sr: int = 16000,
        mono: bool = True,
    ) -> np.ndarray:
        """
        Load audio from file path or bytes, resample, convert to float32.

        Args:
            source:     File path (str/Path) or bytes.
            target_sr:  Target sample rate.
            mono:       Convert to mono if True.

        Returns:
            Normalized float32 numpy array in [-1, 1].
        """
        from pydub import AudioSegment

        if isinstance(source, bytes):
            audio = AudioSegment.from_file(io.BytesIO(source))
        else:
            audio = AudioSegment.from_file(str(source))

        if mono and audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
        audio = audio.set_sample_width(2)  # 16-bit

        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0

        if np.max(np.abs(samples)) > 0:
            samples = samples / np.max(np.abs(samples)) * 0.95

        return samples

    @staticmethod
    def load_audio_librosa(
        file_path: str,
        target_sr: int = 16000,
    ) -> np.ndarray:
        """Fallback audio loading using librosa."""
        import librosa
        audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return audio.astype(np.float32)

    @staticmethod
    def to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        """
        Convert float32 audio array to WAV bytes in memory.

        Args:
            audio:       Float32 mono audio array.
            sample_rate: Sample rate.

        Returns:
            WAV file bytes.
        """
        import struct

        audio_16bit = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_16bit.tobytes())
        return buf.getvalue()

    @staticmethod
    def check_format_support(file_path: str) -> bool:
        """Check if the audio format is supported."""
        return any(file_path.lower().endswith(ext) for ext in SUPPORTED_AUDIO_FORMATS)

    @staticmethod
    def _get_audio_segment(source) -> "pydub.AudioSegment":
        """Load a pydub AudioSegment from file path, bytes, or bytes stream."""
        from pydub import AudioSegment

        if isinstance(source, bytes):
            return AudioSegment.from_file(io.BytesIO(source))
        elif isinstance(source, io.BytesIO):
            source.seek(0)
            return AudioSegment.from_file(source)
        else:
            return AudioSegment.from_file(str(source))

    def convert_format(
        self,
        source,
        output_path: Optional[str] = None,
        output_format: str = "wav",
        sample_rate: int = 16000,
        mono: bool = True,
    ) -> bytes:
        """
        Convert audio from any supported format to WAV bytes.

        Args:
            source:        Input file path, bytes, or BytesIO.
            output_path:   Optional file path to write the output.
            output_format: Output format (default: wav).
            sample_rate:   Target sample rate.
            mono:          Convert to mono if True.

        Returns:
            WAV audio bytes.
        """
        self.check_pydub()
        audio_seg = self._get_audio_segment(source)

        if mono:
            audio_seg = audio_seg.set_channels(1)
        if audio_seg.frame_rate != sample_rate:
            audio_seg = audio_seg.set_frame_rate(sample_rate)
        audio_seg = audio_seg.set_sample_width(2)

        buf = io.BytesIO()
        audio_seg.export(buf, format=output_format)
        result = buf.getvalue()

        if output_path:
            with open(output_path, "wb") as f:
                f.write(result)
            logger.info(f"Audio exported to {output_path}")

        return result

    def split_audio(
        self,
        source,
        chunk_duration_ms: int = 30000,
        overlap_ms: int = 1000,
    ) -> list:
        """
        Split audio into overlapping chunks for long-form transcription.

        Args:
            source:          Input file path, bytes, or BytesIO.
            chunk_duration_ms: Chunk duration in ms.
            overlap_ms:      Overlap between chunks in ms.

        Returns:
            List of WAV bytes, one per chunk.
        """
        self.check_pydub()
        audio_seg = self._get_audio_segment(source)
        audio_seg = audio_seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)

        chunks = []
        total_ms = len(audio_seg)
        start = 0

        while start < total_ms:
            end = min(start + chunk_duration_ms, total_ms)
            chunk = audio_seg[start:end]
            buf = io.BytesIO()
            chunk.export(buf, format="wav")
            chunks.append(buf.getvalue())
            start += chunk_duration_ms - overlap_ms

        logger.info(
            f"Split {total_ms / 1000:.1f}s audio into {len(chunks)} chunks "
            f"({chunk_duration_ms / 1000:.0f}s each, {overlap_ms / 1000:.1f}s overlap)"
        )
        return chunks

    def resample_audio(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int = 16000,
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio

        ratio = target_sr / orig_sr
        n_samples = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, n_samples)
        resampled = np.interp(indices, np.arange(len(audio)), audio)
        return resampled.astype(np.float32)