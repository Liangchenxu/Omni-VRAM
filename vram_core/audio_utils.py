"""
Audio Format Processing Utilities for vram_core
================================================

Handles audio format detection, conversion, resampling, and
channel manipulation. Designed for zero-copy compatibility with
CUDA memory pipelines.

Dependencies:
    - numpy
    - soundfile (for FLAC/OGG/WAV reading)
    - pydub (for MP3 support)
"""

import io
import struct
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = {"wav", "mp3", "ogg", "flac", "raw"}

# Standard sample rates commonly used in ASR
COMMON_SAMPLE_RATES = [8000, 16000, 22050, 44100, 48000, 96000]


class AudioFormatError(Exception):
    """Raised when audio format is unsupported or corrupted."""
    pass


class AudioProcessor:
    """
    Unified audio format processor.

    Provides automatic format detection, sample rate conversion,
    stereo-to-mono conversion, and normalization. Outputs data
    compatible with CUDA zero-copy pipelines.
    """

    def __init__(self, target_sample_rate: int = 16000, target_channels: int = 1):
        """
        Args:
            target_sample_rate: Desired output sample rate (default 16000 for Whisper).
            target_channels: Desired output channel count (default 1 = mono).
        """
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels

    # ------------------------------------------------------------------
    # Format Detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_format(file_path: Union[str, Path]) -> str:
        """
        Detect audio format from file magic bytes and extension.

        Args:
            file_path: Path to the audio file.

        Returns:
            Detected format string (e.g. 'wav', 'mp3', 'flac', 'ogg').
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Try magic bytes first
        with open(file_path, "rb") as f:
            header = f.read(12)

        fmt = AudioProcessor._detect_from_magic(header)
        if fmt:
            return fmt

        # Fallback to extension
        ext = file_path.suffix.lstrip(".").lower()
        if ext in SUPPORTED_FORMATS:
            return ext

        raise AudioFormatError(
            f"Cannot detect format for '{file_path}'. "
            f"Supported formats: {SUPPORTED_FORMATS}"
        )

    @staticmethod
    def _detect_from_magic(header: bytes) -> Optional[str]:
        """Detect format from file header magic bytes."""
        if len(header) < 4:
            return None

        # WAV: RIFF....WAVE
        if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
            return "wav"

        # FLAC: fLaC
        if header[:4] == b"fLaC":
            return "flac"

        # OGG: OggS
        if header[:4] == b"OggS":
            return "ogg"

        # MP3: ID3 tag or frame sync (0xFF 0xFB / 0xFF 0xF3 / 0xFF 0xF2)
        if header[:3] == b"ID3":
            return "mp3"
        if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
            return "mp3"

        return None

    @staticmethod
    def detect_sample_rate(file_path: Union[str, Path]) -> int:
        """
        Detect the sample rate of an audio file.

        Args:
            file_path: Path to the audio file.

        Returns:
            Sample rate in Hz.
        """
        file_path = Path(file_path)
        fmt = AudioProcessor.detect_format(file_path)

        if fmt == "wav":
            return AudioProcessor._wav_sample_rate(file_path)

        # For other formats, use soundfile
        try:
            import soundfile as sf
            info = sf.info(str(file_path))
            return info.samplerate
        except ImportError:
            raise ImportError(
                "soundfile is required for non-WAV format detection. "
                "Install it with: pip install soundfile"
            )

    @staticmethod
    def _wav_sample_rate(file_path: Path) -> int:
        """Extract sample rate from WAV header."""
        with open(file_path, "rb") as f:
            header = f.read(44)  # Standard WAV header size
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise AudioFormatError("Invalid WAV file header")
        sample_rate = struct.unpack_from("<I", header, 24)[0]
        return sample_rate

    # ------------------------------------------------------------------
    # Loading & Conversion
    # ------------------------------------------------------------------

    def load(
        self,
        file_path: Union[str, Path],
        normalize: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Load an audio file, convert to target format, and return as numpy array.

        Args:
            file_path: Path to the audio file.
            normalize: If True, normalize to [-1.0, 1.0] range.

        Returns:
            Tuple of (audio_data as float32 numpy array, sample_rate).
        """
        file_path = Path(file_path)
        fmt = self.detect_format(file_path)

        # Load raw audio data
        audio, sr = self._load_raw(file_path, fmt)

        # Convert to float32
        audio = self._to_float32(audio)

        # Stereo to mono (also flatten single-channel 2D arrays)
        if audio.ndim > 1:
            if audio.shape[1] > 1:
                audio = self.stereo_to_mono(audio)
            else:
                audio = audio.flatten()

        # Resample if needed
        if sr != self.target_sample_rate:
            audio = self.resample(audio, sr, self.target_sample_rate)
            sr = self.target_sample_rate

        # Normalize
        if normalize:
            audio = self.normalize(audio)

        logger.info(
            "Loaded '%s': %d samples, %d Hz, mono, float32",
            file_path.name, len(audio), sr,
        )
        return audio, sr

    def load_from_bytes(
        self,
        data: bytes,
        format_hint: Optional[str] = None,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio from raw bytes (e.g. from network stream).

        Args:
            data: Raw audio bytes.
            format_hint: Optional format hint ('wav', 'mp3', etc.).
            normalize: If True, normalize to [-1.0, 1.0].

        Returns:
            Tuple of (audio_data as float32 numpy array, sample_rate).
        """
        if format_hint is None:
            # Try magic bytes
            header = data[:12]
            fmt = AudioProcessor._detect_from_magic(header)
            if fmt is None:
                raise AudioFormatError("Cannot detect format from bytes. Provide format_hint.")
        else:
            fmt = format_hint.lower()

        if fmt == "wav":
            audio, sr = self._load_wav_bytes(data)
        else:
            try:
                import soundfile as sf
                audio, sr = sf.read(io.BytesIO(data))
            except ImportError:
                raise ImportError("soundfile is required for non-WAV byte loading.")

        audio = self._to_float32(audio)

        # Stereo to mono (also flatten single-channel 2D arrays)
        if audio.ndim > 1:
            if audio.shape[1] > 1:
                audio = self.stereo_to_mono(audio)
            else:
                audio = audio.flatten()

        if sr != self.target_sample_rate:
            audio = self.resample(audio, sr, self.target_sample_rate)
            sr = self.target_sample_rate

        if normalize:
            audio = self.normalize(audio)

        return audio, sr

    def _load_raw(
        self, file_path: Path, fmt: str
    ) -> Tuple[np.ndarray, int]:
        """Load raw audio data based on detected format."""
        if fmt == "wav":
            return self._load_wav(file_path)

        try:
            import soundfile as sf
            return sf.read(str(file_path))
        except ImportError:
            pass

        # Fallback: try pydub for mp3
        if fmt == "mp3":
            return self._load_mp3_pydub(file_path)

        raise ImportError(
            f"Cannot load '{fmt}' format. Install soundfile or pydub: "
            "pip install soundfile pydub"
        )

    @staticmethod
    def _load_wav(file_path: Path) -> Tuple[np.ndarray, int]:
        """Load WAV file using pure Python + struct parsing."""
        with open(file_path, "rb") as f:
            data = f.read()

        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise AudioFormatError("Invalid WAV file")

        # Parse fmt chunk
        num_channels = struct.unpack_from("<H", data, 22)[0]
        sample_rate = struct.unpack_from("<I", data, 24)[0]
        bits_per_sample = struct.unpack_from("<H", data, 34)[0]

        # Find data chunk
        data_offset = AudioProcessor._find_chunk(data, b"data", 12)
        if data_offset < 0:
            raise AudioFormatError("WAV: 'data' chunk not found")

        chunk_size = struct.unpack_from("<I", data, data_offset + 4)[0]
        raw = data[data_offset + 8 : data_offset + 8 + chunk_size]

        # Parse samples
        dtype_map = {16: np.int16, 24: np.int32, 32: np.int32}
        if bits_per_sample not in dtype_map:
            raise AudioFormatError(f"Unsupported WAV bit depth: {bits_per_sample}")

        if bits_per_sample == 24:
            # 24-bit: manually convert to 32-bit
            n_samples = len(raw) // 3
            samples = np.zeros(n_samples, dtype=np.int32)
            for i in range(n_samples):
                b0 = raw[i * 3]
                b1 = raw[i * 3 + 1]
                b2 = raw[i * 3 + 2]
                val = b0 | (b1 << 8) | (b2 << 16)
                if val >= 0x800000:
                    val -= 0x1000000
                samples[i] = val << 8
        else:
            samples = np.frombuffer(raw, dtype=dtype_map[bits_per_sample])

        if num_channels > 1:
            samples = samples.reshape(-1, num_channels)

        return samples, sample_rate

    @staticmethod
    def _load_wav_bytes(data: bytes) -> Tuple[np.ndarray, int]:
        """Load WAV from bytes without using file system."""
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise AudioFormatError("Invalid WAV data")

        num_channels = struct.unpack_from("<H", data, 22)[0]
        sample_rate = struct.unpack_from("<I", data, 24)[0]
        bits_per_sample = struct.unpack_from("<H", data, 34)[0]

        data_offset = AudioProcessor._find_chunk(data, b"data", 12)
        if data_offset < 0:
            raise AudioFormatError("WAV: 'data' chunk not found")

        chunk_size = struct.unpack_from("<I", data, data_offset + 4)[0]
        raw = data[data_offset + 8 : data_offset + 8 + chunk_size]

        dtype_map = {16: np.int16, 24: np.int32, 32: np.int32}
        if bits_per_sample not in dtype_map:
            raise AudioFormatError(f"Unsupported WAV bit depth: {bits_per_sample}")

        if bits_per_sample == 24:
            n_samples = len(raw) // 3
            samples = np.zeros(n_samples, dtype=np.int32)
            for i in range(n_samples):
                b0 = raw[i * 3]
                b1 = raw[i * 3 + 1]
                b2 = raw[i * 3 + 2]
                val = b0 | (b1 << 8) | (b2 << 16)
                if val >= 0x800000:
                    val -= 0x1000000
                samples[i] = val << 8
        else:
            samples = np.frombuffer(raw, dtype=dtype_map[bits_per_sample])

        if num_channels > 1:
            samples = samples.reshape(-1, num_channels)

        return samples, sample_rate

    @staticmethod
    def _load_mp3_pydub(file_path: Path) -> Tuple[np.ndarray, int]:
        """Load MP3 using pydub as fallback."""
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("pydub is required for MP3 support. Install with: pip install pydub")

        audio_segment = AudioSegment.from_mp3(str(file_path))
        samples = np.array(audio_segment.get_array_of_samples())
        sr = audio_segment.frame_rate
        channels = audio_segment.channels

        if channels > 1:
            samples = samples.reshape(-1, channels)

        return samples, sr

    @staticmethod
    def _find_chunk(data: bytes, chunk_id: bytes, start: int) -> int:
        """Find a chunk in WAV data by its 4-byte ID."""
        offset = start
        while offset < len(data) - 8:
            if data[offset : offset + 4] == chunk_id:
                return offset
            chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
            offset += 8 + chunk_size
        return -1

    # ------------------------------------------------------------------
    # Conversion Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float32(audio: np.ndarray) -> np.ndarray:
        """Convert integer audio to float32 [-1.0, 1.0]."""
        if audio.dtype == np.float32:
            return audio
        if audio.dtype == np.float64:
            return audio.astype(np.float32)
        if audio.dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        if audio.dtype == np.int32:
            # Use float64 intermediate to avoid precision loss
            # (float32 only has 24-bit mantissa, int32 needs 32-bit)
            return (audio.astype(np.float64) / 2147483648.0).astype(np.float32)
        if audio.dtype == np.uint8:
            return (audio.astype(np.float32) - 128.0) / 128.0
        return audio.astype(np.float32)

    @staticmethod
    def stereo_to_mono(audio: np.ndarray) -> np.ndarray:
        """
        Convert stereo (or multi-channel) audio to mono by averaging channels.

        Args:
            audio: Input array with shape (samples, channels).

        Returns:
            Mono audio array with shape (samples,).
        """
        if audio.ndim == 1:
            return audio
        return np.mean(audio, axis=1).astype(np.float32)

    @staticmethod
    def resample(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Uses linear interpolation. For production quality, consider
        integrating libsamplerate or torchaudio.

        Args:
            audio: Input audio (1D float32 array).
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            Resampled audio array.
        """
        if orig_sr == target_sr:
            return audio

        try:
            import torch
            import torchaudio.functional as F
            # Use torchaudio's high-quality resampler
            tensor = torch.from_numpy(audio).unsqueeze(0)
            resampled = F.resample(tensor, orig_sr, target_sr)
            return resampled.squeeze(0).numpy()
        except ImportError:
            pass

        # Fallback: linear interpolation (lower quality but no extra deps)
        duration = len(audio) / orig_sr
        n_target = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, n_target)
        resampled = np.interp(indices, np.arange(len(audio)), audio)
        return resampled.astype(np.float32)

    @staticmethod
    def normalize(audio: np.ndarray, peak: float = 1.0) -> np.ndarray:
        """
        Normalize audio to peak amplitude.

        Args:
            audio: Input audio array.
            peak: Target peak amplitude (default 1.0).

        Returns:
            Normalized audio array.
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = (audio / max_val * peak).astype(np.float32)
        return audio

    # ------------------------------------------------------------------
    # CUDA-Ready Buffer
    # ------------------------------------------------------------------

    def to_cuda_buffer(self, audio: np.ndarray) -> "torch.Tensor":
        """
        Convert numpy audio to a CUDA tensor for zero-copy processing.

        Args:
            audio: float32 numpy array.

        Returns:
            CUDA tensor (float32) ready for vram_core kernels.
        """
        import torch
        return torch.from_numpy(audio).to(device="cuda", dtype=torch.float32)

    # ------------------------------------------------------------------
    # WAV Export
    # ------------------------------------------------------------------

    @staticmethod
    def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
        """
        Encode float32 audio to WAV bytes.

        Args:
            audio: float32 mono audio array.
            sample_rate: Sample rate in Hz.

        Returns:
            WAV file bytes.
        """
        try:
            import soundfile as sf
            buf = io.BytesIO()
            sf.write(buf, audio, sample_rate, format="WAV")
            return buf.getvalue()
        except ImportError:
            pass

        # Pure Python WAV encoding fallback
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        int16_audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        n = len(int16_audio)
        data_size = n * 2
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,  # PCM
            1,  # mono
            sample_rate,
            sample_rate * 2,
            2,
            16,
            b"data",
            data_size,
        )
        return header + int16_audio.tobytes()

    # ------------------------------------------------------------------
    # Save from Bytes
    # ------------------------------------------------------------------

    @staticmethod
    def save_bytes(path: Union[str, Path], audio_bytes: bytes, format: str = "wav") -> None:
        """
        Save raw audio bytes to a file by decoding and re-encoding.

        Args:
            path: Output file path.
            audio_bytes: Raw audio bytes.
            format: Output format (default: WAV).
        """
        try:
            import soundfile as sf
            import io as _io

            buf = _io.BytesIO(audio_bytes)
            with sf.SoundFile(buf) as f:
                data = f.read(dtype="float32")
                sr = f.samplerate
            sf.write(str(path), data, sr, format=format.upper())
        except ImportError:
            raise ImportError("soundfile is required to save from bytes.")
