"""
Unit Tests for vram_core Audio Utilities
=========================================

Tests audio format detection, conversion, resampling, and
stereo-to-mono functionality.
"""

import os
import struct
import tempfile
import pytest
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vram_core.audio_utils import AudioProcessor, AudioFormatError, SUPPORTED_FORMATS


class TestAudioFormatDetection:
    """Test audio format detection from magic bytes."""

    def test_wav_magic_detection(self):
        """Test WAV format detection from RIFF header."""
        header = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 4
        result = AudioProcessor._detect_from_magic(header)
        assert result == "wav"

    def test_flac_magic_detection(self):
        """Test FLAC format detection from fLaC header."""
        header = b"fLaC\x00\x00\x00\x22" + b"\x00" * 4
        result = AudioProcessor._detect_from_magic(header)
        assert result == "flac"

    def test_ogg_magic_detection(self):
        """Test OGG format detection from OggS header."""
        header = b"OggS\x00\x02\x00\x00" + b"\x00" * 4
        result = AudioProcessor._detect_from_magic(header)
        assert result == "ogg"

    def test_mp3_id3_magic_detection(self):
        """Test MP3 format detection from ID3 tag."""
        header = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 2
        result = AudioProcessor._detect_from_magic(header)
        assert result == "mp3"

    def test_mp3_sync_magic_detection(self):
        """Test MP3 format detection from frame sync bytes."""
        header = bytes([0xFF, 0xFB, 0x90, 0x00]) + b"\x00" * 8
        result = AudioProcessor._detect_from_magic(header)
        assert result == "mp3"

    def test_unknown_magic(self):
        """Test unknown format returns None."""
        header = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        result = AudioProcessor._detect_from_magic(header)
        assert result is None

    def test_short_header(self):
        """Test short header returns None."""
        header = b"\x00\x00"
        result = AudioProcessor._detect_from_magic(header)
        assert result is None


class TestAudioFormatDetectionFile:
    """Test audio format detection from files."""

    def _create_temp_wav(self, sample_rate=16000, duration_s=1.0, channels=1):
        """Create a temporary WAV file for testing."""
        n_samples = int(sample_rate * duration_s)
        samples = np.random.randint(-32768, 32767, size=n_samples * channels, dtype=np.int16)

        data_size = len(samples) * 2
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,  # PCM
            channels,
            sample_rate,
            sample_rate * channels * 2,
            channels * 2,
            16,
            b"data",
            data_size,
        )

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(header)
        tmp.write(samples.tobytes())
        tmp.close()
        return tmp.name

    def test_detect_wav_format(self):
        """Test WAV format detection from file."""
        path = self._create_temp_wav()
        try:
            fmt = AudioProcessor.detect_format(path)
            assert fmt == "wav"
        finally:
            os.unlink(path)

    def test_detect_format_nonexistent_file(self):
        """Test format detection raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            AudioProcessor.detect_format("/nonexistent/file.wav")

    def test_detect_sample_rate_wav(self):
        """Test sample rate detection from WAV file."""
        path = self._create_temp_wav(sample_rate=44100)
        try:
            sr = AudioProcessor.detect_sample_rate(path)
            assert sr == 44100
        finally:
            os.unlink(path)


class TestWAVLoading:
    """Test WAV file loading."""

    def _create_temp_wav(self, sample_rate=16000, duration_s=1.0, channels=1):
        """Create a temporary WAV file."""
        n_samples = int(sample_rate * duration_s)
        samples = np.random.randint(-32768, 32767, size=n_samples * channels, dtype=np.int16)

        data_size = len(samples) * 2
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            channels,
            sample_rate,
            sample_rate * channels * 2,
            channels * 2,
            16,
            b"data",
            data_size,
        )

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(header)
        tmp.write(samples.tobytes())
        tmp.close()
        return tmp.name

    def test_load_mono_wav(self):
        """Test loading mono WAV file."""
        path = self._create_temp_wav(sample_rate=16000, duration_s=1.0, channels=1)
        try:
            proc = AudioProcessor(target_sample_rate=16000)
            audio, sr = proc.load(path, normalize=False)
            assert sr == 16000
            assert len(audio) == 16000
            assert audio.dtype == np.float32
        finally:
            os.unlink(path)

    def test_load_stereo_wav_converts_to_mono(self):
        """Test loading stereo WAV converts to mono."""
        path = self._create_temp_wav(sample_rate=16000, duration_s=1.0, channels=2)
        try:
            proc = AudioProcessor(target_sample_rate=16000)
            audio, sr = proc.load(path, normalize=False)
            assert sr == 16000
            assert audio.ndim == 1  # Mono
            assert len(audio) == 16000
        finally:
            os.unlink(path)

    def test_load_wav_with_resample(self):
        """Test loading WAV with resampling."""
        path = self._create_temp_wav(sample_rate=44100, duration_s=1.0, channels=1)
        try:
            proc = AudioProcessor(target_sample_rate=16000)
            audio, sr = proc.load(path, normalize=False)
            assert sr == 16000
            # Allow some tolerance in resampled length
            expected_len = 16000
            assert abs(len(audio) - expected_len) <= 100
        finally:
            os.unlink(path)

    def test_load_wav_normalized(self):
        """Test that normalization works correctly."""
        path = self._create_temp_wav(sample_rate=16000, duration_s=0.5, channels=1)
        try:
            proc = AudioProcessor(target_sample_rate=16000)
            audio, _ = proc.load(path, normalize=True)
            max_val = np.max(np.abs(audio))
            assert round(abs(max_val - 1.0), 5) == 0
        finally:
            os.unlink(path)


class TestConversionUtilities:
    """Test audio conversion utilities."""

    def test_stereo_to_mono(self):
        """Test stereo to mono conversion."""
        stereo = np.array([[0.5, 0.3], [0.8, 0.2], [-0.1, 0.9]], dtype=np.float32)
        mono = AudioProcessor.stereo_to_mono(stereo)
        expected = np.mean(stereo, axis=1)
        np.testing.assert_array_almost_equal(mono, expected)
        assert mono.dtype == np.float32

    def test_stereo_to_mono_already_mono(self):
        """Test that mono input passes through unchanged."""
        mono = np.array([0.5, 0.3, 0.8], dtype=np.float32)
        result = AudioProcessor.stereo_to_mono(mono)
        np.testing.assert_array_equal(result, mono)

    def test_to_float32_from_int16(self):
        """Test int16 to float32 conversion."""
        int16_data = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        result = AudioProcessor._to_float32(int16_data)
        assert result.dtype == np.float32
        assert round(abs(result[0] - 0.0), 5) == 0
        assert round(abs(result[3] - 32767 / 32768.0), 3) == 0

    def test_to_float32_from_int32(self):
        """Test int32 to float32 conversion."""
        int32_data = np.array([0, 1073741824, -1073741824], dtype=np.int32)
        result = AudioProcessor._to_float32(int32_data)
        assert result.dtype == np.float32

    def test_to_float32_already_float32(self):
        """Test float32 passes through unchanged."""
        data = np.array([0.5, -0.5, 1.0], dtype=np.float32)
        result = AudioProcessor._to_float32(data)
        np.testing.assert_array_equal(result, data)

    def test_normalize(self):
        """Test audio normalization."""
        audio = np.array([0.1, -0.5, 0.3, -0.2], dtype=np.float32)
        result = AudioProcessor.normalize(audio, peak=1.0)
        assert round(abs(np.max(np.abs(result)) - 1.0), 5) == 0

    def test_normalize_silent_audio(self):
        """Test normalization of silent audio."""
        audio = np.zeros(100, dtype=np.float32)
        result = AudioProcessor.normalize(audio)
        np.testing.assert_array_equal(result, audio)

    def test_resample_same_rate(self):
        """Test resample with same input and output rate."""
        audio = np.random.randn(16000).astype(np.float32)
        result = AudioProcessor.resample(audio, 16000, 16000)
        np.testing.assert_array_equal(result, audio)

    def test_resample_downsample(self):
        """Test downsampling."""
        audio = np.random.randn(44100).astype(np.float32)
        result = AudioProcessor.resample(audio, 44100, 16000)
        expected_len = int(44100 / 44100 * 16000)
        assert abs(len(result) - expected_len) <= 10
        assert result.dtype == np.float32

    def test_resample_upsample(self):
        """Test upsampling."""
        audio = np.random.randn(8000).astype(np.float32)
        result = AudioProcessor.resample(audio, 8000, 16000)
        expected_len = 16000
        assert abs(len(result) - expected_len) <= 10


class TestWAVExport:
    """Test WAV encoding."""

    def test_to_wav_bytes(self):
        """Test encoding audio to WAV bytes."""
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        wav_bytes = AudioProcessor.to_wav_bytes(audio, 16000)
        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 44  # At least header + some data
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"

    def test_wav_bytes_roundtrip(self):
        """Test that WAV bytes can be loaded back."""
        # Use small values that stay within [-1, 1] to avoid clipping
        original = np.random.randn(8000).astype(np.float32) * 0.1
        wav_bytes = AudioProcessor.to_wav_bytes(original, 16000)

        proc = AudioProcessor(target_sample_rate=16000)
        loaded, sr = proc.load_from_bytes(wav_bytes, normalize=False)
        assert sr == 16000
        # Allow small precision loss from int16 conversion
        np.testing.assert_array_almost_equal(loaded, original, decimal=3)


class TestByteLoading:
    """Test loading audio from bytes."""

    def test_load_from_wav_bytes(self):
        """Test loading audio from WAV bytes."""
        # Create WAV in memory
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        wav_bytes = AudioProcessor.to_wav_bytes(audio, 16000)

        proc = AudioProcessor(target_sample_rate=16000)
        loaded, sr = proc.load_from_bytes(wav_bytes)
        assert sr == 16000
        assert len(loaded) == 16000
        assert loaded.dtype == np.float32


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_audio(self):
        """Test handling of empty audio."""
        empty = np.array([], dtype=np.float32)
        result = AudioProcessor.stereo_to_mono(empty)
        assert len(result) == 0

    def test_single_sample(self):
        """Test handling of single sample."""
        single = np.array([0.5], dtype=np.float32)
        result = AudioProcessor.normalize(single)
        assert round(abs(result[0] - 1.0), 5) == 0

    def test_silent_audio_stereo_to_mono(self):
        """Test stereo to mono with silent audio."""
        stereo = np.zeros((100, 2), dtype=np.float32)
        mono = AudioProcessor.stereo_to_mono(stereo)
        np.testing.assert_array_equal(mono, np.zeros(100, dtype=np.float32))

    def test_supported_formats_constant(self):
        """Test that supported formats are defined."""
        assert "wav" in SUPPORTED_FORMATS
        assert "mp3" in SUPPORTED_FORMATS
        assert "flac" in SUPPORTED_FORMATS
        assert "ogg" in SUPPORTED_FORMATS


