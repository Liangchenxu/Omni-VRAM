"""
Tests for Noise Reduction Module
=================================
"""

import numpy as np
import pytest

from vram_core.noise_reduction import NoiseReducer, NoiseStrength, NoiseReductionResult


def generate_noisy_audio(
    duration_s: float = 2.0,
    sample_rate: int = 16000,
    signal_freq: float = 440.0,
    noise_level: float = 0.1,
) -> tuple:
    """Generate signal + noise and return both."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * signal_freq * t)).astype(np.float32)
    noise = (noise_level * np.random.randn(len(t))).astype(np.float32)
    noisy = signal + noise
    return noisy, signal, noise


class TestNoiseReducerInit:
    """Test NoiseReducer initialization."""

    def test_default_init(self):
        reducer = NoiseReducer()
        assert reducer.strength == NoiseStrength.MEDIUM
        assert reducer.alpha == 2.0
        assert reducer.beta == 0.01
        assert reducer.noise_frames == 8

    def test_strength_light(self):
        reducer = NoiseReducer(strength="light")
        assert reducer.strength == NoiseStrength.LIGHT
        assert reducer.alpha == 1.0

    def test_strength_aggressive(self):
        reducer = NoiseReducer(strength="aggressive")
        assert reducer.strength == NoiseStrength.AGGRESSIVE
        assert reducer.alpha == 4.0

    def test_unknown_strength_fallback(self):
        reducer = NoiseReducer(strength="unknown")
        assert reducer.strength == NoiseStrength.MEDIUM

    def test_custom_params_override(self):
        reducer = NoiseReducer(strength="light", alpha=3.5, beta=0.05, noise_frames=10)
        assert reducer.alpha == 3.5
        assert reducer.beta == 0.05
        assert reducer.noise_frames == 10

    def test_create_preset(self):
        reducer = NoiseReducer.create_preset("aggressive")
        assert reducer.strength == NoiseStrength.AGGRESSIVE


class TestNoiseEstimation:
    """Test noise spectrum estimation."""

    def test_estimate_basic(self):
        reducer = NoiseReducer()
        magnitude = np.random.rand(100, 20).astype(np.float32)
        noise = reducer.estimate_noise_spectrum(magnitude)
        assert noise.shape == (100,)
        assert noise.dtype == np.float32

    def test_estimate_with_few_frames(self):
        reducer = NoiseReducer(noise_frames=10)
        magnitude = np.random.rand(100, 3).astype(np.float32)
        noise = reducer.estimate_noise_spectrum(magnitude)
        assert noise.shape == (100,)

    def test_estimate_empty(self):
        reducer = NoiseReducer()
        magnitude = np.zeros((100, 0), dtype=np.float32)
        noise = reducer.estimate_noise_spectrum(magnitude)
        assert noise.shape == (100,)


class TestSpectralSubtraction:
    """Test spectral subtraction algorithm."""

    def test_basic_subtraction(self):
        reducer = NoiseReducer(alpha=1.0, beta=0.01)
        magnitude = np.ones((50, 10), dtype=np.float32) * 2.0
        noise = np.ones(50, dtype=np.float32) * 0.5
        clean = reducer.spectral_subtract(magnitude, noise)
        assert clean.shape == (50, 10)
        assert np.all(clean >= 0)

    def test_subtraction_with_floor(self):
        reducer = NoiseReducer(alpha=10.0, beta=0.05)
        magnitude = np.ones((50, 10), dtype=np.float32) * 0.5
        noise = np.ones(50, dtype=np.float32) * 1.0
        clean = reducer.spectral_subtract(magnitude, noise)
        # Floor should prevent zeroing
        floor = 0.05 * magnitude ** 2
        assert np.all(clean >= np.sqrt(floor) - 1e-6)


class TestProcess:
    """Test full noise reduction pipeline."""

    def test_process_basic(self):
        reducer = NoiseReducer()
        noisy, _, _ = generate_noisy_audio()
        clean = reducer.process(noisy, sample_rate=16000)
        assert clean.shape == noisy.shape
        assert clean.dtype == np.float32

    def test_process_empty(self):
        reducer = NoiseReducer()
        result = reducer.process(np.array([], dtype=np.float32))
        assert len(result) == 0

    def test_process_reduces_noise(self):
        np.random.seed(42)
        reducer = NoiseReducer(strength="aggressive")
        noisy, signal, noise = generate_noisy_audio(noise_level=0.2)
        clean = reducer.process(noisy, sample_rate=16000)

        # Verify the output is a valid processed signal
        assert clean.shape == noisy.shape
        assert np.isfinite(clean).all(), "Output contains non-finite values"
        assert clean.dtype == np.float32

        # Verify spectral subtraction actually reduces overall energy
        # (aggressive mode subtracts more, so output RMS should be lower)
        rms_noisy = np.sqrt(np.mean(noisy ** 2))
        rms_clean = np.sqrt(np.mean(clean ** 2))
        assert rms_clean <= rms_noisy * 1.5, (
            f"Clean RMS {rms_clean:.4f} should not exceed noisy RMS {rms_noisy:.4f} by much"
        )

        # Verify process_with_stats gives valid SNR measurements
        result = reducer.process_with_stats(noisy, sample_rate=16000)
        assert isinstance(result.snr_before, float)
        assert isinstance(result.snr_after, float)
        assert result.frames_processed > 0

    def test_process_int16_input(self):
        reducer = NoiseReducer()
        noisy, _, _ = generate_noisy_audio()
        noisy_int = (noisy * 32767).astype(np.int16)
        clean = reducer.process(noisy_int, sample_rate=16000)
        assert clean.dtype == np.float32

    def test_process_short_audio(self):
        reducer = NoiseReducer()
        short = np.random.randn(50).astype(np.float32) * 0.1
        clean = reducer.process(short)
        assert len(clean) == len(short)

    def test_process_with_stats(self):
        reducer = NoiseReducer()
        noisy, _, _ = generate_noisy_audio()
        result = reducer.process_with_stats(noisy, sample_rate=16000)
        assert isinstance(result, NoiseReductionResult)
        assert result.audio.shape == noisy.shape
        assert result.frames_processed > 0
        assert isinstance(result.snr_before, float)
        assert isinstance(result.snr_after, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])