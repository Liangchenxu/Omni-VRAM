"""
Tests for the multi-band adaptive denoiser (v2.6.0).

Covers:
    - MultibandSpectralSuppressor: band mapping, adaptive over-subtraction,
      recursive noise-PSD estimation, decision-directed Wiener gains and the
      musical-noise (gain smoothing) behaviour
    - NoiseReducer integration: MBSS default, ``reduce_noise`` aggressiveness
      mapping, per-band diagnostics, streaming state continuity and the
      unchanged legacy path

Measurement methodology for the SNR assertions: the output SNR is computed by
orthogonally projecting the processed signal onto the clean reference, so a
global gain difference cannot fake an improvement::

    a = <clean, ref> / <ref, ref>
    snr_out = 10 log10( RMS(a * ref)^2 / RMS(clean - a * ref)^2 )
"""

import numpy as np
import pytest

from vram_core.noise_reduction import (
    AlgorithmType,
    MultibandSpectralSuppressor,
    NoiseReducer,
    NoiseReductionResult,
)

SAMPLE_RATE = 16000


# ─── Helpers ────────────────────────────────────────────────────────────────

def tone(duration_s: float, frequency: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def noise(duration_s: float, level: float = 0.1, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (level * rng.standard_normal(int(SAMPLE_RATE * duration_s))).astype(np.float32)


def noisy_speech(seed: int = 7, noise_level: float = 0.1):
    """
    Noise-only lead-in (0.3 s) followed by tone + noise (1.5 s).

    The lead-in models the silence a noise estimator normally calibrates on.
    """
    lead = noise(0.3, noise_level, seed=seed)
    body = tone(1.5) + noise(1.5, noise_level, seed=seed + 1)
    return np.concatenate([lead, body]).astype(np.float32), lead, body


def output_snr_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    """SNR of ``estimate`` w.r.t. ``reference`` using orthogonal projection."""
    ref_energy = float(np.dot(reference, reference)) + 1e-12
    scale = float(np.dot(estimate, reference)) / ref_energy
    signal_part = scale * reference
    residual = estimate - signal_part
    signal_power = float(np.mean(signal_part ** 2)) + 1e-12
    residual_power = float(np.mean(residual ** 2)) + 1e-12
    return float(10.0 * np.log10(signal_power / residual_power))


def random_spectrogram(frames: int = 40, bins: int = 257, seed: int = 3) -> np.ndarray:
    """Speech-like magnitude spectrogram (noise floor + strong formant peaks)."""
    rng = np.random.default_rng(seed)
    magnitude = 0.05 + 0.02 * rng.random((bins, frames))
    for center in (16, 40, 90):
        magnitude[center - 2:center + 3, :] += rng.random((5, frames)) * 0.6
    return magnitude.astype(np.float32)


# ─── Suppressor unit tests ──────────────────────────────────────────────────

class TestBandMapping:
    """Non-uniform band split used by the MBSS core."""

    def test_slices_cover_every_bin_exactly_once(self):
        slices = MultibandSpectralSuppressor.band_slices(257, SAMPLE_RATE, 512)
        assert len(slices) == 3
        covered = [index for band in slices for index in range(band.start, band.stop)]
        assert covered == list(range(257))

    def test_band_edges_follow_hz_boundaries(self):
        slices = MultibandSpectralSuppressor.band_slices(257, SAMPLE_RATE, 512)
        # 512-point STFT @16 kHz -> 31.25 Hz per bin
        assert slices[0].start == 0
        assert slices[1].start == 16     # ~500 Hz
        assert slices[2].start == 64     # ~2000 Hz
        assert slices[2].stop == 257

    def test_degrades_with_tiny_bin_counts(self):
        slices = MultibandSpectralSuppressor.band_slices(3, SAMPLE_RATE, 512)
        covered = [index for band in slices for index in range(band.start, band.stop)]
        assert covered == [0, 1, 2]


class TestAdaptiveOverSubtraction:
    """alpha_i(SNR_i) must protect high-SNR bands and squeeze low-SNR bands."""

    def test_alpha_decreases_with_snr(self):
        suppressor = MultibandSpectralSuppressor(alpha0=2.0, snr_slope=10.0)
        alphas = suppressor.band_over_subtraction(np.array([-10.0, 0.0, 5.0]))
        assert alphas[0] > alphas[1] > alphas[2]

    def test_alpha_is_clamped(self):
        suppressor = MultibandSpectralSuppressor(alpha0=2.0, snr_slope=10.0)
        alphas = suppressor.band_over_subtraction(np.array([-100.0, 100.0]))
        assert alphas[0] == pytest.approx(3.0)          # 1.5 * alpha0 cap
        assert alphas[1] == pytest.approx(1.0)          # never below 1 (no energy add)

    def test_high_snr_band_is_barely_touched(self):
        suppressor = MultibandSpectralSuppressor(alpha0=2.0)
        assert suppressor.band_over_subtraction(np.array([30.0]))[0] == pytest.approx(1.0)


class TestNoisePsdEstimation:
    """Recursive noise PSD with speech-presence guarding."""

    def test_psd_is_initialised_and_tracked(self):
        suppressor = MultibandSpectralSuppressor()
        magnitude = random_spectrogram(frames=12)
        suppressor.process_spectrum(magnitude, sample_rate=SAMPLE_RATE)
        psd = suppressor.noise_psd
        assert psd is not None
        assert psd.shape == magnitude.shape[0]
        assert np.all(psd > 0)

    def test_frames_seen_accumulates_across_calls(self):
        suppressor = MultibandSpectralSuppressor()
        suppressor.process_spectrum(random_spectrogram(frames=5), sample_rate=SAMPLE_RATE)
        assert suppressor.frames_seen == 5
        suppressor.process_spectrum(random_spectrogram(frames=4), sample_rate=SAMPLE_RATE)
        assert suppressor.frames_seen == 9

    def test_reset_clears_psd(self):
        suppressor = MultibandSpectralSuppressor()
        suppressor.process_spectrum(random_spectrogram(frames=4), sample_rate=SAMPLE_RATE)
        suppressor.reset()
        assert suppressor.noise_psd is None
        assert suppressor.frames_seen == 0

    def test_psd_adapts_downwards_on_pure_noise(self):
        suppressor = MultibandSpectralSuppressor()
        loud = np.full((32, 6), 0.5, dtype=np.float32)
        suppressor.process_spectrum(loud, sample_rate=SAMPLE_RATE)
        quiet = np.full((32, 20), 0.01, dtype=np.float32)
        suppressor.process_spectrum(quiet, sample_rate=SAMPLE_RATE)
        assert float(np.mean(suppressor.noise_psd)) < 0.25


class TestWienerGainAndSmoothing:
    """Decision-directed Wiener gains and musical-noise suppression."""

    def test_gain_range_is_respected(self):
        suppressor = MultibandSpectralSuppressor(beta=0.01, gain_floor=0.05)
        _clean, gain, info = suppressor.process_spectrum(
            random_spectrogram(), sample_rate=SAMPLE_RATE
        )
        floor = max(0.05, float(np.sqrt(0.01)))
        assert np.all(gain >= floor - 1e-6)
        assert np.all(gain <= 1.0 + 1e-6)
        assert info["raw_gain"].shape == gain.shape

    def test_clean_magnitude_is_gain_times_input(self):
        suppressor = MultibandSpectralSuppressor()
        magnitude = random_spectrogram()
        clean, gain, _info = suppressor.process_spectrum(
            magnitude, sample_rate=SAMPLE_RATE
        )
        np.testing.assert_allclose(clean, gain * magnitude, rtol=1e-5, atol=1e-6)

    def test_bands_are_reported(self):
        suppressor = MultibandSpectralSuppressor(alpha0=2.0)
        _clean, _gain, info = suppressor.process_spectrum(
            random_spectrogram(), sample_rate=SAMPLE_RATE
        )
        bands = info["bands"]
        assert len(bands) == 3
        for band in bands:
            assert "snr_db_mean" in band and "alpha_mean" in band
            assert 1.0 <= band["alpha_mean"] <= 3.0

    def test_frequency_and_time_smoothing_reduce_musical_noise(self):
        suppressor = MultibandSpectralSuppressor()
        magnitude = random_spectrogram(frames=60)
        _clean, smooth_gain, info = suppressor.process_spectrum(
            magnitude, sample_rate=SAMPLE_RATE
        )
        raw_gain = info["raw_gain"]

        def roughness(gain: np.ndarray) -> float:
            """Mean absolute frame-to-frame gain jump (musical-noise proxy)."""
            return float(np.mean(np.abs(np.diff(gain, axis=1))))

        assert roughness(smooth_gain) < roughness(raw_gain)

    def test_gain_stays_finite_on_digital_silence(self):
        suppressor = MultibandSpectralSuppressor()
        clean, gain, _info = suppressor.process_spectrum(
            np.zeros((64, 10), dtype=np.float32), sample_rate=SAMPLE_RATE
        )
        assert np.isfinite(clean).all()
        assert np.isfinite(gain).all()


# ─── NoiseReducer integration ───────────────────────────────────────────────

class TestReducerDefaultAlgorithm:
    """The v2.6.0 multi-band engine is the default, legacy stays available."""

    def test_default_algorithm_is_multiband(self):
        reducer = NoiseReducer()
        assert reducer.algorithm is AlgorithmType.MULTIBAND_SPECTRAL
        assert reducer.available_algorithms() == ["legacy", "multiband", "wiener_dd"]

    def test_algorithm_accepts_strings_and_enum(self):
        reducer = NoiseReducer(algorithm="wiener_dd")
        assert reducer.algorithm is AlgorithmType.WIENER_DD
        reducer.algorithm = AlgorithmType.LEGACY
        assert reducer.algorithm is AlgorithmType.LEGACY
        reducer.algorithm = "multiband"
        assert reducer.algorithm is AlgorithmType.MULTIBAND_SPECTRAL

    def test_unknown_algorithm_falls_back_to_multiband(self):
        reducer = NoiseReducer(algorithm="nope")
        assert reducer.algorithm is AlgorithmType.MULTIBAND_SPECTRAL

    def test_unknown_algorithm_assignment_keeps_previous(self):
        reducer = NoiseReducer(algorithm="legacy")
        reducer.algorithm = "does-not-exist"
        assert reducer.algorithm is AlgorithmType.LEGACY

    def test_legacy_path_is_bit_identical(self):
        audio, _lead, _body = noisy_speech()
        reducer = NoiseReducer(algorithm="legacy")
        through_process = reducer.process(audio, sample_rate=SAMPLE_RATE)
        direct = reducer._process_spectral(audio, sample_rate=SAMPLE_RATE)
        np.testing.assert_allclose(through_process, direct, rtol=1e-6, atol=1e-7)

    def test_algorithms_produce_different_output(self):
        audio, _lead, _body = noisy_speech()
        multiband = NoiseReducer(algorithm="multiband").process(audio, sample_rate=SAMPLE_RATE)
        legacy = NoiseReducer(algorithm="legacy").process(audio, sample_rate=SAMPLE_RATE)
        wiener = NoiseReducer(algorithm="wiener_dd").process(audio, sample_rate=SAMPLE_RATE)
        for output in (multiband, legacy, wiener):
            assert output.shape == audio.shape
            assert np.isfinite(output).all()
        assert not np.allclose(multiband, legacy)


class TestNoiseReductionQuality:
    """The MBSS path must actually improve the signal-to-noise ratio."""

    def test_process_improves_snr(self):
        audio, _lead, body = noisy_speech(noise_level=0.1)
        reference = tone(1.5)

        reducer = NoiseReducer(strength="medium")
        clean = reducer.process(audio, sample_rate=SAMPLE_RATE)
        body_clean = clean[len(audio) - len(body):]
        noisy_body = audio[len(audio) - len(body):]

        snr_before = output_snr_db(noisy_body, reference)
        snr_after = output_snr_db(body_clean, reference)
        assert snr_after > snr_before + 2.0, (
            f"MBSS must raise the SNR (before={snr_before:.2f} dB, "
            f"after={snr_after:.2f} dB)"
        )

    def test_noise_floor_is_suppressed(self):
        audio, lead, _body = noisy_speech(noise_level=0.1)
        reducer = NoiseReducer()
        clean = reducer.process(audio, sample_rate=SAMPLE_RATE)
        cleaned_lead = clean[: len(lead)]
        # The noise-only lead-in must lose at least half of its energy
        assert np.sqrt(np.mean(cleaned_lead ** 2)) < 0.5 * np.sqrt(np.mean(lead ** 2))

    def test_reduce_noise_aggressiveness_monotonic(self):
        audio, lead, _body = noisy_speech(noise_level=0.1)
        reducer = NoiseReducer()
        light = reducer.reduce_noise(audio, aggressiveness=0.0, sample_rate=SAMPLE_RATE)
        heavy = reducer.reduce_noise(audio, aggressiveness=1.0, sample_rate=SAMPLE_RATE)

        assert light.shape == audio.shape and light.dtype == np.float32
        assert heavy.shape == audio.shape
        light_rms = float(np.sqrt(np.mean(light[: len(lead)] ** 2)))
        heavy_rms = float(np.sqrt(np.mean(heavy[: len(lead)] ** 2)))
        assert heavy_rms <= light_rms + 1e-6

    def test_reduce_noise_default_signature(self):
        audio, _lead, _body = noisy_speech()
        reducer = NoiseReducer()
        result = reducer.reduce_noise(audio)
        assert isinstance(result, np.ndarray)
        assert result.shape == audio.shape
        assert result.dtype == np.float32
        assert np.isfinite(result).all()

    def test_reduce_noise_restores_configuration(self):
        reducer = NoiseReducer()
        alpha_before = reducer.alpha
        reducer.reduce_noise(noisy_speech()[0], aggressiveness=1.0, sample_rate=SAMPLE_RATE)
        assert reducer.alpha == alpha_before
        assert reducer.algorithm is AlgorithmType.MULTIBAND_SPECTRAL


class TestReducerDiagnosticsAndStreaming:
    """Per-band diagnostics and the continuity of the recursive state."""

    def test_band_diagnostics_are_available(self):
        audio, _lead, _body = noisy_speech()
        reducer = NoiseReducer()
        reducer.process(audio, sample_rate=SAMPLE_RATE)
        assert len(reducer.last_band_info) == 3
        assert reducer.last_gain is not None
        assert reducer.last_gain.shape[1] > 0

    def test_band_over_subtraction_wrapper(self):
        reducer = NoiseReducer(alpha=2.0)
        alphas = reducer.band_over_subtraction(np.array([-10.0, 5.0]))
        assert alphas[0] > alphas[1]

    def test_multiband_spectral_subtract_is_stateless_and_repeatable(self):
        reducer = NoiseReducer()
        magnitude = random_spectrogram(frames=20)
        first = reducer.multiband_spectral_subtract(magnitude, sample_rate=SAMPLE_RATE)
        second = reducer.multiband_spectral_subtract(magnitude, sample_rate=SAMPLE_RATE)
        assert first.shape == magnitude.shape
        assert first.dtype == np.float32
        np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-7)
        assert float(np.max(first)) <= float(np.max(magnitude)) + 1e-6

    def test_multiband_spectral_subtract_empty(self):
        reducer = NoiseReducer()
        empty = reducer.multiband_spectral_subtract(np.zeros((0, 0), dtype=np.float32))
        assert empty.size == 0

    def test_wiener_dd_gain_smoothing_modes(self):
        reducer = NoiseReducer()
        magnitude = random_spectrogram(frames=40)
        raw = reducer.wiener_dd_gain(magnitude, sample_rate=SAMPLE_RATE, smooth=False)
        smoothed = reducer.wiener_dd_gain(magnitude, sample_rate=SAMPLE_RATE, smooth=True)
        assert raw.shape == magnitude.shape == smoothed.shape
        assert np.isfinite(raw).all() and np.isfinite(smoothed).all()
        # The smoothed gain is the temporally regularised one
        import numpy as _np
        assert _np.mean(_np.abs(_np.diff(smoothed, axis=1))) < _np.mean(
            _np.abs(_np.diff(raw, axis=1))
        )

    def test_streaming_chunks_share_recursive_state(self):
        reducer = NoiseReducer()
        rng = np.random.default_rng(1)
        chunk = rng.standard_normal(reducer.frame_length * 2).astype(np.float32)
        reducer.process_chunk(chunk, sample_rate=SAMPLE_RATE)
        first_frames = reducer._suppressor.frames_seen
        assert first_frames > 0
        reducer.process_chunk(chunk, sample_rate=SAMPLE_RATE)
        assert reducer._suppressor.frames_seen > first_frames

    def test_reset_algorithm_state_clears_recursion(self):
        reducer = NoiseReducer()
        reducer.process(noisy_speech()[0], sample_rate=SAMPLE_RATE)
        assert reducer.last_gain is not None
        reducer.reset_algorithm_state()
        assert reducer.last_gain is None
        assert reducer.last_band_info == []
        assert reducer._suppressor.frames_seen == 0

    def test_process_with_stats_still_returns_result(self):
        audio, _lead, _body = noisy_speech()
        result = NoiseReducer().process_with_stats(audio, sample_rate=SAMPLE_RATE)
        assert isinstance(result, NoiseReductionResult)
        assert result.audio.shape == audio.shape
        assert result.frames_processed > 0
        assert np.isfinite(result.snr_before) and np.isfinite(result.snr_after)

    def test_empty_and_short_audio_are_safe(self):
        reducer = NoiseReducer()
        assert len(reducer.process(np.array([], dtype=np.float32))) == 0
        short = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
        assert len(reducer.process(short, sample_rate=SAMPLE_RATE)) == len(short)

    def test_close_releases_state(self):
        reducer = NoiseReducer()
        reducer.process(noisy_speech()[0], sample_rate=SAMPLE_RATE)
        reducer.close()
        assert reducer.last_gain is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



