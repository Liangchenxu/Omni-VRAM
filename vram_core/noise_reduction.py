"""
Noise Reduction Module for vram_core
=====================================

Professional-grade noise reduction with multiple backends:

1. **WebRTC APM** (preferred): Full audio processing pipeline with AEC, NS, AGC
   - Requires: pip install py-webrtc-audio-processing
   - Features: Acoustic Echo Cancellation, Noise Suppression, Automatic Gain Control

2. **Multi-Band Spectral Subtraction** (default): pure numpy/scipy implementation
   - MBSS (non-uniform low/mid/high bands) with adaptive over-subtraction
   - Decision-directed (Ephraim-Malah) a-priori SNR + Wiener gain smoothing
   - ``algorithm="legacy"`` restores the classic v2.5.0 spectral subtraction
   - No external dependencies beyond numpy/scipy

3. **Streaming mode**: Real-time chunk-based noise reduction for live audio

Usage:
    from vram_core.noise_reduction import NoiseReducer

    # Auto-detect best backend
    reducer = NoiseReducer(strength="medium")
    clean_audio = reducer.process(audio_array, sample_rate=16000)

    # Force WebRTC backend
    reducer = NoiseReducer(backend="webrtc")

    # Streaming mode
    reducer = NoiseReducer(streaming=True)
    for chunk in audio_chunks:
        clean_chunk = reducer.process_chunk(chunk, sample_rate=16000)
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import stft, istft

logger = logging.getLogger(__name__)


# 鈹€鈹€ Backend Detection 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_WEBRTC_AVAILABLE = False
try:
    import webrtc_audio_processing
    _WEBRTC_AVAILABLE = True
    logger.info("py-webrtc-audio-processing detected 锟?WebRTC APM backend available")
except ImportError:
    try:
        import py_webrtc_audio_processing
        webrtc_audio_processing = py_webrtc_audio_processing
        _WEBRTC_AVAILABLE = True
        logger.info("py-webrtc-audio detected 锟?WebRTC APM backend available")
    except ImportError:
        logger.info(
            "WebRTC audio processing not available, using spectral subtraction fallback. "
            "Install with: pip install py-webrtc-audio-processing"
        )


# 鈹€鈹€ Enums & Presets 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class NoiseStrength(Enum):
    """Noise reduction strength presets."""
    LIGHT = "light"
    MEDIUM = "medium"
    AGGRESSIVE = "aggressive"


class Backend(Enum):
    """Noise reduction backend."""
    AUTO = "auto"
    WEBRTC = "webrtc"
    SPECTRAL = "spectral"


class AlgorithmType(Enum):
    """
    Frequency-domain denoising algorithm used by the spectral backend.

    LEGACY            : classic magnitude spectral subtraction (v2.5.0 behaviour)
    MULTIBAND_SPECTRAL: multi-band adaptive spectral subtraction (MBSS) with a
                        decision-directed a-priori SNR estimator and Wiener gain
                        smoothing (v2.6.0 default)
    WIENER_DD         : decision-directed Wiener filter without band-dependent
                        over-subtraction (single global over-subtraction factor)
    """
    LEGACY = "legacy"
    MULTIBAND_SPECTRAL = "multiband"
    WIENER_DD = "wiener_dd"


# ── MBSS / decision-directed hyper-parameters ────────────────────────────────
# Non-uniform band edges (Hz) tuned to the speech formant distribution:
# low band 0-500 Hz (F0 + first formant), mid 500-2000 Hz (F1/F2),
# high 2000 Hz-Nyquist (fricatives / consonants).
_MBSS_BAND_EDGES_HZ = (0.0, 500.0, 2000.0)
# Decision-directed smoothing factor for the a-priori SNR (Ephraim-Malah)
_DD_SMOOTHING = 0.96
# Minimum residual Wiener gain -> suppresses isolated musical-noise spikes
_DD_GAIN_FLOOR = 0.05
# 3-tap frequency-domain smoothing kernel applied to the Wiener gain
_MBSS_GAIN_KERNEL = (0.25, 0.5, 0.25)
# Recursive noise-PSD smoothing factor (speech-presence dependent)
_MBSS_NOISE_SMOOTHING = 0.9
_SILENCE_EPS = 1e-12


# Spectral subtraction presets
_SPECTRAL_PRESETS = {
    NoiseStrength.LIGHT: {
        "alpha": 1.0,
        "beta": 0.02,
        "noise_frames": 6,
    },
    NoiseStrength.MEDIUM: {
        "alpha": 2.0,
        "beta": 0.01,
        "noise_frames": 8,
    },
    NoiseStrength.AGGRESSIVE: {
        "alpha": 4.0,
        "beta": 0.005,
        "noise_frames": 12,
    },
}

# WebRTC NS level presets (0=low, 1=moderate, 2=high, 3=very_high)
_WEBRTC_NS_LEVEL = {
    NoiseStrength.LIGHT: 0,
    NoiseStrength.MEDIUM: 1,
    NoiseStrength.AGGRESSIVE: 2,
}


@dataclass
class NoiseReductionResult:
    """Result of noise reduction processing."""
    audio: np.ndarray
    noise_estimate: np.ndarray
    snr_before: float
    snr_after: float
    frames_processed: int
    backend_used: str = "unknown"


class WebRTCProcessor:
    """
    WebRTC Audio Processing Module wrapper.

    Provides AEC (Acoustic Echo Cancellation), NS (Noise Suppression),
    and AGC (Automatic Gain Control) using the WebRTC audio processing engine.

    Args:
        sample_rate: Audio sample rate (8000, 16000, 32000, or 48000).
        ns_level: Noise suppression level (0=low, 1=moderate, 2=high, 3=very_high).
        enable_aec: Enable Acoustic Echo Cancellation.
        enable_agc: Enable Automatic Gain Control.
        enable_ns: Enable Noise Suppression.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        ns_level: int = 1,
        enable_aec: bool = True,
        enable_agc: bool = True,
        enable_ns: bool = True,
    ):
        if not _WEBRTC_AVAILABLE:
            raise RuntimeError(
                "WebRTC audio processing not available. "
                "Install with: pip install py-webrtc-audio-processing"
            )

        self.sample_rate = sample_rate
        self.ns_level = ns_level
        self.enable_aec = enable_aec
        self.enable_agc = enable_agc
        self.enable_ns = enable_ns

        # WebRTC APM processes 10ms frames
        self.frame_size = int(sample_rate * 0.01)  # samples per 10ms frame

        self._apm = None
        self._init_apm()

        logger.info(
            "WebRTC APM initialized: sr=%d, ns_level=%d, aec=%s, agc=%s, ns=%s",
            sample_rate, ns_level, enable_aec, enable_agc, enable_ns,
        )

    def _init_apm(self):
        """Initialize the WebRTC Audio Processing Module."""
        try:
            # Try the newer API first
            self._apm = webrtc_audio_processing.AudioProcessingModule(
                enable_aec=self.enable_aec,
                enable_agc=self.enable_agc,
                enable_ns=self.enable_ns,
            )

            # Configure NS level
            if hasattr(self._apm, 'set_ns_level'):
                self._apm.set_ns_level(self.ns_level)

            # Configure AGC
            if hasattr(self._apm, 'set_agc_config'):
                self._apm.set_agc_config(
                    target_level_dbfs=3,
                    compression_gain_db=9,
                    limiter_enable=True,
                )

        except (AttributeError, TypeError):
            # Fallback: simpler initialization
            try:
                self._apm = webrtc_audio_processing.AudioProcessingModule()
                if self.enable_ns and hasattr(self._apm, 'set_ns'):
                    self._apm.set_ns(True)
                if self.enable_aec and hasattr(self._apm, 'set_aec'):
                    self._apm.set_aec(True)
                if self.enable_agc and hasattr(self._apm, 'set_agc'):
                    self._apm.set_agc(True)
            except Exception as e:
                logger.error("Failed to initialize WebRTC APM: %s", e)
                raise RuntimeError(f"WebRTC APM initialization failed: {e}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single 10ms audio frame through WebRTC APM.

        Args:
            frame: Audio frame (float32, mono, exactly frame_size samples).

        Returns:
            Processed audio frame.
        """
        if self._apm is None:
            return frame

        # Convert to int16 (WebRTC expects 16-bit PCM)
        frame_int16 = (frame * 32767).astype(np.int16)

        try:
            if hasattr(self._apm, 'process_stream'):
                result = self._apm.process_stream(frame_int16)
            elif hasattr(self._apm, 'ProcessStream'):
                result = self._apm.ProcessStream(frame_int16)
            else:
                result = frame_int16
        except Exception as e:
            logger.warning("WebRTC frame processing error: %s", e)
            result = frame_int16

        return result.astype(np.float32) / 32767.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process an entire audio signal through WebRTC APM.

        Splits into 10ms frames, processes each, and reassembles.

        Args:
            audio: Input audio signal (float32, mono).

        Returns:
            Processed audio signal (float32).
        """
        if len(audio) == 0:
            return audio

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Split into 10ms frames
        n_frames = len(audio) // self.frame_size
        remainder = len(audio) % self.frame_size

        output_frames = []
        for i in range(n_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frame = audio[start:end]
            processed = self.process_frame(frame)
            output_frames.append(processed)

        # Handle remainder
        if remainder > 0:
            last_frame = np.zeros(self.frame_size, dtype=np.float32)
            last_frame[:remainder] = audio[n_frames * self.frame_size:]
            processed = self.process_frame(last_frame)
            output_frames.append(processed[:remainder])

        return np.concatenate(output_frames).astype(np.float32)

    def close(self):
        """Release WebRTC APM resources."""
        if self._apm is not None:
            try:
                if hasattr(self._apm, 'close'):
                    self._apm.close()
                elif hasattr(self._apm, 'Destroy'):
                    self._apm.Destroy()
            except Exception:
                pass
            self._apm = None

    def __del__(self):
        self.close()


class MultibandSpectralSuppressor:
    """
    Multi-Band Spectral Subtraction (MBSS) with a decision-directed (DD)
    a-priori SNR estimator and Wiener gain smoothing.

    Mathematical outline (``k`` = frequency bin, ``t`` = frame index):

    1. Posterior SNR ``gamma(k,t) = |Y(k,t)|^2 / N(k,t)``
    2. Band over-subtraction ``alpha_i(SNR_i) = clip(alpha0 - SNR_i_dB / slope,
       alpha_min, alpha_max)`` -- one factor per non-uniform band applied to
       every bin of that band: high-SNR bands are attenuated *less* (formant
       protection), low-SNR bands are attenuated *more* (noise-floor collapse).
    3. Decision-directed a-priori SNR (Ephraim-Malah)::

           xi(k,t) = alpha_dd * G(k,t-1)^2 * |Y(k,t-1)|^2 / (alpha_i * N(k,t))
                   + (1 - alpha_dd) * max(gamma(k,t) / alpha_i - 1, 0)

    4. Wiener gain ``G(k,t) = xi / (1 + xi)``
    5. Gain smoothing: 3-tap frequency smoothing followed by exponential time
       smoothing, floored at ``max(gain_floor, sqrt(beta))`` so that isolated
       gain peaks ("musical noise" / birdies) cannot survive.

    The instance is stateful: the recursive noise PSD, the previous gain and the
    previous clean PSD are carried across calls so that consecutive streaming
    chunks share continuous statistics. Call :meth:`reset` to start over.
    """

    def __init__(
        self,
        alpha0: float = 2.0,
        beta: float = 0.01,
        band_edges_hz: Tuple[float, ...] = _MBSS_BAND_EDGES_HZ,
        alpha_dd: float = _DD_SMOOTHING,
        gain_floor: float = _DD_GAIN_FLOOR,
        time_smoothing: float = 0.7,
        snr_slope: float = 10.0,
        noise_smoothing: float = _MBSS_NOISE_SMOOTHING,
        multiband: bool = True,
    ):
        self.alpha0 = float(alpha0)
        self.beta = float(beta)
        self.band_edges_hz = tuple(float(e) for e in band_edges_hz)
        self.alpha_dd = float(np.clip(alpha_dd, 0.0, 0.999))
        self.gain_floor = float(np.clip(gain_floor, 0.0, 1.0))
        self.time_smoothing = float(np.clip(time_smoothing, 0.0, 0.999))
        self.snr_slope = max(float(snr_slope), 1e-3)
        self.noise_smoothing = float(np.clip(noise_smoothing, 0.0, 0.999))
        # When False the band-dependent over-subtraction is disabled: alpha equals
        # alpha0 for every band, which turns this into a plain DD-Wiener filter.
        self.multiband = bool(multiband)
        self.reset()

    # ── State ─────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Drop all recursive state (noise PSD, previous gain / clean PSD)."""
        self._noise_psd: Optional[np.ndarray] = None
        self._prev_gain: Optional[np.ndarray] = None
        self._prev_clean_psd: Optional[np.ndarray] = None
        self._frames_seen = 0

    @property
    def frames_seen(self) -> int:
        """Number of STFT frames processed since the last :meth:`reset`."""
        return self._frames_seen

    @property
    def noise_psd(self) -> Optional[np.ndarray]:
        """Current recursive noise power spectral density (or ``None``)."""
        return self._noise_psd

    @staticmethod
    def band_slices(
        n_bins: int,
        sample_rate: int,
        frame_length: int,
        band_edges_hz: Tuple[float, ...] = _MBSS_BAND_EDGES_HZ,
    ) -> List[slice]:
        """
        Map the non-uniform Hz band edges to contiguous STFT bin slices.

        Args:
            n_bins: Number of rfft bins (``frame_length // 2 + 1``).
            sample_rate: Sample rate in Hz.
            frame_length: STFT window length in samples.
            band_edges_hz: Non-uniform band edges in Hz.

        Returns:
            One :class:`slice` per band, covering every bin exactly once.
        """
        freq_res = float(sample_rate) / float(frame_length)
        edges = [float(e) for e in band_edges_hz] + [sample_rate / 2.0 + freq_res]
        cuts = [int(np.clip(round(e / freq_res), 0, n_bins)) for e in edges]
        cuts[0] = 0
        cuts[-1] = n_bins
        slices: List[slice] = []
        previous = 0
        for cut in cuts[1:]:
            cut = min(max(cut, previous + 1), n_bins)
            slices.append(slice(previous, cut))
            previous = cut
        return slices

    def band_over_subtraction(self, snr_band_db: np.ndarray) -> np.ndarray:
        """
        Adaptive per-band over-subtraction factor ``alpha_i(SNR_i)``.

        ``alpha_i = clip(alpha0 - SNR_i_dB / slope, alpha_min, alpha_max)``

        Higher SNR gives a smaller subtraction (protect speech), lower SNR gives
        a larger subtraction (kill the noise floor). ``alpha_min`` is 1.0 (a
        factor below 1 would *add* energy) and ``alpha_max`` is ``1.5 * alpha0``.
        """
        snr_band_db = np.asarray(snr_band_db, dtype=np.float64)
        alpha_min = 1.0
        alpha_max = max(self.alpha0 * 1.5, alpha_min)
        alpha = self.alpha0 - snr_band_db / self.snr_slope
        return np.clip(alpha, alpha_min, alpha_max)

    def _estimate_noise_psd(self, power: np.ndarray) -> np.ndarray:
        """
        Recursive noise-PSD estimate with speech-presence guarding.

        ``N(k,t) = lam_t * N(k,t-1) + (1 - lam_t) * |Y(k,t)|^2`` with
        ``lam_t = lam + (1 - lam) * P(speech)`` so that the estimate is frozen
        while speech is present and adapts quickly during pauses.
        """
        n_bins, n_frames = power.shape
        start = 0
        if self._noise_psd is None or self._noise_psd.shape != (n_bins,):
            init_frames = min(2, n_frames)
            anchor = np.mean(power[:, :init_frames], axis=1)
            self._noise_psd = np.maximum(anchor, _SILENCE_EPS)
            start = init_frames
        lam = self.noise_smoothing
        for t in range(start, n_frames):
            p = power[:, t]
            posterior = p / np.maximum(self._noise_psd, _SILENCE_EPS)
            # Soft speech-presence proxy: 0 -> noise only, 1 -> strong speech
            p_speech = np.clip((posterior - 1.0) / 2.0, 0.0, 1.0)
            lam_t = lam + (1.0 - lam) * p_speech
            self._noise_psd = lam_t * self._noise_psd + (1.0 - lam_t) * p
            self._noise_psd = np.maximum(self._noise_psd, _SILENCE_EPS)
        return self._noise_psd

    # ── Main entry point ──────────────────────────────────────────────────

    def process_spectrum(
        self,
        magnitude: np.ndarray,
        sample_rate: int = 16000,
        frame_length: Optional[int] = None,
        noise_estimate: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        """
        Apply MBSS + DD-Wiener gains to a magnitude spectrogram.

        Args:
            magnitude: STFT magnitude matrix ``(n_bins, n_frames)``.
            sample_rate: Sample rate in Hz (drives the band mapping).
            frame_length: STFT window length; inferred from ``n_bins`` when None.
            noise_estimate: Optional magnitude noise estimate used to seed the
                recursive noise PSD.

        Returns:
            ``(clean_magnitude, gain, info)`` where ``gain`` is the smoothed
            Wiener gain and ``info`` carries the raw gain, per-band statistics,
            the ``frame_length`` and the noise PSD.
        """
        mag = np.asarray(magnitude, dtype=np.float64)
        if mag.ndim != 2:
            raise ValueError("magnitude must be a 2-D (n_bins, n_frames) array")
        n_bins, n_frames = mag.shape
        if n_bins == 0 or n_frames == 0:
            empty = np.zeros(mag.shape, dtype=np.float32)
            return empty, empty, {
                "raw_gain": empty, "bands": [], "frame_length": frame_length or 0,
            }

        frame_length = int(frame_length or 2 * (n_bins - 1) or 512)
        power = mag ** 2

        if noise_estimate is not None:
            anchor = np.asarray(noise_estimate, dtype=np.float64).reshape(-1)
            if anchor.size == n_bins and float(anchor.max()) > 0.0:
                seed = np.maximum(anchor ** 2, _SILENCE_EPS)
                if self._noise_psd is None or self._noise_psd.shape != (n_bins,):
                    self._noise_psd = seed
                else:
                    # Blend the caller estimate in as a conservative floor
                    self._noise_psd = np.maximum(self._noise_psd, seed * 0.5)

        noise_psd = self._estimate_noise_psd(power)

        # ── Per-band adaptive over-subtraction factors ────────────────────
        slices = self.band_slices(n_bins, sample_rate, frame_length, self.band_edges_hz)
        alpha_bin = np.ones((n_bins, n_frames), dtype=np.float64)
        band_info: List[Dict[str, float]] = []
        for index, band in enumerate(slices):
            band_noise = max(float(np.mean(noise_psd[band])), _SILENCE_EPS)
            band_power = np.mean(power[band, :], axis=0)
            snr_lin = np.maximum(band_power / band_noise - 1.0, 0.0)
            snr_db = 10.0 * np.log10(snr_lin + 1e-6)
            if self.multiband:
                alpha_t = self.band_over_subtraction(snr_db)
            else:
                alpha_t = np.full(n_frames, self.alpha0, dtype=np.float64)
            alpha_bin[band, :] = alpha_t[np.newaxis, :]
            band_info.append({
                "band": float(index),
                "start_bin": float(band.start or 0),
                "stop_bin": float(band.stop or 0),
                "noise_psd_mean": band_noise,
                "snr_db_mean": float(np.mean(snr_db)),
                "alpha_mean": float(np.mean(alpha_t)),
            })

        # ── Decision-directed Wiener gain (sequential in time) ────────────
        fresh_state = (
            self._prev_gain is None
            or self._prev_gain.shape != power.shape
            or self._prev_clean_psd is None
            or self._prev_clean_psd.shape != power.shape
        )
        if fresh_state:
            prev_gain = np.zeros_like(power)
            prev_clean = np.zeros_like(power)
        else:
            prev_gain = self._prev_gain
            prev_clean = self._prev_clean_psd

        gain = np.empty_like(power)
        raw_gain = np.empty_like(power)
        gain_floor = max(self.gain_floor, float(np.sqrt(max(self.beta, 0.0))))
        kernel = np.asarray(_MBSS_GAIN_KERNEL, dtype=np.float64)

        for t in range(n_frames):
            n_eff = np.maximum(noise_psd * alpha_bin[:, t], _SILENCE_EPS)
            gamma = power[:, t] / n_eff
            xi = (
                self.alpha_dd * (prev_clean[:, t] / n_eff)
                + (1.0 - self.alpha_dd) * np.maximum(gamma - 1.0, 0.0)
            )
            xi = np.maximum(xi, 0.0)
            g_wiener = xi / (1.0 + xi)
            raw_gain[:, t] = g_wiener

            g_freq = np.convolve(g_wiener, kernel, mode="same")
            if fresh_state and t == 0:
                g_hat = g_freq
            else:
                g_hat = (
                    self.time_smoothing * prev_gain[:, t]
                    + (1.0 - self.time_smoothing) * g_freq
                )
            g_hat = np.clip(g_hat, gain_floor, 1.0)

            gain[:, t] = g_hat
            prev_gain[:, t] = g_hat
            prev_clean[:, t] = (g_hat * mag[:, t]) ** 2

        self._prev_gain = prev_gain
        self._prev_clean_psd = prev_clean
        self._frames_seen += n_frames

        clean_magnitude = (gain * mag).astype(np.float32)
        return clean_magnitude, gain.astype(np.float32), {
            "raw_gain": raw_gain.astype(np.float32),
            "bands": band_info,
            "frame_length": frame_length,
            "noise_psd": noise_psd.astype(np.float32),
        }


class NoiseReducer:
    """
    Professional noise reducer with WebRTC APM and spectral subtraction backends.

    Features:
        - WebRTC APM: AEC + NS + AGC (when available)
        - Spectral Subtraction: Pure numpy fallback
        - Streaming mode: Real-time chunk-based processing
        - Auto backend selection

    Args:
        strength: Noise reduction strength ("light", "medium", "aggressive").
        backend: Backend to use ("auto", "webrtc", "spectral").
        streaming: Enable streaming mode for real-time processing.
        sample_rate: Sample rate for WebRTC (default 16000).
        enable_aec: Enable Acoustic Echo Cancellation (WebRTC only).
        enable_agc: Enable Automatic Gain Control (WebRTC only).
        alpha: Over-subtraction factor for spectral method.
        beta: Spectral floor factor for spectral method.
        noise_frames: Number of initial frames for noise estimation (spectral).
        frame_length: STFT frame length (spectral only).
        hop_length: STFT hop length (spectral only).
        algorithm: Frequency-domain algorithm ("multiband", "wiener_dd" or
            "legacy"); see :class:`AlgorithmType`. Defaults to the v2.6.0
            multi-band spectral subtraction (MBSS + decision-directed Wiener).
        dd_smoothing: Decision-directed smoothing factor of the a-priori SNR.
        gain_floor: Minimum Wiener gain (musical-noise suppression).
        time_smoothing: Exponential smoothing factor applied to the gain over
            time (0 = no smoothing, 0.99 = heavy smoothing).
        band_edges_hz: Non-uniform band edges in Hz for the MBSS bands.
        snr_slope: Slope of ``alpha_i(SNR_i) = alpha0 - SNR_i/slope``.
        noise_smoothing: Recursive noise-PSD smoothing factor.
        multiband: Set to False to disable band-dependent over-subtraction
            (plain decision-directed Wiener filter).
    """

    def __init__(
        self,
        strength: str = "medium",
        backend: str = "auto",
        streaming: bool = False,
        sample_rate: int = 16000,
        enable_aec: bool = True,
        enable_agc: bool = True,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        noise_frames: Optional[int] = None,
        frame_length: int = 512,
        hop_length: int = 256,
        algorithm: str = "multiband",
        dd_smoothing: float = _DD_SMOOTHING,
        gain_floor: float = _DD_GAIN_FLOOR,
        time_smoothing: float = 0.7,
        band_edges_hz: Tuple[float, ...] = _MBSS_BAND_EDGES_HZ,
        snr_slope: float = 10.0,
        noise_smoothing: float = _MBSS_NOISE_SMOOTHING,
        multiband: bool = True,
    ):
        # Parse strength
        try:
            self.strength = NoiseStrength(strength)
        except ValueError:
            logger.warning("Unknown strength '%s', falling back to 'medium'", strength)
            self.strength = NoiseStrength.MEDIUM

        # Parse backend
        try:
            self._backend_type = Backend(backend)
        except ValueError:
            logger.warning("Unknown backend '%s', falling back to 'auto'", backend)
            self._backend_type = Backend.AUTO

        # Parse algorithm (v2.6.0)
        try:
            self._algorithm = AlgorithmType(algorithm)
        except ValueError:
            logger.warning(
                "Unknown algorithm '%s', falling back to 'multiband'", algorithm
            )
            self._algorithm = AlgorithmType.MULTIBAND_SPECTRAL

        self.streaming = streaming
        self.sample_rate = sample_rate
        self.enable_aec = enable_aec
        self.enable_agc = enable_agc

        # Spectral subtraction parameters
        defaults = _SPECTRAL_PRESETS[self.strength]
        self.alpha = alpha if alpha is not None else defaults["alpha"]
        self.beta = beta if beta is not None else defaults["beta"]
        self.noise_frames = noise_frames if noise_frames is not None else defaults["noise_frames"]
        self.frame_length = frame_length
        self.hop_length = hop_length

        # Multi-band / decision-directed configuration (v2.6.0)
        self.dd_smoothing = dd_smoothing
        self.gain_floor = gain_floor
        self.time_smoothing = time_smoothing
        self.band_edges_hz = tuple(band_edges_hz)
        self.snr_slope = snr_slope
        self.noise_smoothing = noise_smoothing
        self.multiband = multiband

        # Streaming state
        self._stream_buffer = np.array([], dtype=np.float32)
        self._noise_estimate = None
        self._last_gain: Optional[np.ndarray] = None
        self._last_band_info: List[Dict[str, float]] = []

        # Recursive MBSS/DD filter state (continuous across streaming chunks)
        self._suppressor = self._make_suppressor()

        # Initialize backend
        self._webrtc: Optional[WebRTCProcessor] = None
        self._active_backend = "spectral"
        self._init_backend()

    def _init_backend(self):
        """Initialize the noise reduction backend."""
        if self._backend_type == Backend.WEBRTC or (
            self._backend_type == Backend.AUTO and _WEBRTC_AVAILABLE
        ):
            try:
                self._webrtc = WebRTCProcessor(
                    sample_rate=self.sample_rate,
                    ns_level=_WEBRTC_NS_LEVEL[self.strength],
                    enable_aec=self.enable_aec,
                    enable_agc=self.enable_agc,
                    enable_ns=True,
                )
                self._active_backend = "webrtc"
                logger.info("Using WebRTC APM backend")
            except Exception as e:
                logger.warning(
                    "WebRTC init failed (%s), falling back to spectral subtraction", e
                )
                self._active_backend = "spectral"
        else:
            self._active_backend = "spectral"
            logger.info("Using spectral subtraction backend")

    @property
    def backend(self) -> str:
        """Return the name of the active backend."""
        return self._active_backend

    # ── Algorithm configuration (v2.6.0) ──────────────────────────────────

    def _make_suppressor(self) -> MultibandSpectralSuppressor:
        """Build a fresh MBSS/DD suppressor from the current configuration."""
        return MultibandSpectralSuppressor(
            alpha0=self.alpha,
            beta=self.beta,
            band_edges_hz=self.band_edges_hz,
            alpha_dd=getattr(self, "dd_smoothing", _DD_SMOOTHING),
            gain_floor=getattr(self, "gain_floor", _DD_GAIN_FLOOR),
            time_smoothing=getattr(self, "time_smoothing", 0.7),
            snr_slope=getattr(self, "snr_slope", 10.0),
            noise_smoothing=getattr(self, "noise_smoothing", _MBSS_NOISE_SMOOTHING),
            # WIENER_DD disables the band-dependent over-subtraction
            multiband=(
                getattr(self, "multiband", True)
                and getattr(self, "_algorithm", AlgorithmType.MULTIBAND_SPECTRAL)
                is not AlgorithmType.WIENER_DD
            ),
        )

    @property
    def algorithm(self) -> AlgorithmType:
        """Active frequency-domain algorithm."""
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value) -> None:
        if isinstance(value, AlgorithmType):
            self._algorithm = value
        else:
            try:
                self._algorithm = AlgorithmType(value)
            except ValueError:
                logger.warning("Unknown algorithm '%s', keeping %s", value, self._algorithm)
                return
        self._suppressor.reset()

    @property
    def last_gain(self) -> Optional[np.ndarray]:
        """Smoothed Wiener gain matrix of the last multi-band call."""
        return self._last_gain

    @property
    def last_band_info(self) -> List[Dict[str, float]]:
        """Per-band statistics (SNR + adaptive alpha) of the last call."""
        return self._last_band_info

    def band_over_subtraction(self, snr_band_db) -> np.ndarray:
        """Adaptive per-band over-subtraction factor ``alpha_i(SNR_i)``."""
        return self._suppressor.band_over_subtraction(snr_band_db)

    def reset_algorithm_state(self) -> None:
        """Reset the recursive MBSS/DD state (noise PSD, previous gain)."""
        self._suppressor.reset()
        self._last_gain = None
        self._last_band_info = []

    # 鈹€鈹€ Spectral Subtraction Methods 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def estimate_noise_spectrum(
        self,
        magnitude: np.ndarray,
        n_noise_frames: Optional[int] = None,
    ) -> np.ndarray:
        """Estimate noise power spectrum from initial frames."""
        n = n_noise_frames or self.noise_frames
        n = min(n, magnitude.shape[1])

        if n == 0:
            return np.zeros(magnitude.shape[0], dtype=np.float32)

        noise_estimate = np.mean(magnitude[:, :n], axis=1)
        return noise_estimate.astype(np.float32)

    def spectral_subtract(
        self,
        magnitude: np.ndarray,
        noise_estimate: np.ndarray,
    ) -> np.ndarray:
        """Apply spectral subtraction to magnitude spectrogram."""
        mag_sq = magnitude ** 2
        noise_sq = noise_estimate ** 2
        noise_sq_expanded = noise_sq[:, np.newaxis]

        clean_sq = mag_sq - self.alpha * noise_sq_expanded
        spectral_floor = self.beta * mag_sq
        clean_sq = np.maximum(clean_sq, spectral_floor)
        clean_sq = np.maximum(clean_sq, 0.0)

        return np.sqrt(clean_sq).astype(np.float32)

    # ── Multi-band adaptive processing (v2.6.0) ───────────────────────────

    def _process_multiband(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Denoise audio with multi-band spectral subtraction (MBSS) and a
        decision-directed Wiener filter.

        Shares the recursive filter state between streaming chunks, so
        consecutive ``process_chunk`` calls see continuous noise statistics.
        """
        if len(audio) == 0:
            return np.asarray(audio, dtype=np.float32)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if len(audio) < 8:
            return audio.copy()

        frame_length = min(self.frame_length, len(audio))
        hop_length = min(self.hop_length, max(1, frame_length // 2))

        _freqs, _times, Zxx = stft(
            audio, fs=sample_rate, nperseg=frame_length,
            noverlap=frame_length - hop_length,
        )
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        clean_magnitude, gain, info = self._suppressor.process_spectrum(
            magnitude, sample_rate=sample_rate, frame_length=frame_length,
        )
        self._last_gain = gain
        self._last_band_info = list(info.get("bands", []))

        clean_Zxx = clean_magnitude * np.exp(1j * phase)
        _, clean_audio = istft(
            clean_Zxx, fs=sample_rate, nperseg=frame_length,
            noverlap=frame_length - hop_length,
        )

        clean_audio = np.asarray(clean_audio, dtype=np.float32)[:len(audio)]
        if len(clean_audio) < len(audio):
            clean_audio = np.pad(clean_audio, (0, len(audio) - len(clean_audio)))
        return clean_audio.astype(np.float32)

    def multiband_spectral_subtract(
        self,
        magnitude: np.ndarray,
        noise_estimate: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Stateless MBSS denoising of a magnitude spectrogram.

        Mirrors :meth:`spectral_subtract` (legacy) but applies the v2.6.0
        multi-band adaptive over-subtraction plus the decision-directed Wiener
        gain. A throwaway suppressor is used so that the streaming state is not
        disturbed and repeated calls with the same input are identical.
        """
        mag = np.asarray(magnitude, dtype=np.float32)
        if mag.ndim != 2 or mag.size == 0:
            return mag.astype(np.float32)
        suppressor = self._make_suppressor()
        frame_length = 2 * (mag.shape[0] - 1) or self.frame_length
        clean, _gain, _info = suppressor.process_spectrum(
            mag, sample_rate=sample_rate, frame_length=frame_length,
            noise_estimate=noise_estimate,
        )
        return clean.astype(np.float32)

    def wiener_dd_gain(
        self,
        magnitude: np.ndarray,
        noise_estimate: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        smooth: bool = True,
    ) -> np.ndarray:
        """
        Decision-directed Wiener gain matrix for a magnitude spectrogram.

        Args:
            magnitude: STFT magnitude matrix ``(n_bins, n_frames)``.
            noise_estimate: Optional magnitude noise estimate.
            sample_rate: Sample rate in Hz.
            smooth: True -> frequency + time smoothed gain (default, suppresses
                musical noise); False -> raw per-frame Wiener gain.

        Returns:
            Gain matrix with the shape of ``magnitude`` (float32, range
            ``[gain_floor, 1]``).
        """
        mag = np.asarray(magnitude, dtype=np.float32)
        if mag.ndim != 2 or mag.size == 0:
            return np.zeros_like(mag, dtype=np.float32)
        suppressor = self._make_suppressor()
        frame_length = 2 * (mag.shape[0] - 1) or self.frame_length
        _clean, gain, info = suppressor.process_spectrum(
            mag, sample_rate=sample_rate, frame_length=frame_length,
            noise_estimate=noise_estimate,
        )
        return np.asarray(info["raw_gain"] if not smooth else gain, dtype=np.float32)

    def reduce_noise(
        self,
        audio: np.ndarray,
        aggressiveness: float = 0.7,
        sample_rate: int = 16000,
        algorithm: Optional[str] = None,
    ) -> np.ndarray:
        """
        Convenience entry point: denoise ``audio`` with an aggressiveness knob.

        Args:
            audio: Input audio (float32/int16, mono).
            aggressiveness: 0.0 (barely touch the signal) .. 1.0 (maximum
                suppression). Maps to the over-subtraction factor, the gain
                time-smoothing and the residual gain floor.
            sample_rate: Sample rate in Hz.
            algorithm: Optional per-call algorithm override ("multiband",
                "wiener_dd" or "legacy"); the instance configuration is restored
                afterwards.

        Returns:
            Denoised audio (float32) with the same length as the input.
        """
        level = float(np.clip(aggressiveness, 0.0, 1.0))
        previous = (
            self.alpha,
            self.time_smoothing,
            self.gain_floor,
            self.noise_smoothing,
            self._algorithm,
        )
        try:
            # 0.0 -> light  (alpha 1.0, little smoothing)
            # 1.0 -> heavy  (alpha 3.0, strong smoothing, low residual floor)
            self.alpha = 1.0 + 2.0 * level
            self.time_smoothing = 0.5 + 0.4 * level
            self.gain_floor = max(0.02, 0.12 - 0.10 * level)
            self.noise_smoothing = 0.85 + 0.1 * level
            if algorithm is not None:
                self.algorithm = algorithm
            self._suppressor = self._make_suppressor()
            return np.asarray(self.process(audio, sample_rate), dtype=np.float32)
        finally:
            (
                self.alpha,
                self.time_smoothing,
                self.gain_floor,
                self.noise_smoothing,
                self._algorithm,
            ) = previous
            self._suppressor = self._make_suppressor()

    def _process_spectral(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio with spectral subtraction."""
        if len(audio) == 0:
            return audio

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        frame_length = min(self.frame_length, len(audio))
        hop_length = min(self.hop_length, frame_length // 2)

        freqs, times, Zxx = stft(
            audio, fs=sample_rate, nperseg=frame_length,
            noverlap=frame_length - hop_length,
        )

        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        noise_estimate = self.estimate_noise_spectrum(magnitude)
        clean_magnitude = self.spectral_subtract(magnitude, noise_estimate)

        clean_Zxx = clean_magnitude * np.exp(1j * phase)
        _, clean_audio = istft(
            clean_Zxx, fs=sample_rate, nperseg=frame_length,
            noverlap=frame_length - hop_length,
        )

        clean_audio = clean_audio[:len(audio)]
        if len(clean_audio) < len(audio):
            clean_audio = np.pad(
                clean_audio, (0, len(audio) - len(clean_audio)), mode="constant",
            )

        return clean_audio.astype(np.float32)

    # 鈹€鈹€ Public API 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Apply noise reduction to an audio signal.

        Uses WebRTC APM if available, otherwise falls back to the configured
        frequency-domain algorithm (MBSS + decision-directed Wiener by default,
        classic spectral subtraction with ``algorithm="legacy"``).

        Args:
            audio: Input audio signal (float32, mono).
            sample_rate: Sample rate in Hz.

        Returns:
            Noise-reduced audio signal (float32).
        """
        if self._active_backend == "webrtc" and self._webrtc is not None:
            return self._webrtc.process(audio)
        if self._algorithm is not AlgorithmType.LEGACY:
            return self._process_multiband(audio, sample_rate)
        return self._process_spectral(audio, sample_rate)

    def process_chunk(
        self,
        chunk: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Process a single audio chunk in streaming mode.

        For WebRTC: processes each chunk independently (10ms frames).
        For spectral: buffers chunks and processes when enough data is available.

        Args:
            chunk: Audio chunk (float32, mono).
            sample_rate: Sample rate in Hz.

        Returns:
            Processed audio chunk (float32).
        """
        if self._active_backend == "webrtc" and self._webrtc is not None:
            return self._webrtc.process(chunk)

        # Spectral streaming: buffer and process
        self._stream_buffer = np.concatenate(
            [self._stream_buffer, chunk.astype(np.float32)]
        )

        # Process when we have enough data (at least 2x frame_length)
        min_samples = self.frame_length * 2
        if len(self._stream_buffer) >= min_samples:
            audio_to_process = self._stream_buffer
            self._stream_buffer = np.array([], dtype=np.float32)
            if self._algorithm is not AlgorithmType.LEGACY:
                return self._process_multiband(audio_to_process, sample_rate)
            return self._process_spectral(audio_to_process, sample_rate)

        # Not enough data yet, return zeros
        result = np.zeros_like(chunk, dtype=np.float32)
        return result

    def flush(self, sample_rate: int = 16000) -> np.ndarray:
        """
        Flush the streaming buffer and process remaining audio.

        Args:
            sample_rate: Sample rate in Hz.

        Returns:
            Remaining processed audio from the buffer.
        """
        if len(self._stream_buffer) == 0:
            return np.array([], dtype=np.float32)

        remaining = self._stream_buffer
        self._stream_buffer = np.array([], dtype=np.float32)

        if self._active_backend == "webrtc" and self._webrtc is not None:
            return self._webrtc.process(remaining)
        if self._algorithm is not AlgorithmType.LEGACY:
            return self._process_multiband(remaining, sample_rate)
        return self._process_spectral(remaining, sample_rate)

    def process_with_stats(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> NoiseReductionResult:
        """Apply noise reduction and return detailed statistics."""
        if len(audio) == 0:
            return NoiseReductionResult(
                audio=audio, noise_estimate=np.array([]),
                snr_before=0.0, snr_after=0.0, frames_processed=0,
                backend_used=self._active_backend,
            )

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        clean_audio = self.process(audio, sample_rate)

        # Compute SNR estimates using spectral analysis
        frame_length = min(self.frame_length, len(audio))
        hop_length = min(self.hop_length, frame_length // 2)

        _, _, Zxx_orig = stft(
            audio, fs=sample_rate, nperseg=frame_length,
            noverlap=frame_length - hop_length,
        )
        _, _, Zxx_clean = stft(
            clean_audio, fs=sample_rate, nperseg=frame_length,
            noverlap=frame_length - hop_length,
        )

        mag_orig = np.abs(Zxx_orig)
        mag_clean = np.abs(Zxx_clean)

        noise_frames = min(self.noise_frames, mag_orig.shape[1])
        noise_estimate = self.estimate_noise_spectrum(mag_orig)

        signal_power = np.mean(mag_orig[:, noise_frames:] ** 2) + 1e-10
        noise_power = np.mean(noise_estimate ** 2) + 1e-10
        snr_before = float(10 * np.log10(signal_power / noise_power))

        clean_noise = mag_clean[:, :noise_frames]
        clean_signal = mag_clean[:, noise_frames:]
        clean_noise_power = np.mean(clean_noise ** 2) + 1e-10
        clean_signal_power = np.mean(clean_signal ** 2) + 1e-10
        snr_after = float(10 * np.log10(clean_signal_power / clean_noise_power))

        return NoiseReductionResult(
            audio=clean_audio,
            noise_estimate=noise_estimate,
            snr_before=snr_before,
            snr_after=snr_after,
            frames_processed=mag_orig.shape[1],
            backend_used=self._active_backend,
        )

    @staticmethod
    def create_preset(strength: str = "medium") -> "NoiseReducer":
        """Create a NoiseReducer with preset parameters."""
        return NoiseReducer(strength=strength)

    @staticmethod
    def available_backends() -> List[str]:
        """List available noise reduction backends."""
        backends = ["spectral"]
        if _WEBRTC_AVAILABLE:
            backends.insert(0, "webrtc")
        return backends

    @staticmethod
    def available_algorithms() -> List[str]:
        """List the frequency-domain algorithms supported by the spectral backend."""
        return [algorithm.value for algorithm in AlgorithmType]

    def close(self):
        """Release resources."""
        self._stream_buffer = np.array([], dtype=np.float32)
        self._noise_estimate = None
        self.reset_algorithm_state()
        if self._webrtc is not None:
            self._webrtc.close()
            self._webrtc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()
