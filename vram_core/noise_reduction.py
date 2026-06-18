"""
Noise Reduction Module for vram_core
=====================================

Professional-grade noise reduction with multiple backends:

1. **WebRTC APM** (preferred): Full audio processing pipeline with AEC, NS, AGC
   - Requires: pip install py-webrtc-audio-processing
   - Features: Acoustic Echo Cancellation, Noise Suppression, Automatic Gain Control

2. **Spectral Subtraction** (fallback): Pure numpy/scipy implementation
   - No external dependencies beyond numpy/scipy
   - Good for offline processing

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
from typing import Optional, List

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

        # Streaming state
        self._stream_buffer = np.array([], dtype=np.float32)
        self._noise_estimate = None

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

        Uses WebRTC APM if available, otherwise falls back to spectral subtraction.

        Args:
            audio: Input audio signal (float32, mono).
            sample_rate: Sample rate in Hz.

        Returns:
            Noise-reduced audio signal (float32).
        """
        if self._active_backend == "webrtc" and self._webrtc is not None:
            return self._webrtc.process(audio)
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

    def close(self):
        """Release resources."""
        self._stream_buffer = np.array([], dtype=np.float32)
        self._noise_estimate = None
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
