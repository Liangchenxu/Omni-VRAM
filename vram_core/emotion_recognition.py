"""
Emotion Recognition Module for vram_core
=========================================

Multi-backend emotion classification with deep learning and rule-based fallbacks:

1. **wav2vec2-base-emotion** (preferred): HuggingFace pretrained model
   - Requires: pip install transformers torch
   - Features: Deep learning-based, 7 emotions, high accuracy
   - Supports: happy, sad, angry, neutral, surprise, fear, disgust

2. **Rule-based Engine** (fallback): Handcrafted acoustic features
   - No external dependencies beyond numpy
   - Features: Energy, ZCR, F0, rhythm analysis
   - Supports: happy, sad, angry, neutral, surprised

Usage:
    from vram_core.emotion_recognition import EmotionRecognizer

    # Auto-detect best backend
    recognizer = EmotionRecognizer()
    result = recognizer.analyze(audio_array, sample_rate=16000)
    print(result.emotion, result.confidence)

    # Force specific backend
    recognizer = EmotionRecognizer(backend="wav2vec2")
    recognizer = EmotionRecognizer(backend="rule")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from vram_core.utils import (
    ensure_float32,
    compute_rms_energy_per_frame,
    compute_zcr_per_frame,
)

logger = logging.getLogger(__name__)


# 鈹€鈹€ Backend Detection 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_WAV2VEC2_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import pipeline as hf_pipeline, AutoModelForAudioClassification, AutoFeatureExtractor
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

if _TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE:
    _WAV2VEC2_AVAILABLE = True
    logger.info("wav2vec2 emotion recognition available (transformers + torch)")
else:
    logger.info(
        "wav2vec2 not available, using rule-based fallback. "
        "Install with: pip install transformers torch"
    )


# 鈹€鈹€ Supported Emotions 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
WAV2VEC2_EMOTIONS = ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"]
RULE_EMOTIONS = ["happy", "sad", "angry", "neutral", "surprised"]

# wav2vec2 model name (ehcalabrese/wav2vec2-lg-xlsr-en-speech-emotion-recognition)
_DEFAULT_MODEL = "ehcalabrese/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

# Alternative models
_MODEL_ALTERNATIVES = [
    "ehcalabrese/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    "superb/wav2vec2-base-superb-er",
    "facebook/wav2vec2-base",
]

# Label spellings used by the public emotion checkpoints, mapped onto the
# project's emotion vocabulary:
#   * ``superb/wav2vec2-base-superb-er`` emits the 3-letter codes
#     ``neu`` / ``hap`` / ``ang`` / ``sad``
#   * RAVDESS-style checkpoints emit full words (``fear``, ``disgust``, ...)
# The table doubles as the "is this really an emotion classifier?" test: a bare
# checkpoint such as ``facebook/wav2vec2-base`` only produces ``LABEL_0`` style
# placeholders, which never match and therefore disable the DL backend.
_EMOTION_LABEL_ALIASES = {
    # short codes
    "neu": "neutral",
    "hap": "happy",
    "ang": "angry",
    "sad": "sad",
    "fea": "fear",
    "dis": "disgust",
    # full words and synonyms
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "scared": "fear",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "neutral": "neutral",
    "calm": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
}


def _is_emotion_head(classifier) -> bool:
    """
    Decide whether an ``audio-classification`` pipeline is an *emotion* model.

    Args:
        classifier: A transformers pipeline (or any object exposing
            ``model.config.id2label``).

    Returns:
        True when at least one configured label maps onto the project's emotion
        vocabulary; False for placeholder heads (``LABEL_0``) and unrelated
        classifiers.
    """
    config = getattr(getattr(classifier, "model", None), "config", None)
    labels = getattr(config, "id2label", None) or {}
    for raw_label in labels.values():
        text = str(raw_label).strip().lower()
        if not text or text.startswith("label_") or text.isdigit():
            continue
        if text in _EMOTION_LABEL_ALIASES:
            return True
    return False



@dataclass
class AudioFeatures:
    """Extracted audio features for emotion classification (rule-based)."""
    rms_energy: float = 0.0
    zero_crossing_rate: float = 0.0
    mean_f0: float = 0.0
    std_f0: float = 0.0
    energy_variance: float = 0.0
    energy_range: float = 0.0
    speech_rate_proxy: float = 0.0
    # ---- Robustness features (v2.6.1) ----
    f0_range: float = 0.0            # pitch dynamics: max - min of the F0 contour
    voiced_ratio: float = 0.0        # fraction of frames with a valid pitch
    spectral_flux: float = 0.0       # mean frame-to-frame spectral change
    spectral_flatness: float = 0.0   # geometric/arithmetic mean of the spectrum
    mfcc_summary: Dict[str, float] = field(default_factory=dict)

    def as_vector(self) -> List[float]:
        """
        Joint feature vector used by the rule engine.

        Concatenates the energy / prosody descriptors with the spectral and MFCC
        summaries so no single feature can dominate a decision (v2.6.1).

        Returns:
            List of floats in a stable order: energy, ZCR, pitch, rhythm,
            spectral shape, then the mel-cepstral summary.
        """
        vector = [
            self.rms_energy, self.zero_crossing_rate, self.mean_f0, self.std_f0,
            self.energy_variance, self.energy_range, self.speech_rate_proxy,
            self.f0_range, self.voiced_ratio, self.spectral_flux,
            self.spectral_flatness,
        ]
        for key in sorted(self.mfcc_summary):
            vector.append(float(self.mfcc_summary[key]))
        return vector


@dataclass
class EmotionResult:
    """Result of emotion analysis."""
    emotion: str
    confidence: float
    features: Optional[AudioFeatures] = None
    all_scores: Dict[str, float] = field(default_factory=dict)
    backend_used: str = "unknown"

    def __repr__(self) -> str:
        return (
            f"EmotionResult(emotion='{self.emotion}', "
            f"confidence={self.confidence:.3f}, backend='{self.backend_used}')"
        )


class Wav2Vec2EmotionEngine:
    """
    Deep learning emotion recognition using wav2vec2.

    Uses HuggingFace transformers pipeline for audio classification
    with a pretrained wav2vec2 emotion model.

    Args:
        model_name: HuggingFace model name or path.
        device: Device to run on ("cpu", "cuda", "auto").
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "auto",
    ):
        if not _WAV2VEC2_AVAILABLE:
            raise RuntimeError(
                "wav2vec2 requires transformers and torch. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model_name

        # Auto-detect device
        if device == "auto":
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = 0  # GPU 0
            else:
                self.device = -1  # CPU
        elif device == "cuda":
            self.device = 0
        else:
            self.device = -1

        self._classifier = None
        self._load_model()

    def _load_model(self):
        """Load the wav2vec2 emotion classification model."""
        try:
            logger.info("Loading wav2vec2 emotion model: %s", self.model_name)
            self._classifier = self._build_pipeline(self.model_name)
            logger.info("wav2vec2 emotion model loaded successfully")
        except Exception as e:
            logger.warning("Failed to load model %s: %s", self.model_name, e)
            # Try alternative models
            for alt_model in _MODEL_ALTERNATIVES:
                if alt_model != self.model_name:
                    try:
                        logger.info("Trying alternative model: %s", alt_model)
                        self._classifier = self._build_pipeline(alt_model)
                        self.model_name = alt_model
                        logger.info("Alternative model loaded: %s", alt_model)
                        return
                    except Exception:
                        continue
            raise RuntimeError(f"Failed to load any wav2vec2 emotion model: {e}")

    def _build_pipeline(self, model_name: str):
        """
        Build (and validate) an ``audio-classification`` pipeline.

        A model that does not expose an emotion label set -- e.g. a bare
        ``wav2vec2-base`` checkpoint whose randomly initialised head only emits
        ``LABEL_0`` style outputs -- is rejected so the recognizer can fall back
        to the rule engine instead of reporting meaningless labels.

        Args:
            model_name: HuggingFace model identifier or local path.

        Returns:
            The ready-to-use classification pipeline.

        Raises:
            RuntimeError: When the pipeline cannot be built or is not an emotion
                classifier.
        """
        classifier = hf_pipeline(
            "audio-classification",
            model=model_name,
            device=self.device,
            top_k=None,  # Return all scores
        )
        if not _is_emotion_head(classifier):
            raise RuntimeError(
                f"{model_name} does not expose an emotion label set "
                "(its classifier head is not trained for emotions)"
            )
        return classifier

    def predict(self, audio: np.ndarray, sample_rate: int = 16000) -> EmotionResult:
        """
        Predict emotion from audio using wav2vec2.

        Args:
            audio: Audio signal (float32, mono).
            sample_rate: Sample rate in Hz.

        Returns:
            EmotionResult with predicted emotion and confidence scores.
        """
        if self._classifier is None:
            raise RuntimeError("Model not loaded")

        audio = ensure_float32(audio)

        try:
            # Run inference
            results = self._classifier(
                audio,
                sampling_rate=sample_rate,
            )

            # Parse results
            scores = {}
            for item in results:
                label = item["label"].lower()
                score = item["score"]
                # Map model labels to standard names
                label = self._normalize_label(label)
                scores[label] = float(score)

            # Find best emotion
            best_emotion = max(scores, key=scores.get)
            confidence = scores[best_emotion]

            return EmotionResult(
                emotion=best_emotion,
                confidence=confidence,
                all_scores=scores,
                backend_used="wav2vec2",
            )

        except Exception as e:
            logger.error("wav2vec2 inference failed: %s", e)
            raise

    @staticmethod
    def _normalize_label(label: str) -> str:
        """
        Normalize model labels to standard emotion names.

        Uses :data:`_EMOTION_LABEL_ALIASES`, so the short superb-er codes and the
        RAVDESS-style full words all land on the project vocabulary. Unknown
        labels are returned unchanged (they are surfaced to the caller instead of
        being silently guessed).
        """
        text = str(label).strip()
        return _EMOTION_LABEL_ALIASES.get(text.lower(), text)

    def close(self):
        """Release model resources."""
        self._classifier = None
        if _TORCH_AVAILABLE:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class EmotionRecognizer:
    """
    Multi-backend emotion recognizer.

    Features:
        - wav2vec2 deep learning model (preferred, 7 emotions)
        - Rule-based acoustic feature analysis (fallback, 5 emotions)
        - Auto backend selection
        - Confidence scores for all emotions

    Args:
        backend: Backend to use ("auto", "wav2vec2", "rule").
        model_name: HuggingFace model name (wav2vec2 only).
        device: Device for inference ("auto", "cpu", "cuda").
        frame_size_ms: Frame size for rule-based analysis (ms).
        min_f0: Minimum F0 for pitch estimation (Hz).
        max_f0: Maximum F0 for pitch estimation (Hz).

    Usage:
        recognizer = EmotionRecognizer()
        result = recognizer.analyze(audio, sample_rate=16000)
        print(result.emotion, result.confidence)
        print(result.all_scores)
    """

    def __init__(
        self,
        backend: str = "auto",
        model_name: str = _DEFAULT_MODEL,
        device: str = "auto",
        frame_size_ms: int = 25,
        min_f0: float = 50.0,
        max_f0: float = 500.0,
    ):
        self._backend_type = backend
        self.frame_size_ms = frame_size_ms
        self.min_f0 = min_f0
        self.max_f0 = max_f0

        self._wav2vec2: Optional[Wav2Vec2EmotionEngine] = None
        self._active_backend = "rule"
        self._mfcc_extractor = None
        self._mfcc_unavailable = False
        self._init_backend(model_name, device)

    def _init_backend(self, model_name: str, device: str):
        """Initialize the emotion recognition backend."""
        if self._backend_type == "wav2vec2" or (
            self._backend_type == "auto" and _WAV2VEC2_AVAILABLE
        ):
            try:
                self._wav2vec2 = Wav2Vec2EmotionEngine(
                    model_name=model_name, device=device,
                )
                self._active_backend = "wav2vec2"
                logger.info("Using wav2vec2 emotion backend")
            except Exception as e:
                logger.warning("wav2vec2 init failed (%s), falling back to rule engine", e)
                self._active_backend = "rule"
        else:
            self._active_backend = "rule"
            logger.info("Using rule-based emotion backend")

    @property
    def backend(self) -> str:
        """Return the name of the active backend."""
        return self._active_backend

    @property
    def supported_emotions(self) -> List[str]:
        """Return list of supported emotions for the active backend."""
        if self._active_backend == "wav2vec2":
            return WAV2VEC2_EMOTIONS.copy()
        return RULE_EMOTIONS.copy()

    # 鈹€鈹€ Rule-based Feature Extraction 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def extract_features(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> AudioFeatures:
        """Extract acoustic features from audio signal (rule-based)."""
        if len(audio) == 0:
            return AudioFeatures()

        audio = ensure_float32(audio)

        frame_size = int(sample_rate * self.frame_size_ms / 1000)

        # RMS Energy per frame
        energies = compute_rms_energy_per_frame(audio, frame_size)
        rms_energy = float(np.mean(energies))
        energy_variance = float(np.var(energies))
        energy_range = float(np.max(energies) - np.min(energies)) if len(energies) > 1 else 0.0

        # Zero-Crossing Rate
        zcr_arr = compute_zcr_per_frame(audio, frame_size)
        zcr = float(np.mean(zcr_arr)) if len(zcr_arr) > 0 else 0.0

        # F0 estimation
        f0_values = self._estimate_f0_series(audio, sample_rate, frame_size)
        mean_f0 = float(np.mean(f0_values)) if len(f0_values) > 0 else 0.0
        std_f0 = float(np.std(f0_values)) if len(f0_values) > 0 else 0.0

        # Speech rate proxy
        speech_rate_proxy = self._compute_speech_rate(energies)

        # Robustness features (v2.6.1): pitch dynamics, voiced ratio, spectral
        # shape and an MFCC summary -- combined with energy/ZCR they form the
        # joint feature vector used by the rule engine.
        f0_range = (
            float(np.max(f0_values) - np.min(f0_values)) if len(f0_values) > 0 else 0.0
        )
        n_frames = max(1, len(audio) // max(frame_size, 1))
        voiced_ratio = min(1.0, len(f0_values) / n_frames)
        spectral_flux, spectral_flatness = self._spectral_features(
            audio, sample_rate, frame_size
        )
        mfcc_summary = self._mfcc_summary(audio, sample_rate)

        return AudioFeatures(
            rms_energy=rms_energy,
            zero_crossing_rate=zcr,
            mean_f0=mean_f0,
            std_f0=std_f0,
            energy_variance=energy_variance,
            energy_range=energy_range,
            speech_rate_proxy=speech_rate_proxy,
            f0_range=f0_range,
            voiced_ratio=voiced_ratio,
            spectral_flux=spectral_flux,
            spectral_flatness=spectral_flatness,
            mfcc_summary=mfcc_summary,
        )

    def _spectral_features(
        self, audio: np.ndarray, sample_rate: int, frame_size: int,
    ) -> Tuple[float, float]:
        """
        Mean spectral flux and spectral flatness of an audio buffer.

        Flux (frame-to-frame magnitude change) separates agitated, plosive-rich
        speech from steady vowels; flatness (geometric vs arithmetic mean of the
        spectrum) measures how noise-like each frame is. Both are computed with a
        single vectorised FFT over Hann-windowed frames.

        Args:
            audio: Mono audio buffer (float32).
            sample_rate: Sample rate in Hz (kept for signature symmetry).
            frame_size: Analysis frame length in samples.

        Returns:
            Tuple ``(spectral_flux, spectral_flatness)``, both >= 0.
        """
        frame_size = max(2, int(frame_size))
        if len(audio) < frame_size:
            frames = np.pad(audio, (0, frame_size - len(audio)))[np.newaxis, :]
        else:
            hop = max(1, frame_size // 2)
            count = 1 + (len(audio) - frame_size) // hop
            frames = np.stack(
                [audio[i * hop: i * hop + frame_size] for i in range(count)]
            )

        window = np.hanning(frame_size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) + 1e-10

        if spectrum.shape[0] < 2:
            flux = 0.0
        else:
            diff = np.diff(spectrum, axis=0)
            flux = float(np.mean(np.sqrt(np.sum(diff ** 2, axis=1))))

        geometric = np.exp(np.mean(np.log(spectrum), axis=1))
        arithmetic = np.mean(spectrum, axis=1)
        flatness = float(np.mean(geometric / np.maximum(arithmetic, 1e-10)))
        return flux, flatness

    def _get_mfcc_extractor(self, sample_rate: int):
        """Lazily build the shared MFCC front-end (None when unavailable)."""
        if self._mfcc_extractor is None and not self._mfcc_unavailable:
            try:
                from vram_core.speaker_diarization import MFCCExtractor

                self._mfcc_extractor = MFCCExtractor(
                    n_mfcc=13, sample_rate=int(sample_rate),
                )
            except Exception as error:  # noqa: BLE001 - optional dependency
                logger.info("MFCC summary disabled (%s)", error)
                self._mfcc_unavailable = True
        return self._mfcc_extractor

    def _mfcc_summary(
        self, audio: np.ndarray, sample_rate: int, n_coeffs: int = 5,
    ) -> Dict[str, float]:
        """
        Mean/std of the leading mel-cepstral coefficients.

        The summary joins the energy / spectral descriptors in
        :meth:`AudioFeatures.as_vector`, so the rule engine no longer depends on
        loudness and pitch alone.

        Args:
            audio: Mono audio buffer.
            sample_rate: Sample rate in Hz.
            n_coeffs: Number of leading coefficients summarised.

        Returns:
            ``{"mfcc_mean_k": ..., "mfcc_std_k": ...}`` for ``k < n_coeffs``
            (empty dict when the MFCC front-end is unavailable).
        """
        extractor = self._get_mfcc_extractor(sample_rate)
        if extractor is None:
            return {}
        try:
            mfcc = extractor.extract_mfcc(audio, sample_rate)
        except Exception as error:  # noqa: BLE001 - never break analysis
            logger.debug("MFCC summary failed (%s)", error)
            return {}
        if np.asarray(mfcc).ndim != 2 or np.asarray(mfcc).shape[1] == 0:
            return {}

        summary: Dict[str, float] = {}
        for index in range(min(int(n_coeffs), mfcc.shape[0])):
            row = np.asarray(mfcc[index], dtype=np.float32)
            summary[f"mfcc_mean_{index}"] = float(np.mean(row))
            summary[f"mfcc_std_{index}"] = float(np.std(row))
        return summary

    def _estimate_f0_series(
        self, audio: np.ndarray, sample_rate: int, frame_size: int,
    ) -> np.ndarray:
        """Estimate F0 contour using autocorrelation."""
        min_lag = int(sample_rate / self.max_f0)
        max_lag = int(sample_rate / self.min_f0)

        f0_values = []
        n_frames = max(1, len(audio) // frame_size)

        for i in range(n_frames):
            start = i * frame_size
            end = min(start + frame_size, len(audio))
            frame = audio[start:end]

            if len(frame) < max_lag + 1:
                continue

            frame_centered = frame - np.mean(frame)
            energy = np.sum(frame_centered ** 2)
            if energy < 1e-10:
                continue

            autocorr = np.correlate(frame_centered, frame_centered, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]

            if len(autocorr) <= max_lag:
                continue

            search_region = autocorr[min_lag:max_lag + 1]
            if len(search_region) == 0:
                continue

            peak_idx = np.argmax(search_region)
            peak_val = search_region[peak_idx] / autocorr[0]

            if peak_val > 0.3:
                lag = peak_idx + min_lag
                if lag > 0:
                    f0 = sample_rate / lag
                    if self.min_f0 <= f0 <= self.max_f0:
                        f0_values.append(f0)

        return np.array(f0_values, dtype=np.float32)

    def _compute_speech_rate(self, energies: np.ndarray) -> float:
        """Estimate speech rate proxy from energy envelope."""
        if len(energies) < 3:
            return 0.0

        kernel_size = min(3, len(energies))
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(energies, kernel, mode='same')

        threshold = np.mean(smoothed)
        peaks = 0
        for i in range(1, len(smoothed) - 1):
            if (smoothed[i] > smoothed[i - 1] and
                    smoothed[i] > smoothed[i + 1] and
                    smoothed[i] > threshold):
                peaks += 1

        duration_s = max(len(energies) * (self.frame_size_ms / 1000.0), 0.001)
        return peaks / duration_s

    def _classify_rule(self, features: AudioFeatures) -> EmotionResult:
        """
        Classify emotion using rule-based scoring over the joint feature vector.

        Every emotion score is a weighted combination of energy, prosody
        (F0 mean/spread/range, voiced ratio, rhythm), spectral shape (flux,
        flatness) and the MFCC summary (v2.6.1). Spreading the decision over
        complementary cues removes the single-feature false positives that a
        loudness-only or pitch-only rule set produces.
        """
        scores: Dict[str, float] = {}

        e = min(features.rms_energy * 10, 1.0)
        z = min(features.zero_crossing_rate * 5, 1.0)
        f0_mean = features.mean_f0 / 400.0
        f0_std = min(features.std_f0 / 100.0, 1.0)
        e_var = min(features.energy_variance * 200, 1.0)
        e_range = min(features.energy_range * 10, 1.0)
        rate = min(features.speech_rate_proxy / 10.0, 1.0)
        # ---- joint cues (v2.6.1) ----
        f0_range = min(features.f0_range / 200.0, 1.0)
        voiced = min(max(features.voiced_ratio, 0.0), 1.0)
        flux = min(features.spectral_flux / 0.5, 1.0)
        flatness = min(features.spectral_flatness * 5.0, 1.0)
        mfcc_c0 = min(abs(features.mfcc_summary.get("mfcc_mean_0", 0.0)) / 40.0, 1.0)

        scores["angry"] = (
            0.25 * e + 0.15 * z + 0.15 * f0_std + 0.10 * e_var + 0.10 * e_range +
            0.15 * flux + 0.10 * max(flatness, mfcc_c0)
        )
        scores["happy"] = (
            0.15 * e + 0.15 * z + 0.10 * f0_mean + 0.15 * rate + 0.10 * e_var +
            0.10 * (1.0 - f0_std) + 0.15 * f0_range + 0.10 * (1.0 - flatness)
        )
        scores["sad"] = (
            0.25 * (1.0 - e) + 0.15 * (1.0 - z) + 0.10 * (1.0 - f0_mean) +
            0.10 * (1.0 - rate) + 0.15 * (1.0 - e_var) + 0.10 * (1.0 - f0_range) +
            0.15 * (1.0 - voiced)
        )
        scores["neutral"] = (
            0.20 * (1.0 - abs(e - 0.5) * 2) + 0.15 * (1.0 - abs(z - 0.4) * 2) +
            0.20 * (1.0 - f0_std) + 0.10 * (1.0 - e_var) +
            0.15 * (1.0 - abs(rate - 0.4) * 2) + 0.10 * (1.0 - flux) +
            0.10 * (1.0 - flatness)
        )
        scores["surprised"] = (
            0.20 * e + 0.10 * z + 0.20 * f0_mean + 0.15 * e_range +
            0.10 * f0_std + 0.15 * f0_range + 0.10 * flux
        )

        total = sum(scores.values()) + 1e-10
        scores = {k: v / total for k, v in scores.items()}

        best_emotion = max(scores, key=scores.get)
        confidence = scores[best_emotion]

        return EmotionResult(
            emotion=best_emotion,
            confidence=confidence,
            features=features,
            all_scores=scores,
            backend_used="rule",
        )

    # 鈹€鈹€ Public API 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> EmotionResult:
        """
        Analyze audio and return emotion classification.

        Uses wav2vec2 if available, otherwise falls back to rule engine. The DL
        backend is fail-safe (v2.6.1): a model that cannot score the buffer -- for
        instance a buffer too short for the convolutional feature extractor --
        degrades to the rule engine for that call instead of raising.

        Args:
            audio: Audio signal (float32, mono).
            sample_rate: Sample rate in Hz.

        Returns:
            EmotionResult with detected emotion and confidence.
        """
        if self._active_backend == "wav2vec2" and self._wav2vec2 is not None:
            try:
                result = self._wav2vec2.predict(audio, sample_rate)
            except Exception as error:  # noqa: BLE001 - robustness over purity
                logger.warning(
                    "wav2vec2 inference failed (%s); using the rule engine for "
                    "this buffer", error,
                )
            else:
                if result.features is None:
                    # Always expose the acoustic descriptors, whichever backend
                    # produced the label (v2.6.1): callers rely on
                    # ``EmotionResult.features`` for logging and diagnostics.
                    result.features = self.extract_features(audio, sample_rate)
                return result

        features = self.extract_features(audio, sample_rate)
        return self._classify_rule(features)

    def analyze_batch(
        self,
        audio_list: List[np.ndarray],
        sample_rate: int = 16000,
    ) -> List[EmotionResult]:
        """
        Analyze multiple audio clips.

        Args:
            audio_list: List of audio signals.
            sample_rate: Sample rate in Hz.

        Returns:
            List of EmotionResult for each audio clip.
        """
        results = []
        for audio in audio_list:
            results.append(self.analyze(audio, sample_rate))
        return results

    @staticmethod
    def available_backends() -> List[str]:
        """List available emotion recognition backends."""
        backends = ["rule"]
        if _WAV2VEC2_AVAILABLE:
            backends.insert(0, "wav2vec2")
        return backends

    def close(self):
        """Release resources."""
        if self._wav2vec2 is not None:
            self._wav2vec2.close()
            self._wav2vec2 = None

    def __del__(self):
        self.close()