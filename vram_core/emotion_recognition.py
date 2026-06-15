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
from typing import Dict, List, Optional

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
            self._classifier = hf_pipeline(
                "audio-classification",
                model=self.model_name,
                device=self.device,
                top_k=None,  # Return all scores
            )
            logger.info("wav2vec2 emotion model loaded successfully")
        except Exception as e:
            logger.warning("Failed to load model %s: %s", self.model_name, e)
            # Try alternative models
            for alt_model in _MODEL_ALTERNATIVES:
                if alt_model != self.model_name:
                    try:
                        logger.info("Trying alternative model: %s", alt_model)
                        self._classifier = hf_pipeline(
                            "audio-classification",
                            model=alt_model,
                            device=self.device,
                            top_k=None,
                        )
                        self.model_name = alt_model
                        logger.info("Alternative model loaded: %s", alt_model)
                        return
                    except Exception:
                        continue
            raise RuntimeError(f"Failed to load any wav2vec2 emotion model: {e}")

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
        """Normalize model labels to standard emotion names."""
        label_map = {
            "angry": "angry",
            "anger": "angry",
            "disgust": "disgust",
            "disgusted": "disgust",
            "fear": "fear",
            "fearful": "fear",
            "scared": "fear",
            "happy": "happy",
            "joy": "happy",
            "neutral": "neutral",
            "sad": "sad",
            "sadness": "sad",
            "surprise": "surprise",
            "surprised": "surprise",
        }
        return label_map.get(label, label)

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

        return AudioFeatures(
            rms_energy=rms_energy,
            zero_crossing_rate=zcr,
            mean_f0=mean_f0,
            std_f0=std_f0,
            energy_variance=energy_variance,
            energy_range=energy_range,
            speech_rate_proxy=speech_rate_proxy,
        )

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
        """Classify emotion using rule-based scoring."""
        scores: Dict[str, float] = {}

        e = min(features.rms_energy * 10, 1.0)
        z = min(features.zero_crossing_rate * 5, 1.0)
        f0_mean = features.mean_f0 / 400.0
        f0_std = min(features.std_f0 / 100.0, 1.0)
        e_var = min(features.energy_variance * 200, 1.0)
        e_range = min(features.energy_range * 10, 1.0)
        rate = min(features.speech_rate_proxy / 10.0, 1.0)

        scores["angry"] = 0.30 * e + 0.20 * z + 0.20 * f0_std + 0.15 * e_var + 0.15 * e_range
        scores["happy"] = (
            0.20 * e + 0.25 * z + 0.15 * f0_mean + 0.15 * rate +
            0.10 * e_var + 0.15 * max(0, 1.0 - f0_std)
        )
        scores["sad"] = (
            0.30 * (1.0 - e) + 0.20 * (1.0 - z) + 0.15 * (1.0 - f0_mean) +
            0.15 * (1.0 - rate) + 0.20 * (1.0 - e_var)
        )
        scores["neutral"] = (
            0.25 * (1.0 - abs(e - 0.5) * 2) + 0.20 * (1.0 - abs(z - 0.4) * 2) +
            0.25 * (1.0 - f0_std) + 0.15 * (1.0 - e_var) +
            0.15 * (1.0 - abs(rate - 0.4) * 2)
        )
        scores["surprised"] = (
            0.25 * e + 0.15 * z + 0.25 * f0_mean + 0.20 * e_range + 0.15 * f0_std
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

        Uses wav2vec2 if available, otherwise falls back to rule engine.

        Args:
            audio: Audio signal (float32, mono).
            sample_rate: Sample rate in Hz.

        Returns:
            EmotionResult with detected emotion and confidence.
        """
        if self._active_backend == "wav2vec2" and self._wav2vec2 is not None:
            return self._wav2vec2.predict(audio, sample_rate)

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