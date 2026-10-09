"""
Tests for the v2.6.1 voiceprint / emotion modernization
=======================================================

Covers:
    - ``MFCCExtractor`` enhanced track: Δ / ΔΔ coefficients + CMVN
    - ``VOICEPRINT_BACKENDS`` and the ``enhanced_mfcc`` factory/verifier plumbing
    - ``SpeakerVerifier(enhanced_mfcc=True)`` end-to-end verification
    - ``EmotionRecognizer`` joint feature vector (pitch dynamics, spectral flux,
      spectral flatness and the MFCC summary)
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from vram_core.emotion_recognition import (
    AudioFeatures,
    EmotionRecognizer,
    Wav2Vec2EmotionEngine,
    _is_emotion_head,
)
from vram_core.speaker_diarization import (
    MFCCExtractor,
    VOICEPRINT_BACKENDS,
    create_voiceprint_extractor,
)
from vram_core.speaker_verification import SpeakerVerifier

SAMPLE_RATE = 16000


def synth_voice(seed: int = 0, n: int = 6400, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Deterministic synthetic voiced signal (different partials per seed)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / float(sample_rate)
    partials = (120 + 7 * seed, 240 + 11 * seed, 360 + 13 * seed)
    base = sum(np.sin(2 * np.pi * f * t) for f in partials)
    return (0.3 * base + 0.05 * rng.standard_normal(n)).astype(np.float32)


def make_sine(duration_s: float = 1.0, freq: float = 200.0, amp: float = 0.3) -> np.ndarray:
    """Steady sine tone (spectrally pure)."""
    t = np.linspace(0.0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestEnhancedMFCCExtractor:
    """The enhanced MFCC track (Δ / ΔΔ + CMVN)."""

    def test_baseline_dimensions_are_unchanged(self):
        """Legacy defaults keep the pre-2.6.1 embedding width."""
        assert MFCCExtractor(n_mfcc=13).dim == 26
        assert MFCCExtractor(n_mfcc=13, include_std=False).dim == 13
        assert MFCCExtractor(n_mfcc=13).is_enhanced is False

    def test_enhanced_dimensions_include_the_deltas(self):
        """Δ and ΔΔ each add ``n_mfcc`` coefficients to the pooled statistics."""
        assert MFCCExtractor(n_mfcc=13, use_delta=True).dim == 78
        assert MFCCExtractor(n_mfcc=13, use_delta=True, include_std=False).dim == 39
        assert MFCCExtractor(n_mfcc=13, use_delta=True, cmvn=True).is_enhanced is True

    def test_enhanced_embedding_is_finite_and_normalised(self):
        """The enhanced embedding is a unit vector of the expected width."""
        extractor = MFCCExtractor(n_mfcc=13, use_delta=True, cmvn=True)
        embedding = extractor.embed(synth_voice(1))
        assert embedding.shape == (78,)
        assert np.all(np.isfinite(embedding))
        assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-4)

    def test_dynamic_features_append_zero_deltas_for_constant_input(self):
        """A constant feature matrix has no temporal dynamics."""
        extractor = MFCCExtractor(n_mfcc=3, use_delta=True)
        features = extractor.dynamic_features(np.ones((3, 12), dtype=np.float32))
        assert features.shape == (9, 12)
        np.testing.assert_allclose(features[3:9], 0.0, atol=1e-6)

    def test_cmvn_normalises_the_static_block(self):
        """CMVN yields zero mean / unit variance per coefficient."""
        extractor = MFCCExtractor(n_mfcc=4, use_delta=True, cmvn=True)
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((4, 40)).astype(np.float32) * 5.0 + 3.0
        static_block = extractor.dynamic_features(raw)[:4]

        np.testing.assert_allclose(static_block.mean(axis=1), 0.0, atol=1e-5)
        np.testing.assert_allclose(static_block.std(axis=1), 1.0, atol=1e-4)

    def test_cmvn_without_delta_warns(self, caplog):
        """The degenerate combination is reported instead of silently ignored."""
        with caplog.at_level(logging.WARNING, logger="vram_core.speaker_diarization"):
            MFCCExtractor(n_mfcc=4, cmvn=True)
        assert any("cmvn has no effect" in record.message for record in caplog.records)

    def test_enhanced_extractor_still_distinguishes_voices(self):
        """Δ/ΔΔ features keep the self-similarity ordering."""
        extractor = MFCCExtractor(n_mfcc=13, use_delta=True, cmvn=True)
        a1 = extractor.embed(synth_voice(1))
        a2 = extractor.embed(synth_voice(1))
        b = extractor.embed(synth_voice(4))
        assert float(np.dot(a1, a2)) > float(np.dot(a1, b))

    def test_baseline_extractor_is_bit_identical(self):
        """The baseline path keeps deterministic, legacy-compatible output."""
        extractor = MFCCExtractor(n_mfcc=10)
        audio = synth_voice(2)
        assert np.array_equal(extractor.embed(audio), extractor.embed(audio))


class TestVoiceprintBackendPlumbing:
    """``VOICEPRINT_BACKENDS`` + ``enhanced_mfcc`` factory wiring."""

    def test_backend_identifiers_are_documented(self):
        """The dual-track mechanism advertises auto / onnx / mfcc."""
        assert VOICEPRINT_BACKENDS == ("auto", "onnx", "mfcc")

    def test_factory_defaults_to_the_baseline_mfcc_track(self):
        """Legacy callers still get the pre-2.6.1 extractor."""
        extractor = create_voiceprint_extractor("mfcc")
        assert isinstance(extractor, MFCCExtractor)
        assert extractor.is_enhanced is False

    def test_factory_can_build_the_enhanced_track(self):
        """``enhanced_mfcc=True`` switches on Δ/ΔΔ + CMVN."""
        extractor = create_voiceprint_extractor("mfcc", n_mfcc=13, enhanced_mfcc=True)
        assert isinstance(extractor, MFCCExtractor)
        assert extractor.is_enhanced is True
        assert extractor.cmvn is True
        assert extractor.dim == 78

    def test_onnx_fallback_inherits_the_enhanced_track(self):
        """A missing ONNX model degrades to the *same* feature track."""
        extractor = create_voiceprint_extractor(
            "onnx", model_path="definitely_missing.onnx",
            n_mfcc=13, enhanced_mfcc=True,
        )
        assert isinstance(extractor, MFCCExtractor)
        assert extractor.is_enhanced is True

    def test_unknown_backend_falls_back_to_mfcc(self):
        """Unsupported identifiers degrade instead of raising."""
        extractor = create_voiceprint_extractor("not-a-backend")
        assert isinstance(extractor, MFCCExtractor)


class TestSpeakerVerifierEnhanced:
    """The verifier plumbing for the enhanced voiceprint track."""

    def test_enhanced_backend_registers_and_verifies(self):
        """Enrollment and verification agree on the enhanced embedding."""
        verifier = SpeakerVerifier(extractor_backend="mfcc", enhanced_mfcc=True)
        assert verifier.extractor_name == "mfcc"
        assert verifier.extractor.is_enhanced is True

        voice = synth_voice(11)
        verifier.register("alice", voice)
        result = verifier.verify("alice", voice)
        assert result.verified
        assert result.confidence == pytest.approx(1.0, abs=1e-4)

    def test_legacy_defaults_keep_the_baseline_track(self):
        """``enhanced_mfcc`` defaults to off, so embeddings are unchanged."""
        verifier = SpeakerVerifier(extractor_backend="mfcc")
        assert verifier.extractor.is_enhanced is False
        assert verifier.enhanced_mfcc is False

    def test_builtin_backend_is_untouched(self):
        """The built-in MFCC statistics path stays the default."""
        verifier = SpeakerVerifier()
        assert verifier.extractor is None
        assert verifier.extractor_name == "builtin"


def rule_recognizer() -> EmotionRecognizer:
    """
    Recognizer pinned to the rule engine.

    The joint feature vector lives in the rule-based path, and pinning the
    backend keeps these tests independent of whether ``transformers`` and a
    wav2vec2 checkpoint happen to be installed.
    """
    return EmotionRecognizer(backend="rule")


class TestEmotionJointFeatures:
    """Joint (energy + prosody + spectral + MFCC) emotion features."""

    def test_new_descriptors_are_extracted(self):
        """Every v2.6.1 descriptor is populated for real audio."""
        features = rule_recognizer().extract_features(make_sine())
        assert isinstance(features, AudioFeatures)
        assert features.f0_range >= 0.0
        assert 0.0 <= features.voiced_ratio <= 1.0
        assert features.spectral_flux >= 0.0
        assert features.spectral_flatness >= 0.0
        assert features.mfcc_summary, "MFCC block is missing from the joint vector"

    def test_joint_vector_contains_every_block(self):
        """``as_vector`` concatenates all descriptors plus the MFCC summary."""
        features = rule_recognizer().extract_features(make_sine())
        vector = features.as_vector()
        assert len(vector) == 11 + len(features.mfcc_summary)
        assert all(isinstance(value, float) for value in vector)

    def test_noise_is_flatter_and_more_fluxy_than_a_sine(self):
        """The new cues separate steady vowels from noise-like audio."""
        recognizer = rule_recognizer()
        rng = np.random.default_rng(0)
        noise = (0.3 * rng.standard_normal(SAMPLE_RATE)).astype(np.float32)

        sine_features = recognizer.extract_features(make_sine())
        noise_features = recognizer.extract_features(noise)

        assert noise_features.spectral_flatness > sine_features.spectral_flatness
        assert noise_features.spectral_flux > sine_features.spectral_flux

    def test_empty_audio_returns_zeroed_features(self):
        """Degenerate input is handled without touching the new helpers."""
        features = rule_recognizer().extract_features(np.array([], dtype=np.float32))
        assert features.as_vector() == [0.0] * 11

    def test_classification_still_yields_a_valid_distribution(self):
        """The joint weights keep the rule engine's output well formed."""
        recognizer = rule_recognizer()
        for amplitude in (0.01, 0.3, 0.8):
            result = recognizer.analyze(make_sine(amp=amplitude))
            assert result.emotion in {"happy", "sad", "angry", "neutral", "surprised"}
            assert 0.0 <= result.confidence <= 1.0
            assert sum(result.all_scores.values()) == pytest.approx(1.0, abs=0.01)

    def test_short_audio_is_safe(self):
        """Very short buffers exercise the padded spectral path."""
        recognizer = rule_recognizer()
        result = recognizer.analyze(np.full(200, 0.05, dtype=np.float32))
        assert result.emotion in {"happy", "sad", "angry", "neutral", "surprised"}
        assert np.isfinite(result.confidence)


class TestEmotionBackendRobustness:
    """Label normalisation and fail-safe DL degradation (v2.6.1)."""

    def test_short_codes_map_onto_the_project_vocabulary(self):
        """superb-er's neu/hap/ang/sad codes become the standard emotions."""
        assert Wav2Vec2EmotionEngine._normalize_label("neu") == "neutral"
        assert Wav2Vec2EmotionEngine._normalize_label("hap") == "happy"
        assert Wav2Vec2EmotionEngine._normalize_label("ang") == "angry"
        assert Wav2Vec2EmotionEngine._normalize_label("sad") == "sad"
        assert Wav2Vec2EmotionEngine._normalize_label("HAPPY") == "happy"

    def test_unknown_labels_are_surfaced_unchanged(self):
        """An unrecognised label is never guessed into a wrong emotion."""
        assert Wav2Vec2EmotionEngine._normalize_label("LABEL_0") == "LABEL_0"

    def test_emotion_head_detection(self):
        """Only classifiers with emotion labels are accepted."""
        def pipeline_with(labels):
            config = SimpleNamespace(id2label=labels)
            return SimpleNamespace(model=SimpleNamespace(config=config))

        assert _is_emotion_head(pipeline_with({0: "neu", 1: "hap"})) is True
        assert _is_emotion_head(pipeline_with({0: "LABEL_0", 1: "1"})) is False
        assert _is_emotion_head(SimpleNamespace()) is False

    def test_dl_inference_failure_degrades_to_the_rule_engine(self):
        """A crashing classifier must not break ``analyze()``."""

        class ExplodingEngine:
            """Stand-in DL engine whose inference always fails."""

            def predict(self, audio, sample_rate=16000):
                raise RuntimeError("buffer too short for the conv front-end")

            def close(self):
                """Match the engine lifecycle expected by ``EmotionRecognizer``."""

        recognizer = rule_recognizer()
        recognizer._active_backend = "wav2vec2"
        recognizer._wav2vec2 = ExplodingEngine()

        result = recognizer.analyze(make_sine())
        assert result.backend_used == "rule"
        assert result.emotion in {"happy", "sad", "angry", "neutral", "surprised"}
        assert isinstance(result.features, AudioFeatures)
        assert sum(result.all_scores.values()) == pytest.approx(1.0, abs=0.01)
