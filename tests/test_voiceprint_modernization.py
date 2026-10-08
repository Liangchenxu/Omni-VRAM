"""
Tests for the v2.6.0 voiceprint / diarization modernization.

Covers:
    - ``MFCCExtractor`` / ``ONNXEmbeddingExtractor`` / ``create_voiceprint_extractor``
    - ``AdaptiveCosineClusterer`` (duration-adaptive thresholds, BIC merge test)
    - ``SpeakerVerifier`` pluggable extractors, dynamic cosine pruning,
      duration-adaptive thresholds and the BIC veto
    - Backwards compatibility of the pre-2.6.0 verification defaults
"""

import numpy as np
import pytest

from vram_core.speaker_diarization import (
    AdaptiveCosineClusterer,
    BaseVoiceprintExtractor,
    MFCCExtractor,
    ONNXEmbeddingExtractor,
    available_voiceprint_backends,
    create_voiceprint_extractor,
)
from vram_core.speaker_verification import SpeakerVerifier


def synth_voice(seed: int = 0, n: int = 6400, sample_rate: int = 16000) -> np.ndarray:
    """Deterministic synthetic voiced signal (different partials per seed)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / float(sample_rate)
    partials = (120 + 7 * seed, 240 + 11 * seed, 360 + 13 * seed)
    base = sum(np.sin(2 * np.pi * f * t) for f in partials)
    return (0.3 * base + 0.05 * rng.standard_normal(n)).astype(np.float32)


class TestVoiceprintExtractors:
    """Pluggable voiceprint embedding front-ends."""

    def test_mfcc_extractor_dimensions(self):
        """``dim`` follows ``include_std`` and embeddings are L2-normalised."""
        extractor = MFCCExtractor(n_mfcc=13)
        assert extractor.dim == 26
        assert MFCCExtractor(n_mfcc=13, include_std=False).dim == 13

        embedding = extractor.embed(synth_voice(1))
        assert embedding.shape == (26,)
        assert embedding.dtype == np.float32
        assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-4)

    def test_mfcc_extractor_is_deterministic(self):
        """The cached filterbank / DCT produce bit-identical embeddings."""
        extractor = MFCCExtractor(n_mfcc=10)
        audio = synth_voice(2)
        assert np.array_equal(extractor.embed(audio), extractor.embed(audio))

    def test_mfcc_extractor_empty_audio(self):
        """Empty input yields an empty frame matrix and a zero embedding."""
        extractor = MFCCExtractor(n_mfcc=8)
        mfcc = extractor.extract_mfcc(np.array([], dtype=np.float32))
        assert mfcc.shape == (8, 0)
        embedding = extractor.embed(np.array([], dtype=np.float32))
        assert embedding.shape == (16,)
        assert not np.any(embedding)

    def test_mfcc_extractor_distinguishes_voices(self):
        """Different voices get different embeddings (self-similarity > cross)."""
        extractor = MFCCExtractor(n_mfcc=13)
        a1 = extractor.embed(synth_voice(1))
        a2 = extractor.embed(synth_voice(1))
        b = extractor.embed(synth_voice(4))
        assert float(np.dot(a1, a2)) > float(np.dot(a1, b))

    def test_onnx_extractor_falls_back_without_model(self):
        """A missing ONNX model degrades to the MFCC extractor."""
        extractor = ONNXEmbeddingExtractor(model_path="definitely_missing.onnx")
        assert extractor.is_fallback
        embedding = extractor.embed(synth_voice(3))
        assert embedding.ndim == 1 and embedding.size > 0
        assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-4)

    def test_create_voiceprint_extractor_backends(self):
        """The factory returns MFCC for ``mfcc`` / ``auto`` without a model."""
        assert isinstance(create_voiceprint_extractor("mfcc"), MFCCExtractor)
        assert isinstance(create_voiceprint_extractor("auto"), MFCCExtractor)
        # ONNX unavailable -> transparent MFCC fallback
        fallback = create_voiceprint_extractor("onnx", model_path="missing.onnx")
        assert isinstance(fallback, MFCCExtractor)
        assert "mfcc" in available_voiceprint_backends()

    def test_custom_extractor_interface(self):
        """Any ``BaseVoiceprintExtractor`` subclass plugs into the pipeline."""

        class ConstantExtractor(BaseVoiceprintExtractor):
            name = "constant"

            @property
            def dim(self) -> int:
                return 3

            def embed(self, audio, sample_rate=16000):
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)

        extractor = ConstantExtractor()
        assert extractor.dim == 3
        assert isinstance(extractor, BaseVoiceprintExtractor)
        assert float(np.linalg.norm(extractor.embed(synth_voice(1)))) == pytest.approx(1.0)


class TestAdaptiveCosineClusterer:
    """Duration-aware clustering with the Gaussian BIC merge test."""

    def test_dynamic_threshold_shrinks_with_duration(self):
        """Short segments are held to a stricter threshold."""
        clusterer = AdaptiveCosineClusterer(similarity_threshold=0.7, min_segment_s=1.0)
        assert clusterer.dynamic_threshold(0.1) > clusterer.dynamic_threshold(0.5)
        assert clusterer.dynamic_threshold(0.5) > clusterer.dynamic_threshold(5.0)
        assert clusterer.dynamic_threshold(10.0) == pytest.approx(0.7)
        assert clusterer.dynamic_threshold(0.0) <= 0.999

    def test_assign_identical_embeddings_single_cluster(self):
        """Identical embeddings always land in the same cluster."""
        clusterer = AdaptiveCosineClusterer(similarity_threshold=0.7)
        embedding = MFCCExtractor().embed(synth_voice(1))
        first, confidence = clusterer.assign(embedding, duration_s=2.0)
        second, second_confidence = clusterer.assign(embedding, duration_s=2.0)
        assert first == second
        assert confidence == pytest.approx(1.0, abs=1e-6)
        assert second_confidence >= 0.7
        assert clusterer.cluster_sizes() == {first: 2}

    def test_assign_opposite_embeddings_separate_clusters(self):
        """An anti-correlated embedding cannot be folded into the cluster."""
        clusterer = AdaptiveCosineClusterer(similarity_threshold=0.3, use_bic=False)
        embedding = MFCCExtractor().embed(synth_voice(2))
        first, _ = clusterer.assign(embedding, duration_s=2.0)
        second, _ = clusterer.assign(-embedding, duration_s=2.0)
        assert first != second

    def test_bic_score_merges_similar_distributions(self):
        """Overlapping samples are preferred merged (negative score)."""
        rng = np.random.default_rng(0)
        a = 0.01 * rng.standard_normal((20, 8))
        b = 0.01 * rng.standard_normal((20, 8))
        assert AdaptiveCosineClusterer.bic_score(a, b) < 0.0

    def test_bic_score_vetoes_distant_distributions(self):
        """Well separated samples are preferred split (positive score)."""
        rng = np.random.default_rng(1)
        a = 0.01 * rng.standard_normal((20, 8))
        b = 5.0 + 0.01 * rng.standard_normal((20, 8))
        assert AdaptiveCosineClusterer.bic_score(a, b) > 0.0

    def test_bic_score_degenerate_input(self):
        """Empty input reports the safe "do not merge" default."""
        assert AdaptiveCosineClusterer.bic_score(
            np.empty((0, 4)), np.zeros((1, 4))
        ) == pytest.approx(0.0)

    def test_reset_clears_clusters(self):
        """``reset`` forgets every cluster."""
        clusterer = AdaptiveCosineClusterer()
        clusterer.assign(MFCCExtractor().embed(synth_voice(1)), duration_s=2.0)
        clusterer.reset()
        assert clusterer.cluster_sizes() == {}


class TestSpeakerVerifierModernization:
    """Opt-in verification hardening, with legacy defaults preserved."""

    def test_legacy_defaults_unchanged(self):
        """Default construction keeps the pre-2.6.0 MFCC decision path."""
        verifier = SpeakerVerifier()
        assert verifier.extractor is None
        assert verifier.extractor_name == "builtin"
        assert verifier.dynamic_threshold(0.1) == pytest.approx(verifier.threshold)
        assert verifier.dynamic_threshold(30.0) == pytest.approx(verifier.threshold)

        voice = synth_voice(7)
        verifier.register("alice", voice)
        result = verifier.verify("alice", voice)
        assert result.verified
        assert result.threshold == pytest.approx(0.75)
        assert result.confidence > 0.9

    def test_adaptive_threshold_only_when_enabled(self):
        """Duration-aware thresholds apply to short probes when opted in."""
        verifier = SpeakerVerifier(
            threshold=0.75,
            adaptive_threshold=True,
            min_segment_s=1.0,
            short_penalty=0.2,
        )
        assert verifier.dynamic_threshold(10.0) == pytest.approx(0.75)
        assert verifier.dynamic_threshold(0.2) == pytest.approx(0.75 + 0.2 * 0.8)

        voice = synth_voice(8)
        verifier.register("alice", voice)
        short = verifier.verify("alice", voice[:3200])
        assert short.threshold == pytest.approx(verifier.dynamic_threshold(0.2))
        long = verifier.verify("alice", voice)
        assert long.threshold == pytest.approx(verifier.dynamic_threshold(0.4))

    def test_dynamic_pruning_drops_outlier_templates(self):
        """Pruning averages only the templates agreeing with the best match."""
        probe = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        outlier = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        pool = [probe, probe, outlier]

        pruned = SpeakerVerifier(dynamic_pruning=True)
        assert pruned._pool_similarity(pool, probe) == pytest.approx(1.0)

        plain = SpeakerVerifier()
        assert plain._pool_similarity(pool, probe) == pytest.approx(2.0 / 3.0)

    def test_dynamic_pruning_keeps_best_template(self):
        """A fully disagreeing pool still keeps the best match."""
        probe = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        pool = [
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ]
        pruned = SpeakerVerifier(dynamic_pruning=True)
        assert pruned._pool_similarity(pool, probe) == pytest.approx(0.0)

    def test_dynamic_pruning_registration_and_verification(self):
        """Opt-in pruning keeps the enrollment pool bounded and verifies."""
        verifier = SpeakerVerifier(dynamic_pruning=True, max_templates=2)
        voice = synth_voice(9)
        for _ in range(3):
            verifier.register("alice", voice)
        assert verifier.templates_count("alice") == 2
        result = verifier.verify("alice", voice)
        assert result.verified
        assert result.confidence == pytest.approx(1.0, abs=1e-4)

    def test_register_from_samples_builds_pool(self):
        """Multi-sample enrollment feeds the enrollment pool."""
        verifier = SpeakerVerifier(max_templates=4)
        voices = [synth_voice(3), synth_voice(3, 8000), synth_voice(3, 12000)]
        verifier.register_from_samples("bob", voices)
        assert verifier.templates_count("bob") == 3
        assert verifier.verify("bob", voices[0]).verified

    def test_mfcc_extractor_backend(self):
        """``extractor_backend="mfcc"`` scores with the shared extractor."""
        verifier = SpeakerVerifier(extractor_backend="mfcc")
        assert verifier.extractor_name == "mfcc"
        voiceprint = verifier.register("alice", synth_voice(11))
        assert voiceprint.mfcc_mean is not None  # persistence stays intact

        result = verifier.verify("alice", synth_voice(11))
        assert result.verified
        assert result.confidence == pytest.approx(1.0, abs=1e-4)
        assert verifier.verify_any(synth_voice(11)).speaker_id == "alice"

    def test_onnx_backend_falls_back_to_mfcc(self):
        """Unavailable ONNX models degrade to the MFCC extractor."""
        verifier = SpeakerVerifier(extractor_backend="onnx", model_path="missing.onnx")
        assert verifier.extractor_name == "mfcc"
        verifier.register("carol", synth_voice(12))
        assert verifier.verify("carol", synth_voice(12)).verified

    def test_injected_extractor(self):
        """An injected extractor object drives registration and verification."""

        class OnesExtractor:
            name = "ones"

            def embed(self, audio, sample_rate=16000):
                return np.ones(4, dtype=np.float32)

        verifier = SpeakerVerifier(extractor=OnesExtractor())
        assert verifier.extractor_name == "ones"
        verifier.register("dan", synth_voice(13))
        result = verifier.verify("dan", synth_voice(14))
        assert result.verified
        assert result.confidence == pytest.approx(1.0, abs=1e-4)

    def test_bic_veto_is_opt_in(self):
        """The BIC veto is inert by default and safe when enabled."""
        voice = synth_voice(15)
        legacy = SpeakerVerifier()
        legacy.register_from_samples("alice", [voice, voice, voice])
        assert legacy._bic_veto("alice", voice) is False

        hardened = SpeakerVerifier(use_bic=True)
        hardened.register_from_samples("alice", [voice, voice, voice])
        assert hardened._bic_veto("alice", voice) is False  # identical enrollment
        assert hardened.verify("alice", voice).verified

    def test_bic_veto_needs_enough_templates(self):
        """Fewer than three templates cannot trigger the covariance test."""
        verifier = SpeakerVerifier(use_bic=True)
        verifier.register("alice", synth_voice(16))
        assert verifier.templates_count("alice") == 1
        assert verifier._bic_veto("alice", synth_voice(99)) is False

    def test_reset_and_delete_clear_templates(self):
        """Template bookkeeping follows delete / reset."""
        verifier = SpeakerVerifier()
        verifier.register("alice", synth_voice(17))
        verifier.register("bob", synth_voice(18))
        assert verifier.templates_count() == 2

        assert verifier.delete("alice") is True
        assert verifier.templates_count("alice") == 0
        assert len(verifier.list_speakers()) == 1

        verifier.reset_templates()
        assert verifier.templates_count() == 0
        assert len(verifier.list_speakers()) == 1  # voiceprints untouched

    def test_empty_probe_is_rejected(self):
        """Degenerate probes never raise and are never verified."""
        verifier = SpeakerVerifier(adaptive_threshold=True)
        verifier.register("alice", synth_voice(19))
        result = verifier.verify("alice", np.array([], dtype=np.float32))
        assert result.verified is False
        assert np.isfinite(result.confidence)
