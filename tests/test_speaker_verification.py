"""
Tests for vram_core.speaker_verification module.

Covers:
    - Voiceprint data class (creation, serialization, deserialization)
    - VerificationResult data class
    - SpeakerVerifier (register, verify, verify_any, delete, list, persistence)
    - Edge cases: empty audio, zero vectors, threshold bounds
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from vram_core.speaker_verification import (
    Voiceprint,
    VerificationResult,
    SpeakerVerifier,
)


class TestVoiceprint:
    """Test Voiceprint data class."""

    def test_create_default(self):
        """Default voiceprint has expected empty values."""
        vp = Voiceprint()
        assert vp.speaker_id == ""
        assert vp.mfcc_mean is None
        assert vp.mfcc_std is None
        assert vp.num_samples == 0

    def test_to_dict_with_arrays(self):
        """Serialization converts numpy arrays to lists."""
        vp = Voiceprint(
            speaker_id="alice",
            mfcc_mean=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            mfcc_std=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            num_samples=5,
            created_at=1000.0,
            updated_at=2000.0,
            metadata={"lang": "en"},
        )
        d = vp.to_dict()
        assert d["speaker_id"] == "alice"
        assert d["num_samples"] == 5
        assert isinstance(d["mfcc_mean"], list)
        assert d["mfcc_mean"][0] == pytest.approx(1.0)

    def test_to_dict_none_arrays(self):
        """Serialization handles None arrays gracefully."""
        vp = Voiceprint(speaker_id="bob")
        d = vp.to_dict()
        assert d["mfcc_mean"] is None
        assert d["mfcc_std"] is None

    def test_from_dict_roundtrip(self):
        """Deserialization restores original data."""
        original = Voiceprint(
            speaker_id="carol",
            mfcc_mean=np.array([0.5, 0.6], dtype=np.float32),
            mfcc_std=np.array([0.01, 0.02], dtype=np.float32),
            num_samples=3,
            created_at=100.0,
            updated_at=200.0,
            metadata={"key": "value"},
        )
        restored = Voiceprint.from_dict(original.to_dict())
        assert restored.speaker_id == "carol"
        assert restored.num_samples == 3
        np.testing.assert_array_almost_equal(restored.mfcc_mean, original.mfcc_mean)
        assert restored.metadata["key"] == "value"

    def test_from_dict_missing_fields(self):
        """Deserialization handles missing optional fields."""
        data = {"speaker_id": "minimal"}
        vp = Voiceprint.from_dict(data)
        assert vp.speaker_id == "minimal"
        assert vp.mfcc_mean is None
        assert vp.num_samples == 0


class TestVerificationResult:
    """Test VerificationResult data class."""

    def test_default_values(self):
        """Default result is rejected with 0 confidence."""
        r = VerificationResult()
        assert not r.verified
        assert r.confidence == 0.0

    def test_repr_verified(self):
        """String representation shows verification status."""
        r = VerificationResult(speaker_id="alice", verified=True, confidence=0.92, threshold=0.75)
        s = repr(r)
        assert "alice" in s
        assert "VERIFIED" in s

    def test_repr_rejected(self):
        r = VerificationResult(speaker_id="bob", verified=False, confidence=0.3, threshold=0.75)
        s = repr(r)
        assert "REJECTED" in s


class TestSpeakerVerifier:
    """Test SpeakerVerifier engine."""

    def _make_audio(self, duration_sec=1.0, sr=16000):
        """Generate synthetic sine-wave audio."""
        t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
        return np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def test_init_default(self):
        """Default initialization has no voiceprints."""
        verifier = SpeakerVerifier()
        assert verifier.threshold == 0.75
        assert len(verifier._voiceprints) == 0

    def test_init_custom_threshold(self):
        verifier = SpeakerVerifier(threshold=0.9)
        assert verifier.threshold == 0.9

    def test_register_new_speaker(self):
        """Registering a new speaker creates a voiceprint."""
        verifier = SpeakerVerifier()
        audio = self._make_audio()
        vp = verifier.register("alice", audio)
        assert vp.speaker_id == "alice"
        assert vp.num_samples == 1
        assert "alice" in verifier._voiceprints

    def test_register_update_existing(self):
        """Registering the same speaker updates the voiceprint (EMA)."""
        verifier = SpeakerVerifier()
        audio1 = self._make_audio(duration_sec=1.0)
        audio2 = self._make_audio(duration_sec=2.0)
        verifier.register("alice", audio1)
        vp2 = verifier.register("alice", audio2)
        assert vp2.num_samples == 2

    def test_register_with_metadata(self):
        """Metadata is stored with voiceprint."""
        verifier = SpeakerVerifier()
        audio = self._make_audio()
        vp = verifier.register("alice", audio, metadata={"lang": "zh"})
        assert vp.metadata["lang"] == "zh"

    def test_register_from_samples(self):
        """Multi-sample registration combines MFCC features."""
        verifier = SpeakerVerifier()
        samples = [self._make_audio(0.5) for _ in range(3)]
        vp = verifier.register_from_samples("bob", samples)
        assert vp.num_samples == 3
        assert vp.mfcc_mean is not None

    def test_verify_registered_speaker(self):
        """Verification against registered speaker returns a result."""
        verifier = SpeakerVerifier(threshold=0.0)  # very low threshold
        audio = self._make_audio()
        verifier.register("alice", audio)
        result = verifier.verify("alice", audio)
        assert isinstance(result, VerificationResult)
        assert result.speaker_id == "alice"
        assert result.confidence > 0.0

    def test_verify_unregistered_raises(self):
        """Verifying unregistered speaker raises KeyError."""
        verifier = SpeakerVerifier()
        audio = self._make_audio()
        with pytest.raises(KeyError):
            verifier.verify("ghost", audio)

    def test_verify_custom_threshold(self):
        """Per-call threshold override works."""
        verifier = SpeakerVerifier(threshold=0.5)
        audio = self._make_audio()
        verifier.register("alice", audio)
        result = verifier.verify("alice", audio, threshold=0.99)
        # Same audio vs itself should still pass even at 0.99
        assert result.confidence > 0.9

    def test_verify_any_returns_best_match(self):
        """verify_any returns the best matching speaker."""
        verifier = SpeakerVerifier(threshold=0.0)
        audio_a = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        audio_b = np.sin(2 * np.pi * 880 * np.linspace(0, 1, 16000)).astype(np.float32)
        verifier.register("alice", audio_a)
        verifier.register("bob", audio_b)
        result = verifier.verify_any(audio_a)
        assert result is not None
        assert result.speaker_id == "alice"

    def test_verify_any_no_match(self):
        """verify_any returns None when no speaker matches."""
        verifier = SpeakerVerifier(threshold=0.999)
        audio = self._make_audio()
        verifier.register("alice", audio)
        # Completely different audio (silence)
        silence = np.zeros(16000, dtype=np.float32)
        result = verifier.verify_any(silence)
        # Could be None or a rejected result depending on impl
        if result is not None:
            assert not result.verified

    def test_delete_existing(self):
        """Deleting an existing speaker returns True."""
        verifier = SpeakerVerifier()
        verifier.register("alice", self._make_audio())
        assert verifier.delete("alice")
        assert "alice" not in verifier._voiceprints

    def test_delete_nonexistent(self):
        """Deleting a non-existent speaker returns False."""
        verifier = SpeakerVerifier()
        assert not verifier.delete("ghost")

    def test_list_speakers(self):
        """list_speakers returns registered speakers."""
        verifier = SpeakerVerifier()
        verifier.register("alice", self._make_audio())
        verifier.register("bob", self._make_audio())
        speakers = verifier.list_speakers()
        assert len(speakers) == 2
        ids = [s["speaker_id"] for s in speakers]
        assert "alice" in ids
        assert "bob" in ids

    def test_get_speaker(self):
        """get_speaker returns voiceprint for existing speaker."""
        verifier = SpeakerVerifier()
        verifier.register("alice", self._make_audio())
        vp = verifier.get_speaker("alice")
        assert vp is not None
        assert vp.speaker_id == "alice"

    def test_get_speaker_nonexistent(self):
        """get_speaker returns None for unknown speaker."""
        verifier = SpeakerVerifier()
        assert verifier.get_speaker("ghost") is None

    def test_set_threshold_valid(self):
        """set_threshold updates threshold within bounds."""
        verifier = SpeakerVerifier()
        verifier.set_threshold(0.85)
        assert verifier.threshold == 0.85

    def test_set_threshold_out_of_bounds(self):
        """set_threshold raises ValueError for out-of-bounds."""
        verifier = SpeakerVerifier()
        with pytest.raises(ValueError):
            verifier.set_threshold(-0.1)
        with pytest.raises(ValueError):
            verifier.set_threshold(1.5)

    def test_cosine_similarity_identical(self):
        """Cosine similarity of identical vectors is 1.0."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # Identical vectors: cosine similarity should be 1.0
        assert round(abs(SpeakerVerifier._cosine_similarity(a, a) - 1.0), 5) == 0

    def test_cosine_similarity_orthogonal(self):
        """Cosine similarity of orthogonal vectors is 0.0."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert SpeakerVerifier._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        """Cosine similarity with zero vector returns 0.0."""
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        assert SpeakerVerifier._cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_persistence_save_load(self):
        """Voiceprints persist across save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "voiceprints.json"
            verifier = SpeakerVerifier(storage_path=path)
            verifier.register("alice", self._make_audio())

            # Reload from file
            verifier2 = SpeakerVerifier(storage_path=path)
            assert "alice" in verifier2._voiceprints
            assert verifier2._voiceprints["alice"].num_samples == 1

    def test_persistence_corrupt_file(self):
        """Corrupt storage file is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "voiceprints.json"
            path.write_text("NOT VALID JSON", encoding="utf-8")
            verifier = SpeakerVerifier(storage_path=path)
            # Should not raise, just log error
            assert len(verifier._voiceprints) == 0

    def test_int16_audio_input(self):
        """Integer audio is converted to float32 internally."""
        verifier = SpeakerVerifier()
        audio_int16 = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
        # Should not raise
        vp = verifier.register("alice", audio_int16)
        assert vp.mfcc_mean is not None


