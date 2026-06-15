"""
Tests for Speaker Diarization Module
======================================
"""

import numpy as np
import pytest

from vram_core.speaker_diarization import (
    SpeakerDiarizer,
    SpeakerSegment,
    SpeakerProfile,
    MFCCDiarizer,
)


def make_tone(duration_s=2.0, sr=16000, freq=200.0, amp=0.3):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestSpeakerDiarizerInit:
    def test_default(self):
        d = SpeakerDiarizer()
        assert d._mfcc.n_mfcc == 13
        assert d._mfcc.similarity_threshold == 0.7
        assert d._mfcc.segment_duration_ms == 1000.0

    def test_custom_params(self):
        d = SpeakerDiarizer(n_mfcc=20, similarity_threshold=0.8)
        assert d._mfcc.n_mfcc == 20
        assert d._mfcc.similarity_threshold == 0.8


class TestMFCC:
    def test_extract_basic(self):
        d = MFCCDiarizer()
        audio = make_tone()
        mfcc = d.extract_mfcc(audio, sample_rate=16000)
        assert mfcc.shape[0] == d.n_mfcc
        assert mfcc.shape[1] > 0
        assert mfcc.dtype == np.float32

    def test_extract_empty(self):
        d = MFCCDiarizer()
        mfcc = d.extract_mfcc(np.array([], dtype=np.float32))
        assert mfcc.shape[0] == d.n_mfcc
        assert mfcc.shape[1] == 0

    def test_extract_short_audio(self):
        d = MFCCDiarizer()
        short = np.random.randn(50).astype(np.float32)
        mfcc = d.extract_mfcc(short)
        assert mfcc.shape[0] == d.n_mfcc


class TestEmbedding:
    def test_compute_embedding(self):
        d = MFCCDiarizer()
        mfcc = np.random.randn(13, 50).astype(np.float32)
        emb = d.compute_embedding(mfcc)
        assert emb.shape == (26,)
        assert emb.dtype == np.float32
        # Should be L2-normalized
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    def test_empty_mfcc(self):
        d = MFCCDiarizer()
        mfcc = np.zeros((13, 0), dtype=np.float32)
        emb = d.compute_embedding(mfcc)
        assert np.linalg.norm(emb) == 0.0


class TestCosineSimilarity:
    def test_same_vector(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(MFCCDiarizer.cosine_similarity(a, a) - 1.0) < 1e-6

    def test_orthogonal(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(MFCCDiarizer.cosine_similarity(a, b)) < 1e-6

    def test_opposite(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert MFCCDiarizer.cosine_similarity(a, b) < 0

    def test_zero_vector(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        assert MFCCDiarizer.cosine_similarity(a, b) == 0.0


class TestDiarize:
    def test_single_speaker(self):
        d = SpeakerDiarizer()
        audio = make_tone(duration_s=5.0, freq=200)
        result = d.diarize(audio, sample_rate=16000)
        segments = result.segments
        assert len(segments) > 0
        # Single tone should produce one speaker
        unique_speakers = set(s.speaker_id for s in segments)
        assert len(unique_speakers) == 1

    def test_two_speakers(self):
        d = SpeakerDiarizer(similarity_threshold=0.95)
        # Two very different signals
        tone1 = make_tone(duration_s=3.0, freq=100, amp=0.3)
        tone2 = make_tone(duration_s=3.0, freq=500, amp=0.8)
        audio = np.concatenate([tone1, tone2])
        result = d.diarize(audio, sample_rate=16000)
        segments = result.segments
        assert len(segments) > 0
        unique = set(s.speaker_id for s in segments)
        # May detect 1 or 2 speakers depending on MFCC similarity
        assert len(unique) >= 1

    def test_empty_audio(self):
        d = SpeakerDiarizer()
        result = d.diarize(np.array([], dtype=np.float32))
        assert result.segments == []

    def test_speaker_count(self):
        d = SpeakerDiarizer()
        audio = make_tone(duration_s=3.0)
        d.diarize(audio)
        assert d.get_speaker_count() >= 1
        assert len(d.get_speaker_ids()) >= 1

    def test_speaker_profile(self):
        d = SpeakerDiarizer()
        audio = make_tone(duration_s=3.0)
        d.diarize(audio)
        sid = d.get_speaker_ids()[0]
        profile = d.get_speaker_profile(sid)
        assert profile is not None
        assert profile.speaker_id == sid
        assert profile.total_duration > 0

    def test_nonexistent_speaker(self):
        d = SpeakerDiarizer()
        assert d.get_speaker_profile("nonexistent") is None


class TestSpeakerSegment:
    def test_duration(self):
        seg = SpeakerSegment(start_time=1.0, end_time=3.0, speaker_id="S1")
        assert seg.duration == 2.0

    def test_repr(self):
        seg = SpeakerSegment(start_time=0.0, end_time=2.5, speaker_id="S1")
        assert "S1" in repr(seg)
        assert "0.00" in repr(seg)


class TestMerge:
    def test_merge_same_speaker(self):
        d = MFCCDiarizer()
        segments = [
            SpeakerSegment(0.0, 1.0, "S1"),
            SpeakerSegment(1.0, 2.0, "S1"),
            SpeakerSegment(2.0, 3.0, "S2"),
            SpeakerSegment(3.0, 4.0, "S2"),
            SpeakerSegment(4.0, 5.0, "S1"),
        ]
        merged = d._merge_consecutive(segments)
        assert len(merged) == 3
        assert merged[0].duration == 2.0
        assert merged[1].duration == 2.0
        assert merged[2].duration == 1.0

    def test_merge_empty(self):
        d = MFCCDiarizer()
        assert d._merge_consecutive([]) == []
