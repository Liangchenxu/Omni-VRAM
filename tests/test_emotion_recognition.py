"""
Tests for Emotion Recognition Module
======================================
"""

import numpy as np
import pytest

from vram_core.emotion_recognition import (
    EmotionRecognizer,
    AudioFeatures,
    EmotionResult,
)


def make_sine(duration_s=2.0, sr=16000, freq=200.0, amp=0.3):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestEmotionRecognizerInit:
    def test_default(self):
        r = EmotionRecognizer()
        assert r.frame_size_ms == 25
        assert r.min_f0 == 50.0
        assert r.max_f0 == 500.0

    def test_custom_params(self):
        r = EmotionRecognizer(frame_size_ms=50, min_f0=80, max_f0=400)
        assert r.frame_size_ms == 50


class TestFeatureExtraction:
    def test_basic_features(self):
        r = EmotionRecognizer()
        audio = make_sine()
        f = r.extract_features(audio)
        assert isinstance(f, AudioFeatures)
        assert f.rms_energy > 0
        assert f.zero_crossing_rate > 0
        assert f.mean_f0 > 0

    def test_empty_audio(self):
        r = EmotionRecognizer()
        f = r.extract_features(np.array([], dtype=np.float32))
        assert f.rms_energy == 0.0
        assert f.mean_f0 == 0.0

    def test_high_energy_audio(self):
        r = EmotionRecognizer()
        audio = make_sine(amp=0.9)
        f = r.extract_features(audio)
        assert f.rms_energy > 0.3

    def test_low_energy_audio(self):
        r = EmotionRecognizer()
        audio = make_sine(amp=0.01)
        f = r.extract_features(audio)
        assert f.rms_energy < 0.05


class TestClassification:
    def test_valid_emotion_labels(self):
        r = EmotionRecognizer()
        valid = {"happy", "sad", "angry", "neutral", "surprised"}
        for amp in [0.01, 0.1, 0.3, 0.5, 0.8]:
            audio = make_sine(amp=amp)
            result = r.analyze(audio)
            assert result.emotion in valid
            assert 0 <= result.confidence <= 1

    def test_all_scores_sum_to_one(self):
        r = EmotionRecognizer()
        audio = make_sine()
        result = r.analyze(audio)
        total = sum(result.all_scores.values())
        assert abs(total - 1.0) < 0.01

    def test_result_has_features(self):
        r = EmotionRecognizer()
        audio = make_sine()
        result = r.analyze(audio)
        assert isinstance(result.features, AudioFeatures)

    def test_repr(self):
        result = EmotionResult(
            emotion="happy", confidence=0.45,
            features=AudioFeatures(), all_scores={},
        )
        assert "happy" in repr(result)
        assert "0.45" in repr(result)


class TestAnalyze:
    def test_end_to_end(self):
        r = EmotionRecognizer()
        audio = make_sine(duration_s=3.0, amp=0.5, freq=300)
        result = r.analyze(audio, sample_rate=16000)
        assert result.emotion in ["happy", "sad", "angry", "neutral", "surprised"]

    def test_short_audio(self):
        r = EmotionRecognizer()
        audio = np.random.randn(200).astype(np.float32) * 0.05
        result = r.analyze(audio)
        assert result.emotion is not None

    def test_int16_input(self):
        r = EmotionRecognizer()
        audio = (make_sine() * 32767).astype(np.int16)
        result = r.analyze(audio)
        assert result.emotion is not None