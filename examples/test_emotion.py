"""
Test Emotion Recognition Module
================================

Demonstrates emotion recognition on synthetic audio signals
with different characteristics.

Usage:
    python examples/test_emotion.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from vram_core.emotion_recognition import EmotionRecognizer


def generate_test_audio(
    duration_s: float = 2.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
    frequency: float = 200.0,
    noise_level: float = 0.0,
) -> np.ndarray:
    """Generate a synthetic audio signal."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = amplitude * np.sin(2 * np.pi * frequency * t)
    if noise_level > 0:
        audio += noise_level * np.random.randn(len(audio))
    return audio.astype(np.float32)


def test_basic():
    """Basic emotion recognition test."""
    print("=" * 60)
    print("Emotion Recognition Module - Test Suite")
    print("=" * 60)

    recognizer = EmotionRecognizer()

    # Test 1: Neutral (moderate amplitude, moderate frequency)
    print("\n--- Test 1: Neutral audio ---")
    audio = generate_test_audio(amplitude=0.1, frequency=200.0)
    result = recognizer.analyze(audio)
    print(f"  Detected: {result.emotion} (confidence: {result.confidence:.3f})")
    print(f"  Features: energy={result.features.rms_energy:.4f}, "
          f"ZCR={result.features.zero_crossing_rate:.4f}, "
          f"F0={result.features.mean_f0:.1f}Hz")
    print(f"  All scores: {result.all_scores}")
    assert result.emotion in ["happy", "sad", "angry", "neutral", "surprised"]
    print("  锟?Valid emotion label")

    # Test 2: Energetic / Angry (high amplitude, fast modulation)
    print("\n--- Test 2: High-energy audio (angry-like) ---")
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    audio = (0.8 * np.sin(2 * np.pi * 300 * t) +
             0.3 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
    result = recognizer.analyze(audio)
    print(f"  Detected: {result.emotion} (confidence: {result.confidence:.3f})")
    print(f"  Features: energy={result.features.rms_energy:.4f}, "
          f"F0_std={result.features.std_f0:.1f}")
    print(f"  All scores: {result.all_scores}")
    print("  锟?Completed without error")

    # Test 3: Sad (low amplitude, slow)
    print("\n--- Test 3: Low-energy audio (sad-like) ---")
    audio = generate_test_audio(amplitude=0.02, frequency=120.0)
    result = recognizer.analyze(audio)
    print(f"  Detected: {result.emotion} (confidence: {result.confidence:.3f})")
    print(f"  Features: energy={result.features.rms_energy:.4f}, "
          f"F0={result.features.mean_f0:.1f}Hz")
    print(f"  All scores: {result.all_scores}")
    print("  锟?Completed without error")

    # Test 4: Happy (medium-high energy, higher pitch)
    print("\n--- Test 4: Medium-high energy audio (happy-like) ---")
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    # Frequency modulation for "lively" feel
    audio = (0.4 * np.sin(2 * np.pi * 350 * t) +
             0.2 * np.sin(2 * np.pi * 500 * t * (1 + 0.3 * np.sin(2 * np.pi * 3 * t)))
             ).astype(np.float32)
    result = recognizer.analyze(audio)
    print(f"  Detected: {result.emotion} (confidence: {result.confidence:.3f})")
    print(f"  All scores: {result.all_scores}")
    print("  锟?Completed without error")

    # Test 5: Empty audio
    print("\n--- Test 5: Empty audio ---")
    result = recognizer.analyze(np.array([], dtype=np.float32))
    print(f"  Detected: {result.emotion} (confidence: {result.confidence:.3f})")
    print("  锟?Handled empty audio gracefully")

    # Test 6: Very short audio
    print("\n--- Test 6: Very short audio (100 samples) ---")
    audio = np.random.randn(100).astype(np.float32) * 0.1
    result = recognizer.analyze(audio)
    print(f"  Detected: {result.emotion} (confidence: {result.confidence:.3f})")
    print("  锟?Handled short audio gracefully")

    print("\n" + "=" * 60)
    print("All tests passed! 锟?)
    print("=" * 60)


def test_feature_extraction():
    """Test feature extraction independently."""
    print("\n--- Feature Extraction Detail ---")
    recognizer = EmotionRecognizer()

    audio = generate_test_audio(duration_s=3.0, amplitude=0.3, frequency=250.0)
    features = recognizer.extract_features(audio)

    print(f"  RMS Energy:        {features.rms_energy:.6f}")
    print(f"  Zero-Crossing Rate: {features.zero_crossing_rate:.6f}")
    print(f"  Mean F0:           {features.mean_f0:.2f} Hz")
    print(f"  F0 Std Dev:        {features.std_f0:.2f} Hz")
    print(f"  Energy Variance:   {features.energy_variance:.8f}")
    print(f"  Energy Range:      {features.energy_range:.6f}")
    print(f"  Speech Rate Proxy: {features.speech_rate_proxy:.4f}")
    print("  锟?Feature extraction complete")


if __name__ == "__main__":
    test_basic()
    test_feature_extraction()