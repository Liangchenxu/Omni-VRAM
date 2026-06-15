"""
Shared Utility Functions for vram_core
=======================================

Common audio processing functions used across multiple modules.
Extracted to eliminate code duplication.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


# 鈹€鈹€ Audio Preprocessing 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def ensure_float32(audio: np.ndarray) -> np.ndarray:
    """Convert audio to float32 dtype if not already."""
    if audio.dtype != np.float32:
        return audio.astype(np.float32)
    return audio


# 鈹€鈹€ Acoustic Feature Extraction 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def compute_rms_energy(audio: np.ndarray) -> float:
    """Compute root-mean-square energy of audio signal."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def compute_zero_crossing_rate(audio: np.ndarray) -> float:
    """Compute zero-crossing rate of audio signal."""
    if len(audio) < 2:
        return 0.0
    crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
    return float(crossings / (len(audio) - 1))


def compute_rms_energy_per_frame(
    audio: np.ndarray,
    frame_size: int,
) -> np.ndarray:
    """Compute RMS energy for each frame of the audio signal."""
    n_frames = max(1, len(audio) // frame_size)
    energies = []
    for i in range(n_frames):
        start = i * frame_size
        end = min(start + frame_size, len(audio))
        frame = audio[start:end]
        if len(frame) > 0:
            energies.append(float(np.sqrt(np.mean(frame ** 2))))
    return np.array(energies) if energies else np.array([0.0])


def compute_zcr_per_frame(
    audio: np.ndarray,
    frame_size: int,
) -> np.ndarray:
    """Compute zero-crossing rate for each frame of the audio signal."""
    n_frames = max(1, len(audio) // frame_size)
    zcr_values = []
    for i in range(n_frames):
        start = i * frame_size
        end = min(start + frame_size, len(audio))
        frame = audio[start:end]
        if len(frame) >= 2:
            crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
            zcr_values.append(crossings / (len(frame) - 1))
    return np.array(zcr_values) if zcr_values else np.array([0.0])


# 鈹€鈹€ Event Merging 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def merge_adjacent_events(
    events: list,
    label_key: str = "label",
    start_key: str = "start_time",
    end_key: str = "end_time",
    confidence_key: str = "confidence",
) -> list:
    """
    Merge consecutive events with the same label.

    Works with both dataclass objects and dicts.
    For dataclass objects, uses getattr; for dicts, uses dict access.
    """
    if not events:
        return []

    def _get(obj, key):
        if hasattr(obj, key):
            return getattr(obj, key)
        return obj[key]

    def _set(obj, key, value):
        if hasattr(obj, "__dict__"):
            setattr(obj, key, value)
        else:
            obj[key] = value

    merged = [events[0]]
    for evt in events[1:]:
        if _get(evt, label_key) == _get(merged[-1], label_key):
            # Extend the previous event's end time and take max confidence
            _set(merged[-1], end_key, _get(evt, end_key))
            _set(
                merged[-1],
                confidence_key,
                max(_get(merged[-1], confidence_key), _get(evt, confidence_key)),
            )
        else:
            merged.append(evt)
    return merged


# 鈹€鈹€ Audio Resampling 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def simple_resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Simple linear interpolation resampling.

    For production quality, use librosa.resample or sox.
    This is a lightweight fallback for basic needs.
    """
    if orig_sr == target_sr:
        return audio

    ratio = target_sr / orig_sr
    new_len = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)