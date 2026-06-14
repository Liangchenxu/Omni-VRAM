"""
WhisperResult Dataclass
========================

Typed return value for all Whisper transcription backends.
Includes SRT subtitle export.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("omni-vram.whisper.result")


@dataclass
class WhisperResult:
    """
    Typed return value for all Whisper backends.

    Attributes:
        text:       Full transcription text.
        language:   ISO 639-1 detected language code.
        confidence: Average segment confidence score.
        segments:   Per-segment details [{start, end, text, confidence}].
        backend:    Which backend was used.
        model:      Model name used.
        raw:        Backend-specific raw output.
    """

    text: str
    language: str = "unknown"
    confidence: float = 0.0
    segments: List[Dict[str, Any]] = field(default_factory=list)
    backend: Optional[str] = None
    model: Optional[str] = None
    raw: Any = None

    def to_srt(self) -> str:
        """
        Export segments as SRT subtitle format.

        Returns:
            SRT formatted string.
        """
        lines: List[str] = []
        for i, seg in enumerate(self.segments, 1):
            start = self._format_srt_time(seg.get("start", 0))
            end = self._format_srt_time(seg.get("end", 0))
            text = seg.get("text", "")
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"