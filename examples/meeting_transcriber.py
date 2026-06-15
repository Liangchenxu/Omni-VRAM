#!/usr/bin/env python3
"""
vram_core: Meeting Transcriber
================================

Real-time meeting transcription with automatic speaker segmentation,
live display, and export to SRT + plain text transcript.

Pipeline:
    Microphone 锟?VAD 锟?Whisper Transcription 锟?Speaker Segmentation 锟?Export

Features:
    - Real-time transcription with live display
    - Simple speaker diarization based on silence intervals
    - Automatic paragraph grouping by speaker turns
    - Export SRT subtitle file with speaker labels
    - Export plain text transcript with timestamps

Usage:
    # Basic usage (Chinese, 60 seconds)
    python examples/meeting_transcriber.py

    # English meeting, 5 minutes
    python examples/meeting_transcriber.py --duration 300 --language en

    # Custom output prefix and model
    python examples/meeting_transcriber.py --output-prefix weekly_standup --model medium

    # Verbose debug mode
    python examples/meeting_transcriber.py --verbose

Output files:
    {prefix}.srt  锟?SRT subtitle file with speaker labels and timestamps
    {prefix}.txt  锟?Plain text transcript with timestamps and speaker labels

Requirements:
    pip install pyaudio numpy pydub python-dotenv

    PyAudio installation (Windows):
        pip install pipwin && pipwin install pyaudio
    Or download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

    Whisper.cpp must be installed for local transcription.
    See README.md for setup instructions.

Speaker Diarization:
    This script uses a simple silence-based approach:
    - Speech segments separated by >2s silence are assigned different speakers
    - Within a continuous speech block, the same speaker is assumed
    - For production use, consider integrating pyannote-audio or similar
"""

import argparse
import sys
import time
import signal
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from vram_core.whisper_bridge import WhisperBridge, WhisperBackend, WhisperResult
from vram_core.stream_processor import StreamProcessor, StreamConfig, StreamState
from vram_core.config import setup_logging


# 鈹€鈹€ Data Structures 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

@dataclass
class Utterance:
    """A single utterance segment with speaker label and timestamp."""
    speaker: str
    text: str
    start_time: float  # seconds from meeting start
    end_time: float     # seconds from meeting start
    confidence: float = 0.0
    segments: List[dict] = field(default_factory=list)


@dataclass
class MeetingRecord:
    """Complete meeting record with all utterances."""
    utterances: List[Utterance] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    language: str = ""

    @property
    def duration(self) -> float:
        """Total meeting duration in seconds."""
        if self.utterances:
            return self.utterances[-1].end_time
        return 0.0

    @property
    def speaker_count(self) -> int:
        """Number of unique speakers."""
        return len(set(u.speaker for u in self.utterances))

    def add_utterance(self, utterance: Utterance):
        """Add an utterance to the record."""
        self.utterances.append(utterance)

    def export_srt(self, filepath: str) -> None:
        """Export meeting as SRT subtitle file with speaker labels."""
        with open(filepath, "w", encoding="utf-8") as f:
            for i, utt in enumerate(self.utterances, 1):
                start = self._format_srt_time(utt.start_time)
                end = self._format_srt_time(utt.end_time)
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"[{utt.speaker}] {utt.text}\n")
                f.write("\n")

    def export_txt(self, filepath: str) -> None:
        """Export meeting as plain text transcript with timestamps."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Meeting Transcript\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"Duration: {self._format_duration(self.duration)}\n")
            f.write(f"Speakers: {self.speaker_count}\n")
            f.write(f"Language: {self.language}\n")
            f.write(f"{'=' * 60}\n\n")

            current_speaker = None
            for utt in self.utterances:
                # Add speaker header when speaker changes
                if utt.speaker != current_speaker:
                    if current_speaker is not None:
                        f.write("\n")
                    f.write(f"[{utt.speaker}]\n")
                    current_speaker = utt.speaker

                timestamp = self._format_timestamp(utt.start_time)
                f.write(f"  {timestamp} {utt.text}\n")

            f.write(f"\n{'=' * 60}\n")
            f.write(f"End of Transcript\n")

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = td.total_seconds() % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds to [MM:SS] timestamp."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"[{minutes:02d}:{secs:02d}]"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds to human-readable duration."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"


# 鈹€鈹€ Speaker Diarization 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class SimpleSpeakerDiarizer:
    """
    Simple speaker diarization based on silence intervals.

    Assigns different speaker labels when silence between speech
    segments exceeds a threshold. This is a heuristic approach
    suitable for simple meeting scenarios.
    """

    def __init__(
        self,
        silence_threshold_s: float = 2.0,
        max_speakers: int = 10,
        speaker_prefix: str = "Speaker",
    ):
        self.silence_threshold_s = silence_threshold_s
        self.max_speakers = max_speakers
        self.speaker_prefix = speaker_prefix
        self._current_speaker_idx = 0
        self._last_speech_end: Optional[float] = None
        self._speaker_map: dict = {}

    def reset(self):
        """Reset diarizer state."""
        self._current_speaker_idx = 0
        self._last_speech_end = None

    def assign_speaker(self, speech_start: float, speech_end: float) -> str:
        """
        Assign a speaker label based on timing context.

        Args:
            speech_start: Start time of the speech segment (seconds from meeting start).
            speech_end: End time of the speech segment.

        Returns:
            Speaker label string (e.g., "Speaker 1").
        """
        # First utterance
        if self._last_speech_end is None:
            speaker = self._get_speaker_label(0)
            self._last_speech_end = speech_end
            return speaker

        # Check silence gap
        silence_duration = speech_start - self._last_speech_end

        if silence_duration > self.silence_threshold_s:
            # Long silence -> potentially different speaker
            # Cycle to next speaker
            self._current_speaker_idx = (
                (self._current_speaker_idx + 1) % self.max_speakers
            )

        self._last_speech_end = speech_end
        return self._get_speaker_label(self._current_speaker_idx)

    def _get_speaker_label(self, idx: int) -> str:
        """Get or create speaker label for index."""
        if idx not in self._speaker_map:
            self._speaker_map[idx] = f"{self.speaker_prefix} {idx + 1}"
        return self._speaker_map[idx]


# 鈹€鈹€ Command Line Arguments 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="vram_core: Meeting Transcriber",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/meeting_transcriber.py\n"
            "  python examples/meeting_transcriber.py --duration 300 --language en\n"
            "  python examples/meeting_transcriber.py --output-prefix weekly_standup\n"
            "  python examples/meeting_transcriber.py --model medium --verbose\n"
        ),
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Recording duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="Language code for transcription (default: zh). Common: zh, en, ja, ko",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="meeting",
        help="Output file prefix (default: meeting). Produces {prefix}.srt and {prefix}.txt",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="PyAudio device index (default: system default). Use --list-devices to see available.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        default=False,
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.02,
        help="VAD energy threshold (default: 0.02). Lower = more sensitive.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=2.0,
        help="Silence duration in seconds to trigger speaker change (default: 2.0)",
    )
    parser.add_argument(
        "--device-cpu",
        action="store_true",
        default=False,
        help="Use CPU for whisper instead of CUDA",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose/debug logging",
    )

    return parser.parse_args()


# 鈹€鈹€ Audio Device Helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def list_audio_devices():
    """List all available audio input devices."""
    try:
        import pyaudio
    except ImportError:
        print("Error: pyaudio is not installed.")
        print("Install with: pip install pyaudio")
        sys.exit(1)

    pa = pyaudio.PyAudio()
    print("\n" + "=" * 60)
    print("  Available Audio Input Devices")
    print("=" * 60)

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            default = " (DEFAULT)" if i == pa.get_default_input_device_info()["index"] else ""
            print(f"  [{i:2d}] {info['name']}{default}")
            print(f"       Channels: {info['maxInputChannels']}, "
                  f"Sample Rate: {int(info['defaultSampleRate'])}Hz")

    print("=" * 60)
    pa.terminate()


# 鈹€鈹€ Main 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def main():
    """Main entry point for meeting transcriber."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger("meeting_transcriber")

    # List devices mode
    if args.list_devices:
        list_audio_devices()
        return

    # Check pyaudio
    try:
        import pyaudio
    except ImportError:
        print("=" * 60)
        print("  Error: pyaudio is not installed!")
        print("=" * 60)
        print()
        print("  Install pyaudio:")
        print("    pip install pyaudio")
        print()
        print("  If that fails on Windows:")
        print("    pip install pipwin && pipwin install pyaudio")
        print("  Or download from:")
        print("    https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
        print()
        sys.exit(1)

    # Output file paths
    srt_path = f"{args.output_prefix}.srt"
    txt_path = f"{args.output_prefix}.txt"

    # Print header
    print()
    print("锟? + "锟? * 58 + "锟?)
    print("锟? + "  vram_core: Meeting Transcriber".center(58) + "锟?)
    print("锟? + "锟? * 58 + "锟?)
    print()
    print(f"  Duration:          {args.duration}s")
    print(f"  Language:          {args.language}")
    print(f"  Model:             {args.model}")
    print(f"  Device:            {'CPU' if args.device_cpu else 'CUDA'}")
    print(f"  VAD Threshold:     {args.vad_threshold}")
    print(f"  Silence Threshold: {args.silence_threshold}s")
    print(f"  Output SRT:        {srt_path}")
    print(f"  Output TXT:        {txt_path}")
    print()

    # 鈹€鈹€ Step 1: Initialize Whisper 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("  [1/4] Initializing Whisper...")
    try:
        whisper = WhisperBridge(
            backend=WhisperBackend.AUTO,
            whisper_model=args.model,
            language=args.language,
            device="cpu" if args.device_cpu else "cuda",
        )
        status = whisper.get_status()
        print(f"  锟?Whisper ready (backend: {status['backend']})")
    except Exception as e:
        print(f"  锟?Failed to initialize Whisper: {e}")
        print("  See README.md for whisper.cpp setup instructions.")
        sys.exit(1)

    # 鈹€鈹€ Step 2: Initialize Stream Processor 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("  [2/4] Initializing stream processor...")
    stream_config = StreamConfig(
        sample_rate=16000,
        chunk_duration_ms=100,
        vad_threshold=args.vad_threshold,
    )
    processor = StreamProcessor(
        config=stream_config,
        whisper_bridge=whisper,
    )
    print(f"  锟?Stream processor ready")

    # 鈹€鈹€ Step 3: Initialize Speaker Diarizer 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("  [3/4] Initializing speaker diarizer...")
    diarizer = SimpleSpeakerDiarizer(
        silence_threshold_s=args.silence_threshold,
        max_speakers=10,
        speaker_prefix="Speaker",
    )
    meeting = MeetingRecord(language=args.language)
    print(f"  锟?Speaker diarizer ready (silence gap: {args.silence_threshold}s)")

    # 鈹€鈹€ Step 4: Initialize Microphone 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("  [4/4] Initializing microphone...")
    pa = pyaudio.PyAudio()

    try:
        mic_stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=stream_config.sample_rate,
            input=True,
            input_device_index=args.device,
            frames_per_buffer=stream_config.chunk_size,
        )
        device_name = "default"
        if args.device is not None:
            device_name = pa.get_device_info_by_index(args.device)["name"]
        print(f"  锟?Microphone ready ({device_name})")
    except Exception as e:
        print(f"  锟?Failed to open microphone: {e}")
        pa.terminate()
        sys.exit(1)

    # 鈹€鈹€ Meeting Start Time 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    meeting_start = time.time()
    meeting.start_time = meeting_start
    utterance_count = 0

    # 鈹€鈹€ Callbacks 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def on_speech_start():
        """Called when VAD detects speech start."""
        elapsed = time.time() - meeting_start
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        print(f"\r  馃帳 [{minutes:02d}:{seconds:02d}] Speech detected...", end="", flush=True)

    def on_transcription(result: WhisperResult):
        """Called when transcription completes for a speech segment."""
        nonlocal utterance_count

        elapsed = time.time() - meeting_start

        # Calculate speech timing
        speech_duration = result.audio_duration
        speech_start = max(0, elapsed - speech_duration)
        speech_end = elapsed

        # Assign speaker
        speaker = diarizer.assign_speaker(speech_start, speech_end)

        # Create utterance record
        utterance = Utterance(
            speaker=speaker,
            text=result.text.strip(),
            start_time=speech_start,
            end_time=speech_end,
            confidence=result.confidence,
            segments=result.segments if result.segments else [],
        )
        meeting.add_utterance(utterance)
        utterance_count += 1

        # Live display
        minutes = int(speech_start // 60)
        seconds = int(speech_start % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"

        print()
        print(f"  鈹屸攢 {speaker} [{timestamp}] 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€")
        print(f"  锟?{result.text.strip()}")
        print(f"  鈹斺攢 confidence: {result.confidence:.2f} 鈹€鈹€")

    def on_event(event):
        """Handle stream events."""
        if event.event_type == "error":
            print(f"\n  锟?Error: {event.data}")

    # Wire up callbacks
    processor.on_speech_start = on_speech_start
    processor.on_transcription = on_transcription
    processor.on_event = on_event

    # 鈹€鈹€ Main Recording Loop 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print()
    print("  鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲锟?)
    print("  锟? 馃帣锟? Meeting in progress... Speak normally.           锟?)
    print("  锟? Press Ctrl+C to end the meeting.                     锟?)
    print("  鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲锟?)
    print()

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\n\n  鈿狅笍  Ending meeting...")

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while running:
            elapsed = time.time() - meeting_start
            if elapsed >= args.duration:
                print(f"\n  鈴憋笍  Duration limit reached ({args.duration}s)")
                break

            try:
                chunk_bytes = mic_stream.read(
                    stream_config.chunk_size,
                    exception_on_overflow=False,
                )
                audio_chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                processor.feed(audio_chunk)

            except OSError as e:
                logger.warning(f"Microphone read error: {e}")
                time.sleep(0.01)
                continue

            time.sleep(0.001)

    except Exception as e:
        print(f"\n  锟?Unexpected error: {e}")
        logger.exception("Unexpected error in main loop")

    finally:
        # 鈹€鈹€ Cleanup 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        mic_stream.stop_stream()
        mic_stream.close()
        pa.terminate()

        meeting.end_time = time.time() - meeting_start

    # 鈹€鈹€ Export Results 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print()
    print("  Exporting results...")

    if not meeting.utterances:
        print("  鈿狅笍  No speech detected during the meeting.")
        print("  Try lowering --vad-threshold or speaking closer to the microphone.")
    else:
        # Export SRT
        try:
            meeting.export_srt(srt_path)
            print(f"  锟?SRT exported: {srt_path}")
        except Exception as e:
            print(f"  锟?SRT export failed: {e}")
            logger.exception("SRT export error")

        # Export TXT
        try:
            meeting.export_txt(txt_path)
            print(f"  锟?TXT exported: {txt_path}")
        except Exception as e:
            print(f"  锟?TXT export failed: {e}")
            logger.exception("TXT export error")

    # 鈹€鈹€ Meeting Summary 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    stats = processor.stats
    total_time = time.time() - meeting_start

    print()
    print("锟? + "锟? * 58 + "锟?)
    print("锟? + "  Meeting Summary".center(58) + "锟?)
    print("锟? + "锟? * 58 + "锟?)
    print(f"锟? Duration:           {MeetingRecord._format_duration(total_time):<36}锟?)
    print(f"锟? Utterances:         {utterance_count:<36}锟?)
    print(f"锟? Speakers detected:  {meeting.speaker_count:<36}锟?)
    print(f"锟? Speech time:        {MeetingRecord._format_duration(stats['total_speech_duration_s']):<36}锟?)
    print(f"锟? Chunks processed:   {stats['chunks_processed']:<36}锟?)
    if stats['speech_segments'] > 0:
        print(f"锟? Avg latency:        {stats['avg_latency_ms']:.0f}ms{' ' * 31}锟?)
    print("锟? + "锟? * 58 + "锟?)
    print(f"锟? SRT: {srt_path:<51}锟?)
    print(f"锟? TXT: {txt_path:<51}锟?)
    print("锟? + "锟? * 58 + "锟?)
    print()


if __name__ == "__main__":
    main()