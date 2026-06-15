#!/usr/bin/env python3
"""
vram_core: Voice Chat Bot
===========================

A conversational voice bot that listens to microphone input, transcribes
speech with Whisper, manages multi-turn dialogue history, and optionally
connects to an LLM for AI responses.

Pipeline:
    Microphone 锟?VAD 锟?Whisper Transcription 锟?Display 锟?[LLM Response] 锟?Loop

Features:
    - Multi-turn voice conversation
    - Dialogue history tracking with timestamps
    - Silent exit (auto-stops after prolonged silence)
    - Export full conversation log on exit
    - LLM-ready architecture (plug in any LLM backend)

Usage:
    # Basic usage (Chinese, 5 minutes max)
    python examples/voice_chat_bot.py

    # English conversation, custom timeout
    python examples/voice_chat_bot.py --language en --silence-exit 60

    # Export to specific file
    python examples/voice_chat_bot.py --output my_chat.txt --model medium

    # Verbose debug mode
    python examples/voice_chat_bot.py --verbose

Output format (dialogue log):
    [00:01] User: 浣犲ソ锛屼粖澶╁ぉ姘旀€庝箞锟?
    [00:03] [AI: 鎶辨瓑锛孉I鍥炲鍔熻兘灏氭湭鎺ュ叆]
    [00:08] User: 璋㈣阿

Requirements:
    pip install pyaudio numpy pydub python-dotenv

    PyAudio installation (Windows):
        pip install pipwin && pipwin install pyaudio
    Or download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

    Whisper.cpp must be installed for local transcription.
    See README.md for setup instructions.

LLM Integration:
    This script is designed to be LLM-ready. To add AI responses:
    1. Implement a function that takes user text and returns AI response text
    2. Replace the placeholder in `get_ai_response()` method
    3. The dialogue history is passed to each call for context
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
class DialogueTurn:
    """A single turn in the conversation."""
    role: str          # "user" or "ai"
    text: str
    timestamp: float   # seconds from conversation start
    confidence: float = 0.0

    def format_line(self) -> str:
        """Format as a single dialogue log line."""
        minutes = int(self.timestamp // 60)
        seconds = int(self.timestamp % 60)
        ts = f"{minutes:02d}:{seconds:02d}"
        if self.role == "user":
            return f"[{ts}] User: {self.text}"
        else:
            return f"[{ts}] AI: {self.text}"


@dataclass
class ConversationLog:
    """Complete conversation log with history and export."""
    turns: List[DialogueTurn] = field(default_factory=list)
    language: str = "zh"

    @property
    def user_turns(self) -> List[DialogueTurn]:
        """Get only user turns."""
        return [t for t in self.turns if t.role == "user"]

    @property
    def ai_turns(self) -> List[DialogueTurn]:
        """Get only AI turns."""
        return [t for t in self.turns if t.role == "ai"]

    @property
    def total_turns(self) -> int:
        """Total number of dialogue turns."""
        return len(self.turns)

    def add_user_turn(self, text: str, timestamp: float, confidence: float = 0.0):
        """Add a user utterance."""
        self.turns.append(DialogueTurn(
            role="user",
            text=text,
            timestamp=timestamp,
            confidence=confidence,
        ))

    def add_ai_turn(self, text: str, timestamp: float):
        """Add an AI response."""
        self.turns.append(DialogueTurn(
            role="ai",
            text=text,
            timestamp=timestamp,
        ))

    def get_history_text(self) -> str:
        """Get conversation history as formatted text (for LLM context)."""
        lines = []
        for turn in self.turns:
            prefix = "User" if turn.role == "user" else "AI"
            lines.append(f"{prefix}: {turn.text}")
        return "\n".join(lines)

    def export_txt(self, filepath: str) -> None:
        """Export conversation as text file."""
        duration = self.turns[-1].timestamp if self.turns else 0.0
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Voice Chat Log\n")
            f.write("=" * 50 + "\n")
            f.write(f"Duration:    {minutes}m {seconds}s\n")
            f.write(f"User turns:  {len(self.user_turns)}\n")
            f.write(f"AI turns:    {len(self.ai_turns)}\n")
            f.write(f"Language:    {self.language}\n")
            f.write("=" * 50 + "\n\n")

            for turn in self.turns:
                f.write(turn.format_line() + "\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write("End of conversation\n")


# 鈹€鈹€ AI Response Handler 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class AIResponder:
    """
    AI response handler with pluggable backend.

    Currently returns placeholder responses. To integrate a real LLM:
    1. Implement `generate()` with your LLM API call
    2. The conversation history is available for context
    """

    def __init__(self, enabled: bool = False):
        """
        Initialize AI responder.

        Args:
            enabled: If True, attempt to generate responses.
                     If False, always return placeholder.
        """
        self.enabled = enabled

    def generate(self, user_text: str, history: ConversationLog) -> str:
        """
        Generate an AI response to user input.

        Args:
            user_text: The user's transcribed speech.
            history: Full conversation history for context.

        Returns:
            AI response text.
        """
        if not self.enabled:
            return "[AI鍥炲寰呮帴鍏"

        # 鈹€鈹€ LLM Integration Point 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        # Replace the section below with your LLM API call.
        # Example with OpenAI:
        #
        #     import openai
        #     messages = [
        #         {"role": "system", "content": "浣犳槸涓€涓弸濂界殑AI鍔╂墜锟?},
        #     ]
        #     for turn in history.turns:
        #         role = "user" if turn.role == "user" else "assistant"
        #         messages.append({"role": role, "content": turn.text})
        #
        #     response = openai.ChatCompletion.create(
        #         model="gpt-3.5-turbo",
        #         messages=messages,
        #     )
        #     return response.choices[0].message.content
        #
        # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

        return "[AI鍥炲寰呮帴鍏"


# 鈹€鈹€ Command Line Arguments 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="vram_core: Voice Chat Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/voice_chat_bot.py\n"
            "  python examples/voice_chat_bot.py --language en --silence-exit 60\n"
            "  python examples/voice_chat_bot.py --output my_chat.txt --model medium\n"
            "  python examples/voice_chat_bot.py --verbose\n"
        ),
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Maximum recording duration in seconds (default: 300)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="Language code for transcription (default: zh). Common: zh, en, ja, ko",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chat_log.txt",
        help="Output dialogue log file (default: chat_log.txt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--silence-exit",
        type=int,
        default=30,
        help="Silent exit timeout in seconds (default: 30). Auto-end if no speech detected.",
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
        "--device-cpu",
        action="store_true",
        default=False,
        help="Use CPU for whisper instead of CUDA",
    )
    parser.add_argument(
        "--enable-ai",
        action="store_true",
        default=False,
        help="Enable AI response generation (requires LLM backend integration)",
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
    """Main entry point for voice chat bot."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger("voice_chat_bot")

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

    # Print header
    print()
    print("锟? + "锟? * 58 + "锟?)
    print("锟? + "  vram_core: Voice Chat Bot".center(58) + "锟?)
    print("锟? + "锟? * 58 + "锟?)
    print()
    print(f"  Duration:       {args.duration}s")
    print(f"  Language:       {args.language}")
    print(f"  Model:          {args.model}")
    print(f"  Device:         {'CPU' if args.device_cpu else 'CUDA'}")
    print(f"  VAD Threshold:  {args.vad_threshold}")
    print(f"  Silence Exit:   {args.silence_exit}s")
    print(f"  Output:         {args.output}")
    print(f"  AI Response:    {'Enabled' if args.enable_ai else 'Disabled (placeholder)'}")
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

    # 鈹€鈹€ Step 3: Initialize AI Responder 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("  [3/4] Initializing AI responder...")
    ai_responder = AIResponder(enabled=args.enable_ai)
    conversation = ConversationLog(language=args.language)
    print(f"  锟?AI responder ready")

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

    # 鈹€鈹€ State 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    chat_start = time.time()
    last_speech_time = chat_start
    speech_detected_in_session = False

    # 鈹€鈹€ Callbacks 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def on_speech_start():
        """Called when VAD detects speech start."""
        nonlocal last_speech_time, speech_detected_in_session
        last_speech_time = time.time()
        speech_detected_in_session = True
        elapsed = time.time() - chat_start
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        print(f"\r  馃帳 [{minutes:02d}:{seconds:02d}] Listening...", end="", flush=True)

    def on_transcription(result: WhisperResult):
        """Called when transcription completes."""
        nonlocal last_speech_time
        elapsed = time.time() - chat_start
        last_speech_time = time.time()

        # Add user turn
        user_text = result.text.strip()
        if not user_text:
            return

        conversation.add_user_turn(
            text=user_text,
            timestamp=elapsed,
            confidence=result.confidence,
        )

        # Display user message
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        print()
        print(f"  鈹屸攢 You [{minutes:02d}:{seconds:02d}] 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€")
        print(f"  锟?{user_text}")
        print(f"  鈹斺攢 confidence: {result.confidence:.2f} 鈹€鈹€")

        # Generate AI response
        ai_elapsed = time.time() - chat_start
        ai_response = ai_responder.generate(user_text, conversation)
        conversation.add_ai_turn(text=ai_response, timestamp=ai_elapsed)

        # Display AI response
        ai_minutes = int(ai_elapsed // 60)
        ai_seconds = int(ai_elapsed % 60)
        print(f"  鈹屸攢 AI [{ai_minutes:02d}:{ai_seconds:02d}] 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€")
        print(f"  锟?{ai_response}")
        print(f"  鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€")
        print()
        print(f"  馃挰 Turns: {conversation.total_turns} | "
              f"Listening... (Ctrl+C to end)")

    def on_event(event):
        """Handle stream events."""
        if event.event_type == "error":
            print(f"\n  锟?Error: {event.data}")

    # Wire up callbacks
    processor.on_speech_start = on_speech_start
    processor.on_transcription = on_transcription
    processor.on_event = on_event

    # 鈹€鈹€ Main Loop 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print()
    print("  鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲锟?)
    print("  锟? 馃挰 Voice chat started. Speak to begin.               锟?)
    print(f"  锟? Auto-exit after {args.silence_exit}s of silence.           锟?)
    print("  锟? Press Ctrl+C to end.                                 锟?)
    print("  鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲锟?)
    print()

    running = True
    silent_exit = False

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\n\n  鈿狅笍  Ending conversation...")

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while running:
            elapsed = time.time() - chat_start

            # Check max duration
            if elapsed >= args.duration:
                print(f"\n  鈴憋笍  Duration limit reached ({args.duration}s)")
                break

            # Check silence exit
            if speech_detected_in_session:
                silence_time = time.time() - last_speech_time
                if silence_time >= args.silence_exit:
                    print(f"\n  馃攪 Silent for {args.silence_exit}s, ending conversation...")
                    silent_exit = True
                    break

            # Read audio chunk
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

    # 鈹€鈹€ Export Conversation 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print()
    print("  Exporting conversation...")

    if not conversation.turns:
        print("  鈿狅笍  No conversation recorded.")
        print("  Try lowering --vad-threshold or speaking closer to the microphone.")
    else:
        try:
            conversation.export_txt(args.output)
            print(f"  锟?Conversation saved: {args.output}")
        except Exception as e:
            print(f"  锟?Export failed: {e}")
            logger.exception("Export error")

    # 鈹€鈹€ Summary 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    stats = processor.stats
    total_time = time.time() - chat_start
    total_minutes = int(total_time // 60)
    total_seconds = int(total_time % 60)

    print()
    print("锟? + "锟? * 58 + "锟?)
    print("锟? + "  Chat Summary".center(58) + "锟?)
    print("锟? + "锟? * 58 + "锟?)
    dur_str = f"{total_minutes}m {total_seconds}s"
    user_count = len(conversation.user_turns)
    ai_count = len(conversation.ai_turns)
    speech_str = f"{stats['total_speech_duration_s']:.1f}s"
    print(f"锟? Duration:       {dur_str}{' ' * (38 - len(dur_str))}锟?)
    print(f"锟? User turns:     {user_count}{' ' * (38 - len(str(user_count)))}锟?)
    print(f"锟? AI turns:       {ai_count}{' ' * (38 - len(str(ai_count)))}锟?)
    print(f"锟? Speech time:    {speech_str}{' ' * (38 - len(speech_str))}锟?)
    if silent_exit:
        print(f"锟? Exit reason:    Silent timeout"
              f"{' ' * 25}锟?)
    else:
        print(f"锟? Exit reason:    User interrupted"
              f"{' ' * 23}锟?)
    print("锟? + "锟? * 58 + "锟?)
    print(f"锟? Log: {args.output:<51}锟?)
    print("锟? + "锟? * 58 + "锟?)
    print()


if __name__ == "__main__":
    main()