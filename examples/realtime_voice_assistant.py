#!/usr/bin/env python3
"""
vram_core: Real-Time Voice Assistant
======================================

A real-time voice assistant that captures microphone audio, detects speech
using Voice Activity Detection (VAD), transcribes with Whisper, and
optionally sends text to an LLM for responses.

Pipeline:
    Microphone 锟?VAD 锟?Whisper Transcription 锟?[LLM Response] 锟?Display

Usage:
    # Basic usage (record for 60 seconds, auto-detect device)
    python examples/realtime_voice_assistant.py

    # Record for 120 seconds with verbose output
    python examples/realtime_voice_assistant.py --duration 120 --verbose

    # Specify microphone device index
    python examples/realtime_voice_assistant.py --device 1

    # Use CPU for whisper and adjust VAD sensitivity
    python examples/realtime_voice_assistant.py --device-cpu --vad-threshold 0.01

Requirements:
    pip install pyaudio numpy pydub python-dotenv

    PyAudio installation (Windows):
        pip install pipwin && pipwin install pyaudio
    Or download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

    Whisper.cpp must be installed for local transcription.
    See README.md for setup instructions.

Controls:
    - Speak into the microphone and the assistant will transcribe in real-time
    - Press Ctrl+C to stop
    - Speech is automatically detected via VAD (Voice Activity Detection)
"""

import argparse
import sys
import time
import signal
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from vram_core.whisper_bridge import WhisperBridge, WhisperBackend
from vram_core.stream_processor import StreamProcessor, StreamConfig, StreamState
from vram_core.config import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="vram_core: Real-Time Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/realtime_voice_assistant.py\n"
            "  python examples/realtime_voice_assistant.py --duration 120 --verbose\n"
            "  python examples/realtime_voice_assistant.py --device 1 --vad-threshold 0.01\n"
        ),
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Recording duration in seconds (default: 60)",
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
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force language code (zh/en/ja/etc). Auto-detect if not set.",
    )
    parser.add_argument(
        "--device-cpu",
        action="store_true",
        default=False,
        help="Use CPU for whisper instead of CUDA",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.02,
        help="VAD energy threshold (0.0-1.0, default: 0.02). Lower = more sensitive.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="Audio chunk size in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose/debug logging",
    )

    return parser.parse_args()


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
        if info["maxInputChannels"] > 0:  # Input devices only
            default_marker = " (DEFAULT)" if i == pa.get_default_input_device_info()["index"] else ""
            print(f"  [{i:2d}] {info['name']}{default_marker}")
            print(f"       Channels: {info['maxInputChannels']}, "
                  f"Sample Rate: {int(info['defaultSampleRate'])}Hz")

    print("=" * 60)
    pa.terminate()


def create_mic_stream(
    pa_instance,
    device_index: int = None,
    sample_rate: int = 16000,
    chunk_size: int = 1600,
):
    """
    Create a PyAudio microphone stream.

    Args:
        pa_instance: PyAudio instance.
        device_index: Input device index (None for default).
        sample_rate: Sample rate in Hz.
        chunk_size: Frames per buffer.

    Returns:
        PyAudio stream object.
    """
    # Verify device if specified
    if device_index is not None:
        try:
            dev_info = pa_instance.get_device_info_by_index(device_index)
            if dev_info["maxInputChannels"] == 0:
                print(f"Error: Device {device_index} is not an input device.")
                sys.exit(1)
            # Use device's native sample rate if available
            native_rate = int(dev_info["defaultSampleRate"])
            if native_rate != sample_rate:
                print(f"  Note: Device native rate is {native_rate}Hz, "
                      f"requesting {sample_rate}Hz (resampling may occur)")
        except Exception as e:
            print(f"Error: Invalid device index {device_index}: {e}")
            sys.exit(1)

    stream = pa_instance.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk_size,
    )

    return stream


def on_state_change(state: StreamState):
    """Callback for stream state changes."""
    state_icons = {
        StreamState.IDLE: "馃數",
        StreamState.LISTENING: "馃煝",
        StreamState.SPEAKING: "馃敶",
        StreamState.PROCESSING: "锟?,
        StreamState.ERROR: "锟?,
    }
    icon = state_icons.get(state, "锟?)
    print(f"  {icon} State: {state.value}")


def on_speech_start():
    """Callback when speech is detected."""
    print("\n  馃帳 Speech detected, listening...")


def on_speech_end(audio: np.ndarray):
    """Callback when speech segment ends."""
    duration = len(audio) / 16000
    print(f"  馃帳 Speech ended ({duration:.1f}s), processing...")


def on_transcription(result):
    """Callback when transcription is complete."""
    print()
    print("  鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€")
    print(f"  锟?馃摑 Transcription ({result.language}):")
    print(f"  锟?{result.text}")
    print(f"  锟?Confidence: {result.confidence:.2f} | "
          f"Duration: {result.audio_duration:.1f}s | "
          f"Time: {result.processing_time:.1f}s")
    if result.segments:
        print(f"  锟?Segments: {len(result.segments)}")
    print("  鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€")
    print()
    print("  Listening for speech... (Ctrl+C to stop)")


def on_event(event):
    """Callback for stream events (debug)."""
    if event.event_type == "error":
        print(f"  锟?Error: {event.data}")


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger("voice_assistant")

    # List devices mode
    if args.list_devices:
        list_audio_devices()
        return

    # Check pyaudio availability
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
    print("=" * 60)
    print("  vram_core: Real-Time Voice Assistant")
    print("=" * 60)
    print(f"  Duration:     {args.duration}s")
    print(f"  Model:        {args.model}")
    print(f"  Device:       {'CPU' if args.device_cpu else 'CUDA'}")
    print(f"  VAD Threshold:{args.vad_threshold}")
    print(f"  Sample Rate:  {args.sample_rate}Hz")
    print(f"  Chunk Size:   {args.chunk_ms}ms")
    if args.language:
        print(f"  Language:     {args.language}")
    print("=" * 60)

    # 鈹€鈹€ Step 1: Initialize WhisperBridge 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("\n  [1/3] Initializing Whisper...")
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
        print()
        print("  Make sure whisper.cpp is installed and configured.")
        print("  See README.md for setup instructions.")
        sys.exit(1)

    # 鈹€鈹€ Step 2: Initialize StreamProcessor 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("  [2/3] Initializing stream processor...")
    stream_config = StreamConfig(
        sample_rate=args.sample_rate,
        chunk_duration_ms=args.chunk_ms,
        vad_threshold=args.vad_threshold,
    )
    processor = StreamProcessor(
        config=stream_config,
        whisper_bridge=whisper,
    )

    # Set callbacks
    processor.on_state_change = on_state_change
    processor.on_speech_start = on_speech_start
    processor.on_speech_end = on_speech_end
    processor.on_transcription = on_transcription
    processor.on_event = on_event

    print(f"  锟?Stream processor ready (chunk: {stream_config.chunk_size} samples)")

    # 鈹€鈹€ Step 3: Initialize Microphone 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("  [3/3] Initializing microphone...")
    pa = pyaudio.PyAudio()

    try:
        mic_stream = create_mic_stream(
            pa,
            device_index=args.device,
            sample_rate=args.sample_rate,
            chunk_size=stream_config.chunk_size,
        )
        device_name = "default"
        if args.device is not None:
            device_name = pa.get_device_info_by_index(args.device)["name"]
        print(f"  锟?Microphone ready ({device_name})")
    except Exception as e:
        print(f"  锟?Failed to open microphone: {e}")
        pa.terminate()
        sys.exit(1)

    # 鈹€鈹€ Main Loop 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print()
    print("  馃帣锟? Listening for speech... (Ctrl+C to stop)")
    print()

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\n  鈿狅笍  Stopping...")

    signal.signal(signal.SIGINT, signal_handler)

    start_time = time.time()
    chunks_read = 0

    try:
        while running:
            # Check duration limit
            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                print(f"\n  鈴憋笍  Duration limit reached ({args.duration}s)")
                break

            # Read audio chunk from microphone
            try:
                chunk_bytes = mic_stream.read(
                    stream_config.chunk_size,
                    exception_on_overflow=False,
                )
                # Convert int16 bytes to float32 numpy array
                audio_chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                # Feed to stream processor
                processor.feed(audio_chunk)
                chunks_read += 1

            except OSError as e:
                logger.warning(f"Microphone read error: {e}")
                time.sleep(0.01)
                continue

            # Small sleep to prevent busy-waiting
            # (pyaudio.read is blocking, so this is mostly a safety net)
            time.sleep(0.001)

    except Exception as e:
        print(f"\n  锟?Unexpected error: {e}")
        logger.exception("Unexpected error in main loop")

    finally:
        # 鈹€鈹€ Cleanup 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        print("\n  Cleaning up...")

        mic_stream.stop_stream()
        mic_stream.close()
        pa.terminate()

        # Print statistics
        stats = processor.stats
        total_time = time.time() - start_time

        print()
        print("=" * 60)
        print("  Session Statistics")
        print("=" * 60)
        print(f"  Total time:        {total_time:.1f}s")
        print(f"  Chunks processed:  {stats['chunks_processed']}")
        print(f"  Speech segments:   {stats['speech_segments']}")
        print(f"  Speech duration:   {stats['total_speech_duration_s']:.1f}s")
        print(f"  Processing time:   {stats['total_processing_time_s']:.1f}s")
        if stats['speech_segments'] > 0:
            print(f"  Avg latency:       {stats['avg_latency_ms']:.0f}ms")
        print("=" * 60)
        print("  Done! 馃憢")
        print("=" * 60)


if __name__ == "__main__":
    main()