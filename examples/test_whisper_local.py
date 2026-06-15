#!/usr/bin/env python3
"""
vram_core: Local Whisper Transcription Test
=============================================

Test script for local whisper.cpp transcription.
Accepts an audio file path, runs transcription, and optionally exports SRT.

Usage:
    # Basic usage
    python examples/test_whisper_local.py --audio test.wav

    # Specify model and export SRT
    python examples/test_whisper_local.py --audio test.mp3 --model medium --export-srt

    # Specify output path
    python examples/test_whisper_local.py --audio test.wav --export-srt --output subtitles.srt

    # Force language
    python examples/test_whisper_local.py --audio test.wav --language zh

    # Use CPU instead of CUDA
    python examples/test_whisper_local.py --audio test.wav --device cpu

Requirements:
    1. whisper.cpp installed and configured (see README.md)
    2. GGML model downloaded (e.g., ggml-base.bin)
    3. Set WHISPER_CPP_PATH in .env (optional, auto-detected)
    4. pip install pydub python-dotenv

Supported audio formats: wav, mp3, flac, ogg, m4a, wma, aac
"""

import argparse
import sys
import os
import time
import logging
from pathlib import Path

# Add project root to path so we can import vram_core
sys.path.insert(0, str(Path(__file__).parent.parent))

from vram_core.whisper_bridge import WhisperBridge, WhisperBackend
from vram_core.config import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="vram_core: Local Whisper Transcription Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/test_whisper_local.py --audio test.wav\n"
            "  python examples/test_whisper_local.py --audio test.mp3 --model medium --export-srt\n"
            "  python examples/test_whisper_local.py --audio test.wav --export-srt --output subtitles.srt\n"
            "  python examples/test_whisper_local.py --audio test.wav --language zh\n"
        ),
    )

    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file (wav/mp3/flac/ogg/m4a)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force language code (zh/en/ja/etc). Auto-detect if not set.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Compute device (default: from .env or cuda)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "whisper_cpp", "openai_api"],
        help="Transcription backend (default: auto)",
    )
    parser.add_argument(
        "--export-srt",
        action="store_true",
        default=False,
        help="Export transcription as SRT subtitle file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for SRT (default: <audio_name>.srt)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose/debug logging",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger("test_whisper_local")

    # Validate audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        print(f"\n  Error: Audio file not found: {audio_path}")
        sys.exit(1)

    supported = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac"}
    if audio_path.suffix.lower() not in supported:
        logger.error(f"Unsupported format: {audio_path.suffix}")
        print(f"\n  Error: Unsupported format: {audio_path.suffix}")
        print(f"  Supported: {', '.join(sorted(supported))}")
        sys.exit(1)

    # Map backend string to enum
    backend_map = {
        "auto": WhisperBackend.AUTO,
        "whisper_cpp": WhisperBackend.WHISPER_CPP,
        "openai_api": WhisperBackend.OPENAI_API,
    }
    backend = backend_map[args.backend]

    # Initialize WhisperBridge
    print("=" * 60)
    print("  vram_core: Local Whisper Transcription")
    print("=" * 60)
    print(f"  Audio:    {audio_path}")
    print(f"  Model:    {args.model}")
    print(f"  Backend:  {args.backend}")
    if args.language:
        print(f"  Language: {args.language}")
    if args.device:
        print(f"  Device:   {args.device}")
    print("=" * 60)

    try:
        bridge = WhisperBridge(
            backend=backend,
            whisper_model=args.model,
            language=args.language,
            device=args.device,
        )
    except Exception as e:
        logger.error(f"Failed to initialize WhisperBridge: {e}")
        print(f"\n  Error: Failed to initialize WhisperBridge:")
        print(f"  {e}")
        sys.exit(1)

    # Show bridge status
    status = bridge.get_status()
    print(f"  Active backend: {status['backend']}")
    print(f"  Available:      {status['available_backends']}")
    print()

    # Run transcription
    print("Transcribing...")
    start_time = time.time()

    try:
        result = bridge.transcribe(audio_path)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n  Error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Value error: {e}")
        print(f"\n  Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        print(f"\n  Transcription failed: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time

    # Print results
    print()
    print("-" * 60)
    print("  Transcription Result")
    print("-" * 60)
    print(f"  Language:    {result.language}")
    print(f"  Confidence:  {result.confidence:.2f}")
    print(f"  Segments:    {len(result.segments)}")
    print(f"  Duration:    {result.audio_duration:.2f}s")
    print(f"  Process:     {result.processing_time:.2f}s")
    print(f"  Backend:     {result.backend.value if result.backend else 'N/A'}")
    print("-" * 60)
    print()
    print("  Text:")
    print(f"  {result.text}")
    print()

    # Print segments
    if result.segments:
        print("-" * 60)
        print("  Segments:")
        print("-" * 60)
        for i, seg in enumerate(result.segments, 1):
            start = seg.get("start", "?")
            end = seg.get("end", "?")
            text = seg.get("text", "").strip()
            conf = seg.get("confidence")
            conf_str = f" (p={conf:.2f})" if conf is not None else ""
            print(f"  [{i:3d}] {start} --> {end}{conf_str}")
            print(f"        {text}")
        print()

    # Export SRT
    if args.export_srt:
        if not result.segments:
            print("  Warning: No segments available for SRT export.")
        else:
            srt_path = args.output or f"{audio_path.stem}.srt"
            try:
                result.export_srt(srt_path)
                print(f"  SRT exported: {srt_path}")
            except Exception as e:
                logger.error(f"SRT export failed: {e}")
                print(f"  SRT export failed: {e}")

    print("=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()