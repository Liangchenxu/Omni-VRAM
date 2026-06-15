#!/usr/bin/env python3
"""
vram_core API Server Demo Client
=================================

Demonstrates how to interact with the vram_core transcription API.

Usage:
    1. Start the API server:
       python vram_core/api_server.py --model base --language zh

    2. Run this demo:
       python examples/api_demo.py

Requirements:
    pip install requests websockets
"""

import os
import sys
import json
import base64
import asyncio
import argparse
import tempfile
import numpy as np

# 鈹€鈹€ Configuration 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

API_BASE = "http://127.0.0.1:8000"


def generate_test_wav(duration_s: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Generate a test WAV file with a sine wave."""
    import struct
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # 440 Hz sine wave (A4 note)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype(np.int16)

    # WAV header
    data_size = len(audio) * 2  # 16-bit = 2 bytes per sample
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, 1,  # PCM, mono
        sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size,
    )
    return header + audio.tobytes()


def demo_health_check():
    """Check API server health."""
    import requests
    print("\n" + "=" * 60)
    print("  Health Check")
    print("=" * 60)

    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        data = resp.json()
        print(f"  Status:       {data['status']}")
        print(f"  Version:      {data['version']}")
        print(f"  GPU:          {data['gpu']}")
        print(f"  Backend:      {data['backend']}")
        print(f"  Available:    {data['available_backends']}")
        return True
    except Exception as e:
        print(f"  锟?Connection failed: {e}")
        print(f"  馃挕 Make sure the API server is running:")
        print(f"     python vram_core/api_server.py")
        return False


def demo_root():
    """Check API root endpoint."""
    import requests
    print("\n" + "=" * 60)
    print("  API Root Info")
    print("=" * 60)

    resp = requests.get(f"{API_BASE}/", timeout=5)
    data = resp.json()
    print(f"  Name:    {data['name']}")
    print(f"  Version: {data['version']}")
    print(f"  Docs:    {data['docs']}")
    print(f"  Endpoints:")
    for path, desc in data['endpoints'].items():
        print(f"    {path}: {desc}")


def demo_file_transcribe(audio_path: str = None):
    """Demo: File upload transcription."""
    import requests
    print("\n" + "=" * 60)
    print("  File Upload Transcription (POST /transcribe)")
    print("=" * 60)

    if audio_path and os.path.exists(audio_path):
        print(f"  File: {audio_path}")
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f)}
            resp = requests.post(f"{API_BASE}/transcribe", files=files, timeout=60)
    else:
        # Use generated test audio
        wav_bytes = generate_test_wav(2.0)
        print("  File: [generated 2s test tone]")
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        resp = requests.post(f"{API_BASE}/transcribe", files=files, timeout=60)

    if resp.status_code == 200:
        data = resp.json()
        print(f"  Text:       {data['text']}")
        print(f"  Language:   {data['language']}")
        print(f"  Duration:   {data['duration']:.2f}s")
        print(f"  Confidence: {data['confidence']:.2f}")
        print(f"  Backend:    {data['backend']}")
        print(f"  Time:       {data['processing_time']:.2f}s")
    else:
        print(f"  锟?Error {resp.status_code}: {resp.text}")


def demo_base64_transcribe(audio_path: str = None):
    """Demo: Base64 audio transcription."""
    import requests
    print("\n" + "=" * 60)
    print("  Base64 Transcription (POST /transcribe/base64)")
    print("=" * 60)

    if audio_path and os.path.exists(audio_path):
        with open(audio_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        print(f"  File: {audio_path}")
    else:
        wav_bytes = generate_test_wav(1.5)
        b64 = base64.b64encode(wav_bytes).decode()
        print("  File: [generated 1.5s test tone]")

    print(f"  Base64 length: {len(b64)} chars")

    resp = requests.post(
        f"{API_BASE}/transcribe/base64",
        json={"audio_base64": b64, "language": "zh"},
        timeout=60,
    )

    if resp.status_code == 200:
        data = resp.json()
        print(f"  Text:       {data['text']}")
        print(f"  Language:   {data['language']}")
        print(f"  Duration:   {data['duration']:.2f}s")
        print(f"  Confidence: {data['confidence']:.2f}")
        print(f"  Time:       {data['processing_time']:.2f}s")
    else:
        print(f"  锟?Error {resp.status_code}: {resp.text}")


def demo_websocket_stream():
    """Demo: WebSocket streaming transcription."""
    print("\n" + "=" * 60)
    print("  WebSocket Streaming (WS /stream)")
    print("=" * 60)

    try:
        import websockets
    except ImportError:
        print("  鈿狅笍  websockets package not installed")
        print("  馃挕 Install with: pip install websockets")
        return

    async def _stream():
        uri = API_BASE.replace("http", "ws") + "/stream"
        print(f"  Connecting to {uri}...")

        async with websockets.connect(uri) as ws:
            # Wait for ready message
            msg = json.loads(await ws.recv())
            print(f"  Server: {msg}")

            # Send 3 audio chunks
            for i in range(3):
                # Generate 200ms of audio (3200 samples at 16kHz)
                t = np.linspace(0, 0.2, 3200, endpoint=False)
                audio = (np.sin(2 * np.pi * (300 + i * 100) * t) * 0.3 * 32767).astype(np.int16)
                await ws.send(audio.tobytes())
                print(f"  Sent chunk {i + 1}/3 ({len(audio.tobytes())} bytes)")

                # Check for responses
                try:
                    while True:
                        resp = await asyncio.wait_for(ws.recv(), timeout=0.1)
                        data = json.loads(resp)
                        print(f"  锟?{data}")
                except asyncio.TimeoutError:
                    pass

            # Send stop command
            await ws.send(json.dumps({"action": "stop"}))
            print("  Sent stop command")

            # Collect remaining results
            try:
                while True:
                    resp = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    data = json.loads(resp)
                    print(f"  锟?{data}")
                    if data.get("type") == "stopped":
                        break
            except asyncio.TimeoutError:
                pass

        print("  Connection closed")

    asyncio.run(_stream())


def main():
    parser = argparse.ArgumentParser(
        description="vram_core API Demo Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/api_demo.py
    python examples/api_demo.py --audio path/to/audio.wav
    python examples/api_demo.py --host 192.168.1.100 --port 9000
    python examples/api_demo.py --demo file     # only file upload
    python examples/api_demo.py --demo base64   # only base64
    python examples/api_demo.py --demo stream   # only websocket
        """,
    )
    parser.add_argument("--host", default="127.0.0.1", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--audio", default=None, help="Path to audio file")
    parser.add_argument(
        "--demo", choices=["all", "file", "base64", "stream"],
        default="all", help="Which demo to run",
    )

    args = parser.parse_args()

    global API_BASE
    API_BASE = f"http://{args.host}:{args.port}"

    print("鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽")
    print("锟?        vram_core API Server Demo Client                锟?)
    print("鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆")
    print(f"  Server: {API_BASE}")

    # Health check
    if not demo_health_check():
        sys.exit(1)

    demo_root()

    # Run selected demos
    if args.demo in ("all", "file"):
        demo_file_transcribe(args.audio)

    if args.demo in ("all", "base64"):
        demo_base64_transcribe(args.audio)

    if args.demo in ("all", "stream"):
        demo_websocket_stream()

    print("\n锟?Demo complete!")


if __name__ == "__main__":
    main()