# ============================================================
# Omni-VRAM v2.1.1 — GPU Dockerfile
# ============================================================
# Build:   docker build -t omni-vram:gpu .
# Run:     docker run --gpus all -p 7860:7860 -p 8000:8000 omni-vram:gpu
# ============================================================

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

LABEL maintainer="Liangchenxu <lcx1479632@gmail.com>"
LABEL version="2.1.1"
LABEL description="Omni-VRAM GPU — Production-ready audio AI platform"

# ── Environment Variables ────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ── System Dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    libportaudio2 \
    alsa-utils \
    git \
    wget \
    curl \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Python Dependencies ──────────────────────────────────────
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# Copy dependency files first (Docker layer caching)
COPY requirements.txt setup.py pyproject.toml ./
COPY README.md ./

# Install core dependencies
RUN pip install -r requirements.txt

# Install PyTorch with CUDA 11.8 support
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install optional dependencies
RUN pip install \
    openai-whisper \
    edge-tts \
    deep-translator \
    gradio \
    fastapi \
    uvicorn \
    grpcio \
    grpcio-tools \
    flask \
    pyaudio \
    faster-whisper \
    2>/dev/null || true

# ── Copy Application ─────────────────────────────────────────
COPY vram_core/ ./vram_core/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY app.py ./
COPY vram_hacker.cu ./

# Install package in development mode
RUN pip install -e ".[full]" 2>/dev/null || pip install -e .

# ── Expose Ports ──────────────────────────────────────────────
# 7860: Gradio Web UI
# 8000: FastAPI REST API
# 50051: gRPC Server
EXPOSE 7860 8000 50051

# ── Health Check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import vram_core; print('OK')" || exit 1

# ── Default Command ───────────────────────────────────────────
# Start Gradio web UI (change to api_server.py for API mode)
CMD ["python3", "app.py", "--server-name", "0.0.0.0", "--server-port", "7860"]