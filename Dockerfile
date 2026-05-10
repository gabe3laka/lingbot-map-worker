FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models \
    MODEL_FILE=lingbot-map-long.pt

WORKDIR /app

# System deps: Python 3.10 + ffmpeg + git + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git curl ca-certificates \
    ffmpeg \
    build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel

# PyTorch 2.8.0 cu128
RUN python3.10 -m pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Worker dependencies
RUN python3.10 -m pip install \
    runpod requests fastapi uvicorn \
    flashinfer-python \
    huggingface_hub \
    open3d

# Clone & install LingBot-Map
RUN git clone https://github.com/robbyant/lingbot-map /app/lingbot-map
RUN python3.10 -m pip install -e /app/lingbot-map

# Model checkpoint is downloaded at runtime by rp_handler.py
# using HF_TOKEN env var set in RunPod endpoint settings.
RUN mkdir -p /models /scratch

# Handler
COPY rp_handler.py /app/rp_handler.py

CMD ["python3.10", "-u", "/app/rp_handler.py"]
