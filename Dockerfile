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
    huggingface_hub

# Clone & install LingBot-Map
RUN git clone https://github.com/robbyant/lingbot-map /app/lingbot-map
RUN python3.10 -m pip install -e /app/lingbot-map

# Download model checkpoint at build time to /models
RUN mkdir -p ${MODEL_DIR} && \
    python3.10 - << 'PY' \
from huggingface_hub import hf_hub_download \
import os \
repo_id = "robbyant/lingbot-map" \
filename = "lingbot-map-long.pt" \
local_dir = os.environ.get("MODEL_DIR", "/models") \
hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, local_dir_use_symlinks=False) \
print("Downloaded", filename, "to", local_dir) \
PY

# Handler
COPY rp_handler.py /app/rp_handler.py

# Scratch root (jobs create /scratch/{scan_id})
RUN mkdir -p /scratch

CMD ["python3.10", "-u", "/app/rp_handler.py"]
