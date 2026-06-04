# MisoTTS serving image (Linux + CUDA).
#
# Running on Linux unlocks what the Windows dev box cannot: triton/Inductor
# (torch.compile / CUDA graphs) and flash-attention SDPA kernels. Set
# MISO_COMPILE=1 at runtime to compile the model; flash SDPA is used
# automatically by torch on Linux where available.
#
# Build (from the repo root):
#   docker build -f deploy/Dockerfile -t miso-tts:latest .
# Optionally build flash-attn (slow): --build-arg INSTALL_FLASH_ATTN=1

# DEVEL (not runtime) base: triton/Inductor JIT-compiles kernels at runtime and
# needs a C compiler + CUDA headers (cuda.h, nvcc). The cudnn-runtime image has
# neither, so torch.compile fails with "Failed to find C compiler". The devel
# image plus build-essential provides them.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/hf MISO_PROMPTS_DIR=/workspace/prompts

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip git ffmpeg libsndfile1 ca-certificates build-essential \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/miso

# 1) Pin the torch stack (cu124 wheels) and numpy<2 (ABI) BEFORE installing the
#    project, so the project deps do not pull a CPU torch or numpy 2.x. The
#    "==2.4.0" spec in pyproject is satisfied by the local "+cu124" build.
RUN python -m pip install --upgrade pip \
    && pip install "numpy<2" torch==2.4.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124

# 2) Install the project and its remaining deps from pyproject (transformers,
#    moshi, torchtune, torchao, bitsandbytes [linux], silentcipher [git]). Then
#    re-pin numpy<2 in case a transitive dep bumped it.
COPY . /opt/miso
RUN pip install -e . \
    && pip install "numpy<2"

# 3) Triton (Inductor backend for torch.compile) and serving deps.
RUN pip install triton \
    && pip install -r deploy/requirements-serving.txt \
    && pip install "numpy<2"

# 3b) Python dev headers. triton JIT-compiles a cuda_utils helper that
#     #includes <Python.h>, so the headers must be present or torch.compile
#     fails. Kept as a separate late layer so the torch/pip layers above (and
#     the already-pushed image layers) stay cached.
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 4) Optional flash-attention (slow to build; flash SDPA works without it too).
ARG INSTALL_FLASH_ATTN=0
RUN if [ "$INSTALL_FLASH_ATTN" = "1" ]; then pip install flash-attn --no-build-isolation; fi

# PYTHONPATH includes the repo root (so `generator`, `models` import) AND deploy/
# (so the `miso_server` package imports).
ENV PYTHONPATH=/opt/miso:/opt/miso/deploy

# Default: the RunPod serverless handler (real RunPod drives it; no --rp_serve_api).
CMD ["python", "-m", "miso_server.handler"]
