# MisoTTS serving image (Linux + CUDA).
#
# Running on Linux unlocks what the Windows dev box cannot do: triton/Inductor
# (torch.compile / CUDA graphs) and flash-attention SDPA kernels. Set
# MISO_COMPILE=1 at runtime to compile the model; flash SDPA is used
# automatically by torch on Linux where available.
#
# Build (from the repo root):
#   docker build -f deploy/Dockerfile -t miso-tts:latest .
# Optionally build flash-attn (slow): --build-arg INSTALL_FLASH_ATTN=1

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/hf MISO_PROMPTS_DIR=/workspace/prompts

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip git ffmpeg libsndfile1 ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/miso

# 1) Pin torch/torchaudio to the cu124 wheels and numpy<2 (ABI) FIRST so the
#    project install below does not pull a CPU torch or numpy 2.x.
RUN python -m pip install --upgrade pip \
    && pip install "numpy<2" torch==2.4.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
        torchtune==0.4.0 torchao==0.9.0

# 2) Project model dependencies (transformers, moshi, safetensors, tokenizers,
#    huggingface_hub, silentcipher, etc.) from the repo metadata.
COPY pyproject.toml ./
RUN pip install $(python - <<'PY'
import tomllib
d = tomllib.load(open("pyproject.toml","rb"))
deps = d.get("project", {}).get("dependencies", [])
# torch/torchaudio/torchtune/torchao already installed above; skip to avoid re-resolve
skip = ("torch", "numpy")
print(" ".join(repr(x) for x in deps if not x.lower().startswith(skip)))
PY
) || pip install -e . ; pip install "numpy<2"

# 3) Triton (Inductor backend for torch.compile) and serving deps.
COPY deploy/requirements-serving.txt deploy/requirements-serving.txt
RUN pip install "triton; platform_system=='Linux'" \
    && pip install -r deploy/requirements-serving.txt \
    && pip install "numpy<2"

# 4) Optional flash-attention (slow to build). flash SDPA also works without it.
ARG INSTALL_FLASH_ATTN=0
RUN if [ "$INSTALL_FLASH_ATTN" = "1" ]; then pip install flash-attn --no-build-isolation; fi

# 5) Application code.
COPY . /opt/miso
ENV PYTHONPATH=/opt/miso

# Default: the RunPod serverless handler (real RunPod drives it; no --rp_serve_api).
CMD ["python", "-m", "miso_server.handler"]
