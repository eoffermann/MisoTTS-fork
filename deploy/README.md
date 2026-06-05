# MisoTTS serving (Docker)

Run MisoTTS as a service, locally or on RunPod. The Linux container unlocks
triton/Inductor (`torch.compile` / CUDA graphs) and flash-attention SDPA, which
the Windows dev box cannot use, so this is where the generation throughput
actually improves. Output loudness is normalized in the serving core (the raw
model produces a ~19 dB clip-to-clip loudness spread).

## Layout

```
deploy/
  Dockerfile.modern           Linux + CUDA 12.8 image (torch 2.7.1+cu128) - RECOMMENDED
  requirements-modern.txt     modern model + quant + serving deps (see file header)
  Dockerfile                  legacy CUDA 12.4 image (torch 2.4.0+cu124); superseded
  requirements-serving.txt    legacy serving-only deps for the cu124 Dockerfile
  miso_server/
    core.py                   one model load; voice registry; loudness norm; synth()/synth_stream()
    audio.py                  wav/flac/opus/pcm encode + base64
    handler.py                RunPod serverless handler (generator yielding chunks)
    runpod_local.py           local RunPod simulator (/runsync + /run + /stream_now)
    openai_api.py             OpenAI-compatible /v1/audio/speech
  prompts/                    pre-assigned voice prompts (<name>.wav + <name>.txt)
docker-compose.runpod.yml     Path 1: local RunPod simulator (port 8000)
docker-compose.openai.yml     Path 2: OpenAI-compatible API (port 8080)
```

## Two compose paths

Both build the same image and pass through one GPU; they differ in the surface
they expose.

### 1. RunPod simulator (`docker-compose.runpod.yml`, port 8000)

Mirrors the RunPod serverless contract so the same client works locally and on
RunPod. `POST /runsync` is buffered (collects all handler yields into
`output[]`), exactly like RunPod. Streaming does not work through `/runsync`, so
a local-dev streaming path `POST /stream_now` drives `generate_stream` and emits
chunked NDJSON (or SSE with `Accept: text/event-stream`).

```bash
docker compose -f docker-compose.runpod.yml up --build
curl -s localhost:8000/runsync -H 'content-type: application/json' \
  -d '{"input":{"text":"Hello from the simulator.","voice":"default"}}'
curl -N localhost:8000/stream_now -H 'content-type: application/json' \
  -d '{"input":{"text":"Streaming test."}}'
```

For real RunPod, the image default CMD runs `python -m miso_server.handler` with
no `--rp_serve_api`, which RunPod's runtime drives directly.

### 2. OpenAI-compatible API (`docker-compose.openai.yml`, port 8080)

```bash
docker compose -f docker-compose.openai.yml up --build
curl -s localhost:8080/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"miso-tts-8b","input":"Hello world.","voice":"default","response_format":"wav"}' \
  --output out.wav
curl -s localhost:8080/v1/audio/voices
```

Point any OpenAI TTS client's `base_url` at `http://host:8080/v1`.

## Pre-assigned voices

Put `deploy/prompts/<name>.wav` + `deploy/prompts/<name>.txt` (transcript). They
are registered at startup and selectable by `voice` id. The reference is
Mimi-encoded once and reused. The built-in `default` voice (the model's own
voice) always exists. See `deploy/prompts/README.md`.

## Key environment variables

| var | default | meaning |
|-----|---------|---------|
| `MISO_COMPILE` | `1` in compose | `torch.compile` the model (Linux only) |
| `MISO_TARGET_LUFS` | `-16.0` | output loudness target; empty/None disables LUFS norm |
| `MISO_PEAK_CEILING_DBFS` | `-1.0` | peak ceiling after gain (anti-clip) |
| `MISO_WARMUP` | `1` | run warmup generations at startup (bakes compile graphs) |
| `MISO_TTS_8B_MODEL` | (HF default) | model path or repo id |
| `HF_HOME` | `/workspace/hf` | model cache (mount a volume to persist) |
| `MISO_QUANTIZE` | (auto by VRAM) | force weight precision: `bf16`/`none`, `int8`, or `int4`; overrides the VRAM auto-pick |
| `MISO_BF16_MIN_GB` | `22` | VRAM at/above which bf16 is used (else int8/int4) |
| `MISO_INT8_MIN_GB` | `13` | VRAM at/above which int8 is used (below -> int4) |
| `MISO_REPO_BF16` / `MISO_REPO_INT8` / `MISO_REPO_INT4` | `BigBlueCeiling/MisoTTS-*` | per-variant HF repos |

Weight precision is auto-selected by VRAM: bf16 on cards that fit it, else int8
(~16 GB), else int4 (~12 GB). int8/int4 are pulled as pre-quantized checkpoints
from their HF repos (falling back to quantizing the bf16 weights at load if a repo
is missing or its packed layout does not load on this GPU). int8 is quality-neutral;
int4 is experimental and audibly degraded (see the main README and
PERFORMANCE_PROGRESS.md). Quantization is a memory lever, not a speed one - it does
not speed up this autoregressive decode, it only fits smaller cards.

## Performance and cold start (measured on an A6000)

Measured on the modern stack (torch 2.7.1 + cu128, torchao 0.13.0, torchtune
0.6.1, moshi 0.2.13 - see `deploy/Dockerfile.modern`). Build it with:

```bash
docker build -f deploy/Dockerfile.modern -t miso-tts:modern .
```

Steady-state, compiled (`MISO_COMPILE=1`, reduce-overhead + flash SDPA), fully
warmed (RTF = wall / audio seconds; lower is faster):
- Batch RTF ~1.11-1.13 (near realtime), about 10x faster than eager (~10.9).
- Streaming TTFB ~3.2-5.4 s (vs ~22 s eager).
- The first couple of generations are slower while reduce-overhead recompiles
  across the first frame shapes; steady state is reached by the third clip.

Cold start (fresh process): about 7 min, split:
- model load ~165-175 s (32 GB read + cast; the bf16 variant halves this), and
- compile warmup ~237 s (first generation 161 s + a reshape recompile).

**The compile warmup now caches across processes** (the old torch-2.4 stack could
not - a warm run there still paid the full ~18 min). With a persisted per-SM
inductor cache, a second cold start's warmup drops from ~237 s to ~75 s (~68%)
and TTFB from 5.4 s to 3.2 s, for ~4 min total cold-start-to-serving. The serving
core already points `TORCHINDUCTOR_CACHE_DIR` at `/workspace/inductor_cache/sm_<cc>`
and enables the FX graph cache; **mount that path as a persistent volume** so the
warmup is paid once per GPU architecture, not per cold start. The residual ~75 s
is CUDA-graph capture, which is inherently per-process and cannot be cached, so a
warm/min worker (RunPod active workers / FlashBoot) is still ideal for the lowest
per-request latency. Eager (`MISO_COMPILE=0`, RTF ~10.9) remains the fallback when
a worker cannot be kept warm.

Quantization (`MISO_QUANTIZE`): on Ampere (A6000) it is **not** a throughput win
for this model. int8 weight-only dequantizes to bf16 for the matmul (the INT8
tensor cores need int8 x int8), adding overhead on a compute-bound model
(compiled RTF 1.36 vs bf16's 1.13); int8 dynamic (W8A8) cannot run at all because
the autoregressive decode's tiny per-step matmuls (M=1/10) violate the CUTLASS
INT8 GEMM's M>16 requirement. The now-working quant+compile path matters for the
GPUs where the precision math pays off: hardware fp8 on Ada/Hopper (sm 8.9+) and
nvfp4 on Blackwell, which the cu128 stack also enables. **Default to bf16 +
compile on Ampere.**

## Notes / TODO

- The model + Mimi codec + SilentCipher watermarker download into `HF_HOME` on
  first run; mount the `miso-hf` volume so containers do not re-download.
- A future, larger win is a disk-cached bf16 ("NVBF16") checkpoint to halve load
  I/O and skip the fp32->bf16 cast; and GPU-sense to pick a model variant
  (bf16 / int8 / compiled) per device. Tracked for a follow-up.
- `MISO_COMPILE` compilation warms up on the first few requests; the compose
  files run a warmup at startup.
