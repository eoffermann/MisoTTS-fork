# MisoTTS serving (Docker)

Run MisoTTS as a service, locally or on RunPod. The Linux container unlocks
triton/Inductor (`torch.compile` / CUDA graphs) and flash-attention SDPA, which
the Windows dev box cannot use, so this is where the generation throughput
actually improves. Output loudness is normalized in the serving core (the raw
model produces a ~19 dB clip-to-clip loudness spread).

## Layout

```
deploy/
  Dockerfile                  Linux + CUDA 12.4 image (torch 2.4.0+cu124, triton, deps)
  requirements-serving.txt    fastapi / uvicorn / runpod / soundfile / pyloudnorm
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

## Notes / TODO

- The model + Mimi codec + SilentCipher watermarker download into `HF_HOME` on
  first run; mount the `miso-hf` volume so containers do not re-download.
- A future, larger win is a disk-cached bf16 ("NVBF16") checkpoint to halve load
  I/O and skip the fp32->bf16 cast; and GPU-sense to pick a model variant
  (bf16 / int8 / compiled) per device. Tracked for a follow-up.
- `MISO_COMPILE` compilation warms up on the first few requests; the compose
  files run a warmup at startup.
