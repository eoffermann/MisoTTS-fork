<div align="center">

# MisoTTS (BigBlueCeiling fork)

### An optimization- and deployment-focused fork of MisoTTS

<p>
  <a href="https://github.com/MisoLabsAI/MisoTTS"><img alt="Upstream" src="https://img.shields.io/badge/Upstream-MisoLabsAI%2FMisoTTS-181717?style=for-the-badge&logo=github&labelColor=555555"></a>
  <a href="https://huggingface.co/MisoLabs/MisoTTS"><img alt="Model" src="https://img.shields.io/badge/Model-MisoLabs%2FMisoTTS-yellow?style=for-the-badge"></a>
</p>

</div>

---

This repository is BigBlueCeiling's fork of [MisoLabsAI/MisoTTS](https://github.com/MisoLabsAI/MisoTTS).
The model and the original inference code are MisoLabs' work. This fork exists to
do engineering on top of that: make it fast and correct in practice, and make it
easy to deploy across a range of hardware. It is not a separate model and not a
specific product; it is the optimization and deployment substrate.

## The model (origin)

MisoTTS is an expressive, English, ~8B-parameter text-to-speech model from
[MisoLabsAI](https://github.com/MisoLabsAI), inspired by the Sesame CSM
architecture: a Llama-3.2-style backbone (`llama-8B`) generates Mimi audio codes
from text and optional audio context, and a smaller autoregressive decoder
(`llama-300M`) predicts the higher-order codebooks per frame. Output is
watermarked with SilentCipher. All credit for the model belongs upstream; see
[MisoLabsAI/MisoTTS](https://github.com/MisoLabsAI/MisoTTS) and the
[model card](https://huggingface.co/MisoLabs/MisoTTS) for the architecture,
weights, and the original documentation.

We track upstream via the `upstream` remote and send general fixes back as pull
requests (for example the KV-cache device fix, upstream PR #13).

## What this fork is for

Two engineering goals, kept separate from any application we happen to build on
top of the model:

1. **Optimization.** Fix the bugs that block real use, remove wasted work in the
   load and generation paths, and unlock the throughput the hardware can actually
   deliver (torch.compile / CUDA graphs / flash-attention where the platform
   supports them). On a recent datacenter GPU this takes the model from many
   times slower than realtime, eager, to near realtime.

2. **Portable, easy deployment.** Run it locally or in the cloud behind a stable
   API, and match the model to the hardware instead of assuming one reference
   card. A contributor or a deployment should be able to pick a variant that fits
   their GPU rather than being locked out by a single fixed precision and memory
   requirement.

The applications will vary and shift over time. The substrate should stay fast,
correct, and easy to deploy regardless.

## Issues from upstream we are eliminating

Concrete correctness and efficiency problems we have fixed or are fixing.
`CHANGES_FROM_MAIN_REPO.md` has the commit-level detail for each.

- **Unusable on GPU out of the box.** KV caches were allocated on the CPU while
  the model ran on the GPU, aborting generation with a device-mismatch error.
  Fixed (and submitted upstream).
- **Clips cut off at the end.** Two causes: generation caps sized for a faster
  speech rate than the model actually produces, and an end-of-stream decode that
  dropped the final frame. Fixed, with a perceptual end-truncation gate to keep
  it fixed.
- **Wasted startup and per-frame work.** The full 8B model was random-initialized
  and then immediately overwritten by the checkpoint; a no-op resample ran on
  every clip; the decoder forced a host sync every frame. Removed.
- **Unpredictable output loudness.** No loudness normalization, so clips varied by
  roughly 19 dB with no relation to content. The serving layer normalizes to a
  target loudness.
- **No deployment story.** No service, no API, and no way to match the model to
  the hardware. This fork adds all three.

## Running across a range of hardware

This is an 8B model, so it is not aimed at a 2015-era GPU and never will be. The
realistic goal is to be a strong, runnable option across a range of reasonably
recent cards, so that contributors and deployments are not classed out by a
single hardware requirement. Lowering that floor matters for continued
development: more people can run it, profile it, and contribute.

The mechanism is selectable, hardware-matched model variants. The serving core
detects the GPU architecture and selects the best variant that exists, falling
back to bf16 and logging the gap when it detects a GPU that could use a
better-but-not-yet-built variant. Variants live in separate Hugging Face repos
and are pulled at runtime, so the image stays small and weights version
independently.

| Variant | Precision | Roughly suits | Status |
|---|---|---|---|
| `bf16` | bfloat16 weights | ~16-20 GB VRAM, Ampere or newer (RTX 3090/4090, A6000, ...) | available |
| `int8` / `fp8` weight-only | 8-bit weights, bf16 compute (runs on most recent GPUs) | lighter memory; broad compatibility | evaluating |
| `fp8` (hardware) | fp8 tensor cores | Ada / Hopper / Blackwell (sm 8.9+) | planned, hardware-gated |
| `nvfp4` | 4-bit Blackwell FP4 | smallest and fastest, Blackwell only | planned, hardware-gated |

Realistic caveats:
- Quantized variants trade some quality for memory and throughput. We measure
  both, with the quality harness below, before recommending one.
- The biggest throughput wins (CUDA graphs, flash-attention, hardware
  low-precision) need a modern GPU and a current CUDA stack. Older but supported
  cards get correctness and bf16, not the fastest paths.

## Quality gating

Optimization is only useful if it does not quietly degrade the audio. Changes are
validated with `perf_eval/`, a port of a perceptual TTS-evaluation harness that
scores candidate renders against a baseline (ASR intelligibility, MOS-style
predictors, reference fidelity, and end-truncation detectors) and gates
regressions. Every fix above was checked through it before landing.

## Deployment (Docker)

A Linux/CUDA container serves the model behind two surfaces: a RunPod serverless
handler (with a local simulator) and an OpenAI-compatible `POST /v1/audio/speech`,
plus a pre-assignable named-voice registry and output loudness normalization. See
[`deploy/README.md`](deploy/README.md) for the build, the two compose paths, the
environment knobs, and the measured performance and cold-start figures.

## Running the model directly

The original local-usage path still works. With `uv`:

```bash
git clone https://github.com/eoffermann/MisoTTS.git
cd MisoTTS
uv sync --python 3.10
uv run python run_misotts.py   # writes full_conversation.wav
```

Minimal Python:

```python
import torch, torchaudio
from generator import load_miso_8b

gen = load_miso_8b(device="cuda" if torch.cuda.is_available() else "cpu")
audio = gen.generate(text="Hello from Miso.", speaker=0, context=[], max_audio_length_ms=10_000)
torchaudio.save("miso.wav", audio.unsqueeze(0).cpu(), gen.sample_rate)
```

`generate()` also takes a `context` of reference `Segment`s for voice prompting,
and `generate_stream()` yields PCM chunks as frames are produced. See the upstream
README and the code for the full API.

## Repository layout

- `generator.py`, `models.py`, `moshi_compat.py`, `watermarking.py` - the model
  and inference code (from upstream, with our fixes).
- `deploy/` - the serving container, the RunPod and OpenAI surfaces, GPU-sense and
  variant tooling.
- `perf_eval/` - the quality and performance evaluation harness.
- `profile_misotts.py` - the batch + streaming benchmark.
- `CHANGES_FROM_MAIN_REPO.md` - every divergence from upstream, with the problem
  and the commit that addresses it.

## Safety

MisoTTS is a speech-generation model. Do not use it to impersonate people, create
deceptive audio, commit fraud, or generate harmful content. Generated audio is
watermarked by default; if you deploy the model, use your own private watermark
key and keep it secret.

## License and credit

The model and the original inference code are MisoLabs' work, under the upstream
license. This fork preserves that and adds engineering on top. Use of the model
is subject to the upstream license; see
[MisoLabsAI/MisoTTS](https://github.com/MisoLabsAI/MisoTTS).
