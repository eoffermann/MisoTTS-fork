# Changes from the main repo

This fork (`eoffermann/MisoTTS-fork`) tracks divergence from the upstream project
[`MisoLabsAI/MisoTTS`](https://github.com/MisoLabsAI/MisoTTS). It lists every
substantive change we carry that is not (yet) in upstream `main`, the problem it
addresses, and the commit that addresses it. Newest work is at the bottom of each
section.

Baseline: this fork diverges from upstream `main` at commit `82a2874`.

## Bug fixes

### KV caches allocated on CPU while the model is on the GPU
- Problem: torchtune's `KVCache` builds its `k_cache` / `v_cache` / `cache_pos`
  buffers with no device argument, so they default to CPU. `Model.setup_caches`
  ran after the model was moved to the GPU and nothing relocated those buffers,
  so generation aborted at the first `kv_cache.update()` with "Expected all
  tensors to be on the same device, cuda:0 and cpu". The 8B model was unusable on
  CUDA without a patch.
- Fix: build the caches under the model's device context in `setup_caches`.
- Commit: `836cd08`. Also submitted upstream as PR #13.

### Clips cut off at the end (generation caps too small)
- Problem: the model speaks roughly 1.1 words/second on these prompts, far slower
  than the ~2.5 words/second the cap sizing assumed. A 30-word "medium" line
  reaches its natural end-of-speech token near 27 seconds, but the profiler cap
  was 15 seconds, so medium and long clips were truncated mid-sentence.
- Fix: raise `max_audio_length_ms` to 20 / 40 / 50 seconds for short / medium /
  long. Clips that finish early still stop at EOS regardless of the cap.
- Commit: `1291e82`.

### Clips cut off at the end (abrupt EOS decode)
- Problem: `generate()` broke on the first all-zero EOS frame without emitting it,
  so the Mimi decoder never rendered the trailing decay of the last real frame.
  When the model emits EOS abruptly, the audio ended mid-decay at speech energy
  (an audible chop) even when well within the cap.
- Fix: yield the EOS frame so the codec completes the final sound, and do so
  conditionally (re-decode with the EOS frame only when the tail would otherwise
  end chopped), so clips that already trail to silence are not given a faint blip.
- Commits: `ff5aa7e` (include the EOS frame), `c82b19a` (make it conditional).

## Performance

### Skip wasted random-init of the 8B model at load
- Problem: `Model(config)` random-initializes all ~8B parameters (kaiming/normal
  fills over billions of CPU elements) and `load_state_dict` then overwrites every
  one from the checkpoint. The init is pure wasted startup time.
- Fix: patch the `nn.init` weight-fill functions to no-ops during construction
  (`_skip_random_init`). Tensors are still allocated; only the random fill is
  skipped. Strict `load_state_dict` still guarantees every parameter is populated.
- Result: model load 200.9s to 175.1s (about 13% faster), output bit-identical.
- Commit: `d0ab8d5`.

### Remove a per-frame device-to-host sync in the decoder cache reset
- Problem: `generate_frame` reset the decoder caches every frame via
  `decoder.reset_caches()`, whose `KVCache.reset()` does
  `cache_pos -= cache_pos[0].item()`: an `.item()` device-to-host sync per decoder
  layer every frame (thousands per utterance) that serializes the GPU pipeline.
- Fix: rewind `cache_pos` in place with an `arange` copy (no `.item()`). The
  decoder writes all positions each frame before reading them under a causal mask,
  so the zeroing that `reset()` also did is unnecessary.
- Note: wall-clock is flat on the local Windows build (generation is compute-bound
  there with no flash-attention or torch.compile), but this removes the sync
  anti-pattern and helps where per-frame compute is cheaper.
- Commit: `0455a68`.

### Skip the no-op watermark resample
- Problem: after watermarking, `generate()` resampled 24kHz to 24kHz on every call
  (a full polyphase pass producing identical output) because the watermarker
  already returns audio at the model sample rate.
- Fix: guard the resample so it runs only when the rates actually differ.
- Commit: `06a8a5c`.

## Tooling

### perf_eval: quality and performance regression-gating harness
- What: a port of a perceptual TTS-eval harness (perceval) into the project. It
  renders a fixed-seed prompt set through `generate()`, scores each clip with
  perceval (ASR WER/CER, MOS, reference fidelity) plus two custom end-truncation
  detectors, and compares a candidate against a stored baseline with a regression
  gate. Every fix above was validated through this harness before commit.
- Commits: `353c28a` (harness), `5f2ce5d` (gate-tolerance calibration).
- Note: the vendored perceval package and generated audio/reports are gitignored;
  only the harness code under `perf_eval/` is tracked.

### profile_misotts.py: batch + streaming benchmark
- What: a benchmark harness profiling 60 prompts through both `generate()` and
  `generate_stream()`, reporting wall time, RTF, and streaming TTFB, with a
  markdown report. Includes flushed startup logging so long, previously silent
  load phases are visible.
- Commits: `7491c92` (initial), `8b1790b` (streaming + TTFB + logging).

## Serving and deployment

### Containerized serving (RunPod + OpenAI-compatible APIs)
- What: a Linux/CUDA Docker image and a `miso_server` package that runs MisoTTS as
  a service locally or on RunPod. Linux unlocks triton/Inductor (torch.compile,
  CUDA graphs) and flash-attention SDPA, which the Windows dev box cannot use, so
  this is where generation throughput actually improves (set `MISO_COMPILE=1`).
- Surfaces: a RunPod serverless handler, a local RunPod simulator (`/runsync`
  buffered like RunPod, plus `/stream_now` for local streaming since `/runsync`
  cannot stream), and an OpenAI-compatible `POST /v1/audio/speech`. Two compose
  paths: `docker-compose.runpod.yml` and `docker-compose.openai.yml`.
- Voices: a named voice registry (pre-assigned reference wav + transcript,
  Mimi-encoded once and reused). Built-in `default` voice always present.
- See `deploy/README.md`.

### Container compile speedup (the big throughput win)
- With `MISO_COMPILE=1` (torch.compile reduce-overhead + flash SDPA) the container
  reaches ~1.2 mean RTF vs ~14.5 on the Windows eager baseline: about a **12x
  speedup** (near realtime on an A6000). Confirms the workload was
  launch/dispatch-bound; CUDA graphs recover almost all of it. The compile warmup
  is a one-time ~20 min cost, persisted via a per-SM TorchInductor cache volume.
- This needs the Linux `devel` CUDA base plus build-essential and python3-dev
  (triton JIT-compiles a Python.h-including helper); the runtime base cannot do it.

### GPU-sense model variants
- `core.detect_device_profile()` originally selected a variant by GPU arch
  (Blackwell -> nvfp4, Hopper/Ada -> fp8, Ampere -> bf16). This was REWORKED to a
  VRAM-based selection (bf16 / int8 / int4) once measurement showed the
  frame-by-frame decode cannot feed the hardware low-precision GEMMs, so fp8/nvfp4
  give no speed benefit for this model (see "Stack modernization and quantized
  variants" below for the current design). The unvalidated nvfp4 scaffold
  (`deploy/quantize_nvfp4.py`, `deploy/Dockerfile.blackwell`) was dropped.

### Output loudness normalization
- Problem: the raw model output has a ~19 dB clip-to-clip integrated-loudness
  (LUFS) spread with no correlation to emotion (two "normal" clips ~18 dB apart),
  and several clips near 0 dBFS. Output volume is unpredictable.
- Fix: the serving core normalizes each clip to a target LUFS with a peak ceiling
  (`MISO_TARGET_LUFS`, default -16). `perf_eval/amplitude.py` measures the spread.
- Note: applied in the serving core (the deployable surface); raw `generate()` is
  unchanged. Exposing it to the local profiler is a possible follow-up.

## Features carried from an open upstream PR

### Streaming generation API (`generate_stream`)
- Source: upstream PR #7 (`feat/streaming-generation`), still open upstream at the
  time we merged it. Adds chunked audio decode with CUDA-graph-safe uniform chunk
  sizing inside Mimi streaming mode.
- Commits: `fe34dea`, `4667c22`, `7f76f25`, merged at `f2a0e16`.

## Docs

### Windows PowerShell examples in the Quickstart
- What: added PowerShell command examples alongside the existing shell examples.
- Commit: `ae00c82`.

## Stack modernization and quantized variants

### Modernized the torch stack to 2.7.1 + cu128
- Problem: the upstream pins (torch 2.4.0, torchao 0.9.0, torchtune 0.4.0, moshi
  0.2.2) could not run quantization together with `torch.compile` - every scheme
  crashed in FakeTensor tracing - and cu124 could not target Blackwell. The model
  also could not be matched to smaller GPUs.
- Fix: bump to torch 2.7.1+cu128, torchaudio 2.7.1, torchao 0.13.0 (its official
  torch-2.7.1 pairing), torchtune 0.6.1, moshi 0.2.13. torchtune 0.6.1 is
  API-identical to 0.4.0 for our use (the 8B checkpoint loads strict, 367/367
  keys), so models.py is unchanged; only moshi_compat.py needed a guard for moshi
  0.2.13's reworked quantize module. Dropped the numpy<2 ABI hack (torch 2.7 is
  numpy-2 clean) and bitsandbytes (never imported). Verified on an A6000: int8 +
  compile now runs, compiled bf16 holds RTF ~1.11-1.13, and the compile warmup
  caches across processes (~68% faster second cold start - impossible on 2.4).
- Commits: `1fa5513`, `44aa97f`, `c3e9385`, `d9b82ed`.

### Memory-matched quantized variants (int8 / int4), GPU-sense by VRAM
- Problem: an 8B bf16 model needs ~20 GB at generation peak, classing out 12-16 GB
  cards entirely. Quantization could lower that floor, but only if it actually
  preserves quality and is wired to be selected automatically.
- Fix: the serving core reads VRAM and loads the highest-quality precision that
  fits - bf16 (>=22 GB), int8 (>=13 GB, ~16 GB cards), int4 (below, ~12 GB cards).
  int8/int4 are weight-only torchao quants of the backbone/decoder Linears
  (embeddings/heads stay bf16), pulled as pre-quantized checkpoints from
  `BigBlueCeiling/MisoTTS-int8` and `-int4`, with a fall-back that quantizes the
  bf16 weights at load (layer-wise, so a small card never holds the full bf16
  model) if a repo is missing or its packed layout does not load on the GPU.
- Quality (perceval, 12 canonical prompts): int8 is quality-neutral (mean CER/WER/
  UTMOS even with bf16); int4 is experimental and audibly degraded (UTMOS ~2.9 vs
  3.9, worse on long lines). Documented as such.
- Note: quantization here is a MEMORY lever, not a speed one - the frame-by-frame
  decode's M=1 matmuls cannot feed the hardware low-precision GEMMs (int8 `_int_mm`,
  fp8/fp4 `_scaled_mm` need M>=16), so weight-only int8/int4 dequantize to bf16 for
  the matmul. That is why no hardware-fp8 or Blackwell-nvfp4 variant is shipped.
