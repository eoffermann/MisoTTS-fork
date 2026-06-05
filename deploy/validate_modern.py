"""In-container validation for the modern torch 2.7.1 + torchao 0.13.0 stack.

Runs ONE configuration per process (selected by MISO_VALIDATE) so each result is
a clean cold-start measurement, and appends a JSON line to MISO_RESULT_JSON.

Configs (MISO_VALIDATE):
  eager          bf16, no compile, no quant            (throughput floor / sanity)
  compile        bf16 + torch.compile(reduce-overhead) (graph path)
  int8           int8 weight-only, no compile          (quant alone)
  int8_compile   int8 weight-only + torch.compile      (THE unlock: broken on 2.4)

What it proves vs measures:
  - strict checkpoint load on the modern stack (367/367 keys),
  - quant+compile actually traces+runs on GPU (the 2.4/0.9 FakeTensor blocker),
  - load time, compile warmup (first generate), steady RTF, streaming TTFB.

Mirrors deploy/miso_server/core.py _apply_quant / _maybe_compile exactly so it
validates the real serving path, not a parallel implementation.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/opt/miso")
sys.path.insert(0, "/opt/miso/deploy")

import torch

CONFIG = os.environ.get("MISO_VALIDATE", "eager")
RESULT_JSON = os.environ.get("MISO_RESULT_JSON", "/workspace/out/validate_modern.jsonl")
OUT_DIR = os.environ.get("MISO_OUT_DIR", "/workspace/out")
os.makedirs(OUT_DIR, exist_ok=True)

# Fixed prompts: short / medium / long. Seed fixed for run-to-run comparability.
PROMPTS = [
    "Hello from Miso.",
    "The modern stack should make quantized generation both correct and fast.",
    "When the pipeline runs end to end, the backbone produces audio codes frame "
    "by frame, and the decoder predicts the higher order codebooks for each one, "
    "so the throughput depends on how quickly that autoregressive loop turns.",
]


def log(msg):
    print(f"[validate:{CONFIG}] {msg}", flush=True)


def _filt(mod, fqn):
    import torch.nn as nn
    return isinstance(mod, nn.Linear) and "head" not in fqn and "projection" not in fqn


def apply_int8(gen):
    """int8 WEIGHT-only (mirror core._apply_quant): weights int8, matmul in bf16.
    Saves weight bandwidth but does NOT use Ampere INT8 tensor cores."""
    from torchao.quantization import quantize_, int8_weight_only
    quantize_(gen._model, int8_weight_only(), filter_fn=_filt)
    log("applied int8 weight-only quantization")


def apply_int8_dynamic(gen):
    """int8 DYNAMIC (W8A8): activations AND weights int8 -> int8xint8 matmul on the
    A6000's INT8 tensor cores. The scheme that can actually beat bf16 on Ampere."""
    from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
    quantize_(gen._model, int8_dynamic_activation_int8_weight(), filter_fn=_filt)
    log("applied int8 dynamic (W8A8) quantization")


def apply_compile(gen):
    """Mirror core._maybe_compile (reduce-overhead, backbone + decoder)."""
    mode = os.environ.get("MISO_COMPILE_MODE", "reduce-overhead")
    gen._model.backbone = torch.compile(gen._model.backbone, mode=mode)
    gen._model.decoder = torch.compile(gen._model.decoder, mode=mode)
    log(f"torch.compile enabled (mode={mode})")


def main():
    result = {"config": CONFIG, "tag": os.environ.get("MISO_TAG", ""),
              "torch": torch.__version__}
    import torchao
    result["torchao"] = torchao.__version__
    log(f"torch={torch.__version__} torchao={torchao.__version__}")

    # Device profile (confirms arch / cc).
    from miso_server.core import detect_device_profile
    prof = detect_device_profile()
    result["device"] = prof
    log(f"device: {prof['name']} cc={prof['cc']} arch={prof['arch']} vram={prof['vram_gb']:.0f}GB")

    # ---- load (strict checkpoint) ----
    from generator import load_miso_8b
    torch.manual_seed(1234)
    t0 = time.perf_counter()
    gen = load_miso_8b("cuda", dtype=torch.bfloat16)
    load_s = time.perf_counter() - t0
    result["load_s"] = round(load_s, 1)
    log(f"strict load OK in {load_s:.1f}s")

    if CONFIG in ("int8", "int8_compile"):
        apply_int8(gen)
    if CONFIG in ("int8dyn", "int8dyn_compile"):
        apply_int8_dynamic(gen)
    if CONFIG in ("compile", "int8_compile", "int8dyn_compile"):
        apply_compile(gen)

    torch.cuda.synchronize()

    # ---- batch generate: first call = warmup (compile), rest = steady ----
    gens = []
    for i, text in enumerate(PROMPTS):
        torch.manual_seed(1234 + i)
        torch.cuda.synchronize()
        t = time.perf_counter()
        audio = gen.generate(text=text, speaker=0, context=[], max_audio_length_ms=20_000)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        dur = audio.shape[-1] / gen.sample_rate
        rtf = wall / dur if dur > 0 else float("nan")
        gens.append({"i": i, "wall_s": round(wall, 2), "audio_s": round(dur, 2),
                     "rtf": round(rtf, 2), "samples": int(audio.shape[-1])})
        log(f"gen[{i}] wall={wall:.2f}s audio={dur:.2f}s RTF={rtf:.2f}")
        import torchaudio
        torchaudio.save(os.path.join(OUT_DIR, f"{CONFIG}_{i}.wav"),
                        audio.unsqueeze(0).cpu(), gen.sample_rate)

    result["gens"] = gens
    result["warmup_s"] = gens[0]["wall_s"] if gens else None
    steady = [g["rtf"] for g in gens[1:]] or [gens[0]["rtf"]]
    result["steady_rtf_mean"] = round(sum(steady) / len(steady), 2)

    # ---- streaming TTFB (time to first chunk) ----
    try:
        torch.manual_seed(4321)
        torch.cuda.synchronize()
        t = time.perf_counter()
        first = None
        n_chunks = 0
        for chunk in gen.generate_stream(text=PROMPTS[2], speaker=0, context=[],
                                          max_audio_length_ms=20_000, chunk_frames=25):
            n_chunks += 1
            if first is None:
                first = time.perf_counter() - t
        result["stream_ttfb_s"] = round(first, 2) if first else None
        result["stream_chunks"] = n_chunks
        log(f"stream TTFB={first:.2f}s over {n_chunks} chunks")
    except Exception as exc:
        result["stream_error"] = repr(exc)
        log(f"streaming failed: {exc!r}")

    with open(RESULT_JSON, "a") as f:
        f.write(json.dumps(result) + "\n")
    log(f"DONE. steady RTF mean={result['steady_rtf_mean']} "
        f"warmup={result['warmup_s']}s load={result['load_s']}s")


if __name__ == "__main__":
    main()
