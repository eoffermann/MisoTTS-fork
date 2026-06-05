"""Measure on-device VRAM for bf16 vs int8 weight-only, to answer 'what card fits?'.

Reports torch.cuda allocated (live tensors) and reserved (what the caching
allocator holds ~ closer to nvidia-smi) at three points:
  1. after load_miso_8b (bf16): LM + Mimi + watermarker + KV caches, all on GPU,
  2. after int8 weight-only quantize_ (backbone+decoder Linears -> int8),
  3. peak during one eager generation (the real 'must fit' number).

Run (repo bind-mounted, no rebuild) with the HF cache and repo mounted, e.g.
  docker run --rm --gpus all -v <repo>:/opt/miso -v <hf_cache>:/workspace/hf
    miso-tts:modern python /opt/miso/deploy/measure_vram.py
"""
import sys, time
sys.path.insert(0, "/opt/miso")
import torch


def gb(n):
    return n / 1e9


def snap(tag):
    a = torch.cuda.memory_allocated()
    r = torch.cuda.memory_reserved()
    print(f"[vram] {tag:32s} allocated={gb(a):6.2f} GB  reserved={gb(r):6.2f} GB", flush=True)
    return a, r


def main():
    from generator import load_miso_8b
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    gen = load_miso_8b("cuda", dtype=torch.bfloat16)
    print(f"[vram] loaded bf16 in {time.perf_counter()-t0:.0f}s", flush=True)
    snap("bf16 resident (LM+Mimi+wm+KV)")

    # int8 weight-only on backbone+decoder Linears (mirror core._apply_quant).
    import torch.nn as nn
    from torchao.quantization import quantize_, int8_weight_only
    quantize_(gen._model, int8_weight_only(),
              filter_fn=lambda m, fqn: isinstance(m, nn.Linear) and "head" not in fqn and "projection" not in fqn)
    torch.cuda.empty_cache()
    snap("int8 resident (after quantize)")

    # Peak during a real generation (eager, the minimal-footprint path).
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    _ = gen.generate(text="Measuring peak memory during a normal length generation.",
                     speaker=0, context=[], max_audio_length_ms=12_000)
    peak = torch.cuda.max_memory_allocated()
    print(f"[vram] int8 PEAK during eager generation   allocated={gb(peak):6.2f} GB", flush=True)
    print(f"[vram] int8 reserved after generation      reserved ={gb(torch.cuda.memory_reserved()):6.2f} GB", flush=True)


if __name__ == "__main__":
    main()
