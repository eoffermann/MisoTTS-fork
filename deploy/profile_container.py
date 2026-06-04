"""In-container generation-speed profile (compile + flash vs the local baseline).

Loads via the serving core (so MISO_COMPILE applies), warms up to bake compile
graphs, then renders the quick prompt subset twice and reports RTF. Pass 2 is the
steady-state (compiled, warm) number to compare against the Windows baseline.

Run inside the container:
  docker run --rm --gpus all -e MISO_COMPILE=1 \
    -v <hf-cache>:/workspace/hf miso-tts:latest \
    python /opt/miso/deploy/profile_container.py
"""
import os
import sys
import time

sys.path.insert(0, "/opt/miso")
sys.path.insert(0, "/opt/miso/deploy")

import torch

from miso_server import core
from perf_eval.render import render_set
from perf_eval.prompts import QUICK_PROMPTS


def summarize(tag, recs):
    tw = sum(r["wall_s"] for r in recs)
    ta = sum(r["audio_s"] for r in recs)
    print(f"[{tag}] per-clip: " + ", ".join(f"{r['id']}={r['rtf']:.2f}" for r in recs), flush=True)
    print(f"[{tag}] total wall={tw:.1f}s audio={ta:.1f}s  MEAN RTF={tw/ta:.2f}", flush=True)
    return tw / ta if ta else float("nan")


def main():
    print(f"MISO_COMPILE={os.environ.get('MISO_COMPILE')} "
          f"NO_TORCH_COMPILE={os.environ.get('NO_TORCH_COMPILE')}", flush=True)
    t0 = time.perf_counter()
    gen = core.get_generator()      # applies torch.compile when MISO_COMPILE=1
    print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

    # Flash availability (informational).
    try:
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
        q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.bfloat16)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(q, q, q); torch.cuda.synchronize()
        print("[sdpa] FLASH available", flush=True)
    except Exception as e:
        print(f"[sdpa] FLASH unavailable: {e}", flush=True)

    tw = time.perf_counter()
    core.warmup()                   # bake compile graphs (first calls are slow)
    print(f"[warmup] {time.perf_counter()-tw:.1f}s", flush=True)

    r1 = render_set(gen, "/tmp/prof1", prompts=QUICK_PROMPTS, log=lambda *_: None)
    summarize("pass1 (may include recompile)", r1)
    r2 = render_set(gen, "/tmp/prof2", prompts=QUICK_PROMPTS, log=lambda *_: None)
    rtf2 = summarize("pass2 (compiled, warm)", r2)

    print(f"\nBASELINE (Windows, eager, mem-efficient): mean RTF ~13-16", flush=True)
    print(f"CONTAINER pass2 mean RTF: {rtf2:.2f}", flush=True)
    if rtf2 == rtf2:
        print(f"SPEEDUP vs baseline ~14.5: {14.5/rtf2:.2f}x", flush=True)


if __name__ == "__main__":
    main()
