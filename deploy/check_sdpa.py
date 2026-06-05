"""Task 5: which SDPA backend does torch pick for MisoTTS's backbone attention,
and how big is attention vs the per-frame budget?

The backbone is GQA: 32 query heads, 8 KV heads, head_dim 128. At batch-1 decode
the query is one token against the cached KV; prefill runs the whole prompt. If
torch's FLASH backend is available for these shapes and the attention time is a
tiny fraction of the ~80-120 ms per-frame budget, the dedicated flash-attn package
buys nothing over torch SDPA flash.
"""
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

DEV, DT = "cuda", torch.bfloat16


def mk(lq, lkv):
    q = torch.randn(1, 32, lq, 128, device=DEV, dtype=DT)
    k = torch.randn(1, 8, lkv, 128, device=DEV, dtype=DT)
    v = torch.randn(1, 8, lkv, 128, device=DEV, dtype=DT)
    return q, k, v


def main():
    print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0), flush=True)
    cases = [("decode seq=1, kv=512", 1, 512),
             ("decode seq=1, kv=2048", 1, 2048),
             ("prefill L=55", 55, 55)]
    for name, lq, lkv in cases:
        q, k, v = mk(lq, lkv)
        causal = lq > 1
        avail = {}
        for be, bn in [(SDPBackend.FLASH_ATTENTION, "flash"),
                       (SDPBackend.EFFICIENT_ATTENTION, "mem_eff"),
                       (SDPBackend.MATH, "math")]:
            try:
                with sdpa_kernel(be):
                    F.scaled_dot_product_attention(q, k, v, enable_gqa=True, is_causal=causal)
                    torch.cuda.synchronize()
                avail[bn] = "ok"
            except Exception as e:
                avail[bn] = f"no({type(e).__name__})"
        # default (auto-selected) latency
        for _ in range(10):
            F.scaled_dot_product_attention(q, k, v, enable_gqa=True, is_causal=causal)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(100):
            F.scaled_dot_product_attention(q, k, v, enable_gqa=True, is_causal=causal)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 100 * 1000
        print(f"  {name:24} backends={avail} default={ms:.4f} ms/call", flush=True)
    print("\nPer-frame budget is ~80-120 ms (one backbone forward + 31 decoder steps). "
          "If decode attention is <<1 ms, flash-attn vs torch SDPA flash is irrelevant.",
          flush=True)


if __name__ == "__main__":
    main()
