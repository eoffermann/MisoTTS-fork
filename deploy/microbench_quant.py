"""Quant revisit: at M=1, does int8/int4 weight-only + torch.compile actually READ
the quantized weight from HBM (a bandwidth win on this memory-bound decode), or
does it materialize bf16 (no win, or a loss)?

Microbenchmark one backbone-shaped Linear (in 4096 -> out 14336, the MLP up-proj,
the largest weight) at M=1 across bf16 / int8 / int4, eager and compiled. Compare
the measured per-call latency to the weight-read floor (weight bytes / HBM
bandwidth): if int8+compile approaches the int8 floor (~half the bf16 floor) it
fuses the dequant and reads int8 (real bandwidth win); if it sits at or above the
bf16 floor it is materializing bf16 (no win). This is the question the prior
"no speed win" finding never tested on the modern compile-capable stack.
"""
import time

import torch
import torch.nn as nn
from torchao.quantization import quantize_, int4_weight_only, int8_weight_only

DEV, DT = "cuda", torch.bfloat16
M, K, N = 1, 4096, 14336   # x:(1,4096)  W:(14336,4096)
BW = 768e9                 # A6000 ~768 GB/s


def mk():
    return nn.Linear(K, N, bias=False).to(DEV).to(DT).eval()


def bench(fn, name, iters=300):
    with torch.no_grad():
        for _ in range(30):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    us = (time.perf_counter() - t) / iters * 1e6
    print(f"  {name:22} {us:7.1f} us/call", flush=True)
    return us


def main():
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
          f"Linear K={K} N={N} M={M}", flush=True)
    x = torch.randn(M, K, device=DEV, dtype=DT)

    print("weight-read floors (weight bytes / bandwidth):", flush=True)
    for name, bpe in [("bf16", 2), ("int8", 1), ("int4", 0.5)]:
        print(f"  {name:6} {K * N * bpe / BW * 1e6:6.1f} us", flush=True)

    print("measured:", flush=True)
    m = mk(); bench(lambda: m(x), "bf16 eager")
    mc = torch.compile(mk()); bench(lambda: mc(x), "bf16 compile")
    mi = mk(); quantize_(mi, int8_weight_only()); bench(lambda: mi(x), "int8_wo eager")
    mic = mk(); quantize_(mic, int8_weight_only()); mic = torch.compile(mic)
    bench(lambda: mic(x), "int8_wo compile")
    m4 = mk(); quantize_(m4, int4_weight_only()); m4 = torch.compile(m4)
    bench(lambda: m4(x), "int4_wo compile")

    print("\nVERDICT: int8_wo compile near the int8 floor (~< bf16 compile) => fused, "
          "reads int8 => bandwidth WIN. At/above the bf16 floor => materializes bf16 "
          "=> no win.", flush=True)


if __name__ == "__main__":
    main()
