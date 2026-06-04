"""Compiled-model quality + streaming-TTFB profile (in container).

Loads via the serving core (compiled when MISO_COMPILE=1), warms up, then:
  1. Batch-renders the eval set with baseline-matched seeds to /workspace/out so
     the WAVs can be pulled and scored with perceval on the host (quality).
  2. Measures streaming time-to-first-byte (TTFB) per clip via generate_stream.

Reports batch RTF, the load/warmup cold-start breakdown, and TTFB stats. Mount a
persistent inductor cache volume so the compile warmup is one-time, and a host
dir at /workspace/out to retrieve the WAVs.
"""
import os
import statistics
import sys
import time

sys.path.insert(0, "/opt/miso")
sys.path.insert(0, "/opt/miso/deploy")

import torch
import torchaudio

from miso_server import core
from perf_eval.prompts import EVAL_PROMPTS
from perf_eval.render import BASE_SEED, _CANON_IDX

OUT = os.environ.get("MISO_OUT_DIR", "/workspace/out")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"MISO_COMPILE={os.environ.get('MISO_COMPILE')} "
          f"MODE={os.environ.get('MISO_COMPILE_MODE')}", flush=True)
    t0 = time.perf_counter()
    gen = core.get_generator()
    t_load = time.perf_counter() - t0
    print(f"[load] {t_load:.1f}s", flush=True)
    tw = time.perf_counter()
    core.warmup()
    t_warm = time.perf_counter() - tw
    print(f"[warmup] {t_warm:.1f}s", flush=True)
    print(f"[cold-start] load={t_load:.1f}s warmup={t_warm:.1f}s "
          f"total={t_load + t_warm:.1f}s", flush=True)
    sr = gen.sample_rate

    # 1. Batch render for quality (baseline-matched seeds) + RTF.
    batch = []
    for pid, text, max_ms in EVAL_PROMPTS:
        torch.manual_seed(BASE_SEED + _CANON_IDX[pid])
        torch.cuda.synchronize()
        s = time.perf_counter()
        audio = gen.generate(text=text, speaker=0, context=[], max_audio_length_ms=max_ms)
        torch.cuda.synchronize()
        wall = time.perf_counter() - s
        torchaudio.save(f"{OUT}/{pid}.wav", audio.unsqueeze(0).cpu(), sr)
        aud = audio.shape[0] / sr
        batch.append((pid, wall, aud, wall / aud if aud else 0))
        print(f"[batch] {pid:14} wall={wall:6.2f}s audio={aud:5.2f}s rtf={wall/aud:5.2f}", flush=True)

    # 2. Streaming TTFB per clip.
    print("--- streaming TTFB ---", flush=True)
    stream = []
    for pid, text, max_ms in EVAL_PROMPTS:
        torch.manual_seed(BASE_SEED + _CANON_IDX[pid])
        torch.cuda.synchronize()
        s = time.perf_counter()
        ttfb = None
        n = 0
        nsamp = 0
        try:
            for chunk in gen.generate_stream(text=text, speaker=0, context=[],
                                             max_audio_length_ms=max_ms, chunk_frames=25):
                if ttfb is None:
                    torch.cuda.synchronize()
                    ttfb = time.perf_counter() - s
                n += 1
                nsamp += chunk.shape[0]
            torch.cuda.synchronize()
            total = time.perf_counter() - s
            aud = nsamp / sr
            stream.append((pid, ttfb, total, aud, n))
            print(f"[stream] {pid:14} TTFB={ttfb*1000:6.0f}ms total={total:6.2f}s "
                  f"audio={aud:5.2f}s chunks={n}", flush=True)
        except Exception as exc:
            print(f"[stream] {pid:14} FAILED: {type(exc).__name__}: {exc}", flush=True)

    bw = sum(r[1] for r in batch); ba = sum(r[2] for r in batch)
    print(f"\nBATCH: total wall={bw:.1f}s audio={ba:.1f}s MEAN RTF={bw/ba:.2f}", flush=True)
    ttfbs = [r[1] for r in stream if r[1]]
    if ttfbs:
        print(f"STREAMING TTFB: mean={1000*statistics.mean(ttfbs):.0f}ms "
              f"min={1000*min(ttfbs):.0f}ms max={1000*max(ttfbs):.0f}ms "
              f"(n={len(ttfbs)})", flush=True)


if __name__ == "__main__":
    main()
