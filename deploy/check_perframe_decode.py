"""Verify the Task 2 assumption: Mimi streaming decode of ONE frame at a time is
numerically equivalent to decoding in 25-frame chunks and to a batch decode.

If per-frame decode matches, Task 2 can decode at shape 1 (stable CUDA graph) and
ramp only the EMIT cadence. If it diverges (Mimi lookahead), Task 2 needs plan B
(one graph per ramp size). Run eager (MISO_COMPILE=0) - decode is identical
regardless of how frames were produced, and this avoids the compile warmup.

  docker run --rm --gpus all -e MISO_COMPILE=0 -v <hf>:/workspace/hf \
    miso-tts:latest python /opt/miso/deploy/check_perframe_decode.py
"""
import sys
import time

sys.path.insert(0, "/opt/miso")
sys.path.insert(0, "/opt/miso/deploy")

import torch

from miso_server import core
from perf_eval.prompts import QUICK_PROMPTS

_T0 = time.perf_counter()


def log(m):
    print(f"[{time.perf_counter()-_T0:6.1f}s] {m}", flush=True)


def stat(name, a, ref):
    n = min(a.shape[0], ref.shape[0])
    d = (a[:n] - ref[:n]).abs()
    rms = ref[:n].pow(2).mean().sqrt().item()
    log(f"{name}: len={a.shape[0]} (ref={ref.shape[0]}) maxdiff={d.max().item():.3e} "
        f"meandiff={d.mean().item():.3e} ref_rms={rms:.3e} "
        f"rel={d.mean().item()/(rms+1e-9):.2e}")


def main():
    log("loading (eager)...")
    gen = core.get_generator()
    pid, text, max_ms = QUICK_PROMPTS[0]
    log(f"prompt: {pid} -> \"{text[:50]}...\"")

    torch.manual_seed(1234)
    with torch.inference_mode():
        pt, ptm, mgl = gen._prepare_prompt(text, 0, [], max_ms)
        frames = list(gen._generate_frames(pt, ptm, mgl, 0.9, 50))
    log(f"generated {len(frames)} frames")

    with torch.inference_mode():
        batch = gen._decode_frames(frames).float().cpu()
    with torch.inference_mode(), gen._audio_tokenizer.streaming(1):
        perframe = torch.cat([gen._decode_frames([f]) for f in frames], dim=0).float().cpu()
    with torch.inference_mode(), gen._audio_tokenizer.streaming(1):
        parts = []
        i = 0
        while i + 25 <= len(frames):
            parts.append(gen._decode_frames(frames[i:i + 25]))
            i += 25
        chunk25 = (torch.cat(parts, dim=0).float().cpu() if parts else torch.zeros(0))

    log("--- decode comparisons (ref = batch decode) ---")
    stat("perframe-vs-batch ", perframe, batch)
    stat("chunk25-vs-batch  ", chunk25, batch)
    stat("perframe-vs-chunk25", perframe, chunk25)
    # verdict heuristic: per-frame is "equivalent" if its relative diff to batch is
    # no worse than the current 25-chunk method's relative diff to batch.
    log("If perframe-vs-batch rel is comparable to chunk25-vs-batch rel, per-frame "
        "decode is safe for Task 2.")


if __name__ == "__main__":
    main()
