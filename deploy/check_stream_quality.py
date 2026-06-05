"""Task 2 quality gate: the streamed-and-concatenated waveform must match batch
generate() for the same prompt and seed (no truncation, no chop, level stable).

Same seed -> identical frames; only the decode granularity (per-frame streaming
vs batch) and per-chunk watermarking differ, so the content should match closely.
Run eager (MISO_COMPILE=0) for speed.
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


def tail_rms(x, sr=24000):
    return x[-int(0.12 * sr):].pow(2).mean().sqrt().item() if x.shape[0] > sr // 8 else 0.0


def main():
    gen = core.get_generator()
    sr = gen.sample_rate
    log("comparing batch generate() vs concatenated generate_stream() (ramp default)")
    for pid, text, max_ms in QUICK_PROMPTS:
        torch.manual_seed(1234)
        batch = gen.generate(text=text, speaker=0, context=[], max_audio_length_ms=max_ms).float().cpu()
        torch.manual_seed(1234)
        chunks = list(gen.generate_stream(text=text, speaker=0, context=[], max_audio_length_ms=max_ms))
        stream = torch.cat([c.float().cpu() for c in chunks], dim=0) if chunks else torch.zeros(0)
        n = min(batch.shape[0], stream.shape[0])
        d = (batch[:n] - stream[:n]).abs()
        rms = batch[:n].pow(2).mean().sqrt().item()
        len_ratio = stream.shape[0] / batch.shape[0] if batch.shape[0] else 0.0
        log(f"{pid:14} batch={batch.shape[0]:6d} stream={stream.shape[0]:6d} "
            f"len_ratio={len_ratio:.4f} n_chunks={len(chunks):3d} "
            f"maxdiff={d.max().item():.3e} rel={d.mean().item()/(rms+1e-9):.2e} "
            f"batch_tail={tail_rms(batch):.3e} stream_tail={tail_rms(stream):.3e}")
    log("PASS if len_ratio ~1.0 (no truncation), rel small (content matches), and "
        "stream_tail ~ batch_tail (no chop / no extra blip).")


if __name__ == "__main__":
    main()
