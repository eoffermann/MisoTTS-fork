"""Task 6: measure SilentCipher's cost on the first streamed chunk and confirm the
watermark is still detectable when the leading audio is deferred.

(a) Times watermarking the 80 ms first chunk - the cost MISO_WM_DEFER_MS removes
    from the critical first emit.
(b) Generates the stream with wm_defer_ms = 0 and 500, concatenates, and runs the
    SilentCipher detector on the whole clip: defer must still leave the mark
    detectable on the bulk, and defer=0 must keep full coverage.

Eager (MISO_COMPILE=0) - this isolates the watermark cost, not generation speed.
"""
import sys
import time

sys.path.insert(0, "/opt/miso")
sys.path.insert(0, "/opt/miso/deploy")

import torch

from miso_server import core
from perf_eval.prompts import QUICK_PROMPTS
from watermarking import MISO_TTS_WATERMARK, verify


def main():
    gen = core.get_generator()
    sr = gen.sample_rate
    # Use the LONG prompt (most detection signal). Isolate batch-vs-stream and the
    # defer effect.
    pid, text, max_ms = [p for p in QUICK_PROMPTS if p[0] == "long_sad"][0]
    print(f"prompt: {pid} -> \"{text[:48]}...\"", flush=True)

    # (a) cost of watermarking one 80 ms first chunk
    torch.manual_seed(1234)
    first = gen.generate_stream(text=text, speaker=0, context=[],
                                max_audio_length_ms=max_ms, wm_defer_ms=0).__next__().clone()
    t = time.perf_counter()
    try:
        gen._watermark_audio(first); res = "OK"
    except Exception as e:
        res = f"FAILED({type(e).__name__})"
    print(f"first chunk = {first.shape[0] / sr * 1000:.0f} ms; watermark attempt {res} in "
          f"{(time.perf_counter() - t) * 1000:.1f} ms (what MISO_WM_DEFER_MS removes from "
          f"the first emit)", flush=True)

    # (b) detection: batch generate() (whole-clip watermark) vs stream (per-chunk).
    torch.manual_seed(1234)
    batch = gen.generate(text=text, speaker=0, context=[], max_audio_length_ms=max_ms)
    print(f"batch generate() (whole-clip wm): clip={batch.shape[0]/sr:5.2f}s "
          f"detected={verify(gen._watermarker, batch, sr, MISO_TTS_WATERMARK)}", flush=True)
    for defer in (0, 500):
        torch.manual_seed(1234)
        chs = list(gen.generate_stream(text=text, speaker=0, context=[],
                                       max_audio_length_ms=max_ms, wm_defer_ms=defer))
        full = torch.cat([c for c in chs], dim=0)
        print(f"stream wm_defer_ms={defer:4d} (per-chunk wm): clip={full.shape[0]/sr:5.2f}s "
              f"detected={verify(gen._watermarker, full, sr, MISO_TTS_WATERMARK)}", flush=True)
    print("If batch=True but stream=False, per-chunk streaming watermarking is the "
          "(pre-existing) weakness, independent of the defer.", flush=True)


if __name__ == "__main__":
    main()
