"""Opportunity #2 correctness gate: the threaded emit-overlap pipeline
(MISO_STREAM_PIPELINE=1) must produce the SAME streamed audio as the synchronous
path (=0) for the same seed.

Generation runs on the main thread in BOTH modes, so the seed -> frames mapping is
identical; only WHERE decode/watermark/copy run differs (worker + side stream vs
inline). The concatenated outputs should therefore be ~bit-identical. A non-trivial
diff means a stream/event race or an ordering/trim bug in the worker.
"""
import os
import sys

sys.path.insert(0, "/opt/miso")
sys.path.insert(0, "/opt/miso/deploy")

import torch

from miso_server import core
from perf_eval.prompts import QUICK_PROMPTS


def _stream(gen, text, max_ms):
    return torch.cat([c.float().cpu() for c in gen.generate_stream(
        text=text, speaker=0, context=[], max_audio_length_ms=max_ms)], dim=0)


def main():
    gen = core.get_generator()
    print("comparing MISO_STREAM_PIPELINE=0 (synchronous) vs =1 (worker pipeline)", flush=True)
    for pid, text, max_ms in QUICK_PROMPTS:
        os.environ["MISO_STREAM_PIPELINE"] = "0"
        torch.manual_seed(1234)
        sync = _stream(gen, text, max_ms)
        os.environ["MISO_STREAM_PIPELINE"] = "1"
        torch.manual_seed(1234)
        pipe = _stream(gen, text, max_ms)
        n = min(sync.shape[0], pipe.shape[0])
        d = (sync[:n] - pipe[:n]).abs()
        same_len = sync.shape[0] == pipe.shape[0]
        print(f"  {pid:14} sync={sync.shape[0]:6d} pipe={pipe.shape[0]:6d} same_len={same_len} "
              f"maxdiff={d.max().item():.2e} meandiff={d.mean().item():.2e}", flush=True)
    print("PASS if same_len=True and maxdiff ~0 (identical audio, no race/ordering bug).",
          flush=True)


if __name__ == "__main__":
    main()
