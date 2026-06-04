"""Render the quick subset with GREEDY decoding (top-k=1) for a deterministic
compile-vs-eager parity check.

With top-k=1 sampling is deterministic (always the argmax), so greedy-compiled
and greedy-eager should produce near-identical audio IF compile is numerically
sound. Run this twice (MISO_COMPILE=1 -> one dir, MISO_COMPILE=0 -> another) and
compare the two dirs with perceval in the DETERMINISTIC regime: low reference
distance == compile did not change the model's computation, so the stochastic-
mode quality differences are just sampling variance.

  docker run ... -e MISO_COMPILE=1 -e MISO_OUT_DIR=/workspace/out miso-tts:latest \
    python /opt/miso/deploy/render_greedy.py
"""
import os
import sys

sys.path.insert(0, "/opt/miso")
sys.path.insert(0, "/opt/miso/deploy")

import torch
import torchaudio

from miso_server import core
from perf_eval.prompts import QUICK_PROMPTS
from perf_eval.render import BASE_SEED, _CANON_IDX

OUT = os.environ.get("MISO_OUT_DIR", "/workspace/out")


def main():
    os.makedirs(OUT, exist_ok=True)
    gen = core.get_generator()
    core.warmup()
    sr = gen.sample_rate
    for pid, text, max_ms in QUICK_PROMPTS:
        torch.manual_seed(BASE_SEED + _CANON_IDX[pid])
        # top-k=1 -> greedy/deterministic regardless of temperature.
        audio = gen.generate(text=text, speaker=0, context=[],
                             max_audio_length_ms=max_ms, temperature=0.9, topk=1)
        torchaudio.save(f"{OUT}/{pid}.wav", audio.unsqueeze(0).cpu(), sr)
        print(f"[greedy] {pid:14} dur={audio.shape[0]/sr:.2f}s", flush=True)


if __name__ == "__main__":
    main()
