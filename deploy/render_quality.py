"""Render the canonical 9-prompt eval set at a given weight precision, so the
WAVs can be quality-scored with perceval on the host.

One quant level per process (env MISO_QLEVEL in {bf16,int8,int4}). Uses the
canonical prompts and per-id seeds from perf_eval (NOT ad-hoc sentences), so the
transcripts are known/verifiable and renders are reproducible and paired across
levels. Writes <out>/<level>/<id>.wav plus a _render.json with load time, GPU
resident/peak VRAM, and per-clip RTF.

  docker run --rm --gpus all -v <repo>:/opt/miso -v <hf>:/workspace/hf \
    -v <out>:/workspace/out -e MISO_QLEVEL=int8 \
    miso-tts:modern python /opt/miso/deploy/render_quality.py
"""
import json
import os
import sys
import time

sys.path.insert(0, "/opt/miso")
import torch

from generator import load_miso_8b
from perf_eval.prompts import EVAL_PROMPTS, QUICK_PROMPTS
from perf_eval.render import render_set

LEVEL = os.environ.get("MISO_QLEVEL", "bf16")
COMPILE = os.environ.get("MISO_COMPILE", "0") == "1"
PROMPTS = QUICK_PROMPTS if os.environ.get("MISO_PROMPT_SET") == "quick" else EVAL_PROMPTS
OUT = os.environ.get("MISO_OUT_DIR", "/workspace/out/quality")
out_dir = os.path.join(OUT, LEVEL + ("_compile" if COMPILE else ""))
os.makedirs(out_dir, exist_ok=True)


def log(m):
    print(f"[render:{LEVEL}{'+c' if COMPILE else ''}] {m}", flush=True)


def main():
    quant = None if LEVEL == "bf16" else LEVEL
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    gen = load_miso_8b("cuda", dtype=torch.bfloat16, quantize=quant)
    load_s = time.perf_counter() - t0
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated() / 1e9
    log(f"loaded in {load_s:.0f}s | resident {resident:.2f} GB")

    if COMPILE:
        mode = os.environ.get("MISO_COMPILE_MODE", "reduce-overhead")
        gen._model.backbone = torch.compile(gen._model.backbone, mode=mode)
        gen._model.decoder = torch.compile(gen._model.decoder, mode=mode)
        log(f"torch.compile enabled (mode={mode})")

    torch.cuda.reset_peak_memory_stats()
    records = render_set(gen, out_dir, prompts=PROMPTS, log=log)
    gen_peak = torch.cuda.max_memory_allocated() / 1e9

    summary = {"level": LEVEL, "load_s": round(load_s, 1),
               "resident_gb": round(resident, 2), "gen_peak_gb": round(gen_peak, 2),
               "rtf_mean": round(sum(r["rtf"] for r in records) / len(records), 2),
               "records": records}
    with open(os.path.join(out_dir, "_render.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"DONE 9 clips -> {out_dir} | resident {resident:.2f}GB gen_peak {gen_peak:.2f}GB "
        f"rtf_mean {summary['rtf_mean']}")


if __name__ == "__main__":
    main()
