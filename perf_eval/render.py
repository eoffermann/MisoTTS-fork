"""In-process MisoTTS renderer for the eval harness.

Replaces perceval's HTTP `render.py`: audio is produced by calling
`generator.generate()` directly. A fixed per-clip seed makes a given prompt id
reproducible across baseline and candidate runs, so bit-preserving changes
yield (near-)identical audio and the DETERMINISTIC regime is meaningful.
"""
from __future__ import annotations

import os
import time

import torch
import torchaudio

from .prompts import EVAL_PROMPTS

BASE_SEED = 1234
# Canonical per-id seed offset so a given clip uses the SAME seed whether it is
# rendered as part of the full set or a subset (--quick). Without this, seeding
# by enumerate index would desync baseline vs candidate seeds for subsets and
# invalidate the paired comparison.
_CANON_IDX = {p[0]: i for i, p in enumerate(EVAL_PROMPTS)}


def _sync(generator):
    if str(getattr(generator, "device", "cpu")).startswith("cuda"):
        torch.cuda.synchronize()


def render_set(generator, out_dir, prompts=EVAL_PROMPTS, base_seed=BASE_SEED, log=print):
    """Render every prompt to <out_dir>/<id>.wav. Returns per-clip perf records."""
    os.makedirs(out_dir, exist_ok=True)
    sr = generator.sample_rate
    records = []
    for (pid, text, max_ms) in prompts:
        torch.manual_seed(base_seed + _CANON_IDX[pid])  # stable per id across subsets
        _sync(generator)
        t0 = time.perf_counter()
        audio = generator.generate(text=text, speaker=0, context=[],
                                   max_audio_length_ms=max_ms)
        _sync(generator)
        wall = time.perf_counter() - t0
        wav = os.path.join(out_dir, f"{pid}.wav")
        torchaudio.save(wav, audio.unsqueeze(0).cpu(), sr)
        audio_s = audio.shape[0] / sr
        rtf = wall / audio_s if audio_s > 0 else float("inf")
        rec = {"id": pid, "text": text, "max_ms": max_ms, "wav": wav,
               "wall_s": wall, "audio_s": audio_s, "rtf": rtf}
        records.append(rec)
        log(f"  rendered {pid:14} wall={wall:6.2f}s audio={audio_s:5.2f}s "
            f"rtf={rtf:5.2f} -> {os.path.basename(wav)}")
    return records
