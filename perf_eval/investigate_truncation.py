"""Truncation root-cause probe (single model load, many hypotheses).

Baseline showed acoustic tail-chopping on ~4/9 clips. Two suspected modes:
  (A) cap-bound: audio length == max_audio_length_ms cap (EOS never fired in time)
  (B) EOS-fires-but-tail-chopped: short clip, EOS fired, yet the waveform ends at
      speech energy (e.g. short_sad: 3.44 s, tail/speech 0.89).

For each suspect prompt we reproduce the baseline frames (same seed) but with a
GENEROUS cap, record where EOS first fires, then decode several tail variants and
measure tail energy:
  A  frames[:eos]        current behavior (EOS frame dropped)
  B  frames[:eos+1]      include the all-zero EOS frame in the decode
  Ck frames[:eos+k]      keep k post-EOS frames (does the codec/model emit decay?)
This tells us whether the chop is the codec needing trailing frames (B/Ck fix it)
or the cap (A: EOS fires later than the cap), and by how much.
"""
import os

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["NO_TORCH_COMPILE"] = "1"

import time
import numpy as np
import torch
import torchaudio

from generator import load_miso_8b
from perf_eval.prompts import EVAL_PROMPTS

SR = 24_000
FRAME_S = 0.08  # 80 ms per frame (12.5 Hz)
SUSPECTS = ["short_sad", "medium_normal", "long_sad"]  # one per length
GENEROUS_CAP_MS = 30_000
EXTRA_AFTER_EOS = 24  # ~1.9 s of post-EOS frames to inspect


def _idx_of(pid):
    return [i for i, (p, _, _) in enumerate(EVAL_PROMPTS) if p == pid][0]


def tail_ratio(audio_t):
    a = audio_t.detach().float().cpu().numpy()
    n = a.shape[0]
    if n < int(0.25 * SR):
        return 0.0, n / SR
    fw = int(0.02 * SR)
    nf = n // fw
    rms = np.sqrt((a[:nf * fw].reshape(nf, fw) ** 2).mean(axis=1) + 1e-12)
    speech = float(np.percentile(rms, 90))
    tail = float(np.sqrt((a[-int(0.12 * SR):] ** 2).mean() + 1e-12))
    return (tail / speech if speech > 0 else 0.0), n / SR


def collect_frames(gen, text, seed, max_ms, extra_after_eos):
    """Replicate generate()'s frame loop but collect ALL frames + the EOS index."""
    torch.manual_seed(seed)
    pt, ptm, max_gen = gen._prepare_prompt(text, 0, [], max_ms)
    curr = pt.unsqueeze(0)
    curr_mask = ptm.unsqueeze(0)
    curr_pos = torch.arange(0, pt.size(0)).unsqueeze(0).long().to(gen.device)
    frames = []
    eos_idx = None
    with torch.inference_mode():
        for _ in range(max_gen):
            sample = gen._model.generate_frame(curr, curr_mask, curr_pos, 0.9, 50)
            if eos_idx is None and torch.all(sample == 0):
                eos_idx = len(frames)  # EOS would break here (frame NOT appended)
            frames.append(sample)
            curr = torch.cat([sample, torch.zeros(1, 1).long().to(gen.device)], dim=1).unsqueeze(1)
            curr_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(gen.device)], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1
            if eos_idx is not None and len(frames) >= eos_idx + extra_after_eos:
                break
    return frames, eos_idx, max_gen


def main():
    t0 = time.perf_counter()
    gen = load_miso_8b("cuda")
    print(f"[load] {time.perf_counter() - t0:.1f}s", flush=True)
    out = "perf_eval/out"
    os.makedirs(out, exist_ok=True)

    for pid in SUSPECTS:
        idx = _idx_of(pid)
        _, text, orig_cap = EVAL_PROMPTS[idx]
        seed = 1234 + idx
        frames, eos_idx, max_gen = collect_frames(gen, text, seed, GENEROUS_CAP_MS, EXTRA_AFTER_EOS)
        orig_cap_frames = int(orig_cap / 80)
        print(f"\n=== {pid} (seed {seed}) ===", flush=True)
        if eos_idx is None:
            print(f"  EOS did NOT fire within {len(frames)} frames "
                  f"({len(frames)*FRAME_S:.2f}s); orig cap was {orig_cap_frames} frames "
                  f"({orig_cap/1000:.1f}s) -> MODE A (cap-bound, EOS never fired).", flush=True)
            eos_for_decode = min(orig_cap_frames, len(frames))
        else:
            eos_t = eos_idx * FRAME_S
            mode = "A (cap-bound)" if eos_idx >= orig_cap_frames else "B (EOS-fired-early)"
            print(f"  EOS first fired at frame {eos_idx} ({eos_t:.2f}s). "
                  f"orig cap {orig_cap_frames} frames ({orig_cap/1000:.1f}s). "
                  f"-> {'within cap' if eos_idx < orig_cap_frames else 'AFTER cap (cap truncated it!)'}", flush=True)
            eos_for_decode = eos_idx

        # Decode tail variants and measure.
        variants = {"A_drop_eos": eos_for_decode,
                    "B_incl_eos": eos_for_decode + 1,
                    "C6": eos_for_decode + 6,
                    "C12": eos_for_decode + 12,
                    "C24": eos_for_decode + 24}
        for name, cut in variants.items():
            cut = max(1, min(cut, len(frames)))
            with torch.inference_mode():
                audio = gen._decode_frames(frames[:cut]).float().cpu()
            ratio, dur = tail_ratio(audio)
            torchaudio.save(f"{out}/trunc_{pid}_{name}.wav",
                            audio.unsqueeze(0), SR)
            print(f"    {name:12} frames={cut:4d} dur={dur:5.2f}s tail/speech={ratio:.2f}", flush=True)

    print("\n[done] Lower tail/speech is better. If B/Ck drop well below A, the fix "
          "is decoding trailing frames; if EOS fires after the cap, raise the cap.", flush=True)


if __name__ == "__main__":
    main()
