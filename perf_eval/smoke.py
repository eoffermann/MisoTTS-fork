"""Smoke test + end-truncation diagnostic.

Confirms the model loads and generates end-to-end (validates the KV-cache device
fix), and probes the user-reported "clips cut off at the end" symptom:

- Generates the SAME short text at two different max_audio_length_ms caps. If the
  durations differ materially, the model is NOT emitting a clean EOS frame and
  the cap is what truncates the audio (so every clip ends abruptly mid-trail).
- Measures tail energy: RMS of the final 80 ms vs the clip's peak 80 ms RMS. A
  natural utterance trails into near-silence (low ratio); an abrupt cutoff ends
  with speech-level energy at the very last sample (ratio near 1.0).
"""
import os

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["NO_TORCH_COMPILE"] = "1"

import time
import torch
import torchaudio

from generator import load_miso_8b

TEXT = "The quick brown fox jumps over the lazy dog."
SR = 24_000


def tail_report(audio, sr):
    win = int(0.08 * sr)  # 80 ms
    x = audio.float()
    n = x.shape[0]
    if n < win:
        return None
    tail_rms = x[-win:].pow(2).mean().sqrt().item()
    # peak 80ms RMS via simple strided scan
    step = max(1, win // 2)
    peak = 0.0
    for s in range(0, n - win + 1, step):
        r = x[s:s + win].pow(2).mean().sqrt().item()
        if r > peak:
            peak = r
    ratio = tail_rms / peak if peak > 0 else 0.0
    return tail_rms, peak, ratio


def main():
    t0 = time.perf_counter()
    gen = load_miso_8b("cuda")
    print(f"[load] {time.perf_counter() - t0:.1f}s", flush=True)
    out = "perf_eval/out"
    os.makedirs(out, exist_ok=True)

    for cap_ms in (8_000, 20_000):
        torch.manual_seed(1234)  # same seed -> same RNG draws; isolates the cap effect
        t1 = time.perf_counter()
        audio = gen.generate(text=TEXT, speaker=0, context=[],
                             max_audio_length_ms=cap_ms)
        torch.cuda.synchronize()
        dur = audio.shape[0] / SR
        tr = tail_report(audio, SR)
        path = f"{out}/smoke_cap{cap_ms}.wav"
        torchaudio.save(path, audio.unsqueeze(0).cpu(), SR)
        msg = (f"[cap={cap_ms:>6}ms] gen={time.perf_counter() - t1:5.1f}s  "
               f"dur={dur:5.2f}s  cap_dur={cap_ms / 1000:5.2f}s")
        if tr:
            msg += f"  tail_rms={tr[0]:.4f}  peak_rms={tr[1]:.4f}  tail/peak={tr[2]:.2f}"
        print(msg, flush=True)

    print("[done] If the two durations differ, EOS is not firing and the cap is "
          "truncating. If tail/peak is high (~>0.5), the clip ends mid-speech.",
          flush=True)


if __name__ == "__main__":
    main()
