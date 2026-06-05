"""Task/Opportunity 3: is the SilentCipher watermark being stripped by the
44.1k -> 24k downsample in watermarking.watermark()?

watermark() encodes the mark on 44.1kHz audio, then resamples the WATERMARKED
audio down to 24kHz (output_sample_rate = min(44100, sample_rate)=24000). verify()
later upsamples 24k -> 44.1k and decodes. If the mark lives above 12 kHz (the 24k
Nyquist), the downsample destroys it and no upsampling can recover it.

This isolates the question by decoding the mark at three points:
  (1) at 44.1k right after encode (no downsample)        -> mark present?
  (2) after a 44.1k -> 24k -> 44.1k resample round-trip   -> mark survives 24k?
  (3) the full watermarking.watermark() pipeline output   -> what we ship today.
"""
import sys

sys.path.insert(0, "/opt/miso")
sys.path.insert(0, "/opt/miso/deploy")

import torch
import torchaudio

from miso_server import core
from perf_eval.prompts import QUICK_PROMPTS
from watermarking import MISO_TTS_WATERMARK, watermark


def show(tag, res):
    ok = res.get("status")
    msg = res["messages"][0] if ok else None
    print(f"  {tag:46} status={ok} message={msg}", flush=True)


def main():
    gen = core.get_generator()
    wm = gen._watermarker
    sr = gen.sample_rate  # 24000
    pid, text, max_ms = [p for p in QUICK_PROMPTS if p[0] == "long_sad"][0]
    print(f"key={MISO_TTS_WATERMARK} sr={sr}; prompt={pid}", flush=True)

    # Raw (UN-watermarked) 24k audio from the model.
    torch.manual_seed(1234)
    with torch.inference_mode():
        pt, ptm, mgl = gen._prepare_prompt(text, 0, [], max_ms)
        frames = list(gen._generate_frames(pt, ptm, mgl, 0.9, 50))
        k = len(frames)
        while k > 0 and bool((frames[k - 1] == 0).all()):
            k -= 1
        raw24 = gen._decode_frames(frames[:k]).float()
    print(f"raw audio: {raw24.shape[0]/sr:.2f}s @ {sr}", flush=True)

    with torch.inference_mode():
        # encode at 44.1k
        a44 = torchaudio.functional.resample(raw24, sr, 44100)
        enc44, _ = wm.encode_wav(a44, 44100, MISO_TTS_WATERMARK, calc_sdr=False, message_sdr=36)
        # (1) decode at 44.1k, no downsample
        show("(1) 44.1k, no downsample", wm.decode_wav(enc44, 44100, phase_shift_decoding=True))
        # (2) 44.1k -> 24k -> 44.1k round-trip (mimics shipping at 24k then verify)
        rt = torchaudio.functional.resample(
            torchaudio.functional.resample(enc44, 44100, 24000), 24000, 44100)
        show("(2) after 44.1k->24k->44.1k round-trip", wm.decode_wav(rt, 44100, phase_shift_decoding=True))
        # (3) the actual watermarking.watermark() pipeline output (encoded, returned at 24k)
        enc_pipe, out_sr = watermark(wm, raw24, sr, MISO_TTS_WATERMARK)
        rt_pipe = torchaudio.functional.resample(enc_pipe, out_sr, 44100)
        show(f"(3) full watermark() output (sr={out_sr})", wm.decode_wav(rt_pipe, 44100, phase_shift_decoding=True))

    print("If (1)=True and (2)/(3)=False, the 24k downsample strips the mark -> "
          "serve at 44.1k (or a rate that preserves it).", flush=True)


if __name__ == "__main__":
    main()
