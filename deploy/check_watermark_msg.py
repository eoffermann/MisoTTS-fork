"""Opportunity 3: is the all-zero watermark key the reason nothing is detected?

MISO_TTS_WATERMARK = [0,0,0,0,0] encodes to an all-zero bit message, which the
SilentCipher decoder may not distinguish from "no watermark". Encode the same clip
at 44.1k with the all-zero key and with non-zero messages and see which decode.
Also sweep message_sdr (the watermark strength) in case 36 dB is too quiet.
"""
import sys

sys.path.insert(0, "/opt/miso")
sys.path.insert(0, "/opt/miso/deploy")

import torch
import torchaudio

from miso_server import core
from perf_eval.prompts import QUICK_PROMPTS


def main():
    gen = core.get_generator()
    wm = gen._watermarker
    sr = gen.sample_rate
    pid, text, max_ms = [p for p in QUICK_PROMPTS if p[0] == "long_sad"][0]
    torch.manual_seed(1234)
    with torch.inference_mode():
        pt, ptm, mgl = gen._prepare_prompt(text, 0, [], max_ms)
        frames = list(gen._generate_frames(pt, ptm, mgl, 0.9, 50))
        k = len(frames)
        while k > 0 and bool((frames[k - 1] == 0).all()):
            k -= 1
        raw = gen._decode_frames(frames[:k]).float()
    a44 = torchaudio.functional.resample(raw, sr, 44100)
    print(f"clip {raw.shape[0]/sr:.1f}s; default config.message_sdr="
          f"{getattr(wm.config, 'message_sdr', '?')}", flush=True)

    trials = [([0, 0, 0, 0, 0], 36), ([123, 234, 111, 222, 11], 36),
              ([1, 2, 3, 4, 5], 36), ([123, 234, 111, 222, 11], None),
              ([123, 234, 111, 222, 11], 20)]
    for msg, sdr in trials:
        with torch.inference_mode():
            enc, _ = wm.encode_wav(a44.clone(), 44100, msg, calc_sdr=False, message_sdr=sdr)
            r = wm.decode_wav(enc, 44100, phase_shift_decoding=True)
        dec = r["messages"][0] if r.get("status") else None
        match = dec == msg if dec is not None else False
        print(f"  msg={msg} sdr={sdr}: status={r.get('status')} decoded={dec} match={match}", flush=True)
    print("If non-zero messages detect+match but [0,0,0,0,0] does not, the all-zero "
          "key is the bug -> use a non-zero MISO_TTS_WATERMARK.", flush=True)


if __name__ == "__main__":
    main()
