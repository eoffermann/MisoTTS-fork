"""Opportunity 3: isolate whether SilentCipher's encode->decode roundtrips AT ALL
in this install, independent of MisoTTS audio. Loads ONLY the watermarker and runs
a clean signal through encode_wav -> decode_wav at the model's native rate, sweeping
phase_shift_decoding. If even this fails, the silentcipher 1.0.4 install/checkpoint
is the problem (version pin / API mismatch); if it passes, our MisoTTS audio
characteristics are the issue.
"""
import sys

sys.path.insert(0, "/opt/miso")
sys.path.insert(0, "/opt/miso/deploy")

import torch

from watermarking import load_watermarker


def main():
    wm = load_watermarker(device="cuda")
    sr = int(getattr(wm, "sr", 44100))
    print(f"watermarker model sr={sr} message_sdr={getattr(wm.config,'message_sdr','?')}", flush=True)

    torch.manual_seed(0)
    n = 5 * sr
    t = torch.arange(n, device="cuda") / sr
    # speech-like: a couple of tones + light noise, normal amplitude
    audio = (0.3 * torch.sin(2 * torch.pi * 180 * t)
             + 0.15 * torch.sin(2 * torch.pi * 320 * t)
             + 0.05 * torch.randn(n, device="cuda")).clamp(-1, 1)
    msg = [123, 234, 111, 222, 11]

    for psd in (True, False, "true"):
        with torch.inference_mode():
            enc, _ = wm.encode_wav(audio.clone(), sr, msg, calc_sdr=False, message_sdr=None)
            r = wm.decode_wav(enc, sr, phase_shift_decoding=psd)
        dec = r["messages"][0] if r.get("status") else None
        print(f"  clean signal, phase_shift_decoding={psd!r}: status={r.get('status')} "
              f"decoded={dec} match={dec == msg}", flush=True)

    print("If all status=False on a clean signal, silentcipher 1.0.4 itself does not "
          "roundtrip here (version/checkpoint issue); else it is our audio.", flush=True)


if __name__ == "__main__":
    main()
