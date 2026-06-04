"""Local smoke test for the serving core (no Docker, no FastAPI).

Validates that miso_server.core loads the model, registers the default voice,
synthesizes, and normalizes output loudness toward the target LUFS. Run with the
venv python from the repo root:

  .venv/Scripts/python.exe -m deploy.smoke_serving
"""
import os
import sys

# On the Windows dev box there is no triton; keep compile off and eager.
os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.environ.setdefault("MISO_COMPILE", "0")
os.environ.setdefault("MISO_WARMUP", "0")
_HERE = os.path.dirname(os.path.abspath(__file__))      # .../deploy
sys.path.insert(0, _HERE)                                # so `miso_server` imports
sys.path.insert(0, os.path.dirname(_HERE))               # repo root, so `generator` imports

import numpy as np

from miso_server import core


def lufs(a, sr):
    try:
        import pyloudnorm as pyln
        return pyln.Meter(sr).integrated_loudness(a)
    except Exception:
        return float("nan")


def main():
    print("voices:", core.discover_voices(), flush=True)
    target = core.DEFAULT_TARGET_LUFS
    for text in ["I just feel so empty inside today, like nothing matters anymore.",
                 "I went to the grocery store this morning to buy some fresh bread."]:
        audio, sr = core.synth(text, voice="default", seed=1234, max_audio_length_ms=20_000)
        peak = 20 * np.log10(max(float(np.abs(audio).max()), 1e-9))
        print(f"synth ok: dur={len(audio)/sr:.2f}s sr={sr} peak={peak:.2f}dBFS "
              f"LUFS={lufs(audio, sr):.1f} (target {target})", flush=True)
    print("SERVING CORE SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
