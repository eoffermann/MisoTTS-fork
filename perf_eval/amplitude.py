"""Measure per-clip output amplitude (peak / RMS / integrated LUFS).

The user reports clips coming out dramatically louder or softer with no obvious
pattern (not correlated with emotion). This characterizes the spread so we can
see how large it is and whether it tracks anything. Run on any directory of WAVs:

  .venv/Scripts/python.exe -m perf_eval.amplitude perf_eval/baseline
"""
import glob
import os
import sys

import numpy as np
import soundfile as sf

try:
    import pyloudnorm as pyln
except Exception:
    pyln = None


def measure(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    peak = 20 * np.log10(max(float(np.abs(a).max()), 1e-9))
    rms = 20 * np.log10(max(float(np.sqrt((a ** 2).mean())), 1e-9))
    lufs = float("nan")
    if pyln is not None and len(a) > sr * 0.4:
        try:
            lufs = pyln.Meter(sr).integrated_loudness(a)
        except Exception:
            pass
    return peak, rms, lufs, len(a) / sr


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "perf_eval/baseline"
    files = sorted(glob.glob(os.path.join(d, "*.wav")))
    if not files:
        print("no wavs in", d)
        return
    print("%-18s %9s %9s %7s %6s" % ("clip", "peak_dBFS", "rms_dBFS", "LUFS", "dur_s"))
    rows = []
    for w in files:
        peak, rms, lufs, dur = measure(w)
        name = os.path.basename(w).replace(".wav", "")
        rows.append((name, peak, rms, lufs, dur))
        print("%-18s %9.2f %9.2f %7.1f %6.2f" % (name, peak, rms, lufs, dur))
    peaks = [r[1] for r in rows]
    rmss = [r[2] for r in rows]
    lufss = [r[3] for r in rows if r[3] == r[3]]  # drop nan
    print("--- spread: peak %.1f dB | rms %.1f dB | LUFS %.1f"
          % (max(peaks) - min(peaks), max(rmss) - min(rmss),
             (max(lufss) - min(lufss)) if lufss else float("nan")))


if __name__ == "__main__":
    main()
