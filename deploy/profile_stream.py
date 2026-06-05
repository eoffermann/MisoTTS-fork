"""In-container STREAMING latency profile: TTFB + per-chunk ramp + RTF.

Models profile_container.py but for the streaming path (core.synth_stream). For
each QUICK_PROMPT it records:
  - TTFB (ms): wall from the synth_stream call to the first yielded chunk,
  - per-chunk emit wall + sample count (the ramp),
  - RTF: total generate wall / total audio seconds (steady state, pass 2).

Two passes (pass 1 may still recompile; pass 2 is the warm steady-state number).
The harness LABELS its regime (compiled vs eager) so a baseline is never
ambiguous. Env knobs for later sweeps: MISO_STREAM_CHUNK_FRAMES (cap),
MISO_STREAM_START_FRAMES, MISO_STREAM_RAMP, MISO_STREAM_MAX_FRAMES.

Run inside the container:
  docker run --rm --gpus all -e MISO_COMPILE=1 \
    -v <hf>:/workspace/hf -v <inductor>:/workspace/inductor_cache \
    miso-tts:latest python /opt/miso/deploy/profile_stream.py
"""
import os
import statistics
import sys
import time

sys.path.insert(0, "/opt/miso")
sys.path.insert(0, "/opt/miso/deploy")

_T0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - _T0:6.1f}s] {msg}", flush=True)


log("importing torch (pulls CUDA libs)...")
import torch  # noqa: E402

from miso_server import core  # noqa: E402
from perf_eval.prompts import QUICK_PROMPTS  # noqa: E402


def compile_regime(gen) -> str:
    """ACTIVE if backbone/decoder are torch.compile OptimizedModules, else EAGER."""
    bb = type(getattr(gen._model, "backbone", None)).__name__
    dc = type(getattr(gen._model, "decoder", None)).__name__
    active = bb == "OptimizedModule" and dc == "OptimizedModule"
    return f"compile={'ACTIVE' if active else 'EAGER'} (backbone={bb}, decoder={dc})"


def stream_one(text, max_ms, chunk_frames):
    """Run synth_stream once; return (ttfb_ms, chunk_walls, chunk_samples, total_wall, audio_s, sr)."""
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    t_prev = t_start
    ttfb_ms = None
    chunk_walls, chunk_samples = [], []
    total_samples, sr = 0, 24000
    for a, s in core.synth_stream(text=text, max_audio_length_ms=max_ms,
                                  chunk_frames=chunk_frames, seed=1234):
        now = time.perf_counter()
        if ttfb_ms is None:
            ttfb_ms = (now - t_start) * 1000.0
        chunk_walls.append((now - t_prev) * 1000.0)
        chunk_samples.append(int(a.shape[0]))
        total_samples += int(a.shape[0])
        sr = s
        t_prev = now
    total_wall = time.perf_counter() - t_start
    audio_s = total_samples / sr if sr else 0.0
    return ttfb_ms, chunk_walls, chunk_samples, total_wall, audio_s, sr


def run_pass(tag, chunk_frames):
    rows = []
    for pid, text, max_ms in QUICK_PROMPTS:
        ttfb, walls, samples, wall, audio_s, sr = stream_one(text, max_ms, chunk_frames)
        rtf = wall / audio_s if audio_s > 0 else float("nan")
        rows.append({"id": pid, "ttfb_ms": ttfb, "n_chunks": len(walls), "audio_s": audio_s,
                     "wall_s": wall, "rtf": rtf, "first_chunk_samples": samples[0] if samples else 0})
        log(f"[{tag}] {pid:14} TTFB={ttfb:7.1f}ms chunks={len(walls):3d} "
            f"first={samples[0] if samples else 0:5d}smp audio={audio_s:5.2f}s "
            f"wall={wall:6.2f}s RTF={rtf:5.2f}")
    return rows


def summarize(tag, rows):
    ttfbs = [r["ttfb_ms"] for r in rows if r["ttfb_ms"] is not None]
    tw = sum(r["wall_s"] for r in rows)
    ta = sum(r["audio_s"] for r in rows)
    if ttfbs:
        p90 = sorted(ttfbs)[max(0, int(round(0.9 * len(ttfbs))) - 1)]
        log(f"[{tag}] TTFB mean={statistics.mean(ttfbs):.1f} median={statistics.median(ttfbs):.1f} "
            f"p90={p90:.1f} ms | MEAN RTF={tw/ta:.2f}")


def main():
    cap = int(os.environ.get("MISO_STREAM_CHUNK_FRAMES", "25"))
    log(f"MISO_COMPILE={os.environ.get('MISO_COMPILE')} chunk_frames(cap)={cap} "
        f"START={os.environ.get('MISO_STREAM_START_FRAMES')} RAMP={os.environ.get('MISO_STREAM_RAMP')} "
        f"MAX={os.environ.get('MISO_STREAM_MAX_FRAMES')}")
    log("loading model via core.get_generator()...")
    gen = core.get_generator()
    log(f"loaded. {compile_regime(gen)}")

    log("warmup (bakes compile graphs)...")
    core.warmup()
    log(f"warmup done. {compile_regime(gen)}")

    rows1 = run_pass("pass1", cap)
    summarize("pass1", rows1)
    rows2 = run_pass("pass2", cap)
    log("=== SUMMARY ===")
    summarize("pass2 (warm)", rows2)
    log(compile_regime(gen))


if __name__ == "__main__":
    main()
