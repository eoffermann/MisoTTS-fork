"""RunPod serverless handler for MisoTTS.

A generator handler (RunPod collects yields into the response list). For
stream=False it yields exactly one chunk with the full clip; for stream=True it
yields one chunk per audio segment as it is produced. This is the same contract
FrndoVoice uses, so the local simulator (runpod_local.py) and real RunPod share
client code.

Input schema (job["input"]):
  text: str (required)
  voice: str = "default"
  stream: bool = false
  audio_format: "wav" | "opus" | "flac" | "pcm" = "wav"
  chunk_frames: int = 25
  temperature: float = 0.9
  topk: int = 50
  seed: int | null

Run on RunPod: python -m miso_server.handler   (no --rp_serve_api; RunPod drives it)
Run local dev API: python -m miso_server.handler --rp_serve_api --rp_api_host 0.0.0.0 --rp_api_port 8000
"""
from __future__ import annotations

import os
import time

from . import audio as audiolib
from . import core


def _coerce_input(job: dict) -> dict:
    inp = job.get("input", job) or {}
    return {
        "text": inp.get("text", ""),
        "voice": inp.get("voice", core.DEFAULT_VOICE),
        "stream": bool(inp.get("stream", False)),
        "audio_format": inp.get("audio_format", "wav"),
        "chunk_frames": int(inp.get("chunk_frames", 25)),
        "temperature": float(inp.get("temperature", 0.9)),
        "topk": int(inp.get("topk", 50)),
        "seed": inp.get("seed"),
    }


def handler(job):
    """RunPod generator handler. Yields one (non-stream) or many (stream) dicts."""
    p = _coerce_input(job)
    req_id = job.get("id", "local")
    if not p["text"]:
        yield {"error": "input.text is required"}
        return

    t0 = time.perf_counter()
    if not p["stream"]:
        audio, sr = core.synth(p["text"], voice=p["voice"], temperature=p["temperature"],
                               topk=p["topk"], seed=p["seed"])
        b64, content_type = audiolib.encode_b64(audio, sr, p["audio_format"])
        yield {
            "id": req_id, "voice": p["voice"], "content_type": content_type,
            "sample_rate": sr, "audio_base64": b64,
            "duration": len(audio) / sr, "cost": time.perf_counter() - t0,
        }
        return

    ttfb = None
    idx = 0
    for chunk, sr in core.synth_stream(p["text"], voice=p["voice"], temperature=p["temperature"],
                                       topk=p["topk"], chunk_frames=p["chunk_frames"], seed=p["seed"]):
        if ttfb is None:
            ttfb = (time.perf_counter() - t0) * 1000.0
        b64, content_type = audiolib.encode_b64(chunk, sr, p["audio_format"])
        yield {
            "id": req_id, "voice": p["voice"], "chunk_index": idx, "is_final": False,
            "content_type": content_type, "sample_rate": sr, "audio_base64": b64,
            "ttfb_ms": ttfb if idx == 0 else None,
            "elapsed_s": time.perf_counter() - t0,
        }
        idx += 1
    yield {"id": req_id, "voice": p["voice"], "chunk_index": idx, "is_final": True,
           "elapsed_s": time.perf_counter() - t0}


if __name__ == "__main__":
    # Eagerly load + warm so the first request is fast.
    if os.environ.get("MISO_EAGER_LOAD", "1") == "1":
        core.discover_voices()
        if os.environ.get("MISO_WARMUP", "1") == "1":
            core.warmup()
    import runpod  # provided in the container image
    runpod.serverless.start({"handler": handler})
