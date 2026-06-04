"""Local RunPod simulator.

Mirrors the RunPod serverless wire contract so the same client code works locally
and on real RunPod:

  POST /runsync     body {"input": {...}} -> {"id","status":"COMPLETED","output":[...]}
                    Collects every handler yield into output[] (buffered), exactly
                    like RunPod's /runsync. Streaming does NOT happen here.
  POST /run         -> {"id","status":"IN_QUEUE"}   (async submit)
  GET  /status/{id} -> {"status","output":[...]}    (poll)
  POST /stream_now  Local-dev streaming path (RunPod /runsync can't stream). Same
                    request body; returns chunked NDJSON (one JSON object per
                    line) or SSE if Accept: text/event-stream. This drives the
                    model's generate_stream so streaming can be tested locally.
  GET  /health

Run: uvicorn miso_server.runpod_local:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .handler import handler

app = FastAPI(title="MisoTTS RunPod simulator")

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_counter = 0


def _next_id() -> str:
    global _counter
    with _JOBS_LOCK:
        _counter += 1
        return f"local-{_counter}"


@app.on_event("startup")
def _startup() -> None:
    from . import core
    if os.environ.get("MISO_EAGER_LOAD", "1") == "1":
        core.discover_voices()
        if os.environ.get("MISO_WARMUP", "1") == "1":
            core.warmup()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/runsync")
async def runsync(request: Request):
    body = await request.json()
    job = {"id": _next_id(), "input": body.get("input", body)}
    output = list(handler(job))
    return {"id": job["id"], "status": "COMPLETED", "output": output}


@app.post("/run")
async def run(request: Request):
    body = await request.json()
    job_id = _next_id()
    job = {"id": job_id, "input": body.get("input", body)}

    def _work():
        with _JOBS_LOCK:
            _JOBS[job_id] = {"status": "IN_PROGRESS", "output": []}
        out = list(handler(job))
        with _JOBS_LOCK:
            _JOBS[job_id] = {"status": "COMPLETED", "output": out}

    threading.Thread(target=_work, daemon=True).start()
    return {"id": job_id, "status": "IN_QUEUE"}


@app.get("/status/{job_id}")
def status(job_id: str):
    with _JOBS_LOCK:
        return _JOBS.get(job_id, {"status": "NOT_FOUND", "output": []})


@app.post("/stream_now")
async def stream_now(request: Request):
    body = await request.json()
    job = {"id": _next_id(), "input": body.get("input", body)}
    job["input"]["stream"] = True
    accept = request.headers.get("accept", "")
    headers = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}

    if "text/event-stream" in accept:
        def sse():
            yield b": stream-open\n\n"
            for piece in handler(job):
                yield ("data: " + json.dumps(piece, default=str) + "\n\n").encode("utf-8")
        return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)

    def ndjson():
        for piece in handler(job):
            yield (json.dumps(piece, default=str) + "\n").encode("utf-8")
    return StreamingResponse(ndjson(), media_type="application/x-ndjson", headers=headers)


@app.exception_handler(Exception)
def _err(_request, exc):  # pragma: no cover
    return JSONResponse(status_code=500, content={"error": str(exc)})
