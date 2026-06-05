"""OpenAI-compatible TTS API surface.

Implements POST /v1/audio/speech with the OpenAI audio-speech request shape so
existing OpenAI TTS clients work unchanged (point base_url at this server). The
`voice` field selects a pre-assigned MisoTTS voice; `model` is accepted and
ignored beyond logging. Output loudness is normalized in the serving core.

Run: uvicorn miso_server.openai_api:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import audio as audiolib
from . import core

app = FastAPI(title="MisoTTS (OpenAI-compatible)")

# OpenAI response_format values mapped to our encoder formats.
_FORMAT_MAP = {"mp3": "wav", "opus": "opus", "aac": "wav", "flac": "flac",
               "wav": "wav", "pcm": "pcm"}


class SpeechRequest(BaseModel):
    model: str = "miso-tts-8b"
    input: str = Field(..., min_length=1, max_length=8000)
    voice: str = core.DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0          # accepted for compatibility; MisoTTS has no rate ctrl yet
    stream: bool = False        # extension: stream raw audio chunks
    seed: int | None = None
    # Extensions (optional, off-schema): tune the low-latency emit ramp. Absent ->
    # the MISO_STREAM_* env defaults, so standard OpenAI clients still get the ramp.
    stream_max_frames: int | None = None    # emit-size cap in frames (default 25)
    stream_start_frames: int | None = None  # first emit size in frames (default 1 = 80 ms)


@app.on_event("startup")
def _startup() -> None:
    if os.environ.get("MISO_EAGER_LOAD", "1") == "1":
        core.discover_voices()
        if os.environ.get("MISO_WARMUP", "1") == "1":
            core.warmup()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/audio/voices")
def voices() -> dict:
    return {"voices": core.discover_voices()}


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": "miso-tts-8b", "object": "model"}]}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    fmt = _FORMAT_MAP.get(req.response_format.lower(), "wav")
    if req.voice not in core.list_voices():
        raise HTTPException(status_code=400,
                            detail=f"unknown voice '{req.voice}'; known: {core.list_voices()}")

    if req.stream:
        def gen_chunks():
            for chunk, sr in core.synth_stream(req.input, voice=req.voice, seed=req.seed,
                                               chunk_frames=req.stream_max_frames,
                                               start_frames=req.stream_start_frames):
                data, _ = audiolib.encode(chunk, sr, "pcm")  # raw PCM frames for streaming
                yield data
        return StreamingResponse(gen_chunks(), media_type="audio/L16",
                                 headers={"X-Sample-Rate": "24000", "X-Accel-Buffering": "no"})

    audio, sr = core.synth(req.input, voice=req.voice, seed=req.seed)
    data, content_type = audiolib.encode(audio, sr, fmt)
    return Response(content=data, media_type=content_type)


@app.exception_handler(Exception)
def _err(_request, exc):  # pragma: no cover - defensive
    return JSONResponse(status_code=500, content={"error": {"message": str(exc)}})
