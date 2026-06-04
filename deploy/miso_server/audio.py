"""Audio encoding helpers for the serving surfaces."""
from __future__ import annotations

import base64
import io

import numpy as np
import soundfile as sf

_FORMAT_TO_SF = {
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "ogg": ("OGG", "VORBIS", "audio/ogg"),
    "opus": ("OGG", "OPUS", "audio/ogg; codecs=opus"),
}


def encode(audio: np.ndarray, sample_rate: int, fmt: str = "wav") -> tuple[bytes, str]:
    """Encode float32 mono audio to container bytes. Returns (bytes, content_type)."""
    fmt = (fmt or "wav").lower()
    if fmt == "pcm":  # raw 16-bit little-endian PCM, no header
        pcm = np.clip(audio, -1.0, 1.0)
        return (pcm * 32767.0).astype("<i2").tobytes(), "audio/L16"
    sf_fmt, subtype, content_type = _FORMAT_TO_SF.get(fmt, _FORMAT_TO_SF["wav"])
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format=sf_fmt, subtype=subtype)
    return buf.getvalue(), content_type


def encode_b64(audio: np.ndarray, sample_rate: int, fmt: str = "wav") -> tuple[str, str]:
    data, content_type = encode(audio, sample_rate, fmt)
    return base64.b64encode(data).decode("ascii"), content_type
