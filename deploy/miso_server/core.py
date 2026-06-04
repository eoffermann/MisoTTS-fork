"""Shared MisoTTS serving core.

One model load, reused by all surfaces (RunPod handler, RunPod local simulator,
OpenAI-compatible API). Provides:
  - lazy, cached model load with optional torch.compile + flash-attention (Linux)
  - a named voice registry (pre-assigned voice prompts: reference wav + transcript)
  - synth() and synth_stream() that apply output loudness normalization

This module is platform-agnostic but is intended to run in the Linux CUDA
container, where triton/Inductor and flash SDPA kernels are available (unlike the
Windows dev box). Set MISO_COMPILE=1 to enable torch.compile of the model.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

# Repo root (two levels up from this file: deploy/miso_server/core.py -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Default loudness target (LUFS) and peak ceiling (dBFS). Normalization fixes the
# ~19 dB clip-to-clip loudness spread the raw model produces. -16 LUFS is a
# common target for speech; the peak limiter prevents clipping after gain.
DEFAULT_TARGET_LUFS = float(os.environ.get("MISO_TARGET_LUFS", "-16.0"))
DEFAULT_PEAK_CEILING_DBFS = float(os.environ.get("MISO_PEAK_CEILING_DBFS", "-1.0"))

_MODEL_LOCK = threading.Lock()
_GEN = None  # cached Generator


@dataclass
class Voice:
    """A pre-assigned voice: an id plus the context that conditions generation.

    For the model's built-in voice, context is empty and speaker is 0. For a
    cloned voice, context holds a reference Segment (audio + transcript) whose
    Mimi codes are cached on the Segment after first use (see generator
    _tokenize_segment caching, findings F4/F8), so registering a voice encodes
    it once and every later request reuses the codes.
    """
    voice_id: str
    speaker: int = 0
    context: tuple = ()  # tuple of generator.Segment


def _maybe_compile(model) -> None:
    """Optionally torch.compile the model for CUDA-graph / fused-kernel speedups.

    Only meaningful with triton/Inductor available (Linux). Guarded by MISO_COMPILE
    so the Windows dev path stays eager. Compilation warms up on the first calls.
    """
    if os.environ.get("MISO_COMPILE", "0") != "1":
        return
    try:
        # Compile the per-frame decode hot path. reduce-overhead uses CUDA graphs.
        # Originals are saved so warmup can revert to eager if compilation (which
        # happens lazily on the first forward) fails on the torchtune decode path.
        # reduce-overhead uses CUDA graphs (biggest win, but fragile with dynamic
        # shapes / the KV cache). MISO_COMPILE_MODE can drop to "default" (plain
        # Inductor fusion, no graphs) if the graph path does not hold.
        mode = os.environ.get("MISO_COMPILE_MODE", "reduce-overhead")
        _COMPILE_ORIG["backbone"] = model._model.backbone
        _COMPILE_ORIG["decoder"] = model._model.decoder
        model._model.backbone = torch.compile(model._model.backbone, mode=mode)
        model._model.decoder = torch.compile(model._model.decoder, mode=mode)
        print(f"[core] torch.compile enabled (mode={mode})", flush=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[core] torch.compile setup skipped: {exc}", flush=True)


_COMPILE_ORIG: dict = {}


def _revert_compile(model) -> None:
    if _COMPILE_ORIG:
        model._model.backbone = _COMPILE_ORIG["backbone"]
        model._model.decoder = _COMPILE_ORIG["decoder"]
        _COMPILE_ORIG.clear()


# Model variants that actually exist on disk / are loadable today. As fp8/fp4/
# int8 quantized checkpoints are produced, add them here and GPU-sense will pick
# them. Today only the bf16 path is built.
AVAILABLE_VARIANTS = set(
    (os.environ.get("MISO_AVAILABLE_VARIANTS") or "bf16").split(",")
)
_DTYPE_FOR_VARIANT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def detect_device_profile() -> dict:
    """Detect the GPU architecture and the model variant it would IDEALLY use,
    plus the best variant we can actually load today.

    arch by compute capability:
      blackwell sm>=100 (B100/B200, RTX 50xx) -> native FP4 + FP8
      hopper    sm 90   (H100)                 -> native FP8
      ada       sm 89   (RTX 40, L40)          -> native FP8
      ampere    sm 80/86/87 (A100, A6000, 30x) -> bf16, no native FP8
    """
    if not torch.cuda.is_available():
        return {"arch": "cpu", "cc": 0, "name": "cpu", "vram_gb": 0.0,
                "fp8_native": False, "fp4_native": False,
                "ideal_variant": "fp32", "variant": "fp32"}
    major, minor = torch.cuda.get_device_capability(0)
    cc = major * 10 + minor
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    arch = ("blackwell" if cc >= 100 else "hopper" if cc >= 90 else
            "ada" if cc == 89 else "ampere" if cc >= 80 else
            "turing" if cc >= 75 else "legacy")
    fp8_native = cc >= 89
    fp4_native = cc >= 100
    ideal = "nvfp4" if fp4_native else "fp8" if fp8_native else "bf16"
    # Best variant we have built, in preference order for this GPU.
    pref = {"nvfp4": ["nvfp4", "fp8", "bf16"], "fp8": ["fp8", "bf16"], "bf16": ["bf16"]}[ideal]
    variant = next((v for v in pref if v in AVAILABLE_VARIANTS), "bf16")
    return {"arch": arch, "cc": cc, "name": name, "vram_gb": vram,
            "fp8_native": fp8_native, "fp4_native": fp4_native,
            "ideal_variant": ideal, "variant": variant}


# HuggingFace distribution: ONE repo per variant under the org. Variant
# checkpoints are pulled at runtime (NOT baked into the image), so the image
# stays small and weights version independently. Each repo is env-overridable.
# Until a variant repo is published, loading falls back to the upstream default.
VARIANT_REPO = {
    "bf16":  os.environ.get("MISO_REPO_BF16",  "BigBlueCeiling/MisoTTS-bf16"),
    "fp8":   os.environ.get("MISO_REPO_FP8",   "BigBlueCeiling/MisoTTS-fp8"),
    "nvfp4": os.environ.get("MISO_REPO_NVFP4", "BigBlueCeiling/MisoTTS-nvfp4"),
}


def resolve_model_source(variant: str):
    """Checkpoint source for a variant, passed to load_miso_8b.

    Precedence: explicit MISO_TTS_8B_MODEL override -> the variant's HF repo
    (downloaded to a local path) -> None (load_miso_8b uses the upstream
    default). hf_hub_download caches, so repeated starts do not re-download.
    """
    explicit = os.environ.get("MISO_TTS_8B_MODEL")
    if explicit:
        return explicit
    repo = VARIANT_REPO.get(variant)
    if repo:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=repo, filename="model.safetensors")
            print(f"[core] pulled variant '{variant}' from {repo}", flush=True)
            return path
        except Exception as exc:
            print(f"[core] {repo} not available ({type(exc).__name__}); "
                  f"falling back to the default model", flush=True)
    return None


def get_generator():
    """Load (once) and return the MisoTTS Generator. Thread-safe."""
    global _GEN
    if _GEN is not None:
        return _GEN
    with _MODEL_LOCK:
        if _GEN is not None:
            return _GEN
        # On Linux we WANT compile/flash; do not force it off. On the Windows dev
        # box callers set NO_TORCH_COMPILE=1 themselves.
        from generator import load_miso_8b

        device = "cuda" if torch.cuda.is_available() else "cpu"
        prof = detect_device_profile()
        # Persist the TorchInductor / CUDA-graph compile cache per GPU arch so the
        # ~20 min compile warmup is paid ONCE (mount /workspace/inductor_cache as
        # a volume). Per-SM dir so different GPU classes do not collide.
        if prof["arch"] != "cpu":
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                                  f"/workspace/inductor_cache/sm_{prof['cc']}")
            os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        # MISO_MODEL_VARIANT can force a variant; otherwise use the auto-selected
        # best-available one for this GPU.
        variant = os.environ.get("MISO_MODEL_VARIANT", prof["variant"])
        if variant not in _DTYPE_FOR_VARIANT:
            # fp8/fp4/int8 need a built quantized checkpoint + a quantized load
            # path (torchao). Not wired yet -> fall back to bf16 with a clear note.
            print(f"[core] variant '{variant}' has no load path yet; using bf16.", flush=True)
            variant = "bf16"
        dtype = _DTYPE_FOR_VARIANT[variant]
        if prof["arch"] != "cpu":
            gap = (f" GPU could use '{prof['ideal_variant']}' but that variant is not "
                   f"built; build it to unlock it." if prof["ideal_variant"] != variant else "")
            print(f"[core] GPU {prof['name']} arch={prof['arch']} sm_{prof['cc']} "
                  f"vram={prof['vram_gb']:.0f}GB fp8={prof['fp8_native']} fp4={prof['fp4_native']} "
                  f"-> loading variant '{variant}' ({dtype}).{gap}", flush=True)
        t0 = time.perf_counter()
        source = resolve_model_source(variant)
        gen = load_miso_8b(device, model_path_or_repo_id=source, dtype=dtype)
        _maybe_compile(gen)
        print(f"[core] model loaded on {device} in {time.perf_counter() - t0:.1f}s", flush=True)
        _GEN = gen
        return _GEN


def warmup() -> None:
    """Run a few generations so torch.compile graphs are captured before serving."""
    gen = get_generator()
    texts = ("Warming up.", "This is a slightly longer warmup utterance for the decoder.")
    try:
        for text in texts:
            _ = gen.generate(text=text, speaker=0, context=[], max_audio_length_ms=6000)
    except Exception as exc:  # pragma: no cover
        # Most likely a compile / CUDA-graph failure on the torchtune decode path.
        # Revert to eager and retry so serving still works (without compile).
        print(f"[core] compiled warmup failed ({type(exc).__name__}: {exc}); "
              f"reverting to eager", flush=True)
        _revert_compile(gen)
        for text in texts:
            try:
                _ = gen.generate(text=text, speaker=0, context=[], max_audio_length_ms=6000)
            except Exception as exc2:
                print(f"[core] eager warmup error: {exc2}", flush=True)
    if str(getattr(gen, "device", "")).startswith("cuda"):
        torch.cuda.synchronize()
    print("[core] warmup complete", flush=True)


# ---------------------------------------------------------------------------
# Loudness normalization (fixes the clip-to-clip amplitude spread)
# ---------------------------------------------------------------------------
def normalize_loudness(audio: np.ndarray, sr: int,
                       target_lufs: Optional[float] = DEFAULT_TARGET_LUFS,
                       peak_ceiling_dbfs: float = DEFAULT_PEAK_CEILING_DBFS) -> np.ndarray:
    """Loudness-normalize to target_lufs (integrated) with a peak ceiling.

    If target_lufs is None, only the peak limiter is applied. Falls back to peak
    normalization if pyloudnorm is unavailable or the clip is too short to measure.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x
    if target_lufs is not None:
        gained = None
        try:
            import pyloudnorm as pyln
            if x.shape[0] > sr * 0.4:
                loudness = pyln.Meter(sr).integrated_loudness(x)
                if loudness > -70:  # not digital silence
                    gain = 10 ** ((target_lufs - loudness) / 20.0)
                    gained = x * gain
        except Exception:
            gained = None
        if gained is None:
            # peak-normalize toward a nominal level as a fallback
            peak = float(np.abs(x).max())
            if peak > 0:
                gained = x * (10 ** (-3.0 / 20.0) / peak)
            else:
                gained = x
        x = gained
    # Peak ceiling (prevent clipping after gain).
    ceiling = 10 ** (peak_ceiling_dbfs / 20.0)
    peak = float(np.abs(x).max())
    if peak > ceiling:
        x = x * (ceiling / peak)
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Voice registry
# ---------------------------------------------------------------------------
_VOICES: dict[str, Voice] = {}
_VOICES_LOCK = threading.Lock()
DEFAULT_VOICE = "default"
_PROMPTS_DIR = Path(os.environ.get("MISO_PROMPTS_DIR", str(_REPO_ROOT / "deploy" / "prompts")))


def _ensure_default_voice() -> None:
    if DEFAULT_VOICE not in _VOICES:
        _VOICES[DEFAULT_VOICE] = Voice(voice_id=DEFAULT_VOICE, speaker=0, context=())


def register_voice(voice_id: str, audio_path: str, transcript: str, speaker: int = 0) -> Voice:
    """Register a named voice from a reference wav + its transcript.

    The reference is wrapped in a generator.Segment used as conversation context;
    its Mimi codes are computed once (cached on the Segment) and reused.
    """
    import torchaudio
    from generator import Segment

    gen = get_generator()
    wav, sr = torchaudio.load(audio_path)
    wav = wav.mean(dim=0)  # mono
    if sr != gen.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, gen.sample_rate)
    seg = Segment(speaker=speaker, text=transcript, audio=wav.to(gen.device))
    voice = Voice(voice_id=voice_id, speaker=speaker, context=(seg,))
    with _VOICES_LOCK:
        _VOICES[voice_id] = voice
    return voice


def discover_voices() -> list[str]:
    """Load voices from the prompts dir. Each <name>.wav needs a <name>.txt
    transcript alongside it. Always includes the built-in 'default' voice."""
    _ensure_default_voice()
    if _PROMPTS_DIR.is_dir():
        for wav in sorted(_PROMPTS_DIR.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            if wav.stem in _VOICES or not txt.exists():
                continue
            try:
                register_voice(wav.stem, str(wav), txt.read_text(encoding="utf-8").strip())
            except Exception as exc:  # pragma: no cover
                print(f"[core] could not register voice {wav.stem}: {exc}", flush=True)
    return list_voices()


def list_voices() -> list[str]:
    _ensure_default_voice()
    return sorted(_VOICES.keys())


def _get_voice(voice_id: Optional[str]) -> Voice:
    _ensure_default_voice()
    return _VOICES.get(voice_id or DEFAULT_VOICE, _VOICES[DEFAULT_VOICE])


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
def synth(text: str, voice: Optional[str] = None, *, max_audio_length_ms: int = 40_000,
          temperature: float = 0.9, topk: int = 50,
          target_lufs: Optional[float] = DEFAULT_TARGET_LUFS,
          seed: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Synthesize the full clip. Returns (float32 mono audio, sample_rate)."""
    gen = get_generator()
    v = _get_voice(voice)
    if seed is not None:
        torch.manual_seed(seed)
    audio = gen.generate(text=text, speaker=v.speaker, context=list(v.context),
                         max_audio_length_ms=max_audio_length_ms,
                         temperature=temperature, topk=topk)
    a = audio.detach().float().cpu().numpy()
    a = normalize_loudness(a, gen.sample_rate, target_lufs=target_lufs)
    return a, gen.sample_rate


def synth_stream(text: str, voice: Optional[str] = None, *, max_audio_length_ms: int = 40_000,
                 temperature: float = 0.9, topk: int = 50, chunk_frames: int = 25,
                 target_lufs: Optional[float] = DEFAULT_TARGET_LUFS,
                 seed: Optional[int] = None) -> Iterator[tuple[np.ndarray, int]]:
    """Streaming synthesis. Yields (float32 mono chunk, sample_rate) per chunk.

    Note: per-chunk loudness normalization is applied with a fixed peak ceiling
    only (no per-chunk LUFS retarget, which would pump level between chunks);
    full-clip LUFS targeting is available via synth().
    """
    gen = get_generator()
    v = _get_voice(voice)
    if seed is not None:
        torch.manual_seed(seed)
    for chunk in gen.generate_stream(text=text, speaker=v.speaker, context=list(v.context),
                                     max_audio_length_ms=max_audio_length_ms,
                                     temperature=temperature, topk=topk, chunk_frames=chunk_frames):
        a = chunk.detach().float().cpu().numpy()
        a = normalize_loudness(a, gen.sample_rate, target_lufs=None)  # peak ceiling only
        yield a, gen.sample_rate
