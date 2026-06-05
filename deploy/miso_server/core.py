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
        # Note: model._model._backbone_prefill (a 1-element list set in Model.__init__)
        # already holds the EAGER backbone; compiling only swaps self.backbone, so the
        # prefill keeps routing to the eager module with no extra bookkeeping here.
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


def _pick_quant_by_vram(vram_gb: float) -> Optional[str]:
    """Choose the runtime weight-quant tier that fits this GPU's VRAM.

    Quantization for MisoTTS is purely a MEMORY lever (see
    generator._quantize_and_move): the frame-by-frame decode's tiny per-step
    matmuls (M=1 backbone, M=10 decoder) cannot feed the hardware low-precision
    GEMMs (int8 `_int_mm`, fp8/fp4 `_scaled_mm` all require M>=16), so dynamic /
    activation quant and hardware fp8/nvfp4 give NO speed win here and are not used.
    Even weight-only is not a bandwidth win: re-measured on the modern compile-
    capable stack (torch 2.7 + torchao 0.13, the combo that motivated the upgrade),
    int8 weight-only + torch.compile streams at RTF ~2.75 vs bf16+compile ~1.49 -
    the fused path still dequantizes to bf16 at M=1 (more HBM traffic), it does not
    read int8. So int8/int4 are used ONLY to fit a smaller card, never for speed.
    We pick the highest-quality precision that fits: bf16 on big cards, then int8,
    then int4 (both weight-only, dequant-to-bf16, runnable on ANY GPU). Measured
    peaks (eager): bf16 ~17-20 GB, int8 ~10-12 GB, int4 ~6-8 GB; the defaults leave
    headroom for torch.compile's CUDA-graph pools. Thresholds are env-tunable.
    """
    bf16_min = float(os.environ.get("MISO_BF16_MIN_GB", "22"))
    int8_min = float(os.environ.get("MISO_INT8_MIN_GB", "13"))
    if vram_gb >= bf16_min:
        return None
    if vram_gb >= int8_min:
        return "int8"
    return "int4"


def detect_device_profile() -> dict:
    """Detect the GPU and the runtime weight-quant tier that fits its VRAM.

    arch is reported for logging only; selection is purely VRAM-driven, because
    quantization here is a memory lever, not a speed one (see _pick_quant_by_vram).
      blackwell sm>=100 (B100/B200, RTX 50xx)
      hopper    sm 90   (H100)
      ada       sm 89   (RTX 40, L40)
      ampere    sm 80/86/87 (A100, A6000, 30x)
    """
    if not torch.cuda.is_available():
        return {"arch": "cpu", "cc": 0, "name": "cpu", "vram_gb": 0.0, "quant": None}
    major, minor = torch.cuda.get_device_capability(0)
    cc = major * 10 + minor
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    arch = ("blackwell" if cc >= 100 else "hopper" if cc >= 90 else
            "ada" if cc == 89 else "ampere" if cc >= 80 else
            "turing" if cc >= 75 else "legacy")
    return {"arch": arch, "cc": cc, "name": name, "vram_gb": vram,
            "quant": _pick_quant_by_vram(vram)}


# HuggingFace distribution: the weights live in ONE bf16 repo, pulled at runtime
# (NOT baked into the image). int8/int4 are produced by quantizing those bf16
# weights at load (see generator._quantize_and_move) rather than shipped as
# separate checkpoints: torchao 0.13 has no portable safetensors serialization for
# quantized tensors, and int4's tinygemm layout is architecture-specific, so a
# distributed quantized checkpoint would be fragile. Env-overridable.
VARIANT_REPO = {
    "bf16": os.environ.get("MISO_REPO_BF16", "BigBlueCeiling/MisoTTS-bf16"),
    # Pre-quantized variants (torch.save'd torchao state_dicts, model.pt). Pulled
    # when GPU-sense picks int8/int4; if a repo is absent we fall back to the bf16
    # weights and quantize at load.
    "int8": os.environ.get("MISO_REPO_INT8", "BigBlueCeiling/MisoTTS-int8"),
    "int4": os.environ.get("MISO_REPO_INT4", "BigBlueCeiling/MisoTTS-int4"),
}


def resolve_model_source(variant: str = "bf16"):
    """bf16 checkpoint source passed to load_miso_8b.

    Precedence: explicit MISO_TTS_8B_MODEL override -> the bf16 HF repo (downloaded
    to a local path) -> None (load_miso_8b uses the upstream default).
    hf_hub_download caches, so repeated starts do not re-download.
    """
    explicit = os.environ.get("MISO_TTS_8B_MODEL")
    if explicit:
        return explicit
    repo = VARIANT_REPO.get(variant)
    if repo:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=repo, filename="model.safetensors")
            print(f"[core] pulled bf16 weights from {repo}", flush=True)
            return path
        except Exception as exc:
            print(f"[core] {repo} not available ({type(exc).__name__}); "
                  f"falling back to the default model", flush=True)
    return None


def resolve_prequant_source(scheme: str):
    """Pull the pre-quantized model.pt for `scheme` (int8/int4) from its HF repo.

    Returns a local path, or None if no repo / the file is unavailable / the user
    pinned an explicit bf16 model (then the caller quantizes the bf16 weights at
    load instead). hf_hub_download caches across starts.
    """
    if os.environ.get("MISO_TTS_8B_MODEL"):
        return None  # explicit bf16 override -> quantize that at load, do not pull
    repo = VARIANT_REPO.get(scheme)
    if not repo:
        return None
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, filename="model.pt")
        print(f"[core] pulled pre-quantized {scheme} from {repo}", flush=True)
        return path
    except Exception as exc:
        print(f"[core] pre-quantized {scheme} repo {repo} unavailable "
              f"({type(exc).__name__}); will quantize bf16 at load", flush=True)
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
        # Runtime weight-quant tier. Explicit MISO_QUANTIZE overrides the VRAM
        # auto-pick; "" / "none" / "bf16" force full bf16. int8/int4 quantize the
        # bf16 weights layer-wise AT LOAD (generator._quantize_and_move) so a small
        # card never has to hold the full bf16 model.
        override = os.environ.get("MISO_QUANTIZE")
        if override is not None:
            override = override.strip().lower()
            quant = None if override in ("", "none", "bf16") else override
        else:
            quant = prof["quant"]
        if quant not in (None, "int8", "int4"):
            print(f"[core] unknown quant '{quant}'; using bf16.", flush=True)
            quant = None
        if prof["arch"] != "cpu":
            tier = quant or "bf16"
            note = " (memory-fit; experimental quant)" if quant else ""
            print(f"[core] GPU {prof['name']} arch={prof['arch']} sm_{prof['cc']} "
                  f"vram={prof['vram_gb']:.0f}GB -> weight precision '{tier}'{note}.",
                  flush=True)
        t0 = time.perf_counter()
        # Prefer a pre-quantized checkpoint (smaller download, no load-time quant).
        # Fall back to bf16-quantized-at-load if the repo is absent OR the
        # pre-quantized load fails -- e.g. the int4 tinygemm layout is packed for
        # the build GPU's architecture and may not load on a different arch, whereas
        # quant-at-load re-packs correctly for whatever GPU is present.
        gen = None
        prequant = resolve_prequant_source(quant) if quant else None
        if prequant:
            try:
                gen = load_miso_8b(device, model_path_or_repo_id=prequant, prequantized=True)
            except Exception as exc:
                print(f"[core] pre-quantized {quant} load failed "
                      f"({type(exc).__name__}: {exc}); quantizing bf16 at load", flush=True)
                gen = None
        if gen is None:
            source = resolve_model_source("bf16")
            gen = load_miso_8b(device, model_path_or_repo_id=source, dtype=torch.bfloat16,
                               quantize=quant)
        _maybe_compile(gen)
        print(f"[core] model loaded on {device} in {time.perf_counter() - t0:.1f}s", flush=True)
        _GEN = gen
        return _GEN


def compile_status(gen) -> str:
    """ACTIVE if backbone AND decoder are torch.compile OptimizedModules, else EAGER.

    torch.compile wraps a module in an OptimizedModule; after a successful warmup
    that also means the reduce-overhead CUDA graphs were captured (warmup would have
    raised otherwise). This is the definitive, unambiguous regime label so a
    benchmark is never silently an eager baseline.
    """
    m = getattr(gen, "_model", None)
    bb = type(getattr(m, "backbone", None)).__name__
    dc = type(getattr(m, "decoder", None)).__name__
    return "ACTIVE" if (bb == "OptimizedModule" and dc == "OptimizedModule") else "EAGER"


def warmup() -> None:
    """Run a few generations so torch.compile graphs are captured before serving.

    MISO_COMPILE_STRICT=1 makes a compile / CUDA-graph failure RAISE instead of
    silently reverting to eager, so benchmarks and deploy smoke tests fail loudly
    rather than measuring an eager regime by accident. Default OFF in serving
    (graceful eager fallback for resilience); the benchmark harness turns it ON.
    """
    gen = get_generator()
    requested = os.environ.get("MISO_COMPILE", "0") == "1"
    strict = os.environ.get("MISO_COMPILE_STRICT", "0") == "1"
    texts = ("Warming up.", "This is a slightly longer warmup utterance for the decoder.")
    try:
        for text in texts:
            _ = gen.generate(text=text, speaker=0, context=[], max_audio_length_ms=6000)
    except Exception as exc:  # pragma: no cover
        # Most likely a compile / CUDA-graph failure on the torchtune decode path.
        if requested and strict:
            raise RuntimeError(
                f"compiled warmup failed and MISO_COMPILE_STRICT=1 (would have reverted "
                f"to eager and silently lost the compile win): {type(exc).__name__}: {exc}"
            ) from exc
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
    status = compile_status(gen)
    if requested and status != "ACTIVE":
        print("[core] WARNING: MISO_COMPILE=1 requested but compile=EAGER (reverted); "
              "set MISO_COMPILE_STRICT=1 to fail instead of silently degrading", flush=True)
    print(f"[core] warmup complete; compile={status} (MISO_COMPILE={int(requested)})", flush=True)


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


def _stream_env(name, default, cast):
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return cast(val)
    except (TypeError, ValueError):
        return default


def synth_stream(text: str, voice: Optional[str] = None, *, max_audio_length_ms: int = 40_000,
                 temperature: float = 0.9, topk: int = 50, chunk_frames: Optional[int] = None,
                 start_frames: Optional[int] = None, ramp: Optional[float] = None,
                 rtf_adapt: Optional[bool] = None, wm_defer_ms: Optional[float] = None,
                 target_lufs: Optional[float] = DEFAULT_TARGET_LUFS,
                 seed: Optional[int] = None) -> Iterator[tuple[np.ndarray, int]]:
    """Streaming synthesis. Yields (float32 mono chunk, sample_rate) per chunk.

    Emits on a low-latency ramp by default (first chunk = one 80 ms frame, then
    growing up to a cap), so every streaming caller gets a low TTFB without asking.
    Ramp parameters resolve as explicit arg > MISO_STREAM_* env > default:
      chunk_frames / MISO_STREAM_MAX_FRAMES  cap (default 25),
      start_frames / MISO_STREAM_START_FRAMES first emit in frames (default 1),
      ramp / MISO_STREAM_RAMP                growth factor (default 2.0),
      rtf_adapt / MISO_STREAM_RTF_ADAPT      jump to cap when RTF > 1 (default off),
      wm_defer_ms / MISO_WM_DEFER_MS         skip watermarking the leading ms (default 0).

    Per-chunk loudness uses a fixed peak ceiling only (no per-chunk LUFS retarget,
    which would pump level between chunks); full-clip LUFS is available via synth().
    """
    gen = get_generator()
    v = _get_voice(voice)
    if seed is not None:
        torch.manual_seed(seed)
    cap = chunk_frames if chunk_frames is not None else _stream_env("MISO_STREAM_MAX_FRAMES", 25, int)
    start = start_frames if start_frames is not None else _stream_env("MISO_STREAM_START_FRAMES", 1, int)
    rmp = ramp if ramp is not None else _stream_env("MISO_STREAM_RAMP", 2.0, float)
    adapt = rtf_adapt if rtf_adapt is not None else bool(_stream_env("MISO_STREAM_RTF_ADAPT", 0, int))
    wm = wm_defer_ms if wm_defer_ms is not None else _stream_env("MISO_WM_DEFER_MS", 0.0, float)
    for chunk in gen.generate_stream(text=text, speaker=v.speaker, context=list(v.context),
                                     max_audio_length_ms=max_audio_length_ms,
                                     temperature=temperature, topk=topk,
                                     max_frames=cap, start_frames=start, ramp=rmp,
                                     rtf_adapt=adapt, wm_defer_ms=wm):
        a = chunk.detach().float().cpu().numpy()
        a = normalize_loudness(a, gen.sample_rate, target_lufs=None)  # peak ceiling only
        yield a, gen.sample_rate
