"""Build a pre-quantized MisoTTS checkpoint (int8 or int4) for distribution.

Loads the bf16 weights, weight-only quantizes the backbone/decoder Linears with
torchao (frugal, layer-wise via generator._load_model), and torch.save's the
resulting state_dict (AffineQuantizedTensor weights + bf16 embeddings/heads) as
model.pt for upload to BigBlueCeiling/MisoTTS-<scheme>. Because _load_model
returns the model BEFORE Generator.setup_caches, the file holds exactly the 367
model params (no causal-mask / KV buffers), loadable by
load_miso_8b(prequantized=True).

  docker run --rm --gpus all -v <repo>:/opt/miso -v <hf>:/workspace/hf \
    -v <out>:/workspace/out -e MISO_QSCHEME=int8 \
    miso-tts:modern python /opt/miso/deploy/build_quant_checkpoint.py
"""
import os
import sys
import time

sys.path.insert(0, "/opt/miso")
import torch

from generator import _load_model, load_miso_8b
from models import MISO_TTS_8B_CONFIG
from perf_eval.prompts import EVAL_PROMPTS

SCHEME = os.environ.get("MISO_QSCHEME", "int8")
OUT = os.environ.get("MISO_OUT_DIR", f"/workspace/out/ckpt/{SCHEME}")
os.makedirs(OUT, exist_ok=True)


def log(m):
    print(f"[ckpt:{SCHEME}] {m}", flush=True)


def main():
    src = os.environ.get("MISO_TTS_8B_MODEL") or os.environ.get(
        "MISO_REPO_BF16", "BigBlueCeiling/MisoTTS-bf16")
    log(f"loading bf16 from {src}, weight-only quantizing to {SCHEME}...")
    t0 = time.perf_counter()
    model = _load_model(src, MISO_TTS_8B_CONFIG, "cuda", torch.bfloat16, quantize=SCHEME)
    log(f"quantized in {time.perf_counter() - t0:.0f}s")

    # Move the state_dict to CPU for a portable, device-independent file. int4's
    # tinygemm layout is CUDA-packed; .cpu() should still move the stored data even
    # though the kernel is CUDA-only. Warn if anything resists the move.
    sd = model.state_dict()
    cpu_sd, cuda_kept = {}, 0
    for k, v in sd.items():
        try:
            cpu_sd[k] = v.detach().cpu()
        except Exception as exc:
            log(f"  keep on CUDA ({k}): {type(exc).__name__}")
            cpu_sd[k] = v.detach()
            cuda_kept += 1
    if cuda_kept:
        log(f"WARNING: {cuda_kept} tensors stayed on CUDA (file will be CUDA-bound)")

    out = os.path.join(OUT, "model.pt")
    torch.save(cpu_sd, out)
    log(f"saved {len(cpu_sd)} tensors -> {out} ({os.path.getsize(out) / 1e9:.2f} GB)")

    # Free the build model before the round-trip reload so peak memory stays sane.
    del model, sd, cpu_sd
    torch.cuda.empty_cache()

    log("round-trip: reload prequantized + render one canonical clip...")
    gen = load_miso_8b("cuda", model_path_or_repo_id=out, prequantized=True)
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated() / 1e9
    pid, text, maxms = EVAL_PROMPTS[0]
    torch.manual_seed(1234)
    audio = gen.generate(text=text, speaker=0, context=[], max_audio_length_ms=maxms)
    import torchaudio
    torchaudio.save(os.path.join(OUT, f"roundtrip_{pid}.wav"),
                    audio.unsqueeze(0).cpu(), gen.sample_rate)
    log(f"round-trip OK: reloaded resident {resident:.2f} GB, generated "
        f"{audio.shape[-1] / gen.sample_rate:.2f}s for '{pid}' "
        f"(\"{text[:40]}...\")")


if __name__ == "__main__":
    main()
