"""Create and upload the pre-quantized MisoTTS variant repos to HuggingFace.

Run on the HOST (where huggingface-cli is authenticated as a BigBlueCeiling
member), after deploy/build_quant_checkpoint.py has produced the model.pt files.

  set HF_HUB_ENABLE_HF_TRANSFER=1
  .venv/Scripts/python.exe deploy/upload_quant_repos.py [int8 int4]

Uploads <ckpt_dir>/<scheme>/model.pt -> BigBlueCeiling/MisoTTS-<scheme>/model.pt
and a model card. The serving core (deploy/miso_server/core.py) pulls these
automatically when GPU-sense picks the matching VRAM tier.
"""
import os
import sys

from huggingface_hub import HfApi

CKPT_DIR = os.environ.get("MISO_CKPT_DIR", os.path.join("output", "ckpt"))
ORG = os.environ.get("MISO_HF_ORG", "BigBlueCeiling")

# Measured on an A6000 (see PERFORMANCE_PROGRESS.md). CER/WER/UTMOS are means over
# the 12 canonical EVAL_PROMPTS scored with perceval; bf16 reference CER 0.10 /
# WER 0.15 / UTMOS 3.94.
CARDS = {
    "int8": dict(
        fits="~16 GB VRAM cards (RTX 4060 Ti 16G, 4070 Ti Super, A4000, ...)",
        quality="Quality-preserving: mean CER 0.11, WER 0.14, UTMOS 3.96 - "
                "statistically even with bf16 (CER 0.10 / WER 0.15 / UTMOS 3.94).",
        warn="Experimental. Weight-only int8; bf16 remains the reference.",
    ),
    "int4": dict(
        fits="~12 GB VRAM cards (RTX 3060 12G, 4070, ...)",
        quality="Noticeably degraded: mean CER 0.18, WER 0.26, UTMOS 2.93 "
                "(vs bf16 UTMOS 3.94). Worst on long utterances (long-clip CER up "
                "to ~0.5). Acceptable only as a last-resort 'runs at all' tier.",
        warn="EXPERIMENTAL and lower quality. Use int8 or bf16 if your card fits. "
             "This file is tinygemm-packed on an Ampere (sm_86) GPU; the serving "
             "core falls back to quantizing the bf16 weights at load if it does not "
             "load on your architecture.",
    ),
}


def card(scheme: str) -> str:
    c = CARDS[scheme]
    return f"""---
license: other
base_model: BigBlueCeiling/MisoTTS-bf16
tags:
- text-to-speech
- quantized
- torchao
---

# MisoTTS {scheme} (BigBlueCeiling)

A weight-only **{scheme}** quantization of
[BigBlueCeiling/MisoTTS-bf16](https://huggingface.co/BigBlueCeiling/MisoTTS-bf16),
produced with torchao (`{scheme}_weight_only`). Only the backbone/decoder Linear
layers are quantized; the embeddings, output heads, and projection stay bf16.

> {c['warn']}

## What it is for

Lowering the hardware floor. Quantization here is a **memory** lever, not a speed
one: MisoTTS decodes one frame at a time, and those tiny per-step matmuls cannot
feed the GPU's low-precision tensor cores, so {scheme} dequantizes to bf16 for the
matmul. You get the VRAM saving, not a throughput win.

- **Fits:** {c['fits']}
- **Quality:** {c['quality']}

## Use

This checkpoint is a `torch.save`'d torchao state_dict (`model.pt`). The serving
core in the [MisoTTS repo](https://github.com/eoffermann/MisoTTS) pulls it
automatically when GPU-sense detects a matching VRAM tier. To load it directly:

```python
from generator import load_miso_8b  # from the MisoTTS repo
gen = load_miso_8b("cuda", model_path_or_repo_id="{ORG}/MisoTTS-{scheme}",
                   prequantized=True)
```

Requires torch>=2.7 and a matching torchao (loading unpickles a torchao tensor
subclass, so `weights_only=False` is used; load only checkpoints you trust).

Model and original inference code are MisoLabs' work; see the upstream license.
"""


def main():
    schemes = sys.argv[1:] or ["int8", "int4"]
    api = HfApi()
    for scheme in schemes:
        pt = os.path.join(CKPT_DIR, scheme, "model.pt")
        if not os.path.isfile(pt):
            print(f"[upload] SKIP {scheme}: {pt} missing", flush=True)
            continue
        rid = f"{ORG}/MisoTTS-{scheme}"
        size = os.path.getsize(pt) / 1e9
        print(f"[upload] {rid} <- {pt} ({size:.1f} GB)", flush=True)
        api.create_repo(rid, repo_type="model", exist_ok=True, private=False)
        api.upload_file(path_or_fileobj=card(scheme).encode(), path_in_repo="README.md",
                        repo_id=rid, repo_type="model")
        api.upload_file(path_or_fileobj=pt, path_in_repo="model.pt",
                        repo_id=rid, repo_type="model")
        print(f"[upload] done: https://huggingface.co/{rid}", flush=True)


if __name__ == "__main__":
    main()
