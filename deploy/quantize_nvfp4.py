"""Produce an nvfp4 (NVIDIA FP4) quantized checkpoint of MisoTTS, PyTorch-native.

nvfp4 = 4-bit float (E2M1) with a per-16-element FP8 (E4M3) block scale plus a
per-tensor FP32 scale. This script PACKS the weights into that format; the
packing is pure numerics and runs on ANY GPU (the A6000 dev box is fine). Only
EXECUTING fp4 matmuls needs Blackwell (sm_100+), so the produced checkpoint is
built here and loaded/run on a B200.

STACK REQUIREMENT (not a GPU requirement): nvfp4 packing needs a recent torch
(>= 2.7, for the float4_e2m1 dtype) and a recent torchao (nvfp4/mx_formats). Our
default inference stack is torch 2.4 / torchao 0.9 and CANNOT do this, so run
this script inside the Blackwell image (deploy/Dockerfile.blackwell), which
carries the newer stack. Building the MisoTTS model there also depends on a
torchtune that works with torch 2.7+ (the porting risk flagged in the README).

Usage (inside the Blackwell-stack environment):
  python deploy/quantize_nvfp4.py --out /workspace/weights/model_nvfp4.pt

This is a SKELETON: the exact torchao nvfp4 config import/name is version
specific, and none of this has been validated (no Blackwell hardware or
compatible stack on the dev box). Pin the torchao version and confirm the API
when we build on the B200.
"""
from __future__ import annotations

import argparse
import sys

import torch


def _load_model():
    """Build the MisoTTS model and load bf16 weights (needs torchtune)."""
    sys.path.insert(0, "/opt/miso")
    from generator import load_miso_8b
    # Load on the available GPU at bf16; quantization reads these weights.
    gen = load_miso_8b("cuda", dtype=torch.bfloat16)
    return gen


def _apply_nvfp4(model) -> None:
    """Quantize the Linear weights to nvfp4 in place.

    torchao exposes nvfp4 via its mx/float-formats. The config name has moved
    across versions; current candidates (confirm at build time):
      from torchao.quantization import quantize_
      from torchao.prototype.mx_formats import NVFP4InferenceConfig   # newer
      quantize_(model, NVFP4InferenceConfig())
    Weight-only nvfp4 is the conservative first target (activations stay bf16);
    full nvfp4 (weights + activations) is the bigger win but needs calibration
    and more validation. Skip embeddings, norms, and the small audio heads.
    """
    from torchao.quantization import quantize_  # noqa: F401  (presence check)
    try:
        from torchao.prototype.mx_formats import NVFP4InferenceConfig as _Cfg  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "torchao nvfp4 config not found; pin a torchao version that ships "
            f"nvfp4/mx_formats and update the import. ({exc})"
        )

    def _is_quant_target(mod, name):
        import torch.nn as nn
        return isinstance(mod, nn.Linear) and "audio_head" not in name and "head" not in name

    quantize_(model, _Cfg(), filter_fn=_is_quant_target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output nvfp4 checkpoint path")
    args = ap.parse_args()

    gen = _load_model()
    model = gen._model
    print("[nvfp4] applying weight-only nvfp4 quantization to Linear layers...", flush=True)
    _apply_nvfp4(model)
    # Persist. torchao tensor subclasses serialize through torch.save; the
    # inference loader must import torchao before torch.load to deserialize them.
    torch.save(model.state_dict(), args.out)
    print(f"[nvfp4] wrote {args.out}", flush=True)
    print("[nvfp4] NOTE: load + execute on Blackwell (sm_100+). Packing done here; "
          "fp4 matmuls require Blackwell tensor cores.", flush=True)


if __name__ == "__main__":
    main()
