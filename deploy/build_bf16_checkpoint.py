"""Build a native-bf16 checkpoint of MisoTTS for the HF variant repo.

Loads the upstream model (cast to bf16 on load) and writes its persistent
state_dict as a ~16 GB bf16 safetensors named model.safetensors, so the bf16
variant repo (BigBlueCeiling/MisoTTS-bf16) carries half the download of the fp32
original. Reduces cold-start download/read on a fresh box. Runs on any GPU.

Usage (in container, writing to a mounted host dir):
  docker run --rm --gpus all -v <hf>:/workspace/hf -v <out>:/workspace/out \
    miso-tts:latest python /opt/miso/deploy/build_bf16_checkpoint.py
Then upload model.safetensors to BigBlueCeiling/MisoTTS-bf16 from the host
(where huggingface-cli is authenticated).
"""
import os
import sys
import time

sys.path.insert(0, "/opt/miso")

import torch
from safetensors.torch import save_file

from generator import load_miso_8b


def main():
    out_dir = os.environ.get("MISO_OUT_DIR", "/workspace/out")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "model.safetensors")

    t0 = time.perf_counter()
    gen = load_miso_8b("cuda", dtype=torch.bfloat16)
    print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

    # Persistent state_dict in bf16, contiguous, on CPU for serialization.
    # Exclude the causal-mask buffers that Model.setup_caches registers AFTER
    # construction: they are not in the upstream checkpoint, so including them
    # makes a strict load_state_dict reject the variant with "unexpected keys".
    SKIP = ("causal_mask",)
    sd = {}
    for k, v in gen._model.state_dict().items():
        if any(s in k for s in SKIP):
            continue
        sd[k] = v.detach().to("cpu", torch.bfloat16).contiguous()
    nbytes = sum(t.numel() * t.element_size() for t in sd.values())
    print(f"[state_dict] {len(sd)} tensors, {nbytes/1e9:.1f} GB bf16", flush=True)

    save_file(sd, out, metadata={"format": "pt", "dtype": "bfloat16",
                                 "source": "MisoLabs/MisoTTS"})
    print(f"[saved] {out} ({os.path.getsize(out)/1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
