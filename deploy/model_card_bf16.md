---
license: other
base_model: MisoLabs/MisoTTS
pipeline_tag: text-to-speech
tags:
- text-to-speech
- prosody
---

# MisoTTS bf16 (BigBlueCeiling)

Full-precision (bfloat16) weights for MisoTTS, the **reference** variant in
BigBlueCeiling's optimization- and deployment-focused fork of
[MisoLabsAI/MisoTTS](https://github.com/MisoLabsAI/MisoTTS). The model and the
original inference code are MisoLabs' work; this fork makes it fast and correct in
practice and easy to run across a range of hardware.

MisoTTS is an expressive, English, ~8B-parameter text-to-speech model: a
Llama-3.2-style backbone generates Mimi audio codes from text, a smaller
autoregressive decoder predicts the higher codebooks per frame, and the output is
watermarked with SilentCipher.

## Variant family

This bf16 repo is the reference and the default. The serving core reads the GPU's
VRAM and loads the highest-quality weight precision that fits, pulling it at
runtime:

| Variant | Weights | Fits (gen peak) | Quality vs bf16 |
|---|---|---|---|
| **bf16** (this repo) | bfloat16 | ~24 GB (A6000, 3090/4090, A100, ...) | reference |
| [int8](https://huggingface.co/BigBlueCeiling/MisoTTS-int8) | int8 weight-only | ~16 GB (4060 Ti 16G, 4070 Ti S, A4000) | even (CER/WER/UTMOS ~unchanged) |
| [int4](https://huggingface.co/BigBlueCeiling/MisoTTS-int4) | int4 weight-only | ~12 GB (3060 12G, 4070) | noticeably lower (experimental) |

int8/int4 are weight-only quantizations of these bf16 weights. They are a **memory**
lever, not a speed one (the frame-by-frame decode cannot feed the GPU's
low-precision tensor cores, so they dequantize to bf16 for the matmul). bf16 is
both the quality reference and the fastest path on a card that fits it.

## Quality and performance

Measured on an A6000 over the 12 canonical eval prompts (3 lengths x 4 emotions),
scored with perceval: mean ASR **CER 0.10, WER 0.15, UTMOS 3.94**. With
`torch.compile` (reduce-overhead) generation runs near realtime (**RTF ~1.1** after
warmup); eager is roughly 14x slower. The compile warmup caches across processes,
so a persisted Inductor cache brings cold start to a few minutes.

## Use

```python
import torch, torchaudio
from generator import load_miso_8b  # from the MisoTTS repo

gen = load_miso_8b("cuda")  # GPU-sense pulls this bf16 repo on a card that fits it
audio = gen.generate(text="Hello from Miso.", speaker=0, context=[],
                     max_audio_length_ms=10_000)
torchaudio.save("miso.wav", audio.unsqueeze(0).cpu(), gen.sample_rate)
```

Requires torch>=2.7. See the [MisoTTS repo](https://github.com/eoffermann/MisoTTS)
for the serving container (RunPod and OpenAI-compatible APIs), the GPU-sense variant
selection, and the quality harness.

## Safety, license, credit

Generated audio is watermarked with SilentCipher; if you deploy the model, use your
own private watermark key and keep it secret. Do not use the model to impersonate
people, create deceptive audio, or generate harmful content. The model and the
original inference code are MisoLabs' work, under the upstream license; see
[MisoLabsAI/MisoTTS](https://github.com/MisoLabsAI/MisoTTS).
