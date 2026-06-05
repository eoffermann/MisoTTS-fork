"""Task 3 probe: what does the model emit AFTER the first all-zero EOS frame?

If post-EOS frames stay all-zero (silence), periodic EOS checking (sync every N
frames) is safe - the overshoot is trimmable silence. If they drift to non-zero
(garbage), overshoot would be audible and Task 3 must not over-generate.

Replicates the generate_frame loop WITHOUT the EOS break and prints the all-zero
status of frames around the first EOS. Eager (MISO_COMPILE=0).
"""
import sys

sys.path.insert(0, "/opt/miso")
sys.path.insert(0, "/opt/miso/deploy")

import torch

from miso_server import core


def main():
    gen = core.get_generator()
    m = gen._model
    dev = gen.device
    text = "I just feel so empty inside today, like nothing matters anymore."
    torch.manual_seed(1234)
    with torch.inference_mode():
        pt, ptm, mgl = gen._prepare_prompt(text, 0, [], 20000)
        curr_tokens = pt.unsqueeze(0)
        curr_tokens_mask = ptm.unsqueeze(0)
        curr_pos = torch.arange(0, pt.size(0)).unsqueeze(0).long().to(dev)

        rows = []
        first_eos = None
        post = 0
        for i in range(int(mgl)):
            sample = m.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, 0.9, 50)
            z = bool((sample == 0).all().item())
            nz = int((sample != 0).sum().item())
            rows.append((i, z, nz))
            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(dev)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(dev)], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1
            if first_eos is not None:
                post += 1
                if post >= 15:
                    break
            elif z:
                first_eos = i

    print(f"total frames generated: {len(rows)}; first EOS at frame {first_eos}", flush=True)
    print("frames around / after first EOS:", flush=True)
    for i, z, nz in rows:
        if first_eos is None or i >= first_eos - 2:
            print(f"  frame {i:3d}: all_zero={z}  nonzero_codes={nz}", flush=True)
    if first_eos is not None:
        post_rows = [r for r in rows if r[0] > first_eos]
        all_silent = all(z for _, z, _ in post_rows)
        print(f"\nVERDICT: {len(post_rows)} post-EOS frames, all_zero={all_silent} "
              f"-> overshoot is {'SILENCE (safe to trim)' if all_silent else 'NON-ZERO (garbage; do not over-generate)'}",
              flush=True)


if __name__ == "__main__":
    main()
