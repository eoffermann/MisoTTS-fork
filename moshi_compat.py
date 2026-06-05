from __future__ import annotations

import torch
from torch import nn

from moshi.utils import quantize


def patch_bitsandbytes_import_for_unquantized_layers() -> None:
    # moshi 0.2.2 exposed module-level `quantize.linear` / `quantize.multi_linear`
    # helpers that, even for UNquantized modules, routed through bitsandbytes; this
    # shim replaced them with plain F.linear so an unquantized Mimi did not require
    # bitsandbytes to be installed.
    #
    # moshi 0.2.13 reworked the module: those helpers are gone, the unquantized
    # path uses nn.Linear directly, and bitsandbytes is imported lazily only inside
    # QLinear (instantiated by replace_linear_with_qlinear, which MisoTTS never
    # calls -- Mimi runs in bf16). So on the newer moshi there is nothing to patch
    # and bitsandbytes is not needed at all. Guard the patch on the old API so this
    # is a no-op there while staying correct on 0.2.2.
    if not (hasattr(quantize, "linear") and hasattr(quantize, "multi_linear")):
        return

    original_linear = quantize.linear
    original_multi_linear = quantize.multi_linear

    def linear(module: nn.Module, x: torch.Tensor, name: str = "weight") -> torch.Tensor:
        if quantize.is_quantized(module, name):
            return original_linear(module, x, name)
        return nn.functional.linear(x, getattr(module, name))

    def multi_linear(
        num_steps: int,
        schedule: list[int] | None,
        module: nn.Module,
        x: torch.Tensor,
        offset: int,
        name: str = "weight",
    ) -> torch.Tensor:
        if quantize.is_quantized(module, name):
            return original_multi_linear(num_steps, schedule, module, x, offset, name)

        weight = getattr(module, name)
        num_linear = num_steps if schedule is None else max(schedule) + 1
        weight = weight.view(num_linear, -1, weight.shape[-1])

        outputs = []
        for t in range(x.shape[1]):
            linear_index = t + offset
            if schedule is not None:
                linear_index = schedule[linear_index]
            outputs.append(nn.functional.linear(x[:, t], weight[linear_index]))
        return torch.stack(outputs, 1)

    quantize.linear = linear
    quantize.multi_linear = multi_linear
