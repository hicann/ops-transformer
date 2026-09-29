# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional

import torch
import torch_npu
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class EngramGateOpBuilder(OpBuilder):
    def __init__(self):
        super(EngramGateOpBuilder, self).__init__("ds41.engram_gate", category="engram")

    def sources(self):
        """Python DSL implementation; no C++ extension is compiled."""
        return []

    def schema(self):
        """PyTorch operator signature."""
        return [
            (
                "ds41.engram_gate(Tensor x, Tensor key, Tensor value, Tensor weight, "
                "Tensor? image_mask=None, *, float eps=1e-06, float clamp_value=1e-06) "
                "-> Tensor"
            ),
        ]

    def register_meta(self):
        """Register the FakeTensor/Meta output implementation."""

        @impl(get_as_library(), self.name, "Meta")
        def engram_gate_meta(
            x,
            key,
            value,
            weight,
            image_mask=None,
            *,
            eps=1.0e-6,
            clamp_value=1.0e-6,
        ):
            return torch.empty_like(x)


engram_gate_op_builder = EngramGateOpBuilder()
engram_gate_op_builder._ensure_initialized()

# Imported at module load, not inside the impl.  ``ops/__init__.py`` registers
# the packaged ``_native`` directory; dynamo traces a function-local import as a
# direct module load and skips that, so a capture-first ACLGraph path would
# miss every precompiled ``.so`` and either JIT (prefer) or raise (require).
from ops.engram_gate import engram_gate as _engram_gate_impl  # noqa: E402


@impl(get_as_library(), engram_gate_op_builder.name, "PrivateUse1")
def _engram_gate(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    image_mask: Optional[torch.Tensor] = None,
    *,
    eps: float = 1.0e-6,
    clamp_value: float = 1.0e-6,
) -> torch.Tensor:
    """Run the CANNBotDSL kernel through the Torch NPU dispatcher."""
    return _engram_gate_impl(
        x, key, value, weight, image_mask, eps=eps, clamp_value=clamp_value
    )


def engram_gate_torch(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    image_mask: Optional[torch.Tensor] = None,
    *,
    eps: float = 1.0e-6,
    clamp_value: float = 1.0e-6,
) -> torch.Tensor:
    """Run the Engram residual gate operator on NPU tensors.

    Same semantics as ``ops.engram_gate.engram_gate``: fused dual-path RMS
    normalization, weighted dot product, signed-sqrt sigmoid gating and
    residual update.  Tokens selected by ``image_mask`` return ``x`` as is.

    Args:
        x: Residual input in ``[..., hc_mult, dim]`` BF16 layout, ndim >= 2.
        key: Gate key with the same shape and dtype as ``x``.
        value: Gate value in ``[..., dim]`` BF16 layout sharing ``x`` leading
            dims.
        weight: Per-head dot weight in ``[hc_mult, dim]`` FP32 layout.
        image_mask: Optional bool tensor with shape ``x.shape[:-2]``; ``True``
            tokens are passed through unchanged.
        eps: RMS epsilon; default ``1e-6``.
        clamp_value: Magnitude floor before the signed sqrt; default ``1e-6``.

    Returns:
        BF16 tensor with the same shape as ``x``.
    """
    return torch.ops.cann_ops_transformer.ds41.engram_gate(
        x, key, value, weight, image_mask, eps=eps, clamp_value=clamp_value
    )
