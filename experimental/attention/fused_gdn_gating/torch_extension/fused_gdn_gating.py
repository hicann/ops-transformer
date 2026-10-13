# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class FusedGdnGatingOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("fused_gdn_gating", category="attention")

    def sources(self):
        return ["csrc/attention/fused_gdn_gating.cpp"]

    def schema(self) -> str:
        return (
            "fused_gdn_gating(Tensor a_log, Tensor a, Tensor b, Tensor dt_bias, "
            "float beta=1.0, float threshold=20.0) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def fused_gdn_gating_meta(
            a_log: torch.Tensor,
            a: torch.Tensor,
            b: torch.Tensor,
            dt_bias: torch.Tensor,
            beta: float = 1.0,
            threshold: float = 20.0,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            del a_log, dt_bias, beta, threshold
            output_shape = (1, a.size(0), a.size(1))
            g = torch.empty(output_shape, dtype=torch.float32, device="meta")
            beta_output = torch.empty(output_shape, dtype=b.dtype, device="meta")
            return g, beta_output


_fused_gdn_gating_op_builder = FusedGdnGatingOpBuilder()
_fused_gdn_gating_op_builder._ensure_initialized()


@impl(get_as_library(), _fused_gdn_gating_op_builder.name, "PrivateUse1")
def _fused_gdn_gating(
    a_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    op_module = _fused_gdn_gating_op_builder.load()
    return op_module.fused_gdn_gating(a_log, a, b, dt_bias, beta, threshold)


def fused_gdn_gating(
    a_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the fused gating calculation used by Gated Delta Networks.

    Args:
        a_log: Log-domain decay parameter with shape ``[num_heads]``.
        a: Softplus gating input with shape ``[batch, num_heads]`` and dtype
            ``float16`` or ``bfloat16``.
        b: Sigmoid gating input with the same shape and dtype as ``a``.
        dt_bias: Bias with the same shape and dtype as ``a_log``.
        beta: Softplus beta value. Defaults to ``1.0``.
        threshold: Softplus linear approximation threshold. Defaults to ``20.0``.

    Returns:
        A tuple containing ``g`` with shape ``[1, batch, num_heads]`` and
        dtype ``float32``, and ``beta_output`` with the same shape and dtype
        as ``b``.
    """
    return torch.ops.cann_ops_transformer.fused_gdn_gating(
        a_log, a, b, dt_bias, beta, threshold
    )
