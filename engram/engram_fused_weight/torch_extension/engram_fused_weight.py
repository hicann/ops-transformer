# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class EngramFusedWeightOpBuilder(OpBuilder):
    """EngramFusedWeight 的 PyTorch 扩展构建器。

    输入范围约束:
      - 两个 2D 同 shape 的 BF16/FP16 权重 [hcMult, hiddenSize];
      - 总元素数 % 256 == 0 (0 允许);
      - 输出为同 shape 的 FP32 张量: y = fp32(x1) * fp32(x2)。
    """

    def __init__(self):
        super(EngramFusedWeightOpBuilder, self).__init__(
            "engram_fused_weight", category="engram"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/engram/engram_fused_weight.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "engram_fused_weight(Tensor weight_hidden, Tensor weight_embed) -> Tensor"
        )

    def register_meta(self):
        """
        Registers the Meta implementation.
        """

        @impl(get_as_library(), self.name, "Meta")
        def engram_fused_weight_meta(
            weight_hidden,
            weight_embed,
        ) -> torch.Tensor:
            return torch.empty_like(weight_hidden, dtype=torch.float32)


# Instantiate the builder
engram_fused_weight_op_builder = EngramFusedWeightOpBuilder()
engram_fused_weight_op_builder._ensure_initialized()


@impl(get_as_library(), engram_fused_weight_op_builder.name, "PrivateUse1")
def _engram_fused_weight(
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
) -> torch.Tensor:
    op_module = engram_fused_weight_op_builder.load()
    return op_module.engram_fused_weight(weight_hidden, weight_embed)


def engram_fused_weight(
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
) -> torch.Tensor:
    """将两个 BF16/FP16 Engram 权重逐元素相乘并返回 FP32 张量。

    ``weight_hidden`` / ``weight_embed`` 须为同 shape 的二维 NPU 张量
    ``(hc_mult, hidden_size)``; 总元素数须能被 256 整除 (0 允许)。
    """
    return torch.ops.cann_ops_transformer.engram_fused_weight(
        weight_hidden,
        weight_embed,
    )
