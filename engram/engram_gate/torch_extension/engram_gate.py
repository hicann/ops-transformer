# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyTorch registration and invocation entry for EngramGate."""

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class EngramGateOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("engram_gate", category="engram")

    def sources(self):
        return ["csrc/engram/engram_gate.cpp"]

    def schema(self):
        return (
            "engram_gate(Tensor hidden_states, Tensor kv, Tensor weight_fused, "
            "float eps=1e-6, float clamp_value=1e-6, bool save_for_backward=True) "
            "-> (Tensor, Tensor?, Tensor?, Tensor?, Tensor?)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def meta(
            hidden_states,
            kv,
            weight_fused,
            eps=1e-6,
            clamp_value=1e-6,
            save_for_backward=True,
        ):
            torch._check(hidden_states.dim() == 3)
            torch._check(hidden_states.shape[1] == 4)
            torch._check(hidden_states.shape[2] > 0)
            torch._check(hidden_states.shape[2] % 64 == 0)
            y = torch.empty_like(hidden_states)
            if not save_for_backward:
                return y, None, None, None, None
            return (y,) + tuple(
                torch.empty(
                    hidden_states.shape[:2],
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                for _ in range(4)
            )


engram_gate_op_builder = EngramGateOpBuilder()
engram_gate_op_builder._ensure_initialized()


@impl(get_as_library(), engram_gate_op_builder.name, "PrivateUse1")
def _forward(
    hidden_states, kv, weight_fused, eps=1e-6, clamp_value=1e-6, save_for_backward=True
):
    return engram_gate_op_builder.load().engram_gate(
        hidden_states, kv, weight_fused, eps, clamp_value, save_for_backward
    )


def engram_gate(
    hidden_states, kv, weight_fused, eps=1e-6, clamp_value=1e-6, save_for_backward=True
):
    return torch.ops.cann_ops_transformer.engram_gate(
        hidden_states, kv, weight_fused, eps, clamp_value, save_for_backward
    )
