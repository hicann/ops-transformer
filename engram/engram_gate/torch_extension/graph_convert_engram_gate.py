# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TorchAir converter for the three-input EngramGate IR."""

import torch
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.compat_ir import ge_op
from torchair.ge import attr


@register_fx_node_ge_converter(torch.ops.cann_ops_transformer.engram_gate.default)
def convert_engram_gate(
    hidden_states,
    kv,
    weight_fused,
    eps=1e-6,
    clamp_value=1e-6,
    save_for_backward=True,
    meta_outputs=None,
):
    outputs = ge_op(
        op_type="EngramGate",
        inputs={"hidden_states": hidden_states, "kv": kv, "weight_fused": weight_fused},
        attrs={
            "eps": attr.Float(eps),
            "clamp_value": attr.Float(clamp_value),
            "save_for_backward": attr.Bool(save_for_backward),
        },
        outputs=["output", "dot", "gate_score", "rstd_x", "rstd_k"],
    )
    return outputs if save_for_backward else (outputs[0], None, None, None, None)
