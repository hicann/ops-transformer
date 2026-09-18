# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

try:
    from typing import List

    import torch
    import torchair
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge import attr
    from torchair.ge._ge_graph import Tensor, TensorSpec

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.kda_input_proj.default
    )
    def convert_kda_input_proj(
        x: Tensor,
        weight_qkv: Tensor,
        weight_beta: Tensor,
        weight_gate: Tensor,
        weight_g: Tensor,
        weight_qkv_scale: Tensor,
        *,
        meta_outputs: List[TensorSpec] = None,
    ):
        return torchair.ge.custom_op(
            "KdaInputProj",
            inputs={
                "x": x,
                "weight_qkv": weight_qkv,
                "weight_beta": weight_beta,
                "weight_gate": weight_gate,
                "weight_g": weight_g,
                "weight_qkv_scale": weight_qkv_scale,
            },
            attrs={
                "trans_weight_qkv": attr.Bool(False),
                "trans_weight_beta": attr.Bool(False),
                "trans_weight_gate": attr.Bool(False),
                "trans_weight_g": attr.Bool(False),
            },
            outputs=["qkv", "beta", "gate", "g"],
        )
