# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import List

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair.ge import attr


# 为自定义算子注册converter，用于torch.compile 场景成图
# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.custom.npu_fused_gdn_decode.default)
def convert_npu_fused_gdn_decode(
    mixed_qkv: Tensor,
    a: Tensor,
    b: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    state_ref: Tensor,
    ssm_state_indices: Tensor,
    scale: float,
    *,
    softplus_threshold: float = 20.0,
    meta_outputs: List[TensorSpec] = None,
):
    out, _ = torchair.ge.custom_op(
        "FusedGdnDecode",
        inputs={
            "mixed_qkv": mixed_qkv,
            "a": a,
            "b": b,
            "a_log": a_log,
            "dt_bias": dt_bias,
            "state": state_ref,
            "ssm_state_indices": ssm_state_indices,
        },
        attrs={
            "scale": attr.Float(scale),
            "softplus_threshold": attr.Float(softplus_threshold),
        },
        outputs=["out", "state_out"],
    )
    return out
