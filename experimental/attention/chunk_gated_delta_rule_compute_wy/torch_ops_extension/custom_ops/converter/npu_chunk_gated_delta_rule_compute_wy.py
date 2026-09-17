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
@register_fx_node_ge_converter(
    torch.ops.custom.npu_chunk_gated_delta_rule_compute_wy.default
)
def convert_npu_chunk_gated_delta_rule_compute_wy(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    chunk_size: int,
    meta_outputs: List[TensorSpec] = None,
):
    return torchair.ge.custom_op(
        "ChunkGatedDeltaRuleComputeWy",
        inputs={
            "q": q,
            "k": k,
            "v": v,
            "g": g,
            "beta": beta,
        },
        attrs={
            "chunk_size": attr.Int(chunk_size),
        },
        outputs=["q_kernel", "k_kernel", "w_kernel", "u_kernel", "g_kernel"],
    )
