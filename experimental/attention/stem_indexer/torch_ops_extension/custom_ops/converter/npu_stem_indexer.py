# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import (
    List,
    Optional,
)

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair.ge import attr


# 为自定义算子注册converter，用于torch.compile 场景成图
# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.custom.npu_stem_indexer.default)
def convert_npu_stem_indexer(
    qflat: Tensor,
    kflat: Tensor,
    vbias: Tensor,
    q_seq_lens: Tensor,
    kv_seq_lens: Tensor,
    *,
    num_prompt_tokens: Optional[Tensor] = None,
    metadata: Optional[Tensor] = None,
    causal: bool = True,
    stem_block_size: int = 128,
    stem_stride: int = 16,
    alpha: float = 1.0,
    initial_blocks: int = 4,
    window_size: int = 4,
    k_block_num_rate_medium: float = 0.2,
    k_block_num_bias_medium: int = 30,
    k_block_num_rate_large: float = 0.1,
    k_block_num_bias_large: int = 30,
    topk_score_precision: int = 1,
    meta_outputs: List[TensorSpec] = None,
):
    return torchair.ge.custom_op(
        "StemIndexer",
        inputs={
            "qflat": qflat,
            "kflat": kflat,
            "vbias": vbias,
            "q_seq_lens": q_seq_lens,
            "kv_seq_lens": kv_seq_lens,
            "num_prompt_tokens": num_prompt_tokens,
            "metadata": metadata,
        },
        attrs={
            "causal": attr.Bool(causal),
            "stem_block_size": attr.Int(stem_block_size),
            "stem_stride": attr.Int(stem_stride),
            "alpha": attr.Float(alpha),
            "initial_blocks": attr.Int(initial_blocks),
            "window_size": attr.Int(window_size),
            "k_block_num_rate_medium": attr.Float(k_block_num_rate_medium),
            "k_block_num_bias_medium": attr.Int(k_block_num_bias_medium),
            "k_block_num_rate_large": attr.Float(k_block_num_rate_large),
            "k_block_num_bias_large": attr.Int(k_block_num_bias_large),
            "topk_score_precision": attr.Int(topk_score_precision),
        },
        outputs=["sparse_indices", "sparse_seq_len"],
    )
