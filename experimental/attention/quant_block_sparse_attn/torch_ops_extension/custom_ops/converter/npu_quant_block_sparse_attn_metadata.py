# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import List, Optional

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair.ge import attr


@register_fx_node_ge_converter(
    torch.ops.custom.npu_quant_block_sparse_attn_metadata.default
)
def convert_npu_quant_block_sparse_attn_metadata(
    sparse_seq_len: Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    *,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    seqused_q: Optional[Tensor] = None,
    seqused_kv: Optional[Tensor] = None,
    batch_size: int = 0,
    sparse_block_size_q: int = 128,
    sparse_block_size_k: int = 128,
    quant_mode: int = 1,
    mask_mode: int = 3,
    layout_q: str = "TND",
    layout_kv: str = "PA_BNBD",
    layout_sparse_indices: str = "B_N_Qb_Kb",
    meta_outputs: List[TensorSpec] = None,
):
    stream_info = torch.npu.get_stream_limit(torch.npu.current_stream())
    aic_core_num = stream_info.get("cube_core_num")
    aiv_core_num = stream_info.get("vector_core_num")
    soc_version = torch.npu.get_device_properties().name
    return torchair.ge.custom_op(
        "QuantBlockSparseAttnMetadata",
        inputs={
            "sparse_seq_len": sparse_seq_len,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_kv": cu_seqlens_kv,
            "seqused_q": seqused_q,
            "seqused_kv": seqused_kv,
        },
        attrs={
            "batch_size": attr.Int(batch_size),
            "num_heads_q": attr.Int(num_heads_q),
            "num_heads_kv": attr.Int(num_heads_kv),
            "head_dim": attr.Int(head_dim),
            "sparse_block_size_q": attr.Int(sparse_block_size_q),
            "sparse_block_size_k": attr.Int(sparse_block_size_k),
            "quant_mode": attr.Int(quant_mode),
            "mask_mode": attr.Int(mask_mode),
            "layout_q": attr.Str(layout_q),
            "layout_kv": attr.Str(layout_kv),
            "layout_sparse_indices": attr.Str(layout_sparse_indices),
            "soc_version": attr.Str(soc_version),
            "aic_core_num": attr.Int(aic_core_num),
            "aiv_core_num": attr.Int(aiv_core_num),
        },
        outputs=["metadata"],
    )
