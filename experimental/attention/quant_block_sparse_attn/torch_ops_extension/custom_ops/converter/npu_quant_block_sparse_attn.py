# copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
import torchair
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import (
    Device,
    Number,
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
)
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import (
    declare_supported,
    register_fx_node_ge_converter,
)
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import (
    _TypedTensor,
    F32,
    F16,
    F64,
    I32,
    I16,
    I64,
    I8,
    U8,
    BOOL,
    Support,
)
from torchair.ge import attr


# 为自定义算子注册converter，用于torch.compile 场景成图
# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.custom.npu_quant_block_sparse_attn.default)
def convert_npu_quant_block_sparse_attn(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    p_scale: Optional[Tensor],
    sparse_indices: Tensor,
    sparse_seq_len: Tensor,
    atten_mask: Optional[Tensor],
    softmax_scale: float,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    *,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    seqused_q: Optional[Tensor] = None,
    seqused_kv: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    metadata: Optional[Tensor] = None,
    layout_kv: str = "PA_BNBD",
    layout_q: str = "TND",
    layout_sparse_indices: str = "B_N_Qb_Kb",
    layout_out: str = "TND",
    quant_mode: int = 1,
    mask_mode: int = 3,
    return_softmax_lse: bool = False,
    meta_outputs: List[TensorSpec] = None,
):
    return torchair.ge.custom_op(
        "QuantBlockSparseAttn",
        inputs={
            "query": query,
            "key": key,
            "value": value,
            "q_descale": q_descale,
            "k_descale": k_descale,
            "v_descale": v_descale,
            "p_scale": p_scale,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_kv": cu_seqlens_kv,
            "seqused_q": seqused_q,
            "seqused_kv": seqused_kv,
            "sparse_indices": sparse_indices,
            "sparse_seq_len": sparse_seq_len,
            "block_table": block_table,
            "atten_mask": atten_mask,
            "metadata": metadata,
        },
        attrs={
            "softmax_scale": attr.Float(softmax_scale),
            "sparse_q_block_size": attr.Int(sparse_q_block_size),
            "sparse_kv_block_size": attr.Int(sparse_kv_block_size),
            "layout_kv": attr.Str(layout_kv),
            "layout_q": attr.Str(layout_q),
            "layout_sparse_indices": attr.Str(layout_sparse_indices),
            "layout_out": attr.Str(layout_out),
            "quant_mode": attr.Int(quant_mode),
            "mask_mode": attr.Int(mask_mode),
            "return_softmax_lse": attr.Bool(return_softmax_lse),
        },
        outputs=["attention_out", "softmax_lse"],
    )
