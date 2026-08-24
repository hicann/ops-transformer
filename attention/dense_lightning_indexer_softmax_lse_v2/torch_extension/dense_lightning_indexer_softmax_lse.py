# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional, Tuple

import torch
import torch_npu
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


DLI_METADATA_SIZE = 64
DLI_METADATA_OP_NAME = "dense_lightning_indexer_softmax_lse_metadata"
DLI_OP_NAME = "dense_lightning_indexer_softmax_lse"


class DenseLightningIndexerSoftmaxLseOpBuilder(OpBuilder):
    def __init__(self):
        super(DenseLightningIndexerSoftmaxLseOpBuilder, self).__init__(
            "dense_lightning_indexer_softmax_lse", category="attention"
        )

    def sources(self):
        return ["csrc/attention/dense_lightning_indexer_softmax_lse.cpp"]

    def schema(self) -> str:
        return [
            "dense_lightning_indexer_softmax_lse_metadata("
            "int num_heads_q, int num_heads_k, int head_dim, *, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, "
            "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, int? batch_size=None, "
            "int? max_seqlen_q=None, int? max_seqlen_k=None, "
            "str? layout_q=None, str? layout_k=None, int? mask_mode=None, "
            "int? cmp_ratio=None) -> Tensor",
            "dense_lightning_indexer_softmax_lse("
            "Tensor query_index, Tensor key_index, Tensor weight, *, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, "
            "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, Tensor? metadata=None, "
            'str layout_q="BSND", str layout_k="BSND", int mask_mode=0, '
            "int cmp_ratio=1) -> Tensor",
        ]

    def register_meta(self):
        @torch.library.register_fake("cann_ops_transformer::" + DLI_METADATA_OP_NAME)
        def dense_lightning_indexer_softmax_lse_metadata_meta(
            num_heads_q,
            num_heads_k,
            head_dim,
            *,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            batch_size=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            layout_q=None,
            layout_k=None,
            mask_mode=None,
            cmp_ratio=None,
        ):
            return torch.empty((DLI_METADATA_SIZE,), dtype=torch.int32, device="meta")

        @impl(get_as_library(), self.name, "Meta")
        def dense_lightning_indexer_softmax_lse_meta(
            query_index,
            key_index,
            weight,
            *,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            metadata=None,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=0,
            cmp_ratio=1,
        ):
            if layout_q == "BSND":
                out_shape = (
                    query_index.shape[0],
                    key_index.shape[2],
                    query_index.shape[1],
                )
            else:
                out_shape = (key_index.shape[1], query_index.shape[0])
            return torch.empty(out_shape, dtype=torch.float32, device="meta")


dense_lightning_indexer_softmax_lse_op_builder = (
    DenseLightningIndexerSoftmaxLseOpBuilder()
)


@impl(get_as_library(), DLI_METADATA_OP_NAME, "PrivateUse1")
def dense_lightning_indexer_softmax_lse_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None,
):
    """DenseLightningIndexerSoftmaxLseV2 的前置 metadata 算子，封装 aclnnDenseLightningIndexerSoftmaxLseV2Metadata。

    根据主算子的 shape、layout、mask 等信息，计算分核负载均衡 metadata，
    输出固定 shape (64,) 的 int32 张量，供主算子 metadata 输入使用。

    Args:
        num_heads_q (int): query 的 head 个数，必须为正数且能被 num_heads_k 整除。
        num_heads_k (int): key 的 head 个数，必须为正数。
        head_dim (int): q/k 的 head dimension，必须为正数。
        cu_seqlens_q (Tensor, optional): TND 场景下 query 的累积序列长度，shape (B+1,)，dtype int32。
        cu_seqlens_k (Tensor, optional): TND 场景下 key 的累积序列长度，shape (B+1,)，dtype int32。
        seqused_q (Tensor, optional): 每个 batch 中 query 实际参与运算的序列长度，shape (B,)，dtype int32。
        seqused_k (Tensor, optional): 预留参数，每个 batch 中 key 实际参与运算的序列长度。
        cmp_residual_k (Tensor, optional): 预留参数，key 序列长度与 cmpRatio 相关的残差。
        batch_size (int, optional): batch 数量，TND 场景可填 None 自动推导。
        max_seqlen_q (int, optional): query 的最大序列长度，BSND 场景必须为正数。
        max_seqlen_k (int, optional): key 的最大序列长度，BSND 场景必须为正数。
        layout_q (str, optional): query 排列格式，支持 "BSND"/"TND"，默认 "BSND"。
        layout_k (str, optional): key 排列格式，支持 "BSND"/"TND"，默认 "BSND"。
        mask_mode (int, optional): mask 模式，0=No mask，3=Causal，默认 0。
        cmp_ratio (int, optional): key 压缩比，取值 [1, 128]，默认 1。

    Returns:
        Tensor: shape (64,) 的 int32 张量，前 5 个字段为负载均衡信息。
    """
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = 0 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_k = 0 if max_seqlen_k is None else max_seqlen_k
    layout_q = "BSND" if layout_q is None else layout_q
    layout_k = "BSND" if layout_k is None else layout_k
    mask_mode = 0 if mask_mode is None else mask_mode
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio

    op_module = dense_lightning_indexer_softmax_lse_op_builder.load()
    return op_module.dense_lightning_indexer_softmax_lse_v2_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
    )


@torch.library.register_kernel("cann_ops_transformer::" + DLI_METADATA_OP_NAME, None)
def dense_lightning_indexer_softmax_lse_metadata_fallback(
    num_heads_q,
    num_heads_k,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None,
):
    return dense_lightning_indexer_softmax_lse_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
    )


torch.compiler.allow_in_graph(dense_lightning_indexer_softmax_lse_metadata)


@impl(get_as_library(), DLI_OP_NAME, "PrivateUse1")
def dense_lightning_indexer_softmax_lse(
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weight: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    layout_q: str = "BSND",
    layout_k: str = "BSND",
    mask_mode: int = 0,
    cmp_ratio: int = 1,
) -> torch.Tensor:
    """DenseLightningIndexerSoftmaxLseV2 前向算子，封装 aclnnDenseLightningIndexerSoftmaxLseV2。

    计算 Lightning Indexer 分支的 Softmax LogSumExp 值。接收 query、key 和 weight 三个输入，
    先通过 BatchMatmul 计算 query 与 key 的内积得分，经 ReLU 激活后与 weight 逐元素加权，
    再对 head 维做 ReduceSum，最后通过数值稳定的 LogSumExp 算法输出 softmax_lse 值。

    Args:
        query_index (Tensor): query 输入，BSND shape (B,S1,N1,D)，TND shape (T1,N1,D)，
            dtype 支持 float16/bfloat16/float32。
        key_index (Tensor): key 输入，BSND shape (B,S2,1,D)，TND shape (T2,1,D)，dtype 与 query 一致。
        weight (Tensor): 权重，BSND shape (B,S1,N1)，TND shape (T1,N1)，dtype 为 float32。
        cu_seqlens_q (Tensor, optional): TND 场景下 query 的累积序列长度，shape (B+1,)，dtype int32。
        cu_seqlens_k (Tensor, optional): TND 场景下 key 的累积序列长度，shape (B+1,)，dtype int32。
        seqused_q (Tensor, optional): 每个 batch 中 query 实际参与运算的序列长度，shape (B,)，dtype int32。
        seqused_k (Tensor, optional): 预留参数，每个 batch 中 key 实际参与运算的序列长度。
        cmp_residual_k (Tensor, optional): maskMode=3 且 cmpRatio≠1 时需要传入。
        metadata (Tensor, optional): 前置 metadata 算子输出的分核信息，shape (64,)，dtype int32。
        layout_q (str): query 排列格式，支持 "BSND"/"TND"，默认 "BSND"。
        layout_k (str): key 排列格式，支持 "BSND"/"TND"，默认 "BSND"。
        mask_mode (int): mask 模式，0=defaultMask，3=rightDownCausal，默认 0。
        cmp_ratio (int): 压缩比，取值 [1, 128]，默认 1。

    Returns:
        Tensor: softmax_lse 输出，BSND shape (B,S1)，TND shape (T1,)，dtype 为 float32。
    """
    op_module = dense_lightning_indexer_softmax_lse_op_builder.load()
    return op_module.dense_lightning_indexer_softmax_lse_v2(
        query_index,
        key_index,
        weight,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        metadata,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
    )


@torch.library.register_kernel("cann_ops_transformer::" + DLI_OP_NAME, None)
def dense_lightning_indexer_softmax_lse_fallback(
    query_index,
    key_index,
    weight,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    metadata=None,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
):
    return dense_lightning_indexer_softmax_lse(
        query_index,
        key_index,
        weight,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        metadata=metadata,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
    )


torch.compiler.allow_in_graph(dense_lightning_indexer_softmax_lse)


dense_lightning_indexer_softmax_lse_op_builder = (
    DenseLightningIndexerSoftmaxLseOpBuilder()
)
dense_lightning_indexer_softmax_lse_op_builder._ensure_initialized()
