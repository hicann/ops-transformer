# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


SMLA_METADATA_SIZE = 64
SMLA_METADATA_OP_NAME = "sparse_flash_mla_softmax_l1_norm_metadata"
SMLA_OP_NAME = "sparse_flash_mla_softmax_l1_norm"


class SparseFlashMlaSoftmaxL1NormOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseFlashMlaSoftmaxL1NormOpBuilder, self).__init__(
            "sparse_flash_mla_softmax_l1_norm", category="attention"
        )

    def sources(self):
        return ["csrc/attention/sparse_flash_mla_softmax_l1_norm.cpp"]

    def schema(self) -> list:
        return [
            "sparse_flash_mla_softmax_l1_norm_metadata("
            "int num_heads_q, int num_heads_k, int head_dim, *, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, "
            "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, Tensor? topk_length=None, "
            "int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_k=None, int? topk=None, "
            "str? layout_q=None, str? layout_k=None, int? mask_mode=None, "
            "int? cmp_ratio=None) -> Tensor",
            "sparse_flash_mla_softmax_l1_norm("
            "Tensor q, Tensor k, Tensor softmax_lse, *, "
            "Tensor? sparse_indices=None, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, "
            "Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, "
            "Tensor? topk_length=None, Tensor? metadata=None, "
            "float softmax_scale=1.0, int max_seqlen_k=0, int cmp_ratio=1, int mask_mode=0, "
            'str layout_q="BSND", str layout_k="BSND") -> Tensor',
        ]

    def register_meta(self):
        @torch.library.register_fake("cann_ops_transformer::" + SMLA_METADATA_OP_NAME)
        def sparse_flash_mla_softmax_l1_norm_metadata_meta(
            num_heads_q,
            num_heads_k,
            head_dim,
            *,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            topk_length=None,
            batch_size=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            topk=None,
            layout_q=None,
            layout_k=None,
            mask_mode=None,
            cmp_ratio=None,
        ):
            return torch.empty((SMLA_METADATA_SIZE,), dtype=torch.int32, device="meta")

        @impl(get_as_library(), self.name, "Meta")
        def sparse_flash_mla_softmax_l1_norm_meta(
            q,
            k,
            softmax_lse,
            *,
            sparse_indices=None,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            topk_length=None,
            metadata=None,
            softmax_scale=1.0,
            max_seqlen_k=0,
            cmp_ratio=1,
            mask_mode=0,
            layout_q="BSND",
            layout_k="BSND",
        ):
            if layout_q == "BSND":
                b = q.size(0)
                s1 = q.size(1)
                s2 = k.size(1)
                out_shape = (b, s1, s2)
            else:
                s1 = q.size(0)
                s2 = k.size(0)
                out_shape = (s1, s2)
            return torch.empty(out_shape, dtype=torch.float32, device="meta")


sparse_flash_mla_softmax_l1_norm_op_builder = SparseFlashMlaSoftmaxL1NormOpBuilder()
sparse_flash_mla_softmax_l1_norm_op_builder._ensure_initialized()


@impl(get_as_library(), SMLA_METADATA_OP_NAME, "PrivateUse1")
def sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    topk_length=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    topk=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None,
):
    """SparseFlashMlaSoftmaxL1Norm 的前置 metadata 算子，封装 aclnnSparseFlashMlaSoftmaxL1NormMetadata。

    根据主算子的 shape、layout、mask 等信息，计算分核负载均衡 metadata，
    输出固定 shape (64,) 的 int32 张量，供主算子 metadata 输入使用。

    Args:
        num_heads_q (int): q 的 head 个数，必须为正数且能被 num_heads_k 整除。
        num_heads_k (int): k 的 head 个数，必须为正数。
        head_dim (int): q/k 的 head dimension，必须为正数。
        cu_seqlens_q (Tensor, optional): TND 场景下 q 的累积序列长度，shape (B+1,)，dtype int32。
        cu_seqlens_k (Tensor, optional): TND 场景下 k 的累积序列长度，shape (B+1,)，dtype int32。
        seqused_q (Tensor, optional): 每个 batch 中 q 实际参与运算的序列长度，shape (B,)，dtype int32。
        seqused_k (Tensor, optional): 预留参数，每个 batch 中 k 实际参与运算的序列长度。
        cmp_residual_k (Tensor, optional): 预留参数，key 序列长度与 cmpRatio 相关的残差。
        topk_length (Tensor, optional): 每行 q 对应的 k 实际可选的 topk 长度。
        batch_size (int, optional): batch 数量，TND 场景可填 None 自动推导。
        max_seqlen_q (int, optional): q 的最大序列长度，BSND 场景必须为正数。
        max_seqlen_k (int, optional): k 的最大序列长度，BSND 场景必须为正数。
        topk (int, optional): 从 k 中筛选的关键 token 个数，0 表示无稀疏。
        layout_q (str, optional): q 排列格式，支持 "BSND"/"TND"，默认 "BSND"。
        layout_k (str, optional): k 排列格式，支持 "BSND"/"TND"，默认 "BSND"。
        mask_mode (int, optional): mask 模式，0=No mask，3=Causal，默认 0。
        cmp_ratio (int, optional): k 压缩比，取值 [1, 128]，默认 1。

    Returns:
        Tensor: shape (64,) 的 int32 张量，前 5 个字段为负载均衡信息。
    """
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = 0 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_k = 0 if max_seqlen_k is None else max_seqlen_k
    topk = 0 if topk is None else topk
    layout_q = "BSND" if layout_q is None else layout_q
    layout_k = "BSND" if layout_k is None else layout_k
    mask_mode = 0 if mask_mode is None else mask_mode
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio

    op_module = sparse_flash_mla_softmax_l1_norm_op_builder.load()
    return op_module.sparse_flash_mla_softmax_l1_norm_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        topk_length,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        topk,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
    )


@torch.library.register_kernel("cann_ops_transformer::" + SMLA_METADATA_OP_NAME, None)
def sparse_flash_mla_softmax_l1_norm_metadata_fallback(
    num_heads_q,
    num_heads_k,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    topk_length=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    topk=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None,
):
    return sparse_flash_mla_softmax_l1_norm_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        topk_length=topk_length,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        topk=topk,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
    )


torch.compiler.allow_in_graph(sparse_flash_mla_softmax_l1_norm_metadata)


@impl(get_as_library(), SMLA_OP_NAME, "PrivateUse1")
def sparse_flash_mla_softmax_l1_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    softmax_lse: torch.Tensor,
    *,
    sparse_indices: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    max_seqlen_k: int = 0,
    cmp_ratio: int = 1,
    mask_mode: int = 0,
    layout_q: str = "BSND",
    layout_k: str = "BSND",
) -> torch.Tensor:
    """SparseFlashMlaSoftmaxL1Norm 前向计算，封装 aclnnSparseFlashMlaSoftmaxL1Norm。

    计算 SparseFlashMla 注意力的 Softmax L1Norm 结果，支持 Sliding Window Attention、
    Compressed Attention 以及 Sparse Compressed Attention 场景。

    Args:
        q (Tensor): 查询张量，shape (B,S1,N1,D) 或 (T1,N1,D)，dtype float16/bfloat16。
        k (Tensor): 键张量，shape (B,S2,N2,D) 或 (T2,N2,D)，dtype 与 q 一致，N2=1，D=512。
        softmax_lse (Tensor): 注意力正向计算的 softmax_lse 输出，shape (B,N2,S1,G) 或 (N2,T1,G)，dtype float32。
        sparse_indices (Tensor, optional): 稀疏场景下选择的注意力索引，dtype int32。
        cu_seqlens_q (Tensor, optional): TND 场景下 q 的累积序列长度，shape (B+1,)，dtype int32。
        cu_seqlens_k (Tensor, optional): TND 场景下 k 的累积序列长度，shape (B+1,)，dtype int32。
        seqused_q (Tensor, optional): 每个 batch 中 q 实际参与运算的 token 数，shape (B,)，dtype int32。
        seqused_k (Tensor, optional): 每个 batch 中 k 实际参与运算的 token 数，shape (B,)，dtype int32。
        cmp_residual_k (Tensor, optional): 每个 batch S2//cmpRatio 后的余数，shape (B,)，dtype int32。
        topk_length (Tensor, optional): 每行 q 对应的 k 实际可选的 topk 长度，dtype int32。
        metadata (Tensor, optional): 前置 AICPU 算子输出的 tiling metadata，shape (64,)，dtype int32。
        softmax_scale (float): 缩放系数，默认 1.0。
        max_seqlen_k (int): k 的最大序列长度，TND dense 场景用于输出 shape 推导，默认 0。
        cmp_ratio (int): k 的压缩率，取值 [1, 128]，默认 1。
        mask_mode (int): mask 模式，0=No mask，3=rightDownCausal，默认 0。
        layout_q (str): q 排列格式，支持 "BSND"/"TND"，默认 "BSND"。
        layout_k (str): k 排列格式，支持 "BSND"/"TND"，默认 "BSND"。

    Returns:
        Tensor: softmax L1Norm 结果，shape (B,S1,N2,S2) 或 (T1,N2,T2)，dtype float32。
    """
    op_module = sparse_flash_mla_softmax_l1_norm_op_builder.load()
    return op_module.sparse_flash_mla_softmax_l1_norm(
        q,
        k,
        softmax_lse,
        sparse_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        topk_length,
        metadata,
        softmax_scale,
        max_seqlen_k,
        cmp_ratio,
        mask_mode,
        layout_q,
        layout_k,
    )


@torch.library.register_kernel("cann_ops_transformer::" + SMLA_OP_NAME, None)
def sparse_flash_mla_softmax_l1_norm_fallback(
    q,
    k,
    softmax_lse,
    *,
    sparse_indices=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    topk_length=None,
    metadata=None,
    softmax_scale=1.0,
    max_seqlen_k=0,
    cmp_ratio=1,
    mask_mode=0,
    layout_q="BSND",
    layout_k="BSND",
):
    return sparse_flash_mla_softmax_l1_norm(
        q,
        k,
        softmax_lse,
        sparse_indices=sparse_indices,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        topk_length=topk_length,
        metadata=metadata,
        softmax_scale=softmax_scale,
        max_seqlen_k=max_seqlen_k,
        cmp_ratio=cmp_ratio,
        mask_mode=mask_mode,
        layout_q=layout_q,
        layout_k=layout_k,
    )


torch.compiler.allow_in_graph(sparse_flash_mla_softmax_l1_norm)
