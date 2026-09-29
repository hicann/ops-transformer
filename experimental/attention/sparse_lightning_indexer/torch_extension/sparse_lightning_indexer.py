# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

LI_METADATA_SIZE = 1024
LI_METADATA_OP_NAME = "lightning_indexer_metadata"


class SparseLightningIndexerOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseLightningIndexerOpBuilder, self).__init__(
            "sparse_lightning_indexer", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/sparse_lightning_indexer.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "sparse_lightning_indexer(Tensor q, Tensor k, Tensor w, int topk, Tensor candidate_topk_indices, "
            "*, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, "
            "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, Tensor? block_table=None, "
            "Tensor? output_idx_offset=None, Tensor? metadata=None, int max_seqlen_q=-1, "
            'str layout_q="BSND", str layout_k="BSND", int mask_mode=0, int cmp_ratio=1, '
            "Tensor? candidate_block_length=None, int candidate_block_size=8) -> (Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(get_as_library(), self.name, "Meta")
        def sparse_lightning_indexer_meta(
            q,
            k,
            w,
            topk,
            candidate_topk_indices,
            *,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            block_table=None,
            output_idx_offset=None,
            metadata=None,
            max_seqlen_q=-1,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=0,
            cmp_ratio=1,
            candidate_block_length=None,
            candidate_block_size=8,
        ):
            key_head_num = k.shape[1] if layout_k == "TND" else k.shape[2]
            if layout_q == "BSND":
                sparse_indices_out = torch.empty(
                    [q.shape[0], q.shape[1], key_head_num, topk],
                    dtype=torch.int32,
                    device="meta",
                )
            else:
                sparse_indices_out = torch.empty(
                    [q.shape[0], key_head_num, topk], dtype=torch.int32, device="meta"
                )
            # return_value 固定 0（C6），sparse_values 恒空占位
            sparse_values_out = torch.empty([0], dtype=torch.float, device="meta")
            return (sparse_indices_out, sparse_values_out)


# Instantiate the builder
sparse_lightning_indexer_op_builder = SparseLightningIndexerOpBuilder()
sparse_lightning_indexer_op_builder._ensure_initialized()


def _sparse_li_check_args(
    topk, candidate_topk_indices, candidate_block_length, candidate_block_size
):
    """封装层断言（蓝图 §4.3 契约的执行点）：host C1-C7 的前置镜像。"""
    if not torch.is_tensor(candidate_topk_indices):
        raise TypeError(
            "candidate_topk_indices must be a Tensor (REQUIRED, int32, "
            "BSND [B,S1,N2,candBlocks] / TND [T,N2,candBlocks])"
        )
    if candidate_topk_indices.dtype != torch.int32:
        raise TypeError(
            "candidate_topk_indices must be int32, but got "
            f"{candidate_topk_indices.dtype}"
        )
    cand_blocks = int(candidate_topk_indices.shape[-1])
    if not (0 < cand_blocks <= 2048 and cand_blocks % 64 == 0):
        raise ValueError(
            "The last dim of candidate_topk_indices must be in (0, 2048] and "
            f"a multiple of 64, but got {cand_blocks}"
        )
    if not (0 < int(topk) <= 2048):
        raise ValueError(f"topk must be in (0, 2048], but got {topk}")
    if not (
        candidate_block_size >= 2
        and candidate_block_size <= 64
        and (candidate_block_size & (candidate_block_size - 1)) == 0
    ):
        raise ValueError(
            "candidate_block_size must be a power of 2 in [2, 64], but got "
            f"{candidate_block_size}"
        )
    # C7：candidate_block_length 预留，仅接受 None/空 tensor
    if candidate_block_length is not None and (
        not torch.is_tensor(candidate_block_length)
        or candidate_block_length.numel() != 0
    ):
        raise ValueError(
            "candidate_block_length is reserved and only empty tensor is supported yet"
        )


@impl(get_as_library(), sparse_lightning_indexer_op_builder.name, "PrivateUse1")
def sparse_lightning_indexer(
    q,
    k,
    w,
    topk,
    candidate_topk_indices,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    candidate_block_length=None,
    candidate_block_size=8,
):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    _sparse_li_check_args(
        topk, candidate_topk_indices, candidate_block_length, candidate_block_size
    )
    # metadata 纯透传（与 lightning_indexer 一致）：None 直接下发 aclnn；
    # arch22 kernel 不消费 metadata。如需分核信息，由调用方显式调用
    # lightning_indexer_metadata 生成后传入（依赖环境提供 aclnnLightningIndexerV2Metadata 符号）。

    op_module = sparse_lightning_indexer_op_builder.load()
    return op_module.sparse_lightning_indexer(
        q,
        k,
        w,
        topk,
        candidate_topk_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        metadata,
        max_seqlen_q,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
        candidate_block_length,
        candidate_block_size,
    )


@torch.library.register_kernel(
    "cann_ops_transformer::" + sparse_lightning_indexer_op_builder.name, None
)
def sparse_lightning_indexer_fallback(
    q,
    k,
    w,
    topk,
    candidate_topk_indices,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    candidate_block_length=None,
    candidate_block_size=8,
):
    # fallback（非 NPU 设备）：调用 NPU 实现（与 LIV2 模式一致，报错路径来自 aclnn）
    return sparse_lightning_indexer(
        q,
        k,
        w,
        topk,
        candidate_topk_indices,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        max_seqlen_q=max_seqlen_q,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        candidate_block_length=candidate_block_length,
        candidate_block_size=candidate_block_size,
    )


torch.compiler.allow_in_graph(sparse_lightning_indexer)
