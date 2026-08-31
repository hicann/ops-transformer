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
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from .minimax_sparse_attention_split_kv_csr import build_k2q_csr

OP_NAME = "minimax_sparse_attention_split_kv"


class MinimaxSparseAttentionSplitKvOpBuilder(OpBuilder):
    def __init__(self):
        super(MinimaxSparseAttentionSplitKvOpBuilder, self).__init__(OP_NAME)

    def sources(self):
        return ["csrc/attention/minimax_sparse_attention_split_kv.cpp"]

    def schema(self) -> str:
        return (
            "minimax_sparse_attention_split_kv("
            "Tensor query, Tensor key, Tensor value, Tensor? block_table, "
            "Tensor k2q_row_ptr, Tensor k2q_q_indices, Tensor k2q_slot_indices, "
            "Tensor actual_seq_lengths, Tensor actual_seq_lengths_kv, "
            "int num_key_value_heads, float scale_value, int block_size, int top_k, "
            'int inner_precise=4, bool softmax_lse_flag=False, str input_layout="TND"'
            ") -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def minimax_sparse_attention_split_kv_meta(
            query,
            key,
            value,
            block_table,
            k2q_row_ptr,
            k2q_q_indices,
            k2q_slot_indices,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            num_key_value_heads,
            scale_value,
            block_size,
            top_k,
            inner_precise=4,
            softmax_lse_flag=False,
            input_layout="TND",
        ):
            out_dtype = (
                torch.bfloat16 if query.dtype == torch.float8_e4m3fn else query.dtype
            )
            attention_out = torch.empty(query.shape, dtype=out_dtype, device="meta")
            if softmax_lse_flag:
                if input_layout == "TND":
                    lse_shape = (query.size(0), query.size(1), 1)
                else:
                    lse_shape = (query.size(0), query.size(1), query.size(2), 1)
                softmax_lse = torch.empty(lse_shape, dtype=torch.float, device="meta")
            else:
                softmax_lse = torch.empty((0,), dtype=torch.float, device="meta")
            return (attention_out, softmax_lse)


_minimax_sparse_attention_split_kv_op_builder = MinimaxSparseAttentionSplitKvOpBuilder()


@impl(
    get_as_library(), _minimax_sparse_attention_split_kv_op_builder.name, "PrivateUse1"
)
def _minimax_sparse_attention_split_kv(
    query,
    key,
    value,
    block_table,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_slot_indices,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    num_key_value_heads,
    scale_value,
    block_size,
    top_k,
    inner_precise=4,
    softmax_lse_flag=False,
    input_layout="TND",
):
    op_module = _minimax_sparse_attention_split_kv_op_builder.load()
    return op_module.minimax_sparse_attention_split_kv(
        query,
        key,
        value,
        block_table,
        k2q_row_ptr,
        k2q_q_indices,
        k2q_slot_indices,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        num_key_value_heads,
        scale_value,
        block_size,
        top_k,
        inner_precise,
        softmax_lse_flag,
        input_layout,
    )


def minimax_sparse_attention_split_kv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    k2q_slot_indices: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    num_key_value_heads: int,
    scale_value: float,
    block_size: int,
    top_k: int,
    *,
    block_table: Optional[torch.Tensor] = None,
    inner_precise: int = 4,
    softmax_lse_flag: bool = False,
    input_layout: str = "TND",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MiniMax sparse attention split-KV (KV-Gather-Q prefill).

    Encapsulates aclnnMinimaxSparseAttentionSplitKv. Phase1 is KV-centric
    QK → softmax → PV into per-slot partials; Phase2 FlashDecode-combines them.

    Args:
        query (Tensor): Query, bf16. TND [T, N, D], BNSD [B, N, S, D], or BSND [B, S, N, D].
        key (Tensor): Key, bf16. Paged [num_blocks, block_size, kv_heads, D] (TND only)
            or contiguous matching query layout.
        value (Tensor): Value, same layout/dtype as key.
        k2q_row_ptr (Tensor): int32 CSR row pointers [kv_heads, total_kv_rows+1].
        k2q_q_indices (Tensor): int32 CSR q-token ids [kv_heads, nnz].
            TND uses packed flatten; BNSD/BSND uses padded flatten b*S+t.
        k2q_slot_indices (Tensor): int32 CSR topK slot ids [kv_heads, nnz].
        actual_seq_lengths (Tensor): int32 [B] actual q lengths (0 = dummy request).
        actual_seq_lengths_kv (Tensor): int32 [B] actual kv lengths (0 = dummy request).
        num_key_value_heads (int): KV head count.
        scale_value (float): Softmax scale, typically 1/sqrt(D).
        block_size (int): KV block size (production 128).
        top_k (int): Sparse block budget per q-token.
        block_table (Tensor, optional): int32 [B, max_blocks_per_batch] physical
            block map. Required for paged KV; must be None for BNSD/BSND or
            contiguous TND.
        inner_precise (int): 0 = fp32 softmax + fp32 O_partial; 1 = bf16 softmax
            + bf16 O_partial; 4 (default) = bf16 softmax + fp32 O_partial.
        softmax_lse_flag (bool): If True, write fp32 LSE. TND [T, N, 1],
            BNSD [B, N, S, 1], BSND [B, S, N, 1]. If False, LSE is a [0] placeholder.
        input_layout (str): "TND", "BNSD", or "BSND". Default "TND".
            Paged KV cache requires TND.

    Returns:
        Tuple[Tensor, Tensor]: (attention_out, softmax_lse). attention_out matches
            query shape/dtype. softmax_lse is fp32 when softmax_lse_flag is True.

    Example:
        Training people usually have indexer ``select_idx``, not CSR. Build CSR
        first, then call this op. Typical BNSD contiguous (no paged cache)::

            from cann_ops_transformer import (
                build_k2q_csr,
                minimax_sparse_attention_split_kv,
            )

            row_ptr, q_idx, slot_idx = build_k2q_csr(
                select_idx, actual_seq_lengths, actual_seq_lengths_kv, block_size,
                input_layout="BNSD",
            )
            attn_out, softmax_lse = minimax_sparse_attention_split_kv(
                query, key, value, row_ptr, q_idx, slot_idx,
                actual_seq_lengths, actual_seq_lengths_kv,
                num_key_value_heads, scale_value, block_size, top_k,
                input_layout="BNSD",
            )

    Note:
        Forward-only prefill kernel. No autograd / backward is registered.
        ``group_size = num_q_heads / num_key_value_heads`` must be in ``[1, 16]``,
        and head dim must be 128.
    """
    return _minimax_sparse_attention_split_kv(
        query,
        key,
        value,
        block_table,
        k2q_row_ptr,
        k2q_q_indices,
        k2q_slot_indices,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        num_key_value_heads,
        scale_value,
        block_size,
        top_k,
        inner_precise,
        softmax_lse_flag,
        input_layout,
    )
