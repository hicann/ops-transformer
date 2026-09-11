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


def npu_minimax_sparse_attention_split_kv(
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
        query (Tensor): Query, bf16 or fp8 e4m3fn. TND [T, N, D], BNSD [B, N, S, D],
            or BSND [B, S, N, D].
        key (Tensor): Key, same dtype as query. Paged
            [num_blocks, block_size, kv_heads, D] (TND only) or contiguous matching
            query layout.
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
            query shape; dtype is bf16 when query is fp8 e4m3fn, otherwise query dtype.
            softmax_lse is fp32 when softmax_lse_flag is True.

    Example:
        Training people usually have indexer ``select_idx``, not CSR. Build CSR
        first, then call this op. Typical BNSD contiguous (no paged cache)::

            import custom_ops
            from custom_ops import build_k2q_csr, npu_minimax_sparse_attention_split_kv

            row_ptr, q_idx, slot_idx = build_k2q_csr(
                select_idx, actual_seq_lengths, actual_seq_lengths_kv, block_size,
                input_layout="BNSD",
            )
            attn_out, softmax_lse = npu_minimax_sparse_attention_split_kv(
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
    return torch.ops.custom.npu_minimax_sparse_attention_split_kv(
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
