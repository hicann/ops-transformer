# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Golden reference for minimax_sparse_attention_split_kv (KV-Gather-Q).

Ported from kdb/msa sparse_attention_score_prefill golden, plus:
- innerPrecise 0 (fp32 softmax, P round to bf16) / 4 (bf16 S softmax) / 1 (bf16 O_partial)
- softmax LSE
- TND / BNSD / BSND layouts and contiguous (non-paged) K/V
- padding requests with q_len=kv_len=0
"""

import math
import unittest
from math import ceil

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Shared helpers (aligned with test_sparse_attention_score_golden.py)
# ---------------------------------------------------------------------------


def generate_block_index_with_causal(
    query_fp32,
    key_fp32,
    q_seqlen,
    kv_seqlen,
    kv_heads,
    group_size,
    block_size=128,
    top_k=16,
):
    """Generate per-kv-head block indices with causal pooling (decode style)."""
    his_seq_len = kv_seqlen - q_seqlen
    total_blocks = ceil(kv_seqlen / block_size)
    head_dim = query_fp32.shape[-1]

    select_idx = torch.full((kv_heads, q_seqlen, top_k), -1, dtype=torch.int32)
    select_num_idx = torch.zeros((kv_heads, q_seqlen), dtype=torch.int32)

    for kv_head in range(kv_heads):
        representative_q_head = kv_head * group_size
        k_head = key_fp32[:, kv_head, :]

        for q_token in range(q_seqlen):
            q_vec = query_fp32[q_token, representative_q_head, :]
            causal_bound = his_seq_len + q_token

            scores = torch.matmul(q_vec, k_head[:kv_seqlen, :].transpose(0, 1))

            pooled = torch.full((total_blocks,), -float("inf"), dtype=torch.float32)
            q_block = causal_bound // block_size

            for block_idx in range(total_blocks):
                block_begin = block_idx * block_size
                block_end = min(block_begin + block_size, kv_seqlen)

                if block_idx > q_block:
                    pooled[block_idx] = -float("inf")
                elif block_idx == q_block:
                    pooled[block_idx] = float("inf")
                else:
                    effective_end = min(block_end, causal_bound + 1)
                    if effective_end > block_begin:
                        pooled[block_idx] = torch.max(
                            scores[block_begin:effective_end]
                        ).item()

            visible_blocks = min(total_blocks, q_block + 1)
            valid_k = min(top_k, visible_blocks)
            select_num_idx[kv_head, q_token] = valid_k
            if valid_k > 0:
                topk_indices = torch.topk(pooled, k=valid_k, largest=True).indices.to(
                    torch.int32
                )
                select_idx[kv_head, q_token, :valid_k] = topk_indices

    return select_idx, select_num_idx


def generate_block_table(batch, max_blocks_per_batch, shuffle=True):
    """Logical -> physical block mapping (decode style)."""
    total_physical = batch * max_blocks_per_batch
    all_physical_ids = list(range(total_physical))
    if shuffle:
        import random

        rng = random.Random(137)
        rng.shuffle(all_physical_ids)
    block_table = torch.zeros(batch, max_blocks_per_batch, dtype=torch.int32)
    for b in range(batch):
        for i in range(max_blocks_per_batch):
            block_table[b, i] = all_physical_ids[b * max_blocks_per_batch + i]
    return block_table


def cpu_sparse_attention_score_bf16(
    query,
    key,
    value,
    select_idx,
    block_table,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    num_key_value_heads,
    select_num_idx=None,
    block_size=128,
    scale_value=1.0,
):
    """Decode Q-centric BF16 golden (online softmax over topK KV blocks)."""
    select_idx_cpu = select_idx.to(torch.int64)
    block_table_cpu = block_table.to(torch.int64)

    total_q_tokens, q_heads, head_dim = query.shape
    kv_heads = num_key_value_heads
    group_size = q_heads // kv_heads
    top_k = select_idx.shape[2]
    batch = len(actual_seq_lengths)
    scale_value = float(scale_value)

    output = torch.zeros(total_q_tokens, q_heads, head_dim, dtype=torch.bfloat16)
    q_offset = 0
    for batch_idx, q_seqlen in enumerate(actual_seq_lengths):
        kv_seqlen = int(actual_seq_lengths_kv[batch_idx])
        history_len = kv_seqlen - q_seqlen

        for q_token_in_batch in range(q_seqlen):
            global_q_token = q_offset + q_token_in_batch
            causal_bound = history_len + q_token_in_batch

            for kv_head in range(kv_heads):
                valid_top_k = top_k
                if select_num_idx is not None:
                    valid_top_k = int(select_num_idx[kv_head, global_q_token].item())
                    valid_top_k = min(valid_top_k, top_k)
                if valid_top_k == 0:
                    continue

                q_start = kv_head * group_size
                q_group_bf16 = query[global_q_token, q_start : q_start + group_size, :]

                last_max_fp32 = torch.full(
                    (group_size,), -float("inf"), dtype=torch.float32
                )
                last_sum_fp32 = torch.zeros(group_size, dtype=torch.float32)
                o_acc_fp32 = torch.zeros(group_size, head_dim, dtype=torch.float32)
                is_first = True

                for topk_idx in range(valid_top_k):
                    logical_id = int(
                        select_idx_cpu[kv_head, global_q_token, topk_idx].item()
                    )
                    if logical_id < 0:
                        continue
                    block_begin = logical_id * block_size
                    block_end = min(block_begin + block_size, kv_seqlen)
                    effective_end = min(block_end, causal_bound + 1)
                    if effective_end <= block_begin:
                        continue

                    physical_id = int(block_table_cpu[batch_idx, logical_id].item())
                    valid_len = effective_end - block_begin
                    k_bf16 = key[physical_id, :valid_len, kv_head, :]
                    v_bf16 = value[physical_id, :valid_len, kv_head, :]

                    s_fp32 = (
                        torch.matmul(q_group_bf16.float(), k_bf16.float().t())
                        * scale_value
                    )
                    now_max_fp32 = s_fp32.max(dim=1).values
                    if not is_first:
                        now_max_fp32 = torch.max(now_max_fp32, last_max_fp32)

                    p_fp32 = torch.exp(s_fp32 - now_max_fp32.unsqueeze(1))
                    now_sum_fp32 = p_fp32.sum(dim=1)
                    p_bf16 = p_fp32.to(torch.bfloat16)

                    if is_first:
                        last_sum_fp32 = now_sum_fp32
                        last_max_fp32 = now_max_fp32
                    else:
                        correction_fp32 = torch.exp(last_max_fp32 - now_max_fp32)
                        last_sum_fp32 = correction_fp32 * last_sum_fp32 + now_sum_fp32
                        last_max_fp32 = now_max_fp32

                    pv_fp32 = torch.matmul(p_bf16.float(), v_bf16.float())
                    if is_first:
                        o_acc_fp32 = pv_fp32
                    else:
                        o_acc_fp32 = o_acc_fp32 * correction_fp32.unsqueeze(1) + pv_fp32

                    is_first = False

                if last_sum_fp32.max() > 0:
                    result_fp32 = o_acc_fp32 / last_sum_fp32.unsqueeze(1)
                    output[global_q_token, q_start : q_start + group_size, :] = (
                        result_fp32.to(torch.bfloat16)
                    )

        q_offset += q_seqlen

    return output


# ---------------------------------------------------------------------------
# BF16 block helpers (shared by phase1 and q-centric reference)
# ---------------------------------------------------------------------------


def _softmax_p_from_scores_fp32(s_fp32):
    """fp32 softmax; cast P to bf16 only for PV matmul dtype."""
    row_max_fp32 = s_fp32.max(dim=1).values
    # print("row_max_fp32:", row_max_fp32)
    p_fp32 = torch.exp(s_fp32 - row_max_fp32.unsqueeze(1))
    row_sum_fp32 = p_fp32.sum(dim=1)
    p_bf16 = p_fp32.to(torch.bfloat16)
    return p_bf16, row_max_fp32, row_sum_fp32


def _bf16_block_qk_sm_pv(
    q_group_bf16, k_bf16, v_bf16, scale_value, q, h, t, inner_precise
):
    """Single KV block: QK -> SM -> PV. Returns unnormalized O_partial, rowMax, rowSum."""
    if inner_precise == 0:
        # fp32 S softmax; P rounded to bf16 before PV (kernel high-prec path).
        s_fp32 = torch.matmul(q_group_bf16.float(), k_bf16.float().t()) * float(
            scale_value
        )
    else:
        s_bf16 = torch.matmul(q_group_bf16.float(), k_bf16.float().t()).to(
            torch.bfloat16
        )
        s_fp32 = (s_bf16 * torch.tensor([scale_value], dtype=torch.bfloat16)).to(
            torch.float32
        )
    p_bf16, row_max_fp32, row_sum_fp32 = _softmax_p_from_scores_fp32(s_fp32)
    # if q ==0 and h == 0 and t == 0:
    #   print("p_bf16[0][0]:", p_bf16[0][:16])
    #   print("p_bf16[0][1]:", p_bf16[0][16:32])
    #   print("p_bf16[0][2]:", p_bf16[0][32:48])
    #   print("p_bf16[0][3]:", p_bf16[0][48:64])
    #   print("p_bf16[0][4]:", p_bf16[0][64:80])
    #   print("p_bf16[0][5]:", p_bf16[0][80:96])
    #   print("p_bf16[0][6]:", p_bf16[0][96:])
    # print("s_fp32:", torch.matmul(q_group_bf16.float(), k_bf16.float().t())[0][:128])
    # print("p_bf16:", p_bf16.shape, p_bf16[0][:128])
    o_unnorm_fp32 = torch.matmul(p_bf16.float(), v_bf16.float())
    if inner_precise == 1:
        o_unnorm_fp32 = o_unnorm_fp32.to(torch.bfloat16).to(torch.float32)
    # if q ==0 and h == 0 and t == 0:
    #  print("o_unnorm_fp32:", o_unnorm_fp32)
    row_max_fp32 = torch.where(
        row_sum_fp32 > 0, row_max_fp32, torch.full_like(row_max_fp32, float("-inf"))
    )
    row_sum_fp32 = torch.where(
        row_sum_fp32 > 0, row_sum_fp32, torch.zeros_like(row_sum_fp32)
    )
    return o_unnorm_fp32, row_max_fp32, row_sum_fp32


# Match kernel ComputeScaleValue_VF init / invalid threshold.
NEG_INF_LSE = -3.4028235e38
FLT_MAX_NEW = 3.402823466e38


def _compute_scale_weights(max_slots, sum_slots):
    """Match ComputeScaleValue_VF (softmaxLseFlag=False).

    scale[k] = rowSum[k] * exp(rowMax[k] - global_max) / sum_j(...)
    Intended for accumOut already divided by rowSum (IFA / CopyAccumOutIn path).
    """
    global_max = max_slots.max()
    if not torch.isfinite(global_max) or global_max <= -FLT_MAX_NEW:
        return torch.zeros_like(sum_slots)

    scaled = sum_slots * torch.exp(max_slots - global_max)
    total = scaled.sum()
    if total <= 0:
        return torch.zeros_like(sum_slots)
    return scaled / total


def _combine_partials_max_sum_ws(o_partial_slots, max_slots, sum_slots):
    """Combine topK workspace partials via IFA ComputeScaleValue + ReduceFinalRes.

    Phase1 stores unnormalized O_partial = P*V and per-row rowMax/rowSum.
    Phase2 CopyAccumOutIn: O_norm = O_partial / rowSum (invalid rowSum<=0 -> 0).
    ReduceFinalRes: out = sum(scale[k] * O_norm[k]).
    """
    # print("max_slots:", max_slots)
    # print("sum_slots:", sum_slots)
    scale = _compute_scale_weights(max_slots, sum_slots)
    # print("scale:", scale)
    if scale.sum() <= 0:
        return None

    valid = sum_slots > 0
    o_norm = torch.zeros_like(o_partial_slots)
    if torch.any(valid):
        o_norm[valid] = o_partial_slots[valid] / sum_slots[valid].unsqueeze(-1)
    return (scale.unsqueeze(-1) * o_norm).sum(dim=0)


# ---------------------------------------------------------------------------
# Prefill-specific: k2q CSR + KV outer loop golden
# ---------------------------------------------------------------------------


def _batch_offsets(q_seqlens, kv_seqlens, block_size):
    batch_size = len(q_seqlens)
    batch_q_offset = [0] * (batch_size + 1)
    batch_kv_block_offset = [0] * (batch_size + 1)
    for b in range(batch_size):
        batch_q_offset[b + 1] = batch_q_offset[b] + q_seqlens[b]
        batch_kv_block_offset[b + 1] = batch_kv_block_offset[b] + ceil(
            kv_seqlens[b] / block_size
        )
    return batch_q_offset, batch_kv_block_offset


def _find_batch_for_token(token_idx, batch_offsets):
    for b in range(len(batch_offsets) - 1):
        if token_idx < batch_offsets[b + 1]:
            return b
    return len(batch_offsets) - 2


def _normalize_select_idx_3d(select_idx, select_num_idx=None, kv_heads=None):
    """Return (select_idx_3d, select_num_3d, kv_heads, total_q, top_k) as numpy arrays."""
    if isinstance(select_idx, torch.Tensor):
        select_idx_np = select_idx.detach().cpu().numpy()
    else:
        select_idx_np = np.asarray(select_idx)

    if select_num_idx is not None:
        if isinstance(select_num_idx, torch.Tensor):
            select_num_np = select_num_idx.detach().cpu().numpy()
        else:
            select_num_np = np.asarray(select_num_idx)
    else:
        select_num_np = None

    if select_idx_np.ndim == 2:
        if kv_heads is None:
            kv_heads = 1
        total_q_tokens, top_k = select_idx_np.shape
        select_idx_3d = np.broadcast_to(
            select_idx_np[np.newaxis, :, :], (kv_heads, total_q_tokens, top_k)
        ).copy()
        select_num_3d = (
            select_num_np
            if select_num_np is not None and select_num_np.ndim == 2
            else None
        )
    elif select_idx_np.ndim == 3:
        kv_heads, total_q_tokens, top_k = select_idx_np.shape
        select_idx_3d = select_idx_np
        select_num_3d = select_num_np
    else:
        raise ValueError(
            f"select_idx must be 2D or 3D, got shape {select_idx_np.shape}"
        )

    return select_idx_3d, select_num_3d, kv_heads, total_q_tokens, top_k


def _rows_per_batch_from_actual_seq_lengths(actual_seq_lengths_kv, kv_block_size):
    seqlens_k = np.asarray(actual_seq_lengths_kv, dtype=np.int64)
    return (seqlens_k + kv_block_size - 1) // kv_block_size


def _q_batch_offsets_from_actual(actual_seq_lengths):
    actual = np.asarray(actual_seq_lengths, dtype=np.int32)
    offsets = np.zeros(int(actual.shape[0]) + 1, dtype=np.int32)
    for i, slen in enumerate(actual.tolist()):
        offsets[i + 1] = offsets[i] + int(slen)
    return offsets


def _build_packed_row_map(rows_per_batch):
    """Map (batch, local_kv_block) -> packed dense row id (CUDA-compatible)."""
    rows_per_batch = np.asarray(rows_per_batch, dtype=np.int64)
    batch = int(rows_per_batch.shape[0])
    max_rows = int(rows_per_batch.max()) if batch > 0 else 0
    row_map = np.full((batch, max_rows), -1, dtype=np.int32)
    row_linear = 0
    for kv_block_idx in range(max_rows):
        for batch_idx, row_count in enumerate(rows_per_batch.tolist()):
            if kv_block_idx < row_count:
                row_map[batch_idx, kv_block_idx] = row_linear
                row_linear += 1
    return row_map, row_linear


def _init_packed_row_coord(rows_per_batch):
    """MSA packing coord for packed_row == 0."""
    batch_size = len(rows_per_batch)
    kv_block_idx = 0
    batch_idx = 0
    while batch_idx < batch_size and kv_block_idx >= rows_per_batch[batch_idx]:
        batch_idx += 1
    return batch_idx, kv_block_idx


def _advance_packed_row_coord(batch_idx, kv_block_idx, batch_size, rows_per_batch):
    """Step to the next (batch_idx, kv_block_idx) in MSA packing order."""
    batch_idx += 1
    while batch_idx < batch_size and kv_block_idx >= rows_per_batch[batch_idx]:
        batch_idx += 1
    if batch_idx >= batch_size:
        kv_block_idx += 1
        batch_idx = 0
        while batch_idx < batch_size and kv_block_idx >= rows_per_batch[batch_idx]:
            batch_idx += 1
    return batch_idx, kv_block_idx


def decode_packed_row(packed_row, actual_seq_lengths_kv, block_size):
    """Decode CUDA packed CSR row id -> (batch_idx, local_kv_block_idx)."""
    kv_seqlens = _as_seq_list(actual_seq_lengths_kv)
    rows_per_batch = [ceil(int(kv) / block_size) for kv in kv_seqlens]
    max_kv_block_rows = max(rows_per_batch) if rows_per_batch else 0
    row_linear = 0
    for kv_block_idx in range(max_kv_block_rows):
        for batch_idx, rows_b in enumerate(rows_per_batch):
            if kv_block_idx >= rows_b:
                continue
            if row_linear == packed_row:
                return batch_idx, kv_block_idx
            row_linear += 1
    raise ValueError(f"packed_row {packed_row} out of range")


def _as_seq_list(seq):
    if isinstance(seq, torch.Tensor):
        return [int(x) for x in seq.detach().cpu().tolist()]
    return [int(x) for x in seq]


def _kv_block_valid_size(local_block_idx, kv_seqlen, block_size):
    tail_remain = int(kv_seqlen) - local_block_idx * block_size
    if tail_remain <= 0:
        return 0
    return min(block_size, tail_remain)


def global_select_idx_to_batch_local(
    select_idx_global,
    q_seqlens,
    kv_seqlens,
    block_size,
    select_num_idx=None,
    kv_heads=None,
):
    """
    Convert global logical KV block ids to batch-local ids (CUDA q2k layout).

    q2k_indices[head_kv, total_q, topK] uses per-token batch-local block indices.
    """
    q_seqlens = _as_seq_list(q_seqlens)
    kv_seqlens = _as_seq_list(kv_seqlens)
    select_idx_3d, select_num_3d, kv_heads, total_q, top_k = _normalize_select_idx_3d(
        select_idx_global, select_num_idx, kv_heads
    )
    batch_q_offset, batch_kv_block_offset = _batch_offsets(
        q_seqlens, kv_seqlens, block_size
    )

    q2k = np.full((kv_heads, total_q, top_k), -1, dtype=np.int32)
    for h in range(kv_heads):
        for q in range(total_q):
            batch_idx = _find_batch_for_token(q, batch_q_offset)
            valid_k = top_k
            if select_num_3d is not None:
                valid_k = min(int(select_num_3d[h, q]), top_k)
            for k in range(valid_k):
                global_block = int(select_idx_3d[h, q, k])
                if global_block < 0:
                    continue
                if (
                    global_block < batch_kv_block_offset[batch_idx]
                    or global_block >= batch_kv_block_offset[batch_idx + 1]
                ):
                    continue
                q2k[h, q, k] = global_block - batch_kv_block_offset[batch_idx]
    return q2k, batch_q_offset, batch_kv_block_offset


def build_k2q_csr_torch_reference(
    q2k_indices,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    kv_block_size,
    *,
    q_global_offset=True,
):
    """
    Torch reference for q2k -> k2q CSR (ported from fmha_sm100 sparse_index_utils.py).

    Dense packed rows: one CSR row per (head_kv, packed_kv_block) across the batch.

    Args:
      q2k_indices: [head_kv, total_q, topK] int32, batch-local KV block ids, -1 padded.
      actual_seq_lengths / actual_seq_lengths_kv: [batch] int32, per-batch Q/KV seqlen.
      kv_block_size: tokens per KV block.

    Returns:
      k2q_row_ptr: [head_kv, total_rows + 1] int32
      k2q_q_indices: [head_kv, total_q * topK] int32 (global q if q_global_offset)
      k2q_slot_indices: [head_kv, total_q * topK] int32
    """
    if kv_block_size <= 0:
        raise ValueError(f"kv_block_size must be > 0, got {kv_block_size}")
    if not isinstance(q2k_indices, torch.Tensor):
        q2k_indices = torch.from_numpy(np.asarray(q2k_indices))
    if q2k_indices.dtype != torch.int32:
        q2k_indices = q2k_indices.to(torch.int32)
    if q2k_indices.ndim != 3:
        raise ValueError(
            f"q2k_indices must be [head_kv, total_q, topK], got {tuple(q2k_indices.shape)}"
        )

    actual_seq_lengths = torch.as_tensor(
        actual_seq_lengths, dtype=torch.int32, device=q2k_indices.device
    )
    actual_seq_lengths_kv = torch.as_tensor(
        actual_seq_lengths_kv, dtype=torch.int32, device=q2k_indices.device
    )
    if actual_seq_lengths.shape != actual_seq_lengths_kv.shape:
        raise ValueError(
            "actual_seq_lengths and actual_seq_lengths_kv must have the same shape [B]"
        )
    if actual_seq_lengths.ndim != 1:
        raise ValueError(
            f"actual_seq_lengths must be rank-1 [B], got shape {tuple(actual_seq_lengths.shape)}"
        )

    head_kv, total_q, topk = q2k_indices.shape
    total_q_expected = int(actual_seq_lengths.sum().item())
    if total_q != total_q_expected:
        raise ValueError(
            f"q2k_indices.shape[1] ({total_q}) must equal sum(actual_seq_lengths) "
            f"({total_q_expected})"
        )

    rows_per_batch = _rows_per_batch_from_actual_seq_lengths(
        actual_seq_lengths_kv.cpu().numpy(), kv_block_size
    )
    row_map, total_rows = _build_packed_row_map(rows_per_batch)
    row_map_t = torch.from_numpy(row_map).to(device=q2k_indices.device)
    nnz_upper_bound = total_q * topk

    k2q_row_ptr = torch.zeros(
        (head_kv, total_rows + 1), dtype=torch.int32, device=q2k_indices.device
    )
    k2q_q_indices = torch.full(
        (head_kv, nnz_upper_bound), -1, dtype=torch.int32, device=q2k_indices.device
    )
    k2q_slot_indices = torch.full(
        (head_kv, nnz_upper_bound), -1, dtype=torch.int32, device=q2k_indices.device
    )
    if total_rows == 0 or total_q == 0 or topk == 0:
        return k2q_row_ptr, k2q_q_indices, k2q_slot_indices

    counts = torch.zeros(
        (head_kv, total_rows), dtype=torch.int32, device=q2k_indices.device
    )
    total_entries = total_q * topk
    row_all = torch.empty(
        (head_kv, total_entries), dtype=torch.int64, device=q2k_indices.device
    )
    q_all = torch.empty(
        (head_kv, total_entries), dtype=torch.int32, device=q2k_indices.device
    )
    slot_all = torch.empty(
        (head_kv, total_entries), dtype=torch.int32, device=q2k_indices.device
    )
    valid_all = torch.empty(
        (head_kv, total_entries), dtype=torch.bool, device=q2k_indices.device
    )

    rows_per_batch_list = rows_per_batch.tolist()
    q_offsets = _q_batch_offsets_from_actual(actual_seq_lengths.cpu().numpy())
    entry_cursor = 0

    for batch_idx, kv_rows in enumerate(rows_per_batch_list):
        q_start = int(q_offsets[batch_idx])
        q_end = int(q_offsets[batch_idx + 1])
        q_len = q_end - q_start
        if q_len == 0:
            continue
        num_entries = q_len * topk
        q2k_batch = q2k_indices[:, q_start:q_end, :]
        valid_batch = q2k_batch >= 0
        if valid_batch.any():
            max_valid_kv = int(q2k_batch[valid_batch].max().item())
            if max_valid_kv >= kv_rows:
                raise ValueError(
                    f"q2k_indices references kv_block {max_valid_kv} for batch {batch_idx}, "
                    f"but that batch only has {kv_rows} logical kv blocks"
                )

        kv_flat = q2k_batch.reshape(head_kv, num_entries).long()
        valid_flat = valid_batch.reshape(head_kv, num_entries)
        safe_kv_flat = torch.where(valid_flat, kv_flat, torch.zeros_like(kv_flat))
        row_flat = row_map_t[batch_idx][safe_kv_flat]
        q_flat = (
            torch.arange(q_len, device=q2k_indices.device, dtype=torch.int32)
            .view(1, q_len, 1)
            .expand(head_kv, q_len, topk)
            .reshape(head_kv, num_entries)
        )
        if q_global_offset:
            q_flat = q_flat + q_start
        slot_flat = (
            torch.arange(topk, device=q2k_indices.device, dtype=torch.int32)
            .view(1, 1, topk)
            .expand(head_kv, q_len, topk)
            .reshape(head_kv, num_entries)
        )
        row_all[:, entry_cursor : entry_cursor + num_entries] = row_flat
        q_all[:, entry_cursor : entry_cursor + num_entries] = q_flat
        slot_all[:, entry_cursor : entry_cursor + num_entries] = slot_flat
        valid_all[:, entry_cursor : entry_cursor + num_entries] = valid_flat
        counts.scatter_add_(1, row_flat.to(torch.int64), valid_flat.to(torch.int32))
        entry_cursor += num_entries

    k2q_row_ptr[:, 1:] = counts.cumsum(dim=1, dtype=torch.int32)

    sort_stride = max(total_q, 1)
    invalid_key = total_rows * sort_stride
    sort_keys = torch.full_like(row_all, invalid_key, dtype=torch.int64)
    sort_keys[valid_all] = row_all[valid_all] * sort_stride + q_all[valid_all].to(
        torch.int64
    )
    _, sort_idx = sort_keys.sort(dim=1, stable=True)
    sorted_q = q_all.gather(1, sort_idx)
    sorted_slot = slot_all.gather(1, sort_idx)

    valid_counts = valid_all.sum(dim=1)
    write_mask = torch.arange(total_entries, device=q2k_indices.device).unsqueeze(
        0
    ).expand(head_kv, -1) < valid_counts.unsqueeze(1)
    k2q_q_indices[write_mask] = sorted_q[write_mask]
    k2q_slot_indices[write_mask] = sorted_slot[write_mask]
    return k2q_row_ptr, k2q_q_indices, k2q_slot_indices


def build_k2q_csr(
    select_idx,
    select_num_idx,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    block_size,
    kv_heads=None,
):
    """
    Build dense packed-row CSR reverse index (CUDA / MSA compatible).

    Row id is packed (batch, local_kv_block); empty rows have row_ptr[i]==row_ptr[i+1].

    Args:
      actual_seq_lengths / actual_seq_lengths_kv: [batch] per-batch Q/KV seqlen (non-cumsum).

    Returns:
      k2q_row_ptr: [kv_heads, total_rows + 1] int32
      k2q_q_indices: [kv_heads, total_q * top_k] int32, global q token ids
      k2q_slot_indices: [kv_heads, total_q * top_k] int32
    """
    q2k, _, _ = global_select_idx_to_batch_local(
        select_idx,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_size,
        select_num_idx=select_num_idx,
        kv_heads=kv_heads,
    )

    row_ptr, q_idx, slot_idx = build_k2q_csr_torch_reference(
        torch.from_numpy(q2k),
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_size,
        q_global_offset=True,
    )
    return (
        row_ptr.cpu().numpy(),
        q_idx.cpu().numpy(),
        slot_idx.cpu().numpy(),
    )


def decode_select_idx_to_prefill_global(
    select_idx_decode, q_seqlens, kv_seqlens, block_size
):
    """
    Convert decode select_idx [kv_heads, total_q, top_k] (local logical ids)
    to prefill global select_idx [kv_heads, total_q, top_k] (global kv block ids).
    """
    kv_heads, total_q, top_k = select_idx_decode.shape
    batch_q_offset, batch_kv_block_offset = _batch_offsets(
        q_seqlens, kv_seqlens, block_size
    )
    batch_size = len(q_seqlens)

    select_idx_global = torch.full((kv_heads, total_q, top_k), -1, dtype=torch.int32)
    for h in range(kv_heads):
        for b in range(batch_size):
            for qi in range(q_seqlens[b]):
                q_global = batch_q_offset[b] + qi
                for k in range(top_k):
                    logical_id = int(select_idx_decode[h, q_global, k].item())
                    if logical_id >= 0:
                        select_idx_global[h, q_global, k] = (
                            batch_kv_block_offset[b] + logical_id
                        )
    return select_idx_global


def cpu_golden_prefill_phase1(
    query,
    key,
    value,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_slot_indices,
    block_table,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    block_size,
    top_k,
    scale_value,
    inner_precise,
):
    """
    Phase1 golden: QK + SM + PV only; write workspace partials (O_partial, max, sum).

    Mirrors kernel Phase1 workspace layout; combine/rescale deferred to Phase2.
    """
    total_q_tokens, num_q_heads, d = query.shape
    kv_heads = key.shape[2]
    group_size = num_q_heads // kv_heads
    scale_value = float(scale_value)

    q_seqlens = _as_seq_list(actual_seq_lengths)
    kv_seqlens = _as_seq_list(actual_seq_lengths_kv)
    batch_q_offset, _ = _batch_offsets(q_seqlens, kv_seqlens, block_size)

    if isinstance(k2q_row_ptr, torch.Tensor):
        k2q_row_ptr_np = k2q_row_ptr.detach().cpu().numpy()
        k2q_q_indices_np = k2q_q_indices.detach().cpu().numpy()
        k2q_slot_indices_np = k2q_slot_indices.detach().cpu().numpy()
    else:
        k2q_row_ptr_np = np.asarray(k2q_row_ptr)
        k2q_q_indices_np = np.asarray(k2q_q_indices)
        k2q_slot_indices_np = np.asarray(k2q_slot_indices)

    ws_o = torch.zeros(
        total_q_tokens, kv_heads, top_k, group_size, d, dtype=torch.float32
    )
    ws_max = torch.full(
        (total_q_tokens, kv_heads, top_k, group_size),
        float("-inf"),
        dtype=torch.float32,
    )
    ws_sum = torch.zeros(
        (total_q_tokens, kv_heads, top_k, group_size), dtype=torch.float32
    )
    # Workspace layout per partial (bytes): O_partial[groupSize,D] fp32 + max[groupSize] + sum[groupSize]

    total_rows = int(k2q_row_ptr_np.shape[1] - 1)
    total_task_num_p1 = total_rows * kv_heads
    rows_per_batch = [ceil(int(kv) / block_size) for kv in kv_seqlens]
    batch_size = len(kv_seqlens)

    cur_packed_row = -1
    batch_idx = 0
    kv_block_idx = 0

    for task_idx in range(total_task_num_p1):
        packed_row = task_idx // kv_heads
        kv_head_idx = task_idx % kv_heads
        if packed_row >= total_rows:
            continue

        if packed_row != cur_packed_row:
            if cur_packed_row < 0:
                batch_idx, kv_block_idx = _init_packed_row_coord(rows_per_batch)
                cur_packed_row = 0
            while cur_packed_row < packed_row:
                batch_idx, kv_block_idx = _advance_packed_row_coord(
                    batch_idx, kv_block_idx, batch_size, rows_per_batch
                )
                cur_packed_row += 1

        local_block_idx = kv_block_idx

        physical_block_id = int(block_table[batch_idx, local_block_idx].item())

        valid_size = _kv_block_valid_size(
            local_block_idx, kv_seqlens[batch_idx], block_size
        )
        if valid_size == 0:
            continue

        csr_start = int(k2q_row_ptr_np[kv_head_idx, packed_row])
        csr_end = int(k2q_row_ptr_np[kv_head_idx, packed_row + 1])
        num_q_tokens = csr_end - csr_start
        if num_q_tokens == 0:
            continue

        kv_start_pos = local_block_idx * block_size
        k_block = key[physical_block_id, :valid_size, kv_head_idx, :]
        v_block = value[physical_block_id, :valid_size, kv_head_idx, :]
        q_head_start = kv_head_idx * group_size

        for qi in range(num_q_tokens):
            edge = csr_start + qi
            q_token = int(k2q_q_indices_np[kv_head_idx, edge])
            slot_k = int(k2q_slot_indices_np[kv_head_idx, edge])

            q_row_bf16 = query[q_token, q_head_start : q_head_start + group_size, :]

            q_batch = _find_batch_for_token(q_token, batch_q_offset)
            local_q_idx = q_token - batch_q_offset[q_batch]
            q_position = kv_seqlens[q_batch] - q_seqlens[q_batch] + local_q_idx
            causal_valid_len = min(valid_size, q_position - kv_start_pos + 1)
            causal_valid_len = max(causal_valid_len, 0)
            if causal_valid_len == 0:
                continue

            o_partial, row_max, row_sum = _bf16_block_qk_sm_pv(
                q_row_bf16,
                k_block[:causal_valid_len],
                v_block[:causal_valid_len],
                scale_value,
                q_token,
                kv_head_idx,
                slot_k,
                inner_precise,
            )

            ws_o[q_token, kv_head_idx, slot_k] = o_partial
            ws_max[q_token, kv_head_idx, slot_k] = row_max
            ws_sum[q_token, kv_head_idx, slot_k] = row_sum

    return ws_o, ws_max, ws_sum


def _lse_from_slots(max_slots, sum_slots):
    """lse = log(sum_k rowSum[k]*exp(rowMax[k]-max_k)) + max_k. 0 if no valid slot."""
    valid = (sum_slots > 0) & torch.isfinite(max_slots) & (max_slots > NEG_INF_LSE)
    if not torch.any(valid):
        return 0.0
    gm = max_slots[valid].max()
    acc = (sum_slots[valid] * torch.exp(max_slots[valid] - gm)).sum()
    if acc <= 0:
        return 0.0
    return float(torch.log(acc) + gm)


def cpu_golden_prefill_phase2(ws_o, ws_max, ws_sum, top_k, kv_heads, group_size, d):
    """Phase2: O/rowSum + ComputeScaleValue scale + ReduceFinalRes over topK partials."""
    total_q_tokens = ws_o.shape[0]
    num_q_heads = kv_heads * group_size
    output = torch.zeros(total_q_tokens, num_q_heads, d, dtype=torch.bfloat16)
    lse = torch.zeros(total_q_tokens, num_q_heads, dtype=torch.float32)
    total_task_num_p2 = total_q_tokens * kv_heads

    for task_idx in range(total_task_num_p2):
        q_token = task_idx // kv_heads
        kv_head_idx = task_idx % kv_heads
        q_head_start = kv_head_idx * group_size
        for gh in range(group_size):
            slot_o = ws_o[q_token, kv_head_idx, :, gh, :]
            slot_max = ws_max[q_token, kv_head_idx, :, gh]
            slot_sum = ws_sum[q_token, kv_head_idx, :, gh]
            combined = _combine_partials_max_sum_ws(slot_o, slot_max, slot_sum)
            if combined is not None:
                output[q_token, q_head_start + gh, :] = combined.to(torch.bfloat16)
            lse[q_token, q_head_start + gh] = _lse_from_slots(slot_max, slot_sum)

    return output, lse


def cpu_golden_prefill_qcentric_bf16(
    query,
    key,
    value,
    select_idx,
    block_table,
    q_seqlens,
    kv_seqlens,
    block_size,
    top_k,
    scale_value,
    select_num_idx=None,
):
    """
    Q-centric BF16 golden using per-kv-head select_idx [kv_heads, total_q, top_k].

    Same online-softmax math as decode bf16 with global logical KV block ids.
    """
    select_idx_cpu = select_idx.to(torch.int64)
    block_table_cpu = block_table.to(torch.int64)

    total_q_tokens, q_heads, head_dim = query.shape
    kv_heads = key.shape[2]
    group_size = q_heads // kv_heads
    scale_value = float(scale_value)

    batch_q_offset, batch_kv_block_offset = _batch_offsets(
        q_seqlens, kv_seqlens, block_size
    )

    output = torch.zeros(total_q_tokens, q_heads, head_dim, dtype=torch.bfloat16)

    for q_token in range(total_q_tokens):
        batch_idx = _find_batch_for_token(q_token, batch_q_offset)
        local_q_idx = q_token - batch_q_offset[batch_idx]
        kv_seqlen = int(kv_seqlens[batch_idx])
        q_seqlen = int(q_seqlens[batch_idx])
        causal_bound = kv_seqlen - q_seqlen + local_q_idx

        for kv_head in range(kv_heads):
            valid_top_k = top_k
            if select_num_idx is not None:
                valid_top_k = int(select_num_idx[kv_head, q_token].item())
                valid_top_k = min(valid_top_k, top_k)
            if valid_top_k == 0:
                continue

            q_start = kv_head * group_size
            q_group_bf16 = query[q_token, q_start : q_start + group_size, :]

            last_max_fp32 = torch.full(
                (group_size,), -float("inf"), dtype=torch.float32
            )
            last_sum_fp32 = torch.zeros(group_size, dtype=torch.float32)
            o_acc_fp32 = torch.zeros(group_size, head_dim, dtype=torch.float32)
            is_first = True

            for topk_idx in range(valid_top_k):
                global_kv_block = int(select_idx_cpu[kv_head, q_token, topk_idx].item())
                if global_kv_block < 0:
                    continue

                block_batch = _find_batch_for_token(
                    global_kv_block, batch_kv_block_offset
                )
                if block_batch != batch_idx:
                    continue

                local_block_idx = global_kv_block - batch_kv_block_offset[block_batch]
                block_begin = local_block_idx * block_size
                block_end = min(block_begin + block_size, kv_seqlen)
                effective_end = min(block_end, causal_bound + 1)
                if effective_end <= block_begin:
                    continue

                physical_id = int(block_table_cpu[batch_idx, local_block_idx].item())
                valid_len = effective_end - block_begin
                k_bf16 = key[physical_id, :valid_len, kv_head, :]
                v_bf16 = value[physical_id, :valid_len, kv_head, :]

                s_fp32 = (
                    torch.matmul(q_group_bf16.float(), k_bf16.float().t()) * scale_value
                )
                now_max_fp32 = s_fp32.max(dim=1).values
                if not is_first:
                    now_max_fp32 = torch.max(now_max_fp32, last_max_fp32)

                p_fp32 = torch.exp(s_fp32 - now_max_fp32.unsqueeze(1))
                now_sum_fp32 = p_fp32.sum(dim=1)
                p_bf16 = p_fp32.to(torch.bfloat16)

                if is_first:
                    last_sum_fp32 = now_sum_fp32
                    last_max_fp32 = now_max_fp32
                else:
                    correction_fp32 = torch.exp(last_max_fp32 - now_max_fp32)
                    last_sum_fp32 = correction_fp32 * last_sum_fp32 + now_sum_fp32
                    last_max_fp32 = now_max_fp32

                pv_fp32 = torch.matmul(p_bf16.float(), v_bf16.float())
                if is_first:
                    o_acc_fp32 = pv_fp32
                else:
                    o_acc_fp32 = o_acc_fp32 * correction_fp32.unsqueeze(1) + pv_fp32

                is_first = False

            if last_sum_fp32.max() > 0:
                result_fp32 = o_acc_fp32 / last_sum_fp32.unsqueeze(1)
                output[q_token, q_start : q_start + group_size, :] = result_fp32.to(
                    torch.bfloat16
                )

    return output


def cpu_golden_prefill_bf16(
    query,
    key,
    value,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_slot_indices,
    block_table,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    block_size,
    top_k,
    scale_value,
    inner_precise=4,
):
    """Full two-phase prefill golden: Phase1 QK/SM/PV -> workspace; Phase2 IFA combine."""
    kv_heads = key.shape[2]

    ws_o, ws_max, ws_sum = cpu_golden_prefill_phase1(
        query,
        key,
        value,
        k2q_row_ptr,
        k2q_q_indices,
        k2q_slot_indices,
        block_table,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_size,
        top_k,
        scale_value,
        inner_precise,
    )

    group_size = query.shape[1] // kv_heads
    d = query.shape[2]
    return cpu_golden_prefill_phase2(
        ws_o, ws_max, ws_sum, top_k, kv_heads, group_size, d
    )


def make_case(
    batch,
    q_seqlens,
    kv_seqlens,
    q_heads,
    kv_heads,
    head_dim=128,
    block_size=128,
    top_k=4,
    seed=42,
    shuffle_block_table=True,
):
    """Generate test tensors using decode-style causal block selection.

    q_seqlens/kv_seqlens may contain 0 (padding / dummy requests). Packed TND
    query only includes tokens with q_len>0; CSR packed rows skip kv_len=0.
    """
    assert len(q_seqlens) == batch and len(kv_seqlens) == batch
    group_size = q_heads // kv_heads
    max_kv_seqlen = max(kv_seqlens) if kv_seqlens else 0
    max_blocks_per_batch = max(
        1, ceil(max_kv_seqlen / block_size) if max_kv_seqlen else 1
    )
    total_q_tokens = sum(q_seqlens)

    torch.manual_seed(seed)
    query_fp32 = (
        torch.rand(total_q_tokens, q_heads, head_dim, dtype=torch.float32) * 2 - 1
    )
    key_fp32 = (
        torch.rand(
            max_blocks_per_batch * batch,
            block_size,
            kv_heads,
            head_dim,
            dtype=torch.float32,
        )
        * 2
        - 1
    )
    value_fp32 = (
        torch.rand(
            max_blocks_per_batch * batch,
            block_size,
            kv_heads,
            head_dim,
            dtype=torch.float32,
        )
        * 2
        - 1
    )

    block_table = generate_block_table(
        batch, max_blocks_per_batch, shuffle=shuffle_block_table
    )

    select_idx_decode = torch.full(
        (kv_heads, total_q_tokens, top_k), -1, dtype=torch.int32
    )
    select_num_idx = torch.zeros((kv_heads, total_q_tokens), dtype=torch.int32)

    q_offset = 0
    for b in range(batch):
        q_seqlen_b = q_seqlens[b]
        kv_seqlen_b = kv_seqlens[b]
        if q_seqlen_b == 0 or kv_seqlen_b == 0:
            q_offset += q_seqlen_b
            continue
        total_blocks_b = ceil(kv_seqlen_b / block_size)

        key_logical_b = torch.zeros(
            total_blocks_b * block_size, kv_heads, head_dim, dtype=torch.float32
        )
        for logical_id in range(total_blocks_b):
            physical_id = int(block_table[b, logical_id].item())
            key_logical_b[logical_id * block_size : (logical_id + 1) * block_size] = (
                key_fp32[physical_id]
            )
        key_flat_b = key_logical_b[:kv_seqlen_b, :, :]

        q_for_batch = query_fp32[q_offset : q_offset + q_seqlen_b, :, :]
        batch_select_idx, batch_select_num = generate_block_index_with_causal(
            q_for_batch,
            key_flat_b,
            q_seqlen_b,
            kv_seqlen_b,
            kv_heads,
            group_size,
            block_size,
            top_k,
        )

        select_idx_decode[:, q_offset : q_offset + q_seqlen_b, :] = batch_select_idx
        select_num_idx[:, q_offset : q_offset + q_seqlen_b] = batch_select_num
        q_offset += q_seqlen_b

    select_idx_global = decode_select_idx_to_prefill_global(
        select_idx_decode, q_seqlens, kv_seqlens, block_size
    )

    scale_value = 1.0 / math.sqrt(head_dim)
    actual_seq_lengths = torch.tensor(q_seqlens, dtype=torch.int32)
    actual_seq_lengths_kv = torch.tensor(kv_seqlens, dtype=torch.int32)

    k2q_row_ptr, k2q_q_indices, k2q_slot_indices = build_k2q_csr(
        select_idx_global,
        select_num_idx,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_size,
        kv_heads=kv_heads,
    )

    return {
        "query": query_fp32.to(torch.bfloat16),
        "key": key_fp32.to(torch.bfloat16),
        "value": value_fp32.to(torch.bfloat16),
        "select_idx_decode": select_idx_decode,
        "select_num_idx": select_num_idx,
        "select_idx": select_idx_global,
        "block_table": block_table,
        "k2q_row_ptr": torch.from_numpy(k2q_row_ptr).to(torch.int32),
        "k2q_q_indices": torch.from_numpy(k2q_q_indices).to(torch.int32),
        "k2q_slot_indices": torch.from_numpy(k2q_slot_indices).to(torch.int32),
        "q_seqlens": q_seqlens,
        "kv_seqlens": kv_seqlens,
        "actual_seq_lengths": actual_seq_lengths,
        "actual_seq_lengths_kv": actual_seq_lengths_kv,
        "kv_heads": kv_heads,
        "block_size": block_size,
        "top_k": top_k,
        "scale_value": scale_value,
    }


def packed_q_ids_to_padded(q_ids, q_seqlens, s_pad):
    """Map packed TND qToken -> BNSD/BSND padded flatten b*S+t."""
    offsets = [0]
    for ql in q_seqlens:
        offsets.append(offsets[-1] + ql)
    out = q_ids.clone()
    flat = out.view(-1)
    for i in range(flat.numel()):
        q = int(flat[i].item())
        if q < 0:
            continue
        b = 0
        while b + 1 < len(offsets) and q >= offsets[b + 1]:
            b += 1
        local = q - offsets[b]
        flat[i] = b * s_pad + local
    return out


def tnd_to_bnsd(query_tnd, q_seqlens, s_pad):
    """Packed [T, N, D] -> padded [B, N, S, D]."""
    n, d = query_tnd.shape[1], query_tnd.shape[2]
    b = len(q_seqlens)
    out = torch.zeros(b, n, s_pad, d, dtype=query_tnd.dtype)
    off = 0
    for bi, ql in enumerate(q_seqlens):
        if ql > 0:
            chunk = query_tnd[off : off + ql]  # [ql, N, D]
            out[bi, :, :ql, :] = chunk.permute(1, 0, 2)
        off += ql
    return out


def tnd_to_bsnd(query_tnd, q_seqlens, s_pad):
    """Packed [T, N, D] -> padded [B, S, N, D]."""
    n, d = query_tnd.shape[1], query_tnd.shape[2]
    b = len(q_seqlens)
    out = torch.zeros(b, s_pad, n, d, dtype=query_tnd.dtype)
    off = 0
    for bi, ql in enumerate(q_seqlens):
        if ql > 0:
            out[bi, :ql, :, :] = query_tnd[off : off + ql]
        off += ql
    return out


def tnd_lse_to_bnsd(lse_tnd, q_seqlens, s_pad):
    """Packed LSE [T, N] -> [B, N, S]."""
    n = lse_tnd.shape[1]
    b = len(q_seqlens)
    out = torch.zeros(b, n, s_pad, dtype=lse_tnd.dtype)
    off = 0
    for bi, ql in enumerate(q_seqlens):
        if ql > 0:
            out[bi, :, :ql] = lse_tnd[off : off + ql].transpose(0, 1)
        off += ql
    return out


def tnd_lse_to_bsnd(lse_tnd, q_seqlens, s_pad):
    """Packed LSE [T, N] -> [B, S, N]."""
    n = lse_tnd.shape[1]
    b = len(q_seqlens)
    out = torch.zeros(b, s_pad, n, dtype=lse_tnd.dtype)
    off = 0
    for bi, ql in enumerate(q_seqlens):
        if ql > 0:
            out[bi, :ql, :] = lse_tnd[off : off + ql]
        off += ql
    return out


def paged_kv_to_tnd(kv_paged, block_table, kv_seqlens, block_size):
    """Paged [phys, Bs, Hkv, D] -> packed TND [T_kv, Hkv, D]."""
    hkv, d = kv_paged.shape[2], kv_paged.shape[3]
    t = int(sum(kv_seqlens))
    out = torch.zeros(t, hkv, d, dtype=kv_paged.dtype)
    off = 0
    for b, kvl in enumerate(kv_seqlens):
        nblk = ceil(kvl / block_size) if kvl > 0 else 0
        for lb in range(nblk):
            take = min(block_size, kvl - lb * block_size)
            phys = int(block_table[b, lb].item())
            out[off : off + take] = kv_paged[phys, :take]
            off += take
    return out


def paged_kv_to_bnsd(kv_paged, block_table, kv_seqlens, block_size, s_kv_pad):
    """Paged -> BNSD [B, Hkv, S_kv, D]."""
    hkv, d = kv_paged.shape[2], kv_paged.shape[3]
    b = len(kv_seqlens)
    out = torch.zeros(b, hkv, s_kv_pad, d, dtype=kv_paged.dtype)
    for bi, kvl in enumerate(kv_seqlens):
        nblk = ceil(kvl / block_size) if kvl > 0 else 0
        for lb in range(nblk):
            take = min(block_size, kvl - lb * block_size)
            phys = int(block_table[bi, lb].item())
            beg = lb * block_size
            out[bi, :, beg : beg + take, :] = kv_paged[phys, :take].permute(1, 0, 2)
    return out


def paged_kv_to_bsnd(kv_paged, block_table, kv_seqlens, block_size, s_kv_pad):
    """Paged -> BSND [B, S_kv, Hkv, D]."""
    hkv, d = kv_paged.shape[2], kv_paged.shape[3]
    b = len(kv_seqlens)
    out = torch.zeros(b, s_kv_pad, hkv, d, dtype=kv_paged.dtype)
    for bi, kvl in enumerate(kv_seqlens):
        nblk = ceil(kvl / block_size) if kvl > 0 else 0
        for lb in range(nblk):
            take = min(block_size, kvl - lb * block_size)
            phys = int(block_table[bi, lb].item())
            beg = lb * block_size
            out[bi, beg : beg + take, :, :] = kv_paged[phys, :take]
    return out


def prepare_npu_case(data, layout="TND", paged=True):
    """Build layout-specific tensors + remapped CSR for the NPU op.

    Golden remains packed TND + paged KV. This helper only rewrites what the
    kernel consumes.
    """
    q_seqlens = data["q_seqlens"]
    kv_seqlens = data["kv_seqlens"]
    block_size = data["block_size"]
    s_q = max(q_seqlens) if any(q_seqlens) else 1
    s_kv = max(kv_seqlens) if any(kv_seqlens) else 1
    query = data["query"]
    key = data["key"]
    value = data["value"]
    bt = data["block_table"]
    q_idx = data["k2q_q_indices"]
    out = dict(data)
    out["layout"] = layout
    out["paged"] = paged
    out["s_q"] = s_q
    out["s_kv"] = s_kv

    if layout == "BNSD":
        out["query"] = tnd_to_bnsd(query, q_seqlens, s_q)
        out["k2q_q_indices"] = packed_q_ids_to_padded(q_idx, q_seqlens, s_q)
        if paged:
            out["key"], out["value"], out["block_table"] = key, value, bt
        else:
            out["key"] = paged_kv_to_bnsd(key, bt, kv_seqlens, block_size, s_kv)
            out["value"] = paged_kv_to_bnsd(value, bt, kv_seqlens, block_size, s_kv)
            out["block_table"] = None
    elif layout == "BSND":
        out["query"] = tnd_to_bsnd(query, q_seqlens, s_q)
        out["k2q_q_indices"] = packed_q_ids_to_padded(q_idx, q_seqlens, s_q)
        if paged:
            out["key"], out["value"], out["block_table"] = key, value, bt
        else:
            out["key"] = paged_kv_to_bsnd(key, bt, kv_seqlens, block_size, s_kv)
            out["value"] = paged_kv_to_bsnd(value, bt, kv_seqlens, block_size, s_kv)
            out["block_table"] = None
    else:
        # TND
        if paged:
            out["key"], out["value"], out["block_table"] = key, value, bt
        else:
            out["key"] = paged_kv_to_tnd(key, bt, kv_seqlens, block_size)
            out["value"] = paged_kv_to_tnd(value, bt, kv_seqlens, block_size)
            out["block_table"] = None
    return out


def layout_golden_attn(attn_tnd, q_seqlens, layout, s_q):
    if layout == "BNSD":
        return tnd_to_bnsd(attn_tnd, q_seqlens, s_q)
    if layout == "BSND":
        return tnd_to_bsnd(attn_tnd, q_seqlens, s_q)
    return attn_tnd


def layout_golden_lse(lse_tnd, q_seqlens, layout, s_q):
    if layout == "BNSD":
        return tnd_lse_to_bnsd(lse_tnd, q_seqlens, s_q).unsqueeze(-1)
    if layout == "BSND":
        return tnd_lse_to_bsnd(lse_tnd, q_seqlens, s_q).unsqueeze(-1)
    return lse_tnd.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMinimaxSparseAttentionSplitKvGolden(unittest.TestCase):
    """Validate prefill two-phase bf16 golden (k2q CSR + KV outer loop)."""

    def _max_abs_diff(self, a, b):
        return (a.float() - b.float()).abs().max().item()

    def _run_case(
        self,
        batch,
        q_seqlens,
        kv_seqlens,
        q_heads,
        kv_heads,
        head_dim=128,
        block_size=128,
        top_k=4,
        seed=42,
        check_decode=False,
        tol=5e-3,
        inner_precise=4,
    ):
        data = make_case(
            batch,
            q_seqlens,
            kv_seqlens,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            top_k,
            seed,
        )

        two_phase, lse = cpu_golden_prefill_bf16(
            data["query"],
            data["key"],
            data["value"],
            data["k2q_row_ptr"],
            data["k2q_q_indices"],
            data["k2q_slot_indices"],
            data["block_table"],
            data["actual_seq_lengths"],
            data["actual_seq_lengths_kv"],
            data["block_size"],
            data["top_k"],
            data["scale_value"],
            inner_precise,
        )

        qcentric = cpu_golden_prefill_qcentric_bf16(
            data["query"],
            data["key"],
            data["value"],
            data["select_idx"],
            data["block_table"],
            data["q_seqlens"],
            data["kv_seqlens"],
            block_size,
            top_k,
            data["scale_value"],
            data["select_num_idx"],
        )

        self.assertFalse(
            torch.any(torch.isnan(two_phase.float())), "two-phase output contains NaN"
        )
        self.assertFalse(torch.any(torch.isnan(lse)), "lse contains NaN")
        if sum(q_seqlens) > 0:
            self.assertFalse(torch.all(two_phase == 0), "two-phase output is all zeros")

        max_diff = self._max_abs_diff(two_phase, qcentric)
        mean_diff = (two_phase.float() - qcentric.float()).abs().mean().item()
        print(
            f"[golden] batch={batch} q={q_seqlens} kv={kv_seqlens} top_k={top_k} "
            f"innerPrecise={inner_precise} two_phase_vs_qcentric "
            f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}"
        )

        # q-centric uses fp32 S; innerPrecise=4 rounds S to bf16 so allow a bit more.
        cmp_tol = tol if inner_precise == 0 else max(tol, 2e-2)
        self.assertLess(
            max_diff, cmp_tol, f"two-phase vs q-centric mismatch: max_diff={max_diff}"
        )

        if check_decode:
            decode_ref = cpu_sparse_attention_score_bf16(
                data["query"],
                data["key"],
                data["value"],
                data["select_idx_decode"],
                data["block_table"],
                data["actual_seq_lengths"],
                data["actual_seq_lengths_kv"],
                data["kv_heads"],
                data["select_num_idx"],
                block_size,
                data["scale_value"],
            )
            decode_vs_qc = self._max_abs_diff(decode_ref, qcentric)
            print(f"  decode_vs_qcentric max_diff={decode_vs_qc:.6f}")
            self.assertLess(
                decode_vs_qc,
                tol,
                f"decode vs q-centric mismatch: max_diff={decode_vs_qc}",
            )

    def test_single_batch_small(self):
        self._run_case(
            batch=1,
            q_seqlens=[4],
            kv_seqlens=[512],
            q_heads=8,
            kv_heads=2,
            top_k=3,
            seed=42,
        )

    def test_single_batch_medium(self):
        self._run_case(
            batch=1,
            q_seqlens=[32],
            kv_seqlens=[2048],
            q_heads=16,
            kv_heads=4,
            top_k=5,
            seed=123,
        )

    def test_multi_batch(self):
        self._run_case(
            batch=3,
            q_seqlens=[8, 16, 4],
            kv_seqlens=[1024, 2048, 512],
            q_heads=8,
            kv_heads=2,
            top_k=4,
            seed=456,
        )

    def test_gqa_large_group(self):
        self._run_case(
            batch=1,
            q_seqlens=[16],
            kv_seqlens=[1024],
            q_heads=32,
            kv_heads=4,
            top_k=6,
            seed=789,
            check_decode=True,
        )

    def test_matches_decode_gqa(self):
        self._run_case(
            batch=1,
            q_seqlens=[16],
            kv_seqlens=[1024],
            q_heads=8,
            kv_heads=2,
            top_k=4,
            seed=606,
            check_decode=True,
        )

    def test_partial_last_block(self):
        self._run_case(
            batch=1,
            q_seqlens=[8],
            kv_seqlens=[300],
            q_heads=8,
            kv_heads=2,
            top_k=3,
            seed=101,
        )

    def test_top_k_1(self):
        self._run_case(
            batch=1,
            q_seqlens=[16],
            kv_seqlens=[1024],
            q_heads=8,
            kv_heads=2,
            top_k=1,
            seed=202,
        )

    def test_long_prefill(self):
        self._run_case(
            batch=1,
            q_seqlens=[128],
            kv_seqlens=[4096],
            q_heads=8,
            kv_heads=2,
            top_k=8,
            seed=303,
        )

    def test_k2q_csr_packed_row_q_list(self):
        """CSR packed row 0 lists non-contiguous Q tokens for a single batch."""
        select_idx = np.full((1, 6, 3), -1, dtype=np.int32)
        select_idx[0] = [
            [0, 2, 3],
            [1, 2, 3],
            [0, 2, 4],
            [1, 3, 4],
            [0, 4, 5],
            [2, 4, 5],
        ]
        select_num = np.full((1, 6), 3, dtype=np.int32)
        q_seqlens = [6]
        kv_seqlens = [768]
        block_size = 128
        row_ptr, q_idx, slot_idx = build_k2q_csr(
            select_idx, select_num, q_seqlens, kv_seqlens, block_size, kv_heads=1
        )
        start = int(row_ptr[0, 0])
        end = int(row_ptr[0, 1])
        self.assertEqual(q_idx[0, start:end].tolist(), [0, 2, 4])
        self.assertEqual(slot_idx[0, start:end].tolist(), [0, 0, 0])

    def test_k2q_csr_gqa_per_head(self):
        select_idx = np.full((2, 3, 2), -1, dtype=np.int32)
        select_idx[0, 0, 0] = 0
        select_idx[0, 2, 0] = 0
        select_idx[1, 1, 0] = 1
        select_idx[1, 2, 0] = 1
        select_num = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int32)
        q_seqlens = [3]
        kv_seqlens = [256]
        block_size = 128
        row_ptr, q_idx, _ = build_k2q_csr(
            select_idx, select_num, q_seqlens, kv_seqlens, block_size, kv_heads=2
        )
        s0, e0 = int(row_ptr[0, 0]), int(row_ptr[0, 1])
        s1, e1 = int(row_ptr[1, 1]), int(row_ptr[1, 2])
        self.assertEqual(q_idx[0, s0:e0].tolist(), [0, 2])
        self.assertEqual(q_idx[1, s1:e1].tolist(), [1, 2])

    def test_k2q_csr_matches_torch_reference(self):
        data = make_case(
            batch=2,
            q_seqlens=[4, 6],
            kv_seqlens=[512, 768],
            q_heads=8,
            kv_heads=2,
            top_k=4,
            seed=909,
        )
        k2q_row_ptr, k2q_q_idx, k2q_slot_idx = build_k2q_csr(
            data["select_idx"],
            data["select_num_idx"],
            data["actual_seq_lengths"],
            data["actual_seq_lengths_kv"],
            data["block_size"],
            kv_heads=data["kv_heads"],
        )

        q2k, _, _ = global_select_idx_to_batch_local(
            data["select_idx"],
            data["actual_seq_lengths"],
            data["actual_seq_lengths_kv"],
            data["block_size"],
            data["select_num_idx"],
            kv_heads=data["kv_heads"],
        )

        ref_row_ptr, ref_q_idx, ref_slot_idx = build_k2q_csr_torch_reference(
            torch.from_numpy(q2k),
            data["actual_seq_lengths"],
            data["actual_seq_lengths_kv"],
            data["block_size"],
            q_global_offset=True,
        )
        np.testing.assert_array_equal(k2q_row_ptr, ref_row_ptr.numpy())
        np.testing.assert_array_equal(k2q_q_idx, ref_q_idx.numpy())
        np.testing.assert_array_equal(k2q_slot_idx, ref_slot_idx.numpy())

    def test_decode_packed_row_roundtrip(self):
        q_seqlens = [4, 6]
        kv_seqlens = [300, 500]
        block_size = 128
        rows_per_batch = [ceil(k / block_size) for k in kv_seqlens]
        row_map, total_rows = _build_packed_row_map(
            np.array(rows_per_batch, dtype=np.int64)
        )
        for batch_idx in range(len(q_seqlens)):
            for local_blk in range(rows_per_batch[batch_idx]):
                packed = int(row_map[batch_idx, local_blk])
                b2, l2 = decode_packed_row(packed, kv_seqlens, block_size)
                self.assertEqual((b2, l2), (batch_idx, local_blk))
        self.assertEqual(total_rows, sum(rows_per_batch))

    def test_combine_required_for_topk_partials(self):
        data = make_case(
            batch=1,
            q_seqlens=[16],
            kv_seqlens=[1024],
            q_heads=8,
            kv_heads=2,
            top_k=4,
            seed=505,
        )
        combined, _ = cpu_golden_prefill_bf16(
            data["query"],
            data["key"],
            data["value"],
            data["k2q_row_ptr"],
            data["k2q_q_indices"],
            data["k2q_slot_indices"],
            data["block_table"],
            data["actual_seq_lengths"],
            data["actual_seq_lengths_kv"],
            data["block_size"],
            data["top_k"],
            data["scale_value"],
        )

        one_block_idx = data["select_idx"].clone()
        one_block_idx[:, :, 1:] = -1
        one_select_num = data["select_num_idx"].clone()
        one_select_num[:] = 1
        one_k2q_row_ptr, one_k2q_q_indices, one_k2q_slot_indices = build_k2q_csr(
            one_block_idx,
            one_select_num,
            data["actual_seq_lengths"],
            data["actual_seq_lengths_kv"],
            data["block_size"],
            kv_heads=data["kv_heads"],
        )
        one_partial, _ = cpu_golden_prefill_bf16(
            data["query"],
            data["key"],
            data["value"],
            one_k2q_row_ptr,
            one_k2q_q_indices,
            one_k2q_slot_indices,
            data["block_table"],
            data["actual_seq_lengths"],
            data["actual_seq_lengths_kv"],
            data["block_size"],
            1,
            data["scale_value"],
        )
        max_diff = self._max_abs_diff(combined, one_partial)
        self.assertGreater(max_diff, 1e-3)

    def test_inner_precise_0_and_4(self):
        for ip in (0, 4):
            with self.subTest(inner_precise=ip):
                self._run_case(
                    batch=1,
                    q_seqlens=[8],
                    kv_seqlens=[256],
                    q_heads=16,
                    kv_heads=1,
                    top_k=2,
                    seed=11,
                    inner_precise=ip,
                    check_decode=False,
                )

    def test_inner_precise_1(self):
        self._run_case(
            batch=1,
            q_seqlens=[8],
            kv_seqlens=[256],
            q_heads=16,
            kv_heads=1,
            top_k=2,
            seed=12,
            inner_precise=1,
            check_decode=False,
            tol=3e-2,
        )

    def test_padding_q_kv_len_zero_csr_and_coord(self):
        """Dummy requests (q_len=kv_len=0) occupy no packed rows."""
        data = make_case(
            batch=4,
            q_seqlens=[0, 8, 0, 4],
            kv_seqlens=[0, 256, 0, 128],
            q_heads=8,
            kv_heads=2,
            top_k=2,
            seed=22,
        )
        rows = [ceil(k / data["block_size"]) if k else 0 for k in data["kv_seqlens"]]
        self.assertEqual(rows, [0, 2, 0, 1])
        total_rows = int(data["k2q_row_ptr"].shape[1] - 1)
        self.assertEqual(total_rows, sum(rows))
        b0, l0 = decode_packed_row(0, data["kv_seqlens"], data["block_size"])
        self.assertEqual((b0, l0), (1, 0))
        self.assertEqual(int(data["query"].shape[0]), 12)
        two_phase, lse = cpu_golden_prefill_bf16(
            data["query"],
            data["key"],
            data["value"],
            data["k2q_row_ptr"],
            data["k2q_q_indices"],
            data["k2q_slot_indices"],
            data["block_table"],
            data["actual_seq_lengths"],
            data["actual_seq_lengths_kv"],
            data["block_size"],
            data["top_k"],
            data["scale_value"],
            4,
        )
        self.assertFalse(torch.any(torch.isnan(two_phase.float())))
        self.assertFalse(torch.all(two_phase == 0))

    def test_padding_leading_zero_init_coord(self):
        rows = [0, 1, 0, 2]
        b, blk = _init_packed_row_coord(rows)
        self.assertEqual((b, blk), (1, 0))
        coords = [(b, blk)]
        for _ in range(sum(rows) - 1):
            b, blk = _advance_packed_row_coord(b, blk, len(rows), rows)
            coords.append((b, blk))
        self.assertEqual(coords, [(1, 0), (3, 0), (3, 1)])

    def test_lse_finite_on_valid_tokens(self):
        data = make_case(
            batch=1,
            q_seqlens=[8],
            kv_seqlens=[256],
            q_heads=8,
            kv_heads=2,
            top_k=2,
            seed=33,
        )
        _, lse = cpu_golden_prefill_bf16(
            data["query"],
            data["key"],
            data["value"],
            data["k2q_row_ptr"],
            data["k2q_q_indices"],
            data["k2q_slot_indices"],
            data["block_table"],
            data["actual_seq_lengths"],
            data["actual_seq_lengths_kv"],
            data["block_size"],
            data["top_k"],
            data["scale_value"],
            0,
        )
        self.assertTrue(torch.all(torch.isfinite(lse)))

    def test_bnsd_qtoken_remap(self):
        data = make_case(
            batch=2,
            q_seqlens=[0, 4],
            kv_seqlens=[0, 128],
            q_heads=8,
            kv_heads=2,
            top_k=1,
            seed=44,
        )
        s_q = 4
        remapped = packed_q_ids_to_padded(data["k2q_q_indices"], data["q_seqlens"], s_q)
        valid = remapped >= 0
        # packed q in [0,4) -> padded batch1 tokens 4..7
        self.assertTrue(int(remapped[valid].min()) >= 4)
        self.assertTrue(int(remapped[valid].max()) < 8)
        bnsd = tnd_to_bnsd(data["query"], data["q_seqlens"], s_q)
        self.assertEqual(tuple(bnsd.shape), (2, 8, 4, 128))
        self.assertTrue(torch.all(bnsd[0] == 0))

    def test_kv_len_zero_golden_is_zeros(self):
        data = make_case(
            batch=1,
            q_seqlens=[8],
            kv_seqlens=[0],
            q_heads=8,
            kv_heads=2,
            top_k=2,
            seed=55,
        )
        self.assertEqual(int(data["k2q_row_ptr"].shape[1] - 1), 0)
        out, lse = cpu_golden_prefill_bf16(
            data["query"],
            data["key"],
            data["value"],
            data["k2q_row_ptr"],
            data["k2q_q_indices"],
            data["k2q_slot_indices"],
            data["block_table"],
            data["actual_seq_lengths"],
            data["actual_seq_lengths_kv"],
            data["block_size"],
            data["top_k"],
            data["scale_value"],
            4,
        )
        self.assertTrue(torch.all(out.float() == 0))
        self.assertTrue(torch.all(lse == 0))

    def test_prepare_npu_case_layouts(self):
        data = make_case(
            batch=2,
            q_seqlens=[0, 4],
            kv_seqlens=[0, 128],
            q_heads=8,
            kv_heads=2,
            top_k=1,
            seed=66,
        )
        bnsd = prepare_npu_case(data, layout="BNSD", paged=False)
        self.assertEqual(tuple(bnsd["query"].shape), (2, 8, 4, 128))
        self.assertEqual(tuple(bnsd["key"].shape), (2, 2, 128, 128))
        self.assertIsNone(bnsd["block_table"])
        bsnd = prepare_npu_case(data, layout="BSND", paged=False)
        self.assertEqual(tuple(bsnd["query"].shape), (2, 4, 8, 128))
        tnd_c = prepare_npu_case(data, layout="TND", paged=False)
        self.assertEqual(tuple(tnd_c["query"].shape), (4, 8, 128))
        self.assertEqual(tuple(tnd_c["key"].shape), (128, 2, 128))
        attn, lse = cpu_golden_prefill_bf16(
            data["query"],
            data["key"],
            data["value"],
            data["k2q_row_ptr"],
            data["k2q_q_indices"],
            data["k2q_slot_indices"],
            data["block_table"],
            data["actual_seq_lengths"],
            data["actual_seq_lengths_kv"],
            data["block_size"],
            data["top_k"],
            data["scale_value"],
            0,
        )
        g_attn = layout_golden_attn(attn, data["q_seqlens"], "BNSD", 4)
        g_lse = layout_golden_lse(lse, data["q_seqlens"], "BNSD", 4)
        self.assertEqual(tuple(g_attn.shape), (2, 8, 4, 128))
        self.assertEqual(tuple(g_lse.shape), (2, 8, 4, 1))
        self.assertTrue(torch.all(g_attn[0] == 0))


if __name__ == "__main__":
    unittest.main()
