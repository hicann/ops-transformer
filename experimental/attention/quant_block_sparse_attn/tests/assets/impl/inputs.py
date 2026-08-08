#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Input customization for quant_block_sparse_attn TTK cases.

E2E customize_inputs contract: modify tensors in-place via x.copy_(value).
No return value. Tensor shapes/dtypes are pre-allocated by TTK from CSV.
"""

import math
import random

import torch

FP8_DTYPE = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else None


FP8_E4M3_MAX = 448.0
SCALE_EPSILON = 1e-8


def _fp8_rand(shape, low=-1.0, high=1.0, generator=None):
    return (
        torch.empty(shape, dtype=torch.float32)
        .uniform_(low, high, generator=generator)
        .to(FP8_DTYPE)
    )


def _quant_fp32_to_fp8(tensor, quant_scale):
    quantized = tensor.to(torch.float32) * quant_scale
    quantized = torch.clamp(quantized, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    return quantized.to(FP8_DTYPE).contiguous()


def _quantize_per_token_head(tensor):
    row_max = torch.abs(tensor).amax(dim=-1, keepdim=True)
    row_max = torch.maximum(
        row_max, torch.tensor(SCALE_EPSILON, dtype=torch.float32, device=tensor.device)
    )
    quant_scale = FP8_E4M3_MAX / row_max
    return _quant_fp32_to_fp8(tensor, quant_scale), (
        1.0 / quant_scale
    ).squeeze(-1).contiguous()


def _quantize_value_per_head(tensor):
    head_max = torch.abs(tensor).amax(dim=(0, 1, 3), keepdim=True)
    head_max = torch.maximum(
        head_max, torch.tensor(SCALE_EPSILON, dtype=torch.float32, device=tensor.device)
    )
    quant_scale = FP8_E4M3_MAX / head_max
    value = _quant_fp32_to_fp8(tensor, quant_scale)
    return value, (1.0 / quant_scale).reshape(tensor.shape[2]).contiguous()


def _rand_fullquant_source(shape, amp_low, amp_high, generator, amp_shape=None):
    base = torch.empty(shape, dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
    if amp_shape is None:
        amp_shape = shape[:-1] + (1,)
    log_low = math.log10(max(float(amp_low), SCALE_EPSILON))
    log_high = math.log10(max(float(amp_high), SCALE_EPSILON))
    exponent = torch.empty(amp_shape, dtype=torch.float32).uniform_(
        log_low, log_high, generator=generator
    )
    amps = torch.pow(torch.tensor(10.0, dtype=torch.float32), exponent)
    return base * amps


def _make_block_table(batch, seq_len, block_size, pattern, rng, blocknum=None, q_block_num=1):
    kv_block_num = math.ceil(seq_len / block_size)
    width = kv_block_num
    needed_block_num = batch * width

    if blocknum is None or blocknum >= needed_block_num:
        block_table = torch.empty((batch, kv_block_num), dtype=torch.int32)
        for batch_idx in range(batch):
            ids = list(
                range(
                    batch_idx * kv_block_num, (batch_idx + 1) * kv_block_num
                )
            )
            if pattern == "reverse":
                ids.reverse()
            elif pattern == "random":
                rng.shuffle(ids)
            block_table[batch_idx] = torch.tensor(ids, dtype=torch.int32)
        return block_table

    if blocknum < width:
        # 物理块池小于每 batch 的逻辑块数：前几列共享（全部指向物理块 0），
        # 其余列互不相同，所有 batch 行一致，总物理块数精确等于 blocknum。
        shared_cols = width - blocknum + 1
        if shared_cols <= 0 or shared_cols > width:
            raise ValueError(f"blocknum={blocknum} out of range [1, {width}]")
        block_table = torch.zeros((batch, width), dtype=torch.int32)
        for batch_idx in range(batch):
            for block_idx in range(shared_cols, width):
                block_table[batch_idx, block_idx] = block_idx - shared_cols + 1
        return block_table

    # batch >= 2 且 blocknum >= width：共享前缀（前几列各 batch 相同）+ 每 batch 独有尾部
    block_table = torch.empty((batch, width), dtype=torch.int32)
    shared_count = needed_block_num - blocknum
    full_shared_cols = shared_count // (batch - 1)
    partial_share = shared_count % (batch - 1)
    next_id = 0
    col = 0

    for _ in range(full_shared_cols):
        for batch_idx in range(batch):
            block_table[batch_idx, col] = next_id
        next_id += 1
        col += 1

    if partial_share > 0:
        shared_id = next_id
        next_id += 1
        for batch_idx in range(partial_share + 1):
            block_table[batch_idx, col] = shared_id
        for batch_idx in range(partial_share + 1, batch):
            block_table[batch_idx, col] = next_id
            next_id += 1
        col += 1

    for batch_idx in range(batch):
        for block_idx in range(col, width):
            block_table[batch_idx, block_idx] = next_id
            next_id += 1

    if next_id != blocknum:
        raise AssertionError(f"block_table generated {next_id} physical blocks, expected {blocknum}")
    return block_table


def _allowed_blocks(
    mask_mode, qb_idx, sparse_q_block_size, sparse_kv_block_size, q_len, kv_len
):
    block_num = math.ceil(kv_len / sparse_kv_block_size)
    if block_num <= 0:
        return []
    if mask_mode == 0:
        return list(range(block_num))
    if mask_mode != 3:
        raise ValueError(f"unsupported mask_mode: {mask_mode}")
    max_token = (qb_idx + 1) * sparse_q_block_size - 1
    max_token += kv_len - q_len
    if max_token < 0:
        return []
    max_block = min(block_num - 1, max_token // sparse_kv_block_size)
    return list(range(max_block + 1))


def _select_blocks(blocks, sparse_count, pattern, rng):
    if sparse_count <= 0 or not blocks or pattern == "empty":
        return []
    if pattern in ("sequential", "dense", "causal"):
        return blocks[: min(sparse_count, len(blocks))]
    if pattern == "reverse":
        return list(reversed(blocks[-min(sparse_count, len(blocks)) :]))
    if pattern == "tail":
        selected = blocks[: max(0, min(sparse_count, len(blocks)) - 1)]
        if blocks[-1] not in selected:
            selected.append(blocks[-1])
        return selected[:sparse_count]
    if pattern == "random":
        selected = blocks[:]
        rng.shuffle(selected)
        return selected[: min(sparse_count, len(selected))]
    raise ValueError(f"unsupported sparse_pattern: {pattern}")


def _make_sparse_indices(
    B,
    N1,
    N2,
    S1,
    S2,
    sparse_q_block_size,
    sparse_kv_block_size,
    mask_mode,
    sparse_pattern,
    sparse_count,
    q_lengths,
    kv_lengths,
    rng,
):
    group = N1 // N2
    qb_max = math.ceil(S1 / sparse_q_block_size)
    kv_max = math.ceil(S2 / sparse_kv_block_size)
    sparse_indices = torch.full(
        (B, N1, qb_max, kv_max), fill_value=-1, dtype=torch.int32
    )
    sparse_seq_len = torch.zeros((B, N1, qb_max), dtype=torch.int32)

    for batch_idx in range(B):
        for qb_idx in range(qb_max):
            allowed = _allowed_blocks(
                mask_mode,
                qb_idx,
                sparse_q_block_size,
                sparse_kv_block_size,
                q_lengths[batch_idx],
                kv_lengths[batch_idx],
            )
            for n2_idx in range(N2):
                for group_idx in range(group):
                    head_idx = n2_idx * group + group_idx
                    selected = _select_blocks(
                        allowed, sparse_count, sparse_pattern, rng
                    )
                    sparse_seq_len[batch_idx, head_idx, qb_idx] = len(selected)
                    if selected:
                        sparse_indices[batch_idx, head_idx, qb_idx, : len(selected)] = (
                            torch.tensor(selected, dtype=torch.int32)
                        )
    return sparse_indices, sparse_seq_len


def _dense_to_pa(dense_key, dense_value, dense_k_scale, block_table, block_size, num_blocks=None):
    B, S2, N2, D = dense_key.shape
    required = int(block_table.max().item()) + 1
    if num_blocks is not None and num_blocks < required:
        raise ValueError(
            f"pre-allocated num_blocks={num_blocks} < required physical blocks {required}"
        )
    # 预分配物理池可能大于 block_table 实际用到的块数，按预分配大小定 PA 张量，尾部补零
    block_num = num_blocks if num_blocks is not None else required
    key_pa = torch.zeros((block_num, N2, block_size, D), dtype=dense_key.dtype)
    value_pa = torch.zeros((block_num, N2, block_size, D), dtype=dense_value.dtype)
    k_scale_pa = torch.zeros((block_num, N2, block_size), dtype=torch.float32)

    for b in range(B):
        for logical_block in range(block_table.shape[1]):
            physical = int(block_table[b, logical_block].item())
            start = logical_block * block_size
            end = min(start + block_size, S2)
            token_count = end - start
            if token_count <= 0:
                continue
            key_pa[physical, :, :token_count] = dense_key[b, start:end].permute(1, 0, 2)
            value_pa[physical, :, :token_count] = dense_value[b, start:end].permute(
                1, 0, 2
            )
            k_scale_pa[physical, :, :token_count] = dense_k_scale[b, start:end].permute(
                1, 0
            )
    return key_pa, value_pa, k_scale_pa


def customize_inputs(
    query,
    key,
    value,
    q_descale,
    k_descale,
    v_descale,
    p_scale,
    sparse_indices,
    sparse_seq_len,
    atten_mask,
    *,
    softmax_scale=1.0,
    sparse_q_block_size=128,
    sparse_kv_block_size=128,
    max_seqlen_q=0,
    max_seqlen_kv=0,
    layout_q="BSND",
    layout_kv="PA_BNSD",
    mask_mode=3,
    quant_mode=1,
    return_softmax_lse=False,
    sparse_count=0,
    sparse_pattern="sequential",
    block_table_pattern="sequential",
    actlen_mode="full",
    p_scale_value=1.0,
    seed=0,
    **kwargs,
):
    """Fill input tensors in-place with consistent test data for quant_block_sparse_attn.

    E2E customize_inputs contract: modify tensors in-place, no return value.
    Generation params (sparse_pattern, etc.) arrive via
    extra_attrs from CSV attributes (non-operator params).
    """
    if FP8_DTYPE is None:
        raise RuntimeError(
            "torch.float8_e4m3fn is required for quant_block_sparse_attn test data"
        )

    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)

    D = int(query.shape[-1])
    if layout_q == "BSND":
        B, S1, N1 = int(query.shape[0]), int(query.shape[1]), int(query.shape[2])
    elif layout_q == "NTD":
        N1 = int(query.shape[0])
        B = int(sparse_indices.shape[0])
        S1 = max_seqlen_q if max_seqlen_q > 0 else int(query.shape[1]) // B
    else:
        N1 = int(query.shape[1])
        B = int(sparse_indices.shape[0])
        S1 = max_seqlen_q if max_seqlen_q > 0 else int(query.shape[0]) // B

    N2 = int(v_descale.shape[0])
    block_size = int(key.shape[2]) if key.dim() == 4 else sparse_kv_block_size
    num_blocks = (
        int(key.shape[0])
        if key.dim() == 4
        else math.ceil(max_seqlen_kv / sparse_kv_block_size)
    )
    # S2 由 sparse_indices 的逻辑 KV 块数（末维）反推，保证 sparse_indices 生成结果与预分配形状一致；
    # 不能用 key.shape[0]（物理块数）推，物理池在共享场景下可小于逻辑块数。
    kv_max = int(sparse_indices.shape[-1])
    S2 = kv_max * sparse_kv_block_size

    q_lengths = [S1] * B
    kv_lengths = [S2] * B

    amp_high = FP8_E4M3_MAX
    if layout_q == "NTD":
        query_source = _rand_fullquant_source(
            (N1, B * S1, D), amp_high * 0.01, amp_high, generator
        )
    else:
        query_source = _rand_fullquant_source(
            (B * S1, N1, D), amp_high * 0.01, amp_high, generator
        )
    query_fp8, q_scale = _quantize_per_token_head(query_source)
    query.copy_(query_fp8)

    dense_key_source = _rand_fullquant_source(
        (B, S2, N2, D), 1.0, amp_high, generator
    )
    dense_value_source = _rand_fullquant_source(
        (B, S2, N2, D), 1.0, amp_high, generator, amp_shape=(B, 1, N2, 1)
    )
    dense_key, dense_k_scale_base = _quantize_per_token_head(dense_key_source)
    dense_value, v_scale = _quantize_value_per_head(dense_value_source)

    dense_k_scale = torch.zeros((B, S2, N2), dtype=torch.float32)
    for b in range(B):
        dense_k_scale[b] = dense_k_scale_base[b]

    block_table = _make_block_table(
        B,
        S2,
        sparse_kv_block_size,
        block_table_pattern,
        rng,
        blocknum=int(key.shape[0]) if key.dim() == 4 else None,
    )
    key_pa, value_pa, k_scale_pa = _dense_to_pa(
        dense_key,
        dense_value,
        dense_k_scale,
        block_table,
        sparse_kv_block_size,
        num_blocks=int(key.shape[0]) if key.dim() == 4 else None,
    )

    key.copy_(key_pa)
    value.copy_(value_pa)

    q_descale.copy_(q_scale)
    # k_scale_pa 为 3D (blockNum, N2, blockSize)，k_descale 要求 4D (blockNum, N2, blockSize, 1)，
    # 补末尾单例维再拷贝，避免 3D 被错误广播成 [blockNum, N2, blockSize, ...]。
    k_descale.copy_(k_scale_pa.unsqueeze(-1))
    v_descale.copy_(v_scale)

    # p_scale 必填（shape [1]）：p_scale_value 未指定时生成 [-1, 1] 随机 float，否则用给定值
    if p_scale_value is None:
        p_scale_data = torch.tensor([rng.uniform(-1.0, 1.0)], dtype=torch.float32)
    else:
        p_scale_data = torch.tensor([float(p_scale_value)], dtype=torch.float32)
    p_scale.copy_(p_scale_data)

    sparse_indices_data, sparse_seq_len_data = _make_sparse_indices(
        B,
        N1,
        N2,
        S1,
        S2,
        sparse_q_block_size,
        sparse_kv_block_size,
        mask_mode,
        sparse_pattern,
        sparse_count,
        q_lengths,
        kv_lengths,
        rng,
    )
    sparse_indices.copy_(sparse_indices_data)
    sparse_seq_len.copy_(sparse_seq_len_data)

    if int(mask_mode) == 3:
        # 因果掩码：仅在需要掩码时生成；mask_mode=0（无掩码）时预分配的 atten_mask 为空，
        # 用形状 [0] 生成 tril 会失败，故跳过。
        atten_mask_data = torch.tril(torch.ones(atten_mask.shape, dtype=torch.uint8)).T
        atten_mask.copy_(atten_mask_data)
    elif atten_mask.numel() > 0:
        atten_mask.zero_()
