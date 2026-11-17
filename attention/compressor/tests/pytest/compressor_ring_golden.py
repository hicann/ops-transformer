# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch

from compressor_golden import cpu_compressor, get_seq_used_by_batch


def cpu_compressor_ring(
    x,
    wkv,
    wgate,
    kv_state,
    score_state,
    update_kv_position,
    update_score_position,
    ape,
    *,
    block_table,
    start_pos,
    cu_seqlens=None,
    seqused=None,
    cmp_ratio=4,
    coff=1,
    cache_mode=2,
):
    """Use the unchanged shared golden on an unrolled coff=1 ring window.

    Copy historical prefixes before any write, compute with disjoint linear
    storage, then persist only the last capacity tokens at their ring positions.
    Reducing start positions modulo cmp_ratio preserves window and APE phases.
    This reference is used only by the new ring tests.
    """
    assert coff == 1 and cache_mode == 2
    batch, capacity = len(start_pos), kv_state.shape[1]
    lengths = [
        int(get_seq_used_by_batch(b, x.shape[1], seqused, cu_seqlens))
        for b in range(batch)
    ]
    phases = start_pos % cmp_ratio
    linear_size = max(int(phases[b]) + lengths[b] for b in range(batch))
    linear_size = max(linear_size, 1)
    shape = (batch + 1, linear_size, kv_state.shape[-1])
    linear_kv = torch.zeros(shape, dtype=kv_state.dtype)
    linear_score = torch.zeros(shape, dtype=score_state.dtype)
    linear_table = torch.arange(1, batch + 1, dtype=torch.int32).reshape(batch, 1)
    for b in range(batch):
        prefix = int(phases[b])
        rows = torch.arange(int(start_pos[b]) - prefix, int(start_pos[b])) % capacity
        block = int(block_table[b])
        linear_kv[b + 1, :prefix] = kv_state[block, rows]
        linear_score[b + 1, :prefix] = score_state[block, rows]

    result = cpu_compressor(
        x,
        wkv,
        wgate,
        linear_kv,
        linear_score,
        torch.zeros_like(linear_kv, dtype=torch.bool),
        torch.zeros_like(linear_score, dtype=torch.bool),
        ape,
        block_table=linear_table,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        start_pos=phases,
        cmp_ratio=cmp_ratio,
        coff=1,
        cache_mode=1,
    )
    for b in range(batch):
        count = min(lengths[b], capacity)
        end = int(start_pos[b]) + lengths[b]
        rows = torch.arange(end - count, end) % capacity
        offset = int(phases[b]) + lengths[b] - count
        block = int(block_table[b])
        kv_state[block, rows] = linear_kv[b + 1, offset : offset + count]
        score_state[block, rows] = linear_score[b + 1, offset : offset + count]
        update_kv_position[block, rows] = True
        update_score_position[block, rows] = True
    return result
