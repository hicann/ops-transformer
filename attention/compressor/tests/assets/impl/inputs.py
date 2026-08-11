#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyend (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


import torch
import random
import numpy as np
import gc


def to_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def fill_tensor_from_value(tensor, value):
    if tensor is None or value is None or not torch.is_tensor(tensor):
        return
    data = torch.tensor(value, dtype=tensor.dtype, device=tensor.device)
    tensor.copy_(data.reshape(tensor.shape))


def get_seq_used_by_batch(batch_idx, S, seqused, cu_seqlens):
    if seqused is not None:
        return seqused[batch_idx]
    if cu_seqlens is not None:
        return cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
    return S


def build_block_table(
    state_cache,
    block_table,
    B,
    S_max,
    block_size,
    cmp_ratio,
    start_pos,
    seqused,
    cu_seqlens,
    cache_mode,
    S,
):
    if block_table is None or not torch.is_tensor(block_table):
        return

    block_table.zero_()
    if block_table.shape[0] < B:
        return

    next_block_id = 1
    if cache_mode == 1:
        block_num = state_cache.shape[0]
        for i in range(B):
            cur_start = start_pos[i] // cmp_ratio * cmp_ratio - cmp_ratio
            cur_end = start_pos[i] // cmp_ratio * cmp_ratio + cmp_ratio
            if start_pos[i] % cmp_ratio == 0:
                cur_end = start_pos[i]
            cur_end = min(cur_end, start_pos[i] + S)
            cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
            cur_end_block_id = (cur_end - 1) // block_size
            for j in range(
                cur_start_block_id, min(cur_end_block_id + 1, block_table.shape[1])
            ):
                if next_block_id < block_num:
                    block_table[i][j] = next_block_id
                    next_block_id += 1
            end_pos = get_seq_used_by_batch(i, S, seqused, cu_seqlens)
            next_start = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio - cmp_ratio
            next_end = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio
            if (start_pos[i] + end_pos) % cmp_ratio == 0:
                next_end = start_pos[i] + end_pos
            next_end = min(next_end, start_pos[i] + end_pos)
            next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
            next_end_block_id = (next_end - 1) // block_size
            for j in range(
                next_start_block_id, min(next_end_block_id + 1, block_table.shape[1])
            ):
                if next_block_id < block_num and int(block_table[i][j].item()) == 0:
                    block_table[i][j] = next_block_id
                    next_block_id += 1
    elif cache_mode == 2:
        block_table.zero_()
        if B > 0:
            perm = random.sample(list(range(B)), B)
            for i in range(B):
                block_table[i] = perm[i]


def _fill_seq_descriptors(
    x,
    state_cache,
    cmp_ratio,
    coff,
    cache_mode,
    cu_seqlens,
    seqused,
    start_pos,
    **kwargs,
):
    if x is None or not torch.is_tensor(x):
        return None, None, None, 0, 0, 0, 128

    block_size = 128
    if (
        state_cache is not None
        and torch.is_tensor(state_cache)
        and state_cache.dim() >= 2
    ):
        block_size = int(state_cache.shape[1])

    is_th = x.dim() == 2
    S_max = 0
    cu_seqlens_values = kwargs.get("cu_seqlens_values")
    seqused_values = kwargs.get("seqused_values")
    start_pos_values = kwargs.get("start_pos_values")

    if is_th:
        T = x.shape[0]
        if cu_seqlens_values is not None:
            cu_seqlens_list = [int(v) for v in cu_seqlens_values]
            if cu_seqlens is not None and torch.is_tensor(cu_seqlens):
                fill_tensor_from_value(cu_seqlens, cu_seqlens_list)
        else:
            print("Error: layout of x is [T, hidden_size], cu_seqlens is required!!!")
        B = len(cu_seqlens_list) - 1
        S = T // B
    else:
        cu_seqlens_list = None
        B = x.shape[0]
        S = x.shape[1]

    if seqused_values is not None:
        seqused_list = [int(v) for v in seqused_values]
    else:
        if is_th:
            seqused_list = [
                cu_seqlens_list[i + 1] - cu_seqlens_list[i] for i in range(B)
            ]
        else:
            seqused_list = [x.shape[1]] * B
    fill_tensor_from_value(seqused, seqused_list)

    if start_pos_values is not None:
        start_pos_list = [max(0, int(v)) for v in start_pos_values]
    else:
        start_pos_list = [0] * B
    fill_tensor_from_value(start_pos, start_pos_list)

    if is_th:
        for i in range(B):
            if start_pos_list[i] + cu_seqlens_list[i + 1] - cu_seqlens_list[i] > S_max:
                S_max = start_pos_list[i] + cu_seqlens_list[i + 1] - cu_seqlens_list[i]
    else:
        S_max = max(start_pos_list) + S

    if is_th:
        S = max(seqused_list)

    return cu_seqlens_list, seqused_list, start_pos_list, B, S, S_max, block_size


def apply_batch_slice_seeded(
    x,
    wkv,
    wgate,
    state_cache,
    state_block_table,
    ape,
    cmp_ratio,
    coff,
    cache_mode,
    start_pos_list,
    **kwargs,
):
    """根据 batch_axis/batch_slice_info/batch_seed 生成切片数据并替换输入。

    - batch_axis: 标记每个输入的 batch/seq 轴位置
    - batch_slice_info: 切片信息，指定要替换的 batch/seq 范围
    - batch_seed: 随机种子，确保跨用例相同 seed 生成相同数据
    - input_ranges: ttk 框架自动传递，按 tensor 索引给出 (lo, hi) 取值范围

    流程：
    1. 使用 batch_seed 生成 slice 片段的数据输入(取值范围取自 input_ranges[idx])
    2. 将 slice 片段的数据，根据 slice 信息替换对应的输入位置
    """
    batch_axis = kwargs.get("batch_axis")
    batch_slice_info = kwargs.get("batch_slice_info")
    batch_seed = kwargs.get("batch_seed")
    input_ranges = kwargs.get("input_ranges")

    if batch_axis is None or batch_slice_info is None or batch_seed is None:
        return
    batch_axis = kwargs.get("batch_axis")
    # 输入 tensor 列表（与 tensor_view_shapes 顺序一致）
    is_th = x.dim() == 2
    input_tensors = [x, wkv, wgate, state_cache, ape]
    for idx, tensor in enumerate(input_tensors):
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if idx >= len(batch_axis) or idx >= len(batch_slice_info):
            continue

        axes = batch_axis[idx]

        slices = batch_slice_info[idx]
        seed = batch_seed[idx]
        if axes is None or slices is None:
            continue

        # 从 input_ranges 取该 tensor 的取值范围，缺省 (-10, 10)
        if input_ranges and idx < len(input_ranges) and input_ranges[idx] is not None:
            rng_range = input_ranges[idx]
            lo, hi = rng_range[0], rng_range[1]
        else:
            lo, hi = -10, 10

        # 获取切片后的 shape
        slice_idx = 0
        for axis_pos in axes:
            if slices[axis_pos] is None or seed[axis_pos] is None:
                continue
            sliced_shape = list(tensor.shape)
            for sl, seed_value in zip(slices[axis_pos], seed[axis_pos]):
                rng = np.random.RandomState(seed_value)
                if sl is None:
                    continue
                start = sl[0]
                end = sl[1]
                length = end - start
                if coff == 1:
                    if is_th:
                        sliced_shape[axis_pos] = length
                        data = rng.uniform(lo, hi, size=tuple(sliced_shape)).astype(
                            np.float32
                        )
                        tensor[start:end, :] = torch.from_numpy(data).to(tensor.dtype)
                    else:
                        if axis_pos == 1:
                            if slices[0] is None:
                                bidx = 0
                            else:
                                bidx = slices[0][slice_idx][0]
                                slice_idx += 1
                            sliced_shape[0] = 1
                            sliced_shape[axis_pos] = length
                            data = rng.uniform(lo, hi, size=tuple(sliced_shape)).astype(
                                np.float32
                            )
                            tensor[bidx, start:end, :] = torch.from_numpy(data).to(
                                tensor.dtype
                            )
                        else:
                            sliced_shape[axis_pos] = length
                            data = rng.uniform(lo, hi, size=tuple(sliced_shape)).astype(
                                np.float32
                            )
                            tensor[start:end, :, :] = torch.from_numpy(data).to(
                                tensor.dtype
                            )
                            if cache_mode == 2:
                                cache_shape = list(state_cache.shape)
                                cache_shape[0] = length
                                cache_rng = np.random.RandomState(seed_value)
                                cache_range = input_ranges[3]
                                cache_lo, cache_hi = cache_range[0], cache_range[1]
                                cache_data = cache_rng.uniform(
                                    cache_lo, cache_hi, size=tuple(cache_shape)
                                ).astype(np.float32)
                                block_id = state_block_table[start]
                                state_cache[block_id:(block_id + length), :, :] = torch.from_numpy(
                                    cache_data
                                ).to(state_cache.dtype)
                            else:
                                cache_shape = list(state_cache.shape)
                                cache_shape[0] = length
                                start_seq_id = start_pos_list[start]
                                cur_seq_id = start_seq_id - start_seq_id % cmp_ratio - cmp_ratio if start_seq_id >=  cmp_ratio else start_seq_id - start_seq_id % cmp_ratio
                                block_id = state_block_table[start][int(cur_seq_id // state_cache.shape[1])]
                                cache_rng = np.random.RandomState(seed_value)
                                cache_range = input_ranges[3]
                                cache_lo, cache_hi = cache_range[0], cache_range[1]
                                cache_data = cache_rng.uniform(
                                    cache_lo, cache_hi, size=tuple(cache_shape)
                                ).astype(np.float32)
                                state_cache[
                                    block_id : (block_id + length),
                                    :,
                                    :,
                                ] = torch.from_numpy(cache_data).to(state_cache.dtype)
                elif coff == 2:
                    if is_th:
                        if start < cmp_ratio:
                            continue
                        sliced_shape[axis_pos] = length + cmp_ratio
                        data = rng.uniform(lo, hi, size=tuple(sliced_shape)).astype(
                            np.float32
                        )
                        tensor[(start - cmp_ratio) : end, :] = torch.from_numpy(
                            data
                        ).to(tensor.dtype)
                    else:
                        if axis_pos == 1:
                            if slices[0] is None:
                                bidx = 0
                            else:
                                bidx = slices[0][slice_idx][0]
                                slice_idx += 1
                            sliced_shape[0] = 1
                            sliced_shape[axis_pos] = length + cmp_ratio
                            data = rng.uniform(lo, hi, size=tuple(sliced_shape)).astype(
                                np.float32
                            )
                            tensor[bidx, (start - cmp_ratio) : end, :] = torch.from_numpy(
                                data
                            ).to(tensor.dtype)
                        else:
                            sliced_shape[axis_pos] = length
                            data = rng.uniform(lo, hi, size=tuple(sliced_shape)).astype(
                                np.float32
                            )
                            tensor[start:end, :, :] = torch.from_numpy(data).to(
                                tensor.dtype
                            )
                            if cache_mode == 2:
                                cache_shape = list(state_cache.shape)
                                cache_shape[0] = length
                                cache_rng = np.random.RandomState(seed_value)
                                cache_range = input_ranges[3]
                                cache_lo, cache_hi = cache_range[0], cache_range[1]
                                cache_data = cache_rng.uniform(
                                    cache_lo, cache_hi, size=tuple(cache_shape)
                                ).astype(np.float32)
                                block_id = state_block_table[start]
                                state_cache[block_id:(block_id + length), :, :] = torch.from_numpy(
                                    cache_data
                                ).to(state_cache.dtype)
                            else:
                                cache_shape = list(state_cache.shape)
                                cache_shape[0] = length
                                start_seq_id = start_pos_list[start]
                                cur_seq_id = start_seq_id - start_seq_id % cmp_ratio - cmp_ratio if start_seq_id >=  cmp_ratio else start_seq_id - start_seq_id % cmp_ratio
                                block_id = state_block_table[start][int(cur_seq_id // state_cache.shape[1])]
                                cache_rng = np.random.RandomState(seed_value)
                                cache_range = input_ranges[3]
                                cache_lo, cache_hi = cache_range[0], cache_range[1]
                                cache_data = cache_rng.uniform(
                                    cache_lo, cache_hi, size=tuple(cache_shape)
                                ).astype(np.float32)
                                state_cache[
                                    block_id : (block_id + length),
                                    :,
                                    :,
                                ] = torch.from_numpy(cache_data).to(state_cache.dtype)

def generate_compressor_inputs(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    cmp_ratio=4,
    *,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    coff=1,
    cache_mode=1,
    **kwargs,
):
    cu_seqlens_list, seqused_list, start_pos_list, B, S, S_max, block_size = (
        _fill_seq_descriptors(
            x,
            state_cache,
            cmp_ratio,
            coff,
            cache_mode,
            cu_seqlens,
            seqused,
            start_pos,
            **kwargs,
        )
    )

    if x is None or not torch.is_tensor(x):
        return

    cmp_ratio_val = int(cmp_ratio) if cmp_ratio is not None else 4
    cache_mode_val = int(cache_mode) if cache_mode is not None else 1

    if state_block_table is not None and torch.is_tensor(state_block_table):
        build_block_table(
            state_cache,
            state_block_table,
            B,
            S_max,
            block_size,
            cmp_ratio_val,
            start_pos_list,
            seqused_list,
            cu_seqlens_list,
            cache_mode_val,
            S,
        )
    # batch一致性切片赋值
    apply_batch_slice_seeded(
        x,
        wkv,
        wgate,
        state_cache,
        state_block_table,
        ape,
        cmp_ratio,
        coff,
        cache_mode_val,
        start_pos_list,
        **kwargs,
    )
    gc.collect()


def aclnn_compressor_input(
    x,
    wkv,
    wgate,
    stateCacheRef,
    ape,
    stateBlockTable,
    cuSeqlens,
    seqused,
    startPos,
    cmpRatio,
    coff,
    cacheMode,
    stateCacheStrideDim0,
    gradEnabled,
    cmpKv,
    softmaxScoreOut,
    kvOut,
    **kwargs,
):
    cu_seqlens_list, seqused_list, start_pos_list, B, S, S_max, block_size = (
        _fill_seq_descriptors(
            x,
            stateCacheRef,
            cmpRatio,
            coff,
            cacheMode,
            cuSeqlens,
            seqused,
            startPos,
            **kwargs,
        )
    )

    if x is None or not torch.is_tensor(x):
        return

    if stateBlockTable is not None and torch.is_tensor(stateBlockTable):
        build_block_table(
            stateCacheRef,
            stateBlockTable,
            B,
            S_max,
            block_size,
            cmpRatio,
            start_pos_list,
            seqused_list,
            cu_seqlens_list,
            cacheMode,
            S,
        )

    apply_batch_slice_seeded(
        x,
        wkv,
        wgate,
        stateCacheRef,
        stateBlockTable,
        ape,
        cmpRatio,
        coff,
        cacheMode,
        start_pos_list,
        **kwargs,
    )
    gc.collect()
