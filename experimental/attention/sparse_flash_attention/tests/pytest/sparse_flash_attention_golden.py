#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math
import os

import numpy as np
import torch

SCALE_VALUE = 192**-0.5
RANDOM_BLOCK_TABLE = False


def calculate_special_sum(seq_lens):
    """计算T长度，空序列按1计（与算子PA场景约定一致）。"""
    if not seq_lens:
        return 0
    total = 0
    for element in seq_lens:
        total += 1 if element == 0 else element
    return total


def trans_to_kv_cache(input_tensor, block_size, cur_kv_seqlen, max_seq_len):
    """将连续KV张量转换为PA KV cache格式，返回(cache_tensor, block_table)。

    支持TND输入shape为(T, N, D)与BSND输入shape为(B, S, N, D)。
    """
    device = input_tensor.device
    ndim = input_tensor.dim()
    B = np.sum(np.array(cur_kv_seqlen) >= 0) - 1
    block_num_per_batch = []
    total_seq_block_cnt = 0
    for i in range(B):
        seqlen = cur_kv_seqlen[i + 1] - cur_kv_seqlen[i]
        if seqlen == 0:
            seqlen = 1
        seq_block_cnt = math.ceil(seqlen / block_size)
        block_num_per_batch.append(seq_block_cnt)
        total_seq_block_cnt += seq_block_cnt
    block_table_list = (
        torch.randperm(total_seq_block_cnt)
        if RANDOM_BLOCK_TABLE
        else torch.arange(0, total_seq_block_cnt, 1)
    )

    block_tables = torch.zeros(B, math.ceil(max_seq_len / block_size)).to(torch.int32)
    N, D = input_tensor.shape[-2], input_tensor.shape[-1]
    cache_tensor = torch.zeros(total_seq_block_cnt, block_size, N, D).to(
        input_tensor.dtype
    )
    start = 0
    cur_block_id = 0
    for i in range(B):
        seqlen = cur_kv_seqlen[i + 1] - cur_kv_seqlen[i]
        if seqlen == 0:
            seqlen = 1
        if seqlen < 0:
            break
        seq_block_cnt = block_num_per_batch[i]
        block_tables[i, :seq_block_cnt] = block_table_list[
            cur_block_id : cur_block_id + seq_block_cnt
        ]
        for block_idx in range(seq_block_cnt):
            block_offset = block_idx * block_size
            if ndim == 4:
                seq_tensor = input_tensor[i, block_offset : block_offset + block_size]
            else:
                seq_tensor = input_tensor[
                    start + block_offset : start + block_offset + block_size
                ]
            remain_size = min(block_size, seqlen - block_offset)
            cache_tensor[block_tables[i][block_idx], :remain_size] = seq_tensor[
                :remain_size
            ]
        cur_block_id += seq_block_cnt
        start += seqlen

    return cache_tensor, block_tables.to(device).detach()


def gather_kv(
    k_tensor,
    v_tensor,
    sparse_indices,
    seq_lengths_q,
    seq_lengths_kv,
    batch,
    n2_idx,
    s1_idx,
    sparse_count,
    sparse_block_size,
    sparse_mode,
    layout,
    kv_start_pos,
):
    """根据sparse_indices离散选取KV用于golden计算。"""
    s2_sparse = []
    if sparse_mode == 0:
        threshold = seq_lengths_q
    elif sparse_mode == 3:
        delta_s = seq_lengths_kv - seq_lengths_q
        threshold = delta_s + s1_idx + 1
    else:
        threshold = seq_lengths_kv

    valid_count = min(sparse_count, math.ceil(threshold / sparse_block_size))
    for i in range(valid_count):
        sparse_id = sparse_indices[i]
        if sparse_id == -1:
            break
        begin_idx = (
            sparse_id.item() * sparse_block_size
            if isinstance(sparse_id, torch.Tensor)
            else sparse_id * sparse_block_size
        )
        end_idx = min(begin_idx + sparse_block_size, seq_lengths_kv)

        if begin_idx >= threshold:
            continue
        if end_idx <= threshold:
            s2_sparse.extend(range(begin_idx, end_idx))
        else:
            s2_sparse.extend(range(begin_idx, threshold))

    if layout == "BSND":
        k_sparse = k_tensor[batch, s2_sparse, n2_idx, :]
        v_sparse = v_tensor[batch, s2_sparse, n2_idx, :]
    else:
        s2_sparse_offset = [s + kv_start_pos for s in s2_sparse]
        k_sparse = k_tensor[s2_sparse_offset, n2_idx, :]
        v_sparse = v_tensor[s2_sparse_offset, n2_idx, :]

    return k_sparse, v_sparse


def softmax(x):
    """数值稳定的softmax，返回(result, max, sum)。"""
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y / x_sum
    return ans, x_max, x_sum


def npu_sparse_flash_attention_golden(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    sparse_block_size,
    cur_seq_lengths_query,
    cur_seq_lengths_kv,
    query_rope=None,
    key_rope=None,
    layout="BSND",
    sparse_mode=3,
    return_softmax_lse=False,
):
    """CPU侧golden参考实现。"""
    batch_size = sum(1 for x in cur_seq_lengths_query if x >= 0) - 1
    device = query.device

    if layout == "BSND":
        num_heads = query.shape[2]
        num_kv_heads = key.shape[2]
    else:
        num_heads = query.shape[1]
        num_kv_heads = key.shape[1]

    sparse_count = sparse_indices.shape[-1]
    g = num_heads // num_kv_heads

    q_tensor = torch.cat([query, query_rope], dim=-1)
    k_tensor = torch.cat([key, key_rope], dim=-1)
    v_tensor = value
    out_shape = list(q_tensor.shape)
    out_shape[-1] = out_shape[-1] - query_rope.shape[-1]
    y = torch.zeros(out_shape, dtype=torch.float32, device=device)

    if layout == "BSND":
        softmax_shape = [query.shape[0], key.shape[2], query.shape[1], g]
    elif layout == "TND":
        softmax_shape = [key.shape[1], query.shape[0], g]
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    softmax_max = torch.zeros(softmax_shape, dtype=torch.float32, device=device)
    softmax_sum = torch.ones(softmax_shape, dtype=torch.float32, device=device)

    q_start_pos = 0
    kv_start_pos = 0
    for batch in range(batch_size):
        if layout == "BSND":
            seq_lengths_q = query.shape[1]
            seq_lengths_kv = key.shape[1]
            kv_start_pos = 0
        elif layout == "TND":
            seq_lengths_q = int(
                cur_seq_lengths_query[batch + 1] - cur_seq_lengths_query[batch]
            )
            seq_lengths_kv = int(
                cur_seq_lengths_kv[batch + 1] - cur_seq_lengths_kv[batch]
            )
            if seq_lengths_q == 0:
                if seq_lengths_kv == 0:
                    q_start_pos += 1
                    kv_start_pos += 1
                else:
                    q_start_pos += 1
                    kv_start_pos += seq_lengths_kv
                continue

        for n2_idx in range(num_kv_heads):
            for s1_idx in range(seq_lengths_q):
                if layout == "BSND":
                    q_curr = q_tensor[batch, s1_idx, n2_idx * g : (n2_idx + 1) * g, :]
                    cur_sparse_indices = sparse_indices[batch, s1_idx, n2_idx, :]
                else:
                    q_curr = q_tensor[
                        q_start_pos + s1_idx, n2_idx * g : (n2_idx + 1) * g, :
                    ]
                    cur_sparse_indices = sparse_indices[q_start_pos + s1_idx, n2_idx, :]

                k_sparse, v_sparse = gather_kv(
                    k_tensor,
                    v_tensor,
                    cur_sparse_indices,
                    seq_lengths_q,
                    seq_lengths_kv,
                    batch,
                    n2_idx,
                    s1_idx,
                    sparse_count,
                    sparse_block_size,
                    sparse_mode,
                    layout,
                    kv_start_pos,
                )
                if k_sparse.numel() == 0:
                    continue
                mm1_res = torch.matmul(q_curr.float(), k_sparse.float().t())
                scale_res = mm1_res * scale_value
                softmax_res, x_max, x_sum = softmax(scale_res)
                mm2_res = torch.matmul(softmax_res, v_sparse.float())

                if layout == "BSND":
                    y[batch, s1_idx, n2_idx * g : (n2_idx + 1) * g, :] = mm2_res
                    if return_softmax_lse:
                        softmax_max[batch, n2_idx, s1_idx, :] = x_max[:, 0]
                        softmax_sum[batch, n2_idx, s1_idx, :] = x_sum[:, 0]
                else:
                    y[q_start_pos + s1_idx, n2_idx * g : (n2_idx + 1) * g, :] = mm2_res
                    if return_softmax_lse:
                        softmax_max[n2_idx, q_start_pos + s1_idx, :] = x_max[:, 0]
                        softmax_sum[n2_idx, q_start_pos + s1_idx, :] = x_sum[:, 0]

        q_start_pos += seq_lengths_q
        kv_start_pos += seq_lengths_kv

    return y, softmax_max, softmax_sum


def get_sparse_indices_bsnd(
    B,
    s1,
    n2,
    seq_lengths_q,
    seq_lengths_kv,
    sparse_block_count,
    sparse_block_size,
    sparse_mode,
    device,
):
    """生成BSND布局下的sparse_indices。"""
    sparse_indices = (
        torch.zeros((B, s1, n2, sparse_block_count), dtype=torch.int32, device=device)
        - 1
    )
    for bidx in range(B):
        for nidx in range(n2):
            for sidx in range(s1):
                if sparse_mode == 0:
                    threshold = seq_lengths_kv
                elif sparse_mode == 3:
                    threshold = seq_lengths_kv - seq_lengths_q + sidx + 1
                else:
                    threshold = seq_lengths_kv
                valid_blocks_max = math.ceil(max(0, threshold) / sparse_block_size)
                block_indices = torch.randperm(valid_blocks_max, device=device).to(
                    torch.int32
                )
                valid_blocks_topk = min(valid_blocks_max, sparse_block_count)
                sparse_indices[bidx, sidx, nidx, :valid_blocks_topk] = block_indices[
                    :valid_blocks_topk
                ]
    return sparse_indices


def get_sparse_indices_tnd(
    T1,
    n2,
    acc_seqlen_q,
    acc_seqlen_kv,
    sparse_mode,
    sparse_block_size,
    sparse_block_count,
    device,
):
    """生成TND布局下的sparse_indices。"""
    B = np.sum(np.array(acc_seqlen_q) >= 0) - 1
    sparse_indices = (
        torch.zeros((T1, n2, sparse_block_count), dtype=torch.int32, device=device) - 1
    )
    seq_prefixsum_q = 0
    for bidx in range(B):
        seq_lengths_q = acc_seqlen_q[bidx + 1] - acc_seqlen_q[bidx]
        seq_lengths_kv = acc_seqlen_kv[bidx + 1] - acc_seqlen_kv[bidx]
        if seq_lengths_q == 0:
            seq_prefixsum_q += 1
            continue
        for n2idx in range(n2):
            for s in range(seq_lengths_q):
                if sparse_mode == 0:
                    threshold = seq_lengths_kv
                elif sparse_mode == 3:
                    threshold = seq_lengths_kv - seq_lengths_q + s + 1
                else:
                    threshold = seq_lengths_kv
                valid_blocks_max = math.ceil(max(0, threshold) / sparse_block_size)
                block_indices = torch.randperm(valid_blocks_max, device=device).to(
                    torch.int32
                )
                valid_blocks_topk = min(valid_blocks_max, sparse_block_count)
                sparse_indices[seq_prefixsum_q + s, n2idx, :valid_blocks_topk] = (
                    block_indices[:valid_blocks_topk]
                )
        seq_prefixsum_q += seq_lengths_q
    return sparse_indices


def gen_data(params):
    """根据params生成输入数据及CPU侧golden输出。

    params结构：
    (layout, dtype, seqlen_q, seqlen_kv, n1, n2, head_dim, rope_head_dim,
     sparse_block_size, sparse_block_count, block_size, page_attention,
     return_softmax_lse, return_float_output, sparse_mode, case_name)
    """
    (
        layout,
        dtype,
        seqlen_q,
        seqlen_kv,
        n1,
        n2,
        head_dim,
        rope_head_dim,
        sparse_block_size,
        sparse_block_count,
        block_size,
        page_attention,
        return_softmax_lse,
        return_float_output,
        sparse_mode,
        _case_name,
    ) = params

    scale_value = SCALE_VALUE
    T1 = calculate_special_sum(seqlen_q)
    T2 = calculate_special_sum(seqlen_kv)
    acc_seqlen_q = np.cumsum([0] + list(seqlen_q))
    acc_seqlen_kv = np.cumsum([0] + list(seqlen_kv))
    max_seq_len_kv = int(np.max(seqlen_kv))

    if layout == "TND":
        query = torch.tensor(np.random.uniform(-1, 1, (T1, n1, head_dim)), dtype=dtype)
        key = torch.tensor(np.random.uniform(-1, 1, (T2, n2, head_dim)), dtype=dtype)
        value = key.clone()
        query_rope = torch.tensor(
            np.random.uniform(-1, 1, (T1, n1, rope_head_dim)), dtype=dtype
        )
        key_rope = torch.tensor(
            np.random.uniform(-1, 1, (T2, n2, rope_head_dim)), dtype=dtype
        )
        sparse_indices = get_sparse_indices_tnd(
            T1,
            n2,
            acc_seqlen_q,
            acc_seqlen_kv,
            sparse_mode,
            sparse_block_size,
            sparse_block_count,
            "cpu",
        )
    elif layout == "BSND":
        B = len(seqlen_q)
        S1 = max(seqlen_q) if seqlen_q else 1
        S2 = max(seqlen_kv) if seqlen_kv else 1
        query = torch.tensor(
            np.random.uniform(-1, 1, (B, S1, n1, head_dim)), dtype=dtype
        )
        key = torch.tensor(np.random.uniform(-1, 1, (B, S2, n2, head_dim)), dtype=dtype)
        value = key.clone()
        query_rope = torch.tensor(
            np.random.uniform(-1, 1, (B, S1, n1, rope_head_dim)), dtype=dtype
        )
        key_rope = torch.tensor(
            np.random.uniform(-1, 1, (B, S2, n2, rope_head_dim)), dtype=dtype
        )
        sparse_indices = get_sparse_indices_bsnd(
            B,
            S1,
            n2,
            S1,
            S2,
            sparse_block_count,
            sparse_block_size,
            sparse_mode,
            "cpu",
        )
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    acc_seqlen_q_tensor = torch.tensor(acc_seqlen_q, dtype=torch.int64)
    acc_seqlen_kv_tensor = torch.tensor(acc_seqlen_kv, dtype=torch.int64)

    key_pa = key
    key_rope_pa = key_rope
    block_table = None
    if page_attention:
        key_pa, block_table = trans_to_kv_cache(
            key, block_size, acc_seqlen_kv, max_seq_len_kv
        )
        key_rope_pa, _ = trans_to_kv_cache(
            key_rope, block_size, acc_seqlen_kv, max_seq_len_kv
        )

    cpu_output, cpu_softmax_max, cpu_softmax_sum = npu_sparse_flash_attention_golden(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        sparse_block_size,
        cur_seq_lengths_query=acc_seqlen_q,
        cur_seq_lengths_kv=acc_seqlen_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        layout=layout,
        sparse_mode=sparse_mode,
        return_softmax_lse=return_softmax_lse,
    )

    input_data = {
        "params": params,
        "input": {
            "query": query,
            "key": key_pa if page_attention else key,
            "value": key_pa if page_attention else value,
            "sparse_indices": sparse_indices,
            "block_table": block_table,
            "cur_seq_lengths_query": acc_seqlen_q_tensor,
            "cur_seq_lengths_kv": acc_seqlen_kv_tensor,
            "query_rope": query_rope,
            "key_rope": key_rope_pa,
        },
        "attr": {
            "scale_value": scale_value,
            "sparse_block_size": sparse_block_size,
            "layout_query": layout,
            "layout_kv": "PA_BSND" if page_attention else layout,
            "sparse_mode": sparse_mode,
            "return_softmax_lse": return_softmax_lse,
            "return_float_output": return_float_output,
        },
        "cpu_output": cpu_output,
        "cpu_softmax_max": cpu_softmax_max,
        "cpu_softmax_sum": cpu_softmax_sum,
    }
    return input_data


def save_test_case(input_data, output_dir):
    """保存单条测试用例到pt文件。"""
    os.makedirs(output_dir, exist_ok=True)
    params = input_data["params"]
    case_name = f"sfa_layout_{params[0]}_N1_{params[4]}_N2_{params[5]}_{params[15]}"
    input_filename = f"sfa_case_{case_name}.pt"
    input_filepath = os.path.join(output_dir, input_filename)
    torch.save(input_data, input_filepath)
    print(f"测试用例已保存到: {input_filepath}")
    return input_filepath
