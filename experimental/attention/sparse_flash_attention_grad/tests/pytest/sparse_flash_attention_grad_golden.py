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


def get_tnd_idx(acc_q_len, t_idx):
    """Get batch index and local sequence index from TND global index."""
    b_idx = 0
    while t_idx >= acc_q_len[b_idx + 1]:
        b_idx += 1
    s1_idx = t_idx - acc_q_len[b_idx]
    return b_idx, s1_idx


def cal_attenmask(topk_idx, s1_idx, selected_block_size, select_s2, G, cur_s1, cur_s2):
    """Calculate causal attention mask for sparse mode=3."""
    atten_mask = torch.zeros(select_s2)
    delta_s = cur_s2 - cur_s1
    threshold = delta_s + s1_idx + 1

    s_idx = 0
    for sparse_id in topk_idx:
        if sparse_id == -1:
            break
        begin_idx = sparse_id * selected_block_size
        end_idx = min(begin_idx + selected_block_size, cur_s2)
        act_block_size = end_idx - begin_idx
        if end_idx > threshold:
            local_offset = 0 if threshold <= begin_idx else threshold - begin_idx
            mask_begin = s_idx + local_offset
            mask_end = s_idx + act_block_size
            atten_mask[mask_begin:mask_end] = 1
        s_idx += act_block_size
    atten_mask = atten_mask.repeat(G, 1)
    return atten_mask


def softmax_fwd(x):
    """Numerically stable softmax returning (result, max, sum)."""
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    return y / x_sum, x_max, x_sum


def simple_softmax(x, x_max, x_sum):
    """Softmax using precomputed max and sum."""
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    return y / x_sum


def softmax_grad(p, dp, out, out_grad):
    """Compute gradient of softmax."""
    muls = out_grad * out
    muls_res = muls.sum(dim=-1, keepdim=True)
    sub_res = dp - muls_res
    return sub_res * p


def get_sparse_indices_tnd(
    T1,
    n2,
    acc_seqlen_q,
    acc_seqlen_kv,
    cur_q_len,
    cur_kv_len,
    sparse_mode,
    sparse_block_size,
    sparse_block_count,
    device,
):
    """Generate sorted sparse indices tensor for TND layout."""
    B = len(cur_q_len)
    sparse_indices = (
        torch.ones((T1, n2, sparse_block_count), dtype=torch.int32, device=device) * -1
    )

    for t in range(T1):
        b_idx, s1_idx = get_tnd_idx(acc_seqlen_q, t)
        cur_s1 = cur_q_len[b_idx]
        cur_s2 = cur_kv_len[b_idx]

        for j in range(n2):
            if sparse_mode == 3:
                delta_s = cur_s2 - cur_s1
                threshold = delta_s + s1_idx + 1 if delta_s + s1_idx + 1 >= 0 else 0
            else:
                threshold = cur_s2
            valid_block_count = math.ceil(threshold / sparse_block_size)
            if valid_block_count >= sparse_block_count:
                sparse_indices[t][j][0] = valid_block_count - 1
                sparse_indices[t][j][1:sparse_block_count] = torch.randperm(
                    valid_block_count - 1, device=device
                )[: sparse_block_count - 1].to(torch.int32)
            else:
                sparse_indices[t][j][:valid_block_count] = torch.randperm(
                    valid_block_count, device=device
                ).to(torch.int32)

    # Sort indices (replace -1 with max int for sorting, restore after)
    neg_mask = sparse_indices == -1
    sparse_indices[neg_mask] = 2**31 - 1
    sorted_indices, sort_order = torch.sort(sparse_indices, dim=-1)
    original_neg_in_sorted = neg_mask.gather(dim=-1, index=sort_order)
    sorted_indices[original_neg_in_sorted] = -1

    return sorted_indices


def gather_kv(cur_k, cur_v, topk_idx, block_size, cur_s2):
    """Gather sparse KV entries based on topk indices."""
    s2_sparse = []
    for sparse_id in topk_idx:
        if sparse_id == -1:
            break
        begin_idx = (
            sparse_id.item() * block_size
            if isinstance(sparse_id, torch.Tensor)
            else sparse_id * block_size
        )
        end_idx = min(begin_idx + block_size, cur_s2)
        s2_sparse.extend(range(begin_idx, end_idx))
    if not s2_sparse:
        return None, None, 0
    k_cal = cur_k[s2_sparse, :]
    v_cal = cur_v[s2_sparse, :]
    return k_cal, v_cal, len(s2_sparse)


def scatter_kv(dkv_out, dkv, topk_idx, block_size, cur_s2):
    """Scatter gradient back to full KV positions."""
    dkv_start = 0
    for sparse_id in topk_idx:
        if sparse_id == -1:
            break
        begin_idx = (
            sparse_id.item() * block_size
            if isinstance(sparse_id, torch.Tensor)
            else sparse_id * block_size
        )
        end_idx = min(begin_idx + block_size, cur_s2)
        dkv_end = dkv_start + (end_idx - begin_idx)
        dkv_out[begin_idx:end_idx, :] += dkv[dkv_start:dkv_end, :]
        dkv_start = dkv_end
    return dkv_out


def npu_sparse_flash_attention_grad_golden(
    query,
    key,
    value,
    sparse_indices,
    out,
    dout,
    scale_value,
    sparse_block_size,
    acc_seqlen_q,
    acc_seqlen_kv,
    cur_q_len,
    cur_kv_len,
    query_rope=None,
    key_rope=None,
    sparse_mode=3,
):
    """CPU golden reference for sparse flash attention backward."""
    device = query.device
    T1, N1, D_v = query.shape[0], query.shape[1], value.shape[-1]
    T2, N2 = value.shape[0], value.shape[1]
    G = N1 // N2
    sparse_block_count = sparse_indices.shape[-1]

    # Float computation
    q = query.float().reshape(T1, N2, G, query.shape[-1])
    k = key.float()
    v = value.float()
    o = out.float().reshape(T1, N2, G, D_v)
    do = dout.float().reshape(T1, N2, G, D_v)

    D_qk = q.shape[-1]
    if query_rope is not None and key_rope is not None:
        qr = query_rope.float().reshape(T1, N2, G, query_rope.shape[-1])
        kr = key_rope.float()
        q = torch.cat([q, qr], dim=-1)
        k = torch.cat([k, kr], dim=-1)
        D_qk = q.shape[-1]

    # Compute softmax_max and softmax_sum via forward pass
    softmax_max = torch.zeros(N2, T1, G, device=device, dtype=torch.float32)
    softmax_sum = torch.ones(N2, T1, G, device=device, dtype=torch.float32)

    for i in range(T1):
        b_idx, s1_idx = get_tnd_idx(acc_seqlen_q, i)
        for n2_idx in range(N2):
            topk_idx = sparse_indices[i][n2_idx]
            q_cal = q[i][n2_idx]
            s2_start = acc_seqlen_kv[b_idx]
            s2_end = acc_seqlen_kv[b_idx + 1]
            cur_s2 = cur_kv_len[b_idx]
            cur_s1 = cur_q_len[b_idx]
            cur_k = k[s2_start:s2_end, n2_idx, :]
            cur_v = v[s2_start:s2_end, n2_idx, :]

            k_cal, v_cal, actual_sel_s2 = gather_kv(
                cur_k, cur_v, topk_idx, sparse_block_size, cur_s2
            )
            if actual_sel_s2 > 0:
                qk = torch.matmul(q_cal, k_cal.permute(1, 0)).mul(scale_value)
                if sparse_mode == 3:
                    atten_mask = cal_attenmask(
                        topk_idx,
                        s1_idx,
                        sparse_block_size,
                        actual_sel_s2,
                        G,
                        cur_s1,
                        cur_s2,
                    )
                    qk = qk + atten_mask.to(device) * (-2e35)
                _, x_max, x_sum = softmax_fwd(qk)
                softmax_max[n2_idx][i] = x_max.squeeze(dim=-1)
                softmax_sum[n2_idx][i] = x_sum.squeeze(dim=-1)

    # Backward pass
    dq_out = torch.zeros(T1, N2, G, D_qk, device=device, dtype=torch.float32)
    dk_out = torch.zeros(T2, N2, D_qk, device=device, dtype=torch.float32)
    dv_out = torch.zeros(T2, N2, D_v, device=device, dtype=torch.float32)

    for i in range(T1):
        b_idx, s1_idx = get_tnd_idx(acc_seqlen_q, i)
        for n2_idx in range(N2):
            topk_idx = sparse_indices[i][n2_idx]
            q_cal = q[i][n2_idx]
            out_cal = o[i][n2_idx]
            dout_cal = do[i][n2_idx]

            s2_start = acc_seqlen_kv[b_idx]
            s2_end = acc_seqlen_kv[b_idx + 1]
            cur_s2 = cur_kv_len[b_idx]
            cur_s1 = cur_q_len[b_idx]
            cur_k = k[s2_start:s2_end, n2_idx, :]
            cur_v = v[s2_start:s2_end, n2_idx, :]
            k_cal, v_cal, actual_sel_s2 = gather_kv(
                cur_k, cur_v, topk_idx, sparse_block_size, cur_s2
            )

            if actual_sel_s2 > 0:
                qk = torch.matmul(q_cal, k_cal.permute(1, 0)).mul(scale_value)
                if sparse_mode == 3:
                    atten_mask = cal_attenmask(
                        topk_idx,
                        s1_idx,
                        sparse_block_size,
                        actual_sel_s2,
                        G,
                        cur_s1,
                        cur_s2,
                    )
                    qk = qk + atten_mask.to(device) * (-2e35)

                x_max = softmax_max[n2_idx][i]
                x_sum = softmax_sum[n2_idx][i]
                softmax_res = simple_softmax(
                    qk, x_max.unsqueeze(-1), x_sum.unsqueeze(-1)
                )
                dp = torch.matmul(dout_cal, v_cal.permute(1, 0))

                sg = softmax_grad(softmax_res, dp, out_cal, dout_cal).to(torch.float32)

                dq = torch.matmul(sg, k_cal)
                dk = torch.matmul(sg.permute(1, 0), q_cal)
                dv = torch.matmul(
                    softmax_res.permute(1, 0).to(torch.bfloat16).to(torch.float32),
                    dout_cal,
                )

                dq_out[i][n2_idx] = dq
                dk_out[s2_start:s2_end, n2_idx, :] = scatter_kv(
                    dk_out[s2_start:s2_end, n2_idx, :],
                    dk,
                    topk_idx,
                    sparse_block_size,
                    cur_s2,
                )
                dv_out[s2_start:s2_end, n2_idx, :] = scatter_kv(
                    dv_out[s2_start:s2_end, n2_idx, :],
                    dv,
                    topk_idx,
                    sparse_block_size,
                    cur_s2,
                )

    dq_out = dq_out * scale_value
    dk_out = dk_out * scale_value
    dq_out = dq_out.reshape(T1, N1, D_qk)

    org_Dqk = D_v
    if query_rope is not None and key_rope is not None:
        dq_rope = dq_out[:, :, org_Dqk:]
        dk_rope = dk_out[:, :, org_Dqk:]
        dq = dq_out[:, :, :org_Dqk]
        dk = dk_out[:, :, :org_Dqk]
    else:
        dq = dq_out
        dk = dk_out
        dq_rope = None
        dk_rope = None

    return softmax_max, softmax_sum, dq, dk, dv_out, dq_rope, dk_rope


def gen_data(params):
    """Generate input data and CPU golden for the backward operator.

    params structure:
    (layout, dtype, seqlen_q, seqlen_kv, n1, n2, head_dim, rope_head_dim,
     sparse_block_size, sparse_block_count, sparse_mode, deterministic, case_name)
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
        sparse_mode,
        deterministic,
        _case_name,
    ) = params

    device = torch.device("npu:0")
    scale_value = SCALE_VALUE
    T1 = int(np.sum(seqlen_q))
    T2 = int(np.sum(seqlen_kv))
    acc_seqlen_q = np.cumsum([0] + list(seqlen_q))
    acc_seqlen_kv = np.cumsum([0] + list(seqlen_kv))
    cur_q_len = list(seqlen_q)
    cur_kv_len = list(seqlen_kv)

    if layout == "TND":
        query = torch.randn((T1, n1, head_dim), dtype=dtype)
        key = torch.randn((T2, n2, head_dim), dtype=dtype)
        value = torch.randn((T2, n2, head_dim), dtype=dtype)
        if rope_head_dim > 0:
            query_rope = torch.randn((T1, n1, rope_head_dim), dtype=dtype)
            key_rope = torch.randn((T2, n2, rope_head_dim), dtype=dtype)
        else:
            query_rope = None
            key_rope = None
    elif layout == "BSND":
        B = len(seqlen_q)
        query = torch.randn((B, max(seqlen_q), n1, head_dim), dtype=dtype)
        key = torch.randn((B, max(seqlen_kv), n2, head_dim), dtype=dtype)
        value = torch.randn((B, max(seqlen_kv), n2, head_dim), dtype=dtype)
        if rope_head_dim > 0:
            query_rope = torch.randn((B, max(seqlen_q), n1, rope_head_dim), dtype=dtype)
            key_rope = torch.randn((B, max(seqlen_kv), n2, rope_head_dim), dtype=dtype)
        else:
            query_rope = None
            key_rope = None
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    sparse_indices = get_sparse_indices_tnd(
        T1,
        n2,
        acc_seqlen_q,
        acc_seqlen_kv,
        cur_q_len,
        cur_kv_len,
        sparse_mode,
        sparse_block_size,
        sparse_block_count,
        "cpu",
    ).to(dtype=torch.int32)

    # Simulate forward output and upstream gradient
    out = torch.randn_like(query)
    dout = torch.randn_like(query)

    softmax_max, softmax_sum, dq, dk, dv, dq_rope, dk_rope = (
        npu_sparse_flash_attention_grad_golden(
            query,
            key,
            value,
            sparse_indices,
            out,
            dout,
            scale_value,
            sparse_block_size,
            acc_seqlen_q,
            acc_seqlen_kv,
            cur_q_len,
            cur_kv_len,
            query_rope,
            key_rope,
            sparse_mode,
        )
    )

    # Move tensors to NPU device
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)
    out = out.to(device)
    dout = dout.to(device)
    if query_rope is not None:
        query_rope = query_rope.to(device)
    if key_rope is not None:
        key_rope = key_rope.to(device)
    sparse_indices = sparse_indices.to(device)
    softmax_max = softmax_max.to(device)
    softmax_sum = softmax_sum.to(device)
    acc_seqlen_q_tensor = torch.tensor(acc_seqlen_q, dtype=torch.int64, device=device)
    acc_seqlen_kv_tensor = torch.tensor(acc_seqlen_kv, dtype=torch.int64, device=device)

    input_data = {
        "query": query,
        "key": key,
        "value": value,
        "out": out,
        "dout": dout,
        "query_rope": query_rope,
        "key_rope": key_rope,
        "sparse_indices": sparse_indices,
        "softmax_max": softmax_max,
        "softmax_sum": softmax_sum,
        "cur_seq_lengths_query": acc_seqlen_q_tensor,
        "cur_seq_lengths_kv": acc_seqlen_kv_tensor,
        "scale_value": scale_value,
        "sparse_block_size": sparse_block_size,
        "layout": layout,
        "sparse_mode": sparse_mode,
        "deterministic": deterministic,
        "params": params,
        "cpu_dq": dq,
        "cpu_dk": dk,
        "cpu_dv": dv,
        "cpu_dq_rope": dq_rope,
        "cpu_dk_rope": dk_rope,
    }
    return input_data


def save_test_case(input_data, output_dir):
    """Save single test case as pt file."""
    os.makedirs(output_dir, exist_ok=True)
    params = input_data["params"]
    case_name = f"sfag_layout_{params[0]}_N1_{params[4]}_N2_{params[5]}_{params[12]}"
    input_filename = f"sfag_case_{case_name}.pt"
    input_filepath = os.path.join(output_dir, input_filename)
    torch.save(input_data, input_filepath)
    print(f"测试数据已保存到: {input_filepath}")
    return input_filepath
