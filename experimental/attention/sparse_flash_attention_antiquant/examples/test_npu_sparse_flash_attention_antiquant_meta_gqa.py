# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

import random
import torch
import torch_npu
import torchair
import math
import custom_ops
import numpy as np
import torch.nn as nn
from torch_npu.testing.testcase import TestCase, run_tests

# import torch._dynamo
# torch._dynamo.reset()

np.random.seed(234)  # 固定随机种子
np.set_printoptions(suppress=True)

DEVICE_ID = 0
# DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))


def pa_to_bsnd(pa_in, block_table, actual_seq_lengths, layout_kv):
    if layout_kv == "PA_BSND":
        block_num, block_size, n, d = pa_in.shape
    elif layout_kv == "PA_BNSD":
        block_num, n, block_size, d = pa_in.shape
    elif layout_kv == "PA_NZ":
        block_num, n, dn, block_size, ds = pa_in.shape
        d = dn * ds

    b, max_block_num = block_table.shape
    out = torch.zeros(
        (b, max(max_block_num, math.ceil(block_num / b)) * block_size, n, d)
    ).to(pa_in.dtype)
    if layout_kv == "PA_BSND":
        for i in range(b):
            loop = actual_seq_lengths[i] // block_size
            for j in range(loop):
                out[i, j * block_size : (j + 1) * block_size, :, :] = pa_in[
                    block_table[i][j], :, :, :
                ].reshape(block_size, n, d)
            tail_len = actual_seq_lengths[i] % block_size
            if tail_len > 0:
                out[i, loop * block_size : actual_seq_lengths[i], :, :] = pa_in[
                    block_table[i][loop], :tail_len, :, :
                ].reshape(tail_len, n, d)
    elif layout_kv == "PA_BNSD":
        for i in range(b):
            loop = actual_seq_lengths[i] // block_size
            for j in range(loop):
                out[i, j * block_size : (j + 1) * block_size, :, :] = (
                    pa_in[block_table[i][j], :, :, :]
                    .reshape(n, block_size, d)
                    .permute(1, 0, 2)
                )
            tail_len = actual_seq_lengths[i] % block_size
            if tail_len > 0:
                out[i, loop * block_size : actual_seq_lengths[i], :, :] = (
                    pa_in[block_table[i][loop], :, :tail_len, :]
                    .reshape(n, tail_len, d)
                    .permute(1, 0, 2)
                )
    elif layout_kv == "PA_NZ":
        for i in range(b):
            loop = actual_seq_lengths[i] // block_size
            for j in range(loop):
                out[i, j * block_size : (j + 1) * block_size, :, :] = (
                    pa_in[block_table[i][j], :, :, :, :]
                    .reshape(n, dn, block_size, ds)
                    .permute(2, 0, 1, 3)
                    .reshape(block_size, n, d)
                )
            tail_len = actual_seq_lengths[i] % block_size
            if tail_len > 0:
                out[i, loop * block_size : actual_seq_lengths[i], :, :] = (
                    pa_in[block_table[i][loop], :, :, :tail_len, :]
                    .reshape(n, dn, tail_len, ds)
                    .permute(2, 0, 1, 3)
                    .reshape(tail_len, n, d)
                )
    return out


def gather_kv(
    k_tensor,
    v_tensor,
    sparse_indices,
    sparse_block_size,
    sparse_count,
    batch,
    n2_idx,
    cur_actual_seq_lengths_q,
    cur_actual_seq_lengths_kv,
    sparse_mode,
    s1_shard_idx,
    cur_sparse_seq_lengths_kv,
):
    s2_sparse = list()
    if sparse_count > sparse_indices.numel():
        raise f"sparse_count({sparse_count}) should less than the length sparse_indices({sparse_indices.numel()})"
    if sparse_mode == 0:
        threshold = cur_actual_seq_lengths_kv
    elif sparse_mode == 3:
        threshold = (
            cur_actual_seq_lengths_kv - cur_actual_seq_lengths_q + s1_shard_idx + 1
        )
    sparse_kv_lengths_sum = 0
    for i in range(sparse_count):
        sparse_id = sparse_indices[i]
        if sparse_id == -1:
            break
        begin_idx = sparse_id * sparse_block_size
        end_idx = (
            begin_idx + sparse_block_size
            if begin_idx + sparse_block_size <= threshold
            else threshold
        )
        sparse_kv_lengths_sum = sparse_kv_lengths_sum + (end_idx - begin_idx)
        if i == sparse_count - 1:
            end_idx = (
                end_idx - (sparse_kv_lengths_sum - cur_sparse_seq_lengths_kv)
                if sparse_kv_lengths_sum > cur_sparse_seq_lengths_kv
                else end_idx
            )
        if begin_idx >= threshold:
            continue
        s2_sparse.extend(torch.arange(begin_idx, end_idx))

    k_sparse, v_sparse = (
        k_tensor[batch, n2_idx, s2_sparse, :],
        v_tensor[batch, n2_idx, s2_sparse, :],
    )

    return k_sparse, v_sparse, torch.tensor(s2_sparse)


def softmax(x):
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    return ans


class SFAANetwork(nn.Module):
    def __init__(self):
        super(SFAANetwork, self).__init__()
        self.dummy_version = 2  # 加个无意义的成员变量，改变代码指纹

    def forward(
        self,
        b,
        s1,
        n1,
        s2,
        n2,
        dn,
        query,
        key,
        value,
        sparse_indices,
        key_dequant_scale,
        value_dequant_scale,
        scale_value,
        sparse_block_size,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        sparse_seq_lengths_kv,
        layout_query,
        layout_kv,
        sparse_mode,
        block_table,
        attention_mode,
        quant_scale_repo_mode,
        tile_size,
        key_quant_mode,
        value_quant_mode,
        rope_head_dim,
        sparse_shard_size,
    ):
        # super kernel test
        # with torchair.scope.super_kernel("sp_QsSFAA", "stream-fusion=1:dcci-before-kernel-start=SparseFlashAttentionAntiquant"):
        metadata = torch.ops.custom.npu_sparse_flash_attention_antiquant_metadata(
            b,
            s1,
            n1,
            s2,
            n2,
            dn,
            0,
            sparse_block_size,
            sparse_shared_size=sparse_shard_size,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            sparse_seq_lengths_kv=sparse_seq_lengths_kv,
        )
        # y = torch.broadcast_to(torch.tensor(1, device='npu:14', dtype=torch.float32), (192*1024*1024,))
        # value_dq_scale_new = y.flatten()[:value_dequant_scale.numel()].reshape(value_dequant_scale.shape).to(torch.float32)
        output0 = torch_npu.npu_sparse_flash_attention_antiquant(
            query,
            key,
            value,
            sparse_indices,
            key_dequant_scale=key_dequant_scale,
            value_dequant_scale=value_dequant_scale,
            scale_value=scale_value,
            sparse_block_size=16,
            metadata=metadata,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            sparse_seq_lengths_kv=sparse_seq_lengths_kv,
            layout_query=layout_query,
            layout_kv=layout_kv,
            sparse_mode=sparse_mode,
            block_table=block_table,
            attention_mode=attention_mode,
            quant_scale_repo_mode=quant_scale_repo_mode,
            tile_size=tile_size,
            key_quant_mode=key_quant_mode,
            value_quant_mode=value_quant_mode,
            rope_head_dim=64,
            sparse_shard_size=sparse_shard_size,
        )
        return output0


def cpu_sparse_flash_attention_antiquant_gqa(
    query,
    key,
    value,
    sparse_indices,
    key_dequant_scale,
    value_dequant_scale,
    scale_value,
    sparse_block_size,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    sparse_seq_lengths_kv,
    layout_query="BSND",
    layout_kv="PA_BSND",
    sparse_mode=3,
    block_table=None,
    attention_mode=0,
    quant_scale_repo_mode=0,
    tile_size=0,
    key_quant_mode=0,
    value_quant_mode=0,
    rope_head_dim=0,
    sparse_shard_size=1,
):
    """
    CPU计算的sparse flash attention函数
    """
    query = query.to(torch.float32)
    query_type = query.dtype
    head_dim = query.shape[-1]

    # 将PA格式转换为BSND格式
    key = pa_to_bsnd(key, block_table, actual_seq_lengths_kv, layout_kv)
    value = pa_to_bsnd(value, block_table, actual_seq_lengths_kv, layout_kv)

    # 反量化
    key = (key.to(torch.float32) * key_dequant_scale).to(query_type)
    value = (value.to(torch.float32) * value_dequant_scale).to(query_type)

    batch_size = actual_seq_lengths_query.shape[0]
    if layout_query == "TND":
        num_heads = query.shape[1]
    else:
        num_heads = query.shape[2]
    num_kv_heads = key.shape[2]
    g = num_heads // num_kv_heads

    # 转置为B, N, S, D格式
    q_bnsd_tensor = torch.transpose(query, 1, 2)
    k_bnsd_tensor = torch.transpose(key, 1, 2)
    v_bnsd_tensor = torch.transpose(value, 1, 2)
    sparse_indices_tensor = torch.transpose(sparse_indices, 1, 2)
    out_shape_bnsd = list(q_bnsd_tensor.shape)
    y = torch.zeros(out_shape_bnsd, dtype=query_type)

    # 遍历每个batch
    for batch in range(batch_size):
        cur_acutal_seq_lengths_q = actual_seq_lengths_query[batch]
        if layout_query == "TND" and batch > 0:
            cur_acutal_seq_lengths_q = (
                actual_seq_lengths_query[batch] - actual_seq_lengths_query[batch - 1]
            )
        cur_actual_seq_lengths_kv = actual_seq_lengths_kv[batch]
        cur_sparse_seq_lengths_kv = sparse_seq_lengths_kv[batch]
        sparse_count = math.ceil(sparse_seq_lengths_kv[batch] / sparse_block_size)

        # 遍历每个KV头
        for n2_idx in range(num_kv_heads):
            # 遍历每个query token
            for s1_shard_idx in range(cur_acutal_seq_lengths_q):
                # 获取当前query
                q_curr = q_bnsd_tensor[
                    batch, n2_idx * g : (n2_idx + 1) * g, s1_shard_idx, :
                ].squeeze(1)

                # 获取当前稀疏索引
                cur_sparse_indices = sparse_indices_tensor[
                    batch, n2_idx, s1_shard_idx // sparse_shard_size, :
                ]

                # 根据稀疏索引收集key和value
                k_sparse, v_sparse, s2_index = gather_kv(
                    k_bnsd_tensor,
                    v_bnsd_tensor,
                    cur_sparse_indices,
                    sparse_block_size,
                    sparse_count,
                    batch,
                    n2_idx,
                    cur_acutal_seq_lengths_q,
                    cur_actual_seq_lengths_kv,
                    sparse_mode,
                    s1_shard_idx,
                    cur_sparse_seq_lengths_kv,
                )

                # 计算attention
                if k_sparse.shape[0] == 0:  # 没有有效的key/value
                    # 返回零向量
                    mm2_res = torch.zeros((g, head_dim), dtype=query_type)
                else:
                    mm1_res = torch.matmul(
                        q_curr.to(torch.float32), k_sparse.to(torch.float32).T
                    )
                    scale_res = mm1_res * scale_value
                    if scale_res.numel() != 0:
                        softmax_res = softmax(scale_res.numpy())
                    else:
                        softmax_res = torch.zeros_like(scale_res)
                    softmax_res = torch.tensor(softmax_res).to(query_type)
                    mm2_res = torch.matmul(
                        softmax_res.to(torch.float32), v_sparse.to(torch.float32)
                    )

                # 处理输出维度
                if mm2_res.dim() == 1:
                    mm2_res = mm2_res.unsqueeze(0)
                mm2_res = mm2_res.reshape(g, 1, head_dim)  # 从(g, d)变为(g, 1, d)
                y[
                    batch,
                    n2_idx * g : (n2_idx + 1) * g,
                    s1_shard_idx : s1_shard_idx + 1,
                    :,
                ] = mm2_res

    # 转置回B, S, N, D格式
    return torch.transpose(y, 1, 2)


def cpu_sparse_flash_attention_antiquant_gqa_s2split(
    query,
    key,
    value,
    sparse_indices,
    key_dequant_scale,
    value_dequant_scale,
    scale_value,
    sparse_block_size,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    sparse_seq_lengths_kv,
    layout_query="BSND",
    layout_kv="PA_BSND",
    sparse_mode=3,
    block_table=None,
    attention_mode=0,
    quant_scale_repo_mode=0,
    tile_size=0,
    key_quant_mode=0,
    value_quant_mode=0,
    rope_head_dim=0,
    sparse_shard_size=1,
):
    query_type = query.dtype
    head_dim = query.shape[-1]

    key = pa_to_bsnd(key, block_table, actual_seq_lengths_kv).to(torch.float32)
    key = (key * key_dequant_scale).to(query_type)

    value = pa_to_bsnd(value, block_table, actual_seq_lengths_kv).to(torch.float32)
    value = (value * value_dequant_scale).to(query_type)

    batch_size = actual_seq_lengths_query.shape[0]
    if layout_query == "TND":
        num_heads = query.shape[1]
    else:
        num_heads = query.shape[2]
    num_kv_heads = key.shape[2]
    g = num_heads // num_kv_heads

    q_bnsd_tensor = torch.transpose(query, 1, 2)
    k_bnsd_tensor = torch.transpose(key, 1, 2)
    v_bnsd_tensor = torch.transpose(value, 1, 2)
    sparse_indices_tensor = torch.transpose(sparse_indices, 1, 2)
    out_shape_bnsd = list(q_bnsd_tensor.shape)
    y = torch.zeros(out_shape_bnsd, dtype=query_type)

    for batch in range(batch_size):
        cur_acutal_seq_lengths_q = actual_seq_lengths_query[batch]
        if layout_query == "TND" and batch > 0:
            cur_acutal_seq_lengths_q = (
                actual_seq_lengths_query[batch] - actual_seq_lengths_query[batch - 1]
            )
        cur_actual_seq_lengths_kv = actual_seq_lengths_kv[batch]
        sparse_count = sparse_seq_lengths_kv[batch]
        for n2_idx in range(num_kv_heads):
            s1_shard_loops = (
                cur_acutal_seq_lengths_q + sparse_shard_size - 1
            ) // sparse_shard_size
            s1_shard_tail = (
                cur_acutal_seq_lengths_q - (s1_shard_loops - 1) * sparse_shard_size
            )
            for s1_shard_idx in range(s1_shard_loops):
                s1_shard_size = (
                    s1_shard_tail
                    if (s1_shard_idx == s1_shard_loops - 1)
                    else sparse_shard_size
                )
                q_curr = q_bnsd_tensor[
                    batch,
                    n2_idx * g : (n2_idx + 1) * g,
                    s1_shard_idx * sparse_shard_size : s1_shard_idx * sparse_shard_size
                    + s1_shard_size,
                    :,
                ]
                cur_sparse_indices = sparse_indices_tensor[
                    batch, n2_idx, s1_shard_idx, :
                ]
                k_sparse, v_sparse, s2_index = gather_kv(
                    k_bnsd_tensor,
                    v_bnsd_tensor,
                    cur_sparse_indices,
                    sparse_block_size,
                    sparse_count,
                    batch,
                    n2_idx,
                    cur_acutal_seq_lengths_q,
                    cur_actual_seq_lengths_kv,
                    sparse_mode,
                    s1_shard_idx,
                )
                s2_base = 512
                s2 = k_sparse.shape[0]
                loop = (s2 + s2_base - 1) // s2_base
                tail = s2 - (loop - 1) * s2_base
                rowSum = torch.from_numpy(
                    np.zeros(shape=(s1_shard_size * g, 1), dtype=np.float32)
                )
                rowMax = torch.from_numpy(
                    np.full(
                        shape=(s1_shard_size * g, 1),
                        fill_value=-np.inf,
                        dtype=np.float32,
                    )
                )
                mm2_res = torch.from_numpy(
                    np.zeros(
                        shape=(s1_shard_size * g, out_shape_bnsd[-1]), dtype=np.float32
                    )
                )
                threshold_base = (
                    cur_actual_seq_lengths_kv
                    - cur_acutal_seq_lengths_q
                    + s1_shard_idx * sparse_shard_size
                    + 1
                )
                for i in range(loop):
                    if i < loop - 1:
                        seq_start = i * s2_base
                        seq_end = (i + 1) * s2_base
                    else:
                        seq_start = i * s2_base
                        seq_end = i * s2_base + tail
                    k_cur = k_sparse[seq_start:seq_end, :]
                    v_cur = v_sparse[seq_start:seq_end, :]
                    s2_index_cur = s2_index[seq_start:seq_end]
                    mm1_res = torch.matmul(
                        q_curr.reshape(g * s1_shard_size, head_dim).to(torch.float32),
                        k_cur.to(torch.float32).T,
                    )
                    scale_res = mm1_res * scale_value
                    if sparse_mode == 3:
                        for s1_idx in range(s1_shard_size):
                            mask_index = s2_index_cur >= threshold_base + s1_idx
                            scale_res[s1_idx * g : (s1_idx + 1) * g, mask_index] = -1e12
                    max_local, _ = scale_res.max(axis=-1, keepdims=True)
                    replace_idx = torch.where(rowMax < max_local)
                    rowMax_old = rowMax.clone()
                    rowMax[replace_idx] = max_local[replace_idx]
                    update_mul = torch.exp(rowMax_old - rowMax)

                    softmax_res = torch.exp(scale_res - rowMax)
                    sum_local = softmax_res.sum(axis=-1, keepdims=True)
                    rowSum = update_mul * rowSum + sum_local

                    mm2_res_local = torch.matmul(
                        softmax_res.to(torch.float32), v_cur.to(torch.float32)
                    )
                    mm2_res = update_mul * mm2_res + mm2_res_local
                res = mm2_res / rowSum
                res = res.reshape(g, s1_shard_size, head_dim)
                y[
                    batch,
                    n2_idx * g : (n2_idx + 1) * g,
                    s1_shard_idx * sparse_shard_size : s1_shard_idx * sparse_shard_size
                    + s1_shard_size,
                    :,
                ] = res.to(query_type)
    return torch.transpose(y, 1, 2)


def calculate_new_sparse_seq_kv(act_seq_kv, sparse_ratio, sparse_block_size):
    """
    计算新的sparse_seq_kv值
    1. 先计算 act_seq_kv * sparse_ratio
    2. 向上取整
    3. 调整到满足条件：sparse_seq_kv % 16 == act_seq_kv % 16
    4. 确保 sparse_seq_kv <= act_seq_kv
    """
    new_sparse_seq = []

    for act_val in act_seq_kv:
        # 基本计算：act_seq_kv * sparse_ratio 向上取整
        base_val = math.ceil(act_val * sparse_ratio)

        # 计算余数
        act_remainder = act_val % sparse_block_size

        # 调整base_val，使其模16的余数与act_val模16的余数相等
        current_remainder = base_val % sparse_block_size

        if current_remainder != act_remainder:
            # 计算需要调整的值
            diff = (act_remainder - current_remainder) % sparse_block_size
            base_val += diff

        # 确保sparse_seq_kv <= act_seq_kv
        # 如果base_val大于act_val，尝试减小base_val
        while base_val > act_val and base_val >= sparse_block_size:
            base_val -= sparse_block_size

        # 再次检查余数条件，确保调整后仍然满足
        if base_val % sparse_block_size != act_remainder:
            # 如果调整后不满足余数条件，尝试向下调整
            # 找到不大于act_val且满足余数条件的最大值
            candidate = act_val
            while candidate > 0 and candidate % sparse_block_size != act_remainder:
                candidate -= 1

            # 确保candidate >= base_val的最小值（向上取整后的值）
            min_val = math.ceil(act_val * sparse_ratio)
            if candidate >= min_val:
                base_val = candidate
            else:
                # 如果找不到满足条件的值，使用原始计算值
                # 但确保不大于act_val
                base_val = min(min_val, act_val)

        # 最终确保base_val不大于act_val
        base_val = min(base_val, act_val)

        # 确保base_val非负
        base_val = max(base_val, 0)

        new_sparse_seq.append(base_val)

    return new_sparse_seq


def compute_sparse_seq_len(qk_len_tensor, is_context, sparsity=4):
    sparse_block_size = 16
    max_select_count = 128 * 1024
    fixed_tail_count = 32

    if not is_context:
        act_tail_seq = qk_len_tensor % sparse_block_size
        fixed_seq = torch.where(
            act_tail_seq == 0,
            torch.tensor(
                fixed_tail_count * sparse_block_size, device=qk_len_tensor.device
            ),
            (fixed_tail_count - 1) * sparse_block_size + act_tail_seq,
        )
        # print("fixed_seq = ", fixed_seq)
        select_seq = torch.clamp(qk_len_tensor - fixed_seq, min=0)
        # print("select_seq = ", select_seq)
        select_N_count = torch.ceil(
            (select_seq / sparse_block_size) * round(1 / sparsity, 2)
        )
        select_N_count = torch.min(
            select_N_count, torch.tensor(max_select_count, device=qk_len_tensor.device)
        )
        # print("select_N_count = ", select_N_count)
        sparse_seq_len = select_N_count * sparse_block_size + torch.min(
            qk_len_tensor, fixed_seq
        )
        # print("sparse_seq_len = ", sparse_seq_len)
        sparse_seq_len = sparse_seq_len.to(torch.int32)
    else:
        sparse_seq_len = None
    return sparse_seq_len


class TestCustomSFA(TestCase):
    def test_sfa_eager(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        query_type = torch.bfloat16
        scale_value = 0.041666666666666664
        sparse_block_size = 16

        # 典型shape性能/功能用例，其余泛化用例见文件末尾，可能需要修改一些参数
        sparse_ratio = 0.25
        b = 21 * 4
        s1 = 4
        n1 = 20
        n2 = 2
        dn = 128
        tile_size = 128
        block_size = 512
        import random

        random_seed = 42
        unbalance_cache = 100
        random.seed(random_seed)
        s2_list = []
        sumseqlength = 0
        for _ in range(b - 1):
            offset_percent = random.randint(-unbalance_cache, unbalance_cache)
            offset = int(s2 * (offset_percent / 100))
            s2curent = s2 + offset
            sumseqlength = sumseqlength + s2curent
            s2_list.append(s2curent)
        totalseq = b * s2
        s2_last = max(totalseq - sumseqlength, 0)
        s2_list.append(s2_last)

        act_seq_q_list = [s1] * b
        act_seq_q = torch.tensor(act_seq_q_list).to(torch.int32)
        act_seq_kv = torch.tensor(s2_list).to(torch.int32)
        print(act_seq_q)
        print("SFA test case, act_seq_kv:", act_seq_kv)
        # sparse_seq_kv = torch.tensor(calculate_new_sparse_seq_kv(act_seq_kv, sparse_ratio, sparse_block_size)).to(torch.int32)
        sparse_seq_kv = compute_sparse_seq_len(act_seq_kv, is_context=False, sparsity=4)
        print("SFA test case, sparse_seq_kv:", sparse_seq_kv)
        s2 = max(s2_list)
        print("SFA test case, s2:", s2)

        layout_query = "BSND"
        layout_kv = "PA_NZ"
        sparse_shard_size = 4
        key_quant_mode = 0
        value_quant_mode = 0
        attention_mode = 0
        quant_scale_repo_mode = 0
        sparse_mode = 3
        maxsparse_block_count = 2072
        sparse_block_count = torch.ceil(((sparse_seq_kv) / (sparse_block_size)))
        max_block_num = math.ceil(s2 / block_size)
        block_num = max_block_num * b

        # max_block_num = 257
        # block_num = 2048*5
        print("max_block_num = ", max_block_num)
        print("block_num = ", block_num)

        query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type)
        if layout_kv == "PA_BSND":
            key = torch.tensor(
                np.random.uniform(
                    -100, 100, math.ceil(b * (s2 / block_size), block_size, n2, dn)
                )
            ).to(torch.int8)
            value = torch.tensor(
                np.random.uniform(
                    -100, 100, math.ceil(b * (s2 / block_size), block_size, n2, dn)
                )
            ).to(torch.int8)
        elif layout_kv == "PA_BNSD":
            key = torch.tensor(
                np.random.uniform(
                    -100, 100, (b * math.ceil(s2 / block_size), n2, block_size, dn)
                )
            ).to(torch.int8)
            value = torch.tensor(
                np.random.uniform(
                    -100, 100, (b * math.ceil(s2 / block_size), n2, block_size, dn)
                )
            ).to(torch.int8)
        elif layout_kv == "PA_NZ":
            key = torch.tensor(
                np.random.uniform(
                    -100, 100, (block_num, n2, (dn // 32), block_size, 32)
                )
            ).to(torch.int8)
            value = torch.tensor(
                np.random.uniform(
                    -100, 100, (block_num, n2, (dn // 32), block_size, 32)
                )
            ).to(torch.int8)
        elif layout_kv == "BSND":
            key = torch.tensor(np.random.uniform(-100, 100, (b, s2, n2, dn))).to(
                torch.int8
            )
            value = torch.tensor(np.random.uniform(-100, 100, (b, s2, n2, dn))).to(
                torch.int8
            )
        key_antiquant_scale = torch.tensor(np.random.uniform(-100, 100, (n2, dn))).to(
            torch.float32
        )
        value_antiquant_scale = torch.tensor(np.random.uniform(-100, 100, (n2, dn))).to(
            torch.float32
        )
        sparse_indices = torch.full(
            (b, n2, s1 // sparse_shard_size, maxsparse_block_count), -1
        ).to(torch.int32)
        # generate sparse_indices
        for b_i in range(b):
            for n_i in range(n2):
                for s_i in range(s1 // sparse_shard_size):
                    if sparse_mode == 0:
                        threshold = act_seq_kv[b_i]
                    elif sparse_mode == 3:
                        threshold = (
                            act_seq_kv[b_i]
                            - act_seq_q[b_i]
                            + s_i * sparse_shard_size
                            + sparse_shard_size
                        )
                    if threshold <= 0:
                        sparse_indices[b_i, n_i, s_i, :] = torch.tensor(
                            [-1] * maxsparse_block_count
                        ).to(torch.int32)
                        continue
                    valid_blocks_max = math.ceil(max(0, threshold) / sparse_block_size)
                    # 处理边界情况：如果有效块数为0，跳过
                    if valid_blocks_max == 0:
                        continue
                    valid_blocks_topk = min(valid_blocks_max, sparse_block_count[b_i])
                    valid_blocks_topk_int = int(valid_blocks_topk)
                    # 情况1: 只能选1个块
                    if valid_blocks_topk_int <= 1:
                        # 只选最后一个块
                        sparse_indices[b_i, n_i, s_i, 0:1] = valid_blocks_max - 1
                    # 情况2: 可以选2个或更多块
                    else:
                        # 计算要随机选择的块数
                        random_blocks = max(0, valid_blocks_topk_int - 2)
                        if random_blocks > 0 and valid_blocks_max - 2 > 0:
                            # 从除了最后两个块之外的块中随机选择
                            block_indices = torch.randperm(valid_blocks_max - 2).to(
                                torch.int32
                            )
                            # 填充随机选择的块
                            sparse_indices[b_i, n_i, s_i, :random_blocks] = (
                                block_indices[0:random_blocks]
                            )
                        # 固定选择最后两个块
                        if valid_blocks_max >= 2:
                            # 先放倒数第二个块
                            sparse_indices[b_i, n_i, s_i, random_blocks] = (
                                valid_blocks_max - 2
                            )
                            # 再放最后一个块
                            sparse_indices[b_i, n_i, s_i, random_blocks + 1] = (
                                valid_blocks_max - 1
                            )
                        else:
                            # 如果只有1个块，只选最后一个块
                            sparse_indices[b_i, n_i, s_i, random_blocks] = (
                                valid_blocks_max - 1
                            )
        sparse_indices = torch.transpose(sparse_indices, 1, 2)
        print("sparse_indices = ", sparse_indices)
        # generate block_table
        if layout_kv == "PA_BSND" or layout_kv == "PA_BNSD" or layout_kv == "PA_NZ":
            block_table = torch.full((b, max_block_num), -1).to(torch.int32)
            block_numPerBlock = []
            block_num_min = 0
            for actual_seq in act_seq_kv:
                block_numPerBlock.append(math.ceil(actual_seq / block_size))
                block_num_min += math.ceil(actual_seq / block_size)
            if block_num_min > block_num:
                raise RuntimeError(
                    f"block_num{block_num} is too small, please increase block_num to at least {block_num_min}"
                )
            block_idx_list = torch.randperm(block_num).to(torch.int32)
            block_idx = 0
            block_table_batch_idx = 0
            for idx in block_numPerBlock:
                for j in range(idx):
                    block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
                    block_idx += 1
                block_table_batch_idx += 1
            block_table = block_table.to("npu:%s" % DEVICE_ID)
        else:
            block_table = None

        # compare result
        cpu_out = cpu_sparse_flash_attention_antiquant_gqa(
            query,
            key,
            value,
            sparse_indices,
            key_dequant_scale=key_antiquant_scale,
            value_dequant_scale=value_antiquant_scale,
            scale_value=scale_value,
            sparse_block_size=sparse_block_size,
            actual_seq_lengths_query=act_seq_q,
            actual_seq_lengths_kv=act_seq_kv,
            sparse_seq_lengths_kv=sparse_seq_kv,
            layout_query=layout_query,
            layout_kv=layout_kv,
            sparse_mode=sparse_mode,
            block_table=block_table,
            attention_mode=attention_mode,
            quant_scale_repo_mode=quant_scale_repo_mode,
            tile_size=tile_size,
            key_quant_mode=key_quant_mode,
            value_quant_mode=value_quant_mode,
            rope_head_dim=64,
            sparse_shard_size=sparse_shard_size,
        )

        query = query.to("npu:%s" % DEVICE_ID)
        key = key.to("npu:%s" % DEVICE_ID)
        value = value.to("npu:%s" % DEVICE_ID)
        sparse_indices = sparse_indices.to("npu:%s" % DEVICE_ID)
        key_antiquant_scale = key_antiquant_scale.to("npu:%s" % DEVICE_ID)
        value_antiquant_scale = value_antiquant_scale.to("npu:%s" % DEVICE_ID)
        act_seq_q = act_seq_q.to("npu:%s" % DEVICE_ID)
        act_seq_kv = act_seq_kv.to("npu:%s" % DEVICE_ID)
        sparse_seq_kv = sparse_seq_kv.to("npu:%s" % DEVICE_ID)

        # print(f'======================== TorchNPU eager BEGIN ========================')
        sals_sfa_meta = torch_npu.npu_sparse_flash_attention_antiquant_metadata(
            b,
            s1,
            n1,
            s2,
            n2,
            dn,
            10,
            sparse_block_size,
            actual_seq_lengths_query=act_seq_q,
            actual_seq_lengths_kv=act_seq_kv,
            sparse_seq_lengths_kv=sparse_seq_kv,
            sparse_mode=sparse_mode,
            attention_mode=attention_mode,
            rope_head_dim=64,
            sparse_shared_size=sparse_shard_size,
        )
        npu_out = torch_npu.npu_sparse_flash_attention_antiquant(
            query,
            key,
            value,
            sparse_indices,
            key_dequant_scale=key_antiquant_scale,
            value_dequant_scale=value_antiquant_scale,
            scale_value=scale_value,
            sparse_block_size=sparse_block_size,
            metadata=sals_sfa_meta,
            actual_seq_lengths_query=act_seq_q,
            actual_seq_lengths_kv=act_seq_kv,
            sparse_seq_lengths_kv=sparse_seq_kv,
            layout_query=layout_query,
            layout_kv=layout_kv,
            sparse_mode=sparse_mode,
            block_table=block_table,
            attention_mode=attention_mode,
            quant_scale_repo_mode=quant_scale_repo_mode,
            tile_size=tile_size,
            key_quant_mode=key_quant_mode,
            value_quant_mode=value_quant_mode,
            rope_head_dim=64,
            sparse_shard_size=sparse_shard_size,
        )
        print("npu_out = ", npu_out)

        npu_out = npu_out.cpu().to(torch.float32).numpy()
        cpu_out = cpu_out.cpu().to(torch.float32).numpy()

        res = np.isclose(npu_out, cpu_out, rtol=0.005, atol=0.0001, equal_nan=False)
        diff_mask = ~res
        true_ratio = np.mean(res)
        print("npu output:\n", npu_out, npu_out.shape)
        print("cpu output:\n", cpu_out, cpu_out.shape)
        print("correct ratio of cpu vs npu is:", true_ratio * 100, "%")
        total_elements = npu_out.size
        diff_elements = np.sum(diff_mask)
        diff_perventage = (diff_elements / total_elements) * 100

        print(f"总元素数: {total_elements}")
        print(f"不同元素数: {diff_elements}")
        print(f"不同比例: {diff_perventage:.2f}%")

        # 输出差别详情
        if diff_elements > 0:
            diff_indices = np.where(diff_mask)
            print("\n差别详情: ")
            for idx in range(len(diff_indices[0])):
                pos = tuple(dim[idx] for dim in diff_indices)
                a_val = npu_out[pos]
                b_val = cpu_out[pos]
                diff_val = abs(a_val - b_val)
                diff_val_relative = diff_val / abs(b_val)
                if diff_val_relative > 0.5:
                    print(
                        f"Warning! 位置 {pos}: npu_out={a_val:.6f}, cpu_out={b_val:.6f}, 差值={diff_val:.6f}, diff_value_relative={diff_val_relative:.6f}"
                    )
                else:
                    print(
                        f"位置 {pos}: npu_out={a_val:.6f}, cpu_out={b_val:.6f}, 差值={diff_val:.6f}, diff_value_relative={diff_val_relative:.6f}"
                    )
        self.assertTrue(true_ratio > 0.99, "precision compare fail")
        print("======================== TorchNPU eager FINISH ========================")

        print(
            "======================== TorchNPU eager Graph BEGIN ========================"
        )
        print("sparse_indices = ", sparse_indices)
        print("block_table = ", block_table)
        npu_mode = SFAANetwork().to("npu:%s" % DEVICE_ID)
        from torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        torch._dynamo.reset()
        npu_mode = torch.compile(
            npu_mode, fullgraph=False, backend=npu_backend, dynamic=False
        )
        npu_out0 = npu_mode(
            b,
            s1,
            n1,
            s2,
            n2,
            dn,
            query,
            key,
            value,
            sparse_indices,
            key_dequant_scale=key_antiquant_scale,
            value_dequant_scale=value_antiquant_scale,
            scale_value=scale_value,
            sparse_block_size=sparse_block_size,
            actual_seq_lengths_query=act_seq_q,
            actual_seq_lengths_kv=act_seq_kv,
            sparse_seq_lengths_kv=sparse_seq_kv,
            layout_query=layout_query,
            layout_kv=layout_kv,
            sparse_mode=sparse_mode,
            block_table=block_table,
            attention_mode=attention_mode,
            quant_scale_repo_mode=quant_scale_repo_mode,
            tile_size=tile_size,
            key_quant_mode=key_quant_mode,
            value_quant_mode=value_quant_mode,
            rope_head_dim=64,
            sparse_shard_size=sparse_shard_size,
        )
        npu_out0 = npu_out0.cpu().to(torch.float32).numpy()
        cpu_out = cpu_out.to(torch.float32).numpy()
        res = np.isclose(npu_out0, cpu_out, rtol=0.005, atol=0.0001, equal_nan=False)
        true_ratio = np.mean(res)
        if true_ratio < 0.99:
            print("npu output:\n", npu_out0, npu_out0.shape)
            print("cpu output:\n", cpu_out, cpu_out.shape)
            print("correct ratio of cpu vs npu is:", true_ratio * 100, "%")
        self.assertTrue(true_ratio > 0.99, "precision compare fail")
        print(
            "======================== TorchNPU eager Graph FINISH ========================"
        )


if __name__ == "__main__":
    run_tests()
