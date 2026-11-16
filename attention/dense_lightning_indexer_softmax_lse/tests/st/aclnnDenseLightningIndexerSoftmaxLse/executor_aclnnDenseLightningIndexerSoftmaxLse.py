#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor, AclIntArray
from atk.tasks.dataset.base_dataset import OpsDataset
import numpy as np
import random
import math


def generate_increasing_lists(B: int, T_Q: int, T_KV: int, seed: int = None):
    """
    生成 seq_len_q 和 seq_len_kv 前缀和序列。

    Args:
        B: batch 大小
        T_Q: query 序列总长度上限
        T_KV: kv 序列总长度上限
        seed: 随机种子

    Returns:
        (seq_len_q, seq_len_kv): 长度为 B 的前缀和数组
    """
    if B == 1:
        return [T_Q], [T_KV]

    if seed is not None:
        random.seed(seed)

    K = 1024
    # 候选的大值区间: [1023K, 1024K], [511K, 512K], [127K, 128K]
    candidate_ranges = [
        (1023 * K, 1024 * K),
        (511 * K, 512 * K),
        (127 * K, 128 * K),
    ]

    # 选择不超过 T_KV 的最大区间
    big_range = None
    for lo, hi in candidate_ranges:
        if lo <= T_KV:
            big_range = (lo, hi)
            break

    # 决定哪个 batch 项放大值
    big_idx = random.randint(0, B - 1)

    skv = [0] * B
    remaining_kv = T_KV
    remaining_q = T_Q

    # 先给大值项分配
    if big_range is not None:
        big_val = random.randint(big_range[0], min(big_range[1], T_KV))
        skv[big_idx] = big_val
        remaining_kv -= big_val
    else:
        # T_KV 太小，随便分配
        skv[big_idx] = random.randint(0, T_KV)
        remaining_kv -= skv[big_idx]

    # 给剩余 B-1 项随机分配 kv 长度
    other_indices = [i for i in range(B) if i != big_idx]
    random.shuffle(other_indices)

    # 当 B >= 2 时，确保至少一个 other 项为 0（保证 skv 至少有一个 0，且该位置 sq 也为 0）
    # B == 1 时无需包含 0
    if B >= 2:
        zero_idx = other_indices[0]
        other_indices = other_indices[1:]

    for idx in other_indices:
        if remaining_kv <= 0:
            skv[idx] = 0
        else:
            skv[idx] = random.randint(0, remaining_kv)
            remaining_kv -= skv[idx]

    random.shuffle(skv)

    # 生成 sq，满足 sq[i] <= skv[i]
    sq = [0] * B
    for i in range(B):
        if skv[i] == 0:
            sq[i] = 0
        else:
            sq[i] = random.randint(0, skv[i])

    # 确保 sq 至少有一个 0（skv 为 0 的位置 sq 必然为 0，已满足）

    # 确保 sq 总和不超过 T_Q
    total_sq = sum(sq)
    if total_sq > T_Q:
        # 按比例缩减
        scale = T_Q / total_sq
        sq = [int(v * scale) for v in sq]
        # 逐个分配余量
        remainder = T_Q - sum(sq)
        for i in range(B):
            if remainder <= 0:
                break
            add = min(remainder, skv[i] - sq[i])
            sq[i] += add
            remainder -= add

    # 计算前缀和
    seq_len_q = np.cumsum(sq).astype(np.int64).tolist()
    seq_len_kv = np.cumsum(skv).astype(np.int64).tolist()

    return seq_len_q, seq_len_kv


def batch_matmul_torch(q_head_num, k_head_num, q, k):
    g_size = q_head_num // k_head_num
    score = None
    for i in range(k_head_num):
        group_score = torch.matmul(
            q[i * g_size : (i + 1) * g_size, :, :].to(torch.float32),
            k[i : (i + 1), :, :].to(torch.float32),
        )
        if score is None:
            score = group_score
        else:
            score = torch.cat((score, group_score), 0)
    return torch.nn.functional.relu(score)


def do_scale_torch(s, w):
    return torch.mul(s, w.to(torch.float32))


def dense_lightning_indexer_softmax(
    input_query,
    input_key,
    input_weight,
    input_actual_seq_lengths_query,
    input_actual_seq_lengths_key,
    layout,
    sparse_mode,
):
    if layout == "BSND":
        batch_size = input_query.shape[0]
        q_max_seq_len = input_query.shape[1]
        q_head_num = input_query.shape[2]
        head_dim = input_query.shape[3]

        k_max_seq_len = input_key.shape[1]
        k_head_num = input_key.shape[2]
        head_dim = input_key.shape[3]

    if layout == "TND":
        q_head_num = input_query.shape[1]
        head_dim = input_query.shape[2]

        k_head_num = input_key.shape[1]
        head_dim = input_key.shape[2]

        batch_size = len(input_actual_seq_lengths_query)

    q_seq_accum = (
        batch_size * q_max_seq_len
        if layout == "BSND"
        else input_actual_seq_lengths_query[-1]
    )
    k_seq_accum = (
        batch_size * k_max_seq_len
        if layout == "BSND"
        else input_actual_seq_lengths_key[-1]
    )
    data_type = input_query.dtype

    # 1.计算过程中query、key 原来BSND统一转换为TND格式使用
    query = torch.zeros((q_seq_accum, q_head_num, head_dim), dtype=data_type)
    key = torch.zeros((k_seq_accum, k_head_num, head_dim), dtype=data_type)
    weight = torch.zeros((q_seq_accum, q_head_num), dtype=data_type)

    if layout == "BSND":
        cur_q_seqlen = 0
        cur_k_seqlen = 0
        for i in range(batch_size):
            q_seqlen = q_max_seq_len
            k_seqlen = k_max_seq_len
            query[cur_q_seqlen : (cur_q_seqlen + q_seqlen), :, :] = input_query[
                i, :q_seqlen, :, :
            ]
            key[cur_k_seqlen : (cur_k_seqlen + k_seqlen), :, :] = input_key[
                i, :k_seqlen, :, :
            ]
            weight[cur_q_seqlen : (cur_q_seqlen + q_seqlen), :] = input_weight[
                i, :q_seqlen, :
            ]
            cur_q_seqlen += q_seqlen
            cur_k_seqlen += k_seqlen
    else:
        query = input_query
        key = input_key
        weight = input_weight

    # 2.初始化out softmax_max、softmax_sum数据
    softmax_max_out = (
        torch.zeros(batch_size * q_max_seq_len, dtype=torch.float32)
        if layout == "BSND"
        else torch.zeros(q_seq_accum, dtype=torch.float32)
    )
    softmax_sum_out = (
        torch.zeros(batch_size * q_max_seq_len, dtype=torch.float32)
        if layout == "BSND"
        else torch.zeros(q_seq_accum, dtype=torch.float32)
    )

    # 3.循环batch_size计算
    cur_q_seqlen = 0
    cur_k_seqlen = 0
    out_q_seqlen = 0

    for i in range(batch_size):
        if layout == "TND" and i > 0:
            q_seqlen = (
                input_actual_seq_lengths_query[i]
                - input_actual_seq_lengths_query[i - 1]
            )
            k_seqlen = (
                input_actual_seq_lengths_key[i] - input_actual_seq_lengths_key[i - 1]
            )
        elif layout == "TND" and i == 0:
            q_seqlen = input_actual_seq_lengths_query[i]
            k_seqlen = input_actual_seq_lengths_key[i]
        else:
            q_seqlen = q_max_seq_len
            k_seqlen = k_max_seq_len

        if q_seqlen == 0 or k_seqlen == 0:
            continue

        q = query[
            cur_q_seqlen : (cur_q_seqlen + q_seqlen), :, :
        ]  # (s1, q_head_num, head_dim)
        k = key[
            cur_k_seqlen : (cur_k_seqlen + k_seqlen), :, :
        ]  # (s2, k_head_num, head_dim)
        w = weight[cur_q_seqlen : (cur_q_seqlen + q_seqlen), :]  # (s1, q_head_num)

        q = torch.permute(q, (1, 0, 2))  # (q_head_num, s1, head_dim)
        k = torch.permute(k, (1, 2, 0))  # (k_head_num, head_dim, s2)
        s = batch_matmul_torch(
            q_head_num, k_head_num, q, k
        )  # (k_head_num * g_size, s1, s2)

        s = torch.permute(s, (2, 1, 0))  # (s2, s1, k_head_num * g_size)
        p = do_scale_torch(s, w)  # (s2, s1, k_head_num * g_size)
        p = torch.permute(p, (1, 0, 2))  # s1, s2, k_head_num * g_size
        p_reduce_sum = torch.sum(p, dim=2)  # s1, s2

        if sparse_mode == 3:
            # 只计算下三角
            for s1_idx in range(q_seqlen):
                p_reduce_sum[s1_idx, (k_seqlen - q_seqlen + s1_idx + 1) :] = float(
                    "-inf"
                )
        else:
            pass

        max_out, _ = torch.max(p_reduce_sum, dim=-1, keepdim=False)
        max_res, _ = torch.max(p_reduce_sum, dim=-1, keepdim=True)
        sub_res = p_reduce_sum - max_res
        exp_res = torch.exp(sub_res)
        sum_out = torch.sum(exp_res, dim=-1, keepdim=False)
        softmax_max_out[out_q_seqlen : (out_q_seqlen + q_seqlen)] = max_out
        softmax_sum_out[out_q_seqlen : (out_q_seqlen + q_seqlen)] = sum_out

        cur_q_seqlen += q_seqlen
        cur_k_seqlen += k_seqlen
        out_q_seqlen += q_seqlen if layout == "TND" else q_max_seq_len

    softmax_max_out = torch.unsqueeze(softmax_max_out, 0)
    softmax_sum_out = torch.unsqueeze(softmax_sum_out, 0)

    if layout == "BSND":
        softmax_max_out = softmax_max_out.reshape(batch_size, 1, q_max_seq_len)
        softmax_sum_out = softmax_sum_out.reshape(batch_size, 1, q_max_seq_len)

    return softmax_max_out, softmax_sum_out


def dense_lightning_indexer_softmax_perf(
    input_query_index,
    input_key_index,
    input_weight,
    input_actual_seq_lengths_query,
    input_actual_seq_lengths_key,
    layout,
    sparse_mode,
):
    device = input_query_index.device
    if layout == "TND":
        slice_loop = len(input_actual_seq_lengths_query)
        t1 = input_query_index.shape[0]
        t2 = input_key_index.shape[0]
        q_head_num = input_query_index.shape[1]
        k_head_num = input_key_index.shape[1]
        softmax_max_out = torch.zeros(k_head_num, t1, dtype=torch.float32)
        softmax_sum_out = torch.zeros(k_head_num, t1, dtype=torch.float32)

    if layout == "BSND":
        slice_loop = 1
        q_head_num = input_query_index.shape[2]
        k_head_num = input_key_index.shape[2]
        batch_size = input_query_index.shape[0]
        s1_max = input_query_index.shape[1]
        softmax_max_out = torch.zeros(
            batch_size, k_head_num, s1_max, dtype=torch.float32
        )
        softmax_sum_out = torch.zeros(
            batch_size, k_head_num, s1_max, dtype=torch.float32
        )

    g_size = q_head_num // k_head_num

    for i in range(slice_loop):
        if layout == "TND":
            s1_start = 0 if i == 0 else input_actual_seq_lengths_query[i - 1]
            s1_end = input_actual_seq_lengths_query[i]
            s2_start = 0 if i == 0 else input_actual_seq_lengths_key[i - 1]
            s2_end = input_actual_seq_lengths_key[i]
            query_index = input_query_index[s1_start:s1_end, :, :].unsqueeze(0)
            key_index = input_key_index[s2_start:s2_end, :, :].unsqueeze(0)
            weight = input_weight[s1_start:s1_end, :].unsqueeze(0)
        else:
            query_index = input_query_index
            key_index = input_key_index
            weight = input_weight

        s1 = query_index.shape[1]
        s2 = key_index.shape[1]
        slice_b = query_index.shape[0]

        # 切分s1
        s1_block_size = 64
        s1_block_num = math.ceil(s1 / s1_block_size)
        # [B,S2,N2,D] -> [B,N2*G,D,S2]
        key_index = key_index.permute(0, 2, 3, 1).contiguous().repeat(1, g_size, 1, 1)
        for s1_block_index in range(s1_block_num):
            s1_block_start = s1_block_index * s1_block_size
            s1_block_end = (
                s1
                if s1 < (s1_block_index + 1) * s1_block_size
                else (s1_block_index + 1) * s1_block_size
            )
            # [B,S1,N1,D] -> [B,N1,S1,D]
            query_index_block = (
                query_index[:, s1_block_start:s1_block_end, :, :]
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            # batch matmul: [B,N1,S1,S2]
            s_tmp = torch.matmul(
                query_index_block.to(torch.float32), key_index.to(torch.float32)
            )
            # relu: [B,N1,S1,S2]
            s_relu = torch.relu(s_tmp)
            del s_tmp
            # WEIGHT: [B,S1,N1] -> [B,N1,S1,1]
            weight_block = (
                weight[:, s1_block_start:s1_block_end, :]
                .permute(0, 2, 1)
                .contiguous()
                .unsqueeze(-1)
            )
            # Mul: [B,N1,S1,S2]
            s_mul = s_relu * weight_block.to(torch.float32)
            s_mul = torch.reshape(
                s_mul, (slice_b, k_head_num, g_size, s1_block_end - s1_block_start, s2)
            )
            # reduce: [B,N2,S1,S2]
            s_reduce = torch.sum(s_mul, dim=2)
            del s_mul
            diag_start_index = s2 - s1 + s1_block_index * s1_block_size
            tril_s = torch.tril(s_reduce, diagonal=diag_start_index)
            triu_mask = torch.triu(
                torch.ones_like(s_reduce), diagonal=(diag_start_index + 1)
            ).to(s_reduce.dtype)
            triu_inf = torch.where(
                triu_mask == 1,
                torch.tensor(float("-inf"), dtype=s_reduce.dtype).to(device),
                torch.tensor(0, dtype=s_reduce.dtype).to(device),
            )
            # reduce mask: [B,N2,S1,S2]
            s_reduce = tril_s + triu_inf
            # max, sub, exp, sum [B,N2,S1]
            max_out, _ = torch.max(s_reduce, dim=-1, keepdim=False)
            max_res, _ = torch.max(s_reduce, dim=-1, keepdim=True)
            sub_res = s_reduce - max_res
            exp_res = torch.exp(sub_res)
            sum_out = torch.sum(exp_res, dim=-1, keepdim=False)

            if layout == "TND":
                softmax_max_out[
                    :, (s1_start + s1_block_start) : (s1_start + s1_block_end)
                ] = max_out
                softmax_sum_out[
                    :, (s1_start + s1_block_start) : (s1_start + s1_block_end)
                ] = sum_out

            if layout == "BSND":
                softmax_max_out[:, :, s1_block_start:s1_block_end] = max_out
                softmax_sum_out[:, :, s1_block_start:s1_block_end] = sum_out

    return softmax_max_out, softmax_sum_out


@register("function_aclnnDenseLightningIndexerSoftmaxLse")
class AclnnDenseLightningIndexerSoftmaxLseGoldenExecutor(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        if input_data.kwargs["layout"] == "TND":
            T1 = input_data.kwargs["query_index"].shape[0]
            T2 = input_data.kwargs["key_index"].shape[0]
            B = len(input_data.kwargs["actual_seq_lengths_query"])
            random.seed(T2)
            list1, list2 = generate_increasing_lists(B, T1, T2, seed=T1)

            input_data.kwargs["actual_seq_lengths_query"] = list1
            input_data.kwargs["actual_seq_lengths_key"] = list2

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        query_index = input_data.kwargs["query_index"]
        key_index = input_data.kwargs["key_index"]
        weights = input_data.kwargs["weights"]
        actual_seq_lengths_query = input_data.kwargs["actual_seq_lengths_query"]
        actual_seq_lengths_key = input_data.kwargs["actual_seq_lengths_key"]
        layout = input_data.kwargs["layout"]
        sparse_mode = input_data.kwargs["sparse_mode"]
        pre_tokens = input_data.kwargs["pre_tokens"]
        next_tokens = input_data.kwargs["next_tokens"]

        if layout == "BSND":
            actual_seq_lengths_query = None
            actual_seq_lengths_key = None
        softmax_max, softmax_sum = dense_lightning_indexer_softmax_perf(
            query_index,
            key_index,
            weights,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            layout,
            sparse_mode,
        )

        return softmax_max, softmax_sum


@register("aclnnfunction_aclnnDenseLightningIndexerSoftmaxLse")
class AclnnDenseLightningIndexerSoftmaxLseExecutor(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        if input_data.kwargs["layout"] == "TND":
            T1 = input_data.kwargs["query_index"].shape[0]
            T2 = input_data.kwargs["key_index"].shape[0]
            B = len(input_data.kwargs["actual_seq_lengths_query"])
            random.seed(T2)

            list1, list2 = generate_increasing_lists(B, T1, T2, seed=T1)

            input_data.kwargs["actual_seq_lengths_query"] = list1
            input_data.kwargs["actual_seq_lengths_key"] = list2

        import ctypes

        layout = input_data.kwargs["layout"]

        input_args, output_packages = super().init_by_input_data(input_data)
        AclTensorPtr = ctypes.POINTER(AclIntArray)  # tensor指针类型
        null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
        null_tensor_ptr = ctypes.cast(
            null_void_ptr, AclTensorPtr
        )  # 把这个空指针类型转换为tensor指针类型
        if layout == "BSND":
            # BSND场景传空指针：actual_seq_lengths_query、actual_seq_lengths_key
            input_args[3] = null_tensor_ptr
            input_args[4] = null_tensor_ptr

        return input_args, output_packages
