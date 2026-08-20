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

import test
import torch
import torch_npu
import pytest
import random
import numpy as np
import math
import ctypes
import copy

ENABLE_DISCONTIGUOUS_KEY = True  # 是否开启key cache非连续
KEY_D_EXPAND = 0  # key cache非连续时head_dim的扩展大小


class GeneralizedLI:
    def __init__(
        self,
        batch_size,
        q_seq,
        k_seq,
        q_t_size,
        k_t_size,
        q_head_num,
        k_head_num,
        head_dim,
        block_size,
        block_num,
        qk_dtype,
        weight_dtype,
        actual_seq_dtype,
        act_seq_q,
        act_seq_k,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
    ):
        self.batch_size = batch_size
        self.q_seq = q_seq
        self.k_seq = k_seq
        self.q_t_size = q_t_size
        self.k_t_size = k_t_size
        self.q_head_num = q_head_num
        self.k_head_num = k_head_num
        self.group_size = q_head_num // k_head_num
        self.head_dim = head_dim
        self.block_size = block_size
        self.block_num = block_num
        self.qk_dtype = qk_dtype
        self.weight_dtype = weight_dtype
        self.actual_seq_dtype = actual_seq_dtype
        self.act_seq_q = act_seq_q
        self.act_seq_k = act_seq_k
        self.layout_query = layout_query
        self.layout_key = layout_key
        self.sparse_count = sparse_count
        self.sparse_mode = sparse_mode

        if layout_query == "BSND":
            self.out_shape = [batch_size, q_seq, k_head_num, sparse_count]
        elif layout_query == "TND":
            self.out_shape = [q_t_size, k_head_num, sparse_count]

    def cal_atten_bnsd(self):
        batch_size = self.batch_size
        qs = self.q_seq
        ks = self.k_seq
        n1 = self.q_head_num
        n2 = self.k_head_num
        actualSeqLengths_q = self.actual_seq_lengths_query
        actualSeqLengths_k = self.actual_seq_lengths_key
        q_bnsd_tensor = self.q_bnsd_tensor
        k_bnsd_tensor = self.k_bnsd_tensor
        wt_bnsd_tensor = self.wt_bnsd_tensor
        mask_tensor = self.m_tensor
        out_shape_bnsd = list(q_bnsd_tensor.shape)
        out_shape_bnsd[1] = n2
        out_shape_bnsd[-1] = self.sparse_count

        out_shape_bnss = list(q_bnsd_tensor.shape)
        out_shape_bnss[1] = n2
        out_shape_bnss[-1] = math.floor(max(actualSeqLengths_k))

        y = torch.full(out_shape_bnsd, -1, dtype=torch.int32)
        y_value = torch.full(out_shape_bnss, -float("inf"), dtype=torch.float32)

        for b_idx in range(batch_size):
            curr_actualSeq_q = actualSeqLengths_q[b_idx]
            curr_actualSeq_k = math.floor(actualSeqLengths_k[b_idx])
            self.cur_actseq_q = curr_actualSeq_q
            self.cur_actseq_k = curr_actualSeq_k
            self.cur_q = q_bnsd_tensor[b_idx : (b_idx + 1), :, :curr_actualSeq_q, :]
            self.cur_k = k_bnsd_tensor[b_idx : (b_idx + 1), :, :curr_actualSeq_k, :]
            self.cur_wt = wt_bnsd_tensor[b_idx : (b_idx + 1), :, :curr_actualSeq_q, :]
            if self.sparse_mode != 0:
                self.cur_m = mask_tensor[
                    b_idx : (b_idx + 1), :curr_actualSeq_q, :curr_actualSeq_k
                ]
            self.cur_b_idx = b_idx

            if curr_actualSeq_q != 0:
                actual_selected_count = min(curr_actualSeq_k, self.sparse_count)
                (
                    y[
                        b_idx : (b_idx + 1),
                        :,
                        :curr_actualSeq_q,
                        :actual_selected_count,
                    ],
                    y_value[
                        b_idx : (b_idx + 1), :, :curr_actualSeq_q, :curr_actualSeq_k
                    ],
                ) = self.cal_atten_per_batch_b16(b_idx)
            else:
                pass
        return y, y_value

    def trans_shape_to_bnsd(
        self, tensor, layout, headnums=None, act_seq=None, tensor_name=None
    ):
        if layout == "BSND":
            B = tensor.shape[0]
            S = tensor.shape[1]
            N = tensor.shape[2]
            D = 1
            if len(tensor.shape) == 3:
                tensor = torch.unsqueeze(tensor, dim=-1)
            else:
                D = tensor.shape[3]
            tensor = tensor.reshape(B, S, N, D).permute(0, 2, 1, 3)
            return tensor
        elif layout == "TND":
            T = tensor.shape[0]
            N = tensor.shape[1]
            D = 1
            if len(tensor.shape) == 2:
                tensor = torch.unsqueeze(tensor, dim=-1)
            else:
                D = tensor.shape[2]
            B = len(act_seq)
            S = max(act_seq)
            new_tensor = torch.zeros((B, N, S, D), dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = act_seq[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:act_s, :] = tensor[
                        t_start:t_end, n_index, :
                    ]
                t_start += act_s
            return new_tensor
        elif layout == "PA_BSND":
            if len(tensor.shape) == 3:
                tensor = torch.unsqueeze(tensor, dim=-1)
            return tensor
        else:
            return tensor

    def trans_tnd_actseq(self, list):
        list_len = len(list)
        if list_len == 0:
            raise ValueError("TND情况下 act_seq需要必传")
        list_new = []
        list_new.append(list[0])
        for i in range(list_len - 1):
            new_item = list[i + 1] - list[i]
            if new_item >= 0:
                list_new.append(new_item)
            else:
                raise ValueError(
                    f"TND情况下 act_seq_len 为非递减数列 act_seq_len={list}"
                )
        return list_new

    def cal_atten_per_batch_b16(self, b_idx):
        cur_q = self.cur_q.to(dtype=torch.float32)
        cur_k = self.cur_k.to(dtype=torch.float32)
        cur_wt = self.cur_wt.to(dtype=torch.float32)

        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode

        qk_bmm_res = torch.bmm(
            cur_q.squeeze(0), cur_k.permute(0, 1, 3, 2).squeeze(0)
        ).unsqueeze(0)

        qk_relu_out = (qk_bmm_res.to(dtype=torch.float32)).clamp_min(0.0)

        brc_vmul = torch.bmm(
            cur_wt.permute(0, 2, 3, 1).to(dtype=torch.float32).squeeze(0),
            qk_relu_out.permute(0, 2, 1, 3).to(dtype=torch.float32).squeeze(0),
        ).unsqueeze(0)

        temp_b, temp_s1, temp_n1, temp_s2 = brc_vmul.shape
        temp_g = self.group_size
        temp_n2 = self.k_head_num
        temp_b_idx = self.cur_b_idx
        actual_selected_count = min(temp_s2, sparse_count)
        reduce_sum = brc_vmul.reshape(temp_b, temp_n2, temp_s1, temp_s2)

        if sparse_mode == 3:
            cur_m = self.cur_m
            cur_m_broadcasted = cur_m.reshape(1, 1, temp_s1, temp_s2)
            cur_m_broadcasted = torch.broadcast_to(
                cur_m_broadcasted, (1, temp_n2, temp_s1, temp_s2)
            )
            # 根据布尔矩阵置-inf
            reduce_sum[cur_m_broadcasted.to(dtype=torch.bool)] = -torch.inf
        to_be_sort_ele = reduce_sum.clone()
        # 稳定排序
        b_sorted_indices = torch.full(to_be_sort_ele.shape, -1, dtype=torch.int32)
        if sparse_mode == 3:
            for i in range(temp_s1):
                row_mask = cur_m_broadcasted[0, 0, i, :].to(dtype=torch.bool)
                true_indices = torch.where(~row_mask)[0]
                row_ele = to_be_sort_ele[0, 0, i, true_indices]
                indices = torch.arange(len(row_ele), device=row_ele.device)

                sorted_vals, sorted_idx = torch.sort(
                    torch.stack([-row_ele, indices], dim=1), dim=0, stable=True
                )
                b_sorted_indices[0, 0, i, true_indices] = true_indices[
                    sorted_idx[:, 0]
                ].to(torch.int32)
        else:
            for i in range(temp_s1):
                row_ele = to_be_sort_ele[0, 0, i, :]
                indices = torch.arange(len(row_ele), device=row_ele.device)
                sorted_vals, sorted_idx = torch.sort(
                    torch.stack([-row_ele, indices], dim=1), dim=0, stable=True
                )
                b_sorted_indices[0, 0, i, :] = sorted_idx[:, 0]
        topk_indices = b_sorted_indices[..., :actual_selected_count]
        return topk_indices, to_be_sort_ele

    def trans_bnsd_to_layout(self, tensor, shape, layout, act_q=None):
        # 此时的输出D轴是K轴
        if layout == "BSH":
            output = tensor.permute(0, 2, 1, 3).contiguous().view(shape)
            return output
        elif layout == "BSND":
            output = tensor.permute(0, 2, 1, 3).contiguous()
            return output
        elif layout in ["BSND_NBSD", "BNSD_NBSD", "BSH_NBSD"]:
            output = tensor.permute(1, 0, 2, 3).contiguous()
            return output
        elif layout in ["TND", "TND_NTD"]:
            T = sum(act_q)
            B = tensor.shape[0]
            N = tensor.shape[1]
            D = tensor.shape[3]
            output = torch.full(size=(T, N, D), fill_value=-1, dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = act_q[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    output[t_start:t_end, n_index, :] = tensor[
                        b_index, n_index, :act_s, :
                    ]
                t_start += act_s
            if layout == "TND_NTD":
                output = output.permute(1, 0, 2).contiguous()
            return output
        else:
            return tensor

    def broadcast_n_axis(self, n1, n2, temp_tensor):
        g = n1 // n2
        temp_shape = temp_tensor.shape
        B = temp_shape[0]
        S = temp_shape[2]
        D = temp_shape[3]
        modify_tensor = torch.zeros([B, n1, S, D], dtype=temp_tensor.dtype)
        for i in range(n1):
            j = i // g
            modify_tensor[:, i : i + 1, :, :] = temp_tensor[:, j : j + 1, :, :]
        return modify_tensor

    def create_mask(self, m_shape, act_k, S1):
        atten_masks = torch.zeros(tuple(m_shape), dtype=torch.uint8)
        tmp_pos_orig = act_k - S1

        for i in range(S1):
            if (tmp_pos_orig + i + 1) < 0:
                atten_masks[i, :] = 1
            else:
                atten_masks[i, math.floor(tmp_pos_orig + i + 1) :] = 1
        return atten_masks

    def create_mask_right_down(
        self, m_shape, actualSeqLengthsQ, actualSeqLengthsK, batch
    ):
        mask_s_q = m_shape[0]
        mask_s_kv = m_shape[1]
        next_tokens_list = []
        re_mask_batch = []
        pre_tokens = 214748647
        for i in range(batch):
            if len(actualSeqLengthsQ) == 0:
                S1 = mask_s_q
            else:
                S1 = actualSeqLengthsQ[i]

            if len(actualSeqLengthsK) == 0:
                S2 = mask_s_kv
            else:
                S2 = math.floor(actualSeqLengthsK[i])
            next_tokens = S2 - S1
            next_tokens_list.append(next_tokens)
            act_k = actualSeqLengthsK[i]
            atten_masks = self.create_mask(m_shape, act_k, S1)
            re_mask_batch.append(atten_masks)
        re_mask_np = torch.stack(re_mask_batch, dim=0).numpy().astype(np.bool_)
        cpu_mask = torch.from_numpy(re_mask_np)
        return cpu_mask, next_tokens_list

    def forward(
        self,
        query,
        key,
        weights,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
    ):
        print("cpu执行中...")

        # 参数的初始化
        batch_size = self.batch_size
        q_seq = self.q_seq
        k_seq = self.k_seq
        layout_query = self.layout_query
        layout_key = self.layout_key
        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        out_shape = self.out_shape
        head_dim = self.head_dim
        q_head_num = self.q_head_num
        k_head_num = self.k_head_num
        q_t_size = self.q_t_size
        k_t_size = self.k_t_size
        block_size = self.block_size
        block_num = self.block_num
        q_dtype = self.qk_dtype
        weight_dtype = self.weight_dtype
        k_dtype = self.qk_dtype
        actual_seq_dtype = self.actual_seq_dtype

        if layout_query == "TND":
            actual_seq_lengths_query = self.trans_tnd_actseq(actual_seq_lengths_query)
            self.actual_seq_lengths_query = torch.tensor(actual_seq_lengths_query)
            actualSeqLengths_q = self.actual_seq_lengths_query

        elif layout_query == "BSND":
            self.actual_seq_lengths_query = actual_seq_lengths_query
            actualSeqLengths_q = self.actual_seq_lengths_query

        if layout_key == "TND":
            actual_seq_lengths_key = self.trans_tnd_actseq(actual_seq_lengths_key)
            self.actual_seq_lengths_key = torch.tensor(actual_seq_lengths_key)
            actualSeqLengths_k = self.actual_seq_lengths_key

        elif layout_key == "BSND":
            self.actual_seq_lengths_key = actual_seq_lengths_key
            actualSeqLengths_k = self.actual_seq_lengths_key

        elif layout_key == "PA_BSND":
            self.actual_seq_lengths_key = actual_seq_lengths_key
            actualSeqLengths_k = self.actual_seq_lengths_key
            k_max_s2 = math.floor(max(actualSeqLengths_k))
        query = query.cpu()
        key = key.cpu()
        weights = weights.cpu()

        # 将输入转化为BNSD
        ## BSND / TND -> BNSD
        q_bnsd_tensor = self.trans_shape_to_bnsd(
            query, layout_query, q_head_num, actualSeqLengths_q
        )

        ## BSND/TND/ -> BNSD
        k_bnsd_tensor = self.trans_shape_to_bnsd(
            key,
            layout_key,
            k_head_num,
            torch.floor(actualSeqLengths_k).to(actual_seq_dtype),
        )

        ## BSN1 -> BNS1   TN1 -> BNS1
        wt_bnsd_tensor = self.trans_shape_to_bnsd(
            weights, layout_query, q_head_num, actualSeqLengths_q
        )

        # 将 k n2轴 广播为 n1
        if q_head_num != k_head_num:
            k_bnsd_tensor = self.broadcast_n_axis(q_head_num, k_head_num, k_bnsd_tensor)
        self.q_bnsd_tensor = q_bnsd_tensor
        self.k_bnsd_tensor = k_bnsd_tensor
        self.wt_bnsd_tensor = wt_bnsd_tensor
        # 生成mask, sparse_mode=3时使能
        m_shape_std = [
            q_bnsd_tensor.shape[2],
            k_bnsd_tensor.shape[2],
        ]  # m_shape应该是[s1,s2]
        batch = q_bnsd_tensor.shape[0]
        m_tensor = []
        if sparse_mode == 3:
            m_tensor, next_tokens_list = self.create_mask_right_down(
                m_shape_std, actualSeqLengths_q, actualSeqLengths_k, batch
            )
        elif sparse_mode == 0:
            pass
        else:
            raise ValueError("unsupported sparse_mode!")
        self.m_tensor = m_tensor
        y, y_value = self.cal_atten_bnsd()
        # TND & PA 需要传入out_shape为BNSD
        out_shape_bnsd = list(q_bnsd_tensor.shape)
        out_shape_bnsd[1] = k_head_num
        out_shape_bnsd[-1] = sparse_count
        y = self.trans_bnsd_to_layout(
            y, out_shape_bnsd, layout_query, actualSeqLengths_q
        )
        return y, y_value


def generate_li_test_data(params):
    """Generate LightningIndexer inputs and CPU golden without running the main op."""
    (
        batch_size,
        q_seq,
        k_seq,
        q_t_size,
        k_t_size,
        q_head_num,
        k_head_num,
        head_dim,
        block_size,
        block_num,
        qk_dtype,
        weight_dtype,
        actual_seq_dtype,
        act_seq_q,
        act_seq_k,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        query_datarange,
        key_datarange,
        weights_datarange,
        return_value,
    ) = params
    import ml_dtypes

    test_li = GeneralizedLI(
        batch_size,
        q_seq,
        k_seq,
        q_t_size,
        k_t_size,
        q_head_num,
        k_head_num,
        head_dim,
        block_size,
        block_num,
        qk_dtype,
        weight_dtype,
        actual_seq_dtype,
        act_seq_q,
        act_seq_k,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
    )

    actual_seq_lengths_query = (
        torch.tensor(np.random.uniform(q_seq, q_seq, batch_size)).to(actual_seq_dtype)
        if act_seq_q is None
        else torch.tensor(act_seq_q).to(actual_seq_dtype)
    )
    actual_seq_lengths_key = (
        torch.tensor(np.random.uniform(k_seq, k_seq, batch_size)).to(actual_seq_dtype)
        if act_seq_k is None
        else torch.tensor(act_seq_k).to(actual_seq_dtype)
    )

    # shape推导
    if layout_query == "BSND":
        query_shape = [batch_size, q_seq, q_head_num, head_dim]
        weights_shape = [batch_size, q_seq, q_head_num]
    elif layout_query == "TND":
        query_shape = [q_t_size, q_head_num, head_dim]
        weights_shape = [q_t_size, q_head_num]

    if layout_key == "BSND":
        key_shape = [batch_size, k_seq, k_head_num, head_dim]
    elif layout_key == "TND":
        key_shape = [k_t_size, k_head_num, head_dim]
    elif layout_key == "PA_BSND":
        # 以不同batch中最大seq为标准初始化key(bnsd)
        k_max_s2 = math.floor(max(act_seq_k))
        key_shape = [batch_size, k_head_num, k_max_s2, head_dim]

    # 随机数据生成
    query = torch.tensor(
        np.random.uniform(query_datarange[0], query_datarange[1], query_shape)
    ).to(qk_dtype)
    weights = torch.tensor(
        np.random.uniform(weights_datarange[0], weights_datarange[1], weights_shape)
    ).to(weight_dtype)
    key = torch.tensor(
        np.random.uniform(key_datarange[0], key_datarange[1], key_shape)
    ).to(qk_dtype)
    block_table = None
    key_cpu = key

    if layout_key == "PA_BSND":
        k_max_block_num_per_batch = math.ceil(
            k_max_s2 / block_size
        )  # 遍历batch得到的最大的block num
        key_block_num_per_batch = []
        key_block_num_sum = 0
        for cur_act_k in act_seq_k:
            cur_cmp_act_k = math.floor(cur_act_k)
            cur_key_block_num = math.ceil(cur_cmp_act_k / block_size)
            key_block_num_per_batch.append(cur_key_block_num)
            key_block_num_sum += cur_key_block_num
        if block_num < key_block_num_sum:
            raise ValueError("key actual block num < needed block num")
        # 构建block table
        block_id_list = np.arange(block_num)
        block_id_list = np.random.permutation(block_id_list).astype(np.int32)
        cur_block_id = 0
        block_table = np.full(
            (batch_size, k_max_block_num_per_batch), fill_value=-1, dtype=np.int32
        )
        batch_idx = 0
        for cur_block_id_threshold in key_block_num_per_batch:
            for i_block_id in range(cur_block_id_threshold):
                block_table[batch_idx][i_block_id] = block_id_list[cur_block_id]
                cur_block_id += 1
            batch_idx += 1
        # 构建PA场景的key
        key_expand = torch.zeros(
            (batch_size, k_head_num, k_max_block_num_per_batch * block_size, head_dim),
            dtype=qk_dtype,
        )
        key_expand[:, :, :k_max_s2, :] = key_cpu
        key = torch.zeros((block_num, block_size, k_head_num, head_dim), dtype=qk_dtype)
        for i_batch in range(batch_size):
            for i_block, cur_block_id in enumerate(block_table[i_batch]):
                block_start_pos = i_block * block_size
                if cur_block_id == -1:
                    continue
                else:
                    for i_n in range(k_head_num):
                        key[cur_block_id, :, i_n, :] = key_expand[
                            i_batch,
                            i_n,
                            block_start_pos : block_start_pos + block_size,
                            :,
                        ]

        # kv_cache 0轴非连续：将key融合到blockFusion (ref v1 commit keyStride0)
        if ENABLE_DISCONTIGUOUS_KEY and KEY_D_EXPAND != 0:
            key_d_new = head_dim + KEY_D_EXPAND
            blockFusion = torch.zeros(
                (block_num, block_size * k_head_num * key_d_new), dtype=qk_dtype
            )  # 整个非连续的长度
            key_flat = key.view(block_num, block_size * k_head_num * head_dim)
            blockFusion[:, : block_size * k_head_num * head_dim] = key_flat
            key = blockFusion[:, : block_size * k_head_num * head_dim].view(
                block_num, block_size, k_head_num, head_dim
            )

        block_table = torch.from_numpy(block_table).to(dtype=torch.int32)

    # cpu golden生成
    cpu_result, topk_value = test_li.forward(
        query,
        key_cpu,
        weights,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
    )
    score_values = test_li.trans_bnsd_to_layout(
        topk_value,
        list(topk_value.shape),
        layout_query,
        test_li.actual_seq_lengths_query,
    )
    return {
        "params": params,
        "cpu_result": cpu_result,
        "topk_value": topk_value,
        "score_values": score_values,
        "query": query,
        "key": key,
        "weights": weights,
        "actual_seq_lengths_query": actual_seq_lengths_query,
        "actual_seq_lengths_key": actual_seq_lengths_key,
        "block_table": block_table,
    }


def li_output_single(params):
    data = generate_li_test_data(params)
    layout_query, layout_key, sparse_count, sparse_mode = params[15:19]
    npu_result, sparse_value = torch_npu.npu_lightning_indexer(
        data["query"].npu(),
        data["key"].npu(),
        data["weights"].npu(),
        actual_seq_lengths_query=data["actual_seq_lengths_query"].npu(),
        actual_seq_lengths_key=data["actual_seq_lengths_key"].npu(),
        block_table=data["block_table"].npu(),
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
    )
    torch.npu.synchronize()
    npu_result, _ = torch.sort(npu_result)
    return data["cpu_result"], npu_result, data["topk_value"], sparse_value
