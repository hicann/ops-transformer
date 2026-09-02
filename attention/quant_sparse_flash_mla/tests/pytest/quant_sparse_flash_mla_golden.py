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

import os
import torch
import random
import numpy as np
import math
from generate_hifloat8_data import (
    trans_float_tensor_to_hifuint8,
    trans_hifuint8_tensor_to_float,
)
import logging

DATA_RANGE_LEFT = -2
DATA_RANGE_RIGHT = 2

FP8_DATA_RANGE_LEFT = -5
FP8_DATA_RANGE_RIGHT = 5


class GeneralizedSFAQuant:
    def __init__(
        self,
        layout_q,
        layout_kv,
        q_type,
        ori_kv_type,
        cmp_kv_type,
        B,
        S1,
        T1,
        N1,
        N2,
        D,
        K,
        block_num1,
        block_num2,
        block_size1,
        block_size2,
        cu_seqlens_q,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        cmp_residual_kv,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        ori_topk_length,
        cmp_topk_length,
        template_run_mode,
        q_descale_val,
        ori_kv_descale_val,
        cmp_kv_descale_val,
    ):
        self.ori_topk_length = ori_topk_length
        self.cmp_topk_length = cmp_topk_length
        self.q_descale_val = q_descale_val
        self.ori_kv_descale_val = ori_kv_descale_val
        self.cmp_kv_descale_val = cmp_kv_descale_val
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.q_type = q_type
        self.ori_kv_type = ori_kv_type
        self.cmp_kv_type = cmp_kv_type
        self.B = B
        self.S1 = S1
        self.T1 = T1
        self.N1 = N1
        self.N2 = N2
        self.D = D
        self.K = K
        self.block_num1 = block_num1
        self.block_num2 = block_num2
        self.block_size1 = block_size1
        self.block_size2 = block_size2
        self.cu_seqlens_q = cu_seqlens_q
        self.seqused_q = seqused_q
        self.seqused_ori_kv = seqused_ori_kv
        self.seqused_cmp_kv = seqused_cmp_kv
        self.cu_seqlens_ori_kv = cu_seqlens_ori_kv
        self.cu_seqlens_cmp_kv = cu_seqlens_cmp_kv
        self.cmp_residual_kv = cmp_residual_kv
        self.softmax_scale = softmax_scale
        self.cmp_ratio = cmp_ratio
        self.ori_mask_mode = ori_mask_mode
        self.cmp_mask_mode = cmp_mask_mode
        self.ori_win_left = ori_win_left
        self.ori_win_right = ori_win_right
        self.template_run_mode = template_run_mode

    def calculate_by_bnsd(
        self,
        q_bnsd,
        ori_k_bnsd,
        cmp_k_bnsd,
        ori_sparse_indices_bnsd,
        cmp_sparse_indices_bnsd,
        cu_seqlens_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        sinks,
        ori_topk_length_bnsd,
        cmp_topk_length_bnsd,
        return_softmax_lse=False,
    ):
        attn_out = torch.zeros(q_bnsd.shape, dtype=q_bnsd.dtype)
        softmax_lse = None
        if return_softmax_lse:
            softmax_lse = torch.zeros(
                (
                    q_bnsd.shape[0],
                    ori_k_bnsd.shape[1],
                    q_bnsd.shape[2],
                    q_bnsd.shape[1] // ori_k_bnsd.shape[1],
                ),
                dtype=torch.float32,
            )
        B = q_bnsd.shape[0]
        act_q = self.seqused_q
        G = int(self.N1 / self.N2)
        s2_base_size = 128

        for i_B in range(B):
            logging.info(f"i_B = {i_B}/{B}")
            cur_act_q = act_q[i_B]
            cur_ori_act_kv = seqused_ori_kv[i_B]
            cur_cmp_act_kv = seqused_cmp_kv[i_B] if seqused_cmp_kv is not None else 0
            cur_cmp_residual = (
                cmp_residual_kv[i_B] if cmp_residual_kv is not None else 0
            )
            cur_cmp_restored = cur_cmp_act_kv * self.cmp_ratio + cur_cmp_residual

            for i_N2 in range(self.N2):
                logging.info(f"    i_N2 = {i_N2}/{self.N2}")
                cur_sinks = (
                    sinks[i_N2 * G : (i_N2 + 1) * G] if sinks is not None else None
                )
                for i_S1 in range(cur_act_q):
                    milestones = [
                        int(cur_act_q * pct / 100) for pct in range(10, 101, 10)
                    ]
                    milestones = list(dict.fromkeys(milestones))
                    if i_S1 in milestones:
                        current_pct = (i_S1 / cur_act_q) * 100
                        logging.info(
                            f"      进度：{current_pct:.1f}% | 步数：{i_S1:>{len(str(cur_act_q))}}/{cur_act_q}"
                        )
                    if self.ori_mask_mode == 0:
                        ori_win_start = 0
                        ori_win_end = cur_ori_act_kv
                    elif self.ori_mask_mode == 3:
                        ori_win_start = 0
                        ori_win_end = min(
                            max(cur_ori_act_kv - cur_act_q + i_S1 + 1, 0),
                            cur_ori_act_kv,
                        )
                    elif self.ori_mask_mode == 4:
                        ori_threshold = cur_ori_act_kv - cur_act_q + i_S1 + 1
                        if self.ori_win_left == -1:
                            ori_win_start = 0
                        else:
                            ori_win_start = max(
                                ori_threshold - self.ori_win_left - 1, 0
                            )
                        if self.ori_win_right == -1:
                            ori_win_end = cur_ori_act_kv
                        else:
                            ori_win_end = min(
                                max(ori_threshold + self.ori_win_right, 0),
                                cur_ori_act_kv,
                            )

                    if ori_win_start >= ori_win_end:
                        cur_ori_k_bnsd = torch.zeros(
                            [0, self.D], dtype=ori_k_bnsd.dtype
                        )
                    else:
                        cur_ori_k_bnsd = ori_k_bnsd[
                            i_B, i_N2, ori_win_start:ori_win_end, :
                        ]

                    if (
                        self.template_run_mode == "CSA"
                        and cmp_sparse_indices_bnsd is not None
                    ):
                        topk_id = cmp_sparse_indices_bnsd[i_B, i_N2, i_S1, :]
                        _, cur_cmp_k = self.gather_cmp_kv(
                            cmp_k_bnsd,
                            topk_id,
                            i_B,
                            i_N2,
                            i_S1,
                            cur_cmp_restored,
                            cur_act_q,
                            cmp_topk_length_bnsd=cmp_topk_length_bnsd,
                        )
                    elif self.template_run_mode == "HCA":
                        _, cur_cmp_k = self.mask_cmp_kv(
                            cmp_k_bnsd, i_B, i_N2, i_S1, cur_cmp_restored, cur_act_q
                        )
                    elif (
                        self.template_run_mode == "ORI_SPARSE"
                        and ori_sparse_indices_bnsd is not None
                    ):
                        topk_id = ori_sparse_indices_bnsd[i_B, i_N2, i_S1, :]
                        empty_flag, cur_ori_k = self.gather_ori_kv(
                            ori_k_bnsd,
                            topk_id,
                            i_B,
                            i_N2,
                            i_S1,
                            cur_ori_act_kv,
                            cur_act_q,
                            ori_topk_length_bnsd,
                        )
                        if empty_flag is not True:
                            cur_ori_k_bnsd = cur_ori_k
                        cur_cmp_k = []
                        empty_flag = True
                    elif (
                        self.template_run_mode == "ORI_CMP_SPARSE"
                        and ori_sparse_indices_bnsd is not None
                        and cmp_sparse_indices_bnsd is not None
                    ):
                        ori_topk_id = ori_sparse_indices_bnsd[i_B, i_N2, i_S1, :]
                        ori_empty_flag, cur_ori_k = self.gather_ori_kv(
                            ori_k_bnsd,
                            ori_topk_id,
                            i_B,
                            i_N2,
                            i_S1,
                            cur_ori_act_kv,
                            cur_act_q,
                            ori_topk_length_bnsd,
                        )
                        cmp_topk_id = cmp_sparse_indices_bnsd[i_B, i_N2, i_S1, :]
                        cmp_empty_flag, cur_cmp_k = self.gather_cmp_kv(
                            cmp_k_bnsd,
                            cmp_topk_id,
                            i_B,
                            i_N2,
                            i_S1,
                            cur_cmp_restored,
                            cur_act_q,
                            cmp_topk_length_bnsd=cmp_topk_length_bnsd,
                        )
                        if ori_empty_flag is not True:
                            cur_ori_k_bnsd = cur_ori_k
                        if cmp_empty_flag is True:
                            cur_cmp_k = []
                        empty_flag = cmp_empty_flag
                    else:
                        empty_flag = True
                        cur_cmp_k = []
                    if cur_cmp_k == []:
                        cmp_s2_loop_time = 0
                        cur_cmp_k_fp32 = []
                    else:
                        cmp_s2_loop_time = math.ceil(cur_cmp_k.size(0) / s2_base_size)
                        cur_cmp_k_fp32 = cur_cmp_k.to(dtype=torch.float32)

                    if cur_ori_k_bnsd.size(0) == 0 and cur_cmp_k == []:
                        attn_out[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :] = torch.zeros(
                            [G, self.D], dtype=torch.float
                        )
                        continue

                    q_curr = q_bnsd[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :]
                    q_curr_fp32 = q_curr.to(dtype=torch.float32)
                    ori_s2_loop_time = math.ceil(cur_ori_k_bnsd.size(0) / s2_base_size)
                    total_s2_loop_time = ori_s2_loop_time + cmp_s2_loop_time
                    cur_ori_k_bnsd_fp32 = cur_ori_k_bnsd.to(dtype=torch.float32)
                    # hifp8FullQuant: online softmax with hif8 quantized attention scores
                    hifp8_scale_value = 16.0
                    score_max_pre = torch.ones((G,)).to(torch.float) * (-torch.inf)
                    score_max = (
                        cur_sinks.clone()
                        if cur_sinks is not None
                        else torch.ones((G,)).to(torch.float) * (-torch.inf)
                    )
                    score_max = (
                        cur_sinks.clone()
                        if cur_sinks is not None
                        else torch.ones((G,)).to(torch.float) * (-torch.inf)
                    )
                    acc_o = torch.zeros((G, self.D)).to(torch.float32)
                    sumexp = torch.ones((G,)).to(torch.float32)

                    for i_S2 in range(total_s2_loop_time):
                        if i_S2 < ori_s2_loop_time:  # ori_kv
                            if i_S2 < ori_s2_loop_time - 1:
                                k_tile = cur_ori_k_bnsd_fp32[
                                    i_S2 * s2_base_size : (i_S2 + 1) * s2_base_size, :
                                ]
                            else:
                                k_tile = cur_ori_k_bnsd_fp32[i_S2 * s2_base_size :, :]
                        else:  # cmp_kv
                            if i_S2 < total_s2_loop_time - 1:
                                k_tile = cur_cmp_k_fp32[
                                    (i_S2 - ori_s2_loop_time) * s2_base_size : (
                                        i_S2 - ori_s2_loop_time + 1
                                    )
                                    * s2_base_size,
                                    :,
                                ]
                            else:
                                k_tile = cur_cmp_k_fp32[
                                    (i_S2 - ori_s2_loop_time) * s2_base_size :, :
                                ]
                        v_tile = k_tile.clone()
                        # MM1
                        mm1_res = torch.matmul(q_curr_fp32, k_tile.T)
                        # scale过程与NPU一致 保证精度统一
                        is_cmp_tile = i_S2 >= ori_s2_loop_time
                        if is_cmp_tile:
                            combined_scale = (
                                self.softmax_scale
                                * self.q_descale_val
                                * self.cmp_kv_descale_val
                            )
                            cur_v_descale = self.cmp_kv_descale_val
                        else:
                            combined_scale = (
                                self.softmax_scale
                                * self.q_descale_val
                                * self.ori_kv_descale_val
                            )
                            cur_v_descale = self.ori_kv_descale_val
                        scale_res = mm1_res * combined_scale

                        # 更新 score_max
                        score_max_pre = score_max.clone()
                        cur_score_max = scale_res.max(dim=-1)[0]
                        score_max = torch.max(score_max, cur_score_max)
                        score_max_pre = score_max_pre - score_max
                        score_max_pre = torch.exp(score_max_pre)

                        # 计算 acc_s 并做 hif8 量化
                        acc_s = torch.exp(scale_res - score_max.unsqueeze(1))
                        sumexp_i = acc_s.sum(dim=-1)
                        sumexp = sumexp * score_max_pre + sumexp_i

                        acc_s_cast = acc_s * hifp8_scale_value
                        acc_s_cast = trans_float_tensor_to_hifuint8(acc_s_cast)
                        acc_s_cast = trans_hifuint8_tensor_to_float(acc_s_cast)

                        # MM2
                        mm2_res = torch.matmul(acc_s_cast, v_tile)
                        mm2_res = mm2_res * cur_v_descale
                        acc_o = acc_o * score_max_pre.unsqueeze(1) + mm2_res

                    acc_o = torch.div(acc_o, sumexp.unsqueeze(1))
                    acc_o = acc_o / hifp8_scale_value
                    attn_out[i_B, i_N2 * G : (i_N2 + 1) * G, i_S1, :] = acc_o.to(
                        torch.bfloat16
                    )
                    if return_softmax_lse:
                        softmax_lse[i_B, i_N2, i_S1, :] = score_max + torch.log(
                            sumexp + 1e-10
                        )
        return attn_out, softmax_lse

    def gather_ori_kv(
        self,
        k_tensor,
        topk_id,
        i_B,
        i_N2,
        i_S1,
        cur_act_kv,
        cur_act_q,
        ori_topk_length_bnsd,
        sparse_block_size=1,
    ):
        s2_sparse = list()
        ori_threshold = cur_act_kv - cur_act_q + i_S1 + 1
        left_bound = 0
        right_bound = cur_act_kv
        if self.ori_mask_mode == 3:
            left_bound = 0
            right_bound = min(max(ori_threshold, 0), cur_act_kv)
        elif self.ori_mask_mode == 4:
            if self.ori_win_left == -1:
                left_bound = 0
            else:
                left_bound = max(ori_threshold - self.ori_win_left - 1, 0)
            if self.ori_win_right == -1:
                right_bound = cur_act_kv
            else:
                right_bound = min(
                    max(ori_threshold + self.ori_win_right, 0), cur_act_kv
                )
        elif self.ori_mask_mode == 0:
            pass
        left_bound = min(left_bound, cur_act_kv)

        valid_count = min(topk_id.shape[0], right_bound)
        if ori_topk_length_bnsd is not None:
            valid_count = min(int(ori_topk_length_bnsd[i_B, 0, i_S1, 0]), valid_count)

        for i_valid in range(valid_count):
            cur_topk_id = topk_id[i_valid]
            if cur_topk_id == -1:
                break
            begin_idx = cur_topk_id * sparse_block_size
            end_idx = min(begin_idx + sparse_block_size, cur_act_kv)
            if begin_idx >= right_bound:
                continue
            if begin_idx < left_bound:
                continue
            if end_idx <= right_bound:
                s2_sparse.extend(np.arange(begin_idx, end_idx))
            else:
                s2_sparse.extend(np.arange(begin_idx, right_bound))
        empty_flag = len(s2_sparse) == 0
        k_sparse = k_tensor[i_B, i_N2, s2_sparse, :] if not empty_flag else []
        return empty_flag, k_sparse

    def gather_cmp_kv(
        self,
        k_tensor,
        topk_id,
        i_B,
        i_N2,
        i_S1,
        cur_act_kv,
        cur_act_q,
        cmp_topk_length_bnsd=None,
        sparse_block_size=1,
    ):
        s2_sparse = list()
        cur_cmp_act_kv = math.floor(cur_act_kv / self.cmp_ratio)
        threshold = 0
        if self.cmp_mask_mode == 3:
            threshold = math.floor((cur_act_kv - cur_act_q + i_S1 + 1) / self.cmp_ratio)
        elif self.cmp_mask_mode == 0:
            threshold = cur_cmp_act_kv
        if cmp_topk_length_bnsd is not None:
            valid_count = min(
                int(cmp_topk_length_bnsd[i_B, 0, i_S1, 0]),
                topk_id.shape[0],
                math.ceil(threshold / sparse_block_size),
            )
        else:
            valid_count = min(self.K, math.ceil(threshold / sparse_block_size))
        for i_valid in range(valid_count):
            cur_topk_id = topk_id[i_valid]

            if cur_topk_id == -1:
                break
            begin_idx = cur_topk_id * sparse_block_size
            end_idx = (
                begin_idx + sparse_block_size
                if begin_idx + sparse_block_size <= cur_cmp_act_kv
                else cur_cmp_act_kv
            )
            if begin_idx >= threshold:
                continue
            if end_idx <= threshold:
                s2_sparse.extend(np.arange(begin_idx, end_idx))
            else:
                s2_sparse.extend(np.arange(begin_idx, threshold))

        empty_flag = False
        if len(s2_sparse) == 0:
            k_sparse = []
            empty_flag = True
        else:
            k_sparse = k_tensor[i_B, i_N2, s2_sparse, :]
        return empty_flag, k_sparse

    def mask_cmp_kv(self, k_tensor, i_B, i_N2, i_S1, cur_act_kv, cur_act_q):
        threshold = 0
        if self.cmp_mask_mode == 3:
            threshold = (cur_act_kv - cur_act_q + i_S1 + 1) // self.cmp_ratio
        elif self.cmp_mask_mode == 0:
            threshold = cur_act_kv // self.cmp_ratio
        empty_flag = True
        k_sparse = []
        if threshold > 0:
            empty_flag = False
            k_sparse = k_tensor[i_B, i_N2, :threshold, :]
        return empty_flag, k_sparse

    def trans_shape_to_bnsd(
        self, tensor, shape, layout, cu_seqlens_q=None, seqused_q=None
    ):
        if layout in ["BSND"]:
            B = shape[0]
            S = shape[1]
            N = shape[2]
            D = shape[3]
            tensor = tensor.permute(0, 2, 1, 3)
            return tensor, [B, N, S, D]
        elif layout in ["TND"]:
            N = shape[1]
            D = shape[2]
            B = len(cu_seqlens_q) - 1
            max_s1 = get_max_adjacent_diff(cu_seqlens_q)
            seqused_per_batch = (
                seqused_q
                if seqused_q is not None
                else prefix_sum_to_original(cu_seqlens_q)
            )
            new_tensor = torch.zeros((B, N, max_s1, D), dtype=tensor.dtype)
            for b_index in range(B):
                t_start = int(cu_seqlens_q[b_index])
                cur_seqused = int(seqused_per_batch[b_index])
                if cur_seqused == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:cur_seqused, :] = tensor[
                        t_start : t_start + cur_seqused, n_index, :
                    ]
            return new_tensor, [B, N, max_s1, D]
        else:
            return tensor, shape

    def trans_topk_length_shape_to_bnsd(
        self, tensor, shape, layout, cu_seqlens_q=None, seqused_q=None
    ):
        if layout in ["BSND"]:
            B = shape[0]
            S = shape[1]
            tensor = tensor.reshape(B, 1, S, 1)
            return tensor, [B, 1, S, 1]
        elif layout in ["TND"]:
            T = shape[0]
            B = len(cu_seqlens_q) - 1
            max_s1 = get_max_adjacent_diff(cu_seqlens_q)
            seqused_per_batch = (
                seqused_q
                if seqused_q is not None
                else prefix_sum_to_original(cu_seqlens_q)
            )
            new_tensor = torch.zeros((B, 1, max_s1, 1), dtype=tensor.dtype)
            for b_index in range(B):
                t_start = int(cu_seqlens_q[b_index])
                cur_seqused = int(seqused_per_batch[b_index])
                if cur_seqused == 0:
                    continue
                new_tensor[b_index, 0, 0:cur_seqused, :] = tensor[
                    t_start : t_start + cur_seqused, :
                ]
            return new_tensor, [B, 1, max_s1, 1]

    def trans_bnsd_to_target_layout(
        self, tensor, layout, cu_seqlens_q=None, seqused_q=None
    ):
        if layout in ["BSND"]:
            output = tensor.permute(0, 2, 1, 3).contiguous()
            return output
        elif layout in ["TND"]:
            T = cu_seqlens_q[-1]
            B = tensor.shape[0]
            N = tensor.shape[1]
            D = tensor.shape[3]
            seqused_per_batch = (
                seqused_q
                if seqused_q is not None
                else prefix_sum_to_original(cu_seqlens_q)
            )
            output = torch.zeros((T, N, D), dtype=torch.float)
            for b_index in range(B):
                t_start = int(cu_seqlens_q[b_index])
                cur_seqused = int(seqused_per_batch[b_index])
                if cur_seqused == 0:
                    continue
                for n_index in range(N):
                    output[t_start : t_start + cur_seqused, n_index, :] = tensor[
                        b_index, n_index, :cur_seqused, :
                    ]
            return output
        else:
            return tensor

    def lse_trans_bnsd_to_target_layout(self, tensor, layout, act_seq=None):
        if layout in ["BSND"]:  # B N S G
            return tensor
        elif layout in ["TND"]:  # B N2 S1 G  --> T1 N2 G --> N2 T1 G
            T = act_seq[-1]
            B = tensor.shape[0]
            N2 = tensor.shape[1]
            G = tensor.shape[3]
            output = torch.zeros((N2, T, G), dtype=torch.float)
            t_start = 0
            act_seq_per_batch = prefix_sum_to_original(
                act_seq
            )  # prefix_sum_to_original 还原成每个batch的真实长度
            for b_index in range(B):
                cur_act_seq = act_seq_per_batch[b_index]
                t_end = t_start + cur_act_seq
                if cur_act_seq == 0:
                    continue
                for n_index in range(N2):
                    output[n_index, t_start:t_end, :] = tensor[
                        b_index, n_index, :cur_act_seq, :
                    ]
                t_start += cur_act_seq
            return output
        else:
            return tensor

    def forward(
        self,
        q,
        ori_k_bnsd,
        cmp_k_bnsd,
        ori_sparse_indices,
        cmp_sparse_indices,
        cu_seqlens_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        sinks,
        ori_topk_length,
        cmp_topk_length,
        return_softmax_lse,
    ):
        logging.info("cpu执行中...")
        logging.info(f"template_run_mode = {self.template_run_mode}")

        q_bnsd, q_bnsd_shape = self.trans_shape_to_bnsd(
            q, q.shape, self.layout_q, cu_seqlens_q, self.seqused_q
        )

        ori_sparse_indices_bnsd = None
        if (
            self.template_run_mode in ("ORI_SPARSE", "ORI_CMP_SPARSE")
            and ori_sparse_indices is not None
        ):
            ori_sparse_indices_bnsd, _ = self.trans_shape_to_bnsd(
                ori_sparse_indices,
                ori_sparse_indices.shape,
                self.layout_q,
                cu_seqlens_q,
                self.seqused_q,
            )

        cmp_sparse_indices_bnsd = None
        if (
            self.template_run_mode in ("CSA", "ORI_CMP_SPARSE")
            and cmp_sparse_indices is not None
        ):
            cmp_sparse_indices_bnsd, cmp_sparse_indices_bnsd_shape = (
                self.trans_shape_to_bnsd(
                    cmp_sparse_indices,
                    cmp_sparse_indices.shape,
                    self.layout_q,
                    cu_seqlens_q,
                    self.seqused_q,
                )
            )

        ori_topk_length_bnsd = None
        if (
            self.template_run_mode in ("ORI_SPARSE", "ORI_CMP_SPARSE")
            and ori_topk_length is not None
        ):
            ori_topk_length_bnsd, _ = self.trans_topk_length_shape_to_bnsd(
                ori_topk_length,
                ori_topk_length.shape,
                self.layout_q,
                cu_seqlens_q,
                self.seqused_q,
            )

        cmp_topk_length_bnsd = None
        if (
            self.template_run_mode in ("ORI_SPARSE", "ORI_CMP_SPARSE", "CSA")
            and cmp_topk_length is not None
        ):
            cmp_topk_length_bnsd, _ = self.trans_topk_length_shape_to_bnsd(
                cmp_topk_length,
                cmp_topk_length.shape,
                self.layout_q,
                cu_seqlens_q,
                self.seqused_q,
            )

        attn_out, softmax_lse = self.calculate_by_bnsd(
            q_bnsd,
            ori_k_bnsd,
            cmp_k_bnsd,
            ori_sparse_indices_bnsd,
            cmp_sparse_indices_bnsd,
            cu_seqlens_q,
            seqused_ori_kv,
            seqused_cmp_kv,
            cmp_residual_kv,
            sinks,
            ori_topk_length_bnsd,
            cmp_topk_length_bnsd,
            return_softmax_lse,
        )

        attn_out = self.trans_bnsd_to_target_layout(
            attn_out, self.layout_q, cu_seqlens_q, self.seqused_q
        )
        if return_softmax_lse:
            softmax_lse = self.lse_trans_bnsd_to_target_layout(
                softmax_lse, self.layout_q, cu_seqlens_q
            )
        return attn_out, softmax_lse


def prefix_sum_to_original(cu_seqlens_q):
    """
    从前缀和张量反向计算出原始的非前缀和张量（替代原列表逻辑）

    Args:
        cu_seqlens_q (torch.Tensor): 形状为 [B+1] 的一维前缀和张量（元素为数字类型，如int/float）

    Returns:
        torch.Tensor: 原始的非前缀和张量，形状为 [B]（与原列表长度一致）

    Raises:
        TypeError: 输入非tensor/非一维tensor
        ValueError: tensor长度<2（无法计算差值）
    """
    # 1. 基础类型校验：必须是torch.Tensor
    if not isinstance(cu_seqlens_q, torch.Tensor):
        raise TypeError(f"输入必须是torch.Tensor，当前类型：{type(cu_seqlens_q)}")

    # 2. 维度校验：必须是一维tensor（原列表对应一维）
    if cu_seqlens_q.ndim != 1:
        raise TypeError(
            f"输入必须是一维tensor，当前维度：{cu_seqlens_q.ndim}，形状：{cu_seqlens_q.shape}"
        )

    # 3. 长度校验（前缀和tensor至少需2个元素才能反向计算）
    if len(cu_seqlens_q) < 2:
        raise ValueError(f"前缀和tensor长度需≥2，当前长度：{len(cu_seqlens_q)}")

    # 4. 核心逻辑：计算相邻元素差值（用tensor向量化运算替代循环，效率更高）
    # 原理：original_val[i] = cu_seqlens_q[i+1] - cu_seqlens_q[i]
    # 切片实现：cu_seqlens_q[1:] 取第2个到最后一个元素，cu_seqlens_q[:-1] 取第1个到倒数第2个元素
    original_tensor = cu_seqlens_q[1:] - cu_seqlens_q[:-1]

    return original_tensor


def get_max_adjacent_diff(cu_seqlens_q):
    """
    计算前缀和列表中相邻元素（后-前）的最大差值

    Args:
        cu_seqlens_q (list): 长度为 B+1 的前缀和列表

    Returns:
        float/int: 相邻元素的最大差值；若列表长度<2，返回 None
    """
    # 边界检查：列表长度不足2时无相邻元素
    if len(cu_seqlens_q) < 2:
        return None

    # 初始化最大差值为第一个相邻对的差值
    max_diff = cu_seqlens_q[1] - cu_seqlens_q[0]

    # 遍历所有相邻元素对（从第2对开始）
    for i in range(1, len(cu_seqlens_q) - 1):
        current_diff = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
        # 更新最大差值
        if current_diff > max_diff:
            max_diff = current_diff

    return max_diff


def gen_sparse_indices_bsnd(
    cmp_ratio,
    B,
    S1,
    N2,
    K,
    seqused_q,
    seqused_kv,
    mask_mode,
    sparse_indices_mode,
    kv_topk_mode,
    topk_length_override=None,
    ori_win_left=-1,
    ori_win_right=-1,
):
    if mask_mode != 0:
        kv_topk_mode = "no"  # mask_mode != 0时，kv_topk_mode只能为no
    if sparse_indices_mode is None:
        sparse_indices_mode = "full"
    if sparse_indices_mode not in ["full", "random"]:
        raise ValueError(
            f"sparse_indices_mode only support full/random, which is {sparse_indices_mode}"
        )
    if kv_topk_mode not in ["fullK", "random", "no"]:
        raise ValueError(
            f"kv_topk_mode only support fullK, random, which is {kv_topk_mode}"
        )

    # 有效索引在叠加了causal后有效tokens中选取，不足sparse_block_count，尾部填充-1
    sparse_data = torch.full((B, S1, N2, K), fill_value=-1, dtype=torch.int32)

    if topk_length_override is not None:
        topk_length = topk_length_override
    elif kv_topk_mode != "no":
        topk_length = torch.zeros((B, S1, N2), dtype=torch.int32)
    else:
        topk_length = None

    for i_B in range(B):
        cur_act_q = seqused_q[i_B]
        cur_act_kv = seqused_kv[i_B]
        for i_N2 in range(N2):
            for i_S1 in range(cur_act_q):
                cur_valid_left = 0
                if mask_mode == 3:
                    cur_valid_s2_max = math.floor(
                        (cur_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio
                    )
                elif mask_mode == 0:
                    cur_valid_s2_max = math.floor(cur_act_kv / cmp_ratio)
                elif mask_mode == 4:
                    ori_threshold = cur_act_kv - cur_act_q + i_S1 + 1
                    if ori_win_left == -1:
                        left_bound = 0
                    else:
                        left_bound = max(ori_threshold - ori_win_left - 1, 0)
                    if ori_win_right == -1:
                        right_bound = cur_act_kv
                    else:
                        right_bound = min(
                            max(ori_threshold + ori_win_right, 0), cur_act_kv
                        )
                    left_bound = min(left_bound, cur_act_kv)
                    cur_valid_left = max(0, left_bound)
                    cur_valid_s2_max = max(0, right_bound - cur_valid_left)
                else:
                    raise ValueError(
                        f"topklen sparse mask mode only support 0/3/4, which is {mask_mode}"
                    )
                cur_valid_s2_max = max(0, cur_valid_s2_max)

                # gen sparse indices
                if sparse_indices_mode == "random":
                    if cur_valid_s2_max > 1:
                        cur_valid_s2_max_update = torch.randint(
                            1, cur_valid_s2_max, (1, 1)
                        )[0]
                    else:
                        cur_valid_s2_max_update = 1
                else:
                    cur_valid_s2_max_update = cur_valid_s2_max

                valid_blocks_max = max(0, cur_valid_s2_max_update)
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                valid_blocks_topk = min(valid_blocks_max, K)
                sparse_data[i_B, i_S1, i_N2, :valid_blocks_topk] = (
                    block_indices[0:valid_blocks_topk] + cur_valid_left
                )

                # gen topk length
                if topk_length is not None and topk_length_override is None:
                    if kv_topk_mode == "fullK":
                        topk_length[i_B, i_S1, i_N2] = min(cur_valid_s2_max_update, K)
                    elif kv_topk_mode == "random":
                        # torch.randint范围 [1, K + 1)左开右闭
                        topk_length[i_B, i_S1, i_N2] = min(
                            torch.randint(1, K + 1, (1, 1))[0], cur_valid_s2_max_update
                        )

    return sparse_data, topk_length


def gen_sparse_indices_tnd(
    cmp_ratio,
    B,
    T1,
    N2,
    K,
    cu_seqlens_q,
    seqused_q,
    seqused_ori_kv,
    mask_mode,
    sparse_indices_mode,
    kv_topk_mode,
    topk_length_override=None,
    ori_win_left=-1,
    ori_win_right=-1,
):
    if mask_mode != 0:
        kv_topk_mode = "no"  # mask_mode != 0时，kv_topk_mode只能为no
    if sparse_indices_mode is None:
        sparse_indices_mode = "full"
    if sparse_indices_mode not in ["full", "random"]:
        raise ValueError(
            f"sparse_indices_mode only support full/random, which is {sparse_indices_mode}"
        )
    if kv_topk_mode not in ["fullK", "random", "no"]:
        raise ValueError(
            f"kv_topk_mode only support fullK, random, which is {kv_topk_mode}"
        )

    sparse_data = torch.full((T1, N2, K), fill_value=-1, dtype=torch.int32)

    if topk_length_override is not None:
        topk_length = topk_length_override
    elif kv_topk_mode != "no":
        topk_length = torch.zeros((T1, N2), dtype=torch.int32)
    else:
        topk_length = None

    for i_B in range(B):
        cur_act_q = seqused_q[i_B]
        s1_prefix = cu_seqlens_q[i_B]
        cur_act_kv = seqused_ori_kv[i_B]
        for i_N2 in range(N2):
            for i_S1 in range(cur_act_q):
                cur_valid_left = 0
                if mask_mode == 3:
                    cur_valid_s2_max = math.floor(
                        (cur_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio
                    )
                elif mask_mode == 0:
                    cur_valid_s2_max = math.floor(cur_act_kv / cmp_ratio)
                elif mask_mode == 4:
                    ori_threshold = cur_act_kv - cur_act_q + i_S1 + 1
                    if ori_win_left == -1:
                        left_bound = 0
                    else:
                        left_bound = max(ori_threshold - ori_win_left - 1, 0)
                    if ori_win_right == -1:
                        right_bound = cur_act_kv
                    else:
                        right_bound = min(
                            max(ori_threshold + ori_win_right, 0), cur_act_kv
                        )
                    left_bound = min(left_bound, cur_act_kv)
                    cur_valid_left = max(0, left_bound)
                    cur_valid_s2_max = max(0, right_bound - cur_valid_left)
                else:
                    raise ValueError(
                        f"ori_mask_mode only support 0/3/4, which is {mask_mode}"
                    )
                cur_valid_s2_max = max(0, cur_valid_s2_max)

                # gen sparse indices
                if sparse_indices_mode == "random":
                    if cur_valid_s2_max > 1:
                        cur_valid_s2_max_update = torch.randint(
                            1, cur_valid_s2_max, (1, 1)
                        )[0]
                    else:
                        cur_valid_s2_max_update = 1
                else:
                    cur_valid_s2_max_update = cur_valid_s2_max

                valid_blocks_max = max(0, cur_valid_s2_max_update)
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                valid_blocks_topk = min(valid_blocks_max, K)
                sparse_data[s1_prefix + i_S1, i_N2, :valid_blocks_topk] = (
                    block_indices[0:valid_blocks_topk] + cur_valid_left
                )

                # gen topk length
                if topk_length is not None and topk_length_override is None:
                    if kv_topk_mode == "fullK":
                        topk_length[s1_prefix + i_S1, i_N2] = min(
                            cur_valid_s2_max_update, K
                        )
                    elif kv_topk_mode == "random":
                        # torch.randint范围 [1, K + 1)左开右闭
                        topk_length[s1_prefix + i_S1, i_N2] = min(
                            torch.randint(1, K + 1, (1, 1))[0], cur_valid_s2_max_update
                        )

    return sparse_data, topk_length


def trans_kv_bnsd_to_tnd(kv_bnsd_npu, cu_seqlens_kv, seqused_kv, B, N2, D, kv_type):
    total_t = int(cu_seqlens_kv[-1])
    kv_tnd = torch.zeros((total_t, N2, D), dtype=kv_type)
    for i_B in range(B):
        t_start = int(cu_seqlens_kv[i_B])
        cur_s = int(seqused_kv[i_B])
        if cur_s > 0:
            kv_tnd[t_start : t_start + cur_s, :, :] = kv_bnsd_npu[
                i_B, :, :cur_s, :
            ].permute(1, 0, 2)
    return kv_tnd


def _gen_hif8_tensor(shape, data_range, scale_range):
    scale = torch.tensor(
        [random.uniform(*scale_range)],
        dtype=torch.float32,
    )

    x = (
        torch.rand(shape, dtype=torch.float32) * (data_range[1] - data_range[0])
        + data_range[0]
    )

    x_uint8 = trans_float_tensor_to_hifuint8(x)
    x_fp32 = torch.tensor(trans_hifuint8_tensor_to_float(x_uint8))
    x_uint8 = torch.tensor(x_uint8)

    return x_fp32, x_uint8, scale


def _build_block_table_and_pa(
    k_bnsd_npu,
    kv_type,
    B,
    N2,
    max_s2,
    max_block_num_per_batch,
    block_num,
    block_size,
    seqused_kv,
):
    """构建 block_table 和 PA 物理布局"""
    block_num_per_batch = [math.ceil(s / block_size) for s in seqused_kv]
    total_needed = sum(block_num_per_batch)

    if block_num < total_needed:
        raise ValueError(
            f"kv actual_block_num < needed_block_num, which is {block_num} < {total_needed}"
        )

    D = k_bnsd_npu.shape[-1]
    block_id_list = np.random.permutation(block_num).astype(np.int32)
    block_table = np.full((B, max_block_num_per_batch), fill_value=-1, dtype=np.int32)

    cur_id = 0
    for batch_idx, num_blocks in enumerate(block_num_per_batch):
        block_table[batch_idx, :num_blocks] = block_id_list[
            cur_id : cur_id + num_blocks
        ]
        cur_id += num_blocks

    # 展开到 PA 物理布局
    k_expand = torch.zeros(
        (B, N2, max_block_num_per_batch * block_size, D), dtype=kv_type
    )
    k_expand[:, :, :max_s2, :] = k_bnsd_npu
    k_in_pa_shape = torch.zeros((block_num, block_size, N2, D), dtype=kv_type)

    for i_B in range(B):
        for i_block, bid in enumerate(block_table[i_B]):
            if bid == -1:
                continue
            start = i_block * block_size
            k_in_pa_shape[bid, :, :, :] = k_expand[
                i_B, :, start : start + block_size, :
            ].permute(1, 0, 2)

    block_table = torch.tensor(block_table).to(torch.int32)
    return k_in_pa_shape, block_table


def gen_ori_kv(
    ori_kv_type,
    B,
    S1,
    T1,
    N2,
    D,
    block_num1,
    block_size1,
    ori_max_s2,
    ori_max_block_num_per_batch,
    seqused_ori_kv,
    cu_seqlens_ori_kv,
    layout_kv="PA_BBND",
    ori_kv_datarange=[-2, 2],
    scale_datarange=[-1, 1],
    K1=None,
    template_run_mode=None,
    layout_q=None,
    cu_seqlens_q=None,
    seqused_q=None,
    ori_mask_mode=0,
    ori_sparse_indices_mode="full",
    ori_kv_topk_mode="no",
    ori_topk_length_override=None,
    ori_win_left=-1,
    ori_win_right=-1,
):
    ori_k_bnsd, ori_k_bnsd_npu, ori_kv_descale = _gen_hif8_tensor(
        (B, N2, ori_max_s2, D), ori_kv_datarange, scale_datarange
    )

    if layout_kv == "TND":
        ori_k_in_pa_shape = trans_kv_bnsd_to_tnd(
            ori_k_bnsd_npu, cu_seqlens_ori_kv, seqused_ori_kv, B, N2, D, ori_kv_type
        )
        ori_block_table = None
    elif layout_kv == "BSND":
        ori_block_table = None
        ori_k_in_pa_shape = ori_k_bnsd_npu.reshape(B, ori_max_s2, N2, D).contiguous()
    else:
        ori_k_in_pa_shape, ori_block_table = _build_block_table_and_pa(
            ori_k_bnsd_npu,
            ori_kv_type,
            B,
            N2,
            ori_max_s2,
            ori_max_block_num_per_batch,
            block_num1,
            block_size1,
            seqused_ori_kv,
        )

    ori_sparse_indices = None
    ori_topk_length = None
    if template_run_mode in ("ORI_SPARSE", "ORI_CMP_SPARSE") and K1 is not None:
        if layout_q == "BSND":
            ori_sparse_indices, ori_topk_length = gen_sparse_indices_bsnd(
                1,
                B,
                S1,
                N2,
                K1,
                seqused_q,
                seqused_ori_kv,
                ori_mask_mode,
                ori_sparse_indices_mode,
                ori_kv_topk_mode,
                ori_topk_length_override,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
            )
        elif layout_q == "TND":
            ori_sparse_indices, ori_topk_length = gen_sparse_indices_tnd(
                1,
                B,
                T1,
                N2,
                K1,
                cu_seqlens_q,
                seqused_q,
                seqused_ori_kv,
                ori_mask_mode,
                ori_sparse_indices_mode,
                ori_kv_topk_mode,
                ori_topk_length_override,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
            )

    return (
        ori_k_bnsd,
        ori_k_in_pa_shape,
        ori_block_table,
        ori_sparse_indices,
        ori_topk_length,
        ori_kv_descale,
    )


def gen_cmp_kv(
    layout_q,
    cmp_kv_type,
    B,
    S1,
    T1,
    N2,
    D,
    K,
    block_num2,
    block_size2,
    cmp_max_s2,
    cmp_max_block_num_per_batch,
    cu_seqlens_q,
    seqused_q,
    seqused_ori_kv,
    seqused_cmp_kv,
    cu_seqlens_cmp_kv,
    cmp_residual_kv,
    cmp_ratio,
    cmp_mask_mode,
    template_run_mode,
    layout_kv="PA_BBND",
    cmp_kv_datarange=[-2, 2],
    scale_datarange=[-1, 1],
    cmp_kv_topk_mode="no",
    cmp_sparse_indices_mode="full",
    cmp_topk_length_override=None,
):
    if cmp_max_s2 == 0:
        return None, None, None, None, None, None

    cmp_k_bnsd, cmp_k_bnsd_npu, cmp_kv_descale = _gen_hif8_tensor(
        (B, N2, cmp_max_s2, D), cmp_kv_datarange, scale_datarange
    )

    if layout_kv == "TND":
        cmp_k_in_pa_shape = trans_kv_bnsd_to_tnd(
            cmp_k_bnsd_npu, cu_seqlens_cmp_kv, seqused_cmp_kv, B, N2, D, cmp_kv_type
        )
        cmp_block_table = None
    elif layout_kv == "BSND":
        cmp_block_table = None
        cmp_k_in_pa_shape = cmp_k_bnsd_npu.reshape(B, cmp_max_s2, N2, D).contiguous()
    else:
        cmp_k_in_pa_shape, cmp_block_table = _build_block_table_and_pa(
            cmp_k_bnsd_npu,
            cmp_kv_type,
            B,
            N2,
            cmp_max_s2,
            cmp_max_block_num_per_batch,
            block_num2,
            block_size2,
            seqused_cmp_kv,
        )

    # generate cmp_sparse_indices
    cmp_sparse_indices = None
    cmp_topk_length = None
    if template_run_mode in ("CSA", "ORI_CMP_SPARSE") and cmp_max_s2 != 0:
        if cmp_residual_kv is not None:
            cmp_restored_len = [
                seqused_cmp_kv[i] * cmp_ratio + cmp_residual_kv[i] for i in range(B)
            ]
        else:
            cmp_restored_len = [seqused_cmp_kv[i] * cmp_ratio for i in range(B)]
        if layout_q == "BSND":
            cmp_sparse_indices, cmp_topk_length = gen_sparse_indices_bsnd(
                cmp_ratio,
                B,
                S1,
                N2,
                K,
                seqused_q,
                cmp_restored_len,
                cmp_mask_mode,
                cmp_sparse_indices_mode,
                cmp_kv_topk_mode,
                cmp_topk_length_override,
            )
        elif layout_q == "TND":
            cmp_sparse_indices, cmp_topk_length = gen_sparse_indices_tnd(
                cmp_ratio,
                B,
                T1,
                N2,
                K,
                cu_seqlens_q,
                seqused_q,
                cmp_restored_len,
                cmp_mask_mode,
                cmp_sparse_indices_mode,
                cmp_kv_topk_mode,
                cmp_topk_length_override,
            )

    return (
        cmp_k_bnsd,
        cmp_k_in_pa_shape,
        cmp_block_table,
        cmp_sparse_indices,
        cmp_topk_length,
        cmp_kv_descale,
    )


def save_test_case(input_data, output_dir):
    """
    保存单条测试用例到文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    case_name = input_data["Testcase_Name"]

    # 生成文件名
    input_filename = f"qsmla_case_{case_name}.pt"
    input_filepath = os.path.join(output_dir, input_filename)

    # 保存数据
    torch.save(input_data, input_filepath)
    logging.info(f"测试用例已保存到: {input_filepath}")

    return input_filepath


def gen_data(params, generate_golden=True):
    """
    生成input param及cpuout
    runNpu: 生成完毕后执行npu计算
    return test_data
    """

    Testcase_Name = params["Testcase_Name"]
    layout_q = params["layout_q"]
    layout_kv = params["layout_kv"]
    q_type = params["q_type"]
    ori_kv_type = params["ori_kv_type"]
    cmp_kv_type = params["cmp_kv_type"]
    B = params["B"]
    S1 = params["S1"]
    T1 = params["T1"]
    N1 = params["N1"]
    N2 = params["N2"]
    D = params["D"]
    K = params["K"]
    block_num1 = params["block_num1"]
    block_num2 = params["block_num2"]
    block_size1 = params["block_size1"]
    block_size2 = params["block_size2"]
    cu_seqlens_q = params["cu_seqlens_q"]
    seqused_q = params["seqused_q"]
    seqused_ori_kv = params["seqused_ori_kv"]
    seqused_cmp_kv = params["seqused_cmp_kv"]
    cu_seqlens_ori_kv = params["cu_seqlens_ori_kv"]
    cu_seqlens_cmp_kv = params["cu_seqlens_cmp_kv"]
    cmp_residual_kv = params["cmp_residual_kv"]
    softmax_scale = params["softmax_scale"]
    cmp_ratio = params["cmp_ratio"]
    ori_mask_mode = params["ori_mask_mode"]
    cmp_mask_mode = params["cmp_mask_mode"]
    ori_win_left = params["ori_win_left"]
    ori_win_right = params["ori_win_right"]
    template_run_mode = params["template_run_mode"]
    topk_value_mode = params.get("topk_value_mode", 1)
    return_softmax_lse = params.get("return_softmax_lse", False)
    quant_mode = params.get("quant_mode", False)
    isSink = params.get("isSink", True)
    return_softmax_lse = params.get("return_softmax_lse", False)
    q_datarange = params.get("q_datarange")
    ori_kv_datarange = params.get("ori_kv_datarange")
    cmp_kv_datarange = params.get("cmp_kv_datarange")
    q_descale_datarange = [
        q_datarange[0] / FP8_DATA_RANGE_LEFT,
        q_datarange[1] / FP8_DATA_RANGE_RIGHT,
    ]
    ori_kv_descale_datarange = [
        ori_kv_datarange[0] / FP8_DATA_RANGE_LEFT,
        ori_kv_datarange[1] / FP8_DATA_RANGE_RIGHT,
    ]
    cmp_kv_descale_datarange = [
        cmp_kv_datarange[0] / FP8_DATA_RANGE_LEFT,
        cmp_kv_datarange[1] / FP8_DATA_RANGE_RIGHT,
    ]
    K1 = params.get("K1")
    ori_kv_topk_mode = params.get("ori_kv_topk_mode", "no")
    cmp_kv_topk_mode = params.get("cmp_kv_topk_mode", "no")
    ori_sparse_indices_mode = params.get("ori_sparse_indices_mode", "full")
    cmp_sparse_indices_mode = params.get("cmp_sparse_indices_mode", "full")
    ori_topk_length_override = params.get("ori_topk_length", None)
    cmp_topk_length_override = params.get("cmp_topk_length", None)

    if seqused_q is None:
        raise ValueError("seqused_q must not be None")
    cu_seqlens_q = torch.tensor(cu_seqlens_q).to(torch.int32)
    seqused_q = torch.tensor(seqused_q).to(torch.int32)
    seqused_ori_kv = torch.tensor(seqused_ori_kv).to(torch.int32)
    seqused_cmp_kv = (
        torch.tensor(seqused_cmp_kv).to(torch.int32)
        if seqused_cmp_kv is not None
        else None
    )

    print(q_datarange, ori_kv_datarange, cmp_kv_datarange)
    print(params)

    assert quant_mode == 1, f"quant_mode only support 1, but got {quant_mode}"
    # convert topk_length override to tensor with proper shape
    if ori_topk_length_override is not None and not isinstance(
        ori_topk_length_override, torch.Tensor
    ):
        if layout_q == "BSND":
            ori_topk_length_override = torch.tensor(
                ori_topk_length_override, dtype=torch.int32
            ).reshape(B, S1, N2)
        elif layout_q == "TND":
            ori_topk_length_override = torch.tensor(
                ori_topk_length_override, dtype=torch.int32
            ).reshape(T1, N2)
    if cmp_topk_length_override is not None and not isinstance(
        cmp_topk_length_override, torch.Tensor
    ):
        if layout_q == "BSND":
            cmp_topk_length_override = torch.tensor(
                cmp_topk_length_override, dtype=torch.int32
            ).reshape(B, S1, N2)
        elif layout_q == "TND":
            cmp_topk_length_override = torch.tensor(
                cmp_topk_length_override, dtype=torch.int32
            ).reshape(T1, N2)
    # generate q (hifp8FullQuant)
    if layout_q == "BSND":
        q, q_npu, q_descale = _gen_hif8_tensor(
            (B, S1, N1, D), q_datarange, q_descale_datarange
        )
    elif layout_q == "TND":
        q, q_npu, q_descale = _gen_hif8_tensor(
            (T1, N1, D), q_datarange, q_descale_datarange
        )
        if len(cu_seqlens_q) != (B + 1):
            raise ValueError(
                f"len(cu_seqlens_q) != B + 1, which is {len(cu_seqlens_q)} != {B + 1}"
            )
    else:
        raise ValueError(f"layout_q is not support {layout_q}")

    if len(seqused_ori_kv) != B:
        raise ValueError(
            f"len(seqused_ori_kv) != B, which is {len(seqused_ori_kv)} != {B}"
        )
    else:
        ori_max_s2 = int(get_max_adjacent_diff(cu_seqlens_ori_kv))
        ori_max_block_num_per_batch = math.ceil(ori_max_s2 / block_size1)

        cmp_max_s2 = (
            int(get_max_adjacent_diff(cu_seqlens_cmp_kv))
            if cu_seqlens_cmp_kv is not None
            else 0
        )
        cmp_max_block_num_per_batch = (
            math.ceil(cmp_max_s2 / block_size2) if cmp_max_s2 > 0 else 0
        )

    # D维度固定512，已对齐128，无需padding
    block_num = block_num1 if block_num1 >= block_num2 else block_num2

    # generate sinks tensor (only when isSink=True)
    if isSink:
        sinks = (
            torch.rand((N1)) * (q_datarange[1] - q_datarange[0]) / 10
            + q_datarange[0] / 10
        ).to(torch.float)
    else:
        sinks = None

    # generate ori_kv tensor
    (
        ori_k_bnsd,
        ori_k_in_pa_shape,
        ori_block_table,
        ori_sparse_indices,
        ori_topk_length,
        ori_kv_descale,
    ) = gen_ori_kv(
        ori_kv_type,
        B,
        S1,
        T1,
        N2,
        D,
        block_num,
        block_size1,
        ori_max_s2,
        ori_max_block_num_per_batch,
        seqused_ori_kv,
        cu_seqlens_ori_kv,
        layout_kv,
        ori_kv_datarange,
        ori_kv_descale_datarange,
        K1=K1,
        template_run_mode=template_run_mode,
        layout_q=layout_q,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        ori_mask_mode=ori_mask_mode,
        ori_sparse_indices_mode=ori_sparse_indices_mode,
        ori_kv_topk_mode=ori_kv_topk_mode,
        ori_topk_length_override=ori_topk_length_override,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
    )

    # generate cmp_kv and sparse_indices
    if template_run_mode in ("HCA", "CSA", "ORI_CMP_SPARSE"):
        (
            cmp_k_bnsd,
            cmp_k_in_pa_shape,
            cmp_block_table,
            cmp_sparse_indices,
            cmp_topk_length,
            cmp_kv_descale,
        ) = gen_cmp_kv(
            layout_q,
            cmp_kv_type,
            B,
            S1,
            T1,
            N2,
            D,
            K,
            block_num,
            block_size2,
            cmp_max_s2,
            cmp_max_block_num_per_batch,
            cu_seqlens_q,
            seqused_q,
            seqused_ori_kv,
            seqused_cmp_kv,
            cu_seqlens_cmp_kv,
            cmp_residual_kv,
            cmp_ratio,
            cmp_mask_mode,
            template_run_mode,
            layout_kv,
            cmp_kv_datarange,
            cmp_kv_descale_datarange,
            cmp_kv_topk_mode=cmp_kv_topk_mode,
            cmp_sparse_indices_mode=cmp_sparse_indices_mode,
            cmp_topk_length_override=cmp_topk_length_override,
        )
    else:
        cmp_k_in_pa_shape = None
        cmp_sparse_indices = None
        cmp_block_table = None
        cmp_k_bnsd = None
        cmp_kv_descale = None
        cmp_topk_length = None

    if cmp_k_bnsd is None:  # 如果cmp_k_bnsd为None
        cmp_mask_mode = 0  # cmp_mask_mode 设置为0，防止拦截
    if (
        layout_kv == "PA_BBND"
        and (template_run_mode in ("HCA", "CSA", "ORI_CMP_SPARSE"))
        and cmp_k_in_pa_shape is not None
    ):
        total_block = block_size1 + block_size2
        fusion_base = torch.zeros((block_num, total_block, N2, D), dtype=ori_kv_type)
        fusion_base[:, :block_size1, :, :] = ori_k_in_pa_shape
        fusion_base[:, block_size1:, :, :] = cmp_k_in_pa_shape
        stride_n = total_block * N2 * D
        stride_bs = N2 * D
        stride_n2 = D
        stride_d = 1
        ori_k_in_pa_shape = torch.as_strided(
            fusion_base,
            size=[block_num, block_size1, N2, D],
            stride=[stride_n, stride_bs, stride_n2, stride_d],
        )
        cmp_k_in_pa_shape = torch.as_strided(
            fusion_base,
            size=[block_num, block_size2, N2, D],
            stride=[stride_n, stride_bs, stride_n2, stride_d],
            storage_offset=block_size1 * N2 * D,
        )

    golden_state = {
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "q_type": q_type,
        "ori_kv_type": ori_kv_type,
        "cmp_kv_type": cmp_kv_type,
        "B": B,
        "S1": S1,
        "T1": T1,
        "N1": N1,
        "N2": N2,
        "D": D,
        "K": K,
        "block_num1": block_num1,
        "block_num2": block_num2,
        "block_size1": block_size1,
        "block_size2": block_size2,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_q": seqused_q,
        "seqused_ori_kv": seqused_ori_kv,
        "seqused_cmp_kv": seqused_cmp_kv,
        "cu_seqlens_ori_kv": cu_seqlens_ori_kv,
        "cu_seqlens_cmp_kv": cu_seqlens_cmp_kv,
        "cmp_residual_kv": cmp_residual_kv,
        "softmax_scale": softmax_scale,
        "cmp_ratio": cmp_ratio,
        "ori_mask_mode": ori_mask_mode,
        "cmp_mask_mode": cmp_mask_mode,
        "ori_win_left": ori_win_left,
        "ori_win_right": ori_win_right,
        "ori_topk_length": ori_topk_length,
        "cmp_topk_length": cmp_topk_length,
        "template_run_mode": template_run_mode,
        "q_descale_val": q_descale.item(),
        "ori_kv_descale_val": ori_kv_descale.item(),
        "cmp_kv_descale_val": (
            cmp_kv_descale.item() if cmp_kv_descale is not None else None
        ),
        "q": q,
        "ori_k_bnsd": ori_k_bnsd,
        "cmp_k_bnsd": cmp_k_bnsd,
        "ori_sparse_indices": ori_sparse_indices,
        "cmp_sparse_indices": cmp_sparse_indices,
        "sinks": sinks,
        "return_softmax_lse": return_softmax_lse,
    }
    if generate_golden:
        generate_cpu_golden({"golden_state": golden_state})
        cpu_result = golden_state["cpu_output"]
        cpu_lse = golden_state["cpu_lse"]
    else:
        cpu_result = None
        cpu_lse = None

    logging.info("mode:%s\n", template_run_mode)

    cu_seqlens_q = torch.tensor(cu_seqlens_q).to(torch.int32)
    seqused_ori_kv = torch.tensor(seqused_ori_kv).to(torch.int32)
    seqused_cmp_kv = (
        torch.tensor(seqused_cmp_kv).to(torch.int32)
        if seqused_cmp_kv is not None
        else None
    )
    cu_seqlens_ori_kv = (
        torch.tensor(cu_seqlens_ori_kv).to(torch.int32)
        if cu_seqlens_ori_kv is not None
        else None
    )
    cu_seqlens_cmp_kv = (
        torch.tensor(cu_seqlens_cmp_kv).to(torch.int32)
        if cu_seqlens_cmp_kv is not None
        else None
    )
    cmp_residual_kv = (
        torch.tensor(cmp_residual_kv).to(torch.int32)
        if cmp_residual_kv is not None
        else None
    )
    max_seqlen_q = S1
    if layout_q == "TND":
        max_seqlen_q = cu_seqlens_q.max().item()
    else:
        cu_seqlens_q = None
        max_seqlen_q = S1
    max_seqlen_ori_kv = seqused_ori_kv.max().item()
    max_seqlen_cmp_kv = seqused_cmp_kv.max().item() if seqused_cmp_kv is not None else 0

    # Sparse templates derive KV lengths from sparse indices and top-k lengths.
    # Keep the derived values for metadata and CPU Golden, but do not expose them
    # as direct API inputs when their CSV slots are intentionally absent.
    no_actual_length = template_run_mode in ("ORI_SPARSE", "ORI_CMP_SPARSE")
    op_seqused_ori_kv = None if no_actual_length else seqused_ori_kv
    op_seqused_cmp_kv = None if no_actual_length else seqused_cmp_kv
    cu_seqlens_ori_kv = cu_seqlens_ori_kv if layout_kv == "TND" else None
    cu_seqlens_cmp_kv = cu_seqlens_cmp_kv if layout_kv == "TND" else None

    input_data = {
        "Testcase_Name": Testcase_Name,
        "params": params,
        "metadata_input": {
            "num_heads_q": N1,
            "num_heads_kv": N2,
            "head_dim": D,
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_q": seqused_q,
            "cu_seqlens_ori_kv": cu_seqlens_ori_kv,
            "cu_seqlens_cmp_kv": cu_seqlens_cmp_kv,
            "seqused_ori_kv": seqused_ori_kv,
            "seqused_cmp_kv": seqused_cmp_kv,
            "cmp_residual_kv": cmp_residual_kv,
            "ori_topk_length": ori_topk_length,
            "cmp_topk_length": cmp_topk_length,
            "batch_size": B,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_ori_kv": max_seqlen_ori_kv,
            "max_seqlen_cmp_kv": max_seqlen_cmp_kv,
            "topk": K if template_run_mode in ("CSA", "ORI_CMP_SPARSE") else 0,
            "ori_topk": K1
            if template_run_mode in ("ORI_SPARSE", "ORI_CMP_SPARSE")
            else 0,
            "cmp_ratio": cmp_ratio
            if template_run_mode not in ("SWA", "ORI_SPARSE")
            else 1,
            "quant_mode": quant_mode,
            "ori_mask_mode": ori_mask_mode,
            "cmp_mask_mode": cmp_mask_mode,
            "ori_win_left": ori_win_left,
            "ori_win_right": ori_win_right,
            "layout_q": layout_q,
            "layout_kv": layout_kv,
            "has_ori_kv": True,
            "has_cmp_kv": False
            if (template_run_mode in ("SWA", "ORI_SPARSE") or cmp_k_in_pa_shape is None)
            else True,
        },
        "op_input": {
            "q": q_npu,
            "ori_kv": ori_k_in_pa_shape,
            "cmp_kv": cmp_k_in_pa_shape,
            "ori_sparse_indices": ori_sparse_indices,
            "cmp_sparse_indices": cmp_sparse_indices,
            "ori_topk_length": ori_topk_length,
            "cmp_topk_length": cmp_topk_length,
            "ori_block_table": ori_block_table,
            "cmp_block_table": cmp_block_table,
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_q": seqused_q,
            "cu_seqlens_ori_kv": cu_seqlens_ori_kv,
            "cu_seqlens_cmp_kv": cu_seqlens_cmp_kv,
            "seqused_ori_kv": op_seqused_ori_kv,
            "seqused_cmp_kv": op_seqused_cmp_kv,
            "cmp_residual_kv": cmp_residual_kv,
            "sinks": sinks,
            "q_descale": q_descale,
            "ori_kv_descale": ori_kv_descale,
            "cmp_kv_descale": cmp_kv_descale,
            "softmax_scale": softmax_scale,
            "quant_mode": quant_mode,
            "cmp_ratio": cmp_ratio
            if template_run_mode not in ("SWA", "ORI_SPARSE")
            else 1,
            "ori_mask_mode": ori_mask_mode,
            "cmp_mask_mode": cmp_mask_mode,
            "ori_win_left": ori_win_left,
            "ori_win_right": ori_win_right,
            "layout_q": layout_q,
            "layout_kv": layout_kv,
            "topk_value_mode": topk_value_mode,
            "return_softmax_lse": return_softmax_lse,
        },
        "golden_state": golden_state,
        "cpu_output": cpu_result,
        "cpu_lse": cpu_lse if return_softmax_lse else None,
    }

    return input_data


def generate_cpu_golden(input_data):
    """Calculate QSMLA CPU Golden from input-stage state without new random data."""
    state = input_data["golden_state"]
    test_qsmla = GeneralizedSFAQuant(
        state["layout_q"],
        state["layout_kv"],
        state["q_type"],
        state["ori_kv_type"],
        state["cmp_kv_type"],
        state["B"],
        state["S1"],
        state["T1"],
        state["N1"],
        state["N2"],
        state["D"],
        state["K"],
        state["block_num1"],
        state["block_num2"],
        state["block_size1"],
        state["block_size2"],
        state["cu_seqlens_q"],
        state["seqused_q"],
        state["seqused_ori_kv"],
        state["seqused_cmp_kv"],
        state["cu_seqlens_ori_kv"],
        state["cu_seqlens_cmp_kv"],
        state["cmp_residual_kv"],
        state["softmax_scale"],
        state["cmp_ratio"],
        state["ori_mask_mode"],
        state["cmp_mask_mode"],
        state["ori_win_left"],
        state["ori_win_right"],
        state["ori_topk_length"],
        state["cmp_topk_length"],
        state["template_run_mode"],
        q_descale_val=state["q_descale_val"],
        ori_kv_descale_val=state["ori_kv_descale_val"],
        cmp_kv_descale_val=state["cmp_kv_descale_val"],
    )
    cpu_output, cpu_lse = test_qsmla.forward(
        state["q"],
        state["ori_k_bnsd"],
        state["cmp_k_bnsd"],
        state["ori_sparse_indices"],
        state["cmp_sparse_indices"],
        state["cu_seqlens_q"],
        state["seqused_ori_kv"],
        state["seqused_cmp_kv"],
        state["cmp_residual_kv"],
        state["sinks"],
        state["ori_topk_length"],
        state["cmp_topk_length"],
        state["return_softmax_lse"],
    )
    state["cpu_output"] = cpu_output
    state["cpu_lse"] = cpu_lse
    input_data["cpu_output"] = cpu_output
    input_data["cpu_lse"] = cpu_lse
    return cpu_output, cpu_lse
