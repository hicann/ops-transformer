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
import torchair
import torch.nn as nn
from torchair.configs.compiler_config import CompilerConfig
import quant_lightning_indexer_golden


class QLINetwork(nn.Module):
    def __init__(self):
        super(QLINetwork, self).__init__()

    def forward(
        self,
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        query_quant_mode,
        key_quant_mode,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        pre_tokens,
        next_tokens,
        query_dtype,
        key_dtype,
    ):
        return torch_npu.npu_quant_lightning_indexer(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
            query_quant_mode,
            key_quant_mode,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query=layout_query,
            layout_key=layout_key,
            sparse_count=sparse_count,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            query_dtype=query_dtype,
            key_dtype=key_dtype,
        )


def qli_output_acl_graph(params):
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
        dequant_dtype,
        actual_seq_dtype,
        act_seq_q,
        act_seq_k,
        query_quant_mode,
        key_quant_mode,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        query_datarange,
        key_datarange,
        weights_datarange,
        q_scale_datarange,
        k_scale_datarange,
    ) = params

    test_qli = quant_lightning_indexer_golden.GeneralizedQLI(
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
        dequant_dtype,
        actual_seq_dtype,
        act_seq_q,
        act_seq_k,
        query_quant_mode,
        key_quant_mode,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
    )

    actual_seq_lengths_query = (
        torch.tensor(np.random.uniform(q_seq, q_seq, batch_size))
        .to(actual_seq_dtype)
        .npu()
        if act_seq_q is None
        else torch.tensor(act_seq_q).to(actual_seq_dtype).npu()
    )
    actual_seq_lengths_key = (
        torch.tensor(np.random.uniform(k_seq, k_seq, batch_size))
        .to(actual_seq_dtype)
        .npu()
        if act_seq_k is None
        else torch.tensor(act_seq_k).to(actual_seq_dtype).npu()
    )

    if layout_query == "BSND":
        query = (
            torch.tensor(
                np.random.uniform(
                    query_datarange[0],
                    query_datarange[1],
                    (batch_size, q_seq, q_head_num, head_dim),
                )
            )
            .to(qk_dtype)
            .npu()
        )
        query_dequant_scale = (
            torch.tensor(
                np.random.uniform(
                    q_scale_datarange[0],
                    q_scale_datarange[1],
                    (batch_size, q_seq, q_head_num),
                )
            )
            .to(dequant_dtype)
            .npu()
        )
        weights = (
            torch.tensor(
                np.random.uniform(
                    weights_datarange[0],
                    weights_datarange[1],
                    (batch_size, q_seq, q_head_num),
                )
            )
            .to(weight_dtype)
            .npu()
        )

    elif layout_query == "TND":
        query = (
            torch.tensor(
                np.random.uniform(
                    query_datarange[0],
                    query_datarange[1],
                    (q_t_size, q_head_num, head_dim),
                )
            )
            .to(qk_dtype)
            .npu()
        )
        query_dequant_scale = (
            torch.tensor(
                np.random.uniform(
                    q_scale_datarange[0], q_scale_datarange[1], (q_t_size, q_head_num)
                )
            )
            .to(dequant_dtype)
            .npu()
        )
        weights = (
            torch.tensor(
                np.random.uniform(
                    weights_datarange[0], weights_datarange[1], (q_t_size, q_head_num)
                )
            )
            .to(weight_dtype)
            .npu()
        )

    if layout_key == "BSND":
        key = (
            torch.tensor(
                np.random.uniform(
                    key_datarange[0],
                    key_datarange[1],
                    (batch_size, k_seq, k_head_num, head_dim),
                )
            )
            .to(qk_dtype)
            .npu()
        )
        key_dequant_scale = (
            torch.tensor(
                np.random.uniform(
                    k_scale_datarange[0],
                    k_scale_datarange[1],
                    (batch_size, k_seq, k_head_num),
                )
            )
            .to(dequant_dtype)
            .npu()
        )
        block_table = None
        cpu_result, topk_value = test_qli.forward(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            block_table,
        )

    elif layout_key == "TND":
        key = (
            torch.tensor(
                np.random.uniform(
                    key_datarange[0], key_datarange[1], (k_t_size, k_head_num, head_dim)
                )
            )
            .to(qk_dtype)
            .npu()
        )
        key_dequant_scale = (
            torch.tensor(
                np.random.uniform(
                    k_scale_datarange[0], k_scale_datarange[1], (k_t_size, k_head_num)
                )
            )
            .to(dequant_dtype)
            .npu()
        )
        block_table = None
        cpu_result, topk_value = test_qli.forward(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            block_table,
        )

    elif layout_key == "PA_BSND":
        # 以不同batch中最大seq为标准初始化key(bnsd)和key_dequant_scale(bns)
        k_max_s2 = math.floor(max(act_seq_k))
        k_max_block_num_per_batch = math.ceil(
            k_max_s2 / block_size
        )  # 遍历batch得到的最大的block num
        key_bnsd = torch.tensor(
            np.random.uniform(
                key_datarange[0],
                key_datarange[1],
                (batch_size, k_head_num, k_max_s2, head_dim),
            )
        ).to(qk_dtype)
        key_dequant_scale_bns = torch.tensor(
            np.random.uniform(
                k_scale_datarange[0],
                k_scale_datarange[1],
                (batch_size, k_head_num, k_max_s2),
            )
        ).to(dequant_dtype)
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
        # [batch_size, s2, k_head_num, head_dim] expand to [batch_size, k_max_block_num_per_batch * block_size, k_head_num, head_dim]
        key_expand = torch.zeros(
            (batch_size, k_head_num, k_max_block_num_per_batch * block_size, head_dim),
            dtype=qk_dtype,
        )
        key_expand[:, :, :k_max_s2, :] = key_bnsd
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
        key = key.npu()
        # 构建PA场景的key_dequant_scale
        key_dequant_scale_expand = torch.zeros(
            (batch_size, k_head_num, k_max_block_num_per_batch * block_size),
            dtype=dequant_dtype,
        )
        key_dequant_scale_expand[:, :, :k_max_s2] = key_dequant_scale_bns
        key_dequant_scale = torch.zeros(
            (block_num, block_size, k_head_num), dtype=dequant_dtype
        )
        for i_batch in range(batch_size):
            for i_block, cur_block_id in enumerate(block_table[i_batch]):
                block_start_pos = i_block * block_size
                if cur_block_id == -1:
                    continue
                else:
                    for i_n in range(k_head_num):
                        key_dequant_scale[cur_block_id, :, i_n] = (
                            key_dequant_scale_expand[
                                i_batch,
                                i_n,
                                block_start_pos : block_start_pos + block_size,
                            ]
                        )
        key_dequant_scale = key_dequant_scale.npu()
        cpu_result, topk_value = test_qli.forward(
            query,
            key_bnsd,
            weights,
            query_dequant_scale,
            key_dequant_scale_bns,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            block_table,
        )
        block_table = torch.from_numpy(block_table).to(dtype=torch.int32).npu()

    print("acl_graph")

    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    config.mode = "reduce-overhead"
    npu_mode = torch.compile(
        QLINetwork().npu(), fullgraph=True, backend=npu_backend, dynamic=False
    )
    npu_result = npu_mode(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        query_quant_mode,
        key_quant_mode,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        query_dtype=None,
        key_dtype=None,
    )
    torch.npu.synchronize()
    return cpu_result, npu_result, topk_value
