#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
import check_valid_param
import pytest


class GeneralizedGQA:
    def __init__(
        self,
        batch_size,
        q_head_num,
        kv_head_num,
        q_seq,
        kv_seq,
        head_dim,
        dtype,
        in_layout,
        act_seq_len,
        act_seq_len_kv,
        scaled_value,
        block_size,
        cache_layout,
    ):
        self.batch_size = batch_size
        self.q_head_num = q_head_num
        self.kv_head_num = kv_head_num
        self.group_size = q_head_num // kv_head_num
        self.q_seq = q_seq
        self.kv_seq = kv_seq
        self.head_dim = head_dim
        self.dtype = dtype
        self.in_layout = in_layout
        self.act_seq_len = act_seq_len
        self.act_seq_len_kv = act_seq_len_kv
        self.scale = scaled_value
        self.block_size = block_size
        self.cache_layout = cache_layout

    def get_kv_block(self, kv_cache, block_id, head_idx):
        # 从KV Cache中获取指定block的数据
        if self.cache_layout == "BNBD":
            # 格式: (blocknum, kv_head_num, blocksize, head_dim)
            block = kv_cache[block_id]  # [kv_head_num, blocksize, head_dim]
            block_slice = block[head_idx]  # [blocksize, D]
        else:
            # 格式: (blocknum, blocksize, H), H = kv_head_num * head_dim
            block = kv_cache[block_id]  # [blocksize, H]
            # 从合并的H中提取对应head的数据
            start_idx = head_idx * self.head_dim
            end_idx = (head_idx + 1) * self.head_dim
            block_slice = block[:, start_idx:end_idx]  # [blocksize, D]
        if head_idx < self.kv_head_num:
            return block_slice  # [blocksize, D]
        else:
            raise ValueError(f"head_idx {head_idx} out of range")

    def forward(self, query, key, value, block_table):
        # 适配 BSND 输入
        if self.in_layout == "BSND":
            query = query.transpose(1, 2)  # BSND -> BNSD

        outputs = []
        for b in range(self.batch_size):
            batch_outputs = []
            current_q_len = self.act_seq_len[b]
            current_kv_len = self.act_seq_len_kv[b]

            # 获取当前batch的block table
            batch_block_table = block_table[b]
            valid_blocks = batch_block_table[batch_block_table != -1]  # 去除填充的-1
            required_blocks = (current_kv_len + self.block_size - 1) // self.block_size
            if len(valid_blocks) < required_blocks:
                raise ValueError(
                    f"Batch {b}: number of blocks in block table is not enough"
                )

            for q_head_idx in range(self.q_head_num):
                # 计算对应的 KV head index
                kv_head_idx = q_head_idx // self.group_size
                head_output = torch.zeros(self.q_seq, self.head_dim, dtype=query.dtype)

                # 处理每个Q token
                for q_pos in range(current_q_len):
                    q_token = query[b, q_head_idx, q_pos]  # [head_dim]

                    # 初始化attention scores和values
                    total_scores = []
                    total_values = []

                    # 遍历所有有效的block
                    for _, block_id in enumerate(valid_blocks):
                        # 从KV Cache中获取对应的block
                        k_block = self.get_kv_block(key, block_id, kv_head_idx)
                        v_block = self.get_kv_block(value, block_id, kv_head_idx)

                        # 计算当前block内的attention scores
                        block_scores = (
                            torch.matmul(q_token, k_block.transpose(-2, -1))
                            * self.scale
                        )  # [block_size]

                        total_scores.append(block_scores)
                        total_values.append(v_block)

                    # 合并所有的block的scores和values
                    if total_scores:
                        all_scores = torch.cat(
                            total_scores, dim=-1
                        )  # [total_kv_tokens]
                        all_values = torch.cat(
                            total_values, dim=0
                        )  # [total_kv_tokens, head_dim]

                        # 只保留实际长度范围内的tokens
                        valid_scores = all_scores[:current_kv_len]
                        valid_values = all_values[:current_kv_len]

                        # softmax和加权求和
                        attention_weights = torch.softmax(valid_scores, dim=-1).type(
                            query.dtype
                        )
                        token_output = torch.matmul(
                            attention_weights, valid_values
                        )  # [head_dim]

                        head_output[q_pos] = token_output

                batch_outputs.append(head_output)

            # 堆叠所有head: [q_head_num, q_seq_len, head_dim]
            batch_output = torch.stack(batch_outputs, dim=0)
            outputs.append(batch_output)

        # 堆叠所有batch: [batch_size, q_head_num, q_seq_len, head_dim]
        final_output = torch.stack(outputs, dim=0)

        # 如果需要，转换回原格式
        if self.in_layout == "BSND":
            final_output = final_output.transpose(1, 2)  # [B, S, N, D]

        return final_output


def test_gqa_no_quant(params):
    (
        batch_size,
        q_head_num,
        kv_head_num,
        q_seq,
        kv_seq,
        head_dim,
        dtype,
        in_layout,
        act_seq_len,
        act_seq_len_kv,
        scaled_value,
        block_size,
        cache_layout,
    ) = params

    block_table, total_blocks = check_valid_param.create_random_block_table(
        batch_size, act_seq_len_kv, block_size
    )
    block_table = block_table.npu()

    if in_layout == "BNSD":
        if test.DEBUG_ON == 0:
            query = torch.randn(batch_size, q_head_num, q_seq, head_dim).to(dtype).npu()
            if cache_layout == "BNBD":
                k_cache = (
                    torch.randn(total_blocks, kv_head_num, block_size, head_dim)
                    .to(dtype)
                    .npu()
                )
                v_cache = (
                    torch.randn(total_blocks, kv_head_num, block_size, head_dim)
                    .to(dtype)
                    .npu()
                )
            else:
                k_cache = (
                    torch.randn(total_blocks, block_size, kv_head_num * head_dim)
                    .to(dtype)
                    .npu()
                )
                v_cache = (
                    torch.randn(total_blocks, block_size, kv_head_num * head_dim)
                    .to(dtype)
                    .npu()
                )
        else:
            query = torch.ones(batch_size, q_head_num, q_seq, head_dim).to(dtype).npu()
            if cache_layout == "BNBD":
                k_cache = (
                    torch.ones(total_blocks, kv_head_num, block_size, head_dim)
                    .to(dtype)
                    .npu()
                )
                v_cache = (
                    torch.ones(total_blocks, kv_head_num, block_size, head_dim)
                    .to(dtype)
                    .npu()
                )
            else:
                k_cache = (
                    torch.ones(total_blocks, block_size, kv_head_num * head_dim)
                    .to(dtype)
                    .npu()
                )
                v_cache = (
                    torch.ones(total_blocks, block_size, kv_head_num * head_dim)
                    .to(dtype)
                    .npu()
                )
    else:
        # BSND只支持[blocknum, blocksize, H]格式的kv
        if test.DEBUG_ON == 0:
            query = torch.randn(batch_size, q_seq, q_head_num, head_dim).to(dtype).npu()
            k_cache = (
                torch.randn(total_blocks, block_size, kv_head_num * head_dim)
                .to(dtype)
                .npu()
            )
            v_cache = (
                torch.randn(total_blocks, block_size, kv_head_num * head_dim)
                .to(dtype)
                .npu()
            )
        else:
            query = torch.ones(batch_size, q_seq, q_head_num, head_dim).to(dtype).npu()
            k_cache = (
                torch.ones(total_blocks, block_size, kv_head_num * head_dim)
                .to(dtype)
                .npu()
            )
            v_cache = (
                torch.ones(total_blocks, block_size, kv_head_num * head_dim)
                .to(dtype)
                .npu()
            )

    test_gqa = GeneralizedGQA(
        batch_size,
        q_head_num,
        kv_head_num,
        q_seq,
        kv_seq,
        head_dim,
        dtype,
        in_layout,
        act_seq_len,
        act_seq_len_kv,
        scaled_value,
        block_size,
        cache_layout,
    )
    expect = test_gqa.forward(query, k_cache, v_cache, block_table)

    # 调用FIA算子
    result, _ = torch_npu.npu_fused_infer_attention_score(
        query,
        k_cache,
        v_cache,
        actual_seq_lengths=act_seq_len,
        actual_seq_lengths_kv=act_seq_len_kv,
        num_heads=q_head_num,
        num_key_value_heads=kv_head_num,
        input_layout=in_layout,
        scale=scaled_value,
        pre_tokens=65535,
        next_tokens=65535,
        block_table=block_table,
        block_size=block_size,
    )
    torch.npu.synchronize()
    return expect, result
