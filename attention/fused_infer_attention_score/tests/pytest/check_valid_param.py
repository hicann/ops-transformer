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

import math
import random
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


def validate_config(params):
    batch_size, q_head_num, kv_head_num, q_seq, kv_seq, head_dim, dtype, in_layout = (
        params
    )
    # 校验
    if batch_size > 65536:
        raise ValueError("batch_size must <= 65536")
    if q_head_num > 256:
        raise ValueError("q_head_num must <= 256")
    if kv_head_num > 256:
        raise ValueError("kv_head_num must <= 256")
    if q_head_num % kv_head_num:
        raise ValueError("q_head_num should be intergral multiple of kv_head_num")
    if head_dim > 512:
        raise ValueError("head_dim must <= 512")
    if dtype not in [torch.bfloat16, torch.float16, torch.int8]:
        raise ValueError("dtype should be: float16/bfloat16/int8")
    if in_layout not in ["BSH", "BSND", "BNSD"]:
        raise ValueError("input_layout should be: BSH/BSND/BNSD")


def create_random_block_table(batch_size, act_seq_len_kv, block_size):
    # 计算最大序列长度
    max_kv_seq_length = max(act_seq_len_kv)

    # 计算每个序列实际需要的block数量
    valid_block_num = [
        math.ceil(act_seq_len_kv[i] / block_size) for i in range(batch_size)
    ]
    valid_block_num_sum = sum(valid_block_num)

    # 计算最大block数量
    max_valid_block_num = math.ceil(max_kv_seq_length / block_size)

    # 随机分配block table
    block_table = -torch.ones(batch_size, max_valid_block_num, dtype=torch.int32)
    block_idx = list(range(valid_block_num_sum))
    random.shuffle(block_idx)
    block_idx_start = 0

    for i in range(batch_size):
        block_table[i][: valid_block_num[i]] = torch.tensor(
            block_idx[block_idx_start : block_idx_start + valid_block_num[i]],
            dtype=torch.int32,
        )
        block_idx_start += valid_block_num[i]

    return block_table, valid_block_num_sum


def check_result(expect, result):
    result_cpu = result.cpu()
    expect_cpu = expect.cpu()
    attn_out_diff = result_cpu.reshape(-1) - expect_cpu.reshape(-1)
    thres = 0.005
    idx = torch.nonzero(abs(attn_out_diff) > thres).squeeze()
    values = attn_out_diff[abs(attn_out_diff) > thres]

    if idx.numel() > 0:
        logger.info(f"element indices where diff > {thres}: {idx}")
        logger.info(f"element values where diff > {thres}: {values}")

    pass_num = torch.sum(abs(attn_out_diff.squeeze()) <= thres)
    total_num = attn_out_diff.numel()
    accuracy = pass_num / total_num
    logger.info(f"pass num: {pass_num}, total num: {total_num}, accuracy: {accuracy}")

    assert torch.allclose(
        result_cpu.flatten(), expect_cpu.flatten(), rtol=0.05, atol=0.05
    )
