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
import torch._dynamo
import torch.nn as nn
import torchair
from torchair.configs.compiler_config import CompilerConfig
import check_valid_param


class NPUOperatorWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, params):
        (
            query,
            key,
            value,
            act_seq_len,
            act_seq_len_kv,
            q_head_num,
            kv_head_num,
            in_layout,
            scaled_value,
            block_table,
            block_size,
        ) = params
        return torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
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


def _setup_torchair_compilation():
    # 创建图模式包装器
    torch.npu.set_compile_mode(jit_compile=True)
    model = NPUOperatorWrapper().npu()

    # 配置TorchAir编译
    config = CompilerConfig()
    config.experimental_config.keep_inference_input_mutations = (
        True  # 允许在推理过程中修改输入， 允许kv cacahe更新
    )
    config.experimental_config.frozen_parameter = True  # 将模型参数视为常量进行优化
    config.experimental_config.tiling_schedule_optimize = True  # tiling下沉开关

    npu_backend = torchair.get_npu_backend(compiler_config=config)
    compiled_model = torch.compile(
        model, dynamic=True, fullgraph=True, backend=npu_backend
    )
    return compiled_model


def test_gqa_no_quant_ge(params):
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

    compiled_model = _setup_torchair_compilation()

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

    q_fa = query.clone()
    k_cache_fa = k_cache.clone()
    v_cache_fa = v_cache.clone()

    expect, _ = torch_npu.npu_fused_infer_attention_score(
        q_fa,
        k_cache_fa,
        v_cache_fa,
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

    # 执行编译后的模型
    # Tiling下沉
    torch._dynamo.mark_static(query)
    torch._dynamo.mark_static(k_cache)
    torch._dynamo.mark_static(v_cache)
    torch._dynamo.mark_static(block_table)
    params = (
        query,
        k_cache,
        v_cache,
        act_seq_len,
        act_seq_len_kv,
        q_head_num,
        kv_head_num,
        in_layout,
        scaled_value,
        block_table,
        block_size,
    )
    result, _ = compiled_model(params)

    torch.npu.synchronize()
    return expect, result
