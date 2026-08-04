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

import torch


SUPPORTED_LAYOUT_QUERY = {"BSND", "TND"}
SUPPORTED_LAYOUT_KV = {"BSND", "TND", "PA_BSND"}
SUPPORTED_Q_TYPE = {torch.float16, torch.bfloat16}
SUPPORTED_G = {1, 2, 4, 8, 16, 32, 64, 128}


def _check_actual_seq(actual_seq, B, upper_bound, name):
    # TND/动态有效长场景下，需要保证 actual_seq 与 batch 和长度约束一致。
    if actual_seq is None:
        return
    if len(actual_seq) != B:
        raise ValueError(f"{name} 长度应等于 B={B}，当前: {len(actual_seq)}")
    for i, seq in enumerate(actual_seq):
        if int(seq) <= 0:
            raise ValueError(f"{name}[{i}] 应大于 0，当前: {seq}")
        if upper_bound is not None and int(seq) > upper_bound:
            raise ValueError(f"{name}[{i}] 不应大于 {upper_bound}，当前: {seq}")


def check_valid_param(params):
    # 按 sparse_flash_attention tiling 约束校验测试参数。
    if len(params) == 21:
        (
            Testcase_Name,
            layout_query,
            layout_kv,
            q_type,
            B,
            S1,
            S2,
            N1,
            N2,
            D,
            K,
            scale_value,
            sparse_block_size,
            rope_head_dim,
            sparse_mode,
            attention_mode,
            return_softmax_lse,
            block_size,
            block_num,
            actual_seq_q,
            actual_seq_kv,
        ) = params
        use_sinks = False
    else:
        (
            Testcase_Name,
            layout_query,
            layout_kv,
            q_type,
            B,
            S1,
            S2,
            N1,
            N2,
            D,
            K,
            scale_value,
            sparse_block_size,
            rope_head_dim,
            sparse_mode,
            attention_mode,
            return_softmax_lse,
            use_sinks,
            block_size,
            block_num,
            actual_seq_q,
            actual_seq_kv,
        ) = params

    if q_type not in SUPPORTED_Q_TYPE:
        raise ValueError(f"q_type 仅支持 torch.float16/torch.bfloat16，当前: {q_type}")
    if layout_query not in SUPPORTED_LAYOUT_QUERY:
        raise ValueError(
            f"layout_query 仅支持 {sorted(SUPPORTED_LAYOUT_QUERY)}，当前: {layout_query}"
        )
    if layout_kv not in SUPPORTED_LAYOUT_KV:
        raise ValueError(
            f"layout_kv 仅支持 {sorted(SUPPORTED_LAYOUT_KV)}，当前: {layout_kv}"
        )
    if layout_kv != "PA_BSND" and layout_kv != layout_query:
        raise ValueError(
            f"非 PA_BSND 场景下 layout_kv({layout_kv}) 必须等于 layout_query({layout_query})"
        )
    for name, value in (
        ("B", B),
        ("S1", S1),
        ("S2", S2),
        ("N1", N1),
        ("N2", N2),
        ("D", D),
        ("K", K),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} 应大于 0，当前: {value}")

    if N2 != 1:
        raise ValueError(f"N2 仅支持 1，当前: {N2}")
    if N1 % N2 != 0:
        raise ValueError(f"N1({N1}) 必须能被 N2({N2}) 整除")
    if (N1 // N2) not in SUPPORTED_G:
        raise ValueError(f"g=N1/N2 仅支持 {sorted(SUPPORTED_G)}，当前: {N1 // N2}")
    if D != 512:
        raise ValueError(f"D 仅支持 512，当前: {D}")
    if rope_head_dim != 64:
        raise ValueError(f"rope_head_dim 仅支持 64，当前: {rope_head_dim}")
    if attention_mode != 2:
        raise ValueError(f"attention_mode 仅支持 2，当前: {attention_mode}")
    if sparse_mode not in (0, 3):
        raise ValueError(f"sparse_mode 仅支持 0/3，当前: {sparse_mode}")
    if (
        sparse_block_size <= 0
        or sparse_block_size > 128
        or (sparse_block_size & (sparse_block_size - 1)) != 0
    ):
        raise ValueError(
            f"sparse_block_size 应在 [1, 128] 且为 2 的幂，当前: {sparse_block_size}"
        )

    max_sparse_block_count = math.ceil(S2 / sparse_block_size)
    if K > max_sparse_block_count:
        raise ValueError(
            f"K({K}) 不应大于 ceil(S2 / sparse_block_size)={max_sparse_block_count}"
        )

    if layout_query == "TND" or actual_seq_q is not None:
        _check_actual_seq(actual_seq_q, B, S1, "actual_seq_q")
    if layout_kv == "TND" or actual_seq_kv is not None:
        if layout_kv == "PA_BSND":
            _check_actual_seq(actual_seq_kv, B, None, "actual_seq_kv")
        else:
            _check_actual_seq(actual_seq_kv, B, S2, "actual_seq_kv")

    if layout_kv == "PA_BSND":
        if block_size is None or int(block_size) <= 0:
            raise ValueError("PA_BSND 场景必须提供正整数 block_size")
        if block_size % 16 != 0:
            raise ValueError(
                f"PA_BSND 场景 block_size 必须 16 对齐，当前: {block_size}"
            )
        if block_size % sparse_block_size != 0:
            raise ValueError(
                f"PA_BSND 场景 block_size({block_size}) 必须被 sparse_block_size({sparse_block_size}) 整除"
            )
        if return_softmax_lse:
            raise ValueError("PA_BSND 场景下 return_softmax_lse 不支持为 True")
        if block_num is None or int(block_num) <= 0:
            raise ValueError("PA_BSND 场景必须提供正整数 block_num")
        if actual_seq_kv is not None:
            needed_blocks = sum(
                math.ceil(int(seq) / block_size) for seq in actual_seq_kv
            )
            if block_num < needed_blocks:
                raise ValueError(
                    f"block_num({block_num}) 小于所需物理块数 {needed_blocks}"
                )
    elif block_num is not None and int(block_num) <= 0:
        raise ValueError(f"block_num 若提供则应大于 0，当前: {block_num}")

    if scale_value <= 0:
        raise ValueError(f"scale_value 应大于 0，当前: {scale_value}")
