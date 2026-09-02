#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""GPU flash_attn API 薄封装层。只接收正确格式的数据，不做 layout 转换。"""

import torch

try:
    from flash_attn import (
        flash_attn_func,
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


def gpu_fixed_attn(q_bshd, k_bshd, v_bshd, softmax_scale, causal):
    """固定长度 attention。输入 [B, S, N, D]。"""
    out_bshd = flash_attn_func(
        q_bshd,
        k_bshd,
        v_bshd,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(-1, -1),
    )
    return out_bshd


def gpu_varlen_attn(
    q_tnd, k_tnd, v_tnd, cu_q, cu_kv, max_sq, max_skv, softmax_scale, causal, device
):
    """变长 attention。输入 [total_tokens, N, D]。"""
    cu_q_t = (
        torch.tensor(cu_q, dtype=torch.int32, device=device)
        if not isinstance(cu_q, torch.Tensor)
        else cu_q.to(device=device, dtype=torch.int32)
    )
    cu_kv_t = (
        torch.tensor(cu_kv, dtype=torch.int32, device=device)
        if not isinstance(cu_kv, torch.Tensor)
        else cu_kv.to(device=device, dtype=torch.int32)
    )
    out_tnd = flash_attn_varlen_func(
        q_tnd,
        k_tnd,
        v_tnd,
        cu_seqlens_q=cu_q_t,
        cu_seqlens_k=cu_kv_t,
        max_seqlen_q=int(max_sq),
        max_seqlen_k=int(max_skv),
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(-1, -1),
    )
    return out_tnd


def gpu_paged_attn(
    q_bshd, k_cache, v_cache, block_table, cache_seqlens, softmax_scale, causal
):
    """Paged KV cache attention。
    q: [B, S, N, D], k_cache/v_cache: [num_blocks, block_size, N, D]
    """
    out_bshd = flash_attn_with_kvcache(
        q_bshd,
        k_cache,
        v_cache,
        k=None,
        v=None,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(-1, -1),
    )
    return out_bshd
