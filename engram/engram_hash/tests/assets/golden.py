#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Golden reference for engram_hash (torch)."""

__spec__ = {
    "engram_hash": "EngramHashTestSpec",
    "aclnnEngramHash": "AclnnEngramHashTestSpec",
    "torch.ops.cann_ops_transformer.engram_hash": "EngramHashE2ESpec",
}

import numpy as np
import torch

# 每 token 分块的最大 token 数 (控制大 T 用例的峰值内存)
_TOKEN_CHUNK = 8192


def _to_torch(t):
    if isinstance(t, torch.Tensor) or hasattr(t, "detach"):
        return t.detach().cpu().contiguous()
    return torch.from_numpy(np.ascontiguousarray(t))


def golden_engram_hash(ids, multipliers, vocab_sizes, offsets):
    """向量化 golden (torch 实现, 位级镜像 op_kernel/engram_hash_kernel.h 的 C 语义)。

    ids[T,M] int32 / mult[L,M] int64 / vocab[L,M-1,tables] int32 / offs[L,C] int32,
    返回 int32 [L, T, C] (T=0 时为空)。
    """
    ids = _to_torch(ids).to(torch.int32)
    mult = _to_torch(multipliers).to(torch.int64)
    vocab = _to_torch(vocab_sizes).to(torch.int32)
    offs = _to_torch(offsets).to(torch.int32)

    num_tokens, m = ids.shape
    layers = mult.shape[0]
    _, _, tables = vocab.shape
    num_out_cols = (m - 1) * tables

    out = np.empty((layers, num_tokens, num_out_cols), np.int32)
    if num_tokens == 0:
        return out

    ids64 = ids.to(torch.int64)
    ngram_idx = np.arange(num_out_cols) // tables  # = last - 1
    col_in_table = np.arange(num_out_cols) % tables

    for l in range(layers):
        mult_l = mult[l]
        offs_l = offs[l].to(torch.int64)
        v = vocab[l][ngram_idx, col_in_table].to(torch.int64)
        v_abs = torch.where(v >= 0, v, -v)  # 负 V 按 |V| 取模 (truncmod)
        v_safe = torch.where(v_abs == 0, torch.ones_like(v_abs), v_abs)  # V=0 除零防护

        for t0 in range(0, num_tokens, _TOKEN_CHUNK):
            t1 = min(t0 + _TOKEN_CHUNK, num_tokens)
            ids_blk = ids64[t0:t1]

            # hash 初值广播到所有列; torch int64 乘法回绕 = C 语义
            h = torch.empty((t1 - t0, num_out_cols), dtype=torch.int64)
            h[:] = (ids_blk[:, 0] * mult_l[0]).unsqueeze(1)

            # 第 k 项影响 c >= (k-1)*tables 的列 (列后缀, 增量 XOR)
            for k in range(1, m):
                h[:, (k - 1) * tables :] ^= ids_blk[:, k : k + 1] * mult_l[k]

            # C truncmod: r = h - trunc(h/v)*v, |q*v| <= |h| 恒不溢出 (含 INT64_MIN)
            q = torch.div(h, v_safe.unsqueeze(0), rounding_mode="trunc")
            r = h - q * v_safe.unsqueeze(0)
            r = torch.where(
                v == 0, torch.full_like(r, -1), r
            )  # V=0: 硬件取模余数为常数 -1
            r = torch.where(
                r < 0, r + v.unsqueeze(0), r
            )  # 负余数补正 r += V (V 可为负)

            # out = int32(r) + offs[l, c]: 先截 int32, int64 加法, 存储再截 (回绕)
            out[l, t0:t1] = (
                (r.to(torch.int32).to(torch.int64) + offs_l.unsqueeze(0))
                .to(torch.int32)
                .numpy()
            )
    return out


class EngramHashTestSpec:
    """kernel/geir 流程 (numpy 入参), 参数序同 engram_hash_def.cpp。"""

    def golden(*args, **kwargs):
        ids, multipliers, vocab_sizes, offsets = args[:4]
        return [golden_engram_hash(ids, multipliers, vocab_sizes, offsets)]

    tolerance = {"int32": {"standard": "binary_equal"}}


class AclnnEngramHashTestSpec:
    """aclnn 流程 (torch 入参), 输出 tensor 作为末位入参传入、被忽略。"""

    def golden(*args, **kwargs):
        tensors = [
            a
            for a in args
            if isinstance(a, (torch.Tensor, np.ndarray)) or hasattr(a, "detach")
        ]
        ids, multipliers, vocab_sizes, offsets = tensors[:4]
        return [golden_engram_hash(ids, multipliers, vocab_sizes, offsets)]

    tolerance = {"int32": {"standard": "binary_equal"}}


class EngramHashE2ESpec:
    """E2E 流程 (torch 入参), 输出由接口返回。"""

    def golden(ngram_token_ids, multipliers, vocab_sizes, offsets, **kwargs):
        return [golden_engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets)]

    tolerance = {"int32": {"standard": "binary_equal"}}
