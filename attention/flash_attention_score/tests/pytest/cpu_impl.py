#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import torch
import torch_npu
import math
import numpy as np
import random
from test_utils import *


def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    res = y.div(x_sum)
    return res, x_max, x_sum


def skip_invalid_row(b, n, sq, softmax_res, x_max, x_sum):
    for i in range(b):
        for j in range(n):
            for k in range(sq):
                if x_max[i, j, k, :] == -40000.0:
                    softmax_res[i, j, k, :] = 0
                    x_max[i, j, k, :] = torch.finfo(torch.float).min
                    x_sum[i, j, k, :] = torch.finfo(torch.float).max
    return softmax_res, x_max, x_sum


def forward(
    q,
    k,
    v,
    drop_mask,
    atten_mask,
    pse_type,
    pse,
    scale,
    keep_prob,
    skip_invalid_row=False,
):
    q = q.float()
    k = k.float()
    v = v.float()

    qk = torch.matmul(q, k.permute(0, 1, 3, 2))
    if pse is None:
        qk = qk.mul(scale)
    else:
        if pse_type == 1:
            qk = qk.add(pse).mul(scale)
        else:
            qk = qk.mul(scale).add(pse)
    if atten_mask is not None:
        qk = qk.masked_fill_(atten_mask.cpu().bool(), value=torch.tensor(-40000))
    softmax_res, x_max, x_sum = tsoftmax(qk)

    if skip_invalid_row:
        b, n, sq, _ = softmax_res.shape
        softmax_res, x_max, x_sum = skip_invalid_row(
            b, n, sq, softmax_res, x_max, x_sum
        )

    if drop_mask is not None:
        drop_res = softmax_res * drop_mask * (1.0 / keep_prob)
    else:
        drop_res = softmax_res

    out = torch.matmul(drop_res, v)

    x_max = x_max.broadcast_to(-1, -1, -1, 8)
    x_sum = x_sum.broadcast_to(-1, -1, -1, 8)
    return out, x_max, x_sum


def get_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int64)
    for i in range(1, len(seqlens_list) + 1):
        cu[i] = cu[i - 1] + seqlens_list[i - 1]
    return cu


def broadcastKV(n1, n2, kv_tensor, dtype):
    factor = n1 // n2
    kv_shape = kv_tensor.shape
    b = kv_shape[0]
    s = kv_shape[2]
    d = kv_shape[3]
    kv_res = torch.zeros(b, n1, s, d).to(dtype)
    for i in range(n1):
        j = i // factor
        kv_res[:, i : i + 1, :, :] = kv_tensor[:, j : j + 1, :, :]
    return kv_res


def tforward_normal(q, k, v, pse, **kwargs):
    b = kwargs.get("B")
    n1 = kwargs.get("N1")
    n2 = kwargs.get("N2", n1)
    s1 = kwargs.get("Sq")
    s2 = kwargs.get("Skv", s1)
    d = kwargs.get("D")
    sparse_mode = kwargs.get("sparse_mode", None)
    pre_tokens = kwargs.get("pre_tokens", 2147483647)
    next_tokens = kwargs.get("next_tokens", 2147483647)
    prefix = kwargs.get("prefix", [])
    seed = kwargs.get("seed", 0)
    offset = kwargs.get("offset", 0)

    scale = kwargs.get("scale", 1 / (d**0.5))
    pse_type = kwargs.get("pse_type", 1)
    keep_prob = kwargs.get("keep_prob", 1)

    if abs(1 - keep_prob) < 2e-14:
        drop_mask = None
    else:
        drop_mask = gen_dropmask(b, n1, s1, s2, keep_prob, seed, offset)

    atten_mask = generate_cpu_mask(
        b, s1, s2, sparse_mode, pre_tokens, next_tokens, prefix
    )
    skip_invalid_row = False
    if atten_mask is not None:
        if sparse_mode == 0:
            skip_invalid_row = next_tokens < 0 or pre_tokens + s2 < s1
        elif sparse_mode == 3:
            skip_invalid_row = s1 > s2
        elif sparse_mode == 4:
            skip_invalid_row = pre_tokens < 0 or next_tokens + s2 < s1
        elif sparse_mode in [5, 6]:
            skip_invalid_row = True if 0 in prefix else False

    if n1 != n2:
        k = broadcastKV(n1, n2, k, k.dtype)
        v = broadcastKV(n1, n2, v, v.dtype)
    out, x_max, x_sum = forward(
        q,
        k,
        v,
        drop_mask,
        atten_mask,
        pse_type,
        pse,
        scale,
        keep_prob,
        skip_invalid_row,
    )
    return out, x_max, x_sum


def tforward_tnd(q, k, v, pse, **kwargs):
    actual_seq_qlen = list(kwargs.get("actual_seq_qlen"))
    actual_seq_kvlen = list(kwargs.get("actual_seq_kvlen", actual_seq_qlen))
    assert isinstance(actual_seq_qlen, list)
    assert isinstance(actual_seq_kvlen, list)
    seqlen_q_list = get_seqlen_list(actual_seq_qlen).numpy()
    seqlen_k_list = get_seqlen_list(actual_seq_kvlen).numpy()
    actual_seq_qlen.insert(0, 0)
    actual_seq_kvlen.insert(0, 0)
    input_layout = kwargs.get("input_layout")
    sparse_mode = kwargs.get("sparse_mode", None)
    pre_tokens = kwargs.get("pre_tokens", 2147483647)
    next_tokens = kwargs.get("next_tokens", 2147483647)
    prefix = kwargs.get("prefix", [])
    seed = kwargs.get("seed", 0)
    offset = kwargs.get("offset", 0)
    max_seqlen_q = seqlen_q_list.max()
    max_seqlen_k = seqlen_k_list.max()
    qk_size = seqlen_q_list * [math.ceil(i / 16) * 16 for i in seqlen_k_list]
    qk_pointer = get_cu_seqlens(qk_size).to(torch.int64)

    b = len(seqlen_q_list)
    n1 = kwargs.get("N1")
    n2 = kwargs.get("N2", n1)
    d = kwargs.get("D")
    d_v = kwargs.get("DV", d)
    s1 = max(actual_seq_qlen)
    s2 = max(actual_seq_kvlen)

    scale = kwargs.get("scale", 1 / (d**0.5))
    pse_type = kwargs.get("pse_type", 1)
    keep_prob = kwargs.get("keep_prob", 1)

    pse_s1 = max(1024, max_seqlen_q)
    band_index = 0
    if sparse_mode == 7:
        for index in range(b - 1, -1, -1):
            if seqlen_q_list[index] != 0:
                band_index = index
                break
    elif sparse_mode == 8:
        for index in range(b):
            if seqlen_k_list[index] != 0:
                band_index = index
                break

    if abs(1 - keep_prob) < 2e-14:
        drop_mask = None
    else:
        drop_mask = gen_dropmask_tnd(
            qk_pointer[-1].item() * n1, keep_prob, seed, offset
        )

    out_golden = torch.zeros([1, n1, s1, d_v], dtype=q.dtype)
    x_max = torch.empty(0)
    x_sum = torch.empty(0)
    for i in range(b):
        if seqlen_q_list[i] != 0 and seqlen_q_list[i] != 0:
            qi = q[:, :, actual_seq_qlen[i] : actual_seq_qlen[i + 1]]
            ki = k[:, :, actual_seq_kvlen[i] : actual_seq_kvlen[i + 1]]
            vi = v[:, :, actual_seq_kvlen[i] : actual_seq_kvlen[i + 1]]

            if n1 != n2:
                ki = broadcastKV(n1, n2, ki, ki.dtype)
                vi = broadcastKV(n1, n2, vi, vi.dtype)

            if pse is None:
                psei = None
            else:
                pse_layout = kwargs.get("pse_layout").lower()
                assert pse_layout in ["bn", "n", "bnhs", "1nhs"]
                if pse_layout == "bn":
                    psei = pse[i : i + 1, :, -seqlen_q_list[i] :, -seqlen_k_list[i] :]
                elif pse_layout == "n":
                    psei = pse[:, :, -seqlen_q_list[i] :, -seqlen_k_list[i] :]
                elif pse_layout == "bnhs":
                    psei = pse[
                        i : i + 1,
                        :,
                        pse_s1 - seqlen_q_list[i] : pse_s1,
                        max_seqlen_k - seqlen_k_list[i] : max_seqlen_k,
                    ]
                elif pse_layout == "1nhs":
                    psei = pse[
                        :,
                        :,
                        pse_s1 - seqlen_q_list[i] : pse_s1,
                        max_seqlen_k - seqlen_k_list[i] : max_seqlen_k,
                    ]

            if drop_mask is None:
                drop_maski = None
            else:
                drop_maski = drop_mask[
                    (qk_pointer[i] * n1) : (qk_pointer[i + 1] * n1)
                ].reshape(n1, seqlen_q_list[i], math.ceil(seqlen_k_list[i] / 16) * 16)[
                    :, :, : seqlen_k_list[i]
                ]

            if sparse_mode is None:
                atten_maski = None
            else:
                atten_maski = generate_cpu_mask(
                    1,
                    seqlen_q_list[i],
                    seqlen_k_list[i],
                    sparse_mode,
                    pre_tokens,
                    next_tokens,
                    prefix,
                    i,
                    band_index,
                )

            outi, x_maxi, x_sumi = forward(
                qi, ki, vi, drop_maski, atten_maski, pse_type, psei, scale, keep_prob
            )
            out_golden[:, :, actual_seq_qlen[i] : actual_seq_qlen[i + 1]] = outi
            x_max = torch.cat((x_max, x_maxi.contiguous().view(-1)))
            x_sum = torch.cat((x_sum, x_sumi.contiguous().view(-1)))
    return out_golden, x_max, x_sum


def tforward(q, k, v, pse, **kwargs):
    input_layout = kwargs.get("input_layout")
    if input_layout == "TND":
        return tforward_tnd(q, k, v, pse, **kwargs)
    else:
        return tforward_normal(q, k, v, pse, **kwargs)
