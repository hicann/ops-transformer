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

import torch
import math
from utils.data import generate_cpu_mask


def tsoftmax(x):
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    res = y.div(x_sum)
    return res, x_max, x_sum


def _fix_invalid_rows(softmax_res, x_max, x_sum):
    b, n, sq, _ = softmax_res.shape
    for i in range(b):
        for j in range(n):
            for k in range(sq):
                if x_max[i, j, k, :] == -1.7e38:
                    softmax_res[i, j, k, :] = 0
                    x_max[i, j, k, :] = torch.finfo(torch.float).min
                    x_sum[i, j, k, :] = torch.finfo(torch.float).max
    return softmax_res, x_max, x_sum


def _attend(q, k, v, atten_mask, scale, need_fix_invalid):
    q = q.float()
    k = k.float()
    v = v.float()

    qk = torch.matmul(q, k.permute(0, 1, 3, 2)).mul(scale)
    if atten_mask is not None:
        qk = qk.masked_fill_(atten_mask.cpu().bool(), value=torch.tensor(-1.7e38))

    if qk.shape[-1] == 0:
        b, n, sq, _ = qk.shape
        softmax_res = torch.zeros((b, n, sq, 0), dtype=qk.dtype)
        x_max = torch.zeros(b, n, sq, 1)  # torch.finfo(torch.float).min
        x_sum = torch.zeros(b, n, sq, 1)  # torch.finfo(torch.float).max
    else:
        softmax_res, x_max, x_sum = tsoftmax(qk)

    # softmax_res, x_max, x_sum = tsoftmax(qk)

    if need_fix_invalid:
        softmax_res, x_max, x_sum = _fix_invalid_rows(softmax_res, x_max, x_sum)

    out = torch.matmul(softmax_res, v)
    x_max = x_max
    x_sum = x_sum
    return out, x_max, x_sum


def get_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int64)
    for i in range(1, len(seqlens_list) + 1):
        cu[i] = cu[i - 1] + seqlens_list[i - 1]
    return cu


def broadcastKV(n1, n2, kv_tensor, dtype):
    kv_shape = kv_tensor.shape
    b = kv_shape[0]
    s = kv_shape[2]
    d = kv_shape[3]
    kv_res = torch.zeros(b, n1, s, d).to(dtype)
    if n1 >= n2:
        factor = n1 // n2
        for i in range(n1):
            j = i // factor
            kv_res[:, i : i + 1, :, :] = kv_tensor[:, j : j + 1, :, :]
    else:
        factor = n2 // n1
        for i in range(n1):
            j = i * factor
            kv_res[:, i : i + 1, :, :] = kv_tensor[:, j : j + 1, :, :]
    return kv_res


def _should_fix_invalid_rows(
    mask_mode, win_left, win_right, act_q_len, act_kv_len, prefix
):
    if mask_mode is None or mask_mode == 0:
        return False
    if mask_mode == 3:
        return act_q_len > act_kv_len
    if mask_mode == 4:
        return win_left < 0 or win_right + act_kv_len < act_q_len
    return False


def tforward_tnd(q, k, v, **kwargs):
    cu_q = kwargs["cu_seqlens_q"]
    seqused_q = kwargs["seqused_q"]
    seqused_kv = kwargs["seqused_kv"]
    b = len(cu_q) - 1

    n1 = kwargs.get("N1")
    n2 = kwargs.get("N2", n1)
    d = kwargs.get("D")
    d_v = kwargs.get("DV", d)
    s1_total = cu_q[-1]

    layout_kv = kwargs.get("layout_kv", None)
    is_pa = layout_kv in ("PA_BBND", "PA_BNBD", "PA_NZ")
    cu_kv = kwargs.get("cu_seqlens_kv", None)

    mask_mode = kwargs.get("mask_mode", None)
    win_left = kwargs.get("win_left", 2147483647)
    win_right = kwargs.get("win_right", 2147483647)
    prefix = kwargs.get("prefix", [])

    scale = kwargs.get("scale", 1 / (d**0.5))

    band_index = 0

    qk_size = [int(seqused_q[i] * math.ceil(seqused_kv[i] / 16) * 16) for i in range(b)]
    qk_pointer = get_cu_seqlens(qk_size).to(torch.int64)

    out_golden = torch.zeros([1, n1, s1_total, d_v], dtype=q.dtype)
    x_max = torch.zeros([n1, s1_total], dtype=torch.float32)
    x_sum = torch.zeros([n1, s1_total], dtype=torch.float32)
    for i in range(b):
        act_q_len = seqused_q[i]
        act_kv_len = seqused_kv[i]
        if act_q_len == 0 or act_kv_len == 0:
            continue

        q_start = cu_q[i]
        kv_start = cu_kv[i] if cu_kv is not None else sum(seqused_kv[:i])

        qi = q[:, :, q_start : q_start + act_q_len]
        ki = k[:, :, kv_start : kv_start + act_kv_len]
        vi = v[:, :, kv_start : kv_start + act_kv_len]

        if n1 != n2:
            ki = broadcastKV(n1, n2, ki, ki.dtype)
            vi = broadcastKV(n1, n2, vi, vi.dtype)

        if mask_mode is None:
            atten_maski = None
        else:
            atten_maski = generate_cpu_mask(
                1,
                act_q_len,
                act_kv_len,
                mask_mode,
                win_left,
                win_right,
                prefix,
                i,
                band_index,
            )

        need_fix = _should_fix_invalid_rows(
            mask_mode, win_left, win_right, act_q_len, act_kv_len, prefix
        )

        outi, x_maxi, x_sumi = _attend(qi, ki, vi, atten_maski, scale, need_fix)
        out_golden[:, :, q_start : q_start + act_q_len] = outi
        x_max[:, q_start : q_start + act_q_len] = x_maxi.squeeze(0).squeeze(-1)
        x_sum[:, q_start : q_start + act_q_len] = x_sumi.squeeze(0).squeeze(-1)
    return out_golden, x_max, x_sum


def tforward(q, k, v, **kwargs):
    input_layout = kwargs.get("input_layout")
    if input_layout != "TND":
        b = kwargs.get("B")
        n1 = kwargs.get("N1")
        n2 = kwargs.get("N2", n1)
        s1 = kwargs.get("S1")
        s2 = kwargs.get("S2", s1)

        q = (
            q.reshape(1, b, n1, s1, q.shape[-1])
            .transpose(1, 2)
            .reshape(1, n1, b * s1, q.shape[-1])
        )
        k = (
            k.reshape(1, b, n2, s2, k.shape[-1])
            .transpose(1, 2)
            .reshape(1, n2, b * s2, k.shape[-1])
        )
        v = (
            v.reshape(1, b, n2, s2, v.shape[-1])
            .transpose(1, 2)
            .reshape(1, n2, b * s2, v.shape[-1])
        )

        cu_q = [i * s1 for i in range(b + 1)]
        cu_kv = [i * s2 for i in range(b + 1)]

        kwargs["cu_seqlens_q"] = cu_q
        kwargs["cu_seqlens_kv"] = cu_kv
        kwargs.setdefault("seqused_q", [s1] * b)
        kwargs.setdefault("seqused_kv", [s2] * b)

    out, x_max, x_sum = tforward_tnd(q, k, v, **kwargs)

    if input_layout != "TND":
        b = kwargs.get("B")
        s1 = kwargs.get("S1")
        n1 = kwargs.get("N1")
        # 1 n t d  ->  b n s d
        out = out.reshape(n1, b, s1, out.shape[-1]).transpose(0, 1)
        x_max = x_max.reshape(n1, b, s1).transpose(1, 0)
        x_sum = x_sum.reshape(n1, b, s1).transpose(1, 0)

    return out, x_max, x_sum
