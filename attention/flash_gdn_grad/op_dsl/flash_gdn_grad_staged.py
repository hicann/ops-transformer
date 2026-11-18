# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Staged Cube/Vector backward over all independent chunks."""

import torch

from .flash_gdn_grad_recompute import (
    CHUNK_SIZE,
    add,
    mul,
    pack,
    reduce_last,
    scale,
    sub,
    transpose,
    unpack,
)
from .flash_gdn_grad_recompute import (
    cube_matmul as mm,
)


def run_staged(q, k, v, do, dht, cache, scale_value):
    groups, tokens, _ = q.shape
    chunks = (tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    q, k, v, do = (pack(t, chunks) for t in (q, k, v, do))
    e, r, c = cache.e, cache.r, cache.c
    beta_decay = mul(cache.beta, e)
    dq_h = mul(scale(e, scale_value), do)

    # Stage 1. All chunks can form the output-gradient terms concurrently.
    d_aqk = mm(do, transpose(cache.corrected))
    q_term = mm(transpose(q), dq_h)
    x_term = mm(transpose(cache.aqk), do)

    # Stage 2. The reverse scan keeps one state-gradient plane per owner.
    rhs_grads, outgoing = [], []
    dh = dht
    for index in range(chunks - 1, -1, -1):
        sl = slice(index * groups, (index + 1) * groups)
        outgoing.append(dh)
        dx = add(x_term[sl], mul(r[sl], mm(k[sl], dh)))
        d_rhs = mm(transpose(cache.inverse[sl]), dx)
        rhs_grads.append(d_rhs)
        d_kh = mul(scale(beta_decay[sl], -1.0), d_rhs)
        dh = add(add(mul(c[sl], dh), q_term[sl]), mm(transpose(k[sl]), d_kh))

    # Stage 3. Reunite the chunk-local derivatives and the scanned state.
    dh_out = torch.cat(outgoing[::-1])
    d_rhs = torch.cat(rhs_grads[::-1])
    d_l = scale(mm(d_rhs, transpose(cache.corrected)), -1.0)
    d_qk = scale(mul(cache.decay, d_aqk), scale_value)
    d_kk = mul(cache.beta, mul(cache.strict_decay, d_l))
    d_kgb = scale(mm(d_rhs, transpose(cache.h_in)), -1.0)
    dq = add(mm(dq_h, transpose(cache.h_in)), mm(d_qk, k))
    dk = add(
        add(mul(beta_decay, d_kgb), mm(mul(r, cache.corrected), transpose(dh_out))),
        add(mm(transpose(d_qk), q), mm(add(d_kk, transpose(d_kk)), k)),
    )
    dv = mul(cache.beta, d_rhs)
    db = add(
        add(reduce_last(mul(d_rhs, v)), mul(e, reduce_last(mul(d_kgb, k)))),
        reduce_last(mul(d_l, mul(cache.kk, cache.strict_decay))),
    )

    d_pair = add(mul(d_qk, cache.qk), mul(d_kk, cache.kk))
    d_r = mul(r, reduce_last(mul(mm(k, dh_out), cache.corrected)))
    d_prefix = add(
        reduce_last(mul(dq_h, mm(q, cache.h_in))),
        mul(beta_decay, reduce_last(mul(d_kgb, k))),
    )
    row_sum = reduce_last(d_pair)
    column_sum = reduce_last(transpose(d_pair))
    d_prefix = add(sub(d_prefix, d_r), sub(row_sum, column_sum))
    state_dot = reduce_last(reduce_last(mul(dh_out, cache.h_in)).transpose(1, 2))
    d_last = add(reduce_last(transpose(d_r)), mul(c, state_dot))
    dg_step = add(transpose(reduce_last(transpose(d_prefix), "suffix")), d_last)

    # Stage 4. Return token planes for layout restoration and head reductions.
    dg = mul(dg_step, cache.gate_derivative)
    dbeta = mul(db, cache.beta_derivative)
    d_a = mul(dg_step, cache.gate_step)
    return (*(unpack(t, groups, tokens) for t in (dq, dk, dv, dg, dbeta, d_a)), dh)
