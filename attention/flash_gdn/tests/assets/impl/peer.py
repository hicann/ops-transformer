# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FP32 CPU chunk peer for native TTK cross_check/L2.

This is an independent Torch implementation of scalar-gate chunk algebra,
using a triangular solve rather than the DUT's DSL forward-substitution loops.
All intermediates are FP32, matching GDN's current vector arithmetic budget;
only the public out is rounded to BF16. No KDA compute code, installed operator,
device kernel, artificial noise, or observed DUT result is used here.
"""

import torch
from gdn_ttk_ops import load_asset


def _chunk(q, k, v, g, beta, state, scale):
    # Head-major inputs [H,C,D], scalar gate/beta [H,C], state [H,V,K].
    count = q.shape[1]
    cumulative = g.cumsum(dim=1)
    # Accumulate each causal interval independently. Subtracting two large
    # prefixes would lose small decays following a gate such as -1e8.
    log_decay = torch.full(
        (q.shape[0], count, count), float("-inf"), dtype=torch.float32
    )
    for column in range(count):
        log_decay[:, column, column] = 0.0
        remaining = slice(column + 1, None)
        log_decay[:, remaining, column] = g[:, remaining].cumsum(dim=1)
    decay = log_decay.exp()
    lower = ((k @ k.transpose(-1, -2)) * decay * beta[:, :, None]).tril(-1)
    system = lower + torch.eye(count, dtype=torch.float32)
    u = torch.linalg.solve_triangular(
        system, beta[:, :, None] * v, upper=False, unitriangular=True
    )
    w = torch.linalg.solve_triangular(
        system,
        (beta * cumulative.exp())[:, :, None] * k,
        upper=False,
        unitriangular=True,
    )
    residual = u - w @ state.transpose(-1, -2)
    attention = (q @ k.transpose(-1, -2)) * decay * scale
    out = (q @ state.transpose(-1, -2)) * cumulative.exp()[
        :, :, None
    ] * scale + attention @ residual
    suffix = torch.zeros_like(g)
    suffix[:, :-1] = g[:, 1:].flip(1).cumsum(dim=1).flip(1)
    restored_k = k * suffix.exp()[:, :, None]
    final = (
        state * cumulative[:, -1].exp()[:, None, None]
        + residual.transpose(-1, -2) @ restored_k
    )
    return out, final


def reference(inputs, attributes):
    helper = load_asset("inputs")
    c = helper.configuration(inputs, attributes)
    f = c["flags"]
    values = [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in inputs]
    q, k, v, g, beta, _, initial, a_log, bias = values[:9]
    sequences = [helper.sequences(x.float(), c) for x in (q, k, v, g, beta)]
    out = torch.zeros_like(v, dtype=torch.float32)
    final = torch.zeros((c["batch"], c["hv"], c["dv"], c["dk"]), dtype=torch.float32)
    chunks = (
        torch.zeros(
            load_asset("compare").shapes(inputs, attributes)[2], dtype=torch.float32
        )
        if f["return_intermediate_states"]
        else None
    )
    scale = c["dk"] ** -0.5 if f["scale"] is None else f["scale"]
    chunk_base = 0
    for b, length in enumerate(c["lengths"]):
        qs, ks, vs, gs, bs = [items[b] for items in sequences]
        if f["use_qk_l2norm_in_kernel"]:
            qs, ks = [
                x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)
                for x in (qs, ks)
            ]
        if f["use_gate_in_kernel"]:
            bias_value = (
                bias.float()
                if attributes.get("dt_bias_mode", "none") != "none"
                else 0.0
            )
            gs = -a_log.float().exp() * torch.nn.functional.softplus(gs + bias_value)
        if f["use_beta_sigmoid_in_kernel"]:
            bs = bs.sigmoid() * (2 if f["allow_neg_eigval"] else 1)
        group = c["hv"] // c["hq"]
        qs, ks = [x.repeat_interleave(group, dim=1) for x in (qs, ks)]
        state = (
            initial[b].float().clone()
            if attributes.get("initial_state_mode", "none") in ("random", "zero")
            else torch.zeros_like(final[b])
        )
        for start in range(0, length, 64):
            if chunks is not None:
                if c["layout"] == "TND":
                    chunks[chunk_base + start // 64] = state
                else:
                    chunks[b, start // 64] = state
            end = min(start + 64, length)
            parts = [x[start:end].transpose(0, 1) for x in (qs, ks, vs, gs, bs)]
            tokens, state = _chunk(*parts, state, scale)
            tokens = tokens.transpose(0, 1)
            if c["layout"] == "TND":
                out[c["cu"][b] + start : c["cu"][b] + end] = tokens
            elif c["layout"] == "BNSD":
                out[b, :, start:end] = tokens.transpose(0, 1)
            else:
                out[b, start:end] = tokens
        final[b] = state
        chunk_base += (length + 63) // 64
    return (out.bfloat16(), final if f["output_final_state"] else None, chunks) + (
        None,
    ) * 13
