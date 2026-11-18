# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent FP64 per-token GDN recurrence for the TTK golden.

State is [B,Nv,V,K]; scalar decay and beta broadcast over the last two axes.
This reference never imports a compute kernel or calls the DUT.
"""

import torch
from gdn_ttk_ops import load_asset


def reference(inputs, attributes):
    helper = load_asset("inputs")
    c = helper.configuration(inputs, attributes)
    f = c["flags"]
    values = [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in inputs]
    q, k, v, g, beta, _, initial, a_log, bias = values[:9]
    sequences = [helper.sequences(x.double(), c) for x in (q, k, v, g, beta)]
    out = torch.zeros_like(v, dtype=torch.float64)
    final = torch.zeros((c["batch"], c["hv"], c["dv"], c["dk"]), dtype=torch.float64)
    chunks = (
        torch.zeros(
            load_asset("compare").shapes(inputs, attributes)[2], dtype=torch.float64
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
                bias.double()
                if attributes.get("dt_bias_mode", "none") != "none"
                else 0.0
            )
            gs = -a_log.double().exp() * torch.nn.functional.softplus(gs + bias_value)
        if f["use_beta_sigmoid_in_kernel"]:
            bs = bs.sigmoid() * (2 if f["allow_neg_eigval"] else 1)
        group = c["hv"] // c["hq"]
        qs, ks = [x.repeat_interleave(group, dim=1) for x in (qs, ks)]
        state = (
            initial[b].double().clone()
            if attributes.get("initial_state_mode", "none") in ("random", "zero")
            else torch.zeros_like(final[b])
        )
        outputs = []
        for t in range(length):
            if chunks is not None and t % 64 == 0:
                if c["layout"] == "TND":
                    chunks[chunk_base + t // 64] = state
                else:
                    chunks[b, t // 64] = state
            state = state * gs[t].exp()[:, None, None]
            residual = vs[t] - torch.einsum("hvk,hk->hv", state, ks[t])
            state = state + (bs[t, :, None] * residual)[:, :, None] * ks[t, :, None, :]
            outputs.append(torch.einsum("hvk,hk->hv", state, qs[t]) * scale)
        tokens = torch.stack(outputs)
        if c["layout"] == "TND":
            out[c["cu"][b] : c["cu"][b + 1]] = tokens
        elif c["layout"] == "BNSD":
            out[b, :, :length] = tokens.transpose(0, 1)
        else:
            out[b, :length] = tokens
        final[b] = state
        chunk_base += (length + 63) // 64
    # cross_check promotes BF16 outputs to FP32 and FP32 states to FP64.
    # Keep that extra state precision; rounding it back would erase the peer's
    # FP32 error and make the three-way comparison degenerate.
    state_dtype = (
        torch.float64 if attributes.get("golden_mode") == "Promote" else torch.float32
    )
    return (
        out.to(v.dtype),
        final.to(state_dtype) if f["output_final_state"] else None,
        chunks.to(state_dtype) if chunks is not None else None,
    ) + (None,) * 13


def golden(
    q,
    k,
    v,
    g,
    beta,
    metadata,
    initial_state,
    A_log,
    dt_bias,
    cu_seqlens,
    seqused,
    **kwargs,
):
    return load_asset("compare").pack(
        reference(
            [
                q,
                k,
                v,
                g,
                beta,
                metadata,
                initial_state,
                A_log,
                dt_bias,
                cu_seqlens,
                seqused,
            ],
            kwargs,
        )
    )
