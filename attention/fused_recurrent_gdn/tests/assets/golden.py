# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent FP64 CPU reference for pure recurrent GDN."""

import math
import torch
import torch.nn.functional as F


def recurrent_gdn(
    q,
    k,
    v,
    beta,
    g,
    initial_state=None,
    scale=None,
    *,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    seqused=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    layout="TND",
):
    value_shape = v.shape

    def packed(x):
        x = x.detach().cpu().double()
        if layout == "NTD":
            return x.transpose(0, 1).contiguous()
        if layout == "BNSD":
            x = x.transpose(1, 2)
        return x.flatten(0, 1) if layout in ("BNSD", "BSND") else x

    q, k, v, beta, g = map(packed, (q, k, v, beta, g))
    if use_qk_l2norm_in_kernel:
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta) * (2 if allow_neg_eigval else 1)
    if use_gate_in_kernel:
        bias = 0 if dt_bias is None else dt_bias.cpu().double()
        g = -A_log.cpu().double().exp() * F.softplus(g + bias)
    pool = ssm_state_indices is not None
    if cu_seqlens is not None:
        offsets = cu_seqlens.cpu().tolist()
    elif layout in ("BSND", "BNSD"):
        batch = value_shape[0]
        step = v.shape[0] // batch
        offsets = [b * step for b in range(batch + 1)]
    else:
        offsets = [0, v.shape[0]]
    batch = len(offsets) - 1
    lengths = [b - a for a, b in zip(offsets, offsets[1:])]
    if seqused is not None:
        lengths = seqused.cpu().tolist()
    state_dtype = torch.float32 if initial_state is None else initial_state.dtype
    state_shape = (batch, v.shape[1], q.shape[2], v.shape[2])
    storage = (
        torch.zeros(state_shape, dtype=torch.float64)
        if initial_state is None
        else initial_state.detach().cpu().double().clone()
    )
    indices = None if not pool else ssm_state_indices.cpu().tolist()
    accepted = (
        None if num_accepted_tokens is None else num_accepted_tokens.cpu().tolist()
    )
    final = torch.empty(state_shape, dtype=torch.float64)
    output = torch.zeros_like(v)
    scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    for b in range(batch):
        if not pool:
            source = b
        elif ssm_state_indices.ndim == 1:
            source = indices[b]
        else:
            source = indices[b][accepted[b] - 1]
        state = storage[source].clone()
        for step in range(lengths[b]):
            token = offsets[b] + step
            for h in range(v.shape[1]):
                qh = h // (v.shape[1] // q.shape[1])
                key = k[token, qh]
                transition = torch.exp(g[token, h]) * (
                    torch.eye(q.shape[-1], dtype=torch.float64)
                    - beta[token, h] * torch.outer(key, key)
                )
                state[h] = transition @ state[h] + beta[token, h] * torch.outer(
                    key, v[token, h]
                )
                output[token, h] = scale * torch.mv(state[h].T, q[token, qh])

            if pool:
                destination = (
                    indices[b] if ssm_state_indices.ndim == 1 else indices[b][step]
                )
                storage[destination] = state.to(state_dtype).double()
        final[b] = state
    if pool:
        initial_state.copy_(storage.to(device=initial_state.device, dtype=state_dtype))
    if layout == "NTD":
        output = output.transpose(0, 1).contiguous()
    elif layout in ("BSND", "BNSD"):
        output = output.reshape(batch, v.shape[0] // batch, v.shape[1], v.shape[2])
        if layout == "BNSD":
            output = output.transpose(1, 2).contiguous()
    return output, final.to(state_dtype) if output_final_state and not pool else None
