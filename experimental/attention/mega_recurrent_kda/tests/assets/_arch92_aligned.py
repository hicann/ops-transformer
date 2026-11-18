# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU/NPU reference for the arch92 sequence stages.

Projection tiles use complete-K mm with BF16 operands and FP32 accumulation;
the NPU peer assigns each tile one AIC and one AIV. Vector formulas follow
SequenceConvStage, _RecurrentMath and OutputGateStage, including the two
64-lane segments, multiply-add boundaries and query scaling before reduction.
Backend reduction trees and transcendental approximations remain independent:
source alignment does not imply bitwise equality or replace device validation.
No DUT intermediates, compensation, FP64 or adjusted thresholds are used.
"""

import torch

from ._step_aligned import tile_fullk_mm


def _half_sum(value):
    """Two 64-lane reductions, then an FP32 addition."""
    return value[..., :64].sum(-1, keepdim=True) + value[..., 64:].sum(-1, keepdim=True)


def _folded_dot(state, vector):
    """Multiply the high half, multiply-add the low half, then reduce 64 lanes."""
    high = state[..., 64:] * vector[..., 64:].unsqueeze(-2)
    return torch.addcmul(high, state[..., :64], vector[..., :64].unsqueeze(-2)).sum(-1)


def _sigmoid(value):
    denominator = torch.exp(-value) + 1.0
    return torch.div(torch.ones_like(denominator), denominator)


@torch.no_grad()
def evaluate_arch92(prepared):
    v, c, kind = prepared["a"], prepared["c"], prepared["kind"]
    dev = v["hidden_states"].device
    batch, heads = c["B"], c["N"]
    width, steps = heads * 128, max(c["lengths"])
    total = sum(c["lengths"])
    lengths = torch.tensor(c["lengths"], device=dev, dtype=torch.long)
    prefix = torch.cat(
        (torch.zeros(1, device=dev, dtype=torch.long), lengths.cumsum(0))
    )
    times = torch.arange(steps, device=dev)
    valid = times.unsqueeze(0) < lengths.unsqueeze(1)
    positions = (prefix[:-1, None] + times).clamp(max=total - 1)
    slots = v["state_indices"].long()
    offsets = (v["conv_num_accepted_tokens"].long() - 1).clamp(min=0)
    wc = v["conv1d_weight"]
    if tuple(wc.shape) != (4, 3 * width):
        if wc.ndim == 3:
            wc = wc[:, None, :, :].expand(3, heads, 4, 128)
        wc = wc.permute(2, 0, 1, 3).reshape(4, 3 * width).contiguous()
    wc = wc.float()
    history = (
        v["conv_state"][slots[:, None], offsets[:, None] + torch.arange(3, device=dev)]
        .float()
        .clone()
    )
    selected = history[:, 1:3].bfloat16().clone()
    raw = prepared["raw"][positions]
    convolved, preacc, silu_values = [], [], []
    for token in range(steps):
        current = raw[:, token]
        acc01 = torch.addcmul(history[:, 1] * wc[1], history[:, 0], wc[0])
        acc23 = torch.addcmul(current * wc[3], history[:, 2], wc[2])
        acc = acc01 + acc23
        activated = acc / (torch.exp(-acc) + 1.0)
        convolved.append(activated.bfloat16())
        preacc.append(acc)
        silu_values.append(activated)
        next_history = torch.cat(
            (history[:, 1:], current.bfloat16().float().unsqueeze(1)), dim=1
        )
        history = torch.where(valid[:, token, None, None], next_history, history)
    convolved = torch.stack(convolved, dim=1)
    q, k, value = convolved.view(batch, steps, 3, heads, 128).float().unbind(2)
    q = torch.div(q, torch.sqrt(_half_sum(q.square()) + 1e-6)) * float(c["scale"])
    k = torch.div(k, torch.sqrt(_half_sum(k.square()) + 1e-6))
    fb = prepared["fb"][positions].float().view(batch, steps, heads, 128)
    negative_alpha = -v["A_log"].float().exp()
    gate_logit = negative_alpha.view(1, 1, heads, 1) * (
        fb + v["dt_bias"].float().view(1, 1, heads, 128)
    )
    gate_denominator = torch.exp(gate_logit) + 1.0
    decay = torch.exp(
        torch.div(torch.ones_like(gate_denominator), gate_denominator)
        * float(c["lower_bound"])
    )
    beta = _sigmoid(prepared["beta"][positions].float())
    state = v["checkpoint_state"][slots].float().clone()
    deltas, recurrent = [], []
    for token in range(steps):
        decay_key = decay[:, token] * k[:, token]
        delta = (value[:, token] - _folded_dot(state, decay_key)) * beta[
            :, token
        ].unsqueeze(-1)
        updates = delta.unsqueeze(-1) * k[:, token].unsqueeze(-2)
        updated = torch.addcmul(updates, state, decay[:, token].unsqueeze(-2))
        deltas.append(delta)
        recurrent.append(_folded_dot(updated, q[:, token]))
        state = torch.where(valid[:, token, None, None, None], updated, state)
    u = torch.stack(deltas, dim=1)
    recurrent = torch.stack(recurrent, dim=1)
    mean_square = _half_sum(recurrent.square()) * (1.0 / 128)
    root = torch.sqrt(mean_square + float(c["rms_norm_eps"]))
    inverse_rms = torch.div(torch.ones_like(root), root)
    normalized = (recurrent * inverse_rms) * v["output_norm_weight"].float()

    def mm(left, right):
        return tile_fullk_mm(left, right, m=128, n=128, device_kind=kind, vector_num=1)

    # Gate and output projections also follow the global padded M dimension,
    # rather than starting a new GEMM at each sequence boundary.
    gate32 = mm(
        v["hidden_states"].reshape(total, 7168), v["output_gate_projection_weight"]
    )
    gate = _sigmoid(gate32.bfloat16().float().view(total, heads, 128))
    gated = (normalized[valid] * gate).bfloat16().flatten(1)
    output = mm(gated, v["output_projection_weight"]).bfloat16()
    conv = v["conv_state"].clone()
    up, kp, dp = (v[name].clone() for name in ("replay_u", "replay_k", "replay_decay"))
    conv[slots, :2] = selected
    for token in range(steps):
        mask = valid[:, token]
        chosen = slots[mask]
        conv[chosen, 2 + token] = raw[mask, token].bfloat16()
        up[chosen, :, token] = u[mask, token]
        kp[chosen, :, token] = k[mask, token].half()
        dp[chosen, :, token] = decay[mask, token]
    outputs = (output.view(v["hidden_states"].shape), conv, up, kp, dp)
    trace = dict(
        qkv_fp32=prepared["raw"],
        fa_fp32=prepared["fa_fp32"],
        fa_bf16=prepared["fa"],
        fb_fp32=prepared["fb_fp32"],
        fb_bf16=prepared["fb"],
        beta_fp32=prepared["beta_fp32"],
        beta_bf16=prepared["beta"],
        conv_acc_fp32=torch.stack(preacc, 1)[valid],
        silu_fp32=torch.stack(silu_values, 1)[valid],
        convolved=convolved[valid],
        gate_fp32=gate32,
    )
    return outputs, trace
