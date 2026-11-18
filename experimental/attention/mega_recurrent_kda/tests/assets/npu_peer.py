# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent NPU small-op reference; no DSL/operator or golden imports."""

import torch
import os
import torch_npu


def _device_copy(value, device):
    if not isinstance(value, torch.Tensor):
        return value
    if (
        value.device.type in ("npu", "privateuseone")
        and torch_npu.get_npu_format(value) == 29
    ):
        value = torch_npu.npu_format_cast(value, 2)
    return value.to(device).clone()


def _project(value, weight, output_dtype=torch.bfloat16):
    """BF16 inputs with FP32 NPU matmul accumulation, then the ABI cast."""
    product = torch.mm(
        value.reshape(-1, value.shape[-1]),
        weight.transpose(0, 1),
        out_dtype=torch.float32,
    )
    return product.to(output_dtype).view(*value.shape[:-1], weight.shape[0])


@torch.no_grad()
def run(
    hidden_states,
    qkv_projection_weight,
    decay_projection_a_weight,
    decay_projection_b_weight,
    beta_projection_weight,
    output_gate_projection_weight,
    output_norm_weight,
    output_projection_weight,
    conv1d_weight,
    conv_state,
    checkpoint_state,
    replay_u,
    replay_k,
    replay_decay,
    state_indices,
    conv_num_accepted_tokens,
    A_log,
    dt_bias,
    scale=128**-0.5,
    lower_bound=-5.0,
    rms_norm_eps=1e-5,
    cu_seqlens=None,
):
    """Return five NPU outputs, preserving caller inputs and unselected PA rows."""
    if os.environ.get("KDA_ARCH", "arch35") in ("arch35", "arch92"):
        arguments = locals().copy()
        from ._step_aligned import run_aligned

        return run_aligned(
            **arguments,
            device_kind="npu",
            architecture=os.environ.get("KDA_ARCH", "arch35"),
        )
    device = (
        hidden_states.device
        if hidden_states.device.type in ("npu", "privateuseone")
        else torch.device("npu", torch.npu.current_device())
    )
    values = (
        hidden_states,
        qkv_projection_weight,
        decay_projection_a_weight,
        decay_projection_b_weight,
        beta_projection_weight,
        output_gate_projection_weight,
        output_norm_weight,
        output_projection_weight,
        conv1d_weight,
        conv_state,
        checkpoint_state,
        replay_u,
        replay_k,
        replay_decay,
        state_indices,
        conv_num_accepted_tokens,
        A_log,
        dt_bias,
    )
    a = [_device_copy(v, device) for v in values]
    (
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        wc,
        conv,
        checkpoint,
        u,
        rk,
        rd,
        indices,
        counts,
        alog,
        bias,
    ) = a
    n = int(wbeta.shape[0])
    width = n * 128
    batch = int(indices.numel())
    prefix = (
        cu_seqlens.detach().cpu().tolist()
        if cu_seqlens is not None
        else [i * hidden_states.shape[1] for i in range(batch + 1)]
    )
    slots = indices.detach().cpu().tolist()
    accepted = counts.detach().cpu().tolist()
    if tuple(wc.shape) != (4, 3 * width):
        if wc.ndim == 3:
            wc = wc[:, None, :, :].expand(3, n, 4, 128)
        wc = wc.permute(2, 0, 1, 3).reshape(4, 3 * width).contiguous()
    wc = wc.float()
    x = x.reshape(-1, hidden_states.shape[-1])
    # Device scalar casts mirror the public FP32 scalar ABI.
    scale = float(torch.tensor(float(scale), dtype=torch.float32).item())
    lower_bound = float(torch.tensor(float(lower_bound), dtype=torch.float32).item())
    rms_norm_eps = float(torch.tensor(float(rms_norm_eps), dtype=torch.float32).item())
    outputs = []
    for b in range(batch):
        slot = int(slots[b])
        begin, end = map(int, prefix[b : b + 2])
        length = end - begin
        tokens = x[begin:end]
        raw = _project(tokens, wqkv, torch.float32)
        offset = max(int(accepted[b]) - 1, 0)
        history = conv[slot, offset : offset + 3].float().clone()
        selected_history = conv[slot, offset + 1 : offset + 3].clone()
        convolved = []
        for t in range(length):
            current = raw[t]
            taps = torch.cat((history, current.unsqueeze(0)), dim=0) * wc
            acc = taps[0] + taps[1] + (taps[2] + taps[3])
            convolved.append(torch.nn.functional.silu(acc).bfloat16())
            history = torch.cat(
                (history[1:], current.bfloat16().float().unsqueeze(0)), dim=0
            )
        conv[slot, :2].copy_(selected_history)
        conv[slot, 2 : 2 + length].copy_(raw)
        q, k, v = torch.stack(convolved).view(length, 3, n, 128).float().unbind(1)
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        fa = _project(tokens, wa)
        fb = _project(fa, wb).float().view(length, n, 128)
        decay = torch.exp(
            lower_bound
            * torch.sigmoid(
                alog.float().exp().view(1, n, 1) * (fb + bias.float().view(1, n, 128))
            )
        )
        beta = _project(tokens, wbeta).float().sigmoid()
        gate = _project(tokens, wz).view(length, n, 128).float().sigmoid()
        state = checkpoint[slot].float().clone()
        recurrent = []
        deltas = []
        for t in range(length):
            state = state * decay[t].unsqueeze(-2)
            delta = (v[t] - (state * k[t].unsqueeze(-2)).sum(-1)) * beta[t].unsqueeze(
                -1
            )
            deltas.append(delta)
            state = state + delta.unsqueeze(-1) * k[t].unsqueeze(-2)
            recurrent.append((state * q[t].unsqueeze(-2)).sum(-1) * scale)
        recurrent = torch.stack(recurrent)
        norm = (
            recurrent
            * torch.rsqrt(recurrent.square().mean(-1, keepdim=True) + rms_norm_eps)
            * gamma.float()
        )
        gated = (norm * gate).bfloat16().flatten(1)
        outputs.append(_project(gated, wo))
        u[slot, :, :length].copy_(torch.stack(deltas).transpose(0, 1))
        rk[slot, :, :length].copy_(k.transpose(0, 1).half())
        rd[slot, :, :length].copy_(decay.transpose(0, 1))
    torch.npu.synchronize()
    result = (torch.cat(outputs).view(hidden_states.shape), conv, u, rk, rd)
    assert all(v.device.type in ("npu", "privateuseone") for v in result)
    return result
