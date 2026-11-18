# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU FP32 golden for Mega KDA."""

from __future__ import annotations


import torch
import os


HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
# Q/K RMS normalisation epsilon is a compile-time constant in the sample
# (``normalize_qk`` uses ``1e-06``); the output RMSNorm uses the runtime scalar.
QK_RMS_EPSILON = 1e-6

_FP32 = torch.float32
_BF16 = torch.bfloat16


def _fp32_runtime_scalar(value) -> float:
    """Return the FP32 value the device actually receives for a runtime scalar."""
    return float(torch.tensor(float(value), dtype=_FP32).item())


# Runtime FP32 value of the fixed Q/K normalisation epsilon, then used as an
# exact FP32 constant (the device adds it in FP32 arithmetic).
_QK_RMS_EPSILON_FP32 = _fp32_runtime_scalar(QK_RMS_EPSILON)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"mega_recurrent_kda cpu_fp32 golden: {message}")


def _project(
    value: torch.Tensor, weight: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    """``value @ weight.T`` accumulated in FP32, rounded once at the storage dtype.

    ``weight`` uses the operator's ``[out, in]`` layout.  This is the peer's
    single projection primitive; ``output_dtype`` selects the declared contract
    boundary (FP32 for the raw QKV workspace, BF16 for the four BF16 projections
    and for the final ABI output).
    """
    accumulated = torch.matmul(value.to(_FP32), weight.to(_FP32).transpose(-1, -2))
    return accumulated.to(output_dtype)


def _causal_conv1d_silu(
    raw_qkv: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
):
    """Pooled K=4 causal Conv1D + SiLU with the operator's Conv-state semantics.

    The window products and the SiLU run in FP32; the only rounding points are the
    BF16 Conv output and the BF16 history rows (a FP32 ``raw_qkv`` value is
    rounded to BF16 when it becomes history).  In this private dense helper, slot 0 is an unused null slot:
    it produces an all-zero sequence and keeps its Conv state rows unchanged.  All
    other selected slots receive the last two selected history rows at ``[0:2]``
    and the raw QKV proposals at ``[2:S+2]``; untouched rows/slots are preserved.

    Returns ``(convolved_bf16 [B,S,3P], conv_state_new [C,L,3P])``.
    """
    if raw_qkv.ndim != 3:
        raise ValueError(
            "mega_recurrent_kda cpu_fp32 golden: raw_qkv must be [B,S,3*P]"
        )
    batch_size, seq_len, channels = map(int, raw_qkv.shape)
    _require(
        tuple(conv1d_weight.shape) == (CONV_KERNEL_SIZE, channels),
        "conv1d_weight must be [4,3*P]",
    )
    _require(
        conv_state.ndim == 3 and int(conv_state.shape[2]) == channels,
        "conv_state must be [pool,state_length,3*P]",
    )
    _require(
        int(conv_state.shape[1]) >= seq_len + CONV_KERNEL_SIZE - 2,
        "conv_state state_length must be at least S+2",
    )
    _require(
        tuple(conv_state_indices.shape) == (batch_size,),
        "conv_state_indices must be [B]",
    )
    _require(
        tuple(conv_num_accepted_tokens.shape) == (batch_size,),
        "conv_num_accepted_tokens must be [B]",
    )

    convolved = torch.zeros((batch_size, seq_len, channels), dtype=_BF16)
    conv_state_new = conv_state.clone()
    filter_fp32 = conv1d_weight.to(_FP32)

    for batch_index in range(batch_size):
        cache_index = int(conv_state_indices[batch_index])
        if cache_index == 0:
            # Null slot: zero Conv output for every proposal token, state untouched.
            continue
        accepted = int(conv_num_accepted_tokens[batch_index])
        offset = max(accepted - 1, 0)
        history_stop = offset + CONV_KERNEL_SIZE - 1
        # The selected history window is read from the *original* pool (never from
        # writes of this call), mirroring the device's snapshot semantics.
        history = conv_state[cache_index, offset:history_stop].to(_FP32)
        _require(
            int(history.shape[0]) == CONV_KERNEL_SIZE - 1,
            "accepted Conv history exceeds state_length",
        )
        for token in range(seq_len):
            current = raw_qkv[batch_index, token].to(_FP32)
            window = torch.cat((history, current.unsqueeze(0)), dim=0)
            taps = window * filter_fp32
            convolved_fp32 = taps[0] + taps[1] + (taps[2] + taps[3])
            # SiLU in FP32, then the BF16 Conv-output storage boundary.
            convolved[batch_index, token] = (
                convolved_fp32 * torch.sigmoid(convolved_fp32)
            ).to(_BF16)
            # The FP32 raw QKV value enters the BF16 history rounded to BF16.
            history = torch.cat(
                (history[1:], current.to(_BF16).to(_FP32).unsqueeze(0)), dim=0
            )
        selected = conv_state[cache_index, offset:history_stop]
        conv_state_new[cache_index, :2].copy_(selected[1:])
        proposal_stop = 2 + seq_len
        conv_state_new[cache_index, 2:proposal_stop].copy_(raw_qkv[batch_index])
    return convolved, conv_state_new


@torch.no_grad()
def _dense_golden_fp32(
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
    recurrent_state,
    replay_u,
    replay_k,
    replay_decay,
    conv_state_indices,
    conv_num_accepted_tokens,
    A_log,
    dt_bias,
    scale=128**-0.5,
    lower_bound=-5.0,
    rms_norm_eps=1e-5,
    **kwargs,
):
    t = (
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
        recurrent_state,
        replay_u,
        replay_k,
        replay_decay,
        conv_state_indices,
        conv_num_accepted_tokens,
        A_log,
        dt_bias,
    )
    assert all(x.device.type == "cpu" for x in t)
    scale, lower_bound, rms_norm_eps = map(
        _fp32_runtime_scalar, (scale, lower_bound, rms_norm_eps)
    )
    b, s, h = t[0].shape
    n = t[4].shape[0]
    f32 = torch.float32
    bf16 = torch.bfloat16
    project = _project
    raw_qkv = project(t[0], t[1], torch.float32)
    qkv, conv = _causal_conv1d_silu(raw_qkv, t[8], t[9], t[14], t[15])
    qkv = qkv.view(b, s, 3, n, 128)
    q, k, v = (qkv[:, :, i].to(f32) for i in range(3))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + _QK_RMS_EPSILON_FP32)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + _QK_RMS_EPSILON_FP32)
    raw_decay = project(project(t[0], t[2], bf16), t[3], bf16).view(b, s, n, 128)
    decay = torch.exp(
        lower_bound
        * torch.sigmoid(
            torch.exp(t[16].to(f32)).view(1, 1, n, 1)
            * (raw_decay.to(f32) + t[17].to(f32).view(1, 1, n, 128))
        )
    )
    beta = torch.sigmoid(project(t[0], t[4], bf16).to(f32))
    gate = project(t[0], t[5], bf16).view(b, s, n, 128)
    current = t[10].to(f32).clone()
    recurrent = torch.empty((b, s, n, 128), dtype=f32)
    u = torch.empty((b, s, n, 128), dtype=torch.float32)
    for token in range(s):
        current = current * decay[:, token].unsqueeze(-2)
        delta = (v[:, token] - (current * k[:, token].unsqueeze(-2)).sum(-1)) * beta[
            :, token
        ].unsqueeze(-1)
        u[:, token] = delta.to(
            torch.float32
        )  # record storage does not round the register carry
        current = current + delta.unsqueeze(-1) * k[:, token].unsqueeze(-2)
        recurrent[:, token] = (current * q[:, token].unsqueeze(-2)).sum(-1) * scale
    normalized = (
        recurrent
        * torch.rsqrt(recurrent.square().mean(-1, keepdim=True) + rms_norm_eps)
        * t[6].to(f32)
    )
    gated = (normalized * torch.sigmoid(gate.to(f32))).to(bf16)
    out = project(gated.flatten(2), t[7], bf16)
    return [out, conv, u, k.to(torch.float16), decay.to(torch.float32)]


@torch.no_grad()
def _run_fp32(
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
    t = [
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
    ]
    assert all(x.device.type == "cpu" for x in t)
    n = t[4].shape[0]
    p = n * 128
    b = state_indices.numel()
    x = hidden_states.reshape(-1, 7168)
    prefix = (
        cu_seqlens.tolist()
        if cu_seqlens is not None
        else [i * hidden_states.shape[1] for i in range(b + 1)]
    )
    # TTK Promote may widen floating inputs before calling the golden.
    # Restore the declared storage dtypes at all mutable ABI boundaries.
    conv = conv_state.to(torch.bfloat16).clone()
    u = replay_u.to(torch.float32).clone()
    k = replay_k.to(torch.float16).clone()
    decay = replay_decay.to(torch.float32).clone()
    out = []
    wc = conv1d_weight
    if wc.shape != (4, 3 * p):
        if wc.ndim == 3:
            wc = wc[:, None, :, :].expand(3, n, 4, 128)
        wc = wc.permute(2, 0, 1, 3).reshape(4, 3 * p).contiguous()
    # Convert immutable operands once, rather than recasting the same matrix for
    # every request. Conversions are exact from BF16/FP32 and formulas are unchanged.
    compute_dtype = torch.float32
    x = x.to(compute_dtype)
    immutable = [v.to(compute_dtype) for v in t[1:9]]
    wc = wc.to(compute_dtype)
    for batch in range(b):
        slot = int(state_indices[batch])
        start, end = prefix[batch : batch + 2]
        length = end - start
        pool = torch.zeros((2, conv.shape[1], 3 * p), dtype=conv.dtype)
        pool[1].copy_(conv_state[slot])
        args = [
            x[start:end].unsqueeze(0),
            *immutable,
            pool,
            checkpoint_state[slot].unsqueeze(0),
            torch.empty((1, length, n, 128), dtype=torch.float32),
            torch.empty((1, length, n, 128), dtype=torch.float16),
            torch.empty((1, length, n, 128), dtype=torch.float32),
            torch.ones(1, dtype=torch.int32),
            conv_num_accepted_tokens[batch : batch + 1],
            A_log,
            dt_bias,
        ]
        args[8] = wc
        result = _dense_golden_fp32(*args, scale, lower_bound, rms_norm_eps)
        out.append(result[0].squeeze(0))
        conv[slot].copy_(result[1][1])
        for target, value in zip((u, k, decay), result[2:]):
            target[slot, :, :length].copy_(value[0].transpose(0, 1))
    return torch.cat(out).view(hidden_states.shape), conv, u, k, decay


def run_fp32(*args, **kwargs):
    with torch.device("cpu"):
        if os.environ.get("KDA_ARCH", "arch35") in ("arch35", "arch92"):
            from ._step_aligned import run_aligned

            return run_aligned(
                *args,
                **kwargs,
                device_kind="cpu",
                architecture=os.environ.get("KDA_ARCH", "arch35"),
            )
        return _run_fp32(*args, **kwargs)
