# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU FP64 and original PyTorch references for Mega KDA."""

from __future__ import annotations

import dataclasses
from contextvars import ContextVar
from typing import NamedTuple

import torch


HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
# Q/K RMS normalisation epsilon is a compile-time constant in the sample
# (``normalize_qk`` uses ``1e-06``); the output RMSNorm uses the runtime scalar.
QK_RMS_EPSILON = 1e-6

_FP64 = torch.float64
_FP32 = torch.float32
_BF16 = torch.bfloat16


def _fp32_runtime_scalar(value) -> float:
    """Return the FP32 value the device actually receives for a runtime scalar."""
    return float(torch.tensor(float(value), dtype=_FP32).item())


# Runtime FP32 value of the fixed Q/K normalisation epsilon, then used as an
# exact FP64 constant (the device adds it in FP32 arithmetic).
_QK_RMS_EPSILON_FP32 = _fp32_runtime_scalar(QK_RMS_EPSILON)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"mega_recurrent_kda cpu_hp peer: {message}")


def _project(
    value: torch.Tensor, weight: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    """``value @ weight.T`` accumulated in FP64, rounded once at the storage dtype.

    ``weight`` uses the operator's ``[out, in]`` layout.  This is the peer's
    single projection primitive; ``output_dtype`` selects the declared contract
    boundary (FP32 for the raw QKV workspace, BF16 for the four BF16 projections
    and for the final ABI output).
    """
    accumulated = torch.matmul(value.to(_FP64), weight.to(_FP64).transpose(-1, -2))
    return accumulated.to(output_dtype)


def _causal_conv1d_silu(
    raw_qkv: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
):
    """Pooled K=4 causal Conv1D + SiLU with the operator's Conv-state semantics.

    The window products and the SiLU run in FP64; the only rounding points are the
    BF16 Conv output and the BF16 history rows (a FP32 ``raw_qkv`` value is
    rounded to BF16 when it becomes history).  Slot 0 is the immutable null slot:
    it produces an all-zero sequence and keeps its Conv state rows unchanged.  All
    other selected slots receive the last two selected history rows at ``[0:2]``
    and the raw QKV proposals at ``[2:S+2]``; untouched rows/slots are preserved.

    Returns ``(convolved_bf16 [B,S,3P], conv_state_new [C,L,3P])``.
    """
    if raw_qkv.ndim != 3:
        raise ValueError("mega_recurrent_kda cpu_hp peer: raw_qkv must be [B,S,3*P]")
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
    filter_fp64 = conv1d_weight.to(_FP64)

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
        history = conv_state[cache_index, offset:history_stop].to(_FP64)
        _require(
            int(history.shape[0]) == CONV_KERNEL_SIZE - 1,
            "accepted Conv history exceeds state_length",
        )
        for token in range(seq_len):
            current = raw_qkv[batch_index, token].to(_FP64)
            window = torch.cat((history, current.unsqueeze(0)), dim=0)
            taps = window * filter_fp64
            convolved_fp64 = taps[0] + taps[1] + (taps[2] + taps[3])
            # SiLU in FP64, then the BF16 Conv-output storage boundary.
            convolved[batch_index, token] = (
                convolved_fp64 * torch.sigmoid(convolved_fp64)
            ).to(_BF16)
            # The FP32 raw QKV value enters the BF16 history rounded to BF16.
            history = torch.cat(
                (history[1:], current.to(_BF16).to(_FP64).unsqueeze(0)), dim=0
            )
        selected = conv_state[cache_index, offset:history_stop]
        conv_state_new[cache_index, :2].copy_(selected[1:])
        proposal_stop = 2 + seq_len
        conv_state_new[cache_index, 2:proposal_stop].copy_(raw_qkv[batch_index])
    return convolved, conv_state_new


@torch.no_grad()
def _dense_golden_fp64(
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
    f64 = torch.float64
    bf16 = torch.bfloat16
    project = _project
    raw_qkv = project(t[0], t[1], torch.float32)
    qkv, conv = _causal_conv1d_silu(raw_qkv, t[8], t[9], t[14], t[15])
    qkv = qkv.view(b, s, 3, n, 128)
    q, k, v = (qkv[:, :, i].to(f64) for i in range(3))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + _QK_RMS_EPSILON_FP32)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + _QK_RMS_EPSILON_FP32)
    raw_decay = project(project(t[0], t[2], bf16), t[3], bf16).view(b, s, n, 128)
    decay = torch.exp(
        lower_bound
        * torch.sigmoid(
            torch.exp(t[16].to(f64)).view(1, 1, n, 1)
            * (raw_decay.to(f64) + t[17].to(f64).view(1, 1, n, 128))
        )
    )
    beta = torch.sigmoid(project(t[0], t[4], bf16).to(f64))
    gate = project(t[0], t[5], bf16).view(b, s, n, 128)
    current = t[10].to(f64).clone()
    recurrent = torch.empty((b, s, n, 128), dtype=f64)
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
        * t[6].to(f64)
    )
    gated = (normalized * torch.sigmoid(gate.to(f64))).to(bf16)
    out = project(gated.flatten(2), t[7], bf16)
    return [out, conv, u, k.to(torch.float16), decay.to(torch.float32)]


@torch.no_grad()
def _run_fp64(
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
    conv = conv_state.clone()
    u = replay_u.clone()
    k = replay_k.clone()
    decay = replay_decay.clone()
    out = []
    wc = conv1d_weight
    if wc.shape != (4, 3 * p):
        if wc.ndim == 3:
            wc = wc[:, None, :, :].expand(3, n, 4, 128)
        wc = wc.permute(2, 0, 1, 3).reshape(4, 3 * p).contiguous()
    # Convert immutable operands once, rather than recasting the same matrix for
    # every request. Conversions are exact from BF16/FP32 and formulas are unchanged.
    compute_dtype = torch.float64
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
        result = _dense_golden_fp64(*args, scale, lower_bound, rms_norm_eps)
        out.append(result[0].squeeze(0))
        conv[slot].copy_(result[1][1])
        for target, value in zip((u, k, decay), result[2:]):
            target[slot, :, :length].copy_(value[0].transpose(0, 1))
    return torch.cat(out).view(hidden_states.shape), conv, u, k, decay


def run_fp64(*args, **kwargs):
    with torch.device("cpu"):
        return _run_fp64(*args, **kwargs)


# Original CPU PyTorch arithmetic and replay record taps.
_ORIGINAL_RECORDS = ContextVar("original_golden_records", default=None)


def _original_record_key(key):
    records = _ORIGINAL_RECORDS.get()
    if records is not None:
        records["k"].copy_(key)


def _original_record_value(name, batch, token, value):
    records = _ORIGINAL_RECORDS.get()
    if records is not None:
        records[name][batch, token].copy_(value)


_ORIGINAL_HEAD_DIM = 128
_ORIGINAL_CONV_KERNEL_SIZE = 4
_ORIGINAL_RMS_EPSILON = 1e-06


@dataclasses.dataclass
class MegaRecurrentKDAWeights:
    """TP-local weights in the layouts consumed by the public operator."""

    qkv_projection_weight: torch.Tensor
    decay_projection_a_weight: torch.Tensor
    decay_projection_b_weight: torch.Tensor
    beta_projection_weight: torch.Tensor
    output_gate_projection_weight: torch.Tensor
    output_norm_weight: torch.Tensor
    output_projection_weight: torch.Tensor
    conv1d_weight: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor


@dataclasses.dataclass
class MegaRecurrentKDAGolden:
    """Outputs and intermediate values used by precision tests."""

    output: torch.Tensor
    raw_qkv: torch.Tensor
    convolved_qkv: torch.Tensor
    decay_a: torch.Tensor
    raw_decay: torch.Tensor
    raw_beta: torch.Tensor
    raw_output_gate: torch.Tensor
    recurrent_output: torch.Tensor
    gated_output: torch.Tensor
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor


class _RecurrentCase(NamedTuple):
    convolved_qkv: torch.Tensor
    raw_decay: torch.Tensor
    raw_beta: torch.Tensor
    recurrent_state: torch.Tensor
    ssm_state_indices: torch.Tensor
    num_accepted_tokens: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    scale: float
    lower_bound: float


class _GoldenCase(NamedTuple):
    hidden_states: torch.Tensor
    weights: MegaRecurrentKDAWeights
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor
    conv_state_indices: torch.Tensor
    ssm_state_indices: torch.Tensor
    conv_num_accepted_tokens: torch.Tensor
    num_accepted_tokens: torch.Tensor
    scale: float
    lower_bound: float
    rms_norm_eps: float


def _original_linear(
    value: torch.Tensor, weight: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    result = torch.matmul(value.float(), weight.float().transpose(-1, -2))
    return result.to(output_dtype)


def _original_conv1d_silu(
    raw_qkv: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
) -> torch.Tensor:
    """Run pooled K=4 Conv with the raw-FP32/BF16-history boundary.

    ``conv_state_indices == 0`` denotes the immutable null slot. All proposal
    tokens are evaluated; the accepted count chooses the three history rows
    from the model-native speculative cache.
    """
    if raw_qkv.ndim != 3:
        raise ValueError("raw_qkv must be [B,S,3*P]")
    batch_size, seq_len, channels = map(int, raw_qkv.shape)
    if tuple(conv1d_weight.shape) != (_ORIGINAL_CONV_KERNEL_SIZE, channels):
        raise ValueError("conv1d_weight must be [4,3*P]")
    if conv_state.ndim != 3 or int(conv_state.shape[2]) != channels:
        raise ValueError("conv_state must be [pool,state_length,3*P]")
    if int(conv_state.shape[1]) < seq_len + _ORIGINAL_CONV_KERNEL_SIZE - 2:
        raise ValueError("conv_state state_length must be at least S+2")
    if tuple(conv_state_indices.shape) != (batch_size,):
        raise ValueError("conv_state_indices must be [B]")
    if tuple(conv_num_accepted_tokens.shape) != (batch_size,):
        raise ValueError("conv_num_accepted_tokens must be [B]")
    result = torch.zeros_like(raw_qkv, dtype=torch.bfloat16)
    original_state = conv_state.clone()
    filter_fp32 = conv1d_weight.float()
    for batch_index in range(batch_size):
        cache_index = int(conv_state_indices[batch_index])
        if cache_index == 0:
            continue
        accepted = int(conv_num_accepted_tokens[batch_index])
        offset = max(accepted - 1, 0)
        history_stop = offset + _ORIGINAL_CONV_KERNEL_SIZE - 1
        history = original_state[cache_index, offset:history_stop].float().clone()
        if int(history.shape[0]) != _ORIGINAL_CONV_KERNEL_SIZE - 1:
            raise ValueError("accepted Conv history exceeds state_length")
        for token in range(seq_len):
            current = raw_qkv[batch_index, token].float()
            window = torch.cat((history, current.unsqueeze(0)), dim=0)
            products = window * filter_fp32
            convolved = products[0] + products[1] + (products[2] + products[3])
            result[batch_index, token] = torch.nn.functional.silu(convolved).to(
                torch.bfloat16
            )
            history = torch.cat(
                (history[1:], current.to(torch.bfloat16).float().unsqueeze(0)), dim=0
            )
        selected = original_state[cache_index, offset:history_stop]
        conv_state[cache_index, :2].copy_(selected[1:])
        proposal_stop = 2 + seq_len
        conv_state[cache_index, 2:proposal_stop].copy_(raw_qkv[batch_index])
    return result


def _original_run_pooled_recurrent(case: _RecurrentCase) -> torch.Tensor:
    (
        convolved_qkv,
        raw_decay,
        raw_beta,
        recurrent_state,
        ssm_state_indices,
        num_accepted_tokens,
        a_log,
        dt_bias,
        scale,
        lower_bound,
    ) = case
    batch_size, seq_len, _, num_heads, head_dim = convolved_qkv.shape
    query = convolved_qkv[:, :, 0].float()
    key = convolved_qkv[:, :, 1].float()
    value = convolved_qkv[:, :, 2].float()
    query *= torch.rsqrt(
        query.square().sum(dim=-1, keepdim=True) + _ORIGINAL_RMS_EPSILON
    )
    key *= torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + _ORIGINAL_RMS_EPSILON)
    _original_record_key(key)
    decay = float(lower_bound) * torch.sigmoid(
        torch.exp(a_log.float()).view(1, 1, num_heads, 1)
        * (raw_decay.float() + dt_bias.float().view(1, 1, num_heads, head_dim))
    )
    beta = torch.sigmoid(raw_beta.float())
    output = torch.empty(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        dtype=torch.float32,
        device=convolved_qkv.device,
    )
    state_indices = ssm_state_indices.reshape(batch_size, seq_len)
    for batch_index in range(batch_size):
        accepted = int(num_accepted_tokens[batch_index])
        initial_token = accepted - 1 if accepted > 0 else 0
        current = (
            recurrent_state[int(state_indices[batch_index, initial_token])]
            .float()
            .clone()
        )
        for token in range(seq_len):
            current *= torch.exp(decay[batch_index, token]).unsqueeze(-2)
            _original_record_value(
                "decay", batch_index, token, torch.exp(decay[batch_index, token])
            )
            key_value = (current * key[batch_index, token].unsqueeze(-2)).sum(dim=-1)
            delta = (value[batch_index, token] - key_value) * beta[
                batch_index, token
            ].unsqueeze(-1)
            _original_record_value("u", batch_index, token, delta)
            current += delta.unsqueeze(-1) * key[batch_index, token].unsqueeze(-2)
            output[batch_index, token] = (
                current * query[batch_index, token].unsqueeze(-2)
            ).sum(dim=-1) * float(scale)
            recurrent_state[int(state_indices[batch_index, token])].copy_(
                current.to(recurrent_state.dtype)
            )
    return output


def _original_dense_golden(case: _GoldenCase) -> MegaRecurrentKDAGolden:
    """Run the full fragment on CPU without calling production helpers."""
    (
        hidden_states,
        weights,
        conv_state,
        recurrent_state,
        conv_state_indices,
        ssm_state_indices,
        conv_num_accepted_tokens,
        num_accepted_tokens,
        scale,
        lower_bound,
        rms_norm_eps,
    ) = case
    assert hidden_states.device.type == "cpu", (
        "golden must use independent CPU arithmetic"
    )
    batch_size, seq_len, hidden_size = map(int, hidden_states.shape)
    num_heads = int(weights.beta_projection_weight.shape[0])
    projection_size = num_heads * _ORIGINAL_HEAD_DIM
    raw_qkv = _original_linear(
        hidden_states, weights.qkv_projection_weight, torch.float32
    )
    convolved_flat = _original_conv1d_silu(
        raw_qkv,
        weights.conv1d_weight,
        conv_state,
        conv_state_indices,
        conv_num_accepted_tokens,
    )
    convolved_qkv = convolved_flat.view(
        batch_size, seq_len, 3, num_heads, _ORIGINAL_HEAD_DIM
    )
    decay_a = _original_linear(
        hidden_states, weights.decay_projection_a_weight, torch.bfloat16
    )
    raw_decay = _original_linear(
        decay_a, weights.decay_projection_b_weight, torch.bfloat16
    ).view(batch_size, seq_len, num_heads, _ORIGINAL_HEAD_DIM)
    raw_beta = _original_linear(
        hidden_states, weights.beta_projection_weight, torch.bfloat16
    ).view(batch_size, seq_len, num_heads)
    raw_output_gate = _original_linear(
        hidden_states, weights.output_gate_projection_weight, torch.bfloat16
    ).view(batch_size, seq_len, num_heads, _ORIGINAL_HEAD_DIM)
    recurrent_output = _original_run_pooled_recurrent(
        _RecurrentCase(
            convolved_qkv,
            raw_decay,
            raw_beta,
            recurrent_state,
            ssm_state_indices,
            num_accepted_tokens,
            weights.a_log,
            weights.dt_bias,
            scale,
            lower_bound,
        )
    )
    inverse_rms = torch.rsqrt(
        recurrent_output.square().mean(dim=-1, keepdim=True) + float(rms_norm_eps)
    )
    normalized = recurrent_output * inverse_rms * weights.output_norm_weight.float()
    gated_output = (normalized * torch.sigmoid(raw_output_gate.float())).to(
        torch.bfloat16
    )
    output = _original_linear(
        gated_output.view(batch_size, seq_len, projection_size),
        weights.output_projection_weight,
        torch.bfloat16,
    ).view(batch_size, seq_len, hidden_size)
    return MegaRecurrentKDAGolden(
        output=output,
        raw_qkv=raw_qkv,
        convolved_qkv=convolved_qkv,
        decay_a=decay_a,
        raw_decay=raw_decay,
        raw_beta=raw_beta,
        raw_output_gate=raw_output_gate,
        recurrent_output=recurrent_output,
        gated_output=gated_output,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
    )


def _original_batch(hidden, weights, conv, checkpoint, counts, scale, lower, epsilon):
    b, s, _ = hidden.shape
    n = weights.beta_projection_weight.shape[0]
    pool = torch.zeros((b + 1, *conv.shape[1:]), dtype=conv.dtype, device="cpu")
    pool[1:].copy_(conv)
    # The old API writes a snapshot per token. Only its initial snapshot is an
    # input in Replay; the temporary pool prevents writes to checkpoint_state.
    state = torch.zeros(
        (b * s, *checkpoint.shape[1:]), dtype=checkpoint.dtype, device="cpu"
    )
    state[::s].copy_(checkpoint)
    records = dict(
        u=torch.empty((b, s, n, 128), dtype=torch.float32),
        k=torch.empty((b, s, n, 128), dtype=torch.float16),
        decay=torch.empty((b, s, n, 128), dtype=torch.float32),
    )
    token = _ORIGINAL_RECORDS.set(records)
    try:
        value = _original_dense_golden(
            _GoldenCase(
                hidden,
                weights,
                pool,
                state,
                torch.arange(1, b + 1, dtype=torch.int32),
                torch.arange(b * s, dtype=torch.int32),
                counts,
                torch.zeros(b, dtype=torch.int32),
                scale,
                lower,
                epsilon,
            )
        )
    finally:
        _ORIGINAL_RECORDS.reset(token)
    return (
        value.output,
        value.conv_state[1:],
        records["u"],
        records["k"],
        records["decay"],
    )


@torch.no_grad()
def run_original(
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
    tensors = (
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
    assert all(t.device.type == "cpu" for t in tensors), (
        "Original golden executes on CPU"
    )
    with torch.device("cpu"):
        n = beta_projection_weight.shape[0]
        p = n * 128
        wc = conv1d_weight
        if wc.shape != (4, 3 * p):
            if wc.ndim == 3:
                wc = wc[:, None, :, :].expand(3, n, 4, 128)
            wc = wc.permute(2, 0, 1, 3).reshape(4, 3 * p).contiguous()
        weights = MegaRecurrentKDAWeights(
            qkv_projection_weight,
            decay_projection_a_weight,
            decay_projection_b_weight,
            beta_projection_weight,
            output_gate_projection_weight,
            output_norm_weight,
            output_projection_weight,
            wc,
            A_log,
            dt_bias,
        )
        indices = state_indices.long()
        b = indices.numel()
        conv = conv_state.clone()
        u = replay_u.clone()
        k = replay_k.clone()
        decay = replay_decay.clone()
        cp = checkpoint_state.index_select(0, indices)
        selected = conv_state.index_select(0, indices)
        if hidden_states.ndim == 3:
            assert cu_seqlens is None
            s = hidden_states.shape[1]
            result = _original_batch(
                hidden_states,
                weights,
                selected,
                cp,
                conv_num_accepted_tokens,
                scale,
                lower_bound,
                rms_norm_eps,
            )
            conv.index_copy_(0, indices, result[1])
            for target, value in zip((u, k, decay), result[2:]):
                target[:, :, :s].index_copy_(0, indices, value.transpose(1, 2))
            out = result[0]
        else:
            assert cu_seqlens is not None and cu_seqlens.device.type == "cpu"
            prefix = cu_seqlens.tolist()
            out = []
            # The original golden accepts rectangular BSH; variable TH segments
            # are passed as individual B=1 calls without changing its arithmetic.
            for batch in range(b):
                lo, hi = prefix[batch : batch + 2]
                result = _original_batch(
                    hidden_states[lo:hi].unsqueeze(0),
                    weights,
                    selected[batch : batch + 1],
                    cp[batch : batch + 1],
                    conv_num_accepted_tokens[batch : batch + 1],
                    scale,
                    lower_bound,
                    rms_norm_eps,
                )
                out.append(result[0].squeeze(0))
                conv[indices[batch]].copy_(result[1][0])
                for target, value in zip((u, k, decay), result[2:]):
                    target[indices[batch], :, : hi - lo].copy_(value[0].transpose(0, 1))
            out = torch.cat(out, dim=0)
        return out, conv, u, k, decay
