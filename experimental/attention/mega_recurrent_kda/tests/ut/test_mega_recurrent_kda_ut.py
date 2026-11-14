# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Independent CPU precision and parameterized PA/TH contract tests.

Run this file with pytest after enabling the required NPU runtime.
Use -m functional / -m interception.
Logs/provenance are collected by the external validation command, not test helpers.
"""

from __future__ import annotations
import fcntl
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope="session")
def npu_session(request):
    import torch_npu

    if getattr(request.config, "_kda_device0_lock", None) is None:
        path = Path("/var/lock/cannbot-ci/npu-test.lock.device-0")
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            torch.npu.set_device(0)
            torch.npu.init()
            assert torch.npu.is_available()
        except BaseException:
            handle.close()
            raise
        request.config._kda_device0_lock = handle
        request.addfinalizer(lambda: (torch.npu.synchronize(), handle.close()))


@pytest.fixture(autouse=True)
def require_real_npu(request):
    if request.node.get_closest_marker("npu"):
        request.getfixturevalue("npu_session")


HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
QK_RMS_EPSILON = 1e-6

_FP64 = torch.float64
_FP32 = torch.float32
_BF16 = torch.bfloat16


def _fp32_runtime_scalar(value) -> float:
    return float(torch.tensor(float(value), dtype=_FP32).item())


_QK_RMS_EPSILON_FP32 = _fp32_runtime_scalar(QK_RMS_EPSILON)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"mega_recurrent_kda cpu_hp peer: {message}")


def _project(
    value: torch.Tensor, weight: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    accumulated = torch.matmul(value.to(_FP64), weight.to(_FP64).transpose(-1, -2))
    return accumulated.to(output_dtype)


def _causal_conv1d_silu(
    raw_qkv: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
):
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
            continue
        accepted = int(conv_num_accepted_tokens[batch_index])
        offset = max(accepted - 1, 0)
        history_stop = offset + CONV_KERNEL_SIZE - 1
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
            convolved[batch_index, token] = (
                convolved_fp64 * torch.sigmoid(convolved_fp64)
            ).to(_BF16)
            history = torch.cat(
                (history[1:], current.to(_BF16).to(_FP64).unsqueeze(0)), dim=0
            )
        selected = conv_state[cache_index, offset:history_stop]
        conv_state_new[cache_index, :2].copy_(selected[1:])
        proposal_stop = 2 + seq_len
        conv_state_new[cache_index, 2:proposal_stop].copy_(raw_qkv[batch_index])
    return convolved, conv_state_new


HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
RMS_EPSILON = 1e-6


def _linear(
    value: torch.Tensor, weight: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    result = torch.matmul(value.float(), weight.float().transpose(-1, -2))
    return result.to(output_dtype)


def run_causal_conv1d_silu(
    raw_qkv: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
) -> torch.Tensor:
    if raw_qkv.ndim != 3:
        raise ValueError("raw_qkv must be [B,S,3*P]")
    batch_size, seq_len, channels = map(int, raw_qkv.shape)
    if tuple(conv1d_weight.shape) != (CONV_KERNEL_SIZE, channels):
        raise ValueError("conv1d_weight must be [4,3*P]")
    if conv_state.ndim != 3 or int(conv_state.shape[2]) != channels:
        raise ValueError("conv_state must be [pool,state_length,3*P]")
    if int(conv_state.shape[1]) < seq_len + CONV_KERNEL_SIZE - 2:
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
        history_stop = offset + CONV_KERNEL_SIZE - 1
        history = original_state[cache_index, offset:history_stop].float().clone()
        if int(history.shape[0]) != CONV_KERNEL_SIZE - 1:
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


@torch.no_grad()
def _dense_baseline_fp32(
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
    b, s, h = t[0].shape
    n = t[4].shape[0]
    project = _linear
    conv = t[9].clone()
    qkv = run_causal_conv1d_silu(
        project(t[0], t[1], torch.float32), t[8], conv, t[14], t[15]
    ).view(b, s, 3, n, 128)
    q, k, v = (qkv[:, :, i].float() for i in range(3))
    q *= torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k *= torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    raw_decay = project(project(t[0], t[2], torch.bfloat16), t[3], torch.bfloat16).view(
        b, s, n, 128
    )
    decay = torch.exp(
        lower_bound
        * torch.sigmoid(
            torch.exp(t[16]).view(1, 1, n, 1)
            * (raw_decay.float() + t[17].view(1, 1, n, 128))
        )
    )
    beta = torch.sigmoid(project(t[0], t[4], torch.bfloat16).float())
    gate = project(t[0], t[5], torch.bfloat16).view(b, s, n, 128)
    current = t[10].clone()
    u = torch.empty((b, s, n, 128), dtype=torch.float32)
    recurrent = torch.empty((b, s, n, 128))
    for token in range(s):
        current *= decay[:, token].unsqueeze(-2)
        delta = (v[:, token] - (current * k[:, token].unsqueeze(-2)).sum(-1)) * beta[
            :, token
        ].unsqueeze(-1)
        u[:, token] = delta.to(torch.float32)
        current += delta.unsqueeze(-1) * k[:, token].unsqueeze(-2)
        recurrent[:, token] = (current * q[:, token].unsqueeze(-2)).sum(-1) * scale
    normalized = (
        recurrent
        * torch.rsqrt(recurrent.square().mean(-1, keepdim=True) + rms_norm_eps)
        * t[6].float()
    )
    gated = (normalized * torch.sigmoid(gate.float())).to(torch.bfloat16)
    out = project(gated.flatten(2), t[7], torch.bfloat16)
    return [out, conv, u, k.to(torch.float16), decay]


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
def _reference_run_impl(
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
    *,
    fp64=True,
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
    dense = _dense_golden_fp64 if fp64 else _dense_baseline_fp32
    compute_dtype = torch.float64 if fp64 else torch.float32
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
        result = dense(*args, scale, lower_bound, rms_norm_eps)
        out.append(result[0].squeeze(0))
        conv[slot].copy_(result[1][1])
        for target, value in zip((u, k, decay), result[2:]):
            target[slot, :, :length].copy_(value[0].transpose(0, 1))
    return torch.cat(out).view(hidden_states.shape), conv, u, k, decay


def _golden_fp64(*args, **kwargs):
    return _reference_run(*args, **kwargs, fp64=True)


def _baseline_fp32(*args, **kwargs):
    return _reference_run(*args, **kwargs, fp64=False)


def _reference_run(*args, **kwargs):
    with torch.device("cpu"):
        return _reference_run_impl(*args, **kwargs)


# Both pytest and TTK execute only cases present in the canonical design CSV.
import os
from importlib.util import module_from_spec, spec_from_file_location

_paths_spec = spec_from_file_location(
    "_mega_kda_paths", Path(__file__).resolve().parents[1] / "_mega_kda_paths.py"
)
_paths = module_from_spec(_paths_spec)
_paths_spec.loader.exec_module(_paths)
adapter, designed_cases = _paths.load_test_support()


@pytest.fixture(autouse=True)
def external_evidence(tmp_path_factory, monkeypatch):
    if not os.environ.get("MKDA_RUN_DIR"):
        monkeypatch.setenv(
            "MKDA_RUN_DIR", str(tmp_path_factory.getbasetemp() / "mega_kda_evidence")
        )


_selected = designed_cases.select("pytest_smoke")
if os.environ.get("MEGA_KDA_SUITE"):
    wanted = {c["case_id"] for c in designed_cases.select(os.environ["MEGA_KDA_SUITE"])}
    _selected = [c for c in _selected if c["case_id"] in wanted]
_parameters = [
    pytest.param(
        c["case_id"],
        id=c["case_id"],
        marks=[
            pytest.mark.npu,
            pytest.mark.interception
            if c["expected"] == "REJECT_ATOMIC"
            else pytest.mark.functional,
        ],
    )
    for c in _selected
]


@pytest.mark.parametrize("case_id", _parameters)
def test_designed_case(case_id):
    torch.set_num_threads(4)
    adapter.run_pytest_case(case_id, reference_verify=_baseline_fp32)
