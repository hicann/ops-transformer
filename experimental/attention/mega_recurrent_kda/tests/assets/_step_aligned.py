# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent architecture-specific references following kernel stage boundaries.

M/N/K partitions and explicit cast/operator boundaries follow the source schedule.
CPU and NPU retain their own reduction semantics; exact equivalence is not assumed.
No DUT outputs, intermediate captures, compensation, or FP64 enter this reference.
The primary path uses complete-K mm per output tile on one AIC, preserving a
single GEMM accumulator rather than materializing extra FP32 K-block partials.
"""

import math
import torch


def schedule(c, architecture="arch35"):
    if architecture == "arch92":
        # The arch92 public entry always uses SequenceProjectionStage.
        return dict(
            branch="variable_length",
            m=128,
            k=128,
            beta_k=128,
            beta_n=128,
            fa_segments=1,
        )
    if architecture != "arch35":
        raise ValueError(f"Unsupported reference architecture: {architecture}")
    total = sum(c["lengths"])
    tp = 96 // c["N"]
    optimized = (
        c["B"] % tp == 0
        and 1 <= c["B"] // tp <= 32
        and len(set(c["lengths"])) == 1
        and c["lengths"][0] in (4, 8)
    )
    if not optimized:
        return dict(
            branch="variable_length",
            m=128,
            k=128,
            beta_k=128,
            beta_n=128,
            fa_segments=1,
        )
    if c["lengths"][0] == 4:
        return dict(
            branch="optimized_s4",
            m=math.gcd(total, 128),
            k=128,
            beta_k=64,
            beta_n=32 if c["N"] == 24 else 16,
            fa_segments=1,
        )
    return dict(
        branch="optimized_s8", m=128, k=128, beta_k=128, beta_n=16, fa_segments=1
    )


def copy_inputs(a, device):
    result = {}
    for key, value in a.items():
        if not isinstance(value, torch.Tensor):
            result[key] = value
            continue
        if value.device.type in ("npu", "privateuseone"):
            import torch_npu

            if torch_npu.get_npu_format(value) == 29:
                value = torch_npu.npu_format_cast(value, 2)
        result[key] = value.to(device).clone()
    return result


@torch.no_grad()
def tile_fullk_mm(value, weight, *, m, n=128, device_kind, vector_num=2):
    rows, inner = value.shape
    cols = weight.shape[0]
    pr = math.ceil(rows / m) * m
    pc = math.ceil(cols / n) * n
    x = torch.zeros((pr, inner), device=value.device, dtype=value.dtype)
    x[:rows] = value
    w = torch.zeros((pc, inner), device=weight.device, dtype=weight.dtype)
    w[:cols] = weight
    if device_kind == "cpu":
        x = x.float()
        w = w.float()
    else:
        assert x.dtype == w.dtype == torch.bfloat16

    def execute():
        out = torch.empty((pr, pc), device=value.device, dtype=torch.float32)
        for row in range(0, pr, m):
            for col in range(0, pc, n):
                left = x[row : row + m].contiguous()
                right = w[col : col + n].T.contiguous()
                out[row : row + m, col : col + n] = (
                    torch.mm(left, right)
                    if device_kind == "cpu"
                    else torch.mm(left, right, out_dtype=torch.float32)
                )
        return out[:rows, :cols].contiguous()

    if device_kind == "cpu":
        return execute()
    stream = torch.npu.Stream()
    torch.npu.synchronize()
    torch.npu.set_stream_limit(stream, cube_num=1, vector_num=vector_num)
    try:
        with torch.npu.stream(stream):
            result = execute()
            torch.npu.synchronize()
        return result
    finally:
        torch.npu.reset_stream_limit(stream)


@torch.no_grad()
def prepare(a, c, device_kind, architecture="arch35"):
    dev = (
        torch.device("cpu")
        if device_kind == "cpu"
        else torch.device("npu", torch.npu.current_device())
    )
    v = copy_inputs(a, dev)
    s = schedule(c, architecture)
    total = sum(c["lengths"])
    x = v["hidden_states"].reshape(total, 7168)

    def mm(value, weight, *, n=128):
        return tile_fullk_mm(
            value,
            weight,
            m=s["m"],
            n=n,
            device_kind=device_kind,
            vector_num=1 if architecture == "arch92" else 2,
        )

    def block(weight, n=128):
        return mm(x, v[weight], n=n)

    raw = block("qkv_projection_weight")
    fa32 = block("decay_projection_a_weight")
    fa = fa32.bfloat16()
    fb32 = mm(fa, v["decay_projection_b_weight"])
    fb = fb32.bfloat16()
    beta32 = block("beta_projection_weight", n=s["beta_n"])
    beta = beta32.bfloat16()
    return dict(
        a=v,
        c=c,
        schedule=s,
        raw=raw,
        fa_fp32=fa32,
        fa=fa,
        fb_fp32=fb32,
        fb=fb,
        beta_fp32=beta32,
        beta=beta,
        kind=device_kind,
    )


def project_original(value, weight, kind, dtype=torch.bfloat16):
    if kind == "cpu":
        y = torch.matmul(value.float().unsqueeze(0), weight.float().T).squeeze(0)
    else:
        y = torch.mm(value, weight.T, out_dtype=torch.float32)
    return y.to(dtype)


@torch.no_grad()
def evaluate(prepared, *, explicit_ops):
    v = prepared["a"]
    c = prepared["c"]
    kind = prepared["kind"]
    dev = v["hidden_states"].device
    B = c["B"]
    N = c["N"]
    P = N * 128
    S = max(c["lengths"])
    total = sum(c["lengths"])
    lengths = torch.tensor(c["lengths"], device=dev, dtype=torch.long)
    prefix = torch.cat(
        [torch.zeros(1, device=dev, dtype=torch.long), lengths.cumsum(0)]
    )
    times = torch.arange(S, device=dev)
    valid = times.unsqueeze(0) < lengths.unsqueeze(1)
    positions = (prefix[:-1, None] + times).clamp(max=total - 1)
    slots = v["state_indices"].long()
    counts = v["conv_num_accepted_tokens"].long()
    offsets = (counts - 1).clamp(min=0)
    wc = v["conv1d_weight"]
    if tuple(wc.shape) != (4, 3 * P):
        if wc.ndim == 3:
            wc = wc[:, None, :, :].expand(3, N, 4, 128)
        wc = wc.permute(2, 0, 1, 3).reshape(4, 3 * P).contiguous()
    wc = wc.float()
    history = (
        v["conv_state"][slots[:, None], offsets[:, None] + torch.arange(3, device=dev)]
        .float()
        .clone()
    )
    selected = history[:, 1:3].bfloat16().clone()
    raw = prepared["raw"][positions]
    cv = []
    preacc = []
    silu_values = []
    for t in range(S):
        cur = raw[:, t]
        p1 = history[:, 1] * wc[1]
        p3 = cur * wc[3]
        if explicit_ops:
            acc = torch.addcmul(p1, history[:, 0], wc[0]) + torch.addcmul(
                p3, history[:, 2], wc[2]
            )
        else:
            acc = history[:, 0] * wc[0] + p1 + (history[:, 2] * wc[2] + p3)
        if explicit_ops:
            activated = acc / (torch.exp(-acc) + 1.0)
        else:
            activated = (
                acc * torch.sigmoid(acc)
                if kind == "cpu"
                else torch.nn.functional.silu(acc)
            )
        out = activated.bfloat16()
        cv.append(out)
        preacc.append(acc)
        silu_values.append(activated)
        next_history = torch.cat(
            [history[:, 1:], cur.bfloat16().float().unsqueeze(1)], dim=1
        )
        history = torch.where(valid[:, t, None, None], next_history, history)
    convolved = torch.stack(cv, dim=1)
    q, k, val = convolved.view(B, S, 3, N, 128).float().unbind(2)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    fb = prepared["fb"][positions].float().view(B, S, N, 128)
    beta_raw = prepared["beta"][positions].float()
    if explicit_ops:
        neg_alpha = -v["A_log"].float().exp()
        z = neg_alpha.view(1, 1, N, 1) * (fb + v["dt_bias"].float().view(1, 1, N, 128))
        den = torch.exp(z) + 1.0
        decay = torch.exp(
            float(c["lower_bound"]) * torch.div(torch.ones_like(den), den)
        )
        den_beta = torch.exp(-beta_raw) + 1.0
        beta = torch.div(torch.ones_like(den_beta), den_beta)
    else:
        decay = torch.exp(
            float(c["lower_bound"])
            * torch.sigmoid(
                v["A_log"].float().exp().view(1, 1, N, 1)
                * (fb + v["dt_bias"].float().view(1, 1, N, 128))
            )
        )
        beta = beta_raw.sigmoid()
    state = v["checkpoint_state"][slots].float().clone()
    deltas = []
    recurrent = []
    for t in range(S):
        decayed = state * decay[:, t].unsqueeze(-2)
        delta = (val[:, t] - (decayed * k[:, t].unsqueeze(-2)).sum(-1)) * beta[
            :, t
        ].unsqueeze(-1)
        updated = decayed + delta.unsqueeze(-1) * k[:, t].unsqueeze(-2)
        state = torch.where(valid[:, t, None, None, None], updated, state)
        deltas.append(delta)
        recurrent.append((state * q[:, t].unsqueeze(-2)).sum(-1) * float(c["scale"]))
    u = torch.stack(deltas, dim=1)
    recurrent = torch.stack(recurrent, dim=1)
    norm = (
        recurrent
        * torch.rsqrt(
            recurrent.square().mean(-1, keepdim=True) + float(c["rms_norm_eps"])
        )
        * v["output_norm_weight"].float()
    )
    outputs = []
    start = 0
    x = v["hidden_states"].reshape(total, 7168)
    for b, length in enumerate(c["lengths"]):
        gate = (
            project_original(
                x[start : start + length], v["output_gate_projection_weight"], kind
            )
            .view(length, N, 128)
            .float()
            .sigmoid()
        )
        gated = (norm[b, :length] * gate).bfloat16().flatten(1)
        outputs.append(project_original(gated, v["output_projection_weight"], kind))
        start += length
    conv_pool = v["conv_state"].clone()
    up = v["replay_u"].clone()
    kp = v["replay_k"].clone()
    dp = v["replay_decay"].clone()
    conv_pool[slots, :2] = selected
    for t in range(S):
        selected_slots = slots[valid[:, t]]
        mask = valid[:, t]
        conv_pool[selected_slots, 2 + t] = raw[mask, t].bfloat16()
        up[selected_slots, :, t] = u[mask, t]
        kp[selected_slots, :, t] = k[mask, t].half()
        dp[selected_slots, :, t] = decay[mask, t]
    outputs = (torch.cat(outputs).view(v["hidden_states"].shape), conv_pool, up, kp, dp)
    trace = dict(
        qkv_fp32=prepared["raw"],
        fa_fp32=prepared["fa_fp32"],
        fa_bf16=prepared["fa"],
        fb_fp32=prepared["fb_fp32"],
        fb_bf16=prepared["fb"],
        beta_fp32=prepared["beta_fp32"],
        beta_bf16=prepared["beta"],
        convolved=convolved[valid],
        conv_acc_fp32=torch.stack(preacc, 1)[valid],
        silu_fp32=torch.stack(silu_values, 1)[valid],
    )
    return outputs, trace


@torch.no_grad()
def run_aligned(
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
    device_kind,
    architecture="arch35",
):
    """Compute independent five-output references; never accept DUT captures."""
    a = locals().copy()
    a.pop("device_kind")
    a.pop("architecture")
    # TTK Promote can widen floating inputs; recover their declared ABI dtypes.
    bf16 = (
        "hidden_states",
        "qkv_projection_weight",
        "decay_projection_a_weight",
        "decay_projection_b_weight",
        "beta_projection_weight",
        "output_gate_projection_weight",
        "output_projection_weight",
        "conv1d_weight",
        "conv_state",
    )
    for name in bf16:
        a[name] = a[name].to(torch.bfloat16)
    for name in (
        "output_norm_weight",
        "checkpoint_state",
        "replay_u",
        "replay_decay",
        "A_log",
        "dt_bias",
    ):
        a[name] = a[name].to(torch.float32)
    a["replay_k"] = a["replay_k"].to(torch.float16)
    if cu_seqlens is None:
        lengths = [hidden_states.shape[1]] * state_indices.numel()
    else:
        prefix = cu_seqlens.detach().cpu().tolist()
        lengths = [b - a for a, b in zip(prefix, prefix[1:])]
    c = dict(
        B=len(lengths),
        N=beta_projection_weight.shape[0],
        lengths=lengths,
        scale=scale,
        lower_bound=lower_bound,
        rms_norm_eps=rms_norm_eps,
    )
    prepared = prepare(a, c, device_kind, architecture)
    if architecture == "arch92":
        from ._arch92_aligned import evaluate_arch92

        out, _ = evaluate_arch92(prepared)
    else:
        out, _ = evaluate(prepared, explicit_ops=True)
    return out
