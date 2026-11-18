# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU implementations for ``flash_gdn_grad`` (TTK golden and peer).

Two roles share one token-level recurrence:

``golden``
    FP64 recurrence plus ``torch.autograd.grad`` — the device-independent truth.

``bf16_peer``
    The same recurrence in FP32 with the operator's storage rounding applied to
    the inputs, i.e. the same error budget the NPU kernel carries. This is the
    ``third_party`` of ``cross_check``; a zero-error peer would collapse the
    ratio denominator.

Both return the operator's output order::

    (dq, dk, dv, dh0, dg, dbeta, dA_log, ddt_bias)
"""

from __future__ import annotations

import torch

__all__ = ["golden", "bf16_peer"]

_LAYOUTS = ("BSND", "BNSD", "TND")


def _to_bsnd(t: torch.Tensor, layout: str) -> torch.Tensor:
    """Return a (B, S, H, D) view; TND input is a packed (T, H, D) tensor."""
    if layout == "BNSD":
        return t.transpose(1, 2)
    return t


def _from_bsnd(t: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "BNSD":
        return t.transpose(1, 2)
    return t


def _sequence_lengths(layout, batch, tokens, cu_seqlens, seqused):
    if layout == "TND":
        offsets = [int(v) for v in cu_seqlens.tolist()]
        return [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
    if seqused is not None:
        return [int(v) for v in seqused.tolist()]
    return [tokens] * batch


def _round_like(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Round to the storage dtype of ``like`` without changing the value range."""
    if like.dtype == torch.bfloat16:
        return value.to(torch.bfloat16).to(torch.float32)
    if like.dtype == torch.float16:
        return value.to(torch.float16).to(torch.float32)
    return value


def _reference(
    q,
    k,
    v,
    g,
    beta,
    h0,
    a_log,
    dt_bias,
    do,
    dht,
    *,
    layout: str,
    cu_seqlens,
    seqused,
    fused: bool,
    accum_dtype,
    round_inputs: bool,
    gate_fused: bool = None,
    l2norm: bool = False,
    neg_eigval: bool = False,
    scale: float = None,
):
    """Token-level GDN backward. Returns grads in operator output order."""
    q, k, v, g, beta, do = (_to_bsnd(t, layout).float() for t in (q, k, v, g, beta, do))
    dht = dht.float()
    h0 = h0.float()
    if round_inputs:
        q, k, v, g, beta = (
            _round_like(t, o) for t, o in zip((q, k, v, g, beta), (q, k, v, g, beta))
        )
    dtype = accum_dtype
    batch, tokens, heads, kdim = q.shape
    hv, vdim = v.shape[2], v.shape[3]
    lengths = _sequence_lengths(layout, batch, tokens, cu_seqlens, seqused)

    gate_fused = fused if gate_fused is None else gate_fused
    # q/k 必须是叶子张量：L2 归一化放在计算图内，autograd 才会把 dq/dk 还原到
    # 归一化前的输入（与 kernel 的 restore_normalized_grad 一致）。
    q_leaf = q.to(dtype).requires_grad_()
    k_leaf = k.to(dtype).requires_grad_()
    if l2norm:
        # rstd = rsqrt(Σx²+1e-6)；peer 路径（round_inputs）再把归一化结果过一遍 BF16，
        # 因为 kernel 的 q_norm 是以 BF16 存储/使用的（cast 视为直通）。
        q_rstd = torch.rsqrt(q_leaf.square().sum(-1, keepdim=True) + 1e-6)
        k_rstd = torch.rsqrt(k_leaf.square().sum(-1, keepdim=True) + 1e-6)
        q_norm, k_norm = q_leaf * q_rstd, k_leaf * k_rstd
        if round_inputs:
            q_norm = q_norm.to(torch.bfloat16).to(dtype)
            k_norm = k_norm.to(torch.bfloat16).to(dtype)
    else:
        q_norm, k_norm = q_leaf, k_leaf
    qr = q_norm.repeat_interleave(hv // heads, dim=2)
    kr = k_norm.repeat_interleave(hv // heads, dim=2)
    vr = v.to(dtype).requires_grad_()
    gr = g.to(dtype).requires_grad_()
    br = beta.to(dtype).requires_grad_()
    hr = h0.to(dtype).requires_grad_()
    ar = a_log.to(dtype).requires_grad_() if a_log is not None else None
    biasr = dt_bias.to(dtype).requires_grad_() if dt_bias is not None else None

    if gate_fused:
        if ar is None:
            raise ValueError("fused gate requires A_log")
        gate = -ar.exp() * torch.nn.functional.softplus(
            gr + (biasr if biasr is not None else 0.0)
        )
    else:
        gate = gr
    if fused:
        # allow_neg_eigval 对应 beta = 2·sigmoid(raw)，特征值可落在 (0,2)
        beta_eff = br.sigmoid() * (2.0 if neg_eigval else 1.0)
    else:
        beta_eff = br

    do_d = do.to(dtype)
    active_lengths = torch.tensor(lengths)
    objective = torch.zeros((), dtype=dtype)
    h = hr
    for i in range(tokens):
        active = (i < active_lengths).reshape(batch, 1, 1, 1)
        decayed = h * gate[:, i, :, None, None].exp()
        ki = kr[:, i, :, None, :]
        qi = qr[:, i, :, None, :]
        correction = beta_eff[:, i, :, None] * (vr[:, i] - (ki @ decayed).squeeze(-2))
        h = torch.where(
            active, decayed + ki.transpose(-1, -2) * correction.unsqueeze(-2), h
        )
        out_i = (qi @ h).squeeze(-2) * (kdim**-0.5 if scale is None else float(scale))
        objective = objective + (out_i * do_d[:, i] * active.squeeze(-1)).sum()
    objective = objective + (h * dht.to(dtype)).sum()

    inputs = [q_leaf, k_leaf, vr, gr, br, hr]
    gate_inputs = [t for t in (ar, biasr) if t is not None] if gate_fused else []
    inputs += gate_inputs
    grads = torch.autograd.grad(objective, inputs, allow_unused=True)
    grads = [
        torch.zeros_like(t) if gr_ is None else gr_ for gr_, t in zip(grads, inputs)
    ]

    # q/k grads are shared across the value-head groups.
    # repeat_interleave lays the value-head groups out per key head, so the
    # reduction axis is the trailing group axis of (heads, groups).
    # q_leaf/k_leaf 是归一化前的叶子张量，autograd 已经完成 GQA 组的求和与
    # L2 归一化的链式还原，这里不再二次 reduce。
    dq, dk = grads[0], grads[1]
    dv, dg, dbeta, dh0 = grads[2], grads[3], grads[4], grads[5]
    dout = [dq, dk, dv, dh0, dg, dbeta]
    gate_grads = iter(grads[6:])
    dout += [
        next(gate_grads) if gate_fused and t is not None else None for t in (ar, biasr)
    ]

    result = []
    for index, item in enumerate(dout):
        if item is None:
            result.append(None)
            continue
        # Only the token-major outputs need the layout transposed back.
        result.append(_from_bsnd(item, layout) if index in (0, 1, 2, 4, 5) else item)
    return tuple(result)


def _dispatch(
    accum_dtype,
    round_inputs,
    q,
    k,
    v,
    g,
    beta,
    do,
    dht,
    h0,
    a_log,
    dt_bias,
    layout,
    cu_seqlens,
    seqused,
    use_beta_sigmoid_in_kernel,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    allow_neg_eigval=False,
    scale=None,
    **_ignored,
):
    """Shared entry: TTK forwards every CSV attribute, so the tail is ignored."""
    return _reference(
        q,
        k,
        v,
        g,
        beta,
        h0,
        a_log,
        dt_bias,
        do,
        dht,
        layout=layout,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        fused=bool(use_beta_sigmoid_in_kernel),
        gate_fused=bool(use_gate_in_kernel),
        l2norm=bool(use_qk_l2norm_in_kernel),
        neg_eigval=bool(allow_neg_eigval),
        scale=scale,
        accum_dtype=accum_dtype,
        round_inputs=round_inputs,
    )


def golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    h0: torch.Tensor,
    a_log=None,
    dt_bias=None,
    *,
    layout="BSND",
    cu_seqlens=None,
    seqused=None,
    use_beta_sigmoid_in_kernel=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    allow_neg_eigval=False,
    scale=None,
    **_ignored,
):
    """FP64 token-level reference (device-independent truth)."""
    return _dispatch(
        torch.float64,
        False,
        q,
        k,
        v,
        g,
        beta,
        do,
        dht,
        h0,
        a_log,
        dt_bias,
        layout,
        cu_seqlens,
        seqused,
        use_beta_sigmoid_in_kernel,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        scale=scale,
    )


def bf16_peer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    h0: torch.Tensor,
    a_log=None,
    dt_bias=None,
    *,
    layout="BSND",
    cu_seqlens=None,
    seqused=None,
    use_beta_sigmoid_in_kernel=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    allow_neg_eigval=False,
    scale=None,
    **_ignored,
):
    """CPU 低精度 peer：FP32 累加 + 与 kernel 相同的舍入/开关语义。

    与 ``golden``（FP64 真值）的差别只有两处，都是为了对齐设备端误差预算：
    ① 计算全程 FP32；② 输入与 L2 归一化结果按 kernel 的存储精度做一次 BF16 舍入。
    """
    return _dispatch(
        torch.float32,
        True,
        q,
        k,
        v,
        g,
        beta,
        do,
        dht,
        h0,
        a_log,
        dt_bias,
        layout,
        cu_seqlens,
        seqused,
        use_beta_sigmoid_in_kernel,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        scale=scale,
    )
