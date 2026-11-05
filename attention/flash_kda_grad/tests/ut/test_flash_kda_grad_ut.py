# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FlashKDA backward unit, integration and reference checks.

Run from the repository root:
    python -m pytest -m 'not whitebox' attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py
    python -m pytest attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py

This module contains reference formulas, cache construction, precision helpers,
test cases and fixtures. The npu, ascendc_toolchain, whitebox and slow markers
keep their original meaning. Selection uses pytest's native -m option; without
-m, every case is collected. Direct execution also supports --profile and
--compare-bin; use this file path for both CLI entry points.
"""

from __future__ import annotations

import ast
import csv
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import shutil
import sys
import threading
import weakref
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F


# Resolve this test's helpers without changing later operator imports.
_test_dir = Path(__file__).resolve().parents[1]
_added_paths = [
    str(path) for path in (_test_dir, _test_dir / "ttk") if str(path) not in sys.path
]
sys.path[:0] = _added_paths
try:
    from _kdag_paths import load_sample
    from flash_kda_grad_ttk import golden as ttk_golden
finally:
    for _path in _added_paths:
        sys.path.remove(_path)

# Mathematical references, cache adapters and standalone CLI helpers.


integration = load_sample("flash_kda_grad/flash_kda_grad.py")
metadata_module = load_sample("flash_kda_grad/flash_kda_grad_metadata.py")
flash_kda_grad_npu = integration.flash_kda_grad_npu
flash_kda_grad_metadata = metadata_module.flash_kda_grad_metadata


@dataclass(frozen=True)
class FlashKDAForward:
    out: torch.Tensor
    final_state: torch.Tensor
    q_norm: Optional[torch.Tensor]
    k_norm: Optional[torch.Tensor]
    q_rstd: Optional[torch.Tensor]
    k_rstd: Optional[torch.Tensor]
    beta_sigmoid: Optional[torch.Tensor]
    Aqk: torch.Tensor
    Akk: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    qg: torch.Tensor
    kg: torch.Tensor
    v_new: torch.Tensor
    g_cumsum: torch.Tensor
    chunk_states: torch.Tensor


def _token_major(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "TND":
        return tensor
    if layout == "BNSD":
        return tensor.transpose(1, 2).reshape(-1, tensor.shape[1], *tensor.shape[3:])
    if layout == "BSND":
        return tensor.reshape(-1, *tensor.shape[2:])
    raise ValueError(f"unsupported layout: {layout}")


def _restore_layout(
    tensor: torch.Tensor, original: torch.Tensor, layout: str
) -> torch.Tensor:
    if layout == "TND":
        return tensor.reshape_as(original)
    if layout == "BNSD":
        return (
            tensor.reshape(
                original.shape[0],
                original.shape[2],
                original.shape[1],
                *original.shape[3:],
            )
            .transpose(1, 2)
            .contiguous()
        )
    return tensor.reshape_as(original)


def _sequence_bounds(
    q: torch.Tensor, layout: str, cu_seqlens: Optional[torch.Tensor]
) -> list[tuple[int, int]]:
    if layout == "TND":
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for TND")
        if cu_seqlens.device.type != "cpu":
            raise ValueError("TND requires CPU cu_seqlens metadata")
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must be a 1D int32/int64 vector")
        offsets = cu_seqlens.to("cpu").tolist()
        if (
            len(offsets) < 2
            or offsets[0] != 0
            or offsets[-1] != q.shape[0]
            or any(a > b for a, b in zip(offsets, offsets[1:]))
        ):
            raise ValueError("cu_seqlens does not describe the packed tokens")
        return list(zip(offsets, offsets[1:]))
    if cu_seqlens is not None:
        raise ValueError("cu_seqlens is only valid for TND")
    batch = q.shape[0]
    length = q.shape[2] if layout == "BNSD" else q.shape[1]
    return [(index * length, (index + 1) * length) for index in range(batch)]


def _unit_lower_inverse_series(
    lower: torch.Tensor, identity: torch.Tensor
) -> torch.Tensor:
    """Invert I + strictly lower `lower` with a finite matrix product."""
    inverse = identity
    power = -lower
    steps = (lower.shape[0] - 1).bit_length()
    for step in range(steps):
        inverse = inverse @ (identity + power)
        if step + 1 < steps:
            power = power @ power
    return inverse


@torch.no_grad()
def flash_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = -5.0,
    layout: str = "BNSD",
    chunk_size: int = 64,
) -> FlashKDAForward:
    """Produce out/final state and persistent backward tensors.

    This is a mathematical PyTorch producer. ``chunk_size`` defaults to the
    FlashKDA hardware chunk of 64; smaller values support CPU math probes.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("allow_neg_eigval requires beta sigmoid in the kernel")
    if use_gate_in_kernel and A_log is None:
        raise ValueError("A_log is required for fused gate")
    if lower_bound is not None and not -5.0 <= float(lower_bound) <= 0.0:
        raise ValueError("lower_bound must be in [-5, 0] or None")
    if q.shape != k.shape or q.dtype != k.dtype:
        raise ValueError("q and k must have the same shape and dtype")

    bounds = _sequence_bounds(q, layout, cu_seqlens)
    q_tokens = _token_major(q, layout)
    k_tokens = _token_major(k, layout)
    v_tokens = _token_major(v, layout)
    g_tokens = _token_major(g, layout)
    beta_tokens = _token_major(beta, layout)
    token_count, value_heads, value_dim = v_tokens.shape
    key_heads = q_tokens.shape[1]
    key_dim = q_tokens.shape[-1]
    if token_count != q_tokens.shape[0] or value_heads % key_heads:
        raise ValueError("QK/value token counts or GQA heads do not match")
    if g_tokens.shape != (token_count, value_heads, key_dim):
        raise ValueError("g must have one key-channel gate per value head")
    if beta_tokens.shape != (token_count, value_heads):
        raise ValueError("beta must have one scalar per token and value head")
    if initial_state.shape != (len(bounds), value_heads, value_dim, key_dim):
        raise ValueError(
            "initial_state must have V-first [sequence, head, Dv, Dk] layout"
        )
    high_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    if initial_state.dtype != high_dtype:
        raise ValueError(
            "initial_state must be FP32 (FP64 only for FP64 reference inputs)"
        )
    if use_gate_in_kernel and A_log.shape != (value_heads,):
        raise ValueError("A_log must have one value per value head")
    if dt_bias is not None and dt_bias.shape != (value_heads, key_dim):
        raise ValueError("dt_bias must have [value_heads, key_dim] shape")

    if use_qk_l2norm_in_kernel:
        q_rstd = torch.rsqrt(q_tokens.to(high_dtype).square().sum(-1) + 1e-6)
        k_rstd = torch.rsqrt(k_tokens.to(high_dtype).square().sum(-1) + 1e-6)
        q_norm = (q_tokens.to(high_dtype) * q_rstd[..., None]).to(q.dtype)
        k_norm = (k_tokens.to(high_dtype) * k_rstd[..., None]).to(k.dtype)
        q_effective, k_effective = q_norm.to(high_dtype), k_norm.to(high_dtype)
    else:
        q_norm = k_norm = q_rstd = k_rstd = None
        q_effective, k_effective = q_tokens.to(high_dtype), k_tokens.to(high_dtype)
    if use_beta_sigmoid_in_kernel:
        beta_sigmoid = (
            torch.sigmoid(beta_tokens.to(high_dtype)) * (2 if allow_neg_eigval else 1)
        ).to(high_dtype)
        beta_effective = beta_sigmoid
    else:
        beta_sigmoid = None
        beta_effective = beta_tokens.to(high_dtype)

    chunk_count = sum(
        (end - start + chunk_size - 1) // chunk_size for start, end in bounds
    )
    low_shape = (token_count, value_heads)
    Aqk = torch.zeros(
        chunk_count, value_heads, chunk_size, chunk_size, dtype=q.dtype, device=q.device
    )
    Akk = torch.zeros(Aqk.shape, dtype=Aqk.dtype, device=Aqk.device)
    w = torch.empty(*low_shape, key_dim, dtype=q.dtype, device=q.device)
    u = torch.empty(*low_shape, value_dim, dtype=v.dtype, device=q.device)
    qg = torch.empty(w.shape, dtype=w.dtype, device=w.device)
    kg = torch.empty(w.shape, dtype=w.dtype, device=w.device)
    v_new = torch.empty(u.shape, dtype=u.dtype, device=u.device)
    g_cumsum = torch.empty(*low_shape, key_dim, dtype=high_dtype, device=q.device)
    chunk_states = torch.empty(
        chunk_count, value_heads, value_dim, key_dim, dtype=high_dtype, device=q.device
    )
    output = torch.empty(v_tokens.shape, dtype=v_tokens.dtype, device=v_tokens.device)
    final_state = torch.empty(
        initial_state.shape, dtype=initial_state.dtype, device=initial_state.device
    )

    chunk_base = 0
    for seq_index, (seq_start, seq_end) in enumerate(bounds):
        chunks = [
            (start, min(start + chunk_size, seq_end))
            for start in range(seq_start, seq_end, chunk_size)
        ]
        for head in range(value_heads):
            qk_head = head // (value_heads // key_heads)
            state = initial_state[seq_index, head].T.clone()
            for local_chunk, (start, end) in enumerate(chunks):
                length = end - start
                chunk = chunk_base + local_chunk
                Q = q_effective[start:end, qk_head]
                K = k_effective[start:end, qk_head]
                V = v_tokens[start:end, head].to(high_dtype)
                B = beta_effective[start:end, head]
                raw_g = g_tokens[start:end, head].to(high_dtype)
                if use_gate_in_kernel:
                    alpha = A_log[head].to(high_dtype).exp()
                    biased = raw_g + (
                        0 if dt_bias is None else dt_bias[head].to(high_dtype)
                    )
                    if lower_bound is None:
                        step_gate = -alpha * F.softplus(biased)
                    else:
                        step_gate = float(lower_bound) * torch.sigmoid(alpha * biased)
                else:
                    if bool((raw_g > 0).any()):
                        raise ValueError(
                            "unfused g must be nonpositive single-step log-decay"
                        )
                    step_gate = raw_g
                G = step_gate.cumsum(0)
                g_cumsum[start:end, head] = G

                # Relative differences keep every causal exponent nonpositive.
                ratio = (G[:, None, :] - G[None, :, :]).exp()
                qk_pairs = (Q[:, None, :] * K[None, :, :] * ratio).sum(-1)
                kk_pairs = (K[:, None, :] * K[None, :, :] * ratio).sum(-1)
                M = torch.tril(float(scale) * qk_pairs).to(q.dtype)
                T = torch.tril(B[:, None] * kk_pairs, diagonal=-1)
                identity = torch.eye(length, dtype=high_dtype, device=q.device)
                if q.device.type == "npu":
                    inverse = _unit_lower_inverse_series(T, identity).to(q.dtype)
                elif length == 1:
                    inverse = identity.to(q.dtype)
                else:
                    inverse = torch.linalg.solve_triangular(
                        identity + T,
                        identity,
                        upper=False,
                        unitriangular=True,
                    ).to(q.dtype)
                P = inverse.to(high_dtype)
                Ktilde = G.exp() * K
                W = (P @ (B[:, None] * Ktilde)).to(q.dtype)
                U = (P @ (B[:, None] * V)).to(v.dtype)
                QG = (G.exp() * Q).to(q.dtype)
                KG = ((G[-1:] - G).exp() * K).to(q.dtype)
                chunk_states[chunk, head] = state.T
                corrected = (U.to(high_dtype) - W.to(high_dtype) @ state).to(v.dtype)
                out_chunk = (
                    float(scale) * QG.to(high_dtype) @ state
                    + M.to(high_dtype) @ corrected.to(high_dtype)
                ).to(v.dtype)
                state = G[-1].exp()[:, None] * state + KG.to(
                    high_dtype
                ).T @ corrected.to(high_dtype)

                Aqk[chunk, head, :length, :length] = M
                Akk[chunk, head, :length, :length] = inverse
                w[start:end, head] = W
                u[start:end, head] = U
                qg[start:end, head] = QG
                kg[start:end, head] = KG
                v_new[start:end, head] = corrected
                output[start:end, head] = out_chunk
            final_state[seq_index, head] = state.T
        chunk_base += len(chunks)

    return FlashKDAForward(
        _restore_layout(output, v, layout),
        final_state,
        q_norm,
        k_norm,
        q_rstd,
        k_rstd,
        beta_sigmoid,
        Aqk,
        Akk,
        w,
        u,
        qg,
        kg,
        v_new,
        g_cumsum,
        chunk_states,
    )


class _FlashKDATrain(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        scale,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        lower_bound,
        layout,
        chunk_size,
    ):
        saved = flash_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            A_log,
            dt_bias,
            cu_seqlens=cu_seqlens,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            lower_bound=lower_bound,
            layout=layout,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            *(() if A_log is None else (A_log,)),
            *(() if dt_bias is None else (dt_bias,)),
        )
        ctx.has_A_log = A_log is not None
        ctx.has_dt_bias = dt_bias is not None
        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        ctx.allow_neg_eigval = allow_neg_eigval
        ctx.lower_bound = lower_bound
        ctx.layout = layout
        ctx.cache = saved
        # The public backward requires both seeds, including zero gradients for
        # outputs that do not contribute to the loss.
        ctx.set_materialize_grads(True)
        return saved.out, saved.final_state

    @staticmethod
    def backward(ctx, do, dht):
        tensors = ctx.saved_tensors
        q, k, v, g, beta, _initial_state = tensors[:6]
        cursor = 6
        A_log = tensors[cursor] if ctx.has_A_log else None
        cursor += int(ctx.has_A_log)
        dt_bias = tensors[cursor] if ctx.has_dt_bias else None

        if q.device.type != "npu":
            # CPU training uses autograd through a fresh differentiable forward.
            # The staged PyTorch backward is a test-only intermediate reference.
            with torch.enable_grad():
                inputs = [
                    tensor.detach().requires_grad_(True)
                    for tensor in (q, k, v, g, beta, _initial_state)
                ]
                active = list(inputs)
                a_log = (
                    A_log.detach().requires_grad_(True) if A_log is not None else None
                )
                bias = (
                    dt_bias.detach().requires_grad_(True)
                    if dt_bias is not None
                    else None
                )
                if a_log is not None:
                    active.append(a_log)
                if bias is not None:
                    active.append(bias)
                replay = flash_kda_fwd.__wrapped__(
                    *inputs,
                    a_log,
                    bias,
                    cu_seqlens=ctx.cu_seqlens,
                    scale=ctx.scale,
                    use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel,
                    use_gate_in_kernel=ctx.use_gate_in_kernel,
                    use_beta_sigmoid_in_kernel=ctx.use_beta_sigmoid_in_kernel,
                    allow_neg_eigval=ctx.allow_neg_eigval,
                    lower_bound=ctx.lower_bound,
                    layout=ctx.layout,
                    chunk_size=ctx.cache.Aqk.shape[-1],
                )
                loss = (
                    (replay.out * do).sum() if do is not None else replay.out.sum() * 0
                )
                loss = loss + (
                    (replay.final_state * dht).sum()
                    if dht is not None
                    else replay.final_state.sum() * 0
                )
                grads = torch.autograd.grad(loss, active, allow_unused=True)
            dq, dk, dv, dg, dbeta, dh0 = grads[:6]
            cursor = 6
            dA = grads[cursor] if a_log is not None else None
            cursor += int(a_log is not None)
            dBias = grads[cursor] if bias is not None else None
            return dq, dk, dv, dg, dbeta, dh0, dA, dBias, *(None for _ in range(9))

        if ctx.cache.Aqk.shape[-1] != 64:
            raise ValueError("NPU DSL backward requires chunk_size=64")
        saved = ctx.cache
        offsets = (
            None
            if ctx.cu_seqlens is None
            else ctx.cu_seqlens.to(device=q.device, dtype=torch.int32)
        )
        metadata = flash_kda_grad_metadata(
            **_metadata_arguments(q, v, ctx.layout, offsets)
        )
        dq, dk, dv, dh0, dg, dbeta, dA, dBias = _reference_backward(
            _reference_qk_to_input_layout(saved.q_norm, q, metadata, ctx.layout)
            if ctx.use_qk_l2norm_in_kernel
            else q,
            _reference_qk_to_input_layout(saved.k_norm, k, metadata, ctx.layout)
            if ctx.use_qk_l2norm_in_kernel
            else k,
            v,
            g,
            beta,
            do,
            dht,
            saved.Aqk,
            saved.Akk,
            metadata,
            None,  # initial_state is unused with saved caches
            saved.chunk_states,
            saved.w,
            saved.u,
            saved.v_new,
            saved.qg,
            saved.kg,
            saved.g_cumsum,
            disable_recompute=True,
            q_rtsd=saved.q_rstd,
            k_rstd=saved.k_rstd,
            beta_sigmoid=saved.beta_sigmoid,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=ctx.scale,
            use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=ctx.use_beta_sigmoid_in_kernel,
            allow_neg_eigval=ctx.allow_neg_eigval,
            lower_bound=ctx.lower_bound,
            layout=ctx.layout,
        )
        return dq, dk, dv, dg, dbeta, dh0, dA, dBias, *(None for _ in range(9))


def flash_kda_train(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = -5.0,
    layout: str = "BNSD",
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable FlashKDA; NPU backward dispatches all DSL stages."""
    if q.device.type == "npu":
        if any(
            t.device != q.device
            for t in (k, v, g, beta, initial_state)
            + tuple(t for t in (A_log, dt_bias) if t is not None)
        ):
            raise ValueError("NPU DSL training requires all tensors on one NPU")
        if chunk_size != 64:
            raise ValueError("NPU DSL training requires chunk_size=64")
        if any(t.dtype != torch.bfloat16 for t in (q, k, v, g, beta)):
            raise ValueError("NPU DSL training requires BF16 q/k/v/g/beta")
        if (
            q.shape[-1] != 128
            or k.shape[-1] != 128
            or v.shape[-1] != 128
            or g.shape[-1] != 128
        ):
            raise ValueError("NPU DSL training requires Dk=Dv=128")
        if initial_state.dtype != torch.float32:
            raise ValueError("NPU DSL training requires FP32 initial_state")
        if A_log is not None and A_log.dtype != torch.float32:
            raise ValueError("NPU DSL training requires FP32 A_log")
        if dt_bias is not None and dt_bias.dtype != torch.float32:
            raise ValueError("NPU DSL training requires FP32 dt_bias")
    return _FlashKDATrain.apply(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        scale,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        lower_bound,
        layout,
        chunk_size,
    )


__all__ = ["FlashKDAForward", "flash_kda_fwd", "flash_kda_train"]


def _staged_token_major(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "TND":
        return tensor
    if layout == "BNSD":
        return tensor.transpose(1, 2).reshape(-1, *tensor.shape[1:2], *tensor.shape[3:])
    if layout == "BSND":
        return tensor.reshape(-1, *tensor.shape[2:])
    raise ValueError(f"unsupported layout: {layout}")


def _staged_restore_layout(
    tensor: torch.Tensor, original: torch.Tensor, layout: str
) -> torch.Tensor:
    if layout == "TND":
        return tensor.reshape_as(original)
    if layout == "BNSD":
        return (
            tensor.reshape(
                original.shape[0],
                original.shape[2],
                original.shape[1],
                *original.shape[3:],
            )
            .transpose(1, 2)
            .contiguous()
        )
    return tensor.reshape_as(original)


def _staged_sequence_bounds(
    q: torch.Tensor, layout: str, cu_seqlens: Optional[torch.Tensor]
) -> list[tuple[int, int]]:
    if layout == "TND":
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for TND")
        offsets = cu_seqlens.to("cpu").tolist()
        if (
            offsets[0] != 0
            or offsets[-1] != q.shape[0]
            or any(a > b for a, b in zip(offsets, offsets[1:]))
        ):
            raise ValueError("cu_seqlens does not describe the packed tokens")
        return list(zip(offsets, offsets[1:]))
    batch = q.shape[0]
    length = q.shape[2] if layout == "BNSD" else q.shape[1]
    return [(index * length, (index + 1) * length) for index in range(batch)]


@torch.no_grad()
def flash_kda_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: Optional[torch.Tensor],
    dht: Optional[torch.Tensor],
    chunk_states: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    v_new: torch.Tensor,
    qg: torch.Tensor,
    kg: torch.Tensor,
    g_cumsum: torch.Tensor,
    q_norm: Optional[torch.Tensor],
    q_rstd: Optional[torch.Tensor],
    k_norm: Optional[torch.Tensor],
    k_rstd: Optional[torch.Tensor],
    beta_sigmoid: Optional[torch.Tensor],
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    scale: float,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "BNSD",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Return ``(dq, dk, dv, dh0, dg, dbeta, dA, dBias)``.

    Saved token arrays use logical ``[tokens, value_heads, ...]`` except
    normalized Q/K, which use QK heads. Chunk arrays use a global logical
    chunk index, then a value head. Tail rows in Aqk/Akk are ignored.
    """
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("allow_neg_eigval requires beta sigmoid in the kernel")
    if use_gate_in_kernel and A_log is None:
        raise ValueError("A_log is required for fused gate")
    if use_qk_l2norm_in_kernel and any(
        x is None for x in (q_norm, k_norm, q_rstd, k_rstd)
    ):
        raise ValueError("Q/K normalization caches are required")
    if use_beta_sigmoid_in_kernel and beta_sigmoid is None:
        raise ValueError("beta_sigmoid cache is required")
    if Aqk.shape != Akk.shape or Aqk.shape[-1] != Aqk.shape[-2]:
        raise ValueError("Aqk and Akk must have matching square chunk matrices")

    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    if chunk_states.dtype != dtype:
        raise ValueError(
            "chunk_states must be FP32 (FP64 only for FP64 reference inputs)"
        )
    if dht is not None and dht.dtype != dtype:
        raise ValueError("dht must match the FP32 state dtype")
    q_tokens = _staged_token_major(q, layout)
    k_tokens = _staged_token_major(k, layout)
    v_tokens = _staged_token_major(v, layout)
    g_tokens = _staged_token_major(g, layout)
    beta_tokens = _staged_token_major(beta, layout)
    do_tokens = (
        torch.zeros_like(v_tokens) if do is None else _staged_token_major(do, layout)
    )
    q_effective = (q_norm if use_qk_l2norm_in_kernel else q_tokens).to(dtype)
    k_effective = (k_norm if use_qk_l2norm_in_kernel else k_tokens).to(dtype)
    b_effective = (beta_sigmoid if use_beta_sigmoid_in_kernel else beta_tokens).to(
        dtype
    )
    if b_effective.ndim == 3 and b_effective.shape[-1] == 1:
        b_effective = b_effective.squeeze(-1)
    token_count, value_heads, value_dim = v_tokens.shape
    key_heads = q_effective.shape[1]
    key_dim = q_effective.shape[-1]
    if value_heads % key_heads:
        raise ValueError("value heads must be divisible by QK heads")
    if g_cumsum.shape != (token_count, value_heads, key_dim):
        raise ValueError("g_cumsum shape does not match token and head counts")
    if chunk_states.shape[1:] != (value_heads, value_dim, key_dim):
        raise ValueError("chunk_states must use V-first state layout")

    dq_heads = torch.zeros(
        token_count, key_heads, key_dim, dtype=dtype, device=q.device
    )
    dk_heads = torch.zeros_like(dq_heads)
    dv = torch.zeros(token_count, value_heads, value_dim, dtype=dtype, device=q.device)
    dg = torch.zeros(token_count, value_heads, key_dim, dtype=dtype, device=q.device)
    dbeta = torch.zeros(token_count, value_heads, dtype=dtype, device=q.device)
    dA = torch.zeros_like(A_log, dtype=dtype) if use_gate_in_kernel else None
    dBias = (
        torch.zeros_like(dt_bias, dtype=dtype)
        if use_gate_in_kernel and dt_bias is not None
        else None
    )
    seq_bounds = _staged_sequence_bounds(q, layout, cu_seqlens)
    chunk_size = Aqk.shape[-1]
    chunk_base = 0
    dh0 = torch.zeros(
        len(seq_bounds), value_heads, value_dim, key_dim, dtype=dtype, device=q.device
    )

    for seq_index, (seq_start, seq_end) in enumerate(seq_bounds):
        chunks = [
            (start, min(start + chunk_size, seq_end))
            for start in range(seq_start, seq_end, chunk_size)
        ]
        for head in range(value_heads):
            qk_head = head // (value_heads // key_heads)
            dstate = (
                torch.zeros(value_dim, key_dim, dtype=dtype, device=q.device)
                if dht is None
                else dht[seq_index, head].to(dtype)
            ).T.contiguous()
            for local_chunk in range(len(chunks) - 1, -1, -1):
                start, end = chunks[local_chunk]
                length = end - start
                chunk = chunk_base + local_chunk
                Q = q_effective[start:end, qk_head]
                K = k_effective[start:end, qk_head]
                V = v_tokens[start:end, head].to(dtype)
                G = g_cumsum[start:end, head].to(dtype)
                B = b_effective[start:end, head]
                S = chunk_states[chunk, head].to(dtype).T
                W = w[start:end, head].to(dtype)
                R = v_new[start:end, head].to(dtype)
                QG = qg[start:end, head].to(dtype)
                KG = kg[start:end, head].to(dtype)
                M = Aqk[chunk, head, :length, :length].to(dtype)
                P = Akk[chunk, head, :length, :length].to(dtype)
                dO = do_tokens[start:end, head].to(dtype)

                # Stage 2: both consumers of corrected values and prior state.
                dR = M.T @ dO + KG @ dstate
                dM = torch.tril(dO @ R.T)
                dQG = float(scale) * (dO @ S.T)
                dKG = R @ dstate.T
                dGamma = (dstate * S).sum(-1)
                dSprev = (
                    float(scale) * (QG.T @ dO)
                    + G[-1].exp()[:, None] * dstate
                    - W.T @ dR
                )
                dW = -(dR @ S.T)
                dU = dR

                # Stage 1B: WY, unit-lower inverse, and relative-gate pairs.
                Ktilde = G.exp() * K
                dP = dU @ (B[:, None] * V).T + dW @ (B[:, None] * Ktilde).T
                dVbeta = P.T @ dU
                dKbeta = P.T @ dW
                dv[start:end, head] += B[:, None] * dVbeta
                dB = (dVbeta * V).sum(-1) + (dKbeta * Ktilde).sum(-1)
                dT = torch.tril(-(P.T @ dP @ P.T), diagonal=-1)
                dKtilde = B[:, None] * dKbeta
                rho = (G[-1:] - G).exp()
                dQ = G.exp() * dQG
                dK = G.exp() * dKtilde + rho * dKG
                dG = QG * dQG + Ktilde * dKtilde - KG * dKG
                dG[-1] += G[-1].exp() * dGamma + (KG * dKG).sum(0)

                for row in range(length):
                    for col in range(row + 1):
                        ratio = (G[row] - G[col]).exp()
                        m_coeff = float(scale) * dM[row, col]
                        t_coeff = B[row] * dT[row, col]
                        dQ[row] += m_coeff * K[col] * ratio
                        dK[row] += t_coeff * K[col] * ratio
                        pair = (t_coeff * K[row] + m_coeff * Q[row]) * ratio
                        dK[col] += pair
                        dB[row] += dT[row, col] * (K[row] * K[col] * ratio).sum()
                        gate_pair = pair * K[col]
                        dG[row] += gate_pair
                        dG[col] -= gate_pair

                dq_heads[start:end, qk_head] += dQ
                dk_heads[start:end, qk_head] += dK
                step_gate_grad = torch.flip(
                    torch.cumsum(torch.flip(dG, (0,)), dim=0), (0,)
                )
                if use_gate_in_kernel:
                    raw = g_tokens[start:end, head].to(dtype)
                    bias = 0 if dt_bias is None else dt_bias[head].to(dtype)
                    alpha = A_log[head].to(dtype).exp()
                    if lower_bound is None:
                        pre = raw + bias
                        dg_head = -step_gate_grad * alpha * torch.sigmoid(pre)
                        gate_step = -alpha * F.softplus(pre)
                        dA[head] += (step_gate_grad * gate_step).sum()
                    else:
                        z = alpha * (raw + bias)
                        sig = torch.sigmoid(z)
                        dz = step_gate_grad * float(lower_bound) * sig * (1 - sig)
                        dg_head = alpha * dz
                        dA[head] += (dz * z).sum()
                    dg[start:end, head] += dg_head
                    if dBias is not None:
                        dBias[head] += dg_head.sum(0)
                else:
                    dg[start:end, head] += step_gate_grad

                if use_beta_sigmoid_in_kernel:
                    factor = 2.0 if allow_neg_eigval else 1.0
                    sigmoid = torch.sigmoid(beta_tokens[start:end, head].to(dtype))
                    dbeta[start:end, head] += dB * factor * sigmoid * (1 - sigmoid)
                else:
                    dbeta[start:end, head] += dB
                dstate = dSprev
            dh0[seq_index, head] = dstate.T
        chunk_base += len(chunks)

    if chunk_base != Aqk.shape[0]:
        raise ValueError("cache chunk count does not match sequence boundaries")
    if use_qk_l2norm_in_kernel:
        q_r = q_rstd.to(dtype)[..., None]
        k_r = k_rstd.to(dtype)[..., None]
        dq_heads = q_r * (
            dq_heads - q_effective * (q_effective * dq_heads).sum(-1, keepdim=True)
        )
        dk_heads = k_r * (
            dk_heads - k_effective * (k_effective * dk_heads).sum(-1, keepdim=True)
        )
    return (
        _staged_restore_layout(dq_heads.to(q.dtype), q, layout),
        _staged_restore_layout(dk_heads.to(k.dtype), k, layout),
        _staged_restore_layout(dv.to(v.dtype), v, layout),
        dh0,
        _staged_restore_layout(dg.to(g.dtype), g, layout),
        _staged_restore_layout(dbeta.to(beta.dtype), beta, layout),
        None if dA is None else dA.to(A_log.dtype),
        None if dBias is None else dBias.to(dt_bias.dtype),
    )


def _reference_forward_arguments(arguments, lengths):
    """Pack CPU-reference caches into the current forward's public ABI."""
    values = dict(arguments)
    layout, q, v = values.get("layout", "TND"), values["q"], values["v"]
    lengths = [int(length) for length in lengths]
    batch = len(lengths)
    storage = q.shape[0] if layout == "TND" else q.shape[1 if layout == "BSND" else 2]
    chunks = [(length + 63) // 64 for length in lengths]
    capacity = (storage + 63) // 64 + batch if layout == "TND" else (storage + 63) // 64

    def tokens(tensor):
        if tensor is None or layout == "TND":
            return tensor
        if (
            tensor.ndim in (2, 3)
            and tensor.shape[0] == sum(lengths)
            and (tensor.ndim == 2 or tensor.shape[-1] == 128)
        ):
            result = tensor.new_zeros(
                (batch, tensor.shape[1], storage, *tensor.shape[2:])
            )
            start = 0
            for sequence, length in enumerate(lengths):
                result[sequence, :, :length] = tensor[start : start + length].transpose(
                    0, 1
                )
                start += length
            return result
        return tensor.transpose(1, 2) if layout == "BSND" else tensor

    def chunk_cache(tensor, state=False):
        if tensor is None:
            return None
        if layout == "TND":
            return (
                tensor
                if tensor.shape[0] == capacity
                else _reserve_tnd_forward_chunks(tensor, lengths)
            )
        if tensor.ndim == 5:
            return tensor
        if not state and tensor.shape == (*v.shape[:-1], 64):
            rows = tensor.transpose(1, 2) if layout == "BSND" else tensor
            result = tensor.new_zeros((batch, rows.shape[1], capacity * 64, 64))
            result[:, :, :storage] = rows
            return result.reshape(batch, rows.shape[1], capacity, 64, 64)
        result = tensor.new_zeros((batch, tensor.shape[1], capacity, *tensor.shape[2:]))
        start = 0
        for sequence, count in enumerate(chunks):
            result[sequence, :, :count] = tensor[start : start + count].transpose(0, 1)
            start += count
        return result

    for name in (
        "w",
        "u",
        "v_new",
        "qg",
        "kg",
        "g_cumsum",
        "q_rtsd",
        "k_rstd",
        "beta_sigmoid",
        "q_norm",
        "k_norm",
    ):
        if name in values:
            values[name] = tokens(values[name])
    for name in ("Aqk", "Akk", "chunk_states"):
        if name in values:
            values[name] = chunk_cache(values[name], name == "chunk_states")
    return values


def _reference_backward(*args, **kwargs):
    bound = inspect.signature(integration.flash_kda_grad_npu).bind(*args, **kwargs)
    values = bound.arguments
    metadata = integration._resolve_backward_metadata(
        values["q"],
        values["v"],
        values.get("layout", "TND"),
        values.get("metadata"),
        values.get("cu_seqlens"),
        values.get("seqused"),
    )
    lengths = [
        length
        for _, length, _, _ in metadata_module.metadata_host_info(metadata).sequences
    ]
    values = _reference_forward_arguments(values, lengths)
    return integration.flash_kda_grad_npu(**values)


def _reference_qk_to_input_layout(cache, reference, metadata, layout, *, scalar=False):
    if cache.shape == (reference.shape[:-1] if scalar else reference.shape):
        return cache
    lengths = [
        length
        for _, length, _, _ in metadata_module.metadata_host_info(metadata).sequences
    ]
    name = "q_rtsd" if scalar else "q_norm"
    converted = _reference_forward_arguments(
        dict(q=reference, v=reference, layout=layout, **{name: cache}), lengths
    )[name]
    return converted.transpose(1, 2).contiguous() if layout == "BSND" else converted


def _metadata_arguments(q, v, layout, cu_seqlens=None, *, seqused=None):
    """Translate test fixture shapes into the explicit metadata attributes."""
    head_axis = 2 if layout == "BSND" else 1
    length_axis = 0 if layout == "TND" else 1 if layout == "BSND" else 2
    batch = cu_seqlens.numel() - 1 if layout == "TND" else q.shape[0]
    if cu_seqlens is not None and layout != "TND":
        seqused = cu_seqlens[1:] - cu_seqlens[:-1]
        cu_seqlens = None
    return dict(
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        batch_size=batch,
        n_heads_qk=q.shape[head_axis],
        n_heads_v=v.shape[head_axis],
        seqlens=q.shape[length_axis],
        headdim_qk=q.shape[-1],
        headdim_v=v.shape[-1],
        layout=layout,
    )


def _reserve_tnd_forward_chunks(tensor, lengths):
    """Model the public forward's C-major storage, poisoning reserved chunks."""
    capacity = (sum(lengths) + 63) // 64 + len(lengths)
    reserved = tensor.new_full((capacity, *tensor.shape[1:]), float("nan"))
    token_start = compact_start = 0
    for sequence, length in enumerate(lengths):
        chunks = (length + 63) // 64
        base = token_start // 64 + sequence
        reserved[base : base + chunks] = tensor[compact_start : compact_start + chunks]
        token_start += length
        compact_start += chunks
    return reserved


def _logical(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "TND":
        return tensor
    if layout == "BNSD":
        return tensor.transpose(1, 2).reshape(-1, tensor.shape[1], *tensor.shape[3:])
    return tensor.reshape(-1, *tensor.shape[2:])


def _physical(
    tensor: torch.Tensor, layout: str, batch: int, length: int
) -> torch.Tensor:
    if layout == "TND":
        return tensor
    batched = tensor.reshape(batch, length, *tensor.shape[1:])
    return batched.transpose(1, 2).contiguous() if layout == "BNSD" else batched


def _precision_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Read strided public gradients without extra device layout kernels."""
    if tensor.device.type in {"npu", "privateuseone"} and not tensor.is_contiguous():
        # Transfer one contiguous backing storage, then restore the view on
        # CPU. Precision checks stay independent of device transpose kernels.
        storage_elements = tensor.untyped_storage().nbytes() // tensor.element_size()
        backing = tensor.as_strided((storage_elements,), (1,), storage_offset=0).cpu()
        return backing.as_strided(
            tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset()
        )
    return tensor.cpu()


def _assert_precision(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_relative_l2: float = 0.05,
    min_atol: float = 4e-3,
) -> None:
    got = _precision_to_cpu(actual).float()
    want = expected.detach().float()
    max_abs = (got - want).abs().max().item() if got.numel() else 0.0
    ref_max = want.abs().max().item() if want.numel() else 0.0
    denominator = torch.linalg.vector_norm(want).item() if want.numel() else 0.0
    numerator = torch.linalg.vector_norm(got - want).item() if got.numel() else 0.0
    relative_l2 = numerator / denominator if denominator > 1e-12 else 0.0
    print(
        f"{name}: max_abs={max_abs:.6g}, ref_max={ref_max:.6g}, rel_l2={relative_l2:.6g}"
    )
    if denominator > 1e-12:
        assert relative_l2 < max_relative_l2, f"{name}: rel_l2={relative_l2:.6g}"
    if name not in ("forward output", "forward final_state"):
        got_max = got.abs().max().item() if got.numel() else 0.0
        if ref_max > 1e-5:
            assert got_max > 0.1 * ref_max, f"{name}: unexpectedly near-zero gradient"
        else:
            assert got_max < 1e-3, f"{name}: unexpected nonzero gradient"
    # BF16 gradients can differ by a couple of representable steps after
    # the forward-cache and Cube operand casts; the L2 gate remains strict.
    atol = max(min_atol, (0.01 if actual.dtype == torch.bfloat16 else 0.006) * ref_max)
    torch.testing.assert_close(
        got,
        want,
        atol=atol,
        rtol=3e-2,
        msg=f"{name}: max_abs={max_abs:.6g}, ref_max={ref_max:.6g}, atol={atol:.6g}",
    )


def _token_golden(
    q,
    k,
    v,
    g,
    beta,
    h0,
    a_log,
    bias,
    *,
    layout,
    sequence_lengths,
    scale,
    lower_bound,
    norm,
    fused_gate,
    fused_beta,
    beta_factor,
):
    q = _logical(q, layout).float()
    k = _logical(k, layout).float()
    v = _logical(v, layout).float()
    g = _logical(g, layout).float()
    beta = _logical(beta, layout).float()
    if norm:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    if fused_gate:
        bias_value = 0 if bias is None else bias
        alpha = a_log.exp()[None, :, None]
        pre = g + bias_value
        gate = (
            -alpha * torch.nn.functional.softplus(pre)
            if lower_bound is None
            else lower_bound * torch.sigmoid(alpha * pre)
        )
    else:
        gate = g
    activated = beta_factor * torch.sigmoid(beta) if fused_beta else beta
    heads = v.shape[1]
    qk_heads = q.shape[1]
    cursor = 0
    rows = []
    final = []
    for seq, length in enumerate(sequence_lengths):
        head_rows = []
        head_final = []
        for head in range(heads):
            qk_head = head // (heads // qk_heads)
            state = h0[seq, head].float().T
            token_rows = []
            for token in range(cursor, cursor + length):
                key = k[token, qk_head]
                state = gate[token, head].exp()[:, None] * state
                delta = activated[token, head] * (v[token, head] - key @ state)
                state = state + key[:, None] * delta[None]
                token_rows.append(scale * q[token, qk_head] @ state)
            if length:
                head_rows.append(torch.stack(token_rows))
            head_final.append(state.T)
        if length:
            rows.append(torch.stack(head_rows, dim=1))
        final.append(torch.stack(head_final))
        cursor += length
    out = torch.cat(rows) if rows else v.new_empty((0, heads, v.shape[-1]))
    if layout != "TND":
        out = _physical(out, layout, len(sequence_lengths), sequence_lengths[0])
    return out, torch.stack(final)


_PROFILE_SCHEMA = "flash_kda_grad.main.v3"
_PROFILE_INPUT_NAMES = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "do",
    "dht",
    "chunk_states",
    "Aqk",
    "Akk",
    "w",
    "u",
    "v_new",
    "qg",
    "kg",
    "g_cumsum",
    "q_rtsd",
    "k_rstd",
    "beta_sigmoid",
)
_PROFILE_OPTIONAL_NAMES = ("A_log", "dt_bias", "cu_seqlens")
_PROFILE_ATTR_NAMES = (
    "scale",
    "use_qk_l2norm_in_kernel",
    "use_gate_in_kernel",
    "use_beta_sigmoid_in_kernel",
    "allow_neg_eigval",
    "lower_bound",
    "layout",
)


def _profile_parse_args(argv=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Profile the public FlashKDA backward main ABI; this command does not check accuracy."
    )
    parser.add_argument("--profile", action="store_true", required=True)
    parser.add_argument(
        "--case", choices=("b1", "tnd_b31_t8192_n96_d128_ragged"), default="b1"
    )
    parser.add_argument(
        "--heads", type=int, default=3, help="Nq=Nv when generating a new fixture"
    )
    parser.add_argument(
        "--length", type=int, default=8192, help="sequence length for a new B=1 fixture"
    )
    parser.add_argument(
        "--layout", choices=("BNSD", "BSND"), default="BNSD", help="new fixture layout"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--lower-bound",
        type=float,
        default=-0.06,
        help="new fixture bounded gate value",
    )
    parser.add_argument(
        "--load-pt",
        type=Path,
        help="load main.v3 tensors/caches and saved attrs exactly",
    )
    parser.add_argument(
        "--save-pt",
        type=Path,
        help="save main.v3 fixture; existing paths are never overwritten",
    )
    args = parser.parse_args(argv)
    if min(args.heads, args.length, args.warmup, args.iterations) < 1:
        parser.error("heads, length, warmup and iterations must all be positive")
    if not math.isfinite(args.lower_bound) or not -5.0 <= args.lower_bound < 0.0:
        parser.error("lower-bound must be finite and in [-5, 0)")
    if not args.device.startswith("npu"):
        parser.error("device must select an NPU")
    if args.load_pt is not None and not args.load_pt.is_file():
        parser.error(f"load-pt does not exist: {args.load_pt}")
    if args.save_pt is not None and args.save_pt.exists():
        parser.error(f"save-pt already exists: {args.save_pt}")
    return args


def _profile_build_fixture(args):
    """Generate one BF16 main-ABI fixture with the existing CPU chunk forward."""
    if getattr(args, "case", "b1") == "tnd_b31_t8192_n96_d128_ragged":
        return _tnd_b31_fixture()
    width = integration.SUPPORTED_HEAD_DIM
    generator = torch.Generator(device="cpu").manual_seed(73)

    def random_rows(*shape):
        return torch.randn(shape, generator=generator)

    def physical(rows):
        return (
            rows.transpose(1, 2).contiguous()
            if args.layout == "BNSD"
            else rows.contiguous()
        )

    token_shape = (1, args.length, args.heads, width)
    q, k = (physical(random_rows(*token_shape).bfloat16()) for _ in range(2))
    v = physical((0.2 * random_rows(*token_shape)).bfloat16())
    g = physical((0.2 * random_rows(*token_shape)).bfloat16())
    beta = physical(random_rows(*token_shape[:-1]).bfloat16())
    initial_state = 0.02 * random_rows(1, args.heads, width, width)
    a_log = torch.linspace(-3.8, -3.2, args.heads)
    dt_bias = 0.1 * random_rows(args.heads, width)
    attrs = dict(
        scale=width**-0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
        lower_bound=args.lower_bound,
        layout=args.layout,
    )
    print(
        f"Preparing CPU chunk caches: B=1 S={args.length} Nq=Nv={args.heads} D={width}",
        flush=True,
    )
    with torch.no_grad():
        cache = flash_kda_fwd(q, k, v, g, beta, initial_state, a_log, dt_bias, **attrs)
    inputs = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        do=physical((0.1 * random_rows(*token_shape)).bfloat16()),
        dht=0.02 * random_rows(*initial_state.shape),
    )
    inputs.update(
        {
            name: getattr(cache, "q_rstd" if name == "q_rtsd" else name)
            for name in _PROFILE_INPUT_NAMES[7:]
        }
    )
    inputs["q"] = physical(cache.q_norm.unsqueeze(0))
    inputs["k"] = physical(cache.k_norm.unsqueeze(0))
    inputs = _reference_forward_arguments(
        {**inputs, "layout": args.layout}, [args.length]
    )
    inputs.pop("layout")
    return dict(
        schema=_PROFILE_SCHEMA,
        main=dict(inputs=inputs),
        optional=dict(A_log=a_log, dt_bias=dt_bias, cu_seqlens=None),
        attrs=attrs,
        provenance=dict(
            seed=73,
            producer="test.flash_kda_fwd CPU chunk forward",
            initial_state=initial_state,
        ),
    )


def _profile_load_fixture(args):
    """Keep saved tensor layouts and attrs unchanged when loading a fixture."""
    fixture = (
        _profile_build_fixture(args)
        if args.load_pt is None
        else torch.load(args.load_pt, map_location="cpu", weights_only=True)
    )
    if not isinstance(fixture, dict) or fixture.get("schema") != _PROFILE_SCHEMA:
        raise ValueError(
            f"expected PT schema {_PROFILE_SCHEMA}; old raw-Q/K fixtures use another ABI; regenerate normalized inputs"
        )
    main = fixture.get("main")
    if not isinstance(main, dict) or not isinstance(main.get("inputs"), dict):
        raise ValueError("PT fixture requires main.inputs")
    for field, expected_names in (
        ("inputs", _PROFILE_INPUT_NAMES),
        ("optional", _PROFILE_OPTIONAL_NAMES),
        ("attrs", _PROFILE_ATTR_NAMES),
    ):
        values = main[field] if field == "inputs" else fixture.get(field)
        if not isinstance(values, dict) or set(values) != set(expected_names):
            raise ValueError(f"PT fixture {field} fields must be {expected_names}")
    for name, tensor in {**main["inputs"], **fixture["optional"]}.items():
        if tensor is not None and (
            not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
        ):
            raise ValueError(f"PT fixture {name} must be a CPU tensor or None")
    if args.load_pt is not None:
        print(
            f"Loaded {args.load_pt}; saved shapes, caches and attrs override generation options",
            flush=True,
        )
    if args.save_pt is not None:
        args.save_pt.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also closes the race after the initial existence check.
        with args.save_pt.open("xb") as output:
            torch.save(fixture, output)
        print(f"Saved reproducible main-ABI fixture: {args.save_pt}", flush=True)
    return fixture


def _profile_main(argv=None):
    args = _profile_parse_args(argv)

    import time
    import torch_npu

    fixture = _profile_load_fixture(args)
    torch.npu.set_device(args.device)
    device = torch.device(args.device)
    inputs = {
        name: None if tensor is None else tensor.to(device)
        for name, tensor in fixture["main"]["inputs"].items()
    }
    optional = {
        name: None if tensor is None else tensor.to(device)
        for name, tensor in fixture["optional"].items()
    }
    attrs = fixture["attrs"]
    # AICPU runs once before warmup. Timing markers contain only backward calls.
    metadata = flash_kda_grad_metadata(
        **_metadata_arguments(
            inputs["q"], inputs["v"], attrs["layout"], optional["cu_seqlens"]
        )
    )
    torch.npu.synchronize()
    print(
        f"q={tuple(inputs['q'].shape)} v={tuple(inputs['v'].shape)} dtype={inputs['q'].dtype}",
        flush=True,
    )
    print(f"attrs={attrs}; metadata_words={metadata.numel()}", flush=True)
    for phase, count in (("WARMUP", args.warmup), ("MEASURE", args.iterations)):
        for iteration in range(count):
            torch.npu.synchronize()
            marker_name = f"KDA_{phase}_{iteration:03d}"
            marker = torch_npu.npu.mstx.range_start(
                marker_name, stream=torch.npu.current_stream()
            )
            started = time.perf_counter()
            try:
                result = flash_kda_grad_npu(
                    disable_recompute=True,
                    **inputs,
                    A_log=optional["A_log"],
                    dt_bias=optional["dt_bias"],
                    metadata=metadata,
                    **attrs,
                )
                torch.npu.synchronize()
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                torch_npu.npu.mstx.range_end(marker)
            torch.npu.synchronize()
            print(
                f"{marker_name} python_wall_ms={elapsed_ms:.6f} (includes host work; NOT task_duration)",
                flush=True,
            )
            del result
    print(
        "Completed profiling calls. Accuracy was not checked by this performance entry.",
        flush=True,
    )
    print(
        "Read msprof task_duration inside KDA_MEASURE_*; exclude KDA_WARMUP_*.",
        flush=True,
    )


_BIN_SCHEMA = "flash_kda_grad.main.bin.v3"
_BIN_WIDTH = integration.SUPPORTED_HEAD_DIM
_BIN_CHUNK_SIZE = metadata_module.CHUNK_SIZE
_BIN_OUTPUT_NAMES = ("dq", "dk", "dv", "dh0", "dg", "dbeta", "dA", "dBias")
_BIN_REQUIRED_NAMES = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "do",
    "dht",
    "chunk_states",
    "Aqk",
    "Akk",
    "w",
    "u",
    "v_new",
    "qg",
    "kg",
    "g_cumsum",
)
_BIN_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
}


def _bin_parse_args(argv=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Compare current BF16 backward against saved golden bins."
    )
    parser.add_argument("--compare-bin", action="store_true", required=True)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument(
        "--naming", choices=("auto", "prefixed", "plain"), default="auto"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--nq", type=int, default=1)
    parser.add_argument("--nv", type=int, default=2)
    parser.add_argument("--layout", choices=("BSND", "BNSD"), default="BSND")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--cache-layout", choices=("forward",), default="forward")
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--beta-sigmoid", action="store_true")
    parser.add_argument("--allow-neg-eigval", action="store_true")
    parser.add_argument("--lower-bound", type=float, default=None)
    parser.add_argument("--scale", type=float, default=_BIN_WIDTH**-0.5)
    parser.add_argument("--atol", type=float, default=4e-3)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--max-relative-l2", type=float, default=0.05)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate files without running the NPU",
    )
    args = parser.parse_args(argv)
    if min(args.batch, args.length, args.nq, args.nv) < 1 or args.nv % args.nq:
        parser.error("batch/length/heads must be positive and nv divisible by nq")
    if any(
        not math.isfinite(x) or x < 0
        for x in (args.atol, args.rtol, args.max_relative_l2)
    ):
        parser.error("comparison tolerances must be finite and nonnegative")
    if not args.device.startswith("npu"):
        parser.error("device must select an NPU")
    return args


def _bin_tensor_path(folder, name, naming):
    prefix = "output_" if name in _BIN_OUTPUT_NAMES else "input_"
    return folder / f"{prefix if naming == 'prefixed' else ''}{name}.bin"


def _bin_specs(args):
    """Unambiguous fixed-length main ABI; raw bins do not encode layout or attrs."""
    batch, length, nq, nv, width = args.batch, args.length, args.nq, args.nv, _BIN_WIDTH
    chunk_size = _BIN_CHUNK_SIZE
    chunks = (length + chunk_size - 1) // chunk_size
    q_shape = (
        (batch, length, nq, width)
        if args.layout == "BSND"
        else (batch, nq, length, width)
    )
    v_shape = (
        (batch, length, nv, width)
        if args.layout == "BSND"
        else (batch, nv, length, width)
    )
    saved_q = (batch, nq, length, width)
    saved_v = (batch, nv, length, width)
    state_shape = (batch, nv, chunks, width, width)
    matrix_shape = (batch, nv, chunks, chunk_size, chunk_size)
    specs = {}

    def add(names, shape, dtype):
        for name in names.split():
            specs[name] = {"shape": list(shape), "dtype": dtype}

    add("q k dq dk", q_shape, "bfloat16")
    add("v g do dv dg", v_shape, "bfloat16")
    add("beta dbeta", v_shape[:-1], "bfloat16")
    add("w u v_new qg kg", saved_v, "bfloat16")
    add("Aqk Akk", matrix_shape, "bfloat16")
    add("chunk_states", state_shape, "float32")
    add("g_cumsum", saved_v, "float32")
    add("dht dh0", (batch, nv, width, width), "float32")
    add("q_rtsd k_rstd", saved_q[:-1], "float32")
    add("beta_sigmoid", saved_v[:-1], "float32")
    add("A_log dA", (nv,), "float32")
    add("dt_bias dBias", (nv, width), "float32")
    return specs


def _bin_read_tensor(path, spec):
    import sys

    shape, dtype = spec.get("shape"), spec.get("dtype")
    if (
        sys.byteorder != "little"
        or dtype not in _BIN_DTYPES
        or not isinstance(shape, list)
        or any(type(size) is not int or size < 0 for size in shape)
    ):
        raise ValueError(
            f"invalid little-endian main-ABI tensor spec for {path.name}: {spec}"
        )
    torch_dtype = _BIN_DTYPES[dtype]
    size = math.prod(shape) * torch.empty((), dtype=torch_dtype).element_size()
    raw = bytearray(path.read_bytes())
    if len(raw) != size:
        raise ValueError(
            f"{path.name}: got {len(raw)} bytes, expected {size} for {shape}/{dtype}"
        )
    if not size:
        return torch.empty(shape, dtype=torch_dtype)
    # BF16 payload is interpreted as BF16 bits, never converted from FP16.
    return torch.frombuffer(raw, dtype=torch_dtype).reshape(shape).clone()


def _bin_read_description(args):
    """Manifest wins over generation defaults, but never over the current ABI."""
    import json

    folder = args.bin_dir.resolve()
    manifest_path = folder / "manifest.json"
    data = None
    attrs = dict(
        layout=args.layout,
        scale=args.scale,
        use_qk_l2norm_in_kernel=args.norm,
        use_gate_in_kernel=args.gate,
        use_beta_sigmoid_in_kernel=args.beta_sigmoid,
        allow_neg_eigval=args.allow_neg_eigval,
        lower_bound=args.lower_bound,
    )
    specs = _bin_specs(args)
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("schema") != _BIN_SCHEMA
            or data.get("format") != "raw-contiguous-le-v1"
        ):
            raise ValueError(
                f"manifest must describe {_BIN_SCHEMA}; old raw-Q/K or FP16/combination ABI is incompatible"
            )
        attrs, specs = data.get("attrs"), data.get("tensors")
        print(
            "Using manifest shapes/dtypes/attrs; CLI generation defaults are ignored.",
            flush=True,
        )
    if not isinstance(attrs, dict) or set(attrs) != set(_PROFILE_ATTR_NAMES):
        raise ValueError("bin attributes must contain the complete public main ABI")
    if attrs["layout"] not in ("BSND", "BNSD"):
        raise ValueError(
            "bin comparison currently supports fixed-length BSND/BNSD only"
        )
    flags = (
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "use_beta_sigmoid_in_kernel",
        "allow_neg_eigval",
    )
    if any(type(attrs[name]) is not bool for name in flags):
        raise ValueError("bin fusion attributes must be booleans")
    if not isinstance(attrs["scale"], (float, int)) or not math.isfinite(
        attrs["scale"]
    ):
        raise ValueError("bin scale must be finite")
    bound = attrs["lower_bound"]
    if bound is not None and (
        not isinstance(bound, (float, int)) or not -5.0 <= bound < 0.0
    ):
        raise ValueError("lower_bound must be None or in [-5, 0)")
    if attrs["allow_neg_eigval"] and not attrs["use_beta_sigmoid_in_kernel"]:
        raise ValueError("allow_neg_eigval requires beta sigmoid")
    if not isinstance(specs, dict) or set(specs) - set(
        (*_PROFILE_INPUT_NAMES, *_PROFILE_OPTIONAL_NAMES, *_BIN_OUTPUT_NAMES)
    ):
        raise ValueError("unknown tensor in bin manifest")
    return folder, data, attrs, specs


def _bin_resolve_naming(folder, requested, manifest):
    naming = requested
    if naming == "auto":
        naming = manifest.get("naming", "auto") if manifest else "auto"
    if naming == "auto":
        matches = [
            mode
            for mode in ("prefixed", "plain")
            if _bin_tensor_path(folder, "q", mode).is_file()
            and _bin_tensor_path(folder, "dq", mode).is_file()
        ]
        if len(matches) != 1:
            raise ValueError(
                "cannot infer bin naming; specify --naming prefixed or plain"
            )
        naming = matches[0]
    if naming not in ("prefixed", "plain"):
        raise ValueError("invalid bin naming")
    return naming


def _bin_load_inputs(folder, naming, attrs, specs, has_manifest):
    import hashlib

    for obsolete in ("q_norm", "k_norm", "q_rstd"):
        if _bin_tensor_path(folder, obsolete, naming).exists():
            raise ValueError(
                "old Q/K cache files found; regenerate bins for the normalized q/k ABI"
            )
    required = set(_BIN_REQUIRED_NAMES)
    if attrs["use_qk_l2norm_in_kernel"]:
        required.update(("q_rtsd", "k_rstd"))
    if attrs["use_beta_sigmoid_in_kernel"]:
        required.add("beta_sigmoid")
    if attrs["use_gate_in_kernel"]:
        required.add("A_log")
    values, hashes = {}, {}
    fp32_names = {
        "dht",
        "chunk_states",
        "g_cumsum",
        "q_rtsd",
        "k_rstd",
        "beta_sigmoid",
        "A_log",
        "dt_bias",
    }
    for name in (*_PROFILE_INPUT_NAMES, "A_log", "dt_bias"):
        path = _bin_tensor_path(folder, name, naming)
        if not path.exists():
            if name in required or (has_manifest and specs.get(name) is not None):
                raise FileNotFoundError(f"required main-ABI tensor: {path}")
            values[name] = None
            continue
        spec = specs.get(name)
        expected_dtype = "float32" if name in fp32_names else "bfloat16"
        if not isinstance(spec, dict) or spec.get("dtype") != expected_dtype:
            raise ValueError(
                f"{name} must be {expected_dtype}; regenerate incompatible old bins"
            )
        values[name] = _bin_read_tensor(path, spec)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return values, hashes


def _bin_validate_shapes(values, layout):
    """Validate the relationships that raw byte counts alone cannot establish."""
    q, v = values["q"], values["v"]
    if q.ndim != 4 or v.ndim != 4 or any(size < 1 for size in (*q.shape, *v.shape)):
        raise ValueError("fixed-length q/v must have four positive dimensions")
    head_axis, sequence_axis = (2, 1) if layout == "BSND" else (1, 2)
    if q.shape[-1] != _BIN_WIDTH or v.shape[-1] != _BIN_WIDTH:
        raise ValueError(f"current bin comparison requires Dqk=Dv={_BIN_WIDTH}")
    if v.shape[head_axis] % q.shape[head_axis]:
        raise ValueError("value heads must be divisible by Q/K heads")
    dimensions = SimpleNamespace(
        batch=q.shape[0],
        length=q.shape[sequence_axis],
        nq=q.shape[head_axis],
        nv=v.shape[head_axis],
        layout=layout,
        cache_layout="forward",
    )
    expected = _bin_specs(dimensions)
    for name, value in values.items():
        if value is not None and list(value.shape) != expected[name]["shape"]:
            raise ValueError(
                f"{name}: main-ABI shape must match the current forward cache {expected[name]['shape']}"
            )
    if values["Aqk"].shape != values["Akk"].shape:
        raise ValueError("Aqk/Akk shapes must match")


def _bin_load_references(folder, naming, values, attrs, specs, has_manifest):
    import hashlib

    refs, hashes = {}, {}
    expected_shapes = {
        "dq": values["q"].shape,
        "dk": values["k"].shape,
        "dv": values["v"].shape,
        "dg": values["g"].shape,
        "dbeta": values["beta"].shape,
        "dh0": (
            values["q"].shape[0],
            values["v"].shape[2 if attrs["layout"] == "BSND" else 1],
            _BIN_WIDTH,
            _BIN_WIDTH,
        ),
        "dA": None if not attrs["use_gate_in_kernel"] else values["A_log"].shape,
        "dBias": None
        if not attrs["use_gate_in_kernel"] or values["dt_bias"] is None
        else values["dt_bias"].shape,
    }
    for name, shape in expected_shapes.items():
        path = _bin_tensor_path(folder, name, naming)
        if shape is None:
            if path.exists() or (has_manifest and specs.get(name) is not None):
                raise ValueError(
                    f"{name} is None for these attrs; remove stale output or correct the attrs"
                )
            refs[name] = None
            continue
        spec = specs.get(name)
        if not isinstance(spec, dict) or tuple(spec.get("shape", ())) != tuple(shape):
            raise ValueError(f"{name} golden shape must be {tuple(shape)}")
        if not has_manifest:
            count = math.prod(shape)
            inferred = {count * 4: "float32", count * 8: "float64"}
            if name in ("dq", "dk", "dv", "dg", "dbeta"):
                inferred[count * 2] = "bfloat16"
            spec = {**spec, "dtype": inferred.get(path.stat().st_size)}
        if spec.get("dtype") not in ("bfloat16", "float32", "float64"):
            raise ValueError(f"unsupported golden dtype for {name}")
        refs[name] = _bin_read_tensor(path, spec)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return refs, hashes


def _bin_load_bundle(args):
    folder, manifest, attrs, specs = _bin_read_description(args)
    naming = _bin_resolve_naming(folder, args.naming, manifest)
    if (
        _bin_tensor_path(folder, "cu_seqlens", naming).exists()
        or specs.get("cu_seqlens") is not None
    ):
        raise ValueError(
            "bin comparison requires fixed-length inputs without cu_seqlens"
        )
    values, input_hashes = _bin_load_inputs(
        folder, naming, attrs, specs, manifest is not None
    )
    _bin_validate_shapes(values, attrs["layout"])
    refs, output_hashes = _bin_load_references(
        folder, naming, values, attrs, specs, manifest is not None
    )
    return values, refs, attrs, naming, {**input_hashes, **output_hashes}


def _bin_metrics(actual, reference, atol, rtol, max_relative_l2):
    got, want = actual.detach().cpu().double(), reference.detach().cpu().double()
    if got.shape != want.shape:
        raise ValueError(
            f"output shape mismatch: {tuple(got.shape)} vs {tuple(want.shape)}"
        )
    delta = (got - want).abs()
    finite = torch.isfinite(got) & torch.isfinite(want) & torch.isfinite(delta)
    within = finite & (delta <= atol + rtol * want.abs())
    count, total = int(within.sum()), want.numel()
    all_finite = bool(finite.all())
    ref_norm = float(torch.linalg.vector_norm(want)) if all_finite else 0.0
    error_norm = float(torch.linalg.vector_norm(delta)) if all_finite else 0.0
    relative_l2 = error_norm / ref_norm if all_finite and ref_norm > 1e-12 else None
    derived_finite = math.isfinite(ref_norm) and math.isfinite(error_norm)
    l2_passed = derived_finite and (
        relative_l2 is None or relative_l2 <= max_relative_l2
    )
    return dict(
        max_abs=float(delta.max())
        if all_finite and total
        else 0.0
        if not total
        else None,
        relative_l2=relative_l2
        if relative_l2 is None or math.isfinite(relative_l2)
        else None,
        nonfinite_elements=total - int(finite.sum()),
        passed_elements=count,
        total_elements=total,
        element_pass_rate_pct=100.0 * count / total if total else 100.0,
        atol=atol,
        rtol=rtol,
        max_relative_l2=max_relative_l2,
        passed=all_finite and count == total and l2_passed,
    )


def _bin_report_path(args, naming, hashes):
    output_path = args.json or args.bin_dir / "compare.json"
    protected = {
        _bin_tensor_path(args.bin_dir, name, naming).resolve() for name in hashes
    }
    protected.add((args.bin_dir / "manifest.json").resolve())
    if output_path.resolve() in protected:
        raise ValueError(
            "comparison JSON must not overwrite an input, golden, or manifest"
        )
    return output_path


def _bin_run_backward(values, attrs, device):
    import torch_npu

    torch_npu.npu.set_device(device)
    tensors = {
        name: None if value is None else value.to(device)
        for name, value in values.items()
    }
    metadata = flash_kda_grad_metadata(
        **_metadata_arguments(tensors["q"], tensors["v"], attrs["layout"])
    )
    actual = flash_kda_grad_npu(
        disable_recompute=True, **tensors, metadata=metadata, **attrs
    )
    torch.npu.synchronize()
    return actual


def _bin_evaluate_outputs(actual, refs, args):
    if len(actual) != len(_BIN_OUTPUT_NAMES):
        raise ValueError("backward must return all eight main-ABI output slots")
    rows = {}
    for name, got in zip(_BIN_OUTPUT_NAMES, actual):
        ref = refs[name]
        if got is None or ref is None:
            if got is not None or ref is not None:
                raise ValueError(
                    f"{name}: optional output differs from the supplied golden contract"
                )
            print(f"{name}: N/A (None for current attrs)", flush=True)
            continue
        row = _bin_metrics(got, ref, args.atol, args.rtol, args.max_relative_l2)
        row.update(actual_dtype=str(got.dtype), golden_dtype=str(ref.dtype))
        rows[name] = row
        print(
            f"{name}: max_abs={row['max_abs']} rel_l2={row['relative_l2']} "
            f"element_pass={row['element_pass_rate_pct']:.4f}% passed={row['passed']}",
            flush=True,
        )
    total = sum(row["total_elements"] for row in rows.values())
    count = sum(row["passed_elements"] for row in rows.values())
    passed = bool(rows) and all(row["passed"] for row in rows.values())
    maximum = (
        max(row["max_abs"] for row in rows.values())
        if all(row["max_abs"] is not None for row in rows.values())
        else None
    )
    summary = dict(
        global_max_abs=maximum,
        outputs_passed=sum(row["passed"] for row in rows.values()),
        outputs_total=len(rows),
        output_pass_rate_pct=100.0
        * sum(row["passed"] for row in rows.values())
        / len(rows),
        elements_passed=count,
        elements_total=total,
        element_pass_rate_pct=100.0 * count / total if total else 100.0,
    )
    return rows, summary, passed


def _bin_compare_main(argv=None):
    import json

    args = _bin_parse_args(argv)
    values, refs, attrs, naming, hashes = _bin_load_bundle(args)
    output_path = _bin_report_path(args, naming, hashes)
    print(
        f"Loaded BF16 bins: q={tuple(values['q'].shape)}, v={tuple(values['v'].shape)}, attrs={attrs}",
        flush=True,
    )
    if args.check_only:
        print(
            "Bin files validated; no NPU execution or accuracy comparison performed.",
            flush=True,
        )
        return 0
    actual = _bin_run_backward(values, attrs, args.device)
    rows, summary, passed = _bin_evaluate_outputs(actual, refs, args)
    report = dict(
        schema="flash_kda_grad.bin.compare.v1",
        bin_dir=str(args.bin_dir.resolve()),
        naming=naming,
        attrs=attrs,
        device=args.device,
        file_sha256=hashes,
        results=rows,
        passed=passed,
        summary=summary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"BIN GOLDEN: {'PASS' if passed else 'FAIL'}; max_abs={summary['global_max_abs']}; "
        f"element_pass={report['summary']['element_pass_rate_pct']:.4f}%; JSON={output_path}",
        flush=True,
    )
    return 0 if passed else 1


TND_B31_LENGTHS = tuple(128 + 9 * index for index in range(30)) + (437,)


def _tnd_b31_fixture():
    lengths = TND_B31_LENGTHS
    assert len(lengths) == 31 and sum(lengths) == 8192
    generator = torch.Generator().manual_seed(73)

    def rand(*shape):
        return torch.randn(shape, generator=generator)

    q, k = (rand(8192, 96, 128).bfloat16() for _ in range(2))
    v = (0.2 * rand(8192, 96, 128)).bfloat16()
    g = (0.2 * rand(8192, 96, 128)).bfloat16()
    beta = rand(8192, 96).bfloat16()
    h0 = 0.02 * rand(31, 96, 128, 128)
    a_log = torch.linspace(-3.8, -3.2, 96)
    bias = 0.1 * rand(96, 128)
    do = (0.1 * rand(8192, 96, 128)).bfloat16()
    dht = 0.02 * rand(31, 96, 128, 128)
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    attrs = dict(
        scale=128**-0.5,
        layout="TND",
        lower_bound=-0.06,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
    )
    print(f"TND B=31 T=8192 Nq=Nv=96 D=128 BF16; lengths={lengths}", flush=True)
    cache = flash_kda_fwd(
        q, k, v, g, beta, h0, a_log, bias, cu_seqlens=offsets, **attrs
    )
    inputs = dict(q=q, k=k, v=v, g=g, beta=beta, do=do, dht=dht)
    inputs.update(
        {
            name: getattr(cache, "q_rstd" if name == "q_rtsd" else name)
            for name in _PROFILE_INPUT_NAMES[7:]
        }
    )
    inputs["q"], inputs["k"] = cache.q_norm, cache.k_norm
    inputs = _reference_forward_arguments({**inputs, "layout": "TND"}, lengths)
    inputs.pop("layout")
    return dict(
        schema=_PROFILE_SCHEMA,
        main=dict(inputs=inputs),
        optional=dict(A_log=a_log, dt_bias=bias, cu_seqlens=offsets),
        attrs=attrs,
        provenance=dict(
            seed=73,
            producer="test.flash_kda_fwd CPU chunk forward",
            initial_state=h0,
            sequence_lengths=lengths,
            raw_q=q,
            raw_k=k,
        ),
    )


# Unit, integration and reference checks.


metadata_capacity = metadata_module.metadata_capacity
HEADER_STATUS = metadata_module.HEADER_STATUS
STATUS_BAD_LENGTH = metadata_module.STATUS_BAD_LENGTH
StageTwoResult = integration.StageTwoResult
_prepare_stage_one = integration._prepare_stage_one
_prepare_stage_two = integration._prepare_stage_two


@pytest.fixture(scope="module", autouse=True)
def _register_npu_before_autograd(request):
    # Mixed runs must register the device before the first CPU backward creates its queues.
    has_npu_cases = any(
        item.get_closest_marker("npu") is not None
        for item in request.session.items
        if item.module is request.module
    )
    if has_npu_cases and importlib.util.find_spec("torch_npu") is not None:
        importlib.import_module("torch_npu")


@pytest.mark.whitebox
def test_vf_branches_are_compile_time_only():
    """Runtime switches belong outside VF; row boundaries use vector masks."""
    source = ast.parse(inspect.getsource(integration))
    violations = []
    for scope in ast.walk(source):
        if not isinstance(scope, ast.With):
            continue
        if not any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "vf"
            for item in scope.items
        ):
            continue
        for branch in ast.walk(scope):
            if isinstance(branch, ast.If) and not (
                isinstance(branch.test, ast.Call)
                and isinstance(branch.test.func, ast.Name)
                and branch.test.func.id == "const_expr"
            ):
                violations.append(branch.lineno)
    assert not violations, f"runtime if inside VF at lines {violations}"


@pytest.mark.whitebox
def test_gate_reduction_assigns_one_half_per_aiv():
    """Each AIV owns disjoint gate channels and adds its own dA partial."""
    source = ast.parse(inspect.getsource(integration))
    vector = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "StageOneVector"
    )
    finish = next(
        node
        for node in vector.body
        if isinstance(node, ast.FunctionDef) and node.name == "finish_gate_task_atomic"
    )
    atomic_store = [
        call
        for call in ast.walk(finish)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "mem_copy"
        and any(
            keyword.arg == "options"
            and isinstance(keyword.value, ast.Call)
            and isinstance(keyword.value.func, ast.Name)
            and keyword.value.func.id == "copy_options"
            and any(
                option.arg == "atomic_add"
                and isinstance(option.value, ast.Constant)
                and option.value.value is True
                for option in keyword.value.keywords
            )
            for keyword in call.keywords
        )
    ]
    owned_halves = [
        loop
        for loop in ast.walk(finish)
        if isinstance(loop, ast.For)
        and isinstance(loop.target, ast.Name)
        and loop.target.id == "half"
        and isinstance(loop.iter, ast.Call)
        and isinstance(loop.iter.func, ast.Name)
        and loop.iter.func.id == "range"
        and len(loop.iter.args) == 3
        and isinstance(loop.iter.args[0], ast.Call)
        and isinstance(loop.iter.args[0].func, ast.Name)
        and loop.iter.args[0].func.id == "get_subblock_id"
    ]
    assert len(atomic_store) == 2
    assert len(owned_halves) == 1


@pytest.mark.whitebox
def test_fused_gate_reduction_keeps_chunk_sums_in_ub():
    """The fused path must avoid per-chunk gate-part writes and reloads."""
    source = ast.parse(inspect.getsource(integration))
    body = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_finish_masked_track_chunk"
    )
    called = {
        call.func.attr
        for call in ast.walk(body)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }
    assert "accumulate_gate_chunk" in called
    assert "finish_gate_task_atomic" in called


@pytest.mark.whitebox
def test_head_reduction_assigns_disjoint_token_rows_to_both_aivs():
    source = ast.parse(inspect.getsource(integration))
    body = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_finish_masked_track_chunk"
    )
    head_writes = [
        call
        for call in ast.walk(body)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "finish_head_row"
    ]
    assert len(head_writes) == 2
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_subblock_id"
        for node in ast.walk(body)
    )
    assert not any(
        branch
        for branch in ast.walk(body)
        if isinstance(branch, ast.If)
        and isinstance(branch.test, ast.Compare)
        and isinstance(branch.test.left, ast.Call)
        and isinstance(branch.test.left.func, ast.Name)
        and branch.test.left.func.id == "get_subblock_id"
        and any(
            write in ast.walk(node) for node in branch.body for write in head_writes
        )
    )


@pytest.mark.parametrize(
    "physical_cores, expected", [(1, 1), (16, 16), (32, 32), (40, 32)]
)
def test_backward_launch_uses_platform_cube_cores_up_to_32(
    monkeypatch,
    physical_cores,
    expected,
):
    stream = object()
    fake_npu = SimpleNamespace(
        current_device=lambda: 0, current_stream=lambda device: stream
    )
    monkeypatch.setattr(integration.torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(
        integration,
        "get_platform_info",
        lambda stream: SimpleNamespace(cube_core_num=physical_cores),
        raising=False,
    )
    assert integration._device_block_num() == expected


@pytest.mark.parametrize(
    "length,qk_heads,value_heads,expected_slots",
    [
        pytest.param(8192, 96, 96, 32, id="8k-96-heads"),
        pytest.param(2113, 2, 2, 32, id="tail-after-core-reuse"),
        pytest.param(65, 2, 2, 32, id="fewer-chunks-than-cores"),
        pytest.param(64, 2, 2, 32, id="single-chunk"),
        pytest.param(65, 32, 32, 32, id="one-owner-per-core"),
        pytest.param(0, 2, 2, 0, id="empty-compact-cache"),
        pytest.param(2113, 1, 2, 64, id="gqa-preserves-head-partials"),
    ],
)
def test_one_track_chunk_workspace_is_bounded_for_equal_and_grouped_heads(
    monkeypatch,
    length,
    qk_heads,
    value_heads,
    expected_slots,
):
    """Exercise actual allocation shapes without reserving multi-GiB memory.

    Meta tensors run the host allocator and TensorSpec construction. In the
    one-track path, each core retains one chunk per value head in its current
    Q/K-head group, independently of sequence length and owner count.
    """
    monkeypatch.setattr(integration, "_device_block_num", lambda ref=None: 32)
    chunk_capacity = (length + 63) // 64

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")

    q = empty((length, qk_heads, 128))
    v = empty((length, value_heads, 128))
    matrix = empty((chunk_capacity, value_heads, 64, 64))
    stage_two = StageTwoResult(
        empty(v.shape, torch.float32),
        empty(matrix.shape, torch.float32),
        empty(v.shape, torch.float32),
        empty(v.shape, torch.float32),
        empty(v.shape, torch.float32),
        empty((chunk_capacity, value_heads, 128), torch.float32),
        empty((1, value_heads, 128, 128), torch.float32),
    )
    launches = []

    class CaptureArguments:
        def __call__(self, *args):
            launches.append(args)

        def close(self):
            pass

    monkeypatch.setattr(
        integration.cannbotdsl,
        "compile",
        lambda *args: CaptureArguments(),
    )
    prepared = _prepare_stage_one(
        query=q,
        key=q,
        values=v,
        raw_gate=v,
        activated_beta=empty((length, value_heads), torch.float32),
        saved_inverse=matrix,
        saved_qg=v,
        saved_kg=v,
        saved_gamma=empty(v.shape, torch.float32),
        A_log=empty((value_heads,), torch.float32),
        dt_bias=empty((value_heads, 128), torch.float32),
        query_rstd=empty(q.shape[:-1], torch.float32),
        key_rstd=empty(q.shape[:-1], torch.float32),
        stage_two=stage_two,
        layout="TND",
        metadata=empty((metadata_capacity(1, value_heads),), torch.int32),
        scale=128**-0.5,
        fused_beta=True,
        fused_gate=True,
        use_l2norm=True,
        inverse_factor=0.5,
        lower_bound=-0.06,
    )
    _, arguments, specs, _ = prepared
    assert not launches
    assert all(
        tuple(spec.shape) == tuple(arg.shape) for spec, arg in zip(specs, arguments)
    )

    # Only live scratch tensors remain; no causal-feature alias planes.
    scratch = arguments[-30:]
    assert len(scratch) == 30
    assert scratch[17].shape == (expected_slots, 128)  # gate endpoint gradient
    for index, tensor in enumerate(scratch):
        if index != 17:
            assert tensor.shape[0] == expected_slots * 64
    scratch_bytes = sum(tensor.numel() * tensor.element_size() for tensor in scratch)
    assert scratch_bytes == expected_slots * 599_424


@pytest.fixture
def _isolated_backward_workspace_leases(monkeypatch):
    """Keep lifecycle checks independent of real NPU calls and cache state."""
    leases = OrderedDict()
    monkeypatch.setattr(integration, "_BACKWARD_WORKSPACE_LEASES", leases)
    monkeypatch.setattr(
        integration.torch,
        "npu",
        SimpleNamespace(device=lambda device: nullcontext()),
        raising=False,
    )
    return leases


@pytest.fixture
def _isolated_backward_retirement_worker(
    monkeypatch, _isolated_backward_workspace_leases
):
    monkeypatch.setattr(integration, "_BACKWARD_RETIREMENT_QUEUE", None)
    monkeypatch.setattr(integration, "_BACKWARD_RETIREMENT_THREAD", None)
    monkeypatch.setattr(integration, "_BACKWARD_PROGRAMS", OrderedDict())
    yield _isolated_backward_workspace_leases
    integration._stop_backward_workspace_retirement()


class _WorkspaceCompletionEvent:
    def __init__(self, *, completed=False, synchronization_error=None):
        self.completed = completed
        self.synchronization_error = synchronization_error

    def query(self):
        return self.completed

    def synchronize(self):
        if self.synchronization_error is not None:
            raise self.synchronization_error
        self.completed = True


def _tracked_workspace_lease(leases, event):
    # A meta tensor owns no device storage, but its Python lifetime is real.
    owner = torch.empty((1,), device="meta")
    owner_reference = weakref.ref(owner)
    lease = integration._BackwardWorkspaceLease(event, "npu:0", (owner,))
    leases[id(lease)] = lease
    return lease, owner_reference


def test_backward_workspace_reap_preserves_pending_owners_and_releases_completed_ones(
    _isolated_backward_workspace_leases,
):
    leases = _isolated_backward_workspace_leases
    event = _WorkspaceCompletionEvent()
    lease, owner_reference = _tracked_workspace_lease(leases, event)

    integration._reap_backward_workspaces()
    gc.collect()
    assert owner_reference() is not None
    assert leases[id(lease)] is lease

    event.completed = True
    integration._reap_backward_workspaces()
    gc.collect()
    assert owner_reference() is None
    assert lease.arguments is None
    assert not leases


def test_backward_workspace_retirement_releases_without_another_operator_call(
    _isolated_backward_workspace_leases,
):
    leases = _isolated_backward_workspace_leases
    wait_started = threading.Event()
    device_completed = threading.Event()
    wait_timeout = 5.0

    class BlockingCompletionEvent(_WorkspaceCompletionEvent):
        def synchronize(self):
            wait_started.set()
            if not device_completed.wait(timeout=wait_timeout):
                raise TimeoutError("fake device completion was not signalled")
            super().synchronize()

    lease, owner_reference = _tracked_workspace_lease(leases, BlockingCompletionEvent())
    worker = threading.Thread(
        target=integration._retire_backward_workspace_lease, args=(lease,)
    )
    worker.start()
    try:
        assert wait_started.wait(timeout=wait_timeout)
        gc.collect()
        assert owner_reference() is not None
        device_completed.set()
        worker.join(timeout=wait_timeout)
        assert not worker.is_alive()
        gc.collect()
        assert owner_reference() is None
        assert lease.arguments is None
        assert not leases
    finally:
        # Always unblock and join before pytest restores the isolated globals.
        device_completed.set()
        worker.join(timeout=wait_timeout)


def test_backward_workspace_failed_wait_keeps_owners_until_successful_retry(
    _isolated_backward_workspace_leases,
):
    leases = _isolated_backward_workspace_leases
    error = RuntimeError("fake device wait failed")
    event = _WorkspaceCompletionEvent(synchronization_error=error)
    lease, owner_reference = _tracked_workspace_lease(leases, event)

    integration._retire_backward_workspace_lease(lease)
    assert lease.error is error
    with pytest.raises(RuntimeError, match="workspace retirement failed"):
        integration._reap_backward_workspaces()
    gc.collect()
    assert owner_reference() is not None
    assert leases[id(lease)] is lease

    event.synchronization_error = None
    integration._retire_backward_workspace_lease(lease)
    gc.collect()
    assert owner_reference() is None
    assert lease.error is None
    assert not leases


@pytest.mark.parametrize("release_path", ["retire", "reap"])
def test_backward_workspace_release_uses_owner_device_outside_cache_lock(
    monkeypatch,
    _isolated_backward_workspace_leases,
    release_path,
):
    leases = _isolated_backward_workspace_leases
    state = threading.local()
    observed = []

    class DeviceGuard:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            self.previous = getattr(state, "device", "npu:0")
            state.device = self.device

        def __exit__(self, *args):
            state.device = self.previous

    class Owner:
        def __del__(self):
            # Python Tensor destruction can enter the native allocator here.
            observed.append(
                (
                    getattr(state, "device", "npu:0"),
                    integration._BACKWARD_PROGRAM_LOCK._is_owned(),
                )
            )

    monkeypatch.setattr(integration.torch.npu, "device", DeviceGuard)
    lease = integration._BackwardWorkspaceLease(
        _WorkspaceCompletionEvent(completed=True),
        "npu:7",
        (Owner(),),
    )
    leases[id(lease)] = lease
    target = (
        integration._retire_backward_workspace_lease
        if release_path == "retire"
        else integration._reap_backward_workspaces
    )
    worker = threading.Thread(
        target=target, args=(lease,) if release_path == "retire" else ()
    )
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert observed == [("npu:7", False)]
    assert lease.arguments is None
    assert not leases


def test_backward_workspace_worker_binds_each_device_through_event_destruction(
    monkeypatch,
    _isolated_backward_retirement_worker,
):
    leases = _isolated_backward_retirement_worker
    state = threading.local()
    observed = []
    destroyed = {device: threading.Event() for device in ("npu:7", "npu:3")}

    class DeviceGuard:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            self.previous = getattr(state, "device", "npu:0")
            state.device = self.device

        def __exit__(self, *args):
            state.device = self.previous

    def set_device(device):
        state.device = device
        observed.append(("bind", device))

    def current_stream(device):
        assert state.device == device
        observed.append(("initialize_stream", device))

    class CompletionEvent:
        def __init__(self, device):
            self.device = device

        def synchronize(self):
            observed.append(("synchronize", state.device))

        def __del__(self):
            observed.append(("event_destructor", state.device))
            destroyed[self.device].set()

    class Owner:
        def __del__(self):
            observed.append(("owner_destructor", state.device))

    monkeypatch.setattr(
        integration.torch,
        "npu",
        SimpleNamespace(
            device=DeviceGuard, set_device=set_device, current_stream=current_stream
        ),
    )
    for device in destroyed:
        with integration._BACKWARD_PROGRAM_LOCK:
            lease = integration._BackwardWorkspaceLease(
                CompletionEvent(device), device, (Owner(),)
            )
            leases[id(lease)] = lease
            integration._queue_backward_workspace_retirement(lease)
            del lease
        assert destroyed[device].wait(timeout=5.0)
    assert observed == [
        (operation, device)
        for device in destroyed
        for operation in (
            "bind",
            "initialize_stream",
            "synchronize",
            "owner_destructor",
            "event_destructor",
        )
    ]
    worker = integration._BACKWARD_RETIREMENT_THREAD
    integration._clear_backward_program_cache()
    assert not worker.is_alive()
    assert integration._BACKWARD_RETIREMENT_THREAD is None
    assert integration._BACKWARD_RETIREMENT_QUEUE is None
    assert not leases


def test_backward_workspace_worker_initialization_failure_preserves_owners(
    monkeypatch,
    _isolated_backward_retirement_worker,
):
    leases = _isolated_backward_retirement_worker
    initialization_attempted = threading.Event()
    error = RuntimeError("fake stream initialization failed")

    def current_stream(device):
        initialization_attempted.set()
        raise error

    monkeypatch.setattr(
        integration.torch.npu, "set_device", lambda device: None, raising=False
    )
    monkeypatch.setattr(
        integration.torch.npu, "current_stream", current_stream, raising=False
    )
    lease, owner_reference = _tracked_workspace_lease(
        leases, _WorkspaceCompletionEvent()
    )
    with integration._BACKWARD_PROGRAM_LOCK:
        integration._queue_backward_workspace_retirement(lease)
    assert initialization_attempted.wait(timeout=5.0)
    # Stop/join also proves the error did not silently kill the worker.
    integration._stop_backward_workspace_retirement()
    assert lease.error is error
    assert owner_reference() is not None
    assert leases[id(lease)] is lease
    with pytest.raises(RuntimeError, match="workspace retirement failed"):
        integration._reap_backward_workspaces()
    integration._clear_backward_program_cache()
    assert owner_reference() is None
    assert not leases


def test_backward_reaps_completed_workspace_before_next_stage_two_allocation(
    monkeypatch,
    _isolated_backward_workspace_leases,
    metadata_x86,
):
    leases = _isolated_backward_workspace_leases
    _, owner_reference = _tracked_workspace_lease(
        leases, _WorkspaceCompletionEvent(completed=True)
    )

    class AllocationBoundaryReached(Exception):
        pass

    def next_stage_two_allocation(**arguments):
        gc.collect()
        assert owner_reference() is None
        assert not leases
        raise AllocationBoundaryReached

    monkeypatch.setattr(integration, "_validate_backward_inputs", lambda *args: None)
    monkeypatch.setattr(integration, "_device_block_num", lambda ref=None: 32)
    monkeypatch.setattr(integration, "_prepare_stage_two", next_stage_two_allocation)
    # ABI validation is isolated here: stop at the allocation boundary before
    # placeholders can reach any math, TensorSpec construction or device call.
    arguments = {
        name: None
        for name, parameter in inspect.signature(
            integration.flash_kda_grad_npu
        ).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    arguments["scale"] = 128**-0.5
    # Host schedule selection reads dimensions and effective lengths even when
    # there are fewer owners than cores, so use actual metadata kernel output.
    arguments["q"] = torch.empty((1, 1, 1, 128))
    arguments["v"] = arguments["q"]
    arguments["layout"] = "BNSD"
    arguments["Aqk"] = arguments["Akk"] = torch.empty((1, 1, 1, 64, 64))
    arguments["metadata"] = metadata_x86(
        device=arguments["q"].device,
        cu_seqlens=None,
        capacity=metadata_capacity(1, 1),
        batch=1,
        value_heads=1,
        storage_length=1,
        is_packed=False,
    )
    monkeypatch.setattr(
        integration._BACKWARD_SCRATCH, "acquire", lambda *args: nullcontext(None)
    )
    with pytest.raises(AllocationBoundaryReached):
        integration.flash_kda_grad_npu(disable_recompute=True, **arguments)


@pytest.mark.parametrize("length", [1, 2, 64])
def test_unit_lower_inverse_series_matches_triangular_solve(length):
    torch.manual_seed(819)
    lower = torch.tril(torch.randn(length, length) * 0.03, diagonal=-1)
    identity = torch.eye(length)
    expected = torch.linalg.solve_triangular(
        identity + lower, identity, upper=False, unitriangular=True
    )
    actual = _unit_lower_inverse_series(lower, identity)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def _inputs():
    torch.manual_seed(4)
    q = torch.randn(5, 1, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(5, 1, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(5, 2, 2, dtype=torch.float64, requires_grad=True)
    g = torch.randn(5, 2, 3, dtype=torch.float64, requires_grad=True)
    beta = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    h0 = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    a_log = torch.randn(2, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    return q, k, v, g, beta, h0, a_log, bias


def _effective(
    q,
    k,
    g,
    beta,
    a_log,
    bias,
    *,
    lower_bound,
    beta_factor,
    norm=True,
    fused_gate=True,
    fused_beta=True,
):
    qn = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) if norm else q
    kn = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6) if norm else k
    if not fused_gate:
        gate = g
    elif lower_bound is None:
        gate = -a_log.exp()[None, :, None] * torch.nn.functional.softplus(
            g + bias[None]
        )
    else:
        gate = lower_bound * torch.sigmoid(
            a_log.exp()[None, :, None] * (g + bias[None])
        )
    b = beta_factor * torch.sigmoid(beta) if fused_beta else beta
    return qn, kn, gate, b


def _token_reference(
    q,
    k,
    v,
    g,
    beta,
    h0,
    a_log,
    bias,
    *,
    scale,
    lower_bound,
    beta_factor,
    norm=True,
    fused_gate=True,
    fused_beta=True,
):
    token_count = q.shape[0]
    qn, kn, gate, b = _effective(
        q,
        k,
        g,
        beta,
        a_log,
        bias,
        lower_bound=lower_bound,
        beta_factor=beta_factor,
        norm=norm,
        fused_gate=fused_gate,
        fused_beta=fused_beta,
    )
    outputs = []
    final = []
    for head in range(2):
        state = h0[0, head].T
        rows = []
        for token in range(token_count):
            key = kn[token, 0]
            state = gate[token, head].exp()[:, None] * state
            delta = b[token, head] * (v[token, head] - key @ state)
            state = state + key[:, None] * delta[None]
            rows.append(scale * qn[token, 0] @ state)
        outputs.append(torch.stack(rows))
        final.append(state.T)
    return torch.stack(outputs, dim=1), torch.stack(final)[None]


def _chunk_cache(
    q,
    k,
    v,
    g,
    beta,
    h0,
    a_log,
    bias,
    *,
    scale,
    lower_bound,
    beta_factor,
    norm=True,
    fused_gate=True,
    fused_beta=True,
):
    """Build exact forward intermediates without using the backward implementation."""
    token_count = q.shape[0]
    chunk_count = (token_count + 1) // 2
    q_rstd = torch.rsqrt(q.square().sum(-1) + 1e-6)
    k_rstd = torch.rsqrt(k.square().sum(-1) + 1e-6)
    qn, kn, gate, b = _effective(
        q,
        k,
        g,
        beta,
        a_log,
        bias,
        lower_bound=lower_bound,
        beta_factor=beta_factor,
        norm=norm,
        fused_gate=fused_gate,
        fused_beta=fused_beta,
    )
    aqk = torch.zeros(chunk_count, 2, 2, 2, dtype=q.dtype)
    akk = torch.zeros_like(aqk)
    w = torch.zeros(token_count, 2, 3, dtype=q.dtype)
    u = torch.zeros(token_count, 2, 2, dtype=q.dtype)
    v_new = torch.zeros_like(u)
    qg = torch.zeros_like(w)
    kg = torch.zeros_like(w)
    cumulative = torch.zeros_like(w)
    h = torch.zeros(chunk_count, 2, 2, 3, dtype=q.dtype)
    outputs = torch.zeros_like(v)
    final = torch.zeros_like(h0)
    for head in range(2):
        state = h0[0, head].T
        for chunk, start in enumerate(range(0, token_count, 2)):
            end = min(start + 2, token_count)
            length = end - start
            qc = qn[start:end, 0]
            kc = kn[start:end, 0]
            vc = v[start:end, head]
            bc = b[start:end, head]
            G = gate[start:end, head].cumsum(dim=0)
            ratio = (G[:, None, :] - G[None, :, :]).exp()
            qk = (ratio * (qc[:, None, :] * kc[None])).sum(-1)
            kk = (ratio * (kc[:, None, :] * kc[None])).sum(-1)
            T = torch.tril(bc[:, None] * kk, diagonal=-1)
            inv = torch.linalg.inv(torch.eye(length, dtype=q.dtype) + T)
            tilted_k = G.exp() * kc
            wc = inv @ (bc[:, None] * tilted_k)
            uc = inv @ (bc[:, None] * vc)
            mc = torch.tril(scale * qk)
            qgc = G.exp() * qc
            kgc = (G[-1:] - G).exp() * kc
            h[chunk, head] = state.T
            corrected = uc - wc @ state
            outputs[start:end, head] = scale * qgc @ state + mc @ corrected
            state = G[-1].exp()[:, None] * state + kgc.T @ corrected
            aqk[chunk, head, :length, :length] = mc
            akk[chunk, head, :length, :length] = inv
            w[start:end, head] = wc
            u[start:end, head] = uc
            v_new[start:end, head] = corrected
            qg[start:end, head] = qgc
            kg[start:end, head] = kgc
            cumulative[start:end, head] = G
        final[0, head] = state.T
    return (
        outputs,
        final,
        h,
        aqk,
        akk,
        w,
        u,
        v_new,
        qg,
        kg,
        cumulative,
        qn,
        q_rstd,
        kn,
        k_rstd,
        b,
    )


@pytest.mark.parametrize("lower_bound,beta_factor", [(-0.7, 1), (None, 2)])
@pytest.mark.parametrize("gradient_source", ["output", "final", "both"])
def test_manual_backward_matches_token_autograd(
    lower_bound, beta_factor, gradient_source
):
    q, k, v, g, beta, h0, a_log, bias = _inputs()
    scale = 0.37
    reference_out, reference_final = _token_reference(
        q,
        k,
        v,
        g,
        beta,
        h0,
        a_log,
        bias,
        scale=scale,
        lower_bound=lower_bound,
        beta_factor=beta_factor,
    )
    seed_out = torch.randn_like(reference_out)
    seed_final = torch.randn_like(reference_final)
    if gradient_source == "output":
        seed_final.zero_()
    elif gradient_source == "final":
        seed_out.zero_()
    loss = (reference_out * seed_out).sum() + (reference_final * seed_final).sum()
    expected = torch.autograd.grad(loss, (q, k, v, h0, g, beta, a_log, bias))
    with torch.no_grad():
        (
            out,
            final,
            h,
            aqk,
            akk,
            w,
            u,
            v_new,
            qg,
            kg,
            cumulative,
            qn,
            q_rstd,
            kn,
            k_rstd,
            b,
        ) = _chunk_cache(
            q,
            k,
            v,
            g,
            beta,
            h0,
            a_log,
            bias,
            scale=scale,
            lower_bound=lower_bound,
            beta_factor=beta_factor,
        )
        torch.testing.assert_close(out, reference_out, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(final, reference_final, atol=1e-12, rtol=1e-12)
        actual = flash_kda_grad(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            do=seed_out,
            dht=seed_final,
            chunk_states=h,
            Aqk=aqk,
            Akk=akk,
            w=w,
            u=u,
            v_new=v_new,
            qg=qg,
            kg=kg,
            g_cumsum=cumulative,
            q_norm=qn,
            q_rstd=q_rstd,
            k_norm=kn,
            k_rstd=k_rstd,
            beta_sigmoid=b,
            A_log=a_log,
            dt_bias=bias,
            cu_seqlens=torch.tensor([0, 5], dtype=torch.int32),
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=beta_factor == 2,
            lower_bound=lower_bound,
            layout="TND",
        )
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "norm,fused_gate,fused_beta",
    [
        (False, False, False),
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_unfused_inputs_return_gradients_of_supplied_values(
    norm, fused_gate, fused_beta
):
    q, k, v, g, beta, h0, a_log, bias = _inputs()
    if not fused_gate:
        g = (-torch.rand_like(g)).detach().requires_grad_()
    if not fused_beta:
        beta = torch.sigmoid(beta.detach()).requires_grad_()
    kwargs = dict(
        scale=0.0,
        lower_bound=0.0 if fused_gate else -0.7,
        beta_factor=1,
        norm=norm,
        fused_gate=fused_gate,
        fused_beta=fused_beta,
    )
    ref_out, ref_final = _token_reference(q, k, v, g, beta, h0, a_log, bias, **kwargs)
    seed_final = torch.randn_like(ref_final)
    seed_out = torch.randn_like(ref_out)
    expected = torch.autograd.grad(
        (ref_out * seed_out).sum() + (ref_final * seed_final).sum(),
        (q, k, v, h0, g, beta),
    )
    with torch.no_grad():
        (
            _,
            _,
            h,
            aqk,
            akk,
            w,
            u,
            v_new,
            qg,
            kg,
            cumulative,
            qn,
            q_rstd,
            kn,
            k_rstd,
            b,
        ) = _chunk_cache(q, k, v, g, beta, h0, a_log, bias, **kwargs)
        actual = flash_kda_grad(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            do=seed_out,
            dht=seed_final,
            chunk_states=h,
            Aqk=aqk,
            Akk=akk,
            w=w,
            u=u,
            v_new=v_new,
            qg=qg,
            kg=kg,
            g_cumsum=cumulative,
            q_norm=qn if norm else None,
            q_rstd=q_rstd if norm else None,
            k_norm=kn if norm else None,
            k_rstd=k_rstd if norm else None,
            beta_sigmoid=b if fused_beta else None,
            A_log=a_log if fused_gate else None,
            dt_bias=bias if fused_gate else None,
            cu_seqlens=torch.tensor([0, 5], dtype=torch.int32),
            scale=kwargs["scale"],
            use_qk_l2norm_in_kernel=norm,
            use_gate_in_kernel=fused_gate,
            use_beta_sigmoid_in_kernel=fused_beta,
            lower_bound=kwargs["lower_bound"],
            layout="TND",
        )
    for got, want in zip(actual[:6], expected):
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)
    if not fused_gate:
        assert actual[6:] == (None, None)


@pytest.mark.parametrize("layout", ["BNSD", "BSND"])
def test_layout_and_missing_final_gradient(layout):
    q, k, v, g, beta, h0, a_log, bias = _inputs()
    q, k, v, g, beta = (x.detach() for x in (q, k, v, g, beta))
    with torch.no_grad():
        (
            _,
            _,
            h,
            aqk,
            akk,
            w,
            u,
            v_new,
            qg,
            kg,
            cumulative,
            qn,
            q_rstd,
            kn,
            k_rstd,
            b,
        ) = _chunk_cache(
            q,
            k,
            v,
            g,
            beta,
            h0,
            a_log,
            bias,
            scale=0.37,
            lower_bound=-0.7,
            beta_factor=1,
        )
        seed = torch.randn_like(v)
        reference = flash_kda_grad(
            q,
            k,
            v,
            g,
            beta,
            seed,
            None,
            h,
            aqk,
            akk,
            w,
            u,
            v_new,
            qg,
            kg,
            cumulative,
            qn,
            q_rstd,
            kn,
            k_rstd,
            b,
            a_log,
            bias,
            torch.tensor([0, 5], dtype=torch.int32),
            0.37,
            layout="TND",
        )

        def physical(x):
            x = x[None]
            return x.transpose(1, 2).contiguous() if layout == "BNSD" else x

        actual = flash_kda_grad(
            physical(q),
            physical(k),
            physical(v),
            physical(g),
            physical(beta),
            physical(seed),
            None,
            h,
            aqk,
            akk,
            w,
            u,
            v_new,
            qg,
            kg,
            cumulative,
            qn,
            q_rstd,
            kn,
            k_rstd,
            b,
            a_log,
            bias,
            None,
            0.37,
            layout=layout,
        )
    for got, want in zip(actual, reference):
        torch.testing.assert_close(
            got,
            physical(want)
            if want is not None and want.ndim in (2, 3) and want.shape[0] == 5
            else want,
            atol=1e-10,
            rtol=1e-10,
        )
    assert actual[3].dtype == torch.float64


def test_packed_ragged_sequences_reset_chunk_and_state():
    q, k, v, g, beta, original_h0, a_log, bias = _inputs()
    h0 = torch.cat(
        (original_h0.detach(), original_h0.detach().flip(-1))
    ).requires_grad_()
    config = dict(scale=0.37, lower_bound=-0.7, beta_factor=1)
    reference = [
        _token_reference(
            q[start:end],
            k[start:end],
            v[start:end],
            g[start:end],
            beta[start:end],
            h0[seq : seq + 1],
            a_log,
            bias,
            **config,
        )
        for seq, (start, end) in enumerate(((0, 2), (2, 5)))
    ]
    ref_out = torch.cat([entry[0] for entry in reference])
    ref_final = torch.cat([entry[1] for entry in reference])
    seed_out = torch.randn_like(ref_out)
    seed_final = torch.randn_like(ref_final)
    expected = torch.autograd.grad(
        (ref_out * seed_out).sum() + (ref_final * seed_final).sum(),
        (q, k, v, h0, g, beta, a_log, bias),
    )
    with torch.no_grad():
        caches = [
            _chunk_cache(
                q[start:end],
                k[start:end],
                v[start:end],
                g[start:end],
                beta[start:end],
                h0[seq : seq + 1],
                a_log,
                bias,
                **config,
            )
            for seq, (start, end) in enumerate(((0, 2), (2, 5)))
        ]
        fields = [torch.cat([cache[index] for cache in caches]) for index in range(16)]
        (
            _,
            _,
            h,
            aqk,
            akk,
            w,
            u,
            v_new,
            qg,
            kg,
            cumulative,
            qn,
            q_rstd,
            kn,
            k_rstd,
            b,
        ) = fields
        actual = flash_kda_grad(
            q,
            k,
            v,
            g,
            beta,
            seed_out,
            seed_final,
            h,
            aqk,
            akk,
            w,
            u,
            v_new,
            qg,
            kg,
            cumulative,
            qn,
            q_rstd,
            kn,
            k_rstd,
            b,
            a_log,
            bias,
            torch.tensor([0, 2, 5], dtype=torch.int32),
            config["scale"],
            layout="TND",
            lower_bound=config["lower_bound"],
        )
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)


def test_missing_final_gradient_keeps_fp32_initial_state_gradient():
    q, k, v, g, beta, h0, a_log, bias = _inputs()
    with torch.no_grad():
        (
            _,
            _,
            h,
            aqk,
            akk,
            w,
            u,
            v_new,
            qg,
            kg,
            cumulative,
            qn,
            q_rstd,
            kn,
            k_rstd,
            b,
        ) = _chunk_cache(
            q,
            k,
            v,
            g,
            beta,
            h0,
            a_log,
            bias,
            scale=0.37,
            lower_bound=-0.7,
            beta_factor=1,
        )
        actual = flash_kda_grad(
            q.bfloat16(),
            k.bfloat16(),
            v.bfloat16(),
            g.bfloat16(),
            beta.bfloat16(),
            torch.randn_like(v).bfloat16(),
            None,
            h.float(),
            aqk.bfloat16(),
            akk.bfloat16(),
            w.bfloat16(),
            u.bfloat16(),
            v_new.bfloat16(),
            qg.bfloat16(),
            kg.bfloat16(),
            cumulative.float(),
            qn.bfloat16(),
            q_rstd.float(),
            kn.bfloat16(),
            k_rstd.float(),
            b.float(),
            a_log.float(),
            bias.float(),
            torch.tensor([0, 5], dtype=torch.int32),
            0.37,
            layout="TND",
            lower_bound=-0.7,
        )
    assert actual[3].dtype == torch.float32


def test_rejects_bf16_chunk_state_cache():
    q, k, v, g, beta, h0, a_log, bias = _inputs()
    with torch.no_grad():
        (
            _,
            _,
            h,
            aqk,
            akk,
            w,
            u,
            v_new,
            qg,
            kg,
            cumulative,
            qn,
            q_rstd,
            kn,
            k_rstd,
            b,
        ) = _chunk_cache(
            q,
            k,
            v,
            g,
            beta,
            h0,
            a_log,
            bias,
            scale=0.37,
            lower_bound=-0.7,
            beta_factor=1,
        )
        with pytest.raises(ValueError, match="chunk_states must be FP32"):
            flash_kda_grad(
                q.float(),
                k.float(),
                v.float(),
                g.float(),
                beta.float(),
                torch.randn_like(v).float(),
                None,
                h.bfloat16(),
                aqk.float(),
                akk.float(),
                w.float(),
                u.float(),
                v_new.float(),
                qg.float(),
                kg.float(),
                cumulative.float(),
                qn.float(),
                q_rstd.float(),
                kn.float(),
                k_rstd.float(),
                b.float(),
                a_log.float(),
                bias.float(),
                torch.tensor([0, 5], dtype=torch.int32),
                0.37,
                layout="TND",
                lower_bound=-0.7,
            )


def test_strong_softplus_decay_stays_finite_and_matches_autograd():
    q, k, v, _, beta, h0, a_log, bias = _inputs()
    g = torch.full((5, 2, 3), 500.0, dtype=torch.float64, requires_grad=True)
    config = dict(scale=0.37, lower_bound=None, beta_factor=1)
    ref_out, ref_final = _token_reference(q, k, v, g, beta, h0, a_log, bias, **config)
    seed_out = torch.randn_like(ref_out)
    seed_final = torch.randn_like(ref_final)
    expected = torch.autograd.grad(
        (ref_out * seed_out).sum() + (ref_final * seed_final).sum(),
        (q, k, v, h0, g, beta, a_log, bias),
    )
    with torch.no_grad():
        (
            _,
            _,
            h,
            aqk,
            akk,
            w,
            u,
            v_new,
            qg,
            kg,
            cumulative,
            qn,
            q_rstd,
            kn,
            k_rstd,
            b,
        ) = _chunk_cache(
            q,
            k,
            v,
            g,
            beta,
            h0,
            a_log,
            bias,
            **config,
        )
        actual = flash_kda_grad(
            q,
            k,
            v,
            g,
            beta,
            seed_out,
            seed_final,
            h,
            aqk,
            akk,
            w,
            u,
            v_new,
            qg,
            kg,
            cumulative,
            qn,
            q_rstd,
            kn,
            k_rstd,
            b,
            a_log,
            bias,
            torch.tensor([0, 5], dtype=torch.int32),
            config["scale"],
            layout="TND",
            lower_bound=None,
        )
    for got, want in zip(actual, expected):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)


def _make_inputs(dtype=torch.float64):
    torch.manual_seed(13)
    q = torch.randn(5, 1, 3, dtype=dtype, requires_grad=True)
    k = torch.randn(5, 1, 3, dtype=dtype, requires_grad=True)
    v = torch.randn(5, 2, 2, dtype=dtype, requires_grad=True)
    g = torch.randn(5, 2, 3, dtype=dtype, requires_grad=True)
    beta = torch.randn(5, 2, dtype=dtype, requires_grad=True)
    state = torch.randn(2, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    a_log = torch.randn(2, dtype=dtype, requires_grad=True)
    bias = torch.randn(2, 3, dtype=dtype, requires_grad=True)
    return q, k, v, g, beta, state, a_log, bias


def _token_autograd(
    q,
    k,
    v,
    g,
    beta,
    state,
    a_log,
    bias,
    *,
    bounds,
    lower_bound,
    allow_neg_eigval,
    scale,
):
    qn = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    kn = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    if lower_bound is None:
        gate = -a_log.exp()[None, :, None] * torch.nn.functional.softplus(
            g + bias[None]
        )
    else:
        gate = lower_bound * torch.sigmoid(
            a_log.exp()[None, :, None] * (g + bias[None])
        )
    b = (2 if allow_neg_eigval else 1) * torch.sigmoid(beta)
    outputs = []
    finals = []
    for seq, (start, end) in enumerate(bounds):
        per_head = []
        for head in range(v.shape[1]):
            s = state[seq, head].T
            rows = []
            for token in range(start, end):
                key = kn[token, 0]
                s = gate[token, head].exp()[:, None] * s
                delta = b[token, head] * (v[token, head] - key @ s)
                s = s + key[:, None] * delta[None]
                rows.append(scale * qn[token, 0] @ s)
            per_head.append(torch.stack(rows))
            finals.append(s.T)
        outputs.append(torch.stack(per_head, dim=1))
    return torch.cat(outputs), torch.stack(finals).reshape(
        len(bounds), v.shape[1], v.shape[-1], q.shape[-1]
    )


@pytest.mark.parametrize("lower_bound,allow_neg_eigval", [(-0.7, False), (None, True)])
def test_forward_cache_drives_explicit_backward(lower_bound, allow_neg_eigval):
    q, k, v, g, beta, state, a_log, bias = _make_inputs()
    bounds = [(0, 2), (2, 5)]
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    scale = 0.37
    out_ref, final_ref = _token_autograd(
        q,
        k,
        v,
        g,
        beta,
        state,
        a_log,
        bias,
        bounds=bounds,
        lower_bound=lower_bound,
        allow_neg_eigval=allow_neg_eigval,
        scale=scale,
    )
    seed_out = torch.randn_like(out_ref)
    seed_final = torch.randn_like(final_ref)
    expected = torch.autograd.grad(
        (out_ref * seed_out).sum() + (final_ref * seed_final).sum(),
        (q, k, v, state, g, beta, a_log, bias),
    )
    with torch.no_grad():
        saved = flash_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            state,
            a_log,
            bias,
            cu_seqlens=cu,
            scale=scale,
            lower_bound=lower_bound,
            allow_neg_eigval=allow_neg_eigval,
            layout="TND",
            chunk_size=2,
        )
        torch.testing.assert_close(saved.out, out_ref, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(saved.final_state, final_ref, atol=1e-11, rtol=1e-11)
        assert saved.chunk_states.shape == (3, 2, 2, 3)
        assert saved.chunk_states.dtype == torch.float64
        assert saved.Aqk.shape == saved.Akk.shape == (3, 2, 2, 2)
        assert saved.qg.shape == saved.kg.shape == (5, 2, 3)
        actual = flash_kda_grad(
            q,
            k,
            v,
            g,
            beta,
            seed_out,
            seed_final,
            saved.chunk_states,
            saved.Aqk,
            saved.Akk,
            saved.w,
            saved.u,
            saved.v_new,
            saved.qg,
            saved.kg,
            saved.g_cumsum,
            saved.q_norm,
            saved.q_rstd,
            saved.k_norm,
            saved.k_rstd,
            saved.beta_sigmoid,
            a_log,
            bias,
            cu,
            scale,
            allow_neg_eigval=allow_neg_eigval,
            lower_bound=lower_bound,
            layout="TND",
        )
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)


def test_bf16_cache_keeps_state_and_gate_in_fp32():
    q, k, v, g, beta, state, a_log, bias = _make_inputs()
    with torch.no_grad():
        saved = flash_kda_fwd(
            q.bfloat16(),
            k.bfloat16(),
            v.bfloat16(),
            g.bfloat16(),
            beta.bfloat16(),
            state.float(),
            a_log.float(),
            bias.float(),
            cu_seqlens=torch.tensor([0, 2, 5], dtype=torch.int32),
            scale=0.37,
            lower_bound=-0.7,
            layout="TND",
            chunk_size=2,
        )
    assert saved.out.dtype == saved.Aqk.dtype == saved.qg.dtype == torch.bfloat16
    assert (
        saved.final_state.dtype
        == saved.chunk_states.dtype
        == saved.g_cumsum.dtype
        == torch.float32
    )


@pytest.mark.parametrize("layout", ["TND", "BNSD", "BSND"])
def test_train_operator_autograd_matches_token_recurrence(layout):
    q, k, v, g, beta, state, a_log, bias = _make_inputs()
    state = state[:1].detach().requires_grad_()
    scale = 0.37
    reference_out, reference_final = _token_autograd(
        q,
        k,
        v,
        g,
        beta,
        state,
        a_log,
        bias,
        bounds=[(0, 5)],
        lower_bound=-0.7,
        allow_neg_eigval=False,
        scale=scale,
    )
    seed_out = torch.randn_like(reference_out)
    seed_final = torch.randn_like(reference_final)
    expected = torch.autograd.grad(
        (reference_out * seed_out).sum() + (reference_final * seed_final).sum(),
        (q, k, v, state, g, beta, a_log, bias),
    )

    def physical(tensor):
        if layout == "TND":
            return tensor
        batched = tensor[None]
        return batched.transpose(1, 2).contiguous() if layout == "BNSD" else batched

    out, final = flash_kda_train(
        physical(q),
        physical(k),
        physical(v),
        physical(g),
        physical(beta),
        state,
        a_log,
        bias,
        cu_seqlens=torch.tensor([0, 5], dtype=torch.int32) if layout == "TND" else None,
        scale=scale,
        lower_bound=-0.7,
        layout=layout,
        chunk_size=2,
    )
    out_logical = (
        out
        if layout == "TND"
        else (out.transpose(1, 2)[0] if layout == "BNSD" else out[0])
    )
    torch.testing.assert_close(out_logical, reference_out, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(final, reference_final, atol=1e-11, rtol=1e-11)
    actual = torch.autograd.grad(
        (out * physical(seed_out)).sum() + (final * seed_final).sum(),
        (q, k, v, state, g, beta, a_log, bias),
    )
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)


def test_full_chunk_and_tail_match_fp32_token_recurrence():
    torch.manual_seed(21)
    token_count, key_dim, value_dim = 65, 128, 128
    q = torch.randn(token_count, 1, key_dim)
    k = torch.randn_like(q)
    v = torch.randn(token_count, 2, value_dim)
    g = torch.randn(token_count, 2, key_dim) * 0.02
    beta = torch.randn(token_count, 2)
    state = torch.randn(1, 2, value_dim, key_dim) * 0.01
    a_log = torch.tensor([-0.2, 0.3])
    bias = torch.full((2, key_dim), -6.0)
    scale = key_dim**-0.5
    ref_out, ref_final = _token_autograd(
        q,
        k,
        v,
        g,
        beta,
        state,
        a_log,
        bias,
        bounds=[(0, token_count)],
        lower_bound=-5.0,
        allow_neg_eigval=False,
        scale=scale,
    )
    saved = flash_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        state,
        a_log,
        bias,
        cu_seqlens=torch.tensor([0, token_count], dtype=torch.int32),
        scale=scale,
        lower_bound=-5.0,
        layout="TND",
    )
    assert saved.Aqk.shape == (2, 2, 64, 64)
    assert saved.chunk_states.dtype == torch.float32
    torch.testing.assert_close(saved.out, ref_out, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(saved.final_state, ref_final, atol=2e-5, rtol=2e-4)


def test_train_operator_all_preprocessing_disabled():
    q, k, v, _, _, state, _, _ = _make_inputs()
    state = state[:1].detach().requires_grad_()
    g = (-torch.rand(5, 2, 3, dtype=q.dtype)).requires_grad_()
    beta = torch.sigmoid(torch.randn(5, 2, dtype=q.dtype)).requires_grad_()
    scale = 0.37
    reference_rows = []
    reference_final = []
    for head in range(2):
        s = state[0, head].T
        rows = []
        for token in range(5):
            key = k[token, 0]
            s = g[token, head].exp()[:, None] * s
            delta = beta[token, head] * (v[token, head] - key @ s)
            s = s + key[:, None] * delta[None]
            rows.append(scale * q[token, 0] @ s)
        reference_rows.append(torch.stack(rows))
        reference_final.append(s.T)
    ref_out = torch.stack(reference_rows, dim=1)
    ref_final = torch.stack(reference_final)[None]
    seed_out = torch.randn_like(ref_out)
    expected = torch.autograd.grad(
        (ref_out * seed_out).sum(), (q, k, v, state, g, beta)
    )
    out, final = flash_kda_train(
        q,
        k,
        v,
        g,
        beta,
        state,
        None,
        None,
        cu_seqlens=torch.tensor([0, 5], dtype=torch.int32),
        scale=scale,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        layout="TND",
        chunk_size=2,
    )
    torch.testing.assert_close(out, ref_out, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(final, ref_final, atol=1e-11, rtol=1e-11)
    actual = torch.autograd.grad((out * seed_out).sum(), (q, k, v, state, g, beta))
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)


def _metadata(q, v, layout, offsets=None):
    if offsets is not None:
        offsets = offsets.to(device=q.device, dtype=torch.int32)
    return flash_kda_grad_metadata(**_metadata_arguments(q, v, layout, offsets))


def test_forward_tuple_maps_to_backward_cache_abi(monkeypatch):
    adapter = getattr(integration, "flash_kda_grad_from_forward", None)
    assert adapter is not None, "backward needs the training forward's cache order"
    signature = inspect.signature(integration.flash_kda_grad_npu)
    inputs = tuple(object() for _ in range(7))
    saved = tuple(object() for _ in range(16))
    params = tuple(object() for _ in range(3))
    initial_state = object()
    calls = []

    def capture(*args, **kwargs):
        calls.append(signature.bind(*args, **kwargs).arguments)
        return "gradients"

    monkeypatch.setattr(integration, "flash_kda_grad_npu", capture)
    result = adapter(
        *inputs,
        saved,
        *params,
        0.125,
        layout="TND",
        use_gate_in_kernel=False,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=False,
    )
    assert result == "gradients"
    assert len(calls) == 1
    options = calls[0]
    expected = dict(zip(("q", "k", "v", "g", "beta", "do", "dht"), inputs))
    expected.update(
        Aqk=saved[7],
        Akk=saved[8],
        metadata=params[2],
        initial_state=initial_state,
        chunk_states=saved[15],
        w=saved[9],
        u=saved[10],
        v_new=saved[13],
        qg=saved[11],
        kg=saved[12],
        g_cumsum=saved[14],
        q_rtsd=saved[4],
        k_rstd=saved[5],
        beta_sigmoid=saved[6],
        A_log=params[0],
        dt_bias=params[1],
    )
    for name, value in expected.items():
        assert options[name] is value, name
    assert options["scale"] == 0.125
    assert options["layout"] == "TND"
    assert options["use_gate_in_kernel"] is False


def test_forward_adapter_uses_separately_supplied_fp32_chunk_states(monkeypatch):
    signature = inspect.signature(integration.flash_kda_grad_npu)
    saved = [object() for _ in range(16)]
    saved[15] = torch.empty((1, 1, 128, 128), dtype=torch.bfloat16)
    chunk_states_fp32 = torch.empty((1, 1, 128, 128), dtype=torch.float32)
    calls = []

    def capture(*args, **kwargs):
        calls.append(signature.bind(*args, **kwargs).arguments)
        return "gradients"

    monkeypatch.setattr(integration, "flash_kda_grad_npu", capture)
    inputs = tuple(object() for _ in range(7))
    params = tuple(object() for _ in range(3))
    result = integration.flash_kda_grad_from_forward(
        *inputs,
        tuple(saved),
        *params,
        0.125,
        disable_recompute=True,
        chunk_states_fp32=chunk_states_fp32,
        use_qk_l2norm_in_kernel=False,
    )
    assert result == "gradients"
    assert calls[0]["chunk_states"] is chunk_states_fp32


@pytest.mark.whitebox
def test_forward_adapter_rejects_incomplete_cache_tuple():
    with pytest.raises(ValueError, match="16 tensors/cache slots"):
        integration.flash_kda_grad_from_forward(
            *(object() for _ in range(7)),
            tuple(range(15)),
            None,
            None,
            object(),
            0.125,
            disable_recompute=True,
        )


@pytest.mark.parametrize("layout", ("BNSD", "BSND", "TND"))
def test_forward_adapter_preserves_current_forward_caches(monkeypatch, layout):
    args = _make_host_abi_inputs(torch.float32, layout=layout)
    saved = (
        None,
        None,
        args["q"] if layout != "BSND" else args["q"].transpose(1, 2),
        args["k"] if layout != "BSND" else args["k"].transpose(1, 2),
        args["q_rtsd"],
        args["k_rstd"],
        args["beta_sigmoid"],
        args["Aqk"],
        args["Akk"],
        args["w"],
        args["u"],
        args["qg"],
        args["kg"],
        args["v_new"],
        args["g_cumsum"],
        args["chunk_states"],
    )
    signature = inspect.signature(integration.flash_kda_grad_npu)
    monkeypatch.setattr(
        integration,
        "flash_kda_grad_npu",
        lambda *a, **kw: signature.bind(*a, **kw).arguments,
    )
    result = integration.flash_kda_grad_from_forward(
        *(args[name] for name in ("q", "k", "v", "g", "beta", "do", "dht")),
        saved,
        args["A_log"],
        args["dt_bias"],
        args["metadata"],
        layout=layout,
        disable_recompute=True,
    )
    for name in (
        "chunk_states",
        "Aqk",
        "Akk",
        "w",
        "u",
        "v_new",
        "qg",
        "kg",
        "g_cumsum",
        "q_rtsd",
        "k_rstd",
        "beta_sigmoid",
    ):
        assert result[name] is args[name]


@pytest.mark.npu
@pytest.mark.parametrize(
    "seed,lengths,qk_heads",
    [
        (395, [17, 64, 1], 1),
        (397, [0, 64, 0, 64, 0], 2),
        # A single-chunk sequence still reserves the per-core state slots.
        (399, [64], 2),
    ],
)
def test_strong_packed_gate_parameter_gradient(seed, lengths, qk_heads):
    """Causal row/column rounding must not bias the reverse gate scan."""
    pytest.importorskip("torch_npu")
    torch.manual_seed(seed)
    tokens, heads, dim = sum(lengths), 2, 128
    q = torch.randn(tokens, qk_heads, dim).bfloat16().requires_grad_()
    k = torch.randn_like(q).requires_grad_()
    v = (torch.randn(tokens, heads, dim) * 0.2).bfloat16().requires_grad_()
    g = (torch.randn(tokens, heads, dim) * 0.1).bfloat16().requires_grad_()
    beta = torch.randn(tokens, heads).bfloat16().requires_grad_()
    h0 = (torch.randn(len(lengths), heads, dim, dim) * 0.01).requires_grad_()
    a_log = torch.full((heads,), -1.0).requires_grad_()
    bias = torch.full((heads, dim), -4.0).requires_grad_()
    inputs = (q, k, v, g, beta, h0, a_log, bias)
    scale = dim**-0.5
    output, final = _token_golden(
        *inputs,
        layout="TND",
        sequence_lengths=lengths,
        scale=scale,
        lower_bound=-5.0,
        norm=True,
        fused_gate=True,
        fused_beta=True,
        beta_factor=2.0,
    )
    do = torch.randn_like(output).bfloat16()
    dht = torch.randn_like(final)
    expected = torch.autograd.grad(
        (output * do.float()).sum() + (final * dht).sum(), inputs
    )
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    # Use the existing chunk reference so this backward regression does not
    # depend on the separate DSL forward's packed-cache adapter.
    with torch.no_grad():
        cache = flash_kda_fwd(
            *(tensor.detach() for tensor in inputs),
            cu_seqlens=offsets,
            scale=scale,
            lower_bound=-5.0,
            allow_neg_eigval=True,
            layout="TND",
        )
    saved = tuple(
        getattr(cache, name).npu()
        for name in (
            "out",
            "final_state",
            "q_norm",
            "k_norm",
            "q_rstd",
            "k_rstd",
            "beta_sigmoid",
            "Aqk",
            "Akk",
            "w",
            "u",
            "qg",
            "kg",
            "v_new",
            "g_cumsum",
            "chunk_states",
        )
    )
    saved = list(saved)
    for index in (7, 8, 15):
        saved[index] = _reserve_tnd_forward_chunks(saved[index], lengths)
    saved = tuple(saved)
    qn, kn, vn, gn, bn = (tensor.detach().npu() for tensor in inputs[:5])
    device_offsets = offsets.npu()
    metadata = _metadata(qn, vn, "TND", device_offsets)
    actual = integration.flash_kda_grad_from_forward(
        qn,
        kn,
        vn,
        gn,
        bn,
        do.npu(),
        dht.npu(),
        saved,
        a_log.detach().npu(),
        bias.detach().npu(),
        metadata,
        scale,
        disable_recompute=True,
        cu_seqlens=device_offsets,
        lower_bound=-5.0,
        allow_neg_eigval=True,
        layout="TND",
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    for name, got, want in zip(
        ("dq", "dk", "dv", "dh0", "dg", "dbeta", "dA", "dBias"),
        actual,
        (
            expected[0],
            expected[1],
            expected[2],
            expected[5],
            expected[3],
            expected[4],
            expected[6],
            expected[7],
        ),
    ):
        _assert_precision(name, got, want)


@pytest.mark.parametrize(
    "offsets",
    [
        torch.tensor([0.0, 2.0]),
        torch.tensor([[0, 2]], dtype=torch.int32),
        torch.tensor([0, 3, 2], dtype=torch.int32),
    ],
)
def test_tnd_metadata_rejects_invalid_offsets(offsets):
    q = torch.empty((2, 1, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        _sequence_bounds(q, "TND", offsets)


@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,lengths,qk_heads,value_heads,norm,fused_gate,fused_beta,beta_factor,lower_bound,seed_kind,has_bias",
    [
        ("BNSD", [1], 1, 1, True, True, True, 1.0, -0.7, "both", True),
        ("BSND", [65], 1, 2, True, True, True, 1.0, -0.7, "both", True),
        ("TND", [1, 65], 1, 2, True, True, True, 2.0, None, "both", True),
        pytest.param(
            "BNSD",
            [65],
            32,
            32,
            True,
            True,
            True,
            1.0,
            -0.7,
            "both",
            True,
            id="one-track-full-and-tail",
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "BSND",
            [128],
            32,
            32,
            True,
            True,
            True,
            1.0,
            -0.06,
            "both",
            True,
            id="one-track-bsnd",
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [0, 1, 63, 65, 130],
            8,
            8,
            True,
            True,
            True,
            1.0,
            -0.7,
            "both",
            True,
            id="one-track-packed-boundaries",
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [0, 0, 0, 65],
            8,
            8,
            True,
            True,
            True,
            1.0,
            -0.7,
            "both",
            True,
            id="one-track-empty-owner-slots",
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [65],
            32,
            32,
            True,
            True,
            True,
            1.0,
            -0.7,
            "output",
            True,
            id="one-track-zero-dht",
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [65],
            32,
            32,
            True,
            True,
            True,
            1.0,
            -0.7,
            "final",
            True,
            id="one-track-zero-do",
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [2],
            32,
            32,
            True,
            True,
            True,
            1.0,
            None,
            "both",
            True,
            id="one-track-softplus",
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "BNSD",
            [65],
            32,
            32,
            True,
            False,
            False,
            1.0,
            -0.7,
            "both",
            False,
            id="one-track-unfused",
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "BNSD",
            [128],
            32,
            32,
            True,
            True,
            False,
            1.0,
            -0.06,
            "both",
            True,
            id="one-track-full-raw-beta",
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [128],
            32,
            32,
            True,
            True,
            True,
            1.0,
            -0.06,
            "output",
            True,
            id="one-track-full-zero-dht",
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [64, 128, 64, 128],
            32,
            32,
            True,
            True,
            True,
            1.0,
            -0.06,
            "both",
            True,
            id="one-track-full-packed",
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "BNSD",
            [128] * 4,
            8,
            8,
            True,
            True,
            True,
            1.0,
            -0.06,
            "both",
            True,
            id="one-track-full-batch",
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [128],
            32,
            32,
            True,
            True,
            True,
            2.0,
            -0.06,
            "both",
            False,
            id="one-track-full-no-bias-negative-eigval",
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [3],
            1,
            2,
            False,
            False,
            False,
            1.0,
            -0.7,
            "both",
            False,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [2],
            1,
            1,
            False,
            True,
            True,
            1.0,
            -0.7,
            "both",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [2],
            1,
            1,
            True,
            False,
            True,
            1.0,
            -0.7,
            "both",
            False,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [2],
            1,
            1,
            True,
            True,
            False,
            1.0,
            -0.7,
            "both",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [0, 2],
            1,
            1,
            True,
            True,
            True,
            1.0,
            -0.7,
            "both",
            False,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [2],
            2,
            4,
            True,
            True,
            True,
            1.0,
            -0.7,
            "both",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "BNSD",
            [2],
            1,
            1,
            True,
            True,
            True,
            1.0,
            -0.7,
            "output",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "BNSD",
            [2],
            1,
            1,
            True,
            True,
            True,
            1.0,
            -0.7,
            "final",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [63],
            1,
            1,
            True,
            True,
            True,
            1.0,
            -0.7,
            "both",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "BNSD",
            [64],
            1,
            1,
            True,
            True,
            True,
            1.0,
            -0.7,
            "both",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "BSND",
            [129],
            1,
            2,
            True,
            True,
            True,
            1.0,
            -0.7,
            "both",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [0, 64, 0, 65],
            1,
            1,
            True,
            True,
            True,
            1.0,
            -0.7,
            "both",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [37, 37],
            1,
            2,
            True,
            True,
            True,
            1.0,
            -0.7,
            "both",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [128],
            1,
            1,
            True,
            True,
            True,
            1.0,
            -0.7,
            "both",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [2],
            1,
            1,
            False,
            False,
            True,
            1.0,
            -0.7,
            "both",
            False,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [2],
            1,
            1,
            False,
            True,
            False,
            1.0,
            -0.7,
            "both",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [2],
            1,
            1,
            True,
            False,
            False,
            1.0,
            -0.7,
            "both",
            False,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [2],
            1,
            1,
            True,
            True,
            True,
            2.0,
            -0.7,
            "both",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [65],
            1,
            4,
            True,
            True,
            True,
            1.0,
            -0.7,
            "both",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "BSND",
            [2],
            1,
            1,
            True,
            True,
            True,
            1.0,
            -0.7,
            "output",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "BSND",
            [2],
            1,
            1,
            True,
            True,
            True,
            1.0,
            -0.7,
            "final",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [2],
            1,
            1,
            True,
            True,
            True,
            1.0,
            -0.7,
            "output",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [2],
            1,
            1,
            True,
            True,
            True,
            1.0,
            -0.7,
            "final",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "BNSD",
            [2],
            1,
            1,
            True,
            True,
            True,
            1.0,
            -5.0,
            "both",
            True,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [2],
            1,
            1,
            False,
            False,
            True,
            2.0,
            -0.7,
            "both",
            False,
            marks=pytest.mark.whitebox,
        ),
        pytest.param(
            "TND",
            [2],
            1,
            1,
            False,
            False,
            False,
            1.0,
            None,
            "both",
            False,
            marks=pytest.mark.whitebox,
        ),
    ],
    ids=[
        "bnsd_single",
        "bsnd_tail_gqa",
        "tnd_ragged_softplus",
        "one-track-full-and-tail",
        "one-track-bsnd",
        "one-track-packed-boundaries",
        "one-track-empty-owner-slots",
        "one-track-zero-dht",
        "one-track-zero-do",
        "one-track-softplus",
        "one-track-unfused",
        "one-track-full-raw-beta",
        "one-track-full-zero-dht",
        "one-track-full-packed",
        "one-track-full-batch",
        "one-track-full-no-bias-negative-eigval",
        "unfused",
        "raw_qk",
        "raw_gate",
        "raw_beta",
        "empty_sequence",
        "gqa",
        "output_only",
        "final_only",
        "tail_63",
        "full_chunk_64",
        "three_chunks_129",
        "interspersed_empty",
        "adjacent_sequences_37",
        "two_full_chunks",
        "raw_qk_gate",
        "raw_qk_beta",
        "raw_gate_beta",
        "bounded_negative_eigval",
        "gqa_ratio_four",
        "bsnd_output_only",
        "bsnd_final_only",
        "tnd_output_only",
        "tnd_final_only",
        "default_lower_bound",
        "external_gate_negative_eigval",
        "external_gate_ignores_softplus_bound",
    ],
)
def test_full_train_backward_npu(
    layout,
    lengths,
    qk_heads,
    value_heads,
    norm,
    fused_gate,
    fused_beta,
    beta_factor,
    lower_bound,
    seed_kind,
    has_bias,
    seed=391,
    beta_zero=False,
    qk_scale=1.0,
):
    pytest.importorskip("torch_npu")
    torch.manual_seed(seed)
    tokens = sum(lengths)
    q = torch.randn(tokens, qk_heads, 128).bfloat16()
    k = torch.randn_like(q)
    if qk_scale != 1.0:
        q = (q.float() * qk_scale).bfloat16()
        k = (k.float() * qk_scale).bfloat16()
    v = (torch.randn(tokens, value_heads, 128) * 0.2).bfloat16()
    g = (torch.randn(tokens, value_heads, 128) * 0.1).bfloat16()
    if not fused_gate:
        g = -g.abs() * 0.1
    beta = torch.randn(tokens, value_heads).bfloat16()
    if not fused_beta:
        beta = torch.sigmoid(beta.float()).bfloat16()
    if beta_zero:
        # Effective beta is exactly zero only on the unfused path. Keep
        # all existing random draws unchanged when this option is off.
        assert not fused_beta
        beta.zero_()
    h0 = (torch.randn(len(lengths), value_heads, 128, 128) * 0.01).float()
    a_log = (
        torch.full((value_heads,), -1.0, dtype=torch.float32) if fused_gate else None
    )
    bias = (
        torch.full(
            (value_heads, 128),
            4.0 if lower_bound is None else -4.0,
            dtype=torch.float32,
        )
        if fused_gate and has_bias
        else None
    )
    batch, seq_length = len(lengths), lengths[0]
    physical_cpu = tuple(
        _physical(t, layout, batch, seq_length).detach().requires_grad_()
        for t in (q, k, v, g, beta)
    )
    h0_cpu = h0.detach().requires_grad_()
    a_cpu = None if a_log is None else a_log.detach().requires_grad_()
    bias_cpu = None if bias is None else bias.detach().requires_grad_()
    scale = 128**-0.5
    ref_out, ref_final = _token_golden(
        *physical_cpu,
        h0_cpu,
        a_cpu,
        bias_cpu,
        layout=layout,
        sequence_lengths=lengths,
        scale=scale,
        lower_bound=lower_bound,
        norm=norm,
        fused_gate=fused_gate,
        fused_beta=fused_beta,
        beta_factor=beta_factor,
    )
    seed_out = torch.randn_like(ref_out).bfloat16() if seed_kind != "final" else None
    seed_final = torch.randn_like(ref_final) if seed_kind != "output" else None
    cpu_inputs = (*physical_cpu, h0_cpu) + tuple(
        x for x in (a_cpu, bias_cpu) if x is not None
    )
    objective = 0
    if seed_out is not None:
        objective = objective + (ref_out * seed_out.float()).sum()
    if seed_final is not None:
        objective = objective + (ref_final * seed_final).sum()
    expected_raw = torch.autograd.grad(objective, cpu_inputs, allow_unused=True)
    expected = tuple(
        torch.zeros_like(t) if grad is None else grad
        for t, grad in zip(cpu_inputs, expected_raw)
    )

    physical_npu = tuple(t.detach().npu().requires_grad_() for t in physical_cpu)
    h0_npu = h0.detach().npu().requires_grad_()
    a_npu = None if a_cpu is None else a_cpu.detach().npu().requires_grad_()
    bias_npu = None if bias_cpu is None else bias_cpu.detach().npu().requires_grad_()
    offsets = None
    if layout == "TND":
        offsets = torch.tensor(
            [0] + [sum(lengths[: i + 1]) for i in range(batch)], dtype=torch.int32
        )
    out, final = flash_kda_train(
        *physical_npu,
        h0_npu,
        a_npu,
        bias_npu,
        cu_seqlens=offsets,
        scale=scale,
        layout=layout,
        chunk_size=64,
        use_qk_l2norm_in_kernel=norm,
        use_gate_in_kernel=fused_gate,
        use_beta_sigmoid_in_kernel=fused_beta,
        allow_neg_eigval=beta_factor == 2.0,
        lower_bound=lower_bound,
    )
    _assert_precision("forward output", out, ref_out)
    _assert_precision("forward final_state", final, ref_final)
    npu_inputs = (*physical_npu, h0_npu) + tuple(
        x for x in (a_npu, bias_npu) if x is not None
    )
    npu_objective = 0
    if seed_out is not None:
        npu_objective = npu_objective + (out * seed_out.npu()).sum()
    if seed_final is not None:
        npu_objective = npu_objective + (final * seed_final.npu()).sum()
    actual = torch.autograd.grad(npu_objective, npu_inputs)
    # Forward's BF16 saved matrices introduce more error than FP32 token math.
    names = (
        ("dq", "dk", "dv", "dg", "dbeta", "dh0")
        + (("dA",) if a_cpu is not None else ())
        + (("dBias",) if bias_cpu is not None else ())
    )
    for name, got, want in zip(names, actual, expected):
        # With raw BF16 Q/K, this exact case has a 5.17% floor even when the
        # staged PyTorch backward consumes the same quantized forward caches.
        quantized_raw_qk = not norm and fused_gate and fused_beta
        _assert_precision(
            name,
            got,
            want,
            max_relative_l2=0.06 if name == "dq" and quantized_raw_qk else 0.05,
            min_atol=5e-3 if name == "dq" and not norm else 4e-3,
        )
    if not norm:
        staged_cache = flash_kda_fwd(
            *physical_cpu,
            h0_cpu,
            a_cpu,
            bias_cpu,
            cu_seqlens=offsets,
            scale=scale,
            layout=layout,
            chunk_size=64,
            use_qk_l2norm_in_kernel=norm,
            use_gate_in_kernel=fused_gate,
            use_beta_sigmoid_in_kernel=fused_beta,
            allow_neg_eigval=beta_factor == 2.0,
            lower_bound=lower_bound,
        )
        staged_gradients = flash_kda_grad(
            *physical_cpu,
            seed_out,
            seed_final,
            staged_cache.chunk_states,
            staged_cache.Aqk,
            staged_cache.Akk,
            staged_cache.w,
            staged_cache.u,
            staged_cache.v_new,
            staged_cache.qg,
            staged_cache.kg,
            staged_cache.g_cumsum,
            staged_cache.q_norm,
            staged_cache.q_rstd,
            staged_cache.k_norm,
            staged_cache.k_rstd,
            staged_cache.beta_sigmoid,
            a_cpu,
            bias_cpu,
            offsets,
            scale,
            use_qk_l2norm_in_kernel=norm,
            use_gate_in_kernel=fused_gate,
            use_beta_sigmoid_in_kernel=fused_beta,
            allow_neg_eigval=beta_factor == 2.0,
            lower_bound=lower_bound,
            layout=layout,
        )
        _assert_precision(
            "dq versus quantized cache",
            actual[0],
            staged_gradients[0],
            max_relative_l2=0.01,
        )
        staged_beta = staged_gradients[5].float()
        beta_relative = (
            (actual[4].cpu().float() - staged_beta).norm() / staged_beta.norm()
        ).item()
        assert beta_relative < 0.02, (
            f"dbeta versus quantized cache: rel_l2={beta_relative:.6g}"
        )


@pytest.mark.whitebox
@pytest.mark.npu
def test_one_track_full_raw_qk_beta_npu():
    # Keep the new unnormalized recurrent fixture finite without changing any
    # original fixture, reference function, or precision assertion.
    test_full_train_backward_npu(
        "BNSD",
        [128],
        32,
        32,
        False,
        True,
        False,
        1.0,
        -0.06,
        "both",
        True,
        qk_scale=128**-0.5,
    )


@pytest.mark.whitebox
@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,lengths,qk_heads,value_heads,norm,fused_beta,beta_factor,seed_kind,has_bias,seed",
    [
        ("BSND", [1], 1, 1, True, True, 1.0, "both", False, 73),
        ("BNSD", [2], 1, 1, True, True, 1.0, "both", False, 73),
        ("TND", [2, 0, 3], 1, 2, True, True, 2.0, "both", True, 391),
        ("BNSD", [7, 7], 1, 2, True, True, 1.0, "both", True, 73),
        ("BSND", [16], 2, 4, True, True, 1.0, "both", False, 117),
        ("BNSD", [32], 1, 1, True, True, 1.0, "both", False, 73),
        ("BSND", [64], 1, 1, True, True, 1.0, "both", True, 391),
        ("TND", [1], 1, 1, False, False, 1.0, "both", False, 73),
        ("BNSD", [3], 1, 1, True, False, 1.0, "both", True, 391),
        ("BSND", [2], 1, 2, True, True, 1.0, "output", False, 391),
        ("TND", [2], 1, 2, True, True, 1.0, "final", False, 73),
        ("TND", [0, 2, 0], 1, 2, True, True, 1.0, "both", False, 117),
        ("TND", [2, 65], 1, 2, True, True, 1.0, "both", True, 73),
    ],
    ids=[
        "single",
        "different_seed",
        "packed_empty_gqa_negative",
        "padded_batch",
        "multiple_qk_heads",
        "half_chunk",
        "full_chunk",
        "raw_qk_beta_singleton",
        "raw_beta",
        "output_only",
        "final_only",
        "leading_trailing_empty",
        "mixed_short_long",
    ],
)
def test_single_chunk_softplus_gate_sensitivity_npu(
    layout,
    lengths,
    qk_heads,
    value_heads,
    norm,
    fused_beta,
    beta_factor,
    seed_kind,
    has_bias,
    seed,
):
    """Check all gradients against the unchanged token golden and tolerances."""
    test_full_train_backward_npu(
        layout,
        lengths,
        qk_heads,
        value_heads,
        norm,
        True,
        fused_beta,
        beta_factor,
        None,
        seed_kind,
        has_bias,
        seed,
    )


@pytest.mark.whitebox
@pytest.mark.npu
@pytest.mark.parametrize(
    "lengths,seed,seed_kind",
    [
        ([2, 0, 3], 391, "both"),
        ([0, 2, 0], 117, "both"),
        ([2, 65], 73, "both"),
        ([2], 73, "final"),
    ],
)
def test_reserved_chunks_softplus_matches_token_autograd(lengths, seed, seed_kind):
    test_full_train_backward_npu(
        "TND",
        lengths,
        1,
        2,
        True,
        True,
        True,
        2.0 if seed == 391 else 1.0,
        None,
        seed_kind,
        seed_kind == "both" and seed != 117,
        seed,
    )


@pytest.mark.whitebox
@pytest.mark.npu
def test_single_chunk_softplus_zero_raw_beta_npu():
    """Beta can be zero although its derivative is nonzero; never divide by it."""
    test_full_train_backward_npu(
        "BNSD",
        [3],
        1,
        1,
        True,
        True,
        False,
        1.0,
        None,
        "both",
        True,
        seed=391,
        beta_zero=True,
    )


@pytest.mark.whitebox
@pytest.mark.npu
@pytest.mark.parametrize("layout", ["BNSD", "BSND"])
@pytest.mark.parametrize(
    "heads,storage_length,lower_bound", [(1, 4, None), (32, 128, -0.06)]
)
def test_empty_forward_caches_with_padded_storage_npu(
    layout, heads, storage_length, lower_bound
):
    """Zero valid tokens: all padded gradients are zero and dh0 equals dht."""
    pytest.importorskip("torch_npu")
    batch, width = 1, 128
    raw_shape = (
        (batch, heads, storage_length, width)
        if layout == "BNSD"
        else (batch, storage_length, heads, width)
    )

    def raw(value, *, scalar=False):
        shape = raw_shape[:-1] if scalar else raw_shape
        return torch.full(shape, value, dtype=torch.bfloat16).npu()

    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype).npu()

    dht_cpu = (
        torch.arange(1, heads * width * width + 1, dtype=torch.float32).reshape(
            batch, heads, width, width
        )
        / 32768.0
    )
    token_shape = (0, heads, width)
    scalar_shape = (0, heads)
    matrix_shape = (0, heads, 64, 64)
    # Nonzero physical rows are deliberate: none may leak into padded outputs.
    # Compact caches are mathematically empty; no forward execution is needed.
    inputs = {
        "q": raw(0.125),
        "k": raw(0.25),
        "v": raw(-0.25),
        "g": raw(-0.5),
        "beta": raw(0.375, scalar=True),
        "do": raw(0.5),
        "dht": dht_cpu.npu(),
        "chunk_states": empty((0, heads, width, width), torch.float32),
        "Aqk": empty(matrix_shape, torch.bfloat16),
        "Akk": empty(matrix_shape, torch.bfloat16),
        "w": empty(token_shape, torch.bfloat16),
        "u": empty(token_shape, torch.bfloat16),
        "v_new": empty(token_shape, torch.bfloat16),
        "qg": empty(token_shape, torch.bfloat16),
        "kg": empty(token_shape, torch.bfloat16),
        "g_cumsum": empty(token_shape, torch.float32),
        "q_rtsd": empty(scalar_shape, torch.float32),
        "k_rstd": empty(scalar_shape, torch.float32),
        "beta_sigmoid": empty(scalar_shape, torch.float32),
        "A_log": torch.full((heads,), -0.2, dtype=torch.float32).npu(),
        "dt_bias": torch.full((heads, width), -4.0, dtype=torch.float32).npu(),
    }
    offsets = torch.tensor([0, 0], dtype=torch.int32)
    metadata = _metadata(inputs["q"], inputs["v"], layout, offsets)
    gradients = _reference_backward(
        disable_recompute=True,
        **inputs,
        metadata=metadata,
        scale=width**-0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        lower_bound=lower_bound,
        layout=layout,
    )
    names = ("dq", "dk", "dv", "dh0", "dg", "dbeta", "dA", "dBias")
    references = tuple(
        inputs[name] for name in ("q", "k", "v", "dht", "g", "beta", "A_log", "dt_bias")
    )
    assert isinstance(gradients, (tuple, list)) and len(gradients) == len(names)
    for name, got, reference in zip(names, gradients, references):
        assert isinstance(got, torch.Tensor), name
        assert got.device == inputs["q"].device, name
        assert got.shape == reference.shape, name
        assert got.dtype == reference.dtype, name
        expected = (
            dht_cpu if name == "dh0" else torch.zeros_like(reference, device="cpu")
        )
        got_cpu = got.detach().cpu()
        assert bool(torch.isfinite(got_cpu).all()), name
        assert torch.equal(got_cpu, expected), (
            f"{layout} {name}: expected exact empty-sequence identity"
        )


@pytest.mark.whitebox
@pytest.mark.npu
def test_empty_tnd_sequence_transmits_terminal_gradient():
    pytest.importorskip("torch_npu")
    empty = torch.empty((0, 1, 128), dtype=torch.bfloat16)
    beta = torch.empty((0, 1), dtype=torch.bfloat16)
    h0 = torch.randn((1, 1, 128, 128), dtype=torch.float32)
    a_log = torch.zeros((1,), dtype=torch.float32)
    bias = torch.zeros((1, 128), dtype=torch.float32)
    q, k, v, g = (empty.npu().requires_grad_() for _ in range(4))
    beta = beta.npu().requires_grad_()
    state = h0.npu().requires_grad_()
    a_log = a_log.npu().requires_grad_()
    bias = bias.npu().requires_grad_()
    out, final = flash_kda_train(
        q,
        k,
        v,
        g,
        beta,
        state,
        a_log,
        bias,
        cu_seqlens=torch.tensor([0, 0], dtype=torch.int32),
        layout="TND",
    )
    seed = torch.randn_like(final)
    actual = torch.autograd.grad(
        (final * seed).sum(), (q, k, v, g, beta, state, a_log, bias)
    )
    assert out.shape == empty.shape
    torch.testing.assert_close(final.cpu(), h0)
    for gradient in (*actual[:5], *actual[6:]):
        torch.testing.assert_close(gradient.cpu(), torch.zeros_like(gradient.cpu()))
    torch.testing.assert_close(actual[5].cpu(), seed.cpu())


@pytest.mark.whitebox
@pytest.mark.npu
@pytest.mark.parametrize("layout", ["TND", "BNSD", "BSND"])
def test_zero_upstream_gradients_return_zero(layout):
    """Explicit zero seeds must produce zero public gradients in every layout."""
    pytest.importorskip("torch_npu")
    torch.manual_seed(926)
    rows = tuple(torch.randn((2, 1, 128)).bfloat16() for _ in range(4))
    beta = torch.randn((2, 1)).bfloat16()
    inputs = tuple(_physical(row, layout, 1, 2).npu() for row in (*rows, beta))
    state = (torch.randn((1, 1, 128, 128)) * 0.01).npu()
    a_log = torch.tensor([-1.0], dtype=torch.float32).npu()
    bias = torch.full((1, 128), -4.0, dtype=torch.float32).npu()
    offsets = torch.tensor([0, 2], dtype=torch.int32) if layout == "TND" else None
    cache = flash_kda_fwd(
        *inputs, state, a_log, bias, cu_seqlens=offsets, scale=0.125, layout=layout
    )
    metadata = _metadata(inputs[0], inputs[2], layout, offsets)
    gradients = _reference_backward(
        _reference_qk_to_input_layout(cache.q_norm, inputs[0], metadata, layout),
        _reference_qk_to_input_layout(cache.k_norm, inputs[1], metadata, layout),
        inputs[2],
        inputs[3],
        inputs[4],
        torch.zeros_like(inputs[2]),
        torch.zeros_like(state),
        cache.Aqk,
        cache.Akk,
        metadata,
        None,  # initial_state is unused with saved caches
        cache.chunk_states,
        cache.w,
        cache.u,
        cache.v_new,
        cache.qg,
        cache.kg,
        cache.g_cumsum,
        disable_recompute=True,
        q_rtsd=cache.q_rstd,
        k_rstd=cache.k_rstd,
        beta_sigmoid=cache.beta_sigmoid,
        A_log=a_log,
        dt_bias=bias,
        scale=0.125,
        layout=layout,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    for gradient in gradients:
        torch.testing.assert_close(
            gradient.cpu(), torch.zeros_like(gradient.cpu()), atol=0, rtol=0
        )


@pytest.mark.whitebox
@pytest.mark.npu
def test_public_backward_dispatches_one_compute_kernel(monkeypatch):
    pytest.importorskip("torch_npu")
    torch.manual_seed(928)
    q, k, v, g = (torch.randn((2, 1, 128)).bfloat16().npu() for _ in range(4))
    beta = torch.randn((2, 1)).bfloat16().npu()
    state = (torch.randn((1, 1, 128, 128)) * 0.01).npu()
    a_log = torch.tensor([-1.0], dtype=torch.float32).npu()
    bias = torch.full((1, 128), -4.0).npu()
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    cache = flash_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        state,
        a_log,
        bias,
        cu_seqlens=offsets,
        scale=0.125,
        layout="TND",
    )
    metadata = _metadata(q, v, "TND", offsets)
    real_compile = integration.cannbotdsl.compile
    launches = []

    def counted_compile(host_entry, *specs):
        compiled = real_compile(host_entry, *specs)

        class CountedKernel:
            def __call__(self, *args):
                launches.append(args[-1])
                return compiled(*args)

            def close(self):
                compiled.close()

        return CountedKernel()

    monkeypatch.setattr(integration.cannbotdsl, "compile", counted_compile)
    gradients = _reference_backward(
        _reference_qk_to_input_layout(cache.q_norm, q, metadata, "TND"),
        _reference_qk_to_input_layout(cache.k_norm, k, metadata, "TND"),
        v,
        g,
        beta,
        torch.randn_like(v),
        torch.randn_like(state),
        cache.Aqk,
        cache.Akk,
        metadata,
        None,  # initial_state is unused with saved caches
        cache.chunk_states,
        cache.w,
        cache.u,
        cache.v_new,
        cache.qg,
        cache.kg,
        cache.g_cumsum,
        disable_recompute=True,
        q_rtsd=cache.q_rstd,
        k_rstd=cache.k_rstd,
        beta_sigmoid=cache.beta_sigmoid,
        A_log=a_log,
        dt_bias=bias,
        scale=0.125,
        layout="TND",
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    assert launches == [integration._device_block_num(q)]
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.whitebox
@pytest.mark.npu
def test_strong_softplus_decay_across_causal_tile_matches_autograd():
    pytest.importorskip("torch_npu")
    torch.manual_seed(929)
    length, scale = 17, 0.125
    q = torch.randn((length, 1, 128)).bfloat16().requires_grad_()
    k = torch.randn_like(q).detach().requires_grad_()
    v = (torch.randn_like(q) * 0.2).detach().requires_grad_()
    g = torch.zeros_like(q).detach().requires_grad_()
    beta = torch.randn((length, 1)).bfloat16().requires_grad_()
    h0 = (torch.randn((1, 1, 128, 128)) * 0.01).requires_grad_()
    a_log = torch.tensor([math.log(10.0 / math.log(2.0))], requires_grad=True)
    bias = torch.zeros((1, 128), requires_grad=True)
    offsets = torch.tensor([0, length], dtype=torch.int32)
    inputs = (q, k, v, g, beta, h0, a_log, bias)
    reference_output, reference_final = _token_golden(
        q,
        k,
        v,
        g,
        beta,
        h0,
        a_log,
        bias,
        layout="TND",
        sequence_lengths=[length],
        scale=scale,
        lower_bound=None,
        norm=True,
        fused_gate=True,
        fused_beta=True,
        beta_factor=1.0,
    )
    output_gradient = torch.randn_like(reference_output).bfloat16()
    final_gradient = torch.randn_like(reference_final)
    objective = (reference_output * output_gradient.float()).sum() + (
        reference_final * final_gradient
    ).sum()
    expected = torch.autograd.grad(objective, inputs)

    device_inputs = tuple(tensor.detach().npu() for tensor in inputs)
    q_npu, k_npu, v_npu, g_npu, beta_npu, state, a_npu, bias_npu = device_inputs
    cache = flash_kda_fwd(
        q_npu,
        k_npu,
        v_npu,
        g_npu,
        beta_npu,
        state,
        a_npu,
        bias_npu,
        cu_seqlens=offsets,
        scale=scale,
        lower_bound=None,
        layout="TND",
    )
    metadata = _metadata(q_npu, v_npu, "TND", offsets)
    actual = _reference_backward(
        _reference_qk_to_input_layout(cache.q_norm, q_npu, metadata, "TND"),
        _reference_qk_to_input_layout(cache.k_norm, k_npu, metadata, "TND"),
        v_npu,
        g_npu,
        beta_npu,
        output_gradient.npu(),
        final_gradient.npu(),
        cache.Aqk,
        cache.Akk,
        metadata,
        None,  # initial_state is unused with saved caches
        cache.chunk_states,
        cache.w,
        cache.u,
        cache.v_new,
        cache.qg,
        cache.kg,
        cache.g_cumsum,
        disable_recompute=True,
        q_rtsd=cache.q_rstd,
        k_rstd=cache.k_rstd,
        beta_sigmoid=cache.beta_sigmoid,
        A_log=a_npu,
        dt_bias=bias_npu,
        scale=scale,
        lower_bound=None,
        layout="TND",
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    for name, gradient, reference in zip(
        ("dq", "dk", "dv", "dh0", "dg", "dbeta", "dA", "dBias"),
        actual,
        (
            expected[0],
            expected[1],
            expected[2],
            expected[5],
            expected[3],
            expected[4],
            expected[6],
            expected[7],
        ),
    ):
        if name == "dA":
            # The strongly decayed scalar is ~1e-4 after cancellation across
            # 17 × 128 channels. Check its absolute error at a tighter scale.
            torch.testing.assert_close(
                gradient.cpu().float(), reference.float(), atol=2e-5, rtol=5e-2
            )
        else:
            _assert_precision(name, gradient, reference)


@pytest.mark.whitebox
@pytest.mark.npu
@pytest.mark.parametrize(
    "heads,storage_length,lower_bound", [(1, 2, -5.0), (32, 128, -0.06)]
)
def test_padded_bnsd_gradient_rows_are_zero(heads, storage_length, lower_bound):
    """Invalid physical token rows are written by Stage 1, not left uninitialized."""
    pytest.importorskip("torch_npu")
    torch.manual_seed(919)

    def token():
        return torch.randn((1, heads, 128)).bfloat16()

    q, k, v, g = (token() for _ in range(4))
    beta = torch.randn((1, heads)).bfloat16()
    initial_state = torch.randn((1, heads, 128, 128)).float() * 0.01
    A_log = torch.full((heads,), -1.0, dtype=torch.float32).npu()
    dt_bias = torch.full((heads, 128), -4.0, dtype=torch.float32).npu()
    cache = flash_kda_fwd(
        q.npu(),
        k.npu(),
        v.npu(),
        g.npu(),
        beta.npu(),
        initial_state.npu(),
        A_log,
        dt_bias,
        cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        layout="TND",
        scale=128**-0.5,
        lower_bound=lower_bound,
    )

    def padded_rows(row):
        return (
            torch.cat(
                (
                    row,
                    torch.randn((storage_length - 1, *row.shape[1:]), dtype=row.dtype),
                ),
                dim=0,
            )
            .transpose(0, 1)
            .unsqueeze(0)
            .contiguous()
            .npu()
        )

    q_pad, k_pad, v_pad, g_pad = (padded_rows(row) for row in (q, k, v, g))
    beta_pad = (
        torch.cat(
            (beta, torch.randn((storage_length - 1, heads), dtype=beta.dtype)), dim=0
        )
        .transpose(0, 1)
        .unsqueeze(0)
        .contiguous()
        .npu()
    )
    do_pad = padded_rows(token())
    dht = torch.randn((1, heads, 128, 128), dtype=torch.float32).npu()
    metadata = _metadata(q_pad, v_pad, "BNSD", torch.tensor([0, 1], dtype=torch.int32))
    gradients = _reference_backward(
        _reference_qk_to_input_layout(cache.q_norm, q_pad, metadata, "BNSD"),
        _reference_qk_to_input_layout(cache.k_norm, k_pad, metadata, "BNSD"),
        v_pad,
        g_pad,
        beta_pad,
        do_pad,
        dht,
        cache.Aqk,
        cache.Akk,
        metadata,
        None,  # initial_state is unused with saved caches
        cache.chunk_states,
        cache.w,
        cache.u,
        cache.v_new,
        cache.qg,
        cache.kg,
        cache.g_cumsum,
        disable_recompute=True,
        q_rtsd=cache.q_rstd,
        k_rstd=cache.k_rstd,
        beta_sigmoid=cache.beta_sigmoid,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=128**-0.5,
        layout="BNSD",
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    for gradient in (*gradients[:3], gradients[4], gradients[5]):
        torch.testing.assert_close(
            gradient[:, :, 1:].cpu(), torch.zeros_like(gradient[:, :, 1:].cpu())
        )


@pytest.mark.whitebox
@pytest.mark.npu
@pytest.mark.parametrize("layout", ["BNSD", "BSND"])
@pytest.mark.parametrize("length_source", ["metadata", "both"])
def test_padded_ragged_backward_matches_token_autograd(layout, length_source):
    """Compact caches address valid rows across tails and empty sequences."""
    pytest.importorskip("torch_npu")
    torch.manual_seed(927)
    lengths = (1, 65, 0)
    batch, storage, qk_heads, value_heads = 3, 70, 1, 2
    scale = 0.125
    offsets = torch.tensor([0, 1, 66, 66], dtype=torch.int32)

    def physical(rows):
        return rows.transpose(1, 2).contiguous() if layout == "BNSD" else rows

    q = (
        physical(torch.randn(batch, storage, qk_heads, 128).bfloat16())
        .detach()
        .requires_grad_()
    )
    k = (
        physical(torch.randn(batch, storage, qk_heads, 128).bfloat16())
        .detach()
        .requires_grad_()
    )
    v = (
        physical((torch.randn(batch, storage, value_heads, 128) * 0.2).bfloat16())
        .detach()
        .requires_grad_()
    )
    g = (
        physical((torch.randn(batch, storage, value_heads, 128) * 0.1).bfloat16())
        .detach()
        .requires_grad_()
    )
    beta = (
        physical(torch.randn(batch, storage, value_heads).bfloat16())
        .detach()
        .requires_grad_()
    )
    h0 = (torch.randn(batch, value_heads, 128, 128) * 0.01).requires_grad_()
    a_log = torch.full((value_heads,), -1.0, requires_grad=True)
    bias = torch.full((value_heads, 128), -4.0, requires_grad=True)

    def packed(tensor):
        rows = (
            tensor[seq, :, :length].transpose(0, 1)
            if layout == "BNSD"
            else tensor[seq, :length]
            for seq, length in enumerate(lengths)
        )
        return torch.cat(tuple(rows), dim=0).contiguous()

    cpu_inputs = (q, k, v, g, beta)
    packed_inputs = tuple(packed(tensor) for tensor in cpu_inputs)
    packed_output, expected_final = _token_golden(
        *packed_inputs,
        h0,
        a_log,
        bias,
        layout="TND",
        sequence_lengths=lengths,
        scale=scale,
        lower_bound=-0.7,
        norm=True,
        fused_gate=True,
        fused_beta=True,
        beta_factor=1.0,
    )
    cursor = 0
    padded_output = []
    for length in lengths:
        padded_output.append(
            torch.cat(
                (
                    packed_output[cursor : cursor + length],
                    torch.zeros(
                        (storage - length, value_heads, 128), dtype=packed_output.dtype
                    ),
                ),
                dim=0,
            )
        )
        cursor += length
    expected_output = physical(torch.stack(padded_output))
    output_gradient = physical(torch.randn(batch, storage, value_heads, 128).bfloat16())
    final_gradient = torch.randn_like(expected_final)
    objective = (expected_output * output_gradient.float()).sum() + (
        expected_final * final_gradient
    ).sum()
    expected_raw = torch.autograd.grad(
        objective, (*cpu_inputs, h0, a_log, bias), allow_unused=True
    )
    expected = tuple(
        torch.zeros_like(tensor) if gradient is None else gradient
        for tensor, gradient in zip((*cpu_inputs, h0, a_log, bias), expected_raw)
    )

    device_inputs = tuple(tensor.detach().npu() for tensor in cpu_inputs)
    state, a_log_npu, bias_npu = (tensor.detach().npu() for tensor in (h0, a_log, bias))
    cache = flash_kda_fwd(
        *(packed(tensor) for tensor in device_inputs),
        state,
        a_log_npu,
        bias_npu,
        cu_seqlens=offsets,
        scale=scale,
        lower_bound=-0.7,
        layout="TND",
    )
    metadata = _metadata(device_inputs[0], device_inputs[2], layout, offsets)
    explicit_lengths = (
        torch.tensor(lengths, dtype=torch.int32, device=device_inputs[0].device)
        if length_source != "metadata"
        else None
    )
    actual = _reference_backward(
        _reference_qk_to_input_layout(cache.q_norm, device_inputs[0], metadata, layout),
        _reference_qk_to_input_layout(cache.k_norm, device_inputs[1], metadata, layout),
        device_inputs[2],
        device_inputs[3],
        device_inputs[4],
        output_gradient.npu(),
        final_gradient.npu(),
        cache.Aqk,
        cache.Akk,
        metadata,
        None,  # initial_state is unused with saved caches
        cache.chunk_states,
        cache.w,
        cache.u,
        cache.v_new,
        cache.qg,
        cache.kg,
        cache.g_cumsum,
        disable_recompute=True,
        q_rtsd=cache.q_rstd,
        k_rstd=cache.k_rstd,
        beta_sigmoid=cache.beta_sigmoid,
        A_log=a_log_npu,
        dt_bias=bias_npu,
        scale=scale,
        seqused=explicit_lengths,
        lower_bound=-0.7,
        layout=layout,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    for name, gradient, reference in zip(
        ("dq", "dk", "dv", "dh0", "dg", "dbeta", "dA", "dBias"),
        actual,
        (
            expected[0],
            expected[1],
            expected[2],
            expected[5],
            expected[3],
            expected[4],
            expected[6],
            expected[7],
        ),
    ):
        _assert_precision(name, gradient, reference)


@pytest.mark.whitebox
@pytest.mark.npu
@pytest.mark.parametrize("layout,core_num", [("BNSD", 4), ("BSND", 32)])
def test_fused_stage_one_workspace_reuse_matches_saved_cache_golden(
    monkeypatch, layout, core_num
):
    """Repeated chunk/head reuse and a one-row tail must preserve all gradients."""
    pytest.importorskip("torch_npu")
    monkeypatch.setattr(integration, "_device_block_num", lambda ref=None: core_num)
    torch.manual_seed(941)
    length, heads, scale = 2113, 2, 128**-0.5

    def physical(rows):
        return rows.transpose(1, 2).contiguous() if layout == "BNSD" else rows

    q = physical(torch.randn(1, length, heads, 128).bfloat16())
    k = physical(torch.randn(1, length, heads, 128).bfloat16())
    v = physical((torch.randn(1, length, heads, 128) * 0.2).bfloat16())
    g = physical((torch.randn(1, length, heads, 128) * 0.1).bfloat16())
    beta = physical(torch.randn(1, length, heads).bfloat16())
    h0 = torch.randn(1, heads, 128, 128) * 0.01
    a_log = torch.full((heads,), -1.0, dtype=torch.float32)
    bias = torch.full((heads, 128), -4.0, dtype=torch.float32)
    attrs = dict(
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
        lower_bound=-0.06,
        layout=layout,
    )
    cache = flash_kda_fwd(q, k, v, g, beta, h0, a_log, bias, **attrs)
    assert cache.Aqk.shape[0] == 34
    cache_names = (
        "chunk_states",
        "Aqk",
        "Akk",
        "w",
        "u",
        "v_new",
        "qg",
        "kg",
        "g_cumsum",
        "q_norm",
        "q_rstd",
        "k_norm",
        "k_rstd",
        "beta_sigmoid",
    )
    cpu_arguments = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        do=torch.randn_like(v),
        dht=torch.randn_like(h0),
        A_log=a_log,
        dt_bias=bias,
        **{name: getattr(cache, name) for name in cache_names},
    )
    expected = flash_kda_grad(**cpu_arguments, cu_seqlens=None, **attrs)
    # Normalize reference-cache views on CPU before H2D, so this backward
    # test does not dispatch an unrelated device transpose during the copy.
    device_arguments = {
        name: tensor.contiguous().npu() for name, tensor in cpu_arguments.items()
    }
    metadata = _metadata(device_arguments["q"], device_arguments["v"], layout)
    device_arguments["q"] = _reference_qk_to_input_layout(
        device_arguments.pop("q_norm"), device_arguments["q"], metadata, layout
    )
    device_arguments["k"] = _reference_qk_to_input_layout(
        device_arguments.pop("k_norm"), device_arguments["k"], metadata, layout
    )
    device_arguments["q_rtsd"] = device_arguments.pop("q_rstd")
    actual = _reference_backward(
        disable_recompute=True, **device_arguments, metadata=metadata, **attrs
    )
    for name, got, want in zip(
        ("dq", "dk", "dv", "dh0", "dg", "dbeta", "dA", "dBias"),
        actual,
        expected,
    ):
        _assert_precision(name, got, want)


@pytest.mark.npu
def test_explicit_cache_backward_matches_token_autograd():
    """Exercise the flat saved-cache ABI independently of autograd dispatch."""
    pytest.importorskip("torch_npu")
    torch.manual_seed(771)
    q = torch.randn(2, 1, 128).bfloat16().requires_grad_()
    k = torch.randn(2, 1, 128).bfloat16().requires_grad_()
    v = (torch.randn(2, 1, 128) * 0.1).bfloat16().requires_grad_()
    g = (torch.randn(2, 1, 128) * 0.1).bfloat16().requires_grad_()
    beta = torch.randn(2, 1).bfloat16().requires_grad_()
    h0 = (torch.randn(1, 1, 128, 128) * 0.01).requires_grad_()
    a_log = torch.tensor([-1.0], requires_grad=True)
    bias = torch.full((1, 128), -4.0, requires_grad=True)
    scale = 128**-0.5
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    ref_out, ref_final = _token_golden(
        q,
        k,
        v,
        g,
        beta,
        h0,
        a_log,
        bias,
        layout="TND",
        sequence_lengths=[2],
        scale=scale,
        lower_bound=-0.7,
        norm=True,
        fused_gate=True,
        fused_beta=True,
        beta_factor=1.0,
    )
    do = torch.randn_like(ref_out).bfloat16()
    dht = torch.randn_like(ref_final)
    expected = torch.autograd.grad(
        (ref_out * do.float()).sum() + (ref_final * dht).sum(),
        (q, k, v, h0, g, beta, a_log, bias),
    )
    q_npu, k_npu, v_npu, g_npu, beta_npu, h0_npu, a_npu, bias_npu = (
        t.detach().npu() for t in (q, k, v, g, beta, h0, a_log, bias)
    )
    cache = flash_kda_fwd(
        q_npu,
        k_npu,
        v_npu,
        g_npu,
        beta_npu,
        h0_npu,
        a_npu,
        bias_npu,
        cu_seqlens=offsets,
        scale=scale,
        lower_bound=-0.7,
        layout="TND",
    )
    metadata = _metadata(q_npu, v_npu, "TND", offsets)
    assert cache.chunk_states.dtype == torch.float32
    assert cache.final_state.dtype == torch.float32
    actual = _reference_backward(
        _reference_qk_to_input_layout(cache.q_norm, q_npu, metadata, "TND"),
        _reference_qk_to_input_layout(cache.k_norm, k_npu, metadata, "TND"),
        v_npu,
        g_npu,
        beta_npu,
        do.npu(),
        dht.npu(),
        cache.Aqk,
        cache.Akk,
        metadata,
        None,  # initial_state is unused with saved caches
        cache.chunk_states,
        cache.w,
        cache.u,
        cache.v_new,
        cache.qg,
        cache.kg,
        cache.g_cumsum,
        disable_recompute=True,
        q_rtsd=cache.q_rstd,
        k_rstd=cache.k_rstd,
        beta_sigmoid=cache.beta_sigmoid,
        A_log=a_npu,
        dt_bias=bias_npu,
        scale=scale,
        lower_bound=-0.7,
        layout="TND",
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    for name, got, want in zip(
        ("dq", "dk", "dv", "dh0", "dg", "dbeta", "dA", "dBias"),
        actual,
        expected,
    ):
        _assert_precision(name, got, want)

    bad_metadata = metadata.clone()
    bad_metadata[HEADER_STATUS] = STATUS_BAD_LENGTH
    with pytest.raises(ValueError, match="backward metadata status"):
        _reference_backward(
            _reference_qk_to_input_layout(cache.q_norm, q_npu, metadata, "TND"),
            _reference_qk_to_input_layout(cache.k_norm, k_npu, metadata, "TND"),
            v_npu,
            g_npu,
            beta_npu,
            do.npu(),
            dht.npu(),
            cache.Aqk,
            cache.Akk,
            bad_metadata,
            None,  # initial_state is unused with saved caches
            cache.chunk_states,
            cache.w,
            cache.u,
            cache.v_new,
            cache.qg,
            cache.kg,
            cache.g_cumsum,
            disable_recompute=True,
            q_rtsd=cache.q_rstd,
            k_rstd=cache.k_rstd,
            beta_sigmoid=cache.beta_sigmoid,
            A_log=a_npu,
            dt_bias=bias_npu,
            scale=scale,
            lower_bound=-0.7,
            layout="TND",
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )


@pytest.mark.parametrize("disable_recompute", (False, True))
@pytest.mark.parametrize("spelling", ("positional", "keyword", "mixed"))
def test_public_argument_order_matches_spreadsheet(
    monkeypatch, disable_recompute, spelling
):
    """KDA.xls rows 2-34 must bind identically for all public calling styles."""
    names = (
        "q",
        "k",
        "v",
        "g",
        "beta",
        "do",
        "dht",
        "Aqk",
        "Akk",
        "metadata",
        "initial_state",
        "chunk_states",
        "w",
        "u",
        "v_new",
        "qg",
        "kg",
        "g_cumsum",
        "q_rtsd",
        "k_rstd",
        "beta_sigmoid",
        "A_log",
        "dt_bias",
        "cu_seqlens",
        "seqused",
        "scale",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "use_beta_sigmoid_in_kernel",
        "allow_neg_eigval",
        "disable_recompute",
        "lower_bound",
        "layout",
    )
    signature = inspect.signature(integration.flash_kda_grad_npu)
    assert tuple(signature.parameters) == names
    assert all(
        p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        for p in signature.parameters.values()
    )
    arguments = {name: object() for name in names}
    arguments.update(
        scale=0.37,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
        disable_recompute=disable_recompute,
        lower_bound=-0.7,
        layout="TND",
    )
    validation_signature = inspect.signature(integration._validate_backward_inputs)

    class ValidationReached(Exception):
        pass

    def resolve(q, v, layout, metadata, cu_seqlens, seqused):
        for name, value in zip(
            ("q", "v", "layout", "metadata", "cu_seqlens", "seqused"),
            (q, v, layout, metadata, cu_seqlens, seqused),
        ):
            assert value is arguments[name], name
        return metadata

    def validate(*args):
        values = validation_signature.bind(*args).arguments
        for name, value in values.items():
            assert value is arguments[name], name
        raise ValidationReached

    monkeypatch.setattr(integration, "_resolve_backward_metadata", resolve)
    monkeypatch.setattr(integration, "_validate_backward_inputs", validate)
    positional_count = {"positional": len(names), "keyword": 0, "mixed": 11}[spelling]
    with pytest.raises(ValidationReached):
        integration.flash_kda_grad_npu(
            *(arguments[name] for name in names[:positional_count]),
            **{name: arguments[name] for name in names[positional_count:]},
        )


@pytest.mark.parametrize("from_forward", (False, True))
@pytest.mark.parametrize(
    "gate_options,expected_gate,expected_bound",
    [
        ({"use_gate_in_kernel": False}, False, None),
        ({}, True, None),
        ({"use_gate_in_kernel": True, "lower_bound": None}, True, None),
        ({"use_gate_in_kernel": True, "lower_bound": -5.0}, True, -5.0),
        ({"use_gate_in_kernel": True, "lower_bound": -0.7}, True, -0.7),
    ],
    ids=(
        "gate-off",
        "default-softplus",
        "explicit-softplus",
        "bounded-minus-five",
        "bounded",
    ),
)
@pytest.mark.parametrize(
    "scale_options,expected_scale",
    [
        ({}, 128**-0.5),
        ({"scale": None}, 128**-0.5),
        ({"scale": 0.0}, 0.0),
        ({"scale": 0.37}, 0.37),
    ],
)
def test_public_kdag_defaults_reach_stage_two(
    monkeypatch,
    from_forward,
    scale_options,
    expected_scale,
    gate_options,
    expected_gate,
    expected_bound,
):
    """Both public entries must resolve defaults before preparing the kernel."""
    validation_signature = inspect.signature(integration._validate_backward_inputs)

    def validate(*args):
        options = validation_signature.bind(*args).arguments
        assert options["layout"] == "TND"
        assert options["use_gate_in_kernel"] is expected_gate
        assert options["lower_bound"] == expected_bound
        for name in ("use_qk_l2norm_in_kernel", "use_beta_sigmoid_in_kernel"):
            assert options[name] is True

    class StageTwoReached(Exception):
        pass

    def prepare(**kwargs):
        assert kwargs["scale"] == expected_scale
        assert kwargs["layout"] == "TND"
        raise StageTwoReached

    monkeypatch.setattr(integration, "_validate_backward_inputs", validate)
    monkeypatch.setattr(integration, "_reap_backward_workspaces", lambda: None)
    monkeypatch.setattr(integration, "_device_block_num", lambda ref: 32)
    monkeypatch.setattr(integration, "_track_chunk_plan", lambda *args: (False, True))
    monkeypatch.setattr(
        integration, "_select_backward_template", lambda *args: "persistent"
    )
    monkeypatch.setattr(
        integration._BACKWARD_SCRATCH, "acquire", lambda *args: nullcontext(None)
    )
    monkeypatch.setattr(integration, "_prepare_stage_two", prepare)
    fixture = _make_host_abi_inputs(torch.float32, layout="TND")
    fixture.update(q_norm=fixture["q"], k_norm=fixture["k"])
    inputs = tuple(fixture[name] for name in ("q", "k", "v", "g", "beta", "do", "dht"))
    saved = tuple(
        fixture.get({"q_rstd": "q_rtsd"}.get(name, name))
        for name in (
            "out",
            "final_state",
            "q_norm",
            "k_norm",
            "q_rstd",
            "k_rstd",
            "beta_sigmoid",
            "Aqk",
            "Akk",
            "w",
            "u",
            "qg",
            "kg",
            "v_new",
            "g_cumsum",
            "chunk_states",
        )
    )
    caches = tuple(
        fixture[name]
        for name in (
            "Aqk",
            "Akk",
            "metadata",
            "chunk_states",
            "w",
            "u",
            "v_new",
            "qg",
            "kg",
            "g_cumsum",
        )
    )
    with pytest.raises(StageTwoReached):
        if from_forward:
            integration.flash_kda_grad_from_forward(
                *inputs,
                saved,
                None,
                None,
                object(),
                disable_recompute=True,
                **scale_options,
                **gate_options,
            )
        else:
            integration.flash_kda_grad_npu(
                *inputs,
                *caches[:3],
                None,
                *caches[3:],
                disable_recompute=True,
                **scale_options,
                **gate_options,
            )


@pytest.mark.parametrize("spelling", ("keyword", "positional"))
def test_public_normalization_cache_names(monkeypatch, spelling):
    """Canonical keywords and positional calls pass the same inverse norms."""
    query_cache, key_cache = object(), object()
    signature = inspect.signature(integration._validate_backward_inputs)

    class ValidationReached(Exception):
        pass

    def validate(*args):
        values = signature.bind(*args).arguments
        assert values["q_rtsd"] is query_cache
        assert values["k_rstd"] is key_cache
        raise ValidationReached

    monkeypatch.setattr(integration, "_validate_backward_inputs", validate)
    inputs = tuple(object() for _ in range(18))
    options = {}
    if spelling == "keyword":
        options.update(q_rtsd=query_cache, k_rstd=key_cache)
    if spelling == "positional":
        inputs += (query_cache, key_cache)
    with pytest.raises(ValidationReached):
        integration.flash_kda_grad_npu(*inputs, disable_recompute=True, **options)


@pytest.mark.parametrize("layout", ("BNSD", "BSND", "TND"))
@pytest.mark.parametrize("norm", (False, True))
def test_forward_adapter_selects_qk_values_and_preserves_padding(
    monkeypatch, layout, norm
):
    args = _make_host_abi_inputs(torch.float32, layout=layout)
    q, k = args["q"].fill_(10), args["k"].clone().fill_(20)
    q_saved = torch.full_like(q, 0.125)
    k_saved = torch.full_like(k, -0.25)
    if layout == "BSND":
        q_saved, k_saved = q_saved.transpose(1, 2), k_saved.transpose(1, 2)
    saved = (
        None,
        None,
        q_saved,
        k_saved,
        args["q_rtsd"],
        args["k_rstd"],
        args["beta_sigmoid"],
        args["Aqk"],
        args["Akk"],
        args["w"],
        args["u"],
        args["qg"],
        args["kg"],
        args["v_new"],
        args["g_cumsum"],
        args["chunk_states"],
    )
    signature = inspect.signature(integration.flash_kda_grad_npu)
    monkeypatch.setattr(
        integration,
        "flash_kda_grad_npu",
        lambda *a, **kw: signature.bind(*a, **kw).arguments,
    )
    actual = integration.flash_kda_grad_from_forward(
        q,
        k,
        args["v"],
        args["g"],
        args["beta"],
        args["do"],
        args["dht"],
        saved,
        None,
        None,
        args["metadata"],
        layout=layout,
        use_qk_l2norm_in_kernel=norm,
        disable_recompute=True,
    )
    torch.testing.assert_close(actual["q"], torch.full_like(q, 0.125) if norm else q)
    torch.testing.assert_close(actual["k"], torch.full_like(k, -0.25) if norm else k)
    assert actual["q"].data_ptr() == (q_saved if norm else q).data_ptr()
    assert actual["k"].data_ptr() == (k_saved if norm else k).data_ptr()
    assert actual["q_rtsd"] is args["q_rtsd"] and actual["k_rstd"] is args["k_rstd"]


@pytest.mark.parametrize("layout", ("BNSD", "BSND", "TND"))
def test_public_normalized_qk_reach_stage_one_without_rescaling(
    monkeypatch, host_sequence_metadata, layout
):
    args = _make_host_abi_inputs(torch.float32, layout=layout)
    lengths = (
        dict(cu_seqlens=torch.tensor([0, 2], dtype=torch.int32))
        if layout == "TND"
        else dict(seqused=torch.tensor([1], dtype=torch.int32))
    )
    args["metadata"] = host_sequence_metadata.flash_kda_grad_metadata(
        **_metadata_arguments(args["q"], args["v"], layout, **lengths)
    )
    q, k = args["q"].fill_(0.125), args["k"].clone().fill_(-0.25)
    args["q"], args["k"] = q, k
    args["q_rtsd"] = torch.full_like(args["q_rtsd"], 3.0)
    args["k_rstd"] = torch.full_like(args["k_rstd"], 5.0)

    class Prepared(Exception):
        pass

    def prepare(**values):
        assert values["query"] is q and values["key"] is k
        for name, source, value in (
            ("query_rstd", "q_rtsd", 3.0),
            ("key_rstd", "k_rstd", 5.0),
        ):
            torch.testing.assert_close(values[name], torch.full(q.shape[:-1], value))
            assert values[name].data_ptr() == args[source].data_ptr()
        raise Prepared

    monkeypatch.setattr(integration, "_reap_backward_workspaces", lambda: None)
    monkeypatch.setattr(integration, "_device_block_num", lambda _: 1)
    monkeypatch.setattr(
        integration._BACKWARD_SCRATCH, "acquire", lambda *a: nullcontext(None)
    )
    monkeypatch.setattr(integration, "_prepare_stage_two", lambda **kw: (None, (), ()))
    monkeypatch.setattr(integration, "_prepare_stage_one", prepare)
    with pytest.raises(Prepared):
        integration.flash_kda_grad_npu(disable_recompute=True, **args)


@pytest.mark.parametrize("layout", ("BNSD", "BSND"))
def test_profile_fixture_stores_normalized_qk_with_v3_schema(layout):
    args = SimpleNamespace(layout=layout, length=3, heads=2, lower_bound=-0.7)
    fixture = _profile_build_fixture(args)
    inputs = fixture["main"]["inputs"]
    assert fixture["schema"] == "flash_kda_grad.main.v3"
    assert "q_norm" not in inputs and "k_norm" not in inputs
    assert "q_rtsd" in inputs and "k_rstd" in inputs
    shape = (1, 2, 3, 128) if layout == "BNSD" else (1, 3, 2, 128)
    for name in ("q", "k"):
        assert tuple(inputs[name].shape) == shape
        torch.testing.assert_close(
            inputs[name].float().norm(dim=-1),
            torch.ones(shape[:-1]),
            atol=0.002,
            rtol=0,
        )


def test_bin_reader_rejects_removed_qk_cache_files(tmp_path):
    args = _write_bin_test_bundle(tmp_path)
    _bin_tensor_path(tmp_path, "q_norm", "prefixed").write_bytes(b"")
    with pytest.raises(ValueError, match="old Q/K cache files"):
        _bin_load_bundle(args)


def _make_host_abi_inputs(gate_dtype=None, *, layout="BNSD"):
    """Build one valid CPU-backed host-validation fixture, with shared cache views.

    These tensors carry no numerical golden: each caller corrupts one ABI field.
    None preserves the caller's default gate dtype; cache tests request FP32.
    """
    token = torch.empty((1, 1, 2, 128), dtype=torch.bfloat16)
    saved_row = torch.empty((1, 1, 2, 128), dtype=torch.bfloat16)
    matrix = torch.empty((1, 1, 1, 64, 64), dtype=torch.bfloat16)
    state = torch.empty((1, 1, 128, 128), dtype=torch.float32)
    rstd = torch.empty((1, 1, 2), dtype=torch.float32)
    beta = torch.empty((1, 1, 2), dtype=torch.bfloat16)
    metadata = torch.tensor(
        [
            metadata_module.MAGIC,
            metadata_module.ABI_VERSION,
            metadata_module.STATUS_OK,
            1,
            1,
            2,
            1,
            0,
            0,
            0,
            0,
            1,
            2,
            0,
            0,
        ],
        dtype=torch.int32,
    )
    args = dict(
        q=token,
        k=token,
        v=token,
        g=token,
        beta=beta,
        do=token,
        dht=state,
        chunk_states=state.unsqueeze(2),
        Aqk=matrix,
        Akk=matrix,
        w=saved_row,
        u=saved_row,
        v_new=saved_row,
        qg=saved_row,
        kg=saved_row,
        g_cumsum=saved_row.float(),
        q_rtsd=rstd,
        k_rstd=rstd,
        beta_sigmoid=rstd,
        A_log=torch.zeros((1,), dtype=gate_dtype),
        dt_bias=torch.zeros((1, 128), dtype=gate_dtype),
        metadata=metadata,
        scale=0.125,
        layout="BNSD",
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    if layout == "BSND":
        for name in ("q", "k", "v", "g", "beta", "do"):
            args[name] = args[name].transpose(1, 2)
    elif layout == "TND":
        for name in (
            "q",
            "k",
            "v",
            "g",
            "beta",
            "do",
            "w",
            "u",
            "v_new",
            "qg",
            "kg",
            "g_cumsum",
            "q_rtsd",
            "k_rstd",
            "beta_sigmoid",
        ):
            args[name] = args[name][0].transpose(0, 1)
        for name in ("Aqk", "Akk", "chunk_states"):
            tensor = args[name]
            args[name] = tensor.new_empty((2, 1, *tensor.shape[-2:]))
        args["metadata"][metadata_module.HEADER_FLAGS] = 3
    args["layout"] = layout
    return args


@pytest.mark.parametrize("from_forward", (False, True))
@pytest.mark.parametrize("omitted", (False, True))
@pytest.mark.parametrize("explicit_lengths", (False, True))
def test_public_metadata_is_required_before_generation(
    monkeypatch, from_forward, omitted, explicit_lengths
):
    args = _make_host_abi_inputs(torch.float32)
    if omitted:
        args.pop("metadata")
    else:
        args["metadata"] = None
    options = (
        dict(seqused=torch.tensor([2], dtype=torch.int32)) if explicit_lengths else {}
    )
    monkeypatch.setattr(
        metadata_module,
        "flash_kda_grad_metadata",
        lambda *a, **kw: pytest.fail("missing metadata triggered AICPU generation"),
    )
    with pytest.raises(TypeError if omitted else ValueError, match="metadata"):
        if from_forward:
            metadata_option = {} if omitted else dict(metadata=None)
            integration.flash_kda_grad_from_forward(
                *(args[name] for name in ("q", "k", "v", "g", "beta", "do", "dht")),
                (None,) * 16,
                None,
                None,
                layout="BNSD",
                **metadata_option,
                **options,
            )
        else:
            integration.flash_kda_grad_npu(**args, **options)


@pytest.mark.parametrize("layout", ["BNSD", "BSND", "TND"])
@pytest.mark.parametrize("from_forward", [False, True])
@pytest.mark.parametrize("supplied_metadata", [False, True])
def test_public_sequence_lengths_reach_compute_preparation(
    monkeypatch, host_sequence_metadata, layout, from_forward, supplied_metadata
):
    args = _make_host_abi_inputs(torch.float32, layout=layout)
    options = (
        dict(cu_seqlens=torch.tensor([0, 2], dtype=torch.int32))
        if layout == "TND"
        else dict(seqused=torch.tensor([1], dtype=torch.int32))
    )
    metadata = (
        host_sequence_metadata.flash_kda_grad_metadata(
            **_metadata_arguments(args["q"], args["v"], layout, **options)
        )
        if supplied_metadata
        else None
    )
    args["metadata"] = metadata

    class Prepared(Exception):
        pass

    def prepare(**values):
        info = host_sequence_metadata.metadata_host_info(values["metadata"])
        assert info.sequences == ((0, 2 if layout == "TND" else 1, 1, 0),)
        if supplied_metadata:
            assert values["metadata"] is metadata
        raise Prepared

    monkeypatch.setattr(integration, "_reap_backward_workspaces", lambda: None)
    monkeypatch.setattr(integration, "_device_block_num", lambda q: 1)
    monkeypatch.setattr(
        integration._BACKWARD_SCRATCH, "acquire", lambda *a: nullcontext(None)
    )
    monkeypatch.setattr(integration, "_prepare_stage_two", prepare)
    with pytest.raises(
        Prepared if supplied_metadata else ValueError,
        match=None if supplied_metadata else "metadata is required",
    ):
        if from_forward:
            saved = tuple(
                args.get(
                    {"q_norm": "q", "k_norm": "k", "q_rstd": "q_rtsd"}.get(name, name)
                )
                for name in (
                    "out",
                    "final_state",
                    "q_norm",
                    "k_norm",
                    "q_rstd",
                    "k_rstd",
                    "beta_sigmoid",
                    "Aqk",
                    "Akk",
                    "w",
                    "u",
                    "qg",
                    "kg",
                    "v_new",
                    "g_cumsum",
                    "chunk_states",
                )
            )
            if layout == "BSND":
                saved = (
                    *saved[:2],
                    saved[2].transpose(1, 2),
                    saved[3].transpose(1, 2),
                    *saved[4:],
                )
            integration.flash_kda_grad_from_forward(
                *(args[name] for name in ("q", "k", "v", "g", "beta", "do", "dht")),
                saved,
                args["A_log"],
                args["dt_bias"],
                metadata,
                disable_recompute=True,
                layout=layout,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                **options,
            )
        else:
            integration.flash_kda_grad_npu(disable_recompute=True, **args, **options)


@pytest.mark.parametrize("layout", ["BNSD", "BSND", "TND"])
def test_explicit_sequence_lengths_reject_stale_metadata(
    host_sequence_metadata, layout
):
    q = torch.empty(
        (65, 1, 128)
        if layout == "TND"
        else (2, 1, 65, 128)
        if layout == "BNSD"
        else (2, 65, 1, 128),
        dtype=torch.bfloat16,
    )
    old = torch.tensor([0, 0, 65] if layout == "TND" else [0, 65], dtype=torch.int32)
    new = torch.tensor([0, 1, 65] if layout == "TND" else [1, 64], dtype=torch.int32)
    name = "cu_seqlens" if layout == "TND" else "seqused"
    metadata = host_sequence_metadata.flash_kda_grad_metadata(
        **_metadata_arguments(q, q, layout, **{name: old})
    )
    with pytest.raises(ValueError, match="metadata does not match"):
        integration._resolve_backward_metadata(
            q,
            q,
            layout,
            metadata,
            **{
                "cu_seqlens": new if layout == "TND" else None,
                "seqused": None if layout == "TND" else new,
            },
        )


def test_explicit_full_lengths_accept_legacy_metadata(host_sequence_metadata):
    q = torch.empty((2, 1, 64, 128), dtype=torch.bfloat16)
    metadata = host_sequence_metadata.flash_kda_grad_metadata(
        **_metadata_arguments(q, q, "BNSD")
    )
    resolved = integration._resolve_backward_metadata(
        q, q, "BNSD", metadata, None, torch.tensor([64, 64], dtype=torch.int32)
    )
    assert resolved is metadata


@pytest.mark.parametrize(
    "layout,options",
    [
        ("BNSD", dict(cu_seqlens=torch.tensor([0, 2], dtype=torch.int32))),
        ("BSND", dict(cu_seqlens=torch.tensor([0, 2], dtype=torch.int32))),
        ("TND", dict(seqused=torch.tensor([2], dtype=torch.int32))),
    ],
)
def test_public_sequence_arguments_match_layout(monkeypatch, layout, options):
    monkeypatch.setattr(
        integration,
        "_validate_backward_inputs",
        lambda *a: pytest.fail("invalid length API reached cache validation"),
    )
    with pytest.raises(ValueError, match="only supported"):
        integration.flash_kda_grad_npu(
            *(object() for _ in range(10)),
            disable_recompute=True,
            layout=layout,
            **options,
        )


@pytest.mark.parametrize("length", [-1, 3])
def test_public_invalid_lengths_stop_before_compute(
    host_sequence_metadata, monkeypatch, length
):
    args = _make_host_abi_inputs(gate_dtype=torch.float32)
    monkeypatch.setattr(
        integration,
        "_prepare_stage_two",
        lambda **kw: pytest.fail("invalid length reached compute"),
    )
    with pytest.raises(ValueError, match="metadata status"):
        integration.flash_kda_grad_npu(
            disable_recompute=True,
            **args,
            seqused=torch.tensor([length], dtype=torch.int32),
        )


@pytest.mark.parametrize("offsets", [[1, 2], [0, 1], [0, 2, 1]])
def test_public_invalid_offsets_stop_before_compute(
    host_sequence_metadata, monkeypatch, offsets
):
    args = _make_host_abi_inputs(gate_dtype=torch.float32)
    for name in ("q", "k", "v", "g", "do", "beta"):
        args[name] = args[name].squeeze(0).transpose(0, 1).contiguous()
    args.update(layout="TND")
    monkeypatch.setattr(
        integration,
        "_prepare_stage_two",
        lambda **kw: pytest.fail("invalid offsets reached compute"),
    )
    with pytest.raises(ValueError, match="metadata status"):
        integration.flash_kda_grad_npu(
            disable_recompute=True,
            **args,
            cu_seqlens=torch.tensor(offsets, dtype=torch.int32),
        )


@pytest.mark.parametrize("from_forward", (False, True))
@pytest.mark.parametrize("missing", ("do", "dht", "both"))
def test_public_entries_require_upstream_gradients(monkeypatch, from_forward, missing):
    """Missing seeds must fail on the host before any allocation or launch."""
    args = _make_host_abi_inputs()
    if missing in ("do", "both"):
        args["do"] = None
    if missing in ("dht", "both"):
        args["dht"] = None
    monkeypatch.setattr(
        integration,
        "_prepare_stage_two",
        lambda **kwargs: pytest.fail("missing gradient reached compute preparation"),
    )
    expected = "do" if missing in ("do", "both") else "dht"
    with pytest.raises(ValueError, match=f"{expected} must be a Tensor"):
        if from_forward:
            integration.flash_kda_grad_from_forward(
                *(args[name] for name in ("q", "k", "v", "g", "beta", "do", "dht")),
                (None,) * 16,
                None,
                None,
                args["metadata"],
                disable_recompute=True,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            integration.flash_kda_grad_npu(disable_recompute=True, **args)


@pytest.mark.whitebox
@pytest.mark.parametrize(
    "case,expected_message",
    [
        ("q_on_cpu", "one NPU"),
        ("k_wrong_dtype", "BF16"),
        ("state_wrong_dtype", "FP32"),
        ("gamma_wrong_dtype", "FP32"),
        ("unsupported_layout", "layout must"),
        ("wrong_width", "Q/K layout or width"),
        ("zero_value_heads", "value heads"),
        ("invalid_gqa_ratio", "value heads"),
        ("value_shape", "V/g/beta physical shapes"),
        ("gate_shape", "V/g/beta physical shapes"),
        ("beta_shape", "V/g/beta physical shapes"),
        ("output_gradient_dtype", "do must"),
        ("missing_output_gradient", "do must be a Tensor"),
        ("output_gradient_not_tensor", "do must be a Tensor"),
        ("metadata_dtype", "metadata must"),
        ("missing_metadata", "metadata is required"),
        ("metadata_capacity", "metadata must"),
        ("metadata_magic", "metadata ABI"),
        ("metadata_version", "metadata ABI"),
        ("metadata_status", "metadata status"),
        ("metadata_storage_length", "metadata input shape"),
        ("metadata_layout_flag", "metadata input shape"),
        ("short_token_cache", "qg must"),
        ("short_matrix_cache", "Aqk/Akk"),
        ("short_state_cache", "chunk_states shape"),
        ("short_physical_state_cache", "chunk_states shape"),
        ("short_norm_cache", "FP32 q_rtsd and k_rstd caches"),
        ("short_beta_cache", "activated beta"),
        ("token_cache_shape", "qg must"),
        ("saved_row_mismatch", "saved token caches"),
        ("matrix_cache_shape", "Aqk/Akk"),
        ("inverse_shape", "Aqk/Akk"),
        ("state_shape", "chunk_states shape"),
        ("final_gradient_dtype", "dht must"),
        ("missing_final_gradient", "dht must be a Tensor"),
        ("final_gradient_not_tensor", "dht must be a Tensor"),
        ("missing_query_rtsd", "FP32 q_rtsd and k_rstd caches"),
        ("missing_key_rstd", "FP32 q_rtsd and k_rstd caches"),
        ("query_rtsd_not_tensor", "FP32 q_rtsd and k_rstd caches"),
        ("key_rstd_dtype", "FP32 q_rtsd and k_rstd caches"),
        ("missing_activated_beta", "activated beta"),
        ("missing_gate_parameter", "A_log must"),
        ("gate_bias_shape", "dt_bias must"),
        ("negative_eigval_without_sigmoid", "allow_neg_eigval requires"),
        ("lower_bound_out_of_range", "lower_bound must"),
        ("lower_bound_zero", "lower_bound must"),
        ("lower_bound_nan", "lower_bound must"),
    ],
)
def test_invalid_public_abi_rejected_before_compute(
    monkeypatch, case, expected_message
):
    """Every host contract failure must stop before compiling the AICore kernel."""
    # These CPU fixtures emulate device metadata only; no producer stream exists.
    monkeypatch.setattr(integration, "wait_metadata_ready", lambda metadata: None)
    if case != "q_on_cpu":
        fake_device = SimpleNamespace(type="npu", index=0)
        monkeypatch.setattr(
            torch.Tensor, "device", property(lambda tensor: fake_device)
        )
    args = _make_host_abi_inputs()

    def fake_npu_empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype)

    def corrupted_header(word, value):
        result = args["metadata"].clone()
        result[word] = value
        return {"metadata": result}

    changed = {
        "q_on_cpu": lambda: {"q": args["q"].cpu()},
        "k_wrong_dtype": lambda: {"k": args["q"].float()},
        "state_wrong_dtype": lambda: {"chunk_states": args["chunk_states"].bfloat16()},
        "gamma_wrong_dtype": lambda: {"g_cumsum": args["qg"]},
        "unsupported_layout": lambda: {"layout": "BSD"},
        "wrong_width": lambda: {"q": fake_npu_empty((1, 1, 2, 64))},
        "zero_value_heads": lambda: {"v": fake_npu_empty((1, 0, 2, 128))},
        "invalid_gqa_ratio": lambda: {
            "q": fake_npu_empty((1, 2, 2, 128)),
            "k": fake_npu_empty((1, 2, 2, 128)),
        },
        "value_shape": lambda: {"v": fake_npu_empty((1, 1, 3, 128))},
        "gate_shape": lambda: {"g": fake_npu_empty((1, 1, 3, 128))},
        "beta_shape": lambda: {"beta": fake_npu_empty((1, 1, 3))},
        "output_gradient_dtype": lambda: {"do": args["q"].float()},
        "missing_output_gradient": lambda: {"do": None},
        "output_gradient_not_tensor": lambda: {"do": 0.0},
        "metadata_dtype": lambda: {"metadata": args["metadata"].float()},
        "missing_metadata": lambda: {"metadata": None},
        "metadata_capacity": lambda: {"metadata": fake_npu_empty((7,), torch.int32)},
        "metadata_magic": lambda: corrupted_header(metadata_module.HEADER_MAGIC, 0),
        "metadata_version": lambda: corrupted_header(
            metadata_module.HEADER_ABI_VERSION, 0
        ),
        "metadata_status": lambda: corrupted_header(HEADER_STATUS, STATUS_BAD_LENGTH),
        "metadata_storage_length": lambda: corrupted_header(
            metadata_module.HEADER_STORAGE_LENGTH, 3
        ),
        "metadata_layout_flag": lambda: corrupted_header(
            metadata_module.HEADER_FLAGS, 2
        ),
        "short_token_cache": lambda: {"qg": fake_npu_empty((1, 1, 128))},
        "short_matrix_cache": lambda: {
            "Aqk": fake_npu_empty((0, 1, 64, 64)),
            "Akk": fake_npu_empty((0, 1, 64, 64)),
        },
        "short_state_cache": lambda: {
            "chunk_states": fake_npu_empty((0, 1, 128, 128), torch.float32)
        },
        "short_physical_state_cache": lambda: {
            "chunk_states": fake_npu_empty((1, 1, 0, 128, 128), torch.float32)
        },
        "short_norm_cache": lambda: {"q_rtsd": fake_npu_empty((1, 1), torch.float32)},
        "short_beta_cache": lambda: {
            "beta_sigmoid": fake_npu_empty((1, 1), torch.float32)
        },
        "token_cache_shape": lambda: {"qg": fake_npu_empty((3, 1, 128))},
        "saved_row_mismatch": lambda: {"w": fake_npu_empty((1, 1, 128))},
        "matrix_cache_shape": lambda: {"Aqk": fake_npu_empty((1, 1, 32, 64))},
        "inverse_shape": lambda: {"Akk": fake_npu_empty((2, 1, 64, 64))},
        "state_shape": lambda: {
            "chunk_states": fake_npu_empty((1, 1, 64, 128), torch.float32)
        },
        "final_gradient_dtype": lambda: {"dht": args["chunk_states"].bfloat16()},
        "missing_final_gradient": lambda: {"dht": None},
        "final_gradient_not_tensor": lambda: {"dht": 0.0},
        "missing_query_rtsd": lambda: {"q_rtsd": None},
        "missing_key_rstd": lambda: {"k_rstd": None},
        "query_rtsd_not_tensor": lambda: {"q_rtsd": 1.0},
        "key_rstd_dtype": lambda: {"k_rstd": args["q_rtsd"].bfloat16()},
        "missing_activated_beta": lambda: {"beta_sigmoid": None},
        "missing_gate_parameter": lambda: {"A_log": None},
        "gate_bias_shape": lambda: {"dt_bias": fake_npu_empty((1, 64), torch.float32)},
        "negative_eigval_without_sigmoid": lambda: {
            "allow_neg_eigval": True,
            "use_beta_sigmoid_in_kernel": False,
        },
        "lower_bound_out_of_range": lambda: {"lower_bound": -6.0},
        "lower_bound_zero": lambda: {"lower_bound": 0.0},
        "lower_bound_nan": lambda: {"lower_bound": float("nan")},
    }
    args.update(changed[case]())
    monkeypatch.setattr(
        integration,
        "FlashKDAGrad",
        lambda: pytest.fail("invalid ABI reached the compute launch"),
    )
    with pytest.raises(ValueError, match=expected_message):
        integration.flash_kda_grad_npu(disable_recompute=True, **args)


@pytest.mark.whitebox
@pytest.mark.parametrize(
    "case,expected_message",
    [
        ("stale_storage", "metadata input shape"),
        ("stale_layout", "metadata input shape"),
        ("short_tokens", "qg must"),
        ("short_matrices", "Aqk/Akk"),
        ("short_states", "chunk_states shape"),
        ("short_physical_states", "chunk_states shape"),
        ("short_norm", "FP32 q_rtsd and k_rstd caches"),
        ("short_beta", "activated beta"),
    ],
)
def test_forward_cache_capacity_and_metadata_match_before_launch(
    monkeypatch,
    case,
    expected_message,
):
    """Reject incompatible saved storage before it can reach device address math."""
    # CPU-backed ABI fixtures have no AICPU readiness event to wait on.
    monkeypatch.setattr(integration, "wait_metadata_ready", lambda metadata: None)
    device = SimpleNamespace(type="npu", index=0)
    monkeypatch.setattr(torch.Tensor, "device", property(lambda tensor: device))
    args = _make_host_abi_inputs(gate_dtype=torch.float32)
    if case == "stale_storage":
        args["metadata"][metadata_module.HEADER_STORAGE_LENGTH] = 3
    elif case == "stale_layout":
        args["metadata"][metadata_module.HEADER_FLAGS] = 2
    elif case == "short_tokens":
        args["qg"] = args["qg"][:, :, :1]
    elif case == "short_matrices":
        args["Aqk"] = args["Aqk"][:0]
        args["Akk"] = args["Akk"][:0]
    elif case == "short_states":
        args["chunk_states"] = args["chunk_states"][:0]
    elif case == "short_physical_states":
        args["chunk_states"] = args["chunk_states"].unsqueeze(2)[:, :, :0]
    elif case == "short_norm":
        args["q_rtsd"] = args["q_rtsd"][..., :1]
    else:
        args["beta_sigmoid"] = args["q_rtsd"][..., :1]
    monkeypatch.setattr(
        integration,
        "FlashKDAGrad",
        lambda: pytest.fail("invalid ABI reached the compute launch"),
    )
    with pytest.raises(ValueError, match=expected_message):
        integration.flash_kda_grad_npu(disable_recompute=True, **args)


def _make_forward_layout_inputs(
    host_sequence_metadata, layout, lengths, *, qk_heads=1, value_heads=2
):
    """Build GQA caches with the exact current forward capacity and axis order."""
    batch, nq, nv, width = len(lengths), qk_heads, value_heads, 128
    storage = sum(lengths) if layout == "TND" else max(lengths)

    def raw(heads, scalar=False):
        shape = (
            (storage, heads)
            if layout == "TND"
            else (batch, heads, storage)
            if layout == "BNSD"
            else (batch, storage, heads)
        )
        return torch.zeros(shape if scalar else (*shape, width), dtype=torch.bfloat16)

    q, v = raw(nq), raw(nv)
    token_shape = (
        (storage, nv, width) if layout == "TND" else (batch, nv, storage, width)
    )
    norm_shape = (storage, nq) if layout == "TND" else (batch, nq, storage)
    capacity = (storage + 63) // 64 + batch if layout == "TND" else (storage + 63) // 64
    chunk_prefix = (capacity, nv) if layout == "TND" else (batch, nv, capacity)
    matrix = torch.zeros((*chunk_prefix, 64, 64), dtype=torch.bfloat16)
    state = torch.zeros((*chunk_prefix, width, width))
    token = torch.zeros(token_shape, dtype=torch.bfloat16)
    args = dict(
        q=q,
        k=q.clone(),
        v=v,
        g=v.clone(),
        beta=raw(nv, True),
        do=v.clone(),
        dht=torch.zeros((batch, nv, width, width)),
        chunk_states=state,
        Aqk=matrix,
        Akk=matrix.clone(),
        w=token,
        u=token.clone(),
        v_new=token.clone(),
        qg=token.clone(),
        kg=token.clone(),
        g_cumsum=token.float(),
        q_rtsd=torch.ones(norm_shape),
        k_rstd=torch.ones(norm_shape),
        beta_sigmoid=torch.full(token_shape[:-1], 0.5),
        A_log=torch.zeros(nv),
        dt_bias=torch.zeros(nv, width),
        layout=layout,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    lengths_arg = (
        dict(
            cu_seqlens=torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
            )
        )
        if layout == "TND"
        else dict(seqused=torch.tensor(lengths, dtype=torch.int32))
    )
    args["metadata"] = host_sequence_metadata.flash_kda_grad_metadata(
        **_metadata_arguments(q, v, layout, **lengths_arg)
    )
    return args


@pytest.mark.parametrize("layout", ("BNSD", "BSND", "TND"))
@pytest.mark.parametrize(
    "disabled,field",
    [
        (True, "Aqk"),
        (True, "Akk"),
        (True, "chunk_states"),
        (False, "Aqk"),
        (False, "Akk"),
    ],
)
def test_public_rejects_nonforward_chunk_caches(
    monkeypatch, host_sequence_metadata, layout, disabled, field
):
    args = _make_forward_layout_inputs(host_sequence_metadata, layout, (0, 64))
    cache = args[field]
    # TND compact capacity and padded flattened/compact caches are former ABIs.
    args[field] = cache[:1] if layout == "TND" else cache[:, :, 0]
    monkeypatch.setattr(
        integration,
        "_allocate_recompute_caches",
        lambda *a, **kw: pytest.fail("invalid cache reached recomputation"),
    )
    monkeypatch.setattr(
        integration,
        "_prepare_stage_two",
        lambda **kw: pytest.fail("invalid cache reached backward"),
    )
    with pytest.raises(ValueError, match="Aqk/Akk|chunk_states shape"):
        integration.flash_kda_grad_npu(**args, disable_recompute=disabled)


@pytest.mark.parametrize("layout", ("BNSD", "BSND"))
@pytest.mark.parametrize(
    "field", ("qg", "g_cumsum", "q_rtsd", "k_rstd", "beta_sigmoid")
)
def test_public_rejects_compact_padded_token_caches(
    monkeypatch, host_sequence_metadata, layout, field
):
    args = _make_forward_layout_inputs(host_sequence_metadata, layout, (0, 64))
    args[field] = args[field][1].transpose(0, 1)
    monkeypatch.setattr(
        integration,
        "_prepare_stage_two",
        lambda **kw: pytest.fail("invalid cache reached backward"),
    )
    with pytest.raises(
        ValueError, match="qg must|saved token caches|FP32 q_rtsd|activated beta"
    ):
        integration.flash_kda_grad_npu(**args, disable_recompute=True)


@pytest.mark.parametrize("layout", ("BNSD", "BSND", "TND"))
@pytest.mark.parametrize("disabled", (True, False))
@pytest.mark.parametrize("lengths", ((0, 64), (17, 65), (0, 0)))
@pytest.mark.parametrize("initial_present", (False, True))
@pytest.mark.parametrize("spelling", ("keyword", "positional"))
def test_public_forward_views_and_launch_count(
    monkeypatch,
    host_sequence_metadata,
    layout,
    disabled,
    lengths,
    initial_present,
    spelling,
):
    args = _make_forward_layout_inputs(host_sequence_metadata, layout, lengths)
    if initial_present:
        args["initial_state"] = torch.full_like(args["dht"], 0.25)
    launched = []

    def same_storage(left, right):
        assert left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()

    def stage_two(**values):
        same_storage(values["aqk"], args["Aqk"])
        if disabled:
            assert values["chunk_states"] is args["chunk_states"]
        else:
            assert values["chunk_states"] is not args["chunk_states"]
            assert values["chunk_states"].shape == args["chunk_states"].shape
        if layout == "TND":
            assert values["aqk"] is args["Aqk"]
        else:
            assert values["aqk"].shape == (*args["v"].shape[:-1], 64)
        if disabled:
            same_storage(values["qg"], args["qg"])
        assert values["qg"].shape == args["v"].shape
        return SimpleNamespace(dh0=args["dht"]), (), ()

    def stage_one(**values):
        assert values["query"] is args["q"] and values["key"] is args["k"]
        same_storage(values["saved_inverse"], args["Akk"])
        same_storage(values["query_rstd"], args["q_rtsd"])
        same_storage(values["activated_beta"], args["beta_sigmoid"])
        assert values["query_rstd"].shape == args["q"].shape[:-1]
        assert values["activated_beta"].shape == args["beta"].shape
        return (
            (
                args["q"],
                args["k"],
                args["v"],
                args["g"],
                args["beta"],
                args["A_log"],
                args["dt_bias"],
            ),
            (),
            (),
            (),
        )

    def recompute(device, tensors, config, scale, core_num, has_initial_state):
        launched.append("fused")
        assert has_initial_state == initial_present
        assert tensors[-2] is args.get("initial_state", args["dht"])
        assert tensors[-1].shape[1] == 2

    monkeypatch.setattr(integration, "_reap_backward_workspaces", lambda: None)
    monkeypatch.setattr(integration, "_device_block_num", lambda _: 1)
    monkeypatch.setattr(
        integration._BACKWARD_SCRATCH, "acquire", lambda *a: nullcontext(None)
    )
    monkeypatch.setattr(integration, "_prepare_stage_two", stage_two)
    monkeypatch.setattr(integration, "_prepare_stage_one", stage_one)
    monkeypatch.setattr(
        integration, "_launch_backward_program", lambda *a: launched.append("backward")
    )
    monkeypatch.setattr(integration, "_launch_recompute_program", recompute)
    args["disable_recompute"] = disabled
    if spelling == "positional":
        bound = inspect.signature(integration.flash_kda_grad_npu).bind(**args)
        bound.apply_defaults()
        integration.flash_kda_grad_npu(*bound.args)
    else:
        integration.flash_kda_grad_npu(**args)
    assert launched == (["backward"] if disabled else ["fused"])


@pytest.mark.parametrize(
    "name", ("reserved_matrix_chunks", "reserved_state_chunks", "q_rstd", "k_rtsd")
)
def test_removed_keywords_are_rejected(name):
    with pytest.raises(TypeError, match="unexpected keyword"):
        integration.flash_kda_grad_npu(**{name: True})


@pytest.mark.parametrize("layout", ("BNSD", "BSND", "TND"))
def test_reference_bridge_preserves_forward_slot_values(layout):
    lengths, nq, nv, storage = (0, 17, 65), 1, 2, 65
    shape = (
        (82, nv, 128)
        if layout == "TND"
        else (3, nv, storage, 128)
        if layout == "BNSD"
        else (3, storage, nv, 128)
    )
    qshape = list(shape)
    qshape[1 if layout != "BSND" else 2] = nq
    q, v = torch.zeros(qshape), torch.zeros(shape)
    token = torch.arange(82 * nv * 128).reshape(82, nv, 128)
    matrix = torch.arange(3 * nv * 64 * 64, dtype=torch.float32).reshape(3, nv, 64, 64)
    state = torch.arange(3 * nv * 128 * 128, dtype=torch.float32).reshape(
        3, nv, 128, 128
    )
    converted = _reference_forward_arguments(
        dict(q=q, v=v, qg=token, Aqk=matrix, chunk_states=state, layout=layout), lengths
    )
    start = first = 0
    for sequence, length in enumerate(lengths):
        count = (length + 63) // 64
        base = start // 64 + sequence
        if layout == "TND":
            torch.testing.assert_close(
                converted["qg"][start : start + length], token[start : start + length]
            )
            torch.testing.assert_close(
                converted["Aqk"][base : base + count], matrix[first : first + count]
            )
            torch.testing.assert_close(
                converted["chunk_states"][base : base + count],
                state[first : first + count],
            )
        else:
            torch.testing.assert_close(
                converted["qg"][sequence, :, :length],
                token[start : start + length].transpose(0, 1),
            )
            torch.testing.assert_close(
                converted["Aqk"][sequence, :, :count],
                matrix[first : first + count].transpose(0, 1),
            )
            torch.testing.assert_close(
                converted["chunk_states"][sequence, :, :count],
                state[first : first + count].transpose(0, 1),
            )
        start += length
        first += count


@pytest.mark.parametrize("layout", ("BNSD", "BSND", "TND"))
def test_backward_cache_shapes_match_real_forward_meta(host_sequence_metadata, layout):
    if not os.environ.get("KDA_DSL_ROOT"):
        pytest.skip("optional forward integration requires KDA_DSL_ROOT")
    forward = load_sample("flash_kda_fwd/flash_kda_fwd.py")
    args = _make_forward_layout_inputs(host_sequence_metadata, layout, (0, 17, 65))
    saved = forward._flash_kda_fwd_meta(
        args["q"],
        args["k"],
        args["v"],
        args["g"],
        args["beta"],
        128**-0.5,
        args["dht"],
        args["A_log"],
        args["dt_bias"],
        -0.7,
        layout,
        None,
        disable_recompute=True,
        return_intermediate_state=True,
        output_final_state=True,
    )
    names = (
        "q_rtsd",
        "k_rstd",
        "beta_sigmoid",
        "Aqk",
        "Akk",
        "w",
        "u",
        "qg",
        "kg",
        "v_new",
        "g_cumsum",
        "chunk_states",
    )
    for name, tensor in zip(names, saved[4:]):
        assert args[name].shape == tensor.shape, name
        assert args[name].dtype == tensor.dtype, name


_RECOMPUTE_CACHE_NAMES = ("w", "u", "v_new", "qg", "kg", "g_cumsum")


@pytest.mark.parametrize("mode", (True, False, None))
@pytest.mark.parametrize("supplied", (True, False))
def test_public_recompute_dispatch(monkeypatch, host_sequence_metadata, mode, supplied):
    args = _make_host_abi_inputs(gate_dtype=torch.float32)
    args["metadata"] = host_sequence_metadata.flash_kda_grad_metadata(
        **_metadata_arguments(
            args["q"], args["v"], "BNSD", seqused=torch.tensor([2], dtype=torch.int32)
        )
    )
    names = (*_RECOMPUTE_CACHE_NAMES, "chunk_states")
    originals = tuple(args[name] for name in names)
    if not supplied:
        for name in names:
            args.pop(name)
    if mode is not None:
        args["disable_recompute"] = mode
    replacements = tuple(
        torch.full_like(args["v"], i, dtype=torch.float32 if i == 5 else torch.bfloat16)
        for i in range(6)
    ) + (torch.zeros_like(originals[-1]),)
    calls = []

    class Prepared(Exception):
        pass

    def rebuild(values, final_state_gradient, layout):
        assert mode is False, "saved-cache path entered recomputation"
        assert values is args["v"] and final_state_gradient is args["dht"]
        assert layout == "BNSD"
        calls.append("allocate")
        return replacements

    def prepare(**values):
        expected = replacements if mode is False else originals
        assert values["w"] is expected[0]
        assert values["v_new"] is expected[2]
        assert values["qg"] is expected[3]
        assert values["kg"] is expected[4]
        assert values["g_cumsum"] is expected[5]
        assert values["chunk_states"] is expected[6]
        calls.append("backward")
        raise Prepared

    monkeypatch.setattr(integration, "_allocate_recompute_caches", rebuild)
    monkeypatch.setattr(integration, "_reap_backward_workspaces", lambda: None)
    monkeypatch.setattr(integration, "_device_block_num", lambda _: 1)
    monkeypatch.setattr(
        integration._BACKWARD_SCRATCH, "acquire", lambda *a: nullcontext(None)
    )
    monkeypatch.setattr(integration, "_prepare_stage_two", prepare)
    if mode is not False and not supplied:
        with pytest.raises(ValueError, match="requires all saved"):
            integration.flash_kda_grad_npu(**args)
        assert not calls
    else:
        with pytest.raises(Prepared):
            integration.flash_kda_grad_npu(**args)
        assert calls == (["allocate", "backward"] if mode is False else ["backward"])


@pytest.mark.parametrize(
    "missing,match",
    [
        ("Aqk", "saved"),
        ("Akk", "saved"),
        ("q_rtsd", "q_rtsd"),
        ("beta_sigmoid", "activated beta"),
    ],
)
@pytest.mark.parametrize("omitted", (False, True))
def test_recompute_keeps_nonrecomputable_inputs_required(
    monkeypatch, host_sequence_metadata, missing, match, omitted
):
    args = _make_host_abi_inputs(gate_dtype=torch.float32)
    args["metadata"] = host_sequence_metadata.flash_kda_grad_metadata(
        **_metadata_arguments(
            args["q"], args["v"], "BNSD", seqused=torch.tensor([2], dtype=torch.int32)
        )
    )
    for name in _RECOMPUTE_CACHE_NAMES:
        args.pop(name)
    if omitted:
        args.pop(missing)
    else:
        args[missing] = None
    monkeypatch.setattr(
        integration,
        "_allocate_recompute_caches",
        lambda *a, **kw: pytest.fail("invalid input launched recomputation"),
    )
    required_argument = omitted and missing in ("Aqk", "Akk")
    with pytest.raises(
        TypeError if required_argument else ValueError,
        match=missing if required_argument else match,
    ):
        integration.flash_kda_grad_npu(**args, disable_recompute=False)


@pytest.mark.parametrize("case", ("dtype", "shape", "rank", "not_tensor", "stride"))
def test_recompute_rejects_invalid_initial_state(
    monkeypatch, host_sequence_metadata, case
):
    args = _make_forward_layout_inputs(host_sequence_metadata, "BNSD", (0, 65))
    state = torch.zeros_like(args["dht"])
    args["initial_state"] = {
        "dtype": lambda: state.bfloat16(),
        "shape": lambda: state[:1],
        "rank": lambda: state.unsqueeze(0),
        "not_tensor": lambda: 0,
        "stride": lambda: torch.empty((*state.shape[:-1], 256))[..., ::2],
    }[case]()
    monkeypatch.setattr(
        integration,
        "_allocate_recompute_caches",
        lambda *a: pytest.fail("invalid state reached allocation"),
    )
    with pytest.raises(ValueError, match="initial_state must be FP32"):
        integration.flash_kda_grad_npu(**args, disable_recompute=False)


def test_recompute_is_disabled_by_default_but_not_a_backward_kernel_argument():
    signature = inspect.signature(integration.flash_kda_grad_npu)
    assert signature.parameters["disable_recompute"].default is True
    assert (
        inspect.signature(integration.flash_kda_grad_from_forward)
        .parameters["disable_recompute"]
        .default
        is True
    )
    for name in (*_RECOMPUTE_CACHE_NAMES, "chunk_states", "initial_state"):
        assert signature.parameters[name].default is None
    assert (
        "disable_recompute"
        not in inspect.signature(integration.FlashKDAGrad.run).parameters
    )
    specs, scalars = integration.backward_dynamic_compile_request()
    assert len(specs) == 89 and len(specs) + len(scalars) == 106


@pytest.mark.parametrize("failure", (None, "launch", "record", "synchronize"))
def test_recompute_launch_retains_owners_and_program_until_completion(
    monkeypatch, failure
):
    programs, leases, queued, compiled, closed = {}, {}, [], [], []
    saved_programs = {"saved-cache program": object()}
    sync_fails = failure == "synchronize"

    class Stream:
        def synchronize(self):
            if sync_fails:
                raise RuntimeError("synchronize failed")

        def query(self):
            return False

    stream = Stream()

    class Event:
        def record(self, actual_stream):
            assert actual_stream is stream
            if failure == "record":
                raise RuntimeError("record failed")

        def query(self):
            return False

        def synchronize(self):
            pass

    class Tensor:
        dtype = torch.bfloat16

        def record_stream(self, actual_stream):
            assert actual_stream is stream

    class Program:
        def __call__(self, *args):
            assert integration._BACKWARD_PROGRAM_LOCK._is_owned()
            if failure in ("launch", "synchronize"):
                raise RuntimeError("launch failed")

        def close(self):
            closed.append(self)

    def compile_program(*args):
        program = Program()
        compiled.append(program)
        return program

    def queue(lease):
        assert integration._BACKWARD_PROGRAM_LOCK._is_owned()
        queued.append(lease)

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            device=lambda _: nullcontext(),
            current_stream=lambda _: stream,
            Event=Event,
        ),
        raising=False,
    )
    monkeypatch.setattr(integration, "_RECOMPUTE_PROGRAMS", programs)
    monkeypatch.setattr(integration.cannbotdsl, "compile", compile_program)
    monkeypatch.setattr(integration, "_BACKWARD_PROGRAMS", saved_programs)
    monkeypatch.setattr(integration, "_BACKWARD_WORKSPACE_LEASES", leases)
    monkeypatch.setattr(integration, "_device_block_num", lambda _: 2)
    monkeypatch.setattr(integration, "_queue_backward_workspace_retirement", queue)
    arguments = tuple(Tensor() for _ in range(91))
    config = (0.125, 0, 1.0, 1, 0, 0, 0, 0, -5.0, False, 1, False, True, False, False)
    if failure is None:
        integration._launch_recompute_program(
            "npu:3", arguments, config, 0.125, 2, True
        )
        integration._launch_recompute_program(
            "npu:3", arguments, config, 0.125, 2, True
        )
        assert len(leases) == len(queued) == 2
        assert all(
            lease.arguments is arguments and lease.event_recorded for lease in queued
        )
        assert len(compiled) == 1
        fp32_arguments = list(arguments)
        gate = Tensor()
        gate.dtype = torch.float32
        fp32_arguments[integration._BACKWARD_RAW_GATE_INDEX] = gate
        fp32_arguments = tuple(fp32_arguments)
        for _ in range(2):
            integration._launch_recompute_program(
                "npu:3", fp32_arguments, config, 0.125, 2, True
            )
        assert len(compiled) == 2
        assert all(lease.arguments is fp32_arguments for lease in queued[2:])
        # Fused FP32 beta caches can have either original beta/output dtype.
        fused_config = (*config[:9], True, *config[10:])
        integration._launch_recompute_program(
            "npu:3", fp32_arguments, fused_config, 0.125, 2, True
        )
        assert len(compiled) == 3
        beta_arguments = list(fp32_arguments)
        beta_gradient = Tensor()
        beta_gradient.dtype = torch.float32
        beta_arguments[integration._BACKWARD_BETA_GRAD_INDEX] = beta_gradient
        beta_arguments = tuple(beta_arguments)
        for _ in range(2):
            integration._launch_recompute_program(
                "npu:3", beta_arguments, fused_config, 0.125, 2, True
            )
        assert len(compiled) == 4
        assert all(lease.arguments is beta_arguments for lease in queued[-2:])
    else:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            integration._launch_recompute_program(
                "npu:3", arguments, config, 0.125, 2, True
            )
        assert not queued
        if failure == "synchronize":
            lease = next(iter(leases.values()))
            assert lease.arguments is arguments and not lease.event_recorded
            with pytest.raises(RuntimeError, match="synchronize failed"):
                integration._clear_recompute_program_cache()
            assert programs and not closed and lease.arguments is arguments
            sync_fails = False
        else:
            assert not leases
    assert len(compiled) == (4 if failure is None else 1)
    assert list(saved_programs) == ["saved-cache program"]
    integration._clear_recompute_program_cache()
    assert not programs and not leases
    assert all(lease.arguments is None for lease in queued)
    assert closed == compiled


@pytest.fixture
def recompute_vector_cpu(monkeypatch):
    """Execute the actual VF method bodies with CPU register operations.

    Device compilation is tested separately. This checks arithmetic, rounding,
    in-place triangular ordering and scratch reuse against independent formulas.
    """

    def load(t, offset):
        return t.flatten()[int(offset) : int(offset) + 64].clone()

    def cast(x, dtype, **kw):
        return x.to(
            torch.bfloat16 if dtype == integration.dtypes.bfloat16 else torch.float32
        )

    def store(t, offset, x, mask=64, **kw):
        t.view(-1)[int(offset) : int(offset) + int(mask)].copy_(x[: int(mask)])

    operations = dict(
        vf=lambda **kw: nullcontext(),
        const_expr=bool,
        create_mask=lambda **kw: 64,
        update_mask=lambda n, **kw: (int(n), 0),
        vload=load,
        vload_unpack=lambda t, off, **kw: load(t, off),
        vload_broadcast=lambda t, off: t.flatten()[int(off)].expand(64).clone(),
        vcast=cast,
        vdups=lambda x, dtype, **kw: torch.full((64,), float(x), dtype=torch.float32),
        vadd=lambda x, y, **kw: x + y,
        vadds=lambda x, y, **kw: x + y,
        vsub=lambda x, y, **kw: x - y,
        vmul=lambda x, y, **kw: x * y,
        vmuls=lambda x, y, **kw: x * y,
        vdiv=lambda x, y, **kw: x / y,
        vmaxs=lambda x, y, **kw: torch.clamp_min(x, y),
        vmins=lambda x, y, **kw: torch.clamp_max(x, y),
        vlts=lambda x, y, **kw: x < y,
        vselect=lambda x, y, cond_mask: torch.where(
            torch.arange(x.numel()) < cond_mask
            if isinstance(cond_mask, int)
            else cond_mask,
            x,
            y,
        ),
        vexp=lambda x, **kw: x.exp(),
        vlog=lambda x, **kw: x.log(),
        vreduce_sum=lambda x, **kw: x.sum().expand(64),
        vstore=store,
        vstore_pack=store,
        vstore_first=lambda t, off, x: t.view(-1)[int(off)].copy_(x[0]),
        vmem_bar=lambda *a: None,
    )
    for name, implementation in operations.items():
        monkeypatch.setattr(integration, name, implementation)
    recompute = inspect.getmodule(integration._recompute_implementation().fn)
    for name, implementation in operations.items():
        if hasattr(recompute, name):
            monkeypatch.setattr(recompute, name, implementation)
    monkeypatch.setattr(recompute, "dsl_range", range)
    monkeypatch.setattr(recompute, "vabs", lambda x, **kw: x.abs())
    monkeypatch.setattr(recompute, "vexp_sub", lambda x, y, **kw: (x - y).exp())
    monkeypatch.setattr(recompute, "vselect_raw", operations["vselect"])
    # CPU arithmetic emulates the fused FP32 state update with one rounding.
    monkeypatch.setattr(
        recompute,
        "vmadd",
        lambda x, y, z, **kw: (x.double() * y.double() + z.double()).float(),
    )
    monkeypatch.setattr(recompute, "cast_and_pack_bf16", lambda x, mask: x.bfloat16())
    monkeypatch.setattr(
        recompute,
        "vstore_strided",
        lambda dst, offset, x, mask, **kw: dst.values.view(-1, 64)[
            int(offset) // 16
        ].copy_(x),
    )
    return recompute


def _cpu_dsl_input(tensor):
    # VF tests use real Torch storage and the DSL's compile-time dtype tag.
    return SimpleNamespace(
        dtype=integration.dtypes.float32
        if tensor.dtype == torch.float32
        else integration.dtypes.bfloat16,
        flatten=tensor.flatten,
        view=tensor.view,
    )


def _execute_recompute_preprocess(
    module, q, k, v, g, beta, alpha, bias, bound, fused_gate, beta_cached
):
    rows = q.shape[0]
    operands = torch.empty((8, 64, 128), dtype=torch.bfloat16)
    gamma = torch.empty((64, 128))
    qg, kg = torch.empty_like(q), torch.empty_like(k)
    for part in range(2):
        half = slice(part * 64, (part + 1) * 64)

        def padded(x):
            result = x.new_zeros((64, *x.shape[1:]))
            result[:rows] = x
            return result

        prep = SimpleNamespace(
            ub_g_raw=_cpu_dsl_input(padded(g[:, half])),
            ub_beta_raw=_cpu_dsl_input(padded(beta)),
            ub_beta=torch.empty(64),
            ub_neg_exp_a=-torch.tensor([alpha]).exp(),
            ub_dt_bias=torch.zeros(64) if bias is None else bias[half],
            ub_g_cumsum=torch.empty(64, 64),
            ub_q_norm=padded(q[:, half]),
            ub_k_norm=padded(k[:, half]),
            ub_v_raw=padded(v[:, half]),
            ub_recurrent_operands_nz=SimpleNamespace(
                values=torch.empty(8, 64, 64, dtype=torch.bfloat16),
                physical_stride=(1024,),
            ),
            saved_qg_nd=torch.empty(64, 64, dtype=torch.bfloat16),
            saved_kg_nd=torch.empty(64, 64, dtype=torch.bfloat16),
            has_tail=True,
        )
        module._Preprocess._vf_activate_gate_beta.fn(
            prep,
            0.0 if bound is None else bound,
            rows,
            0 if not fused_gate else 1 if bound is None else 2,
            beta_cached,
            bias is not None,
        )
        module._Preprocess._vf_pack_recurrent_operands.fn(
            prep, 1.0, torch.empty(64), True
        )
        operands[:, :, half] = prep.ub_recurrent_operands_nz.values
        gamma[:, half] = prep.ub_g_cumsum
        qg[:, half] = prep.saved_qg_nd[:rows]
        kg[:, half] = prep.saved_kg_nd[:rows]
    return operands, gamma, qg, kg


@pytest.mark.parametrize("rows", (1, 17, 64))
@pytest.mark.parametrize(
    "gate,bound,bias_on",
    [(False, -5.0, False), (True, -0.7, True), (True, None, False)],
)
@pytest.mark.parametrize(
    "beta_dtype,beta_cached",
    [(torch.bfloat16, False), (torch.float32, False), (torch.float32, True)],
)
@pytest.mark.parametrize("gate_dtype", (torch.bfloat16, torch.float32))
def test_recompute_vector_math(
    recompute_vector_cpu,
    rows,
    gate,
    bound,
    bias_on,
    beta_dtype,
    beta_cached,
    gate_dtype,
):
    torch.manual_seed(77)
    q = F.normalize(torch.randn(rows, 128), dim=-1).bfloat16()
    k = F.normalize(torch.randn(rows, 128), dim=-1).bfloat16()
    v = torch.randn(rows, 128).bfloat16()
    g = (-torch.rand(rows, 128) * 0.1).to(gate_dtype)
    beta = torch.rand(rows).to(beta_dtype)
    bias = torch.randn(128) * 0.1 if bias_on else None
    alpha = -2.0
    # Retain the public input precision until the cache storage boundary.
    step = g.float()
    if gate:
        x = step + (0 if bias is None else bias)
        step = (
            -torch.tensor(alpha).exp() * F.softplus(x)
            if bound is None
            else bound * torch.sigmoid(torch.tensor(alpha).exp() * x)
        )
    gamma = step.cumsum(0)
    effective_beta = beta.float()
    expected = torch.stack(
        (
            gamma.exp() * q.float(),
            (gamma[-1:] - gamma).exp() * k.float(),
            (k.float() * gamma.exp()) * effective_beta[:, None],
            v.float() * effective_beta[:, None],
        )
    )
    high = expected.bfloat16()
    residual = (expected[2:] - high[2:].float()).bfloat16()
    remainder = (expected[2:] - high[2:].float() - residual.float()).bfloat16()
    expected = torch.cat((high, residual, remainder), dim=0)
    operands, actual_gamma, qg, kg = _execute_recompute_preprocess(
        recompute_vector_cpu,
        q,
        k,
        v,
        g,
        beta,
        alpha,
        bias,
        bound,
        gate,
        beta_cached,
    )
    torch.testing.assert_close(actual_gamma[:rows], gamma, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        actual_gamma[rows:],
        actual_gamma[rows - 1 : rows].expand(64 - rows, -1),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(operands[:, :rows], expected, atol=1e-4, rtol=0.008)
    torch.testing.assert_close(qg, operands[0, :rows], atol=0, rtol=0)
    torch.testing.assert_close(kg, operands[1, :rows], atol=0, rtol=0)
    assert torch.count_nonzero(operands[:, rows:]) == 0
    # The three Cube products recover FP32 beta/decay inputs, instead of
    # introducing another BF16 rounding before the W/U cache is stored.
    torch.testing.assert_close(
        operands[2:4, :rows].float()
        + operands[4:6, :rows].float()
        + operands[6:8, :rows].float(),
        torch.stack(
            (
                (k.float() * gamma.exp()) * effective_beta[:, None],
                v.float() * effective_beta[:, None],
            )
        ),
        atol=2e-6,
        rtol=3e-5,
    )


@pytest.mark.parametrize("zero_initial", (False, True))
def test_recompute_state_recurrence_matches_forward(recompute_vector_cpu, zero_initial):
    torch.manual_seed(983)
    module = recompute_vector_cpu
    state = torch.zeros(128, 128) if zero_initial else torch.randn(128, 128) * 0.02
    for _ in range(3):
        decay = torch.rand(128)
        delta = torch.randn_like(state) * 0.01
        expected = (state.double() * decay.double() + delta.double()).float()
        following = torch.empty_like(state)
        snapshot = torch.empty_like(state, dtype=torch.bfloat16)
        for part in range(2):
            half = slice(part * 64, (part + 1) * 64)
            vector = SimpleNamespace(
                gamma_ub=decay[half],
                ub_state_fp32=state[:, half].clone(),
                ub_state_delta=SimpleNamespace(
                    next_read=lambda: delta[:, half].contiguous()
                ),
                ub_state_nz=SimpleNamespace(
                    values=torch.empty(128, 64, dtype=torch.bfloat16),
                    physical_stride=(1024,),
                ),
            )
            module._State.update_recurrent_state.fn(vector)
            following[:, half] = vector.ub_state_fp32
            snapshot[:, half] = vector.ub_state_nz.values
        torch.testing.assert_close(following, expected, atol=0, rtol=0)
        torch.testing.assert_close(snapshot, expected.bfloat16(), atol=0, rtol=0)
        state = following


def test_fused_phases_publish_caches_for_the_selected_owners():
    source = ast.parse(inspect.getsource(integration))
    kernel = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "_RecomputeBackwardKernel"
    )
    body = next(node for node in kernel.body if isinstance(node, ast.FunctionDef))
    scope = body.body[0]
    assert isinstance(scope, ast.With)
    assert ast.unparse(scope.items[0].context_expr) == "_SharedCubeScope()"
    calls = [
        node.value
        for node in scope.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert [ast.unparse(call.func) for call in calls] == [
        "_initialize_backward_gradients",
        "_recompute_tracks",
        "_sync_cube_vector_handoff",
        "_backward_template(staged)",
    ]
    assert calls[-1].args[-1].value is True
    assert calls[1].args[-1].value is True  # common chunk/value schedule
    assert ast.unparse(calls[-2]) == "_sync_cube_vector_handoff(2, grid=True)"


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize("gate_fp32", (False, True))
@pytest.mark.parametrize("beta_fp32", (False, True))
def test_recompute_compiles_to_ascendc(monkeypatch, beta_fp32, gate_fp32):
    monkeypatch.setenv("CANNBOTDSL_PIPE_STAGE", "translate")
    specs, scalars = integration._recompute_dynamic_compile_request(
        beta_fp32, gate_fp32=gate_fp32
    )
    program = integration.cannbotdsl.compile(
        integration._RecomputeBackward().run, *specs, *scalars
    )
    try:
        assert (
            "stage_one_beta_fp32"
            not in program._parameter_lowering.signature.parameters
        )
    finally:
        program.close()


@pytest.mark.npu
@pytest.mark.parametrize("layout", ("BNSD", "BSND", "TND"))
@pytest.mark.parametrize("bound", (-0.7, None))
@pytest.mark.parametrize(
    "g_dtype,beta_dtype",
    [
        (g, b)
        for g in (torch.bfloat16, torch.float32)
        for b in (torch.bfloat16, torch.float32)
    ],
)
@pytest.mark.parametrize("fused_beta", (False, True))
@pytest.mark.parametrize("head_mode", ("grouped", "balanced"))
def test_recompute_backward_matches_saved_cache_on_npu(
    layout, bound, g_dtype, beta_dtype, fused_beta, head_mode, monkeypatch
):
    pytest.importorskip("torch_npu")
    torch.manual_seed(91)
    lengths, heads, nq, width = (0, 17, 65), 2, 1, 128
    if head_mode == "balanced":
        # One head per active core gives every persistent owner the same work.
        # Grouped heads take the staged path through the public auto selector.
        heads = nq = integration._device_block_num()
    rebalanced = head_mode.startswith("rebalanced")
    if rebalanced:
        lengths = (192, 64) * 4 if "full" in head_mode else (129, 17, 0, 64) * 2
        nq, heads = 1, 2 if head_mode.endswith("gqa") else 1
    if head_mode.startswith("staged_bn16"):
        lengths, heads, nq = (129,), 16, 16
    if head_mode == "staged_unsplit":
        lengths, heads, nq = (17, 129), 16, 16
    expected_schedule = (
        "persistent" if head_mode == "balanced" or rebalanced else "staged"
    )
    batch, storage = len(lengths), max(lengths)
    tokens = sum(lengths)
    q = F.normalize(torch.randn(tokens, nq, width), dim=-1).bfloat16().npu()
    k = F.normalize(torch.randn(tokens, nq, width), dim=-1).bfloat16().npu()
    v = (torch.randn(tokens, heads, width) * 0.2).bfloat16().npu()
    g = (torch.randn_like(v.float()) * 0.1).to(g_dtype)
    beta = torch.randn(tokens, heads).to(beta_dtype).npu()
    if not fused_beta:
        beta = beta.sigmoid()
    h0 = (torch.randn(batch, heads, width, width) * 0.01).npu()
    if head_mode.endswith("_zero"):
        h0.zero_()
    initial_state = None if head_mode.endswith("_zero") else h0
    a_log, bias = (
        torch.full((heads,), -2.0).npu(),
        torch.full((heads, width), -4.0).npu(),
    )
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    attrs = dict(
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=fused_beta,
        scale=width**-0.5,
        lower_bound=bound,
    )
    # The public recompute contract follows the STC reference: keep gate,
    # beta products and recurrent state in FP32, rounding the saved cache
    # tensors once. Real forward's additional BF16 operands are not this oracle.
    reference = flash_kda_fwd(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        g.cpu(),
        beta.cpu(),
        h0.cpu(),
        a_log.cpu(),
        bias.cpu(),
        cu_seqlens=offsets,
        layout="TND",
        **attrs,
    )
    cache = SimpleNamespace(
        **{
            name: None if value is None else value.npu()
            for name, value in vars(reference).items()
        }
    )

    def physical(x):
        if layout == "TND":
            return x
        out = x.new_zeros((batch, storage, *x.shape[1:]))
        start = 0
        for b, length in enumerate(lengths):
            out[b, :length] = x[start : start + length]
            start += length
        return out.transpose(1, 2).contiguous() if layout == "BNSD" else out

    q, k, v, g, beta = (physical(x) for x in (q, k, v, g, beta))
    args = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        do=torch.randn_like(v),
        dht=torch.randn_like(h0),
        chunk_states=cache.chunk_states,
        Aqk=cache.Aqk,
        Akk=cache.Akk,
        A_log=a_log,
        dt_bias=bias,
        beta_sigmoid=cache.beta_sigmoid,
        layout=layout,
        **attrs,
    )
    args.update(
        cu_seqlens=offsets.npu() if layout == "TND" else None,
        seqused=None
        if layout == "TND"
        else torch.tensor(lengths, dtype=torch.int32).npu(),
    )
    args["metadata"] = _metadata(q, v, layout, offsets)
    assert (
        integration._select_backward_template(
            q, v, layout, args["metadata"], integration._device_block_num(q)
        )
        == expected_schedule
    )
    saved = _reference_backward(
        **args,
        **{name: getattr(cache, name) for name in _RECOMPUTE_CACHE_NAMES},
        disable_recompute=True,
    )
    assert saved[4].dtype == g_dtype and saved[5].dtype == beta_dtype
    # Snapshot each call before the next launch and compare both results on CPU.
    saved = tuple(_precision_to_cpu(t).detach() for t in saved)
    rebuilt_caches = []
    allocate = integration._allocate_recompute_caches

    def capture_caches(*args, **kwargs):
        caches = allocate(*args, **kwargs)
        # Detect reads of unwritten tail/empty-sequence storage deterministically.
        for tensor in caches:
            tensor.fill_(float("nan"))
        rebuilt_caches.append(caches)
        return caches

    monkeypatch.setattr(integration, "_allocate_recompute_caches", capture_caches)
    if rebalanced:
        cores = integration._device_block_num(q)
        plan = integration._persistent_owner_plan(q, v, layout, args["metadata"], cores)
        assert plan.words and plan.balanced
        assert max(plan.loads) < max(plan.modulo_loads)
        assert sorted(plan.words[cores + 1 :]) == list(range(batch * nq))
        # Independent schedule comparison on exactly the same inputs/caches.
        with monkeypatch.context() as patch:
            patch.setattr(
                integration,
                "_owner_schedule_for_launch",
                lambda q, v, layout, metadata, cores, staged: metadata[:1],
            )
            modulo = _reference_backward(
                **args,
                **{name: getattr(cache, name) for name in _RECOMPUTE_CACHE_NAMES},
                disable_recompute=True,
            )
            modulo = tuple(_precision_to_cpu(t).detach() for t in modulo)
        for name, got, want in zip(
            ("dq", "dk", "dv", "dh0", "dg", "dbeta", "dA", "dBias"), saved, modulo
        ):
            _assert_precision(name, got, want)
        # Reuse the immutable uploaded table from a different consumer stream.
        previous = torch.npu.current_stream(q.device)
        consumer = torch.npu.Stream(device=q.device)
        consumer.wait_stream(previous)
        with torch.npu.stream(consumer):
            actual = _reference_backward(
                **{**args, "chunk_states": None},
                initial_state=initial_state,
                disable_recompute=False,
            )
        previous.wait_stream(consumer)
    else:
        actual = _reference_backward(
            **{**args, "chunk_states": None},
            initial_state=initial_state,
            disable_recompute=False,
        )
    assert actual[4].dtype == g_dtype and actual[5].dtype == beta_dtype
    actual = tuple(_precision_to_cpu(t).detach() for t in actual)
    expected_caches = _reference_forward_arguments(
        dict(
            q=q,
            v=v,
            layout=layout,
            chunk_states=cache.chunk_states,
            **{name: getattr(cache, name) for name in _RECOMPUTE_CACHE_NAMES},
        ),
        lengths,
    )
    for name, rebuilt in zip(
        (*_RECOMPUTE_CACHE_NAMES, "chunk_states"), rebuilt_caches[-1]
    ):
        expected = expected_caches[name]
        if layout == "BSND" and name != "chunk_states":
            rebuilt = rebuilt.transpose(1, 2)
        for sequence, length in enumerate(lengths):
            if layout != "TND" and name != "chunk_states":
                assert torch.count_nonzero(rebuilt[sequence, :, length:].cpu()) == 0, (
                    name
                )
            if name == "chunk_states":
                count = (length + 63) // 64
                if layout == "TND":
                    first = int(offsets[sequence]) // 64 + sequence
                    got, want = (
                        rebuilt[first : first + count],
                        expected[first : first + count],
                    )
                else:
                    got, want = (
                        rebuilt[sequence, :, :count],
                        expected[sequence, :, :count],
                    )
            elif layout == "TND":
                first = int(offsets[sequence])
                got, want = (
                    rebuilt[first : first + length],
                    expected[first : first + length],
                )
            else:
                got, want = (
                    rebuilt[sequence, :, :length],
                    expected[sequence, :, :length],
                )
            got, want = got.cpu().float(), want.cpu().float()
            # Cache rounding can cross BF16 midpoints and cancellation can
            # leave near-zero state entries. Use the established precision
            # helper, with a tighter aggregate bound for reconstructed caches.
            _assert_precision(name, got, want, max_relative_l2=5e-4, min_atol=2e-5)
    for name, got, want in zip(
        ("dq", "dk", "dv", "dh0", "dg", "dbeta", "dA", "dBias"), actual, saved
    ):
        _assert_precision(name, got, want)


@pytest.mark.npu
@pytest.mark.parametrize("layout", ("BNSD", "BSND", "TND"))
@pytest.mark.parametrize(
    "head_mode", ("staged_bn16", "staged_bn16_zero", "staged_unsplit")
)
def test_staged_recompute_value_shards_match_forward_npu(
    monkeypatch, layout, head_mode
):
    test_recompute_backward_matches_saved_cache_on_npu(
        layout, -0.7, torch.float32, torch.bfloat16, True, head_mode, monkeypatch
    )


@pytest.mark.npu
@pytest.mark.parametrize("layout", ("BNSD", "BSND", "TND"))
@pytest.mark.parametrize("grouped", (False, True))
@pytest.mark.parametrize(
    "kind,bound", (("full", -0.7), ("mixed", -0.7), ("mixed", None))
)
def test_rebalanced_persistent_saved_recompute_and_modulo_npu(
    monkeypatch, layout, grouped, kind, bound
):
    pytest.importorskip("torch_npu")
    monkeypatch.setattr(integration, "_device_block_num", lambda ref=None: 4)
    head_mode = "rebalanced_" + kind + ("_gqa" if grouped else "")
    test_recompute_backward_matches_saved_cache_on_npu(
        layout, bound, torch.float32, torch.float32, True, head_mode, monkeypatch
    )


@pytest.mark.npu
def test_persistent_owner_schedule_refreshes_after_metadata_copy_npu():
    pytest.importorskip("torch_npu")
    q = torch.empty((8, 1, 192, 128), dtype=torch.bfloat16, device="npu")
    v = torch.empty((8, 2, 192, 128), dtype=torch.bfloat16, device=q.device)

    def metadata_for(lengths):
        offsets = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
        )
        return _metadata(q, v, "BNSD", offsets)

    metadata = metadata_for((192, 64) * 4)
    original = metadata.clone()
    plan = integration._persistent_owner_plan(q, v, "BNSD", metadata, 4)
    assert plan.words
    first = integration._owner_schedule_for_launch(q, v, "BNSD", metadata, 4, False)
    assert first.cpu().tolist() == list(plan.words)
    assert (
        integration._owner_schedule_for_launch(q, v, "BNSD", metadata, 4, False)
        is first
    )

    # Same pointer and shape, different contents: the cached plan must expire.
    metadata.copy_(metadata_for((128,) * 8))
    balanced = integration._persistent_owner_plan(q, v, "BNSD", metadata, 4)
    assert balanced is not plan and not balanced.words
    modulo = integration._owner_schedule_for_launch(q, v, "BNSD", metadata, 4, False)
    assert modulo.numel() == 1 and modulo.data_ptr() == metadata.data_ptr()
    metadata.copy_(original)
    restored = integration._owner_schedule_for_launch(q, v, "BNSD", metadata, 4, False)
    assert restored.cpu().tolist() == list(plan.words)


@pytest.mark.parametrize("layout", ("BNSD", "BSND", "TND"))
def test_recompute_host_views_and_cache_storage(monkeypatch, layout):
    shape = (
        (82, 2, 128)
        if layout == "TND"
        else (3, 2, 65, 128)
        if layout == "BNSD"
        else (3, 65, 2, 128)
    )
    values = torch.zeros(shape, dtype=torch.bfloat16)
    dht = torch.zeros((3, 2, 128, 128))
    monkeypatch.setattr(
        integration,
        "_launch_recompute_program",
        lambda *a: pytest.fail("allocation launched a kernel"),
    )
    caches = integration._allocate_recompute_caches(values, dht, layout)
    assert len(caches) == 7
    assert all(cache.shape == values.shape for cache in caches[:6])
    assert all(cache.dtype == torch.bfloat16 for cache in caches[:5])
    assert caches[5].dtype == caches[6].dtype == torch.float32
    assert caches[6].shape == (
        (5, 2, 128, 128) if layout == "TND" else (3, 2, 2, 128, 128)
    )


@pytest.mark.parametrize("layout", ("BNSD", "BSND", "TND"))
def test_forward_adapter_accepts_absent_recomputable_slots(monkeypatch, layout):
    rows = torch.ones(1, 1, 128, dtype=torch.bfloat16)
    q = rows if layout == "TND" else rows.unsqueeze(0)
    matrix = (
        torch.ones((1, 1, 64, 64)) if layout == "TND" else torch.ones((1, 1, 1, 64, 64))
    )
    states = (
        torch.zeros((1, 1, 128, 128))
        if layout == "TND"
        else torch.zeros((1, 1, 1, 128, 128))
    )
    saved = (
        None,
        None,
        q,
        q,
        q[..., 0].float(),
        q[..., 0].float(),
        q[..., 0].float(),
        matrix,
        matrix,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    monkeypatch.setattr(integration, "wait_metadata_ready", lambda _: None)
    monkeypatch.setattr(
        integration._metadata,
        "metadata_host_info",
        lambda _: type("Info", (), {"sequences": ((0, 1, 1, 0),)})(),
    )
    signature = inspect.signature(integration.flash_kda_grad_npu)
    monkeypatch.setattr(
        integration,
        "flash_kda_grad_npu",
        lambda *a, **kw: signature.bind(*a, **kw).arguments,
    )
    values = integration.flash_kda_grad_from_forward(
        q,
        q,
        q,
        q,
        q[..., 0],
        q,
        states,
        saved,
        None,
        None,
        object(),
        layout=layout,
        use_qk_l2norm_in_kernel=True,
        disable_recompute=False,
    )
    assert values["disable_recompute"] is False
    assert all(
        values[name] is None for name in (*_RECOMPUTE_CACHE_NAMES, "chunk_states")
    )


@pytest.mark.parametrize("layout", ("BNSD", "BSND", "TND"))
@pytest.mark.parametrize("disabled", (False, True))
@pytest.mark.parametrize("fused_beta", (False, True))
@pytest.mark.parametrize(
    "g_dtype,beta_dtype",
    [
        (g, b)
        for g in (torch.bfloat16, torch.float32)
        for b in (torch.bfloat16, torch.float32)
    ],
)
def test_public_gate_beta_storage_and_return_dtypes(
    monkeypatch,
    host_sequence_metadata,
    layout,
    disabled,
    fused_beta,
    g_dtype,
    beta_dtype,
):
    args = _make_host_abi_inputs(torch.float32, layout=layout)
    args["g"] = torch.full_like(args["g"], 0.123456, dtype=g_dtype)
    args["beta"] = torch.full_like(args["beta"], 0.654321, dtype=beta_dtype)
    args["use_beta_sigmoid_in_kernel"] = fused_beta
    options = (
        dict(cu_seqlens=torch.tensor([0, 2], dtype=torch.int32))
        if layout == "TND"
        else dict(seqused=torch.tensor([2], dtype=torch.int32))
    )
    args["metadata"] = host_sequence_metadata.flash_kda_grad_metadata(
        **_metadata_arguments(args["q"], args["v"], layout, **options)
    )

    kernel_gradients = tuple(
        torch.full_like(args[name], float("nan")) for name in ("g", "beta")
    )
    expected_gradients = tuple(
        (
            torch.arange(args[name].numel(), dtype=torch.float32).reshape(
                args[name].shape
            )
            * 0.03127
            - 0.50013
        ).to(args[name].dtype)
        for name in ("g", "beta")
    )
    launched = []

    def prepare(**values):
        assert values["raw_gate"] is args["g"]
        assert values["beta_dtype"] == beta_dtype
        expected = args["beta_sigmoid"] if fused_beta else args["beta"]
        assert values["activated_beta"].dtype == expected.dtype
        assert values["activated_beta"].data_ptr() == expected.data_ptr()
        return (
            (
                args["q"],
                args["k"],
                args["v"],
                *kernel_gradients,
                args["A_log"],
                args["dt_bias"],
            ),
            (),
            (),
            (),
        )

    def launch(mode):
        launched.append(mode)
        for output, expected in zip(kernel_gradients, expected_gradients):
            output.copy_(expected)

    monkeypatch.setattr(integration, "_reap_backward_workspaces", lambda: None)
    monkeypatch.setattr(integration, "_device_block_num", lambda _: 1)
    monkeypatch.setattr(
        integration._BACKWARD_SCRATCH, "acquire", lambda *a: nullcontext(None)
    )
    monkeypatch.setattr(
        integration,
        "_prepare_stage_two",
        lambda **kw: (SimpleNamespace(dh0=args["dht"]), (), ()),
    )
    monkeypatch.setattr(integration, "_prepare_stage_one", prepare)
    monkeypatch.setattr(
        integration, "_launch_backward_program", lambda *a: launch("saved")
    )
    monkeypatch.setattr(
        integration, "_launch_recompute_program", lambda *a: launch("recompute")
    )
    result = integration.flash_kda_grad_npu(**args, disable_recompute=disabled)
    assert launched == (["saved"] if disabled else ["recompute"])
    for actual, kernel, expected, source in zip(
        result[4:6], kernel_gradients, expected_gradients, (args["g"], args["beta"])
    ):
        assert actual.dtype == source.dtype
        assert actual.shape == source.shape and actual.device == source.device
        torch.testing.assert_close(actual, expected.to(source.dtype), rtol=0, atol=0)
        assert actual is kernel


@pytest.mark.parametrize("name", ("g", "beta"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.float64, torch.int32))
def test_public_gate_beta_reject_unsupported_storage(monkeypatch, name, dtype):
    args = _make_host_abi_inputs(torch.float32)
    args[name] = args[name].to(dtype)
    monkeypatch.setattr(
        torch.Tensor,
        "device",
        property(lambda tensor: SimpleNamespace(type="npu", index=0)),
    )
    with pytest.raises(ValueError, match="g and beta must be BF16 or FP32"):
        integration.flash_kda_grad_npu(**args)


@pytest.mark.parametrize(
    "g_dtype,beta_dtype",
    [
        (g, b)
        for g in (torch.bfloat16, torch.float32)
        for b in (torch.bfloat16, torch.float32)
    ],
)
@pytest.mark.parametrize("layout", ("BNSD", "BSND", "TND"))
@pytest.mark.parametrize("fused_beta", (False, True))
def test_gate_beta_dynamic_specs_preserve_input_and_gradient_dtypes(
    g_dtype, beta_dtype, layout, fused_beta
):
    concrete, config = _prepare_backward_request(
        dict(
            length=65,
            layout=layout,
            g_dtype=g_dtype,
            beta_dtype=beta_dtype,
            use_beta_sigmoid_in_kernel=fused_beta,
        )
    )
    names = list(inspect.signature(integration.FlashKDAGrad.run).parameters)[1:89]
    actual = dict(zip(names, concrete))
    assert actual["stage_one_gm_raw_gate"].dtype == (
        integration.dtypes.float32
        if g_dtype == torch.float32
        else integration.dtypes.bfloat16
    )
    assert actual["stage_one_gm_activated_beta"].dtype == (
        integration.dtypes.float32
        if fused_beta or beta_dtype == torch.float32
        else integration.dtypes.bfloat16
    )
    assert actual["stage_one_gm_public_d_gate"].dtype == (
        integration.dtypes.float32
        if g_dtype == torch.float32
        else integration.dtypes.bfloat16
    )
    assert actual["stage_one_gm_public_d_beta"].dtype == (
        integration.dtypes.float32
        if beta_dtype == torch.float32
        else integration.dtypes.bfloat16
    )
    assert (
        actual["stage_one_gm_d_beta"].dtype
        == actual["stage_one_gm_public_d_beta"].dtype
    )
    dynamic, _ = integration.backward_dynamic_compile_request(
        config[9],
        gate_fp32=g_dtype == torch.float32,
        beta_grad_fp32=beta_dtype == torch.float32,
    )
    fused, _ = integration._recompute_dynamic_compile_request(
        config[9],
        gate_fp32=g_dtype == torch.float32,
        beta_grad_fp32=beta_dtype == torch.float32,
    )
    assert [spec.dtype for spec in dynamic] == [spec.dtype for spec in concrete]
    assert fused[:89] == dynamic


@pytest.mark.parametrize("batched", (False, True))
@pytest.mark.parametrize("gate_fp32", (False, True))
@pytest.mark.parametrize("bound", (-0.7, None))
@pytest.mark.parametrize("fused_gate", (False, True))
def test_gate_backward_vf_matches_fp32_autograd(
    recompute_vector_cpu, batched, gate_fp32, bound, fused_gate
):
    # Non-BF16-representable inputs detect accidental FP32 -> BF16 truncation.
    torch.manual_seed(184)
    rows = integration.GATE_BATCH_ROWS if batched else 1
    raw = torch.randn(rows, 64).to(torch.float32 if gate_fp32 else torch.bfloat16)
    bias = torch.randn(64) * 0.03
    base, pair = torch.randn(rows, 64) * 0.1, torch.randn(rows, 64) * 0.1
    terminal = torch.randn(64) * 0.05
    sensitivity = torch.flip(torch.flip(base + pair, (0,)).cumsum(0), (0,)) + terminal
    x, a, b = (
        raw.float().requires_grad_(),
        torch.tensor(-0.37, requires_grad=True),
        bias.requires_grad_(),
    )
    if fused_gate:
        activated = (
            -a.exp() * F.softplus(x + b)
            if bound is None
            else bound * torch.sigmoid(a.exp() * (x + b))
        )
        expected_g, expected_a, expected_b = torch.autograd.grad(
            (activated * sensitivity).sum(), (x, a, b)
        )
    else:
        expected_g, expected_a, expected_b = (
            sensitivity,
            torch.tensor(0.0),
            torch.zeros(64),
        )
    vector = object.__new__(integration.StageOneVector)
    vector.fuse_base = False
    vector.gate_running = terminal.clone()
    vector.gate_bias_sum = torch.zeros(64)
    vector.gate_alog_sum = torch.zeros(8)
    # Full/tail gate epilogues reuse the FP32 raw-gate plane after reading each row.
    out = raw.detach() if gate_fp32 and batched else torch.empty_like(raw)
    method = (
        integration.StageOneVector._compute_gate_batch_constexpr
        if batched
        else integration.StageOneVector._compute_gate_row_constexpr
    )
    method.fn(
        vector,
        base,
        pair,
        _cpu_dsl_input(raw),
        bias,
        _cpu_dsl_input(out),
        float(a.detach()),
        fused_gate,
        bound is None,
        0.0 if bound is None else bound,
    )
    if gate_fp32:
        torch.testing.assert_close(out, expected_g, atol=2e-6, rtol=2e-5)
        assert not torch.equal(out, out.bfloat16().float())
    else:
        torch.testing.assert_close(
            out.float(), expected_g.bfloat16().float(), atol=1e-4, rtol=0.01
        )
    torch.testing.assert_close(vector.gate_bias_sum, expected_b, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        vector.gate_alog_sum[0], expected_a, atol=2e-6, rtol=2e-5
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("fused,factor", ((False, 1.0), (True, 1.0), (True, 0.5)))
@pytest.mark.parametrize("valid_rows", (0, 1, 17, 64))
def test_beta_gradient_vector_preserves_output_precision(
    recompute_vector_cpu, dtype, fused, factor, valid_rows
):
    torch.manual_seed(924)
    wy, pair = torch.randn(64) * 0.13, torch.randn(64) * 0.07
    beta = torch.sigmoid(torch.randn(64)) / factor
    expected = wy + pair
    if fused:
        expected *= beta * (1.0 - beta * factor)
    expected[valid_rows:] = 0
    vector = object.__new__(integration.BetaCombineVector)
    vector.beta_gradient_sum = wy.clone()
    # The FP32 path publishes its running-sum buffer directly.
    out = (
        vector.beta_gradient_sum
        if dtype == torch.float32
        else torch.empty(64, dtype=dtype)
    )
    integration.BetaCombineVector._combine_single_pair_constexpr.fn(
        vector, beta, pair, _cpu_dsl_input(out), valid_rows, fused, factor
    )
    torch.testing.assert_close(out, expected.to(dtype), rtol=0, atol=0)
    if dtype == torch.float32 and valid_rows:
        assert not torch.equal(out, out.bfloat16().float())


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("fused,factor", ((False, 1.0), (True, 1.0), (True, 0.5)))
def test_short_softplus_beta_gradient_preserves_output_precision(
    recompute_vector_cpu, dtype, fused, factor
):
    vector = object.__new__(integration.SingleChunkGateSensitivity)
    vector.beta_work = torch.full((3, 64), 0.1234567)
    vector.beta = torch.full((1, 64), 0.654321 / factor)
    out = torch.full((1, 8), float("nan"), dtype=dtype)
    integration.SingleChunkGateSensitivity._cast_beta_gradient_constexpr.fn(
        vector, _cpu_dsl_input(out), 17, fused, factor
    )
    expected = vector.beta_work[2, 17].clone()
    if fused:
        beta = vector.beta[0, 17]
        expected *= beta * (1.0 - beta * factor)
    torch.testing.assert_close(out[0, 0], expected.to(dtype), rtol=0, atol=0)
    assert torch.isnan(out[0, 1:]).all()
    if dtype == torch.float32:
        assert out[0, 0] != out[0, 0].bfloat16().float()


@pytest.mark.whitebox
@pytest.mark.usefixtures("_uncached_compiler_validation")
@pytest.mark.parametrize(
    "recomputed,beta_fp32,full,masked",
    [
        (False, False, True, False),
        (False, True, True, True),
        (True, True, True, False),
        (True, False, True, True),
    ],
)
def test_fp32_gate_translates_full_and_mixed_paths(
    monkeypatch, recomputed, beta_fp32, full, masked
):
    monkeypatch.setenv("CANNBOTDSL_PIPE_STAGE", "translate")
    request = (
        integration._recompute_dynamic_compile_request
        if recomputed
        else integration.backward_dynamic_compile_request
    )
    specs, scalars = request(beta_fp32, full, masked, True)
    program = integration.cannbotdsl.compile(
        (
            integration._RecomputeBackward()
            if recomputed
            else integration.FlashKDAGrad()
        ).run,
        *specs,
        *scalars,
    )
    program.close()


def _prepare_backward_request(profile):
    """Prepare concrete meta tensors to check the public runtime ABI."""
    p = dict(
        batch=1,
        length=8192,
        qk_heads=96,
        value_heads=96,
        layout="BNSD",
        core_num=32,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
        lower_bound=-0.06,
        has_bias=True,
        use_full_chunks=False,
    )
    p.update(profile)
    b, s, nq, nv = (p[k] for k in ("batch", "length", "qk_heads", "value_heads"))
    d, c, layout = integration.SUPPORTED_HEAD_DIM, integration.CHUNK_SIZE, p["layout"]

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")

    def physical(heads, width=d):
        shape = (
            (s, heads, width)
            if layout == "TND"
            else (b, heads, s, width)
            if layout == "BNSD"
            else (b, s, heads, width)
        )
        return empty(shape)

    q, v = physical(nq), physical(nv)
    token_shape = tuple(v.shape)
    chunk_capacity = (s + c - 1) // c + b
    matrix_shape = (chunk_capacity, nv, c, c) if layout == "TND" else (*v.shape[:-1], c)
    state_shape = (
        (chunk_capacity, nv, d, d)
        if layout == "TND"
        else (b, nv, (s + c - 1) // c, d, d)
    )
    token = empty(token_shape)
    gamma = empty(token_shape, torch.float32)
    inverse = empty(matrix_shape)
    state = empty(state_shape, torch.float32)
    task_metadata = empty((metadata_module.metadata_capacity(b, nv),), torch.int32)
    scalar = empty(token_shape[:-1], torch.float32)
    qk_scalar = empty(q.shape[:-1], torch.float32)
    scale = d**-0.5
    stage_two, _, two_specs = integration._prepare_stage_two(
        aqk=inverse,
        qg=token,
        kg=token,
        w=token,
        v_new=token,
        g_cumsum=gamma,
        chunk_states=state,
        output_gradient=v,
        final_state_gradient=empty((b, nv, d, d), torch.float32),
        layout=layout,
        metadata=task_metadata,
        scale=scale,
        core_num=p["core_num"],
        _needs_masked_chunks=(not p["use_full_chunks"] or s == 0 or s % c != 0),
    )
    bound, fused_gate = p["lower_bound"], p["use_gate_in_kernel"]
    norm, fused_beta = p["use_qk_l2norm_in_kernel"], p["use_beta_sigmoid_in_kernel"]
    _, _, one_specs, one_config = integration._prepare_stage_one(
        query=q,
        key=q,
        values=v,
        raw_gate=empty(v.shape, p.get("g_dtype", torch.bfloat16)),
        activated_beta=scalar
        if fused_beta
        else empty(v.shape[:-1], p.get("beta_dtype", torch.bfloat16)),
        beta_dtype=p.get("beta_dtype", torch.bfloat16),
        saved_inverse=inverse,
        saved_qg=token,
        saved_kg=token,
        saved_gamma=gamma,
        A_log=empty((nv,), torch.float32) if fused_gate else None,
        dt_bias=empty((nv, d), torch.float32) if p["has_bias"] else None,
        query_rstd=qk_scalar if norm else None,
        key_rstd=qk_scalar if norm else None,
        stage_two=stage_two,
        layout=layout,
        metadata=task_metadata,
        scale=scale,
        fused_beta=fused_beta,
        fused_gate=fused_gate,
        use_l2norm=norm,
        inverse_factor=0.5 if p["allow_neg_eigval"] else 1.0,
        lower_bound=bound,
    )
    return (
        *two_specs,
        *one_specs,
        integration.cannbotdsl.TensorSpec((1,), integration.dtypes.int32),
    ), (
        *one_config,
        p["use_full_chunks"],
        not p["use_full_chunks"] or s == 0 or s % c != 0,
        False,
        False,
    )


@pytest.mark.parametrize("layout", ("BNSD", "BSND", "TND"))
@pytest.mark.parametrize("beta_fp32", (False, True))
@pytest.mark.parametrize("has_initial", (False, True))
def test_fused_runtime_abi_preserves_backward_arguments(layout, beta_fp32, has_initial):
    concrete, config = _prepare_backward_request(
        dict(
            layout=layout,
            batch=2,
            length=65,
            qk_heads=1,
            value_heads=2,
            use_beta_sigmoid_in_kernel=beta_fp32,
        )
    )
    specs, scalars = integration._recompute_dynamic_compile_request(beta_fp32)
    saved, _ = integration.backward_dynamic_compile_request(beta_fp32)
    assert specs[:89] == saved
    assert len(specs) == 91 and len(specs) + len(scalars) == 109
    signature = inspect.signature(integration._RecomputeBackward.run)
    constexpr = {
        "stage_one_beta_fp32",
        "use_full_chunks",
        "needs_masked_chunks",
        "staged",
        "cube_causal",
    }
    runtime_signature = signature.replace(
        parameters=[
            parameter
            for name, parameter in signature.parameters.items()
            if name != "self" and name not in constexpr
        ]
    )
    initial, u = object(), object()
    bound = runtime_signature.bind(
        *concrete,
        initial,
        u,
        0.125,
        *integration._backward_runtime_config(config),
        int(has_initial),
        32,
    ).arguments
    assert bound["initial_state"] is initial and bound["recomputed_u"] is u
    assert bound["has_initial_state"] == int(has_initial)
    assert bound["padded_layout"] == int(layout != "TND")
    assert bound["stage_one_heads_per_key_head"] == 2
    assert bound["stage_one_scale"] == 128**-0.5
    assert bound["stage_one_lower_bound"] == -0.06
    assert bound["core_num"] == 32


@pytest.mark.parametrize("heads,length", [(1, 64), (32, 64), (32, 8192), (96, 8192)])
@pytest.mark.parametrize("beta_fp32", [True, False])
def test_dynamic_export_matches_pooled_workspace_abi(
    monkeypatch, heads, length, beta_fp32
):
    """Verify the runtime workspace against the dynamic export."""

    def forbidden_compile(*args, **kwargs):
        raise AssertionError("Preparing descriptors must not compile or launch")

    monkeypatch.setattr(integration.cannbotdsl, "compile", forbidden_compile)
    concrete, config = _prepare_backward_request(
        dict(
            length=length,
            qk_heads=heads,
            value_heads=heads,
            use_beta_sigmoid_in_kernel=beta_fp32,
        )
    )
    dynamic, scalars = integration.backward_dynamic_compile_request(beta_fp32)
    launcher = ast.parse(inspect.getsource(integration.FlashKDAGrad))
    run = next(
        node
        for node in launcher.body[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    names = [argument.arg for argument in run.args.args[1:]]
    assert names.index("stage_two_scale") == len(concrete) == len(dynamic) == 89
    assert len(names) == len(dynamic) + len(scalars) == 106
    specs = dict(zip(names, concrete))
    slots = integration.MAX_AIC_CORES
    rows = slots * integration.CHUNK_SIZE
    start = names.index("stage_one_gm_inverse")
    stop = names.index("gm_owner_schedule")
    assert stop - start == 30
    for name in names[start:stop]:
        shape = specs[name].shape
        if name == "stage_one_gm_d_gate_last":
            assert shape == (slots, integration.SUPPORTED_HEAD_DIM)
        else:
            assert shape[0] == rows, name
    assert specs["stage_two_gm_d_Aqk"].shape == (
        rows,
        integration.CHUNK_SIZE,
    )
    state_slots = 2 * integration.MAX_AIC_CORES
    assert specs["stage_two_gm_state_gradients"].shape == (
        state_slots * integration.SUPPORTED_HEAD_DIM,
        integration.SUPPORTED_HEAD_DIM,
    )
    assert scalars == (
        integration.dtypes.float32,
        *integration._backward_scalar_specs(config),
        integration.dtypes.int64,
    )


@pytest.mark.parametrize(
    "layout,nq,nv,lengths,storage,block_size,expected",
    [
        ("BNSD", 96, 96, [8192], 8192, 64, (True, False)),
        ("BNSD", 6, 6, [8193], 8193, 64, (True, True)),
        ("BSND", 8, 8, [128] * 4, 128, 64, (True, False)),
        ("TND", 1, 1, [65, 128], 193, 64, (True, True)),
        ("TND", 8, 8, [64, 1984, 2048, 4096], 8192, 64, (True, False)),
        ("BNSD", 1, 1, [64], 129, 64, (True, False)),
        ("BNSD", 32, 32, [128], 128, 1, (False, True)),
        ("BNSD", 32, 32, [0], 128, 64, (True, True)),
        ("TND", 32, 32, [0, 128], 128, 64, (True, True)),
        ("TND", 1, 4, [128], 128, 64, (False, True)),
    ],
)
def test_chunk_plan_preserves_full_tiles_in_tail_and_padded_sequences(
    monkeypatch, layout, nq, nv, lengths, storage, block_size, expected
):
    info = SimpleNamespace(
        sequences=tuple((0, length, (length + 63) // 64, 0) for length in lengths)
    )
    monkeypatch.setattr(
        integration._metadata, "metadata_host_info", lambda metadata: info
    )

    def tensor(heads):
        shape = (
            (sum(lengths), heads, 128)
            if layout == "TND"
            else (len(lengths), heads, storage, 128)
            if layout == "BNSD"
            else (len(lengths), storage, heads, 128)
        )
        return torch.empty(shape, device="meta")

    metadata = torch.empty(
        metadata_capacity(len(lengths), nv), dtype=torch.int32, device="meta"
    )
    assert (
        integration._track_chunk_plan(
            tensor(nq),
            tensor(nv),
            layout,
            metadata,
            block_size,
        )
        == expected
    )


@pytest.mark.parametrize("length", [0, 64, 65, 128, 129])
def test_masked_workspace_is_omitted_only_when_every_chunk_is_full(length):
    concrete, config = _prepare_backward_request(
        dict(length=length, qk_heads=32, value_heads=32, use_full_chunks=True)
    )
    names = list(inspect.signature(integration.FlashKDAGrad.run).parameters)[1:]
    specs = dict(zip(names, concrete))
    masked_cores = integration.MAX_AIC_CORES if length == 0 or length % 64 else 0
    assert config[11] is True
    assert config[12] == bool(masked_cores)
    assert config[13] is False
    # 新增 cube_causal 后，core_num 前共有四个编译开关。
    assert names[-5:-1] == [
        "use_full_chunks",
        "needs_masked_chunks",
        "staged",
        "cube_causal",
    ]
    assert specs["stage_two_gm_state_gradients"].shape == (masked_cores * 2 * 128, 128)
    assert specs["stage_two_gm_masked_aqk"].shape == (masked_cores * 64, 64)
    assert specs["stage_two_gm_state_high"].shape == (masked_cores * 128, 128)


@pytest.mark.parametrize(
    "layout,batch,nq,nv,length,use_full_chunks",
    [
        ("BNSD", 1, 96, 96, 8192, True),
        ("BSND", 1, 32, 32, 129, True),
        ("TND", 4, 8, 8, 257, True),
        ("TND", 1, 1, 4, 8193, False),
    ],
)
def test_track_stage_two_gradients_use_one_chunk_per_core(
    layout, batch, nq, nv, length, use_full_chunks
):
    concrete, _ = _prepare_backward_request(
        dict(
            layout=layout,
            batch=batch,
            length=length,
            qk_heads=nq,
            value_heads=nv,
            use_full_chunks=use_full_chunks,
        )
    )
    names = list(inspect.signature(integration.FlashKDAGrad.run).parameters)[1:]
    specs = dict(zip(names, concrete))
    cores, chunk, width = (
        integration.MAX_AIC_CORES,
        integration.CHUNK_SIZE,
        integration.SUPPORTED_HEAD_DIM,
    )
    for gradient in ("d_v_new", "d_qg", "d_kg", "d_w"):
        for prefix in ("stage_two_gm_", "stage_one_gm_stage_two_"):
            assert specs[prefix + gradient].shape == (1, 1, cores * chunk, width)
    assert specs["stage_two_gm_d_Aqk"].shape == (cores * chunk, chunk)
    assert specs["stage_one_gm_stage_two_d_Aqk"].shape == (cores, 1, chunk, chunk)
    for prefix in ("stage_two_gm_", "stage_one_gm_stage_two_"):
        assert specs[prefix + "d_gate_last"].shape == (1, 1, cores, width)


def test_export_collects_beta_and_chunk_path_variants_without_launch(monkeypatch):
    calls, closed = [], []

    class Program:
        def close(self):
            closed.append(True)

    def compile_only(entry, *args):
        calls.append(args)
        return Program()

    def forbidden_launch(*args, **kwargs):
        raise AssertionError("AOT collection must not launch a device program")

    monkeypatch.setattr(integration.cannbotdsl, "compile", compile_only)
    monkeypatch.setattr(integration, "_launch_backward_program", forbidden_launch)
    integration.export_flash_kda_grad()
    expected_variants = {
        (beta, full, masked, gate, staged, beta_grad, cube)
        for beta, beta_grad in ((False, False), (True, False), (True, True))
        for gate in (False, True)
        for full, masked, staged in (
            (False, True, False),
            (True, False, False),
            (True, True, False),
            (False, True, True),
        )
        for cube in ((False, True) if full else (False,))
    }
    variants = list(integration._backward_variants())
    assert len(variants) == 36
    assert set(variants) == expected_variants
    assert len(calls) == len(closed) == 72
    for args, variant in zip(calls[:36], variants):
        specs, scalars = integration.backward_dynamic_compile_request(*variant)
        assert args == (*specs, *scalars)
    for args, variant in zip(calls[36:], variants):
        specs, scalars = integration._recompute_dynamic_compile_request(*variant)
        assert args == (*specs, *scalars)


@pytest.mark.parametrize(
    "use_full_chunks,needs_masked_chunks",
    [(False, True), (True, False), (True, True)],
)
def test_dynamic_request_specializes_only_beta_and_chunk_paths(
    use_full_chunks, needs_masked_chunks
):
    from cannbotdsl.core.contracts.binding import bind_aot_call

    specs, scalars = integration.backward_dynamic_compile_request(
        True, use_full_chunks, needs_masked_chunks
    )
    entry = integration.FlashKDAGrad()
    bound = bind_aot_call(
        integration.FlashKDAGrad.run.fn,
        (entry, *specs, *scalars),
        {},
        surface_signature=entry.run._surface,
    )
    assert [p.name for p in bound.contract.parameters if p.role == "constexpr"] == [
        "stage_one_beta_fp32",
        "use_full_chunks",
        "needs_masked_chunks",
        "staged",
        "cube_causal",
    ]
    dimensions = {dimension.name for dimension in bound.contract.dimensions}
    assert {
        "P",
        "L",
        "B",
        "NQK",
        "NV",
        "META",
        "CORES",
        "MASKED_CORE_ROWS",
    } <= dimensions


def test_export_closes_collected_program_if_second_variant_fails(monkeypatch):
    closed = []
    calls = []

    def compile_only(entry, *args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("second chunk-path variant failed")
        return SimpleNamespace(close=lambda: closed.append(True))

    monkeypatch.setattr(integration.cannbotdsl, "compile", compile_only)
    with pytest.raises(RuntimeError, match="second chunk-path variant failed"):
        integration.export_flash_kda_grad()
    assert closed == [True]


@pytest.mark.parametrize("shards", [1, 2, 3, 7, 8, 72])
def test_native_export_shards_cover_every_variant_once(monkeypatch, shards):
    """Worker partitions must preserve both cache modes and every dtype/path."""
    variants = list(integration._backward_variants())
    expected = [
        (recompute, variant) for recompute in (False, True) for variant in variants
    ]
    monkeypatch.setattr(integration, "_close_backward_program", lambda entry: None)
    monkeypatch.setattr(
        integration, "_recompute_dynamic_compile_request", lambda *v: (v, ())
    )
    seen = []
    for shard in range(shards):
        calls = []
        monkeypatch.setenv(
            "CANNBOTDSL_FLASH_KDA_GRAD_EXPORT_SHARD", f"{shard}/{shards}"
        )
        monkeypatch.setattr(
            integration,
            "_compile_backward_program",
            lambda *v: calls.append((False, v)),
        )
        monkeypatch.setattr(
            integration.cannbotdsl,
            "compile",
            lambda fn, *args: (
                calls.append((True, args)) or SimpleNamespace(close=lambda: None)
            ),
        )
        integration.export_flash_kda_grad()
        assert calls == expected[shard::shards]
        seen.extend(shard + i * shards for i in range(len(calls)))
    assert sorted(seen) == list(range(72))


@pytest.mark.parametrize("shard", ["bad", "0/0", "-1/8", "8/8", "0/73", "0/1/2"])
def test_native_export_rejects_invalid_shard_before_compiling(monkeypatch, shard):
    monkeypatch.setenv("CANNBOTDSL_FLASH_KDA_GRAD_EXPORT_SHARD", shard)
    monkeypatch.setattr(
        integration.cannbotdsl,
        "compile",
        lambda *a, **k: pytest.fail("invalid shard must not invoke the compiler"),
    )
    with pytest.raises(ValueError, match="export shard"):
        integration.export_flash_kda_grad()


@pytest.mark.parametrize(
    "layout,batch,length,nq,nv,use_full_chunks",
    [
        ("BNSD", 1, 64, 1, 1, False),
        ("BNSD", 2, 65, 2, 6, False),
        ("BSND", 3, 17, 1, 2, False),
        ("TND", 1, 129, 2, 4, False),
        ("BNSD", 2, 0, 1, 3, False),
        ("TND", 1, 0, 1, 1, False),
        ("BNSD", 1, 8192, 96, 96, True),
        ("BNSD", 1, 8193, 96, 96, True),
        ("BSND", 1, 128, 32, 32, True),
        ("TND", 4, 256, 8, 8, True),
    ],
)
def test_dynamic_specs_accept_prepared_layouts_and_cache_capacities(
    layout,
    batch,
    length,
    nq,
    nv,
    use_full_chunks,
):
    concrete, config = _prepare_backward_request(
        dict(
            layout=layout,
            batch=batch,
            length=length,
            qk_heads=nq,
            value_heads=nv,
            use_full_chunks=bool(use_full_chunks),
        )
    )
    # Bind the actual positional launch values against the host signature.
    # This detects shifted flags or float fields when internal modes disappear.
    signature = inspect.signature(integration.FlashKDAGrad.run)
    constexpr = {
        "stage_one_beta_fp32",
        "use_full_chunks",
        "needs_masked_chunks",
        "staged",
        "cube_causal",
    }
    runtime_signature = signature.replace(
        parameters=[
            parameter
            for name, parameter in signature.parameters.items()
            if name != "self" and name not in constexpr
        ]
    )
    runtime = runtime_signature.bind(
        *([None] * len(concrete)),
        0.125,
        *integration._backward_runtime_config(config),
        32,
    ).arguments
    assert runtime["padded_layout"] == int(layout != "TND")
    assert runtime["stage_one_heads_per_key_head"] == nv // nq
    assert runtime["stage_one_scale"] == 128**-0.5
    assert runtime["stage_one_inverse_factor"] == 0.5
    assert runtime["stage_one_lower_bound"] == -0.06
    assert runtime["core_num"] == 32
    dynamic, _ = integration.backward_dynamic_compile_request()
    dimensions = {}
    # Bind directly occurring dimensions, then validate every derived shape
    # and contiguous scratch stride against actual host-prepared requests.
    for expected, actual in zip(dynamic, concrete):
        for expression, value in zip(expected.shape, actual.shape):
            if isinstance(expression, integration.cannbotdsl.Dim):
                assert expression.min <= value
                assert expression.max is None or value <= expression.max
                assert value % expression.multiple_of == 0
                assert dimensions.setdefault(expression.name, value) == value
        if expected.stride is not None:
            for expression, value in zip(expected.stride, actual.stride):
                if isinstance(expression, integration.cannbotdsl.Dim):
                    assert dimensions.setdefault(expression.name, value) == value

    def evaluate(expression):
        if isinstance(expression, int):
            return expression
        if isinstance(expression, integration.cannbotdsl.Dim):
            return dimensions[expression.name]
        left, right = evaluate(expression.lhs), evaluate(expression.rhs)
        return {
            "add": lambda: left + right,
            "mul": lambda: left * right,
            "floordiv": lambda: left // right,
        }[expression.op]()

    assert len(dynamic) == len(concrete)
    for expected, actual in zip(dynamic, concrete):
        assert expected.dtype == actual.dtype
        assert tuple(map(evaluate, expected.shape)) == actual.shape
        if expected.stride is not None:
            assert tuple(map(evaluate, expected.stride)) == actual.stride


def test_runtime_reuses_exported_request_across_shapes_and_switches(monkeypatch):
    compile_calls, launches = [], []
    monkeypatch.setattr(integration, "_BACKWARD_PROGRAMS", integration.OrderedDict())
    monkeypatch.setattr(
        integration, "_BACKWARD_WORKSPACE_LEASES", integration.OrderedDict()
    )

    class Program:
        def __call__(self, *args):
            launches.append(args)

        def close(self):
            pass

    def compile_only(entry, *args):
        compile_calls.append(args)
        return Program()

    stream = SimpleNamespace(synchronize=lambda: None)

    # fmt: off
    event = lambda: SimpleNamespace(query=lambda: True, record=lambda stream: None,  # noqa: E731 - Preserve the existing mock factory.
                                   synchronize=lambda: None)
    # fmt: on

    monkeypatch.setattr(integration.cannbotdsl, "compile", compile_only)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            device=lambda device: nullcontext(),
            set_device=lambda device: None,
            current_device=lambda: 0,
            current_stream=lambda device: stream,
            Event=event,
        ),
        raising=False,
    )
    for profile in (
        dict(length=64, core_num=8),
        dict(
            batch=2,
            length=65,
            layout="BSND",
            value_heads=192,
            use_gate_in_kernel=False,
        ),
        dict(length=64, use_full_chunks=True),
        dict(length=65, use_full_chunks=True),
        dict(length=129, use_full_chunks=True),
    ):
        specs, config = _prepare_backward_request(profile)
        arguments = tuple(
            SimpleNamespace(record_stream=lambda stream: None) for spec in specs
        )
        core_num = profile.get("core_num", 32)
        integration._launch_backward_program(
            "npu:0", arguments, specs, config, 0.125, core_num
        )
        assert launches[-1] == (
            *arguments,
            0.125,
            *integration._backward_runtime_config(config),
            core_num,
        )
    assert len(compile_calls) == 3
    for call, paths in zip(compile_calls, ((False, True), (True, False), (True, True))):
        specs, scalars = integration.backward_dynamic_compile_request(True, *paths)
        assert call == (*specs, *scalars)
    concrete_specs, config = _prepare_backward_request(
        dict(use_beta_sigmoid_in_kernel=False)
    )
    arguments = tuple(
        SimpleNamespace(record_stream=lambda stream: None) for spec in concrete_specs
    )
    integration._launch_backward_program(
        "npu:0", arguments, concrete_specs, config, 0.125, 32
    )
    assert len(compile_calls) == 4
    specs, scalars = integration.backward_dynamic_compile_request(False)
    assert compile_calls[3] == (*specs, *scalars)
    for fused_beta in (True, False):
        concrete_specs, config = _prepare_backward_request(
            dict(
                g_dtype=torch.float32,
                beta_dtype=torch.float32,
                use_beta_sigmoid_in_kernel=fused_beta,
            )
        )
        for _ in range(2):
            integration._launch_backward_program(
                "npu:0", arguments, concrete_specs, config, 0.125, 32
            )
    # FP32 beta uses the same storage variant with or without sigmoid fusion.
    assert len(compile_calls) == 5
    specs, scalars = integration.backward_dynamic_compile_request(
        True, gate_fp32=True, beta_grad_fp32=True
    )
    assert compile_calls[4] == (*specs, *scalars)
    # Same effective FP32 beta cache, but a BF16 original beta needs its own output ABI.
    concrete_specs, config = _prepare_backward_request(
        dict(
            g_dtype=torch.float32,
            beta_dtype=torch.bfloat16,
            use_beta_sigmoid_in_kernel=True,
        )
    )
    integration._launch_backward_program(
        "npu:0", arguments, concrete_specs, config, 0.125, 32
    )
    assert len(compile_calls) == 6
    specs, scalars = integration.backward_dynamic_compile_request(True, gate_fp32=True)
    assert compile_calls[5] == (*specs, *scalars)
    integration._clear_backward_program_cache()


@pytest.fixture
def _uncached_compiler_validation(monkeypatch):
    """Execute compiler validation hooks even when compiled artifacts exist."""
    # Cache hits and packaged providers return before _run_cannir_passes.
    # Scope the bypass to these checks; monkeypatch restores it afterwards.
    monkeypatch.setenv("CANNBOTDSL_CACHE_MEM_DISABLE", "1")
    monkeypatch.setenv("CANNBOTDSL_CACHE_DISK_DISABLE", "1")
    monkeypatch.setenv("CANNBOTDSL_NATIVE_BINARY_MODE", "off")


@pytest.mark.whitebox
@pytest.mark.usefixtures("_uncached_compiler_validation")
@pytest.mark.parametrize(
    "recomputed,use_full_chunks,needs_masked_chunks,staged,"
    "beta_fp32,beta_grad_fp32,gate_fp32",
    [
        pytest.param(False, False, True, False, False, False, False, id="saved-masked"),
        pytest.param(False, True, False, False, True, True, True, id="saved-full"),
        pytest.param(False, True, True, False, True, False, False, id="saved-mixed"),
        pytest.param(False, False, True, True, False, False, True, id="saved-staged"),
        pytest.param(
            True, False, True, False, True, False, True, id="recompute-masked"
        ),
        pytest.param(
            True, True, False, False, False, False, False, id="recompute-full"
        ),
        pytest.param(True, True, True, False, True, True, True, id="recompute-mixed"),
        pytest.param(True, False, True, True, True, True, False, id="recompute-staged"),
    ],
)
def test_dynamic_export_verifies_frontend_and_lowering(
    monkeypatch,
    recomputed,
    use_full_chunks,
    needs_masked_chunks,
    staged,
    beta_fp32,
    beta_grad_fp32,
    gate_fp32,
):
    """Verify each schedule in both modes, with representative storage dtypes.

    Verify frontend IR and lowered IR in the same compilation. Each mode covers
    all three beta input/output dtype pairs and both gate dtypes; the combined
    cases cover all six dtype combinations without crossing each with every
    schedule. Export/request tests retain exhaustive variant and ABI checks.
    """
    from cannbotdsl._mlir import ir
    from cannbotdsl.core.compiler import passes

    run_passes = passes._run_cannir_passes

    class LoweringVerified(Exception):
        pass

    def verify(module):
        module.operation.verify()
        run_passes(module)
        module.operation.verify()
        device_entries = [
            op.operation
            for op in module.body.operations
            if op.operation.name == "func.func" and "cannir.device" in op.attributes
        ]
        assert device_entries, "lowering must produce device entry points"
        for entry in device_entries:
            # Tensor ABI fields (pointers, dimensions and strides) are all
            # 64-bit. Mixed scalar fields are packed without implicit padding.
            offset = 0
            signature = ir.FunctionType(entry.attributes["function_type"].value)
            for argument_type in signature.inputs:
                kind = str(argument_type)
                if kind == "i64":
                    assert offset % 8 == 0, "unaligned int64 device scalar"
                    offset += 8
                elif kind == "f32":
                    assert offset % 4 == 0, "unaligned float32 device scalar"
                    offset += 4
        raise LoweringVerified

    monkeypatch.setattr(passes, "_run_cannir_passes", verify)
    request = (
        integration._recompute_dynamic_compile_request
        if recomputed
        else integration.backward_dynamic_compile_request
    )
    specs, scalar_specs = request(
        beta_fp32,
        use_full_chunks=use_full_chunks,
        needs_masked_chunks=needs_masked_chunks,
        gate_fp32=gate_fp32,
        staged=staged,
        beta_grad_fp32=beta_grad_fp32,
    )
    with pytest.raises(LoweringVerified):
        integration.cannbotdsl.compile(
            (
                integration._RecomputeBackward()
                if recomputed
                else integration.FlashKDAGrad()
            ).run,
            *specs,
            *scalar_specs,
        )


@pytest.mark.whitebox
@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize("beta_fp32", [True, False])
def test_dynamic_export_translates_with_runtime_switches(monkeypatch, beta_fp32):
    """Exercise the real DSL frontend, lowering passes and AscendC codegen."""
    monkeypatch.setenv("CANNBOTDSL_PIPE_STAGE", "translate")
    specs, scalar_specs = integration.backward_dynamic_compile_request(beta_fp32)
    program = integration.cannbotdsl.compile(
        integration.FlashKDAGrad().run, *specs, *scalar_specs
    )
    try:
        names = set(program._parameter_lowering.signature.parameters)
        assert {
            "stage_one_heads_per_key_head",
            "stage_one_lower_bound",
            "stage_one_fused_gate",
            "padded_layout",
            "core_num",
        } <= names
        assert "stage_one_beta_fp32" not in names
    finally:
        program.close()


def _packaged_metadata(tmp_path, monkeypatch):
    source = tmp_path / "flash_kda_grad_metadata.py"
    source.write_bytes(Path(metadata_module.__file__).read_bytes())
    monkeypatch.setattr(metadata_module, "__file__", str(source))
    monkeypatch.setenv("CANNBOTDSL_NATIVE_BINARY_MODE", "require")
    monkeypatch.setenv("CANNBOTDSL_NPU_ARCH", "dav-3510")
    root = tmp_path / "_aicpu" / "flash_kda_grad_metadata"
    root.mkdir(parents=True)
    name = metadata_module.flash_kda_grad_metadata_kernel.fn.__name__
    binary = root / (name + ".so")
    binary.write_bytes(b"metadata test binary")
    info = dict(
        schema=1,
        target="dav-3510",
        name=name,
        launch_mode="interface",
        metadata_abi=metadata_module.ABI_VERSION,
        binary=binary.name,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(info))
    return binary, manifest, info


def test_packaged_metadata_loads_without_compiler(tmp_path, monkeypatch):
    binary, _, _ = _packaged_metadata(tmp_path, monkeypatch)
    monkeypatch.setattr(metadata_module, "_COMPILED", None)

    def forbidden_compile(*args, **kwargs):
        raise AssertionError("installed metadata must not compile")

    monkeypatch.setattr(metadata_module, "compile_aicpu_kernel", forbidden_compile)
    program = metadata_module._get_compiled_kernel()
    assert program.so_path == str(binary)
    assert metadata_module._get_compiled_kernel() is program


@pytest.mark.parametrize("damage", ["binary", "source", "abi", "target", "filename"])
def test_corrupt_or_incompatible_metadata_rejected(tmp_path, monkeypatch, damage):
    binary, manifest, info = _packaged_metadata(tmp_path, monkeypatch)
    if damage == "binary":
        binary.write_bytes(b"corrupt")
    else:
        changes = dict(
            source=("source_sha256", "0" * 64),
            abi=("metadata_abi", -1),
            target=("target", "dav-2201"),
            filename=("binary", "../escape.so"),
        )
        key, value = changes[damage]
        info[key] = value
        manifest.write_text(json.dumps(info))
    with pytest.raises(RuntimeError):
        metadata_module._load_packaged_kernel()


def test_missing_metadata_obeys_native_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata_module, "__file__", str(tmp_path / "source.py"))
    monkeypatch.setenv("CANNBOTDSL_NATIVE_BINARY_MODE", "prefer")
    assert metadata_module._load_packaged_kernel() is None
    monkeypatch.setenv("CANNBOTDSL_NATIVE_BINARY_MODE", "require")
    with pytest.raises(RuntimeError, match="required but not installed"):
        metadata_module._load_packaged_kernel()
    monkeypatch.setenv("CANNBOTDSL_NATIVE_BINARY_MODE", "off")
    assert metadata_module._load_packaged_kernel() is None


@pytest.mark.whitebox
@pytest.mark.ascendc_toolchain
def test_native_backward_export_aot_compiles(monkeypatch, tmp_path):
    """The production parallel build must publish all 72 loadable providers."""
    import subprocess
    import sys

    monkeypatch.setenv("CANNBOTDSL_NATIVE_BINARY_MODE", "off")
    if not os.environ.get("KDA_DSL_ROOT"):
        pytest.skip("optional native package export requires KDA_DSL_ROOT")
    project = Path(os.environ["KDA_DSL_ROOT"]).expanduser().resolve()
    subprocess.run(
        [
            sys.executable,
            str(project / "net/native_package/build.py"),
            "--operator",
            "flash_kda_grad",
            "--output-root",
            str(tmp_path / "output"),
            "--work-root",
            str(tmp_path / "work"),
        ],
        check=True,
    )
    native = tmp_path / "output/operators/flash_kda_grad/native"
    index = json.loads((native / "index.json").read_text())
    inventory = json.loads(
        (native / "operators/flash_kda_grad/operator.json").read_text()
    )
    assert len(index) == inventory["requests"] == len(inventory["variants"]) == 72
    for entry in index.values():
        descriptor = json.loads((native / entry["descriptor"]).read_text())
        program = integration.cannbotdsl.load(str(native / descriptor["binary"]))
        program.close()


def _empty_metadata(batch, heads):
    return torch.empty(metadata_capacity(batch, heads), dtype=torch.int32)


@pytest.mark.whitebox
@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize(
    "layout,norm,fused_gate,fused_beta,lower_bound,inverse_factor,has_bias",
    [
        pytest.param("TND", False, False, False, -5.0, 1.0, False, id="raw-packed"),
        pytest.param("BNSD", True, True, True, -5.0, 1.0, True, id="fused-bnsd"),
        pytest.param("BSND", True, True, True, -5.0, 1.0, True, id="fused-bsnd"),
        pytest.param(
            "TND",
            True,
            True,
            True,
            None,
            0.5,
            False,
            id="softplus-negative-eigval-no-bias",
        ),
        pytest.param("TND", False, True, True, -0.7, 1.0, True, id="raw-query-key"),
        pytest.param("TND", True, False, True, -0.7, 1.0, False, id="raw-gate"),
        pytest.param("TND", True, True, False, -0.7, 1.0, True, id="raw-beta"),
    ],
)
def test_full_backward_single_kernel_aot(
    layout,
    norm,
    fused_gate,
    fused_beta,
    lower_bound,
    inverse_factor,
    has_bias,
):
    """The public backward stages compile together as one device kernel."""
    token = torch.empty((1, 1, 128), dtype=torch.bfloat16)
    raw_token = (
        token
        if layout == "TND"
        else torch.empty((1, 1, 2, 128), dtype=torch.bfloat16)
        if layout == "BNSD"
        else torch.empty((1, 2, 1, 128), dtype=torch.bfloat16)
    )
    token = raw_token
    gamma = torch.empty_like(token, dtype=torch.float32)
    inverse = torch.empty(
        (2, 1, 64, 64) if layout == "TND" else (*token.shape[:-1], 64),
        dtype=torch.bfloat16,
    )
    state = torch.empty(
        (2, 1, 128, 128) if layout == "TND" else (1, 1, 1, 128, 128),
        dtype=torch.float32,
    )
    metadata = _empty_metadata(1, 1)
    stage_two, two_args, two_specs = _prepare_stage_two(
        aqk=inverse,
        qg=token,
        kg=token,
        w=token,
        v_new=token,
        g_cumsum=gamma,
        chunk_states=state,
        output_gradient=raw_token,
        final_state_gradient=torch.empty((1, 1, 128, 128)),
        layout=layout,
        metadata=metadata,
        scale=0.125,
    )
    _, one_args, one_specs, one_config = _prepare_stage_one(
        query=raw_token,
        key=raw_token,
        values=raw_token,
        raw_gate=raw_token,
        activated_beta=torch.empty(
            raw_token.shape[:-1],
            dtype=(torch.float32 if fused_beta else torch.bfloat16),
        ),
        saved_inverse=inverse,
        saved_qg=token,
        saved_kg=token,
        saved_gamma=gamma,
        A_log=torch.empty((1,), dtype=torch.float32) if fused_gate else None,
        dt_bias=(
            torch.empty((1, 128), dtype=torch.float32)
            if fused_gate and has_bias
            else None
        ),
        query_rstd=torch.empty(raw_token.shape[:-1], dtype=torch.float32)
        if norm
        else None,
        key_rstd=torch.empty(raw_token.shape[:-1], dtype=torch.float32)
        if norm
        else None,
        stage_two=stage_two,
        layout=layout,
        metadata=metadata,
        scale=0.125,
        fused_beta=fused_beta,
        fused_gate=fused_gate,
        use_l2norm=norm,
        inverse_factor=inverse_factor,
        lower_bound=lower_bound,
    )
    compiled = integration.cannbotdsl.compile(
        integration.FlashKDAGrad().run,
        *two_specs,
        *one_specs,
        integration.cannbotdsl.TensorSpec((1,), integration.dtypes.int32),
        integration.dtypes.float32,
        *integration._backward_scalar_specs((*one_config, False, True, False, False)),
        integration.dtypes.int64,
    )
    try:
        assert compiled.so_path
        compiled._parameter_lowering.signature.bind(
            *two_args,
            *one_args,
            metadata[:1],
            0.125,
            *integration._backward_runtime_config(
                (*one_config, False, True, False, False)
            ),
            1,
        )
    finally:
        compiled.close()


_BIN_SAVED_ROWS = ("w", "u", "v_new", "qg", "kg", "g_cumsum")


@pytest.mark.parametrize("missing", ("do", "dht"))
def test_bin_requires_both_upstream_gradients(monkeypatch, tmp_path, missing):
    attrs = dict(
        use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        use_gate_in_kernel=False,
    )
    specs = {}
    for name in _BIN_REQUIRED_NAMES:
        specs[name] = {
            "dtype": "float32"
            if name in ("dht", "chunk_states", "g_cumsum")
            else "bfloat16"
        }
        if name != missing:
            _bin_tensor_path(tmp_path, name, "prefixed").write_bytes(b"fixture")
    monkeypatch.setitem(
        _bin_load_inputs.__globals__, "_bin_read_tensor", lambda *args: object()
    )
    with pytest.raises(FileNotFoundError, match=f"input_{missing}\\.bin"):
        _bin_load_inputs(tmp_path, "prefixed", attrs, specs, False)


def _write_bin_test_bundle(folder):
    """Synthetic reader fixtures only; these constant outputs are not a golden."""
    args = _bin_parse_args(
        [
            "--compare-bin",
            "--bin-dir",
            str(folder),
            "--norm",
            "--gate",
            "--beta-sigmoid",
        ]
    )
    for name, spec in _bin_specs(args).items():
        tensor = torch.ones(spec["shape"], dtype=_BIN_DTYPES[spec["dtype"]])
        _bin_tensor_path(folder, name, "prefixed").write_bytes(
            tensor.view(torch.uint8).numpy().tobytes()
        )
    return args


def test_bin_bf16_payload_and_byte_count(tmp_path):
    path = tmp_path / "input_q.bin"
    value = torch.tensor([0.0, 1.0, -2.5, 0.003], dtype=torch.bfloat16)
    raw = value.view(torch.uint8).numpy().tobytes()
    path.write_bytes(raw)
    spec = {"shape": [4], "dtype": "bfloat16"}
    assert torch.equal(
        _bin_read_tensor(path, spec).view(torch.uint16), value.view(torch.uint16)
    )
    path.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="bytes, expected"):
        _bin_read_tensor(path, spec)


def test_bin_precision_rejects_wrong_and_nonfinite():
    reference = torch.tensor([1.0, 2.0, 0.0])
    assert _bin_metrics(reference, reference, 0.004, 0.03, 0.05)["passed"]
    wrong = _bin_metrics(reference + 1, reference, 0.004, 0.03, 0.05)
    assert not wrong["passed"] and wrong["element_pass_rate_pct"] < 100
    bad = _bin_metrics(
        torch.tensor([1.0, float("nan"), float("inf")]), reference, 0.004, 0.03, 0.05
    )
    assert (
        not bad["passed"] and bad["nonfinite_elements"] == 2 and bad["max_abs"] is None
    )
    assert _bin_metrics(torch.zeros(3), torch.zeros(3), 0.004, 0.03, 0.05)["passed"]
    overflow = _bin_metrics(
        torch.tensor([1e308], dtype=torch.float64),
        torch.tensor([-1e308], dtype=torch.float64),
        0.004,
        0.03,
        0.05,
    )
    assert not overflow["passed"] and overflow["max_abs"] is None


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_bin_main_cache_layout_contract(layout):
    args = _bin_parse_args(
        [
            "--compare-bin",
            "--bin-dir",
            ".",
            "--layout",
            layout,
            "--batch",
            "2",
            "--length",
            "65",
        ]
    )
    shapes = _bin_specs(args)
    assert shapes["chunk_states"]["shape"] == [2, 2, 2, 128, 128]
    assert shapes["qg"]["shape"] == [2, 2, 65, 128]
    assert shapes["Aqk"]["shape"] == [2, 2, 2, 64, 64]
    assert shapes["g"]["shape"] == shapes["v"]["shape"]
    assert shapes["q_rtsd"]["shape"] == [2, 1, 65]


def test_bin_rejects_fp16_cli():
    with pytest.raises(SystemExit):
        _bin_parse_args(["--compare-bin", "--bin-dir", ".", "--dtype", "float16"])


def test_bin_reader_requires_inputs_and_rejects_old_manifest(tmp_path):
    import json

    args = _write_bin_test_bundle(tmp_path)
    values, refs, attrs, naming, hashes = _bin_load_bundle(args)
    assert values["q"].shape == (1, 64, 1, 128) and values["v"].shape == (1, 64, 2, 128)
    assert refs["dv"].shape == values["v"].shape and attrs["use_gate_in_kernel"]
    assert naming == "prefixed" and len(hashes) == 29
    _bin_tensor_path(tmp_path, "qg", "prefixed").unlink()
    with pytest.raises(FileNotFoundError, match="qg"):
        _bin_load_bundle(args)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"format": "raw-contiguous-le-v1"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="old raw-Q/K or FP16/combination ABI"):
        _bin_load_bundle(args)


def test_bin_manifest_preserves_attrs_and_rejects_wrong_input_dtype(tmp_path):
    import json

    args = _write_bin_test_bundle(tmp_path)
    _, _, attrs, _, _ = _bin_load_bundle(args)
    manifest = dict(
        schema=_BIN_SCHEMA,
        format="raw-contiguous-le-v1",
        naming="prefixed",
        attrs=attrs,
        tensors=_bin_specs(args),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    args.norm = args.gate = args.beta_sigmoid = False
    assert _bin_load_bundle(args)[2] == attrs
    manifest["tensors"]["g"]["dtype"] = "float32"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="g must be bfloat16"):
        _bin_load_bundle(args)
    manifest["tensors"]["g"]["dtype"] = "bfloat16"
    manifest["tensors"]["qg"]["shape"] = [64, 128, 2]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="qg: main-ABI shape"):
        _bin_load_bundle(args)


def test_bin_check_only_and_json_input_protection(tmp_path, monkeypatch):
    _write_bin_test_bundle(tmp_path)

    def unexpected_launch(*args, **kwargs):
        raise AssertionError("check-only must not launch metadata")

    monkeypatch.setitem(
        globals(),
        "flash_kda_grad_metadata",
        unexpected_launch,
    )
    argv = [
        "--compare-bin",
        "--bin-dir",
        str(tmp_path),
        "--norm",
        "--gate",
        "--beta-sigmoid",
        "--check-only",
    ]
    assert _bin_compare_main(argv) == 0
    with pytest.raises(ValueError, match="must not overwrite"):
        _bin_compare_main([*argv, "--json", str(tmp_path / "input_q.bin")])


def test_bin_wrong_outputs_write_failed_report(tmp_path, monkeypatch):
    import json

    args = _write_bin_test_bundle(tmp_path)
    _, refs, _, _, _ = _bin_load_bundle(args)
    wrong = tuple(torch.zeros_like(refs[name]) for name in _BIN_OUTPUT_NAMES)
    monkeypatch.setitem(
        globals(),
        "_bin_run_backward",
        lambda *unused: wrong,
    )
    status = _bin_compare_main(
        [
            "--compare-bin",
            "--bin-dir",
            str(tmp_path),
            "--norm",
            "--gate",
            "--beta-sigmoid",
        ]
    )
    report = json.loads((tmp_path / "compare.json").read_text(encoding="utf-8"))
    assert status == 1 and report["passed"] is False
    assert report["summary"]["global_max_abs"] == 1.0
    assert report["summary"]["element_pass_rate_pct"] == 0.0


@pytest.mark.whitebox
@pytest.mark.parametrize(
    "heads,value_heads,length", [(1, 1, 65), (6, 6, 8192), (1, 4, 129), (3, 12, 2113)]
)
def test_every_owner_uses_bounded_state_and_chunk_scratch(heads, value_heads, length):
    concrete, _ = _prepare_backward_request(
        dict(
            length=length,
            qk_heads=heads,
            value_heads=value_heads,
        )
    )
    names = list(inspect.signature(integration.FlashKDAGrad.run).parameters)[1:]
    specs = dict(zip(names, concrete))
    cores, width = integration.MAX_AIC_CORES, integration.SUPPORTED_HEAD_DIM
    assert specs["stage_two_gm_state_gradients"].shape == (2 * cores * width, width)
    slots = cores * value_heads // heads
    assert specs["stage_one_gm_d_gate_last"].shape == (slots, width)
    assert specs["stage_one_gm_inverse"].shape == (
        slots * integration.CHUNK_SIZE,
        integration.CHUNK_SIZE,
    )


@pytest.mark.whitebox
@pytest.mark.npu
@pytest.mark.parametrize(
    "heads,value_heads,lengths",
    [
        (1, 2, [129]),
        (3, 12, [65]),
        (33, 66, [65]),
        (1, 2, [0, 65, 129]),
    ],
)
def test_grouped_one_track_reverse_scan_matches_token_autograd(
    heads, value_heads, lengths
):
    test_full_train_backward_npu(
        layout="TND",
        lengths=lengths,
        qk_heads=heads,
        value_heads=value_heads,
        norm=True,
        fused_gate=True,
        fused_beta=True,
        beta_factor=2.0,
        lower_bound=-0.06,
        seed_kind="both",
        has_bias=True,
    )


@pytest.mark.whitebox
@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,lengths,heads",
    [
        ("BNSD", [128], 1),
        ("BSND", [64], 2),
        ("TND", [64, 128, 64, 128], 1),
        ("TND", [64, 64, 64, 1024], 1),
        ("TND", [65, 128, 129], 1),
        ("BNSD", [129], 1),
        ("BSND", [127], 2),
    ],
)
def test_full_one_track_handles_fewer_owners_than_cores(layout, lengths, heads):
    test_full_train_backward_npu(
        layout=layout,
        lengths=lengths,
        qk_heads=heads,
        value_heads=heads,
        norm=True,
        fused_gate=True,
        fused_beta=True,
        beta_factor=2.0,
        lower_bound=-0.06,
        seed_kind="both",
        has_bias=True,
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "lower_bound,has_bias",
    [
        (bound, bias)
        for bound in (-1.12, -2.0, -3.0, -4.0, -5.0)
        for bias in (True, False)
    ]
    + [(bound, False) for bound in (-2.15, -2.20, -2.25, -2.30)],
)
def test_bounded_gate_runtime_span_matches_token_autograd(lower_bound, has_bias):
    """Check safe/full, fallback, and near-threshold spans with the original golden."""
    test_full_train_backward_npu(
        "TND",
        [64, 128, 97, 35],
        2,
        2,
        True,
        True,
        True,
        2.0,
        lower_bound,
        "both",
        has_bias,
    )


@pytest.mark.npu
@pytest.mark.parametrize("layout", ["TND", "BNSD", "BSND"])
@pytest.mark.parametrize("has_bias", [True, False])
def test_bounded_gate_raw_beta_matches_token_autograd(layout, has_bias):
    """Cover safe and split causal paths with BF16 activated beta."""
    lengths = [64, 128, 97, 35] if layout == "TND" else [128, 128]
    test_full_train_backward_npu(
        layout,
        lengths,
        2,
        2,
        True,
        True,
        False,
        1.0,
        -5.0,
        "both",
        has_bias,
    )


@pytest.mark.whitebox
@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize("recomputed", [False, True])
def test_full_chunk_cube_causal_aot(recomputed):
    """Compile the real mixed full/tail Cube pipeline, including shared UB/L1."""
    request = (
        integration._recompute_dynamic_compile_request
        if recomputed
        else integration.backward_dynamic_compile_request
    )
    specs, scalars = request(True, True, True, True, False, True, True)
    entry = (
        integration._RecomputeBackward() if recomputed else integration.FlashKDAGrad()
    )
    program = integration.cannbotdsl.compile(entry.run, *specs, *scalars)
    try:
        assert program.so_path
    finally:
        program.close()


def test_causal_factorization_matches_autograd_with_zero_beta():
    """Validate the migrated contraction/signs against independent differentiation."""
    generator = torch.Generator().manual_seed(73)
    q, k = (
        torch.randn(64, 128, generator=generator, dtype=torch.float64).requires_grad_()
        for _ in range(2)
    )
    g = (
        -0.03 * torch.rand(64, 128, generator=generator, dtype=torch.float64).cumsum(0)
    ).requires_grad_()
    beta = torch.randn(64, generator=generator, dtype=torch.float64)
    beta[::7] = 0
    beta.requires_grad_()
    a = torch.randn(64, 64, generator=generator, dtype=torch.float64).tril()
    lower = torch.randn(64, 64, generator=generator, dtype=torch.float64).tril(-1)
    loss = q.new_zeros(())
    for row in range(64):
        decay = (g[row] - g[: row + 1]).exp()
        loss = loss + (a[row, : row + 1, None] * q[row] * k[: row + 1] * decay).sum()
        loss = (
            loss
            + (
                lower[row, :row, None] * beta[row] * k[row] * k[:row] * decay[:row]
            ).sum()
        )
    expected = torch.autograd.grad(loss, (q, k, g, beta))
    f, r = (g - g[0]).exp(), (g[0] - g).exp()
    dq = f * (a @ (k * r))
    unweighted = f * (lower @ (k * r))
    kr = beta[:, None] * unweighted
    kc = r * (a.T @ (q * f) + lower.T @ (beta[:, None] * k * f))
    actual = (dq, kr + kc, q * dq + k * (kr - kc), (k * unweighted).sum(-1))
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)


@pytest.mark.npu
@pytest.mark.parametrize("length", [65, 64], ids=["full-and-tail", "aligned"])
def test_restored_cube_causal_small_npu(monkeypatch, length):
    """Use enough equal heads to select persistent Cube, then check all gradients."""
    launches = []
    original = integration._launch_backward_program

    def launch(*args, **kwargs):
        config = args[3]
        launches.append(config)
        print(f"restored Cube small: T={length} N=32, config={config}", flush=True)
        assert config[14] and not config[13]
        return original(*args, **kwargs)

    monkeypatch.setattr(integration, "_launch_backward_program", launch)
    generator = torch.Generator().manual_seed(73)

    def random(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32)

    q, k = (random(length, 32, 128).bfloat16() for _ in range(2))
    v, g = ((0.2 * random(length, 32, 128)).bfloat16() for _ in range(2))
    beta = random(length, 32).bfloat16()
    h0 = 0.02 * random(1, 32, 128, 128)
    a_log = torch.linspace(-3.8, -3.2, 32, dtype=torch.float32)
    bias = 0.1 * random(32, 128)
    do = (0.1 * random(length, 32, 128)).bfloat16()
    dht = 0.02 * random(*h0.shape)
    offsets = torch.tensor([0, length], dtype=torch.int32)
    attrs = dict(
        scale=128**-0.5,
        layout="TND",
        lower_bound=-0.06,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
    )
    with torch.no_grad():
        cache = flash_kda_fwd(
            q, k, v, g, beta, h0, a_log, bias, cu_seqlens=offsets, **attrs
        )
    inputs = dict(q=cache.q_norm, k=cache.k_norm, v=v, g=g, beta=beta, do=do, dht=dht)
    inputs.update(
        {
            name: getattr(cache, "q_rstd" if name == "q_rtsd" else name)
            for name in _PROFILE_INPUT_NAMES[7:]
        }
    )
    inputs = _reference_forward_arguments({**inputs, "layout": "TND"}, [length])
    inputs.pop("layout")
    inputs = {name: value.npu() for name, value in inputs.items()}
    metadata = flash_kda_grad_metadata(
        **_metadata_arguments(inputs["q"], inputs["v"], "TND", offsets.npu())
    )
    actual = flash_kda_grad_npu(
        **inputs,
        metadata=metadata,
        A_log=a_log.npu(),
        dt_bias=bias.npu(),
        disable_recompute=True,
        **attrs,
    )
    torch.npu.synchronize()
    leaves = [
        x.detach().clone().requires_grad_() for x in (q, k, v, g, beta, h0, a_log, bias)
    ]
    output, final = _equal_head_token_reference(*leaves)
    expected = torch.autograd.grad(
        (output * do.float()).sum() + (final * dht).sum(), leaves
    )
    for name, got, index in zip(_BIN_OUTPUT_NAMES, actual, (0, 1, 2, 5, 3, 4, 6, 7)):
        _assert_precision(name, got, expected[index])
    assert launches


def _equal_head_token_reference(
    q, k, v, g, beta, h0, a_log, bias, lower_bound=-0.06, beta_factor=2.0
):
    """Batch independent heads; retain the token recurrence of _token_reference.

    This reference covers normalized Q/K and fused bounded gate/beta only.
    Processing one sequence at a time bounds the autograd graph's lifetime.
    """
    assert q.shape[1] == k.shape[1] == v.shape[1]
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    gate = lower_bound * torch.sigmoid(a_log.exp()[None, :, None] * (g + bias))
    activated = beta_factor * torch.sigmoid(beta)
    state = h0[0].float().transpose(-1, -2)
    rows = []
    for token in range(q.shape[0]):
        key = k[token]
        state = gate[token].exp()[:, :, None] * state
        delta = activated[token, :, None] * (
            v[token] - torch.bmm(key[:, None, :], state).squeeze(1)
        )
        state = state + key[:, :, None] * delta[:, None, :]
        rows.append(128**-0.5 * torch.bmm(q[token, :, None, :], state).squeeze(1))
    return torch.stack(rows), state.transpose(-1, -2).unsqueeze(0)


@pytest.mark.npu
@pytest.mark.slow
def test_tnd_b31_t8192_n96_d128_ragged():
    """All 31 sequences and 96 heads, including 112 full and 30 tail chunks."""
    pytest.importorskip("torch_npu")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 8))
    try:
        _check_tnd_b31_t8192_n96_d128_ragged()
    finally:
        torch.set_num_threads(previous_threads)


def _check_tnd_b31_t8192_n96_d128_ragged():
    fixture = _tnd_b31_fixture()
    inputs = fixture["main"]["inputs"]
    q, k, v, g, beta, do, dht = (inputs[name] for name in _PROFILE_INPUT_NAMES[:7])
    h0 = fixture["provenance"]["initial_state"]
    q, k = fixture["provenance"]["raw_q"], fixture["provenance"]["raw_k"]
    lengths = TND_B31_LENGTHS
    a_log, bias, offsets = (
        fixture["optional"][name] for name in _PROFILE_OPTIONAL_NAMES
    )
    attrs = fixture["attrs"]
    device_inputs = {name: tensor.npu() for name, tensor in inputs.items()}
    metadata = flash_kda_grad_metadata(
        **_metadata_arguments(
            device_inputs["q"], device_inputs["v"], "TND", offsets.npu()
        )
    )
    result = flash_kda_grad_npu(
        disable_recompute=True,
        **device_inputs,
        A_log=a_log.npu(),
        dt_bias=bias.npu(),
        metadata=metadata,
        **attrs,
    )
    torch.npu.synchronize()
    names = ("dq", "dk", "dv", "dh0", "dg", "dbeta", "dA", "dBias")
    actual = {name: _precision_to_cpu(x) for name, x in zip(names, result)}
    del result, device_inputs, metadata, inputs, fixture
    gc.collect()
    for name, tensor in actual.items():
        assert torch.isfinite(tensor.float()).all(), name
    expected = {name: torch.zeros_like(x) for name, x in actual.items()}
    cursor = 0
    for sequence, length in enumerate(lengths):
        end = cursor + length
        leaves = [
            x.detach().clone().requires_grad_()
            for x in (
                q[cursor:end],
                k[cursor:end],
                v[cursor:end],
                g[cursor:end],
                beta[cursor:end],
                h0[sequence : sequence + 1],
                a_log,
                bias,
            )
        ]
        output, final = _equal_head_token_reference(*leaves)
        gradients = torch.autograd.grad(
            (output * do[cursor:end].float()).sum()
            + (final * dht[sequence : sequence + 1]).sum(),
            leaves,
        )
        for name, gradient in zip(("dq", "dk", "dv", "dg", "dbeta"), gradients[:5]):
            expected[name][cursor:end] = gradient
        expected["dh0"][sequence : sequence + 1] = gradients[5]
        expected["dA"] += gradients[6]
        expected["dBias"] += gradients[7]
        del leaves, output, final, gradients
        cursor = end
        gc.collect()
        print(f"reference sequence {sequence + 1}/31 complete", flush=True)
    for name in names:
        _assert_precision(name, actual[name], expected[name])


# Tests and helpers from test_flash_kda_grad_metadata.py.


CHUNK_SIZE = metadata_module.CHUNK_SIZE
HEAD_DIM = metadata_module.SUPPORTED_HEAD_DIM
HEADER_WORDS = metadata_module.HEADER_WORDS
TASK_WORDS = metadata_module.TASK_WORDS
STATUS_OK = metadata_module.STATUS_OK
STATUS_BAD_CU_START = metadata_module.STATUS_BAD_CU_START
STATUS_BAD_PADDED_LENGTH = metadata_module.STATUS_BAD_PADDED_LENGTH
STATUS_BAD_PACKED_END = metadata_module.STATUS_BAD_PACKED_END


@pytest.fixture(scope="session")
def metadata_x86(tmp_path_factory):
    if not (shutil.which("c++") or shutil.which("g++")):
        pytest.skip("metadata cross-check requires a host C++ compiler")
    from cannbotdsl.aicpu.toolchain import CompiledAicpuKernel, compile_x86

    module = load_sample("flash_kda_grad/flash_kda_grad_metadata.py")
    kernel = module.flash_kda_grad_metadata_kernel
    directory = tmp_path_factory.mktemp("kda_metadata_x86")
    for name, source in kernel.generate_sources().items():
        (directory / name).write_text(source)
    binary = compile_x86(kernel, str(directory))
    compiled = CompiledAicpuKernel(
        binary,
        kernel.spec,
        name=kernel.fn.__name__,
        launch_mode="interface",
        x86_so_path=binary,
    )

    def launch(
        device,
        cu_seqlens,
        capacity,
        batch,
        value_heads,
        storage_length,
        is_packed,
        *,
        seqused=None,
    ):
        metadata = torch.empty(capacity, dtype=torch.int32)
        compiled.run_x86(
            cu_seqlens=SimpleNamespace(
                x86_addr=0 if cu_seqlens is None else cu_seqlens.data_ptr()
            ),
            seqused=SimpleNamespace(
                x86_addr=0 if seqused is None else seqused.data_ptr()
            ),
            metadata=SimpleNamespace(x86_addr=metadata.data_ptr()),
            capacity=capacity,
            batch=batch,
            value_heads=value_heads,
            storage_length=storage_length,
            is_packed=int(is_packed),
            has_cu=int(cu_seqlens is not None),
            has_seqused=int(seqused is not None),
        )
        return metadata

    return launch


@pytest.fixture
def host_sequence_metadata(monkeypatch, metadata_x86):
    """Emulate NPU placement/streams while executing the real portable kernel."""
    module = load_sample("flash_kda_grad/flash_kda_grad_metadata.py")
    integration = load_sample("flash_kda_grad/flash_kda_grad.py")
    device = SimpleNamespace(type="npu", index=0)
    monkeypatch.setattr(torch.Tensor, "device", property(lambda tensor: device))
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False
    )
    monkeypatch.setattr(module, "_TENSOR_CACHE", module.MetadataTensorCache())
    monkeypatch.setattr(module, "_launch_metadata", metadata_x86)
    monkeypatch.setattr(module, "wait_metadata_ready", lambda metadata: None)
    monkeypatch.setattr(integration, "wait_metadata_ready", lambda metadata: None)
    return module


def test_metadata_capacity_has_one_record_per_sequence_and_value_head():
    assert metadata_capacity(3, 2) == HEADER_WORDS + 3 * 2 * TASK_WORDS


def test_metadata_default_tnd_uses_explicit_dimensions(host_sequence_metadata):
    metadata = flash_kda_grad_metadata(
        torch.tensor([0, 0, 65], dtype=torch.int32),
        batch_size=2,
        n_heads_qk=1,
        n_heads_v=3,
        seqlens=65,
    )
    _assert_metadata_header(
        metadata, batch=2, value_heads=3, storage=65, chunk_count=2, flags=3
    )
    assert metadata_module.metadata_host_info(metadata).sequences == (
        (0, 0, 0, 0),
        (0, 65, 2, 0),
    )


@pytest.mark.parametrize("layout", ("BNSD", "BSND"))
@pytest.mark.parametrize("storage", (0, 65))
def test_metadata_without_lengths_uses_current_device_and_full_storage(
    monkeypatch,
    host_sequence_metadata,
    metadata_x86,
    layout,
    storage,
):
    devices = []

    def launch(device, *args, **kwargs):
        devices.append(device)
        return metadata_x86(device, *args, **kwargs)

    monkeypatch.setattr(metadata_module, "_launch_metadata", launch)
    attrs = dict(
        batch_size=2, n_heads_qk=1, n_heads_v=3, seqlens=storage, layout=layout
    )
    first = flash_kda_grad_metadata(**attrs)
    assert flash_kda_grad_metadata(**attrs) is first
    monkeypatch.setattr(torch.npu, "current_device", lambda: 1)
    second = flash_kda_grad_metadata(**attrs)
    assert second is not first and devices == ["npu:0", "npu:1"]
    chunks = (storage + 63) // 64
    _assert_metadata_header(
        second, batch=2, value_heads=3, storage=storage, chunk_count=2 * chunks, flags=0
    )
    assert metadata_module.metadata_host_info(second).sequences == (
        (0, storage, chunks, 0),
        (storage, storage, chunks, chunks),
    )


def test_metadata_lengths_select_device_and_attributes_invalidate_cache(
    monkeypatch,
    host_sequence_metadata,
    metadata_x86,
):
    device = SimpleNamespace(type="npu", index=2)
    monkeypatch.setattr(torch.Tensor, "device", property(lambda tensor: device))
    devices = []

    def launch(target, *args, **kwargs):
        devices.append(target)
        return metadata_x86(target, *args, **kwargs)

    monkeypatch.setattr(metadata_module, "_launch_metadata", launch)
    attrs = dict(
        seqused=torch.tensor([0, 65], dtype=torch.int32),
        batch_size=2,
        n_heads_qk=1,
        n_heads_v=2,
        seqlens=65,
        layout="BNSD",
    )
    original = flash_kda_grad_metadata(**attrs)
    assert flash_kda_grad_metadata(**attrs) is original
    for changed in (
        dict(n_heads_qk=2),
        dict(n_heads_v=4),
        dict(seqlens=66),
        dict(layout="BSND"),
    ):
        assert flash_kda_grad_metadata(**{**attrs, **changed}) is not original
    assert len(devices) == 5 and all(target is device for target in devices)


@pytest.mark.parametrize("layout", ["BNSD", "BSND", "TND"])
@pytest.mark.parametrize("lengths", [[0, 63, 64, 65, 129], [0, 0], [64, 64]])
def test_explicit_lengths_generate_exact_records(
    host_sequence_metadata, layout, lengths
):
    batch, heads = len(lengths), 3
    storage = sum(lengths) if layout == "TND" else max(lengths)
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    options = (
        dict(cu_seqlens=torch.tensor(prefix, dtype=torch.int32))
        if layout == "TND"
        else dict(seqused=torch.tensor(lengths, dtype=torch.int32))
    )
    metadata = flash_kda_grad_metadata(
        batch_size=batch,
        n_heads_qk=1,
        n_heads_v=heads,
        seqlens=storage,
        layout=layout,
        **options,
    )
    chunks = [(length + 63) // 64 for length in lengths]
    expected = [
        metadata_module.MAGIC,
        metadata_module.ABI_VERSION,
        STATUS_OK,
        batch,
        heads,
        storage,
        sum(chunks),
        3 if layout == "TND" else 1,
    ]
    for sequence, (length, count) in enumerate(zip(lengths, chunks)):
        first = sum(chunks[:sequence])
        for head in range(heads):
            expected.extend(
                [
                    sequence,
                    prefix[sequence],
                    first,
                    count,
                    length,
                    head,
                    first * heads + head * count,
                ]
            )
    assert metadata.tolist() == expected


@pytest.mark.parametrize(
    "lengths,status",
    [
        ([-1, 1], metadata_module.STATUS_BAD_LENGTH),
        ([1, 71], STATUS_BAD_PADDED_LENGTH),
    ],
)
def test_explicit_lengths_reject_invalid_values(
    host_sequence_metadata, lengths, status
):
    metadata = flash_kda_grad_metadata(
        batch_size=2,
        n_heads_qk=1,
        n_heads_v=1,
        seqlens=70,
        layout="BNSD",
        seqused=torch.tensor(lengths, dtype=torch.int32),
    )
    with pytest.raises(ValueError, match=f"metadata status {status}"):
        metadata_module.metadata_host_info(metadata)


def test_explicit_lengths_guard_compact_token_overflow(metadata_x86):
    # The per-head chunk count fits, but the accumulated token offset does not.
    metadata = metadata_x86(
        None,
        None,
        metadata_capacity(2, 1),
        2,
        1,
        metadata_module.INT32_MAX,
        False,
        seqused=torch.tensor([metadata_module.INT32_MAX, 1], dtype=torch.int32),
    )
    assert (
        metadata[metadata_module.HEADER_STATUS].item()
        == metadata_module.STATUS_BAD_METADATA_CAPACITY
    )


@pytest.mark.parametrize(
    "case", ["dtype", "rank", "count", "stride", "not_tensor", "packed", "both"]
)
def test_explicit_lengths_reject_invalid_host_abi(host_sequence_metadata, case):
    lengths = torch.tensor([0, 65], dtype=torch.int32)
    options = dict(seqused=lengths)
    layout = "BNSD"
    if case == "dtype":
        options["seqused"] = lengths.to(torch.int64)
    elif case == "rank":
        options["seqused"] = lengths[:, None]
    elif case == "count":
        options["seqused"] = lengths[:1]
    elif case == "stride":
        options["seqused"] = torch.tensor([0, 9, 65, 9], dtype=torch.int32)[::2]
    elif case == "not_tensor":
        options["seqused"] = [0, 65]
    elif case == "packed":
        layout = "TND"
    else:
        options["cu_seqlens"] = torch.tensor([0, 0, 65], dtype=torch.int32)
    with pytest.raises(ValueError, match="seqused"):
        flash_kda_grad_metadata(
            batch_size=2,
            n_heads_qk=1,
            n_heads_v=1,
            seqlens=70,
            layout=layout,
            **options,
        )


@pytest.mark.parametrize("layout", ["BNSD", "TND"])
def test_sequence_cache_tracks_mutation_and_tensor_identity(
    host_sequence_metadata, layout
):
    lengths = torch.tensor(
        [0, 0, 65] if layout == "TND" else [0, 65], dtype=torch.int32
    )
    name = "cu_seqlens" if layout == "TND" else "seqused"

    def prepare(tensor):
        return flash_kda_grad_metadata(
            batch_size=2,
            n_heads_qk=1,
            n_heads_v=1,
            seqlens=65,
            layout=layout,
            **{name: tensor},
        )

    original = prepare(lengths)
    assert prepare(lengths) is original
    assert prepare(lengths.clone()) is not original
    lengths[1 if layout == "TND" else 0] = 1
    updated = prepare(lengths)
    assert updated is not original
    assert metadata_module.metadata_host_info(original).sequences[0][1] == 0
    assert metadata_module.metadata_host_info(updated).sequences[0][1] == 1


@pytest.mark.parametrize("batch,value_heads", [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_metadata_capacity_rejects_nonpositive_dimensions(batch, value_heads):
    with pytest.raises(ValueError, match="must be positive"):
        metadata_capacity(batch, value_heads)


def test_metadata_capacity_checks_int32_boundary_without_allocating():
    max_tasks = (metadata_module.INT32_MAX - HEADER_WORDS) // TASK_WORDS
    assert metadata_capacity(1, max_tasks) <= metadata_module.INT32_MAX
    with pytest.raises(ValueError, match="exceeds the int32 ABI"):
        metadata_capacity(1, max_tasks + 1)


def test_clear_metadata_cache_accepts_compiled_kernel_without_close(monkeypatch):
    # Isolate mock state from real AICPU launches in the same pytest process.
    monkeypatch.setattr(metadata_module, "_COMPILED", object())
    monkeypatch.setattr(metadata_module, "_PENDING_EVENTS", [])
    metadata_module.clear_metadata_caches()
    assert metadata_module._COMPILED is None


def test_clear_metadata_cache_waits_for_launches_before_closing(monkeypatch):
    calls = Mock()
    pending_events = [calls.first_event, calls.second_event]
    monkeypatch.setattr(metadata_module, "_COMPILED", calls.kernel)
    monkeypatch.setattr(metadata_module, "_PENDING_EVENTS", pending_events)

    metadata_module.clear_metadata_caches()

    assert [entry[0] for entry in calls.mock_calls] == [
        "first_event.synchronize",
        "second_event.synchronize",
        "kernel.close",
    ]
    assert pending_events == []
    assert metadata_module._COMPILED is None
    # Clearing twice must not close the same executable twice.
    metadata_module.clear_metadata_caches()
    calls.kernel.close.assert_called_once_with()


def test_metadata_consumer_waits_on_the_producer_event(monkeypatch):
    # Make a missing event wait fail even if the real producer would finish early.
    consumer = Mock()
    ready = object()
    metadata = SimpleNamespace(
        device="npu:0", _kda_metadata_ready_event=ready, record_stream=Mock()
    )
    npu = SimpleNamespace(
        device=lambda device: nullcontext(),
        current_device=lambda: 0,
        current_stream=lambda device_index: consumer,
    )
    monkeypatch.setattr(metadata_module.torch, "npu", npu, raising=False)

    metadata_module.wait_metadata_ready(metadata)

    consumer.wait_event.assert_called_once_with(ready)
    metadata.record_stream.assert_called_once_with(consumer)


def _assert_metadata_header(
    metadata, *, batch, value_heads, storage, chunk_count, flags
):
    assert metadata.dtype == torch.int32
    assert metadata.numel() == metadata_capacity(batch, value_heads)
    words = metadata.cpu().tolist()
    # Assert the complete serialized ABI, including magic, version and flags.
    assert words[:HEADER_WORDS] == [
        metadata_module.MAGIC,
        metadata_module.ABI_VERSION,
        STATUS_OK,
        batch,
        value_heads,
        storage,
        chunk_count,
        flags,
    ]
    return words


@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,lengths,query_heads,value_heads",
    [
        ("BNSD", [65, 3], 2, 2),
        ("BSND", [0, 65], 2, 2),
        ("TND", [1, 65, 0], 2, 2),
        pytest.param(
            "TND",
            [0, 63, 64, 65, 128, 129, 0],
            2,
            2,
            marks=pytest.mark.whitebox,
            id="all_chunk_boundaries",
        ),
        pytest.param(
            "BNSD",
            [0, 64],
            2,
            2,
            marks=pytest.mark.whitebox,
            id="empty_then_full_chunk",
        ),
        pytest.param(
            "BSND", [63, 64, 65], 2, 2, marks=pytest.mark.whitebox, id="tail_full_tail"
        ),
        pytest.param("BNSD", [65, 3], 1, 3, marks=pytest.mark.whitebox, id="bnsd_gqa"),
        pytest.param("BSND", [0, 65], 1, 3, marks=pytest.mark.whitebox, id="bsnd_gqa"),
        pytest.param("TND", [1, 65, 0], 1, 3, marks=pytest.mark.whitebox, id="tnd_gqa"),
        pytest.param(
            "BNSD", [0, 0], 2, 2, marks=pytest.mark.whitebox, id="bnsd_all_empty"
        ),
        pytest.param(
            "BSND", [0, 0], 2, 2, marks=pytest.mark.whitebox, id="bsnd_all_empty"
        ),
        pytest.param(
            "TND", [0, 0], 2, 2, marks=pytest.mark.whitebox, id="tnd_all_empty"
        ),
    ],
)
def test_backward_metadata_records(layout, lengths, query_heads, value_heads):
    pytest.importorskip("torch_npu")
    batch = len(lengths)
    storage = sum(lengths) if layout == "TND" else max(lengths)
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    lengths_arg = (
        dict(cu_seqlens=torch.tensor(prefix, dtype=torch.int32, device="npu"))
        if layout == "TND"
        else dict(seqused=torch.tensor(lengths, dtype=torch.int32, device="npu"))
    )
    metadata = flash_kda_grad_metadata(
        batch_size=batch,
        n_heads_qk=query_heads,
        n_heads_v=value_heads,
        seqlens=storage,
        layout=layout,
        **lengths_arg,
    )
    # Flags encode supplied offsets (bit 0) and packed TND storage (bit 1).
    actual = _assert_metadata_header(
        metadata,
        batch=batch,
        value_heads=value_heads,
        storage=storage,
        chunk_count=sum((length + CHUNK_SIZE - 1) // CHUNK_SIZE for length in lengths),
        flags=0b11 if layout == "TND" else 0b01,
    )
    token_start = first_chunk = flat_chunk_start = 0
    for sequence, length in enumerate(lengths):
        chunks = (length + CHUNK_SIZE - 1) // CHUNK_SIZE
        for head in range(value_heads):
            record = HEADER_WORDS + (sequence * value_heads + head) * TASK_WORDS
            assert actual[record : record + TASK_WORDS] == [
                sequence,
                token_start,
                first_chunk,
                chunks,
                length,
                head,
                flat_chunk_start,
            ]
            flat_chunk_start += chunks
        token_start += length
        first_chunk += chunks


@pytest.mark.whitebox
@pytest.mark.npu
@pytest.mark.parametrize("layout", ["BNSD", "BSND"])
def test_backward_metadata_without_offsets_uses_physical_batch_start(layout):
    pytest.importorskip("torch_npu")
    batch, heads, storage = 2, 3, 65
    metadata = flash_kda_grad_metadata(
        batch_size=batch,
        n_heads_qk=heads,
        n_heads_v=heads,
        seqlens=storage,
        layout=layout,
    )
    chunks = (storage + CHUNK_SIZE - 1) // CHUNK_SIZE
    words = _assert_metadata_header(
        metadata,
        batch=batch,
        value_heads=heads,
        storage=storage,
        chunk_count=batch * chunks,
        flags=0,
    )
    for sequence in range(batch):
        for head in range(heads):
            record = HEADER_WORDS + (sequence * heads + head) * TASK_WORDS
            assert words[record : record + TASK_WORDS] == [
                sequence,
                sequence * storage,
                sequence * chunks,
                chunks,
                storage,
                head,
                (sequence * heads + head) * chunks,
            ]


@pytest.mark.whitebox
@pytest.mark.npu
def test_backward_metadata_handoff_between_nondefault_streams():
    pytest.importorskip("torch_npu")
    producer = torch.npu.Stream()
    consumer = torch.npu.Stream()
    producer.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(producer):
        metadata = flash_kda_grad_metadata(
            batch_size=1, n_heads_qk=1, n_heads_v=1, seqlens=65, layout="BNSD"
        )
    assert getattr(metadata, "_kda_metadata_ready_event", None) is not None
    with torch.npu.stream(consumer):
        # The consumer must wait on the producer event before reading metadata.
        metadata_module.wait_metadata_ready(metadata)
        consumed = metadata.clone()
    consumer.synchronize()
    words = _assert_metadata_header(
        consumed,
        batch=1,
        value_heads=1,
        storage=65,
        chunk_count=2,
        flags=0,
    )
    assert words[HEADER_WORDS : HEADER_WORDS + TASK_WORDS] == [0, 0, 0, 2, 65, 0, 0]


@pytest.mark.parametrize(
    "case,expected_message",
    [
        ("layout", "layout must"),
        ("missing_cu", "requires cu_seqlens"),
        ("padded_cu", "only supported for TND"),
        ("batch_zero", "batch_size"),
        ("batch_overflow", "batch_size"),
        ("qk_heads_zero", "n_heads_qk"),
        ("v_heads_zero", "n_heads_v"),
        ("gqa", "divisible"),
        ("length_negative", "seqlens"),
        ("length_overflow", "seqlens"),
        ("storage_overflow", "sequence storage"),
        ("float_attr", "seqlens"),
        ("bool_attr", "n_heads_qk"),
        ("tensor_attr", "batch_size"),
        ("qk_width", "headdim_qk"),
        ("v_width", "headdim_v"),
        ("float_width", "headdim_v"),
        ("offset_dtype", "cu_seqlens must"),
        ("offset_shape", "cu_seqlens must"),
        ("offset_stride", "cu_seqlens must"),
        ("offset_not_tensor", "cu_seqlens must"),
        ("offset_device", "on NPU"),
    ],
)
def test_backward_metadata_host_rejects_invalid_abi_before_launch(
    monkeypatch,
    host_sequence_metadata,
    case,
    expected_message,
):
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    args = dict(batch_size=1, n_heads_qk=1, n_heads_v=1, seqlens=2, cu_seqlens=offsets)
    changed = {
        "layout": dict(layout="BSD"),
        "missing_cu": dict(cu_seqlens=None),
        "padded_cu": dict(layout="BNSD"),
        "batch_zero": dict(batch_size=0),
        "batch_overflow": dict(batch_size=metadata_module.INT32_MAX + 1),
        "qk_heads_zero": dict(n_heads_qk=0),
        "v_heads_zero": dict(n_heads_v=0),
        "gqa": dict(n_heads_qk=2, n_heads_v=3),
        "length_negative": dict(seqlens=-1),
        "length_overflow": dict(seqlens=metadata_module.INT32_MAX + 1),
        "storage_overflow": dict(
            layout="BNSD",
            cu_seqlens=None,
            batch_size=2,
            seqlens=metadata_module.INT32_MAX,
        ),
        "float_attr": dict(seqlens=2.0),
        "bool_attr": dict(n_heads_qk=True),
        "tensor_attr": dict(batch_size=torch.tensor(1)),
        "qk_width": dict(headdim_qk=64),
        "v_width": dict(headdim_v=256),
        "float_width": dict(headdim_v=128.0),
        "offset_dtype": dict(cu_seqlens=offsets.to(torch.int64)),
        "offset_shape": dict(cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32)),
        "offset_stride": dict(
            cu_seqlens=torch.tensor([0, 9, 2, 9], dtype=torch.int32)[::2]
        ),
        "offset_not_tensor": dict(cu_seqlens=[0, 2]),
        "offset_device": {},
    }
    args.update(changed[case])
    if case == "offset_device":
        monkeypatch.setattr(
            torch.Tensor, "device", property(lambda tensor: torch.device("cpu"))
        )
    monkeypatch.setattr(
        metadata_module,
        "_launch_metadata",
        lambda *a, **kw: pytest.fail("invalid metadata ABI reached AICPU"),
    )
    with pytest.raises(ValueError, match=expected_message):
        flash_kda_grad_metadata(**args)


@pytest.mark.whitebox
@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,offsets,expected_status",
    [
        ("TND", [1, 2], STATUS_BAD_CU_START),
        ("TND", [0, 3], STATUS_BAD_PACKED_END),
        ("BNSD", [0, 3], STATUS_BAD_PADDED_LENGTH),
        ("BNSD", [0, -1], metadata_module.STATUS_BAD_LENGTH),
        ("BSND", [0, 3], STATUS_BAD_PADDED_LENGTH),
    ],
)
def test_backward_metadata_rejects_invalid_offsets_on_device(
    layout,
    offsets,
    expected_status,
):
    pytest.importorskip("torch_npu")
    prefix = torch.tensor(offsets, dtype=torch.int32, device="npu")
    options = (
        dict(cu_seqlens=prefix)
        if layout == "TND"
        else dict(seqused=prefix[1:] - prefix[:-1])
    )
    metadata = flash_kda_grad_metadata(
        batch_size=1, n_heads_qk=1, n_heads_v=1, seqlens=2, layout=layout, **options
    )
    assert metadata[metadata_module.HEADER_STATUS].cpu().item() == expected_status


# Tests and helpers from test_flash_kda_grad_scheduling.py.


metadata = load_sample("flash_kda_grad/flash_kda_grad_metadata.py")


def _table(metadata_x86, lengths, heads):
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    return metadata_x86(
        "cpu",
        offsets,
        metadata.metadata_capacity(len(lengths), heads),
        len(lengths),
        heads,
        sum(lengths),
        True,
    )


def _reference(weights, heads, cores):
    """Exhaustive owner enumeration and linear least-load search, without the optimized heap."""
    owners = len(weights) * heads
    modulo = tuple(
        sum(weights[owner // heads] for owner in range(c, owners, cores))
        for c in range(cores)
    )

    def balanced(loads):
        return sum(loads) * 10 >= max(loads) * cores * 9

    if (
        balanced(modulo)
        or owners < cores
        or sum(modulo) * 10 < max(weights) * cores * 9
    ):
        return modulo, modulo, ()
    queues = [[] for _ in range(cores)]
    loads = [0] * cores
    for owner in sorted(
        range(owners), key=lambda owner: (-weights[owner // heads], owner)
    ):
        core = min(
            range(cores), key=lambda core: (loads[core], len(queues[core]), core)
        )
        queues[core].append(owner)
        loads[core] += weights[owner // heads]
    if not balanced(loads):
        return modulo, modulo, ()
    offsets = [0]
    for queue in queues:
        offsets.append(offsets[-1] + len(queue))
    return (
        modulo,
        tuple(loads),
        tuple(offsets + [owner for queue in queues for owner in queue]),
    )


@pytest.mark.parametrize("cores", [1, 2, 4, 16, 32])
def test_metadata_schedule_matches_independent_reference(metadata_x86, cores):
    rng = random.Random(20261027 + cores)
    for _ in range(40):
        heads = rng.choice([1, 2, 3, 8, 16, 29, 30, 31, 32, 96])
        ratio = rng.choice([1, 2, 4])
        lengths = [
            rng.choice([0, 1, 63, 64, 65, 128, 192, 8192])
            for _ in range(rng.randint(1, 12))
        ]
        table = _table(metadata_x86, lengths, heads * ratio)
        weights = [max((n + 63) // 64, 1) * ratio for n in lengths]
        expected = _reference(weights, heads, cores)
        plan = metadata.backward_schedule(table, heads, heads * ratio, cores)
        assert (plan.modulo_loads, plan.loads, plan.words) == expected
        assert plan.strategy == (
            "persistent"
            if sum(plan.loads) * 10 >= max(plan.loads) * cores * 9
            else "staged"
        )
        assert metadata.backward_schedule(table, heads, heads * ratio, cores) is plan


@pytest.mark.parametrize(
    "heads,expected",
    [
        (28, "staged"),
        (29, "persistent"),
        (30, "persistent"),
        (31, "persistent"),
        (32, "persistent"),
        (96, "persistent"),
    ],
)
def test_metadata_preserves_ninety_percent_admission(metadata_x86, heads, expected):
    table = _table(metadata_x86, [8192], heads)
    plan = metadata.backward_schedule(table, heads, heads, 32)
    assert plan.strategy == expected
    assert not plan.words


def test_ragged_t8192_b32_n96_keeps_balanced_modulo(metadata_x86):
    lengths = [
        72,
        362,
        300,
        196,
        160,
        485,
        290,
        356,
        485,
        93,
        107,
        145,
        231,
        32,
        668,
        99,
        359,
        318,
        231,
        311,
        213,
        62,
        92,
        886,
        182,
        83,
        435,
        536,
        22,
        310,
        48,
        23,
    ]
    assert len(lengths) == 32 and sum(lengths) == 8192
    table = _table(metadata_x86, lengths, 96)
    plan = metadata.backward_schedule(table, 96, 96, 32)
    assert plan.strategy == "persistent" and not plan.words
    assert plan.loads == (426,) * 32
    schedule = metadata.owner_schedule_for_launch(table, 96, 96, 32, False)
    assert schedule.numel() == 1 and schedule.data_ptr() == table.data_ptr()


def test_schedule_cache_tracks_metadata_mutation_and_core_count(metadata_x86):
    table = _table(metadata_x86, [192, 64] * 4, 2)
    old = metadata.backward_schedule(table, 1, 2, 4)
    assert old.words and old.strategy == "persistent"
    assert len(metadata.backward_schedule(table, 1, 2, 2).loads) == 2
    table.copy_(_table(metadata_x86, [128] * 8, 2))
    new = metadata.backward_schedule(table, 1, 2, 4)
    assert new is not old and not new.words
    # A different Q/K grouping must not reuse the previous plan.
    grouped = metadata.backward_schedule(table, 2, 2, 4)
    assert grouped is not new and sum(grouped.loads) == sum(new.loads)
    metadata.clear_schedule_caches()
    assert metadata.owner_schedule_for_launch(table, 1, 2, 4, False).numel() == 1


@pytest.mark.parametrize(
    "heads,value_heads,cores",
    [(0, 2, 4), (3, 2, 4), (1, 2, 0), (1, 2, 33), (1, 2, True), (1, 4, 4)],
)
def test_schedule_rejects_incompatible_attributes(
    metadata_x86, heads, value_heads, cores
):
    table = _table(metadata_x86, [64], 2)
    with pytest.raises(ValueError):
        metadata.backward_schedule(table, heads, value_heads, cores)


@pytest.mark.parametrize("cores", [1, 4, 32])
@pytest.mark.parametrize("ratio", [1, 4])
def test_staged_chunk_partition_is_complete_and_keeps_gqa_together(
    metadata_x86, cores, ratio
):
    heads = 3
    table = _table(metadata_x86, [0, 1, 64, 65, 1024], heads * ratio)
    words = table.tolist()
    start = metadata._staged_chunk_start.fn
    for task in range(5 * heads * ratio):
        record = metadata.HEADER_WORDS + task * metadata.TASK_WORDS
        chunks = words[record + metadata.TASK_CHUNKS]
        phase = words[record + metadata.TASK_FIRST_CHUNK]
        if ratio == 1:
            phase += words[record + metadata.TASK_VALUE_HEAD]
        ownership = {}
        for core in range(cores):
            for chunk in range(
                start(words, record, ratio, core, cores), max(chunks, 1), cores
            ):
                assert chunk not in ownership
                ownership[chunk] = core
                assert core == (phase + chunk) % cores
        assert sorted(ownership) == list(range(max(chunks, 1)))
        if ratio > 1:
            group_record = (
                metadata.HEADER_WORDS + task // ratio * ratio * metadata.TASK_WORDS
            )
            assert [start(words, record, ratio, c, cores) for c in range(cores)] == [
                start(words, group_record, ratio, c, cores) for c in range(cores)
            ]


@pytest.mark.parametrize(
    "tasks,cores,parts",
    [(1, 32, 8), (3, 32, 8), (31, 32, 8), (32, 32, 1), (64, 32, 1), (1, 1, 1)],
)
def test_staged_reverse_partition_keeps_each_shard_on_one_core(
    monkeypatch, tasks, cores, parts
):
    assert metadata._staged_reverse_is_unsplit.fn(tasks, cores) == (parts == 1)
    monkeypatch.setattr(metadata, "get_block_num", lambda: cores)
    seen = set()
    for core in range(cores):
        monkeypatch.setattr(metadata, "get_block_idx", lambda: core)
        begin, end, step = metadata._staged_reverse_loop_bounds.fn(tasks, parts)
        for owner in range(begin, end, step):
            task, part = metadata._staged_reverse_owner.fn(owner, parts)
            assert (task, part) not in seen
            seen.add((task, part))
            assert core == owner % cores
    assert seen == {(task, part) for task in range(tasks) for part in range(parts)}


def test_workspace_and_chunk_paths_are_owned_by_metadata(metadata_x86):
    table = _table(metadata_x86, [0, 64, 65], 4)
    assert metadata.backward_gradient_slots(table, 4, 32, False) == 32
    assert metadata.backward_gradient_slots(table, 4, 32, True) == (1 + 2 + 3) * 4
    assert metadata.backward_chunk_paths(table, 4, 4, 64, False) == (True, True)
    assert metadata.backward_chunk_paths(table, 4, 4, 64, True) == (False, True)


@pytest.mark.npu
def test_metadata_owner_schedule_can_be_reused_on_another_stream():
    pytest.importorskip("torch_npu")
    with torch.npu.stream(torch.npu.Stream()):
        table = metadata.flash_kda_grad_metadata(
            seqused=torch.tensor([192, 64] * 4, dtype=torch.int32, device="npu"),
            batch_size=8,
            n_heads_qk=1,
            n_heads_v=2,
            seqlens=192,
            layout="BNSD",
        )
        plan = metadata.backward_schedule(table, 1, 2, 4)
        first = metadata.owner_schedule_for_launch(table, 1, 2, 4, False)
    with torch.npu.stream(torch.npu.Stream()):
        reused = metadata.owner_schedule_for_launch(table, 1, 2, 4, False)
        assert reused is first
        assert reused.cpu().tolist() == list(plan.words)
    metadata.clear_schedule_caches()


# CPU reference checks previously run by ttk/selfcheck/test_reference.py.


_TTK_OUTPUT_NAMES = ("dq", "dk", "dv", "dh0", "dg", "dbeta", "dA", "dBias")


def _ttk_inputs(
    batch=2, heads=2, key_heads=1, length=130, dk=16, dv=12, dtype=torch.float64
):
    torch.manual_seed(4)
    generator = torch.Generator().manual_seed(20260)
    q = torch.randn(batch, key_heads, length, dk, generator=generator, dtype=dtype)
    k = torch.randn(batch, key_heads, length, dk, generator=generator, dtype=dtype)
    v = torch.randn(batch, heads, length, dv, generator=generator, dtype=dtype)
    g = torch.randn(batch, heads, length, dk, generator=generator, dtype=dtype)
    beta = torch.randn(batch, heads, length, generator=generator, dtype=dtype)
    h0 = torch.randn(batch, heads, dv, dk, generator=generator, dtype=dtype) * 0.1
    a_log = torch.randn(heads, generator=generator, dtype=dtype)
    dt_bias = torch.randn(heads, dk, generator=generator, dtype=dtype)
    do = torch.randn(batch, heads, length, dv, generator=generator, dtype=dtype)
    dht = torch.randn(batch, heads, dv, dk, generator=generator, dtype=dtype) * 0.1
    return q, k, v, g, beta, a_log, dt_bias, h0, do, dht


def _ttk_rel(a, b):
    return (a - b).norm() / b.norm().clamp_min(1e-12)


def _check_ttk_reference(
    emulate_bf16: bool, label: str, tol: float, with_golden: bool = False
):
    q, k, v, g, beta, a_log, dt_bias, h0, do, dht = _ttk_inputs()
    kwargs = dict(
        scale=1.0 / (q.shape[-1] ** 0.5),
        gate_mode="bounded",
        lower_bound=-5.0,
        layout="BNSD",
    )
    cache = ttk_golden.build_cache(
        q, k, v, g, beta, h0, a_log, dt_bias, emulate_bf16=emulate_bf16, **kwargs
    )
    staged = ttk_golden.staged_backward(
        q, k, v, g, beta, do, dht, cache, A_log=a_log, dt_bias=dt_bias, **kwargs
    )
    golden = ttk_golden.autograd_backward(
        q, k, v, g, beta, a_log, dt_bias, h0, do, dht, **kwargs
    )

    # dA/dBias leave the operator as FP32 (``A_log``/``dt_bias`` dtype), so their
    # achievable agreement is FP32 round-off even in exact-math mode.
    per_name = {"dA": max(tol, 5e-7), "dBias": max(tol, 5e-7)}
    worst = 0.0
    for name, got, want in zip(_TTK_OUTPUT_NAMES, staged, golden):
        assert got is not None and got.shape == want.shape, (
            name,
            got.shape,
            want.shape,
        )
        err = float(_ttk_rel(got.double(), want.double()))
        worst = max(worst, err)
        limit = per_name.get(name, tol)
        flag = "ok " if err <= limit else "FAIL"
        print(f"  [{label}] {flag} {name:6s} rel_l2={err:.3e}  (<= {limit:.0e})")
        assert err <= limit, f"{label}: {name} rel_l2={err:.3e} > {limit:.1e}"
    print(f"  [{label}] worst rel_l2={worst:.3e}")

    if not with_golden:
        return
    emu = ttk_golden.emulated_autograd_backward(
        q, k, v, g, beta, a_log, dt_bias, h0, do, dht, **kwargs
    )
    tok = ttk_golden.token_golden_backward(
        q, k, v, g, beta, a_log, dt_bias, h0, do, dht, **kwargs
    )
    print(f"  [{label}] ---- golden candidates: vs FP64 truth / vs staged peer")
    for tag, got in (("emu-bf16", emu), ("tok-fp32", tok)):
        for name, g1, peer in zip(_TTK_OUTPUT_NAMES, got, staged):
            want = dict(zip(_TTK_OUTPUT_NAMES, golden))[name]
            print(
                f"  [{label}] {tag:9s} {name:6s} golden-vs-truth={_ttk_rel(g1, want):.3e}"
                f"  golden-vs-peer={_ttk_rel(g1, peer):.3e}"
            )


@pytest.mark.parametrize(
    "emulate_bf16,label,tol,with_golden",
    [
        pytest.param(False, "exact", 1e-10, False, id="exact"),
        pytest.param(True, "bf16", 2e-2, True, id="bf16"),
    ],
)
def test_ttk_reference(emulate_bf16, label, tol, with_golden):
    _check_ttk_reference(emulate_bf16, label, tol, with_golden)


def test_ttk_spec_uses_framework_l2(monkeypatch):
    """The file-path plugin uses only the framework preset, without overrides."""
    assets = Path(__file__).resolve().parents[1] / "ttk"
    monkeypatch.syspath_prepend(str(assets))
    loader = importlib.util.spec_from_file_location(
        "_flash_kda_grad_spec_test", assets / "flash_kda_grad_ttk" / "spec.py"
    )
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    assert module.__spec__ == {"flash_kda_grad_ttk.npu": "FlashKdaGradTestSpec"}
    selected = module.FlashKdaGradTestSpec
    assert selected.third_party == {"torch": "flash_kda_grad_ttk.golden.bf16_peer"}
    assert selected.tolerance == {
        "bfloat16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
    }


def _exception_csv_rows():
    path = (
        Path(__file__).resolve().parents[1] / "ttk/cases/flash_kda_grad_exception.csv"
    )
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


_EXCEPTION_ROWS = _exception_csv_rows()
_HOST_EXCEPTION_ROWS = [
    row
    for row in _EXCEPTION_ROWS
    if json.loads(row["remark"])["mutation"]["stage"]
    not in ("precision_checker", "autograd", "metadata_boundary")
]


class _UnexpectedCompute(RuntimeError):
    pass


def _exception_compute_guard(*args, **kwargs):
    raise _UnexpectedCompute("invalid input reached allocation or compute preparation")


def _check_csv_host_exception(row, monkeypatch, builder, device=None):
    """Apply the catalog mutation to the actual ABI, not to TTK remark text."""
    remark = json.loads(row["remark"])
    mutation = remark["mutation"]
    attrs = ast.literal_eval(row["attributes"])
    shapes = ast.literal_eval(row["tensor_view_shapes"])
    layout = remark["layout"]
    axis = 2 if layout == "BSND" else 1
    args = _make_forward_layout_inputs(
        builder,
        layout,
        remark["lengths"],
        qk_heads=shapes[0][axis],
        value_heads=shapes[2][axis],
    )
    args.update(
        initial_state=torch.zeros_like(args["dht"]),
        disable_recompute=attrs["disable_recompute"],
        allow_neg_eigval=False,
        lower_bound=-0.7,
        scale=0.125,
    )
    if device is not None:
        args = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in args.items()
        }
    # Positive control: the unmodified tensors must satisfy the same validator.
    parameters = inspect.signature(integration._validate_backward_inputs).parameters
    integration._validate_backward_inputs(
        **{k: v for k, v in args.items() if k in parameters}
    )

    def place(tensor):
        return tensor.to(device) if device is not None else tensor

    lengths = remark["lengths"]
    offsets = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    meta_args = dict(
        batch_size=len(lengths),
        n_heads_qk=shapes[0][axis],
        n_heads_v=shapes[2][axis],
        seqlens=sum(lengths) if layout == "TND" else max(lengths),
        layout=layout,
        cu_seqlens=place(torch.tensor(offsets, dtype=torch.int32))
        if layout == "TND"
        else None,
        seqused=place(torch.tensor(lengths, dtype=torch.int32))
        if layout != "TND"
        else None,
    )
    monkeypatch.setattr(integration, "_prepare_stage_two", _exception_compute_guard)
    monkeypatch.setattr(
        integration, "_allocate_recompute_caches", _exception_compute_guard
    )
    target, operation = mutation["target"], mutation["op"]
    if mutation["stage"].startswith("metadata_"):
        if target.startswith("metadata."):
            field = target.split(".", 1)[1]
            if field == "heads":
                meta_args.update(n_heads_qk=mutation["nq"], n_heads_v=mutation["nv"])
            else:
                meta_args[field] = mutation["value"]
        elif operation == "add_cu_seqlens":
            meta_args["cu_seqlens"] = place(
                torch.tensor(mutation["value"], dtype=torch.int32)
            )
        elif operation == "set_none":
            meta_args[target] = None
        elif operation == "cast":
            meta_args[target] = meta_args[target].to(getattr(torch, mutation["dtype"]))
        elif operation == "replace":
            meta_args[target] = place(
                torch.tensor(mutation["value"], dtype=torch.int32)
            )
            if target == "cu_seqlens":
                meta_args["batch_size"] = len(mutation["value"]) - 1
        else:
            meta_args[target] = mutation["value"]
        with pytest.raises(ValueError) as caught:
            metadata = metadata_module.flash_kda_grad_metadata(**meta_args)
            # Length contents are validated by AICPU and rejected by its host consumer.
            metadata_module.metadata_host_info(metadata)
        if mutation.get("expected_status") is not None:
            assert f"metadata status {mutation['expected_status']}" in str(caught.value)
    else:
        if operation == "rename":
            args[mutation["new"]] = args.pop(mutation["old"])
        elif operation == "set_none":
            args[target] = None
        elif operation == "cast":
            for name in ("q", "k") if target == "qk" else (target,):
                args[name] = args[name].to(getattr(torch, mutation["dtype"]))
        elif operation == "make_noncontiguous_same_values":
            value = args[target]
            view = value.new_empty(value.numel() * 2)[::2]
            view.copy_(value)
            args[target] = view
        elif operation == "make_last_dim_stride_2_same_values":
            value = args[target]
            view = value.new_empty((*value.shape[:-1], value.shape[-1] * 2))[..., ::2]
            view.copy_(value)
            args[target] = view
        elif operation == "to_device":
            if device is not None:
                args[target] = args[target].cpu()
            else:

                class HostTensor(torch.Tensor):
                    @property
                    def device(self):
                        return torch.device("cpu")

                args[target] = args[target].as_subclass(HostTensor)
        elif operation == "slice_last_dim":
            args[target] = args[target][..., : mutation["size"]]
        elif target == "metadata.header.magic":
            args["metadata"][metadata_module.HEADER_MAGIC] = mutation["value"]
        elif operation in ("set_batch", "set_heads"):
            value = args[target]
            shape = list(value.shape)
            shape[0 if operation == "set_batch" else axis] = mutation["value"]
            args[target] = value.new_empty(shape)
        else:
            args[target] = mutation["value"]
        if target == "allow_neg_eigval":
            args["use_beta_sigmoid_in_kernel"] = attrs["use_beta_sigmoid_in_kernel"]
        expected = (
            TypeError if remark["expected_exception"] == "TypeError" else ValueError
        )
        with pytest.raises(expected) as caught:
            integration.flash_kda_grad_npu(**args)
    print(
        f"{row['testcase_name']}: {type(caught.value).__name__}: {caught.value}; main compute not entered"
    )


@pytest.mark.whitebox
@pytest.mark.parametrize(
    "row", _HOST_EXCEPTION_ROWS, ids=lambda row: row["testcase_name"]
)
def test_exception_csv_host_contract(host_sequence_metadata, monkeypatch, row):
    _check_csv_host_exception(row, monkeypatch, host_sequence_metadata)


@pytest.mark.whitebox
@pytest.mark.npu
@pytest.mark.parametrize(
    "row", _HOST_EXCEPTION_ROWS, ids=lambda row: row["testcase_name"]
)
def test_exception_csv_npu_contract(monkeypatch, row):
    pytest.importorskip("torch_npu")
    device = torch.device("npu", torch.npu.current_device())

    def metadata_on_device(**kwargs):
        kwargs = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in kwargs.items()
        }
        return metadata_module.flash_kda_grad_metadata(**kwargs)

    builder = SimpleNamespace(flash_kda_grad_metadata=metadata_on_device)
    _check_csv_host_exception(row, monkeypatch, builder, device)


@pytest.mark.whitebox
@pytest.mark.parametrize(
    "number", (29, 30, 31), ids=lambda number: f"kdag_exc_{number:03d}"
)
def test_exception_csv_special_contract(monkeypatch, number):
    if number == 29:
        row = next(
            row for row in _EXCEPTION_ROWS if row["testcase_name"] == "kdag_exc_029"
        )
        shapes = ast.literal_eval(row["tensor_view_shapes"])
        values = [
            None if shape is None else torch.zeros(shape) for shape in shapes[:10]
        ]
        q, k, v, g, beta, a_log, bias, h0, do, dht = values
        if h0 is None:
            h0 = torch.zeros_like(dht)
        cache = ttk_golden.build_cache(
            q,
            k,
            v,
            g,
            beta,
            h0,
            a_log,
            bias,
            layout="BNSD",
            gate_mode="softplus",
            lower_bound=None,
            allow_neg_eigval=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            scale=128**-0.5,
            emulate_bf16=True,
        )
        for value in (float("nan"), float("inf"), -float("inf")):
            seed = do.clone()
            seed.reshape(-1)[0] = value
            actual = ttk_golden.bf16_peer(
                q,
                k,
                v,
                g,
                beta,
                a_log,
                bias,
                h0,
                seed,
                dht,
                layout="BNSD",
                gate_mode="softplus",
                lower_bound=None,
                scale=128**-0.5,
                cache=cache,
            )[0]
            with pytest.raises(AssertionError):
                _assert_precision(
                    "nonfinite gradient", actual, torch.zeros_like(actual)
                )
    elif number == 30:
        q = torch.zeros((17, 1, 128), requires_grad=True)
        tensors = [torch.zeros_like(q, requires_grad=True) for _ in range(3)]
        beta = torch.zeros((17, 1), requires_grad=True)
        h0 = torch.zeros((1, 1, 128, 128), requires_grad=True)
        out, _ = flash_kda_train(
            q,
            *tensors,
            beta,
            h0,
            torch.zeros(1),
            torch.zeros(1, 128),
            layout="TND",
            cu_seqlens=torch.tensor([0, 17], dtype=torch.int32),
        )
        with torch.no_grad():
            q.add_(1)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            out.sum().backward()
    else:
        monkeypatch.setattr(
            metadata_module, "_launch_metadata", _exception_compute_guard
        )
        with pytest.raises(ValueError, match="int32"):
            metadata_module.metadata_capacity(306783378, 1)
        defaults = dict(
            cu_seqlens=None,
            seqused=None,
            batch_size=1,
            n_heads_qk=1,
            n_heads_v=1,
            seqlens=64,
            headdim_qk=128,
            headdim_v=128,
            layout="BNSD",
        )
        for name, value in (("seqlens", 2147483648), ("batch_size", True)):
            with pytest.raises(ValueError, match=name):
                metadata_module._validated_metadata_launch_shape(
                    **{**defaults, name: value}
                )


@pytest.mark.whitebox
@pytest.mark.parametrize(
    "name",
    (
        "q",
        "k",
        "v",
        "g",
        "beta",
        "Aqk",
        "Akk",
        "w",
        "u",
        "v_new",
        "qg",
        "kg",
        "g_cumsum",
        "chunk_states",
        "beta_sigmoid",
        "A_log",
        "dt_bias",
    ),
)
def test_public_rejects_nontensor_arguments(host_sequence_metadata, monkeypatch, name):
    args = _make_host_abi_inputs(gate_dtype=torch.float32)
    args[name] = 1.0
    monkeypatch.setattr(integration, "_prepare_stage_two", _exception_compute_guard)
    with pytest.raises(ValueError):
        integration.flash_kda_grad_npu(disable_recompute=True, **args)


@pytest.mark.whitebox
@pytest.mark.parametrize(
    "scale", ("not_a_number", float("nan"), float("inf"), -float("inf"))
)
def test_invalid_scale_rejected_before_recompute_allocation(
    host_sequence_metadata, monkeypatch, scale
):
    args = _make_host_abi_inputs(gate_dtype=torch.float32)
    args["scale"] = scale
    monkeypatch.setattr(
        integration, "_allocate_recompute_caches", _exception_compute_guard
    )
    with pytest.raises(ValueError):
        integration.flash_kda_grad_npu(disable_recompute=False, **args)


@pytest.mark.whitebox
@pytest.mark.parametrize("name", ("q", "v"))
def test_explicit_lengths_reject_nontensor_arguments(
    host_sequence_metadata, monkeypatch, name
):
    args = _make_host_abi_inputs(gate_dtype=torch.float32)
    args[name] = 1.0
    monkeypatch.setattr(integration, "_prepare_stage_two", _exception_compute_guard)
    with pytest.raises(ValueError, match="must be Tensors"):
        integration.flash_kda_grad_npu(
            disable_recompute=True, **args, seqused=torch.tensor([2], dtype=torch.int32)
        )


@pytest.mark.whitebox
@pytest.mark.parametrize("shape", ((), (128,), (1, 128)))
def test_public_rejects_invalid_value_rank(host_sequence_metadata, monkeypatch, shape):
    args = _make_host_abi_inputs(gate_dtype=torch.float32)
    args["v"] = torch.empty(shape, dtype=torch.bfloat16)
    monkeypatch.setattr(integration, "_prepare_stage_two", _exception_compute_guard)
    with pytest.raises(ValueError, match="V rank"):
        integration.flash_kda_grad_npu(disable_recompute=True, **args)


if __name__ == "__main__":
    if "--compare-bin" in sys.argv:
        raise SystemExit(_bin_compare_main())
    _profile_main()
