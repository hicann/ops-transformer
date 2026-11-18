# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ruff: noqa: F401, F841
# fmt: off
# Preserve the imported golden implementation; migration changes only device code.

"""Pure-PyTorch reference for ``flash_kda_grad`` (no CANN, no CANNBotDSL).

Three independent pieces, deliberately kept separate so that a TTK run can use
them as *golden* and *third-party peer*:

``build_cache``
    Chunkwise forward of KDA, producing exactly the tensors that the NPU
    backward operator consumes as its forward cache (``Aqk``/``Akk``/``w``/
    ``u``/``v_new``/``qg``/``kg``/``g_cumsum``/``chunk_states``/normalized Q-K).
    Cache dtypes and the BF16 rounding points mirror the operator ABI, so the
    same cache can be fed to both the NPU kernel and ``staged_backward``.

``staged_backward``
    Explicit reverse pass over the chunk formulas (stage 2 + stage 1), reading
    the *same* BF16 cache as the kernel. Used as the TTK third-party peer: it
    rounds every Cube operand to BF16 exactly where the kernel does, so it has
    the same quantization budget as the device kernel, which is what
    ``cross_check`` needs (a peer whose error is comparable, not zero).

``autograd_backward``
    Token-level recurrence differentiated by ``torch.autograd`` in float64.
    Used as the TTK golden: an implementation-independent, high-precision
    truth, sharing no code path with ``staged_backward``.

Transcribed from the forward cache and ``flash_kda_grad`` reference now kept in
``test/flash_kda_grad/ut/test_flash_kda_grad_ut.py``. Keep the two in sync when
the math changes.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from threading import local
from typing import Optional, Tuple

import torch

# --- causal pair loop ------------------------------------------------------
# The CPU peer batches independent causal-pair products by row and scripts
# the remaining ordered updates to reduce Python dispatch. Keep exp and sum
# at their original shapes and preserve the update order at each destination.
# Disable JIT fusion to retain the original multiply/add rounding points.
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)


def _pair_loop_impl(
    g_cumsum: torch.Tensor,
    dAqk_scaled: torch.Tensor,
    dlower_beta: torch.Tensor,
    dq_chunk: torch.Tensor,
    dk_chunk: torch.Tensor,
    q_chunk: torch.Tensor,
    k_chunk: torch.Tensor,
    dbeta_chunk: torch.Tensor,
    dlower_operand: torch.Tensor,
    dg_cumsum: torch.Tensor,
    length: int,
    gate_ratio: str,
):
    # Batch independent products within a row. Keep exp and sum on the
    # original per-pair vectors and preserve the addition order at each
    # destination (including row == col aliasing).
    for row in range(length):
        end = row + 1
        gamma_row = g_cumsum[row]
        ratios = torch.jit.annotate(list[torch.Tensor], [])
        if gate_ratio == "factorized":
            gamma_exp = gamma_row.exp()
            for col in range(end):
                ratios.append(gamma_exp / g_cumsum[col].exp())
        elif gate_ratio == "clamped":
            for col in range(end):
                ratios.append((gamma_row - g_cumsum[col]).clamp(min=-20.0).exp())
        else:
            for col in range(end):
                ratios.append((gamma_row - g_cumsum[col]).exp())
        decay = torch.stack(ratios)
        k_cols = k_chunk[:end]
        k_row = k_chunk[row]
        da = dAqk_scaled[row, :end].unsqueeze(-1)
        dl_beta = dlower_beta[row, :end].unsqueeze(-1)
        dq_terms = da * k_cols * decay
        dk_row_terms = dl_beta * k_cols * decay
        dk_col_terms = (dl_beta * k_row + da * q_chunk[row]) * decay
        beta_products = k_row * k_cols * decay
        dg_terms = dk_col_terms * k_cols
        dq_row = dq_chunk[row]
        dk_row = dk_chunk[row]
        dbeta_row = dbeta_chunk[row]
        dg_row = dg_cumsum[row]
        dl_row = dlower_operand[row]
        for col in range(end):
            dq_row.add_(dq_terms[col])
            dk_row.add_(dk_row_terms[col])
            dbeta_row.add_(dl_row[col] * beta_products[col].sum())
            dg_row.add_(dg_terms[col])
        # Off-diagonal columns update distinct rows. Apply the diagonal
        # last: its destination aliases the row updated in the loop.
        if row > 0:
            dk_chunk[:row].add_(dk_col_terms[:row])
            dg_cumsum[:row].sub_(dg_terms[:row])
        dk_row.add_(dk_col_terms[row])
        dg_row.sub_(dg_terms[row])
    return dq_chunk, dk_chunk, dbeta_chunk, dg_cumsum


try:
    _pair_loop = torch.jit.script(_pair_loop_impl)
except Exception:  # pragma: no cover - keep the inline loop if scripting fails
    _pair_loop = None


# Numba is optional. The ordered TorchScript path also works without it.
try:
    import numpy as np
    from numba import njit
except ImportError:
    njit = None


def _pair_differences(gamma, diff, first, end, clamped):
    i = 0
    for r in range(first, end):
        for c in range(r + 1):
            for j in range(gamma.shape[1]):
                x = gamma[r, j] - gamma[c, j]
                diff[i, j] = np.float32(-20) if clamped and x < -20 else x
            i += 1
    return i


def _pair_update(da, db, dq, dk, q, k, dg, decay, products, first, end):
    i = 0
    for r in range(first, end):
        qr, kr = q[r], k[r]
        dqr, dkr, dgr = dq[r], dk[r], dg[r]
        for c in range(r):
            kc, dkc, dgc = k[c], dk[c], dg[c]
            ratios, prod = decay[i], products[i]
            a, b = da[r, c], db[r, c]
            for j in range(k.shape[1]):
                ratio = ratios[j]
                kval = kc[j]
                dqr[j] += (a * kval) * ratio
                dkr[j] += (b * kval) * ratio
                p = (b * kr[j] + a * qr[j]) * ratio
                dkc[j] += p
                prod[j] = (kr[j] * kval) * ratio
                g = p * kval
                dgr[j] += g
                dgc[j] -= g
            i += 1

        a, b = da[r, r], db[r, r]
        ratios, prod = decay[i], products[i]
        for j in range(k.shape[1]):
            ratio, kval = ratios[j], kr[j]
            dqr[j] += (a * kval) * ratio
            p = (b * kval + a * qr[j]) * ratio
            dkr[j] = (dkr[j] + (b * kval) * ratio) + p
            prod[j] = (kval * kval) * ratio
            g = p * kval
            dgr[j] = (dgr[j] + g) - g
        i += 1


def _pair_update_beta(beta, dl, sums, first, end):
    i = 0
    for r in range(first, end):
        for c in range(r + 1):
            beta[r] += dl[r, c] * sums[i]
            i += 1


_pair_scratch = local()


@torch.inference_mode(False)
def _pair_workspace(capacity, width):
    if getattr(_pair_scratch, "shape", None) != (capacity, width):
        df = np.empty((capacity, width), dtype=np.float32)
        dr, pr = np.empty_like(df), np.empty_like(df)
        sm = np.empty(capacity, dtype=np.float32)
        arrays = (df, dr, pr, sm)
        _pair_scratch.buffers = (*arrays, *map(torch.from_numpy, arrays))
        _pair_scratch.shape = capacity, width
    return _pair_scratch.buffers


def _numba_pair(gamma, da, db, dq, dk, q, k, beta, dl, dg, length, mode):
    # Reuse a workspace for the largest chunk (64 tokens) across heads.
    capacity = 16 * max(length, 64)
    df, dr, pr, sm, diff, decay, products, sums = _pair_workspace(capacity, k.size(1))
    g, a, b, dq_a, dk_a, q_a, k_a, beta_a, dl_a, dg_a = (
        t.numpy() for t in (gamma, da, db, dq, dk, q, k, beta, dl, dg)
    )
    for first in range(0, length, 16):
        end = min(first + 16, length)
        n = _pair_differences(g, df, first, end, mode == "clamped")
        torch.exp(diff[:n], out=decay[:n])
        _pair_update(a, b, dq_a, dk_a, q_a, k_a, dg_a, dr, pr, first, end)
        torch.sum(products[:n], dim=1, out=sums[:n])
        _pair_update_beta(beta_a, dl_a, sm, first, end)
    return dq, dk, beta, dg


if njit is not None:
    _pair_differences = njit(fastmath=False)(_pair_differences)
    _pair_update = njit(fastmath=False)(_pair_update)
    _pair_update_beta = njit(fastmath=False)(_pair_update_beta)


def _run_cpu_pair(*args):
    if (
        njit is not None
        and args[-1] != "factorized"
        and all(
            t.device.type == "cpu" and t.dtype == torch.float32
            and t.layout == torch.strided and not t.requires_grad
            for t in args[:10]
        )
    ):
        return _numba_pair(*args)
    return _pair_loop(*args)


import torch.utils.checkpoint
import torch.nn.functional as F

CHUNK_SIZE = 64
NORM_EPS = 1e-6

__all__ = [
    "Cache",
    "build_cache",
    "staged_backward",
    "autograd_backward",
    "token_golden_backward",
    "emulated_autograd_backward",
    "bf16_peer",
    "token_major",
    "restore_layout",
]


# ---------------------------------------------------------------------------
# Layout helpers (BNSD / BSND only; TND stays out of the TTK entry signature)
# ---------------------------------------------------------------------------


def token_major(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    """[B,N,S,...] (BNSD) or [B,S,N,...] (BSND) -> [B*S, N, ...] token major."""
    if layout == "TND":
        return tensor
    if layout == "BNSD":
        return tensor.transpose(1, 2).reshape(-1, tensor.shape[1], *tensor.shape[3:]).contiguous()
    if layout == "BSND":
        return tensor.reshape(-1, tensor.shape[2], *tensor.shape[3:]).contiguous()
    raise ValueError(f"unsupported layout: {layout}")


def restore_layout(tensor: torch.Tensor, like: torch.Tensor, layout: str) -> torch.Tensor:
    """[B*S, N, ...] -> the layout/shape of ``like``."""
    if layout == "TND":
        return tensor.reshape_as(like)
    if layout == "BNSD":
        batch, heads, length = like.shape[0], like.shape[1], like.shape[2]
        return (
            tensor.reshape(batch, length, heads, *tensor.shape[2:])
            .transpose(1, 2)
            .contiguous()
        )
    return tensor.reshape_as(like)


def _sequence_bounds(length: int, batch: int) -> list[tuple[int, int]]:
    return [(index * length, (index + 1) * length) for index in range(batch)]


def resolve_initial_state(q, v, initial_state, layout, cu_seqlens=None):
    """Materialize the operator's None-as-zero state for CPU cache/golden math."""
    if initial_state is not None:
        return initial_state
    if layout == "TND":
        if cu_seqlens is None:
            raise ValueError("TND requires cu_seqlens to infer the logical batch")
        batch = cu_seqlens.numel() - 1
        heads = v.shape[1]
    elif layout in ("BNSD", "BSND"):
        batch = q.shape[0]
        heads = v.shape[1 if layout == "BNSD" else 2]
    else:
        raise ValueError(f"unsupported layout: {layout}")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    return torch.zeros(batch, heads, v.shape[-1], q.shape[-1], dtype=dtype, device=q.device)


def normalize_backward_optionals(q, v, initial_state, dt_bias, dht, layout, cu_seqlens):
    """Prepare optional terminal gradients and channelwise bias for reference math."""
    if dht is None:
        state = resolve_initial_state(q, v, initial_state, layout, cu_seqlens)
        dht = torch.zeros_like(state)
    if dt_bias is not None and dt_bias.ndim == 1:
        dt_bias = dt_bias.reshape(v.shape[2 if layout == "BSND" else 1], q.shape[-1])
    return dt_bias, dht


def restore_bias_gradient(grads, dt_bias):
    if dt_bias is not None and dt_bias.ndim == 1 and grads[-1] is not None:
        return (*grads[:-1], grads[-1].reshape_as(dt_bias))
    return grads


def sequence_bounds(q, state, layout, cu_seqlens=None, seqused=None):
    """Validate sequence metadata and return token-major storage intervals."""
    batch = state.shape[0]
    if layout not in ("TND", "BNSD", "BSND"):
        raise ValueError(f"unsupported layout: {layout}")
    if cu_seqlens is not None and seqused is not None:
        raise ValueError("cu_seqlens and seqused are mutually exclusive")
    lengths = cu_seqlens if cu_seqlens is not None else seqused
    if lengths is not None:
        if lengths.dtype != torch.int32 or lengths.ndim != 1 or not lengths.is_contiguous():
            raise ValueError("sequence metadata must be a contiguous rank-1 int32 tensor")
        values = lengths.detach().cpu().tolist()
    if layout == "TND":
        if cu_seqlens is None or seqused is not None:
            raise ValueError("TND requires cu_seqlens and does not accept seqused")
        if len(values) != batch + 1 or values[0] != 0 or values[-1] != q.shape[0] or any(a > b for a,b in zip(values,values[1:])):
            raise ValueError("cu_seqlens must describe all TND tokens and state batches")
        return list(zip(values, values[1:]))
    if cu_seqlens is not None:
        raise ValueError("BNSD/BSND accepts seqused, not cu_seqlens")
    storage = q.shape[2 if layout == "BNSD" else 1]
    if q.shape[0] != batch:
        raise ValueError("input batch must match state batch")
    used = [storage] * batch if seqused is None else values
    if len(used) != batch or any(n < 0 or n > storage for n in used):
        raise ValueError("seqused must contain B lengths in [0, S]")
    return [(i * storage, i * storage + n) for i,n in enumerate(used)]


# ---------------------------------------------------------------------------
# Forward cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cache:
    """Forward tensors in the operator's cache ABI (token major where applicable)."""

    out: torch.Tensor
    final_state: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    q_rstd: torch.Tensor
    k_rstd: torch.Tensor
    beta_sigmoid: torch.Tensor
    Aqk: torch.Tensor
    Akk: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    qg: torch.Tensor
    kg: torch.Tensor
    v_new: torch.Tensor
    g_cumsum: torch.Tensor
    chunk_states: torch.Tensor
    beta_effective: torch.Tensor


def _gate_beta(
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    *,
    high_dtype: torch.dtype,
    gate_mode: str,
    lower_bound: float,
    allow_neg_eigval: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (per-step log decay, effective beta) for one chunk/head."""
    if use_gate_in_kernel:
        alpha = A_log.exp().to(high_dtype)
        biased = raw_g + (0.0 if dt_bias is None else dt_bias.to(high_dtype))
        if gate_mode == "softplus":
            step_gate = -alpha * F.softplus(biased)
        else:
            step_gate = float(lower_bound) * torch.sigmoid(alpha * biased)
    else:
        # The kernel imposes no sign restriction on an unfused gate: the caller
        # supplies the per-step log decay directly, positive values included.
        step_gate = raw_g

    if use_beta_sigmoid_in_kernel:
        factor = 2.0 if allow_neg_eigval else 1.0
        beta_effective = torch.sigmoid(beta.to(high_dtype)) * factor
    else:
        beta_effective = beta.to(high_dtype)
    return step_gate, beta_effective


@dataclass(frozen=True)
class _Step:
    """One (sequence, head, chunk) step at the operator's storage rounding points."""

    out: torch.Tensor
    state: torch.Tensor
    m: torch.Tensor
    inverse: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    qg: torch.Tensor
    kg: torch.Tensor
    corrected: torch.Tensor
    gate: torch.Tensor
    beta_effective: torch.Tensor


def _bf16_operand(x: torch.Tensor) -> torch.Tensor:
    """Round one MMAD operand the way the Cube consumes it: a single BF16 value.

    Every Cube product in the kernel is fed BF16 L1/L0 operands, so an FP32
    intermediate that reaches a matmul loses the same bits the device loses.
    Cache tensors are already BF16-exact, so this is a no-op for them; it only
    perturbs the FP32 intermediates the device really re-quantises.
    """
    return x.to(torch.bfloat16).to(torch.float32)


def _split_operand(x: torch.Tensor) -> torch.Tensor:
    """Round a *state-gradient* operand: BF16 high part + BF16 residual.

    ``StageTwoMatmul.compute_token_state_split`` feeds ``dH`` as two MMADs
    (high and residual), which keeps ~14 effective mantissa bits. The kernel
    uses that split only for the ``d_kg`` product; every other product reads a
    plain BF16 operand (see :func:`_bf16_operand`).
    """
    high = x.to(torch.bfloat16).to(torch.float32)
    return high + (x - high).to(torch.bfloat16).to(torch.float32)


def _bf16_store(x: torch.Tensor) -> torch.Tensor:
    """Round-trip through a single BF16 value (what a BF16 buffer store does)."""
    return x.to(torch.bfloat16).to(torch.float32)


@torch.jit.script
def _ordered_mm(left: torch.Tensor, right: torch.Tensor, result: torch.Tensor, tile: int):
    product = torch.empty_like(result)
    for start in range(0, left.size(1), tile):
        end = min(start + tile, left.size(1))
        torch.mm(left[:, start:end], right[start:end, :], out=product)
        result.add_(product)
    return result


def _cube_matmul(left, right, *, quantize: bool, initial=None, k_tile: int = 16):
    """Emulate Cube BF16 operands and FP32 L0C accumulation by K tiles."""
    if not quantize:
        product = left @ right
        return product if initial is None else initial + product
    if initial is None:
        result = torch.zeros(
            (left.shape[0], right.shape[1]),
            dtype=torch.promote_types(left.dtype, right.dtype),
            device=left.device,
        )
    else:
        result = initial.clone()
    reuse = (
        left.device.type == "cpu"
        and not left.requires_grad
        and not right.requires_grad
        and not result.requires_grad
    )
    if reuse:
        return _ordered_mm(left, right, result, k_tile)
    for k_start in range(0, left.shape[1], k_tile):
        k_end = min(k_start + k_tile, left.shape[1])
        result = result + left[:, k_start:k_end] @ right[k_start:k_end, :]
    return result


def _storage_dtype(storage_dtype: Optional[torch.dtype]) -> torch.dtype:
    """Storage dtype of the forward cache.

    Fixed by the operator ABI (BF16), **not** by the input dtype: TTK's
    ``cross_check`` promotes inputs to FP32 for the golden, and the device
    still stores BF16. Deriving it from ``q.dtype`` would silently turn the
    emulation off under ``Promote``.
    """
    return torch.bfloat16 if storage_dtype is None else storage_dtype


def _adjacent_square_sum(values):
    """Match the staged Vector reduction tree before BF16 normalization."""
    partials = values.square()
    while partials.shape[-1] > 1:
        partials = partials.reshape(*partials.shape[:-1], -1, 2).sum(-1)
    return partials[..., 0]


def _normalized_qk(
    q_tokens, k_tokens, *, use_l2norm, high_dtype, low_dtype,
    emulate_npu_forward=False,
):
    if not use_l2norm:
        return q_tokens, k_tokens, None, None, None, None
    if emulate_npu_forward and q_tokens.shape[-1] == 128:
        q_square_sum = _adjacent_square_sum(q_tokens)
        k_square_sum = _adjacent_square_sum(k_tokens)
    else:
        q_square_sum = q_tokens.square().sum(-1)
        k_square_sum = k_tokens.square().sum(-1)
    q_rstd = torch.rsqrt(q_square_sum + NORM_EPS)
    k_rstd = torch.rsqrt(k_square_sum + NORM_EPS)
    q_norm = (q_tokens * q_rstd[..., None]).to(low_dtype)
    k_norm = (k_tokens * k_rstd[..., None]).to(low_dtype)
    return q_norm.to(high_dtype), k_norm.to(high_dtype), q_norm, k_norm, q_rstd, k_rstd


def _head_chunks(token_count, batch, value_heads, key_heads, chunk_size, bounds=None):
    """Yield ``(seq, head, qk_head, chunks, chunk_base)`` — one loop skeleton."""
    sequence_length = token_count // batch
    heads_per_key = value_heads // key_heads
    base = 0
    for seq_index, (seq_start, seq_end) in enumerate(
        (_sequence_bounds(sequence_length, batch) if bounds is None else bounds)
    ):
        chunks = [
            (start, min(start + chunk_size, seq_end))
            for start in range(seq_start, seq_end, chunk_size)
        ]
        for head in range(value_heads):
            yield seq_index, head, head // heads_per_key, chunks, base
        base += len(chunks)


def _chunk_step(
    Q,
    K,
    V,
    B,
    raw_g,
    a_log_head,
    dt_bias_head,
    state,
    *,
    scale,
    high_dtype,
    low_dtype,
    gate_mode,
    lower_bound,
    allow_neg_eigval,
    use_gate_in_kernel,
    use_beta_sigmoid_in_kernel,
    emulate_npu_forward=False,
    persistent_forward=False,
    emit_output=True,
    advance_state=True,
):
    """Chunkwise KDA step; every ABI-stored tensor is rounded to ``low_dtype``.

    Rounding is a straight-through cast (``x.to(low).to(high)``), so the same
    code is usable both as a value-only cache builder and inside an autograd
    graph.
    """
    step_gate, b_eff = _gate_beta(
        raw_g,
        B,
        a_log_head,
        dt_bias_head,
        high_dtype=high_dtype,
        gate_mode=gate_mode,
        lower_bound=lower_bound,
        allow_neg_eigval=allow_neg_eigval,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
    )
    if emulate_npu_forward:
        # Vector updates the prefix one token at a time in FP32.
        prefix = torch.zeros_like(step_gate[0])
        rows = []
        for row in step_gate:
            prefix = prefix + row
            rows.append(prefix)
        gate = torch.stack(rows)
    else:
        gate = step_gate.cumsum(0)
    length = Q.shape[0]

    if emulate_npu_forward and use_gate_in_kernel and gate_mode != "softplus":
        # Bounded-gate Cube scores cast each decay-weighted operand before MMAD.
        shared = not persistent_forward and not allow_neg_eigval and abs(scale) <= 0.125
        scores, kk_pairs = _forward_cube_scores(Q, K, gate, scale, shared)
    elif emulate_npu_forward and not use_gate_in_kernel:
        scores, kk_pairs = _forward_safe_cube_scores(Q, K, gate, scale)
    else:
        delta = (gate[:, None, :] - gate[None, :, :]).clamp(max=0.0)
        ratio = delta.exp()
        scores = (Q[:, None, :] * K[None, :, :] * ratio).sum(-1) * float(scale)
        kk_pairs = (K[:, None, :] * K[None, :, :] * ratio).sum(-1)
    m = scores.tril().to(low_dtype)
    lower = torch.tril(b_eff[:, None] * kk_pairs, diagonal=-1)
    if emulate_npu_forward and not use_gate_in_kernel:
        # The forward kernel stores beta-weighted KKT in FP16 before inversion.
        lower = lower.to(torch.float16).to(high_dtype)

    identity = torch.eye(length, dtype=high_dtype, device=Q.device)
    if length == 1:
        inverse = identity.to(low_dtype)
    elif emulate_npu_forward and not use_gate_in_kernel and not persistent_forward:
        inverse = _forward_staged_inverse(lower).to(low_dtype)
    else:
        inverse = torch.linalg.solve_triangular(
            identity + lower, identity, upper=False, unitriangular=True
        ).to(low_dtype)
    p = inverse.to(high_dtype)

    gate_exp = gate.exp()
    k_tilde = gate_exp * K
    weighted_k = b_eff[:, None] * k_tilde
    weighted_v = b_eff[:, None] * V
    if emulate_npu_forward:
        weighted_k = _bf16_operand(weighted_k)
        weighted_v = _bf16_operand(weighted_v)
    w = (p @ weighted_k).to(low_dtype)
    u_fp32 = p @ weighted_v
    u = u_fp32.to(low_dtype)
    qg = (gate_exp * Q).to(low_dtype)
    kg = ((gate[-1:] - gate).exp() * K).to(low_dtype)

    state_operand = _bf16_operand(state) if emulate_npu_forward else state
    # The staged forward reloads U from GM. Persistent keeps the C3 result
    # in L0C while accumulating (-W) @ H into the same tile.
    u_operand = (
        u_fp32 if emulate_npu_forward and persistent_forward else u.to(high_dtype)
    )
    corrected = (u_operand - w.to(high_dtype) @ state_operand).to(low_dtype)
    out = (
        float(scale) * qg.to(high_dtype) @ state_operand
        + m.to(high_dtype) @ corrected.to(high_dtype)
    ).to(low_dtype) if emit_output else None
    if advance_state:
        state = gate[-1].exp()[:, None] * state + kg.to(high_dtype).T @ corrected.to(high_dtype)
    return _Step(out, state, m, inverse, w, u, qg, kg, corrected, gate, b_eff)


def _forward_cube_scores(q, k, gate, scale, shared):
    length = q.shape[0]
    scores = q.new_zeros((length, length))
    kk = torch.zeros_like(scores)
    for start in range(0, length, 16):
        rows = slice(start, min(start + 16, length))
        # The staged shared path uses a midpoint for each 32-row panel.
        reference = min(start // 32 * 32 + 16, length - 1) if shared else start
        left_decay = (gate[rows] - gate[reference]).exp()
        columns = slice(0, rows.stop)
        right_decay = (gate[reference] - gate[columns]).exp()
        right = _bf16_operand(k[columns] * right_decay).T
        scores[rows, columns] = _bf16_operand(q[rows] * left_decay * scale) @ right
        kk[rows, columns] = _bf16_operand(k[rows] * left_decay) @ right
    return scores, kk


def _forward_safe_cube_scores(q, k, gate, scale):
    """Model the BF16 Cube operands for externally prepared gate values."""
    length = q.shape[0]
    scores = q.new_zeros((length, length))
    kk = torch.zeros_like(scores)
    for start in range(0, length, 16):
        stop = min(start + 16, length)
        rows = slice(start, stop)
        if start:
            left_decay = (gate[rows] - gate[start]).double().exp().to(q.dtype)
            right = _bf16_operand(k[:start] * (gate[start] - gate[:start]).double().exp().to(q.dtype)).T
            scores[rows, :start] = _bf16_operand(q[rows] * left_decay * scale) @ right
            kk[rows, :start] = _bf16_operand(k[rows] * left_decay) @ right
        # The diagonal tile uses each active row's own gate as the reference.
        for row in range(start, stop):
            columns = slice(start, row + 1)
            right = _bf16_operand(k[columns] * (gate[row] - gate[columns]).double().exp().to(q.dtype)).T
            scores[row, columns] = _bf16_operand(q[row] * scale) @ right
            kk[row, columns] = k[row] @ right
    return scores, kk


def _forward_staged_inverse(lower):
    """Match the staged forward's FP16 handoffs in its 16/32/64 inverse."""
    length = lower.shape[0]
    negative = lower.new_zeros((CHUNK_SIZE, CHUNK_SIZE))
    negative[:length, :length] = -lower
    inverse = torch.zeros_like(negative)

    def fp16(value):
        return value.to(torch.float16).to(lower.dtype)

    for start in range(0, CHUNK_SIZE, 16):
        block = slice(start, start + 16)
        value = fp16(negative[block, block])
        diagonal = torch.eye(16, dtype=lower.dtype, device=lower.device) + value
        power2 = value @ value
        diagonal = diagonal + fp16(diagonal) @ fp16(power2)
        power4 = fp16(power2) @ fp16(power2)
        diagonal = diagonal + fp16(diagonal) @ fp16(power4)
        power8 = fp16(power4) @ fp16(power4)
        inverse[block, block] = diagonal + fp16(diagonal) @ fp16(power8)

    for size in (16, 32):
        for start in range(0, CHUNK_SIZE, 2 * size):
            left = slice(start, start + size)
            right = slice(start + size, start + 2 * size)
            partial = fp16(negative[right, left]) @ fp16(inverse[left, left])
            inverse[right, left] = fp16(inverse[right, right]) @ fp16(partial)
    return inverse[:length, :length]


def _forward_pieces(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    A_log,
    dt_bias,
    *,
    scale,
    gate_mode,
    lower_bound,
    allow_neg_eigval,
    layout,
    use_qk_l2norm_in_kernel,
    use_gate_in_kernel,
    use_beta_sigmoid_in_kernel,
    storage_dtype,
    chunk_size,
    high_dtype,
    emulate_npu_forward=False,
):
    """Shared prologue: token-major views, normalized Q/K, iteration metadata."""
    q_tokens = token_major(q, layout).to(high_dtype)
    k_tokens = token_major(k, layout).to(high_dtype)
    v_tokens = token_major(v, layout).to(high_dtype)
    g_tokens = token_major(g, layout).to(high_dtype)
    beta_tokens = token_major(beta, layout).to(high_dtype)
    q_eff, k_eff, q_norm, k_norm, q_rstd, k_rstd = _normalized_qk(
        q_tokens,
        k_tokens,
        use_l2norm=use_qk_l2norm_in_kernel,
        high_dtype=high_dtype,
        low_dtype=_storage_dtype(storage_dtype),
        emulate_npu_forward=emulate_npu_forward,
    )
    token_count, value_heads, value_dim = v_tokens.shape
    key_heads, key_dim = q_tokens.shape[1], q_tokens.shape[-1]
    batch = initial_state.shape[0]
    ctx = dict(
        q_eff=q_eff,
        k_eff=k_eff,
        v_tokens=v_tokens,
        g_tokens=g_tokens,
        beta_tokens=beta_tokens,
        q_norm=q_norm,
        k_norm=k_norm,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        token_count=token_count,
        value_heads=value_heads,
        value_dim=value_dim,
        key_heads=key_heads,
        key_dim=key_dim,
        batch=batch,
    )
    return ctx


def build_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    *,
    scale: float,
    gate_mode: str = "bounded",
    lower_bound: float = -5.0,
    allow_neg_eigval: bool = False,
    layout: str = "BNSD",
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    chunk_size: int = CHUNK_SIZE,
    emulate_bf16: bool = True,
    storage_dtype: Optional[torch.dtype] = None,
    _peer_only: bool = False,
    emulate_npu_forward: bool = False,
) -> Cache:
    """Chunkwise forward producing the backward cache.

    Math runs in FP32 (FP64 for FP64 inputs). The reference-cache route rounds
    the ABI buffers, which can then be shared by the device and CPU backward.
    The forward integration peer also rounds the operands consumed by Cube.
    """
    initial_state = resolve_initial_state(q, v, initial_state, layout, cu_seqlens)
    high_dtype = (
        torch.float64
        if q.dtype == torch.float64 and not emulate_npu_forward
        else torch.float32
    )
    low_dtype = high_dtype if not emulate_bf16 else _storage_dtype(storage_dtype)

    ctx = _forward_pieces(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        A_log,
        dt_bias,
        scale=scale,
        gate_mode=gate_mode,
        lower_bound=lower_bound,
        allow_neg_eigval=allow_neg_eigval,
        layout=layout,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        storage_dtype=low_dtype,
        chunk_size=chunk_size,
        high_dtype=high_dtype,
        emulate_npu_forward=emulate_npu_forward,
    )
    q_eff = ctx["q_eff"]
    k_eff = ctx["k_eff"]
    v_tokens = ctx["v_tokens"]
    g_tokens = ctx["g_tokens"]
    beta_tokens = ctx["beta_tokens"]
    token_count = ctx["token_count"]
    value_heads = ctx["value_heads"]
    value_dim = ctx["value_dim"]
    key_heads = ctx["key_heads"]
    key_dim = ctx["key_dim"]
    batch = ctx["batch"]

    bounds = sequence_bounds(q, initial_state, layout, cu_seqlens, seqused)
    # The NPU forward scheduler explicitly selects persistent for short
    # multi-sequence TND training. The peer uses the staged model otherwise.
    persistent_forward = (
        emulate_npu_forward
        and layout == "TND"
        and len(bounds) > 1
        and max(end - start for start, end in bounds) <= 2 * chunk_size
    )
    chunk_count = sum(
        (end - start + chunk_size - 1) // chunk_size
        for start, end in bounds
    )
    shape_tokens = (token_count, value_heads)
    Aqk = torch.zeros(chunk_count, value_heads, chunk_size, chunk_size, dtype=low_dtype, device=q.device)
    Akk = torch.zeros(Aqk.shape, dtype=low_dtype, device=q.device)
    w = torch.zeros(*shape_tokens, key_dim, dtype=low_dtype, device=q.device)
    u = torch.zeros(*shape_tokens, value_dim, dtype=low_dtype, device=q.device)
    qg = torch.zeros(w.shape, dtype=low_dtype, device=w.device)
    kg = torch.zeros(w.shape, dtype=low_dtype, device=w.device)
    v_new = torch.zeros(u.shape, dtype=low_dtype, device=u.device)
    g_cumsum = torch.zeros(*shape_tokens, key_dim, dtype=high_dtype, device=q.device)
    chunk_states = torch.zeros(
        chunk_count, value_heads, value_dim, key_dim, dtype=low_dtype, device=q.device
    )
    output = torch.zeros(token_count, value_heads, value_dim, dtype=low_dtype, device=v.device)
    final_state = torch.empty_like(initial_state)
    beta_effective = torch.zeros(*shape_tokens, dtype=high_dtype, device=q.device)
    beta_sigmoid = torch.zeros(*shape_tokens, dtype=high_dtype, device=q.device)

    common = dict(
        scale=scale,
        high_dtype=high_dtype,
        low_dtype=low_dtype,
        gate_mode=gate_mode,
        lower_bound=lower_bound,
        allow_neg_eigval=allow_neg_eigval,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
    )
    for seq_index, head, qk_head, chunks, chunk_base in _head_chunks(
        token_count, batch, value_heads, key_heads, chunk_size, bounds
    ):
        state = initial_state[seq_index, head].to(high_dtype).T.contiguous()
        for local_chunk, (start, end) in enumerate(chunks):
            length = end - start
            chunk = chunk_base + local_chunk
            B = beta_tokens[start:end, head]
            # Public saved states are BF16; peer must consume the same rounded
            # cache as backward. Keep the live recurrence state in high_dtype.
            # Exact-math reference mode retains high_dtype via low_dtype above.
            chunk_states[chunk, head] = state.T
            step = _chunk_step(
                q_eff[start:end, qk_head],
                k_eff[start:end, qk_head],
                v_tokens[start:end, head],
                B,
                g_tokens[start:end, head],
                A_log[head] if A_log is not None else None,
                dt_bias[head] if dt_bias is not None else None,
                state,
                emulate_npu_forward=emulate_npu_forward,
                persistent_forward=persistent_forward,
                emit_output=not _peer_only,
                advance_state=not _peer_only or local_chunk + 1 < len(chunks),
                **common,
            )
            state = step.state
            Aqk[chunk, head, :length, :length] = step.m
            Akk[chunk, head, :length, :length] = step.inverse
            w[start:end, head] = step.w
            u[start:end, head] = step.u
            qg[start:end, head] = step.qg
            kg[start:end, head] = step.kg
            v_new[start:end, head] = step.corrected
            g_cumsum[start:end, head] = step.gate
            beta_effective[start:end, head] = step.beta_effective
            beta_sigmoid[start:end, head] = (
                torch.sigmoid(B) * (2.0 if allow_neg_eigval else 1.0)
                if use_beta_sigmoid_in_kernel
                else B
            )
            if not _peer_only:
                output[start:end, head] = step.out
        final_state[seq_index, head] = state.T

    return Cache(
        out=restore_layout(output, v, layout),
        final_state=final_state,
        q_norm=ctx["q_norm"],
        k_norm=ctx["k_norm"],
        q_rstd=ctx["q_rstd"],
        k_rstd=ctx["k_rstd"],
        beta_sigmoid=beta_sigmoid,
        Aqk=Aqk,
        Akk=Akk,
        w=w,
        u=u,
        qg=qg,
        kg=kg,
        v_new=v_new,
        g_cumsum=g_cumsum,
        chunk_states=chunk_states,
        beta_effective=beta_effective,
    )


# ---------------------------------------------------------------------------
# Differentiable chunkwise forward (golden path: CPU emulating the device)
# ---------------------------------------------------------------------------


def _forward_graph(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    A_log,
    dt_bias,
    *,
    scale,
    gate_mode="bounded",
    lower_bound=-5.0,
    allow_neg_eigval=False,
    layout="BNSD",
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    storage_dtype=None,
    chunk_size=CHUNK_SIZE,
):
    """Same math as :func:`build_cache`, but keeps the autograd graph.

    No in-place writes into pre-allocated tensors (that would cut the graph);
    per-head rows are concatenated instead. Plain fp32 math with BF16 storage
    rounding, i.e. what the device does.
    """
    high_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    low_dtype = _storage_dtype(storage_dtype)
    ctx = _forward_pieces(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        A_log,
        dt_bias,
        scale=scale,
        gate_mode=gate_mode,
        lower_bound=lower_bound,
        allow_neg_eigval=allow_neg_eigval,
        layout=layout,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        storage_dtype=low_dtype,
        chunk_size=chunk_size,
        high_dtype=high_dtype,
    )
    q_eff, k_eff = ctx["q_eff"], ctx["k_eff"]
    v_tokens, g_tokens, beta_tokens = ctx["v_tokens"], ctx["g_tokens"], ctx["beta_tokens"]
    token_count, value_heads = ctx["token_count"], ctx["value_heads"]
    key_heads, batch = ctx["key_heads"], ctx["batch"]

    common = dict(
        scale=scale,
        high_dtype=high_dtype,
        low_dtype=low_dtype,
        gate_mode=gate_mode,
        lower_bound=lower_bound,
        allow_neg_eigval=allow_neg_eigval,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
    )
    rows = [[] for _ in range(value_heads)]
    finals = [[None] * value_heads for _ in range(batch)]
    for seq_index, head, qk_head, chunks, _base in _head_chunks(
        token_count, batch, value_heads, key_heads, chunk_size
    ):
        state = initial_state[seq_index, head].to(high_dtype).T.contiguous()
        for start, end in chunks:
            step = _chunk_step(
                q_eff[start:end, qk_head],
                k_eff[start:end, qk_head],
                v_tokens[start:end, head],
                beta_tokens[start:end, head],
                g_tokens[start:end, head],
                A_log[head] if A_log is not None else None,
                dt_bias[head] if dt_bias is not None else None,
                state,
                **common,
            )
            state = step.state
            rows[head].append(step.out)
        finals[seq_index][head] = state.T

    out = torch.stack([
        torch.cat(head_rows, dim=0) if head_rows
        else v_tokens.new_zeros((0, v_tokens.shape[-1]))
        for head_rows in rows
    ], dim=1)
    final_state = torch.stack([torch.stack(head_rows, dim=0) for head_rows in finals], dim=0)
    return restore_layout(out, v, layout), final_state


def emulated_autograd_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    *,
    scale: float,
    gate_mode: str = "bounded",
    lower_bound: float = -5.0,
    allow_neg_eigval: bool = False,
    layout: str = "BNSD",
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    storage_dtype: Optional[torch.dtype] = None,
    chunk_size: int = CHUNK_SIZE,
):
    """Golden: autograd through the BF16-storage chunkwise forward.

    This is the same construction as ``test_flash_kda_grad_ut``'s CPU autograd path
    (``_FlashKDATrain.backward`` for non-NPU tensors): the forward keeps the
    operator's BF16 rounding points, and the gradients come from
    ``torch.autograd`` over that graph. It therefore answers "does the device
    backward match the derivative of the *same* cache the device consumed",
    instead of comparing a BF16 device against an FP64 ideal.
    """
    initial_state = resolve_initial_state(q, v, initial_state, layout)
    if use_gate_in_kernel and A_log is None:
        raise ValueError("A_log is required when gate fusion is enabled")
    # Leaves stay in the caller's layout: _forward_graph token-majors internally
    # and returns output in that same layout.
    leaves = [
        t.detach().to(torch.float32).requires_grad_(True)
        for t in (q, k, v, g, beta)
    ]
    a_log = A_log.to(torch.float32).requires_grad_(True) if use_gate_in_kernel else None
    bias = (dt_bias.to(torch.float32).requires_grad_(True)
            if use_gate_in_kernel and dt_bias is not None else None)
    h0 = initial_state.to(torch.float32).requires_grad_(True)

    out, final_state = _forward_graph(
        leaves[0],
        leaves[1],
        leaves[2],
        leaves[3],
        leaves[4],
        h0,
        a_log,
        bias,
        scale=scale,
        gate_mode=gate_mode,
        lower_bound=lower_bound,
        allow_neg_eigval=allow_neg_eigval,
        layout=layout,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        storage_dtype=storage_dtype,
        chunk_size=chunk_size,
    )
    # ``out`` is already restored to the caller's layout; ``do``/``dht`` are
    # given in that same layout, so the loss is layout-agnostic.
    loss = (out.to(torch.float32) * do.to(torch.float32)).sum() + (
        final_state * dht.to(torch.float32)
    ).sum()
    all_leaves = [*leaves, a_log, bias, h0]
    active_leaves = [t for t in all_leaves if t is not None]
    loss = loss + sum(t.sum() * 0 for t in active_leaves)
    active_grads = iter(torch.autograd.grad(loss, active_leaves, allow_unused=True))
    grads = []
    for tensor in all_leaves:
        grad = next(active_grads) if tensor is not None else None
        grads.append(torch.zeros_like(tensor) if tensor is not None and grad is None else grad)
    dq, dk, dv, dg, dbeta, d_a_log, d_bias, dh0 = grads
    # Gradients are already shaped like their leaves (caller's layout).
    return (
        dq.to(q.dtype),
        dk.to(k.dtype),
        dv.to(v.dtype),
        dh0.to(torch.float32),
        dg.to(g.dtype),
        dbeta.to(beta.dtype),
        None if d_a_log is None else d_a_log.to(torch.float32),
        None if d_bias is None else d_bias.to(torch.float32),
    )


# ---------------------------------------------------------------------------
# Explicit staged backward (third-party peer: same BF16 cache as the kernel)
# ---------------------------------------------------------------------------


def _persistent_peer_plan(bounds, key_heads, value_heads, cores=32):
    """Mirror metadata's modulo/greedy 90% owner admission on the CPU."""
    import heapq
    weights = [max((end-start+63)//64, 1) * (value_heads//key_heads) for start,end in bounds]
    loads = [0] * cores
    for seq, weight in enumerate(weights):
        for head in range(key_heads):
            loads[(seq*key_heads+head) % cores] += weight
    def balanced(values):
        return sum(values)*10 >= max(values)*len(values)*9
    if balanced(loads):
        return True
    if len(bounds)*key_heads < cores or sum(loads)*10 < max(weights)*cores*9:
        return False
    heap = [(0,0,core) for core in range(cores)]
    for seq in sorted(range(len(weights)), key=lambda i: (-weights[i],i)):
        for head in range(key_heads):
            load,count,core = heap[0]
            heapq.heapreplace(heap,(load+weights[seq],count+1,core))
    return balanced([x[0] for x in heap])


def _cube_causal_delta(q, k, gamma, beta, da, dl, scale):
    """Full-chunk Cube causal path, including coefficient/storage rounding."""
    forward = (gamma-gamma[:1]).exp()
    inverse = (gamma[:1]-gamma).exp()
    output_coeff = _bf16_operand((float(scale)*da).tril())
    lower_coeff = _bf16_operand(dl.tril(diagonal=-1))
    def split(x):
        high = _bf16_operand(x)
        return high, _bf16_operand(x-high)
    qh,ql = split(q*forward)
    kh,kl = split((k*forward)*beta[:,None])
    ih,il = split(k*inverse)
    def product(a, high, low, initial=None):
        result = _cube_matmul(a,high,quantize=True,initial=initial)
        return _cube_matmul(a,low,quantize=True,initial=result)
    dq_exp = product(output_coeff,ih,il)
    db_matrix = product(lower_coeff,ih,il)
    column = product(output_coeff.T,qh,ql)
    column = product(lower_coeff.T,kh,kl,initial=column)
    dq = forward*dq_exp
    dk_row = (forward*db_matrix)*beta[:,None]
    dk_column = inverse*column
    dk = dk_row+dk_column
    dg = (q*dq+k*dk_row)-k*dk_column
    terms = (forward*k)*db_matrix
    db = terms[:,:64].sum(-1)+terms[:,64:].sum(-1)
    return dq, dk, dg, db


def _short_softplus_sensitivity(q, k, v, gamma, beta, state, do, dht, scale):
    """FP32 short-softplus dA/dbeta path used by SingleChunkGateSensitivity.

    Differentiate the unrounded unit-lower solve using saved normalized Q/K,
    cumulative gate and chunk-entry state, as the device's sensitivity phase
    does. This path deliberately does not use the rounded WY/value caches.
    """
    with torch.enable_grad():
        perturbation = gamma.new_zeros((), requires_grad=True)
        effective_beta = beta.detach().clone().requires_grad_(True)
        gate = gamma.detach() * (1 + perturbation)
        ratio = (gate[:, None] - gate[None, :]).exp()
        kk = (k[:, None] * k[None, :] * ratio).sum(-1)
        qk = float(scale) * (q[:, None] * k[None, :] * ratio).sum(-1)
        lower = torch.eye(len(gate), dtype=gate.dtype, device=gate.device)
        lower = lower + torch.tril(effective_beta[:, None] * kk, diagonal=-1)
        rhs = effective_beta[:, None] * (v - (k * gate.exp()) @ state.T)
        corrected = torch.linalg.solve_triangular(
            lower, rhs, upper=False, unitriangular=True
        )
        output = float(scale) * (q * gate.exp()) @ state.T + qk.tril() @ corrected
        final = state * gate[-1].exp()[None, :] + corrected.T @ (
            k * (gate[-1] - gate).exp()
        )
        loss = (output * do).sum() + (final * dht).sum()
        da, db = torch.autograd.grad(loss, (perturbation, effective_beta))
    return da.detach(), db.detach()


def staged_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    cache: Cache,
    *,
    scale: float,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    layout: str = "BNSD",
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    gate_mode: str = "bounded",
    lower_bound: float = -5.0,
    allow_neg_eigval: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    device_quant: Optional[bool] = None,
    r_source: str = "cache",
    gate_ratio: str = "exp_diff",
    inverse_source: str = "cache",
    gate_grad_bf16: bool = False,
    p_scale: str = "none",
    v_path_bf16: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Return (dq, dk, dv, dh0, dg, dbeta, dA_log, ddt_bias) given the forward cache.

    Every operand of a Cube product is rounded to BF16 exactly where the kernel
    rounds it (``_bf16_operand``), and the ``d_kg`` state operand keeps the
    kernel's two-MMAD high/residual split (``_split_operand``). Without that
    emulation the peer keeps FP32 products on a BF16 cache and its error is
    1-2 orders of magnitude smaller than the device's.

    ``device_quant=None`` derives the mode from the cache: a BF16 cache means
    the caller emulates the device (the TTK peer and the host replay of
    ``npu``), an FP32 cache means the exact-math self-check, which must stay
    free of quantization noise.

    ``A_log`` / ``dt_bias`` are only needed for the fused-gate parameter gradients.
    """
    # Cache names match forward; *_chunk denotes the current token slice.
    # lower is the strict-lower system matrix: Akk = (I + lower)^-1.
    # dstate_out is carried backward; dchunk_state is the entry-state gradient.
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    # Apply the device operand rounding without changing cache names.
    quantize = (
        bool(cache.Aqk is not None and cache.Aqk.dtype == torch.bfloat16)
        if device_quant is None
        else bool(device_quant)
    )
    round_operand = _bf16_operand if quantize else (lambda tensor: tensor)
    split_operand = _split_operand if quantize else (lambda tensor: tensor)
    q_tokens = token_major(q, layout)
    k_tokens = token_major(k, layout)
    v_tokens = token_major(v, layout)
    g_tokens = token_major(g, layout)
    beta_tokens = token_major(beta, layout)
    do_tokens = token_major(do, layout)
    q_effective = (cache.q_norm if use_qk_l2norm_in_kernel else q_tokens).to(dtype)
    k_effective = (cache.k_norm if use_qk_l2norm_in_kernel else k_tokens).to(dtype)
    beta_effective = cache.beta_effective.to(dtype)

    token_count, value_heads, value_dim = v_tokens.shape
    key_heads, key_dim = q_effective.shape[1], q_effective.shape[-1]
    batch = dht.shape[0]

    dq_heads = torch.zeros(token_count, key_heads, key_dim, dtype=dtype, device=q.device)
    dk_heads = torch.zeros_like(dq_heads)
    dv = torch.zeros(token_count, value_heads, value_dim, dtype=dtype, device=q.device)
    dg = torch.zeros(token_count, value_heads, key_dim, dtype=dtype, device=q.device)
    dbeta = torch.zeros(token_count, value_heads, dtype=dtype, device=q.device)
    dA_log = torch.zeros(value_heads, dtype=dtype, device=q.device) if use_gate_in_kernel else None
    ddt_bias = (
        torch.zeros(value_heads, key_dim, dtype=dtype, device=q.device)
        if use_gate_in_kernel and dt_bias is not None
        else None
    )
    dh0 = torch.zeros(batch, value_heads, value_dim, key_dim, dtype=dtype, device=q.device)

    sequence_length = token_count // batch
    bounds = sequence_bounds(q, dht, layout, cu_seqlens, seqused)
    cube_causal = (quantize and use_gate_in_kernel and gate_mode != "softplus"
                   and key_heads == value_heads and abs(lower_bound)*63 <= 70
                   and _persistent_peer_plan(bounds, key_heads, value_heads))
    chunk_size = cache.Aqk.shape[-1]
    chunk_base = 0

    for seq_index, (seq_start, seq_end) in enumerate(bounds):
        chunks = [
            (start, min(start + chunk_size, seq_end))
            for start in range(seq_start, seq_end, chunk_size)
        ]
        for head in range(value_heads):
            qk_head = head // (value_heads // key_heads)
            dstate_out = (
                torch.zeros(value_dim, key_dim, dtype=dtype, device=q.device)
                if dht is None
                else dht[seq_index, head].to(dtype).T.contiguous()
            )
            for local_chunk in range(len(chunks) - 1, -1, -1):
                start, end = chunks[local_chunk]
                length = end - start
                chunk = chunk_base + local_chunk

                q_chunk = q_effective[start:end, qk_head]
                k_chunk = k_effective[start:end, qk_head]
                v_chunk = v_tokens[start:end, head].to(dtype)
                g_cumsum = cache.g_cumsum[start:end, head].to(dtype)
                beta_chunk = beta_effective[start:end, head]
                chunk_state = cache.chunk_states[chunk, head].to(dtype).T
                w = cache.w[start:end, head].to(dtype)
                if r_source == "reconstruct":
                    # Diagnostic path: reconstruct v_new = u - w @ chunk_state
                    # instead of reading the saved v_new cache.
                    v_new = cache.u[start:end, head].to(dtype) - round_operand(
                        w
                    ) @ round_operand(chunk_state)
                elif r_source == "reconstruct_bf16":
                    # The kernel stores the recomputed v_new back to BF16.
                    v_new = _bf16_operand(
                        cache.u[start:end, head].to(dtype)
                        - round_operand(w) @ round_operand(chunk_state)
                    ).to(dtype)
                else:
                    v_new = cache.v_new[start:end, head].to(dtype)
                qg = cache.qg[start:end, head].to(dtype)
                kg = cache.kg[start:end, head].to(dtype)
                Aqk = cache.Aqk[chunk, head, :length, :length].to(dtype)
                if inverse_source == "recompute":
                    # Diagnostic path: rebuild Akk = (I + lower)^-1 from
                    # k_chunk/beta_chunk instead of reading the saved Akk.
                    gate_decay_pairs = (
                        (g_cumsum[:, None, :] - g_cumsum[None, :, :]).clamp(max=0.0).exp()
                    )
                    kk_pairs = (k_chunk[:, None, :] * k_chunk[None, :, :] * gate_decay_pairs).sum(
                        -1
                    )
                    lower = torch.tril(beta_chunk[:, None] * kk_pairs, diagonal=-1)
                    eye = torch.eye(length, dtype=dtype, device=q_chunk.device)
                    Akk = (
                        torch.linalg.solve_triangular(
                            eye + lower, eye, upper=False, unitriangular=True
                        )
                        .to(torch.bfloat16)
                        .to(dtype)
                    )
                else:
                    Akk = cache.Akk[chunk, head, :length, :length].to(dtype)
                do_chunk = do_tokens[start:end, head].to(dtype)

                do_operand = round_operand(do_chunk)
                state_operand = round_operand(chunk_state)
                v_operand = round_operand(v_new)
                # Stage 2: consumers of corrected values and of the prior state.
                # ``d_v_new`` is the staged pipeline product (``kg @ dH`` plus
                # ``Aqk.T @ do``); it reads ``dH`` as a single BF16 operand.
                dv_new = _cube_matmul(
                    round_operand(Aqk).T, do_operand, quantize=quantize
                ) + _cube_matmul(
                    round_operand(kg), round_operand(dstate_out), quantize=quantize
                )
                dv_operand = round_operand(dv_new)
                dAqk = torch.tril(
                    _cube_matmul(
                        do_operand, v_operand.T, quantize=quantize
                    )
                )
                dqg = float(scale) * _cube_matmul(
                    do_operand, state_operand.T, quantize=quantize
                )
                # ``compute_token_state_split`` is the one product that feeds
                # dH as BF16 high + BF16 residual (two MMADs).
                if quantize:
                    dstate_high = round_operand(dstate_out)
                    dstate_residual = round_operand(dstate_out - dstate_high)
                    v_new_operand = v_operand
                    dkg = _cube_matmul(v_new_operand, dstate_high.T, quantize=True)
                    dkg = _cube_matmul(
                        v_new_operand, dstate_residual.T, quantize=True, initial=dkg
                    )
                else:
                    dkg = _cube_matmul(
                        v_operand, split_operand(dstate_out).T, quantize=False
                    )
                dstate_decay = (dstate_out * chunk_state).sum(-1)
                dchunk_state = (
                    float(scale) * _cube_matmul(
                        round_operand(qg).T, do_operand, quantize=quantize
                    )
                    + g_cumsum[-1].exp()[:, None] * dstate_out
                    - _cube_matmul(
                        round_operand(w).T, dv_operand, quantize=quantize
                    )
                )
                dw = -_cube_matmul(
                    dv_operand, state_operand.T, quantize=quantize
                )
                du = dv_new

                # Stage 1B: WY system, unit-lower inverse, relative-gate pairs.
                k_exp_g = g_cumsum.exp() * k_chunk
                if p_scale == "beta_row":
                    Akk_used = Akk * beta_chunk[:, None]
                elif p_scale == "beta_col":
                    Akk_used = Akk * beta_chunk[None, :]
                else:
                    Akk_used = Akk
                if v_path_bf16:
                    du_used = _bf16_store(du).to(dtype)
                    dw_used = _bf16_store(dw).to(dtype)
                else:
                    du_used, dw_used = du, dw
                dAkk = _cube_matmul(
                    round_operand(du_used),
                    round_operand(beta_chunk[:, None] * v_chunk).T,
                    quantize=quantize,
                )
                dAkk = _cube_matmul(
                    round_operand(dw_used),
                    round_operand(beta_chunk[:, None] * k_exp_g).T,
                    quantize=quantize,
                    initial=dAkk,
                )
                akk_wy_operand = round_operand(Akk_used).T
                du_wy_operand = round_operand(du_used)
                dw_wy_operand = round_operand(dw_used)
                d_beta_v = _cube_matmul(
                    akk_wy_operand, du_wy_operand, quantize=quantize
                )
                d_beta_k_exp_g = _cube_matmul(
                    akk_wy_operand, dw_wy_operand, quantize=quantize
                )
                # NPU keeps d_beta_v/d_beta_k in FP32. WY residuals come from
                # the pre-cast StageTwo inputs and are added only to the local
                # beta reduction; the dv epilogue consumes the uncorrected result.
                dv[start:end, head] += beta_chunk[:, None] * d_beta_v
                if quantize and length == chunk_size:
                    d_beta_v_for_beta = d_beta_v + (du - _bf16_store(du))
                    d_beta_k_for_beta = d_beta_k_exp_g + (dw - _bf16_store(dw))
                else:
                    d_beta_v_for_beta = d_beta_v
                    d_beta_k_for_beta = d_beta_k_exp_g
                dbeta_chunk = (d_beta_v_for_beta * v_chunk).sum(-1) + (d_beta_k_for_beta * k_exp_g).sum(-1)
                dAkk_used = _bf16_store(dAkk).to(dtype) if v_path_bf16 else dAkk
                # ``compute_inverse_chain`` rounds the FP32 L0C product to BF16
                # in L1 before the second MMAD, so the intermediate is an
                # operand too, not just the two ends of the chain.
                akk_operand = round_operand(Akk_used)
                dlower_head = _cube_matmul(
                    akk_operand.T, round_operand(dAkk_used), quantize=quantize
                )
                dlower = torch.tril(
                    -_cube_matmul(
                        round_operand(dlower_head), akk_operand.T, quantize=quantize
                    ),
                    diagonal=-1,
                )
                dk_exp_g = beta_chunk[:, None] * d_beta_k_exp_g
                state_decay_to_end = (g_cumsum[-1:] - g_cumsum).exp()
                dq_chunk = g_cumsum.exp() * dqg
                dk_chunk = g_cumsum.exp() * dk_exp_g + state_decay_to_end * dkg
                dg_cumsum = qg * dqg + k_exp_g * dk_exp_g - kg * dkg
                dg_cumsum[-1] += g_cumsum[-1].exp() * dstate_decay + (kg * dkg).sum(0)

                # The causal pairs are a Vector contraction in the kernel
                # (``DirectCausalDeltaVector``): ``dAqk``/``d_lower`` reach it
                # as FP32 and are consumed in FP32, so no BF16 operand cast.
                dAqk_scaled = float(scale) * dAqk
                dlower_beta = beta_chunk[:, None] * dlower
                dlower_operand = dlower
                if cube_causal and length == CHUNK_SIZE:
                    pair_q, pair_k, pair_g, pair_b = _cube_causal_delta(
                        q_chunk, k_chunk, g_cumsum, beta_chunk, dAqk, dlower, scale
                    )
                    dq_chunk = dq_chunk + pair_q
                    dk_chunk = dk_chunk + pair_k
                    dg_cumsum = dg_cumsum + pair_g
                    dbeta_chunk = dbeta_chunk + pair_b
                else:
                    if _pair_loop is not None and g_cumsum.device.type == "cpu":
                        dq_chunk, dk_chunk, dbeta_chunk, dg_cumsum = _run_cpu_pair(
                            g_cumsum,
                            dAqk_scaled,
                            dlower_beta,
                            dq_chunk,
                            dk_chunk,
                            q_chunk,
                            k_chunk,
                            dbeta_chunk,
                            dlower_operand,
                            dg_cumsum,
                            length,
                            gate_ratio,
                        )
                    else:
                        for row in range(length):
                            for col in range(row + 1):
                                if gate_ratio == "factorized":
                                    gate_decay_ratio = g_cumsum[row].exp() / g_cumsum[col].exp()
                                elif gate_ratio == "clamped":
                                    gate_decay_ratio = (
                                        (g_cumsum[row] - g_cumsum[col]).clamp(min=-20.0).exp()
                                    )
                                else:
                                    gate_decay_ratio = (g_cumsum[row] - g_cumsum[col]).exp()
                                dAqk_coeff = dAqk_scaled[row, col]
                                dlower_beta_coeff = dlower_beta[row, col]
                                dq_chunk[row] += dAqk_coeff * k_chunk[col] * gate_decay_ratio
                                dk_chunk[row] += dlower_beta_coeff * k_chunk[col] * gate_decay_ratio
                                dk_col_pair = (
                                    dlower_beta_coeff * k_chunk[row] + dAqk_coeff * q_chunk[row]
                                ) * gate_decay_ratio
                                dk_chunk[col] += dk_col_pair
                                dbeta_chunk[row] += (
                                    dlower_operand[row, col]
                                    * (k_chunk[row] * k_chunk[col] * gate_decay_ratio).sum()
                                )
                                dg_cumsum_pair = dk_col_pair * k_chunk[col]
                                dg_cumsum[row] += dg_cumsum_pair
                                dg_cumsum[col] -= dg_cumsum_pair

                dq_heads[start:end, qk_head] += dq_chunk
                dk_heads[start:end, qk_head] += dk_chunk
                dstep_gate = torch.flip(torch.cumsum(torch.flip(dg_cumsum, (0,)), dim=0), (0,))
                if gate_grad_bf16:
                    dstep_gate = _bf16_store(dstep_gate).to(dtype)

                raw_g_chunk = g_tokens[start:end, head].to(dtype)
                if use_gate_in_kernel:
                    if A_log is None:
                        raise ValueError("A_log is required for fused-gate parameter gradients")
                    dt_bias_head = (
                        torch.zeros(key_dim, dtype=dtype, device=q.device)
                        if dt_bias is None
                        else dt_bias[head].to(dtype)
                    )
                    alpha = A_log[head].to(dtype).exp()
                    if gate_mode == "softplus":
                        biased_gate = raw_g_chunk + dt_bias_head
                        dg_chunk = -dstep_gate * alpha * torch.sigmoid(biased_gate)
                        step_gate = -alpha * F.softplus(biased_gate)
                        softplus_da = (dstep_gate * step_gate).sum()
                        dA_log[head] += softplus_da
                    else:
                        scaled_gate = alpha * (raw_g_chunk + dt_bias_head)
                        gate_sigmoid = torch.sigmoid(scaled_gate)
                        dscaled_gate = (
                            dstep_gate * float(lower_bound) * gate_sigmoid * (1 - gate_sigmoid)
                        )
                        dg_chunk = alpha * dscaled_gate
                        dA_log[head] += (dscaled_gate * scaled_gate).sum()
                    dg[start:end, head] += dg_chunk
                    if ddt_bias is not None:
                        ddt_bias[head] += dg_chunk.sum(0)
                else:
                    dg[start:end, head] += dstep_gate

                if use_beta_sigmoid_in_kernel:
                    beta_factor = 2.0 if allow_neg_eigval else 1.0
                    beta_sigmoid = torch.sigmoid(beta_tokens[start:end, head].to(dtype))
                    dbeta[start:end, head] += (
                        dbeta_chunk * beta_factor * beta_sigmoid * (1 - beta_sigmoid)
                    )
                else:
                    dbeta[start:end, head] += dbeta_chunk
                if quantize and use_gate_in_kernel and gate_mode == "softplus" and len(chunks) == 1:
                    sensitivity_da, sensitivity_db = _short_softplus_sensitivity(
                        q_chunk, k_chunk, v_chunk, g_cumsum, beta_chunk,
                        chunk_state.T, do_chunk, dht[seq_index, head].to(dtype), scale,
                    )
                    dA_log[head] += sensitivity_da - softplus_da
                    if use_beta_sigmoid_in_kernel:
                        sensitivity_db *= beta_chunk * (
                            1 - beta_chunk * (0.5 if allow_neg_eigval else 1.0)
                        )
                    dbeta[start:end, head] = sensitivity_db
                dstate_out = dchunk_state
            dh0[seq_index, head] = dstate_out.T
        chunk_base += len(chunks)

    if use_qk_l2norm_in_kernel:
        q_rstd = cache.q_rstd[..., None].to(dtype)
        k_rstd = cache.k_rstd[..., None].to(dtype)
        dq_heads = q_rstd * (
            dq_heads - q_effective * (q_effective * dq_heads).sum(-1, keepdim=True)
        )
        dk_heads = k_rstd * (
            dk_heads - k_effective * (k_effective * dk_heads).sum(-1, keepdim=True)
        )

    return (
        restore_layout(dq_heads.to(q.dtype), q, layout),
        restore_layout(dk_heads.to(k.dtype), k, layout),
        restore_layout(dv.to(v.dtype), v, layout),
        dh0,
        restore_layout(dg.to(g.dtype), g, layout),
        restore_layout(dbeta.to(beta.dtype), beta, layout),
        None if dA_log is None else dA_log.to(torch.float32),
        None if ddt_bias is None else ddt_bias.to(torch.float32),
    )


# ---------------------------------------------------------------------------
# High-precision autograd golden (token recurrence, FP64)
# ---------------------------------------------------------------------------



class _GoldenProgress:
    """Throttled CPU golden progress; no tensor synchronization is needed."""

    def __init__(self, total):
        self.total = total
        self.done = 0
        self.start = self.last_log = time.monotonic()
        self.next_percent = 0.0
        self.last_done = -1
        self.update(0, force=True)

    @staticmethod
    def duration(seconds):
        seconds = max(0, int(seconds))
        return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"

    def update(self, count, force=False):
        self.done += count
        if self.done == self.last_done:
            return
        now = time.monotonic()
        percent = 100.0 if self.total == 0 else min(100.0, 100.0 * self.done / self.total)
        if not force and self.done < self.total:
            if percent < self.next_percent and now - self.last_log < 30.0:
                return
        elapsed = now - self.start
        eta = (self.duration(elapsed * (self.total - self.done) / self.done)
               if 0 < self.done < self.total else
               "00:00:00" if self.done >= self.total else "--:--:--")
        filled = min(20, int(percent / 5.0))
        print(
            f"[CPU Golden] recurrence [{'#' * filled}{'-' * (20 - filled)}] "
            f"{percent:6.2f}% ({self.done}/{self.total} token-heads), "
            f"elapsed={self.duration(elapsed)}, ETA={eta}",
            flush=True,
        )
        self.last_log, self.last_done = now, self.done
        self.next_percent = min(100.0, (math.floor(percent / 5.0) + 1) * 5.0)


# Chunkwise reference for the KDA recurrence (ported from cannbot-dsl PR395).
# The maths is the same token-level recurrence as before; evaluating it chunk-major
# keeps the Python interpreter out of the inner loop, which is what made the 8K
# cases take close to an hour. Retain the selected reference dtype throughout.
_CHUNK_SIZE = 64
_REFERENCE_BLOCK_SIZE = 16


def _chunk_major(x):
    batch, heads, seq_len, dim = x.shape
    return x.reshape(batch, heads, seq_len // _CHUNK_SIZE, _CHUNK_SIZE, dim).contiguous()


def _chunk_lower_matmul(left, right, cumulative_decay, *, diagonal):
    """Causal QK / KKT tiles; the diagonal block uses the exact per-row decay."""
    result = left.new_zeros((*left.shape[:-1], right.shape[-2]))
    for row_start in range(0, _CHUNK_SIZE, _REFERENCE_BLOCK_SIZE):
        row_end = row_start + _REFERENCE_BLOCK_SIZE
        reference = cumulative_decay[..., row_start:row_start + 1, :]
        left_rows = left[..., row_start:row_end, :] * (
            cumulative_decay[..., row_start:row_end, :] - reference).exp()
        for col_start in range(0, row_end, _REFERENCE_BLOCK_SIZE):
            col_end = col_start + _REFERENCE_BLOCK_SIZE
            if col_start == row_start:
                for row in range(row_start, row_end):
                    relative = (cumulative_decay[..., row:row + 1, :]
                                - cumulative_decay[..., row_start:row + 1, :]).exp()
                    right_cols = right[..., row_start:row + 1, :] * relative
                    result[..., row:row + 1, row_start:row + 1] = (
                        left[..., row:row + 1, :] @ right_cols.transpose(-1, -2))
                continue
            right_cols = right[..., col_start:col_end, :] * (
                reference - cumulative_decay[..., col_start:col_end, :]).exp()
            result[..., row_start:row_end, col_start:col_end] = (
                left_rows @ right_cols.transpose(-1, -2))
    return result * torch.tril(
        left.new_ones((_CHUNK_SIZE, _CHUNK_SIZE)),
        diagonal=diagonal)


def _chunk_stage1(K, V, Q, beta, g, *, scale):
    K_i, V_i, Q_i = _chunk_major(K), _chunk_major(V), _chunk_major(Q)
    beta = _chunk_major(beta)
    g_i = _chunk_major(g)
    cumdec = torch.cumsum(g_i, dim=-2)
    gamma = torch.exp(cumdec)
    K_decayed = gamma * K_i
    K_decayed_beta = beta * K_decayed
    Q_decayed = gamma * Q_i * scale
    T = _chunk_lower_matmul(beta * K_i, K_i, cumdec, diagonal=-1)
    eye = torch.eye(_CHUNK_SIZE, dtype=K.dtype, device=K.device)
    inverse = torch.linalg.solve_triangular(
        eye + T, eye.expand_as(T), upper=False, unitriangular=True)
    M = inverse * beta.transpose(-1, -2)
    U_pre = M @ V_i
    W = M @ K_decayed
    Mqk = _chunk_lower_matmul(Q_i * scale, K_i, cumdec, diagonal=0)
    gamma_C = gamma[..., -1, :, None].contiguous()
    K_restored = K_i * (cumdec[..., -1:, :] - cumdec).exp()
    return dict(K_decayed_beta=K_decayed_beta, K_decayed=K_decayed,
                Q_decayed=Q_decayed, U_pre=U_pre, W=W, Mqk=Mqk,
                K_restored=K_restored, gamma_C=gamma_C)


def _chunk_stage2(s1, initial_state):
    chunks = s1["U_pre"].shape[2]
    U = torch.zeros_like(s1["U_pre"])
    O_state = torch.zeros_like(s1["U_pre"])
    state = initial_state.clone()
    U_pre, W, Q_decayed = s1["U_pre"], s1["W"], s1["Q_decayed"]
    K_restored, gamma_C = s1["K_restored"], s1["gamma_C"]
    for c in range(chunks):
        U_c = U_pre[:, :, c] - W[:, :, c] @ state
        O_c = Q_decayed[:, :, c] @ state
        state = gamma_C[:, :, c] * state + K_restored[:, :, c].transpose(-1, -2) @ U_c
        U[:, :, c] = U_c
        O_state[:, :, c] = O_c
    return dict(U=U, O_state=O_state, S=state)


def _chunk_stage3(s1, s2):
    O_i = s2["O_state"] + s1["Mqk"] @ s2["U"]
    batch, heads, chunks, chunk, dim = O_i.shape
    return O_i.reshape(batch, heads, chunks * chunk, dim)


def _chunkwise_forward(q, k, v, step_gate, beta_eff, h0, bounds, scale, heads_per_key=1):
    """Evaluate the chunkwise recurrence in the input arithmetic dtype."""
    if heads_per_key != 1:
        q = q.repeat_interleave(heads_per_key, dim=1)
        k = k.repeat_interleave(heads_per_key, dim=1)
    token_count, value_heads, value_dim = v.shape
    outputs = v.new_zeros((token_count, value_heads, value_dim))
    final = torch.zeros_like(h0)
    for seq_index, (start, end) in enumerate(bounds):
        length = end - start
        if length <= 0:
            final[seq_index] = h0[seq_index]
            continue
        pad = (-length) % _CHUNK_SIZE
        sl = slice(start, end)
        q_s = q[sl].transpose(0, 1).unsqueeze(0)          # (1,Nv,S,D)
        k_s = k[sl].transpose(0, 1).unsqueeze(0)
        v_s = v[sl].transpose(0, 1).unsqueeze(0)
        g_s = step_gate[sl].transpose(0, 1).unsqueeze(0)
        b_s = beta_eff[sl].transpose(0, 1).unsqueeze(-1).unsqueeze(0)
        if pad:
            # padded rows must be the identity: zero log-decay and zero beta
            q_s = F.pad(q_s, (0, 0, 0, pad))
            k_s = F.pad(k_s, (0, 0, 0, pad))
            v_s = F.pad(v_s, (0, 0, 0, pad))
            g_s = F.pad(g_s, (0, 0, 0, pad))
            b_s = F.pad(b_s, (0, 0, 0, pad))
        s1 = _chunk_stage1(k_s, v_s, q_s, b_s, g_s, scale=scale)
        # h0 is stored transposed; the chunk algebra works on the natural
        # (Dk, Dv) orientation, so transpose in and back out.
        s2 = _chunk_stage2(s1, h0[seq_index].transpose(-1, -2).unsqueeze(0))
        out = _chunk_stage3(s1, s2)[0, :, :length, :]     # (Nv, S, D)
        outputs[sl] = out.transpose(0, 1)
        final[seq_index] = s2["S"][0].transpose(-1, -2)
    return outputs, final


def autograd_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    *,
    scale: float,
    gate_mode: str = "bounded",
    lower_bound: float = -5.0,
    allow_neg_eigval: bool = False,
    layout: str = "BNSD",
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, ...]:
    """Differentiate the token-level KDA recurrence.

    ``dtype`` selects the arithmetic precision (FP64 by default, FP32 for the
    ``_token_golden``-equivalent golden). No BF16 storage anywhere: every
    intermediate stays in ``dtype``.
    """
    initial_state = resolve_initial_state(q, v, initial_state, layout, cu_seqlens)
    if use_gate_in_kernel and A_log is None:
        raise ValueError("A_log is required when gate fusion is enabled")
    dt = dtype
    leaves = [
        token_major(q, layout).to(dt).requires_grad_(True),
        token_major(k, layout).to(dt).requires_grad_(True),
        token_major(v, layout).to(dt).requires_grad_(True),
        token_major(g, layout).to(dt).requires_grad_(True),
        token_major(beta, layout).to(dt).requires_grad_(True),
        A_log.to(dt).requires_grad_(True) if use_gate_in_kernel else None,
        dt_bias.to(dt).requires_grad_(True)
        if use_gate_in_kernel and dt_bias is not None else None,
        initial_state.to(dt).requires_grad_(True),
    ]
    q_t, k_t, v_t, g_t, beta_t, a_log, bias, h0 = leaves

    if use_qk_l2norm_in_kernel:
        # q_t / k_t stay the raw leaves: the grouped path rebuilds this
        # normalisation per head group so no subgraph is shared between
        # backward passes.
        qn_t = q_t * torch.rsqrt(q_t.square().sum(-1, keepdim=True) + NORM_EPS)
        kn_t = k_t * torch.rsqrt(k_t.square().sum(-1, keepdim=True) + NORM_EPS)
    else:
        qn_t, kn_t = q_t, k_t

    token_count, value_heads, value_dim = v_t.shape
    key_heads, key_dim = q_t.shape[1], q_t.shape[-1]
    batch = initial_state.shape[0]
    sequence_length = token_count // batch
    heads_per_key = value_heads // key_heads
    do_t = token_major(do, layout).to(dt)
    dht_t = dht.to(dt)

    if use_gate_in_kernel:
        alpha = a_log.exp()
        biased = g_t if bias is None else g_t + bias[None]
        step_gate = (
            -alpha[None, :, None] * F.softplus(biased)
            if gate_mode == "softplus"
            else float(lower_bound) * torch.sigmoid(alpha[None, :, None] * biased)
        )
    else:
        step_gate = g_t
    b_eff = (
        torch.sigmoid(beta_t) * (2.0 if allow_neg_eigval else 1.0)
        if use_beta_sigmoid_in_kernel
        else beta_t
    )

    outputs = torch.zeros(token_count, value_heads, value_dim, dtype=dt, device=q.device)
    final = torch.zeros_like(h0)
    bounds = sequence_bounds(q, initial_state, layout, cu_seqlens, seqused)
    progress = _GoldenProgress(sum(end - start for start, end in bounds) * value_heads)
    print("[CPU Golden] chunkwise recurrence started", flush=True)
    chunk_start = time.monotonic()
    outputs, final = _chunkwise_forward(
        qn_t, kn_t, v_t, step_gate, b_eff, h0, bounds, float(scale), heads_per_key
    )
    progress.update(token_count * value_heads)
    print(
        f"[CPU Golden] chunkwise recurrence done, "
        f"elapsed={progress.duration(time.monotonic() - chunk_start)}",
        flush=True,
    )

    loss = (outputs * do_t).sum() + (final * dht_t).sum() + sum(
        t.sum() * 0 for t in leaves if t is not None
    )
    print("[CPU Golden] autograd backward started (ETA unavailable)", flush=True)
    backward_start = time.monotonic()
    active_leaves = [t for t in leaves if t is not None]
    active_grads = iter(torch.autograd.grad(loss, active_leaves, allow_unused=True))
    grads = tuple(next(active_grads) if t is not None else None for t in leaves)
    print(
        f"[CPU Golden] autograd backward completed, "
        f"elapsed={progress.duration(time.monotonic() - backward_start)}, "
        f"total elapsed={progress.duration(time.monotonic() - progress.start)}",
        flush=True,
    )

    grads = tuple(torch.zeros_like(t) if t is not None and grad is None else grad for grad, t in zip(grads, leaves))
    return (
        restore_layout(grads[0], q, layout),
        restore_layout(grads[1], k, layout),
        restore_layout(grads[2], v, layout),
        grads[7],
        restore_layout(grads[3], g, layout),
        restore_layout(grads[4], beta, layout),
        grads[5],
        grads[6],
    )


def token_golden_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    *,
    scale: float,
    gate_mode: str = "bounded",
    lower_bound: float = -5.0,
    allow_neg_eigval: bool = False,
    layout: str = "BNSD",
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    dtype: torch.dtype = torch.float32,
):
    """TTK golden: the token recurrence evaluated by chunk algebra.

    Mirrors ``test_flash_kda_grad_ut._token_golden`` exactly — inputs are cast to
    ``dtype`` (FP32), L2 norm / gate / beta activation are computed in that
    precision, and the recurrence

        S_t = diag(exp(g_t)) S_{t-1}
        delta_t = beta_t (v_t - k_t^T S_{t-1})
        S_t = S_{t-1} + k_t delta_t^T
        o_t = scale * q_t^T S_t

    is evaluated chunkwise in ``dtype``. Gradients come from ``torch.autograd``
    over that recurrence without casting intermediates to BF16.
    """
    raw_bias = dt_bias
    dt_bias, dht = normalize_backward_optionals(
        q, v, initial_state, dt_bias, dht, layout, cu_seqlens
    )
    grads = autograd_backward(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
        do,
        dht,
        scale=scale,
        gate_mode=gate_mode,
        lower_bound=lower_bound,
        allow_neg_eigval=allow_neg_eigval,
        layout=layout,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        dtype=dtype,
    )
    return restore_bias_gradient(grads, raw_bias)


# ---------------------------------------------------------------------------
# TTK third-party peer entry (dotted API path, executed on the XPU server)
# ---------------------------------------------------------------------------


def bf16_peer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    scale: Optional[float] = None,
    gate_mode: str = "bounded",
    lower_bound: float = -5.0,
    allow_neg_eigval: bool = False,
    layout: str = "BNSD",
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    device_quant: Optional[bool] = None,
    r_source: str = "cache",
    gate_ratio: str = "exp_diff",
    inverse_source: str = "cache",
    gate_grad_bf16: bool = False,
    p_scale: str = "none",
    v_path_bf16: bool = False,
    disable_recompute: bool = False,
    cache: Optional["Cache"] = None,
    cache_source: str = "ref",
    **_ignored,
):
    """Third-party reference with the operator's BF16 quantization budget.

    Same signature as the TTK API under test, so ``cross_check`` can use it as
    the like-for-like peer: it rounds the cache to BF16 at the operator's
    rounding points and casts every Cube operand to BF16 where the kernel does
    (``device_quant=None`` -> inferred from the BF16 cache), giving it the same
    error scale as the device kernel.
    """
    scale = float(scale) if scale is not None else float(q.shape[-1]) ** -0.5
    kwargs = dict(
        scale=scale,
        gate_mode=str(gate_mode).lower(),
        lower_bound=None if lower_bound is None else float(lower_bound),
        allow_neg_eigval=bool(allow_neg_eigval),
        layout=layout,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
        use_gate_in_kernel=bool(use_gate_in_kernel),
        use_beta_sigmoid_in_kernel=bool(use_beta_sigmoid_in_kernel),
    )
    # pr_45 给算子加了 ``disable_recompute``：True 表示消费外部保存的前向缓存，
    # False 表示 kernel 自己重算缓存。CPU 侧两种模式的数学完全一致（缓存是
    # 确定性的，同一份输入必然得到同一份缓存），所以这里只把契约对齐：
    # - 未传 ``cache`` 时按 ``disable_recompute=False`` 自己构建；
    # - 传了 ``cache`` 时按 ``disable_recompute=True`` 只消费、不重建。
    raw_bias = dt_bias
    dt_bias, dht = normalize_backward_optionals(
        q, v, initial_state, dt_bias, dht, layout, cu_seqlens
    )
    if cache is None:
        emulate_npu_forward = str(cache_source).lower() == "fwd"
        # Forward consumes BF16 g even when the public tensor is FP32.
        # Keep the original gate for its backward chain rule below.
        forward_g = g.to(torch.bfloat16) if emulate_npu_forward else g
        cache = build_cache(
            q,
            k,
            v,
            forward_g,
            beta,
            initial_state,
            A_log,
            dt_bias,
            emulate_bf16=True,
            storage_dtype=torch.bfloat16,
            _peer_only=True,
            emulate_npu_forward=emulate_npu_forward,
            **kwargs,
        )
    grads = staged_backward(
        q, k, v, g, beta, do, dht, cache,
        A_log=A_log, dt_bias=dt_bias,
        device_quant=device_quant,
        r_source=str(r_source),
        gate_ratio=str(gate_ratio),
        inverse_source=str(inverse_source),
        gate_grad_bf16=bool(gate_grad_bf16),
        p_scale=str(p_scale),
        v_path_bf16=bool(v_path_bf16),
        **kwargs
    )
    return restore_bias_gradient(grads, raw_bias)
