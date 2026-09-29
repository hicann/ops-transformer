# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU golden for the fused indexer_prologue_qw operator.

Q path: MXFP8 matmul → reshape → inplace RoPE on the last Dr dims → MXFP4 quant.
W path: BF16 matmul → scalar mul.

This module is independent of the NPU kernel's GM→L1 scale transform.
Host `q` is packed uint8 (T, N, D/2): low nibble = even D, high nibble = odd D.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

MX_GROUP_SIZE = 32
MX_SCALE_PAIR = 2
MX_K_ALIGN = 64
FP4_E2M1_MAX = 6.0
E8M0_BIAS = 127
FP32_SHIFT_BITS = 23
FP32_MANTISSA_MASK = 0x7FFFFF
# OCP MXFP4 E2M1 positive codebook. Index is the 3-bit magnitude field.
E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def paired_scale_groups(length: int) -> int:
    return ceil_div(length, MX_K_ALIGN)


def _decode_e8m0_scale(scale: torch.Tensor, outer_size: int, k: int) -> torch.Tensor:
    """Decode public [outer, G, 2] E8M0 bytes to one FP32 scale per 32 elements."""
    valid = ceil_div(k, MX_GROUP_SIZE)
    exponent_bytes = scale.contiguous().view(torch.uint8).reshape(outer_size, -1)
    exponents = exponent_bytes[:, :valid].to(torch.int16) - E8M0_BIAS
    return torch.pow(2.0, exponents.float())


def _fp32_scale_to_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Round FP32 scales up to the next power of two and store IEEE exponent bytes.

    Matches kv_compress_epilog: if the mantissa is nonzero, increment the exponent.
    """
    bits = scale.to(torch.float32).view(torch.int32).to(torch.int64)
    exp_bits = (bits >> FP32_SHIFT_BITS) & 0xFF
    mantissa = bits & FP32_MANTISSA_MASK
    exp_bits = torch.where(mantissa != 0, (exp_bits + 1) & 0xFF, exp_bits)
    zero = scale <= 0
    return torch.where(zero, torch.zeros_like(exp_bits), exp_bits).to(torch.uint8)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_inplace(
    q: torch.Tensor, rope_sin: torch.Tensor, rope_cos: torch.Tensor
) -> torch.Tensor:
    """Inplace RoPE on the last Dr columns of D. q is (T, N, D)."""
    dr = int(rope_sin.shape[-1])
    if dr == 0:
        return q
    if dr % 2 != 0:
        raise ValueError(f"Dr must be even, got {dr}")
    if dr > q.shape[-1]:
        raise ValueError(f"Dr={dr} exceeds D={q.shape[-1]}")
    pe = q[..., -dr:]
    cos = rope_cos.to(torch.float32).unsqueeze(1)
    sin = rope_sin.to(torch.float32).unsqueeze(1)
    q[..., -dr:] = pe * cos + rotate_half(pe) * sin
    return q


def mx_matmul_fp32(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    scale_lhs: torch.Tensor,
    scale_rhs: torch.Tensor,
) -> torch.Tensor:
    """Y[M,N] = dequant(lhs)[M,K] @ dequant(rhs)[N,K]^T."""
    m, k = lhs.shape
    n = rhs.shape[0]
    scale_a = _decode_e8m0_scale(scale_lhs, m, k).repeat_interleave(
        MX_GROUP_SIZE, dim=1
    )[:, :k]
    scale_b = _decode_e8m0_scale(scale_rhs, n, k).repeat_interleave(
        MX_GROUP_SIZE, dim=1
    )[:, :k]
    return (lhs.float() * scale_a) @ (rhs.float() * scale_b).T


def fp32_to_e2m1_nibble(values: torch.Tensor) -> torch.Tensor:
    """Round FP32 to nearest OCP E2M1 code (4-bit nibble). Ties go away from zero."""
    codebook = torch.tensor(E2M1_POS, dtype=torch.float32, device=values.device)
    sign = values < 0
    mag = values.abs()
    dist = (mag.unsqueeze(-1) - codebook).abs()
    min_dist = dist.min(dim=-1).values.unsqueeze(-1)
    tied = dist == min_dist
    arange = torch.arange(8, device=values.device, dtype=torch.int64)
    idx = torch.where(tied, arange, torch.full_like(arange, -1)).max(dim=-1).values
    nibble = idx.to(torch.uint8)
    nibble = torch.where(sign, nibble | 0x08, nibble)
    return nibble


def pack_e2m1_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack (..., D) nibbles into (..., D/2) bytes. D must be even."""
    if nibbles.shape[-1] % 2 != 0:
        raise ValueError(f"D must be even to pack e2m1, got {nibbles.shape[-1]}")
    low = nibbles[..., 0::2].to(torch.int16)
    high = nibbles[..., 1::2].to(torch.int16)
    return (low | (high << 4)).to(torch.uint8).contiguous()


def unpack_e2m1_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack (..., D/2) bytes into (..., D) nibbles."""
    low = packed & 0x0F
    high = (packed.to(torch.int16) >> 4).to(torch.uint8)
    return torch.stack((low, high), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )


def e2m1_nibble_to_fp32(nibbles: torch.Tensor) -> torch.Tensor:
    codebook = torch.tensor(E2M1_POS, dtype=torch.float32, device=nibbles.device)
    mag = codebook[(nibbles & 0x7).long()]
    sign = torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    return mag * sign


def mx_quant_along_last(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize (T, N, D) FP32 to MXFP4 along D.

    Returns:
        q: uint8 packed e2m1, shape (T, N, D/2)
        descale: uint8 E8M0 packed as (T, N, ceil(D/64), 2)
    """
    t, n, d = x.shape
    if d % 2 != 0:
        raise ValueError(f"D must be even for FP4 packing, got {d}")
    n_groups = ceil_div(d, MX_GROUP_SIZE)
    pad = n_groups * MX_GROUP_SIZE - d
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    grouped = x.reshape(t, n, n_groups, MX_GROUP_SIZE)
    amax = grouped.abs().amax(dim=-1).clamp_min(0.0)
    raw_scale = torch.where(
        amax > 0,
        amax / FP4_E2M1_MAX,
        torch.ones_like(amax),
    )
    e8m0 = _fp32_scale_to_e8m0(raw_scale)
    decoded = torch.pow(2.0, e8m0.to(torch.float32) - E8M0_BIAS)
    q_fp32 = grouped / decoded.unsqueeze(-1)
    nibbles = fp32_to_e2m1_nibble(
        q_fp32.reshape(t, n, n_groups * MX_GROUP_SIZE)[..., :d]
    )
    q = pack_e2m1_nibbles(nibbles)

    pair_groups = paired_scale_groups(d)
    pair_elems = pair_groups * MX_SCALE_PAIR
    if n_groups < pair_elems:
        e8m0 = torch.nn.functional.pad(
            e8m0, (0, pair_elems - n_groups), value=E8M0_BIAS
        )
    descale = e8m0.reshape(t, n, pair_groups, MX_SCALE_PAIR).contiguous()
    return q, descale


@dataclass
class IndexerPrologueQwInputs:
    x: torch.Tensor
    qr: torch.Tensor
    wqb: torch.Tensor
    ww: torch.Tensor
    descale_qr: torch.Tensor
    descale_wqb: torch.Tensor
    rope_sin: torch.Tensor
    rope_cos: torch.Tensor
    softmax_scale: float


@dataclass
class IndexerPrologueQwOutputs:
    q: torch.Tensor
    descale_q: torch.Tensor
    w: torch.Tensor
    q_fp32: torch.Tensor  # after RoPE, before quant; debug / staged probes


def _make_paired_scale(
    outer_size: int, k: int, generator: torch.Generator
) -> torch.Tensor:
    scale_k_len = paired_scale_groups(k) * MX_SCALE_PAIR
    exponents = torch.randint(
        -2,
        3,
        (outer_size, scale_k_len),
        dtype=torch.int16,
        generator=generator,
    )
    return (
        (exponents + E8M0_BIAS)
        .to(torch.uint8)
        .reshape(outer_size, paired_scale_groups(k), MX_SCALE_PAIR)
        .contiguous()
    )


def make_inputs(
    t: int,
    dim: int,
    q_lora: int,
    n_heads: int,
    d: int,
    dr: int,
    *,
    softmax_scale: float = 1.0,
    rope_identity: bool = False,
    seed: int = 0,
) -> IndexerPrologueQwInputs:
    if dr > d:
        raise ValueError(f"Dr={dr} exceeds D={d}")
    if dr % 2 != 0:
        raise ValueError(f"Dr must be even, got {dr}")
    if d % 2 != 0:
        raise ValueError(f"D must be even for FP4 packing, got {d}")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    qr = (torch.randn((t, q_lora), generator=gen) * 0.5).to(torch.float8_e4m3fn)
    wqb = (torch.randn((n_heads * d, q_lora), generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    x = torch.randn((t, dim), generator=gen, dtype=torch.float32).to(torch.bfloat16)
    ww = torch.randn((n_heads, dim), generator=gen, dtype=torch.float32).to(
        torch.bfloat16
    )
    if rope_identity:
        rope_cos = torch.ones((t, dr), dtype=torch.float32)
        rope_sin = torch.zeros((t, dr), dtype=torch.float32)
    else:
        theta = torch.rand((t, dr), generator=gen, dtype=torch.float32) * (2 * math.pi)
        rope_cos = torch.cos(theta)
        rope_sin = torch.sin(theta)
    return IndexerPrologueQwInputs(
        x=x,
        qr=qr,
        wqb=wqb,
        ww=ww,
        descale_qr=_make_paired_scale(t, q_lora, gen),
        descale_wqb=_make_paired_scale(n_heads * d, q_lora, gen),
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        softmax_scale=float(softmax_scale),
    )


def indexer_prologue_qw_golden(
    inputs: IndexerPrologueQwInputs,
) -> IndexerPrologueQwOutputs:
    t = inputs.qr.shape[0]
    n_d = inputs.wqb.shape[0]
    # Infer N, D from q output contract: wqb is (N*D, q_lora), ww is (N, dim).
    n_heads = inputs.ww.shape[0]
    if n_d % n_heads != 0:
        raise ValueError(f"wqb.shape[0]={n_d} is not divisible by N={n_heads}")
    d = n_d // n_heads

    y = mx_matmul_fp32(inputs.qr, inputs.wqb, inputs.descale_qr, inputs.descale_wqb)
    q_fp32 = y.reshape(t, n_heads, d).contiguous()
    q_fp32 = apply_rope_inplace(q_fp32, inputs.rope_sin, inputs.rope_cos)
    q, descale_q = mx_quant_along_last(q_fp32)

    w = inputs.softmax_scale * (inputs.x.float() @ inputs.ww.float().T)
    return IndexerPrologueQwOutputs(q=q, descale_q=descale_q, w=w, q_fp32=q_fp32)


def dequant_mx_last(q: torch.Tensor, descale: torch.Tensor, d: int) -> torch.Tensor:
    """Decode packed MXFP4 q (T,N,D/2) with paired descale back to FP32 (T,N,D)."""
    t, n, packed_d = q.shape
    if packed_d * 2 != d:
        raise ValueError(f"packed last dim {packed_d} does not match D={d}")
    nibbles = unpack_e2m1_nibbles(q)
    values = e2m1_nibble_to_fp32(nibbles)
    scale = _decode_e8m0_scale(
        descale.reshape(t * n, descale.shape[2], descale.shape[3]),
        t * n,
        d,
    ).reshape(t, n, -1)
    scale = scale.repeat_interleave(MX_GROUP_SIZE, dim=-1)[..., :d]
    return values * scale
