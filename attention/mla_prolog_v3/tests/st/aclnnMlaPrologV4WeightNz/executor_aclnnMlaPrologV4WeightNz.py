#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

# ===== golden (inlined) =====

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

try:
    import torch_npu

    _HAS_TORCH_NPU = True
except ImportError:
    _HAS_TORCH_NPU = False

MXFP8_GRP = 32
FP8_E4M3_MAX = 448.0
E4M3_EMAX = 8
FRACTAL_NZ_FORMAT = 29
HIF8_MAX_FINITE = 32768.0  # HiFloat8 最大有限值 2^15


@dataclass
class CaseConfig:
    t: int = 8
    he: int = 7168
    hcq: int = 1536
    hckv: int = 512
    d: int = 128
    dr: int = 64
    n: int = 8
    nkv: int = 1
    block_num: int = 2
    block_size: int = 128
    eps_cq: float = 1e-5
    eps_ckv: float = 1e-5
    qc_qr_scale: float = 1.0
    kc_scale: float = 1.0
    do_rope: bool = True
    rope_style: str = "original"
    seed: int = 2025
    weight_quant_mode: int = 0
    kv_cache_quant_mode: int = 0
    query_quant_mode: int = 0
    ckvkr_repo_mode: int = 0
    quant_scale_repo_mode: int = 0
    tile_size: int = 128
    query_norm_flag: bool = False
    batch: int = 1
    seq: int = 0  # 0 → seq=t；BSND / PA_BLK 使用
    cache_mode: str = "PA_BSND"
    token_rank: int = 2


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def interleave_to_half(x: torch.Tensor) -> torch.Tensor:
    dr = x.shape[-1]
    return (
        x.reshape(*x.shape[:-1], dr // 2, 2)
        .transpose(-1, -2)
        .reshape(*x.shape[:-1], dr)
    )


def apply_rope_original(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    q = interleave_to_half(x)
    if q.dim() == 3:
        cos_e = cos.unsqueeze(1).expand_as(q)
        sin_e = sin.unsqueeze(1).expand_as(q)
        return (q * cos_e) + (rotate_half(q) * sin_e)
    return (q * cos) + (rotate_half(q) * sin)


def apply_rope_vf(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Match arch35 vf_rope.h RopeVFImpl: same as the original layout except the low half adds
    sin*odd instead of subtracting it."""
    even, odd = x[..., ::2], x[..., 1::2]
    half = x.shape[-1] // 2
    cos_l, cos_u = cos[..., :half], cos[..., half:]
    sin_l, sin_u = sin[..., :half], sin[..., half:]
    if x.dim() == 3:
        cos_l, cos_u = (
            cos_l.unsqueeze(1).expand_as(even),
            cos_u.unsqueeze(1).expand_as(even),
        )
        sin_l, sin_u = (
            sin_l.unsqueeze(1).expand_as(even),
            sin_u.unsqueeze(1).expand_as(even),
        )
    return torch.cat([cos_l * even + sin_l * odd, sin_u * even + cos_u * odd], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    do_rope: bool,
    rope_style: str = "original",
) -> torch.Tensor:
    """V4 CPU 真值 RoPE（对齐 aclnnMlaPrologV4WeightNz.cal_mlaprolog rotary1/rotary2）。

    doRope 是 V4 相对 V3 的开关：true 做 interleave-half + rotate_half；
    false 时 Qrope / krcache 直通，不读 ropeSin/ropeCos。
    """
    x = x.to(torch.float32)
    if not do_rope:
        return x
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)
    if rope_style == "vf":
        return apply_rope_vf(x, cos, sin)
    return apply_rope_original(x, cos, sin)


def rms_norm(
    x: torch.Tensor, gamma: torch.Tensor, eps: float, scale: float
) -> torch.Tensor:
    """对齐 V4 `_rms_norm_kernel` / kernel RmsNormVF：sum(x^2)*(1/n)+eps。"""
    x32 = x.to(torch.float32)
    n = int(x32.shape[-1])
    recip = torch.tensor(1.0 / n, dtype=torch.float32, device=x32.device)
    rms = torch.sqrt(torch.sum(x32 * x32, dim=-1, keepdim=True) * recip + float(eps))
    y = x32 / rms * gamma.to(torch.float32)
    if abs(float(scale) - 1.0) >= 1e-7:
        y = y * float(scale)
    return y


def s8_saturation(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -128, 127).to(torch.int8)


def s9_saturation(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -256, 255)


def _cpu_f32(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.float32)


def _cast_rint_bf16(x: torch.Tensor) -> torch.Tensor:
    """Cube Fixpipe F322BF16 / Vector CAST_RINT: nearest, ties away from 0.

    torch.to(bfloat16) 是 ties-to-even；kernel CAST_RINT 在低 16 位 >= 0x8000 时进位。
    纯 torch 位运算，避免 numpy 与 DUT 同进程污染 ACL。
    """
    x32 = x.detach().to(device="cpu", dtype=torch.float32).contiguous()
    bits = x32.view(torch.int32)
    sign_mask = torch.tensor(-2147483648, dtype=torch.int32)  # 0x80000000
    mag_mask = torch.tensor(0x7FFFFFFF, dtype=torch.int32)
    special = torch.tensor(0x7F800000, dtype=torch.int32)
    round_bias = torch.tensor(0x8000, dtype=torch.int32)
    mag = bits & mag_mask
    out_bits = torch.where(
        mag >= special, bits, (bits & sign_mask) | (mag + round_bias)
    )
    out = torch.empty(x32.shape, dtype=torch.bfloat16)
    out.view(torch.int16).copy_((out_bits >> 16).to(torch.int16))
    return out


def _to_bf16_fp32_rint(x: torch.Tensor) -> torch.Tensor:
    """F322BF16 再回到 fp32，对齐 Cube Fixpipe 后再进下一拍 Vector。"""
    return _cast_rint_bf16(x).to(torch.float32)


def _cube_fp32_matmul(
    a: torch.Tensor, b: torch.Tensor, k_tile: int = 256, device: Optional[str] = None
) -> torch.Tensor:
    """与 CPU 同一套切 K + FP32 累加。NPU 标杆只把每段换成 torch.matmul(bf16)，不 format_cast NZ。"""
    use_npu = device is not None and str(device).startswith("npu")
    if use_npu:
        a_m = a.detach().to(device=device, dtype=torch.bfloat16).contiguous()
        b_m = b.detach().to(device=device, dtype=torch.bfloat16).contiguous()
    else:
        a_m = a.to(torch.float32)
        b_m = b.to(torch.float32)
    k = int(a_m.shape[-1])

    def _mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y).to(torch.float32)

    if k <= k_tile:
        acc = _mm(a_m, b_m)
    else:
        acc = None
        for k0 in range(0, k, k_tile):
            k1 = min(k0 + k_tile, k)
            part = _mm(a_m[..., k0:k1], b_m[k0:k1, :])
            acc = part if acc is None else acc + part
    if use_npu:
        torch.npu.synchronize()
        return acc.cpu()
    return acc


def quant_s8(x: torch.Tensor, qscale: torch.Tensor) -> torch.Tensor:
    scaled = (_cpu_f32(x) * _cpu_f32(qscale)).round()
    return s8_saturation(s9_saturation(scaled))


def _cast_f32_to_i8_kernel(v: torch.Tensor) -> torch.Tensor:
    """对齐 dynamic quant 落盘：f32 → i8 CAST_RINT（round-to-nearest）。

    实测相对 f16 两级 cast / round-half-away-from-zero，纯 rint 与 Cube 更接近。
    """
    return torch.round(v).clamp(-128, 127)


def _optional_smooth(
    smooth_scale: Optional[torch.Tensor], h: int
) -> Optional[torch.Tensor]:
    """ATK 空可选槽可能落成 bool / 错误尺寸张量，不能当 smooth scale 用。"""
    if (
        isinstance(smooth_scale, torch.Tensor)
        and smooth_scale.dtype != torch.bool
        and smooth_scale.numel() == h
    ):
        return _cpu_f32(smooth_scale).reshape(1, h)
    return None


def dynamic_quant(
    inputs: torch.Tensor, smooth_scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic quant，对齐 arch35 DynamicQuantPerTokenVf（int8）。

    scale = amax / 127；q = cast_i8(x / scale)；dequant 时再乘回 scale。
    """
    t, h = inputs.shape
    y = torch.zeros(t, h, dtype=torch.int32)
    scale = torch.zeros(t, 1, dtype=torch.float32)
    x = inputs.to(torch.float32)
    ss = _optional_smooth(smooth_scale, h)
    for i in range(t):
        row = x[i] * ss[0] if ss is not None else x[i]
        amax = row.abs().max().clamp(min=1e-12)
        scale[i, 0] = amax / 127.0
        y[i] = _cast_f32_to_i8_kernel(row / scale[i, 0])
    return y, scale


def dynamic_quant_fp8(
    inputs: torch.Tensor, smooth_scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic quant → FP8E4M3，对齐 DynamicQuantPerTokenVf (max=448)。"""
    x = inputs.to(torch.float32)
    ss = _optional_smooth(smooth_scale, x.shape[-1])
    if ss is not None:
        x = x * ss
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_E4M3_MAX
    q = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


def dynamic_quant_hif8(
    inputs: torch.Tensor, smooth_scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic quant → HiFloat8，对齐 DynamicQuantPerTokenVf (max=32768)。"""
    x = inputs.to(torch.float32)
    ss = _optional_smooth(smooth_scale, x.shape[-1])
    if ss is not None:
        x = x * ss
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / HIF8_MAX_FINITE
    return _hif8_encode(x / scale), scale


def int8_matmul_i32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """int8×int8 → int32 累加，对齐 Cube L0C=int32（不做 dequant）。"""
    return torch.matmul(a.to(torch.int32), b.to(torch.int32))


def quantize_weight_per_channel(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    wf = w.to(torch.float32)
    scale = wf.abs().amax(dim=0, keepdim=True).clamp(min=1e-8) / 127.0
    wq = torch.round(wf / scale).clamp(-128, 127).to(torch.int8)
    return wq, scale.to(torch.float32)


def quantize_activation_per_token(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xf = x.to(torch.float32)
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    xq = torch.round(xf / scale).clamp(-128, 127).to(torch.int8)
    return xq, scale.to(torch.float32)


def _hif8_encode(x: torch.Tensor) -> torch.Tensor:
    """fp32/bf16 ND → uint8 HiFloat8 存储。需要 torch_npu。"""
    if not (_HAS_TORCH_NPU and torch.npu.is_available()):
        raise RuntimeError("HiFloat8 encode requires torch_npu")
    xf = x.to(torch.float32).contiguous().npu()
    return torch_npu.npu_dtype_cast(xf, torch_npu.hifloat8).cpu()


def _hif8_decode(x_u8: torch.Tensor) -> torch.Tensor:
    """uint8 HiFloat8 存储 → fp32。需要 torch_npu。"""
    if not (_HAS_TORCH_NPU and torch.npu.is_available()):
        raise RuntimeError("HiFloat8 decode requires torch_npu")
    q = x_u8.to(torch.uint8).contiguous().npu()
    return torch_npu.npu_dtype_cast(
        q, torch.float32, input_dtype=torch_npu.hifloat8
    ).cpu()


def quantize_hif8_per_channel(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    wf = w.to(torch.float32)
    scale = wf.abs().amax(dim=0, keepdim=True).clamp(min=1e-8) / HIF8_MAX_FINITE
    return _hif8_encode(wf / scale), scale.to(torch.float32)


def quantize_hif8_per_token(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xf = x.to(torch.float32)
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / HIF8_MAX_FINITE
    return _hif8_encode(xf / scale), scale.to(torch.float32)


def quantize_fp8_per_channel(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """wq=4 权重 per-channel 量化：scale = amax/448，落盘 FP8E4M3。"""
    wf = w.to(torch.float32)
    scale = wf.abs().amax(dim=0, keepdim=True).clamp(min=1e-8) / FP8_E4M3_MAX
    q = (wf / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32)


def quantize_fp8_per_token(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """wq=4 激活 per-token 量化：scale = amax/448，落盘 FP8E4M3。"""
    xf = x.to(torch.float32)
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / FP8_E4M3_MAX
    q = (xf / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32)


def _fp8_family_to_fp32(
    x: torch.Tensor, w: torch.Tensor, wq: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cube FP8/HIF8 MAC 的软件等价：先解码/升到 fp32，再在外面乘 FLOAT scale。"""
    if wq == 5:
        x_fp = x if x.dtype == torch.float32 else _hif8_decode(x)
        w_fp = w if w.dtype == torch.float32 else _hif8_decode(w)
        return x_fp.to(torch.float32), w_fp.to(torch.float32)
    return x.to(torch.float32), w.to(torch.float32)


def _e8m0_scale_from_amax(amax: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    a = amax.to(torch.float32)
    tiny = 2.0 ** (-126)
    log2a = torch.floor(torch.log2(a + tiny * (a == 0).to(torch.float32)))
    share = (log2a - float(E4M3_EMAX) + 127.0).clamp(0.0, 254.0)
    share = torch.where(a == 0, torch.full_like(share, 127.0), share)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32), share - 127.0)
    scale_e8m0 = torch.empty(share.shape, dtype=torch.float8_e8m0fnu)
    scale_e8m0.view(torch.uint8).copy_(share.to(torch.uint8))
    return scale, scale_e8m0


def quantize_mxfp8_along_last(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xf = x.to(torch.float32)
    *lead, h = xf.shape
    assert h % MXFP8_GRP == 0
    g = h // MXFP8_GRP
    blocks = xf.reshape(*lead, g, MXFP8_GRP)
    amax = blocks.abs().amax(dim=-1)
    scale_fp, scale_e8 = _e8m0_scale_from_amax(amax)
    q = (blocks / scale_fp.unsqueeze(-1).clamp(min=1e-30)).clamp(
        -FP8_E4M3_MAX, FP8_E4M3_MAX
    )
    q8 = q.to(torch.float8_e4m3fn).reshape(*lead, h)
    return q8, scale_e8, scale_fp


def quantize_weight_mxfp8(
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wf = w.to(torch.float32)
    k, n = wf.shape
    assert k % MXFP8_GRP == 0
    g = k // MXFP8_GRP
    blocks = wf.reshape(g, MXFP8_GRP, n)
    amax = blocks.abs().amax(dim=1)
    scale_fp, scale_e8 = _e8m0_scale_from_amax(amax)
    scale_fp_out = scale_fp.transpose(0, 1).contiguous()
    scale_e8_out = scale_e8.transpose(0, 1).contiguous()
    q = (blocks / scale_fp.unsqueeze(1).clamp(min=1e-30)).clamp(
        -FP8_E4M3_MAX, FP8_E4M3_MAX
    )
    q8 = q.to(torch.float8_e4m3fn).reshape(k, n)
    return q8, scale_e8_out, scale_fp_out


def dequant_mxfp8_along_last(
    x_fp8: torch.Tensor, scale_fp: torch.Tensor
) -> torch.Tensor:
    xf = x_fp8.to(torch.float32)
    *lead, h = xf.shape
    g = h // MXFP8_GRP
    blocks = xf.reshape(*lead, g, MXFP8_GRP)
    return (blocks * scale_fp.unsqueeze(-1)).reshape(*lead, h)


def dequant_weight_mxfp8(w_fp8: torch.Tensor, scale_fp: torch.Tensor) -> torch.Tensor:
    wf = w_fp8.to(torch.float32)
    k, n = wf.shape
    g = k // MXFP8_GRP
    blocks = wf.reshape(g, MXFP8_GRP, n)
    s = scale_fp.transpose(0, 1).unsqueeze(1)
    return (blocks * s).reshape(k, n)


def dynamic_mx_quant_dequant_cq(x: torch.Tensor) -> torch.Tensor:
    q8, _, scale_fp = quantize_mxfp8_along_last(x)
    return dequant_mxfp8_along_last(q8, scale_fp)


def quant_ckv_fp8_per_tensor(
    x: torch.Tensor, quant_scale: torch.Tensor
) -> torch.Tensor:
    y = _cpu_f32(x) * _cpu_f32(quant_scale).reshape(1, -1)
    y = torch.round(y * 1e8) / 1e8
    return y.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)


def dynamic_quant_q_nope_fp8(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    qf = q.to(torch.float32)
    amax = qf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_E4M3_MAX
    y = (qf / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return y, scale.squeeze(-1).unsqueeze(-1)


def dynamic_quant_q_nope_int8(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """wq=2 + queryQuantMode=1：per-token-head int8，对齐 V4 dynamic_quant_without_smooth_scale。"""
    qf = q.to(torch.float32)
    amax = qf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 127.0
    scaled = qf / scale
    y = s8_saturation(s9_saturation(scaled.round()))
    return y, scale


def dynamic_quant_q_nope_hif8(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    qf = q.to(torch.float32)
    amax = qf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / HIF8_MAX_FINITE
    return _hif8_encode(qf / scale), scale


def quant_ckv_hif8_per_tensor(
    x: torch.Tensor, quant_scale: torch.Tensor
) -> torch.Tensor:
    y = _cpu_f32(x) * _cpu_f32(quant_scale).reshape(1, -1)
    return _hif8_encode(y)


def _kv_dtile(cfg: CaseConfig) -> int:
    """kvq=3 合仓宽度：Hckv + Dr*2 + (Hckv/tile)*4 = 656（tile=128）。"""
    if int(cfg.kv_cache_quant_mode) != 3:
        return int(cfg.hckv)
    tile = max(int(cfg.tile_size) or 128, 1)
    return int(cfg.hckv) + int(cfg.dr) * 2 + (int(cfg.hckv) // tile) * 4


def _clip_alpha_value(clip_alpha) -> float:
    if clip_alpha is None:
        return 1.0
    if isinstance(clip_alpha, torch.Tensor):
        if clip_alpha.dtype == torch.bool or clip_alpha.numel() == 0:
            return 1.0
        return float(clip_alpha.reshape(-1)[0].item())
    return float(clip_alpha)


def _pack_kv_pertile(
    norm2: torch.Tensor,
    rotary_k: torch.Tensor,
    cfg: CaseConfig,
    clip_alpha=None,
) -> torch.Tensor:
    """对齐 V4 kvq=3：per-tile dynamic quant 后与 kR / scale 拼进 kvCache。"""
    t = int(norm2.shape[0])
    hckv, dr = int(cfg.hckv), int(cfg.dr)
    tile = max(int(cfg.tile_size) or 128, 1)
    tiles = hckv // tile
    x = norm2.reshape(t, tiles, tile).to(torch.float32)
    amax = torch.amax(torch.abs(x), dim=-1, keepdim=True).clamp(min=1e-4)
    amax = amax * _clip_alpha_value(clip_alpha)
    clip_res = torch.clamp(x, min=-amax, max=amax)
    wq = int(cfg.weight_quant_mode)
    if wq == 5:
        scale = torch.clamp(amax / HIF8_MAX_FINITE, min=1e-12)
        packed = _hif8_encode((clip_res / scale).reshape(t, hckv)).view(torch.int8)
        deq = scale.reshape(t, -1).contiguous()
    elif wq in (3, 4):
        scale = torch.clamp(amax / FP8_E4M3_MAX, min=1e-12)
        y = (
            (clip_res / scale)
            .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
        packed = y.reshape(t, hckv).view(torch.uint8).view(torch.int8).contiguous()
        deq = scale.reshape(t, -1).contiguous()
    else:
        scale = amax / 127.0
        packed = (
            torch.round(clip_res / scale)
            .clamp(-128, 127)
            .to(torch.int8)
            .reshape(t, hckv)
        )
        deq = scale.reshape(t, -1).contiguous()
    if int(cfg.ckvkr_repo_mode) == 1:
        packed = torch.cat(
            (packed, rotary_k.to(torch.bfloat16).contiguous().view(torch.int8)), dim=-1
        )
    if int(cfg.quant_scale_repo_mode) == 1:
        packed = torch.cat((packed, deq.view(torch.int8)), dim=-1)
    return packed


def gen_rope_cos_sin(t: int, dr: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    half = dr // 2
    pos = torch.arange(t, dtype=torch.float32).unsqueeze(1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
    angles = pos * inv_freq.unsqueeze(0)
    cos_h = torch.cos(angles)
    sin_h = torch.sin(angles)
    cos = torch.cat([cos_h, cos_h], dim=-1).to(torch.bfloat16)
    sin = torch.cat([sin_h, sin_h], dim=-1).to(torch.bfloat16)
    return cos, sin


def _cfg_seq(cfg: CaseConfig) -> int:
    return int(cfg.seq) if int(cfg.seq) > 0 else int(cfg.t)


def _cfg_batch(cfg: CaseConfig) -> int:
    return max(int(cfg.batch), 1)


def _cache_mode(cfg: CaseConfig) -> str:
    return str(getattr(cfg, "cache_mode", "PA_BSND") or "PA_BSND").upper()


def _token_t(token_x: torch.Tensor) -> int:
    if token_x.dim() == 3:
        return int(token_x.shape[0]) * int(token_x.shape[1])
    return int(token_x.shape[0])


def _reshape_dequant_for_query(
    scale: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    """算子 dequantScaleQNopeOut 为 (T, N, 1) 或 (T, 1)；BSND 下 queryOut 为 (B, S, N, H)。"""
    s = scale.to(torch.float32)
    if query.dim() == 4:
        b, s_len, n, _ = (int(x) for x in query.shape)
        t_flat = b * s_len
        if s.dim() == 3 and int(s.shape[0]) == t_flat:
            if int(s.shape[1]) == n:
                return s.reshape(b, s_len, n, 1)
            if int(s.shape[1]) == 1:
                return s.reshape(b, s_len, 1, 1).expand(b, s_len, n, 1)
        if s.dim() == 2 and int(s.shape[0]) == t_flat:
            if int(s.shape[1]) == 1:
                return s.reshape(b, s_len, 1, 1).expand(b, s_len, n, 1)
            if int(s.shape[1]) == n:
                return s.reshape(b, s_len, n, 1)
    if query.dim() == 3:
        t, n, _ = (int(x) for x in query.shape)
        if s.dim() == 2 and int(s.shape[0]) == t:
            if int(s.shape[1]) == 1:
                return s.unsqueeze(-1).expand(t, n, 1)
            if int(s.shape[1]) == n:
                return s.unsqueeze(-1)
    if s.dim() == 2:
        return s.unsqueeze(-1)
    return s


def _get_cache_index(inp: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    ci = inp.get("cache_index")
    if isinstance(ci, torch.Tensor) and ci.dtype != torch.bool and ci.numel() > 0:
        return ci
    return None


def _cache_kv_kr_shapes(cfg: CaseConfig) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    mode = _cache_mode(cfg)
    b, s = _cfg_batch(cfg), _cfg_seq(cfg)
    dtile = _kv_dtile(cfg)
    if int(cfg.ckvkr_repo_mode) == 1:
        kr_shape: Tuple[int, ...] = (0,)
    elif mode == "TND":
        kr_shape = (cfg.t, cfg.nkv, cfg.dr)
    elif mode == "BSND":
        kr_shape = (b, s, cfg.nkv, cfg.dr)
    else:
        kr_shape = (cfg.block_num, cfg.block_size, cfg.nkv, cfg.dr)
    if mode == "TND":
        return (cfg.t, cfg.nkv, dtile), kr_shape
    if mode == "BSND":
        return (b, s, cfg.nkv, dtile), kr_shape
    return (cfg.block_num, cfg.block_size, cfg.nkv, dtile), kr_shape


def gen_inputs(cfg: CaseConfig, device: str = "cpu") -> Dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(cfg.seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(torch.bfloat16)

    mode = _cache_mode(cfg)
    b, s = _cfg_batch(cfg), _cfg_seq(cfg)
    t = int(cfg.t)
    if mode in ("BSND", "PA_BLK_BSND", "PA_BLK_NZ") and b * s != t:
        t = b * s

    # ropeSin/ropeCos：doRope=true 时生成有效 cos/sin 表；doRope=false 时必须为空 tensor
    if cfg.do_rope:
        cos, sin = gen_rope_cos_sin(t, cfg.dr, cfg.seed)
        if mode == "BSND":
            cos = cos.view(b, s, cfg.dr)
            sin = sin.view(b, s, cfg.dr)
    else:
        cos = torch.empty((0,), dtype=torch.bfloat16)
        sin = torch.empty((0,), dtype=torch.bfloat16)
    token_x_bf16 = rnd(t, cfg.he)
    w_dq_bf16 = rnd(cfg.he, cfg.hcq)
    w_uq_qr_bf16 = rnd(cfg.hcq, cfg.n * (cfg.d + cfg.dr))
    w_uk = rnd(cfg.n, cfg.d, cfg.hckv)
    w_dkv_kr_bf16 = rnd(cfg.he, cfg.hckv + cfg.dr)

    out: Dict[str, torch.Tensor] = {
        "rmsnorm_gamma_cq": torch.ones(cfg.hcq, dtype=torch.bfloat16),
        "rmsnorm_gamma_ckv": torch.ones(cfg.hckv, dtype=torch.bfloat16),
        "rope_sin": sin,
        "rope_cos": cos,
        "weight_uk": w_uk,
    }
    if mode not in ("BSND", "TND"):
        if mode in ("PA_BLK_BSND", "PA_BLK_NZ"):
            pages = b * ((s + cfg.block_size - 1) // cfg.block_size)
            out["cache_index"] = torch.arange(pages, dtype=torch.int64)
            out["actual_seq_len"] = torch.tensor(
                [s * (i + 1) for i in range(b)], dtype=torch.int32
            )
        else:
            out["cache_index"] = torch.arange(t, dtype=torch.int64)

    wq, kvq = cfg.weight_quant_mode, cfg.kv_cache_quant_mode

    if wq == 0:
        out["token_x"] = token_x_bf16
        out["weight_dq"] = w_dq_bf16
        out["weight_uq_qr"] = w_uq_qr_bf16
        out["weight_dkv_kr"] = w_dkv_kr_bf16
    elif wq == 1:
        out["token_x"] = token_x_bf16
        out["weight_dq"] = w_dq_bf16
        w_uq_q, deq_uq = quantize_weight_per_channel(w_uq_qr_bf16)
        out["weight_uq_qr"] = w_uq_q
        out["weight_dkv_kr"] = w_dkv_kr_bf16
        out["dequant_scale_w_uq_qr"] = deq_uq
    elif wq == 2:
        token_q, deq_x = quantize_activation_per_token(token_x_bf16)
        w_dq_q, deq_dq = quantize_weight_per_channel(w_dq_bf16)
        w_uq_q, deq_uq = quantize_weight_per_channel(w_uq_qr_bf16)
        w_dkv_q, deq_dkv = quantize_weight_per_channel(w_dkv_kr_bf16)
        out["token_x"] = token_q
        out["weight_dq"] = w_dq_q
        out["weight_uq_qr"] = w_uq_q
        out["weight_dkv_kr"] = w_dkv_q
        out["dequant_scale_x"] = deq_x
        out["dequant_scale_w_dq"] = deq_dq
        out["dequant_scale_w_uq_qr"] = deq_uq
        out["dequant_scale_w_dkv_kr"] = deq_dkv
    elif wq == 3:
        tok_q, deq_x, deq_x_fp = quantize_mxfp8_along_last(token_x_bf16)
        w_dq_q, deq_dq, deq_dq_fp = quantize_weight_mxfp8(w_dq_bf16)
        w_uq_q, deq_uq, deq_uq_fp = quantize_weight_mxfp8(w_uq_qr_bf16)
        w_dkv_q, deq_dkv, deq_dkv_fp = quantize_weight_mxfp8(w_dkv_kr_bf16)
        out["token_x"] = tok_q
        out["weight_dq"] = w_dq_q
        out["weight_uq_qr"] = w_uq_q
        out["weight_dkv_kr"] = w_dkv_q
        out["dequant_scale_x"] = deq_x
        out["dequant_scale_w_dq"] = deq_dq
        out["dequant_scale_w_uq_qr"] = deq_uq
        out["dequant_scale_w_dkv_kr"] = deq_dkv
        out["_dequant_scale_x_fp"] = deq_x_fp
        out["_dequant_scale_w_dq_fp"] = deq_dq_fp
        out["_dequant_scale_w_uq_qr_fp"] = deq_uq_fp
        out["_dequant_scale_w_dkv_kr_fp"] = deq_dkv_fp
    elif wq == 4:
        # FP8 full quant：token/weight 为 FP8E4M3，scale 为 FLOAT（与 int8/hif8 同族，非 mxfp8 E8M0）
        token_q, deq_x = quantize_fp8_per_token(token_x_bf16)
        w_dq_q, deq_dq = quantize_fp8_per_channel(w_dq_bf16)
        w_uq_q, deq_uq = quantize_fp8_per_channel(w_uq_qr_bf16)
        w_dkv_q, deq_dkv = quantize_fp8_per_channel(w_dkv_kr_bf16)
        out["token_x"] = token_q
        out["weight_dq"] = w_dq_q
        out["weight_uq_qr"] = w_uq_q
        out["weight_dkv_kr"] = w_dkv_q
        out["dequant_scale_x"] = deq_x
        out["dequant_scale_w_dq"] = deq_dq
        out["dequant_scale_w_uq_qr"] = deq_uq
        out["dequant_scale_w_dkv_kr"] = deq_dkv
    elif wq == 5:
        token_q, deq_x = quantize_hif8_per_token(token_x_bf16)
        w_dq_q, deq_dq = quantize_hif8_per_channel(w_dq_bf16)
        w_uq_q, deq_uq = quantize_hif8_per_channel(w_uq_qr_bf16)
        w_dkv_q, deq_dkv = quantize_hif8_per_channel(w_dkv_kr_bf16)
        out["token_x"] = token_q
        out["weight_dq"] = w_dq_q
        out["weight_uq_qr"] = w_uq_q
        out["weight_dkv_kr"] = w_dkv_q
        out["dequant_scale_x"] = deq_x
        out["dequant_scale_w_dq"] = deq_dq
        out["dequant_scale_w_uq_qr"] = deq_uq
        out["dequant_scale_w_dkv_kr"] = deq_dkv
    else:
        raise ValueError(f"unsupported weight_quant_mode={wq}")

    kv_shape, kr_shape = _cache_kv_kr_shapes(cfg)
    kvq = cfg.kv_cache_quant_mode
    if kvq == 0:
        out["kv_cache"] = torch.zeros(*kv_shape, dtype=torch.bfloat16)
        out["kr_cache"] = torch.zeros(*kr_shape, dtype=torch.bfloat16)
    elif kvq == 1:
        if wq == 5:
            out["kv_cache"] = torch.zeros(*kv_shape, dtype=torch.uint8)
        elif wq == 2:
            out["kv_cache"] = torch.zeros(*kv_shape, dtype=torch.int8)
        else:
            out["kv_cache"] = torch.zeros(*kv_shape, dtype=torch.float8_e4m3fn)
        out["kr_cache"] = torch.zeros(*kr_shape, dtype=torch.bfloat16)
        out["quant_scale_ckv"] = torch.tensor([1.0], dtype=torch.float32)
    elif kvq == 2:
        out["kv_cache"] = torch.zeros(*kv_shape, dtype=torch.int8)
        out["kr_cache"] = torch.zeros(*kr_shape, dtype=torch.int8)
        out["quant_scale_ckv"] = torch.full((1, cfg.hckv), 10.0, dtype=torch.float32)
        out["quant_scale_ckr"] = torch.full((1, cfg.dr), 10.0, dtype=torch.float32)
    elif kvq == 3:
        if wq == 5:
            kv_dtype = torch.uint8
        elif wq in (3, 4):
            kv_dtype = torch.float8_e4m3fn
        else:
            kv_dtype = torch.int8
        out["kv_cache"] = torch.zeros(*kv_shape, dtype=kv_dtype)
        out["kr_cache"] = torch.zeros(*kr_shape, dtype=torch.bfloat16)
        # 文档：仅部分量化(wq=1)和 int8 全量化(wq=2) 的 pertile 需要 clip；wq=3/4/5 必须 nullptr
        if wq in (1, 2):
            out["k_nope_clip_alpha"] = torch.tensor([1.0], dtype=torch.float32)
    else:
        raise ValueError(f"unsupported kv_cache_quant_mode={kvq} in ATK golden")

    if mode == "BSND":
        out["token_x"] = out["token_x"].reshape(b, s, cfg.he)

    if device != "cpu":
        out = {k: v.to(device) if torch.is_tensor(v) else v for k, v in out.items()}
    return out


def _nz_tile(dtype: torch.dtype) -> int:
    """PA_NZ / PA_BLK_NZ 列打包粒度：1 字节类型 32，2 字节 16。"""
    if dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn):
        return 32
    return 16


def _prepare_golden_inp(inp: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """BSND token/rope 是 (B,S,*)，计算前合轴成 (T,*)。"""
    out = dict(inp)
    tx = out["token_x"]
    if tx.dim() == 3:
        out["token_x"] = tx.reshape(-1, tx.shape[-1])
        for name in ("rope_cos", "rope_sin"):
            ten = out.get(name)
            if isinstance(ten, torch.Tensor) and ten.dim() == 3:
                out[name] = ten.reshape(-1, ten.shape[-1])
    return out


def _unflatten_query(
    out_q: torch.Tensor, out_qrope: torch.Tensor, cfg: CaseConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    # BS 非合轴时 query / queryRope 为 (B, S, N, *)，与文档一致
    if int(getattr(cfg, "token_rank", 2)) == 3:
        b, s = _cfg_batch(cfg), _cfg_seq(cfg)
        return (
            out_q.reshape(b, s, cfg.n, cfg.hckv),
            out_qrope.reshape(b, s, cfg.n, cfg.dr),
        )
    return out_q, out_qrope


def _scatter_nz_pa(
    cache: torch.Tensor, rows: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    """对齐 V4 scatter_pa_nz：cache (BlockNum,BlockSize,N,H)，index 为全局 slot。"""
    bn, bs, n_heads, h = cache.shape
    t = rows.shape[0]
    tile = _nz_tile(rows.dtype)
    data_num = (h + tile - 1) // tile
    idx = index.reshape(-1).to(dtype=torch.long, device="cpu")
    cache_w, src_w = _byte_store(cache, rows)
    max_slot = bn * bs - 1
    for i in range(t):
        slot_g = int(idx[i].item())
        if slot_g < 0 or slot_g > max_slot:
            continue
        blk = slot_g // bs
        slot = slot_g % bs
        for d_i in range(data_num):
            in_blk = d_i * bs + slot
            bs_i = in_blk // data_num
            h0 = (in_blk % data_num) * tile
            in0 = d_i * tile
            in1 = min(in0 + tile, h)
            h1 = h0 + (in1 - in0)
            if h1 > h or bs_i >= bs:
                continue
            src = src_w[i, in0:in1]
            for nh in range(n_heads):
                cache_w[blk, bs_i, nh, h0:h1] = src
    return cache


def _byte_store(
    dst: torch.Tensor, src: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp8 不能直接 index_copy / 切片赋值，改走 uint8 视图。"""
    if src.dtype == torch.float8_e4m3fn:
        return dst.view(torch.uint8), src.view(torch.uint8)
    return dst, src


def _scatter_blk_bsnd(
    cache: torch.Tensor,
    rows: torch.Tensor,
    index: torch.Tensor,
    seq_lens: list,
    starts: list,
) -> torch.Tensor:
    """对齐 V4 scatter_pa_blk_bsnd：index 元素是物理 block id。"""
    n_blocks, blk, n_heads, h = cache.shape
    idx = index.reshape(-1).to(dtype=torch.long, device="cpu")
    cache_w, src_w = _byte_store(cache, rows)
    page_ptr = 0
    for b, (len_b, base) in enumerate(zip(seq_lens, starts)):
        if len_b <= 0:
            continue
        pages_b = (len_b + blk - 1) // blk
        for p in range(pages_b):
            pa = int(idx[page_ptr + p].item())
            if pa < 0 or pa >= n_blocks:
                continue
            start = base + p * blk
            end = min(start + blk, base + len_b)
            sz = end - start
            if sz <= 0:
                continue
            src = src_w[start:end, :h]
            cache_w[pa, :sz, :, :h] = src.unsqueeze(1).expand(
                sz, n_heads, src.shape[-1]
            )
        page_ptr += pages_b
    return cache


def _scatter_blk_nz(
    cache: torch.Tensor,
    rows: torch.Tensor,
    index: torch.Tensor,
    seq_lens: list,
    starts: list,
) -> torch.Tensor:
    """对齐 V4 scatter_pa_blk_nz。"""
    n_blocks, blk, n_heads, h = cache.shape
    tile = _nz_tile(rows.dtype)
    data_num = (h + tile - 1) // tile
    idx = index.reshape(-1).to(dtype=torch.long, device="cpu")
    cache_w, src_w = _byte_store(cache, rows)
    page_ptr = 0
    for len_b, base in zip(seq_lens, starts):
        pages_b = (max(len_b, 0) + blk - 1) // blk if len_b > 0 else 0
        for t in range(len_b):
            page_id = t // blk
            tok_off = t - page_id * blk
            pa = int(idx[page_ptr + page_id].item())
            if pa < 0 or pa >= n_blocks:
                continue
            inp_row = base + t
            for d_i in range(data_num):
                in_blk = d_i * blk + tok_off
                bs_i = in_blk // data_num
                h0 = (in_blk % data_num) * tile
                in0 = d_i * tile
                in1 = min(in0 + tile, h)
                h1 = h0 + (in1 - in0)
                src = src_w[inp_row, in0:in1]
                for nh in range(n_heads):
                    cache_w[pa, bs_i, nh, h0:h1] = src
        page_ptr += pages_b
    return cache


def _parse_blk_seq(cfg: CaseConfig, actual_seq_len: Optional[torch.Tensor], t: int):
    if isinstance(actual_seq_len, torch.Tensor) and actual_seq_len.numel() > 0:
        prefix = [int(x) for x in actual_seq_len.reshape(-1).tolist()]
        lens = [prefix[0]] + [prefix[i] - prefix[i - 1] for i in range(1, len(prefix))]
        starts = [0] + prefix[:-1]
        return lens, starts
    b, s = _cfg_batch(cfg), _cfg_seq(cfg)
    if b * s != t and b == 1:
        s = t
    return [s] * b, [i * s for i in range(b)]


def _scatter_kv(
    norm2: torch.Tensor,
    rotary_k: torch.Tensor,
    cache_index: Optional[torch.Tensor],
    cfg: CaseConfig,
    actual_seq_len: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按 cacheMode 写入 kv/kr。对齐 V4 cal_mlaprolog scatter1/scatter2。"""
    mode = _cache_mode(cfg)
    t = int(norm2.shape[0])
    nkv, dr = cfg.nkv, cfg.dr
    h_kv = int(norm2.shape[-1])
    device, dtype_k = norm2.device, norm2.dtype
    ckv = norm2.reshape(t, h_kv)
    skip_kr = int(cfg.ckvkr_repo_mode) == 1 or rotary_k.numel() == 0
    dtype_r = rotary_k.dtype if not skip_kr else torch.bfloat16
    kr = rotary_k.reshape(t, dr) if not skip_kr else None

    def _empty_kr():
        return torch.zeros(0, dtype=torch.bfloat16, device=device)

    def _write_slots(dst, src_rows):
        """dst (slots, nkv, H) sequential or via index_copy; fp8 走 uint8 视图。"""
        n_heads = dst.shape[1]
        h = dst.shape[-1]
        src = src_rows.reshape(t, h)
        use_u8 = src.dtype in (torch.float8_e4m3fn, torch.uint8)
        dst_w = dst.view(torch.uint8) if use_u8 else dst
        src_w = src.view(torch.uint8) if use_u8 else src
        payload = src_w.unsqueeze(1).expand(t, n_heads, src_w.shape[-1]).contiguous()
        return dst_w, payload

    if mode == "TND":
        kv_cache = torch.zeros(t, nkv, h_kv, dtype=dtype_k, device=device)
        dst_w, payload = _write_slots(kv_cache, ckv)
        dst_w.copy_(payload)
        if skip_kr:
            return kv_cache, _empty_kr()
        kr_cache = torch.zeros(t, nkv, dr, dtype=dtype_r, device=device)
        kr_cache.copy_(kr.unsqueeze(1).expand(t, nkv, dr))
        return kv_cache, kr_cache

    if mode == "BSND":
        b, s = _cfg_batch(cfg), _cfg_seq(cfg)
        kv_cache = torch.zeros(b, s, nkv, h_kv, dtype=dtype_k, device=device)
        dst_w, payload = _write_slots(kv_cache.view(t, nkv, h_kv), ckv)
        dst_w.copy_(payload)
        if skip_kr:
            return kv_cache, _empty_kr()
        kr_cache = torch.zeros(b, s, nkv, dr, dtype=dtype_r, device=device)
        kr_cache.view(t, nkv, dr).copy_(kr.unsqueeze(1).expand(t, nkv, dr))
        return kv_cache, kr_cache

    bn, bs = cfg.block_num, cfg.block_size
    kv_cache = torch.zeros(bn, bs, nkv, h_kv, dtype=dtype_k, device=device)
    kr_cache = (
        _empty_kr()
        if skip_kr
        else torch.zeros(bn, bs, nkv, dr, dtype=dtype_r, device=device)
    )
    # EMPTY_CACHE：BlockNum=0，kernel 不写 cache，标杆同样保持空 cache
    if cache_index is None or bn == 0:
        return kv_cache, kr_cache

    if mode in ("PA_BLK_BSND", "PA_BLK_NZ"):
        lens, starts = _parse_blk_seq(cfg, actual_seq_len, t)
        kv_cpu = kv_cache.cpu()
        ckv_cpu = ckv.detach().cpu()
        if mode == "PA_BLK_NZ":
            _scatter_blk_nz(kv_cpu, ckv_cpu, cache_index, lens, starts)
            if not skip_kr:
                kr_cpu = kr_cache.cpu()
                _scatter_blk_nz(kr_cpu, kr.detach().cpu(), cache_index, lens, starts)
                kr_cache = kr_cpu.to(device=device)
        else:
            _scatter_blk_bsnd(kv_cpu, ckv_cpu, cache_index, lens, starts)
            if not skip_kr:
                kr_cpu = kr_cache.cpu()
                _scatter_blk_bsnd(kr_cpu, kr.detach().cpu(), cache_index, lens, starts)
                kr_cache = kr_cpu.to(device=device)
        return kv_cpu.to(device=device), kr_cache

    if mode == "PA_NZ":
        kv_cpu = kv_cache.cpu()
        _scatter_nz_pa(kv_cpu, ckv.detach().cpu(), cache_index)
        if not skip_kr:
            kr_cpu = kr_cache.cpu()
            _scatter_nz_pa(kr_cpu, kr.detach().cpu(), cache_index)
            kr_cache = kr_cpu.to(device=device)
        return kv_cpu.to(device=device), kr_cache

    # PA_BSND：全局 slot 写入 (BlockNum*BlockSize, Nkv, H)
    kv_flat = kv_cache.reshape(bn * bs, nkv, h_kv)
    idx = cache_index.to(device=device, dtype=torch.long).reshape(-1)
    dst_w, payload = _write_slots(kv_flat, ckv)
    dst_w.index_copy_(0, idx, payload)
    if not skip_kr:
        kr_flat = kr_cache.reshape(bn * bs, nkv, dr)
        kr_flat.index_copy_(0, idx, kr.unsqueeze(1).expand(t, nkv, dr).contiguous())
    return kv_cache, kr_cache


def cal_mlaprolog_cpu(
    inp: Dict[str, torch.Tensor],
    cfg: CaseConfig,
    backend: str = "cpu",
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """V4 CPU 真值；backend='npu' 时同一条图换成 torch_npu 小算子拼接。"""
    inp = _prepare_golden_inp(inp)
    if int(cfg.t) == 0:
        # EMPTY_QUERY：kernel 直接返回，标杆给出同 shape 的空 query，cache 保持输入
        n, hckv, dr = cfg.n, cfg.hckv, cfg.dr
        tx = inp["token_x"]
        if tx.dim() == 3:
            b, s = int(tx.shape[0]), int(tx.shape[1])
            q = torch.zeros(b, s, n, hckv, dtype=torch.float32)
            qr = torch.zeros(b, s, n, dr, dtype=torch.float32)
        else:
            q = torch.zeros(0, n, hckv, dtype=torch.float32)
            qr = torch.zeros(0, n, dr, dtype=torch.float32)
        return q, qr, inp["kv_cache"], inp["kr_cache"]
    t = int(cfg.t)
    hckv, d, dr, n = cfg.hckv, cfg.d, cfg.dr, cfg.n
    wq, kvq = cfg.weight_quant_mode, cfg.kv_cache_quant_mode
    qq = cfg.query_quant_mode
    npu = backend == "npu"
    if npu and not _HAS_TORCH_NPU:
        raise RuntimeError("torch_npu is required for NPU golden")

    def to_dev(x: torch.Tensor, dtype=None) -> torch.Tensor:
        y = x.contiguous()
        if npu:
            y = y.to(device=device)
        return y.to(dtype) if dtype is not None else y

    def mm_fp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(to_dev(a), to_dev(b))

    def mm_i8(a_i8: torch.Tensor, w_i8: torch.Tensor) -> torch.Tensor:
        if npu:
            w_nz = _npu_to_nz(w_i8.to(torch.int8), device)
            return _npu_i8_mm_i32(to_dev(a_i8, torch.int8), w_nz, int(w_i8.shape[-1]))
        return int8_matmul_i32(a_i8.to(torch.int8), w_i8.to(torch.int8))

    def do_rms(
        x: torch.Tensor, gamma: torch.Tensor, eps: float, scale: float
    ) -> torch.Tensor:
        if npu:
            return _npu_rms(
                to_dev(x, torch.float32), to_dev(gamma, torch.float32), eps, scale
            ).to(torch.float32)
        return rms_norm(x, gamma, eps, scale)

    def do_rope(x: torch.Tensor, cos_t, sin_t) -> torch.Tensor:
        if npu:
            return _npu_rope(
                to_dev(x.contiguous(), torch.bfloat16),
                to_dev(cos_t, torch.bfloat16) if use_rope else None,
                to_dev(sin_t, torch.bfloat16) if use_rope else None,
                use_rope,
                cfg.rope_style,
            ).to(torch.float32)
        return apply_rope(x, cos_t, sin_t, use_rope, cfg.rope_style).to(torch.float32)

    def dyn_i8(x: torch.Tensor, smooth):
        if npu:
            xb = to_dev(x, torch.bfloat16)
            smooth_arg = None
            if (
                isinstance(smooth, torch.Tensor)
                and smooth.dtype != torch.bool
                and smooth.numel() == xb.shape[-1]
            ):
                smooth_arg = to_dev(smooth, torch.bfloat16).reshape(1, -1)
            yq, deq_pt = torch_npu.npu_dynamic_quant(xb, smooth_scales=smooth_arg)
            return yq, deq_pt.reshape(-1, 1).to(torch.float32)
        y, scale = dynamic_quant(x, smooth)
        return y.to(torch.int8), scale

    gamma_cq = inp["rmsnorm_gamma_cq"].to(torch.float32)
    gamma_ckv = inp["rmsnorm_gamma_ckv"].to(torch.float32)
    _rs = inp.get("rope_sin")
    use_rope = bool(cfg.do_rope) or (
        isinstance(_rs, torch.Tensor) and _rs.dtype != torch.bool and _rs.numel() > 0
    )
    cos = inp["rope_cos"].to(torch.float32) if use_rope else None
    sin = inp["rope_sin"].to(torch.float32) if use_rope else None
    cache_index = _get_cache_index(inp)
    if cache_index is not None:
        cache_index = cache_index.cpu()
    w_uk = inp["weight_uk"].to(torch.float32)

    # ---- mm1 (MatmulCq) ----
    if wq == 2:
        mm1_i32 = mm_i8(inp["token_x"], inp["weight_dq"])
        mm1 = (
            mm1_i32.to(torch.float32)
            * to_dev(inp["dequant_scale_x"], torch.float32)
            * to_dev(inp["dequant_scale_w_dq"], torch.float32)
        )
        token_x_for_mm4 = inp["token_x"].to(torch.int8)
    elif wq == 3:
        scale_x = inp.get("_dequant_scale_x_fp", inp["dequant_scale_x"])
        scale_w = inp.get("_dequant_scale_w_dq_fp", inp["dequant_scale_w_dq"])
        if npu:
            token_x_dq = _dequant_mxfp8_tensor(inp["token_x"], scale_x, device)
            w_dq_dq = _dequant_mxfp8_weight(inp["weight_dq"], scale_w, device)
        else:
            token_x_dq = dequant_mxfp8_along_last(
                inp["token_x"].detach().cpu(), scale_x.detach().cpu()
            )
            w_dq_dq = dequant_weight_mxfp8(
                inp["weight_dq"].detach().cpu(), scale_w.detach().cpu()
            )
        mm1 = _cube_fp32_matmul(
            token_x_dq.to(torch.float32),
            w_dq_dq.to(torch.float32),
            k_tile=256,
            device=device if npu else None,
        )
        token_x_for_mm4 = token_x_dq
    elif wq in (4, 5):
        token_x_fp, w_dq_fp = _fp8_family_to_fp32(inp["token_x"], inp["weight_dq"], wq)
        if npu:
            mm1 = (
                _npu_bf16_matmul(token_x_fp, w_dq_fp, device)
                * to_dev(inp["dequant_scale_x"], torch.float32)
                * to_dev(inp["dequant_scale_w_dq"], torch.float32)
            )
        else:
            mm1 = (
                mm_fp(token_x_fp, w_dq_fp)
                * to_dev(inp["dequant_scale_x"], torch.float32)
                * to_dev(inp["dequant_scale_w_dq"], torch.float32)
            )
        token_x_for_mm4 = token_x_fp
    else:
        token_x_fp = inp["token_x"].to(torch.float32)
        w_dq = inp["weight_dq"].to(torch.float32)
        token_x_for_mm4 = token_x_fp
        if wq == 0:
            mm1 = _to_bf16_fp32_rint(
                _cube_fp32_matmul(
                    token_x_fp, w_dq, k_tile=256, device=device if npu else None
                )
            )
        elif npu:
            mm1 = torch.matmul(
                to_dev(inp["token_x"], torch.bfloat16),
                to_dev(inp["weight_dq"], torch.bfloat16),
            ).to(torch.float32)
        else:
            mm1 = mm_fp(token_x_fp, w_dq).to(torch.bfloat16).to(torch.float32)

    norm1 = do_rms(mm1, gamma_cq, cfg.eps_cq, cfg.qc_qr_scale)

    if wq in (1, 2):
        norm1_q, deq_qcqr = dyn_i8(norm1, inp.get("smooth_scales_cq"))
        mm2_i32 = mm_i8(norm1_q, inp["weight_uq_qr"])
        scale_w = to_dev(inp["dequant_scale_w_uq_qr"], torch.float32)
        deq_qcqr = deq_qcqr if deq_qcqr.dim() == 2 else deq_qcqr.reshape(-1, 1)
        mm2 = mm2_i32.to(torch.float32) * to_dev(deq_qcqr, torch.float32) * scale_w
    elif wq == 3:
        scale_uq = inp.get("_dequant_scale_w_uq_qr_fp", inp["dequant_scale_w_uq_qr"])
        if npu:
            w_uq = _dequant_mxfp8_weight(inp["weight_uq_qr"], scale_uq, device)
            norm1_dq = _npu_dynamic_mx_dequant(to_dev(norm1, torch.float32))
        else:
            w_uq = dequant_weight_mxfp8(
                inp["weight_uq_qr"].detach().cpu(), scale_uq.detach().cpu()
            )
            norm1_dq = dynamic_mx_quant_dequant_cq(
                norm1.to(torch.bfloat16).to(torch.float32)
            )
        mm2 = _cube_fp32_matmul(
            norm1_dq.to(torch.float32),
            w_uq.to(torch.float32),
            k_tile=64,
            device=device if npu else None,
        )
    elif wq in (4, 5):
        norm1_src = norm1.cpu() if npu else norm1
        if wq == 4:
            norm1_q, deq_qcqr = dynamic_quant_fp8(
                norm1_src, inp.get("smooth_scales_cq")
            )
            a_fp, w_uq_fp = (
                norm1_q.to(torch.float32),
                inp["weight_uq_qr"].to(torch.float32),
            )
        else:
            norm1_q, deq_qcqr = dynamic_quant_hif8(
                norm1_src, inp.get("smooth_scales_cq")
            )
            a_fp, w_uq_fp = _hif8_decode(norm1_q), _hif8_decode(inp["weight_uq_qr"])
        if npu:
            mm2 = (
                _npu_bf16_matmul(a_fp, w_uq_fp, device)
                * to_dev(deq_qcqr, torch.float32)
                * to_dev(inp["dequant_scale_w_uq_qr"], torch.float32)
            )
        else:
            mm2 = (
                mm_fp(a_fp, w_uq_fp)
                * to_dev(deq_qcqr, torch.float32)
                * to_dev(inp["dequant_scale_w_uq_qr"], torch.float32)
            )
    else:
        # wq=0：CPU/NPU 同一套 CAST_RINT(norm1) → 切 K GEMM → CAST_RINT；NPU 段内 torch.matmul(bf16)
        norm1_mm = (
            _to_bf16_fp32_rint(norm1)
            if wq == 0
            else norm1.to(torch.bfloat16).to(torch.float32)
        )
        if wq == 0:
            mm2 = _cube_fp32_matmul(
                norm1_mm,
                inp["weight_uq_qr"].to(torch.float32),
                k_tile=64,
                device=device if npu else None,
            )
        elif npu:
            mm2 = torch.matmul(
                to_dev(norm1_mm, torch.bfloat16),
                to_dev(inp["weight_uq_qr"], torch.bfloat16),
            )
        else:
            mm2 = mm_fp(norm1_mm, inp["weight_uq_qr"].to(torch.float32))
        mm2 = (
            _to_bf16_fp32_rint(mm2)
            if wq == 0
            else mm2.to(torch.bfloat16).to(torch.float32)
        )

    mm2 = mm2.reshape(t, n, d + dr)
    q_nope, q_rope_in = mm2[:, :, :d], mm2[:, :, d:]
    if wq in (1, 2, 4, 5):
        q_nope = q_nope.to(torch.bfloat16).to(torch.float32)

    if wq == 0:
        q_nope_t = q_nope.transpose(0, 1)
        out_q = torch.zeros((n, t, hckv), dtype=torch.float32)
        mm_dev = device if npu else None
        for i in range(n):
            out_q[i] = _cube_fp32_matmul(q_nope_t[i], w_uk[i], k_tile=64, device=mm_dev)
        out_q = _to_bf16_fp32_rint(out_q.transpose(0, 1))
    elif wq == 2 and qq == 1:
        qn = _to_bf16_fp32_rint(q_nope)
        w_uk_f = w_uk.to(torch.float32)
        out_q = torch.zeros((t, n, hckv), dtype=torch.float32)
        mm_dev = device if npu else None
        for i in range(n):
            out_q[:, i, :] = _cube_fp32_matmul(
                qn[:, i, :], w_uk_f[i], k_tile=64, device=mm_dev
            )
    elif npu:
        out_q = _npu_matmul_qn(
            to_dev(q_nope, torch.bfloat16), to_dev(w_uk, torch.bfloat16)
        ).to(torch.float32)
    else:
        q_nope_bf16 = q_nope.to(torch.bfloat16)
        w_uk_b = w_uk.to(torch.bfloat16)
        out_q = torch.zeros((n, t, hckv), dtype=torch.bfloat16)
        q_nope_t = q_nope_bf16.transpose(0, 1)
        for i in range(n):
            out_q[i] = torch.matmul(
                q_nope_t[i].to(torch.float32), w_uk_b[i].to(torch.float32)
            ).to(torch.bfloat16)
        out_q = out_q.transpose(0, 1).to(torch.float32)

    deq_scale_q_nope = None
    # V4：queryQuantMode=1 且 full-quant (wq=2/3/4/5) + kv pertensor
    if qq == 1 and wq in (2, 3, 4, 5) and kvq == 1:
        out_q_cpu = out_q.cpu() if npu else out_q
        if wq == 2:
            out_for_q = _to_bf16_fp32_rint(out_q_cpu)
            if npu and _HAS_TORCH_NPU:
                lead = out_for_q.shape[:-1]
                h = int(out_for_q.shape[-1])
                flat = to_dev(out_for_q.reshape(-1, h), torch.bfloat16)
                yq, deq_pt = torch_npu.npu_dynamic_quant(flat)
                out_q_q = yq.reshape(*lead, h)
                deq_scale_q_nope = deq_pt.reshape(*lead, 1)
            else:
                out_q_q, deq_scale_q_nope = dynamic_quant_q_nope_int8(out_for_q)
            # 标杆比对用 fp32 反量化（与 wq=5 HiFloat8 升 fp32 一致），算子侧仍为 int8
            out_q = out_q_q.to(torch.float32) * deq_scale_q_nope.to(torch.float32)
            if npu:
                out_q = out_q.to(device)
        elif wq == 5:
            # kernel 的 q^N 以 bf16 进入 DynamicQuant，scale = rowmax/32768，再 cast 到 hifloat8。
            # 不能走 npu_dynamic_quant：那是 int8（/127），会同时带偏 query 和 queryRope 的 scale。
            q_bf = _to_bf16_fp32_rint(out_q_cpu)
            out_q_q, deq_scale_q_nope = dynamic_quant_q_nope_hif8(q_bf)
            out_q = _hif8_decode(out_q_q) * deq_scale_q_nope.to(torch.float32)
            if npu:
                out_q = out_q.to(device)
        else:
            out_q_q, deq_scale_q_nope = dynamic_quant_q_nope_fp8(out_q_cpu)
            out_q = out_q_q.to(torch.float32) * deq_scale_q_nope.to(torch.float32)

    out_qrope = do_rope(q_rope_in, cos, sin)
    # MulQr 的输入是 bf16 rope 结果，再乘 quantScaleCkv/dequantScaleQNope，写回 bf16（CAST_RINT）
    if wq == 5:
        out_qrope = _to_bf16_fp32_rint(out_qrope)
    if deq_scale_q_nope is not None:
        qckv = inp["quant_scale_ckv"].to(torch.float32).reshape(1, 1).item()
        scale = deq_scale_q_nope.to(out_qrope.device).squeeze(-1).unsqueeze(-1)
        out_qrope = out_qrope * (qckv / scale)
    if wq in (0, 1, 5):
        out_qrope = _to_bf16_fp32_rint(out_qrope)
    else:
        out_qrope = out_qrope.to(torch.bfloat16).to(torch.float32)

    # ---- mm4 (MatmulCkvKr) ----
    if wq == 2:
        mm4_i32 = mm_i8(token_x_for_mm4, inp["weight_dkv_kr"])
        mm4 = (
            mm4_i32.to(torch.float32)
            * to_dev(inp["dequant_scale_x"], torch.float32)
            * to_dev(inp["dequant_scale_w_dkv_kr"], torch.float32)
        )
    elif wq == 3:
        scale_dkv = inp.get("_dequant_scale_w_dkv_kr_fp", inp["dequant_scale_w_dkv_kr"])
        if npu:
            w_dkv = _dequant_mxfp8_weight(inp["weight_dkv_kr"], scale_dkv, device)
        else:
            w_dkv = dequant_weight_mxfp8(
                inp["weight_dkv_kr"].detach().cpu(), scale_dkv.detach().cpu()
            )
        mm4 = _cube_fp32_matmul(
            token_x_for_mm4.to(torch.float32),
            w_dkv.to(torch.float32),
            k_tile=256,
            device=device if npu else None,
        )
    elif wq in (4, 5):
        if wq == 5:
            w_dkv_fp = _hif8_decode(inp["weight_dkv_kr"])
        else:
            w_dkv_fp = inp["weight_dkv_kr"].to(torch.float32)
        mm4 = (
            mm_fp(token_x_for_mm4.to(torch.float32), w_dkv_fp)
            * to_dev(inp["dequant_scale_x"], torch.float32)
            * to_dev(inp["dequant_scale_w_dkv_kr"], torch.float32)
        )
    else:
        if wq == 0:
            mm4 = _to_bf16_fp32_rint(
                _cube_fp32_matmul(
                    token_x_for_mm4.to(torch.float32),
                    inp["weight_dkv_kr"].to(torch.float32),
                    k_tile=256,
                    device=device if npu else None,
                )
            )
        elif npu:
            mm4 = torch.matmul(
                to_dev(token_x_for_mm4, torch.bfloat16),
                to_dev(inp["weight_dkv_kr"], torch.bfloat16),
            ).to(torch.float32)
        else:
            mm4 = mm_fp(
                token_x_for_mm4.to(torch.float32),
                inp["weight_dkv_kr"].to(torch.float32),
            )
            mm4 = mm4.to(torch.bfloat16).to(torch.float32)

    k_nope = mm4[:, :hckv]
    k_rope_in = mm4[:, hckv:]
    rotary_k = do_rope(k_rope_in, cos, sin)
    if wq == 1 and kvq == 2:
        rotary_k_store = quant_s8(
            rotary_k.cpu() if npu else rotary_k, inp["quant_scale_ckr"].detach().cpu()
        )
    else:
        rotary_k_store = rotary_k.to(torch.bfloat16).to(torch.float32)

    norm2 = do_rms(k_nope, gamma_ckv, cfg.eps_ckv, cfg.kc_scale).to(torch.float32)
    norm2_cpu = norm2.cpu() if npu else norm2
    rotary_cpu = rotary_k_store.cpu() if npu else rotary_k_store
    if kvq == 3:
        packed = _pack_kv_pertile(
            norm2_cpu, rotary_cpu.to(torch.float32), cfg, inp.get("k_nope_clip_alpha")
        )
        kv_dtype = inp["kv_cache"].dtype
        if kv_dtype == torch.float8_e4m3fn:
            norm2 = packed.view(torch.uint8).view(torch.float8_e4m3fn)
        elif kv_dtype == torch.uint8:
            norm2 = packed.view(torch.uint8)
        else:
            norm2 = packed
        rotary_k_store = rotary_cpu
    elif kvq == 2:
        norm2 = quant_s8(norm2_cpu, inp["quant_scale_ckv"].detach().cpu())
        rotary_k_store = rotary_cpu
    elif kvq == 1:
        scale = inp["quant_scale_ckv"].detach().cpu()
        if wq == 5:
            norm2 = quant_ckv_hif8_per_tensor(norm2_cpu, scale)
        elif wq == 2:
            norm2 = quant_s8(norm2_cpu, scale)
        else:
            norm2 = quant_ckv_fp8_per_tensor(norm2_cpu, scale)
        rotary_k_store = rotary_cpu
    else:
        norm2 = norm2_cpu.to(torch.bfloat16).to(torch.float32)
        rotary_k_store = rotary_cpu

    if qq == 0 and wq in (3, 4, 5):
        out_q = _to_bf16_fp32_rint(out_q.cpu() if npu else out_q)
        if npu:
            out_q = out_q.to(device)
    kv_cache, kr_cache = _scatter_kv(
        norm2, rotary_k_store, cache_index, cfg, inp.get("actual_seq_len")
    )
    out_q, out_qrope = _unflatten_query(
        out_q.cpu() if npu else out_q, out_qrope.cpu() if npu else out_qrope, cfg
    )
    return out_q, out_qrope, kv_cache, kr_cache


def _npu_rms(
    x: torch.Tensor, gamma: torch.Tensor, eps: float, scale: float
) -> torch.Tensor:
    if not _HAS_TORCH_NPU:
        raise RuntimeError("torch_npu is required for NPU golden")
    if x.dtype == torch.float32:
        y = rms_norm(x, gamma.to(dtype=x.dtype, device=x.device), eps, 1.0)
        return y * scale if scale != 1.0 else y
    y, _ = torch_npu.npu_rms_norm(x.to(torch.bfloat16), gamma.to(torch.bfloat16), eps)
    return y * scale if scale != 1.0 else y


def _npu_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    do_rope: bool,
    rope_style: str = "original",
) -> torch.Tensor:
    """NPU small-op RoPE. V4 doRope=false: Qrope/krcache passthrough (no rotary)."""
    if not do_rope:
        return x
    if not _HAS_TORCH_NPU:
        raise RuntimeError("torch_npu is required for NPU golden")
    if rope_style == "vf" or x.dtype == torch.float32:
        return apply_rope(x, cos, sin, True, rope_style)
    x = x.to(torch.bfloat16)
    cos = cos.to(torch.bfloat16)
    sin = sin.to(torch.bfloat16)
    xx = interleave_to_half(x)
    if xx.dim() == 2:
        xx3 = xx.unsqueeze(1)
        c = cos.unsqueeze(1) if cos.dim() == 2 else cos
        s = sin.unsqueeze(1) if sin.dim() == 2 else sin
        y = torch_npu.npu_rotary_mul(xx3, c, s, rotary_mode="half")
        return y.squeeze(1)
    c = cos.unsqueeze(1).expand_as(xx) if cos.dim() == 2 else cos
    s = sin.unsqueeze(1).expand_as(xx) if sin.dim() == 2 else sin
    return torch_npu.npu_rotary_mul(xx, c, s, rotary_mode="half")


def _npu_matmul_qn(q_nope: torch.Tensor, w_uk: torch.Tensor) -> torch.Tensor:
    """q_nope (T,N,D) × w_uk (N,D,Hckv)：ND torch.matmul(bf16)，不 format_cast NZ。"""
    qn = q_nope.transpose(0, 1).contiguous()
    out_n = torch.matmul(qn, w_uk)
    return out_n.transpose(0, 1).contiguous()


def _npu_to_nz(weight: torch.Tensor, device: str) -> torch.Tensor:
    """Cast ND weight to FRACTAL_NZ. 仅量化 int8 mm 使用，bf16 标杆不走这条。"""
    w = weight.contiguous().to(device)
    return torch_npu.npu_format_cast(w, FRACTAL_NZ_FORMAT)


def _npu_i8_mm_i32(x_i8: torch.Tensor, w_nz: torch.Tensor, n_out: int) -> torch.Tensor:
    """int8×int8(NZ) → int32 via npu_quant_matmul (scale=1, dequant done by caller)."""
    scale1 = torch.ones(int(n_out), dtype=torch.float32, device=x_i8.device)
    return torch_npu.npu_quant_matmul(x_i8, w_nz, scale1, output_dtype=torch.int32)


def _e8m0_bytes_to_fp_scale(scale: torch.Tensor) -> torch.Tensor:
    """float8_e8m0fnu / uint8 share → float32 scale (= 2^(share-127))."""
    if scale.dtype == torch.float8_e8m0fnu:
        u = scale.view(torch.uint8)
    else:
        u = scale.to(torch.uint8)
    return torch.pow(
        torch.tensor(2.0, device=scale.device, dtype=torch.float32),
        u.to(torch.float32) - 127.0,
    )


def _mx_fp_scale(
    inp: Dict[str, torch.Tensor], fp_key: str, e8_key: str, device: str
) -> torch.Tensor:
    if fp_key in inp and isinstance(inp[fp_key], torch.Tensor):
        return inp[fp_key].contiguous().to(device=device, dtype=torch.float32)
    return _e8m0_bytes_to_fp_scale(inp[e8_key].contiguous().to(device))


def _dequant_mxfp8_tensor(
    x_fp8: torch.Tensor, scale: torch.Tensor, device: str
) -> torch.Tensor:
    """mxfp8 张量 + E8M0/FP scale → fp32（标杆 NPU 与 Cube 同设备反量化）。"""
    x = x_fp8.contiguous().to(device=device)
    if scale.dtype == torch.float32:
        scale_fp = scale.contiguous().to(device=device, dtype=torch.float32)
    else:
        scale_fp = _e8m0_bytes_to_fp_scale(scale.contiguous().to(device))
    return dequant_mxfp8_along_last(x, scale_fp)


def _dequant_mxfp8_weight(
    w_fp8: torch.Tensor, scale: torch.Tensor, device: str
) -> torch.Tensor:
    w = w_fp8.contiguous().to(device=device)
    if scale.dtype == torch.float32:
        scale_fp = scale.contiguous().to(device=device, dtype=torch.float32)
    else:
        scale_fp = _e8m0_bytes_to_fp_scale(scale.contiguous().to(device))
    return dequant_weight_mxfp8(w, scale_fp)


def _npu_bf16_matmul(a: torch.Tensor, b: torch.Tensor, device: str) -> torch.Tensor:
    return torch.matmul(
        a.contiguous().to(device=device, dtype=torch.bfloat16),
        b.contiguous().to(device=device, dtype=torch.bfloat16),
    ).to(torch.float32)


def _npu_dynamic_mx_dequant(x: torch.Tensor) -> torch.Tensor:
    """Hardware mxfp8 dynamic quant then dequant back to float32 (block=32)."""
    x_bf16 = x.to(torch.bfloat16)
    y_fp8, scale_u8 = torch_npu.npu_dynamic_mx_quant(
        x_bf16,
        axis=-1,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
        block_size=MXFP8_GRP,
    )
    scale_fp = _e8m0_bytes_to_fp_scale(scale_u8).reshape(x_bf16.shape[0], -1)
    return dequant_mxfp8_along_last(y_fp8, scale_fp)


def cal_mlaprolog_npu(
    inp: Dict[str, torch.Tensor], cfg: CaseConfig, device: str = "npu:0"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """与 CPU 真值同一条计算图，节点换成 torch_npu 小算子（NZ int8 mm / npu_rms / npu_dynamic_quant / rope）。"""
    outs = cal_mlaprolog_cpu(inp, cfg, backend="npu", device=device)
    if _HAS_TORCH_NPU:
        torch.npu.synchronize()
    return outs


def cast_weight_nz(tensor: torch.Tensor, device: str) -> torch.Tensor:
    if not _HAS_TORCH_NPU:
        return tensor.to(device)
    w = tensor.contiguous().to(device)
    try:
        return torch_npu.npu_format_cast(w, FRACTAL_NZ_FORMAT)
    except Exception:
        # uint8 伪装的 HiFloat8 可能不支持 format_cast，退回 ND，acl format 仍标 NZ
        return w


# ===== noncontig (inlined) =====

import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

# 非连续 backing 的填充值：算子只允许写逻辑行，padding 行必须原样保留。
# 取 3 是因为 bf16 / int8 / fp8_e4m3fn 都能精确表示，便于逐值比对。
PADDING_SENTINEL = 3.0

_NC_NAME_RE = re.compile(r"\bnc(\d+)\b")


def parse_stride_factor(case_name: str) -> int:
    """从用例名解析首轴 stride 放大倍数，``nc1``/无标记表示连续。"""
    match = _NC_NAME_RE.search(case_name or "")
    return max(1, int(match.group(1))) if match else 1


@dataclass
class NonContigCache:
    """一个 cache 的非连续视图及其 backing。"""

    view: torch.Tensor
    backing: torch.Tensor
    factor: int

    @property
    def padding_rows(self) -> torch.Tensor:
        """backing 中不属于逻辑视图的行（拉回 CPU，避免 NPU 上 fp8 不支持 index）。"""
        backing = self.backing.detach().cpu()
        keep = [i for i in range(backing.shape[0]) if i % self.factor != 0]
        return backing[keep]

    def padding_is_intact(self) -> bool:
        rows = self.padding_rows
        if rows.numel() == 0:
            return True
        if rows.dtype == torch.float8_e4m3fn:
            sentinel = (
                torch.tensor(PADDING_SENTINEL, dtype=torch.float32)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
                .item()
            )
            return bool(torch.all(rows.view(torch.uint8) == sentinel))
        return bool(torch.all(rows.float() == PADDING_SENTINEL))


def make_dim0_noncontig(tensor: torch.Tensor, factor: int) -> NonContigCache:
    """把 tensor 放进一个首轴放大 factor 倍的 backing，返回只有 dim0 非连续的视图。

    视图 stride = (factor * 连续stride0, 其余连续 stride)，逻辑行落在 backing 的
    第 0, factor, 2*factor ... 行，行间空洞填 sentinel。

    构造一律在 CPU 上做：NPU 上对 float8 的 as_strided/copy_ 会走 index_high_dims
    并触发 161002。
    """
    if factor <= 1:
        raise ValueError(f"stride factor must be > 1, got {factor}")
    if tensor.numel() == 0:
        return NonContigCache(view=tensor, backing=tensor, factor=1)

    shape = tuple(tensor.shape)
    rest = shape[1:]
    inner = 1
    for dim in rest:
        inner *= dim

    device = tensor.device
    src = tensor.detach().cpu()
    backing = torch.full(
        (shape[0] * factor,) + rest, PADDING_SENTINEL, dtype=torch.float32
    ).to(dtype=src.dtype)
    strides = (factor * inner,) + tuple(backing.stride()[1:])
    view = backing.as_strided(shape, strides)
    view.copy_(src)
    if view.is_contiguous():
        raise RuntimeError("failed to build a dim0 non-contiguous view")

    if device.type != "cpu":
        backing = backing.to(device)
        view = backing.as_strided(shape, strides)
    return NonContigCache(view=view, backing=backing, factor=factor)


def describe(caches: Dict[str, Optional[NonContigCache]]) -> str:
    parts = []
    for name, cache in caches.items():
        if cache is None:
            parts.append(f"{name}=contiguous")
        else:
            parts.append(
                f"{name}=stride0:{cache.view.stride(0)}(x{cache.factor}) "
                f"backing_dim0:{cache.backing.shape[0]}"
            )
    return " ".join(parts)


def padding_report(caches: Dict[str, Optional[NonContigCache]]) -> Tuple[bool, str]:
    ok = True
    details = []
    for name, cache in caches.items():
        if cache is None:
            details.append(f"{name}:n/a")
            continue
        intact = cache.padding_is_intact()
        ok = ok and intact
        details.append(f"{name}:{'intact' if intact else 'POLLUTED'}")
    return ok, " ".join(details)


# ===== executor_base (inlined) =====

import ctypes
import os
import random
import re
import sys
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface import acl_wrapper as _acl_wrapper
from atk.tasks.backends.lib_interface.acl_wrapper import (
    AclDataType,
    AclFormat,
    AclTensor,
    AclTensorStruct,
    NNOPBASE_PATH,
    TORCH_TO_ACLTYPE,
    nnopbase,
)

# ATK 内置 DATATYPE_REGISTRY / TorchDtype 没有 hifp8。create_dataset 随 -p 加载
# 本文件时必须完成注册，否则 KeyError。存储用 uint8；wq=5 时再改写成 HiFloat8 码点。
try:
    from atk.case_generator.generator.data_types.data_torch import BaseDataTypeTensor
    from atk.case_generator.utils.enums import NpDtype, TorchDtype
    from atk.case_generator.utils.plugins_register import DATATYPE_REGISTRY

    _HIFP8_ALIASES = ("hifp8", "hif8", "hifloat8")

    def _is_hifp8_key(key) -> bool:
        if key is None:
            return False
        name = (
            str(getattr(key, "name", key))
            .lower()
            .replace("torch.", "")
            .replace("acl_", "")
        )
        return name in _HIFP8_ALIASES

    def _patch_atk_hifp8_dtype():
        if getattr(TorchDtype, "_hifp8_patched", False):
            return
        orig_torch = TorchDtype.get

        @classmethod
        def _torch_get(cls, key):
            if _is_hifp8_key(key):
                return torch.uint8
            return orig_torch(key)

        TorchDtype.get = _torch_get
        TorchDtype._hifp8_patched = True
        orig_np = NpDtype.get

        @classmethod
        def _np_get(cls, key):
            if _is_hifp8_key(key):
                return np.uint8
            return orig_np(key)

        NpDtype.get = _np_get
        NpDtype._hifp8_patched = True

    def _as_uint8_tensor(shape, fill=None):
        shape = tuple(int(x) for x in shape)
        if fill is not None:
            try:
                val = int(float(fill)) % 256
            except (TypeError, ValueError):
                val = 0
            return torch.full(shape, val, dtype=torch.uint8)
        return torch.randint(0, 256, shape, dtype=torch.uint8)

    _patch_atk_hifp8_dtype()

    @DATATYPE_REGISTRY.register("hifp8")
    @DATATYPE_REGISTRY.register("hif8")
    @DATATYPE_REGISTRY.register("hifloat8")
    class DatasetHifp8(BaseDataTypeTensor):
        """HiFloat8 ATK 造数：uint8 存储，值域占位；真实码点由 gen_inputs 重写。"""

        def _source_tensor(self, fill_value=None):
            rv = getattr(self.config, "range_values", None)
            if fill_value is None and (rv == 0 or rv == [0]):
                return torch.zeros(tuple(self.config.shape), dtype=torch.uint8)
            return _as_uint8_tensor(self.config.shape, fill_value)

        def gen_nonbound_tensor_data(self):
            return self._source_tensor()

        def gen_boundary_tensor_data(self):
            rv = self.config.range_values
            if isinstance(rv, list) and rv:
                rv = rv[0]
            if isinstance(rv, str):
                low = rv.lower()
                if low in ("null", "default"):
                    return torch.tensor([], dtype=torch.uint8)
                if low in ("true", "false", "nan", "inf", "-inf"):
                    return torch.zeros(tuple(self.config.shape), dtype=torch.uint8)
            return self._source_tensor(fill_value=rv)

        def gen_nonbound_scalar_data(self):
            return 0

        def gen_data(self):
            if getattr(self.config, "is_boundary", False):
                return self.gen_boundary_tensor_data()
            return self.gen_nonbound_tensor_data()
except Exception as _hifp8_reg_err:  # pragma: no cover
    pass

# ATK 26.7.8 的 TORCH_TO_ACLTYPE 没有 e8m0（key 是字符串 'torch.xxx'）；
# pyaclnn 转 acl 走这张表。运行期补上，mxfp8 scale 才能建出 ACL_FLOAT8_E8M0。
# 宿主机 pyACL（CANN 9.1.0-beta3）无 ACL_FLOAT8_E8M0 时跳过，避免 import 失败。
if hasattr(AclDataType, "ACL_FLOAT8_E8M0"):
    _acl_wrapper.TORCH_TO_ACLTYPE["torch.float8_e8m0fnu"] = int(
        AclDataType.ACL_FLOAT8_E8M0
    )
# 部分 ATK 版本缺 e4m3；DUT 强制分配 queryOut=fp8 时需要此映射。
if hasattr(AclDataType, "ACL_FLOAT8_E4M3"):
    _acl_wrapper.TORCH_TO_ACLTYPE.setdefault(
        "torch.float8_e4m3fn", int(AclDataType.ACL_FLOAT8_E4M3)
    )
if hasattr(_acl_wrapper, "ACLTYPE_TO_CTYPE"):
    if hasattr(AclDataType, "ACL_FLOAT8_E8M0"):
        _acl_wrapper.ACLTYPE_TO_CTYPE.setdefault(
            int(AclDataType.ACL_FLOAT8_E8M0), ctypes.c_ubyte
        )
    if hasattr(AclDataType, "ACL_FLOAT8_E4M3"):
        _acl_wrapper.ACLTYPE_TO_CTYPE.setdefault(
            int(AclDataType.ACL_FLOAT8_E4M3), ctypes.c_ubyte
        )
    if hasattr(AclDataType, "ACL_HIFLOAT8"):
        _acl_wrapper.ACLTYPE_TO_CTYPE.setdefault(
            int(AclDataType.ACL_HIFLOAT8), ctypes.c_ubyte
        )


logging = Logger().get_logger()
CACHE_INDEX_SEED = 2025
_SEED_NAME_RE = re.compile(r"\.seed(\d+)\.")


def parse_case_seed(case_name: str, default: int = CACHE_INDEX_SEED) -> int:
    """从用例名解析 seed<N>；泛化矩阵里同 shape 的 nc* 共用同一 seed。"""
    match = _SEED_NAME_RE.search(case_name or "")
    return int(match.group(1)) if match else default


def _task_case_name(task_result: TaskResult) -> str:
    case_config = getattr(task_result, "case_config", None)
    return str(getattr(case_config, "name", "") or "")


# Positional layout of the aclnn prototypes. Every tensor slot has to be passed, so the
# executor fills the slots a case does not declare with a null aclTensor pointer.
V1_TENSOR_SLOTS = (
    "token_x",
    "weight_dq",
    "weight_uq_qr",
    "weight_uk",
    "weight_dkv_kr",
    "rmsnorm_gamma_cq",
    "rmsnorm_gamma_ckv",
    "rope_sin",
    "rope_cos",
    "cache_index",
    "kv_cache",
    "kr_cache",
    "dequant_scale_x",
    "dequant_scale_w_dq",
    "dequant_scale_w_uq_qr",
    "dequant_scale_w_dkv_kr",
    "quant_scale_ckv",
    "quant_scale_ckr",
    "smooth_scales_cq",
)
V3_TENSOR_SLOTS = (
    "token_x",
    "weight_dq",
    "weight_uq_qr",
    "weight_uk",
    "weight_dkv_kr",
    "rmsnorm_gamma_cq",
    "rmsnorm_gamma_ckv",
    "rope_sin",
    "rope_cos",
    "kv_cache",
    "kr_cache",
    "cache_index",
    "dequant_scale_x",
    "dequant_scale_w_dq",
    "dequant_scale_w_uq_qr",
    "dequant_scale_w_dkv_kr",
    "quant_scale_ckv",
    "quant_scale_ckr",
    "smooth_scales_cq",
    "actual_seq_len",
    "k_nope_clip_alpha",
)

V1_ATTR_SLOTS = ("rmsnormEpsilonCq", "rmsnormEpsilonCkv", "cacheMode")
V3_ATTR_SLOTS = V1_ATTR_SLOTS + (
    "weightQuantMode",
    "kvCacheQuantMode",
    "queryQuantMode",
    "ckvkrRepoMode",
    "quantScaleRepoMode",
    "tileSize",
    "qcQrScale",
    "kcScale",
)
V4_ATTR_SLOTS = V3_ATTR_SLOTS + ("doRope",)

# queryOut / queryRopeOut are produced by the reference node; the remaining output slots
# (dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut) stay null in these cases.
VERSION_LAYOUT = {
    "v1": (V1_TENSOR_SLOTS, V1_ATTR_SLOTS, 0, ()),
    "v2": (
        V1_TENSOR_SLOTS,
        V1_ATTR_SLOTS,
        1,
        ("weight_dq", "weight_uq_qr", "weight_dkv_kr"),
    ),
    "v3": (
        V3_TENSOR_SLOTS,
        V3_ATTR_SLOTS,
        3,
        ("weight_dq", "weight_uq_qr", "weight_dkv_kr"),
    ),
    "v4": (
        V3_TENSOR_SLOTS,
        V4_ATTR_SLOTS,
        3,
        ("weight_dq", "weight_uq_qr", "weight_dkv_kr"),
    ),
}


def _null_tensor_ptr():
    return ctypes.POINTER(AclTensor)()


# CaseGenerator 用 dtype=bool + shape=[] 标记“不传”的可选槽；ATK 仍可能物化成
# numel=1 的 bool 张量。执行前必须丢掉，否则 golden/aclnn 会当成有效入参。
# rope_sin/rope_cos 也纳入：doRope=false 时生成器将其置为 null 标记，须 drop 为 null
# （op_api 侧 doRope=false 要求 rope 必须同时为空）。
OPTIONAL_TENSOR_KEYS = (
    "rope_sin",
    "rope_cos",
    "dequant_scale_x",
    "dequant_scale_w_dq",
    "dequant_scale_w_uq_qr",
    "dequant_scale_w_dkv_kr",
    "quant_scale_ckv",
    "quant_scale_ckr",
    "smooth_scales_cq",
    "actual_seq_len",
    "cache_index",
    "k_nope_clip_alpha",
)


def _is_atk_null_tensor(value) -> bool:
    if value is None:
        return True
    if not isinstance(value, torch.Tensor):
        return False
    return value.dtype == torch.bool


def _sanitize_optional_kwargs(kw: dict) -> None:
    dropped = [
        k for k in OPTIONAL_TENSOR_KEYS if k in kw and _is_atk_null_tensor(kw[k])
    ]
    for key in dropped:
        kw.pop(key, None)
    if dropped:
        logging.info("[mla_prolog] drop ATK null optional tensors: %s", dropped)


def _resolve_do_rope(kw: Dict[str, Any]) -> bool:
    """V4 doRope（相对 V3 新增）：控制 queryRopeOut 与 krCache 是否做旋转位置编码。

    默认 true。false 时 CPU 真值对 Qr / Kr 直通，且 ropeSin/ropeCos 必须同时为空
    （nullptr 或空 Tensor）；DUT 侧将 rope 槽替换为 null。
    """
    if "doRope" not in kw:
        return True
    value = kw["doRope"]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"doRope tensor must be scalar, got shape={tuple(value.shape)}"
            )
        return bool(value.item())
    return bool(value)


def _rewrite_rope_tables(
    input_data: InputDataset, seed: int = CACHE_INDEX_SEED
) -> None:
    """ATK yaml range=[-1,1] rope tables are not valid cos/sin; rewrite when doRope=true.

    doRope=false 时保持 rope 槽为空，使 DUT 走到 ENABLE_ROPE=0。
    """
    kw = input_data.kwargs
    # doRope=false 必须保持 ropeSin/ropeCos 为空，才能走到 ENABLE_ROPE=0 的 tilingKey
    if not _resolve_do_rope(kw):
        return
    token = kw.get("token_x")
    if not isinstance(token, torch.Tensor):
        return
    t = _token_t(token)
    w_uk = kw.get("weight_uk")
    w_uq = kw.get("weight_uq_qr")
    if not isinstance(w_uk, torch.Tensor) or not isinstance(w_uq, torch.Tensor):
        return
    n_heads = int(w_uk.shape[0])
    d_nope = int(w_uk.shape[1])
    dr = int(w_uq.shape[-1]) // max(n_heads, 1) - d_nope
    cos, sin = gen_rope_cos_sin(t, dr, seed)
    if token.dim() == 3:
        b, s = int(token.shape[0]), int(token.shape[1])
        cos, sin = cos.view(b, s, dr), sin.view(b, s, dr)
    kw["rope_cos"] = cos.to(dtype=torch.bfloat16)
    kw["rope_sin"] = sin.to(dtype=torch.bfloat16)
    logging.info(
        "[mla_prolog][rope] rewrote cos/sin tables t=%s dr=%s seed=%s shape=%s doRope=%s",
        t,
        dr,
        seed,
        tuple(sin.shape),
        _resolve_do_rope(kw),
    )


def _apply_deterministic_cache_index(
    input_data: InputDataset, seed: int = CACHE_INDEX_SEED
):
    """按 cacheMode 写合法 cacheIndex / actualSeqLen。

    PA_BSND/PA_NZ：全局 slot，token 互不冲突。
    PA_BLK_*：page-id ∈ [0, BlockNum)，actualSeqLen 为前缀和且末值=T。
    BSND/TND：cacheIndex / actualSeqLen 必须为 nullptr。
    """
    kw = input_data.kwargs
    mode = str(kw.get("cacheMode", "PA_BSND") or "PA_BSND").upper()
    token_num = _token_t(kw["token_x"])

    if mode in ("BSND", "TND"):
        dropped = [k for k in ("cache_index", "actual_seq_len") if k in kw]
        for key in dropped:
            kw.pop(key, None)
        if dropped:
            logging.info(
                "[mla_prolog][cache] %s -> drop %s (must be null)", mode, dropped
            )
        return

    kv = kw["kv_cache"]
    block_num = int(kv.shape[0])
    block_size = int(kv.shape[1]) if kv.dim() >= 2 else 0
    rng = np.random.RandomState(seed)

    if mode in ("PA_BLK_BSND", "PA_BLK_NZ"):
        token = kw["token_x"]
        # token 3D：actualSeqLen 关闭（ACTUAL_SEQ_LEN_MODE=0），cacheIndex 为 [B, ceil(S/BlockSize)]
        if isinstance(token, torch.Tensor) and token.dim() == 3:
            b, s = int(token.shape[0]), int(token.shape[1])
            kw.pop("actual_seq_len", None)
            pages = ((s + block_size - 1) // block_size) if block_size > 0 else 1
            pages = max(pages, 1)
            if block_num <= 0:
                index = torch.zeros((b, pages), dtype=torch.int64)
            else:
                index = torch.from_numpy(
                    rng.choice(block_num, size=(b, pages), replace=True).astype(
                        np.int64
                    )
                )
            kw["cache_index"] = index
            logging.info(
                "[mla_prolog][cache] %s 3D B=%s S=%s pages=%s block_num=%s seed=%s",
                mode,
                b,
                s,
                pages,
                block_num,
                seed,
            )
            return
        actual = kw.get("actual_seq_len")
        batch = (
            int(actual.numel())
            if isinstance(actual, torch.Tensor) and actual.numel() > 0
            else 1
        )
        if token_num % batch != 0:
            batch = 1
        seq = token_num // batch
        prefix = torch.tensor([seq * (i + 1) for i in range(batch)], dtype=torch.int32)
        kw["actual_seq_len"] = prefix
        pages = (
            batch * ((seq + block_size - 1) // block_size) if block_size > 0 else batch
        )
        pages = max(pages, 1)
        replace = pages > block_num
        index = torch.from_numpy(
            rng.choice(max(block_num, 1), pages, replace=replace).astype(np.int64)
        )
        kw["cache_index"] = index
        logging.info(
            "[mla_prolog][cache] %s pages=%s block_num=%s prefix=%s seed=%s",
            mode,
            pages,
            block_num,
            prefix.tolist(),
            seed,
        )
        return

    slots = block_num * block_size
    if slots == 0:
        # EMPTY_CACHE：没有可写 slot，index 仅占位，kernel 不更新 cache
        kw["cache_index"] = torch.zeros(token_num, dtype=torch.int64)
        logging.info("[mla_prolog][cache] %s empty cache T=%s", mode, token_num)
        return
    if token_num > slots:
        raise ValueError(
            f"cacheMode={mode}: T={token_num} exceeds BlockNum*BlockSize={slots}"
        )
    index = torch.from_numpy(
        rng.choice(slots, token_num, replace=False).astype(np.int64)
    )
    existing = kw.get("cache_index")
    if isinstance(existing, torch.Tensor) and existing.dtype != torch.bool:
        index = index.to(existing.dtype)
    kw["cache_index"] = index
    logging.info(
        "[mla_prolog][cache] %s T=%s slots=%s seed=%s",
        mode,
        token_num,
        slots,
        seed,
    )


def _infer_quant_modes(kw: Dict[str, Any]) -> Tuple[int, int]:
    """V1/V2 have no quant-mode attributes, so derive the mode from the input dtypes.
    The same derivation matches the attributes V4 carries."""
    token_x, w_uq = kw["token_x"], kw["weight_uq_qr"]
    if token_x.dtype == torch.int8:
        wq = 2
    elif w_uq.dtype == torch.int8:
        wq = 1
    elif token_x.dtype == torch.uint8:
        wq = 5
    elif w_uq.dtype == torch.float8_e4m3fn:
        # wq=4: FLOAT per-token/per-channel scales; wq=3: E8M0 block scales
        scale_x = kw.get("dequant_scale_x")
        if (
            isinstance(scale_x, torch.Tensor)
            and scale_x.dtype == torch.float32
            and scale_x.dim() == 2
            and int(scale_x.shape[-1]) == 1
        ):
            wq = 4
        else:
            wq = 3
    else:
        wq = 0

    kv_dtype = kw["kv_cache"].dtype
    hckv = int(kw["weight_uk"].shape[2])
    h_last = int(kw["kv_cache"].shape[-1])
    if h_last > hckv:
        kvq = 3
    elif kv_dtype == torch.int8:
        kvq = 2
    elif kv_dtype == torch.float8_e4m3fn:
        kvq = 1
    else:
        kvq = 0
    return wq, kvq


def _build_case_config(input_data: InputDataset) -> CaseConfig:
    kw = input_data.kwargs
    tx = kw["token_x"]
    kv = kw["kv_cache"]
    mode = str(kw.get("cacheMode", "PA_BSND") or "PA_BSND").upper()
    hckv = int(kw["weight_uk"].shape[2])
    wq_i, kvq_i = _infer_quant_modes(kw)
    wq_attr = int(kw.get("weightQuantMode", -1))
    kvq_attr = int(kw.get("kvCacheQuantMode", -1))
    wq = wq_attr if wq_attr >= 0 else wq_i
    kvq = kvq_attr if kvq_attr >= 0 else kvq_i

    if tx.dim() == 3:
        batch, seq, he = int(tx.shape[0]), int(tx.shape[1]), int(tx.shape[2])
        t = batch * seq
    else:
        t, he = int(tx.shape[0]), int(tx.shape[1])
        batch, seq = 1, t
        actual = kw.get("actual_seq_len")
        if (
            isinstance(actual, torch.Tensor)
            and actual.dtype != torch.bool
            and actual.numel() > 0
            and t % int(actual.numel()) == 0
        ):
            batch = int(actual.numel())
            seq = t // batch

    if kv.dim() == 3:
        nkv = int(kv.shape[1])
        block_num, block_size = 0, 0
    elif mode == "BSND":
        nkv = int(kv.shape[2])
        block_num, block_size = 0, 0
    else:
        nkv = int(kv.shape[2])
        block_num, block_size = int(kv.shape[0]), int(kv.shape[1])

    return CaseConfig(
        t=t,
        he=he,
        hcq=int(kw["weight_dq"].shape[1]),
        hckv=hckv,
        d=int(kw["weight_uk"].shape[1]),
        dr=int(kw["weight_dkv_kr"].shape[1]) - hckv,
        n=int(kw["weight_uk"].shape[0]),
        nkv=nkv,
        block_num=block_num,
        block_size=block_size,
        eps_cq=float(kw.get("rmsnormEpsilonCq", 1e-5)),
        eps_ckv=float(kw.get("rmsnormEpsilonCkv", 1e-5)),
        qc_qr_scale=float(kw.get("qcQrScale", 1.0)),
        kc_scale=float(kw.get("kcScale", 1.0)),
        do_rope=_resolve_do_rope(kw),
        seed=CACHE_INDEX_SEED,
        weight_quant_mode=wq,
        kv_cache_quant_mode=kvq,
        query_quant_mode=int(kw.get("queryQuantMode", 0)),
        ckvkr_repo_mode=int(kw.get("ckvkrRepoMode", 0)),
        quant_scale_repo_mode=int(kw.get("quantScaleRepoMode", 0)),
        tile_size=int(kw.get("tileSize", 128)),
        query_norm_flag=False,
        rope_style=str(kw.get("ropeStyle", kw.get("rope_style", "original"))),
        batch=batch,
        seq=seq,
        cache_mode=mode,
        token_rank=int(tx.dim()),
    )


def _unfused_query_shape(kw: Dict[str, Any], shape) -> tuple:
    """BS 非合轴时 ATK 按 (T, N, *) 申报 query，文档要求 (B, S, N, *)。"""
    shape = tuple(int(x) for x in shape)
    token = kw.get("token_x")
    if not isinstance(token, torch.Tensor) or token.dim() != 3 or len(shape) != 3:
        return shape
    b, s = int(token.shape[0]), int(token.shape[1])
    if shape[0] != b * s:
        return shape
    return (b, s, shape[1], shape[2])


def _query_out_runtime_dtype(kw: Dict[str, Any]) -> torch.dtype:
    """Dtype the operator actually writes into queryOut."""
    qq = int(kw.get("queryQuantMode", 0))
    wq = int(kw.get("weightQuantMode", -1))
    if qq == 1:
        if wq == 2:
            return torch.int8
        if wq == 5:
            return torch.uint8  # HiFloat8 存储；比对侧升 fp32
        return torch.float8_e4m3fn
    if kw["token_x"].dtype == torch.int8:
        return torch.bfloat16
    if kw["token_x"].dtype in (torch.bfloat16, torch.float16):
        return kw["token_x"].dtype
    return torch.bfloat16


def _cast_outputs(
    outs: Sequence[torch.Tensor], kw: Dict[str, Any]
) -> List[torch.Tensor]:
    """Cast golden outs to ATK **compare** dtypes.

    ATK 的 cv_fused_double_benchmark 在比对前会调用 torch.isinf，对 Float8_e4m3fn
    直接 NotImplementedError。因此参考节点对 FP8 输出改以 float32 参与比对；
    DUT 侧仍按算子真实 dtype 分配 FP8 缓冲，并在 after_call 再 cast 到 float32。
    （ATK 无 float8_e4m3fn 专用标准；hifp8_ulp 仅覆盖 HiFloat8，不能套用。）
    """
    query_dtype = _query_out_runtime_dtype(kw)
    qq = int(kw.get("queryQuantMode", 0))
    wq = int(kw.get("weightQuantMode", -1))
    if qq == 1 and wq in (2, 3, 4, 5):
        query_dtype = torch.float32
    elif query_dtype in (torch.float8_e4m3fn, torch.uint8):
        query_dtype = torch.float32  # compare dtype
    rope_dtype = torch.bfloat16
    dtypes = [query_dtype, rope_dtype, kw["kv_cache"].dtype, kw["kr_cache"].dtype]
    # kv/kr 若为 fp8，同样升到 fp32 再比对（当前参考节点只返回 query/query_rope，预留一致性）
    dtypes = [torch.float32 if dt == torch.float8_e4m3fn else dt for dt in dtypes]
    # contiguous clone：避免 view/storage 不一致时 torch.save 报 unexpected pos
    return [t.to(dtype).contiguous().clone() for t, dtype in zip(outs, dtypes)]


def _to_compare_dtype(t: torch.Tensor) -> torch.Tensor:
    """ATK isinf/双标杆比对前，把 FP8 升到 float32。"""
    if t.dtype in (
        torch.float8_e4m3fn,
        getattr(torch, "float8_e5m2", torch.float8_e4m3fn),
    ):
        return t.to(torch.float32)
    return t


def _move_to_npu(input_data: InputDataset, nz_names: Sequence[str] = ()):
    """Only the operator under test consumes FRACTAL_NZ weights; the torch_npu reference
    chain needs plain ND tensors it can matmul and copy back."""
    kw = input_data.kwargs
    for name, tensor in kw.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.device.type != "npu":
            tensor = tensor.to(device=f"npu:{torch.npu.current_device()}")
        if name in nz_names:
            tensor = cast_weight_nz(tensor, tensor.device)
        kw[name] = tensor


def _is_mxfp8_case(kw: Dict[str, Any]) -> bool:
    return int(kw.get("weightQuantMode", -1)) == 3


def _rewrite_quant_gen_inputs(
    input_data: InputDataset, seed: int = CACHE_INDEX_SEED
) -> bool:
    """ATK 随机量化张量/scale 与 kernel 编码不一致：用 gen_inputs 按 wq 整包替换。

    覆盖 wq=3 (mxfp8 e8m0)、wq=4 (fp8 full FLOAT scale)、wq=5 (HiFloat8)。
    参考节点和 pyaclnn 节点都走这里，且共用同一 seed，两边输入一致。
    """
    kw = input_data.kwargs
    wq = int(kw.get("weightQuantMode", -1))
    if wq not in (3, 4, 5):
        return False

    cfg = _build_case_config(input_data)
    cfg.weight_quant_mode = wq
    cfg.kv_cache_quant_mode = int(kw.get("kvCacheQuantMode", cfg.kv_cache_quant_mode))
    cfg.query_quant_mode = int(kw.get("queryQuantMode", cfg.query_quant_mode))
    cfg.seed = seed

    generated = gen_inputs(cfg, device="cpu")
    # 只写回 case 中真实声明为输入的可选键；doRope=false 时 rope_sin/rope_cos 等可选槽
    # 已被 _sanitize_optional_kwargs drop 为 null，若重新插入会改变 kwargs 键顺序，
    # 导致 _expand_to_prototype 的 "case tensors must be declared in aclnn prototype order" 校验失败。
    for name, value in generated.items():
        if name not in kw and name in OPTIONAL_TENSOR_KEYS:
            continue
        kw[name] = value
    tag = {3: "mxfp8", 4: "fp8full", 5: "hif8"}[wq]
    extra = ""
    if wq == 3:
        extra = f" scale_dtype={kw['dequant_scale_x'].dtype}"
    logging.info(
        "[mla_prolog][%s] rewrote inputs via gen_inputs wq=%s kvq=%s qq=%s seed=%s%s",
        tag,
        cfg.weight_quant_mode,
        cfg.kv_cache_quant_mode,
        cfg.query_quant_mode,
        seed,
        extra,
    )
    return True


def _load_nnopbase() -> ctypes.CDLL:
    try:
        return ctypes.CDLL(str(NNOPBASE_PATH), mode=ctypes.RTLD_GLOBAL)
    except Exception:  # noqa: BLE001
        home = os.environ.get("ASCEND_HOME_PATH", "")
        for rel in ("lib64", "x86_64-linux/lib64", "aarch64-linux/lib64"):
            path = os.path.join(home, rel, "libnnopbase.so")
            if os.path.isfile(path):
                return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        return ctypes.CDLL("libnnopbase.so", mode=ctypes.RTLD_GLOBAL)


def _acl_create_tensor(tensor: torch.Tensor, acl_dtype: int, acl_format: int = 2):
    """自建 aclTensor；ATK 的 TORCH_TO_ACLTYPE 没有 float8_e8m0fnu。"""
    lib = _load_nnopbase()
    lib.aclCreateTensor.restype = ctypes.POINTER(AclTensor)
    lib.aclCreateTensor.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    shape = tuple(int(x) for x in tensor.shape)
    stride = tuple(int(x) for x in tensor.stride())
    shape_arr = (ctypes.c_int64 * len(shape))(*shape)
    stride_arr = (ctypes.c_int64 * len(stride))(*stride)
    ptr = lib.aclCreateTensor(
        shape_arr,
        len(shape),
        int(acl_dtype),
        stride_arr,
        int(tensor.storage_offset()),
        int(acl_format),
        shape_arr,
        len(shape),
        ctypes.c_void_p(tensor.untyped_storage().data_ptr()),
    )
    if not ptr:
        raise RuntimeError(
            f"aclCreateTensor failed: shape={shape} stride={stride} acl_dtype={acl_dtype}"
        )
    return ptr


class FunctionMlaPrologBaseApi(BaseApi):
    """CPU / torch_npu reference node."""

    op_version: str = "v1"

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.set_seeds(parse_case_seed(_task_case_name(task_result)))

    def set_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.manual_seed(seed)
            torch.npu.manual_seed_all(seed)

    def init_by_input_data(self, input_data: InputDataset):
        seed = parse_case_seed(_task_case_name(self.task_result))
        self.set_seeds(seed)
        _sanitize_optional_kwargs(input_data.kwargs)
        _rewrite_quant_gen_inputs(input_data, seed=seed)
        _apply_deterministic_cache_index(input_data, seed=seed)
        _rewrite_rope_tables(input_data, seed=seed)
        self.enable_rope = _resolve_do_rope(input_data.kwargs)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """Return ATK dual-benchmark refs: CPU golden (-bd cpu) or NPU small-op (npu node)."""
        kw = input_data.kwargs
        cfg = _build_case_config(input_data)
        cpu_inp = {
            k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
            for k, v in kw.items()
        }

        if self.device == "npu":
            token = kw.get("token_x")
            device = (
                str(token.device)
                if isinstance(token, torch.Tensor) and token.device.type == "npu"
                else "npu:0"
            )
            # 从小算子图的 host 副本出发，避免 NPU scale 与 CPU 量化混设备
            npu_outs = cal_mlaprolog_npu(cpu_inp, cfg, device=device)
            npu_outs = _cast_outputs([x.cpu() for x in npu_outs], cpu_inp)
            return npu_outs[0].npu(), npu_outs[1].npu()

        cpu_outs = _cast_outputs(cal_mlaprolog_cpu(cpu_inp, cfg), cpu_inp)
        return cpu_outs[0], cpu_outs[1]


class AclnnMlaPrologBaseApi(AclnnBaseApi):
    """pyaclnn node: the operator under test."""

    op_version: str = "v1"

    def _layout(self):
        return VERSION_LAYOUT[self.op_version]

    def get_format(self, input_data: InputDataset, index=None, name=None):
        if name in self._layout()[3]:
            return AclFormat.ACL_FORMAT_FRACTAL_NZ
        return AclFormat.ACL_FORMAT_ND

    def _case_name(self) -> str:
        case_config = getattr(self.task_result, "case_config", None)
        return str(getattr(case_config, "name", "") or "")

    def torch_tensor_to_acl(self, tensor, fmt=AclFormat.ACL_FORMAT_ND):
        """ATK 不认识 float8_e8m0fnu / HiFloat8；这两类走自建 aclCreateTensor。"""
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float8_e8m0fnu:
            held = tensor.contiguous()
            self._acl_holders.append(held)
            ptr = _acl_create_tensor(held, int(AclDataType.ACL_FLOAT8_E8M0), int(fmt))
            logging.info(
                "[mla_prolog][mxfp8] acl e8m0 tensor shape=%s", tuple(held.shape)
            )
            return AclTensorStruct(
                tensor=ptr,
                addr=int(held.untyped_storage().data_ptr()),
                pytensor=held,
                data_size=int(held.numel()),
            )
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.dtype == torch.uint8
            and int(getattr(self, "weight_quant_mode", -1)) == 5
            and hasattr(AclDataType, "ACL_HIFLOAT8")
        ):
            held = tensor.contiguous()
            self._acl_holders.append(held)
            ptr = _acl_create_tensor(held, int(AclDataType.ACL_HIFLOAT8), int(fmt))
            logging.info(
                "[mla_prolog][hif8] acl HIFLOAT8 tensor shape=%s fmt=%s",
                tuple(held.shape),
                int(fmt),
            )
            return AclTensorStruct(
                tensor=ptr,
                addr=int(held.untyped_storage().data_ptr()),
                pytensor=held,
                data_size=int(held.numel()),
            )
        return super().torch_tensor_to_acl(tensor, fmt)

    def _apply_noncontig_cache(self, input_data: InputDataset):
        """把 kv/kr cache 换成只有 dim0 非连续的视图。

        参考节点仍然拿到逻辑值，所以 golden 不受布局影响；这里改的只是被测算子
        看到的物理布局。ATK 建 aclTensor 时会带上 torch tensor 的真实 stride，
        算子 Tiling 侧再用 GetInputStride 取到 stride(0)。
        """
        kw = input_data.kwargs
        self.nc_caches: Dict[str, Optional[NonContigCache]] = {
            "kv_cache": None,
            "kr_cache": None,
        }
        if self.nc_factor <= 1:
            return
        for name in ("kv_cache", "kr_cache"):
            tensor = kw.get(name)
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                continue
            cache = make_dim0_noncontig(tensor, self.nc_factor)
            self.nc_caches[name] = cache
            kw[name] = cache.view
        logging.info("[mla_prolog][nc] %s", describe(self.nc_caches))

    def init_by_input_data(self, input_data: InputDataset):
        self._acl_holders: List[torch.Tensor] = []
        seed = parse_case_seed(self._case_name())
        kw = input_data.kwargs
        _sanitize_optional_kwargs(kw)
        _rewrite_quant_gen_inputs(input_data, seed=seed)
        # gen_inputs 塞进 kwargs 的 _dequant_scale_*_fp 只给 golden 用，不能进 acl 原型
        for key in [k for k in list(kw) if k.startswith("_")]:
            kw.pop(key)

        _apply_deterministic_cache_index(input_data, seed=seed)
        self.enable_rope = _resolve_do_rope(kw)
        self.weight_quant_mode = int(kw.get("weightQuantMode", -1))
        tensor_slots, attr_slots, _, nz_names = self._layout()
        # inner infershape 缺省 doRope=true；doRope=false 的 bool 空槽会被 ATK 转成空
        # aclTensor。始终生成合法 (B,S,Dr)/(T,Dr) 表，super() 之后再写回 ACL 槽。
        _rewrite_rope_tables(input_data, seed=seed)
        rope_hold = {}
        for name in ("rope_sin", "rope_cos"):
            tensor = kw.get(name)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype == torch.bool:
                continue
            # doRope=true 时 B/S/T=0 的 rope 仍是合法空 tensor，不能收成 nullptr，
            # 否则 aclnn 报 ropeSin required。
            if tensor.numel() > 0 or self.enable_rope:
                rope_hold[name] = tensor.contiguous()
                if not self.enable_rope:
                    kw.pop(name, None)
        ordered = {}
        for name in tensor_slots:
            if name in kw and isinstance(kw[name], torch.Tensor):
                ordered[name] = kw[name]
        for key, value in kw.items():
            if key not in ordered:
                ordered[key] = value
        kw.clear()
        kw.update(ordered)
        self.nc_factor = parse_stride_factor(self._case_name())
        self.query_quant_mode = int(kw.get("queryQuantMode", 0))

        declared = [n for n, v in kw.items() if isinstance(v, torch.Tensor)]
        _move_to_npu(input_data, nz_names)
        self._apply_noncontig_cache(input_data)
        for name, held in list(rope_hold.items()):
            if held.device.type != "npu":
                rope_hold[name] = held.npu().contiguous()
        # kvCacheRef / krCacheRef are updated in place, so keep the device tensors around to
        # read the operator result back after the call. 非连续用例读的始终是逻辑视图。
        self.cache_refs = tuple(
            self.nc_caches[name].view
            if self.nc_caches.get(name)
            else input_data.kwargs[name]
            for name in ("kv_cache", "kr_cache")
        )

        if self.query_quant_mode == 1:
            t = _token_t(kw["token_x"])
            n = int(kw["weight_uk"].shape[0])
            self.dequant_q_nope = torch.zeros(
                (t, n, 1), dtype=torch.float32, device=kw["token_x"].device
            )
        else:
            self.dequant_q_nope = None

        # 对齐 mlapo：输出缓冲用 zeros 而不是 empty，避免未写满时读到脏数据。
        input_args, output_packages = self._init_with_zero_outputs(input_data)
        # optional 输出槽位自己拼：qq=1 时 dequantScaleQNopeOut 必须非空
        input_args = self._expand_to_prototype(
            list(input_args),
            declared,
            tensor_slots,
            attr_slots,
            len(output_packages),
            0,
        )
        for name, held in rope_hold.items():
            self._acl_holders.append(held)
            ptr = _acl_create_tensor(
                held, int(AclDataType.ACL_BF16), int(AclFormat.ACL_FORMAT_ND)
            )
            slot = tensor_slots.index(name)
            if slot < len(input_args):
                input_args[slot] = AclTensorStruct(
                    tensor=ptr,
                    addr=int(held.untyped_storage().data_ptr()),
                    pytensor=held,
                    data_size=int(held.numel()),
                )
            logging.info(
                "[mla_prolog][rope] DUT overwrite %s shape=%s numel=%s slot=%s",
                name,
                tuple(held.shape),
                int(held.numel()),
                slot,
            )
        if self.dequant_q_nope is not None:
            dq_struct = self.torch_tensor_to_acl(
                self.dequant_q_nope, AclFormat.ACL_FORMAT_ND
            )
            input_args.append(dq_struct.tensor)
            self._acl_holders.append(self.dequant_q_nope)
        else:
            input_args.append(_null_tensor_ptr())
        input_args.append(_null_tensor_ptr())  # queryNormOutOptional
        input_args.append(_null_tensor_ptr())  # dequantScaleQNormOutOptional
        if hasattr(self, "backend") and self.backend is not None:
            self.backend.input_args = input_args
        return input_args, output_packages

    @staticmethod
    def _expand_to_prototype(
        input_args, declared, tensor_slots, attr_slots, n_outputs, null_outputs
    ):
        """ATK converts the declared inputs positionally; splice in a null aclTensor for every
        optional slot the case leaves out so the argument list matches the aclnn prototype."""
        unknown = [n for n in declared if n not in tensor_slots]
        if unknown:
            raise ValueError(
                f"case declares tensors that are not in the prototype: {unknown}"
            )
        if list(declared) != [n for n in tensor_slots if n in declared]:
            raise ValueError("case tensors must be declared in aclnn prototype order")

        n_declared = len(declared)
        tail = input_args[n_declared:]
        if len(tail) != len(attr_slots) + n_outputs:
            raise ValueError(
                f"expected {len(attr_slots)} attrs + {n_outputs} outputs, got {len(tail)}"
            )

        it = iter(input_args[:n_declared])
        remaining = list(declared)
        expanded = []
        for slot in tensor_slots:
            if remaining and remaining[0] == slot:
                expanded.append(next(it))
                remaining.pop(0)
            else:
                expanded.append(_null_tensor_ptr())
        expanded.extend(tail)
        expanded.extend(_null_tensor_ptr() for _ in range(null_outputs))
        return expanded

    def _init_with_zero_outputs(self, input_data: InputDataset):
        """参考 mlapo_atk/exec.py：把 ATK 的 output 分配改成 torch.zeros。

        参考节点为绕开 ATK FP8 isinf 会以 float32 申报 query；这里按算子真实 dtype
        分配 queryOut：qq=1 时 wq=2→int8 / wq=3,4→fp8 / wq=5→HIFLOAT8。
        """
        backend = self.backend
        if not hasattr(backend, "convert_output_data"):
            return super().init_by_input_data(input_data)

        query_runtime_dtype = _query_out_runtime_dtype(input_data.kwargs)
        # ATK 对每个 output 回调时 index 从 0 递增；output_packages 顺序为 query, query_rope
        out_slot = {"i": 0}

        def my_convert_output_data(backend_instance, output_data, index):
            if isinstance(output_data, (list, tuple)):
                data_list = []
                for tmp in output_data:
                    data_list.extend(
                        my_convert_output_data(backend_instance, tmp, index)
                    )
                return [nnopbase.create_x_list(data_list)]
            dtype = output_data.dtype
            torch_dtype = getattr(torch, dtype.replace("torch.", ""))
            slot = out_slot["i"]
            out_slot["i"] += 1
            out_shape = _unfused_query_shape(input_data.kwargs, output_data.shape)
            if slot == 0:
                if query_runtime_dtype == torch.float8_e4m3fn:
                    torch_dtype = torch.float8_e4m3fn
                    dtype = "torch.float8_e4m3fn"
                elif query_runtime_dtype == torch.int8:
                    torch_dtype = torch.int8
                    dtype = "torch.int8"
                elif query_runtime_dtype == torch.uint8:
                    empty_tensor = torch.zeros(
                        out_shape, dtype=torch.uint8, device="npu"
                    )
                    backend_instance.output_cache.append(empty_tensor)
                    cur_index = index + len(backend_instance.input_args)
                    fmt = backend_instance.get_format(index=cur_index)
                    return [self.torch_tensor_to_acl(empty_tensor, fmt)]
            if dtype not in TORCH_TO_ACLTYPE:
                raise ValueError(f"TORCH_TO_ACLTYPE不支持的dtype：{dtype}")
            empty_tensor = torch.zeros(out_shape, dtype=torch_dtype, device="npu")
            backend_instance.output_cache.append(empty_tensor)
            cur_index = index + len(backend_instance.input_args)
            fmt = backend_instance.get_format(index=cur_index)
            storage_shape = (
                out_shape
                if out_shape != tuple(int(x) for x in output_data.shape)
                else backend_instance.get_storage_shape(index=cur_index)
            )
            out_tensor = nnopbase.create_acl_tensor(empty_tensor, fmt, storage_shape)
            return [out_tensor]

        hif8 = int(getattr(self, "weight_quant_mode", -1)) == 5 and hasattr(
            AclDataType, "ACL_HIFLOAT8"
        )
        orig_out = backend.convert_output_data
        old_u8 = TORCH_TO_ACLTYPE.get("torch.uint8")
        try:
            backend.convert_output_data = types.MethodType(
                my_convert_output_data, backend
            )
            if hif8:
                # ATK 默认 uint8 → ACL_UINT8；wq=5 的 uint8 是 HiFloat8 存储。
                TORCH_TO_ACLTYPE["torch.uint8"] = int(AclDataType.ACL_HIFLOAT8)
            return super().init_by_input_data(input_data)
        finally:
            backend.convert_output_data = orig_out
            if old_u8 is not None:
                TORCH_TO_ACLTYPE["torch.uint8"] = old_u8

    def __call__(self):
        torch.npu.synchronize()
        self.backend.aclnn_x_get_workspace_size()
        torch.npu.synchronize()
        self.backend.aclnn_x()
        torch.npu.synchronize()

    def after_call(self, output_packages):
        torch.npu.synchronize()
        outs = list(super().after_call(output_packages))
        torch.npu.synchronize()
        if self.nc_factor > 1:
            kv_cache, kr_cache = self.cache_refs
            padding_ok, padding_detail = padding_report(self.nc_caches)
            logging.info(
                "[mla_prolog][nc] factor=%s kv_stride0=%s kr_stride0=%s padding=%s %s",
                self.nc_factor,
                int(kv_cache.stride(0)) if kv_cache.dim() else 0,
                int(kr_cache.stride(0)) if kr_cache.dim() else 0,
                "ok" if padding_ok else "POLLUTED",
                padding_detail,
            )
        # FP8 / HiFloat8 → FP32 再交给 ATK 双标杆（规避 isinf 不支持这些 dtype）
        qq = int(getattr(self, "query_quant_mode", 0))
        wq = int(getattr(self, "weight_quant_mode", -1))
        if (
            outs
            and qq == 1
            and wq in (2, 3, 4, 5)
            and isinstance(self.dequant_q_nope, torch.Tensor)
        ):
            q = outs[0].detach().cpu()
            scale = _reshape_dequant_for_query(self.dequant_q_nope.detach().cpu(), q)
            if wq == 5:
                q = _hif8_decode(q)
            else:
                q = q.to(torch.float32)
            outs[0] = q * scale
        return tuple(_to_compare_dtype(t) for t in outs)


# ---------------------------------------------------------------------------
# V4 ATK register entries
# ---------------------------------------------------------------------------


@register("function_mla_prolog_v4")
class FunctionMlaPrologV4(FunctionMlaPrologBaseApi):
    """Reference node for ATK default cv_fused_double_benchmark (CPU / NPU small-op)."""

    op_version = "v4"


@register("pyaclnn_function_mla_prolog_v4")
class PyAclnnMlaPrologV4(AclnnMlaPrologBaseApi):
    """DUT: aclnnMlaPrologV4WeightNz via ATK opp path."""

    op_version = "v4"

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnMlaPrologV4WeightNzGetWorkspaceSize("
            "const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr, "
            "const aclTensor *weightUk, const aclTensor *weightDkvKr, const aclTensor *rmsnormGammaCq, "
            "const aclTensor *rmsnormGammaCkv, const aclTensor *ropeSin, const aclTensor *ropeCos, "
            "aclTensor *kvCacheRef, aclTensor *krCacheRef, const aclTensor *cacheIndexOptional, "
            "const aclTensor *dequantScaleXOptional, const aclTensor *dequantScaleWDqOptional, "
            "const aclTensor *dequantScaleWUqQrOptional, const aclTensor *dequantScaleWDkvKrOptional, "
            "const aclTensor *quantScaleCkvOptional, const aclTensor *quantScaleCkrOptional, "
            "const aclTensor *smoothScalesCqOptional, const aclTensor *actualSeqLenOptional, "
            "const aclTensor *kNopeClipAlphaOptional, double rmsnormEpsilonCq, double rmsnormEpsilonCkv, "
            "char *cacheModeOptional, int64_t weightQuantMode, int64_t kvCacheQuantMode, "
            "int64_t queryQuantMode, int64_t ckvkrRepoMode, int64_t quantScaleRepoMode, int64_t tileSize, "
            "double qcQrScale, double kcScale, bool doRope, const aclTensor *queryOut, "
            "const aclTensor *queryRopeOut, "
            "const aclTensor *dequantScaleQNopeOutOptional, const aclTensor *queryNormOutOptional, "
            "const aclTensor *dequantScaleQNormOutOptional, "
            "uint64_t *workspaceSize, aclOpExecutor **executor)"
        )
