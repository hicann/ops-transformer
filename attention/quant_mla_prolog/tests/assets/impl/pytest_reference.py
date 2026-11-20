# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 与 pytest 框架解耦的独立数值契约快照。
"""Quant MLA 的独立 CPU Golden 和完整精度验收契约。

Golden 描述数学量化，不模拟设备调度。HiF8 使用 uint8 载体；Cache 保留物理存储表示，
未写入行按字节核对。三方统计保留特殊值开关、dtype 算术和 warning 判定。
"""

from __future__ import annotations


from enum import Enum


import importlib.util


import logging


import math


import os


from pathlib import Path


import struct


import sys


from unittest.mock import patch


import torch


LOGGER = logging.getLogger(__name__)


MXFP8_GROUP_SIZE = 32


CACHE_GROUP_SIZE = 128


E4M3FN_QMAX = 448.0


HIF8_QMAX = 32768.0


CACHE_TILE_AMAX_FLOOR = 1.0e-4


KV_LORA = 512


VALID_Q_LORA = {640, 1024, 1536, 2048}


VALID_DN_DR = {(128, 64), (192, 64), (256, 0)}


VALID_MODE_PAIRS = {(0, 1), (0, 2), (1, 1), (2, 1), (3, 2)}


def _native_dtype(name: str):
    return getattr(torch, name, None)


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).lower()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_cpu(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is not None:
        _require(
            tensor.device.type == "cpu", f"{name} must be a CPU tensor for the golden"
        )


def _require_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    _require(
        tuple(tensor.shape) == tuple(shape),
        f"{name}.shape must be {shape}, got {tuple(tensor.shape)}",
    )


def _is_e4m3fn(tensor: torch.Tensor) -> bool:
    native = _native_dtype("float8_e4m3fn")
    return (
        native is not None and tensor.dtype == native
    ) or tensor.dtype == torch.uint8


def _is_e8m0(tensor: torch.Tensor) -> bool:
    native = _native_dtype("float8_e8m0fnu")
    return (
        native is not None and tensor.dtype == native
    ) or tensor.dtype == torch.uint8


def _is_hif8_carrier(tensor: torch.Tensor) -> bool:
    name = _dtype_name(tensor)
    return (
        tensor.dtype in (torch.uint8, torch.int8)
        or "hifloat8" in name
        or "hif8" in name
    )


def _raw_u8(tensor: torch.Tensor) -> torch.Tensor:
    _require(tensor.is_contiguous(), "byte-carrier tensor must be contiguous")
    if tensor.dtype == torch.uint8:
        return tensor
    return tensor.view(torch.uint8)


def _bf16_sat_input(value: torch.Tensor) -> torch.Tensor:
    """BF16 饱和语义：NaN 转正零，溢出转带符号的有限极值。"""
    limit = torch.finfo(torch.bfloat16).max
    return torch.nan_to_num(value.float(), nan=0.0, posinf=limit, neginf=-limit).clamp(
        -limit, limit
    )


def _bf16_round_away(value: torch.Tensor) -> torch.Tensor:
    """FP32 转 BF16，进行饱和及最近值舍入，半值远离零并保留带符号零。"""
    value = _bf16_sat_input(value).contiguous()
    bits = value.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rounded = ((bits + 0x8000) >> 16) & 0xFFFF
    bf16_bits = rounded.to(torch.int16)
    return bf16_bits.view(torch.bfloat16).to(torch.float32)


def _e4_positive_table(device: torch.device) -> torch.Tensor:
    code = torch.arange(127, dtype=torch.int64, device=device)
    exp = code >> 3
    mant = code & 7
    subnormal = mant.to(torch.float32) * (2.0**-9)
    normal = torch.ldexp(1.0 + mant.to(torch.float32) / 8.0, exp.to(torch.int32) - 7)
    return torch.where(exp == 0, subnormal, normal)


def _e4m3fn_decode(tensor: torch.Tensor) -> torch.Tensor:
    native = _native_dtype("float8_e4m3fn")
    if native is not None and tensor.dtype == native:
        return tensor.to(torch.float32)
    raw = _raw_u8(tensor).to(torch.int64)
    mag = raw & 0x7F
    exp = mag >> 3
    mant = mag & 7
    subnormal = mant.to(torch.float32) * (2.0**-9)
    normal = torch.ldexp(1.0 + mant.to(torch.float32) / 8.0, exp.to(torch.int32) - 7)
    value = torch.where(exp == 0, subnormal, normal)
    value = torch.where((raw & 0x80) != 0, -value, value)
    return torch.where(mag == 0x7F, torch.full_like(value, float("nan")), value)


def _e4m3fn_encode_raw(value: torch.Tensor) -> torch.Tensor:
    """FP32 转 E4M3FN，使用饱和及最近偶数舍入。"""
    value = value.to(torch.float32)
    sign = torch.signbit(value)
    absolute = value.abs()
    table = _e4_positive_table(value.device)
    finite = torch.nan_to_num(absolute, nan=0.0, posinf=E4M3FN_QMAX, neginf=E4M3FN_QMAX)
    finite = finite.clamp(max=E4M3FN_QMAX)

    high = torch.bucketize(finite, table).clamp(max=table.numel() - 1)
    low = (high - 1).clamp(min=0)
    low_value = table[low]
    high_value = table[high]
    low_distance = finite - low_value
    high_distance = high_value - finite
    tie = low_distance == high_distance
    choose_high = (high_distance < low_distance) | (tie & ((high & 1) == 0))
    magnitude = torch.where(choose_high, high, low).to(torch.uint8)
    raw = magnitude | (sign.to(torch.uint8) << 7)
    # 量化饱和将 NaN 映射为零，这与解码输入 NaN 的语义不同。
    raw = torch.where(torch.isnan(value), torch.zeros_like(raw), raw)
    return raw.contiguous()


def _e4m3fn_from_raw(raw: torch.Tensor) -> torch.Tensor:
    native = _native_dtype("float8_e4m3fn")
    raw = raw.to(torch.uint8).contiguous()
    return raw.view(native) if native is not None else raw


def _e4m3fn_quantize(value: torch.Tensor) -> torch.Tensor:
    return _e4m3fn_from_raw(_e4m3fn_encode_raw(value))


def _e8m0_decode(tensor: torch.Tensor) -> torch.Tensor:
    native = _native_dtype("float8_e8m0fnu")
    if native is not None and tensor.dtype == native:
        return tensor.to(torch.float32)
    raw = _raw_u8(tensor).to(torch.int32)
    value = torch.ldexp(torch.ones_like(raw, dtype=torch.float32), raw - 127)
    return torch.where(raw == 255, torch.full_like(value, float("nan")), value)


def _e8m0_from_raw(raw: torch.Tensor) -> torch.Tensor:
    native = _native_dtype("float8_e8m0fnu")
    raw = raw.to(torch.uint8).contiguous()
    return raw.view(native) if native is not None else raw


def _hif8_decode_raw(raw_tensor: torch.Tensor) -> torch.Tensor:
    raw = _raw_u8(raw_tensor).to(torch.int64)
    magnitude = raw & 0x7F
    sign = torch.where((raw & 0x80) != 0, -1.0, 1.0)
    value = torch.zeros_like(magnitude, dtype=torch.float32)

    subnormal = magnitude < 8
    subnormal_value = torch.where(
        magnitude == 0,
        torch.zeros_like(value),
        torch.ldexp(torch.ones_like(value), magnitude.to(torch.int32) - 23),
    )
    value = torch.where(subnormal, subnormal_value, value)

    d0001 = (magnitude >= 8) & (magnitude < 16)
    value = torch.where(d0001, 1.0 + (magnitude & 7).to(torch.float32) / 8.0, value)

    d001 = (magnitude >= 16) & (magnitude < 32)
    exp_bits = (magnitude >> 3) & 1
    exponent = torch.where(
        exp_bits == 0, torch.ones_like(exp_bits), -torch.ones_like(exp_bits)
    )
    normal = torch.ldexp(
        1.0 + (magnitude & 7).to(torch.float32) / 8.0, exponent.to(torch.int32)
    )
    value = torch.where(d001, normal, value)

    d01 = (magnitude >= 32) & (magnitude < 64)
    exp_bits = (magnitude >> 3) & 3
    exp_abs = 2 + (exp_bits & 1)
    exponent = torch.where((exp_bits >> 1) == 0, exp_abs, -exp_abs)
    normal = torch.ldexp(
        1.0 + (magnitude & 7).to(torch.float32) / 8.0, exponent.to(torch.int32)
    )
    value = torch.where(d01, normal, value)

    d10 = (magnitude >= 64) & (magnitude < 96)
    exp_bits = (magnitude >> 2) & 7
    exp_abs = 4 + (exp_bits & 3)
    exponent = torch.where((exp_bits >> 2) == 0, exp_abs, -exp_abs)
    normal = torch.ldexp(
        1.0 + (magnitude & 3).to(torch.float32) / 4.0, exponent.to(torch.int32)
    )
    value = torch.where(d10, normal, value)

    d11 = magnitude >= 96
    exp_bits = (magnitude >> 1) & 15
    exp_abs = 8 + (exp_bits & 7)
    exponent = torch.where((exp_bits >> 3) == 0, exp_abs, -exp_abs)
    normal = torch.ldexp(
        1.0 + (magnitude & 1).to(torch.float32) / 2.0, exponent.to(torch.int32)
    )
    value = torch.where(d11, normal, value)

    value = value * sign
    value = torch.where(magnitude == 111, sign * float("inf"), value)
    value = torch.where(raw == 128, torch.full_like(value, float("nan")), value)
    return value


def _hif8_positive_table(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    codes = torch.arange(128, dtype=torch.uint8, device=device)
    finite_mask = codes != 111
    finite_codes = codes[finite_mask]
    values = _hif8_decode_raw(finite_codes)
    order = torch.argsort(values, stable=True)
    return values[order], finite_codes[order]


def _hif8_encode_raw(value: torch.Tensor) -> torch.Tensor:
    """FP32 转 HiF8，使用饱和及最近值舍入，半值远离零。"""
    value = value.to(torch.float32)
    sign = torch.signbit(value)
    absolute = value.abs()
    table, codes = _hif8_positive_table(value.device)
    # 先饱和再选择最近值；40960 转 Inf 是非饱和规则，本算子不使用该规则。

    finite = torch.nan_to_num(absolute, nan=0.0, posinf=HIF8_QMAX).clamp(max=HIF8_QMAX)

    high = torch.bucketize(finite, table).clamp(max=table.numel() - 1)
    low = (high - 1).clamp(min=0)
    low_distance = finite - table[low]
    high_distance = table[high] - finite
    choose_high = high_distance <= low_distance
    magnitude = torch.where(choose_high, codes[high], codes[low]).to(torch.uint8)
    raw = magnitude | (sign.to(torch.uint8) << 7)
    raw = torch.where(torch.isnan(value), torch.zeros_like(raw), raw)
    # HiF8 的 0x80 保留为 NaN；有限值舍入至零时，即使原符号为负也使用 0x00。

    raw = torch.where(
        (magnitude == 0) & torch.isfinite(value), torch.zeros_like(raw), raw
    )
    return raw.contiguous()


def _hif8_quantize(value: torch.Tensor) -> torch.Tensor:
    return _hif8_encode_raw(value)


def _empty_quantized(w_qr_quant_mode: int, *, scale: bool) -> torch.Tensor:
    if w_qr_quant_mode == 1:
        if scale:
            return _e8m0_from_raw(torch.empty((0,), dtype=torch.uint8))
        return _e4m3fn_from_raw(torch.empty((0,), dtype=torch.uint8))
    return torch.empty((0,), dtype=torch.float32 if scale else torch.uint8)


def _unpack_nz_weight(weight: torch.Tensor) -> torch.Tensor:
    # 物理形状为 (O/32,K/16,16,32)，即 (O1,K1,K0,O0)。
    o1, k1, k0, o0 = weight.shape
    _require(k0 == 16 and o0 == 32, "NZ weight tail must be (16,32)")
    return weight.permute(0, 3, 1, 2).contiguous().reshape(o1 * o0, k1 * k0)


def _unpack_mxfp8_weight_scale(scale: torch.Tensor, weight_format: str) -> torch.Tensor:
    if weight_format == "ND":
        # 逻辑索引变换：(O,K/64,2) 转为 (O,K/32)。
        return _e8m0_decode(scale).reshape(scale.shape[0], -1)
    # 物理索引变换：(O/16,K/64,16,2) 转为 (O,K/32)。
    o1, k1, o0, scale0 = scale.shape
    _require(o0 == 16 and scale0 == 2, "NZ MXFP8 weight descale tail must be (16,2)")
    raw = _e8m0_decode(scale)
    return raw.permute(0, 2, 1, 3).contiguous().reshape(o1 * o0, k1 * scale0)


def _dequant_mxfp8_activation(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, width = data.shape
    _require(
        width % MXFP8_GROUP_SIZE == 0, "MXFP8 activation K must be divisible by 32"
    )
    decoded = _e4m3fn_decode(data).reshape(
        rows, width // MXFP8_GROUP_SIZE, MXFP8_GROUP_SIZE
    )
    return (decoded * _e8m0_decode(scale).reshape(rows, -1, 1)).reshape(rows, width)


def _dequant_mxfp8_weight(
    data: torch.Tensor, scale: torch.Tensor, weight_format: str
) -> torch.Tensor:
    logical = data if weight_format == "ND" else _unpack_nz_weight(data)
    rows, width = logical.shape
    decoded = _e4m3fn_decode(logical).reshape(
        rows, width // MXFP8_GROUP_SIZE, MXFP8_GROUP_SIZE
    )
    logical_scale = _unpack_mxfp8_weight_scale(scale, weight_format)
    return (decoded * logical_scale.unsqueeze(-1)).reshape(rows, width)


def _mxfp8_projection_result(value: torch.Tensor) -> torch.Tensor:
    """处理 MX 投影 FP32 输出的次正规值。

    保留 FP32 累加、输出及所有非次正规值，不加入 BF16 转换，也不改变进程浮点模式。
    """
    _require(value.dtype == torch.float32, "MX projection result must be FP32")
    magnitude = value.abs()
    subnormal = (magnitude > 0) & (magnitude < torch.finfo(torch.float32).tiny)
    return torch.where(subnormal, torch.zeros_like(value), value)


def _mxfp8_matmul(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_descale: torch.Tensor,
    weight_descale: torch.Tensor,
    weight_format: str,
) -> torch.Tensor:
    """输出 FP32 投影结果的 MX 数学参考。

    有限编码解码后可能超出 FP32，而乘积仍可表示。展开产生 Inf 时，
    CPU 此次运算使用 FP64 范围，再将投影转为 FP32；正常范围仍用 FP32 Matmul。
    两条路径均处理 FP32 输出次正规值，不模拟设备累加顺序或设备重算分支。
    """
    left = _dequant_mxfp8_activation(activation, activation_descale)
    right = _dequant_mxfp8_weight(weight, weight_descale, weight_format)
    if not (torch.isinf(left).any() or torch.isinf(right).any()):
        return _mxfp8_projection_result(left @ right.T)
    logical = weight if weight_format == "ND" else _unpack_nz_weight(weight)
    left = (
        _e4m3fn_decode(activation).double().reshape(activation.shape[0], -1, 32)
        * _e8m0_decode(activation_descale).double().reshape(activation.shape[0], -1, 1)
    ).reshape(activation.shape)
    right = (
        _e4m3fn_decode(logical).double().reshape(logical.shape[0], -1, 32)
        * _unpack_mxfp8_weight_scale(weight_descale, weight_format)
        .double()
        .unsqueeze(-1)
    ).reshape(logical.shape)
    return _mxfp8_projection_result((left @ right.T).float())


def _hif8_matmul_dequant(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_descale: torch.Tensor,
    weight_descale: torch.Tensor,
    weight_format: str,
) -> torch.Tensor:
    """依次执行 FP32 编码值 Matmul、权重 descale 和激活 descale。

    乘法保持分离，合并或移入操作数会改变下一次 BF16 转换前的 FP32 舍入。
    接口使用全局激活 descale，不按 token 独立缩放。
    """
    logical = weight if weight_format == "ND" else _unpack_nz_weight(weight)
    coded_result = _hif8_decode_raw(activation) @ _hif8_decode_raw(logical).T
    channel_dequant = coded_result * weight_descale.to(torch.float32).reshape(1, -1)
    return channel_dequant * activation_descale.to(torch.float32).reshape(1, 1)


def _mxfp8_dynamic_quant(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """每 32 个 BF16 元素动态量化为 E4M3FN 与 E8M0，使用数值指数。

    保留 BF16 预舍入和归一化值舍入，这些是数值转换点，不模拟向量指令。
    """
    _require(
        value.shape[-1] % MXFP8_GROUP_SIZE == 0,
        "MXFP8 dynamic-quant width must divide by 32",
    )
    value_bf16 = _bf16_round_away(value)
    groups = value_bf16.reshape(*value_bf16.shape[:-1], -1, MXFP8_GROUP_SIZE)
    amax = groups.abs().amax(dim=-1)
    has_special = ~torch.isfinite(amax)
    zero_exponent = amax < torch.finfo(torch.bfloat16).tiny
    safe_amax = torch.where(has_special | zero_exponent, torch.ones_like(amax), amax)
    # frexp 的指数 e 满足 floor(log2(amax))=e-1，E8M0 字节为该下取整值加 119。
    _, exponent = torch.frexp(safe_amax)
    scale_raw = (exponent + 118).clamp(min=0, max=254).to(torch.uint8)
    scale_raw = torch.where(has_special, torch.full_like(scale_raw, 255), scale_raw)
    scale_raw = torch.where(zero_exponent, torch.zeros_like(scale_raw), scale_raw)
    inverse = torch.ldexp(torch.ones_like(amax), 127 - scale_raw.to(torch.int32))
    inverse = torch.where(zero_exponent, torch.zeros_like(inverse), inverse)
    inverse = torch.where(has_special, torch.full_like(inverse, float("nan")), inverse)
    normalized = (groups * inverse.unsqueeze(-1)).to(torch.bfloat16).float()
    quantized = _e4m3fn_quantize(normalized.reshape_as(value_bf16))
    scale = _e8m0_decode(_e8m0_from_raw(scale_raw))
    reconstructed = _e4m3fn_decode(quantized).reshape_as(groups) * scale.unsqueeze(-1)
    return quantized, _e8m0_from_raw(scale_raw), reconstructed.reshape_as(value_bf16)


def _normalize_dynamic(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """仅保护零 scale，NaN/Inf 算术由 Torch 处理。"""
    zero_scale = scale == 0
    denominator = torch.where(zero_scale, torch.ones_like(scale), scale)
    return torch.where(zero_scale, torch.zeros_like(value), value / denominator)


def _hif8_dynamic_quant(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    amax = value.abs().amax().to(torch.float32)
    scale = (amax / HIF8_QMAX).reshape(1)
    normalized = _normalize_dynamic(value, scale)
    quantized = _hif8_quantize(normalized)
    reconstructed = _hif8_decode_raw(quantized) * scale
    return quantized, scale, reconstructed


def _e4_per_row_quant(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = value.abs().amax(dim=-1, keepdim=True)
    scale = amax / E4M3FN_QMAX
    normalized = _normalize_dynamic(value, scale)
    return _e4m3fn_quantize(normalized), scale.to(torch.float32)


def _e4_per_cache_group_quant(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _require(
        value.shape[-1] % CACHE_GROUP_SIZE == 0,
        "KV width must be divisible by cache group size 128",
    )
    groups = value.reshape(value.shape[0], -1, CACHE_GROUP_SIZE)
    amax = groups.abs().amax(dim=-1)
    scale = amax.clamp_min(CACHE_TILE_AMAX_FLOOR) / E4M3FN_QMAX
    normalized = groups / scale.unsqueeze(-1)
    quantized = _e4m3fn_quantize(normalized.reshape_as(value))
    return quantized, scale.to(torch.float32)


def _rms_norm(value: torch.Tensor, gamma: torch.Tensor, epsilon: float) -> torch.Tensor:
    value = value.float()
    # 此处为 Torch 数学参考，不模拟硬件通道或归约树。
    mean_square = value.square().mean(dim=-1, keepdim=True)
    return value / torch.sqrt(mean_square + epsilon) * gamma.float()


def _apply_norm_scale(value: torch.Tensor, scale: float) -> torch.Tensor:
    """RMSNorm 后以 FP32 应用一次公开 scale。"""
    scale_fp32 = torch.tensor(scale, dtype=torch.float32)
    _require(bool(torch.isfinite(scale_fp32)), "norm scale must be finite in FP32")
    return value * scale_fp32


def _apply_rope(
    value: torch.Tensor,
    rope_sin: torch.Tensor | None,
    rope_cos: torch.Tensor | None,
    rope_mode: int,
    cache_sat: bool = False,
    high_precision: bool = False,
) -> torch.Tensor:
    if value.shape[-1] == 0 or rope_mode == 0:
        if high_precision:
            return value.float()
        if cache_sat:
            value = _bf16_sat_input(value)
        return value.to(torch.bfloat16).to(torch.float32)
    sin = rope_sin.to(torch.float32)
    cos = rope_cos.to(torch.float32)
    while sin.ndim < value.ndim:
        sin = sin.unsqueeze(-2)
        cos = cos.unsqueeze(-2)
    if rope_mode == 1:
        # 交错半旋转配对奇偶元素，再连接两个半区。
        even = value[..., 0::2]
        odd = value[..., 1::2]
        half = value.shape[-1] // 2
        lower = cos[..., :half] * even - sin[..., :half] * odd
        upper = sin[..., half:] * even + cos[..., half:] * odd
        result = torch.cat((lower, upper), dim=-1)
    else:
        # 标准半区旋转遵循 rotate(x)=[-x2,x1]。
        first, second = value.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        result = value * cos + rotated * sin
    if high_precision:
        return result.float()
    if cache_sat:
        result = _bf16_sat_input(result)
    return result.to(torch.bfloat16).to(torch.float32)


def _validate_cache_storage(kv_cache: torch.Tensor, cache_mode: int) -> int:
    _require_cpu("kv_cache", kv_cache)
    _require(cache_mode in (0, 1), "cache_mode must be 0(PA_BSND) or 1(PA_NZ)")
    _require(
        kv_cache.ndim == 4 and kv_cache.shape[2] == 1,
        "cache storage shadow must have public shape (block_number,block_size,1,D)",
    )
    _require(
        kv_cache.is_contiguous(), "physical cache storage shadow must be contiguous"
    )
    _require(
        kv_cache.element_size() in (1, 2, 4),
        "cache must use 1B, 2B or FP32 shadow elements",
    )
    c0 = 32 // kv_cache.element_size()
    if cache_mode == 1:
        _require(kv_cache.shape[-1] % c0 == 0, "PA_NZ width must be C0-aligned")
    return c0


def logical_cache_to_storage(
    logical_cache: torch.Tensor, cache_mode: int
) -> torch.Tensor:
    """从逻辑 BSND 创建 CPU 物理存储影子副本。

    保持公开形状及 dtype，仅重排字节；这不是设备格式转换或新增公开算子参数。
    """
    c0 = _validate_cache_storage(logical_cache, cache_mode)
    if cache_mode == 0:
        return logical_cache.clone()
    blocks, block_size, _, width = logical_cache.shape
    raw = _raw_u8(logical_cache).reshape(
        blocks, block_size, width // c0, c0, logical_cache.element_size()
    )
    packed = raw.permute(0, 2, 1, 3, 4).contiguous().reshape(-1)
    return packed.view(logical_cache.dtype).reshape_as(logical_cache)


def cache_to_logical(kv_cache: torch.Tensor, cache_mode: int) -> torch.Tensor:
    """创建逻辑 CPU Cache，不将其直接用作原始 PA_NZ 存储。"""
    c0 = _validate_cache_storage(kv_cache, cache_mode)
    if cache_mode == 0:
        return kv_cache.clone()
    blocks, block_size, _, width = kv_cache.shape
    raw = _raw_u8(kv_cache).reshape(
        blocks, width // c0, block_size, c0, kv_cache.element_size()
    )
    logical = raw.permute(0, 2, 1, 3, 4).contiguous().reshape(-1)
    return logical.view(kv_cache.dtype).reshape_as(kv_cache)


def cache_to_values(
    kv_cache: torch.Tensor,
    cache_mode: int,
    qkv_quant_mode: int,
    quant_scale_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    """生成用于比较的 FP32 逻辑、反量化 KV/KR Cache，不作为公开算子输入或输出。

    模式 1 解包混合载荷并应用分组 descale；模式 2/3 除以正的量化乘数。
    乘数为零时只能返回解码载荷，不能恢复原值。形状为 (blocks,block_size,1,512+Dr)。
    只比较非负 cache_index 对应的位置；负索引跳过写入，未写槽位可能未初始化。
    """
    _require(qkv_quant_mode in (0, 1, 2, 3), "qkv_quant_mode must be 0..3")
    _require(
        not (qkv_quant_mode == 1 and cache_mode == 1),
        "qkv_quant_mode=1 does not support PA_NZ",
    )
    logical = cache_to_logical(kv_cache, cache_mode)
    width = logical.shape[-1]
    dr = (width - KV_LORA - 16) // 2 if qkv_quant_mode == 1 else width - KV_LORA
    _require(dr in (0, 64), "cache width must match Dr=0 or Dr=64")
    if qkv_quant_mode == 0:
        _require(logical.dtype == torch.bfloat16, "mode-0 cache must be BF16")
        return logical.float()
    if qkv_quant_mode == 1:
        _require(width == KV_LORA + 2 * dr + 16, "mode-1 cache width mismatch")
        _require(logical.element_size() == 1, "mode-1 cache needs a byte carrier")
        raw = _raw_u8(logical)
        kv = _e4m3fn_decode(raw[..., :KV_LORA].contiguous())
        kr = (
            raw[..., KV_LORA : KV_LORA + 2 * dr]
            .contiguous()
            .view(torch.bfloat16)
            .float()
        )
        descale = raw[..., KV_LORA + 2 * dr :].contiguous().view(torch.float32)
        groups = kv.reshape(*kv.shape[:-1], 4, CACHE_GROUP_SIZE)
        kv = (groups * descale.unsqueeze(-1)).reshape_as(kv)
        return torch.cat((kv, kr), dim=-1)
    _require(quant_scale_kv is not None, "mode-2/3 cache requires quant_scale_kv")
    _require_cpu("quant_scale_kv", quant_scale_kv)
    _require_shape("quant_scale_kv", quant_scale_kv, (1,))
    _require(quant_scale_kv.dtype == torch.float32, "quant_scale_kv must be FP32")
    _require(
        bool(torch.isfinite(quant_scale_kv).all()),
        "quant_scale_kv must be finite",
    )
    decoded = (
        _e4m3fn_decode(logical) if qkv_quant_mode == 2 else _hif8_decode_raw(logical)
    )
    return decoded if quant_scale_kv.item() == 0 else decoded / quant_scale_kv


def _store_cache_rows(
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    rows: torch.Tensor,
    cache_mode: int = 0,
) -> None:
    c0 = _validate_cache_storage(kv_cache, cache_mode)
    block_number, block_size = kv_cache.shape[:2]
    width = kv_cache.shape[-1]
    _require(rows.shape == (cache_index.numel(), width), "cache row shape mismatch")
    _require(rows.dtype == kv_cache.dtype, "cache row dtype must match storage dtype")
    # 按字节复制以支持原生 float8，并保留带符号零和 NaN 位模式。
    row_bytes = _raw_u8(rows.contiguous()).reshape(cache_index.numel(), -1)
    cache_bytes = _raw_u8(kv_cache)
    if cache_mode == 1:
        physical = cache_bytes.reshape(block_number, width // c0, block_size, 32)
    else:
        physical = cache_bytes.reshape(block_number, block_size, -1)
    index = cache_index.to(torch.int64)
    for token in range(index.numel()):
        position = int(index[token].item())
        # 负索引只禁止 Cache 写入，不跳过 Q/QR；在地址运算前检查，包含 INT64_MIN。

        if position < 0:
            continue
        block = position // block_size
        offset = position % block_size
        _require(
            0 <= block < block_number,
            f"cache_index[{token}]={position} is out of range",
        )
        if cache_mode == 1:
            # 元素偏移为 block*block_size*D + (d//C0)*block_size*C0 + offset*C0 + d%C0。

            physical[block, :, offset, :].copy_(row_bytes[token].reshape(-1, 32))
        else:
            physical[block, offset].copy_(row_bytes[token])


def _store_cache_bytes(
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    payload: torch.Tensor,
    cache_mode: int = 0,
) -> None:
    _require(
        kv_cache.element_size() == 1,
        "opaque cache carrier must use one byte per element",
    )
    _store_cache_rows(
        _raw_u8(kv_cache), cache_index, payload.to(torch.uint8), cache_mode
    )


def high_precision_projection(x, weight, *, rows_per_chunk=8192):
    """对完整 K 维使用 FP64 累加，再输出 FP32。

    只对独立 token 行分批以限制扩展精度输入副本，不拆分 K 维或加入中间 BF16 舍入。
    此路径仅用于高精度 KVA 参考，不改变普通 Golden 的数值路径。
    """
    if x.device.type != "cpu" or weight.device.type != "cpu":
        raise ValueError("high-precision projection requires CPU operands")
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("expected x[T,K] and weight[O,K]")
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    result = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.float32)
    weight_fp64 = weight.to(torch.float64)
    for start in range(0, x.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, x.shape[0])
        result[start:stop] = (x[start:stop].to(torch.float64) @ weight_fp64.T).to(
            torch.float32
        )
    return result


def quant_mla_prolog_golden(
    x: torch.Tensor,
    wqa: torch.Tensor,
    wqb: torch.Tensor,
    wkva: torch.Tensor,
    wkb: torch.Tensor,
    descale_x: torch.Tensor,
    descale_wqa: torch.Tensor,
    descale_wqb: torch.Tensor,
    descale_wkva: torch.Tensor,
    norm_weight_qa: torch.Tensor,
    norm_weight_kva: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    quant_scale_kv: torch.Tensor | None = None,
    rope_sin: torch.Tensor | None = None,
    rope_cos: torch.Tensor | None = None,
    *,
    norm_eps: float = 1.0e-5,
    w_qr_quant_mode: int,
    qkv_quant_mode: int,
    cache_mode: int,
    qr_scale: float = 1.0,
    kvr_scale: float = 1.0,
    rope_mode: int = 1,
    return_qr: bool = False,
    action_type: str = "bm_output",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (q, descale_q, qr, descale_qr)，并原地更新 kv_cache。

    bm_output 保留面向设备接口的量化与舍入；bm_output_gold 解码相同输入及 scale，
    跳过中间量化并保留 FP32 实数输出和 Cache。
    参数顺序与 DSL 接口一致，属性仅限关键字；不启用 RoPE 时，其输入可为 None 或合法张量对。
    Cache 保存 cache_mode 对应的物理字节，模式 1 的 cache[block,row,0] 不是逻辑行。
    负索引只跳过对应 Cache 载荷写入；比较所有 Q/QR 行，仅比较非负索引的 Cache。
    """
    _require(action_type in ("bm_output", "bm_output_gold"), "invalid action_type")
    high_precision = action_type == "bm_output_gold"
    tensors = {
        "x": x,
        "wqa": wqa,
        "wqb": wqb,
        "wkva": wkva,
        "wkb": wkb,
        "descale_x": descale_x,
        "descale_wqa": descale_wqa,
        "descale_wqb": descale_wqb,
        "descale_wkva": descale_wkva,
        "norm_weight_qa": norm_weight_qa,
        "norm_weight_kva": norm_weight_kva,
        "kv_cache": kv_cache,
        "cache_index": cache_index,
        "quant_scale_kv": quant_scale_kv,
        "rope_sin": rope_sin,
        "rope_cos": rope_cos,
    }
    for name, tensor in tensors.items():
        _require_cpu(name, tensor)

    _require(
        (qkv_quant_mode, w_qr_quant_mode) in VALID_MODE_PAIRS,
        f"unsupported (qkv_quant_mode,w_qr_quant_mode)=({qkv_quant_mode},{w_qr_quant_mode})",
    )
    _require(cache_mode in (0, 1), "cache_mode must be 0(PA_BSND) or 1(PA_NZ)")
    _require(
        not (qkv_quant_mode == 1 and cache_mode == 1),
        "qkv_quant_mode=1 does not support PA_NZ",
    )
    _require(rope_mode in (0, 1, 2), "rope_mode must be 0, 1, or 2")
    if type(norm_eps) not in (float, int):
        raise TypeError("norm_eps must be a Python float or int, not bool or Tensor")
    try:
        norm_eps = struct.unpack("f", struct.pack("f", norm_eps))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(
            "norm_eps is not representable as positive finite FP32"
        ) from error
    _require(
        math.isfinite(norm_eps) and norm_eps > 0,
        "norm_eps must be finite and strictly positive",
    )
    for name, value in (("qr_scale", qr_scale), ("kvr_scale", kvr_scale)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a Python float or int, not bool or Tensor")
        try:
            fp32_value = struct.unpack("f", struct.pack("f", float(value)))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError(f"{name} must be finite in FP32") from error
        _require(
            math.isfinite(fp32_value),
            f"{name} must be finite in FP32",
        )
    _require(x.ndim == 2, "x must be (T,dim)")

    t, dim = x.shape
    q_lora = norm_weight_qa.numel()
    kv_lora = norm_weight_kva.numel()
    _require(
        1024 <= dim <= 8192 and dim % 256 == 0,
        "dim must be [1024,8192] and 256-aligned",
    )
    _require(q_lora in VALID_Q_LORA, f"q_lora must be one of {sorted(VALID_Q_LORA)}")
    _require(kv_lora == KV_LORA, "kv_lora must be 512")
    _require(wkb.ndim == 3, "wkb must be (N,Dn,kv_lora)")
    n, dn, wkb_kv = wkb.shape
    _require(1 <= n <= 128, "N must be in [1,128]")
    _require(wkb_kv == kv_lora, "wkb.shape[2] must equal kv_lora")

    ranks = (wqa.ndim, wqb.ndim, wkva.ndim)
    _require(
        ranks in ((2, 2, 2), (4, 4, 4)),
        "wqa/wqb/wkva must be uniformly ND or uniformly NZ",
    )
    weight_format = "ND" if ranks[0] == 2 else "NZ"
    logical_wqa_shape = (
        tuple(wqa.shape)
        if weight_format == "ND"
        else (wqa.shape[0] * 32, wqa.shape[1] * 16)
    )
    logical_wqb_shape = (
        tuple(wqb.shape)
        if weight_format == "ND"
        else (wqb.shape[0] * 32, wqb.shape[1] * 16)
    )
    logical_wkva_shape = (
        tuple(wkva.shape)
        if weight_format == "ND"
        else (wkva.shape[0] * 32, wkva.shape[1] * 16)
    )
    _require(logical_wqa_shape == (q_lora, dim), "wqa logical shape mismatch")
    _require(
        logical_wqb_shape[1] == q_lora and logical_wqb_shape[0] % n == 0,
        "wqb logical shape mismatch",
    )
    dr = logical_wkva_shape[0] - kv_lora
    _require(logical_wkva_shape == (kv_lora + dr, dim), "wkva logical shape mismatch")
    _require((dn, dr) in VALID_DN_DR, f"(Dn,Dr)=({dn},{dr}) is unsupported")
    _require(
        logical_wqb_shape == (n * (dn + dr), q_lora), "wqb must be (N*(Dn+Dr),q_lora)"
    )
    if dr == 0:
        _require(rope_mode == 0, "Dr=0 requires rope_mode=0")

    _require_shape("norm_weight_qa", norm_weight_qa, (q_lora,))
    _require_shape("norm_weight_kva", norm_weight_kva, (kv_lora,))
    _require(
        norm_weight_qa.dtype == torch.float32
        and norm_weight_kva.dtype == torch.float32,
        "norm weights must be FP32",
    )
    _require(wkb.dtype == torch.bfloat16, "wkb must be BF16")
    _require(cache_index.dtype == torch.int64, "cache_index must be INT64")
    _require_shape("cache_index", cache_index, (t,))
    _require(
        kv_cache.ndim == 4 and kv_cache.shape[2] == 1,
        "kv_cache must be (block_number,block_size,1,D)",
    )
    _require(
        kv_cache.shape[0] > 0 and kv_cache.shape[1] > 0,
        "cache block_number/block_size must be positive",
    )
    _validate_cache_storage(kv_cache, cache_mode)

    if (dr and rope_mode) or rope_sin is not None or rope_cos is not None:
        _require(
            rope_sin is not None and rope_cos is not None,
            "rope_sin/rope_cos must be supplied together for active RoPE or omitted together",
        )
        _require_shape("rope_sin", rope_sin, (t, dr))
        _require_shape("rope_cos", rope_cos, (t, dr))
        _require(
            rope_sin.dtype == torch.bfloat16 and rope_cos.dtype == torch.bfloat16,
            "rope_sin/rope_cos must be BF16",
        )

    if qkv_quant_mode in (2, 3):
        _require(
            quant_scale_kv is not None,
            "quant_scale_kv is required for qkv_quant_mode=2/3",
        )
        _require_shape("quant_scale_kv", quant_scale_kv, (1,))
        _require(quant_scale_kv.dtype == torch.float32, "quant_scale_kv must be FP32")
        if t != 0:
            kv_scale = float(quant_scale_kv.item())
            _require(
                math.isfinite(kv_scale),
                "quant_scale_kv must be finite",
            )

    dtile = kv_lora + 2 * dr + 4 * math.ceil(kv_lora / CACHE_GROUP_SIZE)
    expected_cache_tail = (
        dtile if qkv_quant_mode == 1 and not high_precision else kv_lora + dr
    )
    _require(
        kv_cache.shape[-1] == expected_cache_tail,
        f"kv_cache last dim must be {expected_cache_tail} for qkv_quant_mode={qkv_quant_mode}",
    )
    if high_precision:
        _require(kv_cache.dtype == torch.float32, "high Golden cache must be FP32")
    elif qkv_quant_mode == 0:
        _require(kv_cache.dtype == torch.bfloat16, "mode 0 kv_cache must be BF16")
    elif qkv_quant_mode in (1, 2):
        _require(
            _is_e4m3fn(kv_cache),
            "mode 1/2 kv_cache must be E4M3FN or uint8 raw carrier",
        )
    else:
        _require(
            _is_hif8_carrier(kv_cache), "mode 3 kv_cache must be HiF8 byte carrier"
        )

    if w_qr_quant_mode == 1:
        _require(
            all(_is_e4m3fn(tensor) for tensor in (x, wqa, wqb, wkva)),
            "mode 1 x/wqa/wqb/wkva must be E4M3FN or uint8 raw carriers",
        )
        _require(
            all(
                _is_e8m0(tensor)
                for tensor in (descale_x, descale_wqa, descale_wqb, descale_wkva)
            ),
            "mode 1 descales must be E8M0 or uint8 raw carriers",
        )
        _require_shape("descale_x", descale_x, (t, dim // 32))
        if weight_format == "ND":
            _require_shape("descale_wqa", descale_wqa, (q_lora, dim // 64, 2))
            _require_shape("descale_wqb", descale_wqb, (n * (dn + dr), q_lora // 64, 2))
            _require_shape("descale_wkva", descale_wkva, (kv_lora + dr, dim // 64, 2))
        else:
            _require_shape("descale_wqa", descale_wqa, (q_lora // 16, dim // 64, 16, 2))
            _require_shape(
                "descale_wqb", descale_wqb, (n * (dn + dr) // 16, q_lora // 64, 16, 2)
            )
            _require_shape(
                "descale_wkva", descale_wkva, ((kv_lora + dr) // 16, dim // 64, 16, 2)
            )
    else:
        _require(
            all(_is_hif8_carrier(tensor) for tensor in (x, wqa, wqb, wkva)),
            "mode 2 x/wqa/wqb/wkva must be HiF8 byte carriers",
        )
        _require(
            all(
                tensor.dtype == torch.float32
                for tensor in (descale_x, descale_wqa, descale_wqb, descale_wkva)
            ),
            "HiF8 descales must be FP32 ND",
        )
        _require_shape("descale_x", descale_x, (1,))
        _require_shape("descale_wqa", descale_wqa, (1, q_lora))
        _require_shape("descale_wqb", descale_wqb, (1, n * (dn + dr)))
        _require_shape("descale_wkva", descale_wkva, (1, kv_lora + dr))

    effective_cache_index = cache_index[cache_index >= 0]
    if effective_cache_index.numel():
        _require(
            torch.unique(effective_cache_index).numel()
            == effective_cache_index.numel(),
            "nonnegative cache_index values must be unique within one call",
        )
        _require(
            int(effective_cache_index.max()) < kv_cache.shape[0] * kv_cache.shape[1],
            "cache_index is out of cache range",
        )

    if t == 0:
        if high_precision:
            return (
                torch.empty((0, n, kv_lora + dr), dtype=torch.float32),
                torch.empty(
                    (0, n, 1) if qkv_quant_mode == 2 else (0,), dtype=torch.float32
                ),
                torch.empty((0, q_lora) if return_qr else (0,), dtype=torch.float32),
                torch.empty(
                    (0, q_lora // 32) if return_qr and w_qr_quant_mode == 1 else (0,),
                    dtype=torch.float32,
                ),
            )
        if qkv_quant_mode in (0, 1):
            q = torch.empty((0, n, KV_LORA + dr), dtype=torch.bfloat16)
        elif qkv_quant_mode == 2:
            q = _e4m3fn_from_raw(torch.empty((0, n, KV_LORA + dr), dtype=torch.uint8))
        else:
            q = torch.empty((0, n, KV_LORA + dr), dtype=torch.uint8)
        descale_q = torch.empty(
            (0, n, 1) if qkv_quant_mode == 2 else (0,), dtype=torch.float32
        )
        qr = _empty_quantized(w_qr_quant_mode, scale=False)
        descale_qr = _empty_quantized(w_qr_quant_mode, scale=True)
        if return_qr:
            qr = qr.reshape(0, q_lora)
            descale_qr = (
                descale_qr.reshape(0, q_lora // 32)
                if w_qr_quant_mode == 1
                else torch.empty((0,), dtype=torch.float32)
            )
        return q, descale_q, qr, descale_qr

    if high_precision:
        if w_qr_quant_mode == 1:
            x_real = _dequant_mxfp8_activation(x, descale_x)
            weights_real = [
                _dequant_mxfp8_weight(w, scale, weight_format)
                for w, scale in (
                    (wqa, descale_wqa),
                    (wqb, descale_wqb),
                    (wkva, descale_wkva),
                )
            ]
        else:
            x_real = _hif8_decode_raw(x) * descale_x.float()
            weights_real = [
                _hif8_decode_raw(w if weight_format == "ND" else _unpack_nz_weight(w))
                * scale.float().reshape(-1, 1)
                for w, scale in (
                    (wqa, descale_wqa),
                    (wqb, descale_wqb),
                    (wkva, descale_wkva),
                )
            ]
        qa = x_real @ weights_real[0].T
        # 所有高精度模式在解码后使用 FP64 进行 KVA 累加。
        kva_proj = high_precision_projection(x_real, weights_real[2])
        qa_scaled = _apply_norm_scale(_rms_norm(qa, norm_weight_qa, norm_eps), qr_scale)
        q_proj = qa_scaled @ weights_real[1].T
        qr_all = qa_scaled
        if w_qr_quant_mode == 1:
            amax = qa_scaled.reshape(t, -1, MXFP8_GROUP_SIZE).abs().amax(dim=-1)
            special = ~torch.isfinite(amax)
            tiny = amax < torch.finfo(torch.bfloat16).tiny
            safe = torch.where(special | tiny, torch.ones_like(amax), amax)
            _, exponent = torch.frexp(safe)
            codes = (exponent + 118).clamp(0, 254).to(torch.uint8)
            codes = torch.where(tiny, torch.zeros_like(codes), codes)
            codes = torch.where(special, torch.full_like(codes, 255), codes)
            descale_qr_all = _e8m0_decode(codes)
        else:
            descale_qr_all = (qa_scaled.abs().amax() / HIF8_QMAX).reshape(1)
    else:
        if w_qr_quant_mode == 1:
            qa = _mxfp8_matmul(x, wqa, descale_x, descale_wqa, weight_format)
            kva_proj = _mxfp8_matmul(x, wkva, descale_x, descale_wkva, weight_format)
        else:
            qa = _hif8_matmul_dequant(x, wqa, descale_x, descale_wqa, weight_format)
            kva_proj = _hif8_matmul_dequant(
                x, wkva, descale_x, descale_wkva, weight_format
            )

        qa_scaled = _apply_norm_scale(_rms_norm(qa, norm_weight_qa, norm_eps), qr_scale)
        if w_qr_quant_mode == 1:
            qr_all, descale_qr_all, _ = _mxfp8_dynamic_quant(qa_scaled)
            q_proj = _mxfp8_matmul(
                qr_all, wqb, descale_qr_all, descale_wqb, weight_format
            )
        else:
            qr_all, descale_qr_all, _ = _hif8_dynamic_quant(qa_scaled)
            q_proj = _hif8_matmul_dequant(
                qr_all, wqb, descale_qr_all, descale_wqb, weight_format
            )

    q_proj = q_proj.reshape(t, n, dn + dr)
    q_nope = q_proj[..., :dn]
    if not high_precision:
        q_nope = q_nope.to(torch.bfloat16).to(torch.float32)
    q_nope = torch.einsum("tnd,ndk->tnk", q_nope, wkb.to(torch.float32))
    if not high_precision:
        q_nope = q_nope.to(torch.bfloat16).to(torch.float32)
    q_rope = _apply_rope(
        q_proj[..., dn:], rope_sin, rope_cos, rope_mode, high_precision=high_precision
    )
    q_fp = torch.cat((q_nope, q_rope), dim=-1)

    kv_raw = kva_proj[..., :kv_lora]
    kr_raw = kva_proj[..., kv_lora:]
    kv_fp = _apply_norm_scale(_rms_norm(kv_raw, norm_weight_kva, norm_eps), kvr_scale)
    kr_fp = _apply_rope(
        kr_raw,
        rope_sin,
        rope_cos,
        rope_mode,
        cache_sat=True,
        high_precision=high_precision,
    )

    if high_precision:
        q = q_fp.float()
        if qkv_quant_mode in (0, 1):
            descale_q = torch.empty((0,), dtype=torch.float32)
        elif qkv_quant_mode == 2:
            _, descale_q = _e4_per_row_quant(q_fp)
        else:
            descale_q = (q_fp.abs().amax() / HIF8_QMAX).reshape(1)
        _store_cache_rows(
            kv_cache, cache_index, torch.cat((kv_fp, kr_fp), dim=-1), cache_mode
        )
        return (
            q,
            descale_q,
            qr_all if return_qr else torch.empty(0, dtype=torch.float32),
            descale_qr_all if return_qr else torch.empty(0, dtype=torch.float32),
        )

    if qkv_quant_mode in (0, 1):
        q = q_fp.to(torch.bfloat16)
        descale_q = torch.empty((0,), dtype=torch.float32)
    elif qkv_quant_mode == 2:
        q, descale_q = _e4_per_row_quant(q_fp)
    else:
        q, descale_q, _ = _hif8_dynamic_quant(q_fp)

    if qkv_quant_mode == 0:
        cache_rows = _bf16_sat_input(torch.cat((kv_fp, kr_fp), dim=-1)).to(
            torch.bfloat16
        )
        _store_cache_rows(kv_cache, cache_index, cache_rows, cache_mode)
    elif qkv_quant_mode == 1:
        kv_quant, kv_descale = _e4_per_cache_group_quant(kv_fp)
        kv_bytes = _raw_u8(kv_quant).reshape(t, kv_lora)
        kr_bytes = (
            kr_fp.to(torch.bfloat16).contiguous().view(torch.uint8).reshape(t, 2 * dr)
        )
        scale_bytes = kv_descale.contiguous().view(torch.uint8).reshape(t, -1)
        payload = torch.cat((kv_bytes, kr_bytes, scale_bytes), dim=-1)
        _require(payload.shape == (t, dtile), "internal mode-1 payload shape mismatch")
        _store_cache_bytes(kv_cache, cache_index, payload, cache_mode)
    elif qkv_quant_mode == 2:
        scale = quant_scale_kv.to(torch.float32)
        cache_rows = _e4m3fn_quantize(torch.cat((kv_fp, kr_fp), dim=-1) * scale)
        if kv_cache.dtype == torch.uint8:
            _store_cache_bytes(kv_cache, cache_index, _raw_u8(cache_rows), cache_mode)
        else:
            _store_cache_rows(kv_cache, cache_index, cache_rows, cache_mode)
    else:
        scale = quant_scale_kv.to(torch.float32)
        cache_rows = _hif8_quantize(torch.cat((kv_fp, kr_fp), dim=-1) * scale)
        _store_cache_bytes(kv_cache, cache_index, cache_rows, cache_mode)

    if return_qr:
        qr, descale_qr = qr_all, descale_qr_all
    else:
        qr = _empty_quantized(w_qr_quant_mode, scale=False)
        descale_qr = _empty_quantized(w_qr_quant_mode, scale=True)
    return q, descale_q, qr, descale_qr


DEFAULT_CONFIG = dict(
    small_value=2.0**-10,
    small_value_atol=2.0**-16,
    max_re_rtol=10.0,
    avg_re_rtol=2.0,
    rmse_rtol=2.0,
)


def _aligned_error_metrics(value, golden, config):
    """在独立副本上计算统计量，并启用全部特殊值处理开关。"""
    if value.shape != golden.shape:
        raise ValueError("shape mismatch")
    v, g = value.detach().flatten().clone(), golden.detach().flatten().clone()
    limit = torch.finfo(v.dtype)
    g[g > limit.max] = torch.inf
    g[g < limit.min] = -torch.inf
    g[torch.isinf(g)] = limit.max
    v[torch.isinf(v)] = limit.max
    g[torch.isnan(g)] = 0
    v[torch.isnan(v)] = 0
    special = {}
    for name, predicate in (
        ("nan", torch.isnan),
        ("inf", torch.isposinf),
        ("ninf", torch.isneginf),
    ):
        special["err_total_" + name] = int((predicate(v) ^ predicate(g)).sum())
    v[~torch.isfinite(v)] = 1
    g[~torch.isfinite(g)] = 1
    big = g.abs() >= config["small_value"]
    vb, gb = v.clone(), g.clone()
    vb[~big], gb[~big] = 1, 1
    delta = (vb - gb).abs()
    relative = delta / gb.abs()
    vs, gs = v.clone(), g.clone()
    mask = g.abs() > config["small_value"]
    vs[mask], gs[mask] = 1, 1
    return dict(
        elements=g.numel(),
        big_elements=int(big.sum()),
        small_elements=int((~big).sum()),
        max_re=float(relative.max()) if g.numel() else 0.0,
        avg_re=float(relative.sum() / big.sum()) if g.numel() else 0.0,
        small_bad=int(((vs - gs).abs() > config["small_value_atol"]).sum()),
        rmse=float(delta.square().mean().sqrt()) if g.numel() else 0.0,
        **special,
    )


def _aligned_compare_three_way(npu, cpu, cpu_gold, output_dtype, config=None):
    """分别返回原始 CPU 误差和按 dtype 下限调整后的阈值。"""
    cfg = dict(DEFAULT_CONFIG if config is None else config)
    if set(cfg) != set(DEFAULT_CONFIG) or not all(
        isinstance(v, (int, float)) and 0 < v < float("inf") for v in cfg.values()
    ):
        raise ValueError("all five comparison parameters must be finite and positive")
    if output_dtype not in ("bf16", "fp16", "fp32", "hf32", "float8_e4m3fn", "e8m0"):
        raise ValueError("explicit canonical comparison dtype required")
    if npu.shape != cpu.shape or npu.shape != cpu_gold.shape:
        raise ValueError("all three shapes must match")
    if output_dtype == "bf16":
        npu, cpu = npu.float().bfloat16(), cpu.float().bfloat16()
    elif output_dtype == "float8_e4m3fn":
        npu, cpu = npu.float(), cpu.float()
    d, b = (
        _aligned_error_metrics(npu, cpu_gold, cfg),
        _aligned_error_metrics(cpu, cpu_gold, cfg),
    )
    floor = {"bf16": 2.0**-8, "fp16": 2.0**-11}.get(output_dtype, 0.0)
    effective = dict(b)
    for k in ("max_re", "avg_re", "rmse"):
        effective[k] = float(torch.tensor(effective[k]).maximum(torch.tensor(floor)))
    limits = dict(
        max_re=effective["max_re"] * cfg["max_re_rtol"],
        avg_re=effective["avg_re"] * cfg["avg_re_rtol"],
        small_bad=effective["small_bad"] * cfg["avg_re_rtol"],
        rmse=effective["rmse"] * cfg["rmse_rtol"],
    )
    errors = [k for k, limit in limits.items() if d[k] > limit]
    warnings = [k for k in limits if d[k] > effective[k] and k not in errors]
    for k in ("err_total_nan", "err_total_inf", "err_total_ninf"):
        if d[k] > b[k]:
            errors.append(k)
        elif d[k] > 0:
            warnings.append(k)
    return dict(
        status="Failed" if errors else "warning" if warnings else "Pass",
        npu=d,
        cpu=b,
        cpu_effective=effective,
        limits=limits,
        errors=errors,
        warnings=warnings,
        output_dtype=output_dtype,
        benchmark_floor=floor,
        config=cfg,
        empty=d["elements"] == 0,
    )


def _aligned_pair(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("comparison requires identical shapes")
    if actual.device.type != "cpu" or expected.device.type != "cpu":
        raise ValueError("comparison requires independent CPU snapshots")


def _aligned_record(method, elements, failed, allowed, passed, **details):
    return dict(
        method=method,
        elements=elements,
        failed_elements=failed,
        allowed_failures=allowed,
        passed=bool(passed),
        status="EMPTY" if elements == 0 else "PASS" if passed else "FAIL",
        **details,
    )


def first_failed_indices(bad, *, limit=8, chunk_elements=1 << 20):
    """返回最先失败的展平索引，不分配全部失败索引。"""
    if limit < 0 or chunk_elements <= 0:
        raise ValueError("limit must be nonnegative and chunk_elements positive")
    flat = bad.flatten()
    indices = []
    if limit == 0:
        return indices
    for start in range(0, flat.numel(), chunk_elements):
        local = flat[start : start + chunk_elements].nonzero().flatten()
        indices.extend(start + i for i in local[: limit - len(indices)].tolist())
        if len(indices) == limit:
            break
    return indices


def _aligned_first_failed_elements(actual, expected, bad, *, limit=8, **details):
    """在当前分段的展平索引空间中提取诊断样本。"""
    indices = first_failed_indices(bad, limit=limit)
    a, b = actual.flatten(), expected.flatten()
    return [
        dict(
            flat_index=i,
            actual=a[i].item(),
            expected=b[i].item(),
            **{name: values.flatten()[i].item() for name, values in details.items()},
        )
        for i in indices
    ]


def _aligned_mixed(actual, expected, *, scale=False):
    """对 BF16/FP32 scale 执行 isclose、分段配额和相对误差硬阈值检查。"""
    _aligned_pair(actual, expected)
    a, b = actual.float().flatten(), expected.float().flatten()
    m = a.numel()
    atol, rtol = (0.000025, 0.005) if scale else (0.0001, 1 / 128)
    finite = torch.isfinite(a) & torch.isfinite(b)
    same_special = (
        (torch.isnan(a) & torch.isnan(b))
        | (torch.isposinf(a) & torch.isposinf(b))
        | (torch.isneginf(a) & torch.isneginf(b))
    )
    # 混合容差使用 FP32 判定，并以整数实现 0.5% 配额。
    close = torch.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
    bad = ~close
    finite_bad = finite & bad
    denominator = torch.maximum(a.abs(), b.abs()).clamp_min((1 / 16384) / 0.005) + 1e-9
    relative = (a - b).abs() / (denominator + 1e-9)
    hard = float(relative[finite_bad].max()) if finite_bad.any() else 0.0
    failed, allowed = int(bad.sum()), math.floor(m * 0.005)
    return _aligned_record(
        1,
        m,
        failed,
        allowed,
        failed * 200 <= m
        and hard < 10
        and bool(torch.isfinite(relative[finite_bad]).all()),
        first_failed_elements=_aligned_first_failed_elements(
            a, b, bad, relative_error=relative
        ),
        dtype_rule="FP32 scale" if scale else "BF16",
        atol=atol,
        rtol=rtol,
        pct_thd=0.005,
        max_bad_relative=hard,
        hard_limit=10,
        special_mismatches=int((~finite & ~same_special).sum()),
    )


def _aligned_hif8(actual_decoded, reference_fp32, *, diff_thd=0.005):
    """按 method6 检查溢出到 Inf、特殊值匹配和通过比例。"""
    if not 0 <= diff_thd <= 1:
        raise ValueError("diff_thd must be in [0, 1]")
    _aligned_pair(actual_decoded, reference_fp32)
    if reference_fp32.dtype != torch.float32:
        raise TypeError("HiF8 reference must be decoded FP32")
    h = actual_decoded.double().flatten()
    x = reference_fp32.clone().flatten()
    x = torch.where(x > 32768, float("inf"), x)
    x = torch.where(x < -32768, -float("inf"), x).double()
    ex = torch.log2(x.abs() + 2.0**-1000).clamp_min(-22)
    exponent = ex.floor()
    width = torch.zeros_like(exponent)
    for limit, bits in ((15, 1), (7, 2), (3, 3)):
        width = torch.where(exponent.abs() <= limit, bits, width)
    ulp = (h - x) * torch.pow(2.0, -exponent + width)
    signed_ex = ex * torch.where(x >= 0, 1, -1)
    signed_eh = torch.log2(h.abs() + 2.0**-1000) * torch.where(h >= 0, 1, -1)
    ulp = torch.where(width == 0, signed_eh - signed_ex, ulp)
    ulp = torch.where(
        torch.isinf(x) | torch.isinf(h), torch.where(x == h, 0.0, float("nan")), ulp
    )
    ulp = torch.where(
        torch.isnan(x) | torch.isnan(h),
        torch.where(torch.isnan(x) & torch.isnan(h), 0.0, float("nan")),
        ulp,
    )
    ulp = torch.where(
        (x == 0) | (x.abs() < 2.0**-23), torch.where(h == 0, 0.0, 1.5), ulp
    )
    bad = (ulp.abs() > 1) | torch.isnan(ulp)
    m, failed = h.numel(), int(bad.sum())
    effective = float((m * 0.001 + 1) / m) if m > 100 else diff_thd
    percent = float(m - failed) / float(m) * 100.0 if m else 100.0
    errors = ulp[bad].tolist()
    return _aligned_record(
        6,
        m,
        failed,
        m // 1000 + 1 if m > 100 else int(m * diff_thd),
        percent >= float(1 - effective) / float(1) * 100.0,
        first_failed_elements=_aligned_first_failed_elements(h, x, bad, ulp=ulp),
        effective_fail_fraction=effective,
        fulfill_percent=percent,
        max_error=max(errors) - 1.0 if errors else 0.0,
        reference_domain="decoded_ordinary_hif8_overflow_to_inf",
    )


POLICY_ID = "20261103_shared_comparison_independent_golden"


ACCEPTANCE_POLICY_ID = "20261103_all_outputs_whole_or_split_two_or_three"


def compare_three_way(npu, cpu, cpu_gold, output_dtype, config=None):
    return _aligned_compare_three_way(npu, cpu, cpu_gold, output_dtype, config)


def evaluate_segments(parts, config=None):
    records = []
    for p in parts:
        r = compare_three_way(p["npu"], p["cpu"], p["cpu_gold"], p["dtype"], config)
        records.append(
            dict(name=p["name"], reference=p["reference"], gating=p["gating"], **r)
        )
    return dict(
        policy_id=POLICY_ID,
        eligible_for_gate=True,
        segments=records,
        failed=[r["name"] for r in records if r["gating"] and r["status"] == "Failed"],
        warnings=[
            r["name"] for r in records if r["gating"] and r["status"] == "warning"
        ],
    )


def output_acceptance(name, whole_two, whole_three, branches, *, homogeneous=True):
    """先验收完整输出，再要求每个分支至少有一条通过路线。"""

    def choose(two, three):
        if any(record.get("passed") is True for record in two):
            return {"passed": True, "route": "two_way", "warning": False}
        accepted = [
            record
            for record in three
            if record.get("status") in ("Pass", "warning")
            and record.get("acceptance_definitive", True)
        ]
        if accepted:
            warning = all(record["status"] == "warning" for record in accepted)
            return {"passed": True, "route": "three_way", "warning": warning}
        return {"passed": False, "route": "none", "warning": False}

    whole = choose(whole_two, whole_three) if homogeneous else None
    branch_results = []
    if whole and whole["passed"]:
        passed, warning, route = True, whole["warning"], "whole_" + whole["route"]
    else:
        branch_results = [
            dict(name=branch["name"], **choose(branch["two_way"], branch["three_way"]))
            for branch in branches
        ]
        if not branch_results:
            raise ValueError("output acceptance requires explicit branch evidence")
        passed = all(record["passed"] for record in branch_results)
        warning = passed and any(record["warning"] for record in branch_results)
        route = "split" if homogeneous else "separate_dtypes"
    return dict(
        name=name + " acceptance",
        method=ACCEPTANCE_POLICY_ID,
        gating=True,
        passed=passed,
        status=(
            ("ACCEPTED_WITH_WARNING" if warning else "ACCEPTED") if passed else "FAILED"
        ),
        route=route,
        whole=whole,
        branches=branch_results,
    )


def apply_output_acceptance(metrics, three_way=None):
    records = [dict(record) for record in metrics]
    two = {record["name"]: record for record in records}
    segments = [] if three_way is None else three_way["segments"]
    three = {record["name"]: record for record in segments}
    if len(two) != len(records) or len(three) != len(segments):
        raise ValueError("comparison component names must be unique")
    for name in ("Q", "Cache"):
        mixed_cache = name == "Cache" and "Cache KV" in two
        pairs = (
            (("Cache KV", "Cache NoPE"), ("Cache KR", "Cache RoPE"))
            if mixed_cache
            else (
                (name + " NoPE/KV", name + " NoPE"),
                (name + " RoPE/KR", name + " RoPE"),
            )
        )
        branches = []
        for two_name, three_name in pairs:
            if two_name not in two:
                raise ValueError("missing branch comparison: " + two_name)
            branches.append(
                dict(
                    name=two_name,
                    two_way=[two[two_name]],
                    three_way=[three[three_name]] if three_name in three else [],
                )
            )
            two[two_name]["gating"] = False
        if not mixed_cache:
            if name not in two:
                raise ValueError("missing whole-output comparison: " + name)
            two[name]["gating"] = False
        records.append(
            output_acceptance(
                name,
                [two[name]] if name in two else [],
                [three[name]] if name in three else [],
                branches,
                homogeneous=not mixed_cache,
            )
        )
    return records


def golden_with_observations(function, args, kwargs):
    """在参考函数返回时读取局部值，不改变代码、全局变量或输出。

    HiF8 ULP 使用量化数值域中转换前的参考值，直接观察普通 Golden 已计算的结果，不从载荷反推。
    """
    observed = {}
    previous = sys.getprofile()

    def capture(frame, event, arg):
        if event == "return" and frame.f_code is function.__code__:
            for name in ("qa_scaled", "q_fp", "kv_fp", "kr_fp"):
                value = frame.f_locals.get(name)
                if isinstance(value, torch.Tensor):
                    observed[name] = value.detach().clone()
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(capture)
    try:
        outputs = function(*args, **kwargs)
    finally:
        sys.setprofile(previous)
    return outputs, observed


def _record(actual, expected, matched, name, method, allowed_failures, errors=None):
    actual, expected = actual.double(), expected.double()
    assert actual.shape == expected.shape, f"{name}: shape mismatch"
    total = actual.numel()
    failed = int((~matched).sum())
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    difference = (actual - expected).abs()
    classification_bad = (
        (torch.isnan(actual) != torch.isnan(expected))
        | (torch.isposinf(actual) != torch.isposinf(expected))
        | (torch.isneginf(actual) != torch.isneginf(expected))
    )
    # 分母使用包含非有限元素的完整分段；混合容差相对误差硬阈值遵循 method14/17。

    relative = difference / (
        torch.maximum(
            torch.maximum(actual.abs(), expected.abs()),
            torch.tensor((1 / 16384) / 0.005, dtype=torch.float64),
        )
        + 2e-9
    )
    bad_finite = (~matched) & finite
    max_failed_relative = relative[bad_finite].max().item() if bad_finite.any() else 0.0
    hard_pass = (
        method in ("hif8_method6_sat", "fp8_method10") or max_failed_relative < 10.0
    )
    result = {
        "name": name,
        "method": method,
        "elements": total,
        "failed_elements": failed,
        "allowed_failures": allowed_failures,
        "matched_ratio": (total - failed) / total if total else 1.0,
        "finite_elements": int(finite.sum()),
        "classification_mismatches": int(classification_bad.sum()),
        "max_failed_relative_error": max_failed_relative,
        "hard_cap_pass": hard_pass,
        "passed": failed <= allowed_failures and hard_pass,
        "empty_segment": total == 0,
    }
    bad_indices = first_failed_indices(~matched)
    result["first_failed_elements"] = [
        {
            "flat_index": i,
            "actual": actual.flatten()[i].item(),
            "expected": expected.flatten()[i].item(),
        }
        for i in bad_indices
    ]
    if errors is not None:
        finite_error = errors[torch.isfinite(errors)]
        result["max_abs_ulp"] = (
            finite_error.abs().max().item() if finite_error.numel() else 0.0
        )
        result["invalid_ulp_elements"] = int(torch.isnan(errors).sum())
    return result


def mixed(actual, expected, name, *, scale=False):
    return dict(_aligned_mixed(actual, expected, scale=scale), name=name)


def e8m0_int8(actual, expected, name="descale_qr E8M0"):
    """将 E8M0 字节视为有符号 INT8，扩宽后再执行 method1 减法。"""
    allowed = {torch.uint8, torch.int8, getattr(torch, "float8_e8m0fnu", None)}
    if actual.dtype not in allowed or expected.dtype not in allowed:
        raise ValueError(
            "E8M0 comparison requires native E8M0 or explicit byte carriers"
        )
    a = actual.contiguous().view(torch.int8).double()
    b = expected.contiguous().view(torch.int8).double()
    if a.shape != b.shape:
        raise ValueError("E8M0 byte shapes must match")
    matched = torch.isclose(a, b, atol=1.0, rtol=0.0, equal_nan=True)
    result = _record(a, b, matched, name, "e8m0_int8_method1", 0)
    result.update(
        gating=True,
        precision_method=1,
        atol=1.0,
        rtol=0.0,
        pct_thd=0.0,
        diff_thd=0.005,
        max_diff_thd=10.0,
        representation="signed_int8_raw_bytes",
        different_codes=int((a != b).sum()),
    )
    return result


def encoded(actual, expected, name):
    """按 FP8 method10 比较数值或已知格式的原始字节载体，pct_thd=0.005。

    保留百分比算术及 err_inx_2.any() 的行为：唯一距离大于 2 的元素在索引零时跳过硬检查。
    仅在适配副本上将 FP8 NaN 规范为正零，不改变调用方张量；带符号零仍计入距离大于 1 的比例。
    空槽位不比较；此方法不是浮点相对误差或位相等检查。
    """
    e5 = getattr(torch, "float8_e5m2", None)
    e4 = getattr(torch, "float8_e4m3fn", None)
    allowed = {torch.uint8, torch.int8, e4, e5}
    if actual.dtype not in allowed or expected.dtype not in allowed:
        raise ValueError("method10 requires FP8 or explicitly FP8 byte carriers")
    if actual.dtype in (e4, e5) and expected.dtype in (e4, e5):
        if actual.dtype != expected.dtype:
            raise ValueError("method10 FP8 formats must match")
    is_e5 = actual.dtype == e5 or expected.dtype == e5
    a = actual.contiguous().view(torch.uint8).to(torch.int16)
    b = expected.contiguous().view(torch.uint8).to(torch.int16)
    assert a.shape == b.shape, f"{name}: payload shape mismatch"
    # 原始字节调用方传入 E4M3FN 的 Q/QR/Cache 分段，不接受任意 uint8、HiF8 或 E8M0；E5M2 使用原生类型调用。

    nan_a = (a & 0x7F) > 0x7C if is_e5 else (a & 0x7F) == 0x7F
    nan_b = (b & 0x7F) > 0x7C if is_e5 else (b & 0x7F) == 0x7F
    a, b = torch.where(nan_a, 0, a), torch.where(nan_b, 0, b)
    linear_distance = (a - b).abs()
    distance = torch.minimum(linear_distance, 256 - linear_distance)
    signed_zero_pair = ((a == 0) & (b == 128)) | ((a == 128) & (b == 0))
    hard_bad = (distance > 2) & ~signed_zero_pair
    result = _record(a, b, distance <= 1, name, "fp8_method10", a.numel() // 200)
    # 保留 any(indices) 对索引零失败的行为，不物化全部索引；单独处理带符号零。

    over_two_count = int((distance > 2).sum())
    hard_check_entered = bool((distance.flatten()[1:] > 2).any())
    hard_count = int(hard_bad.sum())
    indices = first_failed_indices(hard_bad)
    fulfill_percent = (
        (float(a.numel() - result["failed_elements"]) / float(a.numel()) * 100.0)
        if a.numel()
        else None
    )
    result.update(
        pct_thd=0.005,
        code_distance_threshold=1,
        hard_code_distance_threshold=2,
        hard_failed_elements=hard_count,
        hard_cap_pass=not hard_check_entered or hard_count == 0,
        source_hard_check_entered=hard_check_entered,
        source_index_zero_hard_check_skipped=(
            over_two_count == 1 and not hard_check_entered
        ),
        fulfill_percent=fulfill_percent,
        required_fulfill_percent=(1 - 0.005) * 100.0,
        decision_policy="user_method10_source_compat_pct005",
        signed_zero_pairs=int(signed_zero_pair.sum()),
        nan_canonicalized_actual=int(nan_a.sum()),
        nan_canonicalized_reference=int(nan_b.sum()),
        max_circular_code_distance=int(distance.max()) if distance.numel() else 0,
        first_hard_failed_elements=[
            dict(
                flat_index=i,
                actual_code=int(a.flatten()[i]),
                reference_code=int(b.flatten()[i]),
                circular_distance=int(distance.flatten()[i]),
            )
            for i in indices
        ],
    )
    result["passed"] = (
        a.numel() == 0 or fulfill_percent >= (1 - 0.005) * 100.0
    ) and result["hard_cap_pass"]
    return result


def normalize_dynamic(value, scale):
    """数学参考中的零 scale 处理规则。"""
    return torch.where(
        scale == 0,
        torch.zeros_like(value),
        value / torch.where(scale == 0, torch.ones_like(scale), scale),
    )


def hif8(actual_decoded, reference_scaled, name, *, diff_thd=0.005):
    return dict(
        _aligned_hif8(actual_decoded, reference_scaled.float(), diff_thd=diff_thd),
        name=name,
    )


def _validate_q_dtype(actual, expected, quant_mode):
    """在选择比较数值域前检查 Q 的语义格式。

    Q 不像模式 1 Cache 那样包含混合 dtype；uint8 仅在显式声明 FP8 格式时作为同质载体。
    """
    formats = {
        0: {torch.bfloat16},
        1: {torch.bfloat16},
        2: {torch.float8_e4m3fn, torch.uint8},
        3: {torch.uint8},
    }
    if quant_mode not in formats or any(
        value.dtype not in formats[quant_mode] for value in (actual, expected)
    ):
        raise ValueError("Q dtype does not match the homogeneous public output format")


def compare_outputs(
    actual, expected, actual_cache, reference_cache, args, attrs, observed
):
    """比较全部公开分量，不在第一次数值失败时提前停止。"""
    results = []
    q, sq, qr, sqr = actual
    eq, esq, eqr, esqr = expected
    qm, wm = attrs["qkv_quant_mode"], attrs["w_qr_quant_mode"]
    _validate_q_dtype(q, eq, qm)
    dr = q.shape[-1] - 512

    def add_whole(a, b, operation, prefix):
        results.append(operation(a, b, prefix))
        for suffix, sl in (
            ("NoPE/KV", slice(None, 512)),
            ("RoPE/KR", slice(512, None)),
        ):
            record = operation(a[..., sl], b[..., sl], prefix + " " + suffix)
            record.update(
                gating=False,
                diagnostic_reason="branch fallback after both whole-output routes fail",
            )
            results.append(record)

    if qm in (0, 1):
        add_whole(q, eq, mixed, "Q")
    elif qm == 2:
        add_whole(q, eq, encoded, "Q")
    else:
        scaled = _hif8_decode_raw(eq)
        add_whole(_hif8_decode_raw(q), scaled, hif8, "Q")
    if qm in (2, 3):
        results.append(mixed(sq, esq, "descale_q", scale=True))
    if attrs["return_qr"]:
        if wm == 1:
            results.append(encoded(qr, eqr, "QR"))
        else:
            results.append(
                hif8(
                    _hif8_decode_raw(qr),
                    _hif8_decode_raw(eqr),
                    "QR",
                )
            )
            results.append(mixed(sqr, esqr, "descale_qr", scale=True))

    if wm == 1:
        # return_qr=False 时槽位为空，并由独立结构检查确认。
        results.append(e8m0_int8(sqr, esqr))

    capacity = actual_cache.shape[0] * actual_cache.shape[1]
    hits = args[12][args[12] >= 0]
    a = cache_to_logical(actual_cache, attrs["cache_mode"])
    b = cache_to_logical(reference_cache, attrs["cache_mode"])
    # CPU 的 FP8 高级索引并非普遍可用，改为索引原始字节视图。
    a = a.contiguous().view(torch.uint8).reshape(capacity, -1)[hits]
    b = b.contiguous().view(torch.uint8).reshape(capacity, -1)[hits]
    if qm == 0:
        add_whole(
            a.contiguous().view(torch.bfloat16),
            b.contiguous().view(torch.bfloat16),
            mixed,
            "Cache",
        )
    elif qm == 1:
        results.append(encoded(a[..., :512], b[..., :512], "Cache KV"))
        results.append(
            mixed(
                a[..., 512 : 512 + 2 * dr].contiguous().view(torch.bfloat16),
                b[..., 512 : 512 + 2 * dr].contiguous().view(torch.bfloat16),
                "Cache KR",
            )
        )
        results.append(
            mixed(
                a[..., 512 + 2 * dr :].contiguous().view(torch.float32),
                b[..., 512 + 2 * dr :].contiguous().view(torch.float32),
                "Cache descale",
                scale=True,
            )
        )
    elif qm == 2:
        add_whole(a, b, encoded, "Cache")
    else:
        # 比较双方均已完成 HiF8 输出舍入。
        add_whole(_hif8_decode_raw(a), _hif8_decode_raw(b), hif8, "Cache")
    return results


def aligned_three_way_outputs(
    actual, expected, actual_cache, reference_cache, args, attrs
):
    """分别用 D/C 自身 scale 重构数值，并与独立的 FP32 G 比较。"""
    mode, wm = attrs["qkv_quant_mode"], attrs["w_qr_quant_mode"]
    cache_mode = attrs["cache_mode"]
    dr = expected[0].shape[-1] - KV_LORA
    high_args = list(args)
    high_cache = torch.zeros((*args[11].shape[:3], KV_LORA + dr), dtype=torch.float32)
    high_args[11] = high_cache
    high, observed = golden_with_observations(
        quant_mla_prolog_golden, high_args, dict(attrs, action_type="bm_output_gold")
    )
    hits = args[12][args[12] >= 0]
    capacity = high_cache.shape[0] * high_cache.shape[1]

    def query_real(values):
        q, scale = values[:2]
        if mode < 2 or not q.numel():
            return q.float()
        decoded = _e4m3fn_decode(q) if mode == 2 else _hif8_decode_raw(q)
        return decoded * scale

    def qr_real(values):
        qr, scale = values[2:]
        if not qr.numel():
            return qr.float()
        return (
            _dequant_mxfp8_activation(qr, scale)
            if wm == 1
            else _hif8_decode_raw(qr) * scale
        )

    d_q, c_q = query_real(actual), query_real(expected)
    d_cache = cache_to_values(actual_cache, cache_mode, mode, args[13]).reshape(
        capacity, -1
    )[hits]
    c_cache = cache_to_values(reference_cache, cache_mode, mode, args[13]).reshape(
        capacity, -1
    )[hits]
    g_cache = cache_to_logical(high_cache, cache_mode).reshape(capacity, -1)[hits]
    if mode >= 2 and args[13].item() == 0:
        # 零乘数没有逆变换，比较精确编码的零输出。
        g_cache = torch.zeros_like(g_cache)
    parts = []

    def add(name, d, c, g, dtype="fp32", gating=True):
        parts.append(
            dict(
                name=name,
                npu=d,
                cpu=c,
                cpu_gold=g,
                dtype=dtype,
                reference="shared ordinary C / FP32 real-domain G",
                gating=gating,
            )
        )

    qdtype = "bf16" if mode < 2 else "fp32"
    add("Q", d_q, c_q, high[0], qdtype)
    for name, sl in (("NoPE", slice(None, 512)), ("RoPE", slice(512, None))):
        add("Q " + name, d_q[..., sl], c_q[..., sl], high[0][..., sl], qdtype, False)
        cache_dtype = "bf16" if mode == 0 or (mode == 1 and name == "RoPE") else "fp32"
        add(
            "Cache " + name,
            d_cache[..., sl],
            c_cache[..., sl],
            g_cache[..., sl],
            cache_dtype,
            False,
        )
    if mode != 1:
        add("Cache", d_cache, c_cache, g_cache, "bf16" if mode == 0 else "fp32")
    else:

        def cache_scales(cache):
            raw = _raw_u8(cache_to_logical(cache, cache_mode)).reshape(capacity, -1)[
                hits
            ]
            return raw[:, 512 + 2 * dr :].contiguous().view(torch.float32)

        valid = args[12] >= 0
        scale_g = (
            (
                observed["kv_fp"][valid]
                .reshape(-1, 4, 128)
                .abs()
                .amax(dim=-1)
                .clamp_min(CACHE_TILE_AMAX_FLOOR)
                / E4M3FN_QMAX
            )
            if hits.numel()
            else torch.empty((0, 4), dtype=torch.float32)
        )
        add(
            "Cache descale",
            cache_scales(actual_cache),
            cache_scales(reference_cache),
            scale_g,
        )
    if mode >= 2:
        add("descale_q", actual[1], expected[1], high[1])
    if attrs["return_qr"]:
        add("QR", qr_real(actual), qr_real(expected), high[2])
    if wm == 1:
        add(
            "descale_qr E8M0",
            _e8m0_decode(actual[3]),
            _e8m0_decode(expected[3]),
            high[3],
        )
    elif attrs["return_qr"]:
        add("descale_qr", actual[3], expected[3], high[3])
    return evaluate_segments(parts)


def compare_for_acceptance(
    actual,
    expected,
    actual_cache,
    reference_cache,
    args,
    attrs,
    observed,
):
    metrics = compare_outputs(
        actual, expected, actual_cache, reference_cache, args, attrs, observed
    )
    # 两条路线均生成诊断结果，验收仍优先使用双方比较结论。
    three_way = aligned_three_way_outputs(
        actual, expected, actual_cache, reference_cache, args, attrs
    )
    decisions = apply_output_acceptance(metrics, three_way)
    three = {record["name"]: record for record in three_way["segments"]}
    for record in list(decisions):
        name = record["name"]
        if name in (
            "QR",
            "descale_q",
            "descale_qr",
            "descale_qr E8M0",
            "Cache descale",
        ):
            record["gating"] = False
            evidence = dict(
                name=name,
                two_way=[record],
                three_way=[three[name]] if name in three else [],
            )
            decisions.append(
                output_acceptance(
                    name, evidence["two_way"], evidence["three_way"], [evidence]
                )
            )
    decisions.append(
        dict(name="Aligned high Golden diagnostics", gating=False, three_way=three_way)
    )
    return decisions
