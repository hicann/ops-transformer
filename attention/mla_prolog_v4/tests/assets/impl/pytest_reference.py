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
"""四种输出模式的独立 CPU Golden 和精度判定契约。

普通输出保留公开 dtype，高精度输出保留 FP32 实数。
三方比较使用独立副本并接受约定的 warning 结果；完整输出比较失败后才进行分支回退。
"""

from __future__ import annotations


import ast


from enum import Enum


from functools import partial


import hashlib


import json


import math


import os


from pathlib import Path


import sys


from types import SimpleNamespace


from unittest.mock import patch


import torch


_KV_LORA = 512


_GROUP_SIZE = 128


_FP8_MAX = 448.0


_HIF8_MAX = 32768.0


_GROUP_AMAX_FLOOR = 1.0e-4


class CacheMode(Enum):
    """公开的符号枚举，不暴露整数编码。"""

    PA_BSND = "PA_BSND"
    PA_NZ = "PA_NZ"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor(name, value, dtype=None, shape=None):
    _require(isinstance(value, torch.Tensor), f"{name} must be a Tensor")
    _require(value.device.type == "cpu", f"{name} must be on CPU for the golden")
    _require(value.is_contiguous(), f"{name} must be contiguous")
    if dtype is not None:
        _require(
            value.dtype == dtype, f"{name}.dtype must be {dtype}, got {value.dtype}"
        )
    if shape is not None:
        _require(
            tuple(value.shape) == tuple(shape),
            f"{name}.shape must be {shape}, got {tuple(value.shape)}",
        )


def _finite_scalar(name, value, positive=False):
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be a real scalar",
    )
    _require(math.isfinite(value), f"{name} must be finite")
    scalar = torch.tensor(value, dtype=torch.float32).item()
    _require(math.isfinite(scalar), f"{name} must be finite in FP32")
    if positive:
        _require(scalar > 0.0, f"{name} must be positive in FP32")
    return scalar


def _scale(value, scalar):
    return value * scalar


def pack_weight_nz(weight):
    """构造 CPU 测试面板，不创建真正的设备 NZ 张量。"""
    _tensor("weight", weight, torch.bfloat16)
    _require(weight.ndim == 2, "weight must be a logical 2D matrix")
    out_features, in_features = weight.shape
    _require(
        out_features % 16 == 0 and in_features % 16 == 0,
        "NZ fixture dimensions must be multiples of 16",
    )
    return (
        weight.reshape(out_features // 16, 16, in_features // 16, 16)
        .permute(0, 2, 3, 1)
        .contiguous()
    )


def _weight_logical(weight):
    if weight.ndim == 2:
        return weight
    _require(
        weight.ndim == 4 and tuple(weight.shape[-2:]) == (16, 16),
        "CPU NZ fixture must have shape (out/16,in/16,16,16)",
    )
    return (
        weight.permute(0, 3, 1, 2)
        .contiguous()
        .reshape(weight.shape[0] * 16, weight.shape[1] * 16)
    )


def _rms_norm(value, gamma, epsilon):
    """独立于硬件归约顺序的 FP32 RMSNorm 数学参考。"""
    value = value.float()
    _require(value.shape[-1] > 0, "RMSNorm width must be positive")
    mean_square = value.square().mean(dim=-1, keepdim=True)
    return value / torch.sqrt(mean_square + epsilon) * gamma.float()


def _rope(value, sin, cos, mode):
    if mode == 0 or value.shape[-1] == 0:
        return value
    sin, cos = sin.float(), cos.float()
    if value.ndim == 3:
        sin, cos = sin[:, None, :], cos[:, None, :]
    half = value.shape[-1] // 2
    if mode == 1:
        even, odd = value[..., 0::2], value[..., 1::2]
        # 交错半旋转对偶数、奇数配对，并输出两个连续的半区。
        lower = even * cos[..., :half] - odd * sin[..., :half]
        upper = odd * cos[..., half:] + even * sin[..., half:]
        return torch.cat((lower, upper), dim=-1)
    first, second = value[..., :half], value[..., half:]
    rotated = torch.cat((-second, first), dim=-1)
    return value * cos + rotated * sin


def _fp8(value):
    # E4M3FN 使用最近偶数舍入、有限饱和及 NaN 转正零。
    value = value.float().nan_to_num(nan=0.0, posinf=_FP8_MAX, neginf=-_FP8_MAX)
    return value.clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)


def _decode_hif8_values(code):
    """按公开格式解析 HiF8 的符号、dot、指数及尾数。"""
    raw = code.to(torch.int64)
    magnitude = raw & 127
    result = torch.zeros_like(raw, dtype=torch.float32)
    for lo, hi, mant_bits, exponent_bits, base in (
        (16, 32, 3, 1, 1),
        (32, 64, 3, 2, 2),
        (64, 96, 2, 3, 4),
        (96, 128, 1, 4, 8),
    ):
        exponent_field = (magnitude >> mant_bits) & ((1 << exponent_bits) - 1)
        exponent_abs = base + (exponent_field & (base - 1))
        exponent = torch.where(
            (exponent_field & base) != 0, -exponent_abs, exponent_abs
        )
        mantissa = (magnitude & ((1 << mant_bits) - 1)).float()
        normal = torch.ldexp(1.0 + mantissa / (1 << mant_bits), exponent.int())
        result = torch.where((magnitude >= lo) & (magnitude < hi), normal, result)
    result = torch.where(
        (magnitude >= 8) & (magnitude < 16), 1.0 + (magnitude & 7).float() / 8.0, result
    )
    denormal = torch.ldexp(torch.ones_like(result), magnitude.int() - 23)
    result = torch.where((magnitude > 0) & (magnitude < 8), denormal, result)
    result = torch.where((raw & 128) != 0, -result, result)
    result = torch.where(
        magnitude == 111,
        torch.copysign(torch.full_like(result, float("inf")), result),
        result,
    )
    return torch.where(raw == 128, torch.full_like(result, float("nan")), result)


def decode_hif8(tensor):
    """将 CPU uint8 HiF8 位模式解码为 FP32，不应用 descale。"""
    _tensor("HiF8 bytes", tensor, torch.uint8)
    return _decode_hif8_values(tensor)


def _hif8(value):
    # HiF8 使用最近值舍入，半值远离零，并进行有限饱和及 NaN 转零。
    value = value.float()
    codes = torch.arange(128, dtype=torch.int64)
    codes = codes[codes != 111]  # 正无穷不是饱和目标值。
    levels = _decode_hif8_values(codes)
    order = levels.argsort()
    levels, codes = levels[order], codes[order]
    absolute = value.abs().nan_to_num(nan=0.0, posinf=_HIF8_MAX).clamp(max=_HIF8_MAX)
    upper = torch.searchsorted(levels, absolute.contiguous()).clamp(max=len(levels) - 1)
    lower = (upper - 1).clamp(min=0)
    choose_upper = (absolute - levels[lower]) >= (levels[upper] - absolute)
    selected = torch.where(choose_upper, codes[upper], codes[lower])
    signed = selected | (torch.signbit(value).to(torch.int64) << 7)
    signed = torch.where(selected == 0, 0, signed)  # 只使用一种零编码。
    return signed.to(torch.uint8).contiguous()


def _dynamic_descale(maximum, limit):
    return maximum.float() / limit


def _dynamic_q(value, mode, *, high_precision=False):
    # 统计使用公开输出转换前的 FP32 数学 Query。
    value = value.float()
    if mode in (0, 1):
        return (value if high_precision else value.to(torch.bfloat16)), torch.empty(
            0, dtype=torch.float32
        )
    if mode == 2:
        maximum = value.abs().amax(dim=-1, keepdim=True)
        descale = _dynamic_descale(maximum, _FP8_MAX)
    else:
        maximum = value.abs().amax().reshape(1)
        descale = _dynamic_descale(maximum, _HIF8_MAX)
    if high_precision:
        return value, descale
    # 只保护实际零输入，不将非零 amax 的 scale 下溢视为零；内部使用安全分母，返回的 descale 仍为精确零。

    denominator = torch.where(maximum == 0, torch.ones_like(descale), descale)
    normalized = value / denominator
    normalized = torch.where(maximum == 0, torch.zeros_like(normalized), normalized)
    return (_fp8(normalized) if mode == 2 else _hif8(normalized)), descale


def _cache_payload(ckv, kr, mode, quant_scale_kv, *, high_precision=False):
    if high_precision:
        return torch.cat((ckv, kr), dim=-1).float()
    if mode == 0:
        return torch.cat((ckv, kr), dim=-1).to(torch.bfloat16)
    if mode == 1:
        groups = ckv.reshape(ckv.shape[0], _KV_LORA // _GROUP_SIZE, _GROUP_SIZE)
        descales = _dynamic_descale(
            groups.abs().amax(dim=-1).clamp_min(_GROUP_AMAX_FLOOR), _FP8_MAX
        )
        quantized = _fp8(groups / descales[..., None]).reshape(ckv.shape[0], _KV_LORA)
        kr_bytes = kr.to(torch.bfloat16).contiguous().view(torch.uint8)
        scale_bytes = descales.contiguous().view(torch.uint8)
        return torch.cat((quantized.view(torch.uint8), kr_bytes, scale_bytes), dim=-1)
    scaled = torch.cat((ckv, kr), dim=-1) * quant_scale_kv.item()
    return _fp8(scaled) if mode == 2 else _hif8(scaled)


def _write_cache(cache, index, payload, mode):
    # 完整输出和载荷计算完成后，才提交 Cache 写入。
    bytes_per_row = cache.shape[-1] * cache.element_size()
    raw_cache = cache.view(torch.uint8).reshape(-1)
    raw_rows = payload.contiguous().view(torch.uint8).reshape(len(index), bytes_per_row)
    if mode == CacheMode.PA_BSND:
        rows = raw_cache.reshape(-1, bytes_per_row)
        for token, slot in enumerate(index.tolist()):
            if slot < 0:
                continue
            rows[slot].copy_(raw_rows[token])
    else:
        _require(bytes_per_row % 32 == 0, "PA_NZ payload must be 32-byte aligned")
        physical = raw_cache.reshape(
            cache.shape[0], bytes_per_row // 32, cache.shape[1], 32
        )
        for token, slot in enumerate(index.tolist()):
            if slot < 0:
                continue
            block, row = divmod(slot, cache.shape[1])
            physical[block, :, row, :].copy_(raw_rows[token].reshape(-1, 32))


def cache_to_logical(cache, cache_mode):
    """解析 PA 地址布局，混合载荷保持为原始字节。"""
    _tensor("cache", cache)
    _require(isinstance(cache_mode, CacheMode), "cache_mode must be CacheMode")
    _require(
        cache.ndim == 4
        and cache.shape[0] > 0
        and cache.shape[2] == 1
        and 16 <= cache.shape[1] <= 1024
        and cache.shape[1] % 16 == 0,
        "cache shape must be (blocks,block_size,1,width), block_size=16..1024 divisible by 16",
    )
    if cache_mode == CacheMode.PA_BSND:
        return cache
    row_bytes = cache.shape[-1] * cache.element_size()
    _require(row_bytes % 32 == 0, "PA_NZ row must be 32-byte aligned")
    raw = (
        cache.view(torch.uint8)
        .reshape(cache.shape[0], row_bytes // 32, cache.shape[1], 32)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return raw.reshape(-1).view(cache.dtype).reshape(cache.shape)


def high_precision_projection(x, weight, *, rows_per_chunk=8192):
    """对完整 K 维使用 FP64 累加，再输出 FP32。

    仅对相互独立的 token 行分批，限制扩展精度输入副本的大小。
    不拆分 K 维，不引入中间 BF16 舍入，也不模拟设备分块。
    此路径仅用于高精度 KVA 参考，普通 Golden 路径不变。
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


@torch.no_grad()
def golden_mla_prolog(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    quant_scale_kv=None,
    rope_sin=None,
    rope_cos=None,
    *,
    norm_eps=1e-5,
    qkv_quant_mode=0,
    cache_mode=CacheMode.PA_BSND,
    qr_scale=1.0,
    kvr_scale=1.0,
    rope_mode=1,
    return_qr=False,
    action_type="bm_output",
):
    """返回 (q, descale_q, qr)，并原地更新 kv_cache。

    计算输入均为 CPU 张量；设备测试使用独立 NPU 副本。
    bm_output 保留中间 BF16 舍入；bm_output_gold 保留 FP32 中间值，
    KVA 使用 FP64 累加，并返回未量化的 FP32 Q/QR。
    高精度 Cache 必须为 FP32，逻辑宽度为 512+Dr，混合载荷模式 1 也遵循此约定。
    """
    _require(
        action_type in ("bm_output", "bm_output_gold"),
        "action_type must be bm_output or bm_output_gold",
    )
    _require(isinstance(cache_mode, CacheMode), "cache_mode must be a CacheMode enum")
    _require(
        type(qkv_quant_mode) is int and qkv_quant_mode in (0, 1, 2, 3),
        "qkv_quant_mode must be 0/1/2/3",
    )
    _require(
        type(rope_mode) is int and rope_mode in (0, 1, 2), "rope_mode must be 0/1/2"
    )
    _require(type(return_qr) is bool, "return_qr must be bool")
    _require(
        not (qkv_quant_mode == 1 and cache_mode == CacheMode.PA_NZ),
        "mode 1 supports PA_BSND only",
    )
    norm_eps = _finite_scalar("norm_eps", norm_eps, positive=True)
    qr_scale = _finite_scalar("qr_scale", qr_scale)
    kvr_scale = _finite_scalar("kvr_scale", kvr_scale)
    for name, tensor in (
        ("x", x),
        ("wqa", wqa),
        ("wqb", wqb),
        ("wkva", wkva),
        ("wkb", wkb),
    ):
        _tensor(name, tensor, torch.bfloat16)
    _require(x.ndim == 2, "x must have shape (T,dim)")
    _require(
        wqa.ndim == wqb.ndim == wkva.ndim and wqa.ndim in (2, 4),
        "wqa/wqb/wkva must all be ND matrices or all explicit CPU NZ panels",
    )
    qa, qb, kva = map(_weight_logical, (wqa, wqb, wkva))
    t, dim = x.shape
    q_lora = qa.shape[0]
    _require(
        1024 <= dim <= 8192 and dim % 256 == 0, "dim must be 1024..8192, 256 aligned"
    )
    _require(q_lora in (640, 1024, 1536, 2048), "unsupported q_lora")
    _require(wkb.ndim == 3, "wkb must have shape (N,Dn,512)")
    n, dn, kv_lora = wkb.shape
    dr = kva.shape[0] - _KV_LORA
    _require(1 <= n <= 128 and kv_lora == _KV_LORA, "N must be 1..128 and kv_lora=512")
    _require((dn, dr) in ((128, 64), (192, 64), (256, 0)), "unsupported (Dn,Dr)")
    _tensor("wqa logical", qa, shape=(q_lora, dim))
    _tensor("wqb logical", qb, shape=(n * (dn + dr), q_lora))
    _tensor("wkva logical", kva, shape=(_KV_LORA + dr, dim))
    _tensor("norm_weight_qa", norm_weight_qa, torch.float32, (q_lora,))
    _tensor("norm_weight_kva", norm_weight_kva, torch.float32, (_KV_LORA,))
    _tensor("cache_index", cache_index, torch.int64, (t,))
    _tensor("kv_cache", kv_cache)
    _require(
        kv_cache.ndim == 4 and kv_cache.shape[0] > 0,
        "kv_cache must have shape (positive_blocks,block_size,1,width)",
    )
    high_precision = action_type == "bm_output_gold"
    width = (
        _KV_LORA + 2 * dr + 16
        if qkv_quant_mode == 1 and not high_precision
        else _KV_LORA + dr
    )
    _require(
        16 <= kv_cache.shape[1] <= 1024
        and kv_cache.shape[1] % 16 == 0
        and tuple(kv_cache.shape[2:]) == (1, width),
        f"kv_cache trailing shape must be (block_size,1,{width}), block_size=16..1024 divisible by 16",
    )
    cache_dtype = (
        torch.bfloat16
        if qkv_quant_mode == 0
        else torch.uint8
        if qkv_quant_mode == 3
        else torch.float8_e4m3fn
    )
    if high_precision:
        cache_dtype = torch.float32
    qr_dtype = torch.float32 if high_precision else torch.bfloat16
    _require(
        kv_cache.dtype == cache_dtype,
        f"CPU cache dtype for mode {qkv_quant_mode} must be {cache_dtype}",
    )
    _require(dr != 0 or rope_mode == 0, "Dr=0 forces rope_mode=0")
    if dr and rope_mode:
        _tensor("rope_sin", rope_sin, torch.bfloat16, (t, dr))
        _tensor("rope_cos", rope_cos, torch.bfloat16, (t, dr))
    if qkv_quant_mode in (2, 3):
        _tensor("quant_scale_kv", quant_scale_kv, torch.float32, (1,))
    else:
        _require(quant_scale_kv is None, "mode 0/1 does not take quant_scale_kv")
    if t == 0:
        q_dtype = (
            torch.float32
            if high_precision
            else (cache_dtype if qkv_quant_mode != 1 else torch.bfloat16)
        )
        return (
            torch.empty((0, n, _KV_LORA + dr), dtype=q_dtype),
            torch.empty(0, dtype=torch.float32),
            torch.empty((0, q_lora) if return_qr else (0,), dtype=qr_dtype),
        )
    write_index = cache_index[cache_index >= 0]
    _require(
        torch.unique(write_index).numel() == write_index.numel(),
        "repeated nonnegative cache_index is not allowed",
    )
    _require(
        bool((write_index < kv_cache.shape[0] * kv_cache.shape[1]).all()),
        "nonnegative cache_index is outside the cache capacity",
    )
    if quant_scale_kv is not None:
        _require(
            bool(torch.isfinite(quant_scale_kv).all()),
            "quant_scale_kv must be finite",
        )
    for name, tensor in (
        ("x", x),
        ("wqa", qa),
        ("wqb", qb),
        ("wkva", kva),
        ("wkb", wkb),
        ("norm_weight_qa", norm_weight_qa),
        ("norm_weight_kva", norm_weight_kva),
    ):
        _require(bool(torch.isfinite(tensor).all()), f"{name} contains NaN/Inf")
    if dr and rope_mode:
        _require(
            bool(torch.isfinite(rope_sin).all())
            and bool(torch.isfinite(rope_cos).all()),
            "active rope_sin/rope_cos contains NaN/Inf",
        )
    _require(
        sys.byteorder == "little", "packed Cache golden requires little-endian CPU"
    )
    if action_type == "bm_output_gold":
        cq_raw = x.float() @ qa.float().T
    else:
        cq_raw = (x.float() @ qa.float().T).to(torch.bfloat16).float()
    if action_type == "bm_output_gold":
        cq = _scale(_rms_norm(cq_raw, norm_weight_qa, norm_eps), qr_scale)
    else:
        cq = _scale(_rms_norm(cq_raw, norm_weight_qa, norm_eps), qr_scale).to(
            torch.bfloat16
        )
    if action_type == "bm_output_gold":
        q_projected = (cq @ qb.float().T).reshape(t, n, dn + dr)
    else:
        q_projected = (
            (cq.float() @ qb.float().T)
            .to(torch.bfloat16)
            .float()
            .reshape(t, n, dn + dr)
        )
    q_nope = torch.einsum("tnd,ndk->tnk", q_projected[..., :dn], wkb.float())
    q_rope = _rope(q_projected[..., dn:], rope_sin, rope_cos, rope_mode)
    q, descale_q = _dynamic_q(
        torch.cat((q_nope, q_rope), dim=-1),
        qkv_quant_mode,
        high_precision=high_precision,
    )
    if action_type == "bm_output_gold":
        kva_raw = high_precision_projection(x, kva)
    else:
        kva_raw = (x.float() @ kva.float().T).to(torch.bfloat16).float()
    ckv = _scale(_rms_norm(kva_raw[:, :_KV_LORA], norm_weight_kva, norm_eps), kvr_scale)
    kr = _rope(kva_raw[:, _KV_LORA:], rope_sin, rope_cos, rope_mode)
    payload = _cache_payload(
        ckv, kr, qkv_quant_mode, quant_scale_kv, high_precision=high_precision
    )
    qr = cq.to(qr_dtype) if return_qr else torch.empty(0, dtype=qr_dtype)
    _write_cache(kv_cache, cache_index, payload, cache_mode)
    return q, descale_q, qr


_REFERENCE_HELPERS = dict(
    CacheMode=CacheMode,
    _rms_norm=_rms_norm,
    _rope=_rope,
    _dynamic_q=_dynamic_q,
    _cache_payload=_cache_payload,
    decode_hif8=decode_hif8,
    cache_to_logical=cache_to_logical,
)


golden = SimpleNamespace(mla_prolog=golden_mla_prolog, **_REFERENCE_HELPERS)


math_reference = SimpleNamespace(
    mla_prolog=partial(golden_mla_prolog, action_type="bm_output_gold"),
    **_REFERENCE_HELPERS,
)


def ordinary_cpu(golden, inputs, attrs):
    """使用与双方比较相同的普通 Golden 的兼容适配。"""
    copied = dict(inputs, kv_cache=inputs["kv_cache"].clone())
    output, captured = golden_with_reference(golden, copied, attrs)
    payload = (
        golden._cache_payload(
            *captured["cache_fp"], attrs["qkv_quant_mode"], inputs["quant_scale_kv"]
        )
        if len(inputs["x"])
        else torch.empty(
            (0, inputs["kv_cache"].shape[-1]), dtype=inputs["kv_cache"].dtype
        )
    )
    return output, payload


DEFAULT_CONFIG = dict(
    small_value=2.0**-10,
    small_value_atol=2.0**-16,
    max_re_rtol=10.0,
    avg_re_rtol=2.0,
    rmse_rtol=2.0,
)


def error_metrics(value, golden, config):
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


def compare_three_way(npu, cpu, cpu_gold, output_dtype, config=None):
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
    d, b = error_metrics(npu, cpu_gold, cfg), error_metrics(cpu, cpu_gold, cfg)
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


POLICY = "20261103_mla_prolog_supplied_comparison"


ADAPTATION = "real_domain_three_way_independent_copies_warning_accepted"


BM_CMP_STD = {
    name: dict(
        max_re_rtol=10.0,
        avg_re_rtol=2.0,
        rmse_rtol=2.0,
        small_value=0.0009765625,
        small_value_atol=1.52587890625e-05,
    )
    for name in ("fp32", "fp16", "bf16")
}


def _pair(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("comparison requires identical shapes")
    if actual.device.type != "cpu" or expected.device.type != "cpu":
        raise ValueError("comparison requires independent CPU snapshots")


def _record(method, elements, failed, allowed, passed, **details):
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


def _first_failed_elements(actual, expected, bad, *, limit=8, **details):
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


def e8m0_int8(actual, expected):
    """将 E8M0 存储视为有符号 INT8 编码，异常点配额为零。"""
    _pair(actual, expected)
    allowed = {torch.uint8, torch.int8}
    native = getattr(torch, "float8_e8m0fnu", None)
    if native is not None:
        allowed.add(native)
    if actual.dtype not in allowed or expected.dtype not in allowed:
        raise TypeError("E8M0 requires native dtype or explicit byte carriers")
    a = actual.contiguous().view(torch.int8).flatten().float()
    b = expected.contiguous().view(torch.int8).flatten().float()
    bad = ~torch.isclose(a, b, rtol=0.0, atol=1.0)
    relative = (a - b).abs() / (
        torch.maximum(a.abs(), b.abs()).clamp_min((1 / 16384) / 0.005) + 2e-9
    )
    hard = float(relative[bad].max()) if bad.any() else 0.0
    failed = int(bad.sum())
    return _record(
        1,
        a.numel(),
        failed,
        0,
        failed == 0 and hard < 10.0,
        dtype_rule="E8M0 signed INT8 codes",
        rtol=0.0,
        atol=1.0,
        diff_thd=0.005,
        pct_thd=0.0,
        max_diff_thd=10.0,
        max_bad_relative=hard,
        first_failed_elements=_first_failed_elements(
            a, b, bad, relative_error=relative
        ),
    )


def mixed(actual, expected, *, scale=False):
    """对 BF16/FP32 scale 执行 isclose、分段配额和相对误差硬阈值检查。"""
    _pair(actual, expected)
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
    return _record(
        1,
        m,
        failed,
        allowed,
        failed * 200 <= m
        and hard < 10
        and bool(torch.isfinite(relative[finite_bad]).all()),
        first_failed_elements=_first_failed_elements(
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


def e4m3fn(actual, expected):
    """执行 method10 和 0.5% 配额，保留索引零处的 any() 判定行为。"""
    _pair(actual, expected)
    if actual.dtype != torch.float8_e4m3fn or expected.dtype != torch.float8_e4m3fn:
        raise TypeError("method10 requires E4M3FN typed payloads")
    a = actual.contiguous().view(torch.uint8).flatten().to(torch.int16)
    b = expected.contiguous().view(torch.uint8).flatten().to(torch.int16)
    a = torch.where((a & 127) == 127, 0, a)
    b = torch.where((b & 127) == 127, 0, b)
    delta = (a - b).abs()
    distance = torch.minimum(delta, 256 - delta)
    signed_zero = ((a == 0) & (b == 128)) | ((a == 128) & (b == 0))
    failed = int((distance > 1).sum())
    # 保留 any(indices) 语义；唯一失败索引为零时跳过该检查。
    entered = bool((distance.flatten()[1:] > 2).any())
    hard = int(((distance > 2) & ~signed_zero).sum())
    percent = (
        float(a.numel() - failed) / float(a.numel()) * 100.0 if a.numel() else 100.0
    )
    return _record(
        10,
        a.numel(),
        failed,
        a.numel() // 200,
        percent >= (1 - 0.005) * 100.0 and (not entered or hard == 0),
        pct_thd=0.005,
        first_failed_elements=_first_failed_elements(
            actual.float(),
            expected.float(),
            distance > 1,
            actual_code=actual.contiguous().view(torch.uint8),
            expected_code=expected.contiguous().view(torch.uint8),
            code_distance=distance,
        ),
        first_hard_failed_elements=_first_failed_elements(
            actual.float(),
            expected.float(),
            (distance > 2) & ~signed_zero,
            code_distance=distance,
        ),
        hard_failures=hard,
        source_hard_check_entered=entered,
        fulfill_percent=percent,
        signed_zero_pairs=int(signed_zero.sum()),
    )


def hif8(actual_decoded, reference_fp32, *, diff_thd=0.005):
    """按 method6 检查溢出到 Inf、特殊值匹配和通过比例。"""
    if not 0 <= diff_thd <= 1:
        raise ValueError("diff_thd must be in [0, 1]")
    _pair(actual_decoded, reference_fp32)
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
    return _record(
        6,
        m,
        failed,
        m // 1000 + 1 if m > 100 else int(m * diff_thd),
        percent >= float(1 - effective) / float(1) * 100.0,
        first_failed_elements=_first_failed_elements(h, x, bad, ulp=ulp),
        effective_fail_fraction=effective,
        fulfill_percent=percent,
        max_error=max(errors) - 1.0 if errors else 0.0,
        reference_domain="decoded_ordinary_hif8_overflow_to_inf",
    )


def golden_with_reference(golden, inputs, attrs):
    """观察 CPU Golden 调用参数，不从载荷字节反推转换前的数值。

    sys.setprofile 仅作用于当前线程，并在 finally 中恢复。
    不替换函数，不修改中间值；空输入可以没有采集结果。
    """
    captured = {}
    targets = {
        golden._dynamic_q.__code__: ("q_fp", ("value",)),
        golden._cache_payload.__code__: ("cache_fp", ("ckv", "kr")),
    }

    def observe(frame, event, arg):
        if event == "call" and frame.f_code in targets:
            name, names = targets[frame.f_code]
            if name in captured:
                raise RuntimeError("ambiguous repeated Golden reference capture")
            captured[name] = tuple(
                frame.f_locals[key].detach().clone() for key in names
            )

    previous = sys.getprofile()
    if previous is not None:
        raise RuntimeError("Golden capture requires no active Python profiler")
    try:
        sys.setprofile(observe)
        output = golden.mla_prolog(**inputs, **attrs)
    finally:
        sys.setprofile(previous)
    if inputs["x"].shape[0] and set(captured) != {"q_fp", "cache_fp"}:
        raise RuntimeError("missing pre-quantization Golden reference")
    return output, captured


def compare_numeric_outputs(
    golden,
    actual,
    expected,
    cache_actual,
    cache_expected,
    inputs,
    attrs,
    captured,
    *,
    split_diagnostics=False,
):
    """对每个同质输出单独应用配额，混合 Cache 的字段分别处理。

    split_diagnostics 只公开分支诊断，不改变验收条件。
    """
    mode = int(attrs["qkv_quant_mode"])
    layout = golden.CacheMode[attrs["cache_mode"].name]
    dr = inputs["wkva"].shape[0] - 512
    results = {}
    q_sections = (
        (("q_nope", slice(0, 512)), ("q_rope", slice(512, None)))
        if split_diagnostics
        else (("q", slice(None)),)
    )
    for label, section in q_sections:
        a, b = actual[0][..., section], expected[0][..., section]
        results[label] = (
            mixed(a, b)
            if mode < 2
            else e4m3fn(a, b)
            if mode == 2
            else hif8(
                golden.decode_hif8(a.contiguous()), golden.decode_hif8(b.contiguous())
            )
        )
    results["descale_q"] = mixed(actual[1], expected[1], scale=True)
    results["qr"] = mixed(actual[2], expected[2])
    valid = inputs["cache_index"] >= 0
    hits = inputs["cache_index"][valid]
    width = cache_actual.shape[-1]
    a = golden.cache_to_logical(cache_actual, layout).reshape(-1, width)[hits]
    b = golden.cache_to_logical(cache_expected, layout).reshape(-1, width)[hits]
    if mode == 1:
        a, b = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        results["kv"] = e4m3fn(
            a[:, :512].contiguous().view(torch.float8_e4m3fn),
            b[:, :512].contiguous().view(torch.float8_e4m3fn),
        )
        results["kr"] = mixed(
            a[:, 512 : 512 + 2 * dr].contiguous().view(torch.bfloat16),
            b[:, 512 : 512 + 2 * dr].contiguous().view(torch.bfloat16),
        )
        sa = a[:, 512 + 2 * dr :].contiguous().view(torch.float32)
        sb = b[:, 512 + 2 * dr :].contiguous().view(torch.float32)
        # 四个分组 scale 作为一个独立 FP32 scale 分段处理。
        results["cache_descale"] = mixed(sa, sb, scale=True)
    else:
        cache_sections = (
            (("kv", slice(0, 512)), ("kr", slice(512, None)))
            if split_diagnostics
            else (("cache", slice(None)),)
        )
        for label, section in cache_sections:
            av, bv = a[..., section], b[..., section]
            results[label] = (
                mixed(av, bv)
                if mode == 0
                else e4m3fn(av, bv)
                if mode == 2
                else hif8(
                    golden.decode_hif8(av.contiguous()),
                    golden.decode_hif8(bv.contiguous()),
                )
            )
    return results


def precision_status(value):
    """比较记录缺失或格式不正确时，不得静默判为成功。"""
    records = []

    def visit(item):
        if isinstance(item, dict):
            if "method" in item and "passed" in item:
                records.append(item)
            else:
                for child in item.values():
                    visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    if not records:
        return "UNVERIFIED"
    if any(not record["passed"] for record in records):
        return "FAIL"
    return "PASS" if any(record["elements"] for record in records) else "EMPTY"


def three_way(
    device,
    ordinary_cpu,
    cpu_gold,
    dtype,
    *,
    golden_inf_switch=True,
    inf_clip_switch=True,
    nan_toZero_switch=True,
):
    """在独立副本上执行三方比较规则，并启用全部特殊值开关。"""
    _pair(device, cpu_gold)
    _pair(ordinary_cpu, cpu_gold)
    if not (golden_inf_switch and inf_clip_switch and nan_toZero_switch):
        raise ValueError("all three special-value switches must be enabled")
    dtype_name = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
        torch.float64: "fp32",
        torch.float8_e4m3fn: "float8_e4m3fn",
    }.get(dtype)
    if dtype_name is None:
        raise ValueError(
            "unsupported three-way logical dtype; byte carrier is not a dtype policy"
        )
    result = compare_three_way(device, ordinary_cpu, cpu_gold, dtype_name)
    return dict(
        method=12,
        diagnostic_only=False,
        **result,
        configuration=result["config"],
        entry="supplied/compare_three_way_independent_copies",
        input_dtypes=dict(
            device=str(device.dtype),
            cpu=str(ordinary_cpu.dtype),
            gold=str(cpu_gold.dtype),
        ),
    )


def acceptance_status(direct, fallback, *, structural_pass=True):
    """先按完整输出回退，不对逐个元素选择比较路线。"""
    if not structural_pass:
        return "FAIL"
    if direct in ("PASS", "EMPTY"):
        return "PASS"
    return "PASS" if direct == "FAIL" and fallback in ("Pass", "warning") else "FAIL"


policy = SimpleNamespace(
    _pair=_pair,
    _record=_record,
    mixed=mixed,
    e4m3fn=e4m3fn,
    hif8=hif8,
    golden_with_reference=golden_with_reference,
    compare_outputs=compare_numeric_outputs,
    precision_status=precision_status,
    three_way=three_way,
    acceptance_status=acceptance_status,
)


def load_math_reference(golden):
    """返回生成未量化 FP32 输出的统一数学参考。"""
    verify_reference_integrity()
    return math_reference


def fallback_segments(golden, inputs, attrs, actual, after, expected, captured):
    """D/C 分别使用自身 scale，G 使用相同 BF16 输入的未量化 FP32 参考。"""
    high = load_math_reference(golden)
    mode = int(attrs["qkv_quant_mode"])
    layout = golden.CacheMode[attrs["cache_mode"].name]
    dr = expected[0].shape[-1] - 512
    # 只比较写入槽位；高精度 Cache 是独立的 FP32 影子副本。
    high_cache = torch.zeros(
        (*inputs["kv_cache"].shape[:3], 512 + dr), dtype=torch.float32
    )
    high_inputs = dict(inputs, kv_cache=high_cache)
    high_outputs, high_captured = golden_with_reference(high, high_inputs, attrs)
    valid = inputs["cache_index"] >= 0
    hits = inputs["cache_index"][valid]

    def query_values(outputs):
        q, scale, _ = outputs
        values = golden.decode_hif8(q.contiguous()) if mode == 3 else q.float()
        return values * scale if mode >= 2 else values

    triples = {
        "q": (
            query_values(actual),
            query_values(expected),
            high_outputs[0],
            actual[0].dtype if mode < 2 else torch.float32,
        )
    }
    for name, i in (("descale_q", 1), ("qr", 2)):
        triples[name] = (actual[i], expected[i], high_outputs[i], actual[i].dtype)
    width = after.shape[-1]

    def cache_rows(cache):
        return golden.cache_to_logical(cache, layout).reshape(-1, width)[hits]

    da, ca = cache_rows(after), cache_rows(inputs["kv_cache"])
    ge = golden.cache_to_logical(high_cache, layout).reshape(-1, 512 + dr)[hits]
    diagnostics = {}
    for name, section in (("q_nope", slice(0, 512)), ("q_rope", slice(512, None))):
        diagnostics[name] = tuple(v[..., section] for v in triples["q"][:3]) + (
            triples["q"][3],
        )
    if mode == 1:

        def unpack(rows):
            raw = rows.contiguous().view(torch.uint8)
            kv = raw[:, :512].contiguous().view(torch.float8_e4m3fn).float()
            kr = raw[:, 512 : 512 + 2 * dr].contiguous().view(torch.bfloat16).float()
            scales = raw[:, 512 + 2 * dr :].contiguous().view(torch.float32)
            return (
                (kv.reshape(-1, 4, 128) * scales[..., None]).reshape(-1, 512),
                kr,
                scales,
            )

        dk, dr_values, ds = unpack(da)
        ck, cr, cs = unpack(ca)
        high_kv = high_captured["cache_fp"][0]
        gs = (
            high_kv.reshape(-1, 4, 128).abs().amax(-1).clamp_min(_GROUP_AMAX_FLOOR)
            / _FP8_MAX
        )
        triples["kv"] = (dk, ck, ge[:, :512], torch.float32)
        triples["kr"] = (dr_values, cr, ge[:, 512:], torch.bfloat16)
        triples["cache_descale"] = (ds, cs, gs[valid], torch.float32)
    else:

        def real_cache(rows):
            values = (
                golden.decode_hif8(rows.contiguous()) if mode == 3 else rows.float()
            )
            if mode >= 2 and inputs["quant_scale_kv"].item() != 0:
                return values / inputs["quant_scale_kv"]
            return values

        # scale 为零时无法反量化，直接比较解码后的零载荷。
        cache_reference = (
            torch.zeros_like(ge)
            if mode >= 2 and inputs["quant_scale_kv"].item() == 0
            else ge
        )
        triples["cache"] = (
            real_cache(da),
            real_cache(ca),
            cache_reference,
            torch.bfloat16 if mode == 0 else torch.float32,
        )
        for name, section in (("kv", slice(0, 512)), ("kr", slice(512, None))):
            diagnostics[name] = tuple(v[..., section] for v in triples["cache"][:3]) + (
                triples["cache"][3],
            )
    records = {name: policy.three_way(*values) for name, values in triples.items()}
    states = {r["status"] for r in records.values()}
    return dict(
        status="Failed"
        if "Failed" in states
        else "warning"
        if "warning" in states
        else "Pass",
        segments=records,
        diagnostics={
            name: policy.three_way(*values) for name, values in diagnostics.items()
        },
        diagnostics_affect_acceptance=True,
        reference="D=device; C=two-way ordinary Golden; G=unquantized FP32; independent comparison copies",
    ), triples


def accept_output_comparisons(direct, fallback, diagnostics, mode):
    """优先比较完整输出；两条路线均失败后，才拆分同质分支比较。"""
    required = {"q", "qr", "descale_q"}
    required.update({"kv", "kr", "cache_descale"} if mode == 1 else {"cache"})
    if set(direct) != required:
        raise ValueError("missing or extra two-way output")
    fallback = fallback or {}
    three = fallback.get("segments", {})
    split_three = fallback.get("diagnostics", {})
    splits = {"q": ("q_nope", "q_rope")}
    if mode != 1:
        splits["cache"] = ("kv", "kr")

    def either(two_record, three_record):
        two = two_record.get("status", "NOT_RUN")
        triple = three_record.get("status", "NOT_RUN")
        if two in ("PASS", "EMPTY"):
            path = "TWO_WAY"
        elif triple in ("Pass", "warning"):
            path = "THREE_WAY"
        elif two == "FAIL" and triple == "Failed":
            path = "FAIL"
        else:
            path = "UNVERIFIED"
        return dict(two_way=two, three_way=triple, accepted_by=path)

    outputs = {}
    for name in sorted(required):
        decision = either(direct[name], three.get(name, {}))
        if decision["accepted_by"] == "FAIL" and name in splits:
            branches = {
                branch: either(diagnostics.get(branch, {}), split_three.get(branch, {}))
                for branch in splits[name]
            }
            states = {item["accepted_by"] for item in branches.values()}
            decision["branches"] = branches
            decision["accepted_by"] = (
                "FAIL"
                if "FAIL" in states
                else "UNVERIFIED"
                if "UNVERIFIED" in states
                else "SPLIT_FALLBACK"
            )
        outputs[name] = decision
    states = {item["accepted_by"] for item in outputs.values()}
    status = (
        "FAIL"
        if "FAIL" in states
        else "UNVERIFIED"
        if "UNVERIFIED" in states
        else "PASS"
    )
    category = (
        "PRECISION_FAIL"
        if status == "FAIL"
        else "UNVERIFIED"
        if status == "UNVERIFIED"
        else "TWO_WAY_PASS"
        if states == {"TWO_WAY"}
        else "THREE_WAY_PASS"
    )
    return dict(
        status=status,
        category=category,
        outputs=outputs,
        split_fallback_outputs=[
            name
            for name, item in outputs.items()
            if item["accepted_by"] == "SPLIT_FALLBACK"
        ],
    )


def compare_outputs(cpu, actual, attrs, outputs, before):
    expected, captured = policy.golden_with_reference(golden, cpu, attrs)
    host = tuple(value.cpu() for value in outputs)
    after = actual["kv_cache"].cpu()
    for value, reference in zip(host, expected):
        assert value.shape == reference.shape and value.dtype == reference.dtype
    direct = policy.compare_outputs(
        golden,
        host,
        expected,
        after,
        cpu["kv_cache"],
        cpu,
        attrs,
        captured,
    )
    direct_status = policy.precision_status(direct)
    diagnostics = policy.compare_outputs(
        golden,
        host,
        expected,
        after,
        cpu["kv_cache"],
        cpu,
        attrs,
        captured,
        split_diagnostics=True,
    )
    report = dict(
        two_way=direct, two_way_diagnostics=diagnostics, category="TWO_WAY_PASS"
    )
    fallback = None
    if direct_status == "FAIL":
        fallback, _ = fallback_segments(
            golden, cpu, attrs, host, after, expected, captured
        )
        report["three_way"] = fallback
    acceptance = accept_output_comparisons(
        direct, fallback, diagnostics, int(attrs["qkv_quant_mode"])
    )
    report.update(acceptance=acceptance, category=acceptance["category"])
    assert acceptance["status"] == "PASS", report
    layout = attrs["cache_mode"]
    old_bytes = (
        golden.cache_to_logical(before, layout)
        .contiguous()
        .view(torch.uint8)
        .reshape(before.shape[0] * before.shape[1], -1)
    )
    new_bytes = (
        golden.cache_to_logical(after, layout)
        .contiguous()
        .view(torch.uint8)
        .reshape(before.shape[0] * before.shape[1], -1)
    )
    untouched = torch.ones(before.shape[0] * before.shape[1], dtype=torch.bool)
    untouched[cpu["cache_index"]] = False
    assert torch.equal(old_bytes[untouched], new_bytes[untouched]), (
        "Untouched Cache bytes changed"
    )
    return report


_FROZEN_NUMERIC_AST = "1522db2d183f381edadfdefc565c42d4780eb57e243ac6be23bdc626e7d46a24"


_FROZEN_NUMERIC_NAMES = (
    "high_precision_projection",
    "first_failed_indices",
    "_KV_LORA",
    "_GROUP_SIZE",
    "_FP8_MAX",
    "_HIF8_MAX",
    "_GROUP_AMAX_FLOOR",
    "ordinary_cpu",
    "CacheMode",
    "_require",
    "_tensor",
    "_finite_scalar",
    "_scale",
    "pack_weight_nz",
    "_weight_logical",
    "_rms_norm",
    "_rope",
    "_fp8",
    "_decode_hif8_values",
    "decode_hif8",
    "_hif8",
    "_dynamic_descale",
    "_dynamic_q",
    "_cache_payload",
    "_write_cache",
    "cache_to_logical",
    "golden_mla_prolog",
    "DEFAULT_CONFIG",
    "error_metrics",
    "compare_three_way",
    "POLICY",
    "ADAPTATION",
    "BM_CMP_STD",
    "_pair",
    "_record",
    "_first_failed_elements",
    "e8m0_int8",
    "mixed",
    "e4m3fn",
    "hif8",
    "golden_with_reference",
    "compare_numeric_outputs",
    "precision_status",
    "three_way",
    "acceptance_status",
    "fallback_segments",
    "accept_output_comparisons",
)


def verify_reference_integrity():
    tree = ast.parse(Path(__file__).read_text())
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            definitions[node.targets[0].id] = node
    # 只规范 Python 3.12 为非泛型定义新增的空 type_params 元数据，不改变数值 AST 字段。

    payload = "\n".join(
        ast.dump(definitions[name]).replace(", type_params=[]", "")
        for name in _FROZEN_NUMERIC_NAMES
    )
    assert hashlib.sha256(payload.encode()).hexdigest() == _FROZEN_NUMERIC_AST, (
        "Frozen reference or precision policy changed"
    )


CACHE_LAYOUTS = (
    (1, 0, 0),  # 连续布局基线。
    (2, 0, "page"),  # 页步长和存储偏移。
    (1, 1, 3),  # 非对齐间隔和存储偏移。
    (1, 4099, 65),  # 较大间隔和存储偏移。
)
