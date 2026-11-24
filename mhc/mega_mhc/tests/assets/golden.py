# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch CPU reference preserving MegaMhc's arithmetic and activation rounding.

NumPy is used only at the TTK array boundary; all reference computation uses Torch.
"""

import numpy as np
import ml_dtypes
from en_dtypes import float8_e8m0
import torch

__spec__ = {"mega_mhc": "MegaMhcTestSpec"}

_HEADS = 4
_LANES = 64
_PAIR_LANES = 2 * _LANES
_QUANT_GROUP = 32
_SCALE_PAIR = 2
_FP32_EXP_SHIFT = 23
_FP32_EXP_MASK = 255
_FP32_EXP_BIAS = 127
_FP8_MAX_EXP = 8
_FP8_MAX = 448.0
_HF32_MANTISSA_BITS = 10
_NEWTON_HALF = 0.5


def _tensor(value, dtype=None):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu()
    elif value.dtype == np.dtype(ml_dtypes.bfloat16):
        result = torch.from_numpy(np.ascontiguousarray(value).view(np.uint16)).view(
            torch.bfloat16
        )
    else:
        result = torch.from_numpy(np.ascontiguousarray(value))
    return result if dtype is None else result.to(dtype)


def _numpy(value):
    value = value.contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    if value.dtype == torch.float8_e4m3fn:
        return value.view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
    if value.dtype == torch.float8_e8m0fnu:
        return value.view(torch.uint8).numpy().view(float8_e8m0)
    return value.numpy()


def _tree_sum(values):
    """Adjacent binary-tree reduction used by the arch35 vector SUM instruction."""
    while values.shape[-1] > 1:
        values = values[..., ::2] + values[..., 1::2]
    return values[..., :1]


def _squares(value, mode="rms"):
    """Keep independent even/odd vector chains and each caller's tail placement."""
    source = value.float().reshape(-1, value.shape[-1])
    rows, length = source.shape
    padded = torch.nn.functional.pad(source, (0, (-length) % _PAIR_LANES))
    even = torch.zeros((rows, _LANES), dtype=torch.float32)
    odd = torch.zeros_like(even)
    paired = length // _PAIR_LANES
    if mode == "h_in":
        paired = (length + _LANES - 1) // _LANES // 2
    for index in range(paired):
        start = index * _PAIR_LANES
        lo = padded[:, start : start + _LANES]
        hi = padded[:, start + _LANES : start + _PAIR_LANES]
        even = torch.addcmul(even, lo, lo)
        odd = torch.addcmul(odd, hi, hi)
    for start in range(paired * _PAIR_LANES, length, _LANES):
        part = padded[:, start : start + _LANES]
        even = torch.addcmul(even, part, part)
    return _tree_sum(even + odd)


def _post_squares(value):
    """Accumulate the rounded Post heads in the producer's 0/1 and 2/3 pairs."""
    rows, _, length = value.reshape(-1, _HEADS, value.shape[-1]).shape
    source = torch.nn.functional.pad(
        value.float().reshape(rows, _HEADS, length), (0, (-length) % _LANES)
    )
    first = torch.zeros((rows, _LANES), dtype=torch.float32)
    second = torch.zeros_like(first)
    for start in range(0, length, _LANES):
        part = source[..., start : start + _LANES]
        pair01 = torch.addcmul(part[:, 0] * part[:, 0], part[:, 1], part[:, 1])
        pair23 = torch.addcmul(part[:, 2] * part[:, 2], part[:, 3], part[:, 3])
        first = first + pair01
        second = second + pair23
    return _tree_sum(first + second)


def _inverse_rms(squares, length, epsilon, refine=True):
    mean = squares * torch.tensor(1.0 / length, dtype=torch.float32) + torch.tensor(
        epsilon, dtype=torch.float32
    )
    if not refine:
        return 1.0 / torch.sqrt(mean)
    # Evaluate scalar sqrt/div accurately, with a separate FP32 rounding after each.
    # The subsequent Newton step keeps the Kernel's FP32 Mul/FMA operation order.
    root = torch.sqrt(mean.double()).float()
    inverse = (torch.ones_like(root, dtype=torch.float64) / root.double()).float()
    product = inverse * mean
    error = torch.addcmul(torch.ones_like(mean), product, -inverse)
    result = torch.addcmul(inverse, error, inverse * _NEWTON_HALF)
    return torch.where(mean <= torch.finfo(torch.float32).max, result, inverse)


def _rms_norm(value, weight, epsilon, squares=None, mode="rms"):
    source = value.float()
    length = source.shape[-1]
    if squares is None:
        squares = _squares(value, mode)
    inverse = _inverse_rms(squares, length, epsilon).reshape(*source.shape[:-1], 1)
    return (source * inverse * weight.float()).to(value.dtype)


def _projection_squares(value):
    rows, length = value.shape
    source = torch.nn.functional.pad(value.float(), (0, (-length) % _LANES))
    partial = torch.zeros((rows, _LANES), dtype=torch.float32)
    for start in range(0, length, _LANES):
        part = source[:, start : start + _LANES]
        partial = torch.addcmul(partial, part, part)
    return _tree_sum(partial)


def _hf32(value):
    """Match arch35's ten fraction bits and legacy NEAREST_ZERO (ties away) mode."""
    drop = _FP32_EXP_SHIFT - _HF32_MANTISSA_BITS
    bits = value.contiguous().view(torch.int32)
    rounded = (bits + (1 << (drop - 1))) & -(1 << drop)
    return torch.where(torch.isfinite(value), rounded.view(torch.float32), value)


def _mx_quant(value):
    """OCP scales and saturating RNE FP8 with the default NaN-to-zero mode."""
    length = value.shape[-1]
    padded = torch.nn.functional.pad(
        value.float(), (0, (-length) % (_QUANT_GROUP * _SCALE_PAIR))
    )
    blocks = padded.reshape(*value.shape[:-1], -1, _QUANT_GROUP)
    exponent = (
        blocks.contiguous().view(torch.int32) >> _FP32_EXP_SHIFT
    ) & _FP32_EXP_MASK
    maximum = exponent.amax(dim=-1)
    scales = (maximum - _FP8_MAX_EXP).clamp_min(0)
    scales = torch.where(maximum == _FP32_EXP_MASK, _FP32_EXP_MASK, scales)
    inverse = torch.ldexp(
        torch.ones_like(scales, dtype=torch.float32), _FP32_EXP_BIAS - scales
    )
    inverse = torch.where(scales == 0, 0.0, inverse)
    inverse = torch.where(scales == _FP32_EXP_MASK, float("nan"), inverse)
    scaled = blocks * inverse.unsqueeze(-1)
    if value.dtype == torch.bfloat16:
        scaled = scaled.to(torch.bfloat16).float()
    # Match the default saturated FP8 conversion used by the standalone MX quantizers.
    scaled = torch.where(torch.isnan(scaled), 0.0, scaled)
    quantized = scaled.clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    quantized = quantized.reshape(padded.shape)[..., :length].contiguous()
    scale = scales.to(torch.uint8).reshape(*value.shape[:-1], -1, _SCALE_PAIR)
    return quantized, scale.view(torch.float8_e8m0fnu)


class MegaMhcTestSpec:
    tolerance = {
        "float8_e4m3fn": {"standard": "stat_rel_err"},
        "float8_e8m0": {"standard": "binary_equal"},
        "float8_e8m0fnu": {"standard": "binary_equal"},
    }

    @staticmethod
    def customize_inputs(
        x,
        residual,
        post_mix,
        fn,
        mix_scales,
        mix_bases,
        rms_norm_1_weight,
        comb_res_mix=None,
        shifted_prev_mix=None,
        rms_norm_0_weight=None,
        rms_norm_mid_weight=None,
        **kwargs,
    ):
        """Inject quantization boundary values only for the named regression cases."""
        case = kwargs.get("testcase_name", "")
        gamma = rms_norm_1_weight
        if case.startswith("mx_boundary_"):
            kind = case.split("_")[2]
            gamma = gamma.copy()
            gamma.fill(1.0)
            if kind == "zero":
                gamma.fill(0.0)
            elif kind in ("nan", "posinf", "neginf"):
                gamma[0] = {"nan": np.nan, "posinf": np.inf, "neginf": -np.inf}[kind]
            else:
                raise ValueError(f"Unknown MX boundary case: {case}")
        return (
            x,
            residual,
            post_mix,
            fn,
            mix_scales,
            mix_bases,
            gamma,
            comb_res_mix,
            shifted_prev_mix,
            rms_norm_0_weight,
            rms_norm_mid_weight,
        )

    @staticmethod
    def golden(
        x,
        residual,
        post_mix,
        fn,
        mix_scales,
        mix_bases,
        rms_norm_1_weight,
        comb_res_mix=None,
        shifted_prev_mix=None,
        rms_norm_0_weight=None,
        rms_norm_mid_weight=None,
        *,
        rms_norm_eps=1e-6,
        hc_norm_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_post_scale=2.0,
        sinkhorn_eps=1e-6,
        num_sinkhorn_iters=0,
        output_mode=0,
        op_impl_mode=0,
        golden_activation_dtype=None,
        **kwargs,
    ):
        name = str(
            golden_activation_dtype or kwargs.get("input_dtypes", (x.dtype,))[0]
        ).replace("torch.", "")
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[name]
        x = _tensor(x, dtype)
        residual = _tensor(residual, dtype)
        post_mix, fn, mix_scales, mix_bases, gamma1, comb = [
            _tensor(v, torch.float32)
            for v in (
                post_mix,
                fn,
                mix_scales,
                mix_bases,
                rms_norm_1_weight,
                comb_res_mix,
            )
        ]
        gamma0 = _tensor(rms_norm_0_weight, torch.float32)
        gamma_mid = _tensor(rms_norm_mid_weight, torch.float32)
        external = _tensor(shifted_prev_mix, torch.float32)
        leading, length = x.shape[:-1], x.shape[-1]
        x1 = x if gamma0 is None else _rms_norm(x, gamma0, rms_norm_eps)
        mapped = residual[..., 0, :].float().unsqueeze(-2) * comb[..., 0, :].unsqueeze(
            -1
        )
        for head in range(1, _HEADS):
            mapped = torch.addcmul(
                mapped,
                residual[..., head, :].float().unsqueeze(-2),
                comb[..., head, :].unsqueeze(-1),
            )
        x2 = (mapped + post_mix * x1.float().unsqueeze(-2)).to(dtype)
        if gamma_mid is not None:
            squares = _post_squares(x2)
            x2 = _rms_norm(
                x2.reshape(*leading, _HEADS * length), gamma_mid, rms_norm_eps, squares
            ).reshape(x2.shape)
        flat = x2.float().reshape(-1, _HEADS * length)
        inverse = _inverse_rms(
            _projection_squares(flat), flat.shape[-1], hc_norm_eps, refine=False
        )
        left, right = (_hf32(flat), _hf32(fn)) if op_impl_mode == 1 else (flat, fn)
        projected = torch.matmul(left, right.T) * inverse
        scales = torch.repeat_interleave(
            mix_scales, torch.tensor((_HEADS, _HEADS, _HEADS * _HEADS))
        )
        logits = projected * scales + mix_bases
        gates = 1.0 / (1.0 + torch.exp(-logits[:, : 2 * _HEADS]))
        pre = (gates[:, :_HEADS] + hc_pre_eps).reshape(*leading, _HEADS, 1)
        post = (gates[:, _HEADS:] * hc_post_scale).reshape(*leading, _HEADS, 1)
        comb_output = logits[:, 2 * _HEADS :].reshape(*leading, _HEADS, _HEADS)
        if num_sinkhorn_iters > 0:
            comb_output = torch.softmax(comb_output, dim=-1) + sinkhorn_eps
            comb_output = comb_output / (
                comb_output.sum(dim=-2, keepdim=True) + sinkhorn_eps
            )
            for _ in range(1, num_sinkhorn_iters):
                comb_output = comb_output / (
                    comb_output.sum(dim=-1, keepdim=True) + sinkhorn_eps
                )
                comb_output = comb_output / (
                    comb_output.sum(dim=-2, keepdim=True) + sinkhorn_eps
                )
        weights = pre if external is None else external
        h_in = weights[..., 0, :] * x2[..., 0, :].float()
        for head in range(1, _HEADS):
            h_in = torch.addcmul(h_in, weights[..., head, :], x2[..., head, :].float())
        h_in = h_in.to(dtype)
        y = _rms_norm(h_in, gamma1, rms_norm_eps, mode="h_in")
        empty = torch.empty(0, dtype=torch.float32)
        outputs = [x2, post, comb_output, pre if external is not None else empty]
        if output_mode == 1:
            quantized, scale = _mx_quant(y)
            outputs.extend((empty, quantized, scale))
        else:
            outputs.extend((y, empty, empty))
        return [_numpy(v) for v in outputs]
