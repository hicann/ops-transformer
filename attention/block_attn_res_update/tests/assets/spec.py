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
"""Kernel, ACLNN and E2E TestSpecs for BlockAttnResUpdate."""

import numpy


_FP32_TINY = numpy.float32(numpy.finfo(numpy.float32).tiny)


__spec__ = {
    "block_attn_res_update": "BlockAttnResUpdateTestSpec",
    "aclnnBlockAttnResUpdate": "BlockAttnResUpdateAclnnTestSpec",
    "cann_ops_transformer.ops.block_attn_res_update": "BlockAttnResUpdateE2ETestSpec",
}


def _round_to_bfloat16_float32(value):
    """Round FP32 values to BF16 (RNE) in an FP32 NumPy container."""
    value = numpy.ascontiguousarray(value, dtype=numpy.float32)
    bits = value.view(numpy.uint32)
    nan_mask = numpy.isnan(value)
    rounding_bias = numpy.uint32(0x7FFF) + (
        (bits >> numpy.uint32(16)) & numpy.uint32(1)
    )
    rounded_bits = (bits + rounding_bias) & numpy.uint32(0xFFFF0000)
    # Keep NaNs as NaNs even when all payload bits are in the truncated half.
    rounded_bits = numpy.where(
        nan_mask,
        (bits & numpy.uint32(0xFFFF0000)) | numpy.uint32(0x00010000),
        rounded_bits,
    ).astype(numpy.uint32, copy=False)
    return rounded_bits.view(numpy.float32)


def _ftz_float32(value):
    """Flush FP32 subnormal values to signed zero."""
    value = numpy.asarray(value, dtype=numpy.float32)
    signed_zero = numpy.copysign(numpy.float32(0.0), value)
    return numpy.where(numpy.abs(value) < _FP32_TINY, signed_zero, value).astype(
        numpy.float32, copy=False
    )


def _div_ftz_float32(lhs, rhs):
    """Model Ascend 950's default FP32 Div behavior with ``--cce-ftz=true``."""
    lhs = _ftz_float32(lhs)
    rhs = _ftz_float32(rhs)
    with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
        quotient = numpy.divide(lhs, rhs)
    return _ftz_float32(quotient)


def _sqrt_ftz_float32(value):
    """Model Ascend 950's default FP32 Sqrt behavior with ``--cce-ftz=true``."""
    value = _ftz_float32(value)
    with numpy.errstate(invalid="ignore", under="ignore"):
        result = numpy.sqrt(value)
    return _ftz_float32(result)


def _exp_sub_ftz_float32(lhs, rhs):
    """Model the default-FTZ ``ExpSub(lhs, rhs)`` instruction boundary."""
    lhs = _ftz_float32(lhs)
    rhs = _ftz_float32(rhs)
    difference = _ftz_float32(lhs - rhs)
    with numpy.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = numpy.exp(difference).astype(numpy.float32, copy=False)
    return _ftz_float32(result)


def _fma_float32(dst, src0, src1):
    """Model ``Reg::MulDstAdd<float>`` with one FP32 rounding."""
    dst = numpy.asarray(dst, dtype=numpy.float32)
    src0 = numpy.asarray(src0, dtype=numpy.float32)
    src1 = numpy.asarray(src1, dtype=numpy.float32)
    with numpy.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = dst.astype(numpy.float64) * src0.astype(numpy.float64) + src1.astype(
            numpy.float64
        )
    return result.astype(numpy.float32, copy=False)


def _torch_ftz_float32(value):
    """Torch counterpart of :func:`_ftz_float32`."""
    import torch

    tiny = torch.tensor(
        torch.finfo(torch.float32).tiny, dtype=value.dtype, device=value.device
    )
    signed_zero = torch.copysign(torch.zeros_like(value), value)
    return torch.where(torch.abs(value) < tiny, signed_zero, value)


def _torch_div_ftz_float32(lhs, rhs):
    """Torch counterpart of :func:`_div_ftz_float32`."""
    import torch

    quotient = torch.div(_torch_ftz_float32(lhs), _torch_ftz_float32(rhs))
    return _torch_ftz_float32(quotient)


def _torch_sqrt_ftz_float32(value):
    """Torch counterpart of :func:`_sqrt_ftz_float32`."""
    import torch

    return _torch_ftz_float32(torch.sqrt(_torch_ftz_float32(value)))


def _torch_exp_sub_ftz_float32(lhs, rhs):
    """Torch counterpart of :func:`_exp_sub_ftz_float32`."""
    import torch

    difference = _torch_ftz_float32(_torch_ftz_float32(lhs) - _torch_ftz_float32(rhs))
    return _torch_ftz_float32(torch.exp(difference))


def _torch_fma_float32(dst, src0, src1):
    """Torch counterpart of :func:`_fma_float32`."""
    import torch

    result = dst.to(dtype=torch.float64) * src0.to(dtype=torch.float64) + src1.to(
        dtype=torch.float64
    )
    return result.to(dtype=torch.float32)


def _torch_golden(
    partial_block,
    delta,
    pseudo_query,
    numerator,
    logit_max,
    exp_sum,
    eps,
):
    """Return ``(updated_partial_block, h)`` as CPU torch tensors."""
    import torch

    partial = partial_block.to(dtype=torch.float32)
    delta_fp32 = delta.to(dtype=torch.float32)
    pseudo_query_fp32 = pseudo_query.to(dtype=torch.float32)
    numerator_fp32 = numerator.to(dtype=torch.float32)
    logit_max_fp32 = logit_max.to(dtype=torch.float32)
    exp_sum_fp32 = exp_sum.to(dtype=torch.float32)

    # Keep the golden functional: the real API updates partial_block in place,
    # but mutating TTK's CPU input here would affect later input reuse.
    partial_out = partial + delta_fp32
    if partial_out.numel() == 0:
        return partial_out, torch.empty_like(delta)

    square_sum = torch.sum(partial_out * partial_out, dim=-1)
    dot_sum = torch.sum(partial_out * pseudo_query_fp32, dim=-1)
    inv_d = torch.tensor(
        float(numpy.float32(1.0 / partial_out.shape[-1])),
        dtype=torch.float32,
        device=partial_out.device,
    )
    rms = _torch_sqrt_ftz_float32(square_sum * inv_d + float(eps))
    score = _torch_div_ftz_float32(dot_sum, rms)

    current_max = torch.maximum(logit_max_fp32, score)
    alpha = _torch_exp_sub_ftz_float32(logit_max_fp32, current_max)
    beta = _torch_exp_sub_ftz_float32(score, current_max)
    denominator = _torch_fma_float32(exp_sum_fp32, alpha, beta)
    inv_denominator = _torch_div_ftz_float32(torch.ones_like(denominator), denominator)
    alpha = _torch_ftz_float32(alpha * inv_denominator)
    beta = _torch_ftz_float32(beta * inv_denominator)
    partial_scaled = partial_out * beta[:, None]
    h_fp32 = _torch_fma_float32(numerator_fp32, alpha[:, None], partial_scaled)
    return partial_out, h_fp32.to(dtype=torch.bfloat16)


class BlockAttnResUpdateTestSpec:
    """CPU reference for the kernel-only BlockAttnResUpdate test path."""

    @staticmethod
    def golden(
        partial_block,
        delta,
        pseudo_query,
        numerator,
        logit_max,
        exp_sum,
        eps=1e-6,
        **kwargs,
    ):
        del kwargs
        partial = numpy.asarray(partial_block, dtype=numpy.float32)
        delta_fp32 = numpy.asarray(delta, dtype=numpy.float32)
        pseudo_query = numpy.asarray(pseudo_query, dtype=numpy.float32)
        numerator = numpy.asarray(numerator, dtype=numpy.float32)
        logit_max = numpy.asarray(logit_max, dtype=numpy.float32)
        exp_sum = numpy.asarray(exp_sum, dtype=numpy.float32)

        # Do not mutate the input before TTK copies it to the device. Output 0
        # aliases input 0 only in the kernel launch buffers.
        partial_out = numpy.add(partial, delta_fp32).astype(numpy.float32, copy=False)

        square_sum = numpy.sum(partial_out * partial_out, axis=-1, dtype=numpy.float32)
        dot_sum = numpy.sum(partial_out * pseudo_query, axis=-1, dtype=numpy.float32)
        inv_d = numpy.float32(1.0 / partial_out.shape[-1])
        rms = _sqrt_ftz_float32(square_sum * inv_d + numpy.float32(eps))
        score = _div_ftz_float32(dot_sum, rms)

        current_max = numpy.maximum(logit_max, score)
        alpha = _exp_sub_ftz_float32(logit_max, current_max)
        beta = _exp_sub_ftz_float32(score, current_max)
        denominator = _fma_float32(exp_sum, alpha, beta)
        inv_denominator = _div_ftz_float32(numpy.ones_like(denominator), denominator)
        alpha = _ftz_float32(alpha * inv_denominator)
        beta = _ftz_float32(beta * inv_denominator)

        partial_scaled = partial_out * beta[:, None]
        h_fp32 = _fma_float32(numerator, alpha[:, None], partial_scaled)
        h = _round_to_bfloat16_float32(h_fp32)
        return [partial_out, h]


class BlockAttnResUpdateAclnnTestSpec:
    """TestSpec registered by the exact ACLNN API name."""

    @staticmethod
    def golden(
        partialBlockRef,
        delta,
        pseudoQuery,
        numerator,
        logitMax,
        expSum,
        eps=1e-6,
        h=None,
        **kwargs,
    ):
        del h, kwargs
        partial_out, h_golden = _torch_golden(
            partialBlockRef,
            delta,
            pseudoQuery,
            numerator,
            logitMax,
            expSum,
            eps,
        )
        # ACLNN CSV output_tensor_indexes=(0, 6): inplace partialBlockRef, then h.
        return [partial_out, h_golden]


class BlockAttnResUpdateE2ETestSpec:
    """TestSpec registered by the explicit Python wrapper API name."""

    @staticmethod
    def golden(
        partial_block,
        delta,
        pseudo_query,
        numerator,
        logit_max,
        exp_sum,
        *,
        eps=1e-6,
        **kwargs,
    ):
        del kwargs
        partial_out, h_golden = _torch_golden(
            partial_block,
            delta,
            pseudo_query,
            numerator,
            logit_max,
            exp_sum,
            eps,
        )
        # E2E records the API return first, then inplace_input_indexes=(0,).
        return [h_golden, partial_out]
