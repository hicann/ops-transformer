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
    rms = torch.sqrt(square_sum / partial_out.shape[-1] + float(eps))
    score = torch.sum(partial_out * pseudo_query_fp32, dim=-1) / rms

    current_max = torch.maximum(logit_max_fp32, score)
    alpha = torch.exp(logit_max_fp32 - current_max)
    beta = torch.exp(score - current_max)
    denominator = exp_sum_fp32 * alpha + beta
    h_fp32 = (
        numerator_fp32 * (alpha / denominator)[:, None]
        + partial_out * (beta / denominator)[:, None]
    )
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
        inv_d = numpy.float32(1.0 / partial_out.shape[-1])
        rms = numpy.sqrt(square_sum * inv_d + numpy.float32(eps))
        dot_sum = numpy.sum(partial_out * pseudo_query, axis=-1, dtype=numpy.float32)
        score = dot_sum / rms

        current_max = numpy.maximum(logit_max, score)
        alpha = numpy.exp(logit_max - current_max).astype(numpy.float32, copy=False)
        beta = numpy.exp(score - current_max).astype(numpy.float32, copy=False)
        denominator = exp_sum * alpha + beta
        alpha = alpha / denominator
        beta = beta / denominator

        h_fp32 = numerator * alpha[:, None] + partial_out * beta[:, None]
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
