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
"""Kernel TestSpec for BlockAttnResUpdate."""

import numpy


__spec__ = {"block_attn_res_update": "BlockAttnResUpdateTestSpec"}


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
