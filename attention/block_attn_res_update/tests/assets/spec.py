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
"""Kernel, ACLNN and E2E TestSpecs for BlockAttnResUpdate.

CPU paths share FP64 reductions with FP32 arithmetic/storage boundaries.
Div, Sqrt and Exp use FTZ; ordinary multiplication preserves subnormals.
The optional output RMSNorm consumes the BF16-rounded mixture.
Each output allows at most 0.1% of its elements to have relative error
greater than 0.1%, without an absolute-error allowance.

TTK --golden-mode NPU measures the composed Torch operators for plain Update.
With output_norm_weight, the baseline is the unfused BlockAttnResUpdate
operator followed by torch_npu.npu_rms_norm, including the BF16 h boundary.
"""

import numpy


_FP32_TINY = numpy.float32(numpy.finfo(numpy.float32).tiny)
RELATIVE_ERROR_LIMIT = 0.001
BAD_RATIO_DENOMINATOR = 1000
COMPARE_CHUNK_ELEMENTS = 1 << 20
MAX_DIFF_INDICES = 20


__spec__ = {
    "block_attn_res_update": "BlockAttnResUpdateTestSpec",
    "aclnnBlockAttnResUpdate": "BlockAttnResUpdateAclnnTestSpec",
    "aclnnBlockAttnResUpdateV2": "BlockAttnResUpdateV2AclnnTestSpec",
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
    """Apply the explicitly selected PRECISION_0ULP_FTZ_TRUE Div boundary."""
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
    # Sub preserves subnormals; Exp applies FTZ to its input and output.
    lhs = numpy.asarray(lhs, dtype=numpy.float32)
    rhs = numpy.asarray(rhs, dtype=numpy.float32)
    difference = _ftz_float32(lhs - rhs)
    with numpy.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = numpy.exp(difference.astype(numpy.float64))
    return _ftz_float32(result)


def _fma_float32(dst, src0, src1):
    """Approximate FP32 fused multiply-add with FP64 intermediates.

    This avoids a separately rounded FP32 product; it is not a bit-exact
    instruction emulator at FP64-to-FP32 double-rounding midpoints.
    """
    dst = numpy.asarray(dst, dtype=numpy.float32)
    src0 = numpy.asarray(src0, dtype=numpy.float32)
    src1 = numpy.asarray(src1, dtype=numpy.float32)
    with numpy.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = dst.astype(numpy.float64) * src0.astype(numpy.float64) + src1.astype(
            numpy.float64
        )
    return result.astype(numpy.float32, copy=False)


def _numpy_golden(
    partial_block,
    delta,
    pseudo_query,
    numerator,
    logit_max,
    exp_sum,
    eps,
    output_norm_weight=None,
    output_norm_eps=1e-6,
    input_dtypes=None,
):
    """Share FP64 reductions and explicit FP32/BF16 boundaries across all paths."""
    partial = numpy.asarray(_as_numpy(partial_block), dtype=numpy.float32)
    # delta is always stored as BF16, including when TTK promotes its container.
    delta_fp32 = _round_to_bfloat16_float32(_as_numpy(delta))
    query = numpy.asarray(_as_numpy(pseudo_query), dtype=numpy.float32)
    numerator = numpy.asarray(_as_numpy(numerator), dtype=numpy.float32)
    logit_max = numpy.asarray(_as_numpy(logit_max), dtype=numpy.float32)
    exp_sum = numpy.asarray(_as_numpy(exp_sum), dtype=numpy.float32)

    with numpy.errstate(
        over="ignore", under="ignore", invalid="ignore", divide="ignore"
    ):
        # The in-place result is an FP32 storage boundary. Keep the CPU inputs
        # unchanged: TTK may copy/reuse them after invoking the reference.
        partial_out = numpy.add(partial, delta_fp32)
        if partial_out.size == 0:
            return partial_out, numpy.empty_like(partial_out)

        # Reduce products in FP64 so neither NumPy/Torch summation order nor a
        # separately rounded FP32 product defines the oracle. Round the final
        # accumulators before RMS/Div to retain FP32 overflow and FTZ semantics.
        partial64 = partial_out.astype(numpy.float64)
        square_sum = numpy.einsum("td,td->t", partial64, partial64).astype(
            numpy.float32
        )
        dot_sum = (partial64 @ query.astype(numpy.float64)).astype(numpy.float32)
        inv_d = numpy.float32(1.0 / partial_out.shape[-1])
        rms = _sqrt_ftz_float32(square_sum * inv_d + numpy.float32(eps))
        score = _div_ftz_float32(dot_sum, rms)

        current_max = numpy.maximum(logit_max, score)
        alpha = _exp_sub_ftz_float32(logit_max, current_max)
        beta = _exp_sub_ftz_float32(score, current_max)
        denominator = _fma_float32(exp_sum, alpha, beta)
        inv_denominator = _div_ftz_float32(numpy.ones_like(denominator), denominator)
        # Reg::Mul<float> preserves subnormals. Flushing these weights loses
        # contributions which can become normal again when multiplied by p/O.
        alpha = alpha * inv_denominator
        beta = beta * inv_denominator
        partial_scaled = partial_out * beta[:, None]
        h = _round_to_bfloat16_float32(
            _fma_float32(numerator, alpha[:, None], partial_scaled)
        )

        if output_norm_weight is not None:
            # Output RMSNorm consumes the BF16-rounded mixture, not h_fp32.
            h64 = h.astype(numpy.float64)
            square_sum = numpy.einsum("td,td->t", h64, h64).astype(numpy.float32)
            rms = _sqrt_ftz_float32(square_sum * inv_d + numpy.float32(output_norm_eps))
            rstd = _div_ftz_float32(numpy.ones_like(rms), rms)
            weight = numpy.asarray(_as_numpy(output_norm_weight), dtype=numpy.float32)
            if input_dtypes is not None and len(input_dtypes) > 6:
                weight_dtype = str(input_dtypes[6]).lower()
                if "bfloat16" in weight_dtype or weight_dtype == "bf16":
                    weight = _round_to_bfloat16_float32(weight)
            h = _round_to_bfloat16_float32(h * rstd[:, None] * weight)
        return partial_out, h


def _npu_device(value, device=None):
    device = device if device is not None else getattr(value, "device", "cpu")
    return device if str(device).split(":", 1)[0] in ("npu", "privateuseone") else None


def _to_torch_input(value, dtype, device):
    """Keep storage dtype on upload, including NumPy's BF16 containers."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return (
        torch.from_numpy(numpy.asarray(value, dtype=numpy.float32)).to(dtype).to(device)
    )


def _npu_golden(
    partial_block,
    delta,
    pseudo_query,
    numerator,
    logit_max,
    exp_sum,
    eps,
    output_norm_weight=None,
    output_norm_eps=1e-6,
    input_dtypes=None,
    *,
    device,
):
    """Return the measured NPU composition's partial/h in CPU NumPy containers."""
    import torch

    partial = _to_torch_input(partial_block, torch.float32, device)
    delta = _to_torch_input(delta, torch.bfloat16, device)
    query = _to_torch_input(pseudo_query, torch.float32, device)
    numerator = _to_torch_input(numerator, torch.float32, device)
    logit_max = _to_torch_input(logit_max, torch.float32, device)
    exp_sum = _to_torch_input(exp_sum, torch.float32, device)
    eps = float(numpy.float32(eps))
    if partial.numel() == 0:
        h = torch.empty_like(delta)
    elif output_norm_weight is not None:
        import torch_npu
        from cann_ops_transformer.ops import block_attn_res_update

        weight_dtype = str(getattr(output_norm_weight, "dtype", "float32")).lower()
        if input_dtypes is not None and len(input_dtypes) > 6:
            weight_dtype = str(input_dtypes[6]).lower()
        weight_dtype = (
            torch.bfloat16
            if "bfloat16" in weight_dtype or weight_dtype == "bf16"
            else torch.float32
        )
        weight = _to_torch_input(output_norm_weight, weight_dtype, device)
        # TTK supplies private NPU inputs on each warmup/measured invocation.
        # Keep the real in-place update, without adding a clone to the timed chain.
        h = block_attn_res_update(
            partial,
            delta,
            query,
            numerator,
            logit_max,
            exp_sum,
            eps=eps,
            output_norm_weight=None,
        )
        h, _ = torch_npu.npu_rms_norm(h, weight, float(numpy.float32(output_norm_eps)))
    else:
        # Match the kernel's merge order: normalize alpha/beta with a common
        # reciprocal before mixing, so an unnormalized sum cannot overflow h.
        partial = partial + delta.float()
        rstd = torch.rsqrt(partial.square().mean(dim=-1) + eps)
        score = (partial * query).sum(dim=-1) * rstd
        current_max = torch.maximum(logit_max, score)
        alpha = torch.exp(logit_max - current_max)
        beta = torch.exp(score - current_max)
        denominator = alpha * exp_sum + beta
        inv_denominator = torch.ones_like(denominator) / denominator
        alpha = alpha * inv_denominator
        beta = beta * inv_denominator
        partial_scaled = partial * beta[:, None]
        h = (numerator * alpha[:, None] + partial_scaled).to(torch.bfloat16)
    # Convert BF16 to an FP32 container on CPU, after the measured operators.
    # ACLNN's NPU-golden adapter calls Tensor.numpy(), which has no BF16 support.
    return _as_numpy(partial), _as_numpy(h)


def _torch_golden(
    partial_block,
    delta,
    pseudo_query,
    numerator,
    logit_max,
    exp_sum,
    eps,
    output_norm_weight=None,
    output_norm_eps=1e-6,
    input_dtypes=None,
    device=None,
):
    """Use the CPU oracle or the device composition selected by TTK."""
    device = _npu_device(partial_block, device)
    if device is not None:
        return _npu_golden(
            partial_block,
            delta,
            pseudo_query,
            numerator,
            logit_max,
            exp_sum,
            eps,
            output_norm_weight,
            output_norm_eps,
            input_dtypes,
            device=device,
        )
    import torch

    partial_out, h = _numpy_golden(
        partial_block,
        delta,
        pseudo_query,
        numerator,
        logit_max,
        exp_sum,
        eps,
        output_norm_weight,
        output_norm_eps,
        input_dtypes,
    )
    return torch.from_numpy(partial_out), torch.from_numpy(h).to(dtype=torch.bfloat16)


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        return value.numpy()
    return numpy.asarray(value)


def _compare_relative_error(*outputs, **kwargs):
    """TTK supplies all actual outputs followed by all golden outputs.

    Count errors per tensor. Zero requires an exact zero; matching NaNs and
    same-sign infinities agree, while any nonfinite mismatch fails outright.
    """
    del kwargs
    output_count = len(outputs) // 2
    if not output_count or len(outputs) % 2:
        raise ValueError("compare requires equally many actual and golden outputs")

    results = []
    for actual, golden in zip(outputs[:output_count], outputs[output_count:]):
        if actual is None or golden is None:
            results.append(
                {
                    "pass": False,
                    "precision": 0.0,
                    "error_info": "missing output or golden",
                }
            )
            continue
        actual = _as_numpy(actual)
        golden = _as_numpy(golden)
        if actual.shape != golden.shape:
            results.append(
                {
                    "pass": False,
                    "precision": 0.0,
                    "error_info": f"shape mismatch: {actual.shape} vs {golden.shape}",
                }
            )
            continue

        actual = actual.reshape(-1)
        golden = golden.reshape(-1)
        bad_count = 0
        nonfinite_mismatches = 0
        diff_indices = []
        with numpy.errstate(invalid="ignore", over="ignore", under="ignore"):
            for start in range(0, actual.size, COMPARE_CHUNK_ELEMENTS):
                lhs = actual[start : start + COMPARE_CHUNK_ELEMENTS].astype(
                    numpy.float64, copy=False
                )
                rhs = golden[start : start + COMPARE_CHUNK_ELEMENTS].astype(
                    numpy.float64, copy=False
                )
                finite = numpy.isfinite(lhs) & numpy.isfinite(rhs)
                same_nonfinite = (numpy.isnan(lhs) & numpy.isnan(rhs)) | (
                    numpy.isinf(lhs) & (lhs == rhs)
                )
                mismatch = ~(finite | same_nonfinite)
                # Multiplication avoids division by zero without adding an atol.
                bad = finite & (
                    numpy.abs(lhs - rhs) > RELATIVE_ERROR_LIMIT * numpy.abs(rhs)
                )
                bad_count += int(numpy.count_nonzero(bad))
                nonfinite_mismatches += int(numpy.count_nonzero(mismatch))
                if len(diff_indices) < MAX_DIFF_INDICES:
                    indices = numpy.flatnonzero(bad | mismatch)[
                        : MAX_DIFF_INDICES - len(diff_indices)
                    ]
                    diff_indices.extend((indices + start).tolist())

        size = actual.size
        # Integer arithmetic includes the exact 1/1000 boundary without a
        # floating-point rounding error in 1 - matched_ratio.
        passed = bad_count * BAD_RATIO_DENOMINATOR <= size and nonfinite_mismatches == 0
        results.append(
            {
                "pass": passed,
                "precision": 100.0 * (size - bad_count - nonfinite_mismatches) / size
                if size
                else 100.0,
                "error_info": ""
                if passed
                else (
                    f"{bad_count}/{size} elements exceed 0.1% relative error; "
                    f"{nonfinite_mismatches} nonfinite mismatches"
                ),
                "diff_indices": diff_indices,
                "metrics": {
                    "rtol": RELATIVE_ERROR_LIMIT,
                    "max_bad_ratio": 1.0 / BAD_RATIO_DENOMINATOR,
                    "bad_count": bad_count,
                    "element_count": size,
                    "bad_ratio": bad_count / size if size else 0.0,
                    "nonfinite_mismatches": nonfinite_mismatches,
                },
            }
        )
    return results


class BlockAttnResUpdateTestSpec:
    """CPU reference and NPU composition for the kernel-only test path."""

    compare = staticmethod(_compare_relative_error)

    @staticmethod
    def golden(
        partial_block,
        delta,
        pseudo_query,
        numerator,
        logit_max,
        exp_sum,
        output_norm_weight=None,
        eps=1e-6,
        output_norm_eps=1e-6,
        **kwargs,
    ):
        device = _npu_device(partial_block, kwargs.get("device"))
        if device is not None:
            return list(
                _npu_golden(
                    partial_block,
                    delta,
                    pseudo_query,
                    numerator,
                    logit_max,
                    exp_sum,
                    eps,
                    output_norm_weight,
                    output_norm_eps,
                    kwargs.get("input_dtypes"),
                    device=device,
                )
            )
        return list(
            _numpy_golden(
                partial_block,
                delta,
                pseudo_query,
                numerator,
                logit_max,
                exp_sum,
                eps,
                output_norm_weight,
                output_norm_eps,
                kwargs.get("input_dtypes"),
            )
        )


class BlockAttnResUpdateAclnnTestSpec:
    """TestSpec registered by the exact ACLNN API name."""

    compare = staticmethod(_compare_relative_error)

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
        input_dtypes = kwargs.get("input_dtypes")
        del h
        partial_out, h_golden = _torch_golden(
            partialBlockRef,
            delta,
            pseudoQuery,
            numerator,
            logitMax,
            expSum,
            eps,
            input_dtypes=input_dtypes,
            device=kwargs.get("device"),
        )
        # ACLNN CSV output_tensor_indexes=(0, 6): inplace partialBlockRef, then h.
        return [partial_out, h_golden]


class BlockAttnResUpdateE2ETestSpec:
    """TestSpec registered by the explicit Python wrapper API name."""

    compare = staticmethod(_compare_relative_error)

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
        output_norm_weight=None,
        output_norm_eps=1e-6,
        **kwargs,
    ):
        input_dtypes = kwargs.get("input_dtypes")
        partial_out, h_golden = _torch_golden(
            partial_block,
            delta,
            pseudo_query,
            numerator,
            logit_max,
            exp_sum,
            eps,
            output_norm_weight,
            output_norm_eps,
            input_dtypes=input_dtypes,
            device=kwargs.get("device"),
        )
        # E2E records the API return first, then inplace_input_indexes=(0,).
        return [h_golden, partial_out]


class BlockAttnResUpdateV2AclnnTestSpec:
    """V2 consumes an optional gamma before the two scalar attributes."""

    compare = staticmethod(_compare_relative_error)

    @staticmethod
    def golden(
        partialBlockRef,
        delta,
        pseudoQuery,
        numerator,
        logitMax,
        expSum,
        outputNormWeightOptional=None,
        eps=1e-6,
        outputNormEps=1e-6,
        h=None,
        **kwargs,
    ):
        input_dtypes = kwargs.get("input_dtypes")
        del h
        partial_out, h_golden = _torch_golden(
            partialBlockRef,
            delta,
            pseudoQuery,
            numerator,
            logitMax,
            expSum,
            eps,
            outputNormWeightOptional,
            outputNormEps,
            input_dtypes=input_dtypes,
            device=kwargs.get("device"),
        )
        return [partial_out, h_golden]
