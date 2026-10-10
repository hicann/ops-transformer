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
"""Kernel, ACLNN and E2E TestSpecs for BlockAttnResPrepare.

The golden math is kept identical to the previous ``tests/assets/golden.py``
(numpy reference for the kernel path, torch reference for the ACLNN/E2E paths):
RMS normalisation over the last dim, online-softmax statistics over the
``valid_blocks``-clipped history, and the empty online-softmax state when
``valid_blocks[0] == 0``.  Ascend 950 runs the FP32 pipelines with
``--cce-ftz=true``, so subnormal inputs/results are flushed to signed zero and
the ``exp``/``div`` instruction boundaries are modelled explicitly.
"""

import numpy


_FP32_TINY = numpy.float32(numpy.finfo(numpy.float32).tiny)
DEFAULT_EPS = 1.0e-6
T_DIM_INDEX = 0
N_DIM_INDEX = 1
D_DIM_INDEX = 2
S_DIM_INDEX = 0
LAST_DIM_INDEX = -1
FLATTENED_SIZE = -1


__spec__ = {
    "block_attn_res_prepare": "BlockAttnResPrepareTestSpec",
    "aclnnBlockAttnResPrepare": "BlockAttnResPrepareAclnnTestSpec",
    "cann_ops_transformer.ops.block_attn_res_prepare": "BlockAttnResPrepareE2ETestSpec",
}


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


def _numpy_golden(block_res, valid_blocks, pseudo_query, eps):
    """Return ``(numerator, logit_max, exp_sum)`` as FP32 NumPy arrays."""
    residual = numpy.asarray(block_res, dtype=numpy.float32)
    query = numpy.asarray(pseudo_query, dtype=numpy.float32)
    valid = int(
        numpy.asarray(valid_blocks, dtype=numpy.uint64)
        .reshape(FLATTENED_SIZE)[0]
        .item()
    )
    valid = min(valid, residual.shape[N_DIM_INDEX])

    if valid == 0:
        # Identity state of the online softmax: zero numerator, FLOAT32 lowest
        # finite value for logitMax, zero expSum.
        numerator = numpy.zeros(
            (
                query.shape[S_DIM_INDEX],
                residual.shape[T_DIM_INDEX],
                residual.shape[D_DIM_INDEX],
            ),
            dtype=numpy.float32,
        )
        logit_max = numpy.full(
            (query.shape[S_DIM_INDEX], residual.shape[T_DIM_INDEX]),
            numpy.finfo(numpy.float32).min,
            dtype=numpy.float32,
        )
        exp_sum = numpy.zeros_like(logit_max)
        return numerator, logit_max, exp_sum

    history = residual[:, :valid, :]
    square_sum = numpy.sum(history * history, axis=LAST_DIM_INDEX, dtype=numpy.float32)
    rms = _sqrt_ftz_float32(
        square_sum / numpy.float32(history.shape[LAST_DIM_INDEX]) + numpy.float32(eps)
    )
    logits = _div_ftz_float32(
        numpy.einsum("sd,tnd->stn", query, history), numpy.expand_dims(rms, 0)
    )
    logit_max = numpy.max(logits, axis=LAST_DIM_INDEX)
    weights = _exp_sub_ftz_float32(logits, numpy.expand_dims(logit_max, LAST_DIM_INDEX))
    exp_sum = numpy.sum(weights, axis=LAST_DIM_INDEX, dtype=numpy.float32)
    numerator = numpy.einsum("stn,tnd->std", weights, history)
    return (
        numerator.astype(numpy.float32, copy=False),
        logit_max.astype(numpy.float32, copy=False),
        exp_sum.astype(numpy.float32, copy=False),
    )


def _torch_golden(block_res, valid_blocks, pseudo_query, eps):
    """Return ``(numerator, logit_max, exp_sum)`` as CPU torch tensors."""
    import torch

    residual = torch.as_tensor(block_res).to(dtype=torch.float32)
    query = torch.as_tensor(pseudo_query).to(dtype=torch.float32)
    valid = int(torch.as_tensor(valid_blocks).reshape(FLATTENED_SIZE)[0].item())
    valid = min(valid, residual.shape[N_DIM_INDEX])

    if valid == 0:
        numerator = torch.zeros(
            (
                query.shape[S_DIM_INDEX],
                residual.shape[T_DIM_INDEX],
                residual.shape[D_DIM_INDEX],
            ),
            dtype=torch.float32,
            device=residual.device,
        )
        logit_max = torch.full(
            (query.shape[S_DIM_INDEX], residual.shape[T_DIM_INDEX]),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
            device=residual.device,
        )
        exp_sum = torch.zeros_like(logit_max)
        return numerator, logit_max, exp_sum

    history = residual[:, :valid, :]
    square_sum = torch.sum(history * history, dim=LAST_DIM_INDEX)
    rms = _torch_sqrt_ftz_float32(
        square_sum / history.shape[LAST_DIM_INDEX] + float(eps)
    )
    logits = _torch_div_ftz_float32(
        torch.einsum("sd,tnd->stn", query, history), rms.unsqueeze(0)
    )
    logit_max = torch.max(logits, dim=LAST_DIM_INDEX).values
    weights = _torch_exp_sub_ftz_float32(logits, logit_max.unsqueeze(LAST_DIM_INDEX))
    exp_sum = torch.sum(weights, dim=LAST_DIM_INDEX)
    numerator = torch.einsum("stn,tnd->std", weights, history)
    return (
        numerator.to(torch.float32),
        logit_max.to(torch.float32),
        exp_sum.to(torch.float32),
    )


class BlockAttnResPrepareTestSpec:
    """CPU reference for the kernel-only BlockAttnResPrepare test path.

    TTK calls this with the flat numpy input arrays of the kernel CSV
    (``block_res``, ``valid_blocks``, ``pseudo_query``) in order; ``eps``
    arrives through the CSV ``attributes`` dict (``**kwargs`` also receives
    the TTK metadata such as ``input_dtypes``/``testcase_name``).
    """

    @staticmethod
    def golden(block_res, valid_blocks, pseudo_query, eps=DEFAULT_EPS, **kwargs):
        del kwargs
        numerator, logit_max, exp_sum = _numpy_golden(
            block_res, valid_blocks, pseudo_query, eps
        )
        # Kernel CSV output order: numerator, logit_max, exp_sum.
        return [numerator, logit_max, exp_sum]


class BlockAttnResPrepareAclnnTestSpec:
    """TestSpec registered by the exact ACLNN API name.

    ``AclnnParamPlan`` hands over the C-header parameter order
    (``blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum, eps``);
    the three output tensors are stubs on the golden path and are recomputed.
    """

    @staticmethod
    def golden(
        blockRes,
        validBlocks,
        pseudoQuery,
        numerator,
        logitMax,
        expSum,
        eps=DEFAULT_EPS,
        **kwargs,
    ):
        del numerator, logitMax, expSum, kwargs
        numerator_golden, logit_max_golden, exp_sum_golden = _torch_golden(
            blockRes, validBlocks, pseudoQuery, eps
        )
        # ACLNN CSV output_tensor_indexes=(3, 4, 5).
        return [numerator_golden, logit_max_golden, exp_sum_golden]


class BlockAttnResPrepareE2ETestSpec:
    """TestSpec registered by the explicit Python wrapper API name.

    The E2E CSV only carries the three input tensors; the wrapper returns
    ``(numerator, logit_max, exp_sum)`` and TTK compares them in that order.
    """

    @staticmethod
    def golden(
        block_res,
        valid_blocks,
        pseudo_query,
        *,
        eps=DEFAULT_EPS,
        numerator=None,
        logit_max=None,
        exp_sum=None,
        **kwargs,
    ):
        del numerator, logit_max, exp_sum, kwargs
        numerator_golden, logit_max_golden, exp_sum_golden = _torch_golden(
            block_res, valid_blocks, pseudo_query, eps
        )
        return [numerator_golden, logit_max_golden, exp_sum_golden]
