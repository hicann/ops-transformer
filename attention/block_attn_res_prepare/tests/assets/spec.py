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

With an active TTK platform, CPU paths model the BF16 Vector dot's FP32
lane accumulators and the Ascend 950 Cube dot's forward FP32 accumulation.
Standalone CPU calls and other contractions retain FP64 accumulation. Sqrt,
Div and Exp use the Ascend 950 default FTZ semantics;
ordinary multiplication retains subnormal values. For one effective block,
numerator copies V and expSum is one, including when the logit is nonfinite;
logitMax still follows dot/RMS. Other block counts follow the softmax formula.
Each output allows at most 0.1% of its elements to have relative error greater
than 0.1%, without an absolute-error allowance.

With TTK --golden-mode NPU, golden runs the composed FP32 Torch operators
(RMS statistics, matmul, stable exponentials, weighted reduction) on device
for performance comparison. CPU precision references remain independent.
"""

import numpy


DEFAULT_EPS = 1.0e-6
T_DIM_INDEX = 0
N_DIM_INDEX = 1
D_DIM_INDEX = 2
S_DIM_INDEX = 0
LAST_DIM_INDEX = -1
FLATTENED_SIZE = -1
RELATIVE_ERROR_LIMIT = 0.001
BAD_RATIO_DENOMINATOR = 1000
COMPARE_CHUNK_ELEMENTS = 1 << 20
MAX_DIFF_INDICES = 20
_FP32_TINY = numpy.finfo(numpy.float32).tiny
_FP32_VECTOR_ELEMENTS = 64
_CUBE_EXTENDED_BITS = 24 + 5
_DOT_BATCH_ROWS = 4096


def _round_to_bfloat16_float32(value):
    """Round FP32 values to BF16 (RNE), keeping the result in an FP32 array."""
    value = numpy.ascontiguousarray(value, dtype=numpy.float32)
    bits = value.view(numpy.uint32)
    nan_mask = numpy.isnan(value)
    rounding_bias = numpy.uint32(0x7FFF) + (
        (bits >> numpy.uint32(16)) & numpy.uint32(1)
    )
    rounded_bits = (bits + rounding_bias) & numpy.uint32(0xFFFF0000)
    rounded_bits = numpy.where(
        nan_mask,
        (bits & numpy.uint32(0xFFFF0000)) | numpy.uint32(0x00010000),
        rounded_bits,
    ).astype(numpy.uint32, copy=False)
    return rounded_bits.view(numpy.float32)


def _is_bfloat16_dtype(dtype):
    if dtype is None:
        return False
    dtype_name = str(dtype).lower()
    return "bfloat16" in dtype_name or dtype_name in ("bf16", "torch.bfloat16")


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if _is_bfloat16_dtype(value.dtype):
            value = value.float()
        return value.numpy()
    return numpy.asarray(value)


def _numpy_input_to_fp64(value, dtype=None):
    # Native BF16 values already crossed the storage quantization boundary.
    # Convert them directly instead of rounding an FP32 copy a second time.
    if _is_bfloat16_dtype(getattr(value, "dtype", None)):
        if hasattr(value, "detach"):
            return value.detach().cpu().double().numpy()
        return numpy.asarray(value, dtype=numpy.float64)
    value = _as_numpy(value)
    if _is_bfloat16_dtype(dtype) or _is_bfloat16_dtype(value.dtype):
        value = _round_to_bfloat16_float32(value)
    return numpy.asarray(value, dtype=numpy.float64)


def _ftz_float32(value):
    """Apply FTZ only at an instruction boundary that does not retain subnormals."""
    value = numpy.asarray(value, dtype=numpy.float32)
    return numpy.where(
        numpy.abs(value) < _FP32_TINY,
        numpy.copysign(numpy.float32(0.0), value),
        value,
    )


def _ttk_platform():
    """Read the active TTK platform and its optional core limits."""
    try:
        import asc_op_compile_base as asc
        from ttk.switches import get_global_storage
    except ImportError:
        return None
    switches = get_global_storage()
    if switches.dev_plat in (None, "", "AUTO"):
        return None
    platform = asc.common.platform.platform_info
    platform.set_current_compile_soc_info(switches.dev_plat, "AiCore")
    return platform, getattr(switches, "core_limit", None)


def _bf16_vector_platform():
    active = _ttk_platform()
    if active is None:
        return None
    platform, core_limit = active
    cores = int(platform.get_soc_spec("VECTOR_CORE_CNT"))
    ub_size = int(platform.get_soc_spec("UB_SIZE"))
    if core_limit and core_limit[1] is not None:
        cores = min(cores, core_limit[1])
    return cores, ub_size


def _uses_950_cube_dot(tokens, capacity, slots, hidden_size):
    # The smallest MIX tile packs at most 16 residuals into an MM1 column tile.
    if tokens < 32 or capacity > 16 or slots < 16 or hidden_size < 256:
        return False
    if tokens * hidden_size > numpy.iinfo(numpy.uint32).max:
        return False
    active = _ttk_platform()
    if active is None:
        return False
    platform, core_limit = active
    if not str(platform.get_soc_spec("SOC_VERSION")).startswith("Ascend950"):
        return False
    aic = int(platform.get_soc_spec("CORE_NUM"))
    aiv = int(platform.get_soc_spec("VECTOR_CORE_CNT"))
    if core_limit:
        if core_limit[0] is not None:
            aic = min(aic, core_limit[0])
        if core_limit[1] is not None:
            aiv = min(aiv, core_limit[1])
    return aic > 0 and aiv == 2 * aic


def _cube_significand_parts(values):
    fraction, exponent = numpy.frexp(numpy.abs(values))
    fraction = numpy.where(numpy.isfinite(fraction), fraction, 0.0)
    mantissa = numpy.ldexp(fraction, 24).astype(numpy.int64)
    return [((mantissa >> i) & 255) << i for i in (0, 8, 16)], exponent


def _cube_products(left, right):
    """Form the extended products from three 8-bit limbs per FP32 operand."""
    a, exp_a = _cube_significand_parts(left)
    b, exp_b = _cube_significand_parts(right)
    # Align the partial products to the high-limb exponent. The leading bit
    # is at position 46 or 47 in the product of two 24-bit significands.
    high = a[-1] * b[-1]
    discarded = 47 - _CUBE_EXTENDED_BITS + (high >= (1 << 47)).astype(numpy.int64)
    keep = -(1 << discarded)
    significand = sum((x * y) & keep for x in a for y in b)
    exact = left * right
    product = numpy.copysign(numpy.ldexp(significand, exp_a + exp_b - 48), exact)
    return numpy.where(numpy.isfinite(exact), product, exact)


def _cube_align(value, shift):
    return numpy.ldexp(numpy.trunc(numpy.ldexp(value, -shift)), shift)


def _fp32_cube_dot(history, query):
    """Accumulate Q @ V.T in increasing D order, rounding each Cube MAC."""
    tokens, valid, hidden_size = history.shape
    rows = history.reshape(tokens * valid, hidden_size)
    slots = query.shape[0]
    result = numpy.empty((slots, len(rows)), dtype=numpy.float32)
    # Bound temporary storage independently of S, T, N and D.
    for s in range(0, slots, 32):
        for r in range(0, len(rows), 128):
            dots = result[s : s + 32, r : r + 128]
            dots.fill(0.0)
            for d in range(0, hidden_size, 32):
                q = query[s : s + 32, d : d + 32].T[:, :, None]
                v = rows[r : r + 128, d : d + 32].T[:, None, :]
                for product in _cube_products(q, v):
                    # Align both addends before rounding the sum into FP32.
                    previous = dots.astype(numpy.float64)
                    magnitude = numpy.maximum(numpy.abs(previous), numpy.abs(product))
                    shift = numpy.frexp(magnitude)[1] - _CUBE_EXTENDED_BITS
                    dots[:] = _cube_align(previous, shift) + _cube_align(product, shift)
    return result.reshape(slots, tokens, valid)


def _bf16_dot_modes(tokens, capacity, slots, hidden_size, valid, platform):
    """Select row-wise (2), paired-V (1), and odd grouped-V (4) accumulators."""
    cores, ub_size = platform
    modes = numpy.full((slots, tokens, valid), 2, dtype=numpy.uint8)
    stats_bytes = 4 * 144 * 4
    cache_bytes = (2 * capacity + 16) * hidden_size + stats_bytes
    if hidden_size % _FP32_VECTOR_ELEMENTS or cache_bytes > ub_size:
        return modes
    # Match the host's T x S-block scheduling before selecting grouped VF dots.
    s_block_size = min(slots, (tokens * slots + cores - 1) // cores)
    s_block_count = (slots + s_block_size - 1) // s_block_size
    s_block_size = (slots + s_block_count - 1) // s_block_count
    if s_block_size <= 3 or valid == 1:
        return modes
    work = tokens * s_block_count
    used_cores = min(cores, work)
    work_per_core, remainder = divmod(work, used_cores)
    full_s_blocks = s_block_count * s_block_size == slots
    # Three Q/O rows may use UB headroom beyond the baseline V-cache reservation.
    three_s_bytes = 6 * hidden_size * 4 + stats_bytes
    three_s_max_n = min(
        capacity, 5, max(0, (ub_size - three_s_bytes) // (2 * hidden_size))
    )
    token_count = (
        2
        if valid == 2
        and tokens % 2 == 0
        and capacity >= 4
        and full_s_blocks
        and s_block_size % 4 == 0
        else 1
    )
    s_group = (
        3
        if 3 <= valid <= three_s_max_n and full_s_blocks and s_block_size % 3 == 0
        else 2
    )
    s_blocks_per_token_group = token_count * s_block_count
    s_per_block = s_block_size // token_count
    for core in range(used_cores):
        start = core * work_per_core + min(core, remainder)
        remaining = work_per_core + (core < remainder)
        token_group, s_block_index = divmod(start, s_blocks_per_token_group)
        token = token_group * token_count
        while remaining:
            block_count = min(remaining, s_blocks_per_token_group - s_block_index)
            s_start = s_block_index * s_per_block
            s_end = (
                slots
                if s_block_index + block_count == s_blocks_per_token_group
                else (s_block_index + block_count) * s_per_block
            )
            if token_count == 2 or (s_end - s_start) % s_group == 0:
                modes[s_start:s_end, token : token + token_count, : valid // 2 * 2] = 1
                if valid % 2:
                    modes[s_start:s_end, token : token + token_count, -1] = 4
            remaining -= block_count
            s_block_index = 0
            token += token_count
    return modes


def _bf16_vector_dot(history, query, capacity):
    """Model the selected FP32 VF dot, including its non-fused D tail."""
    tokens, valid, hidden_size = history.shape
    slots = query.shape[0]
    rows = history.reshape(tokens * valid, hidden_size)
    platform = _bf16_vector_platform()
    if platform is None:
        return (
            (rows @ query.T)
            .reshape(tokens, valid, slots)
            .transpose(2, 0, 1)
            .astype(numpy.float32)
        )
    modes = _bf16_dot_modes(
        tokens, capacity, slots, hidden_size, valid, platform
    ).reshape(-1)
    result = numpy.empty(modes.shape, dtype=numpy.float32)
    for unroll in (1, 2, 4):
        selected = numpy.flatnonzero(modes == unroll)
        full_size = (
            hidden_size
            // (unroll * _FP32_VECTOR_ELEMENTS)
            * unroll
            * _FP32_VECTOR_ELEMENTS
        )
        for start in range(0, selected.size, _DOT_BATCH_ROWS):
            indices = selected[start : start + _DOT_BATCH_ROWS]
            q_rows, v_rows = indices // rows.shape[0], indices % rows.shape[0]
            accumulators = numpy.zeros(
                (unroll, indices.size, _FP32_VECTOR_ELEMENTS), dtype=numpy.float32
            )
            for offset in range(0, full_size, unroll * _FP32_VECTOR_ELEMENTS):
                for part in range(unroll):
                    begin = offset + part * _FP32_VECTOR_ELEMENTS
                    end = begin + _FP32_VECTOR_ELEMENTS
                    # Retain the product before the FP32 multiply-add boundary.
                    accumulators[part] = (
                        query[q_rows, begin:end] * rows[v_rows, begin:end]
                        + accumulators[part].astype(numpy.float64)
                    ).astype(numpy.float32)
            for begin in range(full_size, hidden_size, _FP32_VECTOR_ELEMENTS):
                end = min(begin + _FP32_VECTOR_ELEMENTS, hidden_size)
                product = (query[q_rows, begin:end] * rows[v_rows, begin:end]).astype(
                    numpy.float32
                )
                accumulators[0, ..., : end - begin] += product
            while accumulators.shape[0] > 1:
                accumulators = accumulators[::2] + accumulators[1::2]
            reduced = accumulators[0]
            while reduced.shape[-1] > 1:
                reduced = reduced[..., ::2] + reduced[..., 1::2]
            result[indices] = reduced[..., 0]
    return result.reshape(slots, tokens, valid)


__spec__ = {
    "block_attn_res_prepare": "BlockAttnResPrepareTestSpec",
    "aclnnBlockAttnResPrepare": "BlockAttnResPrepareAclnnTestSpec",
    "cann_ops_transformer.ops.block_attn_res_prepare": "BlockAttnResPrepareE2ETestSpec",
}


def _numpy_golden(block_res, valid_blocks, pseudo_query, eps, input_dtypes=None):
    """Return FP32 outputs using the active platform's dot accumulation."""
    block_res_dtype = (
        input_dtypes[0] if input_dtypes is not None and len(input_dtypes) > 0 else None
    )
    if not hasattr(block_res, "shape"):
        block_res = numpy.asarray(block_res)
    if not hasattr(pseudo_query, "shape"):
        pseudo_query = numpy.asarray(pseudo_query)
    tokens = block_res.shape[T_DIM_INDEX]
    hidden_size = block_res.shape[D_DIM_INDEX]
    slots = pseudo_query.shape[S_DIM_INDEX]
    valid = int(
        numpy.asarray(_as_numpy(valid_blocks), dtype=numpy.uint64)
        .reshape(FLATTENED_SIZE)[0]
        .item()
    )
    valid = min(valid, block_res.shape[N_DIM_INDEX])

    if valid == 0 or tokens == 0 or slots == 0:
        # Identity state of the online softmax: zero numerator, FLOAT32 lowest
        # finite value for logitMax, zero expSum.
        numerator = numpy.zeros(
            (slots, tokens, hidden_size),
            dtype=numpy.float32,
        )
        logit_max = numpy.full(
            (slots, tokens),
            numpy.finfo(numpy.float32).min,
            dtype=numpy.float32,
        )
        exp_sum = numpy.zeros_like(logit_max)
        return numerator, logit_max, exp_sum

    # Slice before conversion, particularly for large N with few valid blocks.
    history = _numpy_input_to_fp64(block_res[:, :valid, :], block_res_dtype)
    # Promote the actual Q values; only V has an optional BF16 storage boundary.
    query = numpy.asarray(_as_numpy(pseudo_query), dtype=numpy.float64)
    with numpy.errstate(
        over="ignore", under="ignore", invalid="ignore", divide="ignore"
    ):
        # Cast before RMS normalization: an FP32 sum of squares can overflow
        # even when its FP64 mean is finite. invD and eps are FP32 attributes.
        square_sum = numpy.einsum("tnd,tnd->tn", history, history).astype(numpy.float32)
        inv_d = numpy.float32(1.0 / hidden_size)
        mean_square = square_sum * inv_d + numpy.float32(eps)
        rms = _ftz_float32(numpy.sqrt(_ftz_float32(mean_square)))
        if _is_bfloat16_dtype(block_res_dtype) or _is_bfloat16_dtype(
            getattr(block_res, "dtype", None)
        ):
            dot = _bf16_vector_dot(history, query, block_res.shape[N_DIM_INDEX])
        elif _uses_950_cube_dot(
            tokens, block_res.shape[N_DIM_INDEX], slots, hidden_size
        ):
            dot = _fp32_cube_dot(history, query)
        else:
            dot = (
                (history.reshape(tokens * valid, hidden_size) @ query.T)
                .reshape(tokens, valid, slots)
                .transpose(2, 0, 1)
                .astype(numpy.float32)
            )
        logits = _ftz_float32(
            _ftz_float32(dot) / numpy.expand_dims(_ftz_float32(rms), 0)
        )
        if valid == 1:
            # Shared Vector/MIX contract: skip exp(z-z) and the weighted sum,
            # so a nonfinite logit does not contaminate the copied residual.
            numerator = numpy.broadcast_to(
                history[None, :, 0, :], (slots, tokens, hidden_size)
            )
            return (
                numerator.astype(numpy.float32),
                logits[:, :, 0].copy(),
                numpy.ones((slots, tokens), dtype=numpy.float32),
            )
        logit_max = numpy.max(logits, axis=LAST_DIM_INDEX)
        shifted = _ftz_float32(logits - numpy.expand_dims(logit_max, LAST_DIM_INDEX))
        # Only Exp's boundary flushes these weights; the weighted sum below
        # must retain subnormal products. FP64 exp avoids NumPy FP32 libm noise.
        weights = _ftz_float32(numpy.exp(shifted.astype(numpy.float64)))
        weights = weights.astype(numpy.float64)
        exp_sum = numpy.sum(weights, axis=LAST_DIM_INDEX).astype(numpy.float32)
        if tokens == 1:
            numerator = (weights[:, 0, :] @ history[0])[:, None, :]
        else:
            numerator = numpy.einsum("stn,tnd->std", weights, history, optimize=True)
        # Apply output rounding before comparison: a nonzero FP64 value below
        # the FP32 range cannot be an attainable pure-relative-error target.
        return numerator.astype(numpy.float32), logit_max, exp_sum


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
    block_res, valid_blocks, pseudo_query, eps, input_dtypes=None, *, device
):
    """Compose the operators from the Phase2 design's performance baseline."""
    import torch

    tokens, capacity, hidden_size = block_res.shape
    slots = pseudo_query.shape[0]
    # The dynamic slice bound is host metadata. Only this scalar is read back;
    # residual/query tensors and all arithmetic stay on the requested device.
    valid = min(int(_as_numpy(valid_blocks).reshape(-1)[0]), capacity)
    if valid == 0 or tokens == 0 or slots == 0:
        numerator = torch.zeros(
            (slots, tokens, hidden_size), dtype=torch.float32, device=device
        )
        logit_max = torch.full(
            (slots, tokens),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
            device=device,
        )
        exp_sum = torch.zeros_like(logit_max)
    else:
        storage_dtype = getattr(block_res, "dtype", None)
        if input_dtypes is not None and len(input_dtypes) > 0:
            storage_dtype = input_dtypes[0]
        storage_dtype = (
            torch.bfloat16 if _is_bfloat16_dtype(storage_dtype) else torch.float32
        )
        history = _to_torch_input(block_res, storage_dtype, device)[
            :, :valid, :
        ].float()
        query = _to_torch_input(pseudo_query, torch.float32, device)
        rstd = torch.rsqrt(history.square().mean(dim=-1) + float(numpy.float32(eps)))
        # [S,D] @ [T,D,N] -> [T,S,N], with RMS statistics shared by all S slots.
        logits = torch.matmul(query, history.transpose(-1, -2)) * rstd.unsqueeze(1)
        if valid == 1:
            numerator = (
                history[:, 0, :]
                .unsqueeze(0)
                .expand(slots, tokens, hidden_size)
                .contiguous()
            )
            logit_max = logits[:, :, 0].transpose(0, 1).contiguous()
            exp_sum = torch.ones((slots, tokens), dtype=torch.float32, device=device)
            return tuple(_as_numpy(value) for value in (numerator, logit_max, exp_sum))
        maximum = logits.amax(dim=-1, keepdim=True)
        weights = torch.exp(logits - maximum)
        exp_sum = weights.sum(dim=-1).transpose(0, 1).contiguous()
        # FP32 [T,S,N] @ [T,N,D] sums E_n * V_n without materializing
        # the [T,S,N,D] product. Accumulation can differ in the last bits.
        numerator = torch.bmm(weights, history)
        numerator = numerator.transpose(0, 1).contiguous()
        logit_max = maximum.squeeze(-1).transpose(0, 1).contiguous()
    return tuple(_as_numpy(value) for value in (numerator, logit_max, exp_sum))


def _torch_golden(
    block_res, valid_blocks, pseudo_query, eps, input_dtypes=None, device=None
):
    """Use the CPU oracle or the device composition selected by TTK."""
    device = _npu_device(block_res, device)
    if device is not None:
        return _npu_golden(
            block_res, valid_blocks, pseudo_query, eps, input_dtypes, device=device
        )
    import torch

    return tuple(
        torch.from_numpy(value)
        for value in _numpy_golden(
            block_res, valid_blocks, pseudo_query, eps, input_dtypes
        )
    )


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


class BlockAttnResPrepareTestSpec:
    """CPU reference and NPU composition for the kernel-only test path.

    TTK calls this with the flat numpy input arrays of the kernel CSV
    (``block_res``, ``valid_blocks``, ``pseudo_query``) in order; ``eps``
    arrives through the CSV ``attributes`` dict (``**kwargs`` also receives
    the TTK metadata such as ``input_dtypes``/``testcase_name``).
    """

    compare = staticmethod(_compare_relative_error)

    @staticmethod
    def golden(block_res, valid_blocks, pseudo_query, eps=DEFAULT_EPS, **kwargs):
        input_dtypes = kwargs.get("input_dtypes")
        device = _npu_device(block_res, kwargs.get("device"))
        if device is not None:
            return list(
                _npu_golden(
                    block_res,
                    valid_blocks,
                    pseudo_query,
                    eps,
                    input_dtypes,
                    device=device,
                )
            )
        # TTK Promote also rewrites input_dtypes. Q is always stored as FP32,
        # so promoted (FP32 V, FP64 Q) uniquely identifies a BF16 residual.
        if (
            kwargs.get("golden_mode") == "Promote"
            and block_res.dtype == numpy.float32
            and pseudo_query.dtype == numpy.float64
        ):
            input_dtypes = ("bfloat16", "uint64", "float32")
        numerator, logit_max, exp_sum = _numpy_golden(
            block_res, valid_blocks, pseudo_query, eps, input_dtypes
        )
        # Kernel CSV output order: numerator, logit_max, exp_sum.
        return [numerator, logit_max, exp_sum]


class BlockAttnResPrepareAclnnTestSpec:
    """TestSpec registered by the exact ACLNN API name.

    ``AclnnParamPlan`` hands over the C-header parameter order
    (``blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum, eps``);
    the three output tensors are stubs on the golden path and are recomputed.
    """

    compare = staticmethod(_compare_relative_error)

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
        input_dtypes = kwargs.get("input_dtypes")
        del numerator, logitMax, expSum
        numerator_golden, logit_max_golden, exp_sum_golden = _torch_golden(
            blockRes, validBlocks, pseudoQuery, eps, input_dtypes, kwargs.get("device")
        )
        # ACLNN CSV output_tensor_indexes=(3, 4, 5).
        return [numerator_golden, logit_max_golden, exp_sum_golden]


class BlockAttnResPrepareE2ETestSpec:
    """TestSpec registered by the explicit Python wrapper API name.

    The E2E CSV only carries the three input tensors; the wrapper returns
    ``(numerator, logit_max, exp_sum)`` and TTK compares them in that order.
    """

    compare = staticmethod(_compare_relative_error)

    @staticmethod
    def golden(
        block_res,
        valid_blocks,
        pseudo_query,
        *,
        eps=DEFAULT_EPS,
        **kwargs,
    ):
        input_dtypes = kwargs.get("input_dtypes")
        numerator_golden, logit_max_golden, exp_sum_golden = _torch_golden(
            block_res,
            valid_blocks,
            pseudo_query,
            eps,
            input_dtypes,
            kwargs.get("device"),
        )
        return [numerator_golden, logit_max_golden, exp_sum_golden]
