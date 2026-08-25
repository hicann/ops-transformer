#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy

try:
    from ttk.utilities.dtypes import (
        numpy_float8_e4m3fn,
        numpy_float8_e5m2,
        numpy_float8_e8m0,
    )

    _FP8_E4M3 = numpy_float8_e4m3fn()
    _FP8_E5M2 = numpy_float8_e5m2()
    _FP8_E8M0 = numpy_float8_e8m0()
except Exception:
    _FP8_E4M3 = None
    _FP8_E5M2 = None
    _FP8_E8M0 = None

__spec__ = {
    "moe_re_routing_v2": "MoeReRoutingV2TestSpec",
}


def moe_re_routing_v2_numpy(
    tokens,
    expert_token_num_per_rank,
    idx_type,
    per_token_scales=None,
    expert_topk_weight=None,
):
    token_num, token_length = tokens.shape
    rank_num, expert_num = expert_token_num_per_rank.shape

    permute_tokens = numpy.zeros((token_num, token_length), dtype=tokens.dtype)
    permute_token_idx = numpy.zeros(token_num, dtype=numpy.int32)

    if per_token_scales is not None:
        if per_token_scales.ndim == 1:
            permute_per_token_scales = numpy.zeros(
                token_num, dtype=per_token_scales.dtype
            )
        else:
            scalar_length = per_token_scales.shape[1]
            permute_per_token_scales = numpy.zeros(
                (token_num, scalar_length), dtype=per_token_scales.dtype
            )
    else:
        permute_per_token_scales = None

    if expert_topk_weight is not None:
        permute_topk_weight = numpy.zeros_like(expert_topk_weight)
    else:
        permute_topk_weight = None

    expert_token_num = numpy.sum(expert_token_num_per_rank, axis=0)
    per_expert_offset = numpy.zeros(expert_num, dtype=expert_token_num_per_rank.dtype)
    per_expert_offset[1:] = numpy.cumsum(expert_token_num)[:-1]

    src_offset = 0
    for cur_rank in range(rank_num):
        for expert in range(expert_num):
            num_tokens = int(expert_token_num_per_rank[cur_rank, expert])
            if num_tokens == 0:
                continue
            dst_start = int(per_expert_offset[expert])
            dst_end = dst_start + num_tokens
            src_end = src_offset + num_tokens

            permute_tokens[dst_start:dst_end] = tokens[src_offset:src_end]
            if idx_type == 0:
                permute_token_idx[dst_start:dst_end] = numpy.arange(
                    src_offset, src_end, dtype=numpy.int32
                )
            else:
                permute_token_idx[src_offset:src_end] = numpy.arange(
                    dst_start, dst_end, dtype=numpy.int32
                )

            if per_token_scales is not None:
                permute_per_token_scales[dst_start:dst_end] = per_token_scales[
                    src_offset:src_end
                ]

            if expert_topk_weight is not None:
                permute_topk_weight[dst_start:dst_end] = expert_topk_weight[
                    src_offset:src_end
                ]

            src_offset = src_end
            per_expert_offset[expert] = dst_end

    return (
        permute_tokens,
        permute_per_token_scales,
        permute_token_idx,
        expert_token_num,
        permute_topk_weight,
    )


class MoeReRoutingV2TestSpec:
    """MoeReRoutingV2 kernel test spec"""

    @staticmethod
    def golden(
        tokens,
        expert_token_num_per_rank,
        per_token_scales,
        expert_topk_weight,
        *,
        expert_token_num_type=1,
        idx_type=0,
        **kwargs,
    ):
        idx_type_val = int(idx_type) if idx_type is not None else 0
        per_token_scales_np = per_token_scales if per_token_scales is not None else None
        expert_topk_weight_np = (
            expert_topk_weight if expert_topk_weight is not None else None
        )

        return moe_re_routing_v2_numpy(
            tokens,
            expert_token_num_per_rank,
            idx_type_val,
            per_token_scales_np,
            expert_topk_weight_np,
        )

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "int8": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }

    @staticmethod
    def customize_inputs(
        tokens,
        expert_token_num_per_rank,
        per_token_scales,
        expert_topk_weight,
        *,
        input_dtypes=None,
        input_ranges=None,
        **kwargs,
    ):
        dt_list = [str(d) for d in (input_dtypes if input_dtypes else ())]
        needs_fp8 = any("float8" in d for d in dt_list)
        if not needs_fp8:
            return (
                tokens,
                expert_token_num_per_rank,
                per_token_scales,
                expert_topk_weight,
            )

        def _gen(dtype_str, data_range, shape):
            if shape is None or (isinstance(shape, tuple) and None in shape):
                return None
            lo, hi = data_range if data_range else (-1.0, 1.0)
            if "float8" in dtype_str:
                fp8_map = {
                    "float8_e4m3fn": _FP8_E4M3,
                    "float8_e5m2": _FP8_E5M2,
                    "float8_e8m0": _FP8_E8M0,
                }
                fp8_dt = fp8_map.get(dtype_str)
                if fp8_dt is None:
                    return None
                return (
                    numpy.random.uniform(lo, hi, shape)
                    .astype(numpy.float32)
                    .astype(fp8_dt)
                )
            if dtype_str in ("int32", "int64"):
                return numpy.random.randint(
                    int(lo), int(max(hi, lo + 1)), shape, dtype=numpy.dtype(dtype_str)
                )
            return numpy.random.uniform(lo, hi, shape).astype(numpy.dtype(dtype_str))

        def _shape(x):
            if x is None:
                return None
            if hasattr(x, "shape"):
                return tuple(x.shape)
            return x

        ranges = list(input_ranges) if input_ranges else [None] * 4
        new_tokens = _gen(
            dt_list[0] if len(dt_list) > 0 else "",
            ranges[0] if len(ranges) > 0 else None,
            _shape(tokens),
        )
        new_etr = _gen(
            dt_list[1] if len(dt_list) > 1 else "",
            ranges[1] if len(ranges) > 1 else None,
            _shape(expert_token_num_per_rank),
        )
        new_pts = _gen(
            dt_list[2] if len(dt_list) > 2 else "",
            ranges[2] if len(ranges) > 2 else None,
            _shape(per_token_scales),
        )
        new_etw = _gen(
            dt_list[3] if len(dt_list) > 3 else "",
            ranges[3] if len(ranges) > 3 else None,
            _shape(expert_topk_weight),
        )
        return (new_tokens, new_etr, new_pts, new_etw)

    @staticmethod
    def pre_compare(
        permute_tokens,
        permute_per_token_scales,
        permute_token_idx,
        expert_token_num,
        permute_topk_weight,
        g_permute_tokens,
        g_permute_per_token_scales,
        g_permute_token_idx,
        g_expert_token_num,
        g_permute_topk_weight,
    ):
        out = [
            permute_tokens,
            permute_per_token_scales,
            permute_token_idx,
            expert_token_num,
            permute_topk_weight,
        ]
        gold = [
            g_permute_tokens,
            g_permute_per_token_scales,
            g_permute_token_idx,
            g_expert_token_num,
            g_permute_topk_weight,
        ]
        for i in range(len(out)):
            for lst in (out, gold):
                if lst[i] is not None and hasattr(lst[i], "dtype"):
                    dt_str = str(lst[i].dtype)
                    if (
                        "float8" in dt_str
                        or "e4m3" in dt_str
                        or "e5m2" in dt_str
                        or "e8m0" in dt_str
                    ):
                        lst[i] = lst[i].astype(numpy.float32)
        return tuple(out + gold)
