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

__spec__ = {
    "moe_init_routing_v4": "MoeInitRoutingV4TestSpec",
    "aclnnMoeInitRoutingV4": "AclnnMoeInitRoutingV4TestSpec",
}


def _cast_to_int8_hw(arr):
    arr = numpy.nan_to_num(arr, nan=0.0, posinf=127.0, neginf=-128.0, copy=False)
    arr = numpy.clip(arr, -128.0, 127.0)
    arr = numpy.where(arr >= 0, numpy.floor(arr + 0.5), numpy.ceil(arr - 0.5))
    arr = numpy.clip(arr, -128, 127).astype(numpy.int8)
    return arr


def _moe_init_routing_v4_numpy(
    x,
    expert_idx,
    scale,
    offset,
    topk_weight,
    active_num,
    expert_capacity,
    expert_num,
    drop_pad_mode,
    expert_tokens_num_type,
    expert_tokens_num_flag,
    quant_mode,
    active_expert_range,
    row_idx_type,
):
    expert_start = int(active_expert_range[0])
    expert_end = int(active_expert_range[1])
    num_rows = x.shape[0]
    h = x.shape[1]
    k = expert_idx.shape[-1]

    if drop_pad_mode == 1:
        expert_idx_flat = expert_idx.copy().reshape(-1)
        sorted_expert_indices = numpy.argsort(expert_idx_flat, axis=-1, kind="stable")
        sorted_expert_idx = expert_idx_flat[sorted_expert_indices]

        valid_mask = (sorted_expert_idx >= expert_start) & (
            sorted_expert_idx < expert_end
        )
        valid_sorted_row_idx = sorted_expert_indices[valid_mask]
        valid_sorted_expert_idx = sorted_expert_idx[valid_mask]

        sort_row_tmp = numpy.full((expert_num * expert_capacity), -1, dtype=numpy.int64)
        expert_offset = numpy.zeros(expert_num, dtype=numpy.int64)
        for row_idx, expert_id in zip(valid_sorted_row_idx, valid_sorted_expert_idx):
            expert_id = int(expert_id)
            if expert_id < 0 or expert_id >= expert_num:
                continue
            offset_idx = expert_offset[expert_id]
            expert_offset[expert_id] += 1
            if offset_idx >= expert_capacity:
                continue
            sort_row_tmp[expert_id * expert_capacity + offset_idx] = row_idx

        expanded_row_idx = numpy.full((num_rows * k,), -1, dtype=numpy.int32)
        valid_capacity_mask = sort_row_tmp != -1
        expanded_row_idx[sort_row_tmp[valid_capacity_mask]] = numpy.arange(
            expert_num * expert_capacity, dtype=numpy.int32
        )[valid_capacity_mask]

        expanded_x = numpy.zeros((expert_num * expert_capacity, h), dtype=x.dtype)
        expanded_x[valid_capacity_mask] = x[sort_row_tmp[valid_capacity_mask] // k, :]
        expanded_x = expanded_x.reshape((expert_num, expert_capacity, h))

        expanded_topk_weight = None
        if topk_weight is not None:
            topk_weight_flat = topk_weight.reshape(num_rows * k)
            expanded_topk_weight = numpy.zeros(
                (expert_num * expert_capacity, 1), dtype=numpy.float32
            )
            expanded_topk_weight[valid_capacity_mask, 0] = topk_weight_flat[
                sort_row_tmp[valid_capacity_mask]
            ]

        expanded_scale = None
        if scale is not None:
            scale_shape = (expert_num * expert_capacity,) + scale.shape[1:]
            expanded_scale = numpy.zeros(scale_shape, dtype=scale.dtype)
            if scale.shape[0] == 1:
                expanded_scale[valid_capacity_mask] = scale[0]
            else:
                expanded_scale[valid_capacity_mask] = scale[
                    sort_row_tmp[valid_capacity_mask] // k
                ]
            if expanded_scale.ndim == 2 and expanded_scale.shape[-1] == 1:
                expanded_scale = expanded_scale.reshape(-1)

        if expert_tokens_num_flag:
            expert_tokens_count = numpy.bincount(
                valid_sorted_expert_idx - expert_start,
                minlength=expert_end - expert_start,
            )
            expert_tokens_count = expert_tokens_count[
                : expert_end - expert_start
            ].astype(numpy.int64)
        else:
            expert_tokens_count = numpy.array([], dtype=numpy.int64)
        return (
            expanded_x,
            expanded_row_idx,
            expert_tokens_count,
            expanded_scale,
            expanded_topk_weight,
        )

    # drop_pad_mode == 0 (DropLess)
    expert_idx_in = expert_idx.copy().reshape(-1)
    actual_expert_total_num = int(
        numpy.sum((expert_idx_in >= expert_start) & (expert_idx_in < expert_end))
    )

    expert_idx_in[(expert_idx_in < expert_start)] = numpy.int32(
        numpy.iinfo(numpy.int32).max
    )
    sorted_expert_indices = numpy.argsort(expert_idx_in, axis=-1, kind="stable")
    sorted_expert_idx = expert_idx_in[sorted_expert_indices]
    if row_idx_type == 1:
        expanded_row_idx = sorted_expert_indices
    else:
        expanded_row_idx = numpy.ones(num_rows * k).astype(numpy.int32) * -1
        tmp_indices = numpy.arange(actual_expert_total_num)
        expanded_row_idx[sorted_expert_indices[:actual_expert_total_num]] = tmp_indices

    expanded_topk_weight = None
    if topk_weight is not None:
        topk_weight_flat = topk_weight.reshape(num_rows * k)
        effective_num = (
            min(active_num, int(actual_expert_total_num))
            if active_num > 0
            else int(actual_expert_total_num)
        )
        expanded_topk_weight = topk_weight_flat[
            sorted_expert_indices[:effective_num]
        ].reshape(-1, 1)

    if quant_mode == -1:
        if scale is None:
            expanded_scale = None
        else:
            expanded_scale = scale[
                sorted_expert_indices[:actual_expert_total_num] // k
            ].flatten()
        expanded_x = x[sorted_expert_indices[:actual_expert_total_num] // k, :]

    elif quant_mode == 1:
        expanded_x = x[sorted_expert_indices // k, :]
        expanded_x = expanded_x.astype(numpy.float32)
        if scale is None:
            expanded_x = expanded_x[:actual_expert_total_num, :]
            x_abs = numpy.abs(expanded_x)
            x_max = numpy.max(x_abs, axis=-1, keepdims=True)
            expanded_scale = x_max / 127.0
            expanded_x = expanded_x / expanded_scale
            expanded_x = numpy.nan_to_num(
                expanded_x, nan=0.0, posinf=127.0, neginf=-128.0, copy=False
            )
            expanded_x = _cast_to_int8_hw(expanded_x)
        else:
            expended_scale = scale[
                sorted_expert_idx[:actual_expert_total_num] - expert_start, :
            ]
            expanded_x = expanded_x[:actual_expert_total_num, :]
            expanded_x = expanded_x * expended_scale
            x_abs = numpy.abs(expanded_x)
            x_max = numpy.max(x_abs, axis=-1, keepdims=True)
            expanded_scale = x_max / 127.0
            expanded_x = expanded_x / expanded_scale
            expanded_x = numpy.nan_to_num(
                expanded_x, nan=0.0, posinf=127.0, neginf=-128.0, copy=False
            )
            expanded_x = _cast_to_int8_hw(expanded_x)

    else:
        if scale is None:
            expanded_scale = None
        else:
            expanded_scale = scale[
                sorted_expert_indices[:actual_expert_total_num] // k
            ].flatten()
        expanded_x = x[sorted_expert_indices[:actual_expert_total_num] // k, :]

    if expert_tokens_num_type == 0:
        counts = numpy.bincount(
            sorted_expert_idx[:actual_expert_total_num] - expert_start,
            minlength=expert_end - expert_start,
        )
        expert_tokens_count = numpy.cumsum(counts).astype(numpy.int64)
    elif expert_tokens_num_type == 1:
        expert_tokens_count = numpy.bincount(
            sorted_expert_idx[:actual_expert_total_num] - expert_start
        )
        expert_tokens_count = numpy.concatenate(
            [
                expert_tokens_count,
                numpy.zeros(
                    max(0, (expert_end - expert_start) - len(expert_tokens_count))
                ).astype(numpy.int64),
            ]
        )
    elif expert_tokens_num_type == 2:
        expert_id, counts = numpy.unique(
            sorted_expert_idx[:actual_expert_total_num], return_counts=True
        )
        expert_tokens_count = numpy.column_stack((expert_id, counts))
        if expert_tokens_count.shape[0] < expert_num:
            expert_tokens_count = numpy.concatenate(
                (expert_tokens_count, [[0, 0]]), axis=0
            )
    else:
        expert_tokens_count = numpy.array([], dtype=numpy.int64)

    return (
        expanded_x,
        expanded_row_idx.astype(numpy.int32),
        expert_tokens_count.astype(numpy.int64),
        expanded_scale,
        expanded_topk_weight,
    )


class MoeInitRoutingV4TestSpec:
    """MoeInitRoutingV4 kernel test spec"""

    @staticmethod
    def golden(
        x,
        expert_idx,
        scale,
        offset,
        active_num,
        topk_weight,
        *,
        expert_capacity=-1,
        expert_num=-1,
        drop_pad_mode=0,
        expert_tokens_num_type=0,
        expert_tokens_num_flag=False,
        quant_mode=-1,
        active_expert_range=None,
        row_idx_type=0,
        **kwargs,
    ):
        if active_num is None:
            active_num_val = -1
        else:
            # TTK may provide scalar inputs as a one-element NumPy array.
            # Normalize that representation before converting to Python int.
            active_num_val = int(numpy.asarray(active_num).item())
        scale_np = numpy.asarray(scale) if scale is not None else None
        offset_np = numpy.asarray(offset) if offset is not None else None
        topk_weight_np = numpy.asarray(topk_weight) if topk_weight is not None else None

        return _moe_init_routing_v4_numpy(
            numpy.asarray(x),
            numpy.asarray(expert_idx),
            scale_np,
            offset_np,
            topk_weight_np,
            active_num_val,
            expert_capacity,
            expert_num,
            drop_pad_mode,
            expert_tokens_num_type,
            expert_tokens_num_flag,
            quant_mode,
            active_expert_range,
            row_idx_type,
        )

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "int8": {"standard": "stat_rel_err"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }

    @staticmethod
    def pre_compare(
        expanded_x,
        expanded_row_idx,
        expert_tokens_count,
        expanded_scale,
        expanded_topk_weight,
        g_expanded_x,
        g_expanded_row_idx,
        g_expert_tokens_count,
        g_expanded_scale,
        g_expanded_topk_weight,
    ):
        out = [
            expanded_x,
            expanded_row_idx,
            expert_tokens_count,
            expanded_scale,
            expanded_topk_weight,
        ]
        gold = [
            g_expanded_x,
            g_expanded_row_idx,
            g_expert_tokens_count,
            g_expanded_scale,
            g_expanded_topk_weight,
        ]
        for i in range(len(out)):
            if (
                out[i] is not None
                and hasattr(out[i], "dtype")
                and str(out[i].dtype) == "int8"
            ):
                out[i] = out[i].astype(numpy.float32)
            if (
                gold[i] is not None
                and hasattr(gold[i], "dtype")
                and str(gold[i].dtype) == "int8"
            ):
                gold[i] = gold[i].astype(numpy.float32)
        return tuple(out + gold)


class AclnnMoeInitRoutingV4TestSpec:
    """MoeInitRoutingV4 aclnn test spec"""

    @staticmethod
    def golden(
        x,
        expertIdx,
        scaleOptional=None,
        offsetOptional=None,
        activeNumOptional=None,
        topkWeightOptional=None,
        expertCapacity=-1,
        expertNum=-1,
        dropPadMode=0,
        expertTokensNumType=0,
        expertTokensNumFlag=False,
        quantMode=-1,
        activeExpertRangeOptional=None,
        rowIdxType=0,
        expandedXOut=None,
        expandedRowIdxOut=None,
        expertTokensCountOrCumsumOut=None,
        expandedScaleOut=None,
        expandedTopkWeightOut=None,
        *args,
        **kwargs,
    ):
        def _to_np(t):
            if t is None:
                return None
            if hasattr(t, "numpy"):
                try:
                    return t.numpy()
                except (TypeError, RuntimeError):
                    import torch

                    return t.to(torch.float32).numpy()
            return numpy.asarray(t)

        def _get_shape(t):
            if t is None:
                return None
            if hasattr(t, "shape"):
                return tuple(t.shape)
            return None

        def _pad_to_shape(arr, target_shape, pad_value=0):
            if arr is None or target_shape is None:
                return arr
            if len(target_shape) == 0:
                return arr
            padded = numpy.full(
                target_shape,
                pad_value,
                dtype=arr.dtype if arr is not None else numpy.float32,
            )
            slices = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
            padded[slices] = arr[slices] if arr is not None else 0
            return padded

        active_num_val = int(activeNumOptional) if activeNumOptional is not None else -1
        active_expert_range_val = (
            list(activeExpertRangeOptional)
            if activeExpertRangeOptional is not None
            else [0, expertNum]
        )

        x_np = _to_np(x)
        results = _moe_init_routing_v4_numpy(
            x_np,
            _to_np(expertIdx),
            _to_np(scaleOptional),
            _to_np(offsetOptional),
            _to_np(topkWeightOptional),
            active_num_val,
            expertCapacity,
            expertNum,
            dropPadMode,
            expertTokensNumType,
            expertTokensNumFlag,
            quantMode,
            active_expert_range_val,
            rowIdxType,
        )

        (
            expanded_x,
            expanded_row_idx,
            expert_tokens_count,
            expanded_scale,
            expanded_topk_weight,
        ) = results

        x_out_shape = _get_shape(expandedXOut)
        row_idx_out_shape = _get_shape(expandedRowIdxOut)
        scale_out_shape = _get_shape(expandedScaleOut)
        topk_weight_out_shape = _get_shape(expandedTopkWeightOut)

        if x_out_shape is not None and expanded_x is not None:
            if expanded_x.shape[0] < x_out_shape[0] or (
                len(x_out_shape) > 1 and expanded_x.ndim < len(x_out_shape)
            ):
                expanded_x = _pad_to_shape(
                    expanded_x.reshape(x_out_shape)
                    if len(x_out_shape) > 1
                    else expanded_x,
                    x_out_shape,
                )
            elif expanded_x.shape != x_out_shape:
                expanded_x = _pad_to_shape(expanded_x, x_out_shape)

        if row_idx_out_shape is not None and expanded_row_idx is not None:
            if expanded_row_idx.shape[0] < row_idx_out_shape[0]:
                expanded_row_idx = _pad_to_shape(
                    expanded_row_idx, row_idx_out_shape, -1
                )

        if scale_out_shape is not None and expanded_scale is not None:
            if expanded_scale.shape != scale_out_shape:
                expanded_scale = _pad_to_shape(expanded_scale, scale_out_shape)

        if topk_weight_out_shape is not None and expanded_topk_weight is not None:
            if expanded_topk_weight.shape != topk_weight_out_shape:
                expanded_topk_weight = _pad_to_shape(
                    expanded_topk_weight, topk_weight_out_shape
                )

        return (
            expanded_x,
            expanded_row_idx,
            expert_tokens_count,
            expanded_scale,
            expanded_topk_weight,
        )

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "int8": {"standard": "stat_rel_err"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }
