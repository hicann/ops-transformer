#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__input__ = {"kernel": {"quant_reduce_scatter": "quant_reduce_scatter_inputs"}}


def quant_reduce_scatter_inputs(
    x,
    scales,
    group: str,
    reduce_op: str = "sum",
    output_dtype: int = 1,
    world_size: int = 1,
    **kwargs,
):
    """
    QuantReduceScatter inputs validation and adjustment

    Args:
        x: quantized input tensor
        scales: dequantization scales
        group: communication group (HCCL group)
        reduce_op: reduce operation type ("sum", "max", "min")
        output_dtype: output data type (0: fp16, 1: bf16, 2: fp32)
        world_size: number of ranks in the group
        **kwargs: additional parameters

    Returns:
        Tuple of validated and adjusted parameters
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    if x.shape[0] % world_size != 0:
        raise ValueError(
            f"x.shape[0] ({x.shape[0]}) must be divisible by world_size ({world_size})"
        )

    valid_reduce_ops = ["sum", "max", "min"]
    if reduce_op not in valid_reduce_ops:
        raise ValueError(
            f"reduce_op must be one of {valid_reduce_ops}, got {reduce_op}"
        )

    valid_output_dtypes = [0, 1, 2]
    if output_dtype not in valid_output_dtypes:
        raise ValueError(
            f"output_dtype must be one of {valid_output_dtypes}, got {output_dtype}"
        )

    is_mxFp = kwargs.get("is_mxFp", False)
    group_size = 32 if is_mxFp else 128

    expected_scale_size = (x.shape[-1] + group_size - 1) // group_size
    if scales.shape[-1] != expected_scale_size:
        pass

    return x, scales, group, reduce_op, output_dtype, world_size
