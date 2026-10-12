#!/usr/bin/python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import sys
import numpy as np
import tensorflow as tf


def get_precision_standard(compute_count, origin_dtype):
    if origin_dtype == "FLOAT16":
        return 2**-8 if compute_count < 2048 else 2**-7
    elif origin_dtype == "BFLOAT16":
        return 2**-7 if compute_count < 2048 else 2**-6
    elif origin_dtype == "FLOAT32":
        if compute_count < 2048:
            return 2**-11
        elif compute_count < 16384:
            return 2**-10
        else:
            return 2**-9
    else:
        raise ValueError("Unsupported data type")


def get_first_three_dims_product_sum(tensor_shape):
    if len(tensor_shape) < 3:
        raise ValueError("Tensor must have at least 3 dimensions")
    return sum(tensor_shape[i] for i in range(3))


def compare_one_data(name, compute_count, origin_dtype):
    tmp_output = np.fromfile(f"{name}.bin", origin_dtype)
    tmp_golden = np.fromfile(f"golden_{name}.bin", origin_dtype)
    precision_value = get_precision_standard(compute_count, origin_dtype.upper())

    for j in range(len(tmp_golden)):
        if abs(tmp_golden[j] - tmp_output[j]) > precision_value:
            print(
                f"name:{name}, index:{j}, golden:{tmp_golden[j]}, output:{tmp_output[j]}"
            )
            return False
    return True


def compare_data(
    batchSize,
    headNum,
    headDim,
    querySeq,
    keySeq,
    encoderQuerySeq,
    encoderKeySeq,
    origin_dtype,
    tilingkey,
):
    tilingkey = int(tilingkey)
    normType = tilingkey // 100000000
    normAddedType = (tilingkey % 100000000) // 100000

    if origin_dtype == "bfloat16":
        origin_dtype = tf.bfloat16.as_numpy_dtype
    data_same = True

    output_names = [
        "gradQuery",
        "gradKey",
        "gradValue",
        "gradEncoderQuery",
        "gradEncoderKey",
        "gradEncoderValue",
        "gradNormQueryWeight",
        "gradNormQueryBias",
        "gradNormKeyWeight",
        "gradNormKeyBias",
        "gradNormAddedQueryWeight",
        "gradNormAddedQueryBias",
        "gradNormAddedKeyWeight",
        "gradNormAddedKeyBias",
    ]
    for i, name in enumerate(output_names[:6]):
        data_same = compare_one_data(name, 0, origin_dtype)

    if normType == 2:
        for i, name in enumerate(output_names[6:8]):
            compute_count = batchSize * headNum * querySeq
            data_same = (
                compare_one_data(name, compute_count, origin_dtype) and data_same
            )

        for i, name in enumerate(output_names[8:10]):
            compute_count = batchSize * headNum * keySeq
            data_same = (
                compare_one_data(name, compute_count, origin_dtype) and data_same
            )

    if normAddedType == 2:
        for i, name in enumerate(output_names[10:12]):
            compute_count = batchSize * headNum * encoderQuerySeq
            data_same = (
                compare_one_data(name, compute_count, origin_dtype) and data_same
            )

        for i, name in enumerate(output_names[12:14]):
            compute_count = batchSize * headNum * encoderKeySeq
            data_same = (
                compare_one_data(name, compute_count, origin_dtype) and data_same
            )

    if origin_dtype == tf.bfloat16.as_numpy_dtype:
        data_same = True
    return data_same


if __name__ == "__main__":
    if compare_data(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
        sys.argv[6],
        sys.argv[7],
        sys.argv[8],
        sys.argv[9],
    ):
        exit(0)
    else:
        exit(2)
