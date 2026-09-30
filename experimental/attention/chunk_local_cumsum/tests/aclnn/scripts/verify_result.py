# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import math
import os
import sys

import numpy as np
import torch


def get_abs_err(x, y):
    return (x - y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x - y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def assert_close(prefix, ref, tri, ratio):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert get_err_ratio(ref, tri) < ratio, msg


def classify_data(data):
    # 初始化分类字典，用于存放不同数量级的数据
    length = len(data)
    classified_data = {}

    for i in range(length):
        value = data[i]
        # 计算数据的数量级，这里简单地使用对数log10作为数量级
        # print(value)
        magnitude = math.floor(math.log10(abs(value + 1e-10)))  # 使用math库

        # 将数据添加到相应数量级的分类中
        if magnitude in classified_data:
            classified_data[magnitude].append(i)
        else:
            classified_data[magnitude] = [i]
    classified_data = dict(sorted(classified_data.items()))
    return classified_data


def relative_error_percent(in1, in2):
    print(in1.shape, in2.shape)
    print(in1[:128])
    print(in2[:128])

    print(in1[-128:])
    print(in2[-128:])

    in1_magnitude = classify_data(in2)
    for magnitude, idx_set in in1_magnitude.items():
        print(
            f"======== 数量级：{10**magnitude}~{10 ** (magnitude + 1)} 数量: {len(idx_set)} ========"
        )
        in1_subset = in1[idx_set]
        in2_subset = in2[idx_set]

        error = abs(in1_subset - in2_subset)
        absolute_mean_error = np.mean(error)
        print(f"absolute error: {absolute_mean_error}")
        error_percent1 = np.mean(error / abs(in1_subset) * 100)
        error_percent2 = np.mean(error / abs(in2_subset) * 100)
        print(f"相对in1的误差百分比: {error_percent1:.4f} %")
        print(f"相对in2的误差百分比: {error_percent2:.4f} %")


def read(real_result, golden, dtype=np.float16):
    real_result = np.fromfile(real_result, dtype).reshape(64, -1)
    golden = np.fromfile(golden, dtype).reshape(64, -1)

    print(real_result[48])
    print(golden[48])  # 前50个是没问题的


def verify_result(real_result, golden, dtype=np.float32):
    real_result = np.fromfile(real_result, dtype)  # .reshape(16, -1)[:, :1]#.squeeze()
    golden = np.fromfile(golden, dtype)  # .reshape(16, -1)[:,:1]#.squeeze()

    print("a: ", real_result)
    print("b: ", golden)
    # np.testing.assert_allclose(real_result, golden, rtol=1e-3, atol=1e-3, err_msg="result error")
    assert_close(" o", torch.from_numpy(golden), torch.from_numpy(real_result), 0.004)
    print("test pass!")
    # real_result = real_result.reshape(64,16,16)[30:46] # 前16 个数不一样
    # golden = golden.reshape(64,16,16)[30:46]
    # real_result = real_result.reshape(-1)
    # golden = golden.reshape(-1)
    # print(f"real_result: {real_result.shape}")
    # print(f"golden: {golden.shape}")
    # print("overall difference", real_result - golden)
    # print("overall difference mean", np.abs(real_result - golden).mean())

    # relative_error_percent(real_result, golden)

    abs_error = np.abs(real_result - golden)
    abs_mean_error = np.mean(abs_error)

    print(f"abs_mean_error: {abs_mean_error}")

    return True


if __name__ == "__main__":
    # read(sys.argv[1],sys.argv[2])
    verify_result(sys.argv[1], sys.argv[2])
