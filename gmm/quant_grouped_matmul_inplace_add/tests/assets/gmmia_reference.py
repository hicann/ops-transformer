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
"""CPU reference shared by the three GMMIA test entry points."""

import numpy as np


def as_array(value, carrier=None):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        value = value.float() if "float8" in str(value.dtype) else value
        value = value.numpy()
    result = np.asarray(value)
    if carrier == 290:
        import en_dtypes

        result = result.view(en_dtypes.hifloat8)
    elif carrier == 293:
        return np.exp2(result.astype(np.float32) - 127)
    return result.astype(np.float32)


def inplace(x1, x2, scale1, scale2, groups, y, group_list_type):
    groups = np.asarray(groups, dtype=np.int64)
    counts = groups if group_list_type == 1 else np.diff(np.r_[0, groups])
    output = y.copy()
    start = 0
    mx = scale2.ndim == 3
    for expert, count in enumerate(counts):
        stop = start + int(count)
        if not count:
            continue
        a, b = x1[start:stop], x2[start:stop]
        if mx:
            offset = start // 64 + expert
            blocks = (int(count) + 63) // 64
            a_scale = (
                scale1[offset : offset + blocks].swapaxes(1, 2).reshape(-1, a.shape[1])
            )
            b_scale = (
                scale2[offset : offset + blocks].swapaxes(1, 2).reshape(-1, b.shape[1])
            )
            a = a * np.repeat(a_scale, 32, axis=0)[:count]
            b = b * np.repeat(b_scale, 32, axis=0)[:count]
            output[expert] += a.T @ b
        else:
            out = (a.T @ b) * scale2[expert]
            if scale1 is not None:
                out *= scale1.reshape(-1)[expert]
            output[expert] += out
        start = stop
    return [output]
