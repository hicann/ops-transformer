# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Source baseline coordinate leaves shared by fixed and normalized TND plans.

The planner supplies exact geometry and round bounds. These helpers never infer
schedule kind from the public mask or omit physical rounds.
"""

from schedule.deterministic.fag_deter_formulas import (
    pl,
    cal_dense,
    cal_causal,
    cal_band,
    cal_gqa_band,
    _unpack_wxy,
    CAUSAL_BOUNDARY_DEVICE_INTEGER_SQRT,
)


def cal_source_mha_dense(k, m, n, b, j, r):
    return cal_dense(k, m, n, b, j, r)


def cal_source_mha_causal(k, m, n, b, j, r):
    packed = cal_causal(k, m, n, b, j, r, CAUSAL_BOUNDARY_DEVICE_INTEGER_SQRT)
    return _unpack_wxy(packed, m, n)


def cal_source_mha_band(k, m, n, p, q, b, j, r):
    packed = cal_band(k, m, n, p, q, b, j, r)
    return _unpack_wxy(packed, m, n)


def cal_source_gqa_band(k, m, n, p, q, b, g, j, r):
    # k is the source clamp computed before translating the band geometry.
    packed = cal_gqa_band(k, m, n, p, q, b, g, j, r)
    return _unpack_wxy(packed, m, n)


def source_baseline_valid(s1, s2, token1, token2, x, y):
    """Source tile-valid predicate; x/y are translated zero-based coordinates."""
    valid = pl.astype(0, pl.DT_INT64)
    if x >= 0 and y >= 0 and x * 128 < s1 and y * 128 < s2:
        if mask_mode == 0:
            valid = pl.astype(1, pl.DT_INT64)
        elif mask_mode == 3:
            end = pl.min(pl.max(s2 - s1 + 128 * (x + 1), 0), s2)
            if y * 128 < end:
                valid = pl.astype(1, pl.DT_INT64)
        else:
            left = (pl.max(128 * x - token1, 0) // 64) * 64
            right = ((pl.min(128 * (x + 1), s1) + token2 + 63) // 64) * 64
            right = pl.min(right, s2)
            if y * 128 < right and pl.min((y + 1) * 128, s2) > left:
                valid = pl.astype(1, pl.DT_INT64)
    return valid
