/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_pre_backward_arch22_tiling_utils.h
 * \brief Common utilities for arch22 tiling
 */
#ifndef MHC_PRE_BACKWARD_OP_HOST_OP_TILING_ARCH22_TILING_UTILS_H
#define MHC_PRE_BACKWARD_OP_HOST_OP_TILING_ARCH22_TILING_UTILS_H

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"

inline std::map<ge::DataType, uint64_t> kDataSizeMap = {
    {ge::DT_FLOAT, sizeof(float)}, {ge::DT_INT32, sizeof(int32_t)}, {ge::DT_INT64, sizeof(int64_t)}};

/**
 * if b is 0, return a
 */
template <typename T>
inline T DivCeil(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

/**
 * if b is 0, return 0
 */
template <typename T>
inline T CeilAlign(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return DivCeil(a, b) * b;
}

/**
 * if b is 0, return a
 */
template <typename T>
inline T DivFloor(T a, T b)
{
    return b == 0 ? a : a / b;
}

/**
 * if b is 0, return 0
 */
template <typename T>
inline T FloorAlign(T a, T b)
{
    return b == 0 ? 0 : a / b * b;
}

/**
 * \brief Compute factorial of n
 * \param n input value
 * \return n! (factorial of n)
 */
inline int64_t Factorial(int64_t n)
{
    if (n <= 1) {
        return 1;
    }
    int64_t result = 1;
    for (int64_t i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

#endif // MHC_PRE_BACKWARD_OP_HOST_OP_TILING_ARCH22_TILING_UTILS_H
