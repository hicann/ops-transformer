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
 * \file apace_common_utils.h
 * \brief Host-side argument parsing, shape helpers, and error utilities for matmul examples.
 */

#pragma once
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace mm {
// Quantized MX element layouts for non-type template parameters.
enum class DataType {
    DT_FLOAT4_E2M1,
    DT_FLOAT8_E4M3FN,
};
} // namespace mm

#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)
#define CHECK_COND(cond, msg) \
    do { \
        if (!(cond)) { \
            throw std::runtime_error(std::string("Error: ") + msg + "\nFile: " + __FILE__ + \
                                     "\nLine: " + std::to_string(__LINE__)); \
        } \
    } while (0)

template <typename T>
inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return a / b + static_cast<T>(a % b != 0);
}

template <typename T>
inline T Align(T a, T b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
inline T FloorAlign(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return a / b * b;
}

enum class DataType {
    FP4,
    FP8
};

template <mm::DataType dataType, typename T>
constexpr T GetShapeWithDataType(T size)
{
    if constexpr (dataType == mm::DataType::DT_FLOAT4_E2M1) {
        return size << 1UL;
    } else {
        return size;
    }
}

template <mm::DataType dataType, typename T>
constexpr T GetSizeWithDataType(T shape)
{
    if constexpr (dataType == mm::DataType::DT_FLOAT4_E2M1) {
        return (shape + 1) >> 1UL;
    } else {
        return shape;
    }
}
