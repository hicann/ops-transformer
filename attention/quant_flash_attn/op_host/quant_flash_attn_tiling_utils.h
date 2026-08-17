/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_tiling_utils.h
 * \brief Tiling utility templates used by quant_flash_attn. Local to
 *        quant_flash_attn; only the ToString(gert::Shape) call chain actually
 *        referenced by the op_host sources is retained.
 */

#ifndef QUANT_FLASH_ATTN_TILING_UTILS_H
#define QUANT_FLASH_ATTN_TILING_UTILS_H

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
#include <exe_graph/runtime/tiling_context.h>

namespace optiling {

template <typename T>
inline auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

static std::vector<int64_t> ToVector(const gert::Shape &shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

template <typename T>
static std::string ToString(const std::vector<T> &v)
{
    std::ostringstream oss;
    oss << "[";
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    oss << "]";
    return oss.str();
}

inline std::string ToString(const gert::Shape &shape)
{
    return ToString(ToVector(shape));
}

} // namespace optiling

#endif // QUANT_FLASH_ATTN_TILING_UTILS_H
