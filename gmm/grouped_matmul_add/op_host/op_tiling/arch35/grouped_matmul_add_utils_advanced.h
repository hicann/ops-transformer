/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_add_utils_advanced.h
 * \brief
 */

#ifndef __GROUPED_MATMUL_ADD_UTILS_ADVANCED_H
#define __GROUPED_MATMUL_ADD_UTILS_ADVANCED_H

#include <map>

namespace GroupedMatmulAdd {
constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
constexpr uint64_t STEP_K_DEFAULT = 4UL;
constexpr uint64_t DEPTH_DEFAULT = 8UL;
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint32_t MIN_DIM = 2U;
constexpr uint64_t BASE_M_DEFAULT = 256UL;
constexpr uint64_t BASE_N_DEFAULT = 256UL;
constexpr uint64_t BASE_K_DEFAULT = 64UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint32_t INDEX_X = 0U;
constexpr uint32_t INDEX_WEIGHT = 1U;
constexpr uint32_t MAX_TENSOR = 1024U;
constexpr uint32_t INDEX_GROUPLIST = 2U;
constexpr uint32_t INDEX_YREF = 3U;
constexpr uint32_t DIM_ONE = 1U;
constexpr uint32_t DIM_TWO = 2U;
constexpr uint32_t DIM_THREE = 3U;
constexpr uint32_t DIM_FOUR = 4U;
constexpr uint64_t ATTR_IDX_TRANS_X = 0UL;
constexpr uint64_t ATTR_IDX_TRANS_W = 1UL;
constexpr uint64_t ATTR_IDX_GROUPTYPE = 2UL;
constexpr uint64_t ATTR_IDX_GROUP_LIST_TYPE = 3UL;
constexpr int32_t SPLIT_K = 2;

template <typename T>
auto CeilDiv(T a, T b) -> T
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
auto CeilAlign(T a, T b) -> T
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

template <typename T>
auto FloorAlign(T x, T align) -> typename std::enable_if<std::is_integral<T>::value, T>::type
{
    return align == 0 ? 0 : x / align * align;
}
} // namespace GroupedMatmulAdd
#endif
