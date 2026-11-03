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
 * \file flash_mla_tiling_utils.h
 * \brief
 */

#ifndef FLASH_MLA_TILING_UTILS_H
#define FLASH_MLA_TILING_UTILS_H

#include "tiling/tiling_api.h"

namespace optiling {
namespace flash_mla {

// 逐维比对strides与连续场景的期望stride, 跳过size为1的维度(该维索引恒为0, stride不参与地址计算,
// 语义同 PyTorch is_contiguous), stride维数小于shape维数时按非连续处理,
// index返回首个非连续维, 全连续返回GRAPH_SUCCESS
inline ge::graphStatus CheckTensorContiguous(const uint32_t& tensorDimNum, const gert::Shape& inputShape,
                                             const gert::Stride* strides, int32_t& index)
{
    if (strides == nullptr || strides->GetDimNum() == 0) {
        return ge::GRAPH_SUCCESS;
    }
    if (strides->GetDimNum() < tensorDimNum) {
        index = static_cast<int32_t>(tensorDimNum);
        return ge::GRAPH_FAILED;
    }
    uint64_t expectedStride = 1;
    for (index = static_cast<int32_t>(tensorDimNum) - 1; index >= 0; index--) {
        if (inputShape.GetDim(index) != 1 && strides->GetStride(index) != expectedStride) {
            return ge::GRAPH_FAILED;
        }
        expectedStride *= static_cast<uint64_t>(inputShape.GetDim(index));
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
inline auto CalcTailSize(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

template <typename T>
inline auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
inline auto IncreGcd(T a, T b) -> T
{
    if (b == 0) {
        return a;
    }
    if (a % b == 0) {
        return b;
    }
    return IncreGcd(b, a % b);
}

} // namespace flash_mla
} // namespace optiling

#endif // FLASH_MLA_TILING_UTILS_H
