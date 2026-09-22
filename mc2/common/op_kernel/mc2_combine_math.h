/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_COMBINE_MATH_H
#define MC2_COMBINE_MATH_H

#include <cstdint>
#include <type_traits>

namespace Mc2Combine {

template <typename T>
__aicore__ inline T RoundUp(const T val, const T align)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <typename T>
__aicore__ inline T CeilDiv(const T dividend, const T divisor)
{
    return (divisor == 0) ? 0 : ((dividend + divisor - 1) / divisor);
}

// coreCount is nonzero, as required by the callers' tiling validation.
// The first total % coreCount cores receive one extra item.
__aicore__ inline void SplitCoreRange(uint32_t total, uint32_t coreCount, uint32_t coreIndex, uint32_t &count,
                                      uint32_t &begin, uint32_t &end)
{
    count = total / coreCount;
    uint32_t remainder = total % coreCount;
    begin = count * coreIndex;
    if (coreIndex < remainder) {
        count++;
        begin += coreIndex;
    } else {
        begin += remainder;
    }
    end = begin + count;
}

} // namespace Mc2Combine
#endif // MC2_COMBINE_MATH_H
