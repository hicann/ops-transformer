/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_DETER_FIXED_UTILS_H_
#define FLASH_ATTN_GRAD_DETER_FIXED_UTILS_H_

// Integer arithmetic only; preserve the original signed-division semantics.
#include <cstdint>

namespace optiling {
namespace fag_pto_arch35 {
constexpr int64_t NUM_TWO = 2, MULT_BASE = 2;

inline int64_t CeilDivideBy(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

inline int64_t PositiveCeilDiv(int64_t dividend, int64_t divisor)
{
    return dividend / divisor + static_cast<int64_t>(dividend % divisor != 0);
}

inline int64_t PositiveGcd(int64_t lhs, int64_t rhs)
{
    while (rhs > 0) {
        const int64_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

inline int64_t GqaBandMin(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

inline int64_t GqaBandCeil(int64_t a, int64_t b)
{
    return a / b + (a % b != 0 ? 1 : 0);
}

inline int64_t GqaBandGcd(int64_t a, int64_t b)
{
    while (b != 0) {
        const int64_t remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

inline int64_t GqaBandMax(int64_t a, int64_t b)
{
    return a > b ? a : b;
}
} // namespace fag_pto_arch35
} // namespace optiling

#endif // FLASH_ATTN_GRAD_DETER_FIXED_UTILS_H_
