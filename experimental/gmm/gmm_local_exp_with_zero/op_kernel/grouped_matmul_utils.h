/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _GMM_UTILS_H_
#define _GMM_UTILS_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace gmm_utils {

template <typename T>
__aicore__ inline T GreatestCommonDivisor(T a, T b)
{
    T c = a;
    if (a < b) {
        a = b;
        b = c;
    }
    while (b != 0) {
        c = a;
        a = b;
        b = c % b;
    }
    return a;
}

template <typename T>
__aicore__ inline T LeastCommonMultiple(T a, T b)
{
    return a * b / GreatestCommonDivisor(a, b);
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <uint32_t base, typename T = uint32_t>
__aicore__ inline T AlignUp(T a)
{
    return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline T AlignUp(T a, T base)
{
    return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline T AlignDown(T a, T base)
{
    if (unlikely(base == 0)) {
        return a;
    }
    return a / base * base;
}

template <>
__aicore__ inline uint32_t AlignUp<4, uint32_t>(uint32_t a)
{
    // to be Multiple of 4, result should be in a format of b(xxxx,x100).
    // This means last two bits should be zero, requiring that
    // result = num & b(1111,1100) = num & (~3).
    // &(~3) operator may reduces num into the range [num, num - 3].
    // As the result should be no less than a (result >= a), it means num - 3 >= a in the worst case.
    // In this case, num >= a+3. On the other hand, num should also be less then a+4, otherwise,
    // the result will not be least multiple of 4 for 3. In other cases like [num, num - 2],
    // num = a + 3 also satisfies the goal condition.
    return (a + 3) & ~3; // & ~3: set last two bits of (a+3) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<16, uint32_t>(uint32_t a)
{
    // In general, if we want to get the least multiple of b (b is the power of 2) for a,
    // it comes to a conclusion from the above comment: result = (a + (b - 1)) & (~b)
    return (a + 15) & ~15; // & ~15: set last four bits of (a+15) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<32, uint32_t>(uint32_t a)
{
    // refer to the above comments.
    return (a + 31) & ~31; // & ~31: set last five bits of (a+31) to be zero}
}

} // namespace gmm_utils

#endif
