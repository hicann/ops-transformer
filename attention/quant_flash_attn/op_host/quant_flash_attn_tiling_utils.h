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
 * \file quant_flash_attn_tiling_utils.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_TILING_UTILS_H
#define QUANT_FLASH_ATTN_TILING_UTILS_H

namespace optiling {
namespace quant_flash_attn {
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

} // namespace quant_flash_attn
} // namespace optiling

#endif // QUANT_FLASH_ATTN_TILING_UTILS_H
