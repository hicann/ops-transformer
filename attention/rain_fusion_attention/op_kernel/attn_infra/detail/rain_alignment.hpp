/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RAIN_ALIGNMENT_HPP
#define RAIN_ALIGNMENT_HPP

#include <limits>

#include "../../attn_infra/detail/rain_macros.hpp"

template <uint32_t ALIGN, typename T>
HOST_DEVICE constexpr T RoundDown(const T val)
{
    static_assert(ALIGN != 0U, "ALIGN must not be 0");
    return val / ALIGN * ALIGN;
}

template <class T>
HOST_DEVICE constexpr T RoundDown(const T val, const T align)
{
    if (align == 0) {
        return val;
    }
    return val / align * align;
}

template <uint32_t ALIGN, typename T = uint32_t>
HOST_DEVICE constexpr T RoundUp(const T val)
{
    static_assert(ALIGN != 0U, "ALIGN must not be 0");
    T align = ALIGN;
    if (val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <class T>
HOST_DEVICE constexpr T RoundUp(const T val, const T align)
{
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <uint32_t DIVISOR, typename T = uint32_t>
HOST_DEVICE constexpr T CeilDiv(const T dividend)
{
    static_assert(DIVISOR != 0U, "DIVISOR must not be 0");
    T divisor = DIVISOR;
    if (dividend + divisor - 1 < dividend) {
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

template <class T>
HOST_DEVICE constexpr T CeilDiv(const T dividend, const T divisor)
{
    if (divisor == 0 || dividend + divisor - 1 < dividend) {
        return std::numeric_limits<T>::max();
    }
    return (dividend + divisor - 1) / divisor;
}

#endif // ALIGNMENT_HPP
