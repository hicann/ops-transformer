/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_COMBINE_TYPE_TRAITS_H
#define MC2_COMBINE_TYPE_TRAITS_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace Mc2Combine {
template <typename T>
struct OutputType {
    using type = T;
};

template <>
struct OutputType<bfloat16_t> {
    using type = float;
};

template <typename T>
using OutputType_t = typename OutputType<T>::type;
} // namespace Mc2Combine
#endif // MC2_COMBINE_TYPE_TRAITS_H
