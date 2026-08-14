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
 * \file kv_rms_norm_rope_cache_platform.h
 * \brief platform apator
 */
#include "kernel_operator.h"

namespace ops {

__aicore__ inline int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y == 0) {
        return 0;
    }
    return (x + y - 1) / y;
}

__aicore__ inline int64_t FloorDiv(int64_t x, int64_t y)
{
    if (y == 0) {
        return 0;
    }
    return x / y;
}

__aicore__ inline int64_t Aligned(int64_t x, int64_t y)
{
    if (y == 0) {
        return 0;
    }
    return (x + y - 1) / y * y;
}

} // namespace ops

namespace platform {
__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

} // namespace platform
