/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_POST_FOUR_CONSTANTS_H
#define MHC_POST_FOUR_CONSTANTS_H

#include <stdint.h>

namespace MhcPost {
constexpr int64_t FOUR_HEAD_COUNT = 4;
constexpr int64_t FOUR_HEAD_TILE_SIZE = 2048;
constexpr int64_t FOUR_HEAD_D_ALIGN = 64;
constexpr uint32_t FOUR_HEAD_MIN_UB_BYTES = 166 * 1024;
} // namespace MhcPost

#endif // MHC_POST_FOUR_CONSTANTS_H
