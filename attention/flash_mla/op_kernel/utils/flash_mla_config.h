/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_CONFIG_H_
#define FLASH_MLA_CONFIG_H_

#include <cstdint>

namespace FlashMlaConfig {
struct TileConfig {
    uint32_t m;
    uint32_t s2;
    uint32_t cvRatio;
    uint32_t ubBytes;
};
constexpr TileConfig ARCH35 = {96U, 112U, 2U, 248U * 1024U};
constexpr TileConfig ARCH92 = {64U, 128U, 1U, 376U * 1024U};
constexpr uint32_t QK_DIM = 576U;
constexpr uint32_t V_DIM = 512U;
constexpr uint32_t SCORE_STRIDE = 128U;
} // namespace FlashMlaConfig

#endif // FLASH_MLA_CONFIG_H_
