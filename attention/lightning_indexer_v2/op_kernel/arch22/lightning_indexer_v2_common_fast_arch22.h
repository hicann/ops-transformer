/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file lightning_indexer_v2_common_fast_arch22.h
 * \brief Arch22 cube-fill-vector extensions to the ordinary common types.
 */
#ifndef LIGHTNING_INDEXER_V2_COMMON_FAST_ARCH22_H
#define LIGHTNING_INDEXER_V2_COMMON_FAST_ARCH22_H

#include <cstdint>
#include "lightning_indexer_v2_common_arch22.h"

namespace LIV2FastCommon {

template <typename Q_T, typename K_T, typename OUT_T, typename QK_T, typename SCORE_T,
          const bool PAGE_ATTENTION = false, LIV2Common::LI_V2_LAYOUT LAYOUT_T = LIV2Common::LI_V2_LAYOUT::BSND,
          LIV2Common::LI_V2_LAYOUT K_LAYOUT_T = LIV2Common::LI_V2_LAYOUT::PA_BBND, bool DT_W_FLAG = false,
          bool CUBE_FILL_VECTOR = false, typename... Args>
struct LIV2Type
    : LIV2Common::LIV2Type<Q_T, K_T, OUT_T, QK_T, SCORE_T, PAGE_ATTENTION, LAYOUT_T, K_LAYOUT_T, DT_W_FLAG, Args...> {
    static constexpr bool cubeFillVector = CUBE_FILL_VECTOR;
};

struct RunInfo : LIV2Common::RunInfo {
    uint32_t s2LoopEnd = 0U; // Last visible K tile consumed by the fast pipeline.
};

struct ConstInfo : LIV2Common::ConstInfo {
    static constexpr uint32_t SCORE_RING_DEPTH = 8;
    uint32_t realSparseCount = 0;
    uint64_t kCacheStride = 0;
};

} // namespace LIV2FastCommon

#endif // LIGHTNING_INDEXER_V2_COMMON_FAST_ARCH22_H
