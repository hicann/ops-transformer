/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MOE_EP_DISPATCH_SCHEDULE_H
#define MOE_EP_DISPATCH_SCHEDULE_H

#include <cstdint>

namespace MoeEpDispatchSchedule {
constexpr uint32_t METADATA_TILE_BYTES = 16U * 1024U;
constexpr uint32_t ROUTE_TILE_ELEMENTS = 4096U;

// Host tiling validates positive BS, core count and K; supported metadata slots are at most 1024 B.
constexpr uint32_t TileTokens(uint32_t bs, uint32_t cores, uint32_t topK, uint32_t metadataBytes)
{
    // Floor keeps at least one tile per AIV when BS >= cores; ceil can leave count owners idle.
    uint32_t balanced = bs >= cores ? bs / cores : 1U;
    uint32_t routeLimit = ROUTE_TILE_ELEMENTS / topK;
    uint32_t metadataLimit = METADATA_TILE_BYTES / metadataBytes;
    uint32_t tokens = balanced < routeLimit ? balanced : routeLimit;
    return tokens < metadataLimit ? tokens : metadataLimit;
}

constexpr uint32_t TileCount(uint32_t bs, uint32_t tileTokens)
{
    return bs / tileTokens + (bs % tileTokens != 0U ? 1U : 0U);
}

} // namespace MoeEpDispatchSchedule

#endif
