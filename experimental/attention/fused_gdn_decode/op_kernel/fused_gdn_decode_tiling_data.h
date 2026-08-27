/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_GDN_DECODE_TILING_DATA_H
#define FUSED_GDN_DECODE_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace FusedGdnDecode {
#pragma pack(push, 8)
struct alignas(8) FusedGdnDecodeTilingData {
    uint32_t b;
    uint32_t h;
    uint32_t hv;
    uint32_t k;
    uint32_t v;
    uint32_t bv;
    uint32_t vTiles;
    uint32_t stateBufferNum;
    uint32_t totalTasks;
    uint32_t indexBufferElems;
    uint64_t mixedStride;
    uint64_t stateSlotStride;
    uint64_t stateHeadStride;
    uint64_t outBatchStride;
    float scale;
    float softplusThreshold;
    uint32_t ubRestBytes;
};
#pragma pack(pop)
} // namespace FusedGdnDecode

#endif // FUSED_GDN_DECODE_TILING_DATA_H
