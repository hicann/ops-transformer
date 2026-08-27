/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_UPDATE_TILING_DATA_H
#define BLOCK_ATTN_RES_UPDATE_TILING_DATA_H

#include <cstdint>

#pragma pack(push, 8)
struct alignas(8) BlockAttnResUpdateTilingData {
    uint32_t dSize;
    // All participating cores except the last process tPerCore rows along T.
    uint32_t tPerCore;
    // The last participating core processes the remaining rows along T.
    uint32_t lastTPerCore;
    // Maximum number of T rows processed by one UB tile.
    uint32_t tileT;
    // Element stride between FP32 stats planes, aligned to a 32-byte boundary.
    uint32_t statsTStride;
    float eps;
    // Host-computed reciprocal of D used by the RMS normalization.
    float invD;
    uint16_t usedCoreNum;
};
#pragma pack(pop)

#endif // BLOCK_ATTN_RES_UPDATE_TILING_DATA_H
