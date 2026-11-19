/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_TILING_DATA_H
#define BLOCK_ATTN_RES_PREPARE_TILING_DATA_H

#include <cstdint>

namespace optiling {

// Tiling data is deliberately declared as a plain struct.  The host writes
// the scalar fields directly into the raw tiling buffer and the kernel reads
// the same layout, so no generated setter/serialization methods are required
// for the operator-owned fields below.
// The host precomputes scheduling, row alignment and 1/D.
#pragma pack(push, 8)
struct alignas(8) BlockAttnResPrepareTilingData {
    uint32_t totalT;
    uint32_t totalS;
    uint32_t sBlockSize; // Nominal S block width; paired tokens split it into two S spans.
    float eps;
    uint32_t sBlockCount; // Number of S blocks per token.
    uint32_t workPerCore; // Every core gets this many work units before distributing extra work.
    float reciprocalD;

    uint32_t totalD;
    uint32_t dAlign; // Shared UB row stride in elements for V and FP32 Q/O.

    // Host validates core counts within uint16_t; remainder < usedCoreNum.
    uint16_t usedCoreNum;
    uint16_t remainder;           // Each of the first remainder cores gets one additional work unit.
    uint16_t qCacheCoresPerGroup; // Cores per three-S group; zero disables Q caching across tokens.

    // N <= 64, and the V-cache row count never exceeds N.
    uint8_t totalN;
    uint8_t vCacheRows; // Maximum validN eligible for keeping all V rows in UB.
    uint8_t qBufferNum; // Initial Q/O slots before grouping S rows for V reuse.
    uint8_t threeSMaxN; // Maximum validN fitting three Q rows and three output rows.
};

// The Mix template has an independent data layout because its AIC/AIV work mapping and its L1/UB
// capacity plan are different from the Vector template. MM1 dot uses compact N rows,
// while E uses nAlign rows for the second Cube matmul.
struct alignas(8) BlockAttnResPrepareMixTilingData {
    uint64_t eL1Elems;
    uint64_t vUbElems;
    uint64_t dotUbElems;
    uint64_t reduceUbElems;
    uint64_t softmaxUbElems;
    uint64_t workspacePerCoreElems;

    uint32_t totalT;
    uint32_t totalS;
    uint32_t totalWorkUnits;
    uint32_t totalD;
    uint32_t baseT;
    uint32_t baseS;
    uint32_t baseD;
    uint32_t baseDAlign;
    uint32_t sTileNum;
    uint32_t dTileNum;
    uint32_t sAlign;
    uint32_t nAlign;
    uint32_t mm1NAlign;
    float eps;

    // Tensor Scheduler derives the balanced per-core work range from totalWorkUnits and usedCoreNum.
    uint16_t usedCoreNum;

    uint8_t totalN;
    uint8_t qL1BufferNum;
    uint8_t vL1BufferNum;
    uint8_t vUbBufferNum;
};
#pragma pack(pop)

static_assert(sizeof(BlockAttnResPrepareTilingData) == 48, "BlockAttnResPrepareTilingData must fit in 48 bytes");
static_assert(sizeof(BlockAttnResPrepareMixTilingData) == 112,
              "BlockAttnResPrepareMixTilingData must fit in 112 bytes");

} // namespace optiling

#endif // BLOCK_ATTN_RES_PREPARE_TILING_DATA_H
