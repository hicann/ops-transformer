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

constexpr uint64_t BLOCK_ATTN_RES_PREPARE_VECTOR_KEY = 0UL;
constexpr uint64_t BLOCK_ATTN_RES_PREPARE_MIX_KEY = 1UL;

// Tiling data is deliberately declared as a plain struct.  The host writes
// the scalar fields directly into the raw tiling buffer and the kernel reads
// the same layout, so no generated setter/serialization methods are required
// for the operator-owned fields below.
// Fields are ordered from wide to narrow; dimensions use the smallest type
// allowed by their host-side bounds.
struct BlockAttnResPrepareTilingData {
    uint32_t totalT;
    uint32_t totalS;
    uint32_t totalWorkUnits;
    uint32_t totalD;

    uint32_t blockFactor;
    uint32_t tailBlockFactor;
    uint32_t baseD;
    // Vector kernel derives dAlign/baseDAlign (one FP32 vector-register alignment), dTileNum and
    // the Q/V/O single-buffer element count from totalD/baseD.
    uint32_t statUbElems;
    float eps;

    uint16_t usedCoreNum;
    uint16_t bigCoreNum;

    // N is at most one FP32 vector register, and the queue depths are 1 or 2.
    uint8_t totalN;
    uint8_t vCacheRows;
    uint8_t qBufferNum;
    uint8_t vBufferNum;
    uint8_t oBufferNum;
};

// The Mix template has an independent data layout because its AIC/AIV work mapping and its L1/UB
// capacity plan are different from the Vector template. MM1 dot uses compact N rows,
// while E uses nAlign rows for the second Cube matmul.
struct BlockAttnResPrepareMixTilingData {
    uint64_t qL1Elems;
    uint64_t vL1Elems;
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
    uint32_t dAlign;
    float eps;

    // Tensor Scheduler derives the balanced per-core work range from totalWorkUnits and usedCoreNum.
    uint16_t usedCoreNum;
    uint16_t aicCoreNum;
    uint16_t aivCoreNum;

    uint8_t totalN;
    uint8_t qL1BufferNum;
    uint8_t vL1BufferNum;
    uint8_t vUbBufferNum;
};

} // namespace optiling

#endif // BLOCK_ATTN_RES_PREPARE_TILING_DATA_H
