/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_DATA_ARCH35_H
#define MHC_PRE_SINKHORN_BACKWARD_DATA_ARCH35_H

#include <cstdint>

// MM1/MM2 tiling and buffer configuration.
struct MhcPreSinkhornBackwardMMConfig {
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t l1K = 0;
    uint32_t depthB1 = 0;
    uint32_t dbL0A = 0;
    uint32_t dbL0C = 0;
};

// This is a private ABI shared only by the new Ascend950 host tiling and kernel.
struct MhcPreSinkhornBackwardArch35TilingData {
    int64_t bs = 0;
    int64_t n = 0;
    int64_t c = 0;
    int64_t k = 0;
    int64_t hcMix = 0;
    int64_t hcMixAlign = 0;
    int64_t iterTimes = 0;

    int64_t splitMode = 0;
    int64_t deterministic = 0;
    int64_t mBlockCount = 0;
    int64_t kBlockCount = 0;
    int64_t mBlockRows = 0;
    int64_t kBlockSize = 0;

    int64_t rowFactor = 0;
    int64_t rowLoop = 0;
    int64_t tailRowFactor = 0;
    int64_t cFactor = 0;
    int64_t cLoop = 0;
    int64_t tailCFactor = 0;

    int64_t kL1Size = 0;
    int64_t mL1Size = 0;
    int64_t mm1TileM = 0;
    int64_t mm1TileK = 0;
    int64_t mm1LoopK = 0;

    int64_t usedPrepAiv = 0;
    int64_t usedAic = 0;
    int64_t gradPhiTileK = 0;
    int64_t gradPhiLoopK = 0;
    int64_t reduceOwnerCount = 0;
    int64_t maxMWave = 0;

    uint64_t offGQ = 0;
    uint64_t offGR = 0;
    uint64_t offRowPart = 0;
    uint64_t offMm1Ring = 0;
    uint64_t offGradPhiPart = 0;
    uint64_t userWorkspaceBytes = 0;

    // Cube configuration and buffer layout for MM1 (GQ@phi) and MM2 (GQ^T@X).
    MhcPreSinkhornBackwardMMConfig mmConfigMm1;
    MhcPreSinkhornBackwardMMConfig mmConfigMm2;
    uint32_t l1BOffsetElems = 0;
    uint32_t l1SingleBufferElems = 0;
    uint32_t l0ABUsedBytes = 0;
    uint32_t l0ABSingleBufferElems = 0;
    uint32_t l0CUsedBytes = 0;
    uint32_t l0CSingleBufferElems = 0;

    float hcEps = 0.0f;
    float normEps = 0.0f;
};

#endif // MHC_PRE_SINKHORN_BACKWARD_DATA_ARCH35_H
