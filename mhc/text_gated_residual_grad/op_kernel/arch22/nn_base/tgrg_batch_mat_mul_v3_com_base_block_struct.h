/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file tgrg_batch_mat_mul_v3_com_base_block_struct.h
 * \brief
 */
#ifndef TGRG_ARCH22_BATCH_MAT_MUL_V3_COM_BASE_BLOCK_STRUCT_H
#define TGRG_ARCH22_BATCH_MAT_MUL_V3_COM_BASE_BLOCK_STRUCT_H

struct CommonKernelBlockOffset {
    uint64_t offsetA;
    uint64_t offsetB;
    uint64_t offsetC;
    uint64_t offsetBias;
};

struct CommonKernelBaseBlockArgs {
    // a0..a5 = tensor dim0..dim5. Filled from blob slots D5,D4,D0,D1,D2,D3 (not DimN=aN).
    uint64_t batchA0;
    uint64_t batchA1;
    uint64_t batchA2;
    uint64_t batchA3;
    uint64_t batchA4;
    uint64_t batchA5;
    uint64_t batchB0;
    uint64_t batchB1;
    uint64_t batchB2;
    uint64_t batchB3;
    uint64_t batchB4;
    uint64_t batchB5;
    uint64_t batchC0;
    uint64_t batchC1;
    uint64_t batchC2;
    uint64_t batchC3;
    uint64_t batchC4;
    uint64_t batchC5;
    uint64_t aBatchDimAll;
    uint64_t bBatchDimAll;
    uint64_t cBatchDimAll;

    uint64_t batchCnt;
    uint64_t mCnt;      // Number of base blocks along M.
    uint64_t nCnt;      // Number of base blocks along N.
    uint64_t nBaseTail; // Size of the final baseN block along N.
    uint64_t mBaseTail; // Size of the final baseM block along M.
    uint64_t mTileCntL2;
    uint64_t nTileCntL2;
    uint64_t mCntTail;
    uint64_t nCntTail;
    uint64_t mTileCnt;
    uint64_t nTileCnt;
    uint64_t totalCnt; // Total number of base blocks.
    uint64_t totalTileCnt;
    uint64_t round;      // Maximum number of base blocks processed by one core.
    uint64_t realRound;  // Number of base blocks processed by this core.
    uint64_t preCoreNum; // Number of cores, starting at core 0, that process one extra base block.
    uint64_t mCntUse;    // Number of M-axis blocks in the tile.
    uint64_t nCntUse;    // Number of N-axis blocks in the tile.
    uint32_t rowOrder; // Core assignment order. 0 staggers the assignment, ROW_FIRST traverses C by rows, and COL_FIRST
                       // traverses C by columns.
    uint64_t blockIdxStart;
    uint64_t blockIdxEnd;
    uint64_t preTotalBlock;

    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t singleCoreK;

    uint64_t mTileAddrOffset; // Tile start address offset along M.
    uint64_t nTileAddrOffset; // Tile start address offset along N.
    uint64_t c0Size;
    uint64_t alignedOriM;
    uint64_t alignedOriN;
    uint64_t alignedKaSize;
    uint64_t alignedKbSize;

    uint64_t index; // Starting base-block index for this block_idx, traversing N before M.
    bool isHf32;
    bool biasWithBatch;
    bool enableL2Cache;
};

#endif // TGRG_ARCH22_BATCH_MAT_MUL_V3_COM_BASE_BLOCK_STRUCT_H
