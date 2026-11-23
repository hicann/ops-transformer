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
 * \file tgrg_batch_mat_mul_v3_com_base_block.h
 * \brief
 */
#ifndef TGRG_ARCH22_BATCH_MAT_MUL_V3_COM_BASE_BLOCK_H
#define TGRG_ARCH22_BATCH_MAT_MUL_V3_COM_BASE_BLOCK_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "tgrg_mat_mul_v3_common.h"
#include "tgrg_batch_mat_mul_v3_com_base_block_struct.h"
#include "tgrg_batch_mat_mul_v3_tiling_data.h"

using namespace AscendC;
using namespace matmul;

class BatchMatMulCommonBaseBlock {
public:
    __aicore__ inline BatchMatMulCommonBaseBlock() {}
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const void* tilingData);
    template <class C_TYPE>
    __aicore__ inline void InitBatchCount();
    __aicore__ inline void InitL2Cache();
    __aicore__ inline void InitBlockIndex();
    __aicore__ inline void UpdateBlockParams();
    __aicore__ inline void UpdateBlockParams(uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void UpdateBlockCnt(uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void UpdateBlockIndex();
    __aicore__ inline void UpdateBasicIndex(const uint64_t roundIdx, uint64_t mTileIndex, uint64_t nTileIndex);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset(uint64_t mTileIndex, uint64_t nTileIndex);
    template <class A_TYPE>
    __aicore__ inline void CalcAOffset(uint64_t batchAIndex, uint64_t mCntIndex, uint64_t mInnerBatchOffset);
    template <class B_TYPE>
    __aicore__ inline void CalcBOffset(uint64_t batchBIndex, uint64_t nCntIndex);
    template <class C_TYPE>
    __aicore__ inline void CalcCOffset(uint64_t batchCIndex, uint64_t mCntIndex, uint64_t nCntIndex,
                                       uint64_t mInnerBatchOffset);

public:
    static constexpr uint64_t DIAGONAL_THRESHOLD = 5;

    CommonKernelBlockOffset offset_;
    CommonKernelBaseBlockArgs params_;
    const BatchMatmulTilingData* batchMatmulTilingData_;
    bool indexInit_ = false;
    bool batchLastFlag_ = false;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void BatchMatMulCommonBaseBlock::Init(const void* tilingData)
{
    batchMatmulTilingData_ = static_cast<const BatchMatmulTilingData*>(tilingData);

    InitBatchCount<C_TYPE>();

    // Use row-major order by default.
    params_.rowOrder = ROW_FIRST;
    params_.mCntUse = params_.batchCnt * params_.mCnt;
    params_.nCntUse = params_.nCnt;
    params_.mTileCnt = params_.mCntUse;
    params_.nTileCnt = params_.nCntUse;
    params_.mCntTail = params_.mTileCnt;
    params_.nCntTail = params_.nTileCnt;
    params_.totalTileCnt = params_.mTileCnt * params_.nTileCnt;

    // Enable the L2 cache.
    InitL2Cache();
    if (params_.preCoreNum == 0) {
        params_.preCoreNum = batchMatmulTilingData_->multiBatchInfo.batchUsedCoreNum;
    }

    params_.c0Size = 0;
    using A_T = typename A_TYPE::T;
    GetSizeC0<A_T>(params_.c0Size);
    params_.alignedOriM = MMV3DivCeil(batchMatmulTilingData_->matmulTiling.matmulTiling.M, ALIGNED_H) * ALIGNED_H;
    params_.alignedOriN =
        MMV3DivCeil(batchMatmulTilingData_->matmulTiling.matmulTiling.N, params_.c0Size) * params_.c0Size;
    params_.alignedKaSize =
        MMV3DivCeil(batchMatmulTilingData_->matmulTiling.matmulTiling.Ka, params_.c0Size) * params_.c0Size;
    params_.alignedKbSize = MMV3DivCeil(batchMatmulTilingData_->matmulTiling.matmulTiling.Kb, ALIGNED_H) * ALIGNED_H;
    // Both A and B are aligned.
    if constexpr (A_TYPE::isTrans) {
        params_.alignedOriM =
            MMV3DivCeil(batchMatmulTilingData_->matmulTiling.matmulTiling.M, params_.c0Size) * params_.c0Size;
        params_.alignedKaSize =
            MMV3DivCeil(batchMatmulTilingData_->matmulTiling.matmulTiling.Ka, ALIGNED_H) * ALIGNED_H;
    }
    if constexpr (B_TYPE::isTrans) {
        params_.alignedOriN = MMV3DivCeil(batchMatmulTilingData_->matmulTiling.matmulTiling.N, ALIGNED_H) * ALIGNED_H;
        params_.alignedKbSize =
            MMV3DivCeil(batchMatmulTilingData_->matmulTiling.matmulTiling.Kb, params_.c0Size) * params_.c0Size;
    }
}

template <class C_TYPE>
__aicore__ inline void BatchMatMulCommonBaseBlock::InitBatchCount()
{
    params_.isHf32 = batchMatmulTilingData_->matmulTiling.matmulRunInfo.isHf32;
    const MultiBatchInfo& multiBatch = batchMatmulTilingData_->multiBatchInfo;
    // Locals a0..a5 = tensor dim0..dim5. Blob Dim* are slots: a0=D5, a1=D4, a2=D0 .. a5=D3.
    // Dim4/5 are appended; callers that omit them leave 0 (loop skip / divide-by-zero). Treat as missing axis (=1).
    params_.batchA0 = multiBatch.aBatchDim5 == 0 ? 1 : multiBatch.aBatchDim5;
    params_.batchA1 = multiBatch.aBatchDim4 == 0 ? 1 : multiBatch.aBatchDim4;
    params_.batchA2 = multiBatch.aBatchDim0;
    params_.batchA3 = multiBatch.aBatchDim1;
    params_.batchA4 = multiBatch.aBatchDim2;
    params_.batchA5 = multiBatch.aBatchDim3;
    params_.batchB0 = multiBatch.bBatchDim5 == 0 ? 1 : multiBatch.bBatchDim5;
    params_.batchB1 = multiBatch.bBatchDim4 == 0 ? 1 : multiBatch.bBatchDim4;
    params_.batchB2 = multiBatch.bBatchDim0;
    params_.batchB3 = multiBatch.bBatchDim1;
    params_.batchB4 = multiBatch.bBatchDim2;
    params_.batchB5 = multiBatch.bBatchDim3;
    params_.batchC0 = multiBatch.cBatchDim5 == 0 ? 1 : multiBatch.cBatchDim5;
    params_.batchC1 = multiBatch.cBatchDim4 == 0 ? 1 : multiBatch.cBatchDim4;
    params_.batchC2 = multiBatch.cBatchDim0;
    params_.batchC3 = multiBatch.cBatchDim1;
    params_.batchC4 = multiBatch.cBatchDim2;
    params_.batchC5 = multiBatch.cBatchDim3;
    params_.aBatchDimAll = multiBatch.aBatchDimAll;
    params_.bBatchDimAll = multiBatch.bBatchDimAll;
    params_.cBatchDimAll = multiBatch.cBatchDimAll;
    params_.biasWithBatch = multiBatch.biasWithBatch;

    params_.batchCnt =
        params_.batchC0 * params_.batchC1 * params_.batchC2 * params_.batchC3 * params_.batchC4 * params_.batchC5;
    params_.mCnt = MMV3DivCeil(
        batchMatmulTilingData_->matmulTiling.matmulTiling.M,
        static_cast<uint64_t>(
            batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreM)); // Number of base blocks along M.
    params_.nCnt = MMV3DivCeil(
        batchMatmulTilingData_->matmulTiling.matmulTiling.N,
        static_cast<uint64_t>(
            batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN)); // Number of base blocks along N.
    params_.index = 0; // Starting base-block index for this block_idx, traversing N before M.
    params_.nBaseTail =
        batchMatmulTilingData_->matmulTiling.matmulTiling.N -
        (params_.nCnt - 1) *
            batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN; // Size of the final baseN block along N.
    params_.mBaseTail =
        batchMatmulTilingData_->matmulTiling.matmulTiling.M -
        (params_.mCnt - 1) *
            batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreM; // Size of the final baseM block along M.
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    if constexpr (C_TYPE::format == CubeFormat::ND) {
        params_.mBaseTail = static_cast<uint64_t>(batchMatmulTilingData_->multiBatchInfo.mOri) -
                            (params_.mCnt - 1) * batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreM;
    }
#endif
    params_.totalCnt = params_.batchCnt * params_.mCnt * params_.nCnt; // Total number of base blocks.
    params_.round =
        (params_.totalCnt + batchMatmulTilingData_->multiBatchInfo.batchUsedCoreNum - 1) /
        batchMatmulTilingData_->multiBatchInfo.batchUsedCoreNum; // Maximum number of base blocks processed by one core.
    params_.realRound = 0;                                       // Number of base blocks processed by this core.
    params_.preCoreNum =
        params_.totalCnt %
        batchMatmulTilingData_->multiBatchInfo
            .batchUsedCoreNum; // Number of cores, starting at core 0, that process one extra base block.
    params_.blockIdxStart = 0;
    params_.blockIdxEnd = 0;
    params_.preTotalBlock = 0;
}

__aicore__ inline void BatchMatMulCommonBaseBlock::InitL2Cache()
{
    const L2cacheTilePara& tilingL2 = batchMatmulTilingData_->matmulTiling.tileL2cacheTiling;
    params_.mTileCntL2 = static_cast<uint64_t>(tilingL2.mTileCntL2); // Number of tiles along M.
    params_.nTileCntL2 = static_cast<uint64_t>(tilingL2.nTileCntL2); // Number of tiles along N.
    params_.enableL2Cache = ((params_.mTileCntL2 > 1) || (params_.nTileCntL2 > 1));
    if (params_.enableL2Cache) {
        params_.mTileCnt = tilingL2.mTileBlock; // Number of base blocks in each M tile.
        params_.nTileCnt = tilingL2.nTileBlock; // Number of base blocks in each N tile.
        params_.totalTileCnt = params_.mTileCnt * params_.nTileCnt;
        params_.mCntTail = params_.mCnt - (params_.mTileCnt * (MMV3DivCeil(params_.mCnt, params_.mTileCnt) -
                                                               1)); // Number of base blocks in the final M tile.
        params_.nCntTail =
            params_.nCnt - (params_.nTileCntL2 - 1) * params_.nTileCnt; // Number of base blocks in the final N tile.
        params_.mCntUse = params_.mTileCnt;
        params_.nCntUse = params_.nTileCnt;
        params_.rowOrder = 0;
    }
}

__aicore__ inline void BatchMatMulCommonBaseBlock::InitBlockIndex()
{
    if (indexInit_) {                                // ND
        params_.blockIdxStart = params_.blockIdxEnd; // Index of the first core at the start of computation.
    } else {
        params_.blockIdxStart = 0; // Initialize to zero on the first call.
        indexInit_ = true;
    }
    params_.blockIdxEnd = (params_.blockIdxStart + params_.preCoreNum) %
                          batchMatmulTilingData_->matmulTiling.matmulTiling
                              .usedCoreNum; // Index of the last core at the end of computation.
    if (params_.rowOrder == 0) {
        uint64_t indexStart = params_.blockIdxStart;
        uint64_t indexEnd = params_.blockIdxEnd;
        if (indexStart < indexEnd) {
            if ((indexStart <= GetBlockIdx()) && (GetBlockIdx() < indexEnd)) {
                params_.realRound = params_.round;
            } else {
                params_.realRound = params_.round - 1;
            }
        } else if (indexEnd < indexStart) {
            if ((indexEnd <= GetBlockIdx()) && (GetBlockIdx() < indexStart)) {
                params_.realRound = params_.round - 1;
            } else {
                params_.realRound = params_.round;
            }
        } else {
            params_.realRound = params_.round;
        }
    } else {
        if (GetBlockIdx() < params_.preCoreNum) {
            if (params_.rowOrder == ROW_FIRST) {
                params_.index = GetBlockIdx() * params_.round;
            } else {
                params_.preTotalBlock = GetBlockIdx() * params_.round;
                params_.index =
                    params_.preTotalBlock / params_.mCnt + params_.preTotalBlock % params_.mCnt * params_.nCnt;
            }
            params_.realRound = params_.round; // The first preCoreNum cores process one extra base block.
        } else {
            if (params_.rowOrder == ROW_FIRST) {
                params_.index = GetBlockIdx() * (params_.round - 1) + params_.preCoreNum;
            } else {
                params_.preTotalBlock = GetBlockIdx() * (params_.round - 1) + params_.preCoreNum;
                params_.index =
                    params_.preTotalBlock / params_.mCnt + params_.preTotalBlock % params_.mCnt * params_.nCnt;
            }
            params_.realRound = params_.round - 1; // The remaining cores process one fewer base block.
        }
    }
}

__aicore__ inline void BatchMatMulCommonBaseBlock::UpdateBasicIndex(const uint64_t roundIdx, uint64_t mTileIndex,
                                                                    uint64_t nTileIndex)
{
    uint64_t newBlockIdx =
        (GetBlockIdx() + batchMatmulTilingData_->matmulTiling.matmulTiling.usedCoreNum - params_.blockIdxStart) %
            batchMatmulTilingData_->matmulTiling.matmulTiling.usedCoreNum +
        roundIdx * batchMatmulTilingData_->matmulTiling.matmulTiling.usedCoreNum;
    uint64_t mIdx = newBlockIdx % params_.mCntUse;
    uint64_t nIdx = 0;
    if ((params_.mCntUse != 0) && (params_.nCntUse != 0)) {
        nIdx = (newBlockIdx + newBlockIdx / MMLcm(params_.mCntUse, params_.nCntUse)) % params_.nCntUse;
    }
    params_.index = mIdx * params_.nCntUse + nIdx;
}

__aicore__ inline void BatchMatMulCommonBaseBlock::UpdateBlockParams(uint64_t mTileIndex, uint64_t nTileIndex)
{
    if (!(params_.enableL2Cache)) {
        if ((params_.index + 1) % (params_.mCnt * params_.nCnt) == 0) {
            // The last block is a tail block.
            params_.singleCoreM = params_.mBaseTail;
            params_.singleCoreN = params_.nBaseTail;
        } else if (((params_.index + 1) % (params_.mCnt * params_.nCnt)) > ((params_.mCnt - 1) * params_.nCnt)) {
            // M-axis tail block.
            params_.singleCoreM = params_.mBaseTail;
            params_.singleCoreN = batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN;
        } else if ((params_.index + 1) % params_.nCnt == 0) {
            // N-axis tail block.
            params_.singleCoreM = batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreM;
            params_.singleCoreN = params_.nBaseTail;
        } else {
            // Aligned full block.
            params_.singleCoreM = batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreM;
            params_.singleCoreN = batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN;
        }
    } else {
        // Convert the current M tile index to its index within the batch.
        // Within a batch, use baseTail for the final row of the last M tile. Along N, use nTileCnt to determine whether
        // to use baseTail.
        if (batchLastFlag_ && (nTileIndex == (params_.nTileCntL2 - 1)) &&
            (params_.index == (params_.totalTileCnt - 1))) {
            params_.singleCoreM = params_.mBaseTail;
            params_.singleCoreN = params_.nBaseTail;
        } else if (batchLastFlag_ && (params_.index >= (params_.mCntUse - 1) * params_.nCntUse)) {
            params_.singleCoreM = params_.mBaseTail;
            params_.singleCoreN = batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN;
        } else if ((nTileIndex == (params_.nTileCntL2 - 1)) && ((params_.index + 1) % params_.nCntUse == 0)) {
            params_.singleCoreM = batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreM;
            params_.singleCoreN = params_.nBaseTail;
        } else {
            params_.singleCoreM = batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreM;
            params_.singleCoreN = batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN;
        }
    }
}

__aicore__ inline void BatchMatMulCommonBaseBlock::UpdateBlockCnt(uint64_t bmTileIndex, uint64_t nTileIndex)
{
    // The current tile partition stays within a batch.
    // Compute each tile start offset uniformly. Use M across batches and singleCoreM within a batch.
    uint32_t batchTileCnt = CeilDiv(params_.mCnt, params_.mTileCnt); // Both mCnt and mTileCnt are positive.
    uint64_t curBatchCnt = bmTileIndex / batchTileCnt;
    uint64_t curBatchTileIdx = bmTileIndex % batchTileCnt;
    params_.mTileAddrOffset =
        curBatchCnt * batchMatmulTilingData_->matmulTiling.matmulTiling.M +
        curBatchTileIdx * params_.mTileCnt * batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreM;
    params_.nTileAddrOffset =
        nTileIndex * params_.nTileCnt * batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN;
    batchLastFlag_ = ((bmTileIndex + 1) % MMV3DivCeil(params_.mCnt, params_.mTileCnt) == 0);
    if (batchLastFlag_ && (nTileIndex == (params_.nTileCntL2 - 1))) {
        params_.totalTileCnt = params_.mCntTail * params_.nCntTail;
        params_.mCntUse = params_.mCntTail;
        params_.nCntUse = params_.nCntTail;
    } else if (batchLastFlag_) {
        params_.totalTileCnt = params_.mCntTail * params_.nTileCnt;
        params_.mCntUse = params_.mCntTail;
        params_.nCntUse = params_.nTileCnt;
    } else if (nTileIndex == (params_.nTileCntL2 - 1)) {
        params_.totalTileCnt = params_.mTileCnt * params_.nCntTail;
        params_.mCntUse = params_.mTileCnt;
        params_.nCntUse = params_.nCntTail;
    } else {
        params_.totalTileCnt = params_.mTileCnt * params_.nTileCnt;
        params_.mCntUse = params_.mTileCnt;
        params_.nCntUse = params_.nTileCnt;
    }
    params_.round = MMV3DivCeil(params_.totalTileCnt,
                                static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.usedCoreNum));
    params_.preCoreNum = params_.totalTileCnt % batchMatmulTilingData_->matmulTiling.matmulTiling.usedCoreNum;
    if (params_.preCoreNum == 0) {
        params_.preCoreNum = static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.usedCoreNum);
    }
}

__aicore__ inline void BatchMatMulCommonBaseBlock::UpdateBlockParams()
{
    if ((params_.index + 1) % (params_.mCnt * params_.nCnt) == 0) {
        // The last block is a tail block.
        params_.singleCoreM = params_.mBaseTail;
        params_.singleCoreN = params_.nBaseTail;
    } else if ((params_.index + 1) % (params_.mCnt * params_.nCnt) > (params_.mCnt - 1) * params_.nCnt) {
        // M-axis tail block.
        params_.singleCoreM = params_.mBaseTail;
        params_.singleCoreN = batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN;
    } else if ((params_.index + 1) % params_.nCnt == 0) {
        // N-axis tail block.
        params_.singleCoreM = batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreM;
        params_.singleCoreN = params_.nBaseTail;
    } else {
        // Aligned full block.
        params_.singleCoreM = batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreM;
        params_.singleCoreN = batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void BatchMatMulCommonBaseBlock::CalcGMOffset(uint64_t mTileIndex, uint64_t nTileIndex)
{
    // Compute the index along the combined batch and M axis.
    uint64_t bmTileStartIndex =
        params_.mTileAddrOffset / static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.M);
    uint64_t tileInnerMIndex = params_.index / (params_.nCntUse * params_.mCnt);
    uint64_t totalBmTileIndex = bmTileStartIndex + tileInnerMIndex;

    // Batch and M share one axis. Decode the coordinates before recalculating the index.
    uint64_t cInner4 = params_.batchC2 * params_.batchC3 * params_.batchC4 * params_.batchC5;
    uint64_t batchC0Index = totalBmTileIndex / (params_.batchC1 * cInner4);
    uint64_t batchC1Index = (totalBmTileIndex / cInner4) % params_.batchC1;
    uint64_t low4Index = totalBmTileIndex % cInner4;
    uint64_t batchC2Index = low4Index / (params_.batchC3 * params_.batchC4 * params_.batchC5);
    uint64_t batchC3Index =
        low4Index % (params_.batchC3 * params_.batchC4 * params_.batchC5) / (params_.batchC4 * params_.batchC5);
    uint64_t batchC4Index = low4Index % (params_.batchC4 * params_.batchC5) / params_.batchC5;
    uint64_t batchC5Index = low4Index % params_.batchC5;
    uint64_t batchCIndex = totalBmTileIndex;

    // M index within the tile.
    uint64_t mCntIndex = (params_.index / params_.nCntUse) % params_.mCnt;
    uint64_t nCntIndex = params_.index % params_.nCntUse;
    uint64_t mInnerBatchOffset =
        params_.mTileAddrOffset % static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.M);
    CalcCOffset<C_TYPE>(batchCIndex, mCntIndex, nCntIndex, mInnerBatchOffset);

    uint64_t batchA0Index = batchC0Index % params_.batchA0;
    uint64_t batchA1Index = batchC1Index % params_.batchA1;
    uint64_t batchA2Index = batchC2Index % params_.batchA2;
    uint64_t batchA3Index = batchC3Index % params_.batchA3;
    uint64_t batchA4Index = batchC4Index % params_.batchA4;
    uint64_t batchA5Index = batchC5Index % params_.batchA5;
    uint64_t aInner4 = params_.batchA2 * params_.batchA3 * params_.batchA4 * params_.batchA5;
    uint64_t batchAIndex = batchA0Index * (params_.batchA1 * aInner4) + batchA1Index * aInner4 +
                           batchA2Index * (params_.batchA3 * params_.batchA4 * params_.batchA5) +
                           batchA3Index * (params_.batchA4 * params_.batchA5) + batchA4Index * params_.batchA5 +
                           batchA5Index;
    CalcAOffset<A_TYPE>(batchAIndex, mCntIndex, mInnerBatchOffset);

    uint64_t batchB0Index = batchC0Index % params_.batchB0;
    uint64_t batchB1Index = batchC1Index % params_.batchB1;
    uint64_t batchB2Index = batchC2Index % params_.batchB2;
    uint64_t batchB3Index = batchC3Index % params_.batchB3;
    uint64_t batchB4Index = batchC4Index % params_.batchB4;
    uint64_t batchB5Index = batchC5Index % params_.batchB5;
    uint64_t bInner4 = params_.batchB2 * params_.batchB3 * params_.batchB4 * params_.batchB5;
    uint64_t batchBIndex = batchB0Index * (params_.batchB1 * bInner4) + batchB1Index * bInner4 +
                           batchB2Index * (params_.batchB3 * params_.batchB4 * params_.batchB5) +
                           batchB3Index * (params_.batchB4 * params_.batchB5) + batchB4Index * params_.batchB5 +
                           batchB5Index;
    CalcBOffset<B_TYPE>(batchBIndex, nCntIndex);

    if (batchMatmulTilingData_->matmulTiling.matmulTiling.isBias) {
        offset_.offsetBias =
            params_.biasWithBatch ?
                batchCIndex * static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.N) +
                    nCntIndex * static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN) +
                    params_.nTileAddrOffset :
                nCntIndex * static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN) +
                    params_.nTileAddrOffset;
    }
}

template <class A_TYPE>
__aicore__ inline void BatchMatMulCommonBaseBlock::CalcAOffset(uint64_t batchAIndex, uint64_t mCntIndex,
                                                               uint64_t mInnerBatchOffset)
{
    if constexpr (A_TYPE::format == CubeFormat::ND) {
        uint64_t offsetABatch = batchAIndex *
                                static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.M) *
                                batchMatmulTilingData_->matmulTiling.matmulTiling.Ka;
        offset_.offsetA =
            A_TYPE::isTrans ?
                (offsetABatch + (mCntIndex * batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreM) +
                 mInnerBatchOffset) :
                (offsetABatch +
                 (mCntIndex * static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.Ka) *
                  batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreM) +
                 mInnerBatchOffset * batchMatmulTilingData_->matmulTiling.matmulTiling.Ka);
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        uint64_t offsetABatch = batchAIndex * params_.alignedOriM * params_.alignedKaSize;
        offset_.offsetA =
            A_TYPE::isTrans ?
                offsetABatch +
                    (mCntIndex * batchMatmulTilingData_->matmulTiling.matmulTiling.baseM * params_.alignedKaSize) +
                    (mInnerBatchOffset * params_.alignedKaSize) :
                offsetABatch + (mCntIndex * batchMatmulTilingData_->matmulTiling.matmulTiling.baseM * params_.c0Size) +
                    (mInnerBatchOffset * params_.c0Size);
    }
}

template <class B_TYPE>
__aicore__ inline void BatchMatMulCommonBaseBlock::CalcBOffset(uint64_t batchBIndex, uint64_t nCntIndex)
{
    if constexpr (B_TYPE::format == CubeFormat::ND) {
        uint64_t offsetBBatch = batchBIndex *
                                static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.N) *
                                batchMatmulTilingData_->matmulTiling.matmulTiling.Kb;
        offset_.offsetB =
            B_TYPE::isTrans ?
                offsetBBatch +
                    (nCntIndex * static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.Ka) *
                     batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN) +
                    (params_.nTileAddrOffset * batchMatmulTilingData_->matmulTiling.matmulTiling.Ka) :
                offsetBBatch + (nCntIndex * batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN) +
                    params_.nTileAddrOffset;
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        uint64_t offsetBBatch = batchBIndex * params_.alignedOriN * params_.alignedKbSize;
        offset_.offsetB =
            B_TYPE::isTrans ?
                offsetBBatch +
                    (nCntIndex * batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN * params_.c0Size) +
                    (params_.nTileAddrOffset * params_.c0Size) :
                offsetBBatch +
                    (nCntIndex * batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN *
                     params_.alignedKbSize) +
                    (params_.nTileAddrOffset * params_.alignedKbSize);
    }
}

template <class C_TYPE>
__aicore__ inline void BatchMatMulCommonBaseBlock::CalcCOffset(uint64_t batchCIndex, uint64_t mCntIndex,
                                                               uint64_t nCntIndex, uint64_t mInnerBatchOffset)
{
    if constexpr (C_TYPE::format == CubeFormat::ND) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
        offset_.offsetC = batchCIndex * static_cast<uint64_t>(batchMatmulTilingData_->multiBatchInfo.mOri) *
                              batchMatmulTilingData_->matmulTiling.matmulTiling.N +
                          nCntIndex * batchMatmulTilingData_->matmulTiling.matmulTiling.baseN +
                          mCntIndex * static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.baseM) *
                              batchMatmulTilingData_->matmulTiling.matmulTiling.N +
                          mInnerBatchOffset * batchMatmulTilingData_->matmulTiling.matmulTiling.N +
                          params_.nTileAddrOffset;
#else
        offset_.offsetC =
            batchCIndex * static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.M) *
                batchMatmulTilingData_->matmulTiling.matmulTiling.N +
            nCntIndex * batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreN +
            mCntIndex * static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreM) *
                batchMatmulTilingData_->matmulTiling.matmulTiling.N +
            mInnerBatchOffset * batchMatmulTilingData_->matmulTiling.matmulTiling.N + params_.nTileAddrOffset;
#endif
    } else if constexpr (C_TYPE::format == CubeFormat::NZ) {
        offset_.offsetC = batchCIndex * params_.alignedOriM * params_.alignedOriN +
                          nCntIndex * static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.baseN) *
                              batchMatmulTilingData_->matmulTiling.matmulTiling.M +
                          mCntIndex * static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.baseM) *
                              params_.c0Size +
                          mInnerBatchOffset * params_.c0Size +
                          params_.nTileAddrOffset * batchMatmulTilingData_->matmulTiling.matmulTiling.M;
    }
}

__aicore__ inline void BatchMatMulCommonBaseBlock::UpdateBlockIndex()
{
    if (params_.rowOrder == ROW_FIRST) {
        params_.index += 1;
    } else if (params_.rowOrder == COL_FIRST) {
        params_.index += params_.nCnt;
        if (params_.index >= params_.totalCnt) {
            params_.index = params_.index % params_.totalCnt + 1;
        }
    }
}

#endif // TGRG_ARCH22_BATCH_MAT_MUL_V3_COM_BASE_BLOCK_H
