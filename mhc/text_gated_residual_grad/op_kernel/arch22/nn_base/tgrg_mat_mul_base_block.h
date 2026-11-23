/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file tgrg_mat_mul_base_block.h
 * \brief
 */
#ifndef TGRG_ARCH22_MAT_MUL_BASE_BLOCK_H
#define TGRG_ARCH22_MAT_MUL_BASE_BLOCK_H

#include "tgrg_mat_mul_v3_common.h"
#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

struct BlockOffset {
    uint64_t offsetA = 0;
    uint64_t offsetB = 0;
    uint64_t offsetC = 0;
    uint64_t offsetBias = 0;
};

struct BaseBlockArgs {
    uint64_t index;
    uint64_t mCnt;
    uint64_t nCnt;
    uint64_t totalTileCnt;
    uint64_t blockBaseM;
    uint64_t blockBaseN;
    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t nBaseTail;
    uint64_t mBaseTail;
    uint64_t mTileCntL2;
    uint64_t nTileCntL2;
    uint64_t mCntTail;
    uint64_t nCntTail;
    uint64_t mTotalCnt;
    uint64_t nTotalCnt;
    uint64_t round;
    uint64_t realRound;
    uint64_t preCoreNum;
    uint64_t mCntUse;
    uint64_t nCntUse;
    uint32_t rowOrder;
    uint64_t blockIdxStart;
    uint64_t blockIdxEnd;

    uint64_t mTileAddrOffset;
    uint64_t nTileAddrOffset;

    uint64_t c0Size;
    uint64_t alignedOriM;
    uint64_t alignedOriN;
    uint64_t alignedKaSize;
    uint64_t alignedKbSize;

    bool isTransposeA;
    bool isTransposeB;
    uint32_t isHf32;
};

class MatmulBaseBlock {
public:
    __aicore__ inline MatmulBaseBlock() {}
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const void* tilingData);
    __aicore__ inline void UpdateBlockCnt(uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void InitBlockIndex(uint64_t index);
    __aicore__ inline void UpdateBlockParams(uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void UpdateBlockIndex();
    __aicore__ inline void UpdateBasicIndex(const uint64_t roundIdx);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset(uint64_t mTileIndex, uint64_t nTileIndex);

public:
    BlockOffset offset_;
    BaseBlockArgs params_;
    const MatmulTilingData* matmulTilingData_;
    bool indexInit_ = false;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulBaseBlock::Init(const void* tilingData)
{
    matmulTilingData_ = static_cast<const MatmulTilingData*>(tilingData);
    const L2cacheTilePara& tilingL2 = matmulTilingData_->tileL2cacheTiling;

    params_.isTransposeA = matmulTilingData_->matmulRunInfo.transA;
    params_.isTransposeB = matmulTilingData_->matmulRunInfo.transB;
    params_.isHf32 = matmulTilingData_->matmulRunInfo.isHf32;

    params_.blockBaseM = static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseM);
    params_.blockBaseN = static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseN);
    params_.mTileCntL2 = static_cast<uint64_t>(tilingL2.mTileCntL2); // Number of L2 tiles along M.
    params_.nTileCntL2 = static_cast<uint64_t>(tilingL2.nTileCntL2); // Number of L2 tiles along N.
    params_.mTotalCnt =
        (static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) + matmulTilingData_->matmulTiling.singleCoreM - 1) /
        matmulTilingData_->matmulTiling.singleCoreM; // Total number of base tiles along M.
    params_.nTotalCnt =
        (static_cast<uint64_t>(matmulTilingData_->matmulTiling.N) + matmulTilingData_->matmulTiling.singleCoreN - 1) /
        matmulTilingData_->matmulTiling.singleCoreN; // Total number of base tiles along N.
    params_.nBaseTail = static_cast<uint64_t>(matmulTilingData_->matmulTiling.N) -
                        (params_.nTotalCnt - 1) * matmulTilingData_->matmulTiling.singleCoreN; // N tail base tile size.
    params_.mBaseTail = static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) -
                        (params_.mTotalCnt - 1) * matmulTilingData_->matmulTiling.singleCoreM; // M tail base tile size.
    params_.singleCoreM = 0;
    params_.singleCoreN = 0;
    params_.blockIdxStart = 0;
    params_.blockIdxEnd = 0;
    params_.index = 0;
    // The tiling configuration bounds mCnt and nCnt to keep addresses in range.
    // Split the L2 tiles so that full tiles and tail tiles are covered exactly.
    params_.mCnt = (params_.mTotalCnt + params_.mTileCntL2 - 1) / params_.mTileCntL2; // Base tiles per M tile.
    params_.nCnt = (params_.nTotalCnt + params_.nTileCntL2 - 1) / params_.nTileCntL2; // Base tiles per N tile.
    if (tilingL2.mTileBlock > 0 && tilingL2.nTileBlock > 0) {
        params_.mCnt = tilingL2.mTileBlock;
        params_.nCnt = tilingL2.nTileBlock;
    }
    params_.totalTileCnt = params_.mCnt * params_.nCnt;

    params_.mCntTail = params_.mTotalCnt - (params_.mTileCntL2 - 1) * params_.mCnt; // Base tiles in the M tail tile.
    params_.nCntTail = params_.nTotalCnt - (params_.nTileCntL2 - 1) * params_.nCnt; // Base tiles in the N tail tile.
    params_.round = (params_.totalTileCnt + matmulTilingData_->matmulTiling.usedCoreNum - 1) /
                    matmulTilingData_->matmulTiling.usedCoreNum;
    params_.realRound = 0;
    params_.preCoreNum = params_.totalTileCnt % matmulTilingData_->matmulTiling.usedCoreNum;
    params_.mCntUse = params_.mCnt;
    params_.nCntUse = params_.nCnt;
    // When B is larger than A, column-major output reduces repeated cache replacement.

    params_.rowOrder = tilingL2.calOrder;

    params_.c0Size = 0;
    using A_T = typename A_TYPE::T;
    GetSizeC0<A_T>(params_.c0Size);
    params_.alignedOriM = MMV3DivCeil(matmulTilingData_->matmulTiling.M, ALIGNED_H) * ALIGNED_H;
    params_.alignedOriN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, params_.c0Size) * params_.c0Size;
    params_.alignedKaSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, params_.c0Size) * params_.c0Size;
    params_.alignedKbSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb, ALIGNED_H) * ALIGNED_H;
    // Matrices A and B are both aligned.
    if (params_.isTransposeA) {
        params_.alignedOriM = MMV3DivCeil(matmulTilingData_->matmulTiling.M, params_.c0Size) * params_.c0Size;
        params_.alignedKaSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, ALIGNED_H) * ALIGNED_H;
    }
    if (params_.isTransposeB) {
        params_.alignedOriN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, ALIGNED_H) * ALIGNED_H;
        params_.alignedKbSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb, params_.c0Size) * params_.c0Size;
    }
}

__aicore__ inline void MatmulBaseBlock::UpdateBlockCnt(uint64_t mTileIndex, uint64_t nTileIndex)
{
    params_.mTileAddrOffset = mTileIndex * params_.mCnt * matmulTilingData_->matmulTiling.singleCoreM;
    params_.nTileAddrOffset = nTileIndex * params_.nCnt * matmulTilingData_->matmulTiling.singleCoreN;

    if ((mTileIndex == (params_.mTileCntL2 - 1)) && (nTileIndex == (params_.nTileCntL2 - 1))) {
        params_.totalTileCnt = params_.mCntTail * params_.nCntTail;
        params_.mCntUse = params_.mCntTail;
        params_.nCntUse = params_.nCntTail;
    } else if (mTileIndex == (params_.mTileCntL2 - 1)) {
        params_.totalTileCnt = params_.mCntTail * params_.nCnt;
        params_.mCntUse = params_.mCntTail;
        params_.nCntUse = params_.nCnt;
    } else if (nTileIndex == (params_.nTileCntL2 - 1)) {
        params_.totalTileCnt = params_.mCnt * params_.nCntTail;
        params_.mCntUse = params_.mCnt;
        params_.nCntUse = params_.nCntTail;
    } else {
        params_.totalTileCnt = params_.mCnt * params_.nCnt;
        params_.mCntUse = params_.mCnt;
        params_.nCntUse = params_.nCnt;
    }

    params_.round =
        MMV3DivCeil(params_.totalTileCnt, static_cast<uint64_t>(matmulTilingData_->matmulTiling.usedCoreNum));
    params_.preCoreNum = params_.totalTileCnt % matmulTilingData_->matmulTiling.usedCoreNum;
    if (params_.preCoreNum == 0) {
        params_.preCoreNum = static_cast<uint64_t>(matmulTilingData_->matmulTiling.usedCoreNum);
    }
}

__aicore__ inline void MatmulBaseBlock::UpdateBasicIndex(const uint64_t roundIdx)
{
    uint64_t newBlockIdx =
        (GetCurrentBlockIdx() + matmulTilingData_->matmulTiling.usedCoreNum - params_.blockIdxStart) %
            matmulTilingData_->matmulTiling.usedCoreNum +
        roundIdx * matmulTilingData_->matmulTiling.usedCoreNum;
    uint64_t mIdx = newBlockIdx % params_.mCntUse;
    uint64_t nIdx = 0;
    if (params_.mCntUse != 0 && params_.nCntUse != 0) {
        nIdx = (newBlockIdx + newBlockIdx / MMLcm(params_.mCntUse, params_.nCntUse)) % params_.nCntUse;
    }
    params_.index = mIdx * params_.nCntUse + nIdx;
}

__aicore__ inline void MatmulBaseBlock::InitBlockIndex(uint64_t index)
{
    if (indexInit_) {
        params_.blockIdxStart = params_.blockIdxEnd; // Index of the first core for this calculation.
    } else {
        params_.blockIdxStart = index * params_.preCoreNum %
                                matmulTilingData_->matmulTiling.usedCoreNum; // First core index for this calculation.
        indexInit_ = true;
    }
    params_.blockIdxEnd = (params_.blockIdxStart + params_.preCoreNum) %
                          matmulTilingData_->matmulTiling.usedCoreNum; // Exclusive end core index, with wraparound.
    uint64_t indexStart = params_.blockIdxStart;
    uint64_t indexEnd = params_.blockIdxEnd;

    // Balance tail tiles by varying the number of rounds per core.
    if (indexStart < indexEnd) {
        // The extra-round cores form a contiguous range.
        if (GetCurrentBlockIdx() < indexStart) {
            params_.index = GetCurrentBlockIdx() * (params_.round - 1);
            params_.realRound = params_.round - 1;
        } else if (GetCurrentBlockIdx() < indexEnd) {
            params_.index = indexStart * (params_.round - 1) + (GetCurrentBlockIdx() - indexStart) * params_.round;
            params_.realRound = params_.round;
        } else {
            params_.index = (indexStart * (params_.round - 1) + params_.preCoreNum * params_.round +
                             (GetCurrentBlockIdx() - indexEnd) * (params_.round - 1));
            params_.realRound = params_.round - 1;
        }
        if (params_.rowOrder == COL_FIRST) {
            // Assign tiles in column-major order.
            params_.index = params_.index / params_.mCntUse + params_.index % params_.mCntUse * params_.nCntUse;
        }
    } else if (indexEnd < indexStart) {
        // The core range wraps around.
        if (GetCurrentBlockIdx() < indexEnd) {
            params_.index = GetCurrentBlockIdx() * params_.round;
            params_.realRound = params_.round;
        } else if (GetCurrentBlockIdx() < indexStart) {
            params_.index = indexEnd * params_.round + (GetCurrentBlockIdx() - indexEnd) * (params_.round - 1);
            params_.realRound = params_.round - 1;
        } else {
            params_.index = (indexEnd * params_.round + (indexStart - indexEnd) * (params_.round - 1) +
                             (GetCurrentBlockIdx() - indexStart) * params_.round);
            params_.realRound = params_.round;
        }
        if (params_.rowOrder == COL_FIRST) {
            // Assign tiles in column-major order.
            params_.index = params_.index / params_.mCntUse + params_.index % params_.mCntUse * params_.nCntUse;
        }
    } else {
        // Each core processes the same number of full base tiles.
        params_.index = GetCurrentBlockIdx() * params_.round;
        params_.realRound = params_.round;
        if (params_.rowOrder == COL_FIRST) {
            // Assign tiles in column-major order.
            params_.index = params_.index / params_.mCntUse + params_.index % params_.mCntUse * params_.nCntUse;
        }
    }
}

__aicore__ inline void MatmulBaseBlock::UpdateBlockParams(uint64_t mTileIndex, uint64_t nTileIndex)
{
    if ((mTileIndex == (params_.mTileCntL2 - 1)) && (nTileIndex == (params_.nTileCntL2 - 1)) &&
        (params_.index == (params_.totalTileCnt - 1))) {
        params_.singleCoreM = params_.mBaseTail;
        params_.singleCoreN = params_.nBaseTail;
    } else if ((mTileIndex == (params_.mTileCntL2 - 1)) && (params_.index >= (params_.mCntUse - 1) * params_.nCntUse)) {
        params_.singleCoreM = params_.mBaseTail;
        params_.singleCoreN = matmulTilingData_->matmulTiling.singleCoreN;
    } else if ((nTileIndex == (params_.nTileCntL2 - 1)) && ((params_.index + 1) % params_.nCntUse == 0)) {
        params_.singleCoreM = matmulTilingData_->matmulTiling.singleCoreM;
        params_.singleCoreN = params_.nBaseTail;
    } else {
        params_.singleCoreM = matmulTilingData_->matmulTiling.singleCoreM;
        params_.singleCoreN = matmulTilingData_->matmulTiling.singleCoreN;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulBaseBlock::CalcGMOffset(uint64_t mTileIndex, uint64_t nTileIndex)
{
    uint64_t mCntIndex = params_.index / params_.nCntUse;
    uint64_t nCntIndex = params_.index % params_.nCntUse;
    if constexpr (A_TYPE::format == CubeFormat::ND) {
        if (params_.isTransposeA) {
            offset_.offsetA = mCntIndex * matmulTilingData_->matmulTiling.singleCoreM + params_.mTileAddrOffset;
        } else {
            offset_.offsetA =
                mCntIndex * matmulTilingData_->matmulTiling.singleCoreM * matmulTilingData_->matmulTiling.Ka +
                params_.mTileAddrOffset * matmulTilingData_->matmulTiling.Ka;
        }
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        if (params_.isTransposeA) {
            offset_.offsetA = mCntIndex * matmulTilingData_->matmulTiling.singleCoreM * params_.alignedKaSize +
                              params_.mTileAddrOffset * params_.alignedKaSize;
        } else {
            offset_.offsetA = mCntIndex * matmulTilingData_->matmulTiling.singleCoreM * params_.c0Size +
                              params_.mTileAddrOffset * params_.c0Size;
        }
    }
    if constexpr (B_TYPE::format == CubeFormat::ND) {
        if (params_.isTransposeB) {
            offset_.offsetB =
                nCntIndex * matmulTilingData_->matmulTiling.singleCoreN * matmulTilingData_->matmulTiling.Kb +
                params_.nTileAddrOffset * matmulTilingData_->matmulTiling.Kb;
        } else {
            offset_.offsetB = nCntIndex * matmulTilingData_->matmulTiling.singleCoreN + params_.nTileAddrOffset;
        }
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        if (params_.isTransposeB) {
            offset_.offsetB = nCntIndex * matmulTilingData_->matmulTiling.singleCoreN * params_.c0Size +
                              params_.nTileAddrOffset * params_.c0Size;
        } else {
            offset_.offsetB = nCntIndex * matmulTilingData_->matmulTiling.singleCoreN * params_.alignedKbSize +
                              params_.nTileAddrOffset * params_.alignedKbSize;
        }
    }
    if constexpr (C_TYPE::format == CubeFormat::ND) {
        offset_.offsetC = (nCntIndex * matmulTilingData_->matmulTiling.singleCoreN +
                           mCntIndex * matmulTilingData_->matmulTiling.singleCoreM * matmulTilingData_->matmulTiling.N +
                           (params_.mTileAddrOffset * matmulTilingData_->matmulTiling.N + params_.nTileAddrOffset));
    } else {
        offset_.offsetC =
            (nCntIndex * matmulTilingData_->matmulTiling.singleCoreN * matmulTilingData_->matmulTiling.M +
             mCntIndex * matmulTilingData_->matmulTiling.singleCoreM * params_.c0Size +
             (params_.mTileAddrOffset * params_.c0Size + params_.nTileAddrOffset * matmulTilingData_->matmulTiling.M));
    }
    if (matmulTilingData_->matmulTiling.isBias) {
        offset_.offsetBias = nCntIndex * matmulTilingData_->matmulTiling.singleCoreN + params_.nTileAddrOffset;
    }
}

__aicore__ inline void MatmulBaseBlock::UpdateBlockIndex()
{
    if (params_.rowOrder == ROW_FIRST) {
        params_.index += 1;
    } else if (params_.rowOrder == COL_FIRST) {
        params_.index += params_.nCntUse;
        if (params_.index >= params_.totalTileCnt) {
            params_.index = params_.index % params_.totalTileCnt + 1;
        }
    }
}

#endif // TGRG_ARCH22_MAT_MUL_BASE_BLOCK_H
