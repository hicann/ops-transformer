/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_block_l2cache.h
 * \brief
 */
#ifndef MC2_MATMUL_BLOCK_L2CACHE_H
#define MC2_MATMUL_BLOCK_L2CACHE_H
#include "matmul_block.h"

namespace AscendC {
namespace MC2 {
class MatmulBlockL2Cache : public MatmulBlock {
public:
    __aicore__ inline MatmulBlockL2Cache() {}
    __aicore__ inline void Init(AscendC::tiling::TCubeTiling &tiling, Mc2Tiling::RCSTiling &cfg,
                                Mc2Tiling::Mc2L2cacheTilePara &tileL2cacheTiling);
    __aicore__ inline void UpdateBlockIndex();
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset(int32_t mTileIndex = 0, int32_t nTileIndex = 0);
    __aicore__ inline void UpdateBlockCnt(uint32_t index, int32_t mTileIndex = 0, int32_t nTileIndex = 0);

private:
    __aicore__ inline void UpdateCntUse(int32_t mTileIndex, int32_t nTileIndex);
    __aicore__ inline void UpdateSingleCorePara(int32_t mTileIndex, int32_t nTileIndex);

public:
    Mc2Tiling::Mc2L2cacheTilePara tilingL2;
    uint32_t singleCoreM = 0;      // 单次计算的singleCoreM
    uint32_t singleCoreN = 0;      // 单次计算的singleCoreN
    uint32_t mTileBaseCnt = 0;     // 每一份Tile M方向包含的base块个数
    uint32_t nTileBaseCnt = 0;     // 每一份Tile N方向包含的base块个数
    uint32_t totalTileCnt = 0;     // 每一份Tile总的base块个数
    uint32_t mTileBaseTailCnt = 0; // M方向上mTile尾块里的base块的个数
    uint32_t nTileBaseTailCnt = 0; // M方向上nTile尾块里的base块的个数
    uint32_t mCntUse = 0;
    uint32_t nCntUse = 0;
};

__aicore__ inline void MatmulBlockL2Cache::Init(AscendC::tiling::TCubeTiling &tiling, Mc2Tiling::RCSTiling &cfg,
                                                Mc2Tiling::Mc2L2cacheTilePara &tileL2cacheTiling)
{
    MatmulBlock::Init(tiling, cfg, tileL2cacheTiling);
    this->tilingL2 = tileL2cacheTiling;
    mTileBaseCnt = (mBlockCnt + tilingL2.mTileCntL2 - 1) / tilingL2.mTileCntL2;
    nTileBaseCnt = (nBlockCnt + tilingL2.nTileCntL2 - 1) / tilingL2.nTileCntL2;
    if (tilingL2.mTileBlock > 0 && tilingL2.nTileBlock > 0) {
        mTileBaseCnt = tilingL2.mTileBlock;
        nTileBaseCnt = tilingL2.nTileBlock;
    }
    totalTileCnt = mTileBaseCnt * nTileBaseCnt;
    mTileBaseTailCnt = mBlockCnt - (tilingL2.mTileCntL2 - 1) * mTileBaseCnt;
    nTileBaseTailCnt = nBlockCnt - (tilingL2.nTileCntL2 - 1) * nTileBaseCnt;
    mCntUse = mTileBaseCnt;
    nCntUse = nTileBaseCnt;
    int32_t ratio = 2;
    isRowOrder = true;
    if (tiling.N > ratio * tiling.M) {
        isRowOrder = false;
    }
}

__aicore__ inline void MatmulBlockL2Cache::UpdateCntUse(int32_t mTileIndex, int32_t nTileIndex)
{
    if ((mTileIndex == (tilingL2.mTileCntL2 - 1)) && (nTileIndex == (tilingL2.nTileCntL2 - 1))) {
        totalTileCnt = mTileBaseTailCnt * nTileBaseTailCnt;
        mCntUse = mTileBaseTailCnt;
        nCntUse = nTileBaseTailCnt;
    } else if (mTileIndex == (tilingL2.mTileCntL2 - 1)) {
        totalTileCnt = mTileBaseTailCnt * nTileBaseCnt;
        mCntUse = mTileBaseTailCnt;
        nCntUse = nTileBaseCnt;
    } else if (nTileIndex == (tilingL2.nTileCntL2 - 1)) {
        totalTileCnt = mTileBaseCnt * nTileBaseTailCnt;
        mCntUse = mTileBaseCnt;
        nCntUse = nTileBaseTailCnt;
    } else {
        totalTileCnt = mTileBaseCnt * nTileBaseCnt;
        mCntUse = mTileBaseCnt;
        nCntUse = nTileBaseCnt;
    }
}

__aicore__ inline void MatmulBlockL2Cache::UpdateBlockCnt(uint32_t index, int32_t mTileIndex, int32_t nTileIndex)
{
    (void)index;
    UpdateCntUse(mTileIndex, nTileIndex);
    int round = (totalTileCnt + tiling.usedCoreNum - 1) / tiling.usedCoreNum;
    int preCoreNum = totalTileCnt % tiling.usedCoreNum;
    if (preCoreNum == 0) {
        preCoreNum = tiling.usedCoreNum;
    }
    int roundCnt = mTileIndex * tilingL2.nTileCntL2 + nTileIndex;  // 做完一次tile块计算的次数
    int indexStart = roundCnt * preCoreNum % tiling.usedCoreNum;   // 开始运算时，preCore开始的位置
    int indexEnd = (indexStart + preCoreNum) % tiling.usedCoreNum; // 运算结束后，preCore结束的位置
    // 利用roudCnt来解决尾块负载均衡问题
    if (indexStart < indexEnd) {
        // 正常排序, preCore在整个Cores的中间
        if (block_idx < indexStart) {
            blockIndex = block_idx * (round - 1);
            blockCnt = round - 1;
        } else if (block_idx < indexEnd) {
            blockIndex = indexStart * (round - 1) + (block_idx - indexStart) * round;
            blockCnt = round;
        } else {
            blockIndex = indexStart * (round - 1) + preCoreNum * round + (block_idx - indexEnd) * (round - 1);
            blockCnt = round - 1;
        }
        if (!isRowOrder) {
            // 列优先分配
            blockIndex = blockIndex / mCntUse + blockIndex % mCntUse * nCntUse;
        }
    } else if (indexEnd < indexStart) {
        // indexEnd会翻转
        if (block_idx < indexEnd) {
            blockIndex = block_idx * round;
            blockCnt = round;
        } else if (block_idx < indexStart) {
            blockIndex = indexEnd * round + (block_idx - indexEnd) * (round - 1);
            blockCnt = round - 1;
        } else {
            blockIndex = indexEnd * round + (indexStart - indexEnd) * (round - 1) + (block_idx - indexStart) * round;
            blockCnt = round;
        }
        if (!isRowOrder) {
            // 列优先分配
            blockIndex = blockIndex / mCntUse + blockIndex % mCntUse * nCntUse;
        }
    } else {
        // 不存在尾核，基本块对齐
        blockIndex = block_idx * round;
        blockCnt = round;
        if (!isRowOrder) {
            // 列优先分配
            blockIndex = blockIndex / mCntUse + blockIndex % mCntUse * nCntUse;
        }
    }
}

__aicore__ inline void MatmulBlockL2Cache::UpdateBlockIndex()
{
    if (isRowOrder) {
        blockIndex = blockIndex + 1;
    } else {
        blockIndex = blockIndex + nCntUse;
        if (blockIndex >= totalTileCnt) {
            blockIndex = blockIndex % totalTileCnt + 1;
        }
    }
}

__aicore__ inline void MatmulBlockL2Cache::UpdateSingleCorePara(int32_t mTileIndex, int32_t nTileIndex)
{
    if ((mTileIndex == (tilingL2.mTileCntL2 - 1)) && (nTileIndex == (tilingL2.nTileCntL2 - 1)) &&
        (blockIndex == (totalTileCnt - 1))) {
        singleCoreM = mBaseTail;
        singleCoreN = nBaseTail;
    } else if ((mTileIndex == (tilingL2.mTileCntL2 - 1)) && (blockIndex >= (mCntUse - 1) * nCntUse)) {
        singleCoreM = mBaseTail;
        singleCoreN = tiling.baseN;
    } else if ((nTileIndex == (tilingL2.nTileCntL2 - 1)) && ((blockIndex + 1) % nCntUse == 0)) {
        singleCoreM = tiling.baseM;
        singleCoreN = nBaseTail;
    } else {
        singleCoreM = tiling.baseM;
        singleCoreN = tiling.baseN;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulBlockL2Cache::CalcGMOffset(int32_t mTileIndex, int32_t nTileIndex)
{
    UpdateSingleCorePara(mTileIndex, nTileIndex);
    // now only support half dtye;
    constexpr int c0Size = 16;
    auto alignedKb = (tiling.Kb + 15) / 16 * 16;
    auto alignedKa = (tiling.Ka + 15) / 16 * 16;
    uint32_t mCntIndex = blockIndex / nCntUse;
    uint32_t nCntIndex = blockIndex % nCntUse;
    int64_t mTileAddrOffset = mTileIndex * mTileBaseCnt * tiling.baseM;
    int64_t nTileAddrOffset = nTileIndex * nTileBaseCnt * tiling.baseN;
    if constexpr (A_TYPE::format == CubeFormat::ND) {
        if (isTransposeA) {
            offset.offsetA = mCntIndex * tiling.baseM + mTileAddrOffset;
        } else {
            offset.offsetA = mCntIndex * tiling.baseM * tiling.Ka + mTileAddrOffset * tiling.Ka;
        }
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        if (isTransposeA) {
            offset.offsetA = mCntIndex * tiling.baseM * alignedKa + mTileAddrOffset * alignedKa;
        } else {
            offset.offsetA = mCntIndex * tiling.baseM * c0Size + mTileAddrOffset * c0Size;
        }
    }
    if constexpr (B_TYPE::format == CubeFormat::ND) {
        if (isTransposeB) {
            offset.offsetB = nCntIndex * tiling.baseN * tiling.Kb + nTileAddrOffset * tiling.Kb;
        } else {
            offset.offsetB = nCntIndex * tiling.baseN + nTileAddrOffset;
        }
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        if (isTransposeB) {
            offset.offsetB = nCntIndex * tiling.baseN * c0Size + nTileAddrOffset * c0Size;
        } else {
            offset.offsetB = nCntIndex * tiling.baseN * alignedKb + nTileAddrOffset * alignedKb;
        }
    }
    if constexpr (C_TYPE::format == CubeFormat::ND) {
        offset.offsetC = nCntIndex * tiling.baseN + mCntIndex * tiling.baseM * tiling.N +
                         (mTileAddrOffset * tiling.N + nTileAddrOffset);
    } else if constexpr (C_TYPE::format == CubeFormat::NZ) {
        offset.offsetC = nCntIndex * tiling.baseN * tiling.M + mCntIndex * tiling.baseM * c0Size +
                         (mTileAddrOffset * c0Size + nTileAddrOffset * tiling.M);
    }
    offset.offsetBias = nCntIndex * tiling.baseN + nTileAddrOffset;
}

} // namespace MC2

template <bool Mc2L2Cache>
struct BlockType {
    __aicore__ inline BlockType(){};
};

template <>
struct BlockType<false> {
    __aicore__ inline BlockType(){};
    using PARAMS = MC2::MatmulBlock;
};

template <>
struct BlockType<true> {
    __aicore__ inline BlockType(){};
    using PARAMS = MC2::MatmulBlockL2Cache;
};
} // namespace AscendC
#endif // MC2_MATMUL_BLOCK_L2CACHE_H
