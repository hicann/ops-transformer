/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_MATMUL_AIV_KERNEL_COMMON_H
#define MC2_MATMUL_AIV_KERNEL_COMMON_H

#include <cstdint>

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace Mc2MatmulAiv {

constexpr uint32_t SWIZZLE_BUFFER_NUM = 2U;

inline __aicore__ void GetSwizzledBlockIdx(int32_t loopIdx, int32_t mLoop, int32_t nLoop, int32_t swizzleDirection,
                                           int32_t swizzleCount, int64_t &mIdx, int64_t &nIdx)
{
    uint32_t inBatchIdx = loopIdx % (mLoop * nLoop);
    if (swizzleDirection == 0) { // Zn
        uint32_t tileBlockLoop = (mLoop + swizzleCount - 1) / swizzleCount;
        uint32_t tileBlockIdx = inBatchIdx / (swizzleCount * nLoop);
        uint32_t inTileBlockIdx = inBatchIdx % (swizzleCount * nLoop);

        uint32_t nRow = swizzleCount;
        if (tileBlockIdx == tileBlockLoop - 1) {
            nRow = mLoop - swizzleCount * tileBlockIdx;
        }
        mIdx = tileBlockIdx * swizzleCount + inTileBlockIdx % nRow;
        nIdx = inTileBlockIdx / nRow;
        if (tileBlockIdx % SWIZZLE_BUFFER_NUM != 0) {
            nIdx = nLoop - nIdx - 1;
        }
    } else if (swizzleDirection == 1) { // Nz
        uint32_t tileBlockLoop = (nLoop + swizzleCount - 1) / swizzleCount;
        uint32_t tileBlockIdx = inBatchIdx / (swizzleCount * mLoop);
        uint32_t inTileBlockIdx = inBatchIdx % (swizzleCount * mLoop);

        uint32_t nCol = swizzleCount;
        if (tileBlockIdx == tileBlockLoop - 1) {
            nCol = nLoop - swizzleCount * tileBlockIdx;
        }
        mIdx = inTileBlockIdx / nCol;
        nIdx = tileBlockIdx * swizzleCount + inTileBlockIdx % nCol;
        if (tileBlockIdx % SWIZZLE_BUFFER_NUM != 0) {
            mIdx = mLoop - mIdx - 1;
        }
    }
}

template <class TileShape, class Coord>
inline __aicore__ Coord GetBlockLocCoord(Coord blockIdxCoord)
{
    return Coord{blockIdxCoord.m() * TileShape::M, blockIdxCoord.n() * TileShape::N, blockIdxCoord.k() * TileShape::K};
}

template <class TileShape, class Coord>
inline __aicore__ Coord GetBlockSizeCoord(Coord blockIdxCoord, Coord blockLocCoord, int32_t mLoop, int32_t mSize,
                                          int32_t nLoop, int32_t nSize, int32_t kSize)
{
    uint32_t mActual = (blockIdxCoord.m() == (mLoop - 1)) ? (mSize - blockLocCoord.m()) : TileShape::M;
    uint32_t nActual = (blockIdxCoord.n() == (nLoop - 1)) ? (nSize - blockLocCoord.n()) : TileShape::N;
    uint32_t kActual = kSize;
    return Coord{mActual, nActual, kActual};
}

} // namespace Mc2MatmulAiv

#endif // MC2_MATMUL_AIV_KERNEL_COMMON_H
