/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_reduce_fit_balance_tiling.cpp
 * \brief
 */
#include "all_reduce_fit_balance_tiling.h"
#include <cmath>
#include "mc2_log.h"

constexpr static double CALC_COMM_BOUND_BAR = 4;
constexpr static double MM_EXPANSION_TIME = 60;
constexpr static double COMM_EXPANSION_TIME = 100;
constexpr static uint64_t L2_CACHE_SIZE = 128 * ONE_MBYTE;
constexpr static uint64_t COMM_EXPANSION_TILENUM = 4;
constexpr static uint64_t SMALL_M_VALUE = 1024;

void MMAllReduceFitBalanceTiling::EstimateMMCommTime()
{
    matmulPerf_.FindCubeUtil(rankTileNum_);
    matmulPerf_.GetMatmulGradient();

    // Find total matmul time and comm time
    double totalMatmulTime = matmulPerf_.MatmulTime(mmInfo_.mValue, 1);
    double totalTpTime = commPerf_.CommTime(mmInfo_.mValue);
    if (totalMatmulTime >= totalTpTime) {
        tilingM_.cutRes.shortTileAtBack = true;
    }

    ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / std::min(totalTpTime, totalMatmulTime));

    uint64_t sizeOfComm = mmInfo_.mValue * mmInfo_.nValue * commPerf_.GetCommDTypeSize() / ONE_MBYTE;
    OP_LOGD("MatmulAllReduce",
            "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, sizeOfComm %lu, "
            "totalMatmulTime %f, totalCommTime %f, minTileSize %lu, mAlignLen %lu, commTimeFactor_ %f, "
            "rankDim_ %lu, rankTile %lu",
            mmInfo_.mValue, mmInfo_.nValue, mmInfo_.kValue, matmulPerf_.cubeUtil_, sizeOfComm, totalMatmulTime,
            totalTpTime, tilingM_.GetMinLen(), tilingM_.GetAlignLength(), commPerf_.commTimeFactor_, rankDim_,
            rankTileNum_);
}

uint64_t MMAllReduceFitBalanceTiling::Align256(uint64_t value) const
{
    return ((value + 255) / 256) * 256; // 255, 256: align to 256
}

void MMAllReduceFitBalanceTiling::SetCommBoundTile()
{
    // Set the tile of comm bound.
    if ((mmInfo_.mValue <= SMALL_M_VALUE) || tilingM_.cutRes.shortTileAtBack ||
        (ratioCalcComm_ < CALC_COMM_BOUND_BAR) || (matmulPerf_.MatmulTime(mmInfo_.mValue, 1) < MM_EXPANSION_TIME)) {
        return;
    }
    uint64_t tileLen = (mmInfo_.mValue + 1) / THREE;
    uint64_t alignedTileLen = Align256(tileLen);
    tilingM_.cutRes.shortTileLen = alignedTileLen;
    tilingM_.cutRes.numShortTile = 1U;
    tilingM_.cutRes.numLongTile = 1U;
    tilingM_.cutRes.longTileLen = mmInfo_.mValue - alignedTileLen;
    OP_LOGD("MMAllReduceFitBalanceTiling", "SetCommBoundTile: initial longTileLen %lu, shortTileLen %lu",
            tilingM_.cutRes.longTileLen, tilingM_.cutRes.shortTileLen);
}

void MMAllReduceFitBalanceTiling::SetShortTileLen()
{
    tilingM_.SetMinLenByMax(commPerf_.InverseCommTime(COMM_EXPANSION_TIME));
    tilingM_.SetMinLenByMax(matmulPerf_.InverseMatmulTime(MM_EXPANSION_TIME, rankTileNum_));
    // Encourage larger cutLen if the matrix size is greater than the L2 cache size
    uint64_t l2UseSize = mmInfo_.mValue * mmInfo_.kValue * mmInfo_.inMatrixADtypeSize +
                         mmInfo_.kValue * mmInfo_.nValue * mmInfo_.inMatrixBDtypeSize;
    if (l2UseSize > L2_CACHE_SIZE) {
        tilingM_.SetMinLenByMax(tilingM_.GetMinLen() * TWO);
    }
    tilingM_.cutRes.shortTileLen = tilingM_.GetMinLen();
    tilingM_.cutRes.numShortTile = 1U;
    OP_LOGD("MMAllReduceFitBalanceTiling", "initialize shortTileLen %lu", tilingM_.cutRes.shortTileLen);
}

void MMAllReduceFitBalanceTiling::SetLongTileLen()
{
    // Balancing the pipeline
    if (tilingM_.cutRes.shortTileAtBack) {
        double targetTime = matmulPerf_.MatmulTime(tilingM_.cutRes.shortTileLen, rankTileNum_);
        tilingM_.cutRes.longTileLen = commPerf_.InverseCommTime(targetTime);
    } else {
        double targetTime = commPerf_.CommTime(tilingM_.cutRes.shortTileLen);
        tilingM_.cutRes.longTileLen = matmulPerf_.InverseMatmulTime(targetTime, rankTileNum_);
    }
    tilingM_.cutRes.longTileLen = std::max(tilingM_.cutRes.longTileLen, tilingM_.cutRes.shortTileLen);
    OP_LOGD("MMAllReduceFitBalanceTiling", "initialize longTileLen %lu", tilingM_.cutRes.longTileLen);

    // Splitting will lead to time consumption, so the longTile should be adjusted accordingly.
    if ((mmInfo_.mValue - tilingM_.cutRes.shortTileLen) / tilingM_.cutRes.longTileLen >= COMM_EXPANSION_TILENUM) {
        tilingM_.cutRes.longTileLen = tilingM_.cutRes.longTileLen * TWO;
        tilingM_.SetMinLenByMax(tilingM_.GetMinLen() * TWO);
        OP_LOGD("MMAllReduceFitBalanceTiling", "adjust longTileLen %lu", tilingM_.cutRes.longTileLen);
    }
}

void MMAllReduceFitBalanceTiling::AdjustLongShortTileLen()
{
    bool goodLinearityShape = (mmInfo_.kValue * mmInfo_.nValue >= LARGE_NK_BAR_BASE * ONE_MBYTE);
    tilingM_.FitTileLengthDiscrete(false, goodLinearityShape);
    SetCommBoundTile();
    AlignLongTileLen();
    AdjustAicpuPertileEqualSplit();

    // When the long and short tiles are equal, merge them
    if (tilingM_.cutRes.shortTileLen == tilingM_.cutRes.longTileLen) {
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
        tilingM_.cutRes.numLongTile++;
    }
}

void MMAllReduceFitBalanceTiling::AdjustAicpuPertileEqualSplit()
{
    // AICPU通信优化后，等分切分时相同大小的tile可复用缓存，效率更高；
    // 8卡及以上场景通信膨胀开销显著，强制等分切分以降低该开销
    if (!isAicpuMode_ || !isPertileFp8_ || rankDim_ < AICPU_PERTILE_MIN_RANK_DIM) {
        return;
    }
    uint64_t totalLen = mmInfo_.mValue;
    uint64_t alignedBlocks = (totalLen + AICPU_PERTILE_ALIGN_BASE - 1) / AICPU_PERTILE_ALIGN_BASE;
    if (alignedBlocks < AICPU_PERTILE_MAX_TILE_CNT) {
        return;
    }
    uint64_t longTileLen = ((alignedBlocks + 1) / AICPU_PERTILE_MAX_TILE_CNT) * AICPU_PERTILE_ALIGN_BASE;
    uint64_t tailLen = totalLen - longTileLen * (AICPU_PERTILE_MAX_TILE_CNT - 1);
    if (longTileLen == 0 || tailLen == 0) {
        return;
    }
    tilingM_.cutRes.longTileLen = longTileLen;
    tilingM_.cutRes.numLongTile = AICPU_PERTILE_MAX_TILE_CNT - 1;
    tilingM_.cutRes.shortTileLen = tailLen;
    tilingM_.cutRes.numShortTile = 1;
    tilingM_.cutRes.shortTileAtBack = true;
    tilingM_.cutRes.totalTileCnt = AICPU_PERTILE_MAX_TILE_CNT;
    OP_LOGD("MMAllReduceFitBalanceTiling",
            "AICPU pertile equal split: M=%lu, longTileLen=%lu, numLongTile=%lu, "
            "shortTileLen=%lu, numShortTile=%lu",
            totalLen, tilingM_.cutRes.longTileLen, tilingM_.cutRes.numLongTile, tilingM_.cutRes.shortTileLen,
            tilingM_.cutRes.numShortTile);
}

void MMAllReduceFitBalanceTiling::AlignLongTileLen()
{
    if (!isAlign_) {
        return;
    }
    uint64_t baseM = matmulPerf_.GetBaseM();
    if (baseM == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("MMAllReduceFitBalanceTiling", "baseM", std::to_string(baseM).c_str(),
                                              "The value of baseM cannot be 0");
        return;
    }
    if (tilingM_.cutRes.longTileLen % baseM == 0) {
        return;
    }
    uint64_t totalLen = mmInfo_.mValue;
    // Floor-align longTileLen to baseM
    uint64_t alignedLongLen = (tilingM_.cutRes.longTileLen / baseM) * baseM;
    if (alignedLongLen == 0) {
        alignedLongLen = baseM;
    }

    // Redistribute tiles accounting for numLongTile
    tilingM_.cutRes.numLongTile = totalLen / alignedLongLen;
    uint64_t remainLen = totalLen % alignedLongLen;
    if (remainLen > 0) {
        tilingM_.cutRes.shortTileLen = remainLen;
        tilingM_.cutRes.numShortTile = 1U;
    } else {
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
    }
    tilingM_.cutRes.longTileLen = alignedLongLen;
    tilingM_.cutRes.totalTileCnt = tilingM_.cutRes.numLongTile + tilingM_.cutRes.numShortTile;

    // When numLongTile == 1 and shortTileLen >= longTileLen, swap them
    if (tilingM_.cutRes.numLongTile == 1U && tilingM_.cutRes.numShortTile == 1U &&
        tilingM_.cutRes.shortTileLen >= tilingM_.cutRes.longTileLen) {
        std::swap(tilingM_.cutRes.shortTileLen, tilingM_.cutRes.longTileLen);
    }

    OP_LOGD("MMAllReduceFitBalanceTiling",
            "Final: longTileLen %lu, numLongTile %lu, shortTileLen %lu, numShortTile %lu, baseM %lu",
            tilingM_.cutRes.longTileLen, tilingM_.cutRes.numLongTile, tilingM_.cutRes.shortTileLen,
            tilingM_.cutRes.numShortTile, baseM);
}
