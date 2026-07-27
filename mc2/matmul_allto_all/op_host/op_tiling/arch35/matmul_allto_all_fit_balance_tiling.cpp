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
 * \file matmul_allto_all_fit_balance_tiling.cpp
 * \brief arch35 架构的 AllToAll Fit Balance Tiling 实现
 */
#include <iostream>
#include "mc2_log.h"
#include "matmul_allto_all_fit_balance_tiling.h"

namespace MC2Tiling {

constexpr static uint64_t L2_CACHE_SIZE = 128 * ONE_MBYTE;
constexpr static double CALC_COMM_RATIO_THRESHOLD = 1.5;
constexpr static uint64_t LONG_TILE_ALIGN_LEN = 256;
constexpr static uint64_t ALIGN_OFFSET = LONG_TILE_ALIGN_LEN - 1;
constexpr static double COMPUTE_TIME_SCALE_FACTOR = 1.43;
constexpr static uint64_t OVERHEAD_AWARE_MAX_TILES = 3;
constexpr static uint64_t SMALL_M_BAR_FOR_OVERHEAD = 4096;
constexpr static uint64_t NUM_EIGHT = 8;
// AICPU(4P/8P)场景下，定义的常量
constexpr static double AICPU_NO_CUT_TIME_THRESHOLD_US = 80.0;
constexpr static uint64_t AICPU_MN_RATIO_THRESHOLD = 10U;
constexpr static uint64_t AICPU_MN_RATIO_MAX_TILE_CNT = 3U;

void MatmulAlltoAllFitBalanceTiling::EstimateMMCommTime()
{
    matmulPerf_.FindCubeUtil(rankTileNum_);
    matmulPerf_.GetMatmulGradient();

    // Find total matmul time and comm time
    double totalMatmulTime = matmulPerf_.MatmulTime(mmInfo_.mValue, 1);
    // 当 quantMode 为 KC_QUANT 时，使用的模版为mix，需要额外考虑vector的开销
    if (quantMode_ == QuantMode::KC_QUANT) {
        totalMatmulTime *= COMPUTE_TIME_SCALE_FACTOR;
    }
    double totalTpTime = commPerf_.CommTime(mmInfo_.mValue);
    ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / std::min(totalTpTime, totalMatmulTime));
    if (totalMatmulTime >= totalTpTime) {
        tilingM_.cutRes.shortTileAtBack = true;
    }

    uint64_t sizeOfComm = mmInfo_.mValue * mmInfo_.nValue * commPerf_.GetCommDTypeSize() * rankDim_ / ONE_MBYTE;
    OP_LOGD("MatmulAlltoAll",
            "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, sizeOfComm %lu, "
            "totalMatmulTime %f, totalCommTime %f, minTileSize %lu, mAlignLen %lu, commTimeFactor_ %f, "
            "rankDim_ %lu, rankTile %lu",
            mmInfo_.mValue, mmInfo_.nValue, mmInfo_.kValue, matmulPerf_.cubeUtil_, sizeOfComm, totalMatmulTime,
            totalTpTime, tilingM_.GetMinLen(), tilingM_.GetAlignLength(), commPerf_.commTimeFactor_, rankDim_,
            rankTileNum_);
}

void MatmulAlltoAllFitBalanceTiling::SetShortTileLen()
{
    // 通过 maxTileCnt==AICPU_MAX_TILE_CNT 识别 AICPU 模式
    if (tilingM_.tileArgs.maxTileCnt == AICPU_MAX_TILE_CNT) {
        double mmTime = matmulPerf_.MatmulTime(mmInfo_.mValue, 1);
        if (quantMode_ == QuantMode::KC_QUANT) {
            mmTime *= COMPUTE_TIME_SCALE_FACTOR;
        }
        if (mmTime < AICPU_NO_CUT_TIME_THRESHOLD_US ||
            commPerf_.CommTime(mmInfo_.mValue) < AICPU_NO_CUT_TIME_THRESHOLD_US) {
            // 短块置为总长，使基类 GetTiling 走 NoCutTiling 分支(totalLen < shortTileLen * 2)
            tilingM_.cutRes.shortTileLen = tilingM_.totalLen;
            tilingM_.cutRes.numShortTile = 1U;
            return;
        }
        // m/n超过阈值倍，最多切分3份
        if (mmInfo_.mValue > AICPU_MN_RATIO_THRESHOLD * mmInfo_.nValue) {
            tilingM_.SetMaxTileCnt(AICPU_MN_RATIO_MAX_TILE_CNT);
        }
    }

    uint64_t l2UseSize = mmInfo_.mValue * mmInfo_.kValue * mmInfo_.inMatrixADtypeSize +
                         mmInfo_.kValue * mmInfo_.nValue * mmInfo_.inMatrixBDtypeSize;
    isLargerThanL2Cache_ = l2UseSize > L2_CACHE_SIZE;
    if (isLargerThanL2Cache_) {
        tilingM_.SetMinLenByMax(tilingM_.GetMinLen() * TWO);
    }
    if (mmInfo_.mValue * mmInfo_.nValue <= coreNum_ * matmulPerf_.GetBaseM() * matmulPerf_.GetBaseN() * TWO) {
        tilingM_.SetMinLenByMax(mmInfo_.mValue);
    }

    if (mmInfo_.mValue <= SMALL_M_BAR_FOR_OVERHEAD ||
        (mmInfo_.kValue <= tilingM_.GetMinLen() && mmInfo_.mValue >= mmInfo_.kValue * NUM_EIGHT &&
         mmInfo_.mValue < mmInfo_.kValue * mmInfo_.kValue)) {
        uint64_t minTileForOverhead = (mmInfo_.mValue + OVERHEAD_AWARE_MAX_TILES - 1) / OVERHEAD_AWARE_MAX_TILES;
        // 向上对齐到 mAlignLen
        uint64_t alignLen = tilingM_.GetAlignLength();
        minTileForOverhead = (minTileForOverhead + alignLen - 1) / alignLen * alignLen;
        tilingM_.SetMinLenByMax(minTileForOverhead);
    }

    tilingM_.cutRes.shortTileLen = tilingM_.GetMinLen();
    tilingM_.cutRes.numShortTile = 1U;
}

void MatmulAlltoAllFitBalanceTiling::SetLongTileLen()
{
    // balancing the pipeline
    if (tilingM_.cutRes.shortTileAtBack) {
        double targetTime = matmulPerf_.MatmulTime(tilingM_.cutRes.shortTileLen, 1);
        tilingM_.cutRes.longTileLen = commPerf_.InverseCommTime(targetTime);
    } else {
        double targetTime = commPerf_.CommTime(tilingM_.cutRes.shortTileLen);
        tilingM_.cutRes.longTileLen = matmulPerf_.InverseMatmulTime(targetTime, 1);
    }
    OP_LOGD("MatmulAlltoAllFitBalanceTiling", "longTileLen %lu", tilingM_.cutRes.longTileLen);
}

void MatmulAlltoAllFitBalanceTiling::AdjustLongShortTileLenWhenCalcBound()
{
    if (ratioCalcComm_ > CALC_COMM_RATIO_THRESHOLD) {
        if (tilingM_.cutRes.longTileLen % LONG_TILE_ALIGN_LEN != 0) {
            uint64_t alignedLongLen = (tilingM_.cutRes.longTileLen / LONG_TILE_ALIGN_LEN) * LONG_TILE_ALIGN_LEN;
            if (alignedLongLen == 0) {
                alignedLongLen = LONG_TILE_ALIGN_LEN;
            }

            // 计算原始总长度
            uint64_t totalLen = tilingM_.cutRes.shortTileLen * tilingM_.cutRes.numShortTile +
                                tilingM_.cutRes.longTileLen * tilingM_.cutRes.numLongTile;

            // 重新分配：尽可能多地使用对齐后的长块
            tilingM_.cutRes.numLongTile = totalLen / alignedLongLen;
            uint64_t remainLen = totalLen % alignedLongLen;

            // 剩余部分作为短块
            if (remainLen > 0) {
                tilingM_.cutRes.shortTileLen = remainLen;
                tilingM_.cutRes.numShortTile = 1U;
            } else {
                tilingM_.cutRes.shortTileLen = 0U;
                tilingM_.cutRes.numShortTile = 0U;
            }

            tilingM_.cutRes.longTileLen = alignedLongLen;
            tilingM_.cutRes.totalTileCnt = tilingM_.cutRes.numLongTile + tilingM_.cutRes.numShortTile;
        }
    }
}

void MatmulAlltoAllFitBalanceTiling::AdjustLongShortTileLen()
{
    bool goodLinearityShape = (mmInfo_.kValue * mmInfo_.nValue >= LARGE_NK_BAR_BASE * ONE_MBYTE);
    tilingM_.FitTileLengthDiscrete(false, goodLinearityShape);
    AdjustLongShortTileLenWhenCalcBound();

    // When the long and short tiles are equal, the long and short pieces become one.
    if (tilingM_.cutRes.shortTileLen == tilingM_.cutRes.longTileLen) {
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
        tilingM_.cutRes.numLongTile++;
    }
    OP_LOGD("MatmulAlltoAllFitBalanceTiling",
            "Final cut: longTileLen %lu, shortTileLen %lu, numLongTile %lu, numShortTile %lu",
            tilingM_.cutRes.longTileLen, tilingM_.cutRes.shortTileLen, tilingM_.cutRes.numLongTile,
            tilingM_.cutRes.numShortTile);
}

} // namespace MC2Tiling
