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
 * \file reduce_scatter_formulaic_tiling.cpp
 * \brief
 */
#include "mc2_log.h"
#include "reduce_scatter_formulaic_tiling.h"

constexpr static uint64_t AICPU_M_TILE_CAP = 4; // AICPU通路通信切分膨胀较大，限制最大切分轮数
constexpr static uint64_t RANK_SIZE_16P = 16U;  // 16P跨机场景
constexpr static uint64_t EVEN_SPLIT2_M_PER_RANK = 1024U; // 单rank M轴长度上限

void MMPlusReduceScatter::SetCommTimeFactorForA5()
{
    commPerf_.ChangeCommTimeFactorByDivision(REDUCESCATTER_COMMTIME_FACTOR);
    return;
}

void MMPlusReduceScatter::SetCommTimeFactorForOther()
{
    // 通算并行时通信有膨胀，大K场景膨胀明显，做特殊处理
    bool bwGrowthByUtil = matmulPerf_.cubeUtil_ < PART_L2_UTIL;
    bool bwGrowthByShape = clusterInfo_.kValue >= LARGE_K_BOUNDARY && clusterInfo_.nValue > LARGE_N_BOUNDARY;
    bool smallDim = rankDim_ <= MIN_COMM_RANKDIM || rankTileNum_ <= MatmulPerformance::SMALL_RANKTILE;
    bwGrowthByUtil = bwGrowthByUtil && !smallDim;
    bwGrowthByShape = bwGrowthByShape && !smallDim;
    if (bwGrowthByUtil) { // 0.85 is max cube utilizaion
        commPerf_.ChangeCommTimeFactorByDivision(scatterLargerNKCommGrowRatio1);
    } else if (bwGrowthByShape) { // LargeK and LargeN
        commPerf_.ChangeCommTimeFactorByDivision(scatterLargerNKCommGrowRatio2);
    }
    commPerf_.ChangeCommTimeFactorByDivision(commGrowRatio); // 1.15x time of factor
    if (clusterInfo_.socType == SocVersion::SOC910_93) {
        commPerf_.ChangeCommTimeFactorByDivision(0.6); // 0.6x time of factor
    }
    return;
}

void MMPlusReduceScatter::SetCommTimeFactor()
{
    if (clusterInfo_.socType == SocVersion::SOC950) {
        SetCommTimeFactorForA5();
    } else {
        SetCommTimeFactorForOther();
    }
    return;
}

void MMPlusReduceScatter::EstimateKernelTime()
{
    SetCommTimeFactor();
    // 预测耗时
    double totalMatmulTime = EstimateTotalMatmulTime();
    double totalTpTime = EstimateTotalCommTime();
    ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / std::min(totalTpTime, totalMatmulTime));
    if (totalMatmulTime >= totalTpTime) {
        tilingM_.cutRes.shortTileAtBack = true;
    }

    bool decreaseMinLen = (totalTpTime > totalMatmulTime * UNBALANCE_RATIO) && (clusterInfo_.nValue >= FOUR_KILO) &&
                          (clusterInfo_.kValue >= ONE_KILO);
    if (decreaseMinLen) { // 中、大shape通过减少初始值的方式鼓励多切
        tilingM_.SetMinLenByMin(tilingM_.GetAlignLength());
    }
    // 确定性通信场景适当少切
    bool increaseMinLen =
        (ratioCalcComm_ < commGrowRatio) ||
        ((totalMatmulTime > totalTpTime * UNBALANCE_RATIO) &&
         (clusterInfo_.mValue >= tilingM_.GetAlignLength() * FOUR)); // 4 = 2 x 2，m太小时增加初始值会导致不切
    increaseMinLen = increaseMinLen && (clusterInfo_.nValue <= FOUR_KILO) && deterministicSoc910B_; // n小时要少切
    if (increaseMinLen) { // 通过增加初始值的方式鼓励少切
        tilingM_.SetMinLenByMax(tilingM_.GetAlignLength() * TWO);
    }

    // 当 M * N < 4 * 1024 * 1024时，matmul性能变差，需要平衡matmul性能与流水掩盖
    // 假如 matmul时间 >= 通讯时间的3.5倍， 尾块配平会保证大主块，不用额外改大matmulMinTileSize参数
    // 假如 matmul时间 < 通讯时间的2倍，计算时间接近通讯时间，此时并行理论收益大：
    // 希望多切分获得并行收益，所以不再另外改大matmulMinTileSize参数
    increaseMinLen =
        (totalMatmulTime >= totalTpTime * TIME_LOWER_RATIO) && (totalMatmulTime < totalTpTime * TIME_UPPER_RATIO);
    if (increaseMinLen) { // 通过增加初始值的方式鼓励少切
        tilingM_.SetMinLenByMax(FOUR_MBYTE / clusterInfo_.nValue / rankTileNum_);
    }
    OP_LOGD("MatmulReduceScatter",
            "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, "
            "totalMatmulTime %f, totalTP %f, minTileSize %lu, mAlignLen %lu, commTimeFactor_ %f",
            clusterInfo_.mValue, clusterInfo_.nValue, clusterInfo_.kValue, matmulPerf_.cubeUtil_, totalMatmulTime,
            totalTpTime, tilingM_.GetMinLen(), tilingM_.GetAlignLength(), commPerf_.commTimeFactor_);
}

void MMPlusReduceScatter::SelectTilingMethod()
{
    if (isAicpuComm_ && rankDim_ > 8) { // AICPU大于8P场景通信切分膨胀较大，限制最大切分轮数
        tilingM_.SetMaxTileCnt(AICPU_M_TILE_CAP);
    }
    if (tilingM_.SetShortTileLen()) { // 如果shape太小就不切
        return;
    }
    // 16P AICPU计算bound场景小M场景：通信切分膨胀大，切分轮次越少越好，尽量等分利用aicpu展开缓存
    bool evenSplitTwo = isAicpuComm_ && (rankDim_ == RANK_SIZE_16P) &&
                        (clusterInfo_.mValue <= EVEN_SPLIT2_M_PER_RANK) &&
                        (clusterInfo_.mValue % (TWO * tilingM_.GetAlignLength()) == 0U) &&
                        (clusterInfo_.inMatrixADtypeSize == TWO) && tilingM_.cutRes.shortTileAtBack;
    if (evenSplitTwo) {
        tilingM_.cutRes.longTileLen = tilingM_.totalLen / TWO;
        tilingM_.cutRes.numLongTile = TWO;
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
        tilingM_.cutRes.totalTileCnt = TWO;
        OP_LOGD("MatmulReduceScatter",
                "AICPU 16P even split 2: shortTileAtBack %d, longTileLen %lu, "
                "numLongTile %lu, shortTileLen %lu, numShortTile %lu",
                tilingM_.cutRes.shortTileAtBack, tilingM_.cutRes.longTileLen, tilingM_.cutRes.numLongTile,
                tilingM_.cutRes.shortTileLen, tilingM_.cutRes.numShortTile);
        return;
    }
    // 流水配平，找到理论切分长度
    if (tilingM_.cutRes.shortTileAtBack) {
        ShortAtEndCalcBoundBalancing();
    } else {
        ShortAtFrontCommBoundBalancing();
    }

    // 生成切分
    bool smallDimAlignUp = (rankDim_ <= MIN_COMM_RANKDIM) && tilingM_.cutRes.shortTileAtBack &&
                           (clusterInfo_.nValue < SMALL_SHAPE_BAR ||
                            clusterInfo_.kValue < SMALL_SHAPE_BAR); // 2p，计算Bound，且N轴或者K轴很小
    bool goodLinearityShape = (clusterInfo_.kValue * clusterInfo_.nValue >= LARGE_NK_BAR_BASE * ONE_MBYTE);
    tilingM_.FitTileLengthDiscrete(smallDimAlignUp, goodLinearityShape);
    OP_LOGD("MatmulReduceScatter",
            "Final cut: shortTileAtBack %d, longTileLen %lu, "
            "numLongTile %lu, shortTileLen %lu, numShortTile %lu",
            tilingM_.cutRes.shortTileAtBack, tilingM_.cutRes.longTileLen, tilingM_.cutRes.numLongTile,
            tilingM_.cutRes.shortTileLen, tilingM_.cutRes.numShortTile);
}
