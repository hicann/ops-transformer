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
 * \file allto_all_formulaic_tiling.cpp
 * \brief
 */
#include "allto_all_formulaic_tiling.h"
#include "mc2_log.h"

/**
 * @brief 时间估算
 *
 */
void AlltoAllMM::EstimateKernelTime()
{
    SetCommTimeFactor(); // 设置通信的时间因子
    double totalMatmulTime = 0;
    double totalTpTime = 0;
    // 预测计算、通信任务耗时
    if (socVersion_2201 == SocVersion_2201::SOC910_93) {
        totalMatmulTime = EstimateTotalMatmulTime();
        totalTpTime = EstimateTotalCommTime(); // 通信时间
    } else {
        // EstimateTotalMatmulTime单独估算的是matmul的计算时间，A5时，这里乘一个参数考虑重排的时间
        totalMatmulTime = EstimateTotalMatmulTime() * COMPUTE_TIME_SCALE_FACTOR;
        totalTpTime = EstimateTotalCommTime(); // 通信时间
    }
    ratioCalcComm_ =
        (std::max(totalTpTime, totalMatmulTime) /
         std::min(totalTpTime, totalMatmulTime)); // ratio=1时，通算平衡，完美掩盖；越大，计算或者通信的瓶颈越严重
    if (totalMatmulTime >= totalTpTime) {
        tilingM_.cutRes.shortTileAtBack = true;
    }

    // 根据通信、计算耗时比例，调整切分约束
    // 2x compute time 中、大shape通过减少初始值的方式鼓励多切
    bool decreaseMinLenFlag = (totalTpTime > totalMatmulTime * TWO) && (clusterInfo_.kValue >= LARGE_K_BOUNDARY);
    if (decreaseMinLenFlag) {
        tilingM_.SetMinLenByMin(tilingM_.GetAlignLength());
    }
    bool increaseMinLenFlag =
        (totalMatmulTime >= totalTpTime * TIME_LOWER_RATIO) && (totalMatmulTime < totalTpTime * TIME_UPPER_RATIO);
    if (increaseMinLenFlag) { // 通过增加初始值的方式鼓励少切
        tilingM_.SetMinLenByMax(FOUR_MBYTE / clusterInfo_.nValue);
    }
    PrintEstimateKernelTimeResult(totalMatmulTime, totalTpTime);
}

/**
 * @brief 设置时间因子
 *
 */
void AlltoAllMM::SetCommTimeFactor()
{
    if (socVersion_2201 == SocVersion_2201::SOC910_93) {
        OP_LOGD("AlltoAllMatmul, Current socVersion is SOC910_93.");
        tilingM_.SetMaxTileCnt(MAX_TILE_CNT_A3);                                        // 最多切8轮
        uint64_t rankDim = std::max(static_cast<uint64_t>(rankDim_), MIN_COMM_RANKDIM); // 并行维度最小为2
        // 采用 8 die拟合，通信查表使用总数据量，实际时间 = 总数据量 * 2 * (rankDim - 1) / rankDim
        double fittingRatio = (static_cast<double>(FITTING_RANK) - 1.0) / static_cast<double>(FITTING_RANK);
        double currentRatio = (static_cast<double>(rankDim) - 1.0) / static_cast<double>(rankDim);
        double CommTimeFactor = FULL_MESH_TIME_FACTOR * (fittingRatio / currentRatio) * COMM_TIME_SCALE_FACTOR;
        OP_LOGD("AlltoAllMatmul", "Current commTimeFactor is %f", CommTimeFactor);
        commPerf_.ChangeCommTimeFactorByDivision(CommTimeFactor);
    } else {
        OP_LOGD("AlltoAllMatmul, Current platform is A5 (DAV_3510).");
        // A5上的时间因子AlltoAll暂时定义为2
        commPerf_.ChangeCommTimeFactorByDivision(TWO); // 2x time of factor
    }
}

/**
 * @brief 打印时间估计后的结果
 *
 * @param totalMatmulTime 计算时间
 * @param totalTpTime 通信时间
 */
void AlltoAllMM::PrintEstimateKernelTimeResult(double totalMatmulTime, double totalTpTime)
{
    OP_LOGD("AlltoAllMatmul",
            "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, "
            "totalMatmulTime %f, totalCommTime %f, minTileSize %lu, mAlignLen %lu, commTimeFactor_ %f, "
            "rankDim_ %lu, rankTile %lu",
            clusterInfo_.mValue, clusterInfo_.nValue, clusterInfo_.kValue, matmulPerf_.cubeUtil_, totalMatmulTime,
            totalTpTime, tilingM_.GetMinLen(), tilingM_.GetAlignLength(), commPerf_.commTimeFactor_, rankDim_,
            rankTileNum_);
}

/**
 * @brief 选择tiling切分策略，产生最终切分
 *
 */
void AlltoAllMM::SelectTilingMethod()
{
    if (tilingM_.SetShortTileLen()) { // 如果shape太小就不切
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
    OP_LOGD("AlltoAll with Matmul",
            "Final cut: shortTileAtBack %d, longTileLen %lu"
            ", numLongTile %lu, shortTileLen %lu, numShortTile %lu",
            tilingM_.cutRes.shortTileAtBack, tilingM_.cutRes.longTileLen, tilingM_.cutRes.numLongTile,
            tilingM_.cutRes.shortTileLen, tilingM_.cutRes.numShortTile);
}
