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
 * \file all_gather_fit_balance_tiling.cpp
 * \brief
 */
#include "mc2_log.h"
#include "all_gather_fit_balance_tiling.h"

constexpr static double MM_EXPANSION_TIME = 40;
constexpr static double COMM_EXPANSION_TIME = 40;

void AllGatherMMFitBalanceTiling::EstimateMMCommTime()
{
    matmulPerf_.FindCubeUtil(rankTileNum_);
    matmulPerf_.GetMatmulGradient();

    // Find total matmul time and comm time
    double totalMatmulTime = matmulPerf_.MatmulTime(mmInfo_.mValue, rankDim_);
    double totalTpTime = commPerf_.CommTime(mmInfo_.mValue);
    double frontUtil = matmulPerf_.FindCubeUtilByL2Usage(mmInfo_.mValue, 1);
    if (std::abs(frontUtil) > 1e-10) {
        frontMMTime_ = matmulPerf_.MatmulTime(mmInfo_.mValue, 1) / frontUtil;
    } else {
        frontMMTime_ = matmulPerf_.MatmulTime(mmInfo_.mValue, 1);
    }
    tilingM_.cutRes.shortTileAtBack = true;
    ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / std::min(totalTpTime, totalMatmulTime));

    uint64_t sizeOfComm = mmInfo_.mValue * mmInfo_.kValue * (rankDim_ - 1) / ONE_MBYTE * commPerf_.GetCommDTypeSize();
    OPS_LOG_D("AllGatherMatmul",
              "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, sizeOfComm %lu, "
              "totalMatmulTime %f, totalCommTime %f, minTileSize %lu, mAlignLen %lu, commTimeFactor_ %f, "
              "rankDim_ %lu, rankTile %lu",
              mmInfo_.mValue, mmInfo_.nValue, mmInfo_.kValue, matmulPerf_.cubeUtil_, sizeOfComm, totalMatmulTime,
              totalTpTime, tilingM_.GetMinLen(), tilingM_.GetAlignLength(), commPerf_.commTimeFactor_, rankDim_,
              rankTileNum_);
}

void AllGatherMMFitBalanceTiling::SetLongTileLen()
{
    // Long tile should overlap the cost time of local tile
    tilingM_.cutRes.longTileLen = commPerf_.InverseCommTime(frontMMTime_);
}

void AllGatherMMFitBalanceTiling::SetShortTileLen()
{
    // Encourage split more if the comm and calc is balanced and the cost of cutLen is sufficiently high
    bool isCalcCommBalance = ratioCalcComm_ < CALC_COMM_RATIO;
    uint64_t cutLen = tilingM_.GetAlignLength() / TWO;
    double mmCost = matmulPerf_.MatmulTime(cutLen, rankDim_);
    double commCost = commPerf_.CommTime(cutLen);
    if (isCalcCommBalance && (mmCost > MM_EXPANSION_TIME) && (commCost > COMM_EXPANSION_TIME)) {
        tilingM_.SetAlignLength(cutLen);
        tilingM_.SetMinLenByMin(cutLen);
    }

    tilingM_.cutRes.shortTileLen = tilingM_.GetMinLen();
    tilingM_.cutRes.numShortTile = 1U;
}

void AllGatherMMFitBalanceTiling::AdjustLongShortTileLen()
{
    tilingM_.FitTileLengthDiscrete(false, true, true);
    // When the long and short tiles are equal, the long and short pieces become one.
    if (tilingM_.cutRes.shortTileLen == tilingM_.cutRes.longTileLen) {
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
        tilingM_.cutRes.numLongTile++;
    }
}
