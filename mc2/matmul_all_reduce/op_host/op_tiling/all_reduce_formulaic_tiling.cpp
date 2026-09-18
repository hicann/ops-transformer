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
 * \file all_reduce_formulaic_tiling.cc
 * \brief
 */
#include "all_reduce_formulaic_tiling.h"
#include <cmath>
#include "mc2_log.h"
using namespace std;

void MMPlusAllReduce::SetCommTimeFactorForA5()
{
    commPerf_.ChangeCommTimeFactorByDivision(ALLREDUCE_COMMTIME_FACTOR);
    commPerf_.ChangeCommTimeFactorByDivision(static_cast<double>(FITTING_RANK) / static_cast<double>(rankDim_));
    return;
}

void MMPlusAllReduce::SetCommTimeFactorForOther()
{
    commPerf_.ChangeCommTimeFactorByDivision(commGrowRatio);
    return;
}

void MMPlusAllReduce::SetCommTimeFactor()
{
    if (npuArch_ == Ops::Base::DAV_3510) { // A5, __NPU_ARCH__ == 3510
        SetCommTimeFactorForA5();
        // end __NPU_ARCH__ == 3510
    } else {
        SetCommTimeFactorForOther();
    }
    return;
}

void MMPlusAllReduce::EstimateKernelTime()
{
    SetCommTimeFactor();

    // find total matmul time and tp time
    double totalMatmulTime = matmulPerf_.MatmulTime(clusterInfo_.mValue, 1); // AllReduce先切totalTileCnt, 再切rankDim
    double totalTpTime = commPerf_.CommTime(clusterInfo_.mValue);
    if (totalMatmulTime >= totalTpTime) {
        tilingM_.cutRes.shortTileAtBack = true;
    }

    // 通算均衡且shape较小时鼓励切分
    ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / std::min(totalTpTime, totalMatmulTime));
    bool isCalcCommBalance = ratioCalcComm_ < 2.0;
    bool smallM = (clusterInfo_.mValue < tilingM_.GetMinLen() * 2);
    bool largeNK = (clusterInfo_.nValue * clusterInfo_.kValue) >= (LARGE_NK_BAR * ONE_MBYTE);
    if (isCalcCommBalance && smallM && largeNK) {
        tilingM_.SetMinLenByMin(static_cast<uint64_t>(tilingM_.GetMinLen() * HALF));
    }

    OP_LOGD("MatmulAllReduce",
            "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, "
            "totalMatmulTime %f, totalTP %f, minTileSize %lu, mAlignLen %lu, commTimeFactor_ %f",
            clusterInfo_.mValue, clusterInfo_.nValue, clusterInfo_.kValue, matmulPerf_.cubeUtil_, totalMatmulTime,
            totalTpTime, tilingM_.GetMinLen(), tilingM_.tileArgs.mAlignLen, commPerf_.commTimeFactor_);
}

void MMPlusAllReduce::SelectTilingMethod()
{
    tilingM_.cutRes.shortTileLen = tilingM_.GetMinLen();
    tilingM_.cutRes.numShortTile = 1;
    // do not cut if mValue is too small
    bool smallMPerRank = (clusterInfo_.mValue < tilingM_.cutRes.shortTileLen * 2);
    if (smallMPerRank) {
        tilingM_.NoCutTiling();
        return;
    }
    // 流水配平
    if (tilingM_.cutRes.shortTileAtBack) {
        ShortAtEndCalcBoundBalancing();
    } else {
        ShortAtFrontCommBoundBalancing();
    }
    // 生成切分
    bool smallDimAlignUp = (rankDim_ <= MIN_COMM_RANKDIM) && tilingM_.cutRes.shortTileAtBack &&
                           (clusterInfo_.nValue < SMALL_SHAPE_BAR ||
                            clusterInfo_.kValue < SMALL_SHAPE_BAR); // 2p，计算Bound，且N轴或者K轴很小
    tilingM_.GenerateInitialPartition(smallDimAlignUp);
    OP_LOGD("MatmulAllReduce", "Initial cut: longTileLen %lu, shortTileLen %lu", tilingM_.cutRes.longTileLen,
            tilingM_.cutRes.shortTileLen);

    // 根据首尾块大小、轮次等约束调整切分
    bool largeCalcCommRatio = ratioCalcComm_ > LARGE_BACKTILE_CALC_COMM_RATIO_BAR;
    bool soc310Flag = npuArch_ == Ops::Base::DAV_2002;
    bool kGreaterThanN = clusterInfo_.kValue > clusterInfo_.nValue;
    tilingM_.FitTileLengthContinuous(kGreaterThanN, largeCalcCommRatio, soc310Flag);
    // 310P首地址非对齐时通信很慢，所以把非对齐的短块放到最后
    bool commBoundNotAligned = ((npuArch_ == Ops::Base::DAV_2002) && !tilingM_.cutRes.shortTileAtBack &&
                                (tilingM_.cutRes.shortTileLen % tilingM_.tileArgs.mAlignLen != 0)) ||
                               isPerBlock_;
    if (commBoundNotAligned) {
        tilingM_.cutRes.shortTileAtBack = true; // 非对齐的尾块后置
    }
    OP_LOGD("MatmulAllReduce",
            "Final cut: shortTileAtBack %d, longTileLen %lu, "
            "numLongTile %lu, shortTileLen %lu, numShortTile %lu",
            tilingM_.cutRes.shortTileAtBack, tilingM_.cutRes.longTileLen, tilingM_.cutRes.numLongTile,
            tilingM_.cutRes.shortTileLen, tilingM_.cutRes.numShortTile);
}

void MMPlusQuantAllReduce::EstimateKernelTime()
{
    commPerf_.ChangeCommTimeFactorByDivision(QUANT_COMM_GROWTH_FACTOR_SOC910B); // comm quant
    commPerf_.ChangeCommTimeFactorByDivision(HALF);                             // all2all comm

    // find total matmul time and tp time
    double totalMatmulTime = matmulPerf_.MatmulTime(clusterInfo_.mValue, 1); // AllReduce先切totalTileCnt, 再切rankDim
    double totalA2ATime = commPerf_.CommTime(clusterInfo_.mValue);
    if (totalMatmulTime >= totalA2ATime * QUANT_CALC_BOUND_RATIO) {
        tilingM_.cutRes.shortTileAtBack = true;
    }

    // 计算略短于a2a时鼓励多切
    ratioCalcComm_ = (std::max(totalA2ATime, totalMatmulTime) / std::min(totalA2ATime, totalMatmulTime));
    bool isCalcCommBalance = ratioCalcComm_ < static_cast<double>(TWO);
    if (isCalcCommBalance && !tilingM_.cutRes.shortTileAtBack) {
        tilingM_.SetMinLenByMin(static_cast<uint64_t>(tilingM_.GetMinLen() * HALF));
    }

    OP_LOGD("MatmulQuantAllReduce",
            "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, "
            "totalMatmulTime %f, totalTP %f, minTileSize %lu, mAlignLen %lu, ratioCalcComm_ %f",
            clusterInfo_.mValue, clusterInfo_.nValue, clusterInfo_.kValue, matmulPerf_.cubeUtil_, totalMatmulTime,
            totalA2ATime, tilingM_.GetMinLen(), tilingM_.tileArgs.mAlignLen, ratioCalcComm_);
}

void MMPlusQuantAllReduce::SmallShortCheck(const uint64_t totalLen, uint64_t &longTileLen,
                                           const uint64_t &shortTileLen) const
{
    // 确保长块长度合法
    longTileLen = std::max(longTileLen, shortTileLen);
    longTileLen = std::min(longTileLen, totalLen - shortTileLen);
    // 检查长度
    uint64_t tmpNumLong = (totalLen - shortTileLen) / longTileLen;
    uint64_t remainLen = totalLen - (shortTileLen + longTileLen * tmpNumLong);
    if (remainLen > shortTileLen * SHORT_TILE_GROW_RATIO) { // 需要规避大短块场景
        if (remainLen * TWO > longTileLen) {                // 增加长块个数
            tmpNumLong++;
            longTileLen = (totalLen - shortTileLen) / tmpNumLong;
        } else { // 延长长块长度
            tmpNumLong = std::max(tmpNumLong, ONE);
            longTileLen += (remainLen / tmpNumLong);
        }
    }
    OP_LOGD("MatmulQuantAllReduce", "Small short check: remainLen %lu, longTileLen %lu, shortTileLen %lu", remainLen,
            longTileLen, shortTileLen);
}

void MMPlusQuantAllReduce::UniformCutSetShort(const uint64_t totalLen, const uint64_t minAlign,
                                              uint64_t &shortTileLen) const
{
    // 进入切分的判断条件是 totalLen >= 2 * minAlign = shortTileLen
    // 修改shortTileLen后依然要满足上述约束
    if (totalLen > minAlign * FOUR) {
        shortTileLen =
            static_cast<uint64_t>(static_cast<double>(minAlign) * static_cast<double>(FOUR) / static_cast<double>(TWO));
    } else if (totalLen > minAlign * THREE) {
        shortTileLen = static_cast<uint64_t>(static_cast<double>(minAlign) * static_cast<double>(THREE) /
                                             static_cast<double>(TWO));
    }
}

void MMPlusQuantAllReduce::SelectTilingScene(bool &longTileAlignUpFlag)
{
    bool moveUpFlag = clusterInfo_.mValue < SMALL_M;
    bool shortTileAtFrontBranch = !tilingM_.cutRes.shortTileAtBack || moveUpFlag;
    bool uniformCutBranch = ratioCalcComm_ <= static_cast<double>(TWO) || moveUpFlag;
    // 流水配平
    if (shortTileAtFrontBranch) { // mm < a2a = ag
        OP_LOGD("MatmulQuantAllReduce", "Scene mm < a2a = ag.");
        if (clusterInfo_.kValue <= ONE_KBYTE) {
            commPerf_.ChangeCommTimeFactorByDivision(QUANT_SMALL_K_COMM_GROWTH_FACTOR_SOC910B);
        }
        ShortAtFrontCommBoundBalancing();
        longTileAlignUpFlag = true;
        SmallShortCheck(clusterInfo_.mValue, tilingM_.cutRes.longTileLen, tilingM_.cutRes.shortTileLen);
    } else if (uniformCutBranch) { // a2a = ag < mm < (a2a + ag)
        UniformCutSetShort(clusterInfo_.mValue, tilingM_.GetMinLen(), tilingM_.cutRes.shortTileLen);
        OP_LOGD("MatmulQuantAllReduce", "Scene a2a = ag < mm < (a2a + ag)，short len %lu.",
                tilingM_.cutRes.shortTileLen);
        tilingM_.cutRes.longTileLen = tilingM_.cutRes.shortTileLen;
    } else { // (a2a + ag) < mm
        OP_LOGD("MatmulQuantAllReduce", "Scene (a2a + ag) < mm.");
        if (clusterInfo_.kValue >= FOUR_KILO) {
            commPerf_.ChangeCommTimeFactorByDivision(QUANT_LARGE_K_COMM_GROWTH_FACTOR_SOC910B);
        }
        ShortAtEndCalcBoundBalancing();
        longTileAlignUpFlag = true;
        SmallShortCheck(clusterInfo_.mValue, tilingM_.cutRes.longTileLen, tilingM_.cutRes.shortTileLen);
    }
}

void MMPlusQuantAllReduce::SelectTilingMethod()
{
    tilingM_.cutRes.shortTileLen = tilingM_.GetMinLen();
    tilingM_.cutRes.numShortTile = ONE;
    tilingM_.SetBackTileRatio(SMALL_BACKTILE_RATIO); // 短块尽量短，给aiv留时间
    // do not cut if mValue is too small
    bool smallMPerRank = (clusterInfo_.mValue < tilingM_.cutRes.shortTileLen * 2);
    if (smallMPerRank) {
        tilingM_.NoCutTiling();
        return;
    }
    bool longTileAlignUpFlag = (rankDim_ <= MIN_COMM_RANKDIM) && tilingM_.cutRes.shortTileAtBack &&
                               (clusterInfo_.nValue < SMALL_SHAPE_BAR ||
                                clusterInfo_.kValue < SMALL_SHAPE_BAR); // 2p，计算Bound，且N轴或者K轴很小
    SelectTilingScene(longTileAlignUpFlag);
    // 生成切分
    tilingM_.GenerateInitialPartition(longTileAlignUpFlag);
    OP_LOGD("MatmulQuantAllReduce", "Initial cut: longTileLen %lu, shortTileLen %lu", tilingM_.cutRes.longTileLen,
            tilingM_.cutRes.shortTileLen);

    // 根据首尾块大小、轮次等约束调整切分
    bool largeCalcCommRatio = ratioCalcComm_ > LARGE_BACKTILE_CALC_COMM_RATIO_BAR;
    bool kGreaterThanN = clusterInfo_.kValue > clusterInfo_.nValue;
    tilingM_.FitTileLengthContinuous(kGreaterThanN, largeCalcCommRatio);
    OP_LOGD("MatmulQuantAllReduce",
            "Final cut: shortTileAtBack %d, longTileLen %lu, "
            "numLongTile %lu, shortTileLen %lu, numShortTile %lu",
            tilingM_.cutRes.shortTileAtBack, tilingM_.cutRes.longTileLen, tilingM_.cutRes.numLongTile,
            tilingM_.cutRes.shortTileLen, tilingM_.cutRes.numShortTile);
}
