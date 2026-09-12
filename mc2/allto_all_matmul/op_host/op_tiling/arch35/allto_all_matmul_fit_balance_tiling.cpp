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
 * \file allto_all_matmul_fit_balance_tiling.cpp
 * \brief Arch35 formulaic tiling implementation for AlltoAllMatmul
 */
#include "./allto_all_matmul_fit_balance_tiling.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace MC2Tiling {
using QuantType = MC2Tiling::AlltoAllMatmulFitBalanceTiling::QuantType;

// 下列系数用于根据数据量计算时间，计算公式为y(us) = kx(MB) + b
const static std::map<QuantType, double> matmulKMap = {{QuantType::FP_QUANT, 0.1792226},
                                                       {QuantType::KC_QUANT, 0.0993},
                                                       {QuantType::MXFP8_QUANT, 0.0894435},
                                                       {QuantType::MXFP4_QUANT, 0.046137}};
const static std::map<QuantType, double> matmulBMap = {{QuantType::FP_QUANT, 15.095},
                                                       {QuantType::KC_QUANT, 18.71},
                                                       {QuantType::MXFP8_QUANT, 15.14},
                                                       {QuantType::MXFP4_QUANT, 10.97}};
const static std::map<QuantType, double> all2allDtypeSizeMap = {{QuantType::FP_QUANT, 2.0},
                                                                {QuantType::KC_QUANT, 2.0},
                                                                {QuantType::MXFP8_QUANT, 1.0},
                                                                {QuantType::MXFP4_QUANT, 0.5}};
const static std::map<uint32_t, double> all2allKMap = {{4, 5.8416}, {8, 2.5386}};
const static std::map<uint32_t, double> all2allBMap = {{4, 5.8148}, {8, 7.6265}};
// 拟合动态pertoken和permute性能
const static std::map<uint32_t, double> permuteKMap = {{4, 26.75}, {8, 26.75}};
const static std::map<uint32_t, double> permuteBMap = {{4, 9.85}, {8, 9.85}};
const static std::map<uint32_t, double> dynamicPertokenKMap = {{4, 88.6523}, {8, 81.9156}};
const static std::map<uint32_t, double> dynamicPertokenBMap = {{4, 9.6494}, {8, 9.9191}};
constexpr uint64_t ALIGN_M = 256;
constexpr uint64_t AICPU_TILING_COST = 100;
constexpr uint64_t CCU_TILING_COST = 10;
void AlltoAllMatmulFitBalanceTiling::EstimateMMCommTime()
{
    totalMatmulTime_ = CalcMatmulTime(mmInfo_.mValue);
    totalTpTime_ = commPerf_.CommTime(mmInfo_.mValue);
    ratioCalcComm_ = (std::max(totalTpTime_, totalMatmulTime_) / std::min(totalTpTime_, totalMatmulTime_));

    if (totalTpTime_ > totalMatmulTime_) {
        tilingM_.cutRes.shortTileAtBack = true;
    } else {
        tilingM_.cutRes.shortTileAtBack = false;
    }

    // aicpu所需时间相比ccu的比率
    const double aicpuTimeRate = 1.5;
    // 计算最大切分轮次
    double tempMaxTilingNum = commMode_ == mc2tiling::A5_AICPU_TS_ENGINE ?
                                  (totalTpTime_ * aicpuTimeRate / static_cast<double>(AICPU_TILING_COST)) :
                                  (totalTpTime_ / static_cast<double>(CCU_TILING_COST));
    // 最大切分轮次不少于三次
    const double maxTiling = 3.0;
    tempMaxTilingNum = tempMaxTilingNum < maxTiling ? maxTiling : tempMaxTilingNum;
    alltoallMaxTilingNum_ = static_cast<uint64_t>(tempMaxTilingNum) < alltoallMaxTilingNum_ ?
                                static_cast<uint64_t>(tempMaxTilingNum) :
                                alltoallMaxTilingNum_;
    tilingM_.SetMaxTileCnt(alltoallMaxTilingNum_);
}

void AlltoAllMatmulFitBalanceTiling::SetShortTileLen()
{
    bool isCalcCommBalance = ratioCalcComm_ < CALC_COMM_RATIO;
    // M保持256对齐
    uint64_t cutLen = ALIGN_M;
    tilingM_.SetAlignLength(cutLen);
    tilingM_.SetMinLenByMax(cutLen);
    tilingM_.SetMinLenByMin(mmInfo_.mValue);

    if (isCalcCommBalance) {
        // 通算时间比较平衡时,尝试计算交点
        cutLen = CalcMWhenT1EqualT2() / tilingM_.tileArgs.minTileLen * tilingM_.tileArgs.minTileLen;
        if (cutLen > 0 && tilingM_.cutRes.shortTileAtBack == true) {
            // 总通信时间长，短块在后，短块的通信时长需掩盖长块计算时长
            cutLen += tilingM_.tileArgs.minTileLen;
            tilingM_.SetMinLenByMax(cutLen);
        } else if (cutLen > 0 && tilingM_.cutRes.shortTileAtBack == false) {
            // 总计算时间长，短块在前，通信没有数据依赖短块越短越好
            cutLen = tilingM_.tileArgs.minTileLen;
            tilingM_.SetMinLenByMin(cutLen);
        }
    }

    tilingM_.cutRes.shortTileLen = tilingM_.GetMinLen();
    tilingM_.cutRes.numShortTile = 1U;
}

// 根据短块时间计算长块大小
uint64_t AlltoAllMatmulFitBalanceTiling::CalcLongTileLen(uint64_t shortTileLen)
{
    uint64_t longTileLen;
    if (tilingM_.cutRes.shortTileAtBack) {
        double targetTime = commPerf_.CommTime(shortTileLen);
        longTileLen = CalcMByTime(targetTime);
    } else {
        double singleRankMatmulTime = CalcMatmulTime(shortTileLen);
        // 非动态KC量化短块在前时，短块的permute+matmul需要掩盖长块的all2all+permute
        if (matmulQuantType_ != QuantType::KC_QUANT) {
            double singleRankPermuteTime = CalcPermuteTime(shortTileLen);
            uint32_t rank = GetRank();
            longTileLen =
                mmInfo_.kValue == 0 ?
                    0U :
                    (singleRankPermuteTime + singleRankMatmulTime - all2allBMap.at(rank) - permuteBMap.at(rank)) *
                        ONE_MBYTE /
                        ((all2allKMap.at(rank) + permuteKMap.at(rank)) * mmInfo_.kValue *
                         all2allDtypeSizeMap.at(matmulQuantType_));
        } else {
            longTileLen = commPerf_.InverseCommTime(singleRankMatmulTime);
        }
    }
    longTileLen = longTileLen > ALIGN_M ? longTileLen : static_cast<uint64_t>(ALIGN_M);
    return longTileLen / ALIGN_M * ALIGN_M;
}

void AlltoAllMatmulFitBalanceTiling::SetLongTileLen()
{
    tilingM_.cutRes.longTileLen = CalcLongTileLen(tilingM_.cutRes.shortTileLen);
}

void AlltoAllMatmulFitBalanceTiling::AdjustLongShortTileLen()
{
    FitTileLengthDiscrete();
    if (tilingM_.cutRes.shortTileLen > tilingM_.cutRes.longTileLen) {
        tilingM_.cutRes.shortTileLen = tilingM_.cutRes.shortTileLen - tilingM_.cutRes.longTileLen;
        tilingM_.cutRes.numLongTile++;
    }
    // When the long and short tiles are equal, the long and short pieces become one.
    if (tilingM_.cutRes.shortTileLen == tilingM_.cutRes.longTileLen) {
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
        tilingM_.cutRes.numLongTile++;
    }
}

// 基于拟合公式计算matmul时间
double MC2Tiling::AlltoAllMatmulFitBalanceTiling::CalcMatmulTime(uint64_t tileM)
{
    double totalTime = 0.0;
    if (matmulQuantType_ == QuantType::KC_QUANT) {
        totalTime += dynamicPertokenKMap.at(GetRank()) * static_cast<double>(tileM) *
                         static_cast<double>(mmInfo_.kValue) / (coreNum_ * TWO * ONE_MBYTE) +
                     dynamicPertokenBMap.at(GetRank());
    }
    totalTime += matmulKMap.at(matmulQuantType_) * static_cast<double>(tileM) * static_cast<double>(mmInfo_.kValue) *
                     static_cast<double>(mmInfo_.nValue) / (coreNum_ * ONE_MBYTE) +
                 matmulBMap.at(matmulQuantType_);
    return totalTime;
}

// 基于拟合公式计算permute时间
double MC2Tiling::AlltoAllMatmulFitBalanceTiling::CalcPermuteTime(uint64_t tileM)
{
    return permuteKMap.at(GetRank()) * static_cast<double>(tileM) * static_cast<double>(mmInfo_.kValue) *
               all2allDtypeSizeMap.at(matmulQuantType_) / (coreNum_ * TWO * ONE_MBYTE) +
           permuteBMap.at(GetRank());
}

uint64_t AlltoAllMatmulFitBalanceTiling::CalcMByTime(double time)
{
    if (mmInfo_.kValue == 0 || mmInfo_.nValue == 0) {
        return static_cast<uint64_t>(0);
    }
    double matmulK = matmulKMap.at(matmulQuantType_);
    double matmulB = matmulBMap.at(matmulQuantType_);
    double mValue = (time - matmulB) * static_cast<double>(coreNum_) * static_cast<double>(ONE_MBYTE) /
                    (matmulK * static_cast<double>(mmInfo_.kValue) * static_cast<double>(mmInfo_.nValue));
    return mValue > 0 ? static_cast<uint64_t>(mValue) : static_cast<uint64_t>(0);
}

uint64_t AlltoAllMatmulFitBalanceTiling::CalcMWhenT1EqualT2()
{
    double matmulK = matmulKMap.at(matmulQuantType_);
    double matmulB = matmulBMap.at(matmulQuantType_);
    double all2allK = all2allKMap.at(GetRank());
    double all2allB = all2allBMap.at(GetRank());
    double dtypeSize = all2allDtypeSizeMap.at(matmulQuantType_);
    double denominator = all2allK * static_cast<double>(coreNum_) * dtypeSize - matmulK * mmInfo_.nValue;
    const double denominatorTolerance = 1e-9;
    if (std::fabs(denominator) <= denominatorTolerance || mmInfo_.kValue == 0) {
        return static_cast<uint64_t>(0);
    }
    double mValue = (matmulB - all2allB) * static_cast<double>(ONE_MBYTE) * static_cast<double>(coreNum_) /
                    static_cast<double>(mmInfo_.kValue) / denominator;
    return mValue > 0 ? static_cast<uint64_t>(mValue) : static_cast<uint64_t>(0);
}

uint64_t AlltoAllMatmulFitBalanceTiling::primeMinFactor(uint64_t num) const
{
    const uint64_t FACTOR_TWO = 2;
    const uint64_t FACTOR_THREE = 3;
    if (num % FACTOR_TWO == 0) {
        return FACTOR_TWO;
    }
    for (uint64_t factor = FACTOR_THREE; factor * factor <= num; factor += FACTOR_TWO) {
        if (num % factor == 0) {
            return factor;
        }
    }
    return num;
}

void AlltoAllMatmulFitBalanceTiling::ReTilingByFactor()
{
    uint64_t alignNum = mmInfo_.mValue / ALIGN_M;
    uint64_t alignRemainder = mmInfo_.mValue % ALIGN_M;
    uint64_t maxLongNum;
    uint64_t minFactor;
    for (uint64_t tempAllNumOfLongTile = alignNum - 1; (tempAllNumOfLongTile + tempAllNumOfLongTile) > alignNum;
         tempAllNumOfLongTile--) {
        // 切分轮次上限太小
        if (alltoallMaxTilingNum_ <= tilingM_.cutRes.numShortTile) {
            break;
        }
        // 计算最小因数
        minFactor = primeMinFactor(tempAllNumOfLongTile);
        alignRemainder += ALIGN_M;
        maxLongNum = (CalcLongTileLen(alignRemainder) + ALIGN_M - 1) / ALIGN_M;
        if (maxLongNum < minFactor) {
            // 对齐的长块上限小于最小因数，无法整除
            continue;
        }
        // 长块尽量切大减少轮次带来的膨胀
        for (; maxLongNum > alignRemainder / ALIGN_M; maxLongNum--) {
            // 切分轮次限制
            if (tempAllNumOfLongTile / maxLongNum > alltoallMaxTilingNum_ - tilingM_.cutRes.numShortTile) {
                continue;
            }
            if (tempAllNumOfLongTile % maxLongNum == 0) {
                tilingM_.cutRes.longTileLen = maxLongNum * ALIGN_M;
                tilingM_.cutRes.numLongTile = tempAllNumOfLongTile / maxLongNum;
                tilingM_.cutRes.shortTileLen = alignRemainder;
                return;
            }
        }
    }
    // 只有短块不对齐的切分方案不存在
    bool isAlignUp = ratioCalcComm_ > CALC_COMM_RATIO;
    tilingM_.FitTileLengthDiscrete(isAlignUp, true, false);
}

void AlltoAllMatmulFitBalanceTiling::FitTileLengthDiscrete()
{
    // 最小的需要切分的耗时
    const uint64_t minCutTime = 100;
    if (mmInfo_.mValue < (tilingM_.cutRes.shortTileLen + tilingM_.cutRes.shortTileLen) ||
        (totalMatmulTime_ + totalTpTime_) < minCutTime) {
        // 总长度小于两倍短块或者总时间少于100us，不切
        tilingM_.cutRes.longTileLen = mmInfo_.mValue;
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numLongTile = 1U;
        tilingM_.cutRes.numShortTile = 0U;
    } else if (mmInfo_.mValue <= tilingM_.cutRes.shortTileLen + tilingM_.cutRes.longTileLen) {
        // 总长度大于等于两倍短块 且 小于等于短块+长块 ，切一短一长
        tilingM_.cutRes.longTileLen = mmInfo_.mValue - tilingM_.cutRes.shortTileLen;
        tilingM_.cutRes.numLongTile = 1U;
    } else {
        uint64_t tempLongNum = (mmInfo_.mValue - tilingM_.cutRes.shortTileLen) / tilingM_.cutRes.longTileLen;
        uint64_t remainder = (mmInfo_.mValue - tilingM_.cutRes.shortTileLen) % tilingM_.cutRes.longTileLen;
        if (remainder == 0) {
            // 切分完没有余数
            tilingM_.cutRes.numLongTile = tempLongNum;
        } else if (tilingM_.cutRes.shortTileLen != 0 && remainder % tilingM_.cutRes.shortTileLen == 0) {
            // 切分完余数为短块的倍数
            tilingM_.cutRes.numLongTile = tempLongNum;
            tilingM_.cutRes.numShortTile += remainder / tilingM_.cutRes.shortTileLen;
        } else {
            // 不规整的余数，重新切分
            ReTilingByFactor();
        }
    }
    if (tilingM_.cutRes.numLongTile + tilingM_.cutRes.numShortTile > alltoallMaxTilingNum_) {
        // 切分轮次过多，重新切分
        ReTilingByFactor();
    }
}

uint32_t AlltoAllMatmulFitBalanceTiling::GetRank() const
{
    const uint32_t STANDARD_CARD = 4;
    const uint32_t EIGHT_P = 8;
    return ((rankDim_ == STANDARD_CARD) || (rankDim_ == EIGHT_P)) ? static_cast<uint32_t>(rankDim_) : STANDARD_CARD;
}
} // namespace MC2Tiling
