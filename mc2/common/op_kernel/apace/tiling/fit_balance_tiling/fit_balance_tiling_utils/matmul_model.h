/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// fit_balance_tiling 计算模型: cube 利用率/梯度/耗时正反解
#pragma once

#include "fit_balance_types.h"

namespace MatmulPerfConst {
constexpr double COMPUTES_PER_CYCLE = 4096;
constexpr double MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL = 1.4;
constexpr double MNVALUE_THRESHOLD = 820;
constexpr double KVALUE_THRESHOLD = 1280;
constexpr double LOCAL_MM_UTIL_EXPONENT_COEFFICIENT = 0.6; // 本地窗口(单份数据/流水未满)的 MN 指数
constexpr double CUBE_UTIL_EXPONENT_COEFFICIENT = 0.7;
constexpr double MAX_CUBE_UTIL = 0.95;
constexpr double AVERAGE_CUBE_UTIL = 0.75;
} // namespace MatmulPerfConst

// mm 模型参数: 平台规格 + 派生系数——块耗时公式与估时消费
struct MatmulModelParams {
    double computesPerCycle = 0; // cube 有效算力(MACs/cycle, 标定查表产出)
    double cyclePerMicroSec = 0; // AIC 频率(cycle/us, 标定注入)
    double matmulGradient = 0;   // 派生: 每行耗时系数(含利用率)
    uint64_t mmRankMul = 1;      // 派生: mm 总行数倍数(perRankM ? rankDim : 1)
    double localMMTime = 0;      // 派生: 本地块计算耗时(长块掩盖反解与 BACK 估时消费)
};

// cube 利用率: util = min(0.95, 0.75 × (K/1280)^0.7 × (调和平均/820)^mnExp)
//   调和平均 = (m × shares × n) / (m × shares + n)
//   mnExp 双值(经验参数): 流水线正常运转的主体计算取 0.7, 本地计算窗口(单份数据/流水未满)取
//   0.6(更小块利用率更低)
inline double FindCubeUtil(const MatmulInputParams& input, uint64_t mSize, uint64_t shares, double mnExponent)
{
    double mnharmonicMean =
        static_cast<double>(mSize * shares * input.nValue) / static_cast<double>(mSize * shares + input.nValue);
    double kExponentiated =
        std::min(MatmulPerfConst::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
                 std::pow(static_cast<double>(input.kValue), MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                     std::pow(MatmulPerfConst::KVALUE_THRESHOLD, MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT));
    double mnExponentiated =
        std::min(MatmulPerfConst::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
                 std::pow(mnharmonicMean, mnExponent) / std::pow(MatmulPerfConst::MNVALUE_THRESHOLD, mnExponent));
    double result = MatmulPerfConst::AVERAGE_CUBE_UTIL * kExponentiated * mnExponentiated;
    return std::min(MatmulPerfConst::MAX_CUBE_UTIL, result);
}

// 每行耗时系数: 利用率取 0.7 指数(份数 shares——FRONT 有一个 rank 的数据不经通信 → rankDim−1)
//   gradient = max(baseN,n) × max(baseK,k) / (computesPerCycle × 频率 × 核数 × util)
inline double CalcMatmulGradient(const MatmulInputParams& input, const MatmulModelParams& mmModel, uint64_t shares)
{
    const double cubeUtil = FindCubeUtil(input, input.mValue, shares, MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT);
    const uint64_t tmpN = std::max<uint64_t>(input.baseN, input.nValue);
    const uint64_t tmpK = std::max<uint64_t>(input.baseK, input.kValue);
    return (tmpN * tmpK) / (mmModel.computesPerCycle * mmModel.cyclePerMicroSec * input.coreNum * cubeUtil);
}
