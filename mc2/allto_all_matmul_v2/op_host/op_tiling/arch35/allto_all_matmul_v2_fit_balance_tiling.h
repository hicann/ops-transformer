/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_all_matmul_v2_fit_balance_tiling.h
 * \brief AlltoAllMatmulV2 fit-balance 引擎接入——标定数据与分派（算子接入唯一编辑面）
 */

#ifndef __ALLTO_ALL_MATMUL_V2_FIT_BALANCE_TILING_H__
#define __ALLTO_ALL_MATMUL_V2_FIT_BALANCE_TILING_H__

#pragma once

#include "apace/tiling/fit_balance_tiling/fit_balance_tiling.h"

namespace MC2Tiling {

// 实测发现 LocalMMPosition::None 场景的模型能更精确的模拟 precisionMode = 1 分支
const FitBalanceCalibration::DataflowShape A2A_DATAFLOWS[] = {
    {CommComputeOrder::COMM_FIRST, LocalMMPosition::NONE, false},  // precisionMode = 0 / 1
    {CommComputeOrder::COMM_FIRST, LocalMMPosition::FRONT, false}, // precisionMode = 2
};

const FitBalanceCalibration::CommFitRow A2A_COMM_ROWS[] = {
    {TilingCommEngine::AIV_URMA, 4, 6.18, 8.0, 5.29},
};

const FitBalanceCalibration::MatmulFit A2A_MM_FIT = {{256, 256, 128}};

// a2a 完整 tiling 入口 = 公共编排(COF probe → 切分+减半决策 → 减半重算)
// est 的 K 口径为全量(ka×rank), COF runner 的 K 口径为本卡 ka
inline FitBalanceTilingResult GetA2aMmTiling(fe::PlatFormInfos* platformInfo, bool isFp4, bool hasBias,
                                             uint64_t mPerRank, uint64_t nValue, uint64_t ka, uint64_t rankDim,
                                             uint32_t precisionMode, ge::DataType x1Dtype, uint64_t inputADtypeSize,
                                             uint64_t outputDtypeSize, const char* commMode,
                                             QuantMatmulTilingData& mmTile, CommTilingData& commT,
                                             CommTilingData& scaleT)
{
    FitBalanceTilingResult result{};

    // COF 三件套 probe: mm tiling + 标定 miss 时 comm/scale 即兜底通信切分
    GetCoreOccupancyFirstTiling(platformInfo, mPerRank, nValue, ka, isFp4, hasBias, mmTile, commT, scaleT);
    const uint32_t probeUsedCoreNum = mmTile.usedCoreNum;
    result.probeBaseM = mmTile.baseM;
    result.probeBaseN = mmTile.baseN;
    result.usedCoreNum = probeUsedCoreNum;

    // fit-balance 切分 + 减半决策(probe 底块注入)。
    // a2a 双 K 口径: COF runner 用本卡 ka, est 用全量 ka*rankDim(通信在 K 轴)
    const int dataFlowMode = (precisionMode == 2U) ? 1 : 0;
    const FitBalanceCalibration calib =
        MakeFitBalanceCalibration(A2A_DATAFLOWS[dataFlowMode], A2A_COMM_ROWS, A2A_MM_FIT);
    const TilingCommEngine commEngine = ParseCommEngine(commMode);
    bool halveBaseM = false;
    const FitBalanceTilingArgs args =
        MakeFitBalanceEngineArgs(mPerRank, nValue, ka * rankDim, static_cast<uint32_t>(rankDim), probeUsedCoreNum,
                                 x1Dtype, inputADtypeSize, outputDtypeSize, commEngine);
    result.fitBalanceHit =
        GetFitBalanceCutFromMmBase(args, calib, result.cut, &halveBaseM, result.probeBaseM, result.probeBaseN);

    // 减半重算: COF runner(本卡 ka 口径), comm/scale 写临时变量丢弃
    // (fit-balance 切分结果不被覆盖); 核数保持 probe 值防 blockDim 漂移
    if (result.fitBalanceHit && halveBaseM) {
        CommTilingData halveDataTiling{};
        CommTilingData halveScaleTiling{};
        GetCoreOccupancyFirstTiling(platformInfo, mPerRank, nValue, ka, isFp4, hasBias, mmTile, halveDataTiling,
                                    halveScaleTiling, true);
        mmTile.usedCoreNum = probeUsedCoreNum;
    }
    result.halveBaseM = result.fitBalanceHit && halveBaseM;

    return result;
}

} // namespace MC2Tiling

#endif // __ALLTO_ALL_MATMUL_V2_FIT_BALANCE_TILING_H__
