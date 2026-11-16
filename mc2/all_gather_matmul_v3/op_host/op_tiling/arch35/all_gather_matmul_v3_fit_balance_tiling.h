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
 * \file all_gather_matmul_v3_fit_balance_tiling.h
 * \brief AllGatherMatmulV3 fit-balance 引擎接入——标定数据与编排（算子接入唯一编辑面）
 * 编排三步(SWAT probe → GetFitBalanceCutFromMmBase → 终版SWAT含减半)在算子侧组装
 */

#ifndef __ALL_GATHER_MATMUL_V3_FIT_BALANCE_TILING_H__
#define __ALL_GATHER_MATMUL_V3_FIT_BALANCE_TILING_H__

#pragma once

#include "apace/tiling/matmul_tiling/quant_matmul_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "apace/tiling/fit_balance_tiling/fit_balance_tiling.h"

namespace Mc2Tiling {

// AG 数据流: gather → matmul, 本地块在 HEAD 位(不经通信先算); mValue 为每卡 M
const FitBalanceCalibration::DataflowShape AG_DATAFLOWS[] = {
    {CommComputeOrder::COMM_FIRST, LocalMMPosition::FRONT, true},
};

// 4P aiv_urma 板测标定(斜率为收齐口径, 引擎按本卡 own 体积×4 折算); 2/8/16 例 miss 走 COF 兜底
const FitBalanceCalibration::CommFitRow AG_COMM_ROWS[] = {
    {TilingCommEngine::AIV_URMA, 4, 24.7, 5.1, 5.29},
};

const FitBalanceCalibration::MatmulFit AG_MM_FIT = {{256, 256, 128}};

// AG 完整 tiling 入口: SWAT probe → GetFitBalanceCutFromMmBase(切分+减半决策) → 终版SWAT(含减半) + 切分防护
using AgmmTilingOutput = FitBalanceTilingResult;

// SWAT matmul tiling 执行器: fp4/fp8 分型实例化; forceBaseMHalve 时跳过
// "块数不足双波"条件, 强制 baseM 减半一轮
inline void RunSwatMmTiling(fe::PlatFormInfos* platformInfo, uint32_t aicCoreNum, uint32_t aivCoreNum, bool isFp4,
                            bool hasBias, uint64_t n, uint64_t k, uint64_t mmM, QuantMatmulTilingData& mmTileOut,
                            bool forceBaseMHalve)
{
    if (isFp4) {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT4_E2M1, mm::DataType::DT_FLOAT4_E2M1, mm::BiasDataType::DT_FLOAT>
            tilingEngine;
        tilingEngine.SetPlatformInfoPtr(platformInfo);
        tilingEngine.SetCoreLimit(aicCoreNum, aivCoreNum);
        tilingEngine.SetOptimizeEnable(false);
        tilingEngine.SetMTailAlignEnable(true);
        tilingEngine.SetAdjustBasicBlockEnable(false);
        tilingEngine.SetBiasInfo(hasBias);
        if (forceBaseMHalve) {
            tilingEngine.EnableBaseMHalving(true);
            tilingEngine.SetForceBaseMHalving(true);
        }
        tilingEngine.GetTilingData(mmM, n, k, mmTileOut);
    } else {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN,
                              mm::BiasDataType::DT_FLOAT>
            tilingEngine;
        tilingEngine.SetPlatformInfoPtr(platformInfo);
        tilingEngine.SetCoreLimit(aicCoreNum, aivCoreNum);
        tilingEngine.SetOptimizeEnable(false);
        tilingEngine.SetMTailAlignEnable(true);
        tilingEngine.SetAdjustBasicBlockEnable(false);
        tilingEngine.SetBiasInfo(hasBias);
        if (forceBaseMHalve) {
            tilingEngine.EnableBaseMHalving(true);
            tilingEngine.SetForceBaseMHalving(true);
        }
        tilingEngine.GetTilingData(mmM, n, k, mmTileOut);
    }
}

inline AgmmTilingOutput GetAgmmTiling(fe::PlatFormInfos* platformInfo, uint32_t aicCoreNum, uint32_t aivCoreNum,
                                      bool isFp4, bool hasBias, uint64_t mPerRank, uint64_t nValue, uint64_t kValue,
                                      uint64_t rankSize, ge::DataType x1Dtype, uint64_t inputADtypeSize,
                                      uint64_t outputDtypeSize, const char* commMode)
{
    AgmmTilingOutput result{};
    const uint64_t totalLogicalM = rankSize * mPerRank;

    // SWAT probe(全量 M 基准): 产出真实底块与核数
    QuantMatmulTilingData swatProbe{};
    RunSwatMmTiling(platformInfo, aicCoreNum, aivCoreNum, isFp4, hasBias, nValue, kValue, totalLogicalM, swatProbe,
                    false);
    result.probeBaseM = swatProbe.baseM;
    result.probeBaseN = swatProbe.baseN;
    result.usedCoreNum = (swatProbe.usedCoreNum < static_cast<uint32_t>(rankSize)) ? static_cast<uint32_t>(rankSize) :
                                                                                     swatProbe.usedCoreNum;

    // fit-balance 切分 + 减半决策(probe 底块注入)
    const FitBalanceCalibration calib = MakeFitBalanceCalibration(AG_DATAFLOWS[0], AG_COMM_ROWS, AG_MM_FIT);
    const TilingCommEngine commEngine = ParseCommEngine(commMode);
    bool halveBaseM = false;
    const FitBalanceTilingArgs args =
        MakeFitBalanceEngineArgs(mPerRank, nValue, kValue, static_cast<uint32_t>(rankSize), result.usedCoreNum, x1Dtype,
                                 inputADtypeSize, outputDtypeSize, commEngine);
    result.fitBalanceHit =
        GetFitBalanceCutFromMmBase(args, calib, result.cut, &halveBaseM, result.probeBaseM, result.probeBaseN);

    // 终版 SWAT(对齐全量 M, 命中且减半时强制减半一轮)
    RunSwatMmTiling(platformInfo, aicCoreNum, aivCoreNum, isFp4, hasBias, nValue, kValue, totalLogicalM, result.mmTile,
                    result.fitBalanceHit && halveBaseM);
    result.halveBaseM = result.fitBalanceHit && halveBaseM;

    // 切分结果防护(幂等): 退化零块→单块; 未对齐单块→"对齐主块+未对齐尾"
    if (result.fitBalanceHit) {
        NormalizeDegenerateCut(result.cut, mPerRank);
    }
    return result;
}

} // namespace Mc2Tiling

#endif // __ALL_GATHER_MATMUL_V3_FIT_BALANCE_TILING_H__
