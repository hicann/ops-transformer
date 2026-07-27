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
 * \file matmul_allto_all_fit_balance_tiling.h
 * \brief arch35 架构的 AllToAll Fit Balance Tiling 实现
 */
#ifndef __MATMUL_ALLTO_ALL_FIT_BALANCE_TILING_H__
#define __MATMUL_ALLTO_ALL_FIT_BALANCE_TILING_H__

#pragma once
#include "op_host/op_tiling/mc2_fit_based_balance_tiling.h"
#include "../common/matmul_allto_all_util_tiling.h"
#include "../common/allto_all_formulaic_tiling.h"
#include "mc2_comm_utils.h"

namespace MC2Tiling {

// AICPU通信场景下，受限于通信展开开销，需要限制切分上限
constexpr uint64_t AICPU_MAX_TILE_CNT = 6U;

class MatmulAlltoAllFitBalanceTiling : public Mc2FitBasedBalanceTiling {
public:
    explicit MatmulAlltoAllFitBalanceTiling(const mc2tiling::TilingArgs &args, KernelType kernelType,
                                            TopoType topoType = TopoType::STANDARD_CARD,
                                            SocVersion socVersion = SocVersion::SOC950,
                                            QuantMode quantMode = QuantMode::NON_QUANT,
                                            uint8_t commMode = Mc2Comm::COMM_MODE_CCU)
        : Mc2FitBasedBalanceTiling(args, kernelType, topoType, socVersion), quantMode_(quantMode)
    {
        commPerf_.SetCommShapeLen(args.nValue);
        commPerf_.SetCommDTypeSize(mmInfo_.outMatrixCDtypeSize);
        tilingM_.SetMinLenByMax(matmulPerf_.GetBaseM());
        tilingM_.SetAlignLength(matmulPerf_.GetBaseM());
        isQuantMatmul_ = (mmInfo_.inMatrixADtypeSize == 1) && (mmInfo_.inMatrixBDtypeSize == 1);
        // AICPU通信场景下，受限于通信资源，最大切分轮次限制为6
        if (commMode == Mc2Comm::COMM_MODE_AICPU) {
            tilingM_.SetMaxTileCnt(AICPU_MAX_TILE_CNT);
        }
    }

    void EstimateMMCommTime() override;
    void SetShortTileLen() override;
    void SetLongTileLen() override;
    void AdjustLongShortTileLen() override;

private:
    void AdjustLongShortTileLenWhenCalcBound();

    bool isLargerThanL2Cache_ = false;
    bool isQuantMatmul_ = false;
    QuantMode quantMode_;
};

inline CutResult GetArch35TilingResult(const mc2tiling::TilingArgs &args, KernelType kernelType, SocVersion socVersion,
                                       NpuArch npuArch, QuantMode quantMode, uint8_t commMode = Mc2Comm::COMM_MODE_CCU)
{
    // 标卡4P 与 8P 均走 Fit Balance tiling，仅拓扑类型不同；统一判定并构造，消除重复逻辑
    const bool isStandardCard4P = mc2tiling::IsStandardCard4P(args.rankDim, npuArch);
    const bool is8P = mc2tiling::Is8P(args.rankDim, npuArch);
    if (isStandardCard4P || is8P) {
        const TopoType topoType = is8P ? TopoType::EIGHT_P : TopoType::STANDARD_CARD;
        OP_LOGD("Arch35TilingResult", "Using fit balance tiling for arch35, topoType=%d", static_cast<int>(topoType));
        MatmulAlltoAllFitBalanceTiling fitBalanceTiling(args, kernelType, topoType, socVersion, quantMode, commMode);
        return fitBalanceTiling.GetTiling();
    }

    OP_LOGD("Arch35TilingResult", "Falling back to formulaic tiling");
    AlltoAllMM formulaicTiling(args, args.rankDim, kernelType, socVersion);
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        OP_LOGD("Arch35TilingResult", "AICPU mode, limit maxTileCnt to %lu", AICPU_MAX_TILE_CNT);
        formulaicTiling.tilingM_.SetMaxTileCnt(AICPU_MAX_TILE_CNT);
    }
    formulaicTiling.GetTiling();
    return formulaicTiling.tilingM_.cutRes;
}
} // namespace MC2Tiling

#endif // __MATMUL_ALLTO_ALL_FIT_BALANCE_TILING_H__
