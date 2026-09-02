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
 * \file allto_all_matmul_fit_balance_tiling.h
 * \brief Arch35 formulaic tiling for AlltoAllMatmul based on Mc2FitBasedBalanceTiling
 */
#ifndef ALLTO_ALL_MATMUL_FIT_BALANCE_TILING_H
#define ALLTO_ALL_MATMUL_FIT_BALANCE_TILING_H

#include "mc2/common/op_host/op_tiling/mc2_fit_based_balance_tiling.h"
#include "mc2/common/op_host/op_tiling/mc2_tiling_utils.h"

namespace MC2Tiling {
class AlltoAllMatmulFitBalanceTiling : public Mc2FitBasedBalanceTiling {
public:
    enum class QuantType {
        FP_QUANT,
        KC_QUANT,
        MXFP8_QUANT,
        MXFP4_QUANT
    };
    explicit AlltoAllMatmulFitBalanceTiling(QuantType matmulQuantType, const mc2tiling::TilingArgs &args,
                                            TopoType topoType = TopoType::STANDARD_CARD,
                                            SocVersion socVersion = SocVersion::SOC950,
                                            uint8_t commMode = mc2tiling::A5_AICPU_TS_ENGINE)
        : Mc2FitBasedBalanceTiling(args, KernelType::ALL_TO_ALL, topoType, socVersion),
          matmulQuantType_(matmulQuantType),
          commMode_(commMode)
    {
        commPerf_.SetCommShapeLen(mmInfo_.kValue);
        commPerf_.SetCommDTypeSize(mmInfo_.inMatrixADtypeSize);
        tilingM_.SetMinLenByMax(matmulPerf_.GetBaseM());
        tilingM_.SetAlignLength(matmulPerf_.GetBaseM());
    }

    void EstimateMMCommTime() override;
    void SetShortTileLen() override;
    void SetLongTileLen() override;
    void AdjustLongShortTileLen() override;

protected:
    QuantType matmulQuantType_ = QuantType::FP_QUANT;
    double CalcMatmulTime(uint64_t tileM);
    double CalcPermuteTime(uint64_t tileM);
    uint64_t CalcMByTime(double time);
    uint64_t CalcMWhenT1EqualT2();
    void FitTileLengthDiscrete();
    uint64_t primeMinFactor(uint64_t num);
    void ReTilingByFactor();
    uint64_t CalcLongTileLen(uint64_t shortTileLen);
    uint32_t GetRank();
    double totalMatmulTime_ = 0.0;
    double totalTpTime_ = 0.0;
    uint64_t alltoallMaxTilingNum_ = MAX_TILE_CNT;
    uint8_t commMode_ = mc2tiling::A5_AICPU_TS_ENGINE;
};
} // namespace MC2Tiling
#endif // ALLTO_ALL_FIT_BALANCE_TILING_H
