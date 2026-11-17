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
 * \file reduce_scatter_fit_balance_tiling.h
 * \brief
 */
#ifndef __REDUCE_SCATTER_FIT_BALANCE_TILING_H__
#define __REDUCE_SCATTER_FIT_BALANCE_TILING_H__

#pragma once
#include "op_host/op_tiling/mc2_fit_based_balance_tiling.h"

constexpr double RS_FP4_COMPUTE_PER_CYCLE_RATIO = 1; // 调整mxfp4的computesPerCycle和mxfp8一致

template <typename MatmulPerf = MatmulPerformanceArch35, typename CommPerf = HCCLPerformanceArch35>
class MMReduceScatterFitBalanceTiling : public Mc2FitBasedBalanceTilingBase<MatmulPerf, CommPerf> {
public:
    using Base = Mc2FitBasedBalanceTilingBase<MatmulPerf, CommPerf>;
    using Base::matmulPerf_;
    using Base::commPerf_;
    using Base::tilingM_;
    using Base::mmInfo_;
    using Base::rankDim_;
    using Base::rankTileNum_;
    using Base::coreNum_;
    using Base::ratioCalcComm_;

    explicit MMReduceScatterFitBalanceTiling(
        const mc2tiling::TilingArgs& args, KernelType kernelType, TopoType topoType = TopoType::STANDARD_CARD,
        bool isAicpuComm = false) // 是否AICPU通信模式，AICPU时触发M轴切分封顶至AICPU_M_TILE_CAP
        : Base(args, kernelType, topoType),
          isAicpuComm_(isAicpuComm)
    {
        commPerf_.SetCommShapeLen(args.nValue);
        commPerf_.SetCommDTypeSize(mmInfo_.outMatrixCDtypeSize);
        if (args.geAType == ge::DataType::DT_FLOAT4_E2M1 && args.geBType == ge::DataType::DT_FLOAT4_E2M1) {
            matmulPerf_.mmShapeInfo_.inMatrixADtypeSize = 1;
            matmulPerf_.mmShapeInfo_.inMatrixBDtypeSize = 1;
            mmInfo_.inMatrixADtypeSize = 1;
            mmInfo_.inMatrixBDtypeSize = 1;
            matmulPerf_.calcType_ = MatmulCalcType::QUANT;
            matmulPerf_.mmShapeInfo_.computesPerCycle =
                RS_FP4_COMPUTE_PER_CYCLE_RATIO * MatmulPerformance::COMPUTES_PER_CYCLE;
            mmInfo_.computesPerCycle = matmulPerf_.mmShapeInfo_.computesPerCycle;
        }
        tilingM_.SetMinLenByMax(matmulPerf_.GetBaseM());
        tilingM_.SetAlignLength(matmulPerf_.GetBaseM());
        isQuantMatmul_ = (mmInfo_.inMatrixADtypeSize == 1) && (mmInfo_.inMatrixBDtypeSize == 1);
    }

    // 复用已按平台和通信场景配置的模型，保留频率、量化算力及标定参数。
    explicit MMReduceScatterFitBalanceTiling(const mc2tiling::TilingArgs& args, const MatmulPerf& matmulPerf,
                                             const CommPerf& commPerf, bool isAicpuComm)
        : Base(args, matmulPerf, commPerf),
          isAicpuComm_(isAicpuComm)
    {
        tilingM_.SetMinLenByMax(matmulPerf_.GetBaseM());
        tilingM_.SetAlignLength(matmulPerf_.GetBaseM());
        isQuantMatmul_ = matmulPerf_.calcType_ == MatmulCalcType::QUANT;
    }

    void EstimateMMCommTime() override;
    void SetShortTileLen() override;
    void SetLongTileLen() override;
    void AdjustLongShortTileLen() override;
    void SetApaceTileAlignment(bool enabled);

private:
    void FitApaceTileLengthDiscrete(bool smallDimAlignUp, bool goodLinearityShape);
    bool isLargerThanL2Cache_ = false;
    bool isQuantMatmul_ = false;
    bool isAicpuComm_ = false;
    bool keepTileAligned_ = false;
};

using MMReduceScatterFitBalanceTiling960 =
    MMReduceScatterFitBalanceTiling<MatmulPerformance960dt, HCCLPerformance960dt>;

#endif // __REDUCE_SCATTER_FIT_BALANCE_TILING_H__
