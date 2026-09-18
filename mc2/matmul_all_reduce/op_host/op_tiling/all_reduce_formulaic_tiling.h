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
 * \file all_reduce_formulaic_tiling.h
 * \brief
 */
#ifndef __ALL_REDUCE_FORMULAIC_TILING_H__
#define __ALL_REDUCE_FORMULAIC_TILING_H__

#pragma once
#include "op_host/op_tiling/hccl_formulaic_tiling.h"
constexpr double QUANT_COMM_GROWTH_FACTOR_SOC310P = 0.8;
constexpr double QUANT_COMM_GROWTH_FACTOR_SOC910B = 0.6;
constexpr double QUANT_CALC_BOUND_RATIO = 0.7;
constexpr double QUANT_LARGE_K_COMM_GROWTH_FACTOR_SOC910B = 2.0;
constexpr double QUANT_SMALL_K_COMM_GROWTH_FACTOR_SOC910B = 1.4;
constexpr double HALF = 0.5;
constexpr uint64_t LARGE_NK_BAR = 32;
constexpr double LARGE_BACKTILE_CALC_COMM_RATIO_BAR = 1.75;
constexpr double SMALL_BACKTILE_RATIO = 0.25;
constexpr double SHORT_TILE_GROW_RATIO = 1.25;

class MMPlusAllReduce : public OneCalcOneCommBase {
public:
    // Constructor
    explicit MMPlusAllReduce(const mc2tiling::TilingArgs &args, uint32_t inputRankDim, KernelType inputKernelType,
                             NpuArch npuArch = Ops::Base::DAV_2201, bool isPerBlock = false,
                             SocVersion_2201 socVersion_2201 = SocVersion_2201::SOC910_B)
        : OneCalcOneCommBase(args, inputRankDim, inputKernelType, npuArch, socVersion_2201)
    {
        isPerBlock_ = isPerBlock;
        inferFlag_ = true;
        commPerf_.SetCommShapeLen(args.nValue);
        commPerf_.SetCommDTypeSize(clusterInfo_.outMatrixCDtypeSize);
        rankTileNum_ = commPerf_.GetRankTileNum();

        tilingM_.SetMinLenByMax(commPerf_.GetLinearThresholdLen());
        tilingM_.SetMinLenByMax(matmulPerf_.GetLinearThresholdLen(rankTileNum_));
    }

    void EstimateKernelTime() override;
    void SelectTilingMethod() override;

    bool isPerBlock_{false};

private:
    void SetCommTimeFactorForA5();
    void SetCommTimeFactorForOther();
    void SetCommTimeFactor();
};

class MMPlusQuantAllReduce : public OneCalcOneCommBase {
public:
    // Constructor
    explicit MMPlusQuantAllReduce(const mc2tiling::TilingArgs &args, uint32_t inputRankDim, KernelType inputKernelType,
                                  NpuArch npuArch = Ops::Base::DAV_2201,
                                  SocVersion_2201 socVersion_2201 = SocVersion_2201::SOC910_B)
        : OneCalcOneCommBase(args, inputRankDim, inputKernelType, npuArch, socVersion_2201)
    {
        inferFlag_ = true;
        commPerf_.SetCommShapeLen(args.nValue);
        commPerf_.SetCommDTypeSize(clusterInfo_.outMatrixCDtypeSize);
        rankTileNum_ = commPerf_.GetRankTileNum();

        tilingM_.SetMinLenByMax(commPerf_.GetLinearThresholdLen());
        tilingM_.SetMinLenByMax(matmulPerf_.GetLinearThresholdLen(rankTileNum_));
    }

    void EstimateKernelTime() override;
    void SmallShortCheck(const uint64_t totalLen, uint64_t &longTileLen, const uint64_t &shortTileLen) const;
    void UniformCutSetShort(const uint64_t totalLen, const uint64_t minAlign, uint64_t &shortTileLen) const;
    void SelectTilingMethod() override;
    void SelectTilingScene(bool &longTileAlignUpFlag);
};
#endif //__ALL_REDUCE_FORMULAIC_TILING_H__
