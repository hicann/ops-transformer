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
 * \file allto_all_formulaic_tiling.h
 * \brief
 */
#ifndef ALLTO_ALL_FORMULAIC_TILING_H
#define ALLTO_ALL_FORMULAIC_TILING_H

#include "op_host/op_tiling/hccl_formulaic_tiling.h"

constexpr uint64_t LARGE_K_BOUNDARY = 8192;
constexpr double COMPUTE_TIME_SCALE_FACTOR = 1.5;
static constexpr double COMM_TIME_SCALE_FACTOR = 1.35; // A3设置通信膨胀系数
static constexpr uint64_t MAX_TILE_CNT_A3 = 8;         // A3设置最大切分8轮
constexpr double TIME_LOWER_RATIO = 2.0;
constexpr double TIME_UPPER_RATIO = 3.0;
constexpr uint64_t EXPANSION_RATIO = 2;

class AlltoAllMM : public OneCalcOneCommBase {
public:
    // Constructor
    explicit AlltoAllMM(const mc2tiling::TilingArgs &args, uint32_t inputRankDim, KernelType inputKernelType,
                        SocVersion inputSocVersion = SocVersion::SOC950, bool isCommunicationBefore = false)
        : OneCalcOneCommBase(args, inputRankDim, inputKernelType, inputSocVersion),
          socVersion_(inputSocVersion)
    {
        if (isCommunicationBefore) {
            // 如果是AllToAllMatmul，设置CommShapeLen为k轴的长度
            if (args.geAType == ge::DT_FLOAT4_E2M1) {
                commPerf_.SetCommDtypeSizeExpansionFraction(EXPANSION_RATIO);
            }
            commPerf_.SetCommShapeLen(clusterInfo_.kValue);
            commPerf_.SetCommDTypeSize(clusterInfo_.inMatrixADtypeSize);
        } else {
            // 设置CommShapeLen为N轴的长度(FOR MMAlltoAll)
            commPerf_.SetCommShapeLen(clusterInfo_.nValue);
            commPerf_.SetCommDTypeSize(clusterInfo_.outMatrixCDtypeSize);
        }
        rankTileNum_ = commPerf_.GetRankTileNum();
        // 分别根据通信和计算的最小值设置minTileLen, 各自进行拟合计算出minTileLen
        tilingM_.SetMinLenByMax(commPerf_.GetLinearThresholdLen());
        tilingM_.SetMinLenByMax(matmulPerf_.GetLinearThresholdLen(rankTileNum_));
    }
    void EstimateKernelTime() override;
    void SelectTilingMethod() override;

private:
    SocVersion socVersion_;
    void SetCommTimeFactor();
    void PrintEstimateKernelTimeResult(double totalMatmulTime, double totalTpTime);
};

#endif
