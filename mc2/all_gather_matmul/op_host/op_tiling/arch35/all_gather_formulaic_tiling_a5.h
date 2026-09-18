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
 * \file all_gather_formulaic_tiling_a5.h
 * \brief
 */
#ifndef __ALL_GATHER_FORMULAIC_TILING_A5_H__
#define __ALL_GATHER_FORMULAIC_TILING_A5_H__

#include "../all_gather_formulaic_tiling.h"

class AllGatherPlusMMA5 : public AllGatherPlusMM {
public:
    // Constructor
    // A5专用:A5内蕴于类本身,不再接收档位key
    explicit AllGatherPlusMMA5(const mc2tiling::TilingArgs &args, uint32_t inputRankDim, KernelType inputKernelType)
        : AllGatherPlusMM(args, inputRankDim, inputKernelType, Ops::Base::DAV_3510)
    {
        commPerf_.SetCommShapeLen(clusterInfo_.kValue);
        commPerf_.SetCommDTypeSize(clusterInfo_.inMatrixADtypeSize);
        rankTileNum_ = commPerf_.GetRankTileNum();
        tilingM_.SetMinLenByMax(commPerf_.GetLinearThresholdLen());
        tilingM_.SetMinLenByMax(matmulPerf_.GetLinearThresholdLen(rankDim_));
    }

private:
    void SetCommTimeFactor() override;
    bool GetAllowMoreCuts(bool smallMFlag) override;
};

#endif //__ALL_GATHER_FORMULAIC_TILING_A5_H__
