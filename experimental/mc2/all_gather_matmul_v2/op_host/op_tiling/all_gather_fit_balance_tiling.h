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
 * \file all_gather_fit_balance_tiling.h
 * \brief
 */
#ifndef __ALL_GATHER_FIT_BALANCE_TILING_H__
#define __ALL_GATHER_FIT_BALANCE_TILING_H__

#pragma once
#include "op_host/op_tiling/mc2_fit_based_balance_tiling.h"

class AllGatherMMFitBalanceTiling : public Mc2FitBasedBalanceTiling {
public:
    double frontMMTime_ = 0;

    explicit AllGatherMMFitBalanceTiling(const mc2tiling::TilingArgs &args, KernelType kernelType,
                                         TopoType topoType = TopoType::STANDARD_CARD)
        : Mc2FitBasedBalanceTiling(args, kernelType, topoType)
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
};

#endif // __ALL_GATHER_FIT_BALANCE_TILING_H__
