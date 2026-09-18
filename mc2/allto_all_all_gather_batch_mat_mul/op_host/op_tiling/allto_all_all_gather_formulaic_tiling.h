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
 * \file all_to_all_all_gather_formulaic_tiling.h
 * \brief
 */

#ifndef __ALL_TO_ALL_ALL_GATHER_FORMULAIC_TILING_H__
#define __ALL_TO_ALL_ALL_GATHER_FORMULAIC_TILING_H__

#pragma once
#include "op_host/op_tiling/one_calc_two_comm_tiling.h"

class All2AllAllGatherBMM : public OneCalcTwoCommBase {
public:
    explicit All2AllAllGatherBMM(const mc2tiling::TilingArgs &args, uint64_t inputEpDim, uint64_t inputTpDim,
                                 uint64_t batchSize, SocVersion_2201 socVersion_2201 = SocVersion_2201::SOC910_93)
        : OneCalcTwoCommBase(args, inputEpDim, inputTpDim, batchSize, socVersion_2201)
    {
        epCommPerf.SetCommShapeLen(clusterInfo.kValue / tpDim);
        epCommPerf.SetCommDTypeSize(args.inputDtypeSize);
        tpCommPerf.SetCommShapeLen(clusterInfo.kValue);
        tpCommPerf.SetCommDTypeSize(args.inputDtypeSize);
    }
};

class All2AllAllGatherBMMShardH : public OneCalcTwoCommShardHBase {
public:
    explicit All2AllAllGatherBMMShardH(const mc2tiling::TilingArgs &args, uint64_t inputEpDim, uint64_t inputTpDim,
                                       uint64_t batchSize, SocVersion_2201 socVersion_2201 = SocVersion_2201::SOC910_93)
        : OneCalcTwoCommShardHBase(args, inputEpDim, inputTpDim, batchSize, socVersion_2201)
    {
        epCommPerf.SetCommShapeLen(clusterInfo.kValue);
        epCommPerf.SetCommDTypeSize(clusterInfo.inMatrixADtypeSize);
        tpCommPerf.SetCommShapeLen(clusterInfo.kValue);
        tpCommPerf.SetCommDTypeSize(clusterInfo.inMatrixADtypeSize);
    }
    bool SetShortTilePositionFlag(double totalBMMTime, double totalCommTime) override
    {
        // Short tile at front when totalBMMTime > totalCommTime
        if (totalBMMTime > totalCommTime) {
            return true;
        }
        return false;
    };
};
#endif //__ALL_TO_ALL_ALL_GATHER_FORMULAIC_TILING_H__
