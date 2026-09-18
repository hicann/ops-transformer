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
 * \file batch_matmul_reduce_scatter_all_to_all_formulaic_tiling.h
 * \brief
 */

#ifndef __BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_FORMULAIC_TILING_H__
#define __BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_FORMULAIC_TILING_H__

#pragma once
#include "op_host/op_tiling/one_calc_two_comm_tiling.h"

class ReduceScatterAll2AllBMM : public OneCalcTwoCommBase {
public:
    explicit ReduceScatterAll2AllBMM(const mc2tiling::TilingArgs &args, uint64_t inputEpDim, uint64_t inputTpDim,
                                     uint64_t batchSize, SocVersion_2201 socVersion_2201 = SocVersion_2201::SOC910_93)
        : OneCalcTwoCommBase(args, inputEpDim, inputTpDim, batchSize, socVersion_2201)
    {
        epCommPerf.SetCommShapeLen(clusterInfo.nValue / tpDim);
        epCommPerf.SetCommDTypeSize(clusterInfo.outMatrixCDtypeSize);
        tpCommPerf.SetCommShapeLen(clusterInfo.nValue);
        tpCommPerf.SetCommDTypeSize(clusterInfo.outMatrixCDtypeSize);
    }
};

class ReduceScatterAll2AllBMMShardH : public OneCalcTwoCommShardHBase {
public:
    explicit ReduceScatterAll2AllBMMShardH(const mc2tiling::TilingArgs &args, uint64_t inputEpDim, uint64_t inputTpDim,
                                           uint64_t batchSize,
                                           SocVersion_2201 socVersion_2201 = SocVersion_2201::SOC910_93)
        : OneCalcTwoCommShardHBase(args, inputEpDim, inputTpDim, batchSize, socVersion_2201)
    {
        epCommPerf.SetCommShapeLen(clusterInfo.nValue);
        epCommPerf.SetCommDTypeSize(clusterInfo.outMatrixCDtypeSize);
        tpCommPerf.SetCommShapeLen(clusterInfo.nValue);
        tpCommPerf.SetCommDTypeSize(clusterInfo.outMatrixCDtypeSize);
    }
    bool SetShortTilePositionFlag(double totalBmmTime, double totalCommTime) override
    {
        // short tile at front when totalBmmTime < totalCommTime
        if (totalBmmTime < totalCommTime) {
            return true;
        } else {
            return false;
        }
    };
};
#endif //__BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_FORMULAIC_TILING_H__
