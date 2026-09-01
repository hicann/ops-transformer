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
 * \file all_reduce_fit_balance_tiling.h
 * \brief
 */
#ifndef __ALL_REDUCE_FIT_BALANCE_TILING_H__
#define __ALL_REDUCE_FIT_BALANCE_TILING_H__

#pragma once
#include "op_host/op_tiling/mc2_fit_based_balance_tiling.h"
#include "mc2_comm_utils.h"

class MMAllReduceFitBalanceTiling : public Mc2FitBasedBalanceTiling {
public:
    explicit MMAllReduceFitBalanceTiling(const mc2tiling::TilingArgs &args, KernelType kernelType,
                                         TopoType topoType = TopoType::STANDARD_CARD,
                                         SocVersion socVersion = SocVersion::SOC950,
                                         uint8_t commMode = Mc2Comm::COMM_MODE_AICPU, bool isPertileFp8 = false)
        : Mc2FitBasedBalanceTiling(args, kernelType, topoType, socVersion)
    {
        commPerf_.SetCommShapeLen(args.nValue);
        commPerf_.SetCommDTypeSize(mmInfo_.outMatrixCDtypeSize);

        tilingM_.SetMinLenByMax(matmulPerf_.GetBaseM());
        tilingM_.SetAlignLength(matmulPerf_.GetBaseM());
        isAicpuMode_ = (commMode == Mc2Comm::COMM_MODE_AICPU);
        this->isPertileFp8_ = isPertileFp8;
        if (isAicpuMode_ && isPertileFp8_ && rankDim_ >= AICPU_PERTILE_MIN_RANK_DIM) {
            tilingM_.SetMaxTileCnt(AICPU_PERTILE_MAX_TILE_CNT);
        }
    }

    void EstimateMMCommTime() override;
    void SetShortTileLen() override;
    void AdjustLongShortTileLen() override;
    void SetLongTileLen() override;
    void SetCommBoundTile();
    void AlignLongTileLen();
    void SetIsAlign(bool isAlign)
    {
        isAlign_ = isAlign;
    }
    uint64_t Align256(uint64_t value) const;

private:
    static constexpr uint64_t AICPU_PERTILE_MAX_TILE_CNT = 2U;
    static constexpr uint64_t AICPU_PERTILE_ALIGN_BASE = 256U;
    static constexpr uint64_t AICPU_PERTILE_MIN_RANK_DIM = 8U;

    void AdjustAicpuPertileEqualSplit();

    bool isAlign_ = false;
    bool isAicpuMode_ = false;
    bool isPertileFp8_ = false;
};

#endif // __ALL_REDUCE_FIT_BALANCE_TILING_H__
