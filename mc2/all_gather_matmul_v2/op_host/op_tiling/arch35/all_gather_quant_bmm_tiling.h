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
 * \file all_gather_quant_bmm_tiling.h
 * \brief
 */

#ifndef __ALL_GATHER_QUANT_BMM_TILING_H__
#define __ALL_GATHER_QUANT_BMM_TILING_H__

#pragma once
#include "../all_gather_matmul_tiling_base_v2.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"
#include "quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_tiling.h"
#include "../../../op_kernel/all_gather_matmul_tiling_data.h"

namespace optiling {

class AllGatherQuantBmmTiling : public AllGatherMatmulTilingBase {
    friend class AllGatherQuantBmmHelper;

public:
    explicit AllGatherQuantBmmTiling(gert::TilingContext *context);
    ~AllGatherQuantBmmTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    CutResult GetTilingResult() override;

    ::TCubeTiling &MutableTCubeLocalTilingData();
    DequantBmm::Mc2QuantBatchMatmulV3DataParams &MutableTCubeLocalTilingParams();
    DequantBmm::Mc2L2cacheTileParams &MutableTCubeLocalTilingTileL2();
    DequantBmm::Mc2SlidingWindowParams &MutableTCubeLocalTilingWindowParam();

    ::TCubeTiling &MutableTCubeTileTilingData();
    DequantBmm::Mc2QuantBatchMatmulV3DataParams &MutableTCubeTileTilingParams();
    DequantBmm::Mc2L2cacheTileParams &MutableTCubeTileTilingTileL2();
    DequantBmm::Mc2SlidingWindowParams &MutableTCubeTileTilingWindowParam();

    ::TCubeTiling &MutableTCubeTailTilingData();
    DequantBmm::Mc2QuantBatchMatmulV3DataParams &MutableTCubeTailTilingParams();
    DequantBmm::Mc2L2cacheTileParams &MutableTCubeTailTilingTileL2();
    DequantBmm::Mc2SlidingWindowParams &MutableTCubeTailTilingWindowParam();

    Mc2Tiling::RCSTiling &MutableRCSTilingDataA5();

    ge::graphStatus DoAdaptSlidWindowTiling();
    ge::graphStatus SetMc2Hcomm();
    void SetTilingKeyParams();
    ge::graphStatus CheckGroupSize();
    ge::graphStatus CheckPerBlockScaleInput();
    ge::graphStatus CheckMXFPScaleInput();
    ge::graphStatus CheckMXFP4Input();
    ge::graphStatus CheckPerTensorScaleInput();
    ge::graphStatus CheckBiasInput();
    ge::graphStatus SetQuantScene();
    mc2tiling::Mc2QuantMode GetQuantScene();
    ge::graphStatus CheckScaleInvShape();
    ge::graphStatus CheckInputValid();
    ge::graphStatus CheckX1Input();
    ge::graphStatus CheckInput() override;

private:
    Mc2Tiling::AllGatherMatmulTilingDataFp8 allGatherMatmulTilingDataFp8Self_;
    Mc2Tiling::AllGatherMatmulTilingDataFp8 *allGatherMatmulTilingDataFp8_;
    uint64_t myWorkSpaceSize_{0U};
    uint64_t scale1kSpaceSize_{0U};
    mc2tiling::Mc2QuantMode quantMmMode_{mc2tiling::Mc2QuantMode::INVALID_MODE};
    bool isFp4_ = false;
};

class AllGatherQuantBmmHelper : public Mc2AdaptiveSlidingWindowTiling {
public:
    AllGatherQuantBmmHelper(AllGatherQuantBmmTiling &allGatherQuantBmmTiling,
                            DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &out, bool isLocal = false);
    const gert::Shape GetX1Shape(const size_t index) override;
    const gert::Shape GetX2Shape(const size_t index) override;
    const gert::Shape GetOutputShape(const size_t index) override;
    const gert::Shape &GetScaleShape(const size_t index) override;
    const gert::StorageShape *GetPertokenShape(const size_t index) override;
    const gert::StorageShape *GetBiasShape(const size_t index) override;
    const gert::StorageShape *GetOffsetShape(const size_t index);
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    void PrintTilingInputParam(Mc2QuantBatchMatmulInfo &quantBatchMatmulInfo);
    void AnalyzeBatchInfo(const gert::Shape &oriShapeA, const gert::Shape &oriShapeB) override;
    ge::graphStatus PostTiling() override;
    void SetBatch();

private:
    AllGatherQuantBmmTiling &tilingProcesser_;
    uint32_t batch1_{1};
    uint32_t batch2_{1};
    uint32_t batch3_{1};
    uint32_t batch4_{1};
    bool isLocal_ = false;
};
} // namespace optiling

#endif //__ALL_GATHER_QUANT_BMM_TILING_H__
