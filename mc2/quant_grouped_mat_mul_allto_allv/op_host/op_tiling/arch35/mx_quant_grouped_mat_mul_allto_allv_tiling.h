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
 * \file mx_quant_grouped_mat_mul_allto_allv_gmm_tiling.h
 * \brief
 */

#ifndef MX_QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H
#define MX_QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H

#pragma once
#include "../quant_grouped_mat_mul_allto_allv_tiling_common.h"

namespace Mc2Tiling {
namespace Mc2GroupedMatmul {
constexpr uint64_t MX_SCALE_GROUP = 64;
constexpr uint64_t EVEN_ALIGN = 2;

constexpr uint64_t GROUP_M_OFFSET = 32;
constexpr uint64_t GROUP_N_OFFSET = 16;
constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
constexpr uint64_t MX_GROUP_SIZE_K = 32;
constexpr uint64_t MX_GROUP_SIZE_M = 1;
constexpr uint64_t MX_GROUP_SIZE_N = 1;

class MxQuantGroupedMatmulAllToAllvTiling : public QuantGroupedMatmulAllToAllvTilingCommon {
public:
    explicit MxQuantGroupedMatmulAllToAllvTiling(gert::TilingContext* context)
        : QuantGroupedMatmulAllToAllvTilingCommon(context) {};
    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }
    ~MxQuantGroupedMatmulAllToAllvTiling() override = default;

protected:
    void Reset();
    bool IsCapable() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus CheckAndSetInputOutputInfo() override;

    ge::graphStatus CheckAndSetLocalParamsGmm() override;
    ge::graphStatus CheckAndSetLocalParamsMm() override;
    ge::graphStatus CheckParamsRelationGmm() override;
    ge::graphStatus CheckParamsRelationMm() override;
    ge::graphStatus CheckParamsAttrEpAndSetLocalParams() override;
    ge::graphStatus MxCheckShapeDimensions(const gert::StorageShape* shape, uint64_t dims, const char* shapeName);

private:
    ge::graphStatus CheckMxQuantGmmScaleShapes();
    ge::graphStatus CheckMxQuantMmScaleShapes();
    ge::graphStatus CheckMxQuantDtypeConstraints();

    ge::graphStatus CheckMxfp4SpecificConstraints(); // MXFP4 约束校验
};

} // namespace Mc2GroupedMatmul
} // namespace Mc2Tiling
#endif
