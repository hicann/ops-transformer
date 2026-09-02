/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_grouped_mat_mul_tiling_base.h
 * \brief
 */
#ifndef MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_H
#define MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/op_tiling/mc2_tiling_struct.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../../../allto_allv_quant_grouped_mat_mul/op_host/op_tiling/allto_allv_quant_grouped_mat_mul_tiling_base.h"

namespace optiling {
constexpr uint32_t NON_QUANT_MM_X_INDEX = 4U;
constexpr uint32_t NON_QUANT_MM_WEIGHT_INDEX = 5U;
constexpr uint32_t NON_QUANT_ATTR_TRANS_GMM_WEIGHT_INDEX = 4;
constexpr uint32_t NON_QUANT_ATTR_TRANS_MM_WEIGHT_INDEX = 5;
constexpr uint32_t NON_QUANT_ATTR_PERMUTE_OUT_FLAG_INDEX = 6;

struct SetMMTilingParams {
    matmul_tiling::DataType matmulDtype;
    int32_t curMaxM;
    int32_t curMaxK;
    int32_t curMaxN;
    int32_t curBaseM;
    int32_t curBaseN;
    int32_t type;
};

struct MMTilingParams {
    int32_t curMaxM;
    int32_t curMaxK;
    int32_t curMaxN;
    int32_t *curBaseM;
    int32_t *curBaseK;
    int32_t *curBaseN;
};

class AlltoAllvGmmTilingBase : public AlltoAllvQuantGmmTilingBase {
public:
    explicit AlltoAllvGmmTilingBase(gert::TilingContext *context)
        : AlltoAllvQuantGmmTilingBase(context) {};

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetShapeInfo();
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetPlatformInfo() override;
};

class AlltoAllvGmmTilingCommon {
public:
    virtual ge::graphStatus AlltoAllvGmmOpTilingFunc(gert::TilingContext *context) = 0;
};
} // namespace optiling

#endif // MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_H
